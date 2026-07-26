"""Replay exactly the stopped E1 BF16 reference records in HF FP16.

This is deliberately not a formal E1 runner.  It reads BF16 checkpoints as an
immutable source contract and writes a wholly separate validation directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import socket
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.experiments.qaego4d_e1.core import (
    CONTEXT_SCOPE,
    CanonicalCase,
    E1ProtocolError,
    atomic_write_json,
    build_prompt,
    decode_requested_frames,
    load_json,
    sha256_file,
    stable_hash,
)
from src.experiments.qaego4d_e1.formal_core import _valid_core_record
from src.experiments.qaego4d_e1.runner import (
    DEFAULT_ANNOTATION_ROOT,
    DEFAULT_CONFIG,
    _load_all_cases,
    _parse_closed_prediction,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BF16_ROOT = ROOT / "outputs/experiments/qaego4d_e1_core_v1"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/qaego4d_e1_fp16_validation256_v1"
DEFAULT_AMENDMENT = ROOT / "config/experiments/qaego4d_e1_core_amendment_v1.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").casefold())).strip()


def _record_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["task"]), str(row["question_id"]), str(row["condition"])


def _checkpoint_path(output_root: Path, entry: dict[str, Any]) -> Path:
    return output_root / "checkpoints" / entry["task"] / entry["condition"] / f"{entry['question_id']}.json"


def _extract_immutable_manifest(*, bf16_root: Path, config: dict[str, Any], amendment: dict[str, Any]) -> dict[str, Any]:
    config_hash = stable_hash(config)
    amendment_hash = stable_hash(amendment)
    paths = sorted((bf16_root / "checkpoints").glob("*/*/*.json"))
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    errors: list[str] = []
    for path in paths:
        try:
            record = load_json(path)
        except Exception as error:
            errors.append(f"unreadable {path}: {error!r}")
            continue
        task = str(record.get("task"))
        manifest_hash = str(record.get("manifest_hash"))
        if not _valid_core_record(
            record, config_hash=config_hash, manifest_hash=manifest_hash, amendment_hash=amendment_hash,
        ) or record.get("success") is not True:
            errors.append(f"invalid/non-success BF16 checkpoint {path}")
            continue
        key = _record_key(record)
        if key in seen:
            errors.append(f"duplicate BF16 task/question/condition key: {key}")
            continue
        seen.add(key)
        required = {
            "prompt", "prompt_hash", "requested_timestamps_sec", "selected_frames",
            "selection_allowed_interval_sec", "frame_selection_rule", "created_at",
        }
        if not required <= set(record):
            errors.append(f"incomplete BF16 checkpoint {path}")
            continue
        entries.append({
            "ordinal": 0,
            "task": task,
            "question_id": str(record["question_id"]),
            "condition": str(record["condition"]),
            "clip_uid": str(record["clip_uid"]),
            "video_uid": str(record["video_uid"]),
            "context_scope": str(record["context_scope"]),
            "prompt": str(record["prompt"]),
            "prompt_hash": str(record["prompt_hash"]),
            "config_hash": str(record["config_hash"]),
            "manifest_hash": manifest_hash,
            "frame_selection_rule": str(record["frame_selection_rule"]),
            "selection_allowed_interval_sec": record["selection_allowed_interval_sec"],
            "requested_timestamps_sec": record["requested_timestamps_sec"],
            "selected_frames": record["selected_frames"],
            "frames_shown": int(record["frames_shown"]),
            "bf16_checkpoint_path": str(path.resolve()),
            "bf16_checkpoint_sha256": sha256_file(path),
            "bf16_created_at": str(record["created_at"]),
        })
    if errors:
        raise E1ProtocolError("; ".join(errors))
    if len(entries) != 256:
        raise E1ProtocolError(f"Expected exactly 256 valid BF16 checkpoints, found {len(entries)}")
    entries.sort(key=lambda row: (row["bf16_created_at"], row["task"], row["question_id"], row["condition"]))
    for ordinal, entry in enumerate(entries, start=1):
        entry["ordinal"] = ordinal
    entry_hash = stable_hash(entries)
    return {
        "schema_version": "qaego4d-e1-fp16-validation-manifest-v1",
        "created_at": _utc_now(),
        "purpose": "isolated paired FP16 validation; not formal E1 statistics",
        "source_bf16_output_root": str(bf16_root.resolve()),
        "source_bf16_protocol_snapshot_sha256": sha256_file(bf16_root / "protocol_snapshot.json"),
        "source_bf16_checkpoint_count": len(entries),
        "base_config_path": str(DEFAULT_CONFIG.resolve()),
        "base_config_hash": config_hash,
        "base_config_file_sha256": sha256_file(DEFAULT_CONFIG),
        "source_amendment_hash": amendment_hash,
        "validation_dtype": "float16",
        "entries_hash": entry_hash,
        "entries": entries,
    }


def _load_or_create_manifest(*, output_root: Path, bf16_root: Path, config: dict[str, Any], amendment: dict[str, Any]) -> dict[str, Any]:
    generated = _extract_immutable_manifest(bf16_root=bf16_root, config=config, amendment=amendment)
    path = output_root / "validation_manifest.json"
    if path.exists():
        existing = load_json(path)
        if existing.get("entries_hash") != generated["entries_hash"] or existing.get("entries") != generated["entries"]:
            raise E1ProtocolError("Existing FP16 validation manifest does not match immutable BF16 source checkpoints")
        return existing
    atomic_write_json(path, generated)
    return generated


class FP16ValidationModel:
    """Standalone FP16 implementation, leaving frozen BF16 formal code untouched."""

    def __init__(self, *, model_path: Path, max_pixels: int, seed: int, minimum_free_vram_gib: float) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("FP16 validation requires CUDA")
        free, _ = torch.cuda.mem_get_info()
        if free < int(minimum_free_vram_gib * 1024**3):
            raise RuntimeError(
                f"Insufficient free VRAM for FP16 validation: {free / 1024**3:.2f} GiB available, "
                f"need >= {minimum_free_vram_gib:.2f} GiB"
            )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self._torch = torch
        self.pre_model_load_memory = self._snapshot()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float16, device_map={"": 0}
        )
        self.model.eval()
        torch.cuda.synchronize()
        self.model_load_time_s = time.perf_counter() - started
        self.post_model_load_memory = self._snapshot()
        self.nonfinite_parameter_names = [
            name for name, parameter in self.model.named_parameters()
            if parameter.is_floating_point() and not bool(torch.isfinite(parameter).all().item())
        ]
        self.last_inference_telemetry: dict[str, Any] | None = None

    def _snapshot(self) -> dict[str, Any]:
        torch = self._torch
        free, total = torch.cuda.mem_get_info()
        def value(number: int) -> dict[str, Any]:
            return {"bytes": int(number), "gib": float(number / 1024**3)}
        return {
            "allocated": value(int(torch.cuda.memory_allocated())),
            "reserved": value(int(torch.cuda.memory_reserved())),
            "free": value(int(free)), "total": value(int(total)),
        }

    def is_cuda_oom(self, error: BaseException) -> bool:
        return isinstance(error, self._torch.OutOfMemoryError) or "out of memory" in str(error).lower()

    def recover_after_oom(self) -> dict[str, Any]:
        started = time.perf_counter()
        gc.collect()
        self._torch.cuda.empty_cache()
        row = self._snapshot()
        row["recovery_time_s"] = time.perf_counter() - started
        return row

    def infer(self, *, prompt: str, images: list[Any], max_new_tokens: int) -> dict[str, Any]:
        torch = self._torch
        inputs: dict[str, Any] | None = None
        generated: Any = None
        generated_only: Any = None
        telemetry: dict[str, Any] = {"empty_cache_called": True, "before_preprocess": self._snapshot()}
        torch.cuda.reset_peak_memory_stats()
        try:
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
            rendered = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            processor_inputs: dict[str, Any] = {"text": [rendered], "padding": True, "return_tensors": "pt"}
            if images:
                processor_inputs["images"] = images
            preprocessing_started = time.perf_counter()
            inputs = self.processor(**processor_inputs)
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            text_tokens = len(self.processor.tokenizer(rendered, add_special_tokens=False)["input_ids"])
            total_input_tokens = int(inputs["input_ids"].shape[1])
            nonfinite_input_tensors = [
                key for key, value in inputs.items()
                if getattr(value, "is_floating_point", lambda: False)() and not bool(torch.isfinite(value).all().item())
            ]
            torch.cuda.synchronize()
            preprocessing_time = time.perf_counter() - preprocessing_started
            telemetry["before_generate"] = self._snapshot()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens
                )
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - started
            generated_only = generated[:, total_input_tokens:]
            raw_output = self.processor.batch_decode(
                generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            telemetry["after_generate_before_cleanup"] = {
                **self._snapshot(),
                "peak_allocated": {"bytes": int(torch.cuda.max_memory_allocated()), "gib": float(torch.cuda.max_memory_allocated() / 1024**3)},
                "peak_reserved": {"bytes": int(torch.cuda.max_memory_reserved()), "gib": float(torch.cuda.max_memory_reserved() / 1024**3)},
            }
            return {
                "raw_output": raw_output, "answer_model_time_s": inference_time,
                "answer_preprocess_time_s": preprocessing_time, "text_tokens": text_tokens,
                "visual_tokens": max(0, total_input_tokens - text_tokens), "total_input_tokens": total_input_tokens,
                "output_tokens": int(generated_only.shape[1]),
                "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
                "nonfinite_input_tensors": nonfinite_input_tensors,
            }
        except BaseException as error:
            telemetry["exception"] = repr(error)
            telemetry["is_cuda_oom"] = self.is_cuda_oom(error)
            telemetry["at_exception_before_cleanup"] = {
                **self._snapshot(),
                "peak_allocated": {"bytes": int(torch.cuda.max_memory_allocated()), "gib": float(torch.cuda.max_memory_allocated() / 1024**3)},
                "peak_reserved": {"bytes": int(torch.cuda.max_memory_reserved()), "gib": float(torch.cuda.max_memory_reserved() / 1024**3)},
            }
            raise
        finally:
            cleanup_started = time.perf_counter()
            del inputs, generated, generated_only
            gc.collect()
            torch.cuda.empty_cache()
            telemetry["after_cleanup"] = self._snapshot()
            telemetry["cleanup_time_s"] = time.perf_counter() - cleanup_started
            self.last_inference_telemetry = telemetry

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        self._torch.cuda.empty_cache()


def _case_for_entry(*, entry: dict[str, Any], cases: dict[str, list[CanonicalCase]], config: dict[str, Any]) -> CanonicalCase:
    matches = [case for case in cases[entry["task"]] if case.question_id == entry["question_id"]]
    if len(matches) != 1:
        raise E1ProtocolError(f"Current frozen manifests cannot resolve validation entry: {_record_key(entry)}")
    case = matches[0]
    identity = (case.clip_uid, case.video_uid, CONTEXT_SCOPE)
    if identity != (entry["clip_uid"], entry["video_uid"], entry["context_scope"]):
        raise E1ProtocolError(f"Current canonical identity drifted for {_record_key(entry)}")
    current_prompt, _, _ = build_prompt(
        case, open_template=config["prompts"]["open"], closed_template=config["prompts"]["closed"],
    )
    if current_prompt != entry["prompt"] or stable_hash(current_prompt) != entry["prompt_hash"]:
        raise E1ProtocolError(f"Frozen prompt drifted for {_record_key(entry)}")
    return case


def _valid_fp16_record(row: Any, *, entry: dict[str, Any], validation_manifest_hash: str) -> bool:
    required = {
        "success", "raw_output", "answer_model_time_s", "total_latency_s", "frames_shown",
        "visual_tokens", "peak_vram_gib", "cuda_memory", "validation_manifest_hash",
        "source_bf16_checkpoint_sha256", "dtype", "prompt_hash", "selected_frames",
    }
    return bool(
        isinstance(row, dict) and required <= set(row) and row.get("success") is True
        and row.get("dtype") == "float16" and row.get("validation_manifest_hash") == validation_manifest_hash
        and row.get("source_bf16_checkpoint_sha256") == entry["bf16_checkpoint_sha256"]
        and row.get("prompt_hash") == entry["prompt_hash"] and row.get("selected_frames") == entry["selected_frames"]
    )


def _run_entry(*, entry: dict[str, Any], case: CanonicalCase, model: FP16ValidationModel, config: dict[str, Any], validation_manifest_hash: str) -> dict[str, Any]:
    started = time.perf_counter()
    images: list[Any] = []
    selected_frames: list[dict[str, Any]] = []
    decode_time = 0.0
    decoded_count = 0
    answer: dict[str, Any] | None = None
    error: str | None = None
    oom_recovery: dict[str, Any] | None = None
    try:
        if entry["requested_timestamps_sec"]:
            images, selected_frames, decode_time, decoded_count = decode_requested_frames(
                case=case, requested_timestamps=entry["requested_timestamps_sec"],
                allowed_start_sec=float(entry["selection_allowed_interval_sec"][0]),
                allowed_end_sec=float(entry["selection_allowed_interval_sec"][1]),
            )
        if selected_frames != entry["selected_frames"]:
            raise E1ProtocolError(f"Exact BF16 frame identity mismatch before FP16 inference: {_record_key(entry)}")
        if len(selected_frames) != int(entry["frames_shown"]):
            raise E1ProtocolError(f"Frame count mismatch before FP16 inference: {_record_key(entry)}")
        try:
            answer = model.infer(
                prompt=entry["prompt"], images=images,
                max_new_tokens=int(config["decoding"][f"{case.task}_max_new_tokens"]),
            )
        except BaseException as exception:
            if not model.is_cuda_oom(exception):
                raise
            error = repr(exception)
            oom_recovery = model.recover_after_oom()
    finally:
        for image in images:
            image.close()
    success = answer is not None
    raw_output = answer["raw_output"] if success else None
    parsed = _parse_closed_prediction(raw_output) if success and case.task == "closed" else None
    telemetry = model.last_inference_telemetry or {}
    failed_peak = telemetry.get("at_exception_before_cleanup", {}).get("peak_allocated", {}).get("gib")
    return {
        "schema_version": "qaego4d-e1-fp16-validation-record-v1",
        "created_at": _utc_now(), "run_kind": "isolated_bf16_fp16_matched_validation",
        "formal_result": False, "reference_only": True, "dtype": "float16",
        "source_bf16_checkpoint_path": entry["bf16_checkpoint_path"],
        "source_bf16_checkpoint_sha256": entry["bf16_checkpoint_sha256"],
        "validation_manifest_hash": validation_manifest_hash,
        "ordinal": entry["ordinal"], "task": case.task, "question_id": case.question_id,
        "condition": entry["condition"], "clip_uid": case.clip_uid, "video_uid": case.video_uid,
        "context_scope": CONTEXT_SCOPE, "prompt": entry["prompt"], "prompt_hash": entry["prompt_hash"],
        "config_hash": entry["config_hash"], "manifest_hash": entry["manifest_hash"],
        "selection_allowed_interval_sec": entry["selection_allowed_interval_sec"],
        "frame_selection_rule": entry["frame_selection_rule"],
        "requested_timestamps_sec": entry["requested_timestamps_sec"], "selected_frames": selected_frames,
        "frames_shown": len(selected_frames), "frames_decoded_online": decoded_count,
        "video_decode_time_s": decode_time, "success": success, "error": error,
        "oom_recovery": oom_recovery, "raw_output": raw_output,
        "parsed_closed_option": parsed, "answer_parse_valid": True if case.task == "open" and success else parsed is not None,
        "empty_output": bool(success and not raw_output),
        "malformed_output": bool(success and case.task == "closed" and parsed is None),
        "nan_inf_detected": bool(success and answer and answer["nonfinite_input_tensors"]),
        "nonfinite_input_tensors": answer["nonfinite_input_tensors"] if success else None,
        "answer_model_time_s": answer["answer_model_time_s"] if success else None,
        "answer_preprocess_time_s": answer["answer_preprocess_time_s"] if success else None,
        "total_latency_s": time.perf_counter() - started,
        "text_tokens": answer["text_tokens"] if success else None,
        "visual_tokens": answer["visual_tokens"] if success else None,
        "total_input_tokens": answer["total_input_tokens"] if success else None,
        "output_tokens": answer["output_tokens"] if success else None,
        "model_calls": 1, "peak_vram_gib": answer["peak_vram_gib"] if success else failed_peak,
        "cuda_memory": telemetry,
    }


def _value_stats(values: list[float]) -> dict[str, float | None]:
    return {"mean": mean(values) if values else None, "median": median(values) if values else None,
            "total": sum(values) if values else 0.0, "min": min(values) if values else None, "max": max(values) if values else None}


def _open_disagreement(bf16: str, fp16: str) -> str:
    return "wording-only / synonymous" if _normalise_text(bf16) == _normalise_text(fp16) else "potentially meaning-changing (no LLM judge applied)"


def _summarise(*, output_root: Path, manifest: dict[str, Any], fp16_records: list[dict[str, Any]]) -> dict[str, Any]:
    source: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in manifest["entries"]:
        source[_record_key(entry)] = load_json(Path(entry["bf16_checkpoint_path"]))
    ordered = sorted(fp16_records, key=lambda row: row["ordinal"])
    paired = [(source[_record_key(row)], row) for row in ordered]
    comparison: dict[str, Any] = {"open": {}, "closed": {}}
    for task in ("open", "closed"):
        rows = [(bf, fp) for bf, fp in paired if fp["task"] == task]
        by_condition: dict[str, Any] = {}
        for condition in ("blind", "uniform_8", "oracle_leq8"):
            current = [(bf, fp) for bf, fp in rows if fp["condition"] == condition]
            if task == "closed":
                disagreements = [
                    {"question_id": fp["question_id"], "condition": condition,
                     "bf16_prediction": bf.get("parsed_closed_option"), "fp16_prediction": fp.get("parsed_closed_option"),
                     "ground_truth": bf.get("ground_truth_option_letter_after_prediction")}
                    for bf, fp in current if bf.get("parsed_closed_option") != fp.get("parsed_closed_option")
                ]
                by_condition[condition] = {"matched": len(current), "exact_option_agreement": len(current) - len(disagreements), "disagreements": disagreements}
            else:
                differences = [
                    {"question_id": fp["question_id"], "condition": condition,
                     "bf16_output": bf.get("raw_output"), "fp16_output": fp.get("raw_output"),
                     "classification": _open_disagreement(str(bf.get("raw_output") or ""), str(fp.get("raw_output") or ""))}
                    for bf, fp in current if bf.get("raw_output") != fp.get("raw_output")
                ]
                exact = len(current) - len(differences)
                normalised = sum(_normalise_text(str(bf.get("raw_output") or "")) == _normalise_text(str(fp.get("raw_output") or "")) for bf, fp in current)
                by_condition[condition] = {"matched": len(current), "exact_string_agreement": exact, "normalized_text_agreement": normalised, "non_exact_matches": differences}
        comparison[task] = by_condition
    successful_bf = [bf for bf, _ in paired if bf.get("success")]
    successful_fp = [fp for _, fp in paired if fp.get("success")]
    def perf(rows: list[dict[str, Any]]) -> dict[str, Any]:
        model_times = [float(row["answer_model_time_s"]) for row in rows if row.get("answer_model_time_s") is not None]
        total_times = [float(row["total_latency_s"]) for row in rows if row.get("total_latency_s") is not None]
        peaks = [float(row["peak_vram_gib"]) for row in rows if row.get("peak_vram_gib") is not None]
        reserved = [float(row["cuda_memory"].get("after_generate_before_cleanup", {}).get("peak_reserved", {}).get("gib")) for row in rows if row.get("cuda_memory", {}).get("after_generate_before_cleanup", {}).get("peak_reserved", {}).get("gib") is not None]
        return {
            "successful": len(rows), "model_time_s": _value_stats(model_times), "total_latency_s": _value_stats(total_times),
            "model_queries_per_minute": len(model_times) / sum(model_times) * 60 if model_times and sum(model_times) else None,
            "observed_per_record_queries_per_minute": len(total_times) / sum(total_times) * 60 if total_times and sum(total_times) else None,
            "peak_allocated_gib_max": max(peaks) if peaks else None, "peak_reserved_gib_max": max(reserved) if reserved else None,
        }
    bf_perf, fp_perf = perf(successful_bf), perf(successful_fp)
    bf_total = bf_perf["model_time_s"]["total"]
    fp_total = fp_perf["model_time_s"]["total"]
    summary = {
        "schema_version": "qaego4d-e1-fp16-validation-summary-v1", "created_at": _utc_now(),
        "validation_manifest_entries_hash": manifest["entries_hash"], "paired_records": len(paired),
        "by_task": dict(Counter(row["task"] for row in ordered)), "by_condition": dict(Counter(row["condition"] for row in ordered)),
        "prediction_comparison": comparison,
        "numerical_stability": {
            "bf16_successful": len(successful_bf), "fp16_successful": len(successful_fp),
            "fp16_failures": [row for row in ordered if not row.get("success")],
            "fp16_nan_inf_records": [row["question_id"] for row in ordered if row.get("nan_inf_detected")],
            "fp16_empty_outputs": [row["question_id"] for row in ordered if row.get("empty_output")],
            "fp16_malformed_outputs": [row["question_id"] for row in ordered if row.get("malformed_output")],
        },
        "performance": {
            "bf16": bf_perf, "fp16": fp_perf,
            "model_time_speedup_bf16_over_fp16": bf_total / fp_total if bf_total and fp_total else None,
        },
        "systematic_drift_diagnostics": {
            "fp16_refusal_like_outputs": [
                {"question_id": row["question_id"], "condition": row["condition"], "output": row["raw_output"]}
                for row in ordered if any(term in (row.get("raw_output") or "").casefold() for term in ("i don't know", "do not know", "cannot", "uncertain"))
            ],
            "fp16_repeated_output_counts": dict(Counter(row.get("raw_output") for row in ordered)),
        },
    }
    atomic_write_json(output_root / "comparison_summary.json", summary)
    atomic_write_json(output_root / "fp16_records.json", ordered)
    lines = [
        "# E1 BF16 → FP16 matched 256-checkpoint validation", "",
        "Reference-only engineering validation. BF16 and FP16 outputs are stored separately and are not formal E1 statistics.", "",
        f"- Paired records: {len(paired)}", f"- Task counts: {summary['by_task']}", f"- Condition counts: {summary['by_condition']}",
        f"- FP16 failures / NaN-Inf / empty / malformed: {len(summary['numerical_stability']['fp16_failures'])} / {len(summary['numerical_stability']['fp16_nan_inf_records'])} / {len(summary['numerical_stability']['fp16_empty_outputs'])} / {len(summary['numerical_stability']['fp16_malformed_outputs'])}",
        f"- Model-time speedup (BF16 / FP16): {summary['performance']['model_time_speedup_bf16_over_fp16']}", "",
        "See `comparison_summary.json` for every non-exact Open match and every Closed disagreement.",
    ]
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    amendment = load_json(args.amendment)
    if sha256_file(args.config) != amendment.get("parent_protocol_sha256"):
        raise E1ProtocolError("Current E1 config no longer matches the BF16 source amendment")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(output_root=args.output_root, bf16_root=args.bf16_root, config=config, amendment=amendment)
    validation_hash = str(manifest["entries_hash"])
    atomic_write_json(args.output_root / "validation_config_snapshot.json", {
        "base_e1_config_path": str(args.config.resolve()), "base_e1_config_sha256": sha256_file(args.config),
        "base_e1_config_hash": stable_hash(config), "only_numerical_override": {"dtype": "float16"},
        "validation_manifest_entries_hash": validation_hash, "formal_result": False, "reference_only": True,
    })
    cases, current_manifest_hashes = _load_all_cases(config, args.annotation_root)
    if any(entry["config_hash"] != stable_hash(config) for entry in manifest["entries"]):
        raise E1ProtocolError("BF16 source config hash does not match the current frozen E1 config")
    if any(entry["manifest_hash"] != current_manifest_hashes[entry["task"]] for entry in manifest["entries"]):
        raise E1ProtocolError("BF16 source manifest hashes do not match the current frozen manifests")
    cached: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        path = _checkpoint_path(args.output_root, entry)
        row: Any = None
        if path.is_file():
            try:
                row = load_json(path)
            except Exception:
                path.replace(path.with_suffix(path.suffix + f".corrupt-{int(time.time())}"))
        if _valid_fp16_record(row, entry=entry, validation_manifest_hash=validation_hash):
            cached.append(row)
        else:
            pending.append(entry)
    print(json.dumps({"matched_bf16_records": len(manifest["entries"]), "cache_hits": len(cached), "pending": len(pending), "dtype": "float16"}, ensure_ascii=False), flush=True)
    if args.manifest_only:
        return 0
    model: FP16ValidationModel | None = None
    created: list[dict[str, Any]] = []
    load: dict[str, Any] = {"model_loads": 0}
    session_started = time.perf_counter()
    try:
        if pending:
            spec = config["answer_models"]["formal_candidate"]
            model = FP16ValidationModel(
                model_path=Path(spec["local_path"]), max_pixels=int(config["decoding"]["max_pixels"]),
                seed=int(config["decoding"]["seed"]), minimum_free_vram_gib=float(args.minimum_free_vram_gib),
            )
            load = {"model_loads": 1, "model_load_time_s": model.model_load_time_s,
                    "pre_model_load_memory": model.pre_model_load_memory, "post_model_load_memory": model.post_model_load_memory,
                    "nonfinite_parameter_names": model.nonfinite_parameter_names}
            if model.nonfinite_parameter_names:
                raise RuntimeError("FP16 model parameters contain NaN/Inf")
            for entry in pending:
                case = _case_for_entry(entry=entry, cases=cases, config=config)
                record = _run_entry(entry=entry, case=case, model=model, config=config, validation_manifest_hash=validation_hash)
                atomic_write_json(_checkpoint_path(args.output_root, entry), record)
                created.append(record)
                atomic_write_json(args.output_root / "progress.json", {
                    "updated_at": _utc_now(), "expected": len(manifest["entries"]),
                    "cache_hits_at_session_start": len(cached), "created_this_session": len(created),
                    "completed": len(cached) + len(created), "last": _record_key(record), "success": record["success"],
                })
                if not record["success"]:
                    raise RuntimeError(f"FP16 validation failure: {_record_key(record)}: {record['error']}")
                if len(created) % max(1, int(args.progress_every)) == 0:
                    print(json.dumps({"completed": len(cached) + len(created), "expected": len(manifest["entries"]), "last": _record_key(record)}, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            model.close()
        gc.collect()
    all_rows = cached + created
    if len(all_rows) != len(manifest["entries"]):
        raise RuntimeError(f"Incomplete FP16 validation: {len(all_rows)} / {len(manifest['entries'])}")
    summary = _summarise(output_root=args.output_root, manifest=manifest, fp16_records=all_rows)
    atomic_write_json(args.output_root / "run_metadata.json", {
        "created_at": _utc_now(), "host": socket.gethostname(), "session_wall_time_s": time.perf_counter() - session_started,
        **load, "matched_records": len(all_rows), "validation_manifest_entries_hash": validation_hash,
    })
    print(json.dumps({"completed": len(all_rows), "output_root": str(args.output_root), "speedup": summary["performance"]["model_time_speedup_bf16_over_fp16"]}, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated exact 256-checkpoint HF FP16 validation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--bf16-root", type=Path, default=DEFAULT_BF16_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--manifest-only", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
