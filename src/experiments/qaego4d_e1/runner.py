from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import (
    CONDITIONS,
    CONTEXT_SCOPE,
    TASKS,
    CanonicalCase,
    E1ProtocolError,
    atomic_write_json,
    build_prompt,
    decode_zero_decodable_interval_nearest_frame,
    decode_requested_frames,
    load_cases,
    load_json,
    oracle_requested_timestamps,
    sha256_file,
    stable_hash,
    uniform_requested_timestamps,
    valid_result,
)
from .model import LocalQwenVLEngineeringModel


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATION_ROOT = Path("/cs/student/project_msc/2025/rai/xinanx01/QaEgo4D/processed/data/unified")
DEFAULT_CONFIG = ROOT / "config/experiments/qaego4d_e1_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/qaego4d_e1_preparation_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_closed_prediction(raw_output: str) -> str | None:
    match = re.fullmatch(r"(?:option\s*)?([A-D])\.?", raw_output.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _load_all_cases(config: dict[str, Any], annotation_root: Path) -> tuple[dict[str, list[CanonicalCase]], dict[str, str]]:
    manifests = config["manifests"]
    cases: dict[str, list[CanonicalCase]] = {}
    hashes: dict[str, str] = {}
    for task in TASKS:
        manifest_path = ROOT / manifests[task]
        mapping_path = ROOT / manifests[f"{task}_mapping"]
        loaded, manifest_hash = load_cases(
            task=task,
            annotation_root=annotation_root,
            manifest_path=manifest_path,
            mapping_path=mapping_path,
        )
        cases[task] = loaded
        hashes[task] = manifest_hash
        expected = int(config["expected_question_counts"][task])
        if len(loaded) != expected:
            raise E1ProtocolError(f"{task} count={len(loaded)}, expected={expected}")
    return cases, hashes


def _case_valid_for_dry_run(case: CanonicalCase) -> bool:
    return (
        case.clip_duration_sec > 0
        and case.evidence_start_sec >= case.clip_start_sec
        and case.evidence_end_sec >= case.evidence_start_sec
        and case.evidence_end_sec <= case.clip_end_sec
    )


def select_one_video_dry_run_cases(cases: dict[str, list[CanonicalCase]]) -> list[CanonicalCase]:
    """Choose two Open and two Closed rows from one parent video using metadata only."""
    eligible = {
        task: [case for case in rows if _case_valid_for_dry_run(case)]
        for task, rows in cases.items()
    }
    by_uid: dict[str, dict[str, list[CanonicalCase]]] = {}
    for task, rows in eligible.items():
        for case in rows:
            by_uid.setdefault(case.video_uid, {}).setdefault(task, []).append(case)
    candidates = [
        uid for uid, by_task in by_uid.items()
        if by_task.get("open") and by_task.get("closed")
    ]
    if not candidates:
        raise E1ProtocolError("No parent video has valid Open and Closed dry-run samples")
    uid = sorted(candidates)[0]
    selected: list[CanonicalCase] = []
    for task in TASKS:
        selected.extend(sorted(by_uid[uid][task], key=lambda row: row.question_id)[:2])
    return selected


def select_explicit_case(
    cases: dict[str, list[CanonicalCase]], *, task: str, question_id: str,
) -> list[CanonicalCase]:
    """Select exactly one frozen manifest row; used by bounded model probes."""
    if task not in TASKS:
        raise E1ProtocolError(f"Unknown task: {task}")
    matches = [case for case in cases[task] if case.question_id == question_id]
    if len(matches) != 1:
        raise E1ProtocolError(f"Expected exactly one {task} case for {question_id}, found {len(matches)}")
    case = matches[0]
    if not _case_valid_for_dry_run(case):
        raise E1ProtocolError(f"Explicit case is not valid for canonical-clip dry-run: {question_id}")
    return [case]


def _frame_plan(case: CanonicalCase, condition: str) -> tuple[list[float], str, tuple[float, float]]:
    if condition == "blind":
        return [], "no_visual_input", (case.clip_start_sec, case.clip_end_sec)
    if condition == "uniform_8":
        return (
            uniform_requested_timestamps(case.clip_start_sec, case.clip_end_sec, budget=8, nominal_fps=30.0),
            "canonical_clip_uniform_8", (case.clip_start_sec, case.clip_end_sec),
        )
    if condition == "uniform_32":
        return (
            uniform_requested_timestamps(case.clip_start_sec, case.clip_end_sec, budget=32, nominal_fps=30.0),
            "canonical_clip_uniform_32", (case.clip_start_sec, case.clip_end_sec),
        )
    if condition == "oracle_leq8":
        requested, rule, bounds = oracle_requested_timestamps(case, budget=8, nominal_fps=30.0)
        return requested, rule, bounds
    raise E1ProtocolError(f"Unknown E1 condition: {condition}")


def _checkpoint_path(output_root: Path, case: CanonicalCase, condition: str) -> Path:
    return output_root / "checkpoints" / case.task / condition / f"{case.question_id}.json"


def _record(
    *, case: CanonicalCase, condition: str, config: dict[str, Any], config_hash: str,
    manifest_hash: str, model: LocalQwenVLEngineeringModel, run_id: str,
    model_profile: str = "engineering_dry_run_only", formal_result: bool = False,
    run_kind: str = "engineering_dry_run", protocol_amendment_hash: str | None = None,
) -> dict[str, Any]:
    overall_started = time.perf_counter()
    prompt, options, correct_index = build_prompt(
        case,
        open_template=config["prompts"]["open"],
        closed_template=config["prompts"]["closed"],
    )
    requested, selection_rule, allowed_bounds = _frame_plan(case, condition)
    images = []
    selected_frames: list[dict[str, Any]] = []
    video_decode_time_s = 0.0
    frames_decoded_online = 0
    answer: dict[str, Any] | None = None
    failure_exception: str | None = None
    oom_recovery: dict[str, Any] | None = None
    try:
        if requested:
            try:
                images, selected_frames, video_decode_time_s, frames_decoded_online = decode_requested_frames(
                    case=case,
                    requested_timestamps=requested,
                    allowed_start_sec=allowed_bounds[0],
                    allowed_end_sec=allowed_bounds[1],
                )
            except E1ProtocolError as exc:
                is_zero_decodable_interval = (
                    condition == "oracle_leq8"
                    and case.evidence_start_sec < case.evidence_end_sec
                    and "No decodable frame in allowed range" in str(exc)
                )
                if not is_zero_decodable_interval:
                    raise
                images, selected_frames, video_decode_time_s, frames_decoded_online = (
                    decode_zero_decodable_interval_nearest_frame(case=case)
                )
                selection_rule = "interval_zero_decodable_nearest_frame"
                allowed_bounds = (case.clip_start_sec, case.clip_end_sec)
        budget = int(config["conditions"][condition]["frame_budget"])
        if len(selected_frames) > budget:
            raise E1ProtocolError(f"Frame budget exceeded: {condition}")
        if any(
            not (case.clip_start_sec <= frame["actual_timestamp_sec"] < case.clip_end_sec)
            for frame in selected_frames
        ):
            raise E1ProtocolError("Selected frame escaped canonical clip")
        try:
            answer = model.infer(
                prompt=prompt,
                images=images,
                max_new_tokens=int(config["decoding"][f"{case.task}_max_new_tokens"]),
            )
        except BaseException as exc:
            if not model.is_cuda_oom(exc):
                raise
            failure_exception = repr(exc)
            oom_recovery = model.recover_after_oom()
    finally:
        for image in images:
            image.close()
    success = answer is not None
    parsed = _parse_closed_prediction(answer["raw_output"]) if success and case.task == "closed" else None
    total_latency_s = time.perf_counter() - overall_started
    cuda_memory = getattr(model, "last_inference_telemetry", None) or {}
    failed_peak = cuda_memory.get("at_exception_before_cleanup", {}).get("peak_allocated", {})
    return {
        "schema_version": "qaego4d-e1-record-v2",
        "run_id": run_id,
        "created_at": _utc_now(),
        "engineering_only": not formal_result,
        "answer_model": config["answer_models"][model_profile]["model_id"],
        "answer_model_profile": model_profile,
        "formal_result": formal_result,
        "run_kind": run_kind,
        "protocol_amendment_hash": protocol_amendment_hash,
        "stage": "E1",
        "condition": condition,
        "task": case.task,
        "question_id": case.question_id,
        "clip_uid": case.clip_uid,
        "video_uid": case.video_uid,
        "context_scope": CONTEXT_SCOPE,
        "clip_start_sec": case.clip_start_sec,
        "clip_end_sec": case.clip_end_sec,
        "evidence_interval_sec": [case.evidence_start_sec, case.evidence_end_sec],
        "selection_allowed_interval_sec": list(allowed_bounds),
        "frame_selection_rule": selection_rule,
        "requested_timestamps_sec": requested,
        "selected_frames": selected_frames,
        "video_decode_time_s": video_decode_time_s,
        "frames_decoded_online": frames_decoded_online,
        "frames_shown": len(selected_frames),
        "retrieval_time_s": 0.0,
        "success": success,
        "error": failure_exception,
        "oom_recovery": oom_recovery,
        "answer_model_time_s": answer["answer_model_time_s"] if success else None,
        "answer_preprocess_time_s": answer["answer_preprocess_time_s"] if success else None,
        "total_latency_s": total_latency_s,
        "text_tokens": answer["text_tokens"] if success else None,
        "visual_tokens": answer["visual_tokens"] if success else None,
        "total_input_tokens": answer["total_input_tokens"] if success else None,
        "output_tokens": answer["output_tokens"] if success else None,
        "model_calls": 1,
        "api_calls": 0,
        "peak_vram_gib": answer["peak_vram_gib"] if success else failed_peak.get("gib"),
        "cuda_memory": cuda_memory,
        "prompt": prompt,
        "prompt_hash": stable_hash(prompt),
        "raw_output": answer["raw_output"] if success else None,
        "parsed_closed_option": parsed,
        "answer_parse_valid": (True if case.task == "open" else parsed is not None) if success else False,
        "options": options,
        "ground_truth_after_prediction": case.answer,
        "ground_truth_option_index_after_prediction": correct_index,
        "ground_truth_option_letter_after_prediction": (chr(65 + correct_index) if correct_index is not None else None),
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
    }


def _write_aggregate(output_root: Path, records: list[dict[str, Any]], *, metadata: dict[str, Any]) -> None:
    lines = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n"
    path = output_root / "dry_run_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lines, encoding="utf-8")
    atomic_write_json(output_root / "dry_run_summary.json", {
        **metadata,
        "records": len(records),
        "by_condition": dict(Counter(record["condition"] for record in records)),
        "by_task": dict(Counter(record["task"] for record in records)),
        "selected_parent_video_uids": sorted({record["video_uid"] for record in records}),
        "all_context_scope_canonical_clip": all(record["context_scope"] == CONTEXT_SCOPE for record in records),
        "all_selected_frames_inside_canonical_clip": all(
            all(
                record["clip_start_sec"] <= frame["actual_timestamp_sec"] < record["clip_end_sec"]
                for frame in record["selected_frames"]
            ) for record in records
        ),
    })
    baseline = metadata.get("post_model_load_memory")
    baseline_source = "explicit_post_model_load_snapshot"
    if baseline is None:
        # Used only by summary reconstruction of an already-finished run. The
        # first inference snapshot is after model load and before preprocessing.
        first = min(records, key=lambda row: str(row.get("created_at", "")), default=None)
        candidate = (first or {}).get("cuda_memory", {}).get("before_preprocess")
        if isinstance(candidate, dict):
            baseline = candidate
            baseline_source = "first_before_preprocess_snapshot_reconstructed"
    recovery_rows: list[dict[str, Any]] = []
    if isinstance(baseline, dict):
        baseline_allocated = int(baseline["allocated"]["bytes"])
        baseline_reserved = int(baseline["reserved"]["bytes"])
        execution_order = sorted(records, key=lambda row: str(row.get("created_at", "")))
        for ordinal, record in enumerate(execution_order, start=1):
            after_cleanup = record.get("cuda_memory", {}).get("after_cleanup")
            if not isinstance(after_cleanup, dict):
                continue
            allocated = int(after_cleanup["allocated"]["bytes"])
            reserved = int(after_cleanup["reserved"]["bytes"])
            recovery_rows.append({
                "sequence": ordinal,
                "question_id": record["question_id"],
                "task": record["task"],
                "condition": record["condition"],
                "success": record["success"],
                "allocated_after_cleanup_bytes": allocated,
                "allocated_after_cleanup_gib": after_cleanup["allocated"]["gib"],
                "reserved_after_cleanup_bytes": reserved,
                "reserved_after_cleanup_gib": after_cleanup["reserved"]["gib"],
                "free_after_cleanup_bytes": after_cleanup["free"]["bytes"],
                "free_after_cleanup_gib": after_cleanup["free"]["gib"],
                "allocated_delta_from_post_model_load_bytes": allocated - baseline_allocated,
                "reserved_delta_from_post_model_load_bytes": reserved - baseline_reserved,
            })
    allocated_values = [row["allocated_after_cleanup_bytes"] for row in recovery_rows]
    reserved_values = [row["reserved_after_cleanup_bytes"] for row in recovery_rows]
    fragmentation_threshold = 1024**3
    atomic_write_json(output_root / "memory_recovery_summary.json", {
        "post_model_load_baseline": baseline,
        "post_model_load_baseline_source": baseline_source if baseline is not None else None,
        "per_inference_after_cleanup": recovery_rows,
        "flags": {
            "monotonic_allocated_growth": len(allocated_values) > 1 and all(
                right > left for left, right in zip(allocated_values, allocated_values[1:])
            ),
            "monotonic_reserved_growth": len(reserved_values) > 1 and all(
                right > left for left, right in zip(reserved_values, reserved_values[1:])
            ),
            "fragmentation_like_reserved_minus_allocated_over_1gib": any(
                row["reserved_after_cleanup_bytes"] - row["allocated_after_cleanup_bytes"] > fragmentation_threshold
                for row in recovery_rows
            ),
            "unexpected_retained_allocated_over_1gib_above_baseline": any(
                row["allocated_delta_from_post_model_load_bytes"] > fragmentation_threshold
                for row in recovery_rows
            ),
        },
    })


def _run_frame_preflight(
    *, output_root: Path, selected: list[CanonicalCase], config: dict[str, Any], protocol: dict[str, Any],
) -> None:
    """Exercise actual canonical-clip decoding without loading an answer model."""
    checks: list[dict[str, Any]] = []
    for case in selected:
        for condition in CONDITIONS:
            prompt, options, _ = build_prompt(
                case,
                open_template=config["prompts"]["open"],
                closed_template=config["prompts"]["closed"],
            )
            requested, rule, bounds = _frame_plan(case, condition)
            images = []
            selected_frames: list[dict[str, Any]] = []
            decode_time_s = 0.0
            frames_decoded_online = 0
            try:
                if requested:
                    images, selected_frames, decode_time_s, frames_decoded_online = decode_requested_frames(
                        case=case, requested_timestamps=requested,
                        allowed_start_sec=bounds[0], allowed_end_sec=bounds[1],
                    )
            finally:
                for image in images:
                    image.close()
            budget = int(config["conditions"][condition]["frame_budget"])
            inside_clip = all(
                case.clip_start_sec <= row["actual_timestamp_sec"] < case.clip_end_sec
                for row in selected_frames
            )
            inside_oracle = condition != "oracle_leq8" or all(
                bounds[0] <= row["actual_timestamp_sec"] < bounds[1]
                for row in selected_frames
            )
            checks.append({
                "task": case.task, "question_id": case.question_id, "condition": condition,
                "clip_uid": case.clip_uid, "video_uid": case.video_uid,
                "context_scope": CONTEXT_SCOPE, "frame_selection_rule": rule,
                "requested_timestamps_sec": requested, "selected_frames": selected_frames,
                "frames_shown": len(selected_frames), "frame_budget": budget,
                "unique_source_pts": len({row["source_pts"] for row in selected_frames}),
                "video_decode_time_s": decode_time_s,
                "frames_decoded_online": frames_decoded_online,
                "inside_canonical_clip": inside_clip,
                "inside_oracle_allowed_interval": inside_oracle,
                "open_or_closed_prompt": prompt,
                "closed_options_present": options is not None,
            })
    validations = {
        "all_frames_inside_canonical_clip": all(row["inside_canonical_clip"] for row in checks),
        "uniform_8_budget_valid": all(row["frames_shown"] <= 8 and row["frames_shown"] == row["unique_source_pts"] for row in checks if row["condition"] == "uniform_8"),
        "uniform_32_budget_valid": all(row["frames_shown"] <= 32 and row["frames_shown"] == row["unique_source_pts"] for row in checks if row["condition"] == "uniform_32"),
        "oracle_budget_valid": all(row["frames_shown"] <= 8 and row["frames_shown"] == row["unique_source_pts"] and row["inside_oracle_allowed_interval"] for row in checks if row["condition"] == "oracle_leq8"),
        "blind_no_frames": all(row["frames_shown"] == 0 for row in checks if row["condition"] == "blind"),
        "prompts_distinct": config["prompts"]["open"] != config["prompts"]["closed"],
    }
    if not all(validations.values()):
        raise E1ProtocolError(f"Canonical frame preflight failed: {validations}")
    atomic_write_json(output_root / "frame_preflight.json", {
        **protocol,
        "preflight_only": True,
        "qwen_model_loads": 0,
        "qwen_model_calls": 0,
        "validations": validations,
        "checks": checks,
    })


def run(args: argparse.Namespace) -> int:
    if args.memory_robustness and args.output_root == DEFAULT_OUTPUT:
        args.output_root = DEFAULT_OUTPUT / "memory_robustness_rerun_v1"
    config = load_json(args.config)
    if config.get("experiment_id") != "E1":
        raise E1ProtocolError("Wrong experiment config")
    config_hash = stable_hash(config)
    cases, manifest_hashes = _load_all_cases(config, args.annotation_root)
    selected = (
        select_explicit_case(cases, task=args.single_task, question_id=args.single_question_id)
        if args.single_question_id else select_one_video_dry_run_cases(cases)
    )
    if not args.single_question_id and len({case.video_uid for case in selected}) != 1:
        raise E1ProtocolError("Dry-run selection leaked across parent videos")
    model_profile = args.model_profile
    if model_profile not in config["answer_models"]:
        raise E1ProtocolError(f"Unknown answer-model profile: {model_profile}")
    model_spec = config["answer_models"][model_profile]
    run_id = f"qaego4d-e1-dryrun-{model_profile}-{config_hash[:12]}"
    protocol = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "config_path": str(args.config),
        "config_hash": config_hash,
        "config_file_sha256": sha256_file(args.config),
        "manifest_hashes": manifest_hashes,
        "open_prompt_hash": stable_hash(config["prompts"]["open"]),
        "closed_prompt_hash": stable_hash(config["prompts"]["closed"]),
        "uniform_8_rule_hash": stable_hash(config["conditions"]["uniform_8"]),
        "uniform_32_rule_hash": stable_hash(config["conditions"]["uniform_32"]),
        "oracle_rule_hash": stable_hash(config["conditions"]["oracle_leq8"]),
        "decoding_hash": stable_hash(config["decoding"]),
        "selected_cases": [
            {"task": case.task, "question_id": case.question_id, "clip_uid": case.clip_uid,
             "video_uid": case.video_uid, "clip_interval_sec": [case.clip_start_sec, case.clip_end_sec]}
            for case in selected
        ],
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "answer_model_profile": model_profile,
        "answer_model_spec": model_spec,
        "instrumentation": {
            "version": "cuda-memory-telemetry-v1",
            "empty_cache_after_call": True,
            "protocol_behavior_changed": False,
        },
    }
    atomic_write_json(args.output_root / "protocol_snapshot.json", protocol)
    if args.preflight_only:
        _run_frame_preflight(
            output_root=args.output_root, selected=selected, config=config, protocol=protocol,
        )
        print(json.dumps({
            "output_root": str(args.output_root), "frame_preflight": "passed",
            "selected_ids": [case.question_id for case in selected], "qwen_model_loads": 0,
        }, ensure_ascii=False))
        return 0

    cached: list[dict[str, Any]] = []
    pending: list[tuple[CanonicalCase, str]] = []
    for case in selected:
        for condition in CONDITIONS:
            checkpoint = _checkpoint_path(args.output_root, case, condition)
            if checkpoint.is_file():
                try:
                    record = load_json(checkpoint)
                except (json.JSONDecodeError, OSError):
                    record = None
                if valid_result(record, config_hash=config_hash, manifest_hash=manifest_hashes[case.task]):
                    cached.append(record)
                    continue
            pending.append((case, condition))
    if args.validate_only and pending:
        raise E1ProtocolError(f"Validation-only run has {len(pending)} missing checkpoints")
    if args.validate_only:
        # Validation must never overwrite timing/memory metadata from the real
        # model session with cache-only values.
        print(json.dumps({
            "output_root": str(args.output_root), "records": len(cached),
            "cache_hits": len(cached), "cache_misses": 0, "model_loads": 0,
            "validation_only": True,
        }, ensure_ascii=False))
        return 0

    model: LocalQwenVLEngineeringModel | None = None
    created: list[dict[str, Any]] = []
    model_load_time_s: float | None = None
    pre_model_load_memory: dict[str, Any] | None = None
    post_model_load_memory: dict[str, Any] | None = None
    halted_after_oom = False
    if pending:
        model_path = Path(model_spec["local_path"])
        model = LocalQwenVLEngineeringModel(
            model_path=model_path,
            model_id=str(model_spec["model_id"]),
            max_pixels=int(config["decoding"]["max_pixels"]),
            seed=int(config["decoding"]["seed"]),
            minimum_free_vram_gib=float(args.minimum_free_vram_gib),
        )
        model_load_time_s = model.model_load_time_s
        pre_model_load_memory = model.pre_model_load_memory
        post_model_load_memory = model.post_model_load_memory
        try:
            for case, condition in pending:
                record = _record(
                    case=case, condition=condition, config=config, config_hash=config_hash,
                    manifest_hash=manifest_hashes[case.task], model=model, run_id=run_id,
                    model_profile=model_profile,
                )
                atomic_write_json(_checkpoint_path(args.output_root, case, condition), record)
                created.append(record)
                if not record["success"] and not bool((record.get("oom_recovery") or {}).get("safe_to_continue")):
                    halted_after_oom = True
                    break
        finally:
            model.close()
            del model
            gc.collect()
    all_records = sorted(cached + created, key=lambda row: (row["task"], row["question_id"], row["condition"]))
    expected_records = len(selected) * len(CONDITIONS)
    if len(all_records) != expected_records and not halted_after_oom:
        raise E1ProtocolError(f"Incomplete dry-run: {len(all_records)}/{expected_records}")
    _write_aggregate(args.output_root, all_records, metadata={
        **protocol,
        "cache_hits": len(cached),
        "cache_misses": len(created),
        "model_loads": int(bool(created)),
        "model_load_time_s": model_load_time_s,
        "pre_model_load_memory": pre_model_load_memory,
        "post_model_load_memory": post_model_load_memory,
        "halted_after_unsafe_oom": halted_after_oom,
    })
    print(json.dumps({
        "output_root": str(args.output_root), "records": len(all_records),
        "cache_hits": len(cached), "cache_misses": len(created),
        "selected_ids": [case.question_id for case in selected],
        "config_hash": config_hash,
    }, ensure_ascii=False))
    return 2 if halted_after_oom else 0


def rebuild_summary_only(args: argparse.Namespace) -> int:
    """Repair/rebuild reports from completed atomic checkpoints without Qwen."""
    config = load_json(args.config)
    config_hash = stable_hash(config)
    cases, manifest_hashes = _load_all_cases(config, args.annotation_root)
    selected = select_one_video_dry_run_cases(cases)
    records: list[dict[str, Any]] = []
    for case in selected:
        for condition in CONDITIONS:
            checkpoint = _checkpoint_path(args.output_root, case, condition)
            if not checkpoint.is_file():
                raise E1ProtocolError(f"Missing checkpoint for summary rebuild: {checkpoint}")
            record = load_json(checkpoint)
            if not valid_result(record, config_hash=config_hash, manifest_hash=manifest_hashes[case.task]):
                raise E1ProtocolError(f"Invalid checkpoint for summary rebuild: {checkpoint}")
            records.append(record)
    first = min(records, key=lambda row: str(row["created_at"]))
    _write_aggregate(args.output_root, records, metadata={
        "run_id": first["run_id"],
        "created_at": first["created_at"],
        "config_path": str(args.config),
        "config_hash": config_hash,
        "config_file_sha256": sha256_file(args.config),
        "manifest_hashes": manifest_hashes,
        "instrumentation": {"version": "cuda-memory-telemetry-v1", "reconstructed": True},
        "model_loads": 1,
        "model_load_time_s": None,
        "cache_hits": 0,
        "cache_misses": len(records),
        "records_rebuilt_from_atomic_checkpoints": True,
    })
    print(json.dumps({"output_root": str(args.output_root), "summary_rebuilt": len(records)}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one-video QaEgo4D E1 engineering dry-run only")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true", help="Decode/validate canonical frames without loading Qwen")
    parser.add_argument("--validate-only", action="store_true", help="Verify cached dry-run determinism without loading Qwen")
    parser.add_argument("--memory-robustness", action="store_true", help="Write an isolated instrumentation-only rerun")
    parser.add_argument("--rebuild-summary-only", action="store_true", help="Rebuild reports from completed checkpoints without loading Qwen")
    parser.add_argument(
        "--model-profile", default="engineering_dry_run_only",
        choices=("engineering_dry_run_only", "formal_candidate"),
        help="Frozen config answer-model profile; formal_candidate is allowed only for bounded probes.",
    )
    parser.add_argument("--single-task", choices=TASKS, default="open")
    parser.add_argument("--single-question-id", help="Run exactly this one frozen manifest sample across all four conditions")
    parser.add_argument(
        "--minimum-free-vram-gib", type=float, default=7.0,
        help="Engineering preflight gate; does not alter the frozen inference protocol.",
    )
    args = parser.parse_args(argv)
    if args.memory_robustness and args.output_root == DEFAULT_OUTPUT:
        args.output_root = DEFAULT_OUTPUT / "memory_robustness_rerun_v1"
    if args.rebuild_summary_only:
        return rebuild_summary_only(args)
    return run(args)
