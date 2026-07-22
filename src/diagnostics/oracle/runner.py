from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .core import (
    ORACLE_PROMPT_VERSION,
    OracleCase,
    OracleInputError,
    build_oracle_messages,
    load_oracle_case,
    normalize_single_video_fps,
    parse_oracle_prediction,
    select_materializable_question_ids,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DURATION_PLAN = ("1s", "10s", "10s", "60s", "1s")
MAX_PIXELS = 262144
MAX_NEW_TOKENS = 8


def _duration_class(question_id: str) -> str:
    prefix = question_id.split("_", 1)[0]
    if prefix not in {"1s", "10s", "60s"}:
        raise OracleInputError(f"Cannot infer duration class from {question_id}")
    return prefix


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OracleInputError(completed.stderr.strip() or f"ffprobe failed: {path}")
    return float(completed.stdout.strip())


def resolve_cases(
    data_root: Path, *, question_ids: Sequence[str], duration_plan: Sequence[str],
) -> list[OracleCase]:
    selected = (
        [(_duration_class(question_id), question_id) for question_id in question_ids]
        if question_ids
        else select_materializable_question_ids(data_root, duration_plan)
    )
    if len(selected) != 5:
        raise OracleInputError(f"Oracle five-case diagnostic requires exactly 5 questions, got {len(selected)}")
    cases = [load_oracle_case(data_root, duration_class, question_id) for duration_class, question_id in selected]
    counts = defaultdict(int)
    for case in cases:
        counts[case.duration_class] += 1
        source_duration = _probe_duration(case.source_video_path)
        if case.end_sec > source_duration + 1e-6:
            raise OracleInputError(
                f"Oracle interval exceeds source duration for {case.question_id}: "
                f"{case.end_sec} > {source_duration}"
            )
    if counts["1s"] < 1 or counts["10s"] < 2 or counts["60s"] < 1:
        raise OracleInputError(f"Required duration coverage not met: {dict(counts)}")
    return cases


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    by_duration: dict[str, dict[str, Any]] = {}
    for duration_class in ("1s", "10s", "60s"):
        group = [record for record in completed if record["duration_class"] == duration_class]
        if group:
            correct = sum(bool(record["correct"]) for record in group)
            by_duration[duration_class] = {
                "count": len(group),
                "correct": correct,
                "accuracy": correct / len(group),
            }
    latencies = [float(record["total_per_query_latency_sec"]) for record in completed]
    correct = sum(bool(record["correct"]) for record in completed)
    return {
        "completed_questions": len(completed),
        "overall_correct": correct,
        "overall_accuracy": correct / len(completed) if completed else None,
        "accuracy_by_duration_class": by_duration,
        "mean_per_query_latency_sec": statistics.fmean(latencies) if latencies else None,
        "median_per_query_latency_sec": statistics.median(latencies) if latencies else None,
        "total_model_calls": sum(int(record["model_calls"]) for record in records),
    }


def run_oracle_batch(
    *, cases: list[OracleCase], model_path: Path, output_path: Path,
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise OracleInputError("CUDA is unavailable")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise OracleInputError("Current GPU does not support BF16")
    if not model_path.is_dir():
        raise OracleInputError(f"Model path does not exist: {model_path}")

    torch.cuda.empty_cache()
    # PyTorch 2.13 rejects an integer argument here on this host. The no-arg
    # current-device API is the portable path after set_device(0).
    torch.cuda.reset_peak_memory_stats()
    free_before, total_vram = torch.cuda.mem_get_info()
    records: list[dict[str, Any]] = []
    model = processor = None
    batch_started = time.perf_counter()
    model_load_latency = 0.0
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, max_pixels=MAX_PIXELS
        )
        load_started = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        model.eval()
        torch.cuda.synchronize()
        model_load_latency = time.perf_counter() - load_started

        for case in cases:
            record: dict[str, Any] = {
                "question_id": case.question_id,
                "duration_class": case.duration_class,
                "source_video_path": str(case.source_video_path),
                "oracle_interval_sec": [case.start_sec, case.end_sec],
                "question": case.question,
                "options": list(case.options),
                "ground_truth_index": case.ground_truth_index,
                "ground_truth_text": case.ground_truth_text,
                "dtype": "bfloat16",
                "quantization_mode": "none",
                "actual_model_loading_mode": "bfloat16_unquantized",
                "model_calls": 0,
                "status": "started",
            }
            query_started = time.perf_counter()
            try:
                prep_started = time.perf_counter()
                messages = build_oracle_messages(case, max_pixels=MAX_PIXELS)
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                original_fps_value = video_kwargs.get("fps")
                video_kwargs = normalize_single_video_fps(video_inputs, video_kwargs)
                inputs = processor(
                    text=[rendered],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                )
                text_tokens = len(
                    processor.tokenizer(rendered, add_special_tokens=False)["input_ids"]
                )
                total_input_tokens = int(inputs.input_ids.shape[1])
                video_tensor = (
                    video_inputs[0][0]
                    if isinstance(video_inputs[0], tuple)
                    else video_inputs[0]
                )
                inputs = inputs.to("cuda")
                torch.cuda.synchronize()
                record.update(
                    {
                        "decoded_model_facing_frames": int(video_tensor.shape[0]),
                        "effective_video_fps": float(video_kwargs["fps"]),
                        "fps_scalar_compatibility_applied": isinstance(
                            original_fps_value, (list, tuple)
                        ),
                        "clip_preprocessing_latency_sec": time.perf_counter() - prep_started,
                        "text_token_count": text_tokens,
                        "visual_token_count_estimate": max(0, total_input_tokens - text_tokens),
                        "visual_token_count_is_estimated": True,
                        "visual_token_estimation_method": "total_input_tokens_minus_text_tokens",
                        "total_input_token_count": total_input_tokens,
                    }
                )

                inference_started = time.perf_counter()
                record["model_calls"] = 1
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                record["inference_latency_sec"] = time.perf_counter() - inference_started
                generated_only = generated[:, total_input_tokens:]
                raw = processor.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                prediction = parse_oracle_prediction(raw)
                record.update(
                    {
                        "raw_model_output": raw,
                        "output_token_count": int(generated_only.shape[1]),
                        "prediction_index": prediction,
                        "prediction_text": case.options[prediction],
                        "correct": prediction == case.ground_truth_index,
                        "total_per_query_latency_sec": time.perf_counter() - query_started,
                        "status": "completed",
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": repr(exc),
                        "oom": isinstance(exc, torch.cuda.OutOfMemoryError)
                        or "out of memory" in str(exc).lower(),
                        "total_per_query_latency_sec": time.perf_counter() - query_started,
                    }
                )
                records.append(record)
                break
            records.append(record)
            del inputs, generated, generated_only, image_inputs, video_inputs

        summary = {
            "diagnostic_id": "egopolice-short-clip-oracle-v1",
            "prompt_version": ORACLE_PROMPT_VERSION,
            "model_path": str(model_path),
            "dtype": "bfloat16",
            "quantization_mode": "none",
            "actual_model_loading_mode": "bfloat16_unquantized",
            "model_load_latency_sec": model_load_latency,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_free_before_bytes": int(free_before),
            "gpu_total_bytes": int(total_vram),
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "batch_wall_latency_sec": time.perf_counter() - batch_started,
            "records": records,
            "aggregate": _aggregate(records),
        }
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five ground-truth short-clip EgoPolice Oracle diagnostics")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--duration-plan", nargs="+", default=list(DEFAULT_DURATION_PLAN))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/diagnostics/oracle/egopolice_oracle_5.json",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = resolve_cases(
        args.data_root,
        question_ids=args.question_id,
        duration_plan=args.duration_plan,
    )
    selection = [
        {
            "question_id": case.question_id,
            "duration_class": case.duration_class,
            "source_video_path": str(case.source_video_path),
            "interval_sec": [case.start_sec, case.end_sec],
        }
        for case in cases
    ]
    if args.preflight_only:
        print(json.dumps({"preflight_only": True, "selection": selection}, ensure_ascii=False, indent=2))
        return 0

    # qwen-vl-utils reads this at import time inside run_oracle_batch.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    summary = run_oracle_batch(cases=cases, model_path=args.model_path, output_path=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["aggregate"]["completed_questions"] == 5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
