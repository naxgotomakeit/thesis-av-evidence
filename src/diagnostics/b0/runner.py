from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.baselines.egopolice_b0.core import (
    BaselineInputError,
    build_mcq_prompt,
    evaluate_gt_exposure,
    extract_uniform_frames,
    load_mcq_cases,
    probe_duration,
    resolve_video_path,
    uniform_timestamps,
)
from src.baselines.egopolice_b0.model import Qwen25VL7BBaseline, preflight_environment
from src.baselines.egopolice_b0.runner import _allowed_video_ids


ROOT = Path(__file__).resolve().parents[3]
QUESTION_IDS = ("1s_972", "10s_973", "10s_974", "60s_386", "1s_973")
NUM_FRAMES = 8
MAX_PIXELS = 262144
DTYPE = "bfloat16"
QUANTIZATION = {
    "mode": "none",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
}
GENERATION = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 8, "use_cache": True}
FIXED_ZERO_COSTS = {
    "offline_reusable_preprocessing": 0,
    "reusable_index_size_bytes": 0,
    "embedding_computations": 0,
    "similarity_computations": 0,
    "caption_generation_calls": 0,
    "planner_calls": 0,
}


def _duration_class(question_id: str) -> str:
    duration_class = question_id.split("_", 1)[0]
    if duration_class not in {"1s", "10s", "60s"}:
        raise BaselineInputError(f"Cannot infer duration class from {question_id}")
    return duration_class


def resolve_cases(data_root: Path, question_ids: Sequence[str] = QUESTION_IDS) -> list[dict[str, Any]]:
    if tuple(question_ids) != QUESTION_IDS:
        raise BaselineInputError(f"This diagnostic requires exactly {list(QUESTION_IDS)}")
    cache: dict[str, list[dict[str, Any]]] = {}
    cases: list[dict[str, Any]] = []
    for question_id in question_ids:
        duration_class = _duration_class(question_id)
        if duration_class not in cache:
            cache[duration_class] = load_mcq_cases(data_root / f"mcq_{duration_class}.json")
        matches = [case for case in cache[duration_class] if case["question_id"] == question_id]
        if len(matches) != 1:
            raise BaselineInputError(f"Expected one row for {question_id}, found {len(matches)}")
        cases.append(matches[0])
    return cases


def _case_paths(
    data_root: Path, subset_manifest: Path, cases: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], Path]]:
    allowed = _allowed_video_ids(subset_manifest)
    video_root = data_root / "videos"
    resolved = []
    for case in cases:
        if allowed is not None and case["video_id"] not in allowed:
            raise BaselineInputError(f"Video is outside frozen subset: {case['video_id']}")
        path = resolve_video_path(video_root, case["video_relative_path"])
        if not path.is_file():
            raise BaselineInputError(f"Missing source video: {path}")
        resolved.append((case, path))
    return resolved


def deterministic_sampling_preflight(
    *, data_root: Path, subset_manifest: Path, ffprobe_path: str,
) -> dict[str, Any]:
    cases = resolve_cases(data_root)
    rows = []
    timestamps_by_video: dict[str, list[float]] = {}
    for case, path in _case_paths(data_root, subset_manifest, cases):
        duration = probe_duration(path, ffprobe_path)
        timestamps = uniform_timestamps(duration, NUM_FRAMES)
        previous = timestamps_by_video.setdefault(case["video_id"], timestamps)
        if timestamps != previous:
            raise BaselineInputError(f"Question-dependent timestamps detected: {case['video_id']}")
        rows.append({
            "question_id": case["question_id"],
            "source_video_id": case["video_id"],
            "full_source_video_duration_sec": duration,
            "sampled_frame_timestamps_sec": timestamps,
        })
    return {"passed": True, "question_independent_sampling_confirmed": True, "rows": rows}


def _conditional_accuracy(records: list[dict[str, Any]], hit: int) -> dict[str, Any]:
    group = [record for record in records if record["gt_interval_hit_at_8"] == hit]
    correct = sum(bool(record["correct"]) for record in group)
    return {
        "count": len(group),
        "correct": correct,
        "accuracy": correct / len(group) if group else None,
    }


def _interpretation(oracle_correct: bool, b0_correct: bool, hit: int) -> str:
    if not oracle_correct:
        return "answer-model/perception difficulty is a confound; do not use as clean evidence-selection failure"
    if b0_correct and hit == 0:
        return "possible guessing, indirect evidence, annotation-window limitation, or answer prior; flag for later case analysis"
    if not b0_correct and hit == 0:
        return "likely evidence sampling failure"
    if not b0_correct and hit == 1:
        return "evidence was temporally exposed but answer still failed; do not attribute purely to retrieval"
    return "both Oracle and B0 correct with GT interval exposed"


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["total_per_query_latency_sec"]) for record in records]
    frame_latencies = [float(record["frame_extraction_latency_sec"]) for record in records]
    inference_latencies = [float(record["qwen_inference_latency_sec"]) for record in records]
    correct = sum(bool(record["correct"]) for record in records)
    hits = sum(int(record["gt_interval_hit_at_8"]) for record in records)
    return {
        "completed_questions": len(records),
        "overall_correct": correct,
        "overall_accuracy": correct / len(records),
        "gt_interval_hit_at_8_count": hits,
        "gt_interval_hit_at_8_rate": hits / len(records),
        "accuracy_conditional_on_gt_interval_hit_at_8_1": _conditional_accuracy(records, 1),
        "accuracy_conditional_on_gt_interval_hit_at_8_0": _conditional_accuracy(records, 0),
        "mean_per_query_latency_sec": statistics.fmean(latencies),
        "median_per_query_latency_sec": statistics.median(latencies),
        "mean_frame_extraction_latency_sec": statistics.fmean(frame_latencies),
        "mean_inference_latency_sec": statistics.fmean(inference_latencies),
        "total_model_calls": sum(int(record["model_calls"]) for record in records),
        "frames_per_question": NUM_FRAMES,
    }


def _load_oracle_correctness(path: Path) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record["question_id"]): bool(record["correct"])
        for record in payload["records"]
        if record.get("status") == "completed"
    }


def run_b0_batch(
    *, data_root: Path, model_path: Path, subset_manifest: Path,
    oracle_result_path: Path, output_path: Path, ffmpeg_path: str, ffprobe_path: str,
) -> dict[str, Any]:
    import torch

    cases = resolve_cases(data_root)
    case_paths = _case_paths(data_root, subset_manifest, cases)
    environment = preflight_environment(model_path, dtype=DTYPE, quantization_mode="none")
    if not environment["passed"]:
        raise BaselineInputError(f"B0 preflight failed: {environment['errors']}")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = None
    records: list[dict[str, Any]] = []
    timestamps_by_video: dict[str, list[float]] = {}
    batch_started = time.perf_counter()
    try:
        model = Qwen25VL7BBaseline(
            model_path=model_path,
            max_pixels=MAX_PIXELS,
            seed=0,
            dtype=DTYPE,
            quantization=QUANTIZATION,
        )
        model_load_latency = model.model_load_latency_sec
        for case, video_path in case_paths:
            query_started = time.perf_counter()
            images = []
            try:
                # Sampling is complete before the GT interval is passed to any
                # metric code. The sampler receives only video, duration, and
                # the frozen frame/pixel budget.
                images, timestamps, duration, frame_latency = extract_uniform_frames(
                    video_path=video_path,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    num_frames=NUM_FRAMES,
                    max_pixels=MAX_PIXELS,
                )
                previous = timestamps_by_video.setdefault(case["video_id"], timestamps)
                if timestamps != previous:
                    raise BaselineInputError(f"Question-dependent timestamps: {case['video_id']}")
                prompt = build_mcq_prompt(case["question"], case["options"], len(images))
                inference = model.infer(images=images, prompt=prompt, generation=GENERATION)
                prediction = int(inference["prediction_index"])

                # GT is consulted only now, after frame selection and inference.
                exposure = evaluate_gt_exposure(timestamps, case["ignored_annotation_interval"])
                record = {
                    "status": "completed",
                    "baseline_id": "egopolice_b0_uniform_qwen2_5_vl_7b_v0_1",
                    "question_id": case["question_id"],
                    "source_video_id": case["video_id"],
                    "source_video_path": str(video_path),
                    "question": case["question"],
                    "options": case["options"],
                    "prediction_index": prediction,
                    "prediction_text": case["options"][prediction],
                    "ground_truth_index": case["ground_truth_index"],
                    "ground_truth_text": case["ground_truth_text"],
                    "correct": prediction == case["ground_truth_index"],
                    "raw_model_output": inference["raw_output"],
                    "full_source_video_duration_sec": duration,
                    "sampled_frame_timestamps_sec": timestamps,
                    "sampling_rule": "8_equal_full_video_bins_midpoint_of_each_bin",
                    "sampling_question_independent": True,
                    "model_facing_frames": len(images),
                    **exposure,
                    "frame_extraction_latency_sec": frame_latency,
                    "preprocessing_latency_sec": inference["preprocessing_latency_sec"],
                    "qwen_inference_latency_sec": inference["inference_latency_sec"],
                    "total_per_query_latency_sec": time.perf_counter() - query_started,
                    "model_calls": 1,
                    "text_input_tokens": inference["text_token_count"],
                    "visual_tokens_estimate": inference["visual_token_count"],
                    "visual_tokens_are_estimated": True,
                    "visual_token_estimation_method": "total_input_tokens_minus_text_tokens",
                    "total_input_tokens": inference["total_input_token_count"],
                    "output_tokens": inference["output_token_count"],
                    "peak_gpu_allocated_memory_bytes": inference["peak_gpu_memory_bytes"],
                    "dtype": DTYPE,
                    "quantization_mode": "none",
                    "actual_model_loading_mode": model.actual_model_loading_mode,
                    "oom": False,
                    "errors": [],
                    "warnings": [],
                    "fixed_zero_costs": dict(FIXED_ZERO_COSTS),
                }
                records.append(record)
            finally:
                for image in images:
                    image.close()
        if len(records) != 5:
            raise BaselineInputError(f"Expected 5 completed B0 questions, got {len(records)}")

        oracle = _load_oracle_correctness(oracle_result_path)
        paired = []
        for record in records:
            question_id = record["question_id"]
            if question_id not in oracle:
                raise BaselineInputError(f"Missing Oracle result for {question_id}")
            paired.append({
                "question_id": question_id,
                "oracle_correct": oracle[question_id],
                "b0_correct": record["correct"],
                "gt_interval_hit_at_8": record["gt_interval_hit_at_8"],
                "interpretation": _interpretation(
                    oracle[question_id], record["correct"], record["gt_interval_hit_at_8"]
                ),
            })
        result = {
            "diagnostic_id": "egopolice-b0-long-video-five-v1",
            "model_path": str(model_path),
            "dtype": DTYPE,
            "quantization_mode": "none",
            "actual_model_loading_mode": model.actual_model_loading_mode,
            "one_model_load": True,
            "model_load_latency_sec": model_load_latency,
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_gpu_allocated_memory_bytes": max(
                record["peak_gpu_allocated_memory_bytes"] for record in records
            ),
            "batch_wall_latency_sec": time.perf_counter() - batch_started,
            "question_independent_sampling_confirmed": True,
            "records": records,
            "aggregate": _aggregate(records),
            "paired_oracle_comparison": paired,
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled five-question long-video B0 smoke")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--subset-manifest", type=Path,
        default=ROOT / "config/data/egopolice_50videos.json",
    )
    parser.add_argument(
        "--oracle-result", type=Path,
        default=ROOT / "outputs/diagnostics/oracle/egopolice_oracle_5.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/diagnostics/b0/egopolice_b0_5.json",
    )
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--ffprobe-path", default="ffprobe")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = deterministic_sampling_preflight(
        data_root=args.data_root,
        subset_manifest=args.subset_manifest,
        ffprobe_path=args.ffprobe_path,
    )
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    result = run_b0_batch(
        data_root=args.data_root,
        model_path=args.model_path,
        subset_manifest=args.subset_manifest,
        oracle_result_path=args.oracle_result,
        output_path=args.output,
        ffmpeg_path=args.ffmpeg_path,
        ffprobe_path=args.ffprobe_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
