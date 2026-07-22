from __future__ import annotations

import gc
import hashlib
import json
import statistics
import subprocess
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.baselines.egopolice_b0.core import (
    BaselineInputError,
    build_mcq_prompt,
    evaluate_gt_exposure,
    extract_uniform_frames,
    probe_duration,
    uniform_timestamps,
)
from src.baselines.egopolice_b0.model import Qwen25VL7BBaseline, preflight_environment


ROOT = Path(__file__).resolve().parents[3]
VIDEO_ID = "pasadena/YKI08"
EXPECTED_QUESTION_MANIFEST_SHA256 = "fca734b9764ed132483ba3858db34243b50384ba2f1e115197c15762adcb23a5"
EXPECTED_QUESTION_COUNT = 5
NON_FORMAL_LABEL = "PRE-FLIGHT / NON-FORMAL — pipeline validation only"
MINIMUM_ACTIVATION_HEADROOM_BYTES = 4 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "num_frames": 8,
        "max_pixels": 262144,
        "seed": 0,
        "dtype": "bfloat16",
        "prompt_version": "egopolice-b0-mcq-v1",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise BaselineInputError(f"Frozen B0 config mismatch for {key}: {config.get(key)!r}")
    if config.get("quantization", {}).get("mode") != "none":
        raise BaselineInputError("YKI08 B0 preflight requires no quantization")
    generation = config.get("generation", {})
    if generation.get("do_sample") is not False or generation.get("temperature") != 0.0:
        raise BaselineInputError("Frozen B0 deterministic decoding config changed")
    if int(generation.get("max_new_tokens", -1)) != 8 or generation.get("use_cache") is not True:
        raise BaselineInputError("Frozen B0 generation budget changed")
    return config


def load_yki08_questions(path: Path) -> tuple[list[dict[str, Any]], str]:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_QUESTION_MANIFEST_SHA256:
        raise BaselineInputError(
            f"Frozen question manifest hash mismatch: {actual_sha} != {EXPECTED_QUESTION_MANIFEST_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = [row for row in payload.get("questions", []) if row.get("video_id") == VIDEO_ID]
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise BaselineInputError(
            f"Expected {EXPECTED_QUESTION_COUNT} frozen YKI08 questions, found {len(questions)}"
        )
    ids = [str(row.get("question_id")) for row in questions]
    if len(ids) != len(set(ids)):
        raise BaselineInputError("Duplicate YKI08 question IDs")
    expected_counts = {"1s": 2, "10s": 2, "60s": 1}
    actual_counts: dict[str, int] = defaultdict(int)
    for row in questions:
        duration_class = str(row.get("duration_class"))
        actual_counts[duration_class] += 1
        options = row.get("options")
        gt = row.get("ground_truth_index")
        interval = row.get("gt_interval_sec")
        if not isinstance(options, list) or len(options) != 5:
            raise BaselineInputError(f"Question {row.get('question_id')} does not have five options")
        if not isinstance(gt, int) or not 0 <= gt <= 4:
            raise BaselineInputError(f"Question {row.get('question_id')} has invalid GT")
        if not isinstance(interval, list) or len(interval) != 2:
            raise BaselineInputError(f"Question {row.get('question_id')} has invalid interval")
    if dict(actual_counts) != expected_counts:
        raise BaselineInputError(f"Unexpected YKI08 duration distribution: {dict(actual_counts)}")
    return questions, actual_sha


def _video_path(data_root: Path) -> Path:
    path = data_root / "videos" / "pasadena" / "YKI08.mp4"
    if not path.is_file():
        raise BaselineInputError(f"Missing formal YKI08 source video: {path}")
    return path


def probe_video(path: Path, ffprobe_path: str) -> dict[str, Any]:
    command = [
        ffprobe_path, "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    if completed.returncode != 0:
        raise BaselineInputError(completed.stderr.strip() or "ffprobe failed")
    payload = json.loads(completed.stdout)
    video_streams = [row for row in payload.get("streams", []) if row.get("codec_type") == "video"]
    if not video_streams or float(payload.get("format", {}).get("duration", 0.0)) <= 0:
        raise BaselineInputError("YKI08 ffprobe validation failed")
    return payload


def prepare_preflight(
    *, data_root: Path, model_path: Path, config_path: Path,
    question_manifest_path: Path, ffprobe_path: str,
) -> dict[str, Any]:
    import torch

    config = load_frozen_config(config_path)
    questions, manifest_sha = load_yki08_questions(question_manifest_path)
    video_path = _video_path(data_root)
    ffprobe = probe_video(video_path, ffprobe_path)
    duration = probe_duration(video_path, ffprobe_path)
    timestamps_first = uniform_timestamps(duration, int(config["num_frames"]))
    timestamps_second = uniform_timestamps(duration, int(config["num_frames"]))
    if timestamps_first != timestamps_second or len(timestamps_first) != 8:
        raise BaselineInputError("B0 midpoint timestamps are not deterministic")
    per_question = {row["question_id"]: list(timestamps_first) for row in questions}
    if len({tuple(value) for value in per_question.values()}) != 1:
        raise BaselineInputError("Question-dependent B0 timestamps detected")

    environment = preflight_environment(
        model_path,
        dtype=str(config["dtype"]),
        quantization_mode=str(config["quantization"]["mode"]),
    )
    if not environment["passed"]:
        raise BaselineInputError(f"Qwen BF16 environment preflight failed: {environment['errors']}")
    required = int(environment["checkpoint_weight_bytes"]) + MINIMUM_ACTIVATION_HEADROOM_BYTES
    free = int(environment["free_vram_bytes"])
    if free < required:
        raise BaselineInputError(
            f"Insufficient free VRAM for guarded BF16 preflight: free={free}, required={required}"
        )
    return {
        "label": NON_FORMAL_LABEL,
        "formal_result": False,
        "video_id": VIDEO_ID,
        "video_path": str(video_path),
        "video_ffprobe": ffprobe,
        "full_source_duration_sec": duration,
        "question_manifest_path": str(question_manifest_path),
        "question_manifest_sha256": manifest_sha,
        "question_manifest_hash_unchanged": True,
        "question_count": len(questions),
        "question_ids": [row["question_id"] for row in questions],
        "frozen_b0_config_path": str(config_path),
        "frozen_b0_config": config,
        "sampled_frame_timestamps_sec": timestamps_first,
        "sampling_rule": "full_duration_8_equal_bins_exact_midpoint",
        "sampling_deterministic": True,
        "sampling_question_independent": True,
        "per_question_timestamp_sha256": hashlib.sha256(
            json.dumps(per_question, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "free_vram_bytes": free,
            "total_vram_bytes": int(environment["total_vram_bytes"]),
            "checkpoint_weight_bytes": int(environment["checkpoint_weight_bytes"]),
            "guarded_required_bytes": required,
            "guarded_preflight_passed": True,
        },
        "environment": environment,
    }


def _group_accuracy(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in records)
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else None,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_class = {}
    for duration_class in ("1s", "10s", "60s"):
        by_class[duration_class] = _group_accuracy(
            [row for row in records if row["question_duration_class"] == duration_class]
        )
    latencies = [float(row["total_per_query_latency_sec"]) for row in records]
    hits = sum(int(row["gt_interval_hit_at_8"]) for row in records)
    return {
        "overall": _group_accuracy(records),
        "accuracy_by_question_duration_class": by_class,
        "gt_interval_hit_at_8": {
            "hits": hits,
            "total": len(records),
            "rate": hits / len(records),
        },
        "mean_per_query_latency_sec": statistics.fmean(latencies),
        "median_per_query_latency_sec": statistics.median(latencies),
        "mean_frame_extraction_latency_sec": statistics.fmean(
            float(row["frame_extraction_latency_sec"]) for row in records
        ),
        "mean_preprocessing_latency_sec": statistics.fmean(
            float(row["preprocessing_latency_sec"]) for row in records
        ),
        "mean_inference_latency_sec": statistics.fmean(
            float(row["qwen_inference_latency_sec"]) for row in records
        ),
        "total_model_calls": sum(int(row["model_calls"]) for row in records),
        "frames_per_question": 8,
    }


def run(
    *, data_root: Path, model_path: Path, config_path: Path,
    question_manifest_path: Path, output_dir: Path,
    ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe",
) -> dict[str, Any]:
    import torch

    preflight = prepare_preflight(
        data_root=data_root,
        model_path=model_path,
        config_path=config_path,
        question_manifest_path=question_manifest_path,
        ffprobe_path=ffprobe_path,
    )
    questions, _ = load_yki08_questions(question_manifest_path)
    config = preflight["frozen_b0_config"]
    video_path = Path(preflight["video_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight_checks.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = None
    records: list[dict[str, Any]] = []
    runtime_warnings: list[str] = []
    batch_started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = Qwen25VL7BBaseline(
                model_path=model_path,
                max_pixels=int(config["max_pixels"]),
                seed=int(config["seed"]),
                dtype=str(config["dtype"]),
                quantization=dict(config["quantization"]),
            )
            runtime_warnings.extend(f"{row.category.__name__}: {row.message}" for row in caught)
        model_load_sec = float(model.model_load_latency_sec)
        canonical_timestamps = list(preflight["sampled_frame_timestamps_sec"])

        for question in questions:
            query_started = time.perf_counter()
            images = []
            try:
                # This receives no question, GT interval, or metadata. Sampling
                # is solely full duration -> eight equal-bin midpoint frames.
                images, timestamps, duration, frame_latency = extract_uniform_frames(
                    video_path=video_path,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    num_frames=int(config["num_frames"]),
                    max_pixels=int(config["max_pixels"]),
                )
                if timestamps != canonical_timestamps or len(images) != 8:
                    raise BaselineInputError("Runtime sampling deviated from deterministic B0 preflight")
                prompt = build_mcq_prompt(
                    str(question["question"]), list(question["options"]), len(images)
                )
                inference = model.infer(
                    images=images,
                    prompt=prompt,
                    generation=dict(config["generation"]),
                )
                prediction = int(inference["prediction_index"])

                # GT is consulted only after timestamps are fixed and the one
                # Qwen call has completed.
                exposure = evaluate_gt_exposure(timestamps, list(question["gt_interval_sec"]))
                records.append({
                    "status": "completed",
                    "label": NON_FORMAL_LABEL,
                    "formal_result": False,
                    "baseline_id": config["baseline_id"],
                    "question_id": question["question_id"],
                    "source_video_id": VIDEO_ID,
                    "source_video_path": str(video_path),
                    "question_duration_class": question["duration_class"],
                    "question": question["question"],
                    "options": question["options"],
                    "prediction_index": prediction,
                    "prediction_text": question["options"][prediction],
                    "ground_truth_index": question["ground_truth_index"],
                    "ground_truth_text": question["ground_truth_text"],
                    "correct": prediction == int(question["ground_truth_index"]),
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
                    "dtype": config["dtype"],
                    "quantization_mode": config["quantization"]["mode"],
                    "actual_model_loading_mode": model.actual_model_loading_mode,
                    "model_calls_contract": "exactly_one",
                    "fixed_zero_costs": {
                        "offline_reusable_preprocessing": 0,
                        "reusable_index_size_bytes": 0,
                        "embedding_computations": 0,
                        "similarity_computations": 0,
                        "caption_generation_calls": 0,
                        "planner_calls": 0,
                    },
                })
            finally:
                for image in images:
                    image.close()

        if len(records) != EXPECTED_QUESTION_COUNT:
            raise BaselineInputError(
                f"Expected {EXPECTED_QUESTION_COUNT} completed questions, got {len(records)}"
            )
        result = {
            "schema_version": "egopolice-b0-yki08-preflight-v1",
            "run_id": "B0-preflight-YKI08-nonformal-v1",
            "label": NON_FORMAL_LABEL,
            "formal_result": False,
            "formal_video_readiness": {"ready": 1, "total": 20},
            "limitation": "Only 1/20 formal videos is available; these metrics are not formal B0 results.",
            "model_path": str(model_path),
            "dtype": config["dtype"],
            "quantization_mode": config["quantization"]["mode"],
            "actual_model_loading_mode": model.actual_model_loading_mode,
            "deterministic_decoding": True,
            "one_model_load": True,
            "model_load_latency_sec": model_load_sec,
            "model_calls_per_question": 1,
            "model_facing_frames_per_question": 8,
            "preflight": preflight,
            "records": records,
            "aggregate": aggregate(records),
            "peak_gpu_allocated_memory_bytes": max(
                int(row["peak_gpu_allocated_memory_bytes"]) for row in records
            ),
            "batch_wall_latency_sec": time.perf_counter() - batch_started,
            "warnings": runtime_warnings,
            "errors": [],
        }
        (output_dir / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

