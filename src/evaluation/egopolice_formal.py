from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable


VIDEO_DURATION_BINS = ("le_300", "301_600", "601_1200", "gt_1200")
QUESTION_DURATION_CLASSES = ("1s", "10s", "60s")


def source_video_duration_bin(duration_sec: float) -> str:
    if duration_sec <= 0:
        raise ValueError("Validated source-video duration must be positive")
    if duration_sec <= 300:
        return "le_300"
    if duration_sec <= 600:
        return "301_600"
    if duration_sec <= 1200:
        return "601_1200"
    return "gt_1200"


def _accuracy_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(record["correct"]) for record in records)
    total = len(records)
    return {
        "correct": correct,
        "total": total,
        "n": total,
        "accuracy": correct / total if total else None,
        "accuracy_percent": 100 * correct / total if total else None,
        "distinct_videos": len({record["source_video_id"] for record in records}),
    }


def _mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.fmean(values) if values else None


def _duration_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["total_per_query_latency_sec"]) for record in records]
    return {
        "gt_interval_hit_at_8_rate": _mean(records, "gt_interval_hit_at_8"),
        "mean_nearest_gt_interval_distance_sec": _mean(
            records, "nearest_sample_distance_to_gt_interval_seconds"
        ),
        "mean_online_latency_sec": statistics.fmean(latencies) if latencies else None,
        "median_online_latency_sec": statistics.median(latencies) if latencies else None,
        "mean_frame_extraction_latency_sec": _mean(records, "frame_extraction_latency_sec"),
        "mean_inference_latency_sec": _mean(records, "qwen_inference_latency_sec"),
        "mean_text_input_tokens": _mean(records, "text_input_tokens"),
        "mean_visual_tokens_estimate": _mean(records, "visual_tokens_estimate"),
        "mean_model_facing_frames": _mean(records, "model_facing_frames"),
        "mean_model_calls": _mean(records, "model_calls"),
    }


def aggregate_formal_b0(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_video_length: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        question_class = str(record["question_duration_class"])
        duration_bin = str(record["source_video_duration_bin"])
        if question_class not in QUESTION_DURATION_CLASSES:
            raise ValueError(f"Invalid question duration class: {question_class}")
        if duration_bin not in VIDEO_DURATION_BINS:
            raise ValueError(f"Invalid source-video duration bin: {duration_bin}")
        by_question[question_class].append(record)
        by_video_length[duration_bin].append(record)
        cross[(duration_bin, question_class)].append(record)
    return {
        "overall": _accuracy_group(rows),
        "by_question_duration_class": {
            name: _accuracy_group(by_question[name]) for name in QUESTION_DURATION_CLASSES
        },
        "by_source_video_duration": {
            name: {
                **_accuracy_group(by_video_length[name]),
                "diagnostics": _duration_diagnostics(by_video_length[name]),
            }
            for name in VIDEO_DURATION_BINS
        },
        "source_video_duration_x_question_duration": {
            duration_bin: {
                question_class: _accuracy_group(cross[(duration_bin, question_class)])
                for question_class in QUESTION_DURATION_CLASSES
            }
            for duration_bin in VIDEO_DURATION_BINS
        },
    }
