"""Normalized segmentation schema and method-independent metrics."""

from __future__ import annotations

import statistics
from typing import Any, Iterable

import numpy as np


METHOD_NAMES = ("current", "comet_style_dinov2", "kts_dinov2")


def make_segmentation(
    *,
    video_id: str,
    method: str,
    video_duration: float,
    boundaries: Iterable[float],
    frame_timestamps: np.ndarray,
) -> dict[str, Any]:
    """Build normalized, contiguous segments from internal boundary timestamps."""
    if method not in METHOD_NAMES:
        raise ValueError(f"Unknown segmentation method: {method}")
    duration = float(video_duration)
    interior = sorted({float(value) for value in boundaries if 0.0 < float(value) < duration})
    edges = [0.0, *interior, duration]
    timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    segments = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        center = (start + end) / 2.0
        available = timestamps[(timestamps >= start) & (timestamps < end)]
        representative = center if len(available) == 0 else float(available[np.argmin(np.abs(available - center))])
        segments.append(
            {
                "segment_id": f"{method}_{index:04d}",
                "start": start,
                "end": end,
                "duration": end - start,
                "representative_frame_timestamp": representative,
            }
        )
    record = {
        "video_id": str(video_id),
        "method": method,
        "video_duration": duration,
        "segments": segments,
    }
    errors = validate_segmentation(record)
    if errors:
        raise ValueError(f"Invalid normalized segmentation: {errors}")
    return record


def validate_segmentation(record: dict[str, Any], tolerance: float = 1e-6) -> list[str]:
    """Return normalized-schema and temporal-coverage violations."""
    errors: list[str] = []
    if record.get("method") not in METHOD_NAMES:
        errors.append("unknown_method")
    segments = record.get("segments") or []
    duration = float(record.get("video_duration", 0.0))
    if not segments:
        return [*errors, "empty_segments"]
    if abs(float(segments[0]["start"])) > tolerance:
        errors.append("first_segment_not_at_video_start")
    if abs(float(segments[-1]["end"]) - duration) > tolerance:
        errors.append("last_segment_not_at_video_end")
    identifiers: set[str] = set()
    for index, segment in enumerate(segments):
        identifier = str(segment.get("segment_id"))
        if identifier in identifiers:
            errors.append(f"duplicate_segment_id:{identifier}")
        identifiers.add(identifier)
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            errors.append(f"non_positive_duration:{identifier}")
        if abs(float(segment.get("duration", end - start)) - (end - start)) > tolerance:
            errors.append(f"duration_mismatch:{identifier}")
        if index:
            previous_end = float(segments[index - 1]["end"])
            if start > previous_end + tolerance:
                errors.append(f"gap_before:{identifier}")
            if start < previous_end - tolerance:
                errors.append(f"overlap_before:{identifier}")
    return errors


def segmentation_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Calculate descriptive under/over-segmentation statistics."""
    durations = [float(item["duration"]) for item in record["segments"]]
    total = float(record["video_duration"])
    mean = statistics.fmean(durations)
    short2 = sum(value < 2.0 for value in durations)
    short4 = sum(value < 4.0 for value in durations)
    coverage = sum(durations)
    count = len(durations)
    return {
        "video_id": record["video_id"],
        "method": record["method"],
        "number_of_segments": count,
        "mean_segment_duration": mean,
        "median_segment_duration": statistics.median(durations),
        "max_segment_duration": max(durations),
        "largest_region_ratio": max(durations) / total if total else None,
        "short_segment_count_lt_2s": short2,
        "short_segment_fraction_lt_2s": short2 / count,
        "short_segment_count_lt_4s": short4,
        "short_segment_fraction_lt_4s": short4 / count,
        "boundaries_per_minute": (count - 1) / (total / 60.0) if total else None,
        "duration_coefficient_of_variation": (
            statistics.pstdev(durations) / mean if mean else 0.0
        ),
        "coverage_duration": coverage,
        "coverage_ratio": coverage / total if total else None,
        "coverage_valid": not validate_segmentation(record),
    }
