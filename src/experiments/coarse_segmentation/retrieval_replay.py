"""Controlled CLIP retrieval replay over alternative temporal boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        return values / max(float(np.linalg.norm(values)), 1e-12)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def pool_clip_regions(
    segmentation: dict[str, Any],
    frame_embeddings: np.ndarray,
    frame_timestamps: np.ndarray,
) -> np.ndarray:
    """Pool identical CLIP frame features for every method's temporal regions."""
    frames = normalize(frame_embeddings)
    timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    pooled = []
    for index, segment in enumerate(segmentation["segments"]):
        start, end = float(segment["start"]), float(segment["end"])
        mask = (timestamps >= start) & (timestamps < end)
        if index == len(segmentation["segments"]) - 1:
            mask = (timestamps >= start) & (timestamps <= end)
        if not np.any(mask):
            center = (start + end) / 2.0
            mask[np.argmin(np.abs(timestamps - center))] = True
        mean = frames[mask].mean(axis=0)
        pooled.append(normalize(mean))
    return np.stack(pooled).astype(np.float32)


def union_duration(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    current_end = float("-inf")
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, current_end))
        current_end = max(current_end, end)
    return total


def replay(
    segmentation: dict[str, Any],
    region_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    """Rank alternative regions with one shared normalized CLIP text query."""
    query = normalize(query_embedding)
    regions = normalize(region_embeddings)
    scores = regions @ query
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    ranked = []
    for rank, index in enumerate(order, start=1):
        segment = segmentation["segments"][index]
        ranked.append(
            {
                "rank": rank,
                "segment_id": segment["segment_id"],
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "score": float(scores[index]),
            }
        )
    selected = ranked[: min(top_k, len(ranked))]
    intervals = [(float(item["start"]), float(item["end"])) for item in selected]
    unique = union_duration(intervals)
    raw = sum(end - start for start, end in intervals)
    duration = float(segmentation["video_duration"])
    return {
        "video_id": segmentation["video_id"],
        "method": segmentation["method"],
        "top_k": top_k,
        "ranked_regions": ranked,
        "selected_top_k": selected,
        "total_raw_selected_duration": raw,
        "unique_temporal_duration": unique,
        "refinement_search_space_seconds": unique,
        "refinement_search_space_ratio": unique / duration if duration else None,
        "top_k_overlap_redundancy_seconds": raw - unique,
    }
