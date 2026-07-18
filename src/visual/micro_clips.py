from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def micro_clip_ranges(frame_count: int, window: int = 4, stride: int = 2, duration: float | None = None) -> list[tuple[int, int]]:
    """Return half-open frame ranges, including a final short clip when necessary."""
    if frame_count < 0 or window <= 0 or stride <= 0:
        raise ValueError("frame_count must be non-negative; window and stride must be positive")
    if frame_count == 0:
        return []
    if frame_count <= window:
        return [(0, frame_count)]
    ranges = [(start, start + window) for start in range(0, frame_count - window + 1, stride)]
    if ranges[-1][1] < frame_count:
        ranges.append((ranges[-1][0] + stride, frame_count))
    elif duration is not None and duration > ranges[-1][1]:
        ranges.append((ranges[-1][0] + stride, frame_count))
    return ranges


def representative_index(embeddings: np.ndarray, start: int, end: int) -> tuple[int, np.ndarray]:
    rows = normalize_rows(embeddings[start:end])
    if not len(rows):
        raise ValueError("a micro-clip must contain at least one frame")
    pooled = rows.mean(axis=0)
    pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
    return start + int(np.argmax(rows @ pooled)), pooled.astype(np.float32)


def motion_magnitude(frame_paths: list[Path]) -> float:
    """Mean adjacent-frame absolute grayscale change in [0, 1]; diagnostic only."""
    if len(frame_paths) < 2:
        return 0.0
    images = [np.asarray(Image.open(p).convert("L").resize((160, 90)), dtype=np.float32) / 255.0
              for p in frame_paths]
    return float(np.mean([np.mean(np.abs(b - a)) for a, b in zip(images, images[1:])]))


def build_micro_clips(
    embeddings: np.ndarray, frame_paths: list[Path], duration: float,
    window: int = 4, stride: int = 2,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if len(embeddings) != len(frame_paths):
        raise ValueError(f"frame/embedding count mismatch: {len(frame_paths)} != {len(embeddings)}")
    rows: list[dict[str, Any]] = []
    pooled: list[np.ndarray] = []
    ranges = micro_clip_ranges(len(frame_paths), window, stride, duration)
    for number, (start, end) in enumerate(ranges):
        representative, vector = representative_index(embeddings, start, end)
        pooled.append(vector)
        rows.append({
            "micro_clip_id": f"micro_clip_{number:04d}",
            "start_time": float(start),
            "end_time": float(duration if number == len(ranges) - 1 else min(duration, end)),
            "frame_indices": list(range(start, end)),
            "frame_timestamps": [float(i) for i in range(start, end)],
            "frame_paths": [p.as_posix() for p in frame_paths[start:end]],
            "embedding_index": number,
            "representative_frame_index": representative,
            "representative_frame_timestamp": float(representative),
            "representative_frame_path": frame_paths[representative].as_posix(),
            "motion_magnitude": motion_magnitude(frame_paths[start:end]),
            "duration": float((duration if number == len(ranges) - 1 else min(duration, end)) - start),
            "warnings": ["final_partial_window"] if number == len(ranges) - 1 and duration - start < window else [],
        })
    shape = (0, embeddings.shape[1]) if embeddings.ndim == 2 else (0, 0)
    return rows, np.stack(pooled).astype(np.float32) if pooled else np.empty(shape, np.float32)


def temporal_overlap(item: dict[str, Any], start: float, end: float) -> bool:
    return float(item["start_time"]) < end and float(item["end_time"]) > start


def rank_cosine(query: np.ndarray, embeddings: np.ndarray, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query = np.asarray(query, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    scores = normalize_rows(embeddings) @ query
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [{**rows[int(i)], "rank": rank_no, "similarity_score": float(scores[int(i)])}
            for rank_no, i in enumerate(order, 1)]
