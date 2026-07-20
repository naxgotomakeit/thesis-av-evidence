"""Faithful NumPy adaptation of Kernel Temporal Segmentation (KTS).

Reference: Potapov, Douze, Harchaoui, Schmid, ECCV 2014. The implementation
minimizes kernel within-segment scatter by dynamic programming and selects the
change-point count with the paper's BIC-form penalty. The original paper tunes C
on annotated validation data; this experiment freezes a data-scaled cpd_auto-style
coefficient instead and labels it as an experimental default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KTSConfig:
    max_change_points: int = 30
    minimum_segment_duration_sec: float = 4.0
    maximum_segment_duration_sec: float | None = None
    penalty_strength: float = 1.0
    kernel: str = "linear_cosine"


def _scatter_costs(kernel: np.ndarray) -> np.ndarray:
    """Return KTS within-segment scatter cost for every half-open interval."""
    matrix = np.asarray(kernel, dtype=np.float64)
    n = len(matrix)
    diagonal_prefix = np.concatenate(([0.0], np.cumsum(np.diag(matrix))))
    integral = np.pad(matrix, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    costs = np.full((n, n + 1), np.inf, dtype=np.float64)
    for start in range(n):
        ends = np.arange(start + 1, n + 1)
        lengths = ends - start
        diagonal = diagonal_prefix[ends] - diagonal_prefix[start]
        block_sum = (
            integral[ends, ends]
            - integral[start, ends]
            - integral[ends, start]
            + integral[start, start]
        )
        costs[start, ends] = diagonal - block_sum / lengths
    return np.maximum(costs, 0.0)


def segment(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    config: KTSConfig,
) -> dict[str, Any]:
    """Run KTS DP and BIC-style automatic change-point count selection."""
    features = np.asarray(embeddings, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if len(features) != len(times):
        raise ValueError("DINO embeddings/timestamps length mismatch")
    n = len(features)
    if n < 2:
        return {"boundaries": [], "diagnostics": {"selected_change_points": 0}}
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    kernel = normalized @ normalized.T
    costs = _scatter_costs(kernel)
    frame_step = float(np.median(np.diff(times)))
    min_frames = max(1, int(np.ceil(config.minimum_segment_duration_sec / frame_step)))
    max_frames = None
    if config.maximum_segment_duration_sec is not None:
        max_frames = max(min_frames, int(np.floor(config.maximum_segment_duration_sec / frame_step)))
    max_segments = min(config.max_change_points + 1, n // min_frames)
    dp = np.full((max_segments + 1, n + 1), np.inf, dtype=np.float64)
    previous = np.full((max_segments + 1, n + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for segment_count in range(1, max_segments + 1):
        minimum_end = segment_count * min_frames
        for end in range(minimum_end, n + 1):
            earliest = (segment_count - 1) * min_frames
            if max_frames is not None:
                earliest = max(earliest, end - max_frames)
            latest = end - min_frames
            starts = np.arange(earliest, latest + 1)
            values = dp[segment_count - 1, starts] + costs[starts, end]
            choice = int(np.argmin(values))
            dp[segment_count, end] = float(values[choice])
            previous[segment_count, end] = int(starts[choice])
    one_segment_scatter = max(float(costs[0, n]), 1e-12)
    # cpd_auto's vmax represents the descriptor/kernel scale, not the sum of
    # scatter over N samples. L2-normalized linear kernels have diagonal scale 1.
    kernel_scale_vmax = max(float(np.median(np.diag(kernel))), 1e-12)
    objectives = []
    for segment_count in range(1, max_segments + 1):
        change_points = segment_count - 1
        if change_points == 0:
            penalty = 0.0
        else:
            penalty = (
                config.penalty_strength
                * kernel_scale_vmax
                * change_points
                / (2.0 * n)
                * (np.log(float(n) / change_points) + 1.0)
            )
        objectives.append(float(dp[segment_count, n] / n + penalty))
    selected_segments = int(np.argmin(objectives)) + 1
    edges = [n]
    end = n
    for segment_count in range(selected_segments, 0, -1):
        start = int(previous[segment_count, end])
        if start < 0:
            raise RuntimeError("KTS backtracking failed")
        edges.append(start)
        end = start
    edges = sorted(edges)
    internal = edges[1:-1]
    boundaries = [float(times[index]) for index in internal]
    return {
        "boundaries": boundaries,
        "diagnostics": {
            "kernel": config.kernel,
            "selected_change_points": selected_segments - 1,
            "selected_segment_count": selected_segments,
            "selected_boundary_indices": internal,
            "objective_by_segment_count": objectives,
            "one_segment_scatter": one_segment_scatter,
            "kernel_scale_vmax": kernel_scale_vmax,
            "minimum_segment_frames": min_frames,
            "maximum_segment_frames": max_frames,
            "max_change_points_considered": max_segments - 1,
            "penalty_formula": "C*m*(log(n/m)+1), C=penalty_strength*kernel_scale_vmax/(2*n)",
        },
    }
