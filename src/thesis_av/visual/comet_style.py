"""Experimental CoMET-style coarse appearance/event change-point segmentation.

The high-level pipeline follows the requested CoMET-style formulation. Parameter
values are explicit experimental defaults; they are not presented as paper values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class CometStyleConfig:
    smoothing_sigma_frames: float = 1.0
    adaptive_mad_multiplier: float = 0.5
    minimum_prominence: float = 0.005
    minimum_segment_duration_sec: float = 4.0


def segment(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    config: CometStyleConfig,
) -> dict[str, Any]:
    """Detect local minima in smoothed adjacent DINO cosine similarity."""
    features = np.asarray(embeddings, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float64)
    if len(features) != len(times):
        raise ValueError("DINO embeddings/timestamps length mismatch")
    if len(features) < 2:
        return {"boundaries": [], "raw_similarity": [], "smoothed_similarity": [], "diagnostics": {}}
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    raw = np.sum(normalized[:-1] * normalized[1:], axis=1).astype(np.float64)
    smooth = gaussian_filter1d(raw, sigma=config.smoothing_sigma_frames, mode="nearest")
    median = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - median)))
    prominence = max(config.minimum_prominence, config.adaptive_mad_multiplier * mad)
    frame_step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    minimum_distance = max(1, int(np.ceil(config.minimum_segment_duration_sec / frame_step)))
    minima, properties = find_peaks(-smooth, distance=minimum_distance, prominence=prominence)
    # A local minimum must also be below the robust center; deterministic filtering
    # prevents shallow high-similarity extrema from becoming event boundaries.
    minima = np.asarray([index for index in minima if smooth[index] < median], dtype=np.int64)
    boundaries = [float(times[index + 1]) for index in minima]
    return {
        "boundaries": boundaries,
        "raw_similarity": raw.tolist(),
        "smoothed_similarity": smooth.tolist(),
        "diagnostics": {
            "median_smoothed_similarity": median,
            "mad_smoothed_similarity": mad,
            "adaptive_prominence": prominence,
            "minimum_peak_distance_frames": minimum_distance,
            "candidate_minima_count_before_median_filter": int(len(properties.get("prominences", []))),
            "boundary_indices": minima.tolist(),
        },
    }

