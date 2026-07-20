"""Deterministic lightweight motion proxy and within-Event change points.

This is deliberately labelled ``LIGHTWEIGHT_MOTION_PROXY``.  It is not RAFT
and is not a reproduction of CoMET's action-level module.  It uses only the
frozen 1 FPS frame grid so the immutable Event boundaries remain the parent
temporal lattice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, sobel


MOTION_SCHEMA_VERSION = "lightweight-motion-proxy-v1"
ACTION_SCHEMA_VERSION = "event-action-parent-child-v1"


@dataclass(frozen=True)
class MotionProxyConfig:
    image_width: int = 160
    image_height: int = 90
    smoothing_sigma_frames: float = 1.0
    minimum_action_duration_sec: float = 3.0
    penalty_multiplier: float = 1.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_set_identity(frame_paths: list[Path], timestamps: np.ndarray) -> str:
    """Content identity for the exact ordered frozen frame grid."""
    digest = hashlib.sha256()
    digest.update(np.asarray(timestamps, dtype=np.float64).tobytes())
    for path in frame_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _expected_metadata(
    *, video_id: str, timestamps: np.ndarray, frame_identity: str, config: MotionProxyConfig
) -> dict[str, Any]:
    return {
        "schema_version": MOTION_SCHEMA_VERSION,
        "video_id": video_id,
        "frame_identity_sha256": frame_identity,
        "frame_timestamps_sha256": hashlib.sha256(
            np.asarray(timestamps, dtype=np.float64).tobytes()
        ).hexdigest(),
        "frame_count": int(len(timestamps)),
        "method": "LIGHTWEIGHT_MOTION_PROXY",
        "config": asdict(config),
    }


def motion_cache_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Require exact provenance/config compatibility; never accept a stale cache."""
    return all(metadata.get(key) == value for key, value in expected.items())


def _load_gray(path: Path, config: MotionProxyConfig) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("L").resize(
            (config.image_width, config.image_height), Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.float32) / 255.0


def _difference_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.abs(right - left)
    edge_left = np.hypot(sobel(left, axis=0), sobel(left, axis=1))
    edge_right = np.hypot(sobel(right, axis=0), sobel(right, axis=1))
    edge_difference = np.abs(edge_right - edge_left)
    values = [
        float(np.mean(difference)),
        float(np.quantile(difference, 0.75)),
        float(np.quantile(difference, 0.90)),
        float(np.std(difference)),
        float(np.mean(edge_difference)),
        float(np.quantile(edge_difference, 0.90)),
    ]
    for y_indices in np.array_split(np.arange(difference.shape[0]), 3):
        for x_indices in np.array_split(np.arange(difference.shape[1]), 4):
            values.append(float(np.mean(difference[np.ix_(y_indices, x_indices)])))
    return np.asarray(values, dtype=np.float64)


def extract_motion_proxy(
    frame_paths: list[Path], timestamps: np.ndarray, video_duration: float,
    config: MotionProxyConfig,
) -> dict[str, np.ndarray]:
    if len(frame_paths) != len(timestamps) or len(frame_paths) < 2:
        raise ValueError("Motion proxy requires matching frame paths/timestamps and >=2 frames")
    frames = [_load_gray(path, config) for path in frame_paths]
    features = np.stack(
        [_difference_features(left, right) for left, right in zip(frames[:-1], frames[1:])]
    )
    starts = np.asarray(timestamps[:-1], dtype=np.float64)
    ends = np.asarray(timestamps[1:], dtype=np.float64)
    # The final 1 FPS cell has no future sampled frame.  Reuse the last measured
    # transition only to give the [last timestamp, video end) cell explicit
    # coverage; it cannot create an internal boundary by itself.
    if float(timestamps[-1]) < float(video_duration):
        features = np.vstack([features, features[-1]])
        starts = np.append(starts, float(timestamps[-1]))
        ends = np.append(ends, float(video_duration))
    scalar = features[:, 0]
    smoothed_scalar = gaussian_filter1d(
        scalar, sigma=config.smoothing_sigma_frames, mode="nearest"
    )
    return {
        "features": features,
        "interval_starts": starts,
        "interval_ends": ends,
        "raw_motion": scalar,
        "smoothed_motion": smoothed_scalar,
    }


def extract_or_load_motion_proxy(
    *, video_id: str, frame_paths: list[Path], timestamps: np.ndarray,
    video_duration: float, cache_npz: Path, cache_metadata: Path,
    config: MotionProxyConfig, force_recompute: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    identity = frame_set_identity(frame_paths, timestamps)
    expected = _expected_metadata(
        video_id=video_id, timestamps=timestamps, frame_identity=identity, config=config
    )
    if not force_recompute and cache_npz.exists() and cache_metadata.exists():
        stored = json.loads(cache_metadata.read_text(encoding="utf-8"))
        if motion_cache_compatible(stored, expected):
            with np.load(cache_npz) as bundle:
                arrays = {key: np.asarray(bundle[key]) for key in bundle.files}
            return arrays, {
                **stored,
                "cache_hit": True,
                "load_or_extract_sec": time.perf_counter() - started,
                "cache_size_bytes": cache_npz.stat().st_size + cache_metadata.stat().st_size,
            }
    arrays = extract_motion_proxy(frame_paths, timestamps, video_duration, config)
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, **arrays)
    metadata = {
        **expected,
        "video_duration": float(video_duration),
        "feature_shape": list(arrays["features"].shape),
        "feature_dtype": str(arrays["features"].dtype),
    }
    cache_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return arrays, {
        **metadata,
        "cache_hit": False,
        "load_or_extract_sec": time.perf_counter() - started,
        "cache_size_bytes": cache_npz.stat().st_size + cache_metadata.stat().st_size,
    }


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    standardized = (matrix - median) / np.maximum(1.4826 * mad, 1e-5)
    # Near-constant channels can otherwise dominate through numerical scale.
    return np.clip(standardized, -20.0, 20.0)


def _rbf_kernel(values: np.ndarray) -> tuple[np.ndarray, float]:
    standardized = _robust_standardize(values)
    squared = np.sum(
        (standardized[:, None, :] - standardized[None, :, :]) ** 2, axis=2
    )
    positive = squared[squared > 1e-12]
    gamma = 1.0 / float(np.median(positive)) if len(positive) else 1.0
    return np.exp(-gamma * squared), gamma


def exact_rbf_change_points(
    values: np.ndarray, *, minimum_size: int, penalty_multiplier: float
) -> dict[str, Any]:
    """Exact DP for a penalized RBF within-segment scatter objective.

    The objective matches deterministic multiple-change-point segmentation;
    unlike PELT, no pruning is performed because frozen Event parents are short.
    """
    matrix = np.asarray(values, dtype=np.float64)
    count = len(matrix)
    if count < 2 * minimum_size:
        return {
            "change_point_indices": [], "gamma": None, "penalty": None,
            "objective": None, "unsplit_objective": None,
        }
    kernel, gamma = _rbf_kernel(matrix)
    prefix = np.pad(kernel, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    def cost(start: int, end: int) -> float:
        size = end - start
        block_sum = (
            prefix[end, end] - prefix[start, end]
            - prefix[end, start] + prefix[start, start]
        )
        return float(size - block_sum / size)

    penalty = float(penalty_multiplier * math.log(max(count, 2)))
    best = np.full(count + 1, np.inf, dtype=np.float64)
    previous = np.full(count + 1, -1, dtype=np.int64)
    best[0] = -penalty
    for end in range(minimum_size, count + 1):
        for start in range(0, end - minimum_size + 1):
            if start and start < minimum_size:
                continue
            if not np.isfinite(best[start]):
                continue
            candidate = best[start] + cost(start, end) + penalty
            if candidate < best[end] - 1e-12:
                best[end] = candidate
                previous[end] = start
    change_points: list[int] = []
    cursor = count
    while previous[cursor] > 0:
        change_points.append(int(previous[cursor]))
        cursor = int(previous[cursor])
    change_points.reverse()
    return {
        "change_point_indices": change_points,
        "gamma": float(gamma),
        "penalty": penalty,
        "objective": float(best[count]),
        "unsplit_objective": cost(0, count),
    }


def _boundary_score(smoothed: np.ndarray, index: int, window: int = 2) -> float:
    left = smoothed[max(0, index - window):index]
    right = smoothed[index:min(len(smoothed), index + window)]
    if not len(left) or not len(right):
        return 0.0
    scale = max(float(np.median(np.abs(smoothed - np.median(smoothed)))) * 1.4826, 1e-8)
    return float(abs(float(np.mean(right)) - float(np.mean(left))) / scale)


def build_action_children(
    *, video_id: str, event_segments: list[dict[str, Any]],
    motion: dict[str, np.ndarray], config: MotionProxyConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split only inside immutable Event parents and validate exact coverage."""
    starts = np.asarray(motion["interval_starts"], dtype=np.float64)
    ends = np.asarray(motion["interval_ends"], dtype=np.float64)
    features = np.asarray(motion["features"], dtype=np.float64)
    smooth_scalar = np.asarray(motion["smoothed_motion"], dtype=np.float64)
    actions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    action_counter = 0
    for parent_index, event in enumerate(event_segments):
        parent_id = str(event["segment_id"])
        event_start, event_end = float(event["start"]), float(event["end"])
        indices = np.flatnonzero((starts >= event_start - 1e-8) & (ends <= event_end + 1e-8))
        local_features = features[indices]
        smoothed_features = gaussian_filter1d(
            local_features, sigma=config.smoothing_sigma_frames,
            axis=0, mode="nearest",
        ) if len(local_features) else local_features
        frame_step = float(np.median(ends - starts)) if len(starts) else 1.0
        minimum_size = max(1, int(math.ceil(config.minimum_action_duration_sec / frame_step)))
        result = exact_rbf_change_points(
            smoothed_features, minimum_size=minimum_size,
            penalty_multiplier=config.penalty_multiplier,
        )
        boundary_pairs = [
            (int(index), float(starts[indices[index]]))
            for index in result["change_point_indices"]
        ]
        boundary_pairs = [
            (index, value) for index, value in boundary_pairs
            if value - event_start >= config.minimum_action_duration_sec - 1e-8
            and event_end - value >= config.minimum_action_duration_sec - 1e-8
        ]
        boundary_times = [value for _, value in boundary_pairs]
        points = [event_start, *boundary_times, event_end]
        parent_action_ids: list[str] = []
        boundary_records = []
        for local_index, timestamp in boundary_pairs:
            boundary_records.append(
                {
                    "timestamp": timestamp,
                    "motion_change_score": _boundary_score(
                        smooth_scalar[indices], local_index
                    ),
                    "raw_motion_before": float(motion["raw_motion"][indices[max(0, local_index - 1)]]),
                    "raw_motion_after": float(motion["raw_motion"][indices[min(len(indices) - 1, local_index)]]),
                }
            )
        for start, end in zip(points[:-1], points[1:]):
            action_id = f"{video_id}:action:{action_counter:04d}"
            action_counter += 1
            parent_action_ids.append(action_id)
            actions.append(
                {
                    "action_id": action_id,
                    "parent_event_id": parent_id,
                    "parent_event_index": parent_index,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "representative_frame_timestamp": (start + end) / 2.0,
                    "split_from_parent": len(points) > 2,
                    "schema_version": ACTION_SCHEMA_VERSION,
                }
            )
        decisions.append(
            {
                "video_id": video_id,
                "parent_event_id": parent_id,
                "parent_start": event_start,
                "parent_end": event_end,
                "motion_observation_count": int(len(indices)),
                "feature_smoothing_sigma_frames": config.smoothing_sigma_frames,
                "change_points": boundary_records,
                "action_child_ids": parent_action_ids,
                "split": len(parent_action_ids) > 1,
                "change_point_diagnostics": result,
                "reason": (
                    "penalized_rbf_change_points_detected"
                    if len(parent_action_ids) > 1 else
                    "no_penalized_motion_change_point; parent preserved as one Action"
                ),
            }
        )
    validate_action_partition(event_segments, actions)
    return actions, decisions


def validate_action_partition(
    event_segments: list[dict[str, Any]], actions: list[dict[str, Any]], tolerance: float = 1e-7
) -> dict[str, Any]:
    parent_ids = [str(row["segment_id"]) for row in event_segments]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Event IDs must be unique")
    by_parent: dict[str, list[dict[str, Any]]] = {identifier: [] for identifier in parent_ids}
    for action in actions:
        parent_id = str(action["parent_event_id"])
        if parent_id not in by_parent:
            raise ValueError(f"Action has unknown Event parent: {parent_id}")
        by_parent[parent_id].append(action)
    for event in event_segments:
        children = sorted(by_parent[str(event["segment_id"])], key=lambda row: row["start"])
        if not children:
            raise ValueError("Every Event must have at least one Action child")
        if abs(float(children[0]["start"]) - float(event["start"])) > tolerance:
            raise ValueError("Action children do not begin at parent Event start")
        if abs(float(children[-1]["end"]) - float(event["end"])) > tolerance:
            raise ValueError("Action children do not end at parent Event end")
        for left, right in zip(children[:-1], children[1:]):
            if abs(float(left["end"]) - float(right["start"])) > tolerance:
                raise ValueError("Action children have a gap or overlap")
        if any(float(row["duration"]) <= 0 for row in children):
            raise ValueError("Action duration must be positive")
    return {
        "valid": True,
        "event_count": len(event_segments),
        "action_count": len(actions),
        "parent_coverage_rate": 1.0,
    }
