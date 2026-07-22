from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TARGET_VIDEO_ID = "pasadena/YKI08"
SUPPORTED_DURATION_CLASSES = ("1s", "10s", "60s")


class CRadioDiagnosticError(RuntimeError):
    """Raised when the isolated representation diagnostic contract is violated."""


@dataclass(frozen=True)
class RetrievalQuestion:
    question_id: str
    video_id: str
    duration_class: str
    question: str
    gt_start_sec: float
    gt_end_sec: float


def load_frozen_questions(manifest_path: Path, video_id: str = TARGET_VIDEO_ID) -> list[RetrievalQuestion]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("questions")
    if not isinstance(rows, list):
        raise CRadioDiagnosticError(f"Invalid question manifest: {manifest_path}")

    selected: list[RetrievalQuestion] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("video_id") != video_id:
            continue
        question_id = str(row.get("question_id") or "")
        duration_class = str(row.get("duration_class") or "")
        interval = row.get("gt_interval_sec")
        question = str(row.get("question") or "").strip()
        if not question_id or question_id in seen:
            raise CRadioDiagnosticError(f"Missing or duplicate question ID for {video_id}: {question_id!r}")
        if duration_class not in SUPPORTED_DURATION_CLASSES:
            raise CRadioDiagnosticError(f"Unsupported duration class for {question_id}: {duration_class}")
        if not isinstance(interval, list) or len(interval) != 2:
            raise CRadioDiagnosticError(f"Invalid interval for {question_id}: {interval!r}")
        start_sec, end_sec = map(float, interval)
        if start_sec < 0 or end_sec <= start_sec or not question:
            raise CRadioDiagnosticError(f"Invalid frozen question row: {question_id}")
        selected.append(
            RetrievalQuestion(
                question_id=question_id,
                video_id=video_id,
                duration_class=duration_class,
                question=question,
                gt_start_sec=start_sec,
                gt_end_sec=end_sec,
            )
        )
        seen.add(question_id)
    if not selected:
        raise CRadioDiagnosticError(f"No frozen questions found for {video_id}")
    return selected


def one_fps_timestamps(duration_sec: float) -> np.ndarray:
    """Return a deterministic one-Hz grid at 0, 1, ... while t < duration."""
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise CRadioDiagnosticError(f"Invalid video duration: {duration_sec}")
    return np.arange(math.ceil(duration_sec), dtype=np.float64)


def timestamps_to_frame_indices(
    timestamps_sec: np.ndarray, *, average_fps: float, frame_count: int,
) -> np.ndarray:
    if average_fps <= 0 or frame_count <= 0:
        raise CRadioDiagnosticError("FPS and frame count must be positive")
    indices = np.floor(np.asarray(timestamps_sec, dtype=np.float64) * average_fps).astype(np.int64)
    return np.clip(indices, 0, frame_count - 1)


def rank_timestamps(similarities: Sequence[float]) -> np.ndarray:
    """Rank independently of any annotation interval; ties retain timestamp order."""
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise CRadioDiagnosticError("Similarities must be a non-empty finite 1-D sequence")
    return np.argsort(-values, kind="stable")


def distance_to_half_open_interval(timestamp_sec: float, start_sec: float, end_sec: float) -> float:
    if start_sec <= timestamp_sec < end_sec:
        return 0.0
    if timestamp_sec < start_sec:
        return start_sec - timestamp_sec
    return timestamp_sec - end_sec


def posthoc_gt_interval_metrics(
    ranked_indices: Sequence[int],
    timestamps_sec: Sequence[float],
    *,
    gt_start_sec: float,
    gt_end_sec: float,
    ks: Sequence[int] = (1, 5, 8, 10),
) -> dict[str, Any]:
    """Evaluate a completed ranking against GT; GT cannot affect ranking here."""
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    ranking = np.asarray(ranked_indices, dtype=np.int64)
    if gt_start_sec < 0 or gt_end_sec <= gt_start_sec:
        raise CRadioDiagnosticError("Invalid GT interval")
    if ranking.ndim != 1 or ranking.size != timestamps.size or set(ranking.tolist()) != set(range(timestamps.size)):
        raise CRadioDiagnosticError("Ranking must be a permutation of all timestamp indices")

    ranked_timestamps = timestamps[ranking]
    inside = (ranked_timestamps >= gt_start_sec) & (ranked_timestamps < gt_end_sec)
    inside_positions = np.flatnonzero(inside)
    first_inside_rank = int(inside_positions[0] + 1) if inside_positions.size else None
    top1_distance = distance_to_half_open_interval(
        float(ranked_timestamps[0]), gt_start_sec, gt_end_sec
    )
    return {
        "gt_interval_sec": [gt_start_sec, gt_end_sec],
        "metric_semantics": (
            "Post-hoc timestamp membership in the annotated interval; this is a coarse temporal "
            "exposure diagnostic and does not prove that decisive visual evidence is present."
        ),
        "gt_interval_hit_at_k": {
            str(k): bool(inside[: min(int(k), inside.size)].any()) for k in ks
        },
        "best_ranked_timestamp_distance_to_gt_interval_sec": float(top1_distance),
        "rank_of_first_timestamp_inside_gt_interval": first_inside_rank,
    }

