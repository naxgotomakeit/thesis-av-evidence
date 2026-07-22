from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.diagnostics.cradio_v4.core import rank_timestamps


@dataclass(frozen=True)
class RawQuestionQuery:
    question_id: str
    video_id: str
    raw_question: str


def build_raw_question_queries(rows: Sequence[dict[str, Any]]) -> list[RawQuestionQuery]:
    queries: list[RawQuestionQuery] = []
    for row in rows:
        raw = str(row["question"])
        if not raw:
            raise ValueError(f"Empty raw question: {row['question_id']}")
        queries.append(RawQuestionQuery(
            question_id=str(row["question_id"]),
            video_id=str(row["video_id"]),
            raw_question=raw,
        ))
    return queries


def cosine_ranking(
    *, raw_question_embedding: np.ndarray, normalized_visual_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(raw_question_embedding, dtype=np.float32)
    visual = np.asarray(normalized_visual_embeddings, dtype=np.float32)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1 or visual.ndim != 2 or visual.shape[1] != query.size:
        raise ValueError("Text/visual embedding shape mismatch")
    if not np.isfinite(query).all() or not np.isfinite(visual).all():
        raise ValueError("Non-finite retrieval embeddings")
    query_norm = float(np.linalg.norm(query))
    visual_norms = np.linalg.norm(visual, axis=1, keepdims=True)
    if query_norm <= 0 or np.any(visual_norms <= 0):
        raise ValueError("Zero-norm retrieval embedding")
    query = query / query_norm
    visual = visual / visual_norms
    similarities = visual @ query
    ranking = rank_timestamps(similarities)
    return similarities.astype(np.float32), ranking.astype(np.int64)


def select_top8_unique(
    *, ranking: np.ndarray, timestamps_sec: np.ndarray,
) -> list[int]:
    ranking = np.asarray(ranking, dtype=np.int64)
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    selected: list[int] = []
    seen_timestamps: set[float] = set()
    for index in ranking:
        timestamp = float(timestamps[int(index)])
        if timestamp in seen_timestamps:
            continue
        selected.append(int(index))
        seen_timestamps.add(timestamp)
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("Fewer than eight unique 1 FPS frames are available")
    return selected
