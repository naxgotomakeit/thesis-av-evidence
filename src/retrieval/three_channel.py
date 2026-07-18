from __future__ import annotations

from typing import Any

import numpy as np


ALLOWED_RETRIEVAL_CASE_KEYS = {"case_id", "video_id", "question"}
FORBIDDEN_RETRIEVAL_KEYS = {
    "answer", "answers", "answer_options", "options", "choices",
    "provided_context", "annotation_context", "provided_timestamp",
    "provided_timestamp_start", "provided_timestamp_end",
    "manual_evidence_intervals", "human_review", "selection_reason",
}


def safe_retrieval_case(case: dict[str, Any]) -> dict[str, str]:
    """Build the only payload allowed to cross into query encoding/ranking."""
    payload = {key: case[key] for key in ("case_id", "video_id", "question")}
    assert_no_leakage_payload(payload)
    return payload


def assert_no_leakage_payload(payload: dict[str, Any]) -> None:
    forbidden = FORBIDDEN_RETRIEVAL_KEYS.intersection(payload)
    if forbidden:
        raise ValueError(f"retrieval payload contains forbidden keys: {sorted(forbidden)}")
    extra = set(payload).difference(ALLOWED_RETRIEVAL_CASE_KEYS)
    if extra:
        raise ValueError(f"retrieval payload contains non-allowlisted keys: {sorted(extra)}")
    if not str(payload.get("question", "")).strip():
        raise ValueError("raw question must be non-empty")


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def cosine_similarities(query: np.ndarray, index_embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    index_embeddings = np.asarray(index_embeddings, dtype=np.float32)
    if index_embeddings.ndim != 2:
        raise ValueError("index embeddings must be 2D")
    if query.shape[1] != index_embeddings.shape[1]:
        raise ValueError(f"embedding dimension mismatch: {query.shape[1]} vs {index_embeddings.shape[1]}")
    if index_embeddings.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    return (normalize_rows(index_embeddings) @ normalize_rows(query).T).reshape(-1)


def rank_index(query: np.ndarray, index_embeddings: np.ndarray, metadata_rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if len(index_embeddings) != len(metadata_rows):
        raise ValueError(f"embedding/metadata row mismatch: {len(index_embeddings)} vs {len(metadata_rows)}")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    scores = cosine_similarities(query, index_embeddings)
    # Stable sort gives deterministic row-order tie breaking.
    order = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
    return [
        {"rank": rank, "similarity_score": float(scores[row]), "embedding_row": int(row), "metadata": metadata_rows[int(row)]}
        for rank, row in enumerate(order, start=1)
    ]


def temporal_overlap(start: float, end: float, reference_start: float, reference_end: float) -> bool:
    return float(end) > float(reference_start) and float(start) < float(reference_end)


def temporal_distance(start: float, end: float, reference_start: float, reference_end: float) -> float:
    if temporal_overlap(start, end, reference_start, reference_end):
        return 0.0
    if float(end) <= float(reference_start):
        return float(reference_start) - float(end)
    return float(start) - float(reference_end)


def evaluate_ranked_results(results: list[dict[str, Any]], reference_start: float, reference_end: float) -> dict[str, Any]:
    if not results:
        return {"top1_overlap": False, "top3_overlap": False, "minimum_temporal_distance_sec": None, "closest_retrieved_rank": None, "closest_retrieved_similarity_score": None}
    evaluated = []
    for result in results:
        start, end = float(result["start_time"]), float(result["end_time"])
        evaluated.append((temporal_distance(start, end, reference_start, reference_end), int(result["rank"]), float(result["similarity_score"])))
    closest = min(evaluated, key=lambda x: (x[0], x[1]))
    return {
        "top1_overlap": temporal_overlap(results[0]["start_time"], results[0]["end_time"], reference_start, reference_end),
        "top3_overlap": any(temporal_overlap(x["start_time"], x["end_time"], reference_start, reference_end) for x in results[:3]),
        "minimum_temporal_distance_sec": round(closest[0], 6),
        "closest_retrieved_rank": closest[1],
        "closest_retrieved_similarity_score": closest[2],
    }


def validate_query_dimension(query: np.ndarray, index: np.ndarray) -> bool:
    query = np.asarray(query)
    index = np.asarray(index)
    return query.size == index.shape[1] if index.ndim == 2 else False

