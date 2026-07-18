"""Deterministic regression tests for EgoSchema temporal evidence propagation."""

from src.canonical_pipeline.reranking import resolve_visual_frame_selection_anchor
from src.retrieval.task5b import rank_visual_micro_candidates
from src.retrieval.task6_relation_reranking import deduplicate_visual_frames
from src.retrieval.task6_v1_1 import order_visual_frames


def _micro(candidate_id: str, start: float, score: float, rank: int, distance: float):
    return {
        "candidate_id": candidate_id,
        "microclip_id": candidate_id,
        "start_time": start,
        "end_time": start + 4.0,
        "anchor_distance_sec": distance,
        "semantic_similarity_score": score,
        "semantic_query_rank": rank,
    }


def _record(coarse=None):
    return {"task5b_v1_1_input": {"coarse_visual_candidates": coarse or []}}


def test_no_timestamp_micro_selection_uses_existing_semantic_relevance():
    early = _micro("early", 0.0, 0.10, 2, 0.0)
    semantic = _micro("semantic", 90.0, 0.90, 1, 0.0)
    ranked = rank_visual_micro_candidates(
        [early, semantic], meaningful_temporal_anchor=False
    )
    assert ranked[0]["candidate_id"] == "semantic"


def test_explicit_timestamp_keeps_existing_temporal_distance_priority():
    temporal = _micro("temporal", 10.0, 0.10, 2, 0.0)
    semantic = _micro("semantic", 90.0, 0.90, 1, 80.0)
    ranked = rank_visual_micro_candidates(
        [semantic, temporal], meaningful_temporal_anchor=True
    )
    assert ranked[0]["candidate_id"] == "temporal"


def test_visual_only_semantic_anchor_selects_nearby_four_frames():
    visual = [{
        "candidate_id": "micro_100",
        "start_time": 98.0,
        "end_time": 102.0,
        "task5b_selection_rank": 1,
        "query_rank": 1,
        "similarity_score": 0.8,
    }]
    anchor = resolve_visual_frame_selection_anchor(_record(), [], visual)
    assert anchor["source"] == "visual_semantic"
    assert anchor["timestamp_sec"] == 100.0
    frames = [
        {"video_id": "v", "timestamp": value, "canonical_frame_path": f"{value}.jpg"}
        for value in (0.0, 0.5, 99.0, 99.5, 100.0, 100.5, 179.5)
    ]
    selected, _ = deduplicate_visual_frames(frames, 4, anchor["timestamp_sec"])
    ordered = order_visual_frames(
        selected, [], anchor_time_override=anchor["timestamp_sec"]
    )
    assert [item["timestamp"] for item in ordered] == [99.0, 99.5, 100.0, 100.5]
    assert all(item["anchor_distance_sec"] <= 1.0 for item in ordered)


def test_true_no_anchor_no_evidence_keeps_chronological_fallback():
    anchor = resolve_visual_frame_selection_anchor(_record(), [], [])
    assert anchor == {
        "source": "true_fallback",
        "timestamp_sec": None,
        "candidate_ids": [],
    }
    frames = [
        {"video_id": "v", "timestamp": value, "canonical_frame_path": f"{value}.jpg"}
        for value in (10.0, 0.5, 2.0, 0.0, 1.5, 1.0)
    ]
    selected, _ = deduplicate_visual_frames(frames, 4, None)
    assert [item["timestamp"] for item in selected] == [0.0, 0.5, 1.0, 1.5]
