from __future__ import annotations

import inspect

from src.retrieval import task6_v1_2 as core


def relation(source: str, target: str, kind: str) -> dict:
    return {"source_candidate_id": source, "target_candidate_id": target, "relation_type": kind, "temporal_gap_sec": 0.2, "relation_confidence": "temporal_only", "relation_basis": "transcript_sequence"}


def test_a_b_c_visual_frames_are_serialized_chronologically_with_preserved_selection_rank():
    frames = [
        {"timestamp": 3.5, "selection_rank": 1, "anchor_distance_sec": 0.2},
        {"timestamp": 4.0, "selection_rank": 2, "anchor_distance_sec": 0.3},
        {"timestamp": 3.0, "selection_rank": 3, "anchor_distance_sec": 0.7},
        {"timestamp": 4.5, "selection_rank": 4, "anchor_distance_sec": 0.8},
    ]
    ordered = core.chronological_frames(frames)
    assert [item["timestamp"] for item in ordered] == [3.0, 3.5, 4.0, 4.5]
    assert [item["selection_rank"] for item in ordered] == [3, 1, 2, 4]
    assert [item["presentation_order"] for item in ordered] == [1, 2, 3, 4]
    assert [item["anchor_distance_sec"] for item in ordered] == [0.7, 0.2, 0.3, 0.8]


def test_d_e_response_to_points_response_to_trigger_and_inverse_is_trigger_of():
    corrected = core.correct_response_relations([relation("trigger", "response", "response_to")])
    response_to = next(item for item in corrected if item["relation_type"] == "response_to")
    trigger_of = next(item for item in corrected if item["relation_type"] == "trigger_of")
    assert (response_to["source_candidate_id"], response_to["target_candidate_id"]) == ("response", "trigger")
    assert (trigger_of["source_candidate_id"], trigger_of["target_candidate_id"]) == ("trigger", "response")


def test_f_alternative_relation_is_unchanged():
    alternative = relation("r1", "r2", "alternative_to")
    assert core.correct_response_relations([alternative]) == [alternative]


def test_g_per_case_drop_count_equals_genuine_drops():
    result = core.corrected_budget_accounting({"dropped_candidate_count": 99}, [{"candidate_id": "x"}, {"candidate_id": "y"}])
    assert result["dropped_candidate_count"] == 2
    assert result["dropped_candidate_count_consistent"] is True


def test_h_merged_sources_do_not_increase_drop_count_or_break_reconciliation():
    packet = {
        "candidates_before_reranking": [{"candidate_id": "v1"}, {"candidate_id": "v2"}, {"candidate_id": "speech"}],
        "retained_candidates": [
            {"candidate_id": "visual", "candidate_type": "canonical_visual_evidence"},
            {"candidate_id": "speech"},
        ],
        "merged_source_candidates": [
            {"source_candidate_id": "v1"}, {"source_candidate_id": "v2"},
        ],
        "transformed_candidates": [],
        "actually_dropped_candidates": [],
    }
    assert core.merge_reduction_count(packet["merged_source_candidates"]) == 1
    assert core.candidate_count_consistent(packet)


def test_i_dropped_visual_frames_do_not_affect_candidate_drop_count():
    result = core.corrected_budget_accounting({}, [])
    assert result["dropped_candidate_count"] == 0
    assert result["dropped_candidate_count_consistent"] is True


def test_j_no_model_retrieval_or_media_dependencies():
    source = inspect.getsource(core).lower()
    for token in ("import whisper", "anthropic", "openai", "cv2", "librosa", "clip.load", "requests"):
        assert token not in source


def test_k_preserves_input_frame_objects_except_presentation_order():
    frame = {"timestamp": 1.0, "selection_rank": 2, "anchor_distance_sec": 0.5, "canonical_frame_path": "frame.jpg"}
    result = core.chronological_frames([frame])[0]
    assert frame == {"timestamp": 1.0, "selection_rank": 2, "anchor_distance_sec": 0.5, "canonical_frame_path": "frame.jpg"}
    assert {key: result[key] for key in frame} == frame


def test_l_canonical_visual_entity_from_frame_assets_is_accounted_without_a_fake_drop():
    packet = {
        "candidates_before_reranking": [
            {"candidate_id": "a1"}, {"candidate_id": "a2"}, {"candidate_id": "a3"},
        ],
        "retained_candidates": [
            {"candidate_id": "a1"}, {"candidate_id": "a2"}, {"candidate_id": "a3"},
            {
                "candidate_id": "visual_packet",
                "candidate_type": "canonical_visual_evidence",
                "canonical_visual_frames": [{"timestamp": 1.0}],
                "source_candidate_ids": [],
            },
        ],
        "merged_source_candidates": [],
        "transformed_candidates": [],
        "actually_dropped_candidates": [],
        "dropped_visual_frames": [{"candidate_id": "frame_video_2000"}],
        "relations": [{"source_candidate_id": "a1", "target_candidate_id": "visual_packet"}],
    }
    accounting = core.candidate_identity_accounting(packet)
    assert accounting["consistent"] is True
    assert accounting["introduced_canonical_candidate_ids"] == ["visual_packet"]
    assert accounting["unique_input_candidate_count"] == 3
    assert accounting["unique_retained_candidate_count"] == 4
    assert core.candidate_count_consistent(packet)


def test_m_identity_invariants_reject_duplicates_and_retained_drop_overlap():
    packet = {
        "candidates_before_reranking": [{"candidate_id": "x"}, {"candidate_id": "x"}],
        "retained_candidates": [{"candidate_id": "x"}],
        "merged_source_candidates": [],
        "transformed_candidates": [],
        "actually_dropped_candidates": [{"candidate_id": "x"}],
    }
    accounting = core.candidate_identity_accounting(packet)
    assert accounting["consistent"] is False
    assert "input_candidate_id_duplicated" in accounting["errors"]
    assert "retained_and_dropped_overlap" in accounting["errors"]


def test_n_frames_and_relations_do_not_inflate_candidate_entity_counts():
    packet = {
        "candidates_before_reranking": [{"candidate_id": "speech"}],
        "retained_candidates": [{"candidate_id": "speech"}],
        "merged_source_candidates": [],
        "transformed_candidates": [],
        "actually_dropped_candidates": [],
        "dropped_visual_frames": [
            {"candidate_id": "frame_1"}, {"candidate_id": "frame_2"},
        ],
        "relations": [
            {"source_candidate_id": "speech", "target_candidate_id": "speech"},
        ],
    }
    accounting = core.candidate_identity_accounting(packet)
    assert accounting["consistent"] is True
    assert accounting["retained_candidate_count"] == 1
    assert accounting["excluded_entity_types"] == [
        "visual_frame_asset", "audio_clip_asset", "relation",
    ]
