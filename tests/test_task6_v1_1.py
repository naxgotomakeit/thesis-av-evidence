from __future__ import annotations

import inspect

from src.retrieval import task6_v1_1 as core


def candidate(candidate_id: str, modality: str, roles: list[str], start: float, end: float) -> dict:
    return {"candidate_id": candidate_id, "modality": modality, "roles": roles, "start_time": start, "end_time": end}


def test_a_anchor_visual_relations_have_distinct_direction_and_basis():
    anchor = candidate("a", "speech", ["temporal_anchor"], 1, 2)
    visual = candidate("v", "visual", ["resolver"], 1.5, 3)
    relations = core.corrected_relations([anchor, visual], "identify_object", [])
    temporal = next(item for item in relations if item["source_candidate_id"] == "a")
    resolver = next(item for item in relations if item["relation_type"] == "resolves")
    assert temporal["target_candidate_id"] == "v"
    assert temporal["relation_basis"] == "temporal_overlap"
    assert resolver["source_candidate_id"] == "v"
    assert resolver["target_candidate_id"] == "a"
    assert resolver["relation_basis"] == "operation_role"
    assert resolver["relation_confidence"] == "role_based"


def test_b_supporting_modality_does_not_break_required_retention():
    speech = candidate("s", "speech", ["trigger", "plausible_response"], 0, 1)
    dropped = [{"candidate_id": "a", "modality": "acoustic", "reason": "redundant_supporting_acoustic_candidate"}]
    result = core.classify_modalities(["speech", "acoustic"], [speech], {"a": {"acoustic_evidence_role": "supporting"}}, dropped)
    assert result["answer_required_modalities"] == ["speech"]
    assert result["supporting_modalities"] == ["acoustic"]
    assert result["dropped_supporting_modalities"] == ["acoustic"]
    assert set(result["answer_required_modalities"]).issubset(result["retained_modalities"])


def test_c_merged_sources_are_not_actually_dropped():
    left, right = candidate("m1", "visual", [], 0, 2), candidate("m2", "visual", [], 1, 3)
    canonical = {**candidate("canonical", "visual", ["resolver"], 0, 3), "candidate_type": "canonical_visual_evidence"}
    packet = {"candidates_before_reranking": [left, right], "retained_candidates": [canonical], "dropped_candidates": [{"candidate_id": "m1", "reason": "overlapping_visual_window_merged"}, {"candidate_id": "m2", "reason": "overlapping_visual_window_merged"}]}
    result = core.classify_candidate_accounting(packet)
    assert {item["source_candidate_id"] for item in result["merged_source_candidates"]} == {"m1", "m2"}
    assert result["actually_dropped_candidates"] == []


def test_d_transformed_micro_window_is_not_a_genuine_drop():
    micro = candidate("m", "visual", [], 0, 2)
    canonical = {**candidate("canonical", "visual", ["resolver"], 0, 2), "candidate_type": "canonical_visual_evidence"}
    packet = {"candidates_before_reranking": [micro], "retained_candidates": [canonical], "dropped_candidates": [{"candidate_id": "m", "reason": "visual_micro_window_represented_by_canonical_frames"}]}
    result = core.classify_candidate_accounting(packet)
    assert result["transformed_candidates"][0]["source_candidate_id"] == "m"
    assert result["actually_dropped_candidates"] == []


def test_e_dropped_frame_does_not_count_as_candidate_drop():
    packet = {"candidates_before_reranking": [], "retained_candidates": [], "dropped_candidates": [{"candidate_id": "frame_video_1000", "reason": "visual_frame_budget_anchor_nearest_selection"}]}
    result = core.classify_candidate_accounting(packet)
    assert len(result["dropped_visual_frames"]) == 1
    assert result["actually_dropped_candidates"] == []


def test_f_visual_selection_order_is_separate_from_chronological_presentation():
    frames = [{"timestamp": 3.5}, {"timestamp": 4.0}, {"timestamp": 3.0}, {"timestamp": 4.5}]
    anchor = candidate("a", "speech", ["temporal_anchor"], 3.65, 3.79)
    ordered = core.order_visual_frames(frames, [anchor])
    assert [item["timestamp"] for item in ordered] == [3.0, 3.5, 4.0, 4.5]
    assert [item["selection_rank"] for item in ordered] == [3, 1, 2, 4]
    assert [item["presentation_order"] for item in ordered] == [1, 2, 3, 4]


def test_g_speech_ambiguity_relations_are_preserved():
    previous = [{"source_candidate_id": "t", "target_candidate_id": "r1", "relation_type": "response_to", "temporal_gap_sec": 0.2, "relation_confidence": "temporal_only", "relation_basis": "transcript_sequence"}, {"source_candidate_id": "r1", "target_candidate_id": "r2", "relation_type": "alternative_to", "temporal_gap_sec": 0.3, "relation_confidence": "temporal_only", "relation_basis": "transcript_sequence"}]
    result = core.corrected_relations([candidate("t", "speech", ["trigger"], 0, 1), candidate("r1", "speech", ["plausible_response"], 1.2, 2), candidate("r2", "speech", ["plausible_response"], 2.2, 3)], "measure_delay", previous)
    assert {item["relation_type"] for item in result} == {"response_to", "alternative_to"}


def test_h_fallback_provenance_relation_is_preserved():
    previous = [{"source_candidate_id": "fallback", "target_candidate_id": "speech_branch_missing_evidence", "relation_type": "fallback_recovery_for", "temporal_gap_sec": 0.0, "relation_confidence": "provenance", "relation_basis": "fallback_provenance"}]
    result = core.corrected_relations([candidate("fallback", "speech", ["direct_evidence", "fallback_recovered"], 1, 2)], "count_occurrences", previous)
    assert result == previous


def test_i_weak_reference_is_not_part_of_correction_functions():
    source = inspect.getsource(core.corrected_relations) + inspect.getsource(core.classify_modalities) + inspect.getsource(core.order_visual_frames)
    assert "weak_reference" not in source
    assert "posthoc" not in source


def test_j_zero_model_or_retrieval_dependencies():
    source = inspect.getsource(core).lower()
    for token in ("import whisper", "anthropic", "openai", "cv2", "librosa", "clip.load"):
        assert token not in source
