from __future__ import annotations

import inspect

from scripts import run_task7a_preflight as runner
from src.final_qa import task7a_preflight as core


def test_a_visual_payload_frames_are_chronological_and_sequential():
    frames = [
        {"frame_path": "a.jpg", "timestamp_sec": 1.0, "presentation_order": 1, "selection_rank": 2},
        {"frame_path": "b.jpg", "timestamp_sec": 2.0, "presentation_order": 2, "selection_rank": 1},
    ]
    validation = core.validate_visual_frames(frames, runner.ROOT)
    assert validation["chronological_visual_order"] is True
    assert validation["presentation_order_sequential"] is True


def test_b_visual_packaging_uses_only_canonical_retained_frames():
    candidate = {"candidate_id": "v", "start_time": 0, "end_time": 2, "roles": ["resolver"], "canonical_visual_frames": [{"canonical_frame_path": "one.jpg", "timestamp": 1.0, "selection_rank": 1, "presentation_order": 1}]}
    packaged = runner.visual_evidence(candidate)
    assert [item["frame_path"] for item in packaged["frames"]] == ["one.jpg"]
    assert "dropped" not in str(packaged)


def test_c_d_acoustic_direct_evidence_without_real_local_clip_is_blocked_not_satisfied_by_clap():
    payload = {"case_id": "x", "answer_required_modalities": ["acoustic"], "pipeline_uncertainties": [], "evidence_groups": [{"visual_evidence": [], "speech_evidence": [], "acoustic_evidence": [{"evidence_id": "a", "audio_clip_path": None, "start_sec": 0, "end_sec": 1, "roles": ["direct_evidence"], "clap_retrieval_provenance": {"clap_similarity_score": 0.9}}], "relations": []}]}
    result, _ = runner.preflight(payload)
    assert result["status"] == "blocked"
    assert "required_modality:acoustic" in result["missing_assets"]


def test_e_speech_uses_effective_timestamp_bounds():
    candidate = {"start_time": 0, "end_time": 4, "effective_start_time": 1, "effective_end_time": 3}
    assert core.effective_bounds(candidate) == (1.0, 3.0)


def test_f_speech_packaging_never_verifies_speaker_identity_from_transcript_only():
    candidate = {"candidate_id": "s", "start_time": 0, "end_time": 1, "transcript_text": "Wait", "roles": ["trigger"], "speaker_attribution_warning": "unverified"}
    assert runner.speech_evidence(candidate)["speaker_verified"] is False


def test_g_h_response_alternatives_and_fallback_provenance_are_preserved():
    candidate = {"candidate_id": "s", "start_time": 1, "end_time": 2, "transcript_text": "Oh man", "roles": ["alternative", "plausible_response", "fallback_recovered"], "source": "local_asr_fallback"}
    packaged = runner.speech_evidence(candidate)
    assert set(packaged["roles"]) == {"alternative", "plausible_response", "fallback_recovered"}
    assert packaged["fallback_provenance"] is True


def test_i_pipeline_uncertainty_is_explicit_and_cannot_be_silently_removed():
    packet = {"answer_required_modalities": ["speech"], "retained_candidates": [], "unresolved_ambiguities": ["unresolved_speaker_attribution"], "missing_information": [], "structural_evidence_status": "questionable", "dataset_or_query_inconsistency_status": "unknown"}
    types = {item["type"] for item in core.build_pipeline_uncertainties(packet)}
    assert {"speaker_attribution_uncertainty", "semantic_uncertainty", "dataset_or_query_inconsistency_unknown"}.issubset(types)


def test_j_leakage_audit_rejects_weak_references_gold_and_review_notes():
    audit = core.payload_leakage_audit({"weak_reference_interval": [1, 2], "note": "human review conclusion", "ground_truth_answer": "x"})
    assert audit["leakage_check_passed"] is False
    assert len(audit["forbidden_matches"]) >= 3


def test_k_l_required_assets_and_evidence_ids_are_validated():
    payload = {"case_id": "x", "answer_required_modalities": ["speech"], "pipeline_uncertainties": [], "evidence_groups": [{"visual_evidence": [], "speech_evidence": [{"evidence_id": "same", "transcript": "x", "start_sec": 0, "end_sec": 1, "timestamp_validity": "valid"}, {"evidence_id": "same", "transcript": "y", "start_sec": 1, "end_sec": 2, "timestamp_validity": "valid"}], "acoustic_evidence": [], "relations": []}]}
    result, _ = runner.preflight(payload)
    assert result["status"] == "blocked"
    assert result["consistency"]["unique_evidence_ids"] is False


def test_m_planned_calls_are_one_and_actual_model_calls_are_zero_in_preflight_contract():
    assert runner.MODEL_REQUIREMENTS["single_request_per_question_preferred"] is True
    source = inspect.getsource(runner) + inspect.getsource(core)
    for token in ("import whisper", "anthropic", "openai", "requests", "cv2", "librosa"):
        assert token not in source.lower()


def test_n_source_integrity_contract_tracks_task6_v1_2():
    assert runner.INPUT.name == "task6_evidence_packets.jsonl"
    assert any("task6_v1_2" in str(path) for path in runner.FROZEN_PRIOR_OUTPUTS)
