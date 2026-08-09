from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.claim_level_av_sufficiency_v2.core import (
    SYSTEM_PROMPT,
    build_request_payload,
    claim_schema,
    deduplicate_claim_cache,
    normalize_result_schema,
    validate_result,
)


def packet() -> dict:
    return {
        "question_id": "q_weapon_visible",
        "question": "Was a weapon visibly present?",
        "visual_evidence": [{
            "visual_id": "v1", "timestamp_sec": 10.0,
            "caption_metadata": "EVENT: possible firearm\nUNCERTAIN: holder unclear",
            "branch_path": ["E01", "C01", "M01", "F01"],
            "image_path": "forbidden.jpg",
        }],
        "audio_evidence": [{
            "audio_node_id": "a1", "start_sec": 9.0, "end_sec": 11.0,
            "transcript": "Call an ambulance", "speaker_label": "unknown_speaker",
            "confidence": {"avg_logprob": -0.4},
            "uncertainty": {"fallback_used": False, "warnings": []},
            "provenance": {"transcript_exact_copy": True},
        }],
        "typed_av_links": [{"visual_node_id": "v1", "audio_node_id": "a1"}],
        "prior_visual_sufficiency_annotation": {"overall_status": "sufficient"},
    }


def result() -> dict:
    claim = {
        "claim_id": "c1", "claim_text": "An ambulance is requested.",
        "claim_type": "reported_event", "status": "supported",
        "support_mode": ["audio_event"], "evidence_ids": ["a1"],
        "time_range": [9.0, 11.0], "reasoning_summary": "Imperative request.",
        "confidence": "high", "answer_critical": True, "reusable": True,
        "needs_raw_visual_review": False, "review_reason": None,
        "review_target": None, "remaining_uncertainty": [],
        "rejected_inferences": ["The ambulance arrived."],
    }
    return {
        "question_id": "q_weapon_visible", "claims": [claim],
        "required_claims": ["c1"], "supported_claims": ["c1"],
        "uncertain_claims": [], "conflicted_claims": [], "missing_claims": [],
        "answerability_from_current_index": "sufficient",
        "visual_review_requests": [], "remaining_uncertainty": [],
    }


def test_audio_command_supported_claim():
    assert "command" in SYSTEM_PROMPT


def test_request_does_not_become_arrival():
    assert '"Call an ambulance" supports an ambulance request, not arrival' in SYSTEM_PROMPT


def test_clear_arrival_statement_may_support_arrival():
    assert "ambulance is here" in SYSTEM_PROMPT


def test_question_negation_rule_present():
    assert "non-question, non-negated" in SYSTEM_PROMPT


def test_asr_uncertainty_in_payload():
    p = packet()
    p["audio_evidence"][0]["confidence"]["avg_logprob"] = -0.9
    assert build_request_payload(p)["audio_evidence"][0]["asr_reliability"] == "uncertain"


def test_cross_modal_support_enum():
    assert "cross_modal" in claim_schema()["properties"]["claims"]["items"]["properties"]["support_mode"]["items"]["enum"]


def test_command_does_not_imply_cuffing():
    assert "Hands behind your back" in SYSTEM_PROMPT
    assert "completed handcuffing" in SYSTEM_PROMPT


def test_reported_weapon_does_not_bind_holder():
    assert "does not establish visual weapon presence" in SYSTEM_PROMPT


def test_supported_claim_no_review():
    assert not validate_result(result(), build_request_payload(packet()))


def test_noncritical_uncertainty_no_review_required():
    r = result()
    r["claims"][0].update(status="uncertain", answer_critical=False)
    r["supported_claims"] = []
    r["uncertain_claims"] = ["c1"]
    assert not validate_result(r, build_request_payload(packet()))


def test_unresolvable_speaker_identity_not_reviewed():
    assert "unresolvable speaker identity" in SYSTEM_PROMPT


def test_critical_binding_can_request_review():
    r = result()
    c = r["claims"][0]
    c.update(
        status="uncertain", needs_raw_visual_review=True,
        review_reason="Holder binding unclear", review_target="hand-object relation",
    )
    r["supported_claims"] = []
    r["uncertain_claims"] = ["c1"]
    r["visual_review_requests"] = [{
        "claim_id": "c1", "answer_critical": True,
        "reason": "Holder binding unclear", "target_time_range": [9.0, 11.0],
        "target_visual_question": "Is a hand directly connected to the object?",
        "candidate_existing_evidence_ids": ["v1"],
        "expected_resolution": ["confirm", "reject", "remain_uncertain"],
    }]
    assert not validate_result(r, build_request_payload(packet()))


def test_no_fixed_frame_budget():
    r = result()
    c = r["claims"][0]
    c.update(
        status="uncertain", needs_raw_visual_review=True,
        review_reason="Need 4 frames", review_target="object",
    )
    r["supported_claims"] = []
    r["uncertain_claims"] = ["c1"]
    r["visual_review_requests"] = [{
        "claim_id": "c1", "answer_critical": True, "reason": "Inspect",
        "target_time_range": [9.0, 11.0],
        "target_visual_question": "Review maximum 4 frames",
        "candidate_existing_evidence_ids": ["v1"],
        "expected_resolution": ["confirm"],
    }]
    assert "fixed frame budget forbidden" in validate_result(r, build_request_payload(packet()))


def test_no_raw_images_sent():
    payload = build_request_payload(packet())
    text = json.dumps(payload)
    assert "image_path" not in text and "forbidden.jpg" not in text


def test_one_call_contract_schema_has_one_result():
    assert claim_schema()["type"] == "object"


def test_old_outputs_excluded_from_payload():
    text = json.dumps(build_request_payload(packet()))
    assert "prior_visual_sufficiency_annotation" not in text
    assert "overall_status" not in text


def test_evidence_provenance_rejected_when_unknown():
    r = result()
    r["claims"][0]["evidence_ids"] = ["unknown"]
    assert any("invalid evidence provenance" in x for x in validate_result(r, build_request_payload(packet())))


def test_claim_cache_deterministic_deduplication():
    r1 = result()
    r2 = json.loads(json.dumps(r1))
    r2["question_id"] = "q_handcuffing"
    a = deduplicate_claim_cache([r1, r2])
    b = deduplicate_claim_cache([r1, r2])
    assert a == b and len(a["claims"]) == 1


def test_transcript_exact_copy():
    payload = build_request_payload(packet())
    assert payload["audio_evidence"][0]["exact_transcript"] == "Call an ambulance"


def test_review_request_must_be_critical():
    r = result()
    r["visual_review_requests"] = [{
        "claim_id": "c1", "answer_critical": True, "reason": "Inspect",
        "target_time_range": [9.0, 11.0], "target_visual_question": "Inspect",
        "candidate_existing_evidence_ids": ["v1"], "expected_resolution": ["confirm"],
    }]
    assert "visual review request/claim flag mismatch" in validate_result(r, build_request_payload(packet()))


def test_missing_claim_description_is_schema_normalized_without_evidence():
    r = result()
    r["missing_claims"] = ["Ambulance arrival is not established"]
    normalized, log = normalize_result_schema(r, "q_weapon_visible")
    cid = normalized["missing_claims"][0]
    claim = next(x for x in normalized["claims"] if x["claim_id"] == cid)
    assert claim["status"] == "not_found"
    assert claim["evidence_ids"] == []
    assert cid in normalized["required_claims"]
    assert log[0]["method"] == "wrap_explicit_missing_description_as_not_found_claim"


def test_explicit_review_request_synchronizes_claim_flag():
    r = result()
    r["visual_review_requests"] = [{
        "claim_id": "c1", "answer_critical": True, "reason": "Inspect binding",
        "target_time_range": [9.0, 11.0],
        "target_visual_question": "Is the relation visible?",
        "candidate_existing_evidence_ids": ["v1"],
        "expected_resolution": ["confirm", "reject", "remain_uncertain"],
    }]
    normalized, log = normalize_result_schema(r, "q_weapon_visible")
    assert normalized["claims"][0]["needs_raw_visual_review"] is True
    assert any(x["method"] == "synchronize_explicit_request_with_claim_review_flag" for x in log)
