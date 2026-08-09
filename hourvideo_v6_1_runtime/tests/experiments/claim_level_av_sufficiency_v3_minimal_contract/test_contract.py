from __future__ import annotations

import inspect
from copy import deepcopy

import pytest

from src.experiments.claim_level_av_sufficiency_v2.core import validate_result
from src.experiments.claim_level_av_sufficiency_v3_minimal_contract import core


def requirements() -> list[dict]:
    return [
        {"requirement_id": "req_a", "description": "first", "answer_critical": True},
        {"requirement_id": "req_b", "description": "second", "answer_critical": False},
    ]


def packet() -> dict:
    return {
        "question_id": "q_fixture",
        "question": "What happened?",
        "question_type": "fixture",
        "selected_phase_ids": [],
        "visual_evidence": [{
            "evidence_id": "v1", "timestamp": [10.0, 11.0],
            "caption": "fixture", "caption_uncertainty": [],
            "source_provenance": "fixture",
        }],
        "audio_evidence": [{
            "evidence_id": "a1", "timestamp": [9.5, 12.0],
            "exact_transcript": "fixture", "speaker": "unknown_speaker",
            "asr_reliability": "reliable", "source_provenance": "fixture",
        }],
        "typed_av_links": [], "existing_uncertainty": [], "rejected_bindings": [],
    }


def claim(**changes) -> dict:
    value = {
        "claim_id": "c1", "requirement_ids": ["req_a"],
        "claim_text": "A generic event is supported.", "claim_type": "state",
        "status": "supported", "support_mode": ["visual_caption"],
        "evidence_ids": ["v1"], "confidence": "high",
        "remaining_uncertainty": [], "rejected_inferences": [],
        "claim_event_time": None,
    }
    value.update(changes)
    return value


def semantic(*claims: dict) -> dict:
    return {"question_id": "q_fixture", "claims": list(claims)}


def v3_input() -> dict:
    return core.build_v3_input(packet(), requirements())


def test_deterministic_status_list_projection() -> None:
    payload = semantic(
        claim(),
        claim(claim_id="c2", requirement_ids=["req_b"], status="uncertain"),
    )
    one, _ = core.project_semantic_payload(payload, v3_input())
    two, _ = core.project_semantic_payload(payload, v3_input())
    assert one == two
    assert one["supported_claims"] == ["c1"]
    assert one["uncertain_claims"] == ["c2"]


def test_valid_and_unknown_requirement_ids() -> None:
    assert core.validate_semantic_payload(semantic(claim()), v3_input()) == [
        "requirements without claims: req_b"
    ]
    bad = semantic(claim(requirement_ids=["req_unknown"]))
    errors = core.validate_semantic_payload(bad, v3_input())
    assert any("requirement_ids" in error for error in errors)


def test_evidence_envelope_is_separate_from_event_time() -> None:
    payload = semantic(claim(evidence_ids=["v1", "a1"], support_mode=["cross_modal"], requirement_ids=["req_a", "req_b"]))
    projected, _ = core.project_semantic_payload(payload, v3_input())
    assert projected["claims"][0]["evidence_time_envelope"] == [9.5, 12.0]
    assert projected["claims"][0]["claim_event_time"] is None


def test_reversed_claim_event_time_is_rejected_not_reordered() -> None:
    errors = core.validate_semantic_payload(
        semantic(claim(requirement_ids=["req_a", "req_b"], claim_event_time=[12.0, 10.0])),
        v3_input(),
    )
    assert any("reversed claim_event_time" in error for error in errors)


def test_review_candidates_are_deterministic_and_candidate_only() -> None:
    payload = semantic(claim(requirement_ids=["req_a", "req_b"], status="uncertain"))
    projected, _ = core.project_semantic_payload(payload, v3_input())
    assert projected["review_candidates"] == [{
        "claim_id": "c1", "requirement_ids": ["req_a", "req_b"],
        "candidate_existing_evidence_ids": ["v1"],
        "target_time_range": [10.0, 11.0],
        "available_evidence_modality": ["visual"], "candidate_only": True,
    }]


def test_semantic_claim_hashes_unchanged_by_projection() -> None:
    payload = semantic(claim(requirement_ids=["req_a", "req_b"]))
    before = deepcopy(payload)
    _, audit = core.project_semantic_payload(payload, v3_input())
    assert payload == before
    assert audit["semantic_claims_deep_equal"]
    assert audit["semantic_claims_sha256_before"] == audit["semantic_claims_sha256_after"]


def test_legacy_adapter_passes_old_validator() -> None:
    payload = semantic(claim(requirement_ids=["req_a", "req_b"]))
    projected, _ = core.project_semantic_payload(payload, v3_input())
    legacy = core.build_legacy_adapter_view(projected)
    assert validate_result(legacy, packet()) == []


def test_no_fuzzy_requirement_text_matching() -> None:
    bad = semantic(claim(requirement_ids=["REQ A"]))
    errors = core.validate_semantic_payload(bad, v3_input())
    assert any("requirement_ids" in error for error in errors)


def test_no_question_event_or_video_specific_rules() -> None:
    source = inspect.getsource(core)
    for forbidden in ("q_weapon", "q_medical", "q_handcuff", "540772226"):
        assert forbidden not in source


def test_forbidden_legacy_fields_fail_semantic_validation() -> None:
    value = claim(requirement_ids=["req_a", "req_b"])
    value["answer_critical"] = True
    errors = core.validate_semantic_payload(semantic(value), v3_input())
    assert any("forbidden semantic fields" in error for error in errors)


def test_v3_prompt_keeps_derived_fields_out_of_model_output() -> None:
    prompt = " ".join(core.V3_SYSTEM_PROMPT.split())
    assert "classification lists" in prompt
    assert "review-control fields" in prompt
    assert "Do not answer" in prompt


def test_old_source_hash_unchanged_by_read_only_audit() -> None:
    path = inspect.getsourcefile(validate_result)
    assert path is not None
    before = core.file_sha256(core.Path(path))
    _ = inspect.getsource(validate_result)
    assert core.file_sha256(core.Path(path)) == before
