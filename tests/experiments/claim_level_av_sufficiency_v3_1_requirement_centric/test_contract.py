from __future__ import annotations

import inspect

from src.experiments.claim_level_av_sufficiency_v2.core import validate_result
from src.experiments.claim_level_av_sufficiency_v3_1_requirement_centric import core


def v3_input():
    return {
        "question_id": "q_fixture", "question": "fixture", "question_type": "fixture",
        "requirements": [
            {"requirement_id": "req_a", "answer_critical": True},
            {"requirement_id": "req_b", "answer_critical": False},
        ],
        "visual_evidence": [{"evidence_id": "v1", "timestamp": [1.0, 2.0]}],
        "audio_evidence": [{"evidence_id": "a1", "timestamp": [1.5, 3.0]}],
        "typed_av_links": [], "existing_uncertainty": [], "rejected_bindings": [],
        "selected_phase_ids": [],
    }


def assessment(rid="req_a", cid="c1", status="supported", **changes):
    row = {
        "requirement_id": rid, "claim_id": cid, "claim_text": "A generic claim.",
        "claim_type": "state", "status": status,
        "support_mode": ["visual_caption"] if status != "not_found" else [],
        "evidence_ids": ["v1"] if status != "not_found" else [],
        "confidence": "high" if status != "not_found" else "low",
        "remaining_uncertainty": [], "rejected_inferences": [],
        "claim_event_time": None,
    }
    row.update(changes)
    return row


def payload(rows=None, auxiliary=None):
    return {
        "question_id": "q_fixture",
        "requirement_assessments": rows or [assessment(), assessment("req_b", "c2")],
        "auxiliary_claims": auxiliary or [],
    }


def legacy_packet():
    return {
        "question_id": "q_fixture", "visual_evidence": [{"evidence_id": "v1"}],
        "audio_evidence": [{"evidence_id": "a1"}],
    }


def test_exact_requirement_coverage_passes():
    errors, audit = core.validate_payload(payload(), v3_input())
    assert errors == [] and audit["exact_set_equality"]


def test_missing_requirement_fails():
    errors, _ = core.validate_payload(payload([assessment()]), v3_input())
    assert any("missing requirement" in error for error in errors)


def test_duplicate_requirement_fails():
    errors, _ = core.validate_payload(payload([assessment(), assessment(cid="c2")]), v3_input())
    assert any("duplicate requirement" in error for error in errors)


def test_unknown_requirement_fails():
    errors, _ = core.validate_payload(payload([assessment(), assessment("unknown", "c2")]), v3_input())
    assert any("unknown requirement" in error for error in errors)


def test_model_generated_not_found_assessment_passes():
    value = payload([assessment(status="not_found"), assessment("req_b", "c2")])
    assert core.validate_payload(value, v3_input())[0] == []


def test_auxiliary_claim_cannot_satisfy_requirement():
    auxiliary = assessment("req_b", "aux")
    auxiliary.pop("requirement_id")
    errors, audit = core.validate_payload(payload([assessment()], [auxiliary]), v3_input())
    assert any("missing requirement" in error for error in errors)
    assert not audit["auxiliary_claims_count_toward_coverage"]


def test_support_mode_is_array():
    errors, _ = core.validate_payload(payload([assessment(support_mode="visual_caption"), assessment("req_b", "c2")]), v3_input())
    assert any("support_mode array" in error for error in errors)


def test_confidence_taxonomy_is_unchanged():
    errors, _ = core.validate_payload(payload([assessment(confidence=0.5), assessment("req_b", "c2")]), v3_input())
    assert any("invalid confidence" in error for error in errors)


def test_reversed_claim_event_time_fails():
    errors, _ = core.validate_payload(payload([assessment(claim_event_time=[2.0, 1.0]), assessment("req_b", "c2")]), v3_input())
    assert any("reversed claim_event_time" in error for error in errors)


def test_projection_is_deterministic_and_semantics_unchanged():
    one, audit = core.project_payload(payload(), v3_input())
    two, _ = core.project_payload(payload(), v3_input())
    assert one == two
    assert audit["assessment_semantics_unchanged"]


def test_legacy_adapter_accepts_not_found_without_sentinel():
    source = payload([assessment(status="not_found"), assessment("req_b", "c2")])
    projected, _ = core.project_payload(source, v3_input())
    legacy = core.build_legacy_view(projected)
    assert validate_result(legacy, legacy_packet()) == []
    assert len(legacy["claims"]) == 2
    assert legacy["claims"][0]["claim_id"] == "c1"


def test_no_fuzzy_or_case_specific_rules():
    source = inspect.getsource(core)
    for forbidden in ("fuzzy", "q_weapon", "540772226", "weapon_holder_role"):
        assert forbidden not in source.lower()


def test_old_v3_import_is_read_only_contract_reuse():
    source = inspect.getsource(core)
    assert "claim_level_av_sufficiency_v3_minimal_contract" in source
    assert "write_text" not in source.split("def run_canary", 1)[0]
