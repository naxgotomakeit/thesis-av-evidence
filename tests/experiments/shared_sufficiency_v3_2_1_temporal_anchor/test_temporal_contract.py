from __future__ import annotations

import copy

from experiments.shared_sufficiency_v3_2_1_temporal_anchor.core import project_statuses, validate_result
from experiments.shared_sufficiency_v3_2_contract.core import validate_result as validate_v3_2


def temporal_packet(kind: str = "visual_caption", overlap: bool = False) -> dict:
    first = [1.0, 3.0] if overlap else [1.0, 2.0]
    second = [2.0, 4.0] if overlap else [3.0, 4.0]
    return {
        "question_id": "q",
        "question": "Did A happen before B?",
        "requirements": [{"requirement_id": "q::temporal_order", "description": "temporal_order"}],
        "evidence": [
            {"evidence_id": "a", "evidence_type": kind, "timestamp": first, "source_content": "A source", "source_provenance": {}},
            {"evidence_id": "b", "evidence_type": kind, "timestamp": second, "source_content": "B source", "source_provenance": {}},
        ],
        "existing_uncertainty": [], "rejected_bindings": [],
    }


def anchor(eid: str, interval: list[float], anchor_type: str = "event_occurrence") -> dict:
    return {"evidence_id": eid, "anchor_type": anchor_type, "start_sec": interval[0], "end_sec": interval[1], "rationale": "Direct role classification."}


def temporal_result(
    *, relation: str = "before", a: dict | None = None, b: dict | None = None,
    status: str = "supported", kind: str = "visual_caption",
) -> dict:
    ids = [] if status == "not_found" else [x["evidence_id"] for x in (a, b) if x is not None]
    return {
        "question_id": "q",
        "claims": [{
            "requirement_id": "q::temporal_order", "status": status,
            "supporting_evidence_ids": ids,
            "evidence_types": [] if not ids else [kind],
            "support_scope": "event_order" if status == "supported" else ("no_direct_support" if status == "not_found" else "indirect_context"),
            "direct_support": status == "supported", "rationale": "Temporal assessment.",
            "contradiction_pairs": [],
            "temporal_grounding": {"relation": relation, "event_a_anchor": a, "event_b_anchor": b},
        }],
    }


def valid_before(kind: str = "visual_caption") -> tuple[dict, dict]:
    packet = temporal_packet(kind)
    return packet, temporal_result(a=anchor("a", [1.0, 2.0]), b=anchor("b", [3.0, 4.0]), kind=kind)


def test_01_occurrence_anchors_support_before() -> None:
    packet, result = valid_before()
    assert validate_result(result, packet) == []


def test_02_occurrence_anchors_support_after() -> None:
    packet = temporal_packet()
    result = temporal_result(relation="after", a=anchor("b", [3.0, 4.0]), b=anchor("a", [1.0, 2.0]))
    assert validate_result(result, packet) == []


def test_03_plan_or_command_cannot_support_order() -> None:
    packet, result = valid_before()
    result["claims"][0]["temporal_grounding"]["event_b_anchor"]["anchor_type"] = "plan_or_command"
    assert any("event_occurrence" in e for e in validate_result(result, packet))


def test_04_mention_or_unclear_cannot_support_order() -> None:
    packet, result = valid_before()
    result["claims"][0]["temporal_grounding"]["event_b_anchor"]["anchor_type"] = "mention_or_unclear"
    assert any("event_occurrence" in e for e in validate_result(result, packet))


def test_05_state_confirmation_is_not_exact_occurrence_anchor() -> None:
    packet, result = valid_before()
    result["claims"][0]["temporal_grounding"]["event_a_anchor"]["anchor_type"] = "event_state_confirmation"
    assert any("event_occurrence" in e for e in validate_result(result, packet))


def test_06_asr_recording_timestamp_alone_cannot_support_order() -> None:
    packet, result = valid_before("audio_asr")
    result["claims"][0]["temporal_grounding"]["event_a_anchor"]["anchor_type"] = "mention_or_unclear"
    result["claims"][0]["temporal_grounding"]["event_b_anchor"]["anchor_type"] = "mention_or_unclear"
    assert validate_result(result, packet)


def test_07_missing_event_a_anchor_fails_supported() -> None:
    packet = temporal_packet()
    result = temporal_result(a=None, b=anchor("b", [3.0, 4.0]))
    assert any("both anchors" in e for e in validate_result(result, packet))


def test_08_missing_event_b_anchor_fails_supported() -> None:
    packet = temporal_packet()
    result = temporal_result(a=anchor("a", [1.0, 2.0]), b=None)
    assert any("both anchors" in e for e in validate_result(result, packet))


def test_09_unknown_evidence_id_fails() -> None:
    packet = temporal_packet()
    result = temporal_result(a=anchor("a", [1.0, 2.0]), b=anchor("unknown", [3.0, 4.0]))
    assert any("unknown" in e or "invalid supporting" in e for e in validate_result(result, packet))


def test_10_model_invented_timestamp_fails() -> None:
    packet, result = valid_before()
    result["claims"][0]["temporal_grounding"]["event_a_anchor"]["start_sec"] = 0.5
    assert any("exactly match" in e for e in validate_result(result, packet))


def test_11_overlap_cannot_support_strict_before() -> None:
    packet = temporal_packet(overlap=True)
    result = temporal_result(a=anchor("a", [1.0, 3.0]), b=anchor("b", [2.0, 4.0]))
    assert any("do not establish before" in e for e in validate_result(result, packet))


def test_12_relation_must_match_interval_order() -> None:
    packet, result = valid_before()
    result["claims"][0]["temporal_grounding"]["relation"] = "after"
    assert any("do not establish after" in e for e in validate_result(result, packet))


def test_13_uncertain_may_retain_partial_anchor() -> None:
    packet = temporal_packet()
    partial = anchor("a", [1.0, 2.0], "event_state_confirmation")
    result = temporal_result(relation="unknown", a=partial, b=None, status="uncertain")
    assert validate_result(result, packet) == []


def test_14_not_found_may_contain_null_anchors() -> None:
    packet = temporal_packet()
    result = temporal_result(relation="unknown", a=None, b=None, status="not_found")
    assert validate_result(result, packet) == []


def test_15_non_temporal_v3_2_claim_remains_compatible() -> None:
    packet = temporal_packet()
    packet["requirements"] = [{"requirement_id": "q::object", "description": "object_presence"}]
    result = {"question_id": "q", "claims": [{
        "requirement_id": "q::object", "status": "supported", "supporting_evidence_ids": ["a"],
        "evidence_types": ["visual_caption"], "support_scope": "caption_semantic_observation",
        "direct_support": True, "rationale": "Caption explicitly states it.", "contradiction_pairs": [],
    }]}
    assert validate_result(result, packet) == validate_v3_2(result, packet) == []


def test_16_status_projection_is_unchanged_and_deterministic() -> None:
    packet, result = valid_before()
    projected, audit = project_statuses(result, packet)
    assert projected["supported_claims"] == ["q::temporal_order"]
    assert projected["uncertain_claims"] == projected["conflicted_claims"] == projected["not_found_claims"] == []
    assert audit["semantic_unchanged"] and audit["lists_disjoint_complete"]
