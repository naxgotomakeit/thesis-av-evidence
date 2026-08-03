from __future__ import annotations

import copy

import pytest

from experiments.shared_sufficiency_v3_2_contract.core import project, validate_result


REQ = {"requirement_id": "q::fact", "description": "fact", "text": "Establish the fact."}


def packet(kind: str = "detector_observation", *, description: str = "fact") -> dict:
    requirement = {**REQ, "description": description}
    return {
        "question_id": "q",
        "question": "Question?",
        "requirements": [requirement],
        "evidence": [
            {
                "evidence_id": "e1",
                "evidence_type": kind,
                "timestamp": [1.0, 2.0],
                "source_content": "Exact source one",
                "source_provenance": {},
            },
            {
                "evidence_id": "e2",
                "evidence_type": kind,
                "timestamp": [3.0, 4.0],
                "source_content": "Exact source two",
                "source_provenance": {},
            },
        ],
        "existing_uncertainty": [],
        "rejected_bindings": [],
    }


def claim(
    *, status: str = "supported", evidence_ids: list[str] | None = None,
    evidence_types: list[str] | None = None, scope: str = "detector_object_observation",
    direct: bool = True, pairs: list[dict] | None = None,
) -> dict:
    return {
        "question_id": "q",
        "claims": [{
            "requirement_id": "q::fact",
            "status": status,
            "supporting_evidence_ids": ["e1"] if evidence_ids is None else evidence_ids,
            "evidence_types": ["detector_observation"] if evidence_types is None else evidence_types,
            "support_scope": scope,
            "direct_support": direct,
            "rationale": "Exact direct evidence.",
            "contradiction_pairs": [] if pairs is None else pairs,
        }],
    }


def test_01_detector_is_representable_without_visual_caption() -> None:
    assert validate_result(claim(), packet()) == []


def test_02_visual_caption_remains_direct_semantic_evidence() -> None:
    assert validate_result(
        claim(evidence_types=["visual_caption"], scope="caption_semantic_observation"),
        packet("visual_caption"),
    ) == []


def test_03_reviewed_visual_frame_is_distinct_from_caption() -> None:
    assert validate_result(
        claim(evidence_types=["reviewed_visual_frame"], scope="reviewed_visual_confirmation"),
        packet("reviewed_visual_frame"),
    ) == []


def test_04_embedding_signal_cannot_be_direct_support() -> None:
    errors = validate_result(
        claim(evidence_types=["embedding_retrieval_signal"], scope="indirect_context"),
        packet("embedding_retrieval_signal"),
    )
    assert any("navigation/reference" in error or "direct cited" in error for error in errors)


def test_05_unreviewed_fine_reference_cannot_be_direct_support() -> None:
    errors = validate_result(
        claim(evidence_types=["fine_frame_reference"], scope="reviewed_visual_confirmation"),
        packet("fine_frame_reference"),
    )
    assert any("navigation/reference" in error or "capability" in error for error in errors)


def test_06_structural_map_cannot_enter_evidence() -> None:
    with pytest.raises(ValueError, match="forbidden/unknown evidence type"):
        validate_result(
            claim(evidence_types=["structural_map"]), packet("structural_map")
        )


def test_07_audio_cannot_support_visual_visibility() -> None:
    errors = validate_result(
        claim(evidence_types=["audio_asr"], scope="reviewed_visual_confirmation"),
        packet("audio_asr"),
    )
    assert any("capability" in error for error in errors)


def conflict_pair(left: list[str], right: list[str], rationale: str = "The two sources contradict each other and cannot both be true.") -> dict:
    return {
        "supports_requirement": left,
        "opposes_requirement": right,
        "incompatibility_rationale": rationale,
    }


def conflicted(pairs: list[dict], *, ids: list[str] | None = None) -> dict:
    return claim(
        status="conflicted", evidence_ids=["e1", "e2"] if ids is None else ids,
        evidence_types=["visual_caption"], scope="indirect_context", direct=False,
        pairs=pairs,
    )


def test_08_conflicted_with_empty_pairs_fails() -> None:
    assert any("requires explicit" in error for error in validate_result(conflicted([]), packet("visual_caption")))


def test_09_conflicted_with_one_sided_pair_fails() -> None:
    errors = validate_result(conflicted([conflict_pair(["e1"], [])]), packet("visual_caption"))
    assert any("both valid nonempty sides" in error for error in errors)


def test_10_conflicted_with_invalid_id_fails() -> None:
    errors = validate_result(conflicted([conflict_pair(["e1"], ["absent"])]), packet("visual_caption"))
    assert any("both valid nonempty sides" in error for error in errors)


def test_11_explicit_genuine_contradiction_passes() -> None:
    assert validate_result(
        conflicted([conflict_pair(["e1"], ["e2"])]), packet("visual_caption")
    ) == []


def test_12_insufficiency_is_not_a_genuine_contradiction() -> None:
    errors = validate_result(
        conflicted([conflict_pair(["e1"], ["e2"], "The timing is ambiguous and cannot confirm the event.")]),
        packet("visual_caption"),
    )
    assert any("genuine incompatibility" in error for error in errors)


def test_13_timestamp_recording_order_cannot_support_event_order() -> None:
    errors = validate_result(
        claim(evidence_types=["audio_asr"], scope="recording_order"),
        packet("audio_asr", description="temporal_order"),
    )
    assert any("event-order" in error for error in errors)


def test_14_mention_order_remains_representable() -> None:
    assert validate_result(
        claim(evidence_types=["audio_asr"], scope="mention_order"),
        packet("audio_asr", description="mention_sequence"),
    ) == []


def test_15_every_requirement_must_be_assessed_exactly_once() -> None:
    duplicated = claim()
    duplicated["claims"].append(copy.deepcopy(duplicated["claims"][0]))
    assert any("exactly once" in error for error in validate_result(duplicated, packet()))
    omitted = {"question_id": "q", "claims": []}
    assert any("exactly once" in error for error in validate_result(omitted, packet()))


def test_16_projection_is_deterministic_and_disjoint() -> None:
    projected, audit = project(claim(), packet())
    assert projected["supported_claims"] == ["q::fact"]
    assert projected["uncertain_claims"] == []
    assert projected["conflicted_claims"] == []
    assert projected["not_found_claims"] == []
    assert audit["semantic_unchanged"] and audit["lists_disjoint"]


def test_17_audio_mention_cannot_directly_support_physical_event_occurrence() -> None:
    errors = validate_result(
        claim(evidence_types=["audio_asr"], scope="audible_statement"),
        packet("audio_asr", description="medical_event"),
    )
    assert any("physical or visual requirement" in error for error in errors)
