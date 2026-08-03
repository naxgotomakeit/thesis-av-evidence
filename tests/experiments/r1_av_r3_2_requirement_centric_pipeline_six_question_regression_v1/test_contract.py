from experiments.r1_av_r3_2_requirement_centric_pipeline_six_question_regression_v1.core import (
    direct_packet,
    project_v3_2_1_temporal_extension,
    validate_requirement_scoped_citations,
)


def test_direct_packet_preserves_requirement_specific_policy():
    packet = {
        "question_id": "q", "question": "?", "requirements": [{"requirement_id": "r"}],
        "evidence": [{"evidence_id": "e1"}], "contextual_evidence": [{"evidence_id": "c1"}],
        "requirement_evidence_policy": {"r": {"direct_support_evidence_ids": ["e1"], "contextual_evidence_ids": ["c1"]}},
        "existing_uncertainty": [], "rejected_bindings": [],
    }
    result = direct_packet(packet)
    assert result["allowed_evidence_ids_by_requirement"] == {"r": ["e1"]}
    assert "contextual_evidence" not in result


def test_cross_requirement_citation_fails():
    packet = {"allowed_evidence_ids_by_requirement": {"r1": ["e1"], "r2": ["e2"]}}
    result = {"claims": [{"requirement_id": "r1", "supporting_evidence_ids": ["e2"]}]}
    assert validate_requirement_scoped_citations(result, packet)


def test_non_order_temporal_extension_is_schema_only_projection():
    result = {"claims": [{"requirement_id": "r", "status": "supported", "rationale": "same", "temporal_grounding": {"relation": "same_time"}}]}
    packet = {"requirements": [{"requirement_id": "r", "description": "injury_timestamp"}]}
    projected, audit = project_v3_2_1_temporal_extension(result, packet)
    assert "temporal_grounding" not in projected["claims"][0]
    assert result["claims"][0]["temporal_grounding"] == {"relation": "same_time"}
    assert audit["semantic_fields_changed"] == 0
