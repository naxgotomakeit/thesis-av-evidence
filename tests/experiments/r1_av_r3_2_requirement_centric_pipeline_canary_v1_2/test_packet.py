from experiments.r1_av_r3_2_requirement_centric_pipeline_canary_v1_2.core import deterministic_no_direct_result, direct_packet


def test_contextual_evidence_is_excluded_from_model_packet():
    packet={"question_id":"q","question":"x","requirements":[],"evidence":[],"contextual_evidence":[{"evidence_id":"a"}],"existing_uncertainty":[],"rejected_bindings":[]}
    value=direct_packet(packet)
    assert "contextual_evidence" not in value
    assert value["evidence"]==[]


def test_no_direct_temporal_requirement_is_deterministically_not_found():
    packet={"question_id":"q","requirements":[{"requirement_id":"q::temporal_order","description":"temporal_order"}]}
    claim=deterministic_no_direct_result(packet)["claims"][0]
    assert claim["status"]=="not_found"
    assert claim["temporal_grounding"]["relation"]=="unknown"
