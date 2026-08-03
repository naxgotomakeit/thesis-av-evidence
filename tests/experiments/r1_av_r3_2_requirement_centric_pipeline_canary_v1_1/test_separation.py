from experiments.r1_av_r3_2_requirement_centric_pipeline_canary_v1_1.core import classify_evidence, keyed_provider_schema, normalize_provider_sentinels


def test_audio_is_contextual_for_physical_requirements():
    assert classify_evidence("r3_2","major_actions","audio_asr")=="contextual"


def test_caption_is_direct_semantic_candidate():
    assert classify_evidence("r3_2","major_actions","visual_caption")=="direct"


def test_detector_action_is_contextual():
    assert classify_evidence("r1_av","major_actions","detector_observation")=="contextual"


def test_empty_enum_sentinel_normalization_only_removes_sentinel():
    result={"claims":[{"supporting_evidence_ids":["__no_direct_evidence_available__"],"contextual_evidence_ids":["real"]}]}
    normalized,audit=normalize_provider_sentinels(result)
    assert normalized["claims"][0]["supporting_evidence_ids"]==[]
    assert normalized["claims"][0]["contextual_evidence_ids"]==["real"]
    assert audit["semantic_evidence_added"]==0


def test_keyed_schema_reuses_normal_and_temporal_definitions():
    packet={"requirements":[{"requirement_id":"q::a","description":"major_actions"},{"requirement_id":"q::t","description":"temporal_order"}]}
    schema=keyed_provider_schema(packet)
    props=schema["properties"]["assessments"]["properties"]
    assert props["q::a"]["$ref"]=="#/$defs/normal"
    assert props["q::t"]["$ref"]=="#/$defs/temporal"
