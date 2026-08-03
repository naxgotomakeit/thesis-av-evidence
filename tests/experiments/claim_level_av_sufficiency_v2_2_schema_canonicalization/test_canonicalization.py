import copy
import pytest
from src.experiments.claim_level_av_sufficiency_v2_2_schema_canonicalization.core import canonicalize_v2_2


def base():
    claim={"claim_id":"c1","claim_text":"A visible event","claim_type":"state","status":"supported","support_mode":[],"evidence_ids":[],"time_range":[],"reasoning_summary":"r","confidence":"medium","answer_critical":True,"reusable":True,"needs_raw_visual_review":False,"review_reason":None,"review_target":None,"remaining_uncertainty":[],"rejected_inferences":[]}
    return {"question_id":"q","claims":[claim],"required_claims":["c1"],"supported_claims":[],"uncertain_claims":["c1"],"conflicted_claims":[],"missing_claims":[],"answerability_from_current_index":"partial","visual_review_requests":[],"remaining_uncertainty":[]}


def test_status_projection_and_semantics():
    raw=base(); out,a=canonicalize_v2_2(raw)
    assert out["supported_claims"]==["c1"] and out["claims"][0]["status"]=="supported"
    assert a["claim_semantics_unchanged"]


def test_unique_exact_required_text_maps():
    raw=base(); raw["required_claims"]=["  A visible   event "]
    out,a=canonicalize_v2_2(raw)
    assert out["required_claims"]==["c1"] and not a["errors"]


def test_nonexact_required_text_fails_without_guessing():
    raw=base(); raw["required_claims"]=["A broad visible requirement"]
    out,a=canonicalize_v2_2(raw)
    assert a["errors"] and out["required_claims"]==raw["required_claims"]


def test_explicit_review_request_syncs_only_control_view():
    raw=base(); raw["visual_review_requests"]=[{"claim_id":"c1","answer_critical":True,"reason":"inspect","target_time_range":[1,2],"target_visual_question":"what is visible","candidate_existing_evidence_ids":[],"expected_resolution":["confirm"]}]
    out,a=canonicalize_v2_2(raw)
    assert out["claims"][0]["needs_raw_visual_review"] is True
    assert a["claim_semantics_unchanged"] and a["review_control"][0]["control_sync_required"]


def test_deterministic():
    raw=base(); assert canonicalize_v2_2(raw)==canonicalize_v2_2(copy.deepcopy(raw))
