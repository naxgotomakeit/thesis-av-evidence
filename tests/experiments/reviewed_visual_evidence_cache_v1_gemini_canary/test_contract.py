from experiments.reviewed_visual_evidence_cache_v1_gemini_canary.core import response_text, validate_model_result


def test_valid_confirmed_result():
    value={"status":"confirmed","requirement_effect":"supports_requirement","direct_visual_support":True,"finding":"A firearm is visible.","confidence":"high"}
    assert validate_model_result(value)==[]


def test_nonconfirmed_cannot_claim_direct_support():
    value={"status":"uncertain","requirement_effect":"supports_requirement","direct_visual_support":True,"finding":"unclear","confidence":"low"}
    assert validate_model_result(value)


def test_rest_response_text_extraction():
    response={"steps":[{"type":"model_output","content":[{"type":"text","text":"{\"x\":1}"}]}]}
    assert response_text(response)=='{"x":1}'
