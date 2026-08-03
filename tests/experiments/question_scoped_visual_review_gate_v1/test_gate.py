from experiments.question_scoped_visual_review_gate_v1.core import project_question, validate_projection


def _fine(values):
    return {"gemini_contract": {"allowed_image_ids_by_requirement": values}}


def test_unasked_weapon_role_cannot_trigger_review():
    source = {
        "question_id": "q_weapon_visible",
        "question": "Was a weapon visible, and when?",
        "requirement_assessments": [
            {"requirement_id": "q_weapon_visible::weapon_presence", "status": "supported", "direct_support": True},
            {"requirement_id": "q_weapon_visible::weapon_holder_role", "status": "uncertain", "direct_support": False},
        ],
    }
    out = project_question(source, _fine({"q_weapon_visible::weapon_holder_role": ["F1"]}))
    assert out["decision"] == "answer_now"
    assert out["excluded_supplementary_requirements"][0]["requirement_id"].endswith("weapon_holder_role")
    assert validate_projection(source, out) == []


def test_requested_weapon_presence_can_trigger_review():
    source = {
        "question_id": "q_weapon_visible",
        "question": "Was a weapon visible, and when?",
        "requirement_assessments": [
            {"requirement_id": "q_weapon_visible::weapon_presence", "status": "not_found", "direct_support": False},
        ],
    }
    out = project_question(source, _fine({"q_weapon_visible::weapon_presence": ["F1"]}))
    assert out["reviewable_requirement_ids"] == ["q_weapon_visible::weapon_presence"]


def test_unasked_medical_identities_are_excluded():
    source = {
        "question_id": "q_medical_assistance",
        "question": "Did any officer provide visible medical assistance? If so, when?",
        "requirement_assessments": [
            {"requirement_id": "q_medical_assistance::assistance_action", "status": "supported", "direct_support": True},
            {"requirement_id": "q_medical_assistance::provider_actor", "status": "uncertain", "direct_support": False},
            {"requirement_id": "q_medical_assistance::recipient_actor", "status": "uncertain", "direct_support": False},
        ],
    }
    fine = _fine({
        "q_medical_assistance::provider_actor": ["F1"],
        "q_medical_assistance::recipient_actor": ["F2"],
    })
    out = project_question(source, fine)
    assert out["decision"] == "answer_now"
    assert [x["requirement_id"] for x in out["excluded_supplementary_requirements"]] == [
        "q_medical_assistance::recipient_actor"
    ]
    provider = next(x for x in out["in_scope_requirement_assessments"] if x["requirement_id"].endswith("provider_actor"))
    assert provider["question_scope"] == "retained_without_visual_refinement"
    assert provider["visual_review_eligible"] is False
