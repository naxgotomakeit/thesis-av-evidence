from experiments.hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.core import batch_key, comparison_schema, project_resolved_sufficiency, validate_comparison


QUESTION = {"question_id": "q", "question_text": "Which?", "answer_options": [{"option_id": "A", "text": "alpha"}, {"option_id": "B", "text": "beta"}]}


def comparison(status_a="supported", status_b="not_observed"):
    return {
        "question_id": "q",
        "observations": [{"fine_id": "F1", "finding": "alpha", "visible_actions": [], "visible_objects": [], "uncertainty_notes": []}],
        "option_assessments": [
            {"option_id": "A", "status": status_a, "direct_visual_support": status_a == "supported", "supporting_fine_ids": ["F1"] if status_a != "not_observed" else [], "supported_clauses": ["alpha"] if status_a == "supported" else [], "unverified_clauses": [] if status_a == "supported" else ["alpha"], "contradicted_clauses": [], "rationale": "a"},
            {"option_id": "B", "status": status_b, "direct_visual_support": status_b == "supported", "supporting_fine_ids": ["F1"] if status_b not in {"not_observed"} else [], "supported_clauses": [], "unverified_clauses": ["beta"] if status_b != "supported" else [], "contradicted_clauses": [], "rationale": "b"},
        ],
    }


def test_schema_has_all_options_but_no_final_answer():
    text = repr(comparison_schema(QUESTION, ["F1"]))
    assert "option_assessments" in text
    assert "selected_option_id" not in text


def test_comparison_requires_canonical_option_order():
    value = comparison(); validate_comparison(value, QUESTION, ["F1"])
    value["option_assessments"].reverse()
    try: validate_comparison(value, QUESTION, ["F1"])
    except ValueError: pass
    else: raise AssertionError("reordered options passed")


def test_batch_key_uses_all_options_collectively():
    fine = [{"image_sha256": "abc"}]
    assert batch_key("r1_av", QUESTION, fine, "v3") == batch_key("r1_av", QUESTION, fine, "v3")
    changed = {**QUESTION, "answer_options": [QUESTION["answer_options"][0], {"option_id": "B", "text": "changed"}]}
    assert batch_key("r1_av", QUESTION, fine, "v3") != batch_key("r1_av", changed, fine, "v3")


def test_supported_visual_option_projects_without_haiku():
    initial = {"question_id": "q", "gate": "provisional", "candidate_option_ids": ["A", "B"], "review_requirement_ids": [], "review_query": "", "assessments": [
        {"requirement_id": "q::option_a", "status": "uncertain", "direct_support": False, "supporting_evidence_ids": [], "rationale": "initial"},
        {"requirement_id": "q::option_b", "status": "uncertain", "direct_support": False, "supporting_evidence_ids": [], "rationale": "initial"},
    ]}
    evidence = [{"evidence_id": "reviewed_visual::F1::scope"}]
    result = project_resolved_sufficiency(initial, comparison(), QUESTION, evidence, "scope")
    assert result["gate"] == "answer_ready"
    assert result["candidate_option_ids"] == ["A"]
