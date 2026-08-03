from experiments.r1_av_r3_2_retrieval_bound_fine_evidence_v1.core import validate


def fixture():
    index = {"fine_nodes": [{"fine_id": "F1"}],
             "medium_nodes": [{"medium_id": "M1", "child_fine_ids": ["F1"]}]}
    row = {"question_id": "q", "selected_fine_evidence": [{
        "fine_id": "F1", "medium_id": "M1", "frame_path": __file__, "siglip_score_raw": 0.5}],
        "gemini_contract": {"allowed_image_ids": ["F1"], "whole_medium_expansion_allowed": False,
                            "claim_range_expansion_allowed": False, "unranked_child_fines_allowed": False}}
    return index, {"r1_av": [row] * 6, "r3_2": [row] * 6}


def test_valid_contract():
    index, results = fixture()
    assert validate(results, index, 2) == []


def test_unranked_expansion_fails():
    index, results = fixture()
    results["r1_av"][0]["gemini_contract"]["whole_medium_expansion_allowed"] = True
    assert any("expansion enabled" in x for x in validate(results, index, 2))


def test_unknown_fine_fails():
    index, results = fixture()
    results["r1_av"][0]["selected_fine_evidence"][0]["fine_id"] = "BAD"
    results["r1_av"][0]["gemini_contract"]["allowed_image_ids"] = ["BAD"]
    assert any("unknown Fine" in x for x in validate(results, index, 2))
