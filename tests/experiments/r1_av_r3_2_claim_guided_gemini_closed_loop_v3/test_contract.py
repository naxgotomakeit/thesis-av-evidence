from pathlib import Path

from experiments.r1_av_r3_2_claim_guided_gemini_closed_loop_v3.core import (
    answerability,
    apply_review,
    select_images,
    validate_final,
    validate_route,
)


def assessment(rid="q::r", status="uncertain", direct=False):
    return {"requirement_id": rid, "status": status, "direct_support": direct,
            "answer_critical": True, "supporting_evidence_ids": [], "finding": "x"}


def test_answer_now_has_no_review_request():
    payload = {"question_id": "q", "requirement_assessments": [assessment()],
               "localization": {"available_local_ranges": [{"range": [1, 2]}]}}
    decision = {"question_id": "q", "decision": "answer_now", "review_request": None}
    assert validate_route(decision, payload) == []


def test_broad_review_fails():
    payload = {"question_id": "q", "requirement_assessments": [assessment()],
               "localization": {"available_local_ranges": [{"range": [0, 1235]}]}}
    decision = {"question_id": "q", "decision": "request_local_review",
                "review_request": {"target_requirement_ids": ["q::r"],
                                   "target_time_range": [0, 1000]}}
    assert "whole-video/broad review rejected" in validate_route(decision, payload)


def test_review_range_requires_frozen_localization():
    payload = {"question_id": "q", "requirement_assessments": [assessment()],
               "localization": {"available_local_ranges": [{"range": [10, 20]}]}}
    decision = {"question_id": "q", "decision": "request_local_review",
                "review_request": {"target_requirement_ids": ["q::r"],
                                   "target_time_range": [30, 40]}}
    assert "target range has no frozen localization support" in validate_route(decision, payload)


def _assets() -> Path:
    path = Path("outputs/experiments/r1_av_r3_2_claim_guided_gemini_closed_loop_v3/test_assets")
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_static_selection_has_no_fixed_cap():
    a, b = _assets() / "a.jpg", _assets() / "b.jpg"
    a.write_bytes(b"a"); b.write_bytes(b"b")
    registry = [
        {"image_id": "a", "timestamp_sec": 4.0, "path": str(a), "question_id": None},
        {"image_id": "b", "timestamp_sec": 6.0, "path": str(b), "question_id": None},
    ]
    decision = {"review_request": {"target_requirement_ids": ["q::r"],
                "target_time_range": [0, 10], "target_kind": "static"}}
    result = select_images("q", decision, registry)
    assert {x["image_id"] for x in result["images"]} == {"a", "b"}
    assert result["fixed_frame_count_rule"] is False


def test_dynamic_selection_keeps_every_local_frame():
    registry = []
    for i in range(15):
        path = _assets() / f"{i}.jpg"; path.write_bytes(str(i).encode())
        registry.append({"image_id": str(i), "timestamp_sec": float(i), "path": str(path), "question_id": None})
    decision = {"review_request": {"target_requirement_ids": ["q::r"],
                "target_time_range": [0, 14], "target_kind": "dynamic"}}
    assert len(select_images("q", decision, registry)["images"]) == 15


def test_review_updates_only_target():
    rows = [assessment("q::a"), assessment("q::b")]
    updates = [{"requirement_id": "q::a", "status": "supported", "direct_support": True,
                "finding": "seen", "support_scope": "reviewed", "supporting_image_ids": ["i"],
                "remaining_uncertainty": [], "rationale": "seen"}]
    out = apply_review(rows, updates)
    assert out[0]["status"] == "supported" and out[1] == rows[1]


def test_answerability_is_deterministic():
    assert answerability([assessment(status="supported", direct=True)]) == "answer_directly"
    assert answerability([assessment()]) == "insufficient_to_answer"
    assert answerability([assessment(status="supported", direct=True), assessment("q::b")]) == "partial_answer"
    assert answerability([assessment(status="supported", direct=True)], temporal=True) == "partial_answer"


def test_final_cannot_change_answerability():
    packets = [{"question_id": "q", "answerability": "partial_answer",
                "requirement_assessments": [assessment()], "accepted_temporal_resolution": None}]
    parsed = {"answers": [{"question_id": "q", "answerability": "answer_directly", "answer": "x",
                           "supporting_requirement_ids": [], "supporting_evidence_ids": [], "caveats": []}]}
    assert "q: answerability changed" in validate_final(parsed, packets)


def test_temporal_sidecar_evidence_and_ems_caveat_are_valid():
    packets = [{"question_id": "q", "answerability": "partial_answer",
                "requirement_assessments": [assessment()],
                "accepted_temporal_resolution": {"event_a_review": {"supporting_evidence_ids": ["comet_style_a"]}}}]
    parsed = {"answers": [{"question_id": "q", "answerability": "partial_answer",
                           "answer": "The target order cannot be established; a later EMS request is not treatment onset.",
                           "supporting_requirement_ids": [], "supporting_evidence_ids": ["visual_caption::comet_style_a"],
                           "caveats": []}]}
    assert validate_final(parsed, packets) == []
