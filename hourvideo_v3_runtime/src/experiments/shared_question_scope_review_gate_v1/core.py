from __future__ import annotations

from typing import Any

from experiments.r1_av_r3_2_claim_guided_gemini_closed_loop_v3.core import overlaps
from experiments.reviewed_visual_evidence_cache_v1.cache import normalize_scope


def canonical_review_scope(requirement_id: str) -> str:
    value = requirement_id.removeprefix("req_")
    normalize_scope(value)
    return value


def build_route_payload(
    *,
    question: dict[str, Any],
    assessments: list[dict[str, Any]],
    candidate_option_ids: list[str],
    evidence_ranges: list[dict[str, Any]],
    navigation_ranges: list[dict[str, Any]],
) -> dict[str, Any]:
    requirement_ids = [row["requirement_id"] for row in assessments]
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("duplicate requirement assessment")
    option_ids = {row["option_id"] for row in question["answer_options"]}
    if not set(candidate_option_ids) <= option_ids:
        raise ValueError("unknown candidate option")
    ranges = []
    for source_kind, rows in (("evidence", evidence_ranges), ("navigation_only", navigation_ranges)):
        for row in rows:
            lo, hi = map(float, row["range"])
            if lo > hi:
                raise ValueError("reversed localization range")
            ranges.append({
                "range": [lo, hi],
                "source": row["source"],
                "source_kind": source_kind,
            })
    unique = {(row["range"][0], row["range"][1], row["source"], row["source_kind"]): row for row in ranges}
    return {
        "question_id": question["question_id"],
        "question": {
            "question_text": question["question_text"],
            "answer_options": question["answer_options"],
        },
        "requirement_assessments": [
            {
                **row,
                "answer_critical": bool(row.get("answer_critical", True)),
            }
            for row in assessments
        ],
        "candidate_option_ids_from_sufficiency": candidate_option_ids,
        "localization": {"available_local_ranges": list(unique.values())},
        "policy": {
            "no_whole_video_review": True,
            "one_local_review_max": True,
            "uncertainty_alone_does_not_trigger_review": True,
            "review_only_if_answer_critical_and_materially_resolvable": True,
            "navigation_ranges_are_localization_hints_not_evidence": True,
        },
        "raw_image_inputs": 0,
    }


def validate_route_decision(
    decision: dict[str, Any], payload: dict[str, Any], *, video_duration_sec: float
) -> list[str]:
    errors: list[str] = []
    if decision.get("question_id") != payload["question_id"]:
        errors.append("question_id mismatch")
    assessments = {row["requirement_id"]: row for row in payload["requirement_assessments"]}
    review_request = decision.get("review_request")
    if decision.get("decision") == "answer_now":
        if review_request is not None:
            errors.append("answer_now must not include review_request")
        return errors
    if decision.get("decision") != "request_local_review":
        return errors + ["unknown route decision"]
    if not review_request or not review_request.get("target_requirement_ids"):
        return errors + ["review lacks concrete target"]
    target_ids = review_request["target_requirement_ids"]
    if len(target_ids) != len(set(target_ids)):
        errors.append("duplicate review target")
    if not set(target_ids) <= set(assessments):
        errors.append("unknown target requirement")
    if not any(assessments[rid].get("answer_critical", True) for rid in target_ids if rid in assessments):
        errors.append("review target is not answer-critical")
    target_range = review_request.get("target_time_range")
    if not isinstance(target_range, list) or len(target_range) != 2:
        return errors + ["invalid target range"]
    lo, hi = map(float, target_range)
    if lo < 0 or lo > hi or hi > video_duration_sec:
        errors.append("invalid target range")
    elif hi - lo >= video_duration_sec * 0.5:
        errors.append("whole-video/broad review rejected")
    elif not any(overlaps([lo, hi], row["range"]) for row in payload["localization"]["available_local_ranges"]):
        errors.append("target range has no frozen localization support")
    return errors


def select_local_fines(
    *,
    decision: dict[str, Any],
    fine_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    request = decision["review_request"]
    lo, hi = map(float, request["target_time_range"])
    candidates = [
        row for row in fine_nodes
        if lo <= float(row["timestamp_sec"]) <= hi
    ]
    if not candidates:
        return []
    if request["target_kind"] == "dynamic":
        selected = candidates
    else:
        center = (lo + hi) / 2.0
        distance = min(abs(float(row["timestamp_sec"]) - center) for row in candidates)
        selected = [row for row in candidates if abs(abs(float(row["timestamp_sec"]) - center) - distance) < 1e-9]
    return sorted(selected, key=lambda row: (float(row["timestamp_sec"]), row["fine_id"]))
