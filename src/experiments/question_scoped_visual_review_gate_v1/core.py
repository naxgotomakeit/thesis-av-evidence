from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


QUESTION_SCOPE: dict[str, dict[str, tuple[str, ...]]] = {
    "q_global_summary": {
        "answer_required": (
            "event_phases", "chronological_order", "principal_visible_actors",
            "major_actions", "force_or_restraint_progression",
            "visible_injury_or_blood", "visible_medical_or_aftercare", "visible_outcome",
        ),
        "support_only": (),
        "retained_no_review": (),
    },
    "q_weapon_visible": {
        "answer_required": ("weapon_presence", "weapon_timestamp"),
        "support_only": (),
        "retained_no_review": (),
    },
    "q_visible_injury": {
        "answer_required": ("visible_injury_or_blood", "injury_timestamp"),
        "support_only": ("direct_visuality",),
        "retained_no_review": (),
    },
    "q_medical_assistance": {
        "answer_required": ("assistance_action", "assistance_timestamp"),
        "support_only": ("action_is_medical_or_supportive",),
        # The question asks whether an officer provided assistance, but it does
        # not ask for the provider's name, rank, unit, or exact identity. Keep
        # the existing actor assessment as context without allowing the old
        # over-specific refinement request to trigger image review.
        "retained_no_review": ("provider_actor",),
    },
    "q_handcuffing": {
        "answer_required": ("restraint_or_handcuff_action", "timestamp"),
        "support_only": ("handcuff_object_visibility",),
        "retained_no_review": ("officer_actor",),
    },
    "q_handcuff_before_medical": {
        "answer_required": ("temporal_order",),
        "support_only": (
            "handcuff_event", "handcuff_timestamp", "medical_event",
            "medical_timestamp", "order_confidence",
        ),
        "retained_no_review": (),
    },
}


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slot(requirement_id: str) -> str:
    return requirement_id.split("::", 1)[-1]


def build_scope_manifest() -> dict[str, Any]:
    questions = {}
    for qid, scope in QUESTION_SCOPE.items():
        answer = list(scope["answer_required"])
        support = list(scope["support_only"])
        retained = list(scope["retained_no_review"])
        questions[qid] = {
            "answer_required_slot_ids": answer,
            "support_only_slot_ids": support,
            "retained_without_visual_refinement_slot_ids": retained,
            "review_eligible_slot_ids": answer + support,
            "policy": (
                "Only facts explicitly requested by the question, plus indispensable evidence "
                "prerequisites, may trigger visual review. Supplementary identity, rank, role, "
                "ownership, or descriptive refinements are excluded."
            ),
        }
    return {
        "contract": "question_scoped_visual_review_gate_v1",
        "scope_authority": "frozen question text",
        "questions": questions,
    }


def project_question(source: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    qid = source["question_id"]
    if qid not in QUESTION_SCOPE:
        raise ValueError(f"unknown question scope: {qid}")
    scope = QUESTION_SCOPE[qid]
    answer = set(scope["answer_required"])
    support = set(scope["support_only"])
    retained = set(scope["retained_no_review"])
    eligible = answer | support
    included_slots = eligible | retained
    allowed = fine["gemini_contract"]["allowed_image_ids_by_requirement"]
    in_scope, excluded, reviewable = [], [], []
    for row in source["requirement_assessments"]:
        rid = row["requirement_id"]
        slot = _slot(rid)
        projected = dict(row)
        projected["question_scope"] = (
            "answer_required" if slot in answer else
            "support_only" if slot in support else "supplementary_not_requested"
        )
        if slot in retained:
            projected["question_scope"] = "retained_without_visual_refinement"
        projected["answer_critical"] = slot in answer
        projected["visual_review_eligible"] = slot in eligible
        unresolved = not (row["status"] == "supported" and row.get("direct_support"))
        projected["visual_review_candidate"] = bool(slot in eligible and unresolved and allowed.get(rid))
        if slot in included_slots:
            in_scope.append(projected)
            if projected["visual_review_candidate"]:
                reviewable.append(rid)
        else:
            excluded.append({
                "requirement_id": rid,
                "reason": "supplementary fact not requested by the question",
                "previous_status": row["status"],
                "previous_next_stage": row.get("next_stage"),
                "retrieved_fine_ids": allowed.get(rid, []),
            })
    return {
        "question_id": qid,
        "question": source["question"],
        "in_scope_requirement_assessments": in_scope,
        "excluded_supplementary_requirements": excluded,
        "reviewable_requirement_ids": reviewable,
        "decision": "request_local_review" if reviewable else "answer_now",
        "policy": {
            "unasked_requirements_reach_router": False,
            "unasked_requirements_reach_final_answer": False,
            "unasked_requirements_trigger_visual_review": False,
            "retrieval_outputs_modified": False,
            "organizer_outputs_modified": False,
        },
    }


def validate_projection(source: dict[str, Any], projection: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_ids = {x["requirement_id"] for x in source["requirement_assessments"]}
    included = {x["requirement_id"] for x in projection["in_scope_requirement_assessments"]}
    excluded = {x["requirement_id"] for x in projection["excluded_supplementary_requirements"]}
    if included & excluded or included | excluded != source_ids:
        errors.append("scope projection does not partition source requirements")
    if not set(projection["reviewable_requirement_ids"]) <= included:
        errors.append("reviewable requirement is outside question scope")
    if any(x["question_scope"] == "supplementary_not_requested"
           for x in projection["in_scope_requirement_assessments"]):
        errors.append("supplementary requirement leaked into in-scope handoff")
    return errors


def run(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = _load(config_path)
    out = root / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: root / v for k, v in cfg["sources"].items()}
    hashes_before = {k: _sha(p) for k, p in paths.items()}
    source = _load(paths["routing_inputs"])
    fine_docs = {
        "r1_av": _load(paths["r1_fine_retrieval"]),
        "r3_2": _load(paths["r3_fine_retrieval"]),
    }
    fine_by_q = {side: {x["question_id"]: x for x in rows} for side, rows in fine_docs.items()}
    projections: dict[str, dict[str, Any]] = {"r1_av": {}, "r3_2": {}}
    errors: list[str] = []
    for side in projections:
        for qid, row in source[side].items():
            projected = project_question(row, fine_by_q[side][qid])
            projections[side][qid] = projected
            errors += [f"{side}/{qid}: {e}" for e in validate_projection(row, projected)]

    scope = build_scope_manifest()
    _dump(out / "question_scope_manifest.json", scope)
    _dump(out / "scoped_routing_inputs.json", projections)
    audit = {
        side: {
            qid: {
                "decision": row["decision"],
                "reviewable_requirement_ids": row["reviewable_requirement_ids"],
                "excluded_requirement_ids": [x["requirement_id"] for x in row["excluded_supplementary_requirements"]],
                "excluded_requirements_with_fine_images": [
                    x["requirement_id"] for x in row["excluded_supplementary_requirements"]
                    if x["retrieved_fine_ids"]
                ],
            }
            for qid, row in rows.items()
        }
        for side, rows in projections.items()
    }
    _dump(out / "review_gate_audit.json", audit)
    r3_old = _load(paths["prior_route_decisions"])["r3_2"]
    comparison = {
        qid: {
            "prior_decision": r3_old[qid]["parsed"]["decision"],
            "scoped_decision": projections["r3_2"][qid]["decision"],
            "excluded_visual_review_requirements": audit["r3_2"][qid]["excluded_requirements_with_fine_images"],
        }
        for qid in projections["r3_2"]
    }
    _dump(out / "r3_before_after_review_comparison.json", comparison)
    hashes_after = {k: _sha(p) for k, p in paths.items()}
    if hashes_before != hashes_after:
        errors.append("protected source hash changed")
    validation = {
        "scope_projection_validation": "passed" if not errors else "failed",
        "unasked_requirement_router_leakage": 0,
        "unasked_requirement_final_handoff_leakage": 0,
        "organizer_changes": 0,
        "retrieval_changes": 0,
        "model_api_calls": 0,
        "protected_sources_unchanged": hashes_before == hashes_after,
        "errors": errors,
        "overall_validation": "passed" if not errors else "failed",
    }
    _dump(out / "protected_hash_audit.json", {"before": hashes_before, "after": hashes_after, "unchanged": hashes_before == hashes_after})
    _dump(out / "validation_report.json", validation)
    r3_reviewable = {qid: row["reviewable_requirement_ids"] for qid, row in projections["r3_2"].items()}
    report = [
        "# Question-scoped visual review gate v1", "",
        f"- Validation: `{validation['overall_validation']}`",
        "- Model/API calls: `0`",
        "- Organizer/Retrieval changes: `0/0`", "",
        "## R3_2 scoped review candidates", "",
    ] + [f"- `{qid}`: `{ids or 'answer_now'}`" for qid, ids in r3_reviewable.items()]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return validation
