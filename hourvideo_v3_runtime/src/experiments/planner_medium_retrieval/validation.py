from __future__ import annotations

import math
from typing import Any

from .schemas import (
    FORBIDDEN_ANSWER_KEYS,
    MODALITIES,
    OPERATIONS,
    SCOPES,
    STRATEGIES,
)

CAPABILITY_BY_MODALITY = {
    "visual": "has_visual_embeddings",
    "audio": "has_asr",
    "detector": "has_detector_tags",
    "tracking": "has_tracking",
}


def _walk_forbidden(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_ANSWER_KEYS:
                errors.append(f"forbidden_answer_field:{path}.{key}")
            errors.extend(_walk_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_forbidden(child, f"{path}[{index}]"))
    return errors


def validate_planner_output(
    plan: Any,
    *,
    question: dict[str, Any],
    index: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(plan, dict):
        return {"valid": False, "errors": ["planner_output_not_object"], "warnings": []}
    errors.extend(_walk_forbidden(plan))
    required = {
        "planner_version",
        "question_id",
        "scope",
        "operation",
        "search_units",
        "required_evidence",
        "retrieval_strategy",
        "candidate_storyline_ids",
        "candidate_coarse_ids",
        "needs_temporal_neighbors",
        "uncertainty_notes",
    }
    missing = sorted(required - set(plan))
    errors.extend(f"missing_field:{name}" for name in missing)
    if missing:
        return {"valid": False, "errors": errors, "warnings": warnings}
    if plan["planner_version"] != "planner_v1":
        errors.append("invalid_planner_version")
    if plan["question_id"] != question["question_id"]:
        errors.append("question_id_mismatch")
    if plan["scope"] not in SCOPES:
        errors.append("invalid_scope")
    if plan["operation"] not in OPERATIONS:
        errors.append("invalid_operation")
    if plan["retrieval_strategy"] not in STRATEGIES:
        errors.append("invalid_retrieval_strategy")

    units = plan["search_units"]
    if not isinstance(units, list) or not units:
        errors.append("search_units_empty_or_invalid")
        units = []
    unit_ids = [unit.get("unit_id") for unit in units if isinstance(unit, dict)]
    if len(unit_ids) != len(set(unit_ids)) or any(not unit_id for unit_id in unit_ids):
        errors.append("search_unit_ids_not_unique_nonempty")
    capabilities = index["capabilities"]
    for unit in units:
        if not isinstance(unit, dict):
            errors.append("search_unit_not_object")
            continue
        modalities = unit.get("required_modalities", [])
        if not modalities:
            errors.append(f"search_unit_missing_modalities:{unit.get('unit_id')}")
        for modality in modalities:
            if modality not in MODALITIES:
                errors.append(f"unsupported_modality_enum:{modality}")
                continue
            capability = CAPABILITY_BY_MODALITY[modality]
            if not capabilities.get(capability, False):
                errors.append(f"capability_unavailable:{modality}:{capability}")
        if not unit.get("description") or not unit.get("query_variants"):
            errors.append(f"incomplete_search_unit:{unit.get('unit_id')}")

    evidence = plan["required_evidence"]
    if not isinstance(evidence, list) or not evidence:
        errors.append("required_evidence_empty_or_invalid")
        evidence = []
    slot_ids: list[str] = []
    for slot in evidence:
        if not isinstance(slot, dict):
            errors.append("required_evidence_item_not_object")
            continue
        slot_ids.append(slot.get("slot_id"))
        for unit_id in slot.get("search_unit_ids", []):
            if unit_id not in unit_ids:
                errors.append(f"required_evidence_unknown_search_unit:{unit_id}")
        if not isinstance(slot.get("minimum_support"), int) or slot.get("minimum_support", 0) < 1:
            errors.append(f"invalid_minimum_support:{slot.get('slot_id')}")
    if len(slot_ids) != len(set(slot_ids)) or any(not slot for slot in slot_ids):
        errors.append("slot_ids_not_unique_nonempty")

    story_ids = {node["storyline_event_id"] for node in index["storyline_events"]}
    coarse_ids = {node["coarse_id"] for node in index["coarse_nodes"]}
    for story_id in plan["candidate_storyline_ids"]:
        if story_id not in story_ids:
            errors.append(f"unknown_storyline_id:{story_id}")
    for coarse_id in plan["candidate_coarse_ids"]:
        if coarse_id not in coarse_ids:
            errors.append(f"unknown_coarse_id:{coarse_id}")

    qid = question["question_id"]
    if qid == "q_global_summary":
        if (plan["scope"], plan["operation"], plan["retrieval_strategy"]) != (
            "global",
            "summary",
            "global_coverage",
        ):
            errors.append("global_summary_contract_violation")
    if qid in {
        "q_weapon_visible",
        "q_visible_injury",
        "q_medical_assistance",
        "q_handcuffing",
    }:
        if plan["operation"] != "presence_localisation" or plan["retrieval_strategy"] != "targeted":
            errors.append("presence_localisation_contract_violation")
    if qid == "q_handcuff_before_medical":
        if (
            plan["scope"] != "multi_event"
            or plan["operation"] != "sequence"
            or plan["retrieval_strategy"] != "multi_target_compare"
            or len(units) != 2
        ):
            errors.append("multi_target_sequence_contract_violation")
        descriptions = " ".join(str(unit.get("description", "")).lower() for unit in units)
        if "handcuff" not in descriptions or not any(
            term in descriptions for term in ("medical", "aid", "assistance", "treatment")
        ):
            errors.append("multi_target_units_not_distinct_handcuff_medical")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def validate_ranking(
    ranking: list[dict[str, Any]],
    *,
    medium_ids: set[str],
    allowed_coarse_ids: set[str],
    weights: dict[str, float],
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in ranking:
        medium_id = row.get("medium_id")
        if medium_id not in medium_ids:
            errors.append(f"unknown_medium_id:{medium_id}")
        if medium_id in seen:
            errors.append(f"duplicate_medium_id:{medium_id}")
        seen.add(medium_id)
        if allowed_coarse_ids and row.get("parent_coarse_id") not in allowed_coarse_ids:
            errors.append(f"medium_outside_coarse_universe:{medium_id}")
        numbers = [
            row.get("visual_score_raw"),
            row.get("visual_score_normalized"),
            row.get("lexical_score"),
            row.get("coarse_prior"),
            row.get("combined_score"),
        ]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numbers):
            errors.append(f"invalid_numeric_score:{medium_id}")
            continue
        expected = (
            weights["visual_score"] * row["visual_score_normalized"]
            + weights["lexical_score"] * row["lexical_score"]
            + weights["coarse_prior"] * row["coarse_prior"]
        )
        if abs(expected - row["combined_score"]) > 1e-8:
            errors.append(f"combined_score_formula_mismatch:{medium_id}")
    return errors
