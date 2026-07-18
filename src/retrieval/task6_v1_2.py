"""Task 6 v1.2 serialization-only correction helpers."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any


def chronological_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep retained frame identity/selection metadata, but serialize by time."""
    ordered = sorted((copy.deepcopy(item) for item in frames), key=lambda item: (float(item["timestamp"]), int(item.get("selection_rank", 0))))
    for position, item in enumerate(ordered, start=1):
        item["presentation_order"] = position
    return ordered


def correct_response_relations(relations: list[dict[str, Any]], emit_trigger_of: bool = True) -> list[dict[str, Any]]:
    """Make ``response_to`` point from the response to the trigger."""
    corrected: list[dict[str, Any]] = []
    for relation in relations:
        item = copy.deepcopy(relation)
        if item.get("relation_type") == "response_to":
            item["source_candidate_id"], item["target_candidate_id"] = item["target_candidate_id"], item["source_candidate_id"]
            corrected.append(item)
            if emit_trigger_of:
                inverse = copy.deepcopy(item)
                inverse["source_candidate_id"], inverse["target_candidate_id"] = item["target_candidate_id"], item["source_candidate_id"]
                inverse["relation_type"] = "trigger_of"
                corrected.append(inverse)
        else:
            corrected.append(item)
    return corrected


def corrected_budget_accounting(accounting: dict[str, Any], actually_dropped_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(accounting)
    result["dropped_candidate_count"] = len(actually_dropped_candidates)
    result["dropped_candidate_count_consistent"] = result["dropped_candidate_count"] == len(actually_dropped_candidates)
    return result


def merge_reduction_count(merged_source_candidates: list[dict[str, Any]]) -> int:
    """Two source windows represented by one canonical item reduce count by one."""
    return max(0, len(merged_source_candidates) - int(bool(merged_source_candidates)))


def candidate_identity_accounting(packet: dict[str, Any]) -> dict[str, Any]:
    """Reconcile candidate entities without counting assets or relations.

    A canonical visual evidence entity may be introduced from already-selected
    frame assets even when Task 5C supplied no visual candidate entity.  That
    representation-only introduction is separate from a retained source
    candidate, a merge/transformation, and a genuine drop.
    """
    before_list = [item.get("candidate_id") for item in packet["candidates_before_reranking"]]
    retained_list = [item.get("candidate_id") for item in packet["retained_candidates"]]
    dropped_list = [item.get("candidate_id") for item in packet["actually_dropped_candidates"]]
    merged_list = [item.get("source_candidate_id") for item in packet["merged_source_candidates"]]
    transformed_list = [item.get("source_candidate_id") for item in packet.get("transformed_candidates", [])]

    before = set(before_list)
    retained = set(retained_list)
    dropped = set(dropped_list)
    represented = set(merged_list) | set(transformed_list)
    introduced = retained - before
    introduced_records = [
        item for item in packet["retained_candidates"]
        if item.get("candidate_id") in introduced
    ]
    allowed_introduced = {
        item["candidate_id"]
        for item in introduced_records
        if item.get("candidate_type") == "canonical_visual_evidence"
    }
    expected_retained = (before - dropped - represented) | introduced

    errors: list[str] = []
    lists = {
        "input": before_list,
        "retained": retained_list,
        "actually_dropped": dropped_list,
        "merged_source": merged_list,
        "transformed_source": transformed_list,
    }
    for name, values in lists.items():
        if any(value is None for value in values):
            errors.append(f"{name}_candidate_id_missing")
        if len(values) != len(set(values)):
            errors.append(f"{name}_candidate_id_duplicated")
    if retained & dropped:
        errors.append("retained_and_dropped_overlap")
    if dropped - before:
        errors.append("dropped_candidate_not_in_input")
    if represented - before:
        errors.append("represented_source_not_in_input")
    if dropped & represented:
        errors.append("source_both_dropped_and_represented")
    if introduced != allowed_introduced:
        errors.append("unsupported_introduced_candidate_entity")
    if retained != expected_retained:
        errors.append("candidate_identity_union_mismatch")

    return {
        "input_candidate_ids": sorted(before),
        "retained_candidate_ids": sorted(retained),
        "actually_dropped_candidate_ids": sorted(dropped),
        "represented_source_candidate_ids": sorted(represented),
        "introduced_canonical_candidate_ids": sorted(introduced),
        "expected_retained_candidate_ids": sorted(expected_retained),
        "input_candidate_count": len(before_list),
        "unique_input_candidate_count": len(before),
        "retained_candidate_count": len(retained_list),
        "unique_retained_candidate_count": len(retained),
        "actually_dropped_candidate_count": len(dropped_list),
        "unique_actually_dropped_candidate_count": len(dropped),
        "represented_source_candidate_count": len(represented),
        "introduced_canonical_candidate_count": len(introduced),
        "retained_and_dropped_intersection": sorted(retained & dropped),
        "errors": errors,
        "consistent": not errors,
        "counting_scope": "candidate_entities_only",
        "excluded_entity_types": ["visual_frame_asset", "audio_clip_asset", "relation"],
    }


def candidate_count_consistent(packet: dict[str, Any]) -> bool:
    """Return whether candidate identities reconcile across representation changes."""
    return bool(candidate_identity_accounting(packet)["consistent"])


def relation_counts(relations: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["relation_type"] for item in relations).items()))
