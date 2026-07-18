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


def candidate_count_consistent(packet: dict[str, Any]) -> bool:
    return len(packet["retained_candidates"]) == (
        len(packet["candidates_before_reranking"])
        - merge_reduction_count(packet["merged_source_candidates"])
        - len(packet["actually_dropped_candidates"])
    )


def relation_counts(relations: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["relation_type"] for item in relations).items()))
