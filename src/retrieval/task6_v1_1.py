"""Pure Task 6 v1.1 packet-correction helpers.

These functions only reorganise already selected Task 6 v1 candidates.  They
do not read media, retrieve candidates, or call any model.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any


ANSWER_REQUIRED_ROLES = {"direct_evidence", "resolver", "trigger", "plausible_response"}


def temporal_gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, float(right["start_time"]) - float(left["end_time"]), float(left["start_time"]) - float(right["end_time"]))


def overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, min(float(left["end_time"]), float(right["end_time"])) - max(float(left["start_time"]), float(right["start_time"])))


def corrected_relations(candidates: list[dict[str, Any]], operation: str, previous: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep speech/fallback relations and correct anchor--resolver semantics."""
    relations = [copy.deepcopy(item) for item in previous if item.get("relation_type") != "resolves"]
    anchors = [item for item in candidates if "temporal_anchor" in item.get("roles", [])]
    resolvers = [item for item in candidates if "resolver" in item.get("roles", [])]
    for anchor in anchors:
        for resolver in resolvers:
            overlap = overlap_seconds(anchor, resolver)
            temporal_type = "overlaps" if overlap > 0 else "near"
            gap = 0.0 if overlap > 0 else round(temporal_gap(anchor, resolver), 6)
            relations.append({
                "source_candidate_id": anchor["candidate_id"],
                "target_candidate_id": resolver["candidate_id"],
                "relation_type": temporal_type,
                "temporal_gap_sec": gap,
                "relation_confidence": "temporal_only",
                "relation_basis": "temporal_overlap" if overlap > 0 else "anchor_distance",
            })
            if operation.startswith("identify"):
                relations.append({
                    "source_candidate_id": resolver["candidate_id"],
                    "target_candidate_id": anchor["candidate_id"],
                    "relation_type": "resolves",
                    "temporal_gap_sec": gap,
                    "relation_confidence": "role_based",
                    "relation_basis": "operation_role",
                })
    return relations


def classify_modalities(planner_requested: list[str], retained: list[dict[str, Any]], diagnostics: dict[str, dict[str, Any]], actually_dropped: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Separate answer-essential modality roles from anchors/support."""
    answer_required = {
        item["modality"]
        for item in retained
        if set(item.get("roles", [])) & ANSWER_REQUIRED_ROLES
    }
    retained_modalities = {item["modality"] for item in retained}
    supporting = set(planner_requested) - answer_required
    for candidate_id, diagnostic in diagnostics.items():
        if diagnostic.get("acoustic_evidence_role") in {"supporting", "fallback_only", "not_required", "temporal_anchor"}:
            supporting.add(diagnostic.get("modality", "acoustic"))
    dropped_supporting = {
        item.get("modality")
        for item in actually_dropped
        if item.get("modality") in supporting and item.get("modality") not in retained_modalities
    }
    return {
        "planner_requested_modalities": sorted(set(planner_requested)),
        "answer_required_modalities": sorted(answer_required),
        "supporting_modalities": sorted(supporting - answer_required),
        "retained_modalities": sorted(retained_modalities),
        "dropped_supporting_modalities": sorted(dropped_supporting),
    }


def classify_candidate_accounting(packet: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Reclassify v1's conflated drops without changing any evidence."""
    before_by_id = {item["candidate_id"]: item for item in packet["candidates_before_reranking"]}
    retained_visual = next((item for item in packet["retained_candidates"] if item.get("candidate_type") == "canonical_visual_evidence"), None)
    canonical_id = retained_visual.get("candidate_id") if retained_visual else None
    merged, transformed, actual, frames = [], [], [], []
    for item in packet.get("dropped_candidates", []):
        candidate_id, reason = item.get("candidate_id"), item.get("reason")
        if reason == "overlapping_visual_window_merged":
            merged.append({"source_candidate_id": candidate_id, "canonical_candidate_id": canonical_id, "reason": "merged_into_canonical_visual_evidence"})
        elif reason == "visual_micro_window_represented_by_canonical_frames":
            transformed.append({"source_candidate_id": candidate_id, "canonical_candidate_id": canonical_id, "reason": "replaced_by_canonical_visual_frame_representation"})
        elif candidate_id not in before_by_id or str(candidate_id).startswith("frame_") or str(reason).startswith("visual_frame_"):
            frames.append(copy.deepcopy(item))
        else:
            dropped = copy.deepcopy(item)
            dropped["modality"] = before_by_id[candidate_id].get("modality")
            actual.append(dropped)
    return {
        "merged_source_candidates": merged,
        "transformed_candidates": transformed,
        "actually_dropped_candidates": actual,
        "dropped_visual_frames": frames,
    }


def order_visual_frames(frames: list[dict[str, Any]], retained: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve selection priority but expose chronological downstream order."""
    anchors = [item for item in retained if "temporal_anchor" in item.get("roles", []) or "trigger" in item.get("roles", [])]
    anchor_time = None
    if anchors:
        anchor_time = sum((float(item["start_time"]) + float(item["end_time"])) / 2 for item in anchors) / len(anchors)
    ranked = []
    for rank, frame in enumerate(frames, start=1):
        item = copy.deepcopy(frame)
        item["selection_rank"] = rank
        item["anchor_distance_sec"] = None if anchor_time is None else round(abs(float(item["timestamp"]) - anchor_time), 6)
        ranked.append(item)
    ordered = sorted(ranked, key=lambda item: (float(item["timestamp"]), item["selection_rank"]))
    for position, item in enumerate(ordered, start=1):
        item["presentation_order"] = position
    return ordered


def relation_counts(relations: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["relation_type"] for item in relations).items()))


def union_duration(candidates: list[dict[str, Any]], modalities: set[str] | None = None) -> float:
    spans = sorted(
        (float(item["start_time"]), float(item["end_time"]))
        for item in candidates
        if modalities is None or item.get("modality") in modalities
    )
    total, stop = 0.0, float("-inf")
    for start, end in spans:
        if start > stop:
            total += end - start
        elif end > stop:
            total += end - stop
        stop = max(stop, end)
    return total
