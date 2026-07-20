"""Deterministic adjacent boundary decisions and anti-chaining grouping."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def posture_conflict(left: list[str], right: list[str]) -> bool:
    return bool(left and right and set(left).isdisjoint(right))


def pair_decision(
    *, caption_similarity: float, visual_similarity: float,
    left_postures: list[str], right_postures: list[str],
    caption_min: float, visual_min: float,
) -> tuple[str, list[str]]:
    failures = []
    if float(caption_similarity) < float(caption_min):
        failures.append("caption_continuity_below_threshold")
    if float(visual_similarity) < float(visual_min):
        failures.append("strong_visual_transition_veto")
    if posture_conflict(left_postures, right_postures):
        failures.append("explicit_posture_state_conflict")
    return ("STOP" if failures else "MERGE"), failures


def build_groups_with_anti_chaining(
    *, records: list[dict[str, Any]], pair_rows: list[dict[str, Any]],
    caption_centroid_min: float, visual_centroid_min: float,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    if not records:
        return [], []
    if len(pair_rows) != len(records) - 1:
        raise ValueError("One pair decision is required per adjacent Medium boundary")
    groups: list[list[int]] = []
    current = [0]
    expansion_rows = []
    for right_index in range(1, len(records)):
        pair = pair_rows[right_index - 1]
        pair_allows = pair["pair_decision"] == "MERGE"
        group_caption_similarity = None
        group_visual_similarity = None
        group_allows = True
        anti_chaining_applied = pair_allows and len(current) >= 2
        if anti_chaining_applied:
            caption_centroid = normalized(np.mean([records[index]["caption_embedding"] for index in current], axis=0))
            visual_centroid = normalized(np.mean([records[index]["visual_embedding"] for index in current], axis=0))
            group_caption_similarity = float(caption_centroid @ normalized(records[right_index]["caption_embedding"]))
            group_visual_similarity = float(visual_centroid @ normalized(records[right_index]["visual_embedding"]))
            group_allows = (
                group_caption_similarity >= float(caption_centroid_min)
                and group_visual_similarity >= float(visual_centroid_min)
            )
        merge = pair_allows and group_allows
        expansion_rows.append(
            {
                "left_medium_id": records[right_index - 1]["medium_id"],
                "right_medium_id": records[right_index]["medium_id"],
                "pair_decision": pair["pair_decision"],
                "anti_chaining_applied": anti_chaining_applied,
                "candidate_group_medium_ids_before": [records[index]["medium_id"] for index in current],
                "group_caption_centroid_similarity": group_caption_similarity,
                "group_visual_centroid_similarity": group_visual_similarity,
                "group_caption_centroid_threshold": float(caption_centroid_min),
                "group_visual_centroid_threshold": float(visual_centroid_min),
                "final_boundary_decision": "MERGE" if merge else "STOP",
                "anti_chaining_prevented_merge": bool(pair_allows and not group_allows),
            }
        )
        if merge:
            current.append(right_index)
        else:
            groups.append(current)
            current = [right_index]
    groups.append(current)
    flattened = [index for group in groups for index in group]
    if flattened != list(range(len(records))):
        raise RuntimeError("Semantic grouping lost or reordered Medium records")
    return groups, expansion_rows
