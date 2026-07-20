"""Deterministic CLIP retrieval over immutable Fine leaves and stored parents."""

from __future__ import annotations

from typing import Any
import time

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array / max(float(np.linalg.norm(array)), 1e-12)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def stable_rank(ids: list[str], scores: np.ndarray, nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    order = sorted(
        range(len(ids)),
        key=lambda index: (-float(scores[index]), float(nodes[ids[index]]["start"]), ids[index]),
    )
    return [
        {
            "rank": rank,
            "node_id": ids[index],
            "start": float(nodes[ids[index]]["start"]),
            "end": float(nodes[ids[index]]["end"]),
            "duration": float(nodes[ids[index]]["duration"]),
            "score": float(scores[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def union_duration(rows: list[dict[str, Any]]) -> float:
    total, current_end = 0.0, float("-inf")
    for start, end in sorted((float(row["start"]), float(row["end"])) for row in rows):
        total += max(0.0, end - max(start, current_end))
        current_end = max(current_end, end)
    return total


def score_fine_nodes(
    ids: list[str], query: np.ndarray, fine_embeddings: dict[str, np.ndarray],
    nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not ids:
        return []
    matrix = np.stack([fine_embeddings[identifier] for identifier in ids])
    scores = normalize(matrix) @ normalize(query)
    return stable_rank(ids, scores, nodes)


def score_parent_nodes(
    ids: list[str], query: np.ndarray, parent_prototypes: dict[str, dict[str, np.ndarray]],
    nodes: dict[str, dict[str, Any]], centroid_weight: float, medoid_weight: float,
) -> list[dict[str, Any]]:
    if not ids:
        return []
    # Parent score touches only two stored prototypes per node.  It never reads
    # descendant Fine embeddings or scores at query time.
    centroids = np.stack([parent_prototypes[identifier]["centroid"] for identifier in ids])
    medoids = np.stack([parent_prototypes[identifier]["medoid"] for identifier in ids])
    query = normalize(query)
    scores = centroid_weight * (normalize(centroids) @ query) + medoid_weight * (normalize(medoids) @ query)
    return stable_rank(ids, scores, nodes)


def fine_only_retrieval(
    *, query: np.ndarray, fine_ids: list[str], fine_embeddings: dict[str, np.ndarray],
    nodes: dict[str, dict[str, Any]], final_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    ranking = score_fine_nodes(fine_ids, query, fine_embeddings, nodes)
    selected = ranking[: min(final_budget, len(ranking))]
    return {
        "method": "fine_only",
        "fine_nodes_available": len(fine_ids),
        "fine_nodes_scored": len(fine_ids),
        "parent_nodes_scored": 0,
        "parent_prototype_comparisons": 0,
        "total_node_score_operations": len(fine_ids),
        "total_vector_comparisons": len(fine_ids),
        "ranking": ranking,
        "selected_final": selected,
        "activated_temporal_duration": union_duration(selected),
        "retrieval_sec": time.perf_counter() - started,
    }


def _members_inside(parent_id: str, child_cut_ids: list[str], nodes: dict[str, dict[str, Any]]) -> list[str]:
    parent_leaves = set(nodes[parent_id]["leaf_ids"])
    return [identifier for identifier in child_cut_ids if set(nodes[identifier]["leaf_ids"]).issubset(parent_leaves)]


def hierarchy_guided_retrieval(
    *, query: np.ndarray, fine_ids: list[str], medium_ids: list[str], coarse_ids: list[str],
    fine_embeddings: dict[str, np.ndarray], parent_prototypes: dict[str, dict[str, np.ndarray]],
    nodes: dict[str, dict[str, Any]], beam_width: int, final_budget: int,
    centroid_weight: float, medoid_weight: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    coarse_ranking = score_parent_nodes(
        coarse_ids, query, parent_prototypes, nodes, centroid_weight, medoid_weight
    )
    selected_coarse = coarse_ranking[: min(beam_width, len(coarse_ranking))]
    selected_coarse_ids = [row["node_id"] for row in selected_coarse]
    medium_candidates = sorted(
        {
            identifier
            for parent_id in selected_coarse_ids
            for identifier in _members_inside(parent_id, medium_ids, nodes)
        },
        key=lambda identifier: (float(nodes[identifier]["start"]), identifier),
    )
    medium_ranking = score_parent_nodes(
        medium_candidates, query, parent_prototypes, nodes, centroid_weight, medoid_weight
    )
    selected_medium = medium_ranking[: min(beam_width, len(medium_ranking))]
    selected_medium_ids = [row["node_id"] for row in selected_medium]
    visited_fine_ids = sorted(
        {leaf for identifier in selected_medium_ids for leaf in nodes[identifier]["leaf_ids"]},
        key=lambda identifier: (float(nodes[identifier]["start"]), identifier),
    )
    fine_ranking = score_fine_nodes(visited_fine_ids, query, fine_embeddings, nodes)
    selected_final = fine_ranking[: min(final_budget, len(fine_ranking))]
    cut_node_scores = len(coarse_ids) + len(medium_candidates)
    internal_parent_scores = sum(nodes[identifier]["node_type"] == "internal" for identifier in [*coarse_ids, *medium_candidates])
    return {
        "method": "hierarchy_guided",
        "beam_width": beam_width,
        "fine_nodes_available": len(fine_ids),
        "coarse_nodes_scored": len(coarse_ids),
        "medium_nodes_scored": len(medium_candidates),
        "operating_view_nodes_scored": cut_node_scores,
        "parent_nodes_scored": int(internal_parent_scores),
        "parent_prototype_comparisons": 2 * cut_node_scores,
        "fine_nodes_scored": len(visited_fine_ids),
        "total_node_score_operations": cut_node_scores + len(visited_fine_ids),
        "total_vector_comparisons": 2 * cut_node_scores + len(visited_fine_ids),
        "coarse_ranking": coarse_ranking,
        "selected_coarse": selected_coarse,
        "pruned_coarse": coarse_ranking[len(selected_coarse):],
        "medium_candidates": medium_candidates,
        "medium_ranking": medium_ranking,
        "selected_medium": selected_medium,
        "pruned_medium": medium_ranking[len(selected_medium):],
        "visited_fine_ids": visited_fine_ids,
        "pruned_fine_ids": [identifier for identifier in fine_ids if identifier not in set(visited_fine_ids)],
        "fine_ranking": fine_ranking,
        "selected_final": selected_final,
        "unique_branches_visited": len(selected_coarse_ids) + len(selected_medium_ids),
        "fine_activation_fraction": len(visited_fine_ids) / len(fine_ids) if fine_ids else None,
        "activated_temporal_duration": union_duration(selected_final),
        "retrieval_sec": time.perf_counter() - started,
    }


def add_fine_only_proxy_metrics(
    fine_only: dict[str, Any], guided: dict[str, Any]
) -> dict[str, Any]:
    """Compare against Fine-only output as a diagnostic proxy, never ground truth."""
    reference = [row["node_id"] for row in fine_only["selected_final"]]
    visited = set(guided["visited_fine_ids"])
    selected = {row["node_id"] for row in guided["selected_final"]}
    reference_set = set(reference)
    guided["diagnostic_proxy"] = {
        "label": "Fine-only Top-K reference; not evidence ground truth",
        "fine_only_top1_reached": bool(reference and reference[0] in visited),
        "fine_only_top3_reached_rate": len(reference_set & visited) / len(reference_set) if reference_set else None,
        "final_selected_id_overlap_rate": len(reference_set & selected) / len(reference_set) if reference_set else None,
        "useful_fine_pruned": [identifier for identifier in reference if identifier not in visited],
    }
    return guided
