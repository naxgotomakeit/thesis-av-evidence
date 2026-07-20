"""Temporal adjacency-constrained hierarchy builders over immutable CoMET leaves.

The functions in this module are question-independent and API-free. They do not
modify fine intervals. Internal nodes retain ordered children and enough frame
statistics to reconstruct every fine leaf exactly.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np


TOLERANCE = 1e-6


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array / max(float(np.linalg.norm(array)), 1e-12)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def representation_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def _nearest_frame_index(timestamps: np.ndarray, target: float, indices: np.ndarray) -> int:
    local = timestamps[indices]
    return int(indices[int(np.argmin(np.abs(local - target)))])


def representative_frames(
    *,
    start: float,
    end: float,
    timestamps: np.ndarray,
    frame_paths: list[str],
    features: np.ndarray,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    include_medoid: bool = True,
) -> list[dict[str, Any]]:
    mask = (timestamps >= start) & (timestamps < end)
    if abs(end - float(timestamps[-1] + 1.0)) <= 1.0 + TOLERANCE:
        mask = (timestamps >= start) & (timestamps <= end)
    indices = np.flatnonzero(mask)
    if not len(indices):
        indices = np.asarray([int(np.argmin(np.abs(timestamps - (start + end) / 2.0)))])
    output: list[dict[str, Any]] = []
    for fraction in fractions:
        target = start + (end - start) * fraction
        index = _nearest_frame_index(timestamps, target, indices)
        output.append(
            {
                "kind": f"temporal_{int(round(fraction * 100)):02d}_percent",
                "timestamp": float(timestamps[index]),
                "frame_path": frame_paths[index],
            }
        )
    if include_medoid:
        local = normalize(features[indices])
        centroid = normalize(local.mean(axis=0))
        medoid_local = int(np.argmax(local @ centroid))
        index = int(indices[medoid_local])
        output.append(
            {
                "kind": "dinov2_medoid",
                "timestamp": float(timestamps[index]),
                "frame_path": frame_paths[index],
            }
        )
    return output


def build_boundary_records(
    segments: list[dict[str, Any]], comet_diagnostics: dict[str, Any]
) -> list[dict[str, Any]]:
    boundaries = [float(value) for value in comet_diagnostics["boundaries"]]
    indices = [int(value) for value in comet_diagnostics["diagnostics"]["boundary_indices"]]
    raw = np.asarray(comet_diagnostics["raw_similarity"], dtype=np.float64)
    smooth = np.asarray(comet_diagnostics["smoothed_similarity"], dtype=np.float64)
    if len(boundaries) != len(indices) or len(boundaries) != max(0, len(segments) - 1):
        raise ValueError("Fine segments and original CoMET boundary diagnostics differ")
    records = []
    strengths = []
    for position, (timestamp, signal_index) in enumerate(zip(boundaries, indices)):
        expected = float(segments[position]["end"])
        if abs(timestamp - expected) > TOLERANCE:
            raise ValueError(f"Fine boundary drift at {position}: {timestamp} != {expected}")
        strength = float(1.0 - smooth[signal_index])
        strengths.append(strength)
        records.append(
            {
                "boundary_id": f"fine_boundary_{position:04d}",
                "left_leaf_id": str(segments[position]["segment_id"]),
                "right_leaf_id": str(segments[position + 1]["segment_id"]),
                "timestamp": timestamp,
                "signal_index": signal_index,
                "raw_adjacent_cosine_similarity": float(raw[signal_index]),
                "smoothed_adjacent_cosine_similarity": float(smooth[signal_index]),
                "raw_change_strength": float(1.0 - raw[signal_index]),
                "smoothed_change_strength": strength,
            }
        )
    if not records:
        return records
    values = np.asarray(strengths, dtype=np.float64)
    weak_threshold = float(np.quantile(values, 1.0 / 3.0))
    strong_threshold = float(np.quantile(values, 2.0 / 3.0))
    veto_threshold = float(np.quantile(values, 0.9))
    ordered = np.sort(values)
    for record in records:
        value = float(record["smoothed_change_strength"])
        if value <= weak_threshold:
            category = "weak"
        elif value >= strong_threshold:
            category = "strong"
        else:
            category = "medium"
        record.update(
            {
                "strength_category": category,
                "strength_empirical_percentile": float(np.searchsorted(ordered, value, side="right") / len(ordered)),
                "very_strong_fine_to_medium_veto": bool(value >= veto_threshold),
                "video_thresholds": {
                    "weak_quantile_value": weak_threshold,
                    "strong_quantile_value": strong_threshold,
                    "veto_quantile_value": veto_threshold,
                },
            }
        )
    return records


def build_fine_nodes(
    *,
    segments: list[dict[str, Any]],
    features: np.ndarray,
    timestamps: np.ndarray,
    frame_paths: list[str],
    boundary_records: list[dict[str, Any]],
    representative_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    include_medoid: bool = True,
) -> list[dict[str, Any]]:
    frames = normalize(np.asarray(features, dtype=np.float64))
    times = np.asarray(timestamps, dtype=np.float64)
    if len(frames) != len(times) or len(frame_paths) != len(times):
        raise ValueError("Shared DINO/frame grid identity mismatch")
    by_left = {record["right_leaf_id"]: record for record in boundary_records}
    by_right = {record["left_leaf_id"]: record for record in boundary_records}
    nodes = []
    for index, segment in enumerate(segments):
        start, end = float(segment["start"]), float(segment["end"])
        mask = (times >= start) & (times < end)
        if index == len(segments) - 1:
            mask = (times >= start) & (times <= end)
        frame_indices = np.flatnonzero(mask)
        if not len(frame_indices):
            raise ValueError(f"Fine leaf has no DINO frames: {segment['segment_id']}")
        local = frames[frame_indices]
        vector_sum = local.sum(axis=0)
        sum_squared_norm = float(np.sum(local * local))
        centroid = vector_sum / len(local)
        sse = max(0.0, sum_squared_norm - len(local) * float(centroid @ centroid))
        pooled = normalize(centroid)
        identifier = str(segment["segment_id"])
        nodes.append(
            {
                "node_id": identifier,
                "node_type": "fine_leaf",
                "source_fine_segment_id": identifier,
                "start": start,
                "end": end,
                "duration": end - start,
                "child_ids": [],
                "parent_id": None,
                "leaf_ids": [identifier],
                "leftmost_leaf_index": index,
                "rightmost_leaf_index": index,
                "frame_count": int(len(local)),
                "frame_index_start": int(frame_indices[0]),
                "frame_index_end": int(frame_indices[-1]),
                "pooled_dinov2": np.asarray(pooled, dtype=np.float32).tolist(),
                "pooled_dinov2_sha256": representation_sha256(pooled),
                "internal_variability": sse / len(local),
                "internal_boundary_ids": [],
                "left_boundary": copy.deepcopy(by_left.get(identifier)),
                "right_boundary": copy.deepcopy(by_right.get(identifier)),
                "representative_frames": representative_frames(
                    start=start,
                    end=end,
                    timestamps=times,
                    frame_paths=frame_paths,
                    features=frames,
                    fractions=representative_fractions,
                    include_medoid=include_medoid,
                ),
                "_sum": vector_sum,
                "_sum_squared_norm": sum_squared_norm,
                "_sse": sse,
            }
        )
    return nodes


def _ward_cost(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_mean = left["_sum"] / left["frame_count"]
    right_mean = right["_sum"] / right["frame_count"]
    scale = left["frame_count"] * right["frame_count"] / (left["frame_count"] + right["frame_count"])
    return float(scale * np.sum((left_mean - right_mean) ** 2))


def _semantic_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(1.0 - normalize(left["_sum"]) @ normalize(right["_sum"]))


def _percentile(values: list[float], value: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.searchsorted(ordered, value, side="right") / len(ordered)) if len(ordered) else 0.0


def _merge(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    node_id: str,
    boundary: dict[str, Any],
    timestamps: np.ndarray,
    frame_paths: list[str],
    features: np.ndarray,
    representative_fractions: tuple[float, ...],
    include_medoid: bool,
) -> dict[str, Any]:
    if left["rightmost_leaf_index"] + 1 != right["leftmost_leaf_index"]:
        raise ValueError("Hierarchy attempted a non-adjacent temporal merge")
    if abs(float(left["end"]) - float(right["start"])) > TOLERANCE:
        raise ValueError("Adjacent hierarchy children have a gap or overlap")
    count = int(left["frame_count"] + right["frame_count"])
    vector_sum = left["_sum"] + right["_sum"]
    sum_squared_norm = float(left["_sum_squared_norm"] + right["_sum_squared_norm"])
    centroid = vector_sum / count
    sse = max(0.0, sum_squared_norm - count * float(centroid @ centroid))
    pooled = normalize(centroid)
    start, end = float(left["start"]), float(right["end"])
    return {
        "node_id": node_id,
        "node_type": "internal",
        "source_fine_segment_id": None,
        "start": start,
        "end": end,
        "duration": end - start,
        "child_ids": [left["node_id"], right["node_id"]],
        "parent_id": None,
        "leaf_ids": [*left["leaf_ids"], *right["leaf_ids"]],
        "leftmost_leaf_index": left["leftmost_leaf_index"],
        "rightmost_leaf_index": right["rightmost_leaf_index"],
        "frame_count": count,
        "frame_index_start": left["frame_index_start"],
        "frame_index_end": right["frame_index_end"],
        "pooled_dinov2": np.asarray(pooled, dtype=np.float32).tolist(),
        "pooled_dinov2_sha256": representation_sha256(pooled),
        "internal_variability": sse / count,
        "internal_boundary_ids": [
            *left["internal_boundary_ids"], boundary["boundary_id"], *right["internal_boundary_ids"]
        ],
        "left_boundary": copy.deepcopy(left.get("left_boundary")),
        "right_boundary": copy.deepcopy(right.get("right_boundary")),
        "representative_frames": representative_frames(
            start=start,
            end=end,
            timestamps=np.asarray(timestamps, dtype=np.float64),
            frame_paths=frame_paths,
            features=features,
            fractions=representative_fractions,
            include_medoid=include_medoid,
        ),
        "_sum": vector_sum,
        "_sum_squared_norm": sum_squared_norm,
        "_sse": sse,
    }


def _serialize_node(node: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in node.items() if not key.startswith("_")}


def _cut(active: list[dict[str, Any]], fine_count: int, rule: str, target: int) -> dict[str, Any]:
    return {
        "node_ids": [node["node_id"] for node in active],
        "node_count": len(active),
        "target_node_count": target,
        "compression_ratio_vs_fine": len(active) / fine_count,
        "selection_rule": rule,
        "ground_truth_granularity": False,
    }


def _between_boundary(
    left: dict[str, Any], right: dict[str, Any], by_pair: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    pair = (left["leaf_ids"][-1], right["leaf_ids"][0])
    if pair not in by_pair:
        raise ValueError(f"Original fine boundary missing for adjacent pair {pair}")
    return by_pair[pair]


def _base_hierarchy(
    *, video_id: str, video_duration: float, method: str, fine_nodes: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]], medium_fraction: float, coarse_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], int, int]:
    leaves = copy.deepcopy(fine_nodes)
    nodes = {node["node_id"]: node for node in leaves}
    fine_count = len(leaves)
    medium_target = max(1, int(math.ceil(fine_count * medium_fraction)))
    coarse_target = max(1, int(math.ceil(fine_count * coarse_fraction)))
    hierarchy = {
        "schema_version": "fine-to-coarse-hierarchy-v1",
        "video_id": video_id,
        "method": method,
        "video_duration": float(video_duration),
        "fine_leaf_ids": [node["node_id"] for node in leaves],
        "boundary_records": copy.deepcopy(boundary_records),
        "cuts": {
            "fine": _cut(leaves, fine_count, "immutable original CoMET leaves", fine_count)
        },
        "accepted_merge_order": [],
        "root_id": None,
    }
    by_pair = {(row["left_leaf_id"], row["right_leaf_id"]): row for row in boundary_records}
    return hierarchy, leaves, nodes, by_pair, medium_target, coarse_target


def build_ward_hierarchy(
    *, video_id: str, video_duration: float, fine_nodes: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]], timestamps: np.ndarray, frame_paths: list[str],
    features: np.ndarray, medium_fraction: float = 0.5, coarse_fraction: float = 0.25,
    representative_fractions: tuple[float, ...] = (0.25, 0.5, 0.75), include_medoid: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hierarchy, active, nodes, by_pair, medium_target, coarse_target = _base_hierarchy(
        video_id=video_id, video_duration=video_duration, method="adjacent_ward",
        fine_nodes=fine_nodes, boundary_records=boundary_records,
        medium_fraction=medium_fraction, coarse_fraction=coarse_fraction,
    )
    log: list[dict[str, Any]] = []
    iteration = 0
    while len(active) > 1:
        candidates = []
        for index in range(len(active) - 1):
            left, right = active[index], active[index + 1]
            candidates.append(
                {
                    "index": index,
                    "left_id": left["node_id"],
                    "right_id": right["node_id"],
                    "inter_boundary_id": _between_boundary(left, right, by_pair)["boundary_id"],
                    "ward_sse_increase": _ward_cost(left, right),
                }
            )
        winner = min(candidates, key=lambda row: (row["ward_sse_increase"], row["index"], row["left_id"], row["right_id"]))
        for candidate in candidates:
            log.append(
                {
                    "method": "adjacent_ward", "video_id": video_id, "iteration": iteration,
                    **{key: value for key, value in candidate.items() if key != "index"},
                    "accepted": candidate is winner,
                    "reason": "minimum_adjacent_ward_cost" if candidate is winner else "higher_adjacent_ward_cost",
                }
            )
        index = winner["index"]
        left, right = active[index], active[index + 1]
        boundary = _between_boundary(left, right, by_pair)
        parent_id = f"ward_node_{iteration:04d}"
        parent = _merge(
            left, right, node_id=parent_id, boundary=boundary, timestamps=timestamps,
            frame_paths=frame_paths, features=features,
            representative_fractions=representative_fractions, include_medoid=include_medoid,
        )
        nodes[left["node_id"]]["parent_id"] = parent_id
        nodes[right["node_id"]]["parent_id"] = parent_id
        nodes[parent_id] = parent
        active[index : index + 2] = [parent]
        hierarchy["accepted_merge_order"].append(
            {"parent_id": parent_id, "child_ids": parent["child_ids"], "merge_cost": winner["ward_sse_increase"]}
        )
        iteration += 1
        if len(active) == medium_target:
            hierarchy["cuts"]["medium"] = _cut(
                active, len(fine_nodes), "50% reference compression cut from full Ward dendrogram", medium_target
            )
        if len(active) == coarse_target:
            hierarchy["cuts"]["coarse"] = _cut(
                active, len(fine_nodes), "25% reference compression cut from full Ward dendrogram", coarse_target
            )
    hierarchy["root_id"] = active[0]["node_id"]
    hierarchy["nodes"] = [_serialize_node(nodes[node_id]) for node_id in nodes]
    hierarchy["invariants"] = validate_hierarchy(hierarchy)
    return hierarchy, log


def build_safe_hierarchy(
    *, video_id: str, video_duration: float, fine_nodes: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]], timestamps: np.ndarray, frame_paths: list[str],
    features: np.ndarray, medium_fraction: float = 0.5, coarse_fraction: float = 0.25,
    representative_fractions: tuple[float, ...] = (0.25, 0.5, 0.75), include_medoid: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hierarchy, active, nodes, by_pair, medium_target, coarse_target = _base_hierarchy(
        video_id=video_id, video_duration=video_duration, method="boundary_aware_safe_merge",
        fine_nodes=fine_nodes, boundary_records=boundary_records,
        medium_fraction=medium_fraction, coarse_fraction=coarse_fraction,
    )
    log: list[dict[str, Any]] = []
    iteration = 0
    medium_captured = False
    coarse_captured = False
    while len(active) > 1:
        if not medium_captured and len(active) <= medium_target:
            hierarchy["cuts"]["medium"] = _cut(
                active, len(fine_nodes), "50% reference target with very-strong Fine-boundary veto", medium_target
            )
            medium_captured = True
        if medium_captured and not coarse_captured and len(active) <= coarse_target:
            hierarchy["cuts"]["coarse"] = _cut(
                active, len(fine_nodes), "25% reference compression cut after releasing Fine→Medium veto", coarse_target
            )
            coarse_captured = True
        phase = "fine_to_medium" if not medium_captured else ("medium_to_coarse" if not coarse_captured else "coarse_to_root")
        raw_candidates = []
        for index in range(len(active) - 1):
            left, right = active[index], active[index + 1]
            boundary = _between_boundary(left, right, by_pair)
            combined_count = left["frame_count"] + right["frame_count"]
            combined_sum = left["_sum"] + right["_sum"]
            combined_sum_sq = left["_sum_squared_norm"] + right["_sum_squared_norm"]
            combined_mean = combined_sum / combined_count
            combined_sse = max(0.0, combined_sum_sq - combined_count * float(combined_mean @ combined_mean))
            before_variability = (left["_sse"] + right["_sse"]) / combined_count
            after_variability = combined_sse / combined_count
            raw_candidates.append(
                {
                    "index": index,
                    "left_id": left["node_id"],
                    "right_id": right["node_id"],
                    "inter_boundary_id": boundary["boundary_id"],
                    "semantic_difference": _semantic_difference(left, right),
                    "inter_boundary_strength": boundary["smoothed_change_strength"],
                    "inter_boundary_strength_percentile": boundary["strength_empirical_percentile"],
                    "left_internal_variability": left["internal_variability"],
                    "right_internal_variability": right["internal_variability"],
                    "weighted_internal_variability_before": before_variability,
                    "merged_internal_variability": after_variability,
                    "merged_variability_increase": max(0.0, after_variability - before_variability),
                    "merged_duration_sec": float(right["end"] - left["start"]),
                    "merged_duration_ratio": float((right["end"] - left["start"]) / video_duration),
                    "merged_fine_leaf_count": len(left["leaf_ids"]) + len(right["leaf_ids"]),
                    "merged_component_size_ratio": (len(left["leaf_ids"]) + len(right["leaf_ids"])) / len(fine_nodes),
                    "strong_boundary_veto": bool(
                        phase == "fine_to_medium" and boundary["very_strong_fine_to_medium_veto"]
                    ),
                }
            )
        semantic_values = [row["semantic_difference"] for row in raw_candidates]
        variability_values = [row["merged_variability_increase"] for row in raw_candidates]
        for row in raw_candidates:
            row["semantic_difference_percentile"] = _percentile(semantic_values, row["semantic_difference"])
            row["variability_increase_percentile"] = _percentile(variability_values, row["merged_variability_increase"])
            row["duration_component_pressure"] = (
                row["merged_duration_ratio"] + row["merged_component_size_ratio"]
            ) / 2.0
            row["merge_score"] = (
                row["semantic_difference_percentile"]
                + row["inter_boundary_strength_percentile"]
                + row["variability_increase_percentile"]
                + row["duration_component_pressure"]
            ) / 4.0
        eligible = [row for row in raw_candidates if not row["strong_boundary_veto"]]
        if not eligible and not medium_captured:
            hierarchy["cuts"]["medium"] = _cut(
                active,
                len(fine_nodes),
                "veto-limited Medium cut before 50% target; every remaining boundary was very strong",
                medium_target,
            )
            hierarchy["cuts"]["medium"]["veto_limited"] = True
            medium_captured = True
            continue
        winner = min(
            eligible,
            key=lambda row: (row["merge_score"], row["index"], row["left_id"], row["right_id"]),
        )
        for candidate in raw_candidates:
            if candidate is winner:
                reason = "minimum_boundary_aware_safe_merge_score"
            elif candidate["strong_boundary_veto"]:
                reason = "very_strong_fine_boundary_veto_during_fine_to_medium"
            else:
                reason = "higher_boundary_aware_merge_score"
            log.append(
                {
                    "method": "boundary_aware_safe_merge", "video_id": video_id,
                    "iteration": iteration, "phase": phase,
                    **{key: value for key, value in candidate.items() if key != "index"},
                    "accepted": candidate is winner, "reason": reason,
                }
            )
        index = winner["index"]
        left, right = active[index], active[index + 1]
        boundary = _between_boundary(left, right, by_pair)
        parent_id = f"safe_node_{iteration:04d}"
        parent = _merge(
            left, right, node_id=parent_id, boundary=boundary, timestamps=timestamps,
            frame_paths=frame_paths, features=features,
            representative_fractions=representative_fractions, include_medoid=include_medoid,
        )
        nodes[left["node_id"]]["parent_id"] = parent_id
        nodes[right["node_id"]]["parent_id"] = parent_id
        nodes[parent_id] = parent
        active[index : index + 2] = [parent]
        hierarchy["accepted_merge_order"].append(
            {
                "parent_id": parent_id, "child_ids": parent["child_ids"],
                "merge_score": winner["merge_score"],
                "semantic_difference": winner["semantic_difference"],
                "inter_boundary_strength": winner["inter_boundary_strength"],
                "merged_internal_variability": winner["merged_internal_variability"],
                "phase": phase,
            }
        )
        iteration += 1
    if "medium" not in hierarchy["cuts"]:
        hierarchy["cuts"]["medium"] = _cut(active, len(fine_nodes), "root fallback", medium_target)
    if "coarse" not in hierarchy["cuts"]:
        hierarchy["cuts"]["coarse"] = _cut(active, len(fine_nodes), "root fallback", coarse_target)
    hierarchy["root_id"] = active[0]["node_id"]
    hierarchy["nodes"] = [_serialize_node(nodes[node_id]) for node_id in nodes]
    hierarchy["invariants"] = validate_hierarchy(hierarchy)
    return hierarchy, log


def descendant_fine_ids(node_id: str, nodes: dict[str, dict[str, Any]]) -> list[str]:
    node = nodes[node_id]
    if node["node_type"] == "fine_leaf":
        return [node_id]
    output: list[str] = []
    for child in node["child_ids"]:
        output.extend(descendant_fine_ids(child, nodes))
    return output


def validate_hierarchy(hierarchy: dict[str, Any]) -> dict[str, Any]:
    nodes = {row["node_id"]: row for row in hierarchy.get("nodes", [])}
    fine_ids = list(hierarchy["fine_leaf_ids"])
    errors: list[str] = []
    if len(nodes) != 2 * len(fine_ids) - 1:
        errors.append("full_binary_tree_node_count")
    if hierarchy.get("root_id") not in nodes:
        errors.append("root_missing")
    for node in nodes.values():
        children = node["child_ids"]
        if node["node_type"] == "fine_leaf":
            if children or node["leaf_ids"] != [node["node_id"]]:
                errors.append(f"invalid_leaf:{node['node_id']}")
            continue
        if len(children) != 2 or any(child not in nodes for child in children):
            errors.append(f"invalid_children:{node['node_id']}")
            continue
        left, right = nodes[children[0]], nodes[children[1]]
        if left["rightmost_leaf_index"] + 1 != right["leftmost_leaf_index"]:
            errors.append(f"nonadjacent_children:{node['node_id']}")
        if abs(float(left["end"]) - float(right["start"])) > TOLERANCE:
            errors.append(f"child_gap_overlap:{node['node_id']}")
        if abs(float(node["start"]) - float(left["start"])) > TOLERANCE or abs(float(node["end"]) - float(right["end"])) > TOLERANCE:
            errors.append(f"parent_interval:{node['node_id']}")
        if nodes[children[0]].get("parent_id") != node["node_id"] or nodes[children[1]].get("parent_id") != node["node_id"]:
            errors.append(f"parent_link:{node['node_id']}")
    if hierarchy.get("root_id") in nodes:
        reconstructed = descendant_fine_ids(hierarchy["root_id"], nodes)
        if reconstructed != fine_ids:
            errors.append("root_fine_reconstruction")
    for cut_name, cut in hierarchy.get("cuts", {}).items():
        reconstructed: list[str] = []
        for node_id in cut["node_ids"]:
            reconstructed.extend(descendant_fine_ids(node_id, nodes))
        if reconstructed != fine_ids:
            errors.append(f"cut_partition:{cut_name}")
    return {
        "valid": not errors,
        "errors": errors,
        "fine_leaf_preservation_rate": (
            len(descendant_fine_ids(hierarchy["root_id"], nodes)) / len(fine_ids)
            if fine_ids and hierarchy.get("root_id") in nodes else 0.0
        ),
        "adjacency_only": not any("nonadjacent" in error for error in errors),
        "reversible_to_original_fine_leaves": "root_fine_reconstruction" not in errors,
        "parent_child_edge_count": sum(len(node["child_ids"]) for node in nodes.values()),
    }
