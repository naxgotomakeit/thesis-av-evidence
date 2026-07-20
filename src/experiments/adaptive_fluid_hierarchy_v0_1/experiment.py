"""Select fixed, elbow, and locally fluid frontiers without changing tree topology."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def topology_hash(hierarchy: dict[str, Any]) -> str:
    payload = [
        {
            "node_id": node["node_id"], "parent_id": node["parent_id"],
            "child_ids": node["child_ids"], "leaf_ids": node["leaf_ids"],
            "start": node["start"], "end": node["end"],
        }
        for node in hierarchy["nodes"]
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fine_leaf_hash(hierarchy: dict[str, Any]) -> str:
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    payload = [
        {"node_id": item, "start": nodes[item]["start"], "end": nodes[item]["end"]}
        for item in hierarchy["cuts"]["fine"]["node_ids"]
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_tree_context(hierarchy: dict[str, Any]) -> dict[str, Any]:
    nodes = {str(row["node_id"]): row for row in hierarchy["nodes"]}
    merge = {str(row["parent_id"]): dict(row) for row in hierarchy["accepted_merge_order"]}
    internal_ids = [row["parent_id"] for row in hierarchy["accepted_merge_order"]]
    qualities = {node_id: 1.0 - float(merge[node_id]["merge_score"]) for node_id in internal_ids}
    ordered_quality = sorted(qualities.values())
    ranks = {
        node_id: sum(value <= quality + 1e-12 for value in ordered_quality) / len(ordered_quality)
        for node_id, quality in qualities.items()
    }
    local_drop: dict[str, float] = {}
    child_quality_available: dict[str, bool] = {}
    for node_id in internal_ids:
        child_values = [qualities[child] for child in nodes[node_id]["child_ids"] if child in qualities]
        child_quality_available[node_id] = bool(child_values)
        local_drop[node_id] = max(child_values) - qualities[node_id] if child_values else 0.0
    depth: dict[str, int] = {}

    def assign(node_id: str, value: int) -> None:
        depth[node_id] = value
        for child in nodes[node_id]["child_ids"]:
            assign(str(child), value + 1)

    assign(str(hierarchy["root_id"]), 0)
    return {
        "nodes": nodes,
        "merge": merge,
        "quality": qualities,
        "q_rank": ranks,
        "local_drop": local_drop,
        "child_quality_available": child_quality_available,
        "depth": depth,
        "root_id": str(hierarchy["root_id"]),
        "fine_ids": list(hierarchy["cuts"]["fine"]["node_ids"]),
        "video_duration": float(hierarchy["video_duration"]),
    }


def _ordered(ids: list[str], nodes: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(ids, key=lambda item: (float(nodes[item]["start"]), float(nodes[item]["end"]), item))


def fixed_target_frontier(hierarchy: dict[str, Any], fraction: float) -> list[str]:
    """Replay the frozen accepted merge order until the requested reference count."""
    context = build_tree_context(hierarchy)
    nodes = context["nodes"]
    target = max(1, int(math.ceil(len(context["fine_ids"]) * float(fraction))))
    active = list(context["fine_ids"])
    for merge in hierarchy["accepted_merge_order"]:
        if len(active) <= target:
            break
        left, right = merge["child_ids"]
        if left not in active or right not in active:
            raise RuntimeError("Frozen accepted merge order cannot be replayed")
        left_index, right_index = active.index(left), active.index(right)
        if right_index != left_index + 1:
            raise RuntimeError("Frozen merge is not adjacent in replay frontier")
        active[left_index : right_index + 1] = [merge["parent_id"]]
    return _ordered(active, nodes)


def elbow_diagnostics(hierarchy: dict[str, Any]) -> dict[str, Any]:
    costs = sorted(float(row["merge_score"]) for row in hierarchy["accepted_merge_order"])
    if len(costs) < 3:
        raise RuntimeError("At least three accepted merges are required for two elbows")
    gaps = [costs[index + 1] - costs[index] for index in range(len(costs) - 1)]
    ranked = sorted(range(len(gaps)), key=lambda index: (-gaps[index], index))[:2]
    ranked.sort()
    medium_index, coarse_index = ranked
    median_gap = statistics.median(gaps)
    mad = statistics.median(abs(value - median_gap) for value in gaps)
    scale = max(mad, 1e-12)
    return {
        "sorted_merge_costs": costs,
        "adjacent_gaps": gaps,
        "detected_gap_indices": ranked,
        "medium_cost_threshold": costs[medium_index],
        "coarse_cost_threshold": costs[coarse_index],
        "medium_gap": gaps[medium_index],
        "coarse_gap": gaps[coarse_index],
        "medium_gap_robust_scale": (gaps[medium_index] - median_gap) / scale,
        "coarse_gap_robust_scale": (gaps[coarse_index] - median_gap) / scale,
        "rule": "two largest adjacent gaps in sorted accepted Safe-Merge costs; lower transition=Medium, higher=Coarse",
    }


def threshold_frontier(
    context: dict[str, Any], *, maximum_merge_cost: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    nodes, merge = context["nodes"], context["merge"]
    selected: list[str] = []
    trace: list[dict[str, Any]] = []

    def visit(node_id: str) -> None:
        node = nodes[node_id]
        if not node["child_ids"]:
            selected.append(node_id)
            trace.append({"node_id": node_id, "decision": "KEEP_LEAF", "reason": "leaf"})
            return
        cost = float(merge[node_id]["merge_score"])
        if cost <= maximum_merge_cost + 1e-12:
            selected.append(node_id)
            trace.append(
                {"node_id": node_id, "decision": "COLLAPSE", "merge_cost": cost, "threshold": maximum_merge_cost, "reason": "merge_cost_at_or_below_global_elbow"}
            )
        else:
            trace.append(
                {"node_id": node_id, "decision": "RECURSE", "merge_cost": cost, "threshold": maximum_merge_cost, "reason": "merge_cost_above_global_elbow"}
            )
            for child in node["child_ids"]:
                visit(str(child))

    visit(context["root_id"])
    return _ordered(selected, nodes), trace


def fluid_frontier(
    context: dict[str, Any], *, minimum_q_rank: float, maximum_local_drop: float,
    maximum_duration_sec: float | None = None, maximum_video_ratio: float | None = None,
    level: str = "medium",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Top-down local pruning; separate branches may stop at different depths."""
    nodes = context["nodes"]
    selected: list[str] = []
    trace: list[dict[str, Any]] = []
    duration_guard = None
    if maximum_duration_sec is not None and maximum_video_ratio is not None:
        duration_guard = min(float(maximum_duration_sec), float(maximum_video_ratio) * context["video_duration"])

    def visit(node_id: str) -> None:
        node = nodes[node_id]
        if not node["child_ids"]:
            selected.append(node_id)
            trace.append(
                {"level": level, "node_id": node_id, "start": node["start"], "end": node["end"], "decision": "KEEP_LEAF", "reason": "leaf"}
            )
            return
        q_rank = float(context["q_rank"][node_id])
        drop = float(context["local_drop"][node_id])
        quality_pass = q_rank >= minimum_q_rank
        drop_pass = drop <= maximum_local_drop
        guard_pass = duration_guard is None or float(node["duration"]) <= duration_guard + TOLERANCE
        collapse = quality_pass and drop_pass and guard_pass
        failed = []
        if not quality_pass:
            failed.append("q_rank")
        if not drop_pass:
            failed.append("local_drop")
        if not guard_pass:
            failed.append("duration_guard")
        trace.append(
            {
                "level": level,
                "node_id": node_id,
                "start": node["start"], "end": node["end"], "duration": node["duration"],
                "merge_cost_raw": context["merge"][node_id]["merge_score"],
                "merge_quality_raw": context["quality"][node_id],
                "q_percentile_rank": q_rank,
                "local_quality_drop": drop,
                "internal_child_quality_available": context["child_quality_available"][node_id],
                "minimum_q_rank": minimum_q_rank,
                "maximum_local_drop": maximum_local_drop,
                "duration_guard_sec": duration_guard,
                "decision": "COLLAPSE" if collapse else "RECURSE",
                "failed_conditions": failed,
                "reason": "all collapse conditions passed" if collapse else "recurse because: " + ", ".join(failed),
            }
        )
        if collapse:
            selected.append(node_id)
        else:
            for child in node["child_ids"]:
                visit(str(child))

    visit(context["root_id"])
    return _ordered(selected, nodes), trace


def validate_frontiers(
    hierarchy: dict[str, Any], medium_ids: list[str], coarse_ids: list[str],
) -> dict[str, Any]:
    context = build_tree_context(hierarchy)
    nodes = context["nodes"]
    fine_ids = set(context["fine_ids"])

    def validate_level(ids: list[str]) -> dict[str, Any]:
        ordered = _ordered(ids, nodes)
        leaf_owner: dict[str, str] = {}
        for node_id in ordered:
            for leaf_id in nodes[node_id]["leaf_ids"]:
                if leaf_id in leaf_owner:
                    raise ValueError(f"Fine leaf has multiple frontier owners: {leaf_id}")
                leaf_owner[leaf_id] = node_id
        gaps = overlaps = 0
        for left, right in zip(ordered, ordered[1:]):
            delta = float(nodes[right]["start"]) - float(nodes[left]["end"])
            gaps += int(delta > TOLERANCE)
            overlaps += int(delta < -TOLERANCE)
        covered = sum(float(nodes[item]["duration"]) for item in ordered)
        return {
            "node_count": len(ordered),
            "fine_leaf_preservation_rate": len(leaf_owner) / len(fine_ids),
            "missing_fine_leaf_count": len(fine_ids - set(leaf_owner)),
            "multi_parent_violation_count": len(leaf_owner) - len(set(leaf_owner)),
            "gap_count": gaps,
            "overlap_count": overlaps,
            "temporal_coverage_ratio": covered / context["video_duration"],
            "valid": set(leaf_owner) == fine_ids and gaps == 0 and overlaps == 0,
        }

    medium = validate_level(medium_ids)
    coarse = validate_level(coarse_ids)
    medium_membership: dict[str, str] = {}
    lineage_violations = 0
    for medium_id in medium_ids:
        matches = [
            coarse_id for coarse_id in coarse_ids
            if set(nodes[medium_id]["leaf_ids"]) <= set(nodes[coarse_id]["leaf_ids"])
        ]
        if len(matches) != 1:
            lineage_violations += 1
        else:
            medium_membership[medium_id] = matches[0]
    return {
        "medium": medium,
        "coarse": coarse,
        "medium_to_coarse_membership": medium_membership,
        "lineage_violation_count": lineage_violations,
        "valid": medium["valid"] and coarse["valid"] and lineage_violations == 0,
    }


def duration_metrics(ids: list[str], context: dict[str, Any]) -> dict[str, Any]:
    durations = [float(context["nodes"][item]["duration"]) for item in ids]
    depth_values = [int(context["depth"][item]) for item in ids]
    histogram = {str(value): depth_values.count(value) for value in sorted(set(depth_values))}
    return {
        "node_count": len(ids),
        "mean_duration_sec": statistics.fmean(durations),
        "median_duration_sec": statistics.median(durations),
        "max_duration_sec": max(durations),
        "largest_node_ratio": max(durations) / context["video_duration"],
        **{f"count_gt_{int(threshold)}s": sum(value > threshold for value in durations) for threshold in (20, 45, 90, 180, 300)},
        "selected_depth_distribution": histogram,
        "selected_depth_min": min(depth_values),
        "selected_depth_median": statistics.median(depth_values),
        "selected_depth_max": max(depth_values),
    }


def representative_record(node: dict[str, Any]) -> dict[str, Any]:
    records = list(node.get("representative_frames", []))
    for preferred in ("dinov2_medoid", "temporal_50_percent"):
        match = next((row for row in records if row["kind"] == preferred), None)
        if match:
            return dict(match)
    if not records:
        raise RuntimeError(f"Node has no frozen representative: {node['node_id']}")
    return dict(records[0])


def materialize_representative(
    node: dict[str, Any], *, root: Path, output_dir: Path, video_id: str,
) -> dict[str, Any]:
    record = representative_record(node)
    source = Path(record["frame_path"])
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_dir / "assets" / video_id / f'{node["node_id"]}__{source.name}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        shutil.copy2(source, destination)
    elif sha256_file(destination) != sha256_file(source):
        raise RuntimeError(f"Representative asset collision: {destination}")
    return {
        "timestamp": float(record["timestamp"]), "kind": record["kind"],
        "source_path": source.as_posix(), "html_relative_path": destination.relative_to(output_dir).as_posix(),
    }


def representative_embedding(
    node: dict[str, Any], *, features: np.ndarray, timestamps: np.ndarray,
) -> np.ndarray:
    record = representative_record(node)
    index = int(np.argmin(np.abs(timestamps - float(record["timestamp"]))))
    vector = np.asarray(features[index], dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def redundancy_metrics(
    medium_ids: list[str], *, context: dict[str, Any], features: np.ndarray,
    timestamps: np.ndarray, boundary_by_pair: dict[tuple[str, str], dict[str, Any]],
    similarity_thresholds: list[float], high_similarity: float, weak_boundary_percentile: float,
) -> dict[str, Any]:
    ids = _ordered(medium_ids, context["nodes"])
    embeddings = [representative_embedding(context["nodes"][item], features=features, timestamps=timestamps) for item in ids]
    pairs = []
    for index, (left, right) in enumerate(zip(ids, ids[1:])):
        similarity = float(embeddings[index] @ embeddings[index + 1])
        left_leaf = context["nodes"][left]["leaf_ids"][-1]
        right_leaf = context["nodes"][right]["leaf_ids"][0]
        boundary = boundary_by_pair[(left_leaf, right_leaf)]
        pairs.append(
            {
                "left_id": left, "right_id": right, "cosine_similarity": similarity,
                "boundary_strength": boundary["smoothed_change_strength"],
                "boundary_strength_percentile": boundary["strength_empirical_percentile"],
                "high_similarity_weak_boundary": similarity > high_similarity and boundary["strength_empirical_percentile"] <= weak_boundary_percentile,
            }
        )
    values = [row["cosine_similarity"] for row in pairs]
    if values:
        percentiles = np.percentile(np.asarray(values), [75, 90, 95]).tolist()
        summary = {
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "p75": percentiles[0], "p90": percentiles[1], "p95": percentiles[2],
        }
    else:
        summary = {key: None for key in ("mean", "median", "p75", "p90", "p95")}
    return {
        "adjacent_pair_count": len(pairs),
        "similarity_distribution": summary,
        "counts_above_threshold": {str(value): sum(item > value for item in values) for value in similarity_thresholds},
        "high_similarity_weak_boundary_pair_count": sum(row["high_similarity_weak_boundary"] for row in pairs),
        "pairs": pairs,
    }


def overmerge_metrics(
    ids: list[str], *, context: dict[str, Any], boundary_by_id: dict[str, dict[str, Any]],
    high_boundary_percentile: float, low_coherence_percentile: float,
    giant_ratio: float, chaining_ratio: float,
) -> dict[str, Any]:
    nodes = context["nodes"]
    coherences = {item: 1.0 - float(nodes[item]["internal_variability"]) for item in ids}
    coherence_values = list(coherences.values())
    cutoff = float(np.quantile(coherence_values, low_coherence_percentile)) if coherence_values else 0.0
    records = []
    for node_id in ids:
        node = nodes[node_id]
        boundaries = [boundary_by_id[item] for item in node["internal_boundary_ids"] if item in boundary_by_id]
        strongest = max((row["strength_empirical_percentile"] for row in boundaries), default=0.0)
        mean_strength = statistics.fmean(row["smoothed_change_strength"] for row in boundaries) if boundaries else 0.0
        ratio = float(node["duration"]) / context["video_duration"]
        high_boundary = strongest >= high_boundary_percentile
        low_coherence_giant = ratio >= giant_ratio and coherences[node_id] <= cutoff
        chaining = False
        imbalance = None
        if len(node["child_ids"]) == 2 and len(node["leaf_ids"]) >= 3:
            child_durations = [float(nodes[item]["duration"]) for item in node["child_ids"]]
            imbalance = max(child_durations) / sum(child_durations)
            chaining = imbalance >= chaining_ratio
        records.append(
            {
                "node_id": node_id, "start": node["start"], "end": node["end"], "duration": node["duration"],
                "strongest_internal_boundary_percentile": strongest,
                "mean_internal_boundary_strength": mean_strength,
                "internal_visual_coherence": coherences[node_id],
                "high_boundary_crossing_flag": high_boundary,
                "low_coherence_giant_flag": low_coherence_giant,
                "chaining_risk_flag": chaining,
                "child_duration_imbalance": imbalance,
                "flagged": high_boundary or low_coherence_giant or chaining,
            }
        )
    return {
        "node_count": len(ids),
        "high_boundary_crossing_count": sum(row["high_boundary_crossing_flag"] for row in records),
        "low_coherence_giant_count": sum(row["low_coherence_giant_flag"] for row in records),
        "chaining_risk_count": sum(row["chaining_risk_flag"] for row in records),
        "total_flagged_node_count": sum(row["flagged"] for row in records),
        "nodes": records,
    }


def local_complexity_alignment(
    *, hierarchy: dict[str, Any], method_medium_ids: dict[str, list[str]],
    raw_similarity: list[float], window_sec: float,
) -> dict[str, Any]:
    context = build_tree_context(hierarchy)
    nodes = context["nodes"]
    duration = context["video_duration"]
    windows = []
    start = 0.0
    fine_boundaries = [float(nodes[item]["start"]) for item in context["fine_ids"][1:]]
    change = 1.0 - np.asarray(raw_similarity, dtype=np.float64)
    while start < duration - TOLERANCE:
        end = min(duration, start + window_sec)
        length_min = (end - start) / 60.0
        left, right = int(math.floor(start)), min(len(change), int(math.ceil(end)))
        row: dict[str, Any] = {
            "start": start, "end": end,
            "fine_boundary_density_per_min": sum(start <= value < end for value in fine_boundaries) / length_min,
            "mean_local_visual_change": float(change[left:right].mean()) if right > left else 0.0,
            "medium_density_by_method": {},
        }
        for method, ids in method_medium_ids.items():
            midpoints = [(float(nodes[item]["start"]) + float(nodes[item]["end"])) / 2.0 for item in ids]
            row["medium_density_by_method"][method] = sum(start <= value < end for value in midpoints) / length_min
        windows.append(row)
        start = end
    fine = np.asarray([row["fine_boundary_density_per_min"] for row in windows], dtype=np.float64)
    visual = np.asarray([row["mean_local_visual_change"] for row in windows], dtype=np.float64)
    def normalized(values: np.ndarray) -> np.ndarray:
        spread = float(values.max() - values.min())
        return (values - values.min()) / spread if spread > 1e-12 else np.zeros_like(values)
    combined = (normalized(fine) + normalized(visual)) / 2.0
    for row, value in zip(windows, combined):
        row["combined_complexity"] = float(value)
    correlations = {}
    low_cut, high_cut = np.quantile(combined, [0.25, 0.75]) if len(combined) > 1 else (combined[0], combined[0])
    for method in method_medium_ids:
        density = np.asarray([row["medium_density_by_method"][method] for row in windows], dtype=np.float64)
        def correlation(left: np.ndarray, right: np.ndarray) -> float:
            if len(left) <= 1 or np.ptp(left) <= 1e-12 or np.ptp(right) <= 1e-12:
                return float("nan")
            return float(spearmanr(left, right).statistic)
        fine_corr = correlation(fine, density)
        visual_corr = correlation(visual, density)
        combined_corr = correlation(combined, density)
        correlations[method] = {
            "fine_density_spearman": None if np.isnan(fine_corr) else float(fine_corr),
            "visual_change_spearman": None if np.isnan(visual_corr) else float(visual_corr),
            "combined_complexity_spearman": None if np.isnan(combined_corr) else float(combined_corr),
            "stable_window_mean_medium_density": float(density[combined <= low_cut].mean()),
            "complex_window_mean_medium_density": float(density[combined >= high_cut].mean()),
        }
    return {"window_duration_sec": window_sec, "windows": windows, "correlations": correlations}
