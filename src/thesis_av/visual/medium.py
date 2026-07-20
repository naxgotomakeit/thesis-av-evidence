"""Exact original pure Fluid Loose frontier over a frozen Safe-Merge tree."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
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

