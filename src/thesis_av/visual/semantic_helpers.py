"""Exact frozen Semantic Coarse provenance and validation helpers."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

def frozen_fluid_loose_frontiers(
    *, metrics_path: Path, hierarchy_path: Path, expected_video_count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = [
        row for row in metrics
        if row.get("source_group") == "long_primary" and row.get("method") == "fluid_loose"
    ]
    if len(selected) != expected_video_count:
        raise RuntimeError(f"Expected {expected_video_count} long Fluid Loose records, found {len(selected)}")
    hierarchies = {str(row["video_id"]): row for row in load_jsonl(hierarchy_path)}
    output = []
    for row in sorted(selected, key=lambda item: str(item["video_id"])):
        video_id = str(row["video_id"])
        hierarchy = hierarchies.get(video_id)
        if hierarchy is None:
            raise RuntimeError(f"Frozen Safe-Merge tree missing for {video_id}")
        nodes = {str(node["node_id"]): node for node in hierarchy["nodes"]}
        medium_ids = list(row["medium_ids"])
        if len(medium_ids) != len(set(medium_ids)):
            raise RuntimeError(f"Duplicate Fluid Loose Medium ID for {video_id}")
        medium = []
        seen_leaves: list[str] = []
        for medium_id in medium_ids:
            if medium_id not in nodes:
                raise RuntimeError(f"Fluid Loose node {medium_id} absent from frozen tree")
            node = nodes[medium_id]
            medium.append(node)
            seen_leaves.extend(str(item) for item in node["leaf_ids"])
        medium.sort(key=lambda node: (float(node["start"]), str(node["node_id"])))
        expected_leaves = list(hierarchy["fine_leaf_ids"])
        if seen_leaves != expected_leaves:
            # The source IDs may be stored in chronological order already; validate the sorted frontier too.
            seen_leaves = [str(leaf) for node in medium for leaf in node["leaf_ids"]]
        if seen_leaves != expected_leaves:
            raise RuntimeError(f"Fluid Loose frontier does not preserve Fine leaves exactly for {video_id}")
        for left, right in zip(medium, medium[1:]):
            if abs(float(left["end"]) - float(right["start"])) > 1e-6:
                raise RuntimeError(f"Gap/overlap in Fluid Loose frontier for {video_id}")
        frontier_identity = [
            {
                "medium_id": str(node["node_id"]), "start": float(node["start"]),
                "end": float(node["end"]), "leaf_ids": list(node["leaf_ids"]),
            }
            for node in medium
        ]
        output.append(
            {
                "video_id": video_id,
                "video_duration": float(row["video_duration"]),
                "fine_count": int(row["fine_count"]),
                "medium_count": len(medium),
                "medium_ids": [str(node["node_id"]) for node in medium],
                "medium_nodes": medium,
                "hierarchy": hierarchy,
                "frontier_sha256": sha256_json(frontier_identity),
            }
        )
    provenance = {
        "adaptive_metrics_path": metrics_path.as_posix(),
        "adaptive_metrics_sha256": sha256_file(metrics_path),
        "safe_merge_hierarchies_path": hierarchy_path.as_posix(),
        "safe_merge_hierarchies_sha256": sha256_file(hierarchy_path),
        "selected_method": "fluid_loose",
        "selected_source_group": "long_primary",
        "video_count": len(output),
        "medium_count": sum(item["medium_count"] for item in output),
    }
    return output, provenance

def posture_state_tokens(caption: str, vocabulary: list[str]) -> list[str]:
    lowered = caption.lower()
    return sorted(token for token in vocabulary if re.search(rf"\b{re.escape(token)}\b", lowered))

def validate_semantic_coarse(
    *, frontier: dict[str, Any], medium_records: list[dict[str, Any]], coarse_records: list[dict[str, Any]]
) -> dict[str, Any]:
    frozen_ids = list(frontier["medium_ids"])
    observed_ids = [str(item["medium_id"]) for item in medium_records]
    child_ids = [str(item) for coarse in coarse_records for item in coarse["child_medium_ids"]]
    intervals_unchanged = all(
        str(node["node_id"]) == record["medium_id"]
        and abs(float(node["start"]) - float(record["start"])) <= 1e-6
        and abs(float(node["end"]) - float(record["end"])) <= 1e-6
        for node, record in zip(frontier["medium_nodes"], medium_records)
    )
    adjacent = all(
        abs(float(left["end"]) - float(right["start"])) <= 1e-6
        for left, right in zip(coarse_records, coarse_records[1:])
    )
    result = {
        "medium_ids_exact": observed_ids == frozen_ids,
        "medium_intervals_unchanged": intervals_unchanged,
        "every_medium_exactly_once": child_ids == frozen_ids,
        "coarse_temporally_adjacent_complete": adjacent,
        "fine_count_unchanged": int(frontier["fine_count"]),
        "frontier_sha256": frontier["frontier_sha256"],
    }
    result["passed"] = all(value for key, value in result.items() if isinstance(value, bool))
    return result

