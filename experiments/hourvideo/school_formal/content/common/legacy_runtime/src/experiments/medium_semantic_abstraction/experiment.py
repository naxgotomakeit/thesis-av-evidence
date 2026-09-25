"""Frozen hierarchy adaptation and deterministic frame selection."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def node_map(hierarchy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["node_id"]): row for row in hierarchy["nodes"]}


def hierarchy_views(hierarchy: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = node_map(hierarchy)
    medium_ids = list(hierarchy["cuts"]["medium"]["node_ids"])
    coarse_ids = list(hierarchy["cuts"]["coarse"]["node_ids"])
    output = []
    assigned: set[str] = set()
    for coarse_id in coarse_ids:
        coarse = nodes[coarse_id]
        coarse_leaves = set(coarse["leaf_ids"])
        medium = [
            nodes[medium_id]
            for medium_id in medium_ids
            if set(nodes[medium_id]["leaf_ids"]) <= coarse_leaves
        ]
        medium.sort(key=lambda row: (float(row["start"]), str(row["node_id"])))
        assigned.update(str(row["node_id"]) for row in medium)
        output.append({"coarse": coarse, "medium": medium})
    if assigned != set(medium_ids):
        raise RuntimeError("Medium→Coarse cut membership is incomplete")
    return output


def select_representative(node: dict[str, Any], fraction: float) -> dict[str, Any]:
    target = float(node["start"]) + float(node["duration"]) * fraction
    candidates = [item for item in node["representative_frames"] if item["kind"].startswith("temporal_")]
    if not candidates:
        raise RuntimeError(f"No temporal representative frames for {node['node_id']}")
    selected = min(candidates, key=lambda item: (abs(float(item["timestamp"]) - target), str(item["frame_path"])))
    return {
        "medium_id": str(node["node_id"]),
        "fraction": fraction,
        "selection_rule": "temporal_midpoint" if fraction == 0.5 else f"temporal_{int(fraction * 100)}_percent",
        "target_timestamp": target,
        "timestamp": float(selected["timestamp"]),
        "source_image_path": str(selected["frame_path"]),
    }


def materialize_selected_asset(
    selection: dict[str, Any], *, root: Path, output_dir: Path, video_id: str
) -> dict[str, Any]:
    source = Path(selection["source_image_path"])
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(f"Selected Medium frame unavailable: {source}")
    destination = output_dir / "assets" / video_id / (
        f"{selection['medium_id']}_{int(round(selection['fraction'] * 100)):02d}_{source.name}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"Existing semantic asset differs from source: {destination}")
    else:
        shutil.copy2(source, destination)
    return {
        **selection,
        "source_image_sha256": sha256_file(source),
        "image_path": destination.as_posix(),
        "html_relative_path": destination.relative_to(output_dir).as_posix(),
    }


def prepare_medium_records(
    *, hierarchies: list[dict[str, Any]], root: Path, output_dir: Path,
    one_fractions: list[float], three_fractions: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    medium_records = []
    coarse_records = []
    for hierarchy in hierarchies:
        video_id = str(hierarchy["video_id"])
        for view in hierarchy_views(hierarchy):
            coarse = view["coarse"]
            coarse_records.append(
                {
                    "video_id": video_id,
                    "coarse_id": str(coarse["node_id"]),
                    "start": float(coarse["start"]),
                    "end": float(coarse["end"]),
                    "duration": float(coarse["duration"]),
                    "medium_ids": [str(row["node_id"]) for row in view["medium"]],
                }
            )
            for medium in view["medium"]:
                record = {
                    "video_id": video_id,
                    "coarse_id": str(coarse["node_id"]),
                    "medium_id": str(medium["node_id"]),
                    "start": float(medium["start"]),
                    "end": float(medium["end"]),
                    "duration": float(medium["duration"]),
                    "fine_ids": list(medium["leaf_ids"]),
                    "conditions": {},
                }
                for condition, fractions in (("one_frame", one_fractions), ("three_frame", three_fractions)):
                    selections = [select_representative(medium, float(fraction)) for fraction in fractions]
                    record["conditions"][condition] = {
                        "selected_images": [
                            materialize_selected_asset(item, root=root, output_dir=output_dir, video_id=video_id)
                            for item in selections
                        ]
                    }
                one_midpoint = record["conditions"]["one_frame"]["selected_images"][0]
                three_midpoint = next(
                    row for row in record["conditions"]["three_frame"]["selected_images"] if row["fraction"] == 0.5
                )
                if one_midpoint["source_image_sha256"] != three_midpoint["source_image_sha256"]:
                    raise RuntimeError(f"Shared midpoint invariant failed: {video_id}/{medium['node_id']}")
                medium_records.append(record)
    return medium_records, coarse_records
