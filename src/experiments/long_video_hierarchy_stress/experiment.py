"""Deterministic utilities for the isolated hierarchy stress experiment."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path, *, ffprobe_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe_path), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames,width,height:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"Cannot probe local video: {path}")
    stream = payload["streams"][0]
    numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
    fps = float(numerator) / max(float(denominator), 1e-12)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    frame_count = int(stream.get("nb_frames") or round(duration * fps))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return {
        "video_id": path.stem,
        "source_path": path.as_posix(),
        "duration_sec": duration if duration > 0 else frame_count / fps,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "source_bytes": path.stat().st_size,
    }


def inspect_local_videos(roots: list[Path], *, ffprobe_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                row = probe_video(path, ffprobe_path=ffprobe_path)
                row["source_root"] = root.as_posix()
                rows.append(row)
    return rows


def select_candidates(
    rows: list[dict[str, Any]], *, selected_ids: list[str], minimum_long_sec: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(row["video_id"]): row for row in rows}
    missing = [video_id for video_id in selected_ids if video_id not in by_id]
    if missing:
        raise RuntimeError(f"Frozen fallback video IDs unavailable: {missing}")
    selected = []
    for rank, video_id in enumerate(selected_ids, start=1):
        row = dict(by_id[video_id])
        row.update(
            {
                "selection_rank": rank,
                "selection_reason": (
                    "No local video reaches 300s; deterministic longest-duration "
                    "EgoSound fallback, tied by video_id"
                ),
                "target_long_video_requirement_met": row["duration_sec"] >= minimum_long_sec,
            }
        )
        selected.append(row)
    durations = [float(row["duration_sec"]) for row in rows]
    inventory = {
        "inspected_video_count": len(rows),
        "maximum_available_duration_sec": max(durations) if durations else None,
        "count_ge_300_sec": sum(value >= 300.0 for value in durations),
        "count_ge_600_sec": sum(value >= 600.0 for value in durations),
        "count_ge_900_sec": sum(value >= 900.0 for value in durations),
        "count_ge_1200_sec": sum(value >= 1200.0 for value in durations),
        "true_long_video_stress_possible": any(value >= minimum_long_sec for value in durations),
        "limitation": (
            "No local video exceeds 180s; this run is a same-duration held-out hierarchy "
            "generalization check, not a 5-20 minute stress test."
        ),
    }
    return selected, inventory


def extract_frames_1fps(
    video_path: Path, output_dir: Path, *, duration_sec: float, jpeg_quality: int,
    ffmpeg_path: Path,
) -> tuple[list[Path], np.ndarray, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg's 1 FPS filter emits one frame per complete/nearest one-second bin;
    # round() matches its deterministic count for fractional container durations.
    expected_count = int(round(duration_sec))
    frame_paths = [output_dir / f"frame_{index:05d}.jpg" for index in range(expected_count)]
    if all(path.is_file() and path.stat().st_size > 0 for path in frame_paths):
        return frame_paths, np.arange(expected_count, dtype=np.float64), {
            "cache_hit": True,
            "extraction_sec": 0.0,
            "frame_count": expected_count,
            "sampling_fps": 1.0,
        }
    started = time.perf_counter()
    quality = max(2, min(31, int(round((100 - int(jpeg_quality)) / 3.2)) + 2))
    completed = subprocess.run(
        [
            str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
            "-vf", "fps=1", "-q:v", str(quality), "-start_number", "0",
            str(output_dir / "frame_%05d.jpg"),
        ],
        check=True, capture_output=True, text=True,
    )
    written = sorted(output_dir.glob("frame_*.jpg"))
    if len(written) != expected_count:
        raise RuntimeError(
            f"1 FPS extraction incomplete: {len(written)}/{expected_count}; ffmpeg={completed.stderr[-500:]}"
        )
    return frame_paths, np.arange(expected_count, dtype=np.float64), {
        "cache_hit": False,
        "extraction_sec": time.perf_counter() - started,
        "frame_count": expected_count,
        "sampling_fps": 1.0,
    }


def level_metrics(nodes: list[dict[str, Any]], video_duration: float) -> dict[str, Any]:
    durations = [float(node["duration"]) for node in nodes]
    maximum = max(durations)
    return {
        "node_count": len(nodes),
        "mean_duration_sec": statistics.fmean(durations),
        "median_duration_sec": statistics.median(durations),
        "max_duration_sec": maximum,
        "largest_node_duration_ratio": maximum / video_duration,
        "very_short_count_lt_4s": sum(value < 4.0 for value in durations),
        "long_count_gt_20s": sum(value > 20.0 for value in durations),
        "long_count_gt_45s": sum(value > 45.0 for value in durations),
        "long_count_gt_90s": sum(value > 90.0 for value in durations),
    }


def hierarchy_depth(hierarchy: dict[str, Any]) -> int:
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}

    def depth(node_id: str) -> int:
        children = nodes[node_id]["child_ids"]
        return 1 if not children else 1 + max(depth(child_id) for child_id in children)

    return depth(str(hierarchy["root_id"]))


def hierarchy_metrics(hierarchy: dict[str, Any], *, chaining_ratio: float) -> dict[str, Any]:
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    duration = float(hierarchy["video_duration"])
    fine = [nodes[node_id] for node_id in hierarchy["cuts"]["fine"]["node_ids"]]
    medium = [nodes[node_id] for node_id in hierarchy["cuts"]["medium"]["node_ids"]]
    coarse = [nodes[node_id] for node_id in hierarchy["cuts"]["coarse"]["node_ids"]]
    giant_nodes = []
    chaining_nodes = []
    for level, selected in (("medium", medium), ("coarse", coarse)):
        for node in selected:
            ratio = float(node["duration"]) / duration
            if ratio >= 0.5:
                giant_nodes.append({"level": level, "node_id": node["node_id"], "ratio": ratio})
            if len(node["child_ids"]) == 2:
                child_durations = [float(nodes[item]["duration"]) for item in node["child_ids"]]
                imbalance = max(child_durations) / max(sum(child_durations), 1e-12)
                if imbalance >= chaining_ratio and len(node["leaf_ids"]) >= 3:
                    chaining_nodes.append(
                        {"level": level, "node_id": node["node_id"], "child_imbalance_ratio": imbalance}
                    )
    return {
        "fine": level_metrics(fine, duration),
        "medium": level_metrics(medium, duration),
        "coarse": level_metrics(coarse, duration),
        "hierarchy_depth": hierarchy_depth(hierarchy),
        "giant_parent_cases": giant_nodes,
        "chaining_risk_cases": chaining_nodes,
        "fine_preservation_rate": hierarchy["invariants"]["fine_leaf_preservation_rate"],
    }


def validate_cut_nesting(hierarchy: dict[str, Any]) -> dict[str, Any]:
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    fine_ids = set(hierarchy["cuts"]["fine"]["node_ids"])
    medium_ids = hierarchy["cuts"]["medium"]["node_ids"]
    coarse_ids = hierarchy["cuts"]["coarse"]["node_ids"]
    medium_membership: dict[str, str] = {}
    for medium_id in medium_ids:
        for leaf_id in nodes[medium_id]["leaf_ids"]:
            if leaf_id in medium_membership:
                raise ValueError(f"Fine leaf belongs to multiple Medium nodes: {leaf_id}")
            medium_membership[leaf_id] = medium_id
    coarse_membership: dict[str, str] = {}
    for coarse_id in coarse_ids:
        coarse_leaves = set(nodes[coarse_id]["leaf_ids"])
        for medium_id in medium_ids:
            medium_leaves = set(nodes[medium_id]["leaf_ids"])
            if medium_leaves <= coarse_leaves:
                if medium_id in coarse_membership:
                    raise ValueError(f"Medium node belongs to multiple Coarse nodes: {medium_id}")
                coarse_membership[medium_id] = coarse_id
    return {
        "fine_count": len(fine_ids),
        "fine_assigned_to_medium": len(medium_membership),
        "medium_count": len(medium_ids),
        "medium_assigned_to_coarse": len(coarse_membership),
        "coarse_count": len(coarse_ids),
        "valid": len(medium_membership) == len(fine_ids) and len(coarse_membership) == len(medium_ids),
    }
