"""Selection, section-download and validation utilities for EgoPolice clips."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import statistics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_video_index(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        values = next(csv.reader([line]))
        if len(values) != 4:
            continue
        sample_id, source_url, start_raw, end_raw = (value.strip() for value in values)
        try:
            start, end = float(start_raw), float(end_raw)
        except ValueError:
            continue
        if "youtu" not in source_url.lower() or start < 0 or end <= start:
            continue
        rows.append(
            {
                "line_number": line_number,
                "sample_id": sample_id,
                "source_url": source_url,
                "start_sec": start,
                "end_sec": end,
                "target_clip_duration_sec": end - start,
            }
        )
    if not rows:
        raise RuntimeError("No explicit YouTube intervals found in EgoPolice video index")
    return rows


def candidate_ranking(
    rows: list[dict[str, Any]], *, target_sec: float, preferred_prefixes: list[str]
) -> list[dict[str, Any]]:
    preferred = [row for row in rows if any(row["sample_id"].startswith(prefix) for prefix in preferred_prefixes)]
    pool = preferred or rows
    return sorted(
        pool,
        key=lambda row: (
            abs(float(row["target_clip_duration_sec"]) - target_sec),
            str(row["sample_id"]),
        ),
    )


def select_primary_candidates(
    rows: list[dict[str, Any]], *, targets: list[float], preferred_prefixes: list[str]
) -> list[dict[str, Any]]:
    used_urls: set[str] = set()
    selected = []
    for target in targets:
        candidates = candidate_ranking(rows, target_sec=target, preferred_prefixes=preferred_prefixes)
        candidate = next((row for row in candidates if row["source_url"] not in used_urls), None)
        if candidate is None:
            raise RuntimeError(f"No distinct-source candidate for target {target}")
        used_urls.add(str(candidate["source_url"]))
        selected.append(
            {
                **candidate,
                "duration_class_target_sec": target,
                "absolute_target_error_sec": abs(float(candidate["target_clip_duration_sec"]) - target),
                "selection_reason": (
                    "closest explicit interval to target among preferred official police channels, "
                    "with distinct source URL and stable sample_id tie-break"
                ),
            }
        )
    return selected


def sanitize_sample_id(sample_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", sample_id).strip("._-")
    if not value:
        raise ValueError(f"Cannot sanitize sample ID: {sample_id}")
    return value


def probe_media(path: Path, *, ffprobe_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe_path), "-v", "error", "-show_entries",
            "stream=index,codec_type,codec_name,avg_frame_rate,width,height:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    video = next((row for row in payload.get("streams", []) if row.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"Downloaded clip has no video stream: {path}")
    numerator, denominator = str(video.get("avg_frame_rate", "0/1")).split("/", 1)
    fps = float(numerator) / max(float(denominator), 1e-12)
    return {
        "actual_duration_sec": float(payload["format"]["duration"]),
        "fps": fps,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "video_codec": video.get("codec_name"),
        "audio_present": any(row.get("codec_type") == "audio" for row in payload.get("streams", [])),
        "audio_codec": next((row.get("codec_name") for row in payload.get("streams", []) if row.get("codec_type") == "audio"), None),
    }


def decode_sanity(path: Path, *, ffmpeg_path: Path, duration_sec: float) -> dict[str, Any]:
    timestamp = max(0.0, min(duration_sec / 2.0, duration_sec - 0.1))
    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(ffmpeg_path), "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(path),
            "-frames:v", "1", "-f", "null", "NUL",
        ],
        capture_output=True, text=True,
    )
    return {
        "passed": completed.returncode == 0,
        "sample_timestamp_sec": timestamp,
        "decode_sanity_sec": time.perf_counter() - started,
        "stderr_tail": completed.stderr[-500:],
    }


def download_section(
    candidate: dict[str, Any], *, python_path: Path, yt_dlp_zipapp: Path,
    ffmpeg_path: Path, output_dir: Path, format_selector: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_sample_id(str(candidate["sample_id"]))
    output_template = output_dir / f"{stem}.%(ext)s"
    existing = sorted(output_dir.glob(f"{stem}.*"))
    video_extensions = {".mp4", ".mkv", ".webm", ".mov"}
    reusable = next((path for path in existing if path.suffix.lower() in video_extensions and path.stat().st_size > 0), None)
    if reusable:
        return {
            "path": reusable,
            "download_sec": 0.0,
            "cache_hit": True,
            "download_mode": "reused exact experiment-local section file",
            "command": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    section = f"*{candidate['start_sec']:.3f}-{candidate['end_sec']:.3f}"
    command = [
        str(python_path), str(yt_dlp_zipapp), "--no-playlist", "--no-part", "--newline",
        "--download-sections", section, "--force-keyframes-at-cuts",
        "--ffmpeg-location", str(ffmpeg_path.parent), "-f", format_selector,
        "--merge-output-format", "mp4", "-o", str(output_template),
        str(candidate["source_url"]),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    files = sorted(output_dir.glob(f"{stem}.*"))
    downloaded = next((path for path in files if path.suffix.lower() in video_extensions and path.stat().st_size > 0), None)
    if completed.returncode != 0 or downloaded is None:
        raise RuntimeError(
            json.dumps(
                {
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                },
                ensure_ascii=False,
            )
        )
    return {
        "path": downloaded,
        "download_sec": elapsed,
        "cache_hit": False,
        "download_mode": "yt-dlp direct requested-section download with ffmpeg force-keyframe trim",
        "command": command[:-1] + ["<source_url>"],
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def level_metrics(
    nodes: list[dict[str, Any]], video_duration: float, *, thresholds: list[float]
) -> dict[str, Any]:
    """Count hierarchy entities (not frames/assets) at one reference cut."""
    durations = [float(node["duration"]) for node in nodes]
    if not durations:
        raise ValueError("A hierarchy reference cut cannot be empty")
    maximum = max(durations)
    output = {
        "node_count": len(nodes),
        "mean_duration_sec": statistics.fmean(durations),
        "median_duration_sec": statistics.median(durations),
        "max_duration_sec": maximum,
        "largest_node_duration_ratio": maximum / max(float(video_duration), 1e-12),
    }
    for threshold in thresholds:
        direction = "lt" if threshold == 4.0 else "gt"
        count = sum(value < threshold for value in durations) if direction == "lt" else sum(
            value > threshold for value in durations
        )
        output[f"node_count_{direction}_{int(threshold)}s"] = count
    return output


def hierarchy_depth(hierarchy: dict[str, Any]) -> int:
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}

    def depth(node_id: str) -> int:
        children = nodes[node_id]["child_ids"]
        return 1 if not children else 1 + max(depth(child_id) for child_id in children)

    return depth(str(hierarchy["root_id"]))


def hierarchy_metrics(
    hierarchy: dict[str, Any], *, thresholds: list[float], giant_ratio: float,
    chaining_ratio: float,
) -> dict[str, Any]:
    """Return entity-level scaling diagnostics without subjective quality labels."""
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    duration = float(hierarchy["video_duration"])
    levels = {
        name: [nodes[node_id] for node_id in hierarchy["cuts"][name]["node_ids"]]
        for name in ("fine", "medium", "coarse")
    }
    giant_nodes: list[dict[str, Any]] = []
    chaining_nodes: list[dict[str, Any]] = []
    for level in ("medium", "coarse"):
        for node in levels[level]:
            ratio = float(node["duration"]) / max(duration, 1e-12)
            if ratio >= giant_ratio:
                giant_nodes.append(
                    {"level": level, "node_id": node["node_id"], "duration_sec": node["duration"], "ratio": ratio}
                )
            if len(node["child_ids"]) == 2 and len(node["leaf_ids"]) >= 3:
                child_durations = [float(nodes[item]["duration"]) for item in node["child_ids"]]
                imbalance = max(child_durations) / max(sum(child_durations), 1e-12)
                if imbalance >= chaining_ratio:
                    chaining_nodes.append(
                        {
                            "level": level,
                            "node_id": node["node_id"],
                            "duration_sec": node["duration"],
                            "child_imbalance_ratio": imbalance,
                        }
                    )
    return {
        **{
            level: level_metrics(selected, duration, thresholds=thresholds)
            for level, selected in levels.items()
        },
        "hierarchy_depth": hierarchy_depth(hierarchy),
        "giant_parent_cases": giant_nodes,
        "chaining_risk_cases": chaining_nodes,
        "fine_preservation_rate": hierarchy["invariants"]["fine_leaf_preservation_rate"],
    }


def validate_cut_nesting(hierarchy: dict[str, Any]) -> dict[str, Any]:
    """Verify Fine -> Medium -> Coarse reference-cut entity membership exactly once."""
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    fine_ids = set(hierarchy["cuts"]["fine"]["node_ids"])
    medium_ids = list(hierarchy["cuts"]["medium"]["node_ids"])
    coarse_ids = list(hierarchy["cuts"]["coarse"]["node_ids"])
    medium_membership: dict[str, str] = {}
    for medium_id in medium_ids:
        for fine_id in nodes[medium_id]["leaf_ids"]:
            if fine_id in medium_membership:
                raise ValueError(f"Fine leaf belongs to multiple Medium nodes: {fine_id}")
            medium_membership[fine_id] = medium_id
    coarse_membership: dict[str, str] = {}
    for coarse_id in coarse_ids:
        coarse_fine = set(nodes[coarse_id]["leaf_ids"])
        for medium_id in medium_ids:
            if set(nodes[medium_id]["leaf_ids"]) <= coarse_fine:
                if medium_id in coarse_membership:
                    raise ValueError(f"Medium node belongs to multiple Coarse nodes: {medium_id}")
                coarse_membership[medium_id] = coarse_id
    return {
        "fine_count": len(fine_ids),
        "fine_assigned_to_medium": len(medium_membership),
        "medium_count": len(medium_ids),
        "medium_assigned_to_coarse": len(coarse_membership),
        "coarse_count": len(coarse_ids),
        "valid": set(medium_membership) == fine_ids and len(coarse_membership) == len(medium_ids),
    }
