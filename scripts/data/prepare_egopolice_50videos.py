from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


MCQ_TYPES = ("1s", "10s", "60s")
SELECTION_SEED = 20260721
SOURCE_QUOTAS = {
    "copa": {"le_300": 9, "301_600": 14, "601_1200": 13, "gt_1200": 9},
    "pasadena": {"le_300": 1, "301_600": 1, "601_1200": 2, "gt_1200": 1},
}
DEV_BIN_QUOTAS = {"le_300": 2, "301_600": 3, "601_1200": 3, "gt_1200": 2}


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{namespace}|{value}".encode()).hexdigest()


def duration_bin(value: float) -> str:
    if value <= 300:
        return "le_300"
    if value <= 600:
        return "301_600"
    if value <= 1200:
        return "601_1200"
    return "gt_1200"


def video_id_from_path(value: str) -> str:
    return str(PurePosixPath(value).with_suffix(""))


def parse_video_index(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 4:
                continue
            sample_id, source_url, start_raw, end_raw = (value.strip() for value in row)
            try:
                start_sec, end_sec = float(start_raw), float(end_raw)
            except ValueError:
                continue
            output[sample_id] = {
                "sample_id": sample_id,
                "source_url": source_url,
                "source_start_sec": start_sec,
                "source_end_sec": end_sec,
                "video_index_line": line_number,
            }
    return output


def load_candidates(data_root: Path) -> list[dict[str, Any]]:
    index = parse_video_index(data_root / "video.txt")
    by_video: dict[str, dict[str, Any]] = {}
    for duration_type in MCQ_TYPES:
        path = data_root / f"mcq_{duration_type}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            video_id = video_id_from_path(str(row["video"]))
            candidate = by_video.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "video_metadata_path": str(row["video"]).replace("\\", "/"),
                    "question_ids_by_type": defaultdict(list),
                    "question_count_by_type": Counter(),
                    "maximum_annotation_end_sec": 0.0,
                    "minimum_annotation_start_sec": math.inf,
                },
            )
            candidate["question_ids_by_type"][duration_type].append(str(row["id"]))
            candidate["question_count_by_type"][duration_type] += 1
            candidate["maximum_annotation_end_sec"] = max(
                candidate["maximum_annotation_end_sec"], float(row["end second"])
            )
            candidate["minimum_annotation_start_sec"] = min(
                candidate["minimum_annotation_start_sec"], float(row["start second"])
            )

    output = []
    for video_id, candidate in by_video.items():
        source = index.get(video_id)
        if source is None:
            continue
        parts = video_id.split("/")
        source_collection = parts[0]
        case_folder = "/".join(parts[:-1])
        # Pasadena has no separate case folder in the supplied path. Treat each
        # distinct source video as a leakage group instead of grouping all of
        # Pasadena together under one artificial case.
        leakage_group = case_folder if source_collection == "copa" else video_id
        types = sorted(candidate["question_ids_by_type"], key=MCQ_TYPES.index)
        maximum_end = float(candidate["maximum_annotation_end_sec"])
        output.append(
            {
                **candidate,
                **source,
                "source_collection": source_collection,
                "case_id": case_folder,
                "case_folder": case_folder,
                "leakage_group": leakage_group,
                "mcq_duration_types": types,
                "mcq_metadata_files": [f"mcq_{value}.json" for value in types],
                "question_count": sum(candidate["question_count_by_type"].values()),
                "duration_lower_bound_sec": maximum_end,
                "duration_lower_bound_basis": "maximum MCQ annotation end; not a verified media duration",
                "duration_bin": duration_bin(maximum_end),
            }
        )
    return output


def select_frozen_50(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if set(row["mcq_duration_types"]) == set(MCQ_TYPES)]
    chosen: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    # Allocate scarce long-duration strata first so the one-case-per-video
    # constraint cannot be consumed by the much larger short strata.
    allocation_order = ("gt_1200", "601_1200", "301_600", "le_300")
    for source_collection, bin_quotas in SOURCE_QUOTAS.items():
        for bin_name in allocation_order:
            quota = bin_quotas[bin_name]
            pool = sorted(
                (
                    row
                    for row in eligible
                    if row["source_collection"] == source_collection
                    and row["duration_bin"] == bin_name
                ),
                key=lambda row: stable_key("selection", row["video_id"]),
            )
            selected_here = 0
            for row in pool:
                if row["leakage_group"] in used_groups:
                    continue
                chosen.append(row)
                used_groups.add(row["leakage_group"])
                selected_here += 1
                if selected_here == quota:
                    break
            if selected_here != quota:
                raise RuntimeError(
                    f"Could not satisfy frozen quota {source_collection}/{bin_name}: "
                    f"wanted {quota}, found {selected_here} distinct groups"
                )
    if len(chosen) != 50 or len({row["video_id"] for row in chosen}) != 50:
        raise RuntimeError("Frozen selection must contain exactly 50 distinct source videos")
    return chosen


def usage_order(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["video_id"]: row for row in selected}
    development: list[dict[str, Any]] = []
    used: set[str] = set()

    pasadena = min(
        (row for row in selected if row["source_collection"] == "pasadena"),
        key=lambda row: stable_key("development-pasadena", row["video_id"]),
    )
    development.append(pasadena)
    used.add(pasadena["video_id"])

    targets = dict(DEV_BIN_QUOTAS)
    targets[pasadena["duration_bin"]] -= 1
    for bin_name, quota in targets.items():
        pool = sorted(
            (row for row in selected if row["duration_bin"] == bin_name and row["video_id"] not in used),
            key=lambda row: stable_key("development", row["video_id"]),
        )
        development.extend(pool[:quota])
        used.update(row["video_id"] for row in pool[:quota])
    if len(development) != 10:
        raise RuntimeError("Development designation must contain exactly 10 videos")

    smoke: list[dict[str, Any]] = [pasadena]
    smoke_used = {pasadena["video_id"]}
    for bin_name in DEV_BIN_QUOTAS:
        if any(row["duration_bin"] == bin_name for row in smoke):
            continue
        row = min(
            (row for row in development if row["duration_bin"] == bin_name and row["video_id"] not in smoke_used),
            key=lambda item: stable_key("smoke", item["video_id"]),
        )
        smoke.append(row)
        smoke_used.add(row["video_id"])
    if len(smoke) < 5:
        extras = sorted(
            (row for row in development if row["video_id"] not in smoke_used),
            key=lambda row: stable_key("smoke-extra", row["video_id"]),
        )
        smoke.extend(extras[: 5 - len(smoke)])

    smoke_ids = {row["video_id"] for row in smoke}
    remaining_dev = sorted(
        (row for row in development if row["video_id"] not in smoke_ids),
        key=lambda row: stable_key("development-order", row["video_id"]),
    )
    remaining = sorted(
        (row for row in selected if row["video_id"] not in {item["video_id"] for item in development}),
        key=lambda row: stable_key("evaluation-order", row["video_id"]),
    )
    ordered = smoke + remaining_dev + remaining
    if len(ordered) != 50 or set(by_id) != {row["video_id"] for row in ordered}:
        raise RuntimeError("Usage ordering changed the frozen video set")
    return ordered


def probe_media(path: Path, ffprobe_path: str) -> dict[str, Any]:
    command = [
        ffprobe_path,
        "-v", "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate",
        "-of", "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"readable": False, "probe_error": completed.stderr.strip() or "ffprobe failed"}
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next((row for row in streams if row.get("codec_type") == "video"), {})
    audio = next((row for row in streams if row.get("codec_type") == "audio"), None)
    return {
        "readable": bool(video),
        "probe_error": None if video else "no video stream",
        "duration_sec": float(payload.get("format", {}).get("duration", 0.0)),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": video.get("avg_frame_rate"),
        "video_codec": video.get("codec_name"),
        "audio_present": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "file_size_bytes": int(payload.get("format", {}).get("size", path.stat().st_size)),
    }


def availability(row: dict[str, Any], data_root: Path, ffprobe_path: str) -> dict[str, Any]:
    relative_path = PurePosixPath("videos") / PurePosixPath(row["video_metadata_path"])
    path = data_root.joinpath(*relative_path.parts)
    if not path.is_file():
        provider = "Google Drive" if "drive.google.com" in row["source_url"] else "Vimeo"
        method = (
            "gdown or authenticated Google Drive download using the supplied video.txt URL"
            if provider == "Google Drive"
            else "yt-dlp Vimeo download using the supplied video.txt URL; authenticated cookies may be required"
        )
        return {
            "relative_video_path": relative_path.as_posix(),
            "availability_status": "missing",
            "download_or_copy_status": "not_downloaded",
            "missing_reason": (
                "No local media file exists. This managed session cannot write D:; "
                "download was not attempted into the Git workspace."
            ),
            "required_download_method": method,
            "provider": provider,
            "media_probe": None,
        }
    probe = probe_media(path, ffprobe_path)
    return {
        "relative_video_path": relative_path.as_posix(),
        "availability_status": "available" if probe["readable"] else "invalid",
        "download_or_copy_status": "existing_verified" if probe["readable"] else "existing_probe_failed",
        "missing_reason": None if probe["readable"] else probe["probe_error"],
        "required_download_method": None,
        "provider": "Google Drive" if "drive.google.com" in row["source_url"] else "Vimeo",
        "media_probe": probe,
    }


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["question_ids_by_type"] = {
        value: list(row["question_ids_by_type"].get(value, [])) for value in MCQ_TYPES
    }
    output["question_count_by_type"] = {
        value: int(row["question_count_by_type"].get(value, 0)) for value in MCQ_TYPES
    }
    output["associated_question_ids"] = [
        question_id for value in MCQ_TYPES for question_id in output["question_ids_by_type"][value]
    ]
    output.pop("minimum_annotation_start_sec", None)
    output.pop("maximum_annotation_end_sec", None)
    return output


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified_durations = [
        row["media_probe"]["duration_sec"]
        for row in rows
        if row["availability_status"] == "available" and row["media_probe"]
    ]
    lower_bounds = [float(row["duration_lower_bound_sec"]) for row in rows]
    counts = Counter()
    for row in rows:
        counts.update(row["question_count_by_type"])
    return {
        "selected_video_count": len(rows),
        "distinct_video_count": len({row["video_id"] for row in rows}),
        "unique_leakage_group_count": len({row["leakage_group"] for row in rows}),
        "associated_question_count": sum(row["question_count"] for row in rows),
        "question_count_by_mcq_duration_type": {value: counts[value] for value in MCQ_TYPES},
        "source_collection_counts": dict(Counter(row["source_collection"] for row in rows)),
        "duration_lower_bound_distribution_sec": {
            "total": sum(lower_bounds),
            "mean": statistics.fmean(lower_bounds),
            "median": statistics.median(lower_bounds),
            "min": min(lower_bounds),
            "max": max(lower_bounds),
            "warning": "These are maximum annotation-end lower bounds, not verified video durations.",
        },
        "verified_duration_distribution_sec": (
            {
                "total": sum(verified_durations),
                "mean": statistics.fmean(verified_durations),
                "median": statistics.median(verified_durations),
                "min": min(verified_durations),
                "max": max(verified_durations),
            }
            if verified_durations else None
        ),
        "available_video_count": sum(row["availability_status"] == "available" for row in rows),
        "missing_video_count": sum(row["availability_status"] == "missing" for row in rows),
        "invalid_video_count": sum(row["availability_status"] == "invalid" for row in rows),
        "videos_with_audio": sum(
            bool(row.get("media_probe") and row["media_probe"].get("audio_present")) for row in rows
        ),
    }


def write_outputs(
    rows: list[dict[str, Any]], *, manifest_path: Path, csv_path: Path, missing_path: Path,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    stats = aggregate(rows)
    manifest = {
        "schema_version": "egopolice-frozen-50-v1",
        "selection_seed": SELECTION_SEED,
        "selection_method": (
            "Question-performance-independent deterministic stratification over source collection and "
            "maximum MCQ annotation-end duration lower-bound bins; prefers videos present in 1s, 10s, "
            "and 60s metadata; stable SHA-256 ordering; unique leakage groups where metadata permits."
        ),
        "selection_uses_model_performance": False,
        "selection_uses_gold_answers": False,
        "selection_uses_ground_truth_timestamps_for_b0_sampling": False,
        "ground_truth_timestamp_policy": "retained for later evaluation/oracle/grounding only",
        "usage_policy": {
            "smoke_debug": "selection ranks 1-5",
            "development_fast_iteration": "selection ranks 1-10",
            "main_stagewise_evaluation": "all 50",
            "threshold_tuning_on_full_50_prohibited": True,
        },
        "source_metadata": ["mcq_1s.json", "mcq_10s.json", "mcq_60s.json", "video.txt"],
        "source_metadata_sha256": source_hashes,
        "frozen_stratum_quotas": SOURCE_QUOTAS,
        "statistics": stats,
        "videos": rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "selection_rank", "usage_designation", "video_id", "case_folder", "source_collection",
        "duration_lower_bound_sec", "verified_duration_sec", "question_count", "question_count_1s",
        "question_count_10s", "question_count_60s", "availability_status", "relative_video_path",
        "provider", "audio_present", "file_size_bytes", "missing_reason",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            probe = row.get("media_probe") or {}
            writer.writerow(
                {
                    "selection_rank": row["selection_rank"],
                    "usage_designation": row["usage_designation"],
                    "video_id": row["video_id"],
                    "case_folder": row["case_folder"],
                    "source_collection": row["source_collection"],
                    "duration_lower_bound_sec": row["duration_lower_bound_sec"],
                    "verified_duration_sec": probe.get("duration_sec"),
                    "question_count": row["question_count"],
                    "question_count_1s": row["question_count_by_type"]["1s"],
                    "question_count_10s": row["question_count_by_type"]["10s"],
                    "question_count_60s": row["question_count_by_type"]["60s"],
                    "availability_status": row["availability_status"],
                    "relative_video_path": row["relative_video_path"],
                    "provider": row["provider"],
                    "audio_present": probe.get("audio_present"),
                    "file_size_bytes": probe.get("file_size_bytes"),
                    "missing_reason": row["missing_reason"],
                }
            )
    missing = [row for row in rows if row["availability_status"] != "available"]
    missing_path.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and audit the deterministic EgoPolice 50-video pool")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "")))
    parser.add_argument("--ffprobe-path", default=os.environ.get("FFPROBE_PATH", "ffprobe"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--missing-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root or not args.data_root.is_dir():
        raise SystemExit(f"DATA_ROOT does not exist: {args.data_root}")
    selected = usage_order(select_frozen_50(load_candidates(args.data_root)))
    rows = []
    for rank, candidate in enumerate(selected, start=1):
        row = serialize_row(candidate)
        row.update(availability(row, args.data_root, args.ffprobe_path))
        row["selection_rank"] = rank
        row["usage_designation"] = (
            "smoke_debug" if rank <= 5 else "development_fast_iteration" if rank <= 10 else "main_evaluation"
        )
        row["included_in_main_evaluation_50"] = True
        row["selection_reason"] = (
            f"Frozen {row['source_collection']}/{row['duration_bin']} stratum; all three MCQ duration "
            "types available; stable seeded order; no performance or answer use."
        )
        rows.append(row)
    source_names = ["mcq_1s.json", "mcq_10s.json", "mcq_60s.json", "video.txt"]
    source_hashes = {
        name: hashlib.sha256((args.data_root / name).read_bytes()).hexdigest()
        for name in source_names
    }
    manifest = write_outputs(
        rows, manifest_path=args.manifest, csv_path=args.summary_csv, missing_path=args.missing_json,
        source_hashes=source_hashes,
    )
    print(json.dumps(manifest["statistics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
