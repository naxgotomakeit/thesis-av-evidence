from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.egopolice_formal import (
    QUESTION_DURATION_CLASSES,
    VIDEO_DURATION_BINS,
    source_video_duration_bin,
)


def _probe(path: Path, ffprobe_path: str) -> dict[str, Any]:
    completed = subprocess.run([
        ffprobe_path, "-v", "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"readable": False, "error": completed.stderr.strip() or "ffprobe failed"}
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    try:
        duration = float(payload.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        duration = 0.0
    if not video or duration <= 0:
        return {"readable": False, "error": "missing video stream or non-positive duration"}
    return {
        "readable": True,
        "error": None,
        "duration_sec": duration,
        "source_video_duration_bin": source_video_duration_bin(duration),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": str(video.get("avg_frame_rate") or ""),
        "video_codec": video.get("codec_name"),
        "audio_present": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "file_size_bytes": int(payload.get("format", {}).get("size", path.stat().st_size)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal EgoPolice ablation20 readiness")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--video-manifest", type=Path,
        default=ROOT / "config/data/egopolice_ablation20_v1.json",
    )
    parser.add_argument(
        "--question-manifest", type=Path,
        default=ROOT / "config/data/egopolice_ablation_questions_v1.json",
    )
    parser.add_argument(
        "--download-log", type=Path,
        default=ROOT / "outputs/data_audit/egopolice_ablation20_download_attempt.json",
    )
    parser.add_argument(
        "--json-output", type=Path,
        default=ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json",
    )
    parser.add_argument(
        "--csv-output", type=Path,
        default=ROOT / "outputs/data_audit/egopolice_ablation20_readiness.csv",
    )
    parser.add_argument(
        "--question-output", type=Path,
        default=ROOT / "outputs/data_audit/egopolice_ablation98_pre_evaluation.json",
    )
    parser.add_argument(
        "--distribution-output", type=Path,
        default=ROOT / "outputs/data_audit/egopolice_ablation98_duration_distribution.csv",
    )
    parser.add_argument("--ffprobe-path", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    videos = json.loads(args.video_manifest.read_text(encoding="utf-8"))["videos"]
    questions = json.loads(args.question_manifest.read_text(encoding="utf-8"))["questions"]
    question_counts = Counter(row["duration_class"] for row in questions)
    question_video_ids = {row["video_id"] for row in questions}
    manifest_video_ids = {row["video_id"] for row in videos}
    if (
        len(videos) != 20 or len(manifest_video_ids) != 20
        or len(questions) != 98
        or len({row["question_id"] for row in questions}) != 98
        or question_video_ids != manifest_video_ids
        or question_counts != Counter({"1s": 39, "10s": 38, "60s": 21})
    ):
        raise ValueError("Frozen 20-video/98-question contract failed")
    for question in questions:
        start_sec, end_sec = (float(value) for value in question["gt_interval_sec"])
        answer = int(question["ground_truth_index"])
        if end_sec <= start_sec or not 0 <= answer < 5 or not question["ground_truth_text"]:
            raise ValueError(f"Invalid frozen question metadata: {question['question_id']}")
    download_results = {}
    if args.download_log.is_file():
        download_results = {
            row["video_id"]: row
            for row in json.loads(args.download_log.read_text(encoding="utf-8"))["results"]
        }
    audit_rows = []
    for video in videos:
        relative = PurePosixPath(str(video["relative_video_path"]))
        path = args.data_root.joinpath(*relative.parts)
        download = download_results.get(video["video_id"], {})
        if path.is_file():
            probe = _probe(path, args.ffprobe_path)
            status = "ready" if probe["readable"] else "failed"
        else:
            probe = {"readable": False, "error": None}
            status = "blocked" if download.get("status") == "blocked_not_attempted" or download.get("blocker") else "missing"
        audit_rows.append({
            "manifest_video_id": video["video_id"],
            "actual_file_path": str(path),
            "file_exists": path.is_file(),
            "status": status,
            "duration_sec": probe.get("duration_sec"),
            "source_video_duration_bin": probe.get("source_video_duration_bin"),
            "width": probe.get("width"), "height": probe.get("height"),
            "fps": probe.get("fps"), "video_codec": probe.get("video_codec"),
            "audio_present": probe.get("audio_present"),
            "audio_codec": probe.get("audio_codec"),
            "file_size_bytes": probe.get("file_size_bytes", 0),
            "readable_valid": probe["readable"],
            "mapping_unambiguous": True,
            "error": probe.get("error") or download.get("error") or download.get("blocker") or download.get("stderr_tail"),
        })
    by_video = {row["manifest_video_id"]: row for row in audit_rows}
    enriched_questions = []
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for question in questions:
        video = by_video[question["video_id"]]
        duration_bin = video["source_video_duration_bin"] or "unavailable"
        distribution[duration_bin][question["duration_class"]] += 1
        enriched_questions.append({
            "question_id": question["question_id"],
            "source_video_id": question["video_id"],
            "question_duration_class": question["duration_class"],
            "full_source_video_duration_sec": video["duration_sec"],
            "source_video_duration_bin": video["source_video_duration_bin"],
            "gt_interval_sec": question["gt_interval_sec"],
            "ground_truth_index": question["ground_truth_index"],
            "ground_truth_text": question["ground_truth_text"],
            "source_video_ready": video["status"] == "ready",
        })
    status_counts = Counter(row["status"] for row in audit_rows)
    duration_counts = Counter(
        row["source_video_duration_bin"] for row in audit_rows if row["status"] == "ready"
    )
    payload = {
        "schema_version": "egopolice-ablation20-readiness-v1",
        "video_manifest": str(args.video_manifest),
        "question_manifest": str(args.question_manifest),
        "summary": {
            "target_videos": 20,
            "ready": status_counts["ready"],
            "missing": status_counts["missing"],
            "blocked": status_counts["blocked"],
            "failed": status_counts["failed"],
            "total_ready_storage_bytes": sum(row["file_size_bytes"] for row in audit_rows),
            "actual_duration_bin_video_counts": {
                name: duration_counts[name] for name in VIDEO_DURATION_BINS
            },
            "duration_bin_unavailable_videos": 20 - status_counts["ready"],
        },
        "videos": audit_rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(audit_rows[0])
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(audit_rows)
    question_payload = {
        "schema_version": "egopolice-ablation98-pre-evaluation-v1",
        "formal_evaluation_ready": status_counts["ready"] == 20,
        "question_count": len(enriched_questions),
        "distinct_video_count": len(question_video_ids),
        "question_count_by_duration_class": {
            name: question_counts[name] for name in QUESTION_DURATION_CLASSES
        },
        "questions": enriched_questions,
    }
    args.question_output.write_text(json.dumps(question_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bins = [*VIDEO_DURATION_BINS, "unavailable"]
    with args.distribution_output.open("w", encoding="utf-8", newline="") as handle:
        fields = ["source_video_duration_bin", "number_of_videos", "number_of_questions", "questions_1s", "questions_10s", "questions_60s"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in bins:
            counts = distribution[name]
            video_count = sum(
                1 for row in audit_rows
                if (row["source_video_duration_bin"] or "unavailable") == name
            )
            writer.writerow({
                "source_video_duration_bin": name,
                "number_of_videos": video_count,
                "number_of_questions": sum(counts.values()),
                "questions_1s": counts["1s"],
                "questions_10s": counts["10s"],
                "questions_60s": counts["60s"],
            })
    print(json.dumps({"summary": payload["summary"], "question_distribution": {name: dict(distribution[name]) for name in bins}}, indent=2))
    return 0 if status_counts["ready"] == 20 else 2


if __name__ == "__main__":
    raise SystemExit(main())
