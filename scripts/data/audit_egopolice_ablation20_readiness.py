from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
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


EXPECTED_VIDEO_MANIFEST_SHA256 = "0cb8d55962634d900d440f943a7d91ca5fe5473f3f5d0c096c826685acd3e1c5"
EXPECTED_QUESTION_MANIFEST_SHA256 = "fca734b9764ed132483ba3858db34243b50384ba2f1e115197c15762adcb23a5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    actual_size = path.stat().st_size
    format_size = int(payload.get("format", {}).get("size", actual_size))
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
        "file_size_bytes": actual_size,
        "ffprobe_format_size_bytes": format_size,
        "file_size_matches_ffprobe": actual_size == format_size,
    }


def _packet_scan(path: Path, ffprobe_path: str) -> dict[str, Any]:
    """Read every demuxed packet so a valid header alone cannot pass readiness."""
    started = time.perf_counter()
    completed = subprocess.run([
        ffprobe_path, "-v", "warning", "-count_packets",
        "-show_entries", "stream=index,codec_type,nb_read_packets",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=False)
    warnings = [line for line in completed.stderr.splitlines() if line.strip()]
    try:
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except json.JSONDecodeError:
        payload = {}
    streams = payload.get("streams", [])
    video_packets = sum(
        int(stream.get("nb_read_packets") or 0)
        for stream in streams if stream.get("codec_type") == "video"
    )
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    audio_packets = sum(int(stream.get("nb_read_packets") or 0) for stream in audio_streams)
    passed = (
        completed.returncode == 0
        and video_packets > 0
        and (not audio_streams or audio_packets > 0)
        and not warnings
    )
    return {
        "full_packet_scan_passed": passed,
        "full_packet_scan_returncode": completed.returncode,
        "video_packet_count": video_packets,
        "audio_packet_count": audio_packets,
        "full_packet_scan_warnings": warnings,
        "full_packet_scan_latency_sec": time.perf_counter() - started,
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
    video_manifest_sha = _sha256(args.video_manifest)
    question_manifest_sha = _sha256(args.question_manifest)
    if video_manifest_sha != EXPECTED_VIDEO_MANIFEST_SHA256:
        raise ValueError(
            f"Frozen video manifest SHA256 mismatch: {video_manifest_sha}"
        )
    if question_manifest_sha != EXPECTED_QUESTION_MANIFEST_SHA256:
        raise ValueError(
            f"Frozen question manifest SHA256 mismatch: {question_manifest_sha}"
        )
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
    official_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    for metadata_name in sorted({str(row["metadata_source_file"]) for row in questions}):
        metadata_path = args.data_root / metadata_name
        if not metadata_path.is_file():
            raise ValueError(f"Missing official MCQ metadata: {metadata_path}")
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        official_metadata[metadata_name] = {str(row["id"]): row for row in rows}
        if len(official_metadata[metadata_name]) != len(rows):
            raise ValueError(f"Duplicate IDs in official MCQ metadata: {metadata_path}")
    max_frozen_gt_end_by_video: dict[str, float] = defaultdict(float)
    for question in questions:
        start_sec, end_sec = (float(value) for value in question["gt_interval_sec"])
        answer = int(question["ground_truth_index"])
        options = question.get("options")
        if (
            end_sec <= start_sec or not 0 <= answer < 5
            or not isinstance(options, list) or len(options) != 5
            or question["ground_truth_text"] != options[answer]
        ):
            raise ValueError(f"Invalid frozen question metadata: {question['question_id']}")
        metadata_name = str(question["metadata_source_file"])
        official = official_metadata[metadata_name].get(str(question["question_id"]))
        if official is None:
            raise ValueError(f"Question absent from official metadata: {question['question_id']}")
        official_video_id = str(official.get("video") or "").removesuffix(".mp4")
        official_interval = [float(official["start second"]), float(official["end second"])]
        if (
            official_video_id != question["video_id"]
            or str(official.get("question")) != question["question"]
            or list(official.get("options") or []) != options
            or int(official.get("answer")) != answer
            or official_interval != [start_sec, end_sec]
        ):
            raise ValueError(f"Frozen/official metadata mismatch: {question['question_id']}")
        max_frozen_gt_end_by_video[question["video_id"]] = max(
            max_frozen_gt_end_by_video[question["video_id"]], end_sec
        )
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
            strict_errors: list[str] = []
            if probe["readable"]:
                probe.update(_packet_scan(path, args.ffprobe_path))
                duration = float(probe["duration_sec"])
                lower_bound = float(video["duration_lower_bound_sec"])
                max_frozen_gt_end = max_frozen_gt_end_by_video[video["video_id"]]
                probe["duration_lower_bound_sec"] = lower_bound
                probe["max_frozen_gt_interval_end_sec"] = max_frozen_gt_end
                probe["duration_covers_metadata_lower_bound"] = duration >= lower_bound
                probe["duration_covers_all_frozen_gt_intervals"] = duration >= max_frozen_gt_end
                if not probe["file_size_matches_ffprobe"]:
                    strict_errors.append("filesystem/ffprobe size mismatch")
                if not probe["full_packet_scan_passed"]:
                    strict_errors.append("full ffprobe packet scan failed or warned")
                if not probe["duration_covers_metadata_lower_bound"]:
                    strict_errors.append("duration below all-question metadata lower bound")
                if not probe["duration_covers_all_frozen_gt_intervals"]:
                    strict_errors.append("duration does not cover frozen GT intervals")
            else:
                strict_errors.append(str(probe.get("error") or "metadata probe failed"))
            probe["strict_validation_errors"] = strict_errors
            probe["readable"] = probe["readable"] and not strict_errors
            status = "ready" if probe["readable"] else "failed"
        else:
            probe = {"readable": False, "error": None}
            status = "blocked" if download.get("status") == "blocked_not_attempted" or download.get("blocker") else "missing"
        current_error = (
            "; ".join(probe.get("strict_validation_errors") or [])
            or probe.get("error")
            or (download.get("error") or download.get("blocker") or download.get("stderr_tail") if status != "ready" else None)
        )
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
            "ffprobe_format_size_bytes": probe.get("ffprobe_format_size_bytes"),
            "file_size_matches_ffprobe": probe.get("file_size_matches_ffprobe"),
            "video_packet_count": probe.get("video_packet_count"),
            "audio_packet_count": probe.get("audio_packet_count"),
            "full_packet_scan_passed": probe.get("full_packet_scan_passed"),
            "full_packet_scan_warnings": probe.get("full_packet_scan_warnings"),
            "full_packet_scan_latency_sec": probe.get("full_packet_scan_latency_sec"),
            "duration_lower_bound_sec": probe.get("duration_lower_bound_sec"),
            "max_frozen_gt_interval_end_sec": probe.get("max_frozen_gt_interval_end_sec"),
            "duration_covers_metadata_lower_bound": probe.get("duration_covers_metadata_lower_bound"),
            "duration_covers_all_frozen_gt_intervals": probe.get("duration_covers_all_frozen_gt_intervals"),
            "readable_valid": probe["readable"],
            "mapping_unambiguous": True,
            "error": current_error,
            "historical_download_note": download.get("error") or download.get("blocker") or download.get("stderr_tail"),
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
            "official_metadata_mapping_valid": True,
        })
    status_counts = Counter(row["status"] for row in audit_rows)
    duration_counts = Counter(
        row["source_video_duration_bin"] for row in audit_rows if row["status"] == "ready"
    )
    payload = {
        "schema_version": "egopolice-ablation20-readiness-v1",
        "video_manifest": str(args.video_manifest),
        "video_manifest_sha256": video_manifest_sha,
        "question_manifest": str(args.question_manifest),
        "question_manifest_sha256": question_manifest_sha,
        "validation_method": {
            "metadata_probe": "ffprobe format/stream metadata",
            "truncation_check": "ffprobe full-file packet count with warnings treated as failure",
            "duration_checks": "validated duration must cover parent metadata lower bound and every frozen GT interval",
            "question_mapping": "all frozen fields matched exactly against official mcq_1s/mcq_10s/mcq_60s metadata",
        },
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
        "video_manifest_sha256": video_manifest_sha,
        "question_manifest_sha256": question_manifest_sha,
        "question_count": len(enriched_questions),
        "ready_question_count": sum(row["source_video_ready"] for row in enriched_questions),
        "distinct_video_count": len(question_video_ids),
        "official_metadata_mapping_verified": True,
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
