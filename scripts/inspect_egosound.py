#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.egosound import (AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, distribution, ffprobe_media,
    index_by_stem, load_annotations, match_media, parse_provided_timestamp, relative_path,
    validate_manifest_row, wav_signal_activity)
from scripts.select_dev_cases_10 import select_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect EgoSound annotations and media without modifying raw data.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--duration-mismatch-seconds", type=float, default=0.5)
    parser.add_argument("--dev-case-count", type=int, default=10, choices=range(5, 11), metavar="{5..10}")
    parser.add_argument("--silence-rms-threshold", type=float, default=0.001)
    parser.add_argument("--mostly-silent-active-fraction", type=float, default=0.05)
    return parser.parse_args()


def classify_case(entry):
    qtype = str(entry.get("question_type") or "").lower()
    text = " ".join(str(entry.get(k) or "") for k in ("question", "context")).lower()
    if "cross-modal" in qtype or (any(x in text for x in ("visually", "camera shows", "look", "visible")) and any(x in text for x in ("sound", "heard", "says", "voice"))):
        return "cross_modal_candidate"
    if any(x in text for x in ("says", "said", "voice", "spoken", "conversation", "asks", "replies")):
        return "speech_candidate"
    if any(x in text for x in ("sound", "heard", "clink", "thud", "rustl", "music", "scrap", "tap", "gurg", "swish")):
        return "non_speech_audio_candidate"
    if any(x in text for x in ("visually", "visible", "what card", "what object", "camera", "color")):
        return "visual_candidate"
    return "general_av_candidate"


def main():
    args = parse_args()
    root = args.dataset_root.resolve()
    out = args.output_root.resolve()
    manifest_dir = out / "data" / "manifests"
    report_dir = out / "outputs" / "dataset_inspection"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    all_files = [p for p in root.rglob("*") if p.is_file()]
    json_files = [p for p in all_files if p.suffix.lower() == ".json"]
    videos, duplicate_videos = index_by_stem([p for p in all_files if p.suffix.lower() in VIDEO_EXTENSIONS])
    audios, duplicate_audios = index_by_stem([p for p in all_files if p.suffix.lower() in AUDIO_EXTENSIONS])
    ffprobe = args.ffprobe or shutil.which("ffprobe")
    if not ffprobe:
        raise SystemExit("ffprobe was not found; pass --ffprobe PATH")

    annotation_files = []
    entries = []
    global_warnings = []
    schema_counts = Counter()
    for path in json_files:
        try:
            loaded = load_annotations(path)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            global_warnings.append({"id": relative_path(path, root), "warning": f"annotation_parse_failed:{exc}"})
            continue
        if not loaded:
            continue
        annotation_files.append(relative_path(path, root))
        entries.extend((entry, path) for entry in loaded)
        for entry in loaded:
            schema_counts.update(entry.keys())

    video_meta, audio_meta, activity_meta = {}, {}, {}
    media_warnings = defaultdict(list)
    referenced_video_ids = set()
    for entry, _ in entries:
        video_id, video, audio, _ = match_media(entry, videos, audios)
        if video_id:
            referenced_video_ids.add(video_id)
        if video and video_id not in video_meta:
            video_meta[video_id], warns = ffprobe_media(video, ffprobe)
            media_warnings[(video_id, "video")].extend(warns)
        if audio and video_id not in audio_meta:
            audio_meta[video_id], warns = ffprobe_media(audio, ffprobe)
            media_warnings[(video_id, "audio")].extend(warns)
            if audio.suffix.lower() == ".wav":
                activity_meta[video_id], warns = wav_signal_activity(audio, args.silence_rms_threshold)
                media_warnings[(video_id, "audio")].extend(warns)

    rows = []
    failed_ids = []
    for ordinal, (entry, annotation_path) in enumerate(entries, 1):
        video_id, video, audio, warnings = match_media(entry, videos, audios)
        warnings += [f"video:{x}" for x in media_warnings[(video_id, "video")]]
        warnings += [f"audio:{x}" for x in media_warnings[(video_id, "audio")]]
        vm, am = video_meta.get(video_id, {}), audio_meta.get(video_id, {})
        vd, ad = vm.get("duration"), am.get("duration")
        mismatch = abs(vd - ad) if vd is not None and ad is not None else None
        if mismatch is not None and mismatch > args.duration_mismatch_seconds:
            warnings.append(f"video_audio_duration_mismatch:{mismatch:.3f}s")
        case_id = str(entry.get("question_id") or f"{video_id or 'unknown'}_{ordinal}")
        media_valid = bool(video and audio and vm.get("has_video") and am.get("has_audio") and vd and ad and not any("ffprobe_failed" in x for x in warnings))
        provided_start, provided_end, timestamp_warnings = parse_provided_timestamp(entry.get("timestamp"))
        warnings += timestamp_warnings
        row = {
            "case_id": case_id, "source_subset": "ego4d", "video_id": video_id,
            "video_path": relative_path(video, root), "audio_path": relative_path(audio, root),
            "question": entry.get("question"), "answer": entry.get("answer"),
            "question_type": entry.get("question_type"),
            "provided_timestamp_start": provided_start, "provided_timestamp_end": provided_end,
            "provided_context": entry.get("context"), "evidence_source": None,
            "video_duration": vd, "audio_duration": ad,
            "video_fps": vm.get("fps"), "video_width": vm.get("width"), "video_height": vm.get("height"),
            "video_has_embedded_audio": vm.get("has_audio"), "audio_sample_rate": am.get("sample_rate"),
            "audio_channels": am.get("channels"), "media_valid": media_valid, "warnings": sorted(set(warnings)),
            "provided_timestamp": entry.get("timestamp"),
            "annotation_video_path": entry.get("video_path"), "annotation_file": relative_path(annotation_path, root),
            "signal_active_fraction": activity_meta.get(video_id, {}).get("signal_active_fraction"),
            "mean_window_rms": activity_meta.get(video_id, {}).get("mean_window_rms"),
        }
        schema_errors = validate_manifest_row(row)
        if schema_errors:
            row["warnings"].extend(schema_errors)
            row["media_valid"] = False
        if not row["media_valid"]:
            failed_ids.append(case_id)
        rows.append(row)

    manifest_path = manifest_dir / "egosound_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    valid_rows = [r for r in rows if r["media_valid"] and (r["signal_active_fraction"] is None or r["signal_active_fraction"] >= args.mostly_silent_active_fraction)]
    selected = select_cases(valid_rows, args.dev_case_count)
    type_counts = Counter(x["suspected_evidence_type"] for x in selected)
    dev_path = report_dir / "dev_cases.json"
    dev_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    unique_ref = sorted(referenced_video_ids)
    diffs = [abs(video_meta[v]["duration"] - audio_meta[v]["duration"]) for v in unique_ref if video_meta.get(v, {}).get("duration") is not None and audio_meta.get(v, {}).get("duration") is not None]
    mostly_silent = [v for v, m in activity_meta.items() if m.get("signal_active_fraction", 0) < args.mostly_silent_active_fraction]
    valid_video_ids = [v for v in unique_ref if video_meta.get(v, {}).get("has_video") and video_meta[v].get("duration")]
    valid_audio_ids = [v for v in unique_ref if audio_meta.get(v, {}).get("has_audio") and audio_meta[v].get("duration")]
    unreadable = sorted({v for v in unique_ref if any("ffprobe" in w or "invalid_or_missing" in w or "missing_" in w for kind in ("video", "audio") for w in media_warnings[(v, kind)])})
    report = {
        "dataset_root": str(root), "discovered_structure": {"annotation_files": annotation_files, "video_directories": sorted({relative_path(p.parent, root) for p in videos.values()}), "audio_directories": sorted({relative_path(p.parent, root) for p in audios.values()}), "file_extension_counts": dict(Counter(p.suffix.lower() for p in all_files))},
        "annotation_schema": {"root_type": "array", "fields": dict(schema_counts), "timestamp_policy": "Annotation timestamp is parsed into provided_timestamp_start/end and retained verbatim as provided_timestamp. It is not treated as gold evidence; evidence_source remains null."},
        "media_matching_rule": "Use stem of annotation video_path as video_id; match video and external audio by identical stem. Cross-check question_id begins with video_id plus underscore.",
        "total_annotation_entries": len(rows), "unique_videos": len(unique_ref),
        "valid_cases": sum(r["media_valid"] for r in rows),
        "cases_with_provided_timestamps": sum(bool(r.get("provided_timestamp")) for r in rows),
        "cases_with_parsed_single_interval_timestamps": sum(r["provided_timestamp_start"] is not None and r["provided_timestamp_end"] is not None for r in rows),
        "valid_video_files": len(valid_video_ids), "valid_external_audio_files": len(valid_audio_ids),
        "videos_with_embedded_audio_tracks": sum(bool(video_meta.get(v, {}).get("has_audio")) for v in unique_ref),
        "missing_video_count": sum(v not in videos for v in unique_ref), "missing_audio_count": sum(v not in audios for v in unique_ref),
        "annotation_to_media_matching_failures": sum(not r["video_id"] or r["video_path"] is None or r["audio_path"] is None for r in rows),
        "unreadable_or_corrupt_media_count": len(unreadable),
        "video_audio_duration_mismatch_statistics": {"threshold_seconds": args.duration_mismatch_seconds, "count_over_threshold": sum(x > args.duration_mismatch_seconds for x in diffs), "absolute_difference_seconds": distribution(diffs)},
        "duration_distribution_seconds": {"video": distribution([m["duration"] for m in video_meta.values() if m.get("duration") is not None]), "audio": distribution([m["duration"] for m in audio_meta.values() if m.get("duration") is not None])},
        "signal_activity": {"method": "One-second PCM WAV RMS windows; this measures signal activity, not speech.", "speech_activity_available": False, "speech_activity_unavailable_reason": "No reliable speech detector was used because model/dependency downloads are out of scope.", "analyzed_audio_files": len(activity_meta), "mostly_silent_threshold_active_fraction": args.mostly_silent_active_fraction, "mostly_silent_count": len(mostly_silent), "signal_active_count": len(activity_meta) - len(mostly_silent), "active_fraction_distribution": distribution([m["signal_active_fraction"] for m in activity_meta.values() if m.get("signal_active_fraction") is not None])},
        "development_selection": {"requested": args.dev_case_count, "selected": len(selected), "unique_videos": len({x["video_id"] for x in selected}), "suspected_evidence_type_counts": dict(type_counts), "ground_truth_evidence_type": False},
        "duplicates": {"video_stems": duplicate_videos, "audio_stems": duplicate_audios},
        "warnings": global_warnings + [{"id": r["case_id"], "warnings": r["warnings"]} for r in rows if r["warnings"]],
        "failed_ids": sorted(set(failed_ids + unreadable)),
    }
    report_path = report_dir / "health_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"""# EgoSound dataset health report

## Structure and schema

The extracted subset contains `{', '.join(annotation_files)}`, video directories `{', '.join(report['discovered_structure']['video_directories'])}`, and audio directories `{', '.join(report['discovered_structure']['audio_directories'])}`. The annotation root is a JSON array with observed fields: {', '.join(schema_counts)}.

Media are matched by the five-digit stem from `video_path`: the MP4 stem must equal the WAV stem, while `question_id` is cross-checked against `<video_id>_<question number>`. Annotation `timestamp` is parsed into `provided_timestamp_start` and `provided_timestamp_end` and retained verbatim as `provided_timestamp`; it is not treated as gold evidence. `evidence_source` remains null.

## Health statistics

- Annotation entries: {len(rows)}; unique referenced videos: {len(unique_ref)}
- Valid QA cases: {sum(r['media_valid'] for r in rows)}; cases with provided timestamps: {sum(bool(r.get('provided_timestamp')) for r in rows)}; parsed as one start/end interval: {sum(r['provided_timestamp_start'] is not None and r['provided_timestamp_end'] is not None for r in rows)}
- Valid videos: {len(valid_video_ids)}; valid external audio files: {len(valid_audio_ids)}
- Videos with embedded audio: {report['videos_with_embedded_audio_tracks']}
- Missing video/audio: {report['missing_video_count']} / {report['missing_audio_count']}
- Unreadable/corrupt media: {len(unreadable)}
- Video/audio duration mismatches over {args.duration_mismatch_seconds}s: {report['video_audio_duration_mismatch_statistics']['count_over_threshold']}
- Cheap RMS signal check: {len(activity_meta) - len(mostly_silent)} signal-active, {len(mostly_silent)} mostly silent. This is not speech detection; speech-active versus non-speech cannot be estimated reliably without additional dependencies/models and is reported unavailable.

## Development cases and unresolved issues

Selected {len(selected)} valid cases from {len({x['video_id'] for x in selected})} unique videos. `suspected_evidence_type` labels are annotation-text heuristics, not ground truth. Each row includes `selection_reason`. Unresolved warnings and failed IDs are recorded in `health_report.json`; there are {len(report['warnings'])} warning records and {len(report['failed_ids'])} failed IDs.
"""
    (report_dir / "health_report.md").write_text(md, encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "health_report_json": str(report_path), "health_report_md": str(report_dir / 'health_report.md'), "dev_cases": str(dev_path), "annotations": len(rows), "matched_videos": sum(v in videos for v in unique_ref), "matched_audios": sum(v in audios for v in unique_ref), "valid_cases": sum(r["media_valid"] for r in rows), "cases_with_provided_timestamps": sum(bool(r.get("provided_timestamp")) for r in rows), "cases_with_parsed_single_interval_timestamps": sum(r["provided_timestamp_start"] is not None and r["provided_timestamp_end"] is not None for r in rows), "selected_dev_cases": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
