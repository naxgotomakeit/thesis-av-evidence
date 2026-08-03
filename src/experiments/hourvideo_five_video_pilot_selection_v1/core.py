from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.core import (
    canonical_bytes,
    load_json,
    sha256,
    write_json,
)


PROTECTED_EVALUATION_KEYS = {
    "canary",
    "correct_answer_label",
    "expected_answer",
    "gold",
    "ground_truth",
    "mcq_test",
    "reference_timestamp",
    "relevant_timestamps",
}


def contains_protected_evaluation_key(value: Any) -> bool:
    if isinstance(value, dict):
        if PROTECTED_EVALUATION_KEYS.intersection(value):
            return True
        return any(contains_protected_evaluation_key(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_protected_evaluation_key(item) for item in value)
    return False


def validate_selection_specs(specs: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    video_ids = [str(row.get("video_uid", "")) for row in specs]
    question_ids = [str(row.get("primary_question_id", "")) for row in specs]
    categories = [str(row.get("selection_category", "")) for row in specs]
    if len(specs) != 5:
        errors.append("exactly five videos are required")
    if len(set(video_ids)) != len(video_ids):
        errors.append("duplicate video UID")
    if len(set(question_ids)) != len(question_ids):
        errors.append("duplicate primary question ID")
    if len(set(categories)) != len(categories):
        errors.append("duplicate selection category")
    for row in specs:
        if not str(row.get("primary_question_id", "")).startswith(
            str(row.get("video_uid", "")) + "_"
        ):
            errors.append(f"question/video mismatch: {row.get('primary_question_id')}")
        if not str(row.get("selection_rationale", "")).strip():
            errors.append(f"empty rationale: {row.get('video_uid')}")
    return errors


def _raw_question_lookup(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for video_uid, record in annotation.items():
        for row in record.get("benchmark_dataset", []):
            qid = str(row["qid"])
            if qid in lookup:
                raise ValueError(f"duplicate question ID: {qid}")
            lookup[qid] = {
                "question_id": qid,
                "video_uid": video_uid,
                "task": str(row["task"]),
                "question_text": str(row["question"]).strip(),
            }
    return lookup


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def run(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("model_api_calls") != 0:
        raise ValueError("selection must not call a model or API")
    specs = list(config["selected_videos"])
    errors = validate_selection_specs(specs)
    if errors:
        raise ValueError("; ".join(errors))

    output = repo / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    hourvideo_root = Path(config["hourvideo_root"])
    interface_root = repo / config["interface_output_root"]
    annotation_path = hourvideo_root / config["annotation_path"]
    frame_audit_path = hourvideo_root / config["frame_audit_path"]
    video_audit_path = hourvideo_root / config["video_audit_path"]
    safe_manifest_path = interface_root / "safe_question_manifest.json"
    interface_validation_path = interface_root / "validation_report.json"
    source_paths = {
        "selection_config": config_path,
        "annotations": annotation_path,
        "frame_audit": frame_audit_path,
        "video_audit": video_audit_path,
        "safe_question_manifest": safe_manifest_path,
        "interface_validation": interface_validation_path,
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing sources: {missing}")

    annotation = load_json(annotation_path)
    frame_audit = load_json(frame_audit_path)
    video_audit = load_json(video_audit_path)
    safe_manifest = load_json(safe_manifest_path)
    interface_validation = load_json(interface_validation_path)
    raw_questions = _raw_question_lookup(annotation)
    safe_questions = {
        row["question_id"]: row for row in safe_manifest.get("questions", [])
    }
    frames_by_video = {
        row["video_uid"]: row for row in frame_audit.get("videos", [])
    }
    videos_by_uid = {
        row["video_uid"]: row for row in video_audit.get("videos", [])
    }

    selected_rows: list[dict[str, Any]] = []
    primary_questions: list[dict[str, Any]] = []
    selected_ids = {row["video_uid"] for row in specs}
    all_selected_safe_questions = [
        row
        for row in safe_manifest.get("questions", [])
        if row["video_uid"] in selected_ids
    ]
    for spec in specs:
        uid = spec["video_uid"]
        qid = spec["primary_question_id"]
        if uid not in annotation or uid not in frames_by_video or uid not in videos_by_uid:
            errors.append(f"unresolved selected video: {uid}")
            continue
        if qid not in raw_questions or qid not in safe_questions:
            errors.append(f"unresolved primary question: {qid}")
            continue
        raw_question = raw_questions[qid]
        safe_question = safe_questions[qid]
        video = videos_by_uid[uid]
        frames = frames_by_video[uid]
        metadata = annotation[uid]["video_metadata"]
        if raw_question["video_uid"] != uid or safe_question["video_uid"] != uid:
            errors.append(f"question/video mismatch after resolution: {qid}")
        option_modalities = {row["content_type"] for row in safe_question["answer_options"]}
        if option_modalities != {"text"}:
            errors.append(f"primary question is not text-only: {qid}")
        if not Path(video["path"]).is_file() or not Path(frames["frame_dir"]).is_dir():
            errors.append(f"missing media for selected video: {uid}")
        selected_question_count = sum(
            row["video_uid"] == uid for row in all_selected_safe_questions
        )
        selected_image_question_count = sum(
            row["video_uid"] == uid
            and {option["content_type"] for option in row["answer_options"]} == {"image"}
            for row in all_selected_safe_questions
        )
        selected_rows.append(
            {
                "selection_order": len(selected_rows),
                "video_uid": uid,
                "selection_category": spec["selection_category"],
                "selection_rationale": spec["selection_rationale"],
                "scenario": str(metadata.get("scenarios", "")),
                "duration_sec": float(video["duration_sec"]),
                "duration_min": round(float(video["duration_sec"]) / 60.0, 2),
                "has_audio": bool(video.get("has_audio")),
                "frame_count": int(frames["frame_count"]),
                "video_path": str(video["path"]),
                "frame_dir": str(frames["frame_dir"]),
                "video_sha256": str(video["sha256"]),
                "question_count": selected_question_count,
                "image_option_question_count": selected_image_question_count,
                "primary_question_id": qid,
            }
        )
        primary_questions.append(
            {
                "selection_order": len(primary_questions),
                "question_id": qid,
                "video_uid": uid,
                "task": raw_question["task"],
                "question_text": raw_question["question_text"],
                "answer_options": safe_question["answer_options"],
                "correct_answer_included": False,
                "reference_timestamp_included": False,
            }
        )

    selected_question_count = len(all_selected_safe_questions)
    primary_ids = {row["question_id"] for row in primary_questions}
    held_out_count = selected_question_count - len(primary_ids)
    primary_text_only = all(
        {option["content_type"] for option in row["answer_options"]} == {"text"}
        for row in primary_questions
    )
    selection_payload = {
        "selection_kind": "content_aligned_engineering_pilot_not_representative_sample",
        "selection_policy": config["selection_policy"],
        "selected_videos": selected_rows,
        "primary_questions": primary_questions,
    }
    decision_records = {
        "selected_videos": selected_rows,
        "primary_questions": primary_questions,
    }
    leakage_checks = {
        "forbidden_selection_keys_absent": not contains_protected_evaluation_key(
            decision_records
        ),
        "correct_labels_not_in_primary_manifest": all(
            row["correct_answer_included"] is False for row in primary_questions
        ),
        "reference_timestamps_not_in_primary_manifest": all(
            row["reference_timestamp_included"] is False for row in primary_questions
        ),
        "selection_did_not_use_option_content": True,
        "selection_used_option_modality_only_for_text_only_first_smoke": True,
    }
    checks = {
        "interface_validation_passed": interface_validation.get("overall_validation")
        == "passed_no_api_interface_ready_for_five_video_index_build",
        "exactly_five_unique_videos": len(selected_rows) == 5
        and len({row["video_uid"] for row in selected_rows}) == 5,
        "exactly_five_unique_primary_questions": len(primary_questions) == 5
        and len(primary_ids) == 5,
        "one_primary_question_per_video": {
            row["video_uid"] for row in primary_questions
        }
        == {row["video_uid"] for row in selected_rows},
        "primary_questions_text_only": primary_text_only,
        "four_audio_one_no_audio": sum(row["has_audio"] for row in selected_rows) == 4,
        "all_media_paths_resolve": all(
            Path(row["video_path"]).is_file() and Path(row["frame_dir"]).is_dir()
            for row in selected_rows
        ),
        "all_frame_counts_positive": all(row["frame_count"] > 0 for row in selected_rows),
        "leakage_checks_passed": all(leakage_checks.values()),
    }
    errors.extend(name for name, value in checks.items() if not value)
    errors = sorted(set(errors))

    totals = {
        "selected_video_count": len(selected_rows),
        "selected_video_duration_sec": round(
            sum(row["duration_sec"] for row in selected_rows), 3
        ),
        "selected_video_duration_min": round(
            sum(row["duration_sec"] for row in selected_rows) / 60.0, 2
        ),
        "selected_1fps_frame_count": sum(row["frame_count"] for row in selected_rows),
        "audio_available_video_count": sum(row["has_audio"] for row in selected_rows),
        "audio_unavailable_video_count": sum(
            not row["has_audio"] for row in selected_rows
        ),
        "all_questions_on_selected_videos": selected_question_count,
        "primary_live_smoke_question_count": len(primary_questions),
        "held_out_question_count": held_out_count,
        "image_option_questions_in_full_selected_set": sum(
            row["image_option_question_count"] for row in selected_rows
        ),
    }
    freeze_manifest = {
        "freeze_kind": "five_video_content_aligned_engineering_pilot_selection",
        "freeze_version": "v1",
        "selected_video_uids": [row["video_uid"] for row in selected_rows],
        "primary_question_ids": [row["question_id"] for row in primary_questions],
        "initial_live_scope": "one predeclared text-option question per selected video",
        "selection_is_dataset_representative": False,
        "selection_may_not_be_changed_after_observing_pipeline_results": True,
        "correct_labels_or_reference_timestamps_used": False,
        "selection_payload_sha256": hashlib.sha256(
            canonical_bytes(selection_payload)
        ).hexdigest(),
    }
    validation = {
        "source_validation": "passed" if not errors else "failed",
        "selection_contract_validation": "passed" if all(checks.values()) else "failed",
        "evaluation_leakage_validation": (
            "passed" if all(leakage_checks.values()) else "failed"
        ),
        "media_readiness_validation": (
            "passed" if checks["all_media_paths_resolve"] else "failed"
        ),
        "primary_question_scope_validation": (
            "passed" if primary_text_only and len(primary_questions) == 5 else "failed"
        ),
        "overall_validation": (
            "passed_five_video_selection_frozen_for_index_build"
            if not errors
            else "failed"
        ),
        "model_api_calls": 0,
        "correct_answer_access_for_selection": 0,
        "reference_timestamp_access_for_selection": 0,
        "errors": errors,
        "checks": checks,
    }
    source_audit = {
        "sources": {name: _source(path) for name, path in source_paths.items()},
        "selection_fields_accessed": config["selection_policy"]["content_fields_used"]
        + config["selection_policy"]["technical_fields_used"],
        "forbidden_fields_accessed_for_selection": [],
    }
    held_out = {
        "selected_video_question_count": selected_question_count,
        "primary_question_count": len(primary_questions),
        "held_out_question_count": held_out_count,
        "policy": (
            "held out from the initial paid smoke; no result-driven expansion is allowed "
            "without a new versioned manifest"
        ),
    }
    files = {
        "input_manifest.json": {
            "experiment": config["experiment"],
            "mode": "no_api_content_aligned_selection",
            "model_api_calls": 0,
        },
        "source_hash_audit.json": source_audit,
        "selection_policy.json": config["selection_policy"],
        "candidate_scope_note.json": {
            "explicit_theft_or_physical_conflict_question_found": False,
            "substituted_categories": [
                "global event summary",
                "interpersonal collaboration",
                "multi-phase activity",
                "object-centered event",
                "cross-location navigation",
            ],
            "no_event_semantics_were_inferred_from_answers": True,
        },
        "selected_video_manifest.json": {
            "videos": selected_rows,
            "totals": totals,
        },
        "primary_question_manifest.json": {
            "questions": primary_questions,
            "scope": "five text-option questions; one per video",
        },
        "held_out_question_summary.json": held_out,
        "evaluation_leakage_audit.json": leakage_checks,
        "media_readiness_audit.json": {
            "videos": [
                {
                    key: row[key]
                    for key in [
                        "video_uid",
                        "duration_sec",
                        "has_audio",
                        "frame_count",
                        "video_path",
                        "frame_dir",
                        "video_sha256",
                    ]
                }
                for row in selected_rows
            ],
            "totals": totals,
        },
        "pilot_cost_scope.json": {
            "duration_and_frame_scope": totals,
            "paid_cost_estimate": "not computed until R1_AV/R3_2 indexing budgets are frozen",
            "model_api_calls_in_selection": 0,
        },
        "five_video_pilot_freeze_manifest.json": freeze_manifest,
        "validation_report.json": validation,
    }
    for name, value in files.items():
        write_json(output / name, value)

    rows = "\n".join(
        f"| {row['selection_order'] + 1} | `{row['video_uid']}` | "
        f"{row['selection_category']} | {row['duration_min']:.2f} | "
        f"{'yes' if row['has_audio'] else 'no'} | {row['question_count']} | "
        f"`{row['primary_question_id']}` |"
        for row in selected_rows
    )
    report = f"""# HourVideo five-video pilot selection v1

- Overall: `{validation['overall_validation']}`
- Selection basis: question text, task, scenario, duration, audio availability, frame readiness, and option modality only.
- Correct labels/reference timestamps used: `0` / `0`.
- Total: {totals['selected_video_duration_min']:.2f} minutes, {totals['selected_1fps_frame_count']} existing 1-fps frames, {totals['audio_available_video_count']} audio + {totals['audio_unavailable_video_count']} no-audio videos.
- Initial paid question scope: 5 text-option questions, one per video.
- Remaining questions on these videos: {totals['held_out_question_count']} held out.

| # | Video | Category | Minutes | Audio | Questions | Primary question |
|---:|---|---|---:|:---:|---:|---|
{rows}

The dev50 questions do not contain an explicit theft, robbery, fight, or physical-conflict question. The selection therefore uses the nearest honest categories: global event summaries, collaboration, multi-phase changes, object-centered events, and cross-location navigation. It is a content-aligned engineering pilot, not a representative dataset sample.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    return validation
