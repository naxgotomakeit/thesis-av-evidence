from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _questions(annotation: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for uid, record in annotation.items():
        for row in record.get("benchmark_dataset", []):
            result[str(row["qid"])] = {
                "question_id": str(row["qid"]),
                "video_uid": uid,
                "question_text": str(row["question"]).strip(),
                "task": str(row["task"]),
            }
    return result


def run(repo: Path, config_path: Path) -> dict[str, Any]:
    cfg = load(config_path)
    if cfg.get("model_api_calls") != 0:
        raise ValueError("selection must be no-API")
    root = Path(cfg["hourvideo_root"])
    source_paths = {
        "annotations": root / cfg["annotation_path"],
        "frame_audit": root / cfg["frame_audit_path"],
        "video_audit": root / cfg["video_audit_path"],
        "safe_questions": repo / cfg["safe_question_manifest"],
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing selection sources: {missing}")
    annotation = load(source_paths["annotations"])
    frames = {row["video_uid"]: row for row in load(source_paths["frame_audit"])["videos"]}
    videos = {row["video_uid"]: row for row in load(source_paths["video_audit"])["videos"]}
    safe = {row["question_id"]: row for row in load(source_paths["safe_questions"])["questions"]}
    questions = _questions(annotation)
    excluded = set(cfg["excluded_development_video_uids"])
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    qrows: list[dict[str, Any]] = []
    for order, spec in enumerate(cfg["selected_cases"]):
        uid, qid = spec["video_uid"], spec["question_id"]
        if uid in excluded:
            errors.append(f"development video leaked into pilot: {uid}")
            continue
        if uid not in videos or uid not in frames or qid not in questions or qid not in safe:
            errors.append(f"unresolved case: {uid}/{qid}")
            continue
        video, frame, question, safe_q = videos[uid], frames[uid], questions[qid], safe[qid]
        if question["video_uid"] != uid or safe_q["video_uid"] != uid:
            errors.append(f"question/video mismatch: {qid}")
        kinds = {option["content_type"] for option in safe_q["answer_options"]}
        if kinds != {"text"}:
            errors.append(f"non-text options: {qid}")
        duration_min = float(video["duration_sec"]) / 60.0
        low = float(cfg["selection_policy"]["eligible_duration_min"])
        high = float(cfg["selection_policy"]["eligible_duration_max"])
        if not low <= duration_min <= high:
            errors.append(f"duration outside predeclared band: {uid}")
        if not Path(video["path"]).is_file() or not Path(frame["frame_dir"]).is_dir():
            errors.append(f"media missing: {uid}")
        rows.append({
            "selection_order": order,
            "video_uid": uid,
            "question_id": qid,
            "category": spec["category"],
            "duration_sec": float(video["duration_sec"]),
            "duration_min": round(duration_min, 3),
            "has_audio": bool(video.get("has_audio")),
            "frame_count": int(frame["frame_count"]),
            "video_path": str(video["path"]),
            "frame_dir": str(frame["frame_dir"]),
            "video_sha256": str(video["sha256"]),
        })
        qrows.append({
            "selection_order": order,
            "video_uid": uid,
            "question_id": qid,
            "task": question["task"],
            "question_text": question["question_text"],
            "answer_options": safe_q["answer_options"],
            "gold_included": False,
            "reference_timestamps_included": False,
        })
    if len(rows) != 10 or len({row["video_uid"] for row in rows}) != 10:
        errors.append("exactly ten unique videos required")
    if len(qrows) != 10 or len({row["question_id"] for row in qrows}) != 10:
        errors.append("exactly ten unique questions required")
    if len({row["category"] for row in rows}) != 10:
        errors.append("selection categories must be unique")
    selection = {
        "selection_kind": cfg["selection_policy"]["kind"],
        "selection_policy": cfg["selection_policy"],
        "excluded_development_video_uids": sorted(excluded),
        "videos": rows,
        "questions": qrows,
    }
    output = repo / cfg["output_root"]
    dump(output / "input_manifest.json", {"experiment": cfg["experiment"], "model_api_calls": 0})
    dump(output / "source_hash_audit.json", {name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for name, path in source_paths.items()})
    dump(output / "selected_case_manifest.json", selection)
    totals = {
        "video_count": len(rows),
        "question_count": len(qrows),
        "duration_min": round(sum(row["duration_sec"] for row in rows) / 60.0, 3),
        "frame_count": sum(row["frame_count"] for row in rows),
        "audio_video_count": sum(row["has_audio"] for row in rows),
        "no_audio_video_count": sum(not row["has_audio"] for row in rows),
    }
    freeze = {
        "freeze_kind": "hourvideo_ten_video_ten_question_engineering_pilot_selection",
        "selection_sha256": hashlib.sha256(canonical(selection)).hexdigest(),
        "video_uids": [row["video_uid"] for row in rows],
        "question_ids": [row["question_id"] for row in qrows],
        "gold_or_reference_timestamps_used": False,
        "change_after_observing_results_allowed": False,
    }
    validation = {
        "source_validation": "passed" if not missing else "failed",
        "selection_validation": "passed" if not errors else "failed",
        "evaluation_leakage_validation": "passed",
        "media_readiness_validation": "passed" if not any("media missing" in e for e in errors) else "failed",
        "overall_validation": "passed_ten_video_selection_frozen_for_index_build" if not errors else "failed",
        "model_api_calls": 0,
        "errors": sorted(set(errors)),
        "totals": totals,
    }
    dump(output / "pilot_selection_freeze_manifest.json", freeze)
    dump(output / "validation_report.json", validation)
    table = "\n".join(
        f"| {row['selection_order'] + 1} | `{row['video_uid']}` | {row['category']} | {row['duration_min']:.2f} | {'yes' if row['has_audio'] else 'no'} | `{row['question_id']}` |"
        for row in rows
    )
    report = f"""# HourVideo ten-video pilot selection v1

- Status: `{validation['overall_validation']}`
- Scope: 10 videos / 10 text-option questions.
- Total duration: {totals['duration_min']:.2f} minutes; existing 1-fps frames: {totals['frame_count']}.
- Audio: {totals['audio_video_count']} videos; no-audio: {totals['no_audio_video_count']} videos.
- Gold labels, reference timestamps, and previous model results used for selection: `0`.
- The completed development smoke video is excluded.

| # | Video | Category | Minutes | Audio | Question |
|---:|---|---|---:|:---:|---|
{table}

This is a cost-bounded, content-stratified engineering pilot, not a representative HourVideo benchmark sample.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    return validation
