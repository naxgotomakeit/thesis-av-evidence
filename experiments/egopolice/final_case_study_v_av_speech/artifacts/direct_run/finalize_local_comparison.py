#!/usr/bin/env python3
"""No-model/no-gold local comparison and final freeze for the 226 routes."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone


OUT = Path(__file__).resolve().parent
QUESTION_ORDER = [
    "q_global_summary", "q_weapon_visible", "q_visible_injury",
    "q_medical_assistance", "q_handcuffing", "q_handcuff_before_medical",
]
ANNOTATIONS = [
    {"stage_id": "manual_exit_car", "start_sec": 192.0, "end_sec": 201.0, "description": "camera wearer exits car"},
    {"stage_id": "manual_gun_shot", "start_sec": 399.0, "end_sec": 400.0, "description": "gun shot"},
    {"stage_id": "manual_ground_blood_arrest", "start_sec": 595.0, "end_sec": 642.0, "description": "person on ground / blood / arrest context"},
    {"stage_id": "manual_checking_assistance", "start_sec": 643.0, "end_sec": 695.0, "description": "checking / assistance"},
    {"stage_id": "manual_later_restraint_medical", "start_sec": 696.0, "end_sec": 1235.0, "description": "later sustained restraint / medical response"},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(value)
    os.replace(temp, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def nearest_summary(left: list[float], right: list[float]) -> dict[str, object]:
    if not left or not right:
        return {"available": False, "V_to_AV": [], "AV_to_V": []}
    l2r = [min(abs(x - y) for y in right) for x in left]
    r2l = [min(abs(y - x) for x in left) for y in right]
    return {
        "available": True,
        "V_to_AV": l2r, "AV_to_V": r2l,
        "symmetric_mean_sec": statistics.mean(l2r + r2l),
        "symmetric_median_sec": statistics.median(l2r + r2l),
        "symmetric_max_sec": max(l2r + r2l),
    }


def navigation_category(v: dict[str, object], av: dict[str, object], jaccard: float) -> str:
    vt, at = v["resolved_timestamps_sec"], av["resolved_timestamps_sec"]
    if vt == at:
        return "SAME_NAVIGATION"
    first_close = bool(vt and at and abs(vt[0] - at[0]) <= 45.0)
    count_close = abs(len(vt) - len(at)) <= 3
    if jaccard >= 0.5 and first_close and count_close:
        return "MINOR_NAVIGATION_CHANGE"
    return "MAJOR_NAVIGATION_CHANGE"


# Local semantic comparison only; no gold/correctness was loaded.
ANSWER_CATEGORIES = {
    "q_global_summary": "NOT_COMPARABLE_BOTH_ROUTE_FAILURE",
    "q_weapon_visible": "DIFFERENT_WORDING_SAME_SUBSTANCE",
    "q_visible_injury": "SUBSTANTIVE_ANSWER_CHANGE",
    "q_medical_assistance": "NOT_COMPARABLE_ROUTE_FAILURE",
    "q_handcuffing": "DIFFERENT_WORDING_SAME_SUBSTANCE",
    "q_handcuff_before_medical": "NOT_COMPARABLE_ROUTE_FAILURE",
}


LIMITATION_FLAGS = {
    "q_global_summary": {
        "conditions": ["AV_SPEECH"],
        "basis": "Rejected AV final-answer attempts explicitly relied on tourniquet/medical-response map summaries; this is one of the frozen v2.2 qualitative limitations.",
    },
    "q_visible_injury": {
        "conditions": ["AV_SPEECH"],
        "basis": "Accepted AV answer/reason explicitly invoked the speech-derived 'shots to chest and groin' map context while answering visible injury.",
    },
    "q_medical_assistance": {
        "conditions": ["AV_SPEECH"],
        "basis": "Rejected AV final-answer attempts explicitly described a tourniquet/pressure band and used the tourniquet-bearing AV map region.",
    },
}


def main() -> None:
    complete = json.loads((OUT / "execution_complete.json").read_text())
    if complete.get("route_count") != 12 or complete.get("status") != "ROUTES_FROZEN":
        raise RuntimeError("post-hoc comparison requires all 12 route attempts frozen")
    results = json.loads((OUT / "route_results.json").read_text())
    by = {(row["question_id"], row["condition"]): row for row in results}
    comparisons = []
    navigation_rows = []
    for qid in QUESTION_ORDER:
        v, av = by[(qid, "V")], by[(qid, "AV_SPEECH")]
        vset = {frame["frame_path"] for frame in v["inspected_frames"]}
        aset = {frame["frame_path"] for frame in av["inspected_frames"]}
        union = vset | aset
        intersection = vset & aset
        jaccard = 1.0 if not union else len(intersection) / len(union)
        category = navigation_category(v, av, jaccard)
        row = {
            "question_id": qid, "question_text": v["question_text"],
            "V": {"completed": v["completed"], "terminal_status": v["terminal_status"],
                  "inspection_turn_count": v["inspection_turn_count"], "requested_timestamps_sec": v["requested_timestamps_sec"],
                  "resolved_timestamps_sec": v["resolved_timestamps_sec"], "unique_frames": v["unique_frame_count"],
                  "answer": v["final_answer"], "reason": v["final_reason"]},
            "AV_SPEECH": {"completed": av["completed"], "terminal_status": av["terminal_status"],
                  "inspection_turn_count": av["inspection_turn_count"], "requested_timestamps_sec": av["requested_timestamps_sec"],
                  "resolved_timestamps_sec": av["resolved_timestamps_sec"], "unique_frames": av["unique_frame_count"],
                  "answer": av["final_answer"], "reason": av["final_reason"]},
            "evidence_overlap": {"common_frame_count": len(intersection), "V_only_frame_count": len(vset - aset),
                                 "AV_only_frame_count": len(aset - vset), "frame_jaccard": jaccard,
                                 "common_frames": sorted(intersection), "V_only_frames": sorted(vset - aset), "AV_only_frames": sorted(aset - vset),
                                 "timestamp_distance": nearest_summary(v["resolved_timestamps_sec"], av["resolved_timestamps_sec"])},
            "navigation_change_category": category,
            "answer_change_category": ANSWER_CATEGORIES[qid],
            "known_map_limitation": ({"flag": "POTENTIALLY_INFLUENCED_BY_KNOWN_MAP_LIMITATION", **LIMITATION_FLAGS[qid]}
                                      if qid in LIMITATION_FLAGS else {"flag": None}),
            "correctness_assessed": False,
        }
        comparisons.append(row)
        navigation_rows.append({"question_id": qid, "category": category,
            "V_first_resolved_sec": v["resolved_timestamps_sec"][0] if v["resolved_timestamps_sec"] else None,
            "AV_first_resolved_sec": av["resolved_timestamps_sec"][0] if av["resolved_timestamps_sec"] else None,
            "V_path": v["resolved_timestamps_sec"], "AV_path": av["resolved_timestamps_sec"],
            "frame_jaccard": jaccard})
    atomic_json(OUT / "per_question_v_av_comparison.json", {
        "created_at_utc": now(), "comparison_type": "descriptive_no_gold", "questions": comparisons,
        "answer_category_note": "NOT_COMPARABLE_* is used when frozen route failure leaves no valid matched answer; no answer was imputed.",
    })
    atomic_json(OUT / "navigation_comparison.json", {
        "created_at_utc": now(),
        "category_rule": {
            "SAME_NAVIGATION": "ordered resolved timestamp sequences exactly equal",
            "MINOR_NAVIGATION_CHANGE": "not exact, frame Jaccard >= 0.5, first resolved timestamps within 45s, and frame-count difference <= 3",
            "MAJOR_NAVIGATION_CHANGE": "all other cases",
        },
        "rows": navigation_rows,
    })

    annotation_routes = []
    for result in results:
        timestamps = [float(x) for x in result["resolved_timestamps_sec"]]
        stages = []
        for stage in ANNOTATIONS:
            hits = [x for x in timestamps if stage["start_sec"] <= x <= stage["end_sec"]]
            distance = None if hits or not timestamps else min(
                0.0 if stage["start_sec"] <= x <= stage["end_sec"] else
                min(abs(x - stage["start_sec"]), abs(x - stage["end_sec"])) for x in timestamps)
            stages.append({**stage, "hit_timestamps_sec": hits, "hit": bool(hits), "nearest_distance_sec_if_missed": distance})
        annotation_routes.append({"route_id": result["route_id"], "question_id": result["question_id"],
                                  "condition": result["condition"], "completed": result["completed"],
                                  "resolved_timestamps_sec": timestamps, "stages": stages})
    atomic_json(OUT / "annotation_posthoc_comparison.json", {
        "created_at_utc": now(), "status": "POST_HOC_ONLY_AFTER_12_ROUTE_ATTEMPTS_FROZEN",
        "annotation_source": "user-supplied qualitative reference in execution instruction",
        "provided_to_model": False, "used_for_navigation_or_stopping": False,
        "correctness_gold": False, "interval_boundary_rule": "inclusive for this descriptive post-hoc comparison",
        "annotations": ANNOTATIONS, "routes": annotation_routes,
    })

    lines = [
        "# 226 Direct-v1.2 open-ended V/AV-Speech six-question summary", "",
        "Status: `ROUTES_FROZEN`; completed routes: `8/12`.", "",
        "All 12 route attempts were executed in the frozen order. Four routes terminated after the one allowed structural correction because the provider returned a `reason` longer than the frozen 240-character tool contract. They were not repaired or replayed.", "",
        "Raw ASR was not supplied to the Direct answerer. The same semantic-only projection set `exact_source_asr=[]` in both maps; all Organizer-generated structure, summaries, uncertainty notes, and exact visual captions were retained.", "",
        "| Question | V frames | AV frames | Jaccard | Navigation | Answer comparison |", "|---|---:|---:|---:|---|---|",
    ]
    for row in comparisons:
        lines.append(f"| {row['question_id']} | {row['V']['unique_frames']} | {row['AV_SPEECH']['unique_frames']} | {row['evidence_overlap']['frame_jaccard']:.3f} | {row['navigation_change_category']} | {row['answer_change_category']} |")
    lines += ["", "No gold/correctness was read or evaluated. Human stage annotations were used only after all 12 route attempts were frozen.", ""]
    atomic_bytes(OUT / "DIRECT_OPEN_ENDED_SIXQ_SUMMARY.md", "\n".join(lines).encode())

    # Build a complete non-self-referential inventory, then hash the inventory.
    excluded = {"manifest.json", "sha256.txt"}
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.relative_to(OUT).as_posix() not in excluded:
            files.append({"path": path.relative_to(OUT).as_posix(), "sha256": sha(path), "size_bytes": path.stat().st_size})
    manifest = {
        "schema_version": "direct_open_ended_226_sixq_final_manifest_v1",
        "created_at_utc": now(), "status": "PASS" if all(row["max_unique_frames_pass"] for row in results) else "FAIL",
        "route_attempts_frozen": len(results), "completed_routes": sum(bool(row["completed"]) for row in results),
        "max_unique_frames_all_routes_pass": all(row["max_unique_frames_pass"] for row in results),
        "gold_or_correctness_read": False, "raw_asr_direct_answerer_access": False,
        "files": files,
    }
    atomic_json(OUT / "manifest.json", manifest)
    final_files = files + [{"path": "manifest.json", "sha256": sha(OUT / "manifest.json"), "size_bytes": (OUT / "manifest.json").stat().st_size}]
    atomic_bytes(OUT / "sha256.txt", ("\n".join(f"{row['sha256']}  {row['path']}" for row in final_files) + "\n").encode())
    for line in (OUT / "sha256.txt").read_text().splitlines():
        expected, rel = line.split(None, 1)
        if sha(OUT / rel.strip()) != expected:
            raise RuntimeError(f"final SHA verification failed: {rel}")
    print(json.dumps({"status": manifest["status"], "route_attempts": len(results),
                      "completed": manifest["completed_routes"], "files": len(final_files),
                      "sha_validation": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
