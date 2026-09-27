#!/usr/bin/env python3
"""Read-only structural verifier for the EgoPolice Layer-1 archive."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
QUESTIONS = {"q_weapon_visible", "q_visible_injury", "q_handcuff_before_medical"}
CONDITIONS = {"V", "AV_SPEECH"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    v_map = load(ART / "maps_v2_2/V/parsed_map.json")
    av_map = load(ART / "maps_v2_2/AV_SPEECH/parsed_map.json")
    checks.append(("V map has exactly 6 regions", len(v_map["coarse_regions"]) == 6, str(len(v_map["coarse_regions"]))))
    checks.append(("AV-Speech map has exactly 10 regions", len(av_map["coarse_regions"]) == 10, str(len(av_map["coarse_regions"]))))

    freeze = load(ART / "direct_run/questions_freeze.json")
    frozen = {q["question_id"]: q["question_text"] for q in freeze["questions"]}
    checks.append(("all three thesis questions are frozen", QUESTIONS <= set(frozen), ",".join(sorted(QUESTIONS))))

    route_results = sorted((ART / "primary_routes").glob("*/*/route_result.json"))
    checks.append(("exactly six primary route results", len(route_results) == 6, str(len(route_results))))
    observed_pairs = set()
    for path in route_results:
        row = load(path)
        observed_pairs.add((row["condition"], row["question_id"]))
        checks.append((f"question text frozen: {row['route_id']}", row["question_text"] == frozen[row["question_id"]], row["question_id"]))
    expected_pairs = {(c, q) for c in CONDITIONS for q in QUESTIONS}
    checks.append(("primary route identity is 2 conditions x 3 questions", observed_pairs == expected_pairs, repr(sorted(observed_pairs))))

    frame_manifest = load(ART / "source_data_handoff/route_inspected_frame_manifest.json")
    by_route = {r["route_id"]: r for r in frame_manifest["routes"]}
    for path in route_results:
        row = load(path)
        fm = by_route[row["route_id"]]
        route_times = [float(v) for v in row["resolved_timestamps_sec"]]
        manifest_times = [float(v) for v in fm["resolved_timestamps_sec"]]
        route_frame_shas = [f["observed_sha256"] for f in row["inspected_frames"]]
        manifest_frame_shas = [f["frame_sha256"] for f in fm["inspected_frames"]]
        checks.append((f"route timestamps match frame manifest: {row['route_id']}", route_times == manifest_times, repr(route_times)))
        checks.append((f"route frame SHA sequence matches manifest: {row['route_id']}", route_frame_shas == manifest_frame_shas, str(len(route_frame_shas))))
        checks.append((f"frame manifest binds route_result SHA: {row['route_id']}", sha(path) == fm["source_route_result_sha256"], sha(path)))

    source_rows = {}
    with (ROOT / "SOURCE_MANIFEST.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            source_rows[row["staging_path"]] = row
            staged = ROOT / row["staging_path"]
            checks.append((f"manifested file exists: {row['staging_path']}", staged.is_file(), row["staging_path"]))
            if staged.is_file():
                checks.append((f"staged SHA matches SOURCE_MANIFEST: {row['staging_path']}", sha(staged) == row["staged_sha256"], row["staged_sha256"]))

    artifact_files = {p.relative_to(ROOT).as_posix() for p in ART.rglob("*") if p.is_file()}
    checks.append(("every artifact is in SOURCE_MANIFEST", artifact_files == set(source_rows), f"artifacts={len(artifact_files)} manifest={len(source_rows)}"))
    exact_targets = [
        "artifacts/maps_v2_2/V/parsed_map.json",
        "artifacts/maps_v2_2/AV_SPEECH/parsed_map.json",
        *[p.relative_to(ROOT).as_posix() for p in route_results],
    ]
    for rel in exact_targets:
        row = source_rows[rel]
        checks.append((f"formal map/route is byte-identical to source: {rel}", row["copy_mode"] == "BYTE_IDENTICAL" and row["original_sha256"] == row["staged_sha256"], row["staged_sha256"]))

    v_primary = load(ART / "primary_routes/V/q_handcuff_before_medical/route_result.json")
    recovery = load(ART / "diagnostic_format_only_recovery/V/q_handcuff_before_medical/recovery_result.json")
    recovery_validation = load(ART / "diagnostic_format_only_recovery/V/q_handcuff_before_medical/validation.json")
    checks.append(("V temporal primary remains format failure", v_primary["completed"] is False and v_primary["terminal_status"] == "invalid_action:invalid_reason" and v_primary["final_answer"] is None, v_primary["terminal_status"]))
    checks.append(("recovery remains diagnostic and does not replace primary", recovery["diagnostic"] == "FORMAT_ONLY_RECOVERY_DIAGNOSTIC" and recovery["primary_completion_unchanged"] is True and recovery["new_inspection_calls"] == 0 and recovery["new_frames"] == 0, recovery["diagnostic"]))
    checks.append(("format-only recovery validation passes", recovery_validation["status"] == "PASS" and recovery_validation["new_inspection_calls"] == 0 and recovery_validation["new_frames"] == 0, recovery_validation["status"]))

    required_handoff = {
        "route_inspected_frame_manifest.json", "inspected_frames_all.csv", "per_frame_visual_audit.csv",
        "per_question_quality.json", "case_ranking.json", "figure_candidate_plan.md",
        "FINAL_CASE_STUDY_AUDIT.md", "README_REVIEW.md", "UPSTREAM_REVIEW_BUNDLE_MANIFEST.sha256",
    }
    handoff_names = {p.name for p in (ART / "source_data_handoff").iterdir() if p.is_file()}
    checks.append(("required source-data handoff is complete", required_handoff <= handoff_names, repr(sorted(required_handoff - handoff_names))))

    forbidden = [p for p in ART.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".npz", ".npy", ".pt", ".pth", ".safetensors"}]
    checks.append(("no JPEG/video/embedding/model payload", not forbidden, repr([p.name for p in forbidden])))

    failures = [item for item in checks if not item[1]]
    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "check_count": len(checks),
        "pass_count": len(checks) - len(failures),
        "failure_count": len(failures),
        "failures": [{"check": n, "detail": d} for n, _, d in failures],
    }, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
