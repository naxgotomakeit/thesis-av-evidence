#!/usr/bin/env python3
"""Validate and freeze the blinded 31-record Sol review before any label comparison."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/abd_eval300_disagreement_independent_review_v1"
PUBLIC = OUT / "reviewer_materials"
REVIEW = OUT / "blinded_reviewer_output"
PREP = OUT / "PREPARATION_FREEZE.json"
FREEZE = OUT / "BLINDED_REVIEW_FREEZE.json"
LABELS = {"supported", "partially_supported", "unsupported", "contradicted", "unreviewable"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    if FREEZE.exists():
        raise RuntimeError(f"refuse overwrite: {FREEZE}")
    prep = load(PREP)
    public_manifest = load(PUBLIC / "package_manifest.json")
    batches = load(PUBLIC / "batch_manifest.json")["batches"]
    expected = {x["review_id"]: x for x in public_manifest["items"]}
    issues = []
    for name, meta in prep["public_files"].items():
        p = PUBLIC / name
        if not p.is_file() or sha(p) != meta["sha256"] or p.stat().st_size != meta["size_bytes"]:
            issues.append(f"public_file_changed:{name}")

    merged = []
    batch_hashes = {}
    viewing_hashes = {}
    for batch in batches:
        bid = batch["batch_id"]
        rp = REVIEW / f"batch_{bid}.jsonl"
        vp = REVIEW / f"batch_{bid}_viewing.json"
        if not rp.is_file() or not vp.is_file():
            issues.append(f"missing_batch_output:{bid}")
            continue
        rows = jsonl(rp)
        view = load(vp)
        view_rows = view.get("items", view if isinstance(view, list) else [])
        expected_ids = batch["review_ids"]
        got_ids = [x.get("review_id") for x in rows]
        if got_ids != expected_ids or len(got_ids) != len(set(got_ids)):
            issues.append(f"batch_ids_or_order:{bid}")
        view_by_id = {x.get("review_id"): x for x in view_rows}
        if set(view_by_id) != set(expected_ids):
            issues.append(f"viewing_ids:{bid}")
        for row in rows:
            rid = row.get("review_id")
            if row.get("option_support") not in LABELS:
                issues.append(f"invalid_label:{rid}")
            if not isinstance(row.get("evidence_references"), list) or not row["evidence_references"]:
                issues.append(f"evidence_references:{rid}")
            for field in ("concise_rationale", "missing_key_conditions"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    issues.append(f"missing_{field}:{rid}")
            if not isinstance(row.get("review_flag"), bool):
                issues.append(f"review_flag_type:{rid}")
            if row.get("review_flag") and not str(row.get("review_flag_reason", "")).strip():
                issues.append(f"review_flag_reason:{rid}")
            item = expected.get(rid)
            if item is None:
                continue
            expected_map = bool(item["map_count"])
            expected_images = int(item["image_count"])
            if bool(row.get("map_read")) != expected_map:
                issues.append(f"map_read:{rid}")
            if row.get("images_opened") != expected_images:
                issues.append(f"images_opened:{rid}")
            vr = view_by_id.get(rid, {})
            if bool(vr.get("map_read")) != expected_map or vr.get("images_opened") != expected_images or vr.get("expected_images") != expected_images:
                issues.append(f"viewing_ledger_counts:{rid}")
        merged.extend(rows)
        batch_hashes[bid] = sha(rp)
        viewing_hashes[bid] = sha(vp)

    all_ids = [x.get("review_id") for x in merged]
    expected_order = [x["review_id"] for x in public_manifest["items"]]
    if all_ids != expected_order or len(all_ids) != 31 or len(set(all_ids)) != 31:
        issues.append("merged_identity_or_order")
    merged_path = REVIEW / "blinded_reviews.jsonl"
    if not merged_path.is_file() or jsonl(merged_path) != merged:
        issues.append("reviewer_merged_file_mismatch")
    progress = REVIEW / "progress.json"
    report = REVIEW / "REPORT.md"
    if not progress.is_file() or not report.is_file():
        issues.append("missing_progress_or_report")
    if issues:
        raise RuntimeError("blind review validation failed: " + ";".join(issues))

    atomic_json(FREEZE, {
        "schema_version": "abd_disagreement_independent_blinded_review_freeze_v1",
        "frozen": True,
        "actual_reviewer_model": "gpt-5.6-sol",
        "review_type": "Codex-assisted semantic review",
        "record_count": len(merged),
        "batch_count": len(batches),
        "maps_read": sum(x["map_count"] for x in public_manifest["items"]),
        "images_opened": sum(x["image_count"] for x in public_manifest["items"]),
        "preparation_freeze_sha256": sha(PREP),
        "criteria_sha256": sha(PUBLIC / "AUDIT_CRITERIA_FROZEN.md"),
        "package_manifest_sha256": sha(PUBLIC / "package_manifest.json"),
        "batch_file_sha256": batch_hashes,
        "viewing_file_sha256": viewing_hashes,
        "blinded_reviews_sha256": sha(merged_path),
        "progress_sha256": sha(progress),
        "report_sha256": sha(report),
        "identity_mapping_accessed_by_reviewer": False,
        "prior_labels_accessed_by_reviewer": False,
        "gold_read": False,
        "correctness_read": False,
        "external_api_calls": 0,
    })
    print(json.dumps({"frozen": True, "records": 31, "batches": len(batches), "maps_read": sum(x["map_count"] for x in public_manifest["items"]), "images_opened": sum(x["image_count"] for x in public_manifest["items"]), "blinded_reviews_sha256": sha(merged_path)}, indent=2))


if __name__ == "__main__":
    main()
