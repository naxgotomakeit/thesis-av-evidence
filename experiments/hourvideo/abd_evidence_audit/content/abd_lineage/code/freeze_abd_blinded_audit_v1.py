#!/usr/bin/env python3
"""Validate and freeze the completed blinded ABD evidence audit.

This phase deliberately never opens identity_mapping.json or any gold/scored file.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/abd_eval300_evidence_audit_v1"
ANALYSIS = ROOT / "outputs/abd_eval300_analysis_v1"
VALID = {"supported", "partially_supported", "unsupported", "contradicted", "unreviewable"}
REQUIRED = {
    "audit_id", "option_support", "rationale", "evidence_refs",
    "map_region_time_range", "frame_timestamps", "missing_key_evidence",
    "confidence", "review_flag", "evidence_relationship",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_json(path: Path, obj: object) -> None:
    atomic_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def main() -> None:
    batch_doc = json.loads((AUDIT / "batch_manifest.json").read_text())
    batches = batch_doc["batches"]
    registry = json.loads((AUDIT / "valid_second_pass_batches.json").read_text())
    package_freeze = json.loads((AUDIT / "package_freeze_manifest.json").read_text())
    raw_integrity = json.loads((AUDIT / "raw_integrity.json").read_text())

    expected_batches = [f"B{i:03d}" for i in range(1, 193)]
    if batch_doc.get("batch_count") != 192 or [b["batch_id"] for b in batches] != expected_batches:
        raise SystemExit("batch manifest is not the frozen 192-batch sequence")
    if registry.get("gold_loaded") is not False or set(registry.get("batches", {})) != set(expected_batches):
        raise SystemExit("valid registry is incomplete or crossed the gold boundary")
    if package_freeze.get("gold_loaded") is not False or package_freeze.get("historical_audit_labels_loaded") is not False:
        raise SystemExit("package freeze is not blinded")
    if raw_integrity.get("status") != "PASS" or raw_integrity.get("gold_loaded") is not False:
        raise SystemExit("raw ABD integrity precondition failed")

    anchor_checks = {
        "scientific_manifest_sha256": package_freeze["scientific_manifest_sha256"],
        "raw_closure_file_sha256": sha256(ANALYSIS / "structural_closure_report.json"),
        "archive_file_sha256": sha256(ANALYSIS / "ABD_RESULTS_PACKAGE.tar.gz"),
    }
    for key, actual in anchor_checks.items():
        expected = package_freeze[key]
        if actual != expected:
            raise SystemExit(f"raw anchor drift: {key}: {actual} != {expected}")

    all_rows = []
    seen = set()
    progress = []
    for batch in batches:
        bid = batch["batch_id"]
        review = AUDIT / "reviews" / f"{bid}.jsonl"
        reg = registry["batches"][bid]
        if reg.get("status") != "valid_actual_evidence_review" or sha256(review) != reg["review_sha256"]:
            raise SystemExit(f"invalid registered review SHA: {bid}")
        rows = [json.loads(line) for line in review.read_text().splitlines() if line]
        ids = batch["audit_ids"]
        if len(rows) != len(ids) or [r.get("audit_id") for r in rows] != ids:
            raise SystemExit(f"review/manifest ID mismatch: {bid}")
        for row in rows:
            if set(row) != REQUIRED or row["option_support"] not in VALID:
                raise SystemExit(f"schema/label failure: {row.get('audit_id')}")
            if row["audit_id"] in seen:
                raise SystemExit(f"duplicate audit ID: {row['audit_id']}")
            if not isinstance(row["rationale"], str) or not row["rationale"].strip():
                raise SystemExit(f"empty rationale: {row['audit_id']}")
            if not isinstance(row["evidence_refs"], list):
                raise SystemExit(f"invalid evidence_refs: {row['audit_id']}")
            seen.add(row["audit_id"])
            all_rows.append(row)

        map_count = 0
        image_count = 0
        for audit_id in ids:
            package = json.loads((AUDIT / "review_packages" / f"{audit_id}.json").read_text())
            map_count += int(bool(package.get("map")))
            image_count += len(package.get("images") or [])
        if image_count != batch["image_count"]:
            raise SystemExit(f"image-count drift: {bid}")
        progress.append({
            "batch_id": bid,
            "valid_entries": len(rows),
            "maps_actually_reviewed_attested": map_count,
            "images_actually_viewed_attested": image_count,
            "review_sha256": sha256(review),
            "status": "valid_actual_evidence_review",
        })

    if len(all_rows) != 900 or len(seen) != 900:
        raise SystemExit(f"expected 900 unique reviews, got {len(all_rows)}/{len(seen)}")

    blinded = AUDIT / "scored_evidence_audit_blinded.jsonl"
    raw = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in all_rows).encode()
    atomic_bytes(blinded, raw)
    progress_doc = {
        "schema_version": "abd_evidence_audit_progress_v1",
        "gold_loaded": False,
        "batch_count": 192,
        "audit_count": 900,
        "batches": progress,
    }
    atomic_json(AUDIT / "review_progress.json", progress_doc)

    freeze = {
        "schema_version": "abd_blinded_evidence_audit_freeze_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_loaded_at_freeze": False,
        "historical_labels_loaded": False,
        "batch_count": 192,
        "audit_count": 900,
        "unique_audit_ids": 900,
        "valid_registry_sha256": sha256(AUDIT / "valid_second_pass_batches.json"),
        "batch_manifest_sha256": sha256(AUDIT / "batch_manifest.json"),
        "package_freeze_manifest_sha256": sha256(AUDIT / "package_freeze_manifest.json"),
        "raw_integrity_sha256": sha256(AUDIT / "raw_integrity.json"),
        "raw_anchor_checks": anchor_checks,
        "scored_evidence_audit_blinded_sha256": sha256(blinded),
        "review_progress_sha256": sha256(AUDIT / "review_progress.json"),
        "status": "FROZEN_AND_VERIFIED",
    }
    atomic_json(AUDIT / "audit_freeze.json", freeze)
    # Read-back verification of the closure just written.
    check = json.loads((AUDIT / "audit_freeze.json").read_text())
    if check != freeze or sha256(blinded) != freeze["scored_evidence_audit_blinded_sha256"]:
        raise SystemExit("audit freeze read-back failed")
    print(json.dumps({"status": "PASS", "batches": 192, "audits": 900, "blinded_sha256": sha256(blinded), "audit_freeze_sha256": sha256(AUDIT / "audit_freeze.json")}))


if __name__ == "__main__":
    main()
