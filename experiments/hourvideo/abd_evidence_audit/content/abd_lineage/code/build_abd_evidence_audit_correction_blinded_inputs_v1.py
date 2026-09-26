#!/usr/bin/env python3
"""Build a portable, blinded correction-input package without changing the audit freeze."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/abd_eval300_evidence_audit_v1"
OUT = ROOT / "outputs/abd_eval300_evidence_audit_correction_inputs_v1"
PACKAGE_NAME = "ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS"
PACKAGE = OUT / PACKAGE_NAME
ARCHIVE = OUT / f"{PACKAGE_NAME}.tar.gz"
CONTROL_SEED = "abd-evidence-audit-correction-controls-v1"
ORDER_SEED = "abd-evidence-audit-correction-order-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(b"\0")
        h.update(sha256(path).encode())
        h.update(b"\n")
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing correction package: {OUT}")

    protected = [
        AUDIT / "audit_freeze.json",
        AUDIT / "scored_evidence_audit_blinded.jsonl",
        AUDIT / "valid_second_pass_batches.json",
        AUDIT / "batch_manifest.json",
        *sorted((AUDIT / "reviews").glob("B*.jsonl")),
    ]
    before = tree_digest(protected)

    # Selection uses only the frozen blinded review labels and opaque batch membership.
    # It never opens identity_mapping.json, any gold source, or any scored/statistical output.
    blinded_rows = [json.loads(line) for line in (AUDIT / "scored_evidence_audit_blinded.jsonl").read_text().splitlines() if line]
    status = {row["audit_id"]: row["option_support"] for row in blinded_rows}
    batch_doc = json.loads((AUDIT / "batch_manifest.json").read_text())
    batch_of = {audit_id: int(batch["batch_id"][1:]) for batch in batch_doc["batches"] for audit_id in batch["audit_ids"]}
    targets = sorted(aid for aid, label in status.items() if label == "unreviewable")
    if len(targets) != 116:
        raise SystemExit(f"expected 116 target inputs, got {len(targets)}")
    eligible = [aid for aid, label in status.items() if label != "unreviewable"]
    rng = random.Random(CONTROL_SEED)
    controls = []
    control_counts = []
    for workflow, (lo, hi) in enumerate(((1, 64), (65, 128), (129, 192)), start=1):
        pool = sorted(aid for aid in eligible if lo <= batch_of[aid] <= hi)
        picked = rng.sample(pool, 20)
        controls.extend(picked)
        control_counts.append({"workflow": workflow, "batch_range": f"B{lo:03d}-B{hi:03d}", "selected_count": 20, "eligible_count": len(pool)})
    selected = targets + controls
    if len(selected) != 176 or len(set(selected)) != 176:
        raise SystemExit("selection is not 176 unique opaque IDs")
    random.Random(ORDER_SEED).shuffle(selected)

    PACKAGE.mkdir(parents=True)
    items_dir = PACKAGE / "items"
    items_dir.mkdir()
    inventory = []
    total_images = 0
    total_maps = 0

    allowed_keys = {"audit_id", "question", "options", "predicted_option", "original_reason", "map", "map_raw_file", "images", "tool_feedback"}
    for position, audit_id in enumerate(selected, start=1):
        source = json.loads((AUDIT / "review_packages" / f"{audit_id}.json").read_text())
        item_dir = items_dir / audit_id
        item_dir.mkdir()
        images_dir = item_dir / "images"
        images_dir.mkdir()

        image_rows = []
        for index, image in enumerate(source.get("images") or [], start=1):
            src = Path(image["frame_path"])
            expected_sha = image["frame_sha256"]
            if sha256(src) != expected_sha:
                raise SystemExit(f"source image SHA mismatch: {audit_id} image {index}")
            suffix = src.suffix.lower() or ".jpg"
            dst = images_dir / f"frame_{index:03d}{suffix}"
            shutil.copyfile(src, dst)
            if sha256(dst) != expected_sha:
                raise SystemExit(f"copied image SHA mismatch: {audit_id} image {index}")
            timestamp = float(image["resolved_timestamp_sec"])
            image_rows.append({
                "bundle_path": str(dst.relative_to(PACKAGE)),
                "sha256": expected_sha,
                "resolved_timestamp_sec": timestamp,
                "model_visible_timestamp_text": f"Frame timestamp={timestamp:.3f}s from video start.",
            })
            total_images += 1

        map_value = None
        map_raw_file = None
        if source.get("map") is not None:
            src_map = AUDIT / "review_packages" / f"{audit_id}.map.json"
            if not src_map.is_file():
                raise SystemExit(f"missing raw map: {audit_id}")
            dst_map = item_dir / "map.raw.json"
            shutil.copyfile(src_map, dst_map)
            if sha256(dst_map) != source["map"]["current_sha256"]:
                raise SystemExit(f"raw map SHA mismatch: {audit_id}")
            map_value = json.loads(dst_map.read_text())
            map_raw_file = {
                "bundle_path": str(dst_map.relative_to(PACKAGE)),
                "sha256": sha256(dst_map),
                "byte_preserved_from_frozen_review_package": True,
            }
            total_maps += 1

        item = {
            "audit_id": audit_id,
            "question": source["question"],
            "options": source["options"],
            "predicted_option": source["prediction"],
            "original_reason": source["reason"],
            "map": map_value,
            "map_raw_file": map_raw_file,
            "images": image_rows,
            "tool_feedback": [],
        }
        if set(item) != allowed_keys:
            raise SystemExit(f"unexpected item schema: {audit_id}")
        item_path = item_dir / "input.json"
        write_json(item_path, item)
        inventory.append({
            "position": position,
            "audit_id": audit_id,
            "input_file": str(item_path.relative_to(PACKAGE)),
            "input_sha256": sha256(item_path),
            "map_count": int(map_value is not None),
            "image_count": len(image_rows),
        })

    readme = """# ABD evidence-audit correction inputs (blinded)\n\nThis portable package contains only opaque audit inputs for evidence-support review.\nEach `items/E####/input.json` contains the question/options, predicted option, original\nreason, actual model-visible map (if any), relative paths to copied image bytes, resolved\ntimestamps and the exact model-visible timestamp text. `map.raw.json` preserves the\nfrozen map bytes. No identity mapping, arm name, gold, correctness, previous audit\nlabel/rationale, or statistical result is included. ABD was single-turn, so tool_feedback\nis empty. Item order is deterministically shuffled.\n"""
    (PACKAGE / "README.md").write_text(readme)

    # Per-file hashes cover every payload file. MANIFEST.json and MANIFEST.sha256 are
    # closure files and therefore are intentionally outside their own recursive inventory.
    payload_files = sorted(p for p in PACKAGE.rglob("*") if p.is_file())
    file_inventory = [{"path": str(p.relative_to(PACKAGE)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in payload_files]
    manifest = {
        "schema_version": "abd_evidence_audit_correction_blinded_inputs_v1",
        "blinded": True,
        "route_count": 176,
        "map_count": total_maps,
        "image_count": total_images,
        "tool_feedback_count": 0,
        "ordered_items": inventory,
        "files": file_inventory,
        "source_blinded_audit_sha256": sha256(AUDIT / "scored_evidence_audit_blinded.jsonl"),
        "source_audit_freeze_sha256": sha256(AUDIT / "audit_freeze.json"),
        "selection_order_seed_sha256": hashlib.sha256(ORDER_SEED.encode()).hexdigest(),
        "prohibited_material_included": False,
    }
    write_json(PACKAGE / "MANIFEST.json", manifest)
    (PACKAGE / "MANIFEST.sha256").write_text(f"{sha256(PACKAGE / 'MANIFEST.json')}  MANIFEST.json\n")

    with tarfile.open(ARCHIVE, "w:gz") as tf:
        tf.add(PACKAGE, arcname=PACKAGE_NAME)
    archive_sha = sha256(ARCHIVE)
    (OUT / f"{ARCHIVE.name}.sha256").write_text(f"{archive_sha}  {ARCHIVE.name}\n")

    # Validate archive, selection counts, copied bytes, and protected frozen files.
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        names = tf.getnames()
        if any("identity_mapping" in n or "scored_evidence_audit" in n or ".env" in n for n in names):
            raise SystemExit("prohibited filename in archive")
        tf.getmembers()
    for rec in inventory:
        obj = json.loads((PACKAGE / rec["input_file"]).read_text())
        if set(obj) != allowed_keys or obj["audit_id"] != rec["audit_id"]:
            raise SystemExit(f"post-build schema mismatch: {rec['audit_id']}")
        if any(k in obj for k in ("variant", "gold", "correct", "option_support", "rationale", "question_id", "task_id")):
            raise SystemExit(f"prohibited structured field: {rec['audit_id']}")
        if any(Path(im["bundle_path"]).is_absolute() for im in obj["images"]):
            raise SystemExit(f"absolute image path leaked: {rec['audit_id']}")
    after = tree_digest(protected)
    if after != before:
        raise SystemExit("existing frozen audit changed during package build")

    build_report = {
        "status": "PASS",
        "target_count_verified": len(targets),
        "control_count_verified": len(controls),
        "control_workflows": control_counts,
        "unique_routes": len(selected),
        "maps_copied": total_maps,
        "images_copied": total_images,
        "archive_sha256": archive_sha,
        "manifest_sha256": sha256(PACKAGE / "MANIFEST.json"),
        "protected_frozen_files_tree_sha256_before": before,
        "protected_frozen_files_tree_sha256_after": after,
        "gold_read": False,
        "identity_mapping_read": False,
        "reclassification_performed": False,
    }
    write_json(OUT / "BUILD_REPORT.json", build_report)
    print(json.dumps(build_report, ensure_ascii=False))


if __name__ == "__main__":
    main()
