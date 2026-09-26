#!/usr/bin/env python3
"""Prepare a new anonymous, no-gold 31-record independent-review package."""
from __future__ import annotations

import csv
import hashlib
import json
import secrets
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/abd_eval300_evidence_audit_correction_inputs_v1/ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS"
DIAG = ROOT / "outputs/abd_eval300_evidence_audit_correction_validation_v1/control_disagreements.csv"
CRITERIA = ROOT / "audit/codex_evidence_audit_r1_r3_gens_v3_v1/AUDIT_CRITERIA_FROZEN.md"
OUT = ROOT / "outputs/abd_eval300_disagreement_independent_review_v1"
PUBLIC = OUT / "reviewer_materials"
EXPECTED_CRITERIA_SHA = "7e9e25909a97ade16b86c896e5b086f5148b7d3f1bda8a8118db8c07f318f4e3"

INSTRUCTIONS = """# Unified independent evidence-review instructions

Use `AUDIT_CRITERIA_FROZEN.md` as the governing standard. Review each anonymous record independently.
Judge only whether the model-selected option is supported by evidence that was actually supplied in the
record. Do not re-answer the question. Do not seek identity, method, gold, correctness, prior reviews, or
other video evidence. If an original answer reason is ever present, it is a claim to verify and is not evidence.

Apply these five labels:

- `supported`: clear supplied evidence distinguishes and establishes the selected answer.
- `partially_supported`: supplied evidence supports a key part of the selected answer, but another necessary
  condition is not established. Scene relevance alone is not partial support.
- `unsupported`: evidence is technically reviewable but insufficient to support the selected answer.
- `contradicted`: supplied evidence explicitly refutes the selected answer. Omission, non-observation, or the
  appearance of another object is not automatically a contradiction.
- `unreviewable`: technical inability to inspect the supplied package. A missing modality is not itself a
  technical failure.

For counts, duration, immediately-next events, ordering, and whole-video non-occurrence, require evidence
that covers the corresponding condition. Do not treat sample-frame count or summary length as event count or
duration. When an answer option is only an image-path string, do not treat the option image as visible and do
not retrieve it; decide whether the remaining supplied evidence establishes the selected option's identity.
Do not fill missing facts from common sense or the answer wording.

For every item, read the complete map when present and visually open every supplied image. Timestamps are the
exact model-visible labels and must remain paired with their following images. Record one JSON object with:

- `review_id`
- `option_support`
- `evidence_references`
- `concise_rationale`
- `missing_key_conditions`
- `review_flag`
- `review_flag_reason`
- `map_read` (boolean; true iff a supplied map was completely read)
- `images_opened` (integer; must equal the supplied image count)

Use references such as `map:C07(585-675s)` or `image:frame_003@90.000s`. Preserve unresolved ambiguity in
`review_flag`; do not revise a label merely to make it agree with an unknown earlier reviewer.

This supplements the historical frozen standard by making the five-label operational boundaries, missing-
modality handling, image-path-option handling, timestamp pairing, and viewing-attestation fields explicit.
The historical per-item prompt and decoding parameters were not preserved, so this is an independent review
under the recovered written standard, not an exact replay of historical execution.
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refuse overwrite: {OUT}")
    if sha(CRITERIA) != EXPECTED_CRITERIA_SHA:
        raise RuntimeError("GenS audit criteria SHA mismatch")
    old_ids = [r["correction_audit_id"] for r in csv.DictReader(DIAG.open(encoding="utf-8"))]
    if len(old_ids) != 31 or len(set(old_ids)) != 31:
        raise RuntimeError("expected exactly 31 unique source IDs")

    OUT.mkdir(parents=True)
    PUBLIC.mkdir()
    (PUBLIC / "items").mkdir()
    shutil.copyfile(CRITERIA, PUBLIC / "AUDIT_CRITERIA_FROZEN.md")
    (PUBLIC / "UNIFIED_REVIEW_INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")

    seed = secrets.token_bytes(32)
    keyed = sorted(old_ids, key=lambda x: hashlib.sha256(seed + b"\0order\0" + x.encode()).digest())
    mapping = []
    public_rows = []
    for pos, old_id in enumerate(keyed, 1):
        opaque = "S" + hashlib.sha256(seed + b"\0id\0" + old_id.encode()).hexdigest()[:12].upper()
        src_dir = SOURCE / "items" / old_id
        src = json.loads((src_dir / "input.json").read_text(encoding="utf-8"))
        dst_dir = PUBLIC / "items" / opaque
        dst_dir.mkdir()
        images = []
        if src["images"]:
            (dst_dir / "images").mkdir()
        for index, image in enumerate(src["images"], 1):
            source_image = SOURCE / image["bundle_path"]
            target_name = f"frame_{index:03d}.jpg"
            target_image = dst_dir / "images" / target_name
            shutil.copyfile(source_image, target_image)
            if sha(target_image) != image["sha256"]:
                raise RuntimeError(f"image SHA mismatch: {old_id} {index}")
            images.append({
                "display_order": index,
                "timestamp_text": image["model_visible_timestamp_text"],
                "resolved_timestamp_sec": image["resolved_timestamp_sec"],
                "image_file": f"images/{target_name}",
                "sha256": image["sha256"],
            })
        map_meta = None
        if src.get("map_raw_file"):
            source_map = SOURCE / src["map_raw_file"]["bundle_path"]
            target_map = dst_dir / "map.raw.json"
            shutil.copyfile(source_map, target_map)
            if sha(target_map) != src["map_raw_file"]["sha256"]:
                raise RuntimeError(f"map SHA mismatch: {old_id}")
            map_meta = {"file": "map.raw.json", "sha256": sha(target_map), "complete_raw_bytes": True}
        item = {
            "review_id": opaque,
            "question": src["question"],
            "options": src["options"],
            "predicted_option": src["predicted_option"],
            "map": map_meta,
            "images": images,
            "tool_feedback": src.get("tool_feedback", []),
            "presentation_order": "complete map (if present), then each timestamp text immediately followed by its image, then question/options/predicted option",
            "original_answer_reason_included": False,
        }
        write_json(dst_dir / "input.json", item)
        public_rows.append({
            "review_id": opaque,
            "position": pos,
            "input_file": f"items/{opaque}/input.json",
            "input_sha256": sha(dst_dir / "input.json"),
            "map_count": int(map_meta is not None),
            "image_count": len(images),
        })
        mapping.append({"review_id": opaque, "source_audit_id": old_id, "position": pos})

    batches = []
    for n in range(0, len(public_rows), 4):
        part = public_rows[n:n + 4]
        batches.append({
            "batch_id": f"R{n // 4 + 1:02d}",
            "review_ids": [x["review_id"] for x in part],
            "item_count": len(part),
            "map_count": sum(x["map_count"] for x in part),
            "image_count": sum(x["image_count"] for x in part),
        })
    write_json(PUBLIC / "batch_manifest.json", {"batches": batches})
    write_json(PUBLIC / "package_manifest.json", {
        "schema_version": "abd_disagreement_independent_review_public_v1",
        "blinded": True,
        "record_count": 31,
        "map_count": sum(x["map_count"] for x in public_rows),
        "image_count": sum(x["image_count"] for x in public_rows),
        "items": public_rows,
        "prohibited_identity_or_prior_labels_included": False,
    })
    write_json(OUT / "identity_mapping_private.json", {
        "warning": "PRIVATE: never provide to blinded reviewer",
        "random_seed_hex": seed.hex(),
        "rows": mapping,
    })

    files = {}
    for p in sorted(PUBLIC.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(PUBLIC))] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    write_json(OUT / "PREPARATION_FREEZE.json", {
        "schema_version": "abd_disagreement_independent_review_preparation_freeze_v1",
        "frozen": True,
        "requested_reviewer_model": "gpt-5.6-sol",
        "historical_exact_execution_reproduction_claimed": False,
        "criteria_sha256": sha(PUBLIC / "AUDIT_CRITERIA_FROZEN.md"),
        "unified_instructions_sha256": sha(PUBLIC / "UNIFIED_REVIEW_INSTRUCTIONS.md"),
        "package_manifest_sha256": sha(PUBLIC / "package_manifest.json"),
        "batch_manifest_sha256": sha(PUBLIC / "batch_manifest.json"),
        "private_mapping_sha256": sha(OUT / "identity_mapping_private.json"),
        "public_files": files,
        "supplements_to_original_standard": [
            "explicit five-label boundary wording",
            "missing modality is not technical unreviewable",
            "opaque image-path option handling",
            "timestamp-image adjacency",
            "per-item map/image viewing attestations",
        ],
        "gold_read": False,
        "correctness_read": False,
        "prior_labels_in_public_package": False,
        "external_api_calls": 0,
    })
    print(json.dumps({"records": 31, "maps": sum(x["map_count"] for x in public_rows), "images": sum(x["image_count"] for x in public_rows), "batches": len(batches), "criteria_sha256": sha(PUBLIC / "AUDIT_CRITERIA_FROZEN.md")}, indent=2))


if __name__ == "__main__":
    main()
