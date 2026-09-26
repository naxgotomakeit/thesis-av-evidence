#!/usr/bin/env python3
"""Offline validation and control-consistency analysis for blinded correction reviews.

This script intentionally never opens a gold/correctness/accuracy source and never
modifies the original ABD run or frozen evidence audit.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
INCOMING = (Path.home() / "abd_correction_incoming_v1/correction_results.tar.gz").resolve()
EXPECTED_ARCHIVE_SHA = "aef2cd42ebc3f39b4da6407040d210cb0c3120b2a857f96a7ae20d9578a39a96"
EXPECTED_REVIEWS_SHA = "3ab37d3b323018ba063688790112639cb0053de4a0d1fb0407da2f012d35c991"
AUDIT = ROOT / "outputs/abd_eval300_evidence_audit_v1"
INPUT_PACKAGE = ROOT / "outputs/abd_eval300_evidence_audit_correction_inputs_v1/ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS"
INPUT_BUILD_REPORT = ROOT / "outputs/abd_eval300_evidence_audit_correction_inputs_v1/BUILD_REPORT.json"
FORMAL_INPUTS = ROOT / "drafts/abd_direct_eval300_v1/inputs"
BUILDER = ROOT / "scripts/build_abd_evidence_audit_correction_blinded_inputs_v1.py"
PAYLOAD_BUILDER = ROOT / "src/abd_draft_v1/core.py"
OUT = ROOT / "outputs/abd_eval300_evidence_audit_correction_validation_v1"
EXTRACTED = OUT / "extracted"
LABELS = ["supported", "partially_supported", "unsupported", "contradicted", "unreviewable"]
NEW_REQUIRED = {"audit_id", "option_support", "concise_rationale", "evidence_references", "missing_key_evidence", "confidence", "review_flag"}
CONTROL_SEED = "abd-evidence-audit-correction-controls-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, obj: object) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
    write_text(path, buf.getvalue())


def protected_digest() -> str:
    paths = [
        AUDIT / "audit_freeze.json", AUDIT / "scored_evidence_audit_blinded.jsonl",
        AUDIT / "valid_second_pass_batches.json", AUDIT / "batch_manifest.json",
        *sorted((AUDIT / "reviews").glob("B*.jsonl")),
    ]
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        h.update(str(path.relative_to(ROOT)).encode()); h.update(b"\0")
        h.update(sha256(path).encode()); h.update(b"\n")
    return h.hexdigest()


def safe_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tf.getmembers()
    names = set()
    for m in members:
        p = PurePosixPath(m.name)
        if p.is_absolute() or ".." in p.parts or not m.isfile():
            raise SystemExit(f"unsafe archive member: {m.name!r}")
        if m.name in names:
            raise SystemExit(f"duplicate archive member: {m.name}")
        names.add(m.name)
    return members


def exact_extract(tf: tarfile.TarFile, members: list[tarfile.TarInfo]) -> None:
    EXTRACTED.mkdir(parents=True)
    root = EXTRACTED.resolve()
    for m in members:
        dst = (EXTRACTED / m.name).resolve()
        if root not in dst.parents:
            raise SystemExit(f"archive traversal: {m.name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = tf.extractfile(m)
        if src is None:
            raise SystemExit(f"unreadable member: {m.name}")
        with dst.open("wb") as out:
            shutil.copyfileobj(src, out)


def kappa(matrix: dict[str, dict[str, int]], n: int) -> float | None:
    if not n:
        return None
    po = sum(matrix[x][x] for x in LABELS) / n
    old_marg = {x: sum(matrix[x][y] for y in LABELS) for x in LABELS}
    new_marg = {y: sum(matrix[x][y] for x in LABELS) for y in LABELS}
    pe = sum(old_marg[x] * new_marg[x] for x in LABELS) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"validation directory already exists; inspected but refusing overwrite: {OUT}")
    if not INCOMING.is_file() or sha256(INCOMING) != EXPECTED_ARCHIVE_SHA:
        raise SystemExit("incoming archive missing or SHA mismatch")

    before = protected_digest()
    expected_prior = json.loads(INPUT_BUILD_REPORT.read_text())["protected_frozen_files_tree_sha256_after"]
    if before != expected_prior:
        raise SystemExit("original frozen audit differs from correction-input build anchor")

    OUT.mkdir(parents=True)
    with tarfile.open(INCOMING, "r:gz") as tf:
        members = safe_members(tf)
        exact_extract(tf, members)
    expected_names = {"corrected_blinded_reviews.jsonl", "correction_audit_freeze.json", "review_progress.json", "CORRECTION_AUDIT_REPORT.md", "SHA256SUMS.txt"}
    if {m.name for m in members} != expected_names:
        raise SystemExit("archive member set mismatch")

    declared = {}
    for line in (EXTRACTED / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        declared[name] = digest
    for name in expected_names - {"SHA256SUMS.txt"}:
        if declared.get(name) != sha256(EXTRACTED / name):
            raise SystemExit(f"internal checksum mismatch: {name}")
    reviews_path = EXTRACTED / "corrected_blinded_reviews.jsonl"
    if sha256(reviews_path) != EXPECTED_REVIEWS_SHA:
        raise SystemExit("corrected reviews SHA mismatch")

    corrected = [json.loads(line) for line in reviews_path.read_text().splitlines() if line]
    corrected_ids = [r.get("audit_id") for r in corrected]
    if len(corrected) != 176 or len(set(corrected_ids)) != 176:
        raise SystemExit("corrected reviews are not 176 unique IDs")
    for row in corrected:
        if set(row) != NEW_REQUIRED or row["option_support"] not in LABELS:
            raise SystemExit(f"invalid correction review schema/label: {row.get('audit_id')}")
        if not isinstance(row["evidence_references"], list) or not isinstance(row["review_flag"], bool):
            raise SystemExit(f"invalid correction field type: {row['audit_id']}")

    input_manifest = json.loads((INPUT_PACKAGE / "MANIFEST.json").read_text())
    manifest_ids = [r["audit_id"] for r in input_manifest["ordered_items"]]
    if corrected_ids != manifest_ids or set(corrected_ids) != set(manifest_ids):
        raise SystemExit("correction result IDs/order do not exactly match source correction manifest")
    incoming_freeze = json.loads((EXTRACTED / "correction_audit_freeze.json").read_text())
    if incoming_freeze["source_archive_sha256"] != sha256(INPUT_PACKAGE.parent / "ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS.tar.gz"):
        raise SystemExit("incoming freeze source archive identity mismatch")
    if incoming_freeze["source_manifest_sha256"] != sha256(INPUT_PACKAGE / "MANIFEST.json"):
        raise SystemExit("incoming freeze source manifest identity mismatch")
    if incoming_freeze["results"]["sha256"] != EXPECTED_REVIEWS_SHA:
        raise SystemExit("incoming freeze results identity mismatch")

    # Deterministic reconstruction of the missing standalone target/control map.
    old_rows = [json.loads(line) for line in (AUDIT / "scored_evidence_audit_blinded.jsonl").read_text().splitlines() if line]
    old_by_id = {r["audit_id"]: r for r in old_rows}
    batch_doc = json.loads((AUDIT / "batch_manifest.json").read_text())
    batch_of = {aid: int(b["batch_id"][1:]) for b in batch_doc["batches"] for aid in b["audit_ids"]}
    targets = {aid for aid, r in old_by_id.items() if r["option_support"] == "unreviewable"}
    if len(targets) != 116:
        raise SystemExit("frozen target count is not 116")
    eligible = [aid for aid, r in old_by_id.items() if r["option_support"] != "unreviewable"]
    rng = random.Random(CONTROL_SEED)
    controls = set()
    control_by_workflow = {}
    for workflow, (lo, hi) in enumerate(((1, 64), (65, 128), (129, 192)), start=1):
        pool = sorted(aid for aid in eligible if lo <= batch_of[aid] <= hi)
        picked = set(rng.sample(pool, 20))
        controls |= picked
        control_by_workflow[workflow] = picked
    if targets & controls or len(controls) != 60 or set(corrected_ids) != targets | controls:
        raise SystemExit("deterministically reconstructed target/control identity does not match 176 result IDs")

    # Exact question/arm mapping. Only association fields are read; no gold/correctness source is opened.
    identity_doc = json.loads((AUDIT / "identity_mapping.json").read_text())
    identity = {r["audit_id"]: {"question_id": r["question_id"], "arm": r["variant"]} for r in identity_doc["rows"]}
    reconstructed = []
    for aid in manifest_ids:
        workflow = 1 if batch_of[aid] <= 64 else 2 if batch_of[aid] <= 128 else 3
        reconstructed.append({
            "correction_audit_id": aid, "original_audit_id": aid,
            "question_id": identity[aid]["question_id"], "arm": identity[aid]["arm"],
            "original_workflow": workflow, "original_batch_id": f"B{batch_of[aid]:03d}",
            "sample_role": "target" if aid in targets else "control",
            "mapping_method": "exact_audit_id_key_and_deterministic_build_seed_reconstruction",
        })
    write_json(OUT / "reconstructed_identity_mapping.json", {"warning": "No standalone mapping file was saved at package-build time; this mapping is exactly reconstructed from frozen IDs, batch membership, the preserved build seed, and identity_mapping exact keys.", "rows": reconstructed})

    new_by_id = {r["audit_id"]: r for r in corrected}
    matrix = {old: {new: 0 for new in LABELS} for old in LABELS}
    disagreements = []
    agreement_by_workflow = {}
    control_rows = []
    for workflow in (1, 2, 3):
        ids = sorted(control_by_workflow[workflow])
        agree = 0
        for aid in ids:
            old, new = old_by_id[aid], new_by_id[aid]
            matrix[old["option_support"]][new["option_support"]] += 1
            same = old["option_support"] == new["option_support"]
            agree += same
            rec = next(r for r in reconstructed if r["correction_audit_id"] == aid)
            row = {
                **rec, "old_label": old["option_support"], "new_label": new["option_support"],
                "exact_agreement": same, "old_reason": old["rationale"],
                "new_reason": new["concise_rationale"],
                "old_evidence_references": json.dumps(old["evidence_refs"], ensure_ascii=False),
                "new_evidence_references": json.dumps(new["evidence_references"], ensure_ascii=False),
                "old_missing_key_evidence": old["missing_key_evidence"],
                "new_missing_key_evidence": new["missing_key_evidence"],
            }
            control_rows.append(row)
            if not same: disagreements.append(row)
        agreement_by_workflow[str(workflow)] = {"n": 20, "exact_agreement": agree, "proportion": agree / 20}
    total_agree = sum(r["exact_agreement"] for r in control_rows)
    kap = kappa(matrix, 60)
    consistency = {
        "overall": {"n": 60, "exact_agreement": total_agree, "proportion": total_agree / 60, "unweighted_cohen_kappa": kap},
        "by_workflow": agreement_by_workflow,
        "confusion_matrix_old_rows_new_columns": matrix,
        "disagreement_count": len(disagreements),
    }
    write_json(OUT / "control_consistency_stats.json", consistency)
    disagreement_fields = ["correction_audit_id", "original_audit_id", "question_id", "arm", "original_workflow", "original_batch_id", "old_label", "new_label", "old_reason", "new_reason", "old_evidence_references", "new_evidence_references", "old_missing_key_evidence", "new_missing_key_evidence"]
    write_csv(OUT / "control_disagreements.csv", disagreements, disagreement_fields)

    target_new = Counter(new_by_id[aid]["option_support"] for aid in targets)
    control_new = Counter(new_by_id[aid]["option_support"] for aid in controls)
    target_summary = {
        "target_n": 116, "old_label_distribution": {"unreviewable": 116},
        "new_label_distribution": {x: target_new[x] for x in LABELS},
        "changed_from_old_unreviewable": sum(new_by_id[aid]["option_support"] != "unreviewable" for aid in targets),
        "control_n": 60, "control_new_label_distribution": {x: control_new[x] for x in LABELS},
        "mixed_176_distribution_intentionally_not_reported_as_900_rate": True,
    }
    write_json(OUT / "target_and_control_label_distributions.json", target_summary)

    flag_rows = []
    for aid in manifest_ids:
        new = new_by_id[aid]
        if new["review_flag"]:
            rec = next(r for r in reconstructed if r["correction_audit_id"] == aid)
            flag_rows.append({**rec, "new_label": new["option_support"], "reason": new["concise_rationale"], "missing_key_evidence": new["missing_key_evidence"], "evidence_references": json.dumps(new["evidence_references"], ensure_ascii=False)})
    write_csv(OUT / "review_flags.csv", flag_rows, ["correction_audit_id", "original_audit_id", "question_id", "arm", "original_workflow", "original_batch_id", "sample_role", "new_label", "reason", "missing_key_evidence", "evidence_references"])

    # Determine whether path-valued option images reached the original model as image blocks.
    input_rows = {}
    for arm in "ABD":
        for line in (FORMAL_INPUTS / f"{arm}.jsonl").read_text().splitlines():
            row = json.loads(line); input_rows[(arm, row["question_id"])] = row
    option_path_rows = []
    for rec in reconstructed:
        aid = rec["original_audit_id"]
        package = json.loads((AUDIT / "review_packages" / f"{aid}.json").read_text())
        path_options = {k: v for k, v in package["options"].items() if "navigation_images" in str(v) or "spatial_layout_images" in str(v)}
        if not path_options:
            continue
        frozen = input_rows[(rec["arm"], rec["question_id"])]
        option_path_rows.append({
            **rec, "path_option_count": len(path_options),
            "path_option_types": ";".join(sorted({"navigation_images" if "navigation_images" in str(v) else "spatial_layout_images" for v in path_options.values()})),
            "path_options": json.dumps(path_options, ensure_ascii=False),
            "frozen_evidence_image_count": len(frozen.get("images") or []),
            "determination": "original_model_received_option_paths_as_text_only_not_as_option_image_blocks",
            "basis": "GensPromptBuilder._content emits image blocks only from row['images']; _question_text interpolates option values into the final text block.",
            "certainty": "high_from_frozen_builder_and_input_row; no byte-for-byte wire payload was persisted",
        })
    write_csv(OUT / "missing_option_image_records.csv", option_path_rows, ["correction_audit_id", "original_audit_id", "question_id", "arm", "original_workflow", "original_batch_id", "sample_role", "path_option_count", "path_option_types", "path_options", "frozen_evidence_image_count", "determination", "basis", "certainty"])

    # Human-readable reports.
    def pct(x: float) -> str: return f"{100*x:.1f}%"
    integrity_report = f"""# Correction package integrity and mapping validation\n\n## Package integrity\n\n- Incoming absolute path: `{INCOMING}`\n- Archive SHA-256: `{sha256(INCOMING)}` (expected; PASS).\n- Corrected review SHA-256: `{sha256(reviews_path)}` (expected; PASS).\n- Internal SHA256SUMS: all four declared payload hashes match.\n- Archive safety: 5 unique regular files; no absolute paths, traversal, links, devices, or duplicate members.\n- Records: 176 rows, 176 unique IDs, exact ID and order match to the original correction input manifest.\n- Schema and label values: PASS.\n\n## Identity recovery\n\n- Targets: exactly 116 frozen `unreviewable` records.\n- Controls: exactly 60, with 20 from each original workflow; no overlap or duplicates.\n- Correction audit ID equals the preserved opaque original audit ID; question/arm association used exact identity-map keys.\n- Important provenance limitation: the package builder did not save a standalone target/control mapping file. The mapping in `reconstructed_identity_mapping.json` was deterministically reconstructed from the frozen ID set, batch manifest, preserved builder seed and exact identity-map keys. No question, image, map, or fuzzy matching was used.\n- Frozen audit protected-tree SHA before/after validation: `{before}`. It matches the correction-input build anchor.\n\nFile integrity passing does not establish semantic audit quality; scale consistency is reported separately.\n"""
    write_text(OUT / "PACKAGE_INTEGRITY_AND_MAPPING_REPORT.md", integrity_report)

    matrix_lines = ["| old \\ new | " + " | ".join(LABELS) + " |", "|---|" + "---:|" * len(LABELS)]
    for old in LABELS:
        matrix_lines.append("| " + old + " | " + " | ".join(str(matrix[old][new]) for new in LABELS) + " |")
    drift_pairs = Counter((r["old_label"], r["new_label"]) for r in disagreements)
    drift_text = ", ".join(f"{a}→{b}: {n}" for (a, b), n in drift_pairs.most_common()) or "none"
    workflow_rates = [agreement_by_workflow[str(i)]["proportion"] for i in (1,2,3)]
    concentration = "The 20-item workflow samples are too small for firm workflow-level conclusions."
    consistency_report = f"""# Control-sample audit consistency\n\n- Overall exact agreement: {total_agree}/60 = {pct(total_agree/60)}.\n- Unweighted Cohen's kappa: {kap:.6f}.\n- Workflow 1: {agreement_by_workflow['1']['exact_agreement']}/20 = {pct(workflow_rates[0])}.\n- Workflow 2: {agreement_by_workflow['2']['exact_agreement']}/20 = {pct(workflow_rates[1])}.\n- Workflow 3: {agreement_by_workflow['3']['exact_agreement']}/20 = {pct(workflow_rates[2])}.\n\n## Confusion matrix\n\n{chr(10).join(matrix_lines)}\n\nThere are {len(disagreements)} changed control labels. Most common directions: {drift_text}. {concentration} The disagreements should be read as potential scale/boundary drift, particularly where `supported`, `partially_supported`, and `unsupported` are adjacent interpretations; no post-hoc pass threshold was applied. Full reasons and evidence references are in `control_disagreements.csv`.\n"""
    write_text(OUT / "CONTROL_CONSISTENCY_REPORT.md", consistency_report)

    target_report = "# Target correction-label summary\n\nThe 116 targets and 60 controls are reported separately; their mixed 176-row distribution is not treated as a 900-route support rate.\n\n## 116 target rows\n\n" + "\n".join(f"- {lab}: {target_new[lab]}" for lab in LABELS) + f"\n\nChanged from the original `unreviewable` label: {target_summary['changed_from_old_unreviewable']}/116. No old frozen label was replaced.\n\n## 60 control rows — new labels only\n\n" + "\n".join(f"- {lab}: {control_new[lab]}" for lab in LABELS) + f"\n\nReview flags are preserved without automatic clearing: {len(flag_rows)} total; details and stated reasons are in `review_flags.csv`.\n"
    write_text(OUT / "TARGET_LABEL_CHANGE_SUMMARY.md", target_report)

    unresolved = f"""# Unresolved issues\n\n1. No standalone identity/target-control mapping file was saved when the correction input package was built. Exact deterministic reconstruction succeeded, but this remains a provenance defect.\n2. {len(option_path_rows)} correction records contain `navigation_images` or `spatial_layout_images` option paths. Frozen builder inspection indicates those option assets were not sent as image blocks: the model received the path strings in question/options text. This was an original request limitation, not merely a correction-export omission. The conclusion is high-confidence from frozen code and input rows, although a byte-for-byte serialized wire payload was not persisted.\n3. File/hash validity and full reviewer completion do not by themselves establish semantic quality. Control agreement and rationale differences must be reviewed.\n4. No correction labels have been merged into the 900-row frozen audit, and no final arm-level support rates were generated.\n"""
    write_text(OUT / "UNRESOLVED_ISSUES.md", unresolved)

    if total_agree < 60:
        recommendation = f"The control set changed on {len(disagreements)}/60 rows (agreement {pct(total_agree/60)}, kappa {kap:.3f}). Because these controls were not selected as known problem cases, correcting only the 116 targets would mix two demonstrated audit scales. Before formal merging, independently adjudicate all {len(disagreements)} control disagreements and expand review around the dominant label boundaries/workflow pattern. The 20-row workflow samples are exploratory and should guide, not define, expansion."
    else:
        recommendation = "All 60 controls agree, but this small stratified sample cannot prove global equivalence. Review all flagged targets and a larger independent control sample before formal merging if the final support rates are consequential."
    write_text(OUT / "REVIEW_SCOPE_RECOMMENDATION.md", "# Review-scope recommendation\n\n" + recommendation + "\n\nThis recommendation is based on label/rationale consistency only; no gold, correctness, or accuracy was read.\n")

    after = protected_digest()
    if after != before:
        raise SystemExit("original frozen audit changed during validation")
    shutil.copy2(__file__, OUT / "validate_abd_correction_results_v1.py")
    output_files = sorted(p for p in OUT.rglob("*") if p.is_file())
    validation_manifest = {
        "schema_version": "abd_correction_validation_v1",
        "status": "PASS_WITH_DOCUMENTED_MAPPING_PROVENANCE_LIMITATION",
        "archive_sha256": sha256(INCOMING), "corrected_reviews_sha256": sha256(reviews_path),
        "records": 176, "targets": 116, "controls": 60,
        "gold_read": False, "correctness_read": False, "accuracy_statistics_read": False,
        "model_or_external_api_calls": 0, "reclassification_performed": False,
        "old_labels_replaced_or_merged": False,
        "protected_frozen_tree_sha256_before": before, "protected_frozen_tree_sha256_after": after,
        "files": [{"path": str(p.relative_to(OUT)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in output_files],
    }
    write_json(OUT / "correction_validation_manifest.json", validation_manifest)
    print(json.dumps({
        "status": validation_manifest["status"], "agreement": f"{total_agree}/60",
        "kappa": kap, "workflow_agreement": agreement_by_workflow,
        "target_new_distribution": {x: target_new[x] for x in LABELS},
        "control_new_distribution": {x: control_new[x] for x in LABELS},
        "review_flags": len(flag_rows), "path_option_records": len(option_path_rows),
        "protected_unchanged": before == after,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
