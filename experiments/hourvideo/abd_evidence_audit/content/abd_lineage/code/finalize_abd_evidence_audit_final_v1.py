#!/usr/bin/env python3
"""Deterministically merge and score the final ABD evidence audit.

Phase ``merge`` never opens a correctness/gold-bearing input.  It validates and
freezes the selected review labels first.  Phase ``stats`` refuses to run unless
that freeze validates byte-for-byte; only then does it open the existing ABD
correctness report.  No semantic decisions are made by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/abd_eval300_evidence_audit_final_v1"
ORIG = ROOT / "outputs/abd_eval300_evidence_audit_v1"
CORR = ROOT / "outputs/abd_eval300_evidence_audit_correction_validation_v1"
SOL = ROOT / "outputs/abd_eval300_disagreement_independent_review_v1"
SCORE = ROOT / "outputs/abd_eval300_analysis_v1/abd_score_report.json"
HIST = ROOT / "audit/codex_evidence_audit_r1_r3_gens_v3_v1/evidence_score_cross_stats.json"

ORIG_REVIEWS = ORIG / "scored_evidence_audit_blinded.jsonl"
ORIG_MAP = ORIG / "identity_mapping.json"
ORIG_FREEZE = ORIG / "audit_freeze.json"
CORR_REVIEWS = CORR / "extracted/corrected_blinded_reviews.jsonl"
CORR_MAP = CORR / "reconstructed_identity_mapping.json"
CORR_MANIFEST = CORR / "correction_validation_manifest.json"
CORR_FREEZE = CORR / "extracted/correction_audit_freeze.json"
SOL_REVIEWS = SOL / "blinded_reviewer_output/blinded_reviews.jsonl"
SOL_MAP = SOL / "identity_mapping_private.json"
SOL_FREEZE = SOL / "BLINDED_REVIEW_FREEZE.json"

EXPECTED_SOL_SHA = "75fdab627ddf165f75133f026d0a6e605e6b838bdcb58912e752a67a0783557f"
EXPECTED_CORR_SHA = "3ab37d3b323018ba063688790112639cb0053de4a0d1fb0407da2f012d35c991"
LABELS = {"supported", "partially_supported", "unsupported", "contradicted", "unreviewable"}
ARMS = ("A", "B", "D")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: record is not an object")
                rows.append(value)
    return rows


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_bytes(path, text.encode())


def write_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode())


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def unique_index(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}: missing/invalid {key}")
        if value in result:
            raise ValueError(f"{label}: duplicate {key} {value}")
        result[value] = row
    return result


def selected_source_hashes() -> dict[str, str]:
    paths = [ORIG_REVIEWS, ORIG_MAP, ORIG_FREEZE, CORR_REVIEWS, CORR_MAP,
             CORR_MANIFEST, CORR_FREEZE, SOL_REVIEWS, SOL_MAP, SOL_FREEZE]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def normalize_review(source: str, row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("option_support")
    if label not in LABELS:
        raise ValueError(f"{source}: illegal label {label!r}")
    rationale = row.get("rationale", row.get("concise_rationale"))
    refs = row.get("evidence_refs", row.get("evidence_references", []))
    missing = row.get("missing_key_evidence", row.get("missing_key_conditions"))
    return {
        "final_label": label,
        "rationale": rationale,
        "evidence_references": refs,
        "map_region_time_range": row.get("map_region_time_range"),
        "frame_timestamps": row.get("frame_timestamps"),
        "missing_key_evidence": missing,
        "confidence": row.get("confidence"),
        "review_flag": row.get("review_flag"),
        "review_flag_reason": row.get("review_flag_reason"),
        "evidence_relationship": row.get("evidence_relationship"),
    }


def phase_merge() -> None:
    # Deliberately do not reference or open SCORE/HIST in this phase.
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    before_hashes = selected_source_hashes()
    orig_freeze = load_json(ORIG_FREEZE)
    corr_manifest = load_json(CORR_MANIFEST)
    corr_freeze = load_json(CORR_FREEZE)
    sol_freeze = load_json(SOL_FREEZE)

    checks: list[dict[str, Any]] = []
    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"check": name, "pass": bool(condition), "detail": detail})
        if not condition:
            raise ValueError(f"FAILED {name}: {detail}")

    check("original_blinded_review_sha_matches_freeze",
          before_hashes[str(ORIG_REVIEWS.relative_to(ROOT))] == orig_freeze["scored_evidence_audit_blinded_sha256"],
          before_hashes[str(ORIG_REVIEWS.relative_to(ROOT))])
    check("correction_review_sha_matches_expected_and_manifests",
          before_hashes[str(CORR_REVIEWS.relative_to(ROOT))] == EXPECTED_CORR_SHA
          == corr_manifest["corrected_reviews_sha256"] == corr_freeze["results"]["sha256"],
          before_hashes[str(CORR_REVIEWS.relative_to(ROOT))])
    check("sol_review_sha_matches_expected_and_freeze",
          before_hashes[str(SOL_REVIEWS.relative_to(ROOT))] == EXPECTED_SOL_SHA
          == sol_freeze["blinded_reviews_sha256"],
          before_hashes[str(SOL_REVIEWS.relative_to(ROOT))])

    orig_reviews = unique_index(load_jsonl(ORIG_REVIEWS), "audit_id", "original reviews")
    orig_mapping_rows = load_json(ORIG_MAP)["rows"]
    orig_mapping = unique_index(orig_mapping_rows, "audit_id", "original mapping")
    corr_reviews = unique_index(load_jsonl(CORR_REVIEWS), "audit_id", "correction reviews")
    corr_mapping_rows = load_json(CORR_MAP)["rows"]
    corr_mapping = unique_index(corr_mapping_rows, "correction_audit_id", "correction mapping")
    sol_reviews = unique_index(load_jsonl(SOL_REVIEWS), "review_id", "Sol reviews")
    sol_mapping_rows = load_json(SOL_MAP)["rows"]
    sol_mapping = unique_index(sol_mapping_rows, "review_id", "Sol mapping")

    check("original_review_and_mapping_ids", set(orig_reviews) == set(orig_mapping) and len(orig_reviews) == 900,
          {"reviews": len(orig_reviews), "mapping": len(orig_mapping)})
    check("correction_review_and_mapping_ids", set(corr_reviews) == set(corr_mapping) and len(corr_reviews) == 176,
          {"reviews": len(corr_reviews), "mapping": len(corr_mapping)})
    check("sol_review_and_mapping_ids", set(sol_reviews) == set(sol_mapping) and len(sol_reviews) == 31,
          {"reviews": len(sol_reviews), "mapping": len(sol_mapping)})

    original_ids = set(orig_reviews)
    correction_to_original = {cid: row["original_audit_id"] for cid, row in corr_mapping.items()}
    correction_ids = set(correction_to_original.values())
    check("correction_mapping_one_to_one", len(correction_ids) == 176 and correction_ids <= original_ids,
          {"unique_original_ids": len(correction_ids)})
    roles = Counter(row["sample_role"] for row in corr_mapping_rows)
    target_ids = {row["original_audit_id"] for row in corr_mapping_rows if row["sample_role"] == "target"}
    control_ids = {row["original_audit_id"] for row in corr_mapping_rows if row["sample_role"] == "control"}
    check("correction_roles_and_disjointness", roles == {"target": 116, "control": 60}
          and not (target_ids & control_ids), {"roles": dict(roles), "overlap": len(target_ids & control_ids)})
    sol_to_original = {rid: row["source_audit_id"] for rid, row in sol_mapping.items()}
    sol_ids = set(sol_to_original.values())
    check("sol_is_exact_subset_of_controls", len(sol_ids) == 31 and sol_ids <= control_ids,
          {"sol_ids": len(sol_ids), "outside_controls": sorted(sol_ids - control_ids)})

    final_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    sol_by_original = {sol_to_original[rid]: sol_reviews[rid] for rid in sol_reviews}
    sol_review_id = {sol_to_original[rid]: rid for rid in sol_reviews}
    corr_by_original = {correction_to_original[cid]: corr_reviews[cid] for cid in corr_reviews}
    corr_id_by_original = {correction_to_original[cid]: cid for cid in corr_reviews}
    corr_role_by_original = {row["original_audit_id"]: row["sample_role"] for row in corr_mapping_rows}
    source_paths = {
        "sol_independent_review": SOL_REVIEWS,
        "correction": CORR_REVIEWS,
        "original_abd": ORIG_REVIEWS,
    }
    source_hash = {name: before_hashes[str(path.relative_to(ROOT))] for name, path in source_paths.items()}

    for audit_id in sorted(original_ids):
        identity = orig_mapping[audit_id]
        original = orig_reviews[audit_id]
        if audit_id in sol_by_original:
            source = "sol_independent_review"
            source_record_id = sol_review_id[audit_id]
            source_row = sol_by_original[audit_id]
        elif audit_id in corr_by_original:
            source = "correction"
            source_record_id = corr_id_by_original[audit_id]
            source_row = corr_by_original[audit_id]
        else:
            source = "original_abd"
            source_record_id = audit_id
            source_row = original
        normalized = normalize_review(source, source_row)
        final_rows.append({
            "audit_id": audit_id,
            "question_id": identity["question_id"],
            "task_id": identity["task_id"],
            "arm": identity["variant"],
            **normalized,
            "selected_review_version": source,
            "source_record_id": source_record_id,
            "source_file": str(source_paths[source].relative_to(ROOT)),
            "source_sha256": source_hash[source],
            "source_review_record": source_row,
        })
        replacement_rows.append({
            "audit_id": audit_id,
            "question_id": identity["question_id"],
            "task_id": identity["task_id"],
            "arm": identity["variant"],
            "sample_role": corr_role_by_original.get(audit_id, "not_in_correction"),
            "original_abd_label": original["option_support"],
            "correction_label": corr_by_original.get(audit_id, {}).get("option_support", ""),
            "sol_label": sol_by_original.get(audit_id, {}).get("option_support", ""),
            "final_label": normalized["final_label"],
            "selected_review_version": source,
            "source_record_id": source_record_id,
            "source_file": str(source_paths[source].relative_to(ROOT)),
            "source_sha256": source_hash[source],
        })

    source_counts = Counter(row["selected_review_version"] for row in final_rows)
    arm_counts = Counter(row["arm"] for row in final_rows)
    task_ids = {row["task_id"] for row in final_rows}
    expected_task_ids = {row["task_id"] for row in orig_mapping_rows}
    check("final_source_counts", source_counts == {
        "sol_independent_review": 31, "correction": 145, "original_abd": 724}, dict(source_counts))
    check("final_arm_counts", arm_counts == {"A": 300, "B": 300, "D": 300}, dict(arm_counts))
    check("final_unique_and_complete_ids", len(final_rows) == 900 and len(task_ids) == 900
          and task_ids == expected_task_ids, {"rows": len(final_rows), "tasks": len(task_ids)})
    check("final_labels_legal", all(row["final_label"] in LABELS for row in final_rows), True)

    final_path = OUT / "final_evidence_audit_labels.jsonl"
    replacement_path = OUT / "review_version_replacements.csv"
    write_jsonl(final_path, final_rows)
    write_csv(replacement_path, [
        "audit_id", "question_id", "task_id", "arm", "sample_role",
        "original_abd_label", "correction_label", "sol_label", "final_label",
        "selected_review_version", "source_record_id", "source_file", "source_sha256",
    ], replacement_rows)
    after_hashes = selected_source_hashes()
    check("selected_historical_source_files_unchanged", before_hashes == after_hashes, True)
    integrity = {
        "schema_version": "abd_evidence_audit_final_merge_integrity_v1",
        "created_at_utc": utc_now(),
        "status": "PASS",
        "correctness_or_gold_opened": False,
        "checks": checks,
        "source_counts": dict(source_counts),
        "arm_counts": dict(arm_counts),
        "correction_roles": dict(roles),
        "source_input_sha256": before_hashes,
    }
    write_json(OUT / "merge_integrity.json", integrity)
    freeze = {
        "schema_version": "abd_evidence_audit_final_label_freeze_v1",
        "status": "FROZEN_AND_VERIFIED",
        "frozen_at_utc": utc_now(),
        "selection_priority": ["sol_independent_review", "correction", "original_abd"],
        "record_count": 900,
        "unique_audit_ids": 900,
        "source_counts": dict(source_counts),
        "arm_counts": dict(arm_counts),
        "correctness_or_gold_opened_before_freeze": False,
        "final_evidence_audit_labels_sha256": sha256(final_path),
        "review_version_replacements_sha256": sha256(replacement_path),
        "merge_integrity_sha256": sha256(OUT / "merge_integrity.json"),
        "source_input_sha256": before_hashes,
    }
    write_json(OUT / "label_freeze.json", freeze)
    print(json.dumps({"status": "labels_frozen", "output": str(OUT), "freeze": freeze}, indent=2))


def pct(n: int, d: int) -> str:
    return "NA" if d == 0 else f"{100 * n / d:.2f}%"


def phase_stats() -> None:
    freeze_path = OUT / "label_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("label_freeze.json is required; run merge phase first")
    freeze = load_json(freeze_path)
    labels_path = OUT / "final_evidence_audit_labels.jsonl"
    replacements_path = OUT / "review_version_replacements.csv"
    merge_integrity_path = OUT / "merge_integrity.json"
    if freeze.get("status") != "FROZEN_AND_VERIFIED":
        raise ValueError("label freeze status is not FROZEN_AND_VERIFIED")
    expected = {
        labels_path: freeze["final_evidence_audit_labels_sha256"],
        replacements_path: freeze["review_version_replacements_sha256"],
        merge_integrity_path: freeze["merge_integrity_sha256"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"frozen label artifact changed: {path}")
    if selected_source_hashes() != freeze["source_input_sha256"]:
        raise ValueError("a selected historical source input changed after label freeze")

    # Correctness is first opened only after all frozen-label checks above pass.
    correctness_first_read_at = utc_now()
    score = load_json(SCORE)
    labels = load_jsonl(labels_path)
    by_task = unique_index(labels, "task_id", "final labels")
    score_rows = score["per_question"]
    if len(score_rows) != 300:
        raise ValueError(f"expected 300 score rows, got {len(score_rows)}")

    joined: list[dict[str, Any]] = []
    for question in score_rows:
        qid = question["question_id"]
        for arm in ARMS:
            task_id = question[f"{arm}_task_id"]
            if task_id not in by_task:
                raise ValueError(f"score task not in frozen labels: {task_id}")
            row = by_task[task_id]
            if row["question_id"] != qid or row["arm"] != arm:
                raise ValueError(f"identity mismatch for {task_id}")
            joined.append({
                "audit_id": row["audit_id"], "question_id": qid, "task_id": task_id,
                "arm": arm, "final_label": row["final_label"],
                "selected_review_version": row["selected_review_version"],
                "review_flag": row["review_flag"],
                "prediction": question[f"{arm}_prediction"],
                "correct": bool(question[f"{arm}_correct"]),
                "terminal_state": question[f"{arm}_state"],
                "result_class": question[f"{arm}_result_class"],
                "failure_category": question[f"{arm}_failure_category"],
            })
    if len(joined) != 900 or len({row["task_id"] for row in joined}) != 900:
        raise ValueError("joined score set is not exactly 900 unique tasks")

    metrics: list[dict[str, Any]] = []
    for arm in (*ARMS, "ALL"):
        rows = joined if arm == "ALL" else [row for row in joined if row["arm"] == arm]
        denominator = 900 if arm == "ALL" else 300
        label_counts = Counter(row["final_label"] for row in rows)
        valid = sum(row["result_class"] == "valid_answer" for row in rows)
        correct = sum(row["correct"] for row in rows)
        correct_supported = sum(row["correct"] and row["final_label"] == "supported" for row in rows)
        correct_uoc = sum(row["correct"] and row["final_label"] in {"unsupported", "contradicted"} for row in rows)
        flags = sum(row["review_flag"] is True for row in rows)
        metrics.append({
            "arm": arm, "denominator": denominator, "valid_answers": valid,
            "no_answer_or_run_failure": denominator - valid, "correct": correct,
            "accuracy": correct / denominator,
            **{label: label_counts[label] for label in sorted(LABELS)},
            **{f"{label}_rate": label_counts[label] / denominator for label in sorted(LABELS)},
            "strict_supported_rate": label_counts["supported"] / denominator,
            "correct_and_supported": correct_supported,
            "correct_and_supported_rate_all": correct_supported / denominator,
            "correct_unsupported_or_contradicted": correct_uoc,
            "correct_unsupported_or_contradicted_share_of_correct": None if correct == 0 else correct_uoc / correct,
            "review_flag_count": flags,
        })

    expected_correct = {"A": 98, "B": 103, "D": 103}
    actual_correct = {row["arm"]: row["correct"] for row in metrics if row["arm"] in ARMS}
    if actual_correct != expected_correct:
        raise ValueError(f"accuracy reproduction failed: {actual_correct}")
    if any(row["valid_answers"] != 300 for row in metrics if row["arm"] in ARMS):
        raise ValueError("expected 300 valid answers per ABD arm")

    fields = [
        "arm", "denominator", "valid_answers", "no_answer_or_run_failure", "correct", "accuracy",
        "supported", "supported_rate", "partially_supported", "partially_supported_rate",
        "unsupported", "unsupported_rate", "contradicted", "contradicted_rate",
        "unreviewable", "unreviewable_rate", "strict_supported_rate",
        "correct_and_supported", "correct_and_supported_rate_all",
        "correct_unsupported_or_contradicted",
        "correct_unsupported_or_contradicted_share_of_correct", "review_flag_count",
    ]
    write_csv(OUT / "abd_final_metrics.csv", fields, metrics)
    write_jsonl(OUT / "final_evidence_audit_with_correctness.jsonl", joined)

    table = [
        "# ABD final evidence-audit metrics", "",
        "All arm-level rates use 300 as denominator; ALL uses 900. Labels were frozen before correctness was opened.", "",
        "| Arm | Correct | Accuracy | supported | partially_supported | unsupported | contradicted | unreviewable | Strict support | Correct & supported | Correct unsupported/contradicted | Share of correct | Review flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        table.append(
            f"| {row['arm']} | {row['correct']}/{row['denominator']} | {pct(row['correct'], row['denominator'])} | "
            f"{row['supported']} | {row['partially_supported']} | {row['unsupported']} | {row['contradicted']} | {row['unreviewable']} | "
            f"{pct(row['supported'], row['denominator'])} | {row['correct_and_supported']} ({pct(row['correct_and_supported'], row['denominator'])}) | "
            f"{row['correct_unsupported_or_contradicted']} | {pct(row['correct_unsupported_or_contradicted'], row['correct'])} | {row['review_flag_count']} |"
        )
    write_text(OUT / "ABD_FINAL_METRICS.md", "\n".join(table) + "\n")

    hist = load_json(HIST)
    comparison_rows = []
    for method in ("R1", "R3", "GENS"):
        data = hist["methods"][method]
        counts = data["support_counts"]
        comparison_rows.append({
            "method": method, "audit_version": "codex_evidence_audit_r1_r3_gens_v3_v1",
            "source_file": str(HIST.relative_to(ROOT)), "denominator": 300,
            "correct": data["correct"], "accuracy": data["accuracy"],
            "supported": counts.get("supported", 0),
            "partially_supported": counts.get("partially_supported", 0),
            "unsupported": counts.get("unsupported", 0),
            "contradicted": counts.get("contradicted", 0),
            "unreviewable": counts.get("unreviewable", 0),
            "no_final_answer": counts.get("no_final_answer", 0),
            "strict_supported_rate": counts.get("supported", 0) / 300,
            "correct_unsupported_or_contradicted": data["correct_but_unsupported_or_contradicted"]["count"],
            "correct_unsupported_or_contradicted_share_of_correct": data["correct_but_unsupported_or_contradicted"]["share_of_correct"],
        })
    for row in metrics:
        if row["arm"] in ARMS:
            comparison_rows.append({
                "method": row["arm"], "audit_version": "abd_eval300_evidence_audit_final_v1",
                "source_file": "outputs/abd_eval300_evidence_audit_final_v1/label_freeze.json",
                "denominator": 300, "correct": row["correct"], "accuracy": row["accuracy"],
                "supported": row["supported"], "partially_supported": row["partially_supported"],
                "unsupported": row["unsupported"], "contradicted": row["contradicted"],
                "unreviewable": row["unreviewable"], "no_final_answer": row["no_answer_or_run_failure"],
                "strict_supported_rate": row["strict_supported_rate"],
                "correct_unsupported_or_contradicted": row["correct_unsupported_or_contradicted"],
                "correct_unsupported_or_contradicted_share_of_correct": row["correct_unsupported_or_contradicted_share_of_correct"],
            })
    comparison_fields = list(comparison_rows[0])
    write_csv(OUT / "abd_gens_direct_comparison.csv", comparison_fields, comparison_rows)

    write_text(OUT / "AUDIT_METHOD_AND_LIMITATIONS.md", """# Audit method and limitations

The ABD evidence audit is a Codex-assisted evidence-support judgment over 900 routes (300 each for A, B, and D). Reviewers used only the map, images, and model-visible timestamps actually supplied to the answer model. The five labels remain distinct: `supported`, `partially_supported`, `unsupported`, `contradicted`, and `unreviewable`.

Reviews were completed in batches and included limited blinded re-review. The final per-record selection rule was fixed before correctness linkage: **independent Sol review > correction review > original ABD review**. This produced 31 Sol-selected records, 145 correction-selected records, and 724 original-ABD-selected records. All source records and historical versions remain preserved; the final file records the selected source and SHA for every route.

This must not be described as a fresh Sol review of all 900 records, nor as complete reviewer agreement. The previously reported five-class and strict-supported consistency analyses apply only to the records actually checked. Those selected disagreement/control samples do not estimate the error rate across all 900 routes. The audit is model-assisted interpretation, not objective ground truth and not an observation of the answering model's latent reasoning.

Strict support rate is `supported / 300` per arm. `partially_supported`, `unsupported`, and `contradicted` retain separate meanings; “non-supported” is only a derived binary grouping and must not be renamed `unsupported`. Correctness was opened only after the final label artifact was frozen and verified.
""")

    source_after = selected_source_hashes()
    if source_after != freeze["source_input_sha256"]:
        raise ValueError("historical selected inputs changed during stats phase")
    integrity_md = f"""# Final integrity report

- Status: **PASS**
- Final audit records: 900 unique audit IDs and task IDs.
- Arm coverage: A=300, B=300, D=300.
- Selected review sources: Sol=31, correction=145, original ABD=724.
- Correction split: target=116, control=60, overlap=0.
- Sol coverage: all 31 are members of the 60 correction controls.
- Legal five-class labels: 900/900.
- Correctness first read: `{correctness_first_read_at}`, after label freeze verification.
- Accuracy reproduction: A=98/300, B=103/300, D=103/300.
- Valid answers: A=300, B=300, D=300; failures/no-answer=0 for each.
- Selected historical source hashes: unchanged before merge, after merge, and after statistics.
- Semantic reviews added in this phase: 0.
- External model/API calls in this phase: 0.
- Original predictions and historical audit artifacts modified: no.
"""
    write_text(OUT / "INTEGRITY_CHECK_REPORT.md", integrity_md)

    stats_freeze = {
        "schema_version": "abd_evidence_audit_final_statistics_freeze_v1",
        "status": "FROZEN_AND_VERIFIED",
        "created_at_utc": utc_now(),
        "label_freeze_sha256": sha256(freeze_path),
        "correctness_first_read_at_utc": correctness_first_read_at,
        "correctness_read_after_label_freeze_verification": True,
        "gold_answer_text_copied_to_outputs": False,
        "correctness_source": str(SCORE.relative_to(ROOT)),
        "correctness_source_sha256": sha256(SCORE),
        "historical_comparison_source": str(HIST.relative_to(ROOT)),
        "historical_comparison_source_sha256": sha256(HIST),
        "accuracy_reproduced": actual_correct,
        "semantic_reviews_added": 0,
        "external_model_or_api_calls": 0,
    }
    write_json(OUT / "statistics_freeze.json", stats_freeze)

    files = []
    for path in sorted(p for p in OUT.iterdir() if p.name != "final_manifest.json"):
        if path.is_file():
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": "abd_evidence_audit_final_manifest_v1",
        "status": "FROZEN_AND_VERIFIED",
        "created_at_utc": utc_now(),
        "record_count": 900,
        "source_counts": freeze["source_counts"],
        "arm_counts": freeze["arm_counts"],
        "selection_priority": freeze["selection_priority"],
        "label_freeze_sha256": sha256(freeze_path),
        "generator_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "files": files,
        "selected_historical_sources_unchanged": True,
        "original_predictions_modified": False,
        "historical_audits_modified": False,
        "semantic_reviews_added": 0,
        "external_model_or_api_calls": 0,
    }
    write_json(OUT / "final_manifest.json", manifest)
    print(json.dumps({"status": "statistics_frozen", "metrics": metrics,
                      "manifest_sha256": sha256(OUT / "final_manifest.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("merge", "stats"))
    args = parser.parse_args()
    if args.phase == "merge":
        phase_merge()
    else:
        phase_stats()


if __name__ == "__main__":
    main()
