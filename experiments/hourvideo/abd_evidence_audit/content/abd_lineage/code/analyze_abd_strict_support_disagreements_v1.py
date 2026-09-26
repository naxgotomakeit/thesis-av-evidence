#!/usr/bin/env python3
"""No-gold offline strict-supported boundary statistics for frozen audit comparisons."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/abd_eval300_evidence_audit_v1"
VALID = ROOT / "outputs/abd_eval300_evidence_audit_correction_validation_v1"
SOL = ROOT / "outputs/abd_eval300_disagreement_independent_review_v1"
OUT = ROOT / "outputs/abd_eval300_strict_support_disagreement_stats_v1"
LABELS = {"supported", "partially_supported", "unsupported", "contradicted", "unreviewable"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def atomic(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value) -> None:
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise RuntimeError("field list required for empty CSV")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields or list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic(path, buf.getvalue())


def is_supported(label: str) -> bool:
    if label not in LABELS:
        raise RuntimeError(f"missing or illegal five-class label: {label!r}")
    return label == "supported"


def binary_stats(rows: list[dict], source: str, target: str) -> dict:
    cells = Counter((is_supported(x[source]), is_supported(x[target])) for x in rows)
    ss, sn, ns, nn = cells[(True, True)], cells[(True, False)], cells[(False, True)], cells[(False, False)]
    n = len(rows)
    return {
        "compared_records": n,
        "source_supported_target_supported": ss,
        "source_supported_target_non_supported": sn,
        "source_non_supported_target_supported": ns,
        "source_non_supported_target_non_supported": nn,
        "binary_agreement_count": ss + nn,
        "binary_agreement_proportion": (ss + nn) / n if n else None,
        "supported_to_non_supported": sn,
        "non_supported_to_supported": ns,
        "boundary_crossing_total": sn + ns,
        "supported_net_change": ns - sn,
        "strict_support_rate_net_effect_percentage_points": 100 * (ns - sn) / 300,
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refuse overwrite: {OUT}")

    audit_freeze = load(AUDIT / "audit_freeze.json")
    validation_manifest = load(VALID / "correction_validation_manifest.json")
    sol_freeze = load(SOL / "BLINDED_REVIEW_FREEZE.json")
    post_manifest = load(SOL / "POST_FREEZE_COMPARISON_MANIFEST.json")
    old_path = AUDIT / "scored_evidence_audit_blinded.jsonl"
    corr_path = VALID / "extracted/corrected_blinded_reviews.jsonl"
    map_path = VALID / "reconstructed_identity_mapping.json"
    sol_path = SOL / "blinded_reviewer_output/blinded_reviews.jsonl"
    three_path = SOL / "THREE_WAY_COMPARISON.csv"

    expected_mapping_sha = next(x["sha256"] for x in validation_manifest["files"] if x["path"] == "reconstructed_identity_mapping.json")
    checks = {
        "old_blinded_audit_sha_matches_freeze": sha(old_path) == audit_freeze["scored_evidence_audit_blinded_sha256"],
        "correction_reviews_sha_matches_validation": sha(corr_path) == validation_manifest["corrected_reviews_sha256"],
        "mapping_sha_matches_validation_manifest": sha(map_path) == expected_mapping_sha,
        "sol_reviews_sha_matches_blind_freeze": sha(sol_path) == sol_freeze["blinded_reviews_sha256"],
        "three_way_csv_sha_matches_post_freeze_manifest": sha(three_path) == post_manifest["generated_files"]["THREE_WAY_COMPARISON.csv"],
        "gold_read_flags_false": not any((audit_freeze["gold_loaded_at_freeze"], validation_manifest["gold_read"], sol_freeze["gold_read"], post_manifest["gold_read"])),
        "correctness_read_flags_false": not any((validation_manifest["correctness_read"], sol_freeze["correctness_read"], post_manifest["correctness_read"])),
    }
    if not all(checks.values()):
        raise RuntimeError(f"source integrity failure: {checks}")

    mappings = load(map_path)["rows"]
    map_by_old = {x["original_audit_id"]: x for x in mappings}
    if len(mappings) != 176 or len(map_by_old) != 176:
        raise RuntimeError("mapping is not 176 unique records")
    controls = [x for x in mappings if x["sample_role"] == "control"]
    if len(controls) != 60 or Counter(x["original_workflow"] for x in controls) != Counter({1: 20, 2: 20, 3: 20}):
        raise RuntimeError("control identity/workflow count mismatch")
    if set(x["arm"] for x in controls) - {"A", "B", "D"}:
        raise RuntimeError("invalid mapped arm")

    old_rows = {x["audit_id"]: x for x in load_jsonl(old_path)}
    corr_rows = {x["audit_id"]: x for x in load_jsonl(corr_path)}
    if len(old_rows) != 900 or len(corr_rows) != 176:
        raise RuntimeError("frozen source record count mismatch")
    rows60 = []
    for m in controls:
        old = old_rows[m["original_audit_id"]]["option_support"]
        corr = corr_rows[m["correction_audit_id"]]["option_support"]
        is_supported(old); is_supported(corr)
        rows60.append({
            "record_id": m["original_audit_id"], "question_id": m["question_id"], "arm": m["arm"],
            "original_workflow": m["original_workflow"], "original_abd_label": old,
            "correction_label": corr, "original_abd_is_supported": is_supported(old),
            "correction_is_supported": is_supported(corr),
        })

    sol_rows = {x["review_id"]: x for x in load_jsonl(sol_path)}
    private_map = {x["review_id"]: x["source_audit_id"] for x in load(SOL / "identity_mapping_private.json")["rows"]}
    three = list(csv.DictReader(three_path.open(encoding="utf-8")))
    if len(three) != 31 or len(sol_rows) != 31 or set(private_map) != set(sol_rows):
        raise RuntimeError("Sol identity/count mismatch")
    expected31 = {x["record_id"] for x in rows60 if x["original_abd_label"] != x["correction_label"]}
    if {x["original_audit_id"] for x in three} != expected31 or len(expected31) != 31:
        raise RuntimeError("31 records are not exactly the five-class disagreements")

    rows31 = []
    for x in three:
        eid = x["original_audit_id"]
        m = map_by_old[eid]
        rid = x["independent_review_id"]
        if private_map[rid] != eid:
            raise RuntimeError(f"private mapping mismatch: {eid}")
        old, corr, sol = x["original_abd_label"], x["correction_label"], x["independent_sol_label"]
        if old != old_rows[eid]["option_support"] or corr != corr_rows[eid]["option_support"] or sol != sol_rows[rid]["option_support"]:
            raise RuntimeError(f"three-way label provenance mismatch: {eid}")
        for label in (old, corr, sol): is_supported(label)
        rows31.append({
            "record_id": eid, "question_id": m["question_id"], "arm": m["arm"],
            "original_workflow": m["original_workflow"], "original_abd_label": old,
            "correction_label": corr, "independent_sol_label": sol,
            "original_abd_is_supported": is_supported(old), "correction_is_supported": is_supported(corr),
            "independent_sol_is_supported": is_supported(sol),
            "independent_sol_review_flag": str(x["independent_review_flag"]).lower() == "true",
        })

    comparisons = [
        ("controls60_original_abd_vs_correction", rows60, "original_abd_label", "correction_label"),
        ("disagreements31_original_abd_vs_sol", rows31, "original_abd_label", "independent_sol_label"),
        ("disagreements31_correction_vs_sol", rows31, "correction_label", "independent_sol_label"),
    ]
    overall, by_arm = [], []
    for name, records, source, target in comparisons:
        overall.append({"comparison": name, "source": source, "target": target, **binary_stats(records, source, target)})
        for arm in ("A", "B", "D"):
            subset = [x for x in records if x["arm"] == arm]
            by_arm.append({"comparison": name, "arm": arm, "source": source, "target": target, **binary_stats(subset, source, target)})

    def always_non(x):
        return not any((x["original_abd_is_supported"], x["correction_is_supported"], x["independent_sol_is_supported"]))
    def crosses(x):
        return len({x["original_abd_is_supported"], x["correction_is_supported"], x["independent_sol_is_supported"]}) > 1
    all_different = [x for x in rows31 if len({x["original_abd_label"], x["correction_label"], x["independent_sol_label"]}) == 3]
    flagged = [x for x in rows31 if x["independent_sol_review_flag"]]
    subset_rows = []
    for name, subset in (("all_31_five_class_disagreements", rows31), ("three_way_all_labels_different_10", all_different), ("sol_review_flag_true", flagged)):
        subset_rows.append({
            "subset": name, "records": len(subset),
            "always_non_supported_all_three": sum(always_non(x) for x in subset),
            "at_least_one_supported_boundary_crossing": sum(crosses(x) for x in subset),
        })
    crossing = [x for x in rows31 if crosses(x)]

    OUT.mkdir()
    write_csv(OUT / "controls60_five_class_and_binary.csv", rows60)
    write_csv(OUT / "disagreements31_five_class_and_binary.csv", rows31)
    write_csv(OUT / "pairwise_binary_overall.csv", overall)
    write_csv(OUT / "pairwise_binary_by_arm.csv", by_arm)
    write_csv(OUT / "subset_boundary_summary.csv", subset_rows)
    write_csv(OUT / "boundary_crossing_records_31.csv", crossing)

    pp = lambda n: f"{n * 100 / 300:+.3f}"
    report = [
        "# ABD strict-support disagreement statistics", "",
        "Strict support is derived only as `label == supported`; all original five-class labels are preserved.",
        "No label was merged or replaced. All percentage-point effects use the fixed arm denominator 300 and",
        "mean only: if labels in this comparison subset were substituted while every other record stayed fixed.", "",
        "## Direct answers", "",
        f"- Of the 31 five-class disagreements, {sum(always_non(x) for x in rows31)} remain non-supported in all three rounds and never cross the strict-supported boundary.",
        f"- {len(crossing)} records cross the supported boundary at least once.",
        f"- Of the 10 records with three different five-class labels, {sum(always_non(x) for x in all_different)} remains non-supported in all three rounds and {sum(crosses(x) for x in all_different)} cross the boundary.",
        f"- Of the {len(flagged)} Sol review-flag records, {sum(always_non(x) for x in flagged)} always remain non-supported and {sum(crosses(x) for x in flagged)} cross the boundary. A review flag is not itself a boundary change.", "",
        "## Overall 2x2 comparisons", "",
        "| comparison | n | S→S | S→non-S | non-S→S | non-S→non-S | binary agreement | net supported | net pp/300 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in overall:
        report.append(f"| {x['comparison']} | {x['compared_records']} | {x['source_supported_target_supported']} | {x['source_supported_target_non_supported']} | {x['source_non_supported_target_supported']} | {x['source_non_supported_target_non_supported']} | {x['binary_agreement_count']} ({100*x['binary_agreement_proportion']:.1f}%) | {x['supported_net_change']:+d} | {x['strict_support_rate_net_effect_percentage_points']:+.3f} |")
    report += ["", "## Arm-specific hypothetical effects", "", "| comparison | arm | reviewed | S→non-S | non-S→S | crossings | net supported | net pp/300 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in by_arm:
        report.append(f"| {x['comparison']} | {x['arm']} | {x['compared_records']} | {x['supported_to_non_supported']} | {x['non_supported_to_supported']} | {x['boundary_crossing_total']} | {x['supported_net_change']:+d} | {x['strict_support_rate_net_effect_percentage_points']:+.3f} |")
    report += ["", "## Boundary-crossing records among the 31", "", "| ID | question_id | arm | original ABD | correction | Sol | ABD S? | correction S? | Sol S? | Sol review flag |", "|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for x in crossing:
        report.append(f"| {x['record_id']} | {x['question_id']} | {x['arm']} | {x['original_abd_label']} | {x['correction_label']} | {x['independent_sol_label']} | {int(x['original_abd_is_supported'])} | {int(x['correction_is_supported'])} | {int(x['independent_sol_is_supported'])} | {int(x['independent_sol_review_flag'])} |")
    report += ["", "## Interpretation boundary", "", f"The 31 records were selected because their first two five-class labels disagreed. Within them, {sum(always_non(x) for x in rows31)}/31 disagreements are internal to the non-supported categories, while {len(crossing)}/31 cross strict support at least once. This describes the selected subset only and is not a 900-record error estimate or a validation of the 116 correction targets."]
    atomic(OUT / "STRICT_SUPPORT_DISAGREEMENT_REPORT.md", "\n".join(report) + "\n")

    exceptional = {}
    for name, records, source, target in comparisons:
        exceptional[name] = {
            source: Counter(x[source] for x in records).get("unreviewable", 0),
            target: Counter(x[target] for x in records).get("unreviewable", 0),
            "missing_labels": sum(not x.get(source) or not x.get(target) for x in records),
            "no_final_answer": 0,
        }
    integrity = {
        "schema_version": "abd_strict_support_disagreement_integrity_v1",
        "source_checks": checks,
        "control_records": len(rows60), "control_arm_counts": dict(Counter(x["arm"] for x in rows60)),
        "independent_records": len(rows31), "independent_arm_counts": dict(Counter(x["arm"] for x in rows31)),
        "exceptional_labels": exceptional,
        "workflows_not_treated_as_arms": True,
        "gold_read": False, "correctness_read": False, "labels_modified_or_merged": False,
        "models_or_external_api_calls": 0, "question_answering_reruns": 0,
    }
    write_json(OUT / "integrity.json", integrity)

    generated = sorted(p for p in OUT.iterdir() if p.is_file())
    manifest = {
        "schema_version": "abd_strict_support_disagreement_stats_manifest_v1",
        "files": {p.name: {"sha256": sha(p), "size_bytes": p.stat().st_size} for p in generated},
        "source_sha256": {
            "old_blinded_audit": sha(old_path), "correction_reviews": sha(corr_path),
            "identity_mapping": sha(map_path), "sol_blinded_reviews": sha(sol_path),
            "three_way_comparison": sha(three_path),
        },
        "gold_read": False, "correctness_read": False, "labels_modified_or_merged": False,
        "models_or_external_api_calls": 0,
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"checks": checks, "overall": overall, "by_arm": by_arm, "subsets": subset_rows, "crossing_ids": [x["record_id"] for x in crossing]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
