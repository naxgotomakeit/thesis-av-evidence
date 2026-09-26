#!/usr/bin/env python3
"""Link the frozen neutral audit to existing scores and emit derived summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def dump_json(path: Path, value: object) -> None:
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_root", type=Path)
    args = ap.parse_args()
    root = args.audit_root.resolve()
    freeze = json.loads((root / "evidence_audit_freeze.json").read_text())
    frozen_path = root / "evidence_audit_frozen.jsonl"
    if sha(frozen_path) != freeze["evidence_audit_sha256"]:
        raise SystemExit("frozen audit SHA mismatch")
    audit = load_jsonl(frozen_path)
    mapping = json.loads((root / "identity_mapping.json").read_text())["rows"]
    if [x["neutral_id"] for x in audit] != [x["neutral_id"] for x in mapping]:
        raise SystemExit("identity mapping order mismatch")

    ws = Path(__file__).resolve().parents[1]
    direct_csv = ws / "outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/canonical_summary_v1/route_level_canonical.csv"
    direct_accuracy_path = direct_csv.parent / "accuracy.json"
    direct_manifest_path = direct_csv.parent / "canonical_manifest.json"
    v3_score_path = ws / "outputs/gens_haiku_structured_v3/formal_v3_direct_parser_aligned_evaluation/score.json"
    v3_structural_path = v3_score_path.parent / "structural_validation_no_gold.json"
    v2_score_path = ws / "outputs/gens_haiku_structured_v2/formal_v2_evaluation/score.json"
    v2_integrity_path = ws / "outputs/gens_haiku_structured_v2/final_integrity_check.json"

    with direct_csv.open(encoding="utf-8", newline="") as handle:
        direct_rows = list(csv.DictReader(handle))
    direct = {(row["method"], row["question_id"]): row for row in direct_rows}
    v3_score = json.loads(v3_score_path.read_text())
    v3 = {row["question_id"]: row for row in v3_score["per_question"]}
    v2_score = json.loads(v2_score_path.read_text())
    v2 = {row["question_id"]: row for row in v2_score["per_question"]}
    canonical_order = {row["question_id"]: index for index, row in enumerate(v3_score["per_question"])}

    linked: list[dict] = []
    for tag, ident in zip(audit, mapping):
        group = ident["source_group"]
        qid = ident["question_id"]
        if group in {"R1", "R3"}:
            score = direct[(group, qid)]
            prediction = score["prediction"] or None
            correct = score["correct"].lower() == "true"
            completion = score["prediction_present"].lower() == "true"
        elif group == "GENS":
            score = v3[qid]
            prediction = score["prediction"]
            correct = bool(score["correct"])
            completion = prediction is not None
        else:
            raise SystemExit(f"unknown group {group}")
        row = {
            **tag,
            "source_group": group,
            "question_id": qid,
            "prediction": prediction,
            "prediction_present": completion,
            "correct": correct,
            "canonical_question_index": canonical_order[qid],
        }
        package = json.loads((root / "review_packages" / f"{tag['neutral_id']}.json").read_text())
        if group in {"R1", "R3"}:
            row["input_evidence_channels_available"] = "map_and_images" if package["images"] else "map"
        else:
            row["input_evidence_channels_available"] = "images"
        linked.append(row)

    if len(linked) != 900 or len({(r["source_group"], r["question_id"]) for r in linked}) != 900:
        raise SystemExit("linked population mismatch")
    linked_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in linked
    )
    linked_path = root / "scored_evidence_audit.jsonl"
    atomic(linked_path, linked_bytes)

    method_stats: dict[str, dict] = {}
    for group in ["R1", "R3", "GENS"]:
        rows = [row for row in linked if row["source_group"] == group]
        support = Counter(row["option_support"] for row in rows)
        cross: dict[str, dict] = {}
        for category in ["supported", "partially_supported", "unsupported", "contradicted", "unreviewable", "no_final_answer"]:
            subset = [row for row in rows if row["option_support"] == category]
            cross[category] = {
                "total": len(subset),
                "correct": sum(row["correct"] for row in subset),
                "incorrect": sum(not row["correct"] for row in subset),
            }
        completed = sum(row["prediction_present"] for row in rows)
        correct = sum(row["correct"] for row in rows)
        weak_correct = [r for r in rows if r["correct"] and r["option_support"] in {"unsupported", "contradicted"}]
        supported_wrong = [r for r in rows if not r["correct"] and r["option_support"] == "supported"]
        method_stats[group] = {
            "population": 300,
            "predictions": completed,
            "completion_rate": completed / 300,
            "correct": correct,
            "accuracy": correct / 300,
            "support_counts": dict(support),
            "support_by_correctness": cross,
            "correct_but_unsupported_or_contradicted": {
                "count": len(weak_correct),
                "share_of_all_300": len(weak_correct) / 300,
                "share_of_correct": len(weak_correct) / correct if correct else None,
            },
            "supported_but_incorrect": len(supported_wrong),
            "needs_further_review": sum(r["needs_further_review"] for r in rows),
            "reason_basis_counts": dict(Counter(label for r in rows for label in r["reason_basis"])),
        }
        if group in {"R1", "R3"}:
            method_stats[group]["direct_evidence_source_counts_as_recorded"] = dict(Counter(r["direct_evidence_source"] for r in rows))
            method_stats[group]["input_evidence_channels_available"] = dict(Counter(r["input_evidence_channels_available"] for r in rows))
            method_stats[group]["direct_evidence_source_by_support"] = {
                source: dict(Counter(r["option_support"] for r in rows if r["direct_evidence_source"] == source))
                for source in ["map", "images", "map_and_images", "none"]
            }
            method_stats[group]["source_label_inconsistency_count"] = sum(
                r["direct_evidence_source"] == "not_applicable" for r in rows
            )

    v2_abstain = [qid for qid, row in v2.items() if row["result_class"] == "abstention"]
    v2_answered = [qid for qid, row in v2.items() if row["result_class"] == "valid_answer"]
    transitions = Counter()
    for qid in v2_answered:
        before = "correct" if v2[qid]["correct"] else "wrong"
        after = "correct" if v3[qid]["correct"] else "wrong"
        transitions[f"{before}_to_{after}"] += 1
    abstain_outcomes = Counter("correct" if v3[qid]["correct"] else "wrong" for qid in v2_abstain)
    gens_audit = {r["question_id"]: r for r in linked if r["source_group"] == "GENS"}
    newly_correct = [qid for qid in v3 if v3[qid]["correct"] and not v2[qid]["correct"]]
    lost_correct = [qid for qid in v3 if not v3[qid]["correct"] and v2[qid]["correct"]]
    v2v3 = {
        "v2_abstentions": len(v2_abstain),
        "v2_abstention_to_v3": dict(abstain_outcomes),
        "v2_answered": len(v2_answered),
        "v2_answered_transitions": dict(transitions),
        "v2_correct": sum(r["correct"] for r in v2.values()),
        "v3_correct": sum(r["correct"] for r in v3.values()),
        "net_change": sum(r["correct"] for r in v3.values()) - sum(r["correct"] for r in v2.values()),
        "newly_correct_count": len(newly_correct),
        "lost_correct_count": len(lost_correct),
        "newly_correct_v3_evidence_support": dict(Counter(gens_audit[q]["option_support"] for q in newly_correct)),
        "newly_correct_question_ids": newly_correct,
        "lost_correct_question_ids": lost_correct,
        "interpretation_warning": "v2→v3 changed prompt, interface, and forced-choice policy; this is not a parser-only effect.",
    }

    examples: dict[str, dict[str, list[dict]]] = {}
    categories = ["supported", "partially_supported", "unsupported", "contradicted", "unreviewable", "no_final_answer"]
    for group in ["R1", "R3", "GENS"]:
        examples[group] = {}
        rows = sorted((r for r in linked if r["source_group"] == group), key=lambda r: r["canonical_question_index"])
        for category in categories:
            examples[group][category] = [
                {k: row[k] for k in ["neutral_id", "question_id", "prediction", "correct", "audit_note", "evidence_refs"]}
                for row in rows if row["option_support"] == category
            ][:2]

    direct_manifest = json.loads(direct_manifest_path.read_text())
    v3_structural = json.loads(v3_structural_path.read_text())
    v2_integrity = json.loads(v2_integrity_path.read_text())
    direct_ns = direct_csv.parents[1]
    direct_raw_files = [
        direct_ns / "formal_manifest_final_candidate_no_api_v3.json",
        direct_ns / "formal_launch_lock_v3.json",
        direct_ns / "budget_ledger.json",
        *sorted((direct_ns / "inputs").glob("*.json")),
        *sorted((direct_ns / "journals").glob("*.jsonl")),
        *sorted((direct_ns / "route_status").glob("*.json")),
        *sorted((direct_ns / "route_artifacts").glob("*.json")),
    ]
    direct_inventory = [
        {"path": str(path.relative_to(direct_ns)), "sha256": sha(path), "bytes": path.stat().st_size}
        for path in direct_raw_files
    ]
    direct_current_closure = hashlib.sha256(
        json.dumps(direct_inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw_integrity = {
        "direct": {
            "recorded_raw_closure_sha256": direct_manifest["raw_closure_sha256_after"],
            "current_raw_closure_sha256": direct_current_closure,
            "current_raw_closure_matches": direct_current_closure == direct_manifest["raw_closure_sha256_after"],
            "recorded_raw_unchanged": direct_manifest["raw_closure_unchanged"],
            "raw_file_count": len(direct_raw_files),
            "canonical_manifest_sha256_now": sha(direct_manifest_path),
        },
        "gens_v3": {
            "recorded_raw_closure_sha256": v3_structural["raw_closure_sha256"],
            "score_raw_closure_sha256": v3_score["raw_closure_sha256"],
            "inventory_files_rehashed": len(v3_structural["raw_file_inventory"]),
            "all_inventory_file_hashes_unchanged": all(
                Path(item["path"]).exists() and sha(Path(item["path"])) == item["sha256"]
                for item in v3_structural["raw_file_inventory"]
            ),
        },
        "gens_v2": {
            "recorded_raw_closure_sha256": v2_score["raw_closure_sha256"],
            "integrity_check_unchanged": v2_integrity["v2_raw_closure_unchanged"],
        },
        "analysis_writes_confined_to_audit_root": True,
    }

    summary = {
        "audit_label": "Codex-assisted evidence audit / Codex辅助证据审计",
        "frozen_audit_sha256": freeze["evidence_audit_sha256"],
        "scored_audit_sha256": sha(linked_path),
        "population": 900,
        "methods": method_stats,
        "operational_definition": {
            "evidence_insufficient_for_correct_answer_metric": ["unsupported", "contradicted"],
            "partially_supported_reported_separately": True,
        },
        "score_sources": {
            "direct_route_csv": str(direct_csv),
            "direct_route_csv_sha256": sha(direct_csv),
            "gens_v3_score": str(v3_score_path),
            "gens_v3_score_sha256": sha(v3_score_path),
            "gens_v2_score": str(v2_score_path),
            "gens_v2_score_sha256": sha(v2_score_path),
        },
        "review_integrity_note": (
            "All 900 batch records were written and progress validation reported 900/900 before score linkage. "
            "A broad file search then unintentionally printed some existing per-question gold fields before the "
            "combined audit SHA was written; no review file was changed after that exposure. Batch-file hashes in "
            "evidence_audit_freeze.json preserve the completed pre-linkage records."
        ),
        "source_label_note": (
            "The frozen direct_evidence_source field was not used consistently in early review batches. "
            "It is reported exactly as recorded and is not post-score repaired. input_evidence_channels_available "
            "is a separate objective field derived from the frozen package (Direct always has map; image count "
            "determines map versus map_and_images availability) and must not be interpreted as causal reliance."
        ),
    }
    dump_json(root / "evidence_score_cross_stats.json", summary)
    dump_json(root / "gens_v2_to_v3_transition.json", v2v3)
    dump_json(root / "fixed_examples.json", examples)
    dump_json(root / "raw_integrity.json", raw_integrity)

    csv_path = root / "unified_summary.csv"
    fields = ["method", "population", "predictions", "correct", "accuracy", "supported", "partially_supported", "unsupported", "contradicted", "unreviewable", "no_final_answer", "correct_but_unsupported_or_contradicted", "supported_but_incorrect", "needs_further_review"]
    lines: list[str] = []
    import io
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for group in ["R1", "R3", "GENS"]:
        stat = method_stats[group]
        writer.writerow({
            "method": group,
            "population": 300,
            "predictions": stat["predictions"],
            "correct": stat["correct"],
            "accuracy": stat["accuracy"],
            **{category: stat["support_counts"].get(category, 0) for category in categories},
            "correct_but_unsupported_or_contradicted": stat["correct_but_unsupported_or_contradicted"]["count"],
            "supported_but_incorrect": stat["supported_but_incorrect"],
            "needs_further_review": stat["needs_further_review"],
        })
    atomic(csv_path, out.getvalue().encode())

    freeze_out = {
        "frozen_audit_sha256": freeze["evidence_audit_sha256"],
        "scored_evidence_audit_sha256": sha(linked_path),
        "cross_stats_sha256": sha(root / "evidence_score_cross_stats.json"),
        "v2_to_v3_sha256": sha(root / "gens_v2_to_v3_transition.json"),
        "examples_sha256": sha(root / "fixed_examples.json"),
        "unified_summary_sha256": sha(csv_path),
        "raw_integrity_sha256": sha(root / "raw_integrity.json"),
    }
    report_path = root / "EVIDENCE_AUDIT_REPORT.md"
    if report_path.exists():
        freeze_out["report_sha256"] = sha(report_path)
    gold_validation_path = root / "independent_gold_validation.json"
    if gold_validation_path.exists():
        freeze_out["independent_gold_validation_sha256"] = sha(gold_validation_path)
    dump_json(root / "scored_analysis_freeze.json", freeze_out)
    print(json.dumps({"methods": method_stats, "v2_to_v3": v2v3, "raw_integrity": raw_integrity, "freeze": freeze_out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
