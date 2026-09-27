#!/usr/bin/env python3
"""Post-hoc gold scoring for the structurally frozen recovered Eval300.

This script never edits source route artifacts. Runtime failures and invalid
predictions remain incorrect observations in the fixed denominator of 300.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--structural-summary", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    merged_sha_before = sha256_file(args.merged)
    structural_sha_before = sha256_file(args.structural_summary)
    merged = json.loads(args.merged.read_text(encoding="utf-8"))
    structural = json.loads(args.structural_summary.read_text(encoding="utf-8"))
    if structural.get("gold_loaded") is not False or structural.get("main_denominator") != 300:
        raise RuntimeError("requires frozen no-gold structural merge with denominator 300")
    if merged.get("summary") != structural:
        raise RuntimeError("embedded and standalone structural summaries differ")

    rows = merged.get("results", [])
    by_method = {method: [row for row in rows if row.get("method") == method] for method in ("R1", "R3")}
    for method, group in by_method.items():
        qids = [row["question_id"] for row in group]
        if len(group) != 300 or len(set(qids)) != 300:
            raise RuntimeError(f"{method} is not a unique 300-question population")
    if {row["question_id"] for row in by_method["R1"]} != {row["question_id"] for row in by_method["R3"]}:
        raise RuntimeError("R1/R3 question pairing changed")

    annotations = json.loads(args.gold.read_text(encoding="utf-8"))
    gold: dict[str, str] = {}
    for video in annotations.values():
        for question in video["benchmark_dataset"]:
            qid = question["qid"]
            if qid in gold:
                raise RuntimeError(f"duplicate gold qid: {qid}")
            gold[qid] = question["correct_answer_label"]
    all_qids = {row["question_id"] for row in rows}
    if not all_qids.issubset(gold):
        missing = sorted(all_qids - set(gold))
        raise RuntimeError(f"gold does not cover merged population: {missing[:3]}")

    scored = []
    for row in rows:
        prediction = row.get("final_prediction")
        valid = row.get("terminal_status") == "final_answer" and prediction in {"A", "B", "C", "D", "E"}
        item = dict(row)
        item.update(
            {
                "gold": gold[row["question_id"]],
                "valid_answer": valid,
                "correct": bool(valid and prediction == gold[row["question_id"]]),
            }
        )
        scored.append(item)

    method_summaries = {}
    scored_by_method = {method: [row for row in scored if row["method"] == method] for method in ("R1", "R3")}
    for method, group in scored_by_method.items():
        correct = sum(row["correct"] for row in group)
        valid = sum(row["valid_answer"] for row in group)
        failures = Counter(row["terminal_status"] for row in group if not row["valid_answer"])
        route_cost = sum(float(row.get("known_settled_route_cost_usd") or 0) for row in group)
        artifacts = [json.loads(Path(row["source_artifact"]).read_text(encoding="utf-8")) for row in group]
        resources = {
            "provider_attempts_with_known_responses": sum(len(artifact.get("provider_attempts", [])) for artifact in artifacts),
            "unique_images_transmitted_route_sum": sum(int(artifact.get("unique_images_transmitted") or 0) for artifact in artifacts),
            "input_tokens": sum(int(artifact.get("total_input_tokens") or 0) for artifact in artifacts),
            "output_tokens": sum(int(artifact.get("total_output_tokens") or 0) for artifact in artifacts),
            "cache_creation_input_tokens": sum(int(artifact.get("total_cache_creation_input_tokens") or 0) for artifact in artifacts),
            "cache_read_input_tokens": sum(int(artifact.get("total_cache_read_input_tokens") or 0) for artifact in artifacts),
            "modeled_api_latency_sec": sum(float(artifact.get("total_modeled_api_latency_sec") or 0) for artifact in artifacts),
            "route_wall_time_sec_sum": sum(float(artifact.get("route_wall_time_sec") or 0) for artifact in artifacts),
            "note": "Unknown network-result attempts have no fabricated token, image, latency, or usage values and are represented only by the separate reservation fields.",
        }
        method_summaries[method] = {
            "population": 300,
            "correct": correct,
            "accuracy_fixed_denominator_300": correct / 300,
            "valid_answers": valid,
            "completion_rate": valid / 300,
            "invalid_or_failed": 300 - valid,
            "accuracy_among_valid_answers": correct / valid if valid else None,
            "accuracy_among_valid_answers_denominator": valid,
            "failure_statuses": dict(sorted(failures.items())),
            "known_settled_route_cost_usd": route_cost,
            "resources_with_known_responses": resources,
        }

    r1_by_qid = {row["question_id"]: row for row in scored_by_method["R1"]}
    r3_by_qid = {row["question_id"]: row for row in scored_by_method["R3"]}
    paired = Counter()
    for qid in sorted(r1_by_qid):
        a, b = r1_by_qid[qid]["correct"], r3_by_qid[qid]["correct"]
        paired["both_correct" if a and b else "r1_only" if a else "r3_only" if b else "neither_correct"] += 1
    paired.update(
        {
            "population": 300,
            "r1_minus_r3_correct": method_summaries["R1"]["correct"] - method_summaries["R3"]["correct"],
            "r1_minus_r3_accuracy_percentage_points": 100
            * (method_summaries["R1"]["accuracy_fixed_denominator_300"] - method_summaries["R3"]["accuracy_fixed_denominator_300"]),
        }
    )

    cost = {
        "r1_eval300_known_settled_route_cost_usd": method_summaries["R1"]["known_settled_route_cost_usd"],
        "r3_eval300_known_settled_route_cost_usd": method_summaries["R3"]["known_settled_route_cost_usd"],
        "corrected_175_known_settled_route_cost_usd": structural["cost"]["corrected_175_known_route_cost_usd"],
        "correction_execution_ledger_settled_usd": structural["cost"]["aggregate_ledger_settled_usd"],
        "formal_unknown_reserved_usd": structural["cost"]["formal_unknown_reserved_usd"],
        "all_unknown_reserved_usd": structural["cost"]["aggregate_unknown_reserved_usd"],
        "unknown_reservation_is_not_actual_spend": True,
        "note": "Execution-ledger settled includes historical settled smoke cost; route costs describe frozen method results. Unknown reservations are disclosed separately.",
    }
    report = {
        "schema_version": "r1_visual_only_recovered_eval300_gold_score_v1",
        "scoring_protocol": {
            "main_denominator_per_method": 300,
            "failures_retained_as_incorrect": True,
            "gold_loaded_only_after_structural_pass": True,
            "source_route_artifacts_modified": False,
        },
        "source_identity": {
            "merged_path": str(args.merged.resolve()),
            "merged_sha256": merged_sha_before,
            "structural_summary_path": str(args.structural_summary.resolve()),
            "structural_summary_sha256": structural_sha_before,
            "gold_path": str(args.gold.resolve()),
            "gold_sha256": sha256_file(args.gold),
        },
        "methods": method_summaries,
        "paired_comparison": dict(paired),
        "cost": cost,
        "per_question": scored,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "scored_results_with_gold.json", report)
    summary = dict(report)
    summary.pop("per_question")
    atomic_json(args.output_dir / "accuracy_summary.json", summary)
    with (args.output_dir / "accuracy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "population", "correct", "accuracy", "valid_answers", "completion_rate", "invalid_or_failed", "known_settled_route_cost_usd"])
        for method in ("R1", "R3"):
            value = method_summaries[method]
            writer.writerow([method, value["population"], value["correct"], value["accuracy_fixed_denominator_300"], value["valid_answers"], value["completion_rate"], value["invalid_or_failed"], value["known_settled_route_cost_usd"]])

    if sha256_file(args.merged) != merged_sha_before or sha256_file(args.structural_summary) != structural_sha_before:
        raise RuntimeError("source merge changed during scoring")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
