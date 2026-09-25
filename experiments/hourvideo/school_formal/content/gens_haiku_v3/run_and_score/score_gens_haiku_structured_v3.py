#!/usr/bin/env python3
"""Independent gold scoring after a frozen PASS v3 structural closure."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gens_haiku_eval300.runtime import atomic_json, canonical_sha, read_jsonl, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--structural-validation", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    structural = json.loads(args.structural_validation.read_text(encoding="utf-8"))
    if structural.get("status") != "PASS" or structural.get("gold_loaded") is not False:
        raise RuntimeError("requires frozen PASS no-gold closure")
    inventory = [
        {"path": row["path"], "sha256": sha256_file(Path(row["path"])), "bytes": Path(row["path"]).stat().st_size}
        for row in structural["raw_file_inventory"]
    ]
    if canonical_sha(inventory) != structural["raw_closure_sha256"]:
        raise RuntimeError("v3 raw closure changed before scoring")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    qids = manifest["question_ids"]
    annotations = json.loads(args.gold.read_text(encoding="utf-8"))
    gold = {question["qid"]: question["correct_answer_label"] for video in annotations.values() for question in video["benchmark_dataset"]}
    if not set(qids).issubset(gold):
        raise RuntimeError("gold identity coverage mismatch")
    rows = []
    for qid in qids:
        route = json.loads((args.namespace / "routes" / f"{qid}.json").read_text(encoding="utf-8"))
        prediction = route.get("prediction")
        rows.append({
            "question_id": qid,
            "prediction": prediction,
            "result_class": route["result_class"],
            "gold": gold[qid],
            "correct": prediction is not None and prediction == gold[qid],
        })
    counts = Counter(row["result_class"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    attempts = read_jsonl(args.namespace / "journals/attempt_end.jsonl")
    token_fields = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
    cost_fields = ("ordinary_input_usd", "cache_creation_usd", "cache_read_usd", "output_usd", "cache_aware_usd")
    resources = {
        field: sum(float(row.get(field, 0)) if field.endswith("usd") else int(row.get(field, 0)) for row in attempts)
        for field in (*token_fields, *cost_fields)
    }
    resources.update({
        "images": sum(json.loads((args.namespace / "routes" / f"{qid}.json").read_text())["selected_frame_count"] for qid in qids),
        "provider_attempts": len(attempts),
        "transport_retries": len(attempts) - 300,
        "total_api_latency_sec_all_300": structural["latency"]["total_api_latency_sec_all_300"],
        "observed_experiment_wall_clock_sec": structural["latency"]["observed_experiment_wall_clock_sec"],
    })
    valid = counts["valid_answer"]
    report = {
        "schema_version": "gens_haiku_structured_answer_v3_score",
        "population": 300,
        "correct": correct,
        "accuracy": correct / 300,
        "valid_answer": valid,
        "valid_answer_rate": valid / 300,
        "invalid_format": counts["invalid_format"],
        "invalid_format_rate": counts["invalid_format"] / 300,
        "runtime_failure": counts["runtime_failure"],
        "runtime_failure_rate": counts["runtime_failure"] / 300,
        "accuracy_among_valid_answers": correct / valid if valid else None,
        "accuracy_among_valid_answers_denominator": valid,
        "resource_summary": resources,
        "latency_denominator": "all 300 formal routes",
        "raw_closure_sha256": structural["raw_closure_sha256"],
        "gold_source_path": str(args.gold),
        "gold_source_sha256": sha256_file(args.gold),
        "raw_results_modified": False,
        "per_question": rows,
    }
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
