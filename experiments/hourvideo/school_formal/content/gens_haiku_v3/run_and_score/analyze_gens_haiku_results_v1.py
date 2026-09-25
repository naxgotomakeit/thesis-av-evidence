#!/usr/bin/env python3
"""Create diagnostic sidecars without modifying raw routes, journals, or ledger."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gens_haiku_eval300.classification import classify_result  # noqa: E402
from gens_haiku_eval300.runtime import atomic_json, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    route_files = sorted((args.namespace / "routes").glob("*.json"))
    rows = []
    for path in route_files:
        route = json.loads(path.read_text(encoding="utf-8"))
        runtime_failure = route.get("terminal_status") not in {"final_answer", "invalid_answer"}
        diagnosis = classify_result(raw_response_text=route.get("raw_response_text"),
                                    parsed_prediction=route.get("prediction"), runtime_failure=runtime_failure)
        rows.append({"question_id": route["question_id"], "route_path": str(path),
                     "route_sha256": sha256_file(path), "prediction": route.get("prediction"),
                     "terminal_status": route.get("terminal_status"), **diagnosis})
    counts = Counter(row["result_class"] for row in rows)
    costs = [json.loads(p.read_text())["resource_totals"].get("cache_aware_usd", 0.0) for p in route_files]
    report = {"schema_version": "gens_haiku_result_diagnostics_v1", "source_namespace": str(args.namespace),
              "source_files_modified": False, "expected_routes": args.expected, "observed_routes": len(rows),
              "complete": len(rows) == args.expected, "counts": dict(counts),
              "rates": {key: value / args.expected for key, value in counts.items()},
              "manual_review_question_ids": [r["question_id"] for r in rows if r["manual_review_required"]],
              "route_classifications": rows, "total_answering_usd": sum(costs),
              "mean_usd_per_expected_question": sum(costs) / args.expected,
              "median_observed_route_usd": statistics.median(costs) if costs else None,
              "gold_loaded": False}
    atomic_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("complete", "counts", "total_answering_usd", "gold_loaded")}, indent=2))


if __name__ == "__main__": main()
