#!/usr/bin/env python3
"""Freeze and structurally validate completed ABD results without loading gold."""
from __future__ import annotations

import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.formal_runtime import AbdFormalConfig, verify_manifest  # noqa: E402
from abd_draft_v1.store import atomic_json, read_jsonl  # noqa: E402
from gens_haiku_eval300.runtime import canonical_sha, sha256_file  # noqa: E402


CONFIG = ROOT / "config/abd_formal_candidate_v1.json"
MANIFEST = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/formal_candidate_manifest.json"
RUN = ROOT / "outputs/abd_eval300_formal_v1"
OUT = ROOT / "outputs/abd_eval300_analysis_v1"
EXPECTED_MANIFEST_SHA = "24e6305d798378c352099ee963b00efb54d5b40b545c34221949dfba36cd36dc"


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    values = {str(row["task_id"]): row for row in rows}
    if len(values) != len(rows):
        raise RuntimeError(f"duplicate {label} task identity")
    return values


def main() -> None:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("ABD scientific manifest identity changed")
    config = AbdFormalConfig.load(CONFIG)
    manifest = verify_manifest(ROOT, CONFIG, MANIFEST)
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    starts = read_jsonl(RUN / "journals/request_starts.jsonl")
    responses = read_jsonl(RUN / "journals/provider_responses.jsonl")
    ends = read_jsonl(RUN / "journals/attempt_ends.jsonl")
    parsers = read_jsonl(RUN / "journals/parser_results.jsonl")
    collections = {
        "request_start": keyed(starts, "request-start"),
        "provider_response": keyed(responses, "provider-response"),
        "attempt_end": keyed(ends, "attempt-end"),
        "parser_result": keyed(parsers, "parser-result"),
    }
    if any(set(values) != set(tasks) for values in collections.values()):
        raise RuntimeError("journal task population differs from frozen 900 tasks")
    if any(len(values) != 900 for values in collections.values()):
        raise RuntimeError("journal count differs from 900")

    status_paths = sorted((RUN / "task_status").glob("*.json"))
    artifact_paths = sorted((RUN / "task_artifacts").glob("*.json"))
    statuses = keyed([json.loads(path.read_text(encoding="utf-8")) for path in status_paths], "status")
    artifacts = keyed([json.loads(path.read_text(encoding="utf-8")) for path in artifact_paths], "artifact")
    if set(statuses) != set(tasks) or set(artifacts) != set(tasks):
        raise RuntimeError("status/artifact task population differs from frozen tasks")

    for task_id, task in tasks.items():
        start = collections["request_start"][task_id]
        response = collections["provider_response"][task_id]
        end = collections["attempt_end"][task_id]
        parser = collections["parser_result"][task_id]
        status = statuses[task_id]
        artifact = artifacts[task_id]
        if start["input_row_sha256"] != task["input_row_sha256"] or start["variant"] != task["variant"]:
            raise RuntimeError(f"input identity mismatch: {task_id}")
        attempt_ids = {row["attempt_id"] for row in (start, response, end, parser, status, artifact)}
        if len(attempt_ids) != 1:
            raise RuntimeError(f"attempt identity mismatch: {task_id}")
        if response["response_sha256"] != end["response_sha256"] or artifact["response_sha256"] != end["response_sha256"]:
            raise RuntimeError(f"response identity mismatch: {task_id}")
        if end.get("authoritative_usage_available") is not True or end.get("cache_aware_usd") is None:
            raise RuntimeError(f"missing authoritative usage/cost: {task_id}")
        if end.get("response_model") != config.model:
            raise RuntimeError(f"actual response model mismatch: {task_id}")
        if status.get("state") not in {"terminal_success", "terminal_failed"}:
            raise RuntimeError(f"non-terminal task: {task_id}")
        if artifact.get("input_row_sha256") not in {None, task["input_row_sha256"]}:
            raise RuntimeError(f"artifact input identity mismatch: {task_id}")

    ledger = json.loads((RUN / "budget_ledger.json").read_text(encoding="utf-8"))
    attempt_ids = {row["attempt_id"] for row in starts}
    if set(ledger["charges"]) != attempt_ids or ledger["unresolved_request_reservations"]:
        raise RuntimeError("ledger attempts or reservations do not close")
    charge_sum = sum((Decimal(str(value)) for value in ledger["charges"].values()), Decimal("0"))
    if charge_sum != Decimal(str(ledger["spent_usd"])):
        raise RuntimeError("ledger charges do not sum to spent")
    if charge_sum > Decimal("30"):
        raise RuntimeError("ABD total budget exceeded")

    first_three = [task["task_id"] for task in manifest["tasks"][:3]]
    if [row["task_id"] for row in starts[:3]] != first_three:
        raise RuntimeError("accepted first batch order changed")
    if sum(row.get("continuation_id") is None for row in starts) != 3:
        raise RuntimeError("first-batch request accounting changed")
    if sum(row.get("continuation_id") == "abd_eval300_remaining_897_v1" for row in starts) != 897:
        raise RuntimeError("continuation request accounting differs from 897")

    raw_paths = [
        RUN / "budget_ledger.json",
        *(sorted((RUN / "journals").glob("*.jsonl"))),
        *artifact_paths,
        *status_paths,
        *(sorted((RUN / "batch_controls").glob("*.json"))),
        *(sorted((RUN / "continuation_controls").glob("*.json"))),
    ]
    inventory = [{
        "path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size,
    } for path in raw_paths]
    closure_sha = canonical_sha(inventory)
    result_counts = Counter(end["result_class"] for end in ends)
    failure_counts = Counter(end.get("failure_category") or "none" for end in ends)
    arm_counts = {
        variant: {
            "tasks": sum(task["variant"] == variant for task in manifest["tasks"]),
            "request_starts": sum(row["variant"] == variant for row in starts),
            "responses": sum(row["variant"] == variant for row in responses),
            "terminal_success": sum(statuses[task_id]["state"] == "terminal_success" for task_id, task in tasks.items() if task["variant"] == variant),
            "terminal_failed": sum(statuses[task_id]["state"] == "terminal_failed" for task_id, task in tasks.items() if task["variant"] == variant),
        }
        for variant in "ABD"
    }
    report = {
        "schema_version": "abd_eval300_raw_structural_closure_v1",
        "status": "PASS",
        "gold_loaded": False,
        "candidate_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "population": {"tasks": 900, "questions": 300, "arms": arm_counts},
        "journal_counts": {name: len(values) for name, values in collections.items()},
        "status_count": len(statuses),
        "artifact_count": len(artifacts),
        "result_class_counts": dict(result_counts),
        "failure_category_counts": dict(failure_counts),
        "request_accounting": {"first_batch": 3, "continuation": 897, "total": 900},
        "financial_closure": {
            "spent_usd": str(charge_sum), "unresolved_reservations_usd": "0",
            "hard_cap_usd": "30", "charge_count": len(ledger["charges"]),
        },
        "raw_file_inventory": inventory,
        "raw_closure_sha256": closure_sha,
        "excluded_operational_files": ["runner.lock", "launch_authorized_remaining_897.sh", "runner_stdout_stderr.log"],
        "raw_results_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "raw_file_inventory.json", {"files": inventory, "raw_closure_sha256": closure_sha})
    atomic_json(OUT / "structural_closure_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "raw_file_inventory"}, indent=2))


if __name__ == "__main__":
    main()
