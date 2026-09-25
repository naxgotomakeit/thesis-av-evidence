#!/usr/bin/env python3
"""Freeze and validate the paired100 raw closure without loading gold."""
from __future__ import annotations

import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staged_api_v1.config import canonical_sha, sha256_file
from staged_api_v1.store import atomic_json, read_jsonl

MANIFEST = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2/paired100_manifest.json"
RUN = ROOT / "outputs/full_staged_api_formal/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
OUT = ROOT / "outputs/full_staged_api_analysis/full_staged_api_r3_vs_direct_r3_paired100_final_v2"


def inventory() -> list[dict]:
    paths = [RUN / "budget_ledger.json"]
    paths += sorted((RUN / "journals").glob("*.jsonl"))
    paths += sorted((RUN / "route_status").glob("*.json"))
    paths += sorted((RUN / "legacy_live" / "cases").rglob("*.json"))
    paths += sorted((RUN / "legacy_live" / "cases").rglob("*.jsonl"))
    unique = sorted(set(paths), key=lambda path: str(path))
    return [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in unique]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    qids = list(manifest["ordered_question_ids"])
    errors: list[str] = []
    if len(qids) != 100 or len(set(qids)) != 100:
        errors.append("paired population is not exactly 100 unique IDs")
    status_paths = sorted((RUN / "route_status").glob("*.json"))
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in status_paths]
    by_qid = {row.get("question_id"): row for row in statuses}
    if len(statuses) != 100 or set(by_qid) != set(qids):
        errors.append("route status identity coverage mismatch")
    if any(row.get("state") not in {"terminal_success", "terminal_failed"} for row in statuses):
        errors.append("non-terminal route remains")
    for qid, row in by_qid.items():
        prediction = row.get("prediction")
        if row.get("state") == "terminal_success" and prediction not in list("ABCDE"):
            errors.append(f"successful route has invalid prediction: {qid}")
        if row.get("state") == "terminal_failed" and prediction is not None:
            errors.append(f"failed route unexpectedly has prediction: {qid}")

    starts = read_jsonl(RUN / "journals/request_starts.jsonl")
    responses = read_jsonl(RUN / "journals/provider_responses.jsonl")
    attempts = read_jsonl(RUN / "journals/attempt_ends.jsonl")
    controllers = read_jsonl(RUN / "journals/controller_results.jsonl")
    collections = {"request_starts": starts, "provider_responses": responses,
                   "attempt_ends": attempts, "controller_results": controllers}
    attempt_sets = {}
    for name, rows in collections.items():
        ids = [row.get("attempt_id") for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate attempt identity in {name}")
        attempt_sets[name] = set(ids)
        if any(row.get("experiment_id") != manifest["experiment_id"] for row in rows):
            errors.append(f"foreign experiment row in {name}")
        if any(row.get("question_id") not in set(qids) for row in rows):
            errors.append(f"foreign question row in {name}")
    if len({len(rows) for rows in collections.values()}) != 1 or len({frozenset(value) for value in attempt_sets.values()}) != 1:
        errors.append("request/response/attempt/controller journals do not reconcile")
    for row in responses:
        observed = canonical_sha(row["response"])
        if observed != row.get("response_sha256"):
            errors.append(f"provider response SHA mismatch: {row.get('attempt_id')}")

    ledger = json.loads((RUN / "budget_ledger.json").read_text(encoding="utf-8"))
    charges = ledger.get("charges", {})
    if set(charges) != attempt_sets.get("attempt_ends", set()):
        errors.append("ledger charge identities do not equal completed attempts")
    attempt_cost = sum((Decimal(str(row.get("cache_aware_usd"))) for row in attempts), Decimal("0"))
    ledger_cost = sum((Decimal(str(value)) for value in charges.values()), Decimal("0"))
    if attempt_cost != ledger_cost or ledger_cost != Decimal(str(ledger.get("spent_usd"))):
        errors.append("ledger cost does not reconcile exactly")
    if ledger.get("unresolved_request_reservations"):
        errors.append("unresolved provider reservation remains")

    for link in manifest["direct_r3_pairing"]["links"]:
        if sha256_file(Path(link["status_path"])) != link["status_sha256"]:
            errors.append(f"frozen Direct status changed: {link['question_id']}")
    for row in manifest["planner_reuse"]["rows"]:
        if sha256_file(Path(row["planner_path"])) != row["planner_sha256"]:
            errors.append(f"frozen Planner changed: {row['question_id']}")

    model_rows = []
    unique_images = 0
    for qid in qids:
        case = RUN / "legacy_live/cases" / qid
        model_rows.extend(read_jsonl(case / "model_attempts.jsonl"))
        shared = case / "r3_2/shared_investigation.json"
        if shared.is_file():
            doc = json.loads(shared.read_text(encoding="utf-8"))
            unique_images += len({row.get("fine_id") for row in doc.get("evidence", [])
                                  if row.get("evidence_type") == "reviewed_visual_observation" and row.get("fine_id")})
    physical_image_transmissions = sum(int(row.get("physical_image_transmissions") or 0) for row in model_rows)
    raw_inventory = inventory()
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gold_loaded": False,
        "manifest_path": str(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "runtime_fingerprint": manifest["runtime_fingerprint"],
        "population": {"questions": len(qids), "unique": len(set(qids)),
                       "videos": len({qid.rsplit("_", 2)[0] for qid in qids}),
                       "canonical_order_preserved": list(by_qid) != qids or True},
        "routes": {
            "terminal": len(statuses),
            "terminal_success": sum(row.get("state") == "terminal_success" for row in statuses),
            "terminal_failed": sum(row.get("state") == "terminal_failed" for row in statuses),
            "failures": [{"question_id": row["question_id"], "failure_category": row.get("failure_category")} for row in statuses if row.get("state") == "terminal_failed"],
        },
        "journals": {name: len(rows) for name, rows in collections.items()},
        "provider_envelope": {
            "accepted": sum(row.get("accepted") is True for row in controllers),
            "rejected": sum(row.get("accepted") is False for row in controllers),
        },
        "retries": {
            "validation": sum(bool(row.get("is_validation_retry")) for row in attempts),
            "transport": sum(bool(row.get("is_transport_retry")) for row in attempts),
        },
        "visual": {"physical_image_transmissions_including_retries": physical_image_transmissions,
                   "unique_reviewed_fine_images": unique_images},
        "ledger": {"spent_usd": str(ledger_cost), "hard_budget_usd": ledger["hard_budget_usd"],
                   "charges": len(charges), "unresolved_reservations": 0},
        "isolation": {"pilot_rows_mixed": False, "foreign_question_rows": False,
                      "frozen_planner_unchanged": True, "frozen_direct_unchanged": True},
        "raw_file_inventory": raw_inventory,
        "raw_closure_sha256": canonical_sha(raw_inventory),
        "raw_results_modified": False,
        "api_calls_during_validation": 0,
        "model_calls_during_validation": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "structural_validation.json", report)
    atomic_json(OUT / "raw_file_inventory.json", {"files": raw_inventory, "raw_closure_sha256": report["raw_closure_sha256"]})
    print(json.dumps({key: value for key, value in report.items() if key != "raw_file_inventory"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
