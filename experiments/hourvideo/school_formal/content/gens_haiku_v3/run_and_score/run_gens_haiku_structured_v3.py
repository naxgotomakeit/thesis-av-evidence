#!/usr/bin/env python3
"""Explicitly gated smoke/formal runner for GenS structured-answer v3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direct_api_v1.anthropic_provider import load_existing_anthropic_credentials  # noqa: E402
from gens_haiku_eval300.runtime import GensAnsweringStore, read_jsonl  # noqa: E402
from gens_haiku_eval300.structured_v3 import (  # noqa: E402
    CampaignBudgetLedger,
    StructuredV3Config,
    StructuredV3HaikuProvider,
    StructuredV3RouteRunner,
    runtime_fingerprint,
)

PREFLIGHT = ROOT / "scripts/preflight_gens_haiku_structured_v3.py"
CAMPAIGN_LEDGER = ROOT / "outputs/gens_haiku_structured_v3/campaign_budget_v3.json"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/gens_haiku_structured_v3.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/gens_haiku_structured_v3/preflight_v3/formal_manifest.json")
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--execute-real-api", action="store_true")
    parser.add_argument("--i-understand-this-spends-money", action="store_true")
    args = parser.parse_args()

    config = StructuredV3Config.load(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run_fp = runtime_fingerprint(args.config, manifest, Path(__file__), PREFLIGHT)
    selected_ids = args.question_id or manifest["question_ids"]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("duplicate requested question identity")
    if not args.execute_real_api:
        print(json.dumps({
            "status": "DRY_ONLY",
            "would_run": len(selected_ids),
            "api_calls": 0,
            "run_fingerprint": run_fp,
        }, indent=2))
        return
    if not args.i_understand_this_spends_money:
        raise SystemExit("real API execution requires --i-understand-this-spends-money")
    if args.namespace.resolve().is_relative_to((ROOT / "outputs/gens_haiku_structured_v3/preflight_v3").resolve()):
        raise SystemExit("execution namespace must be separate from preflight")

    rows = load_rows(Path(config.selector_path))
    by_id = {row["qa_uid"]: row for row in rows}
    if any(qid not in by_id for qid in selected_ids):
        raise RuntimeError("unknown question identity")
    credential_path = json.loads(Path(config.direct_reference_config).read_text(encoding="utf-8"))["credential_env_path"]
    key = load_existing_anthropic_credentials(Path(credential_path))

    store = GensAnsweringStore(args.namespace, config.experiment_id, config.campaign_hard_api_budget_usd)
    store.initialise(run_fp)
    store.reconcile_ledger()
    campaign = CampaignBudgetLedger(CAMPAIGN_LEDGER, config.campaign_hard_api_budget_usd)
    campaign.initialise()
    campaign.reconcile(read_jsonl(args.namespace / "journals/attempt_end.jsonl"), str(args.namespace))
    interrupted = store.reconcile_interrupted(selected_ids)
    runner = StructuredV3RouteRunner(
        config,
        store,
        campaign,
        lambda: StructuredV3HaikuProvider(config, api_key=key),
    )
    summary = {
        "eligible": len(selected_ids),
        "executed": 0,
        "skipped": 0,
        "valid_answer": 0,
        "invalid_format": 0,
        "runtime_failure": interrupted,
        "interrupted_materialised": interrupted,
        "budget_stopped": False,
    }
    for qid in selected_ids:
        status = store.status(qid)
        if status and status["state"].startswith("terminal_"):
            summary["skipped"] += 1
            continue
        try:
            campaign.assert_budget(config.per_request_budget_reserve_usd)
        except RuntimeError:
            summary["budget_stopped"] = True
            break
        result = runner.run_one(by_id[qid], run_fingerprint=run_fp)
        summary["executed"] += 1
        summary[result.get("result_class", "runtime_failure")] += 1
        print(json.dumps({
            "question_id": qid,
            "result_class": result.get("result_class"),
            "prediction": result.get("prediction"),
            "spent_usd": json.loads(CAMPAIGN_LEDGER.read_text(encoding="utf-8"))["spent_usd"],
        }))
    summary["namespace_spent_usd"] = json.loads(store.ledger_path.read_text(encoding="utf-8"))["spent_usd"]
    summary["campaign_spent_usd"] = json.loads(CAMPAIGN_LEDGER.read_text(encoding="utf-8"))["spent_usd"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
