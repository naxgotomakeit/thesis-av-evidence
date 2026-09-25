#!/usr/bin/env python3
"""Guarded launch/resume entry for frozen R3 Full Staged paired100.

The command defaults to fingerprint verification only.  Live execution needs
both --execute and the explicit approval token printed by --help.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staged_api_v1.config import StagedConfig
from staged_api_v1.provider import AnthropicStagedProvider
from staged_api_v1.runtime import FullStagedRunner, verify_manifest
from staged_api_v1.store import BudgetExceeded, StagedStore


CONFIG = ROOT / "config/full_staged_api_r3_paired100_final_v2.json"
PREFLIGHT = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
MANIFEST = PREFLIGHT / "paired100_manifest.json"
LEGACY_CONFIG = PREFLIGHT / "legacy_downstream_config.json"
APPROVAL_TOKEN = "APPROVE_FULL_STAGED_R3_PAIRED100_FINAL_V2_25USD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    args = parser.parse_args()
    cfg = StagedConfig.load(CONFIG)
    manifest = verify_manifest(cfg, CONFIG, MANIFEST, ROOT)
    if not args.execute:
        print(json.dumps({"status": "PASS_NO_API_FINGERPRINT_GATE", "questions": len(manifest["ordered_question_ids"]),
                          "runtime_fingerprint": manifest["runtime_fingerprint"], "api_calls": 0}, indent=2))
        return
    if args.approval_token != APPROVAL_TOKEN:
        raise SystemExit(f"live paired100 requires --approval-token {APPROVAL_TOKEN}")
    if Path(sys.executable).resolve() != Path(cfg.live_python).resolve():
        raise SystemExit(f"live paired100 requires frozen Python interpreter: {cfg.live_python}")
    store = StagedStore(Path(cfg.output_root), experiment_id=cfg.experiment_id,
                        hard_budget_usd=cfg.hard_api_budget_usd,
                        request_reserve_usd=cfg.per_request_budget_reserve_usd)
    provider_factory = lambda: AnthropicStagedProvider(
        model=cfg.model,
        max_tokens_by_stage=cfg.max_tokens_by_stage,
        temperature=cfg.temperature,
        timeout_sec=cfg.timeout_sec,
        max_transport_retries=cfg.max_transport_retries,
        pricing=cfg.pricing_object,
        store=store,
        credential_env_path=Path(cfg.credential_env_path),
    )
    runner = FullStagedRunner(config=cfg, config_path=CONFIG, manifest_path=MANIFEST,
                              workspace_root=ROOT, store=store, provider_factory=provider_factory,
                              legacy_config_path=LEGACY_CONFIG)
    try:
        result = runner.run_ordered(list(manifest["ordered_question_ids"]))
    except BudgetExceeded as error:
        # The current started route is durably labelled global_budget_stop by
        # FullStagedRunner. Later, never-started routes retain no status and are
        # therefore pending, not model failures.
        result = {"status": "STOPPED_BY_GLOBAL_BUDGET", "reason": str(error),
                  "policy": "never-started routes remain pending"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
