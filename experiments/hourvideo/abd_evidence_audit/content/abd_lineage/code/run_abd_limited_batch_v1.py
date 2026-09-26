#!/usr/bin/env python3
"""Guarded three-task ABD batch entry; defaults to identity validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.batch_control import (  # noqa: E402
    AbdLimitedBatchRunner,
    BatchScopedStore,
    load_and_verify_batch_control,
    verify_batch_authorization,
)
from abd_draft_v1.formal_runtime import AbdFormalConfig  # noqa: E402
from abd_draft_v1.provider import AnthropicAbdProvider  # noqa: E402


CONFIG = ROOT / "config/abd_formal_candidate_v1.json"
BASE_MANIFEST = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/formal_candidate_manifest.json"
CONTROL_MANIFEST = ROOT / "drafts/abd_direct_eval300_v1/limited_batch_v1/batch_control_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    parser.add_argument("--approval-file", type=Path)
    args = parser.parse_args()
    config = AbdFormalConfig.load(CONFIG)
    _base, control = load_and_verify_batch_control(
        root=ROOT, config_path=CONFIG, base_manifest_path=BASE_MANIFEST,
        control_manifest_path=CONTROL_MANIFEST,
    )
    if not args.execute:
        print(json.dumps({
            "status": "LIMITED_BATCH_OFFLINE_VALIDATED_NOT_EXECUTED",
            "batch_id": control["batch_id"],
            "task_ids": [task["task_id"] for task in control["task_scope"]],
            "max_provider_requests": control["max_provider_requests"],
            "batch_budget_usd": control["batch_budget_usd"],
            "authorization_issued": False,
            "api_calls": 0,
        }, indent=2))
        return
    if args.approval_file is None:
        raise SystemExit("limited-batch --execute requires --approval-file")
    if Path(sys.executable).resolve() != Path(config.live_python).resolve():
        raise SystemExit(f"limited ABD batch requires {config.live_python}")
    verify_batch_authorization(
        config=config, config_path=CONFIG, base_manifest_path=BASE_MANIFEST,
        control_manifest_path=CONTROL_MANIFEST, approval_path=args.approval_file,
        approval_token=args.approval_token, control=control,
    )
    store = BatchScopedStore(
        Path(config.output_root), experiment_id=config.experiment_id,
        hard_budget_usd=config.candidate_hard_budget_usd,
        request_reserve_usd=config.per_request_budget_reserve_usd,
    )
    runner = AbdLimitedBatchRunner(
        root=ROOT, config=config, config_path=CONFIG,
        base_manifest_path=BASE_MANIFEST, control_manifest_path=CONTROL_MANIFEST,
        store=store,
        provider_factory=lambda: AnthropicAbdProvider.from_credentials(
            credential_env_path=Path(config.credential_env_path),
            timeout_sec=config.timeout_sec,
        ),
    )
    print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
