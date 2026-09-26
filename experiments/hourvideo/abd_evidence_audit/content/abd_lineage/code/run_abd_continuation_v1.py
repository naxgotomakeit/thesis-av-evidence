#!/usr/bin/env python3
"""Guarded continuation of frozen ABD indexes 3..899."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.continuation_control import (  # noqa: E402
    AbdContinuationRunner, ContinuationStore, load_and_verify_control, verify_authorization,
)
from abd_draft_v1.formal_runtime import AbdFormalConfig  # noqa: E402
from abd_draft_v1.provider import AnthropicAbdProvider  # noqa: E402


CONFIG = ROOT / "config/abd_formal_candidate_v1.json"
CANDIDATE = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/formal_candidate_manifest.json"
MAPPING = ROOT / "drafts/abd_direct_eval300_v1/limited_batch_v1/execution_identity_mapping.json"
CONTROL = ROOT / "drafts/abd_direct_eval300_v1/continuation_v1/continuation_control_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    parser.add_argument("--approval-file", type=Path)
    args = parser.parse_args()
    config = AbdFormalConfig.load(CONFIG)
    _manifest, control = load_and_verify_control(
        root=ROOT, config_path=CONFIG, candidate_manifest_path=CANDIDATE,
        identity_mapping_path=MAPPING, control_path=CONTROL,
    )
    if not args.execute:
        print(json.dumps({
            "status": "CONTINUATION_VALIDATED_NOT_EXECUTED",
            "remaining_tasks": 897, "first_execution_index": 3,
            "max_new_requests": 897, "total_budget_usd": 30,
        }, indent=2))
        return
    if args.approval_file is None:
        raise SystemExit("continuation --execute requires --approval-file")
    if Path(sys.executable).resolve() != Path(config.live_python).resolve():
        raise SystemExit(f"ABD continuation requires {config.live_python}")
    verify_authorization(
        config=config, config_path=CONFIG, candidate_manifest_path=CANDIDATE,
        identity_mapping_path=MAPPING, control_path=CONTROL,
        approval_path=args.approval_file, approval_token=args.approval_token,
        control=control,
    )
    store = ContinuationStore(
        Path(config.output_root), experiment_id=config.experiment_id,
        hard_budget_usd=config.candidate_hard_budget_usd,
        request_reserve_usd=config.per_request_budget_reserve_usd,
    )
    runner = AbdContinuationRunner(
        root=ROOT, config=config, config_path=CONFIG,
        candidate_manifest_path=CANDIDATE, identity_mapping_path=MAPPING,
        control_path=CONTROL, store=store,
        provider_factory=lambda: AnthropicAbdProvider.from_credentials(
            credential_env_path=Path(config.credential_env_path), timeout_sec=config.timeout_sec,
        ),
    )
    print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
