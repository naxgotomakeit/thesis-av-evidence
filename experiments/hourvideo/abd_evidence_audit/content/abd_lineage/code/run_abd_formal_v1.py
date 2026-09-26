#!/usr/bin/env python3
"""Guarded ABD formal generation entry; dry validation unless explicitly authorized."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.formal_runtime import (  # noqa: E402
    APPROVAL_TOKEN,
    AbdFormalConfig,
    AbdFormalRunner,
    verify_execution_authority,
    verify_manifest,
)
from abd_draft_v1.provider import AnthropicAbdProvider  # noqa: E402
from abd_draft_v1.store import AbdStore  # noqa: E402


CONFIG_PATH = ROOT / "config/abd_formal_candidate_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval-token")
    parser.add_argument("--approval-file", type=Path)
    args = parser.parse_args()
    config = AbdFormalConfig.load(CONFIG_PATH)
    manifest_path = Path(config.formal_candidate_manifest_path)
    manifest = verify_manifest(ROOT, CONFIG_PATH, manifest_path)
    if not args.execute:
        print(json.dumps({
            "status": "OFFLINE_VALIDATED_NOT_EXECUTED",
            "experiment_id": config.experiment_id,
            "tasks": manifest["task_count"],
            "candidate_budget_usd": config.candidate_hard_budget_usd,
            "budget_authorized": False,
            "api_calls": 0,
        }, indent=2))
        return
    if args.approval_file is None:
        raise SystemExit("--execute requires --approval-file")
    if Path(sys.executable).resolve() != Path(config.live_python).resolve():
        raise SystemExit(f"ABD formal execution requires {config.live_python}")
    verify_execution_authority(
        config=config, config_path=CONFIG_PATH, manifest_path=manifest_path,
        approval_path=args.approval_file, approval_token=args.approval_token,
    )
    store = AbdStore(
        Path(config.output_root), experiment_id=config.experiment_id,
        hard_budget_usd=config.candidate_hard_budget_usd,
        request_reserve_usd=config.per_request_budget_reserve_usd,
    )
    runner = AbdFormalRunner(
        root=ROOT, config=config, config_path=CONFIG_PATH,
        manifest_path=manifest_path, store=store,
        provider_factory=lambda: AnthropicAbdProvider.from_credentials(
            credential_env_path=Path(config.credential_env_path),
            timeout_sec=config.timeout_sec,
        ),
    )
    print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
