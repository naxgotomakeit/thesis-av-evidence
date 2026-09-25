#!/usr/bin/env python3
"""Resume the real Direct-v1.2 formal run; execution requires an explicit flag.

Running without ``--execute-real-formal`` is read-only validation and cannot
construct an Anthropic client.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direct_api_v1.anthropic_provider import AnthropicDirectAgent
from direct_api_v1.formal_launch_gate import ValidatedLaunch, validate_launch_preflight
from direct_api_v1.formal_runtime import FormalOrchestrator
from direct_api_v1.frame_resolver import FrozenFrameResolver

EXPERIMENT = "direct_v1_2_3x16_r1_r3_eval300_formal_v1"
NAMESPACE = ROOT / "outputs/direct_v1_formal" / EXPERIMENT
MANIFEST = NAMESPACE / "formal_manifest_final_candidate_no_api_v3.json"
LAUNCH_LOCK = NAMESPACE / "formal_launch_lock_v3.json"
CONFIG = ROOT / "config/direct_v1_anthropic_smoke.json"


def preflight() -> ValidatedLaunch:
    return validate_launch_preflight(root=ROOT, namespace=NAMESPACE, manifest_path=MANIFEST,
                                     launch_lock_path=LAUNCH_LOCK, config_path=CONFIG)


def build_real_orchestrator(validated: ValidatedLaunch, *, agent_class=AnthropicDirectAgent,
                            orchestrator_class=FormalOrchestrator):
    """Build the API-capable runner only from a completed mandatory preflight."""
    manifest, config = validated.manifest, validated.config
    hierarchy_by_video = {row["video_id"]: row["hierarchy"] for row in manifest["videos"]}

    def resolver_factory(route):
        hierarchy = json.loads(Path(hierarchy_by_video[route["video_id"]]).read_text(encoding="utf-8"))
        return FrozenFrameResolver.from_hierarchy(hierarchy)

    def agent_factory(route, lifecycle, state):
        return agent_class(
            credential_env_path=Path(config["credential_env_path"]), model=config["model"],
            pricing=config["anthropic_cache_aware_pricing_usd_per_million_tokens"],
            max_output_tokens=config["max_output_tokens"], timeout_sec=config["timeout_sec"],
            max_retries=config["max_retries"], max_total_usd=50.0, lifecycle=lifecycle,
        )

    return orchestrator_class(
        manifest_path=MANIFEST, namespace=NAMESPACE, agent_factory=agent_factory,
        resolver_factory=resolver_factory, cap_usd=50.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-real-formal", action="store_true")
    args = parser.parse_args()
    validated = preflight()
    if not args.execute_real_formal:
        print(json.dumps({"valid": True, "mode": "read_only_no_api", "routes": 600,
                          "launch_mode": validated.mode, "manifest": str(MANIFEST),
                          "manifest_sha256": validated.manifest_sha256}, indent=2))
        return
    result = build_real_orchestrator(validated).execute()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
