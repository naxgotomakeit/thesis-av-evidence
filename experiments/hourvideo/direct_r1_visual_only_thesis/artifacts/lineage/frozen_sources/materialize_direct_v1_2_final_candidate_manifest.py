#!/usr/bin/env python3
"""Create the versioned no-API candidate and its immutable launch lock."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direct_api_v1.formal_launch_gate import EXPECTED_CACHE, EXPECTED_MODEL, EXPECTED_PRICING, LOCK_SCHEMA
from direct_api_v1.formal_runtime import atomic_json, fingerprints, now, sha

EXPERIMENT = "direct_v1_2_3x16_r1_r3_eval300_formal_v1"
NAMESPACE = ROOT / "outputs/direct_v1_formal" / EXPERIMENT
SOURCE = NAMESPACE / "formal_manifest.json"
TARGET = NAMESPACE / "formal_manifest_final_candidate_no_api_v3.json"
LOCK = NAMESPACE / "formal_launch_lock_v3.json"


def main() -> None:
    if TARGET.exists() or LOCK.exists():
        raise RuntimeError("v3 launch artifacts already exist; refuse overwrite")
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config/direct_v1_anthropic_smoke.json").read_text(encoding="utf-8"))
    manifest["fingerprints"] = fingerprints(ROOT, config, manifest, NAMESPACE / "inputs")
    manifest["state"] = "final_candidate_v3_ready_no_api"
    manifest["formal_execution_started"] = False
    manifest["api_calls"] = 0
    manifest["model_calls"] = 0
    manifest["gold_loaded"] = False
    manifest["resume_policy"] = {
        "terminal_route": "skip_forever",
        "never_started": "run_normally",
        "interrupted_active_route": "terminal_failed:interrupted_active_route;never_replay",
        "mid_route_continuation": False,
    }
    manifest["durable_ordering"] = [
        "route_running", "request_start", "provider_call", "attempt_end", "ledger_reconciliation",
        "controller_validation", "controller_result", "state_frame_update", "atomic_terminal_route_status",
    ]
    manifest["source_dry_manifest"] = str(SOURCE)
    manifest["source_dry_manifest_sha256"] = sha(SOURCE)
    manifest["scientific_contract"] = {
        "model_id": EXPECTED_MODEL,
        "max_new_images_per_turn": 3,
        "max_unique_images_per_question": 16,
        "provider_transport_retries": 1,
        "structural_correction_retries": 1,
        "max_route_turns": 32,
        "question_count": 300,
        "video_count": 12,
        "route_count": 600,
        "r1_route_count": 300,
        "r3_route_count": 300,
        "formal_hard_budget_usd": 50.0,
        "cache_configuration": EXPECTED_CACHE,
        "pricing_configuration": EXPECTED_PRICING,
    }
    atomic_json(TARGET, manifest)
    lock = {
        "schema_version": LOCK_SCHEMA,
        "experiment_id": EXPERIMENT,
        "candidate_manifest_path": str(TARGET.resolve()),
        "candidate_manifest_sha256": sha(TARGET),
        "expected_route_count": 600,
        "expected_question_count": 300,
        "expected_model_id": EXPECTED_MODEL,
        "real_execution_allowed": True,
        "created_at": now(),
    }
    atomic_json(LOCK, lock)
    print(f"{TARGET}\nsha256={sha(TARGET)}\n{LOCK}\nsha256={sha(LOCK)}")


if __name__ == "__main__":
    main()
