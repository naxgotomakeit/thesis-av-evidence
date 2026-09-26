#!/usr/bin/env python3
"""Generate the narrow ABD continuation control and validate the live baseline offline."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.continuation_control import (  # noqa: E402
    CONTINUATION_VERSION, code_fingerprints, expected_authorization,
    load_and_verify_control, remaining_scope,
)
from abd_draft_v1.formal_runtime import AbdFormalConfig, verify_manifest  # noqa: E402
from abd_draft_v1.store import AbdStore, atomic_json, read_jsonl  # noqa: E402
from gens_haiku_eval300.runtime import canonical_sha, sha256_file  # noqa: E402


CONFIG = ROOT / "config/abd_formal_candidate_v1.json"
CANDIDATE = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/formal_candidate_manifest.json"
MAPPING = ROOT / "drafts/abd_direct_eval300_v1/limited_batch_v1/execution_identity_mapping.json"
OUT = ROOT / "drafts/abd_direct_eval300_v1/continuation_v1"
CONTROL = OUT / "continuation_control_manifest.json"
STATE = ROOT / "outputs/abd_eval300_formal_v1"


def main() -> None:
    config = AbdFormalConfig.load(CONFIG)
    manifest = verify_manifest(ROOT, CONFIG, CANDIDATE)
    scope = remaining_scope(manifest)
    first_ids = [task["task_id"] for task in manifest["tasks"][:3]]
    store = AbdStore(
        STATE, experiment_id=config.experiment_id,
        hard_budget_usd=config.candidate_hard_budget_usd,
        request_reserve_usd=config.per_request_budget_reserve_usd,
    )
    ledger = store._ledger()
    starts = read_jsonl(store.request_starts_path)
    if len(starts) != 3 or ledger["spent_usd"] != "0.084717" or ledger["unresolved_request_reservations"]:
        raise RuntimeError("accepted first-batch financial baseline changed")
    if any((store.status(task_id) or {}).get("state") not in {"terminal_success", "terminal_failed"} for task_id in first_ids):
        raise RuntimeError("accepted first-batch terminal baseline changed")
    if len(list(store.status_root.glob("*.json"))) != 3:
        raise RuntimeError("unexpected formal task status before continuation")
    control = {
        "schema_version": "abd_eval300_continuation_control_manifest_v1",
        "execution_control_version": CONTINUATION_VERSION,
        "continuation_id": "abd_eval300_remaining_897_v1",
        "candidate_manifest_sha256": sha256_file(CANDIDATE),
        "execution_identity_mapping_sha256": sha256_file(MAPPING),
        "remaining_task_count": 897,
        "remaining_scope_sha256": canonical_sha(scope),
        "first_remaining_task": scope[0],
        "last_remaining_task": scope[-1],
        "max_new_provider_requests": 897,
        "max_total_provider_requests": 900,
        "total_budget_usd": 30.0,
        "resume_baseline": {
            "completed_task_ids": first_ids,
            "provider_requests": 3,
            "spent_usd": "0.084717",
            "unresolved_reservations_usd": "0",
        },
        "code_fingerprints": code_fingerprints(ROOT),
        "source_git_parent_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "scientific_payload_contract_changed": False,
        "authorization_issued": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(CONTROL, control)
    load_and_verify_control(
        root=ROOT, config_path=CONFIG, candidate_manifest_path=CANDIDATE,
        identity_mapping_path=MAPPING, control_path=CONTROL,
    )
    template = expected_authorization(
        config=config, config_path=CONFIG, candidate_manifest_path=CANDIDATE,
        identity_mapping_path=MAPPING, control_path=CONTROL, control=control, allow=False,
    )
    atomic_json(OUT / "authorization_TEMPLATE_NOT_AUTHORIZED.json", template)
    report = {
        "schema_version": "abd_eval300_continuation_targeted_preflight_v1",
        "status": "PASS_NOT_AUTHORIZED_NOT_EXECUTED",
        "candidate_manifest_sha256": control["candidate_manifest_sha256"],
        "control_manifest_sha256": sha256_file(CONTROL),
        "source_git_parent_commit": control["source_git_parent_commit"],
        "baseline": control["resume_baseline"],
        "remaining": {
            "count": 897, "first_execution_index": 3, "last_execution_index": 899,
            "scope_sha256": control["remaining_scope_sha256"],
        },
        "limits": {"new_requests": 897, "total_requests": 900, "total_budget_usd": 30.0},
        "scientific_payload_contract_changed": False,
        "real_api_calls": 0,
    }
    atomic_json(OUT / "TARGETED_PREFLIGHT_REPORT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
