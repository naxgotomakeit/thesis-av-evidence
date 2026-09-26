#!/usr/bin/env python3
"""Generate and exercise the ABD three-task batch control entirely offline."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.batch_control import (  # noqa: E402
    BATCH_CONTROL_VERSION,
    AbdLimitedBatchRunner,
    BatchScopedStore,
    batch_code_fingerprints,
    expected_authorization,
    load_and_verify_batch_control,
)
from abd_draft_v1.core import GensPromptBuilder, load_jsonl  # noqa: E402
from abd_draft_v1.formal_runtime import AbdFormalConfig  # noqa: E402
from abd_draft_v1.provider import AnthropicAbdProvider  # noqa: E402
from abd_draft_v1.store import atomic_json  # noqa: E402
from gens_haiku_eval300.runtime import sha256_file  # noqa: E402


CONFIG = ROOT / "config/abd_formal_candidate_v1.json"
BASE = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/formal_candidate_manifest.json"
OLD_FREEZE = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/freeze_record.json"
OUT = ROOT / "drafts/abd_direct_eval300_v1/limited_batch_v1"
CONTROL = OUT / "batch_control_manifest.json"
EXPECTED_QUESTION = "6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31"


class CaptureMessages:
    def __init__(self, model: str) -> None:
        self.model = model
        self.payloads: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "id": f"offline-batch-{len(self.payloads)}",
            "model": self.model,
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use", "id": f"tool-{len(self.payloads)}",
                "name": "final_answer",
                "input": {"selected_option_id": "A", "reason": "offline batch capture"},
            }],
            "usage": {"input_tokens": 1000, "output_tokens": 10},
        }


def scope_from_base(base: dict[str, Any]) -> list[dict[str, Any]]:
    scope = [
        {key: task[key] for key in (
            "execution_index", "task_id", "variant", "question_id", "video_id", "input_row_sha256"
        )}
        for task in base["tasks"][:3]
    ]
    if [task["execution_index"] for task in scope] != [0, 1, 2]:
        raise RuntimeError("first frozen indexes changed")
    if [task["variant"] for task in scope] != ["A", "B", "D"]:
        raise RuntimeError("first frozen arm order changed")
    if {task["question_id"] for task in scope} != {EXPECTED_QUESTION}:
        raise RuntimeError("first frozen question changed")
    return scope


def simulate_real_runner_path(
    config: AbdFormalConfig, base: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    rows = {
        variant: {row["question_id"]: row for row in load_jsonl(
            ROOT / f"drafts/abd_direct_eval300_v1/inputs/{variant}.jsonl"
        )}
        for variant in "ABD"
    }
    client = CaptureMessages(config.model)
    with tempfile.TemporaryDirectory() as temporary:
        store = BatchScopedStore(
            Path(temporary), experiment_id=config.experiment_id,
            hard_budget_usd=config.candidate_hard_budget_usd,
            request_reserve_usd=config.per_request_budget_reserve_usd,
        )
        runner = AbdLimitedBatchRunner(
            root=ROOT, config=config, config_path=CONFIG,
            base_manifest_path=BASE, control_manifest_path=CONTROL,
            store=store, provider_factory=lambda: AnthropicAbdProvider(client),
        )
        first = runner.run()
        second = runner.run()
        if len(client.payloads) != 3:
            raise RuntimeError("limited runner did not stop at exactly three simulated sends")
        if first["batch_financials"]["request_count"] != 3 or second["batch_financials"]["request_count"] != 3:
            raise RuntimeError("batch request count did not survive resume")
        builder = GensPromptBuilder(config.abd_prompt_config)
        expected_payloads = [
            builder.build_provider_payload(rows[task["variant"]][task["question_id"]], encode_images=True)
            for task in control["task_scope"]
        ]
        if client.payloads != expected_payloads:
            raise RuntimeError("batch control changed model-visible payloads")
        statuses = [store.status(task["task_id"])["state"] for task in control["task_scope"]]
        if statuses != ["terminal_success"] * 3:
            raise RuntimeError("offline limited runner terminal states differ")
        return {
            "simulated_provider_requests": len(client.payloads),
            "task_ids": [task["task_id"] for task in control["task_scope"]],
            "terminal_states": statuses,
            "fourth_request_occurred": False,
            "resume_added_requests": False,
            "payloads_byte_equivalent_to_frozen_builder_output": True,
            "formal_output_namespace_created": False,
            "temporary_state_removed_on_exit": True,
        }


def main() -> None:
    config = AbdFormalConfig.load(CONFIG)
    base = json.loads(BASE.read_text(encoding="utf-8"))
    old_freeze = json.loads(OLD_FREEZE.read_text(encoding="utf-8"))
    if old_freeze["formal_candidate_manifest_sha256"] != sha256_file(BASE):
        raise RuntimeError("old frozen candidate identity changed")
    scope = scope_from_base(base)
    control = {
        "schema_version": "abd_limited_batch_control_manifest_v1",
        "execution_control_version": BATCH_CONTROL_VERSION,
        "batch_id": "abd_first_question_abd_v1",
        "base_candidate_commit": "27e1350a581fc28ccf710890d19bf644f9f68c2c",
        "base_candidate_manifest_sha256": sha256_file(BASE),
        "change_scope": "execution range, request count, and incremental batch budget only",
        "scientific_payload_contract_changed": False,
        "task_scope": scope,
        "max_provider_requests": 3,
        "batch_budget_usd": 0.75,
        "input_manifest_sha256": {
            variant: base["file_fingerprints"][f"input_{variant}"] for variant in "ABD"
        },
        "batch_code_fingerprints": batch_code_fingerprints(ROOT),
        "source_git_parent_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "authorization_issued": False,
        "real_api_calls": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(CONTROL, control)
    load_and_verify_batch_control(
        root=ROOT, config_path=CONFIG, base_manifest_path=BASE,
        control_manifest_path=CONTROL,
    )
    template = expected_authorization(
        config=config, config_path=CONFIG, base_manifest_path=BASE,
        control_manifest_path=CONTROL, control=control, allow=False,
    )
    atomic_json(OUT / "authorization_TEMPLATE_NOT_AUTHORIZED.json", template)
    simulation = simulate_real_runner_path(config, base, control)
    report = {
        "schema_version": "abd_limited_batch_offline_preflight_v1",
        "status": "PASS_NOT_AUTHORIZED_NOT_RUN",
        "execution_control_version": BATCH_CONTROL_VERSION,
        "base_candidate_identity": {
            "commit": control["base_candidate_commit"],
            "manifest_sha256": control["base_candidate_manifest_sha256"],
        },
        "new_control_identity": {
            "manifest_path": str(CONTROL),
            "manifest_sha256": sha256_file(CONTROL),
            "code_fingerprints": control["batch_code_fingerprints"],
        },
        "scope": scope,
        "limits": {"max_provider_requests": 3, "batch_budget_usd": 0.75, "per_request_reserve_usd": 0.25},
        "simulation": simulation,
        "offline_unit_test_contract": [
            "exact A/B/D only and never fourth send",
            "scope/index/task/manifest/code mismatch rejection",
            "incremental spent plus unresolved plus next reserve admission",
            "persistent request count and costs across resume",
            "terminal skip and unknown-outcome no replay",
            "missing/invalid authorization fail closed",
            "model-visible payload equality with old frozen candidate",
            "temporary directories only",
        ],
        "formal_authorization_issued": False,
        "real_api_calls": 0,
        "formal_experiment_launches": 0,
        "flat15_or_historical_outputs_touched": False,
    }
    atomic_json(OUT / "BATCH_PREFLIGHT_REPORT.json", report)
    markdown = f"""# ABD limited-batch v1 — offline preflight

Status: **PASS; authorization not issued; formal batch not run**.

This adds execution-range control only. The base payload/runtime identity remains commit `{control['base_candidate_commit']}` and manifest SHA `{control['base_candidate_manifest_sha256']}`.

## Exact scope

| execution_index | task_id | arm |
|---:|---|---|
""" + "\n".join(
        f"| {task['execution_index']} | `{task['task_id']}` | {task['variant']} |"
        for task in scope
    ) + f"""

- Question: `{EXPECTED_QUESTION}`
- Maximum provider requests: 3
- Incremental batch cap: US$0.75
- Per-request reservation: US$0.25
- Fourth task is outside the authorization-bound scope and unreachable.

## Offline simulation

The real limited runner path invoked the existing frozen `_execute_task` three times through a capture-only simulated transport. It produced A, B, D in order, then stopped. A second invocation issued zero new requests. Captured model-visible payloads were exactly equal to the existing frozen builder output.

Unit tests cover scope and identity tampering, batch-budget admission, persistent request count/cost, terminal and unknown-outcome recovery, authorization failure, payload equality, and exclusive temporary state.

`authorization_TEMPLATE_NOT_AUTHORIZED.json` is deliberately non-executable. No signed authorization file was created.

Future command template, not executed:

```bash
{config.live_python} scripts/run_abd_limited_batch_v1.py --execute --approval-token APPROVE_ABD_FIRST_QUESTION_BATCH_V1 --approval-file <reviewed-limited-batch-authorization.json>
```

Real API calls: **0**. Formal experiment launches: **0**.
"""
    (OUT / "BATCH_PREFLIGHT_REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "task_ids": simulation["task_ids"],
        "simulated_provider_requests": 3, "real_api_calls": 0,
        "formal_experiment_launches": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
