"""Limited-batch control layered over the frozen ABD task executor."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from gens_haiku_eval300.runtime import canonical_sha, sha256_file

from .core import GensPromptBuilder
from .formal_runtime import AbdFormalConfig, AbdFormalRunner, verify_manifest
from .provider import AnthropicAbdProvider
from .store import (
    AbdStore,
    BudgetExceeded,
    NON_REPLAY_STATES,
    RunnerLock,
    atomic_json,
    read_jsonl,
)


BATCH_CONTROL_VERSION = "abd_limited_batch_execution_v1"
BATCH_APPROVAL_TOKEN = "APPROVE_ABD_FIRST_QUESTION_BATCH_V1"
BATCH_AUTH_SCHEMA = "abd_limited_batch_authorization_v1"


class BatchControlError(RuntimeError):
    pass


class BatchLimitReached(BudgetExceeded):
    pass


def batch_code_fingerprints(root: Path) -> dict[str, str]:
    return {
        "frozen_core": sha256_file(root / "src/abd_draft_v1/core.py"),
        "frozen_provider": sha256_file(root / "src/abd_draft_v1/provider.py"),
        "frozen_store": sha256_file(root / "src/abd_draft_v1/store.py"),
        "frozen_task_runtime": sha256_file(root / "src/abd_draft_v1/formal_runtime.py"),
        "batch_control": sha256_file(root / "src/abd_draft_v1/batch_control.py"),
        "batch_entry": sha256_file(root / "scripts/run_abd_limited_batch_v1.py"),
        "batch_preflight": sha256_file(root / "scripts/preflight_abd_limited_batch_v1.py"),
        "batch_tests": sha256_file(root / "tests/test_abd_limited_batch_v1.py"),
    }


def load_and_verify_batch_control(
    *, root: Path, config_path: Path, base_manifest_path: Path,
    control_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = verify_manifest(root, config_path, base_manifest_path)
    control = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    if control.get("schema_version") != "abd_limited_batch_control_manifest_v1":
        raise BatchControlError("wrong ABD batch-control schema")
    if control.get("execution_control_version") != BATCH_CONTROL_VERSION:
        raise BatchControlError("wrong ABD batch-control version")
    if control.get("base_candidate_manifest_sha256") != sha256_file(base_manifest_path):
        raise BatchControlError("base candidate manifest identity mismatch")
    if control.get("batch_code_fingerprints") != batch_code_fingerprints(root):
        raise BatchControlError("batch execution code identity mismatch")
    scope = control.get("task_scope")
    if not isinstance(scope, list) or len(scope) != 3:
        raise BatchControlError("batch scope must contain exactly three tasks")
    for expected, frozen in zip(scope, base["tasks"][:3]):
        identity = {
            key: frozen[key]
            for key in ("execution_index", "task_id", "variant", "question_id", "video_id", "input_row_sha256")
        }
        if expected != identity:
            raise BatchControlError("batch task scope differs from frozen execution order")
    if [task["execution_index"] for task in scope] != [0, 1, 2]:
        raise BatchControlError("batch execution indexes must be 0,1,2")
    if [task["variant"] for task in scope] != ["A", "B", "D"]:
        raise BatchControlError("batch arm order must be A,B,D")
    if len({task["question_id"] for task in scope}) != 1:
        raise BatchControlError("batch must contain one question")
    if control.get("max_provider_requests") != 3 or Decimal(str(control.get("batch_budget_usd"))) != Decimal("0.75"):
        raise BatchControlError("batch request/budget limit changed")
    expected_inputs = {
        variant: base["file_fingerprints"][f"input_{variant}"] for variant in "ABD"
    }
    if control.get("input_manifest_sha256") != expected_inputs:
        raise BatchControlError("batch input manifest identity mismatch")
    return base, control


def expected_authorization(
    *, config: AbdFormalConfig, config_path: Path, base_manifest_path: Path,
    control_manifest_path: Path, control: dict[str, Any], allow: bool,
) -> dict[str, Any]:
    return {
        "schema_version": BATCH_AUTH_SCHEMA,
        "experiment_id": config.experiment_id,
        "batch_id": control["batch_id"],
        "real_execution_allowed": allow,
        "authorization_status": "AUTHORIZED" if allow else "TEMPLATE_NOT_AUTHORIZED",
        "execution_control_version": BATCH_CONTROL_VERSION,
        "config_sha256": sha256_file(config_path),
        "base_candidate_manifest_sha256": sha256_file(base_manifest_path),
        "batch_control_manifest_sha256": sha256_file(control_manifest_path),
        "batch_code_fingerprints": control["batch_code_fingerprints"],
        "input_manifest_sha256": control["input_manifest_sha256"],
        "task_scope": control["task_scope"],
        "max_provider_requests": 3,
        "batch_budget_usd": 0.75,
    }


def verify_batch_authorization(
    *, config: AbdFormalConfig, config_path: Path, base_manifest_path: Path,
    control_manifest_path: Path, approval_path: Path, approval_token: str | None,
    control: dict[str, Any],
) -> dict[str, Any]:
    if approval_token != BATCH_APPROVAL_TOKEN:
        raise BatchControlError("explicit limited-batch approval token missing")
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchControlError("limited-batch authorization missing or malformed") from error
    expected = expected_authorization(
        config=config, config_path=config_path,
        base_manifest_path=base_manifest_path,
        control_manifest_path=control_manifest_path,
        control=control, allow=True,
    )
    if approval != expected:
        raise BatchControlError("limited-batch authorization does not match frozen scope and code")
    return approval


class BatchScopedStore(AbdStore):
    """Add a persistent incremental batch cap to the existing formal ledger."""

    def configure_batch(self, control: dict[str, Any], control_manifest_sha256: str) -> None:
        self.batch_id = str(control["batch_id"])
        self.batch_budget = Decimal(str(control["batch_budget_usd"]))
        self.max_batch_requests = int(control["max_provider_requests"])
        self.batch_scope = list(control["task_scope"])
        self.batch_state_path = self.root / "batch_controls" / f"{self.batch_id}.json"
        expected = {
            "execution_control_version": BATCH_CONTROL_VERSION,
            "batch_id": self.batch_id,
            "batch_control_manifest_sha256": control_manifest_sha256,
            "task_scope_sha256": canonical_sha(self.batch_scope),
            "max_provider_requests": self.max_batch_requests,
            "batch_budget_usd": str(self.batch_budget),
        }
        if self.batch_state_path.exists():
            if json.loads(self.batch_state_path.read_text(encoding="utf-8")) != expected:
                raise BatchControlError("durable batch state identity mismatch")
        else:
            atomic_json(self.batch_state_path, expected)

    def _batch_requests(self) -> list[dict[str, Any]]:
        return [
            row for row in read_jsonl(self.request_starts_path)
            if row.get("batch_id") == self.batch_id
        ]

    def batch_financials(self) -> dict[str, Any]:
        requests = self._batch_requests()
        attempt_ids = {str(row["attempt_id"]) for row in requests}
        ledger = self._ledger()
        spent = sum(
            (Decimal(str(value)) for key, value in ledger["charges"].items() if key in attempt_ids),
            Decimal("0"),
        )
        held = sum(
            (Decimal(str(value)) for key, value in ledger["unresolved_request_reservations"].items() if key in attempt_ids),
            Decimal("0"),
        )
        return {
            "request_count": len(requests),
            "spent_usd": str(spent),
            "unresolved_reservations_usd": str(held),
            "spent_plus_unresolved_usd": str(spent + held),
        }

    def assert_batch_can_request(self) -> None:
        financials = self.batch_financials()
        if financials["request_count"] >= self.max_batch_requests:
            raise BatchLimitReached("limited batch has reached three provider requests")
        admitted = (
            Decimal(financials["spent_usd"])
            + Decimal(financials["unresolved_reservations_usd"])
            + self.request_reserve
        )
        if admitted > self.batch_budget:
            raise BatchLimitReached("limited batch US$0.75 budget would be exceeded")

    def request_start(self, row: dict[str, Any]) -> None:
        self.assert_batch_can_request()
        super().request_start({**row, "batch_id": self.batch_id})


class AbdLimitedBatchRunner:
    """Run only the authorization-bound task subset through the frozen executor."""

    def __init__(
        self, *, root: Path, config: AbdFormalConfig, config_path: Path,
        base_manifest_path: Path, control_manifest_path: Path,
        store: BatchScopedStore,
        provider_factory: Callable[[], AnthropicAbdProvider],
    ) -> None:
        self.root = root
        self.config = config
        self.config_path = config_path
        self.base_manifest_path = base_manifest_path
        self.control_manifest_path = control_manifest_path
        self.store = store
        self.provider_factory = provider_factory

    def run(self) -> dict[str, Any]:
        with RunnerLock(self.store.root / "runner.lock"):
            base_manifest, control = load_and_verify_batch_control(
                root=self.root, config_path=self.config_path,
                base_manifest_path=self.base_manifest_path,
                control_manifest_path=self.control_manifest_path,
            )
            self.store.initialise()
            self.store.configure_batch(control, sha256_file(self.control_manifest_path))
            frozen = AbdFormalRunner(
                root=self.root, config=self.config, config_path=self.config_path,
                manifest_path=self.base_manifest_path, store=self.store,
                provider_factory=self.provider_factory,
            )
            reconciliation = frozen.reconcile(base_manifest)
            rows = frozen._task_rows(base_manifest)
            builder = GensPromptBuilder(self.config.abd_prompt_config)
            provider_holder: list[AnthropicAbdProvider] = []

            def provider_once() -> AnthropicAbdProvider:
                if not provider_holder:
                    provider_holder.append(self.provider_factory())
                return provider_holder[0]

            executed = 0
            skipped = 0
            stop_reason = "scope_completed"
            for scoped in control["task_scope"]:
                financials = self.store.batch_financials()
                if financials["request_count"] >= control["max_provider_requests"]:
                    stop_reason = "request_limit_reached"
                    break
                task_id = scoped["task_id"]
                status = self.store.status(task_id)
                if status and status.get("state") in NON_REPLAY_STATES:
                    skipped += 1
                    if status.get("state") == "pending_provider_outcome_review":
                        stop_reason = "pending_provider_outcome_review"
                        break
                    continue
                try:
                    # Stop before constructing the provider or starting a task.
                    # The store repeats this check atomically at request_start.
                    self.store.assert_batch_can_request()
                except BatchLimitReached:
                    stop_reason = "batch_budget_or_request_limit"
                    break
                task = base_manifest["tasks"][scoped["execution_index"]]
                try:
                    result = frozen._execute_task(
                        task=task, row=rows[task_id], builder=builder,
                        provider_factory=provider_once,
                    )
                except BatchLimitReached:
                    stop_reason = "batch_budget_or_request_limit"
                    break
                executed += int(result["executed"])
                if result["budget_stopped"]:
                    stop_reason = "batch_or_global_budget_stop"
                    break
                status = self.store.status(task_id)
                if status and status.get("state") == "pending_provider_outcome_review":
                    stop_reason = "pending_provider_outcome_review"
                    break
            final = self.store.batch_financials()
            summary = {
                "execution_control_version": BATCH_CONTROL_VERSION,
                "batch_id": control["batch_id"],
                "authorized_scope": control["task_scope"],
                "executed_this_invocation": executed,
                "skipped_non_replay": skipped,
                "stop_reason": stop_reason,
                "batch_financials": final,
                "reconciliation": reconciliation,
                "fourth_task_reachable": False,
            }
            atomic_json(
                self.store.root / "batch_controls" / f"{control['batch_id']}_last_execution.json",
                summary,
            )
            return summary
