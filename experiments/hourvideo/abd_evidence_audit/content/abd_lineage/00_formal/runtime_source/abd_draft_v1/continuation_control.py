"""Resume the frozen ABD campaign after its accepted three-task batch."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from gens_haiku_eval300.runtime import canonical_sha, sha256_file

from .core import GensPromptBuilder
from .formal_runtime import AbdFormalConfig, AbdFormalRunner, verify_manifest
from .provider import AnthropicAbdProvider
from .store import AbdStore, BudgetExceeded, NON_REPLAY_STATES, RunnerLock, atomic_json, read_jsonl


CONTINUATION_VERSION = "abd_eval300_continuation_v1"
CONTINUATION_TOKEN = "APPROVE_ABD_EVAL300_REMAINING_897_V1"
CONTINUATION_AUTH_SCHEMA = "abd_eval300_continuation_authorization_v1"


class ContinuationControlError(RuntimeError):
    pass


class ContinuationLimitReached(BudgetExceeded):
    pass


def code_fingerprints(root: Path) -> dict[str, str]:
    return {
        "frozen_core": sha256_file(root / "src/abd_draft_v1/core.py"),
        "frozen_provider": sha256_file(root / "src/abd_draft_v1/provider.py"),
        "frozen_store": sha256_file(root / "src/abd_draft_v1/store.py"),
        "frozen_task_runtime": sha256_file(root / "src/abd_draft_v1/formal_runtime.py"),
        "continuation_control": sha256_file(root / "src/abd_draft_v1/continuation_control.py"),
        "continuation_entry": sha256_file(root / "scripts/run_abd_continuation_v1.py"),
        "continuation_preflight": sha256_file(root / "scripts/preflight_abd_continuation_v1.py"),
        "continuation_tests": sha256_file(root / "tests/test_abd_continuation_v1.py"),
    }


def scoped_identity(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task[key] for key in (
        "execution_index", "task_id", "variant", "question_id", "video_id", "input_row_sha256"
    )}


def remaining_scope(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [scoped_identity(task) for task in manifest["tasks"][3:]]


def load_and_verify_control(
    *, root: Path, config_path: Path, candidate_manifest_path: Path,
    identity_mapping_path: Path, control_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify_manifest(root, config_path, candidate_manifest_path)
    mapping = json.loads(identity_mapping_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if control.get("schema_version") != "abd_eval300_continuation_control_manifest_v1":
        raise ContinuationControlError("wrong continuation control schema")
    if control.get("execution_control_version") != CONTINUATION_VERSION:
        raise ContinuationControlError("wrong continuation control version")
    if control.get("candidate_manifest_sha256") != sha256_file(candidate_manifest_path):
        raise ContinuationControlError("scientific candidate identity mismatch")
    if control.get("execution_identity_mapping_sha256") != sha256_file(identity_mapping_path):
        raise ContinuationControlError("execution identity mapping mismatch")
    base = mapping.get("base_scientific_candidate", {})
    if base.get("candidate_manifest_sha256") != control["candidate_manifest_sha256"]:
        raise ContinuationControlError("mapping does not bind scientific candidate")
    if control.get("code_fingerprints") != code_fingerprints(root):
        raise ContinuationControlError("continuation code identity mismatch")
    scope = remaining_scope(manifest)
    if len(scope) != 897 or scope[0]["execution_index"] != 3 or scope[-1]["execution_index"] != 899:
        raise ContinuationControlError("frozen remaining scope is not indexes 3..899")
    if control.get("remaining_scope_sha256") != canonical_sha(scope):
        raise ContinuationControlError("remaining task scope identity mismatch")
    if control.get("first_remaining_task") != scope[0] or control.get("last_remaining_task") != scope[-1]:
        raise ContinuationControlError("remaining scope boundary mismatch")
    if control.get("remaining_task_count") != 897:
        raise ContinuationControlError("remaining task count changed")
    if control.get("max_new_provider_requests") != 897 or control.get("max_total_provider_requests") != 900:
        raise ContinuationControlError("continuation request limits changed")
    if Decimal(str(control.get("total_budget_usd"))) != Decimal("30"):
        raise ContinuationControlError("global budget changed")
    return manifest, control


def expected_authorization(
    *, config: AbdFormalConfig, config_path: Path, candidate_manifest_path: Path,
    identity_mapping_path: Path, control_path: Path, control: dict[str, Any], allow: bool,
) -> dict[str, Any]:
    return {
        "schema_version": CONTINUATION_AUTH_SCHEMA,
        "experiment_id": config.experiment_id,
        "continuation_id": control["continuation_id"],
        "real_execution_allowed": allow,
        "authorization_status": "AUTHORIZED" if allow else "TEMPLATE_NOT_AUTHORIZED",
        "execution_control_version": CONTINUATION_VERSION,
        "config_sha256": sha256_file(config_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "execution_identity_mapping_sha256": sha256_file(identity_mapping_path),
        "control_manifest_sha256": sha256_file(control_path),
        "code_fingerprints": control["code_fingerprints"],
        "remaining_scope_sha256": control["remaining_scope_sha256"],
        "remaining_task_count": 897,
        "first_execution_index": 3,
        "last_execution_index": 899,
        "max_new_provider_requests": 897,
        "max_total_provider_requests": 900,
        "total_budget_usd_including_first_batch": 30.0,
        "resume_baseline": control["resume_baseline"],
    }


def verify_authorization(
    *, config: AbdFormalConfig, config_path: Path, candidate_manifest_path: Path,
    identity_mapping_path: Path, control_path: Path, approval_path: Path,
    approval_token: str | None, control: dict[str, Any],
) -> dict[str, Any]:
    if approval_token != CONTINUATION_TOKEN:
        raise ContinuationControlError("explicit continuation approval token missing")
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContinuationControlError("continuation authorization missing or malformed") from error
    expected = expected_authorization(
        config=config, config_path=config_path, candidate_manifest_path=candidate_manifest_path,
        identity_mapping_path=identity_mapping_path, control_path=control_path,
        control=control, allow=True,
    )
    if approval != expected:
        raise ContinuationControlError("continuation authorization does not match scope and code")
    return approval


class ContinuationStore(AbdStore):
    """Persist continuation identity and enforce global/new request ceilings."""

    def configure_continuation(self, control: dict[str, Any], control_sha256: str) -> None:
        self.continuation_id = str(control["continuation_id"])
        self.max_new_requests = int(control["max_new_provider_requests"])
        self.max_total_requests = int(control["max_total_provider_requests"])
        self.continuation_state_path = self.root / "continuation_controls" / f"{self.continuation_id}.json"
        expected = {
            "execution_control_version": CONTINUATION_VERSION,
            "continuation_id": self.continuation_id,
            "control_manifest_sha256": control_sha256,
            "remaining_scope_sha256": control["remaining_scope_sha256"],
            "max_new_provider_requests": self.max_new_requests,
            "max_total_provider_requests": self.max_total_requests,
            "total_budget_usd": "30.0",
        }
        if self.continuation_state_path.exists():
            if json.loads(self.continuation_state_path.read_text(encoding="utf-8")) != expected:
                raise ContinuationControlError("durable continuation identity mismatch")
        else:
            self._assert_resume_baseline(control["resume_baseline"])
            atomic_json(self.continuation_state_path, expected)

    def _assert_resume_baseline(self, baseline: dict[str, Any]) -> None:
        starts = read_jsonl(self.request_starts_path)
        ledger = self._ledger()
        held = sum((Decimal(str(v)) for v in ledger["unresolved_request_reservations"].values()), Decimal("0"))
        expected_ids = baseline["completed_task_ids"]
        if len(starts) != baseline["provider_requests"]:
            raise ContinuationControlError("resume request-count baseline mismatch")
        if Decimal(str(ledger["spent_usd"])) != Decimal(str(baseline["spent_usd"])) or held != Decimal("0"):
            raise ContinuationControlError("resume financial baseline mismatch")
        if any(not self.status(task_id) or self.status(task_id).get("state") not in {"terminal_success", "terminal_failed"} for task_id in expected_ids):
            raise ContinuationControlError("first-batch terminal baseline mismatch")
        status_files = list(self.status_root.glob("*.json")) if self.status_root.is_dir() else []
        if len(status_files) != len(expected_ids):
            raise ContinuationControlError("unexpected task state exists before continuation")

    def continuation_financials(self) -> dict[str, Any]:
        starts = read_jsonl(self.request_starts_path)
        ledger = self._ledger()
        held = sum((Decimal(str(v)) for v in ledger["unresolved_request_reservations"].values()), Decimal("0"))
        return {
            "new_request_count": sum(row.get("continuation_id") == self.continuation_id for row in starts),
            "total_request_count": len(starts),
            "spent_usd": str(ledger["spent_usd"]),
            "unresolved_reservations_usd": str(held),
            "spent_plus_unresolved_usd": str(Decimal(str(ledger["spent_usd"])) + held),
        }

    def assert_continuation_can_request(self) -> None:
        values = self.continuation_financials()
        if values["new_request_count"] >= self.max_new_requests:
            raise ContinuationLimitReached("continuation reached 897 new requests")
        if values["total_request_count"] >= self.max_total_requests:
            raise ContinuationLimitReached("campaign reached 900 total requests")
        self.assert_budget_for_request()

    def request_start(self, row: dict[str, Any]) -> None:
        self.assert_continuation_can_request()
        super().request_start({**row, "continuation_id": self.continuation_id})


class AbdContinuationRunner:
    """Execute only frozen indexes 3..899 and pause on unknown outcomes."""

    def __init__(
        self, *, root: Path, config: AbdFormalConfig, config_path: Path,
        candidate_manifest_path: Path, identity_mapping_path: Path, control_path: Path,
        store: ContinuationStore, provider_factory: Callable[[], AnthropicAbdProvider],
    ) -> None:
        self.root = root
        self.config = config
        self.config_path = config_path
        self.candidate_manifest_path = candidate_manifest_path
        self.identity_mapping_path = identity_mapping_path
        self.control_path = control_path
        self.store = store
        self.provider_factory = provider_factory

    def run(self) -> dict[str, Any]:
        with RunnerLock(self.store.root / "runner.lock"):
            manifest, control = load_and_verify_control(
                root=self.root, config_path=self.config_path,
                candidate_manifest_path=self.candidate_manifest_path,
                identity_mapping_path=self.identity_mapping_path, control_path=self.control_path,
            )
            self.store.initialise()
            self.store.configure_continuation(control, sha256_file(self.control_path))
            frozen = AbdFormalRunner(
                root=self.root, config=self.config, config_path=self.config_path,
                manifest_path=self.candidate_manifest_path, store=self.store,
                provider_factory=self.provider_factory,
            )
            reconciliation = frozen.reconcile(manifest)
            pending = [task["task_id"] for task in manifest["tasks"] if (self.store.status(task["task_id"]) or {}).get("state") == "pending_provider_outcome_review"]
            if pending:
                return self._summary(control, 0, 0, "pending_provider_outcome_review", reconciliation, pending)
            rows = frozen._task_rows(manifest)
            builder = GensPromptBuilder(self.config.abd_prompt_config)
            provider_holder: list[AnthropicAbdProvider] = []

            def provider_once() -> AnthropicAbdProvider:
                if not provider_holder:
                    provider_holder.append(self.provider_factory())
                return provider_holder[0]

            executed = 0
            skipped = 0
            stop_reason = "scope_completed"
            for task in manifest["tasks"][3:]:
                task_id = task["task_id"]
                status = self.store.status(task_id)
                if status and status.get("state") in NON_REPLAY_STATES:
                    skipped += 1
                    if status.get("state") == "pending_provider_outcome_review":
                        stop_reason = "pending_provider_outcome_review"
                        break
                    continue
                try:
                    self.store.assert_continuation_can_request()
                except ContinuationLimitReached:
                    stop_reason = "request_or_budget_limit"
                    break
                result = frozen._execute_task(
                    task=task, row=rows[task_id], builder=builder,
                    provider_factory=provider_once,
                )
                executed += int(result["executed"])
                if result["budget_stopped"]:
                    stop_reason = "request_or_budget_limit"
                    break
                if (self.store.status(task_id) or {}).get("state") == "pending_provider_outcome_review":
                    stop_reason = "pending_provider_outcome_review"
                    break
            return self._summary(control, executed, skipped, stop_reason, reconciliation, [])

    def _summary(
        self, control: dict[str, Any], executed: int, skipped: int, stop_reason: str,
        reconciliation: dict[str, int], pending: list[str],
    ) -> dict[str, Any]:
        states = [self.store.status(task["task_id"]) for task in json.loads(self.candidate_manifest_path.read_text(encoding="utf-8"))["tasks"]]
        summary = {
            "execution_control_version": CONTINUATION_VERSION,
            "continuation_id": control["continuation_id"],
            "executed_this_invocation": executed,
            "skipped_non_replay_in_remaining_scope": skipped,
            "stop_reason": stop_reason,
            "terminal_count": sum(bool(state) and state.get("state") in {"terminal_success", "terminal_failed"} for state in states),
            "pending_task_ids": pending or [state["task_id"] for state in states if state and state.get("state") == "pending_provider_outcome_review"],
            "financials": self.store.continuation_financials(),
            "reconciliation": reconciliation,
            "next_execution_index": next((index for index, state in enumerate(states) if state is None), None),
        }
        atomic_json(self.store.root / "continuation_controls" / f"{control['continuation_id']}_last_execution.json", summary)
        return summary
