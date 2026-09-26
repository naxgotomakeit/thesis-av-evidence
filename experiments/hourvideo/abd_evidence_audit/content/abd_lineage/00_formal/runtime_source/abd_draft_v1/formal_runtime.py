"""Guarded single-request ABD runtime with response-aware crash recovery."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from direct_api_v1.pricing import AnthropicCacheAwarePricing
from gens_haiku_eval300.runtime import canonical_sha, sha256_file

from .core import AbdDraftConfig, GensPromptBuilder, load_jsonl, parse_response
from .provider import (
    AbdProviderError,
    AnthropicAbdProvider,
    authoritative_usage,
    field,
    serialise_provider_response,
)
from .store import (
    AbdStore,
    BudgetExceeded,
    NON_REPLAY_STATES,
    RunnerLock,
    atomic_json,
    read_jsonl,
)


APPROVAL_TOKEN = "APPROVE_ABD_EVAL300_FORMAL_V1"


class InjectedCrash(BaseException):
    pass


class CrashInjector:
    def __init__(self, boundary: str | None = None) -> None:
        self.boundary = boundary
        self.triggered = False

    def checkpoint(self, boundary: str) -> None:
        if self.boundary == boundary and not self.triggered:
            self.triggered = True
            raise InjectedCrash(boundary)


@dataclass(frozen=True)
class AbdFormalConfig:
    schema_version: str
    experiment_id: str
    abd_prompt_config_path: str
    formal_candidate_manifest_path: str
    output_root: str
    live_python: str
    credential_env_path: str
    provider: str
    model: str
    max_output_tokens: int
    temperature: float
    timeout_sec: float
    sdk_internal_max_retries: int
    outer_transport_retries: int
    cache_policy: str
    single_turn: bool
    structural_correction_retries: int
    concurrency: int
    candidate_hard_budget_usd: float
    per_request_budget_reserve_usd: float
    budget_authorized: bool
    budget_authorization_status: str
    api_enabled_by_default: bool
    assumed_context_window_tokens_for_offline_screening_only: int
    pricing: dict[str, Any]
    abd_prompt_config: AbdDraftConfig

    @classmethod
    def load(cls, path: Path) -> "AbdFormalConfig":
        path = path.resolve()
        root = path.parent.parent
        raw = json.loads(path.read_text(encoding="utf-8"))
        prompt_path = Path(raw["abd_prompt_config_path"])
        if not prompt_path.is_absolute():
            prompt_path = root / prompt_path
        for key in ("formal_candidate_manifest_path", "output_root"):
            value = Path(raw[key])
            if not value.is_absolute():
                raw[key] = str(root / value)
        raw["abd_prompt_config_path"] = str(prompt_path)
        config = cls(abd_prompt_config=AbdDraftConfig.load(prompt_path), **raw)
        base = config.abd_prompt_config
        if config.api_enabled_by_default:
            raise ValueError("ABD formal API must default off")
        if config.provider != base.provider or config.model != base.model:
            raise ValueError("ABD provider/model differs from approved prompt config")
        if config.max_output_tokens != base.max_output_tokens or config.temperature != base.temperature:
            raise ValueError("ABD generation parameters differ from approved config")
        if config.timeout_sec != base.timeout_sec:
            raise ValueError("ABD timeout differs from approved config")
        if config.sdk_internal_max_retries != 0 or config.outer_transport_retries != 0:
            raise ValueError("ABD physical requests must not retry automatically")
        if config.cache_policy != "disabled_no_cache_control":
            raise ValueError("ABD cache must remain disabled")
        if not config.single_turn or config.structural_correction_retries != 0:
            raise ValueError("ABD must be one turn with no correction")
        if config.concurrency != 1:
            raise ValueError("ABD concurrency must equal one")
        if config.candidate_hard_budget_usd <= 0 or config.per_request_budget_reserve_usd <= 0:
            raise ValueError("ABD budget settings must be positive")
        if config.per_request_budget_reserve_usd > config.candidate_hard_budget_usd:
            raise ValueError("ABD reserve exceeds candidate cap")
        if config.pricing != base.pricing:
            raise ValueError("ABD pricing differs from frozen GenS pricing")
        if not Path(config.live_python).is_file():
            raise ValueError("ABD live interpreter does not exist")
        return config

    @property
    def pricing_object(self) -> AnthropicCacheAwarePricing:
        return AnthropicCacheAwarePricing.from_mapping(self.pricing)


def file_fingerprints(root: Path, config_path: Path) -> dict[str, str]:
    config = AbdFormalConfig.load(config_path)
    return {
        "formal_config": sha256_file(config_path),
        "abd_prompt_config": sha256_file(Path(config.abd_prompt_config_path)),
        "common_system_prompt": sha256_file(Path(config.abd_prompt_config.system_prompt_path)),
        "abd_core": sha256_file(root / "src/abd_draft_v1/core.py"),
        "abd_provider": sha256_file(root / "src/abd_draft_v1/provider.py"),
        "abd_store": sha256_file(root / "src/abd_draft_v1/store.py"),
        "abd_formal_runtime": sha256_file(root / "src/abd_draft_v1/formal_runtime.py"),
        "abd_package_init": sha256_file(root / "src/abd_draft_v1/__init__.py"),
        "formal_preflight_entry": sha256_file(root / "scripts/preflight_abd_formal_candidate_v1.py"),
        "formal_generation_entry": sha256_file(root / "scripts/run_abd_formal_v1.py"),
        "independent_scorer_entry": sha256_file(root / "scripts/score_abd_formal_v1.py"),
        "formal_runtime_tests": sha256_file(root / "tests/test_abd_formal_v1.py"),
        "historical_parser": sha256_file(root / "src/gens_haiku_eval300/structured_v3.py"),
        "historical_pricing": sha256_file(root / "src/direct_api_v1/pricing.py"),
        "input_A": sha256_file(root / "drafts/abd_direct_eval300_v1/inputs/A.jsonl"),
        "input_B": sha256_file(root / "drafts/abd_direct_eval300_v1/inputs/B.jsonl"),
        "input_D": sha256_file(root / "drafts/abd_direct_eval300_v1/inputs/D.jsonl"),
    }


def verify_manifest(root: Path, config_path: Path, manifest_path: Path) -> dict[str, Any]:
    config = AbdFormalConfig.load(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != config.experiment_id:
        raise RuntimeError("ABD manifest experiment mismatch")
    if manifest.get("task_count") != 900 or manifest.get("tasks_per_variant") != {"A": 300, "B": 300, "D": 300}:
        raise RuntimeError("ABD frozen task population mismatch")
    if manifest.get("concurrency") != 1:
        raise RuntimeError("ABD frozen concurrency mismatch")
    if manifest.get("file_fingerprints") != file_fingerprints(root, config_path):
        raise RuntimeError("ABD implementation/input fingerprint mismatch")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 900:
        raise RuntimeError("ABD execution order missing")
    if canonical_sha(tasks) != manifest.get("execution_order_sha256"):
        raise RuntimeError("ABD execution order hash mismatch")
    counts = {variant: sum(task.get("variant") == variant for task in tasks) for variant in "ABD"}
    if counts != {"A": 300, "B": 300, "D": 300}:
        raise RuntimeError("ABD task order arm count mismatch")
    return manifest


def verify_execution_authority(
    *, config: AbdFormalConfig, config_path: Path, manifest_path: Path,
    approval_path: Path, approval_token: str | None,
) -> None:
    if approval_token != APPROVAL_TOKEN:
        raise RuntimeError("explicit ABD execution approval token missing")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": config.experiment_id,
        "real_execution_allowed": True,
        "authorized_budget_usd": config.candidate_hard_budget_usd,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if approval != expected:
        raise RuntimeError("ABD budget/launch authorization does not match frozen candidate")


def _response_record(
    *, config: AbdFormalConfig, task: dict[str, Any], attempt_id: str,
    response_tree: Any, response_sha256: str, latency_sec: float,
) -> dict[str, Any]:
    usage = authoritative_usage(response_tree)
    parsed = parse_response(field(response_tree, "content", []))
    cost = config.pricing_object.breakdown(**usage) if usage is not None else None
    usage_record = usage or {
        "ordinary_input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "output_tokens": None,
    }
    return {
        "task_id": task["task_id"],
        "question_id": task["question_id"],
        "variant": task["variant"],
        "attempt_id": attempt_id,
        "provider_status": "response_received",
        "provider_outcome_unknown": False,
        "authoritative_usage_available": usage is not None,
        **usage_record,
        "cost_breakdown": cost,
        "cache_aware_usd": None if cost is None else cost["total_cache_aware_usd"],
        "response_id": field(response_tree, "id"),
        "response_model": field(response_tree, "model"),
        "stop_reason": field(response_tree, "stop_reason"),
        "stop_sequence": field(response_tree, "stop_sequence"),
        "response_sha256": response_sha256,
        "latency_sec": latency_sec,
        **parsed,
    }


class AbdFormalRunner:
    def __init__(
        self,
        *,
        root: Path,
        config: AbdFormalConfig,
        config_path: Path,
        manifest_path: Path,
        store: AbdStore,
        provider_factory: Callable[[], AnthropicAbdProvider],
        crash: CrashInjector | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.config_path = config_path
        self.manifest_path = manifest_path
        self.store = store
        self.provider_factory = provider_factory
        self.crash = crash or CrashInjector()

    def _task_rows(self, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        cached: dict[str, dict[str, dict[str, Any]]] = {}
        for variant in "ABD":
            path = self.root / f"drafts/abd_direct_eval300_v1/inputs/{variant}.jsonl"
            cached[variant] = {row["question_id"]: row for row in load_jsonl(path)}
        for task in manifest["tasks"]:
            row = cached[task["variant"]][task["question_id"]]
            if canonical_sha(row) != task["input_row_sha256"]:
                raise RuntimeError(f"ABD input row changed: {task['task_id']}")
            rows[task["task_id"]] = row
        return rows

    def _finalise_response(
        self,
        task: dict[str, Any],
        response_row: dict[str, Any],
    ) -> dict[str, Any]:
        attempt_id = str(response_row["attempt_id"])
        prior_ends = [
            row for row in self.store.journal_rows(self.store.attempt_ends_path, task["task_id"])
            if row.get("attempt_id") == attempt_id
        ]
        if prior_ends:
            record = prior_ends[-1]
            if record.get("authoritative_usage_available") is True and record.get("cache_aware_usd") is not None:
                self.store.charge_once(attempt_id, float(record["cache_aware_usd"]))
        else:
            record = _response_record(
                config=self.config,
                task=task,
                attempt_id=attempt_id,
                response_tree=response_row["response"],
                response_sha256=response_row["response_sha256"],
                latency_sec=float(response_row.get("latency_sec_at_receipt") or 0.0),
            )
            self.store.attempt_end(record)
        self.crash.checkpoint("after_attempt_end")
        self.store.parser_result({
            "task_id": task["task_id"], "question_id": task["question_id"],
            "variant": task["variant"], "attempt_id": attempt_id,
            "result_class": record["result_class"],
            "prediction": record.get("prediction"),
            "reason": record.get("reason"),
            "failure_category": record.get("failure_category"),
        })
        usage_missing = record.get("authoritative_usage_available") is not True
        valid = record.get("result_class") == "valid_answer" and not usage_missing
        failure = "missing_authoritative_usage" if usage_missing else record.get("failure_category")
        artifact = {
            "schema_version": "abd_task_artifact_v1",
            "experiment_id": self.config.experiment_id,
            "task_id": task["task_id"],
            "question_id": task["question_id"],
            "video_id": task["video_id"],
            "variant": task["variant"],
            "attempt_id": attempt_id,
            "request_identity_sha256": response_row["request_identity_sha256"],
            "response_sha256": response_row["response_sha256"],
            "response_id": record.get("response_id"),
            "response_model": record.get("response_model"),
            "prediction": record.get("prediction") if valid else None,
            "parsed_prediction": record.get("prediction"),
            "reason": record.get("reason"),
            "result_class": record.get("result_class") if not usage_missing else "runtime_failure",
            "failure_category": failure,
            "provider_status": record.get("provider_status"),
            "authoritative_usage_available": not usage_missing,
            "usage": {key: record.get(key) for key in (
                "ordinary_input_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens", "output_tokens",
            )},
            "cache_aware_usd": record.get("cache_aware_usd"),
            "raw_provider_response_location": str(self.store.provider_responses_path),
        }
        return self.store.terminal(task["task_id"], success=valid, artifact=artifact)

    def reconcile(self, manifest: dict[str, Any]) -> dict[str, int]:
        counts = {"terminal": 0, "pending_review": 0, "recovered_response": 0, "never_started": 0}
        tasks = {task["task_id"]: task for task in manifest["tasks"]}
        for task_id, task in tasks.items():
            status = self.store.status(task_id)
            if status is None:
                counts["never_started"] += 1
                continue
            if status.get("state") in {"terminal_success", "terminal_failed"}:
                counts["terminal"] += 1
                continue
            if status.get("state") == "pending_provider_outcome_review":
                counts["pending_review"] += 1
                continue
            responses = self.store.journal_rows(self.store.provider_responses_path, task_id)
            if responses:
                self._finalise_response(task, responses[-1])
                counts["recovered_response"] += 1
                continue
            starts = self.store.journal_rows(self.store.request_starts_path, task_id)
            if starts:
                self.store.set_pending_review(
                    task_id,
                    question_id=task["question_id"], variant=task["variant"],
                    attempt_id=starts[-1]["attempt_id"], provider_outcome_unknown=True,
                    failure_category="request_sent_provider_outcome_unknown",
                )
                counts["pending_review"] += 1
            else:
                artifact = {
                    "schema_version": "abd_task_artifact_v1", "experiment_id": self.config.experiment_id,
                    "task_id": task_id, "question_id": task["question_id"], "video_id": task["video_id"],
                    "variant": task["variant"], "prediction": None, "result_class": "runtime_failure",
                    "failure_category": "interrupted_before_request_start", "attempt_id": None,
                }
                self.store.terminal(task_id, success=False, artifact=artifact)
                counts["terminal"] += 1
        return counts

    def _execute_task(
        self,
        *,
        task: dict[str, Any],
        row: dict[str, Any],
        builder: GensPromptBuilder,
        provider_factory: Callable[[], AnthropicAbdProvider],
    ) -> dict[str, Any]:
        task_id = task["task_id"]
        # Provider construction performs no request and occurs before any
        # request-start record. A missing SDK/credential therefore cannot be
        # mistaken for an unknown provider outcome.
        provider = provider_factory()
        self.store.start_task(
            task_id, question_id=task["question_id"], video_id=task["video_id"],
            variant=task["variant"], input_row_sha256=task["input_row_sha256"],
        )
        payload = builder.build_provider_payload(row, encode_images=True)
        request_sha = canonical_sha(payload)
        attempt_id = f"abd:{task_id}:{uuid.uuid4()}"
        request_record = {
            "task_id": task_id, "question_id": task["question_id"],
            "video_id": task["video_id"], "variant": task["variant"],
            "attempt_id": attempt_id, "request_identity_sha256": request_sha,
            "input_row_sha256": task["input_row_sha256"], "model": self.config.model,
            "max_tokens": self.config.max_output_tokens, "image_count": len(row["images"]),
            "map_present": row.get("map") is not None,
        }
        try:
            self.store.request_start(request_record)
        except BudgetExceeded:
            artifact = {
                "schema_version": "abd_task_artifact_v1", "experiment_id": self.config.experiment_id,
                "task_id": task_id, "question_id": task["question_id"], "video_id": task["video_id"],
                "variant": task["variant"], "prediction": None, "result_class": "runtime_failure",
                "failure_category": "global_budget_stop_no_request", "attempt_id": None,
            }
            self.store.terminal(task_id, success=False, artifact=artifact)
            return {"executed": False, "budget_stopped": True}
        self.crash.checkpoint("after_request_start")
        try:
            receipt = provider.send(payload)
        except AbdProviderError as error:
            self.store.attempt_end({
                **request_record, "provider_status": "provider_error",
                "provider_error_category": type(error.__cause__).__name__ if error.__cause__ else type(error).__name__,
                "provider_outcome_unknown": True, "authoritative_usage_available": False,
                "ordinary_input_tokens": None, "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None, "output_tokens": None,
                "cache_aware_usd": None,
            })
            self.store.set_pending_review(
                task_id, question_id=task["question_id"], variant=task["variant"],
                attempt_id=attempt_id, provider_outcome_unknown=True,
                failure_category="request_sent_provider_outcome_unknown",
            )
            return {"executed": True, "budget_stopped": False}
        response_tree, response_sha = serialise_provider_response(receipt.response)
        response_row = {
            **request_record,
            "provider_status": "response_received",
            "response_sha256": response_sha,
            "response": response_tree,
            "latency_sec_at_receipt": receipt.latency_sec,
        }
        self.store.provider_response(response_row)
        self.crash.checkpoint("after_provider_response")
        self._finalise_response(task, response_row)
        return {"executed": True, "budget_stopped": False}

    def run(self) -> dict[str, Any]:
        with RunnerLock(self.store.root / "runner.lock"):
            manifest = verify_manifest(self.root, self.config_path, self.manifest_path)
            self.store.initialise()
            reconciliation = self.reconcile(manifest)
            rows = self._task_rows(manifest)
            builder = GensPromptBuilder(self.config.abd_prompt_config)
            provider_holder: list[AnthropicAbdProvider] = []

            def provider_once() -> AnthropicAbdProvider:
                if not provider_holder:
                    provider_holder.append(self.provider_factory())
                return provider_holder[0]

            summary = {
                "task_count": 900, "executed": 0, "skipped_non_replay": 0,
                "budget_stopped": False, "reconciliation": reconciliation,
            }
            for task in manifest["tasks"]:
                task_id = task["task_id"]
                status = self.store.status(task_id)
                if status and status.get("state") in NON_REPLAY_STATES:
                    summary["skipped_non_replay"] += 1
                    continue
                row = rows[task_id]
                result = self._execute_task(
                    task=task, row=row, builder=builder,
                    provider_factory=provider_once,
                )
                summary["executed"] += int(result["executed"])
                if result["budget_stopped"]:
                    summary["budget_stopped"] = True
                    break
            ledger = self.store._ledger()
            summary.update({
                "spent_usd": ledger["spent_usd"],
                "unresolved_request_reservations": ledger["unresolved_request_reservations"],
            })
            atomic_json(self.store.root / "last_execution.json", summary)
            return summary
