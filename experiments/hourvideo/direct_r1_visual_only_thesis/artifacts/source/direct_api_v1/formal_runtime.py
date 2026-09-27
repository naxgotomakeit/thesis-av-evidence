"""Crash-safe orchestration for the frozen Direct-v1.2 formal protocol.

The runtime deliberately supports route-boundary resume only.  A route left in
``running`` state is materialised as ``interrupted_active_route`` and is never
sent to a provider again.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .anthropic_provider import direct_action_tools
from .controller import DirectController
from .maps import load_direct_input
from .policy import MAX_NEW_IMAGES_PER_TURN, MAX_UNIQUE_IMAGES_PER_QUESTION
from .prompt import DIRECT_V1_PROMPT_VERSION, DIRECT_V1_SYSTEM_PROMPT
from .state import DirectSessionState


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


class InjectedCrash(BaseException):
    """Test-only hard crash which provider/controller exception handlers cannot absorb."""


class CrashInjector:
    """One-shot crash injection at a named durable boundary."""

    def __init__(self, boundary: str | None = None) -> None:
        self.boundary = boundary
        self.triggered = False

    def checkpoint(self, boundary: str) -> None:
        if self.boundary == boundary and not self.triggered:
            self.triggered = True
            raise InjectedCrash(boundary)


class NamespaceLockError(RuntimeError):
    pass


class NamespaceWriterLock:
    """Single-writer lock acquired before any mutable run asset is touched."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / ".writer.lock"
        self.acquired = False

    def __enter__(self) -> "NamespaceWriterLock":
        if not self.root.is_dir():
            raise NamespaceLockError(f"namespace must exist before writer-lock acquisition: {self.root}")
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "acquired_at": now()}) + "\n"
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise NamespaceLockError(f"namespace writer lock already exists: {self.path}") from error
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


class FormalStore:
    def __init__(self, root: Path, experiment_id: str, cap: float = 50.0) -> None:
        self.root, self.experiment_id, self.cap = root, experiment_id, cap

    @property
    def ledger_path(self) -> Path:
        return self.root / "budget_ledger.json"

    def initialise(self) -> None:
        if not self.ledger_path.exists():
            atomic_json(self.ledger_path, {
                "schema_version": "reserved_budget_ledger_v2", "cap_usd": self.cap,
                "spent_usd": 0.0, "reserved_usd": 0.0,
                "attempt_ids": [], "attempts": {}, "charges": [],
            })

    def _ledger(self) -> dict[str, Any]:
        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        data.setdefault("schema_version", "reserved_budget_ledger_v2")
        data.setdefault("reserved_usd", 0.0)
        data.setdefault("attempts", {})
        data.setdefault("attempt_ids", [])
        data.setdefault("charges", [])
        return data

    def reserve_attempt(self, attempt_id: str, upper_bound: float, **metadata: Any) -> None:
        if upper_bound <= 0:
            raise RuntimeError("attempt upper bound must be positive")
        data = self._ledger()
        if attempt_id in data["attempts"]:
            raise RuntimeError("attempt reservation already exists")
        committed = float(data["spent_usd"]) + float(data["reserved_usd"])
        if committed + upper_bound > float(data["cap_usd"]) + 1e-12:
            raise RuntimeError("formal_frozen_accounting_cap_would_be_exceeded")
        data["attempts"][attempt_id] = {
            "state": "reserved", "reserved_upper_bound_usd": upper_bound,
            "created_at": now(), **metadata,
        }
        data["reserved_usd"] = float(data["reserved_usd"]) + upper_bound
        atomic_json(self.ledger_path, data)

    def mark_uncertain(self, attempt_id: str, reason: str) -> None:
        data = self._ledger()
        attempt = data["attempts"].get(attempt_id)
        if not attempt:
            raise RuntimeError("unknown attempt reservation")
        if attempt["state"] in {"reserved", "uncertain_liability"}:
            attempt.update({"state": "uncertain_liability", "uncertain_reason": reason, "updated_at": now()})
            atomic_json(self.ledger_path, data)

    def settle_attempt(self, attempt_id: str, actual_cost: float, **metadata: Any) -> bool:
        if actual_cost < 0:
            raise RuntimeError("negative provider cost")
        data = self._ledger()
        attempt = data["attempts"].get(attempt_id)
        if not attempt:
            raise RuntimeError("attempt has no reservation")
        if attempt["state"] == "settled":
            return False
        reserved = float(attempt["reserved_upper_bound_usd"])
        if actual_cost > reserved + 1e-12:
            attempt.update({"state": "upper_bound_violated", "actual_cost_usd": actual_cost, "updated_at": now()})
            atomic_json(self.ledger_path, data)
            raise RuntimeError("actual attempt cost exceeded reserved upper bound")
        remaining_reserved = max(0.0, float(data["reserved_usd"]) - reserved)
        next_spend = float(data["spent_usd"]) + actual_cost
        if next_spend + remaining_reserved > float(data["cap_usd"]) + 1e-12:
            raise RuntimeError("formal_frozen_accounting_cap_would_be_exceeded")
        attempt.update({"state": "settled", "actual_cost_usd": actual_cost, "settled_at": now(), **metadata})
        data["reserved_usd"] = remaining_reserved
        data["spent_usd"] = next_spend
        data["attempt_ids"].append(attempt_id)
        data["charges"].append({"attempt_id": attempt_id, "cost_usd": actual_cost, **metadata})
        atomic_json(self.ledger_path, data)
        return True

    def status(self, route_id: str) -> dict[str, Any] | None:
        path = self.root / "route_status" / f"{route_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_status(self, route_id: str, value: dict[str, Any]) -> None:
        atomic_json(self.root / "route_status" / f"{route_id}.json", value)

    def start(self, route_id: str, **metadata: Any) -> None:
        existing = self.status(route_id)
        if existing and str(existing.get("state", "")).startswith("terminal_"):
            raise RuntimeError("terminal route cannot restart")
        self.write_status(route_id, {"state": "running", "route_id": route_id, "started_at": now(), **metadata})

    def request_start(self, route_id: str, **record: Any) -> None:
        append_jsonl(self.root / "journals" / "request_start.jsonl", {"timestamp": now(), "experiment_id": self.experiment_id, "route_id": route_id, **record})

    def attempt_end(self, route_id: str, **record: Any) -> None:
        append_jsonl(self.root / "journals" / "attempt_end.jsonl", {"timestamp": now(), "experiment_id": self.experiment_id, "route_id": route_id, **record})

    def controller_result(self, route_id: str, **record: Any) -> None:
        append_jsonl(self.root / "journals" / "controller_result.jsonl", {"timestamp": now(), "experiment_id": self.experiment_id, "route_id": route_id, **record})

    def terminal(self, route_id: str, success: bool, **record: Any) -> None:
        self.write_status(route_id, {"state": "terminal_success" if success else "terminal_failed", "route_id": route_id, "terminal_at": now(), **record})

    def spend(self) -> float:
        return float(self._ledger()["spent_usd"])

    def uncertain_liabilities(self) -> list[dict[str, Any]]:
        data = self._ledger()
        return [
            {"attempt_id": attempt_id, **record}
            for attempt_id, record in data["attempts"].items()
            if record.get("state") in {"uncertain_liability", "upper_bound_violated"}
        ]

    def route_uncertain_liabilities(self, route_id: str) -> list[dict[str, Any]]:
        return [row for row in self.uncertain_liabilities() if row.get("route_id") == route_id]

    def assert_can_attempt(self, estimated_upper_bound_usd: float) -> None:
        data = self._ledger()
        total = float(data["spent_usd"]) + float(data["reserved_usd"]) + float(estimated_upper_bound_usd)
        if total > float(data["cap_usd"]) + 1e-12:
            raise RuntimeError("formal_frozen_accounting_cap_would_be_exceeded")

    def charge_once(self, attempt_id: str, cost: float, **metadata: Any) -> bool:
        return self.settle_attempt(attempt_id, cost, **metadata)

    def reconcile_ledger(self) -> int:
        """Idempotently charge every durable completed attempt."""
        reconciled = 0
        for record in read_jsonl(self.root / "journals" / "attempt_end.jsonl"):
            attempt_id = str(record["attempt_id"])
            if (record.get("provider_status", "").startswith("provider_error") and
                    not record.get("response_received") and not record.get("billing_outcome_known")):
                self.mark_uncertain(attempt_id, "provider_error_without_confirmed_response")
                continue
            if self.settle_attempt(
                attempt_id, float(record.get("cache_aware_usd") or 0.0),
                route_id=record.get("route_id"), provider_status=record.get("provider_status"),
            ):
                reconciled += 1
        return reconciled

    def response_path(self, attempt_id: str) -> Path:
        safe = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()
        return self.root / "provider_responses" / f"{safe}.json"

    def request_payload_path(self, attempt_id: str) -> Path:
        safe = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()
        return self.root / "request_payloads" / f"{safe}.json"

    def persist_request_payload(self, attempt_id: str, value: dict[str, Any]) -> Path:
        path = self.request_payload_path(attempt_id)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_sha(existing) != canonical_sha(value):
                raise RuntimeError("persisted provider request identity conflict")
            return path
        atomic_json(path, value)
        return path

    def persist_response(self, attempt_id: str, value: dict[str, Any]) -> Path:
        path = self.response_path(attempt_id)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_sha(existing) != canonical_sha(value):
                raise RuntimeError("persisted provider response identity conflict")
            return path
        atomic_json(path, value)
        return path

    def unresolved_attempts(self) -> list[dict[str, Any]]:
        ended = {row.get("attempt_id") for row in read_jsonl(self.root / "journals/attempt_end.jsonl")}
        return [row for row in read_jsonl(self.root / "journals/request_start.jsonl")
                if row.get("attempt_id") not in ended]

    def recover_persisted_attempts(self) -> dict[str, list[dict[str, Any]]]:
        recovered: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for start in self.unresolved_attempts():
            attempt_id = str(start["attempt_id"])
            response_path = self.response_path(attempt_id)
            if not response_path.exists():
                self.mark_uncertain(attempt_id, "request_started_without_durable_response")
                blocked.append({"attempt_id": attempt_id, "route_id": start.get("route_id"),
                                "reason": "pending_unknown_no_durable_response"})
                continue
            snapshot = json.loads(response_path.read_text(encoding="utf-8"))
            record = dict(snapshot["attempt_record"])
            self.attempt_end(str(start["route_id"]), attempt_id=attempt_id, **record)
            self.settle_attempt(attempt_id, float(record["cache_aware_usd"]),
                                route_id=start.get("route_id"), provider_status="offline_recovered_response")
            action = snapshot.get("parsed_action")
            if isinstance(action, dict) and action.get("action") == "final_answer" and action.get("selected_option_id") in "ABCDE":
                self._materialise_recovered_final(str(start["route_id"]), action, "persisted_provider_response")
            else:
                blocked.append({"attempt_id": attempt_id, "route_id": start.get("route_id"),
                                "reason": "persisted_nonterminal_response_requires_operator_resume",
                                "parsed_action": action})
            recovered.append({"attempt_id": attempt_id, "route_id": start.get("route_id"),
                              "parsed_action": action,
                              "response_path": str(response_path)})
        return {"recovered": recovered, "blocked": blocked}

    def result_path(self, route_id: str) -> Path:
        return self.root / "route_artifacts" / f"{route_id}.json"

    def checkpoint_path(self, route_id: str) -> Path:
        return self.root / "route_checkpoints" / f"{route_id}.json"

    def write_checkpoint(self, route_id: str, value: dict[str, Any]) -> None:
        atomic_json(self.checkpoint_path(route_id), value)

    def _materialise_recovered_final(self, route_id: str, action: dict[str, Any], source: str) -> dict[str, Any]:
        status = self.status(route_id) or {}
        checkpoint = None
        if self.checkpoint_path(route_id).exists():
            checkpoint = json.loads(self.checkpoint_path(route_id).read_text(encoding="utf-8"))
        attempts = [row for row in read_jsonl(self.root / "journals/attempt_end.jsonl")
                    if row.get("route_id") == route_id]
        artifact = dict(checkpoint["telemetry"]) if checkpoint and isinstance(checkpoint.get("telemetry"), dict) else {
            "question_id": status.get("question_id"), "method": status.get("method"),
            "started_at_utc": status.get("started_at"), "ended_at_utc": now(),
            "rounds": max([int(row.get("turn_index") or 0) for row in attempts] or [1]),
            "unique_images_transmitted": 0, "correction_attempts": 0,
            "turns": [], "provider_attempts": attempts,
        }
        if checkpoint and isinstance(checkpoint.get("state"), dict):
            artifact["unique_images_transmitted"] = int(checkpoint["state"].get("unique_image_count") or 0)
        recovered_rounds = max(int(artifact.get("rounds") or 0),
                               max([int(row.get("turn_index") or 0) for row in attempts] or [1]))
        artifact.update({
            "terminal_status": "final_answer", "final_prediction": action["selected_option_id"],
            "provider_attempts": attempts,
            "rounds": recovered_rounds,
            "total_api_attempts": len(attempts),
            "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in attempts),
            "total_cache_creation_input_tokens": sum(int(row.get("cache_creation_input_tokens") or 0) for row in attempts),
            "total_cache_read_input_tokens": sum(int(row.get("cache_read_input_tokens") or 0) for row in attempts),
            "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in attempts),
            "total_usd": sum(float(row.get("cache_aware_usd") or 0.0) for row in attempts),
            "recovered_offline": True, "recovery_source": source, "recovered_at": now(),
        })
        turns = artifact.setdefault("turns", [])
        if not turns or turns[-1].get("action_type") != "final_answer":
            last = attempts[-1] if attempts else {}
            turns.append({
                "turn_index": int(artifact["rounds"]), "action_type": "final_answer",
                "requested_timestamps_sec": [], "resolved_frames": [], "images_transmitted": 0,
                "duplicate_requests": 0, "provider_status": "offline_recovered_response",
                "action_reason": action.get("reason", ""),
                "input_tokens": last.get("input_tokens"),
                "cache_creation_input_tokens": last.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": last.get("cache_read_input_tokens"),
                "output_tokens": last.get("output_tokens"),
                "api_latency_sec": last.get("api_latency_sec"),
                "api_cost_usd": last.get("cache_aware_usd"),
                "ordinary_input_cost_usd": last.get("ordinary_input_cost_usd"),
                "cache_creation_cost_usd": last.get("cache_creation_cost_usd"),
                "cache_read_cost_usd": last.get("cache_read_cost_usd"),
                "output_cost_usd": last.get("output_cost_usd"),
            })
        atomic_json(self.result_path(route_id), artifact)
        self.terminal(route_id, True, category="final_answer", prediction=action["selected_option_id"],
                      artifact_path=str(self.result_path(route_id)),
                      resource_totals={"provider_attempts": len(attempts), "usd": artifact["total_usd"],
                                       "images": artifact.get("unique_images_transmitted", 0),
                                       "rounds": artifact.get("rounds", 0)},
                      recovered_offline=True, recovery_source=source)
        return artifact

    def recover_running_terminals(self, route_ids: Iterable[str]) -> list[dict[str, Any]]:
        recovered = []
        response_dir = self.root / "provider_responses"
        snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(response_dir.glob("*.json"))] if response_dir.exists() else []
        for route_id in route_ids:
            status = self.status(route_id)
            if not status or status.get("state") != "running":
                continue
            artifact_path = self.result_path(route_id)
            if artifact_path.exists():
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                success = artifact.get("terminal_status") == "final_answer" and artifact.get("final_prediction") in "ABCDE"
                self.terminal(route_id, success, category=artifact.get("terminal_status"),
                              prediction=artifact.get("final_prediction"), artifact_path=str(artifact_path),
                              resource_totals={"provider_attempts": artifact.get("total_api_attempts", 0),
                                               "usd": artifact.get("total_usd", 0.0),
                                               "images": artifact.get("unique_images_transmitted", 0),
                                               "rounds": artifact.get("rounds", 0)},
                              recovered_offline=True, recovery_source="durable_route_artifact")
                recovered.append({"route_id": route_id, "source": "durable_route_artifact"})
                continue
            candidates = [row for row in snapshots if row.get("route_id") == route_id]
            if candidates:
                action = candidates[-1].get("parsed_action")
                if isinstance(action, dict) and action.get("action") == "final_answer" and action.get("selected_option_id") in "ABCDE":
                    self._materialise_recovered_final(route_id, action, "settled_provider_response")
                    recovered.append({"route_id": route_id, "source": "settled_provider_response"})
        return recovered

    def materialise_interrupted(self, route_id: str, status: dict[str, Any]) -> None:
        completed_attempts = [row for row in read_jsonl(self.root / "journals" / "attempt_end.jsonl")
                              if row.get("route_id") == route_id]
        resource_totals = {
            **status.get("resource_totals", {}),
            "provider_attempts": len(completed_attempts),
            "usd": sum(float(row.get("cache_aware_usd") or 0.0) for row in completed_attempts),
        }
        artifact = {
            "route_id": route_id, "question_id": status.get("question_id"), "method": status.get("method"),
            "terminal_status": "interrupted_active_route", "final_prediction": None,
            "resource_totals": resource_totals, "materialised_at": now(),
        }
        atomic_json(self.result_path(route_id), artifact)
        self.terminal(route_id, False, category="interrupted_active_route", prediction=None,
                      artifact_path=str(self.result_path(route_id)), resource_totals=artifact["resource_totals"])

    def reconcile_active_routes(self, route_ids: Iterable[str]) -> int:
        reconciled = 0
        for route_id in route_ids:
            status = self.status(route_id)
            if status and status.get("state") == "running":
                self.write_status(route_id, {**status, "state": "blocked_pending_recovery",
                                             "blocked_at": now(),
                                             "reason": "active_route_requires_offline_recovery"})
                reconciled += 1
        return reconciled

    # Backward-compatible name retained for the already-materialised primitive.
    def reconcile(self, route_ids: list[str]) -> None:
        self.reconcile_active_routes(route_ids)


class FormalRouteLifecycle:
    """Lifecycle shared by a route's provider and controller."""

    def __init__(self, store: FormalStore, route_id: str, crash: CrashInjector | None = None) -> None:
        self.store, self.route_id = store, route_id
        self.crash = crash or CrashInjector()
        self._sequence = len([row for row in read_jsonl(store.root / "journals" / "request_start.jsonl") if row.get("route_id") == route_id])
        self._attempt_ids: dict[tuple[int, int], str] = {}

    def _event(self, event: str, **record: Any) -> None:
        append_jsonl(self.store.root / "journals" / "lifecycle.jsonl",
                     {"timestamp": now(), "experiment_id": self.store.experiment_id,
                      "route_id": self.route_id, "event": event, **record})

    def provider_request_start(self, *, turn_index: int, attempt_index: int, provider: str, model: str,
                               estimated_upper_bound_usd: float = 0.0) -> str:
        self._sequence += 1
        attempt_id = f"{self.route_id}:attempt:{self._sequence}"
        self.store.reserve_attempt(
            attempt_id, estimated_upper_bound_usd, route_id=self.route_id,
            turn_index=turn_index, attempt_index=attempt_index, provider=provider, model=model,
        )
        self._attempt_ids[(turn_index, attempt_index)] = attempt_id
        self.store.request_start(self.route_id, attempt_id=attempt_id, turn_index=turn_index,
                                 attempt_index=attempt_index, provider=provider, model=model)
        self._event("request_start", attempt_id=attempt_id)
        self.crash.checkpoint("after_request_start")
        return attempt_id

    def provider_response_received(self) -> None:
        self.crash.checkpoint("after_provider_call")

    def controller_failure(self, diagnostic: dict[str, Any]) -> None:
        path = self.store.root / "failure_diagnostics" / (
            hashlib.sha256(self.route_id.encode("utf-8")).hexdigest() + ".json")
        atomic_json(path, {"route_id": self.route_id, **diagnostic})
        self._event("controller_failure_diagnostic", diagnostic_path=str(path),
                    diagnostic_sha256=sha(path))

    def provider_request_prepared(self, attempt_id: str, snapshot: dict[str, Any]) -> None:
        path = self.store.persist_request_payload(attempt_id, snapshot)
        self._event("provider_request_prepared", attempt_id=attempt_id, request_path=str(path),
                    transport_mode=snapshot.get("transport_mode"))

    def provider_response_persisted(self, attempt_id: str, snapshot: dict[str, Any]) -> None:
        path = self.store.persist_response(attempt_id, snapshot)
        self._event("provider_response_persisted", attempt_id=attempt_id, response_path=str(path))
        self.crash.checkpoint("after_response_persisted")

    def provider_attempt_end(self, attempt_id: str, record: Any) -> None:
        payload = _jsonable(record)
        self.store.attempt_end(self.route_id, attempt_id=attempt_id, **payload)
        self._event("attempt_end", attempt_id=attempt_id)
        self.crash.checkpoint("after_attempt_end")
        if payload.get("response_received"):
            self.store.settle_attempt(attempt_id, float(payload.get("cache_aware_usd") or 0.0),
                                      route_id=self.route_id, provider_status=payload.get("provider_status"))
        else:
            self.store.mark_uncertain(attempt_id, "provider_error_without_confirmed_response")
        self._event("ledger_reconciliation", attempt_id=attempt_id)
        self.crash.checkpoint("after_ledger_reconciliation")

    def controller_validation(self, *, validation: str, state: DirectSessionState,
                              action: Any = None, provider_attempts: Iterable[Any] = ()) -> None:
        attempt_ids = []
        for record in provider_attempts:
            # IDs are deliberately recoverable from route/turn/attempt ordering.
            turn = getattr(record, "turn_index", None)
            attempt = getattr(record, "attempt_index", None)
            attempt_ids.append(self._attempt_ids.get((turn, attempt), {"turn_index": turn, "attempt_index": attempt}))
        self.store.controller_result(
            self.route_id, turn_index=state.rounds + 1, validation=validation,
            action=_jsonable(action), provider_attempts=attempt_ids,
        )
        self._event("controller_result", validation=validation)
        self.crash.checkpoint("after_controller_result")

    def controller_checkpoint(self, *, state: DirectSessionState, telemetry: Any) -> None:
        self.store.write_checkpoint(self.route_id, {
            "schema_version": "direct_route_checkpoint_v2",
            "route_id": self.route_id,
            "state": {
                "question_id": state.question_id,
                "mode": state.mode.value,
                "rounds": state.rounds,
                "unique_image_count": state.unique_image_count,
                "seen_frame_paths": sorted(state.seen_frame_paths),
                "final_prediction": state.final_prediction,
                "terminal_status": state.terminal_status,
            },
            "telemetry": telemetry.as_dict(),
            "written_at": now(),
        })
        self._event("controller_checkpoint")
        self.crash.checkpoint("after_controller_checkpoint")


class FormalOrchestrator:
    """Execute a frozen manifest with an injectable provider factory."""

    def __init__(self, *, manifest_path: Path, namespace: Path,
                 agent_factory: Callable[[dict[str, Any], FormalRouteLifecycle, DirectSessionState], Any],
                 resolver_factory: Callable[[dict[str, Any]], Any], cap_usd: float = 50.0,
                 crash: CrashInjector | None = None) -> None:
        self.manifest_path, self.namespace = manifest_path, namespace
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.agent_factory, self.resolver_factory = agent_factory, resolver_factory
        self.store = FormalStore(namespace, str(self.manifest["experiment_id"]), cap_usd)
        self.crash = crash or CrashInjector()

    def execute(self) -> dict[str, Any]:
        with NamespaceWriterLock(self.namespace):
            return self._execute_locked()

    def _execute_locked(self) -> dict[str, Any]:
        self.store.initialise()
        routes = list(self.manifest["routes"])
        route_ids = [str(route["route_id"]) for route in routes]
        recovery = self.store.recover_persisted_attempts()
        ledger_reconciled = self.store.reconcile_ledger()
        recovered_terminals = self.store.recover_running_terminals(route_ids)
        interrupted = self.store.reconcile_active_routes(route_ids)
        uncertain = self.store.uncertain_liabilities()
        if recovery["blocked"] or interrupted or uncertain:
            summary = {
                "route_count": len(routes), "executed": 0, "skipped_terminal": 0,
                "blocked_pending_recovery": recovery["blocked"],
                "offline_recovered_attempts": recovery["recovered"],
                "offline_recovered_terminals": recovered_terminals,
                "interrupted_routes_marked_blocked": interrupted,
                "uncertain_liabilities": uncertain,
                "ledger_attempts_reconciled": ledger_reconciled,
                "spend_before_usd": self.store.spend(), "new_provider_attempts": 0,
                "stopped_without_resend": True, "completed_at": now(),
            }
            append_jsonl(self.namespace / "execution_history.jsonl", summary)
            atomic_json(self.namespace / "last_execution.json", summary)
            return summary
        summary = {"route_count": len(routes), "executed": 0, "skipped_terminal": 0,
                   "interrupted_materialised": 0, "ledger_attempts_reconciled": ledger_reconciled,
                   "offline_recovered_attempts": recovery["recovered"],
                   "offline_recovered_terminals": recovered_terminals,
                   "provider_attempts_before": len(read_jsonl(self.namespace / "journals" / "attempt_end.jsonl")),
                   "spend_before_usd": self.store.spend()}
        input_root = self.manifest_path.parent / "inputs"
        for route in routes:
            route_id = str(route["route_id"])
            status = self.store.status(route_id)
            if status and str(status.get("state", "")).startswith("terminal_"):
                summary["skipped_terminal"] += 1
                continue
            self.store.start(route_id, question_id=route["question_id"], method=route["method"], resource_totals={})
            append_jsonl(self.namespace / "journals" / "lifecycle.jsonl",
                         {"timestamp": now(), "experiment_id": self.store.experiment_id,
                          "route_id": route_id, "event": "route_running"})
            self.crash.checkpoint("after_route_running")
            try:
                direct_input = load_direct_input(
                    question_path=input_root / f"{route['question_id']}.json", map_path=Path(route["map_path"]),
                    method=route["method"], expected_map_sha256=route["map_sha256"],
                )
                state = DirectSessionState(direct_input)
                lifecycle = FormalRouteLifecycle(self.store, route_id, self.crash)
                agent = self.agent_factory(route, lifecycle, state)
                controller = DirectController(resolver=self.resolver_factory(route), max_turns=32,
                                              enable_action_correction=True, lifecycle=lifecycle)
                telemetry = controller.run(state=state, agent=agent)
                artifact = telemetry.as_dict()
                diagnostic_path = self.namespace / "failure_diagnostics" / (
                    hashlib.sha256(route_id.encode("utf-8")).hexdigest() + ".json")
                if diagnostic_path.exists():
                    artifact.update({"failure_diagnostic_path": str(diagnostic_path),
                                     "failure_diagnostic_sha256": sha(diagnostic_path)})
                route_uncertain = self.store.route_uncertain_liabilities(route_id)
                if route_uncertain:
                    artifact.update({
                        "cost_completeness": "unknown_liability_not_in_total_usd",
                        "unresolved_reserved_upper_bound_usd": sum(
                            float(row.get("reserved_upper_bound_usd") or 0.0) for row in route_uncertain),
                        "unresolved_attempt_ids": [row["attempt_id"] for row in route_uncertain],
                    })
                atomic_json(self.store.result_path(route_id), artifact)
                append_jsonl(self.namespace / "journals" / "lifecycle.jsonl",
                             {"timestamp": now(), "experiment_id": self.store.experiment_id,
                              "route_id": route_id, "event": "terminal_artifact"})
                self.crash.checkpoint("after_terminal_artifact")
                resource_totals = {"provider_attempts": telemetry.total_api_attempts,
                                   "usd": telemetry.total_usd,
                                   "images": telemetry.unique_images_transmitted,
                                   "rounds": telemetry.rounds}
                if route_uncertain:
                    self.store.write_status(route_id, {
                        "state": "blocked_unknown_provider_outcome", "route_id": route_id,
                        "blocked_at": now(), "category": telemetry.terminal_status,
                        "prediction": telemetry.final_prediction,
                        "artifact_path": str(self.store.result_path(route_id)),
                        "resource_totals": resource_totals,
                        "unresolved_attempt_ids": [row["attempt_id"] for row in route_uncertain],
                        "unresolved_reserved_upper_bound_usd": artifact["unresolved_reserved_upper_bound_usd"],
                    })
                else:
                    success = telemetry.terminal_status == "final_answer" and telemetry.final_prediction is not None
                    self.store.terminal(route_id, success, category=telemetry.terminal_status,
                                        prediction=telemetry.final_prediction,
                                        artifact_path=str(self.store.result_path(route_id)),
                                        resource_totals=resource_totals)
                append_jsonl(self.namespace / "journals" / "lifecycle.jsonl",
                             {"timestamp": now(), "experiment_id": self.store.experiment_id,
                              "route_id": route_id,
                              "event": "blocked_unknown_provider_outcome" if route_uncertain else "terminal_status"})
                self.crash.checkpoint("after_terminal_status")
                summary["executed"] += 1
                if self.store.uncertain_liabilities():
                    summary["stopped_for_uncertain_liability"] = True
                    break
            except InjectedCrash:
                raise
            except Exception as error:
                artifact = {"route_id": route_id, "question_id": route["question_id"], "method": route["method"],
                            "terminal_status": f"runtime_failure:orchestrator:{type(error).__name__}",
                            "final_prediction": None, "resource_totals": {}, "materialised_at": now()}
                atomic_json(self.store.result_path(route_id), artifact)
                self.store.terminal(route_id, False, category=artifact["terminal_status"], prediction=None,
                                    artifact_path=str(self.store.result_path(route_id)), resource_totals={})
                summary["executed"] += 1
        summary["provider_attempts_after"] = len(read_jsonl(self.namespace / "journals" / "attempt_end.jsonl"))
        summary["new_provider_attempts"] = summary["provider_attempts_after"] - summary["provider_attempts_before"]
        summary["spend_after_usd"] = self.store.spend()
        summary["additional_spend_usd"] = summary["spend_after_usd"] - summary["spend_before_usd"]
        summary["terminal_routes"] = sum(
            bool(self.store.status(route_id)) and str(self.store.status(route_id).get("state", "")).startswith("terminal_")
            for route_id in route_ids
        )
        summary["completed_at"] = now()
        append_jsonl(self.namespace / "execution_history.jsonl", summary)
        atomic_json(self.namespace / "last_execution.json", summary)
        return summary


def fingerprints(root: Path, config: dict[str, Any], manifest: dict[str, Any] | None = None,
                 input_root: Path | None = None) -> dict[str, Any]:
    """Recompute every explicit implementation/configuration launch fingerprint."""
    source = root / "src/direct_api_v1"
    result = {
        "provider_name": config["provider"],
        "model_id": config["model"],
        "provider_config_hash": sha(root / "config/direct_v1_anthropic_smoke.json"),
        "prompt_hash": sha(source / "prompt.py"),
        "system_prompt_text_hash": hashlib.sha256(DIRECT_V1_SYSTEM_PROMPT.encode()).hexdigest(),
        "policy_hash": sha(source / "policy.py"),
        "controller_hash": sha(source / "controller.py"),
        "provider_hash": sha(source / "anthropic_provider.py"),
        "actions_hash": sha(source / "actions.py"),
        "state_machine_hash": sha(source / "state.py"),
        "maps_loader_hash": sha(source / "maps.py"),
        "telemetry_hash": sha(source / "telemetry.py"),
        "frame_resolver_hash": sha(source / "frame_resolver.py"),
        "pricing_hash": sha(source / "pricing.py"),
        "formal_runtime_hash": sha(source / "formal_runtime.py"),
        "formal_launch_gate_hash": sha(source / "formal_launch_gate.py"),
        "real_formal_runner_hash": sha(root / "scripts/execute_direct_v1_2_3x16_r1_r3_eval300_formal_v1.py"),
        "final_candidate_builder_hash": sha(root / "scripts/materialize_direct_v1_2_final_candidate_manifest.py"),
        "action_schema_hash": hashlib.sha256(json.dumps({str(n): direct_action_tools(n) for n in (3, 2, 1, 0)}, sort_keys=True).encode()).hexdigest(),
        "max_new_images_per_turn": MAX_NEW_IMAGES_PER_TURN,
        "max_unique_images_per_question": MAX_UNIQUE_IMAGES_PER_QUESTION,
        "provider_transport_retries": config["max_retries"],
        "structural_correction_retries": 1,
        "max_route_turns": 32,
        "max_output_tokens": config["max_output_tokens"],
        "provider_timeout_sec": config["timeout_sec"],
        "provider_temperature": 0.0,
        "cache_configuration": config["prompt_caching"],
        "pricing_configuration": config["anthropic_cache_aware_pricing_usd_per_million_tokens"],
        "pricing_catalog_path": config["pricing_catalog_path"],
        "configuration_schema_version": config["schema_version"],
        "prompt_version": DIRECT_V1_PROMPT_VERSION,
        "formal_hard_budget_usd": 50.0,
    }
    if manifest is not None:
        routes = list(manifest["routes"])
        if input_root is None:
            input_root = root / "outputs/direct_v1_formal" / str(manifest["experiment_id"]) / "inputs"
        question_ids = list(dict.fromkeys(str(route["question_id"]) for route in routes))
        population_path = Path(manifest["population_manifest"])
        population = json.loads(population_path.read_text(encoding="utf-8"))
        input_rows = [{"question_id": question_id, "path": str(input_root / f"{question_id}.json"),
                       "sha256": sha(input_root / f"{question_id}.json")} for question_id in question_ids]
        map_rows = [{"route_id": route["route_id"], "method": route["method"],
                     "path": route["map_path"], "sha256": sha(Path(route["map_path"]))} for route in routes]
        hierarchy_rows = [{"video_id": video["video_id"], "path": video["hierarchy"],
                           "sha256": sha(Path(video["hierarchy"]))} for video in manifest["videos"]]
        result.update({
            "eval300_population_manifest_hash": sha(population_path),
            "eval300_ordered_routes_source_hash": population["source_ordered_routes_sha256"],
            "question_input_closure_hash": canonical_sha(input_rows),
            "route_order_hash": canonical_sha([
                {key: route[key] for key in ("route_id", "method", "question_id", "video_id")} for route in routes
            ]),
            "r1_map_closure_hash": canonical_sha([row for row in map_rows if row["method"] == "R1"]),
            "r3_map_closure_hash": canonical_sha([row for row in map_rows if row["method"] == "R3"]),
            "hierarchy_closure_hash": canonical_sha(hierarchy_rows),
        })
    return result
