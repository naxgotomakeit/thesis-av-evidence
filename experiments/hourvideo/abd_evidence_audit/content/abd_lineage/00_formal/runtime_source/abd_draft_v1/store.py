"""Crash-durable ABD journals, statuses, budget holds, and process lock."""
from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


TERMINAL_STATES = frozenset({"terminal_success", "terminal_failed"})
NON_REPLAY_STATES = frozenset({*TERMINAL_STATES, "pending_provider_outcome_review"})


class BudgetExceeded(RuntimeError):
    pass


class RunnerAlreadyActive(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RunnerLock:
    """Non-blocking process lock; the kernel releases it after a crash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "RunnerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise RunnerAlreadyActive("another ABD runner holds the experiment lock") from error
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(json.dumps({"pid": os.getpid(), "acquired_at_utc": utc_now()}) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class AbdStore:
    def __init__(
        self,
        root: Path,
        *,
        experiment_id: str,
        hard_budget_usd: float,
        request_reserve_usd: float,
    ) -> None:
        self.root = root
        self.experiment_id = experiment_id
        self.hard_budget = Decimal(str(hard_budget_usd))
        self.request_reserve = Decimal(str(request_reserve_usd))
        self.journal_root = root / "journals"
        self.status_root = root / "task_status"
        self.artifact_root = root / "task_artifacts"
        self.request_starts_path = self.journal_root / "request_starts.jsonl"
        self.provider_responses_path = self.journal_root / "provider_responses.jsonl"
        self.attempt_ends_path = self.journal_root / "attempt_ends.jsonl"
        self.parser_results_path = self.journal_root / "parser_results.jsonl"
        self.ledger_path = root / "budget_ledger.json"

    def initialise(self) -> None:
        if not self.ledger_path.exists():
            atomic_json(self.ledger_path, {
                "experiment_id": self.experiment_id,
                "hard_budget_usd": str(self.hard_budget),
                "spent_usd": "0",
                "charges": {},
                "unresolved_request_reservations": {},
                "request_reserve_usd": str(self.request_reserve),
                "budget_authorized": True,
            })
        self._ledger()

    def _ledger(self) -> dict[str, Any]:
        ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if ledger.get("experiment_id") != self.experiment_id:
            raise RuntimeError("ABD ledger experiment mismatch")
        if Decimal(str(ledger.get("hard_budget_usd"))) != self.hard_budget:
            raise RuntimeError("ABD ledger cap mismatch")
        charges = ledger.get("charges")
        holds = ledger.get("unresolved_request_reservations")
        if not isinstance(charges, dict) or not isinstance(holds, dict):
            raise RuntimeError("ABD ledger structure invalid")
        spent = Decimal(str(ledger.get("spent_usd")))
        if spent != sum((Decimal(str(value)) for value in charges.values()), Decimal("0")):
            raise RuntimeError("ABD ledger charges do not reconcile")
        if set(charges).intersection(holds):
            raise RuntimeError("ABD attempt cannot be charged and unresolved")
        held = sum((Decimal(str(value)) for value in holds.values()), Decimal("0"))
        if spent + held > self.hard_budget:
            raise RuntimeError("ABD spent plus in-flight reservations exceeds hard cap")
        return ledger

    def assert_budget_for_request(self) -> None:
        ledger = self._ledger()
        spent = Decimal(str(ledger["spent_usd"]))
        held = sum(
            (Decimal(str(value)) for value in ledger["unresolved_request_reservations"].values()),
            Decimal("0"),
        )
        if spent + held + self.request_reserve > self.hard_budget:
            raise BudgetExceeded("ABD hard budget would be exceeded by a new in-flight reserve")

    def reserve_request(self, attempt_id: str) -> None:
        self.assert_budget_for_request()
        ledger = self._ledger()
        if attempt_id in ledger["charges"] or attempt_id in ledger["unresolved_request_reservations"]:
            raise RuntimeError("ABD attempt identity already exists")
        ledger["unresolved_request_reservations"][attempt_id] = str(self.request_reserve)
        atomic_json(self.ledger_path, ledger)

    def charge_once(self, attempt_id: str, usd: float) -> bool:
        amount = Decimal(str(usd))
        if amount < 0:
            raise RuntimeError("negative ABD provider cost")
        ledger = self._ledger()
        if attempt_id in ledger["charges"]:
            if Decimal(str(ledger["charges"][attempt_id])) != amount:
                raise RuntimeError("ABD attempt cost changed")
            return False
        held = Decimal(str(ledger["unresolved_request_reservations"].get(attempt_id, "0")))
        next_spent = Decimal(str(ledger["spent_usd"])) + amount
        other_holds = sum(
            (Decimal(str(value)) for key, value in ledger["unresolved_request_reservations"].items() if key != attempt_id),
            Decimal("0"),
        )
        if next_spent + other_holds > self.hard_budget:
            raise BudgetExceeded(
                f"authoritative ABD cost {amount} exceeds the admitted reserve {held} and hard cap"
            )
        ledger["charges"][attempt_id] = str(amount)
        ledger["unresolved_request_reservations"].pop(attempt_id, None)
        ledger["spent_usd"] = str(next_spent)
        atomic_json(self.ledger_path, ledger)
        return True

    def status_path(self, task_id: str) -> Path:
        return self.status_root / f"{task_id.replace(':', '__')}.json"

    def artifact_path(self, task_id: str) -> Path:
        return self.artifact_root / f"{task_id.replace(':', '__')}.json"

    def status(self, task_id: str) -> dict[str, Any] | None:
        path = self.status_path(task_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def start_task(self, task_id: str, **identity: Any) -> None:
        if self.status(task_id) is not None:
            raise RuntimeError("ABD task already has durable status")
        atomic_json(self.status_path(task_id), {
            "experiment_id": self.experiment_id,
            "task_id": task_id,
            "state": "running",
            "started_at_utc": utc_now(),
            **identity,
        })

    def request_start(self, row: dict[str, Any]) -> None:
        task_id = str(row["task_id"])
        status = self.status(task_id)
        if not status or status.get("state") != "running":
            raise RuntimeError("ABD request requires one running task")
        self.reserve_request(str(row["attempt_id"]))
        append_jsonl(self.request_starts_path, {
            "event": "request_start", "timestamp_utc": utc_now(),
            "experiment_id": self.experiment_id, **row,
        })

    def provider_response(self, row: dict[str, Any]) -> None:
        append_jsonl(self.provider_responses_path, {
            "event": "provider_response", "timestamp_utc": utc_now(),
            "experiment_id": self.experiment_id, **row,
        })

    def attempt_end(self, row: dict[str, Any]) -> None:
        append_jsonl(self.attempt_ends_path, {
            "event": "attempt_end", "timestamp_utc": utc_now(),
            "experiment_id": self.experiment_id, **row,
        })
        if row.get("authoritative_usage_available") is True and row.get("cache_aware_usd") is not None:
            self.charge_once(str(row["attempt_id"]), float(row["cache_aware_usd"]))

    def parser_result(self, row: dict[str, Any]) -> None:
        append_jsonl(self.parser_results_path, {
            "event": "parser_result", "timestamp_utc": utc_now(),
            "experiment_id": self.experiment_id, **row,
        })

    def set_pending_review(self, task_id: str, **extra: Any) -> dict[str, Any]:
        status = {
            "experiment_id": self.experiment_id,
            "task_id": task_id,
            "state": "pending_provider_outcome_review",
            "updated_at_utc": utc_now(),
            **extra,
        }
        atomic_json(self.status_path(task_id), status)
        return status

    def terminal(self, task_id: str, *, success: bool, artifact: dict[str, Any]) -> dict[str, Any]:
        existing = self.status(task_id)
        if existing and existing.get("state") in TERMINAL_STATES:
            return existing
        atomic_json(self.artifact_path(task_id), artifact)
        status = {
            "experiment_id": self.experiment_id,
            "task_id": task_id,
            "state": "terminal_success" if success else "terminal_failed",
            "terminal_at_utc": utc_now(),
            "prediction": artifact.get("prediction"),
            "result_class": artifact.get("result_class"),
            "failure_category": artifact.get("failure_category"),
            "attempt_id": artifact.get("attempt_id"),
            "artifact_path": str(self.artifact_path(task_id)),
        }
        atomic_json(self.status_path(task_id), status)
        return status

    def journal_rows(self, path: Path, task_id: str) -> list[dict[str, Any]]:
        return [row for row in read_jsonl(path) if row.get("task_id") == task_id]
