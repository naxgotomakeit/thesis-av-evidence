"""Crash-durable journals and independent budget ledger for Full Staged API."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"terminal_success", "terminal_failed"}


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
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


class BudgetExceeded(RuntimeError):
    pass


class StagedStore:
    def __init__(self, root: Path, *, experiment_id: str, hard_budget_usd: float, request_reserve_usd: float) -> None:
        self.root = root
        self.experiment_id = experiment_id
        self.hard_budget = Decimal(str(hard_budget_usd))
        self.request_reserve = Decimal(str(request_reserve_usd))
        self.journal_root = root / "journals"
        self.status_root = root / "route_status"
        self.request_starts_path = self.journal_root / "request_starts.jsonl"
        self.provider_responses_path = self.journal_root / "provider_responses.jsonl"
        self.attempt_ends_path = self.journal_root / "attempt_ends.jsonl"
        self.controller_results_path = self.journal_root / "controller_results.jsonl"
        self.ledger_path = root / "budget_ledger.json"
        if not self.ledger_path.exists():
            atomic_json(self.ledger_path, {
                "experiment_id": experiment_id,
                "hard_budget_usd": str(self.hard_budget),
                "spent_usd": "0",
                "charges": {},
                "unresolved_request_reservations": {},
                "request_reserve_semantics": "conservative_admission_floor_not_verified_cost_upper_bound",
            })
        self._validate_ledger()

    def _validate_ledger(self) -> dict[str, Any]:
        ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if ledger.get("experiment_id") != self.experiment_id or Decimal(str(ledger.get("hard_budget_usd"))) != self.hard_budget:
            raise RuntimeError("staged budget ledger identity mismatch")
        if Decimal(str(ledger["spent_usd"])) != sum((Decimal(str(value)) for value in ledger["charges"].values()), Decimal("0")):
            raise RuntimeError("staged budget ledger does not reconcile")
        reservations = ledger.get("unresolved_request_reservations")
        if not isinstance(reservations, dict):
            raise RuntimeError("staged budget ledger lacks durable request reservations")
        if set(reservations).intersection(ledger["charges"]):
            raise RuntimeError("an attempt cannot be both charged and unresolved")
        return ledger

    def assert_budget_for_request(self) -> None:
        ledger = self._validate_ledger()
        held = sum((Decimal(str(value)) for value in ledger["unresolved_request_reservations"].values()), Decimal("0"))
        if Decimal(str(ledger["spent_usd"])) + held + self.request_reserve > self.hard_budget:
            raise BudgetExceeded("hard API budget would be exceeded")

    def reserve_request(self, attempt_id: str) -> bool:
        """Durably hold admission budget before one physical request.

        The hold is deliberately described as an admission floor, not a
        verified upper bound on Anthropic billing.  Unknown outcomes retain
        the hold until an authoritative attempt-end can be reconciled.
        """
        ledger = self._validate_ledger()
        if attempt_id in ledger["charges"]:
            raise RuntimeError("charged attempt cannot be reserved")
        if attempt_id in ledger["unresolved_request_reservations"]:
            if Decimal(str(ledger["unresolved_request_reservations"][attempt_id])) != self.request_reserve:
                raise RuntimeError("request reservation changed")
            return False
        self.assert_budget_for_request()
        ledger = self._validate_ledger()
        ledger["unresolved_request_reservations"][attempt_id] = str(self.request_reserve)
        atomic_json(self.ledger_path, ledger)
        return True

    def charge_once(self, attempt_id: str, usd: float) -> bool:
        ledger = self._validate_ledger()
        amount = Decimal(str(usd))
        if attempt_id in ledger["charges"]:
            if Decimal(str(ledger["charges"][attempt_id])) != amount:
                raise RuntimeError("attempt cost changed during reconciliation")
            return False
        updated_spend = Decimal(str(ledger["spent_usd"])) + amount
        ledger["charges"][attempt_id] = str(amount)
        ledger["unresolved_request_reservations"].pop(attempt_id, None)
        ledger["spent_usd"] = str(updated_spend)
        ledger["actual_response_exceeded_hard_cap"] = updated_spend > self.hard_budget
        atomic_json(self.ledger_path, ledger)
        # An already-received provider response is authoritative spend and must
        # never disappear merely because the per-request reserve was too small.
        # Persist it first, then stop the campaign before any subsequent call.
        if updated_spend > self.hard_budget:
            raise BudgetExceeded("authoritative response cost exceeded hard budget")
        return True

    def reconcile_attempt_charges(self) -> dict[str, Any]:
        """Idempotently charge every durable attempt-end after a crash gap."""
        charged = 0
        already_charged = 0
        cap_exceeded = False
        for row in read_jsonl(self.attempt_ends_path):
            if row.get("authoritative_usage_available") is False or row.get("cache_aware_usd") is None:
                continue
            try:
                if self.charge_once(
                    str(row["attempt_id"]), float(row["cache_aware_usd"]),
                ):
                    charged += 1
                else:
                    already_charged += 1
            except BudgetExceeded:
                # charge_once has already persisted the authoritative amount.
                charged += 1
                cap_exceeded = True
        return {
            "charged_from_attempt_ends": charged,
            "already_charged": already_charged,
            "actual_response_exceeded_hard_cap": cap_exceeded,
        }

    def status_path(self, question_id: str) -> Path:
        return self.status_root / f"{question_id}.json"

    def status(self, question_id: str) -> dict[str, Any] | None:
        path = self.status_path(question_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def assert_not_terminal(self, question_id: str) -> None:
        status = self.status(question_id)
        if status and status.get("state") in TERMINAL_STATES:
            raise RuntimeError("terminal route can never issue another provider request")

    def start_route(self, question_id: str, *, input_sha256: str) -> None:
        status = self.status(question_id)
        if status:
            raise RuntimeError("route already has durable status")
        atomic_json(self.status_path(question_id), {
            "experiment_id": self.experiment_id, "question_id": question_id,
            "state": "running", "started_at_utc": utc_now(), "input_sha256": input_sha256,
            "provider_attempts": 0, "transport_retries": 0, "validation_retries": 0,
            "shared_calls": 0, "fine_calls": 0, "final_calls": 0, "image_transmissions": 0,
        })

    def request_start(self, row: dict[str, Any]) -> None:
        self.assert_not_terminal(str(row["question_id"]))
        self.reserve_request(str(row["attempt_id"]))
        append_jsonl(self.request_starts_path, {"event": "request_start", "timestamp_utc": utc_now(), "experiment_id": self.experiment_id, **row})

    def attempt_end(self, row: dict[str, Any]) -> None:
        append_jsonl(self.attempt_ends_path, {"event": "attempt_end", "timestamp_utc": utc_now(), "experiment_id": self.experiment_id, **row})
        if row.get("authoritative_usage_available") is not False and row.get("cache_aware_usd") is not None:
            self.charge_once(str(row["attempt_id"]), float(row["cache_aware_usd"]))

    def provider_response(self, row: dict[str, Any]) -> None:
        """Persist the complete provider response before any local parsing.

        This journal is intentionally separate from attempt-end telemetry and
        controller-envelope decisions.  Its append is fsync-backed, so an
        invalid or unparsable response remains available after a crash.
        """
        append_jsonl(self.provider_responses_path, {
            "event": "provider_response",
            "timestamp_utc": utc_now(),
            "experiment_id": self.experiment_id,
            **row,
        })

    def controller_result(self, row: dict[str, Any]) -> None:
        append_jsonl(self.controller_results_path, {"event": "controller_result", "timestamp_utc": utc_now(), "experiment_id": self.experiment_id, **row})

    def route_totals(self, question_id: str) -> dict[str, Any]:
        attempts = [row for row in read_jsonl(self.attempt_ends_path) if row.get("question_id") == question_id]
        fields = ("ordinary_input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
        totals = {field: sum(int(row[field]) for row in attempts if row.get(field) is not None) for field in fields}
        totals["usage_incomplete"] = any(
            any(row.get(field) is None for field in fields) for row in attempts
        )
        totals.update({
            "provider_attempts": len(attempts),
            "transport_retries": sum(bool(row.get("is_transport_retry")) for row in attempts),
            "validation_retries": len({
                (row.get("stage"), row.get("stage_call_index"), row.get("validation_retry_index"))
                for row in attempts if int(row.get("validation_retry_index") or 0) > 0
            }),
            "cache_aware_usd": sum(float(row["cache_aware_usd"]) for row in attempts if row.get("cache_aware_usd") is not None),
            "provider_outcome_unknown_attempts": sum(bool(row.get("provider_outcome_unknown")) for row in attempts),
            "api_latency_sec": sum(float(row.get("latency_sec") or 0.0) for row in attempts),
        })
        return totals

    def terminal(self, question_id: str, *, success: bool, extra: dict[str, Any]) -> dict[str, Any]:
        existing = self.status(question_id)
        if existing and existing.get("state") in TERMINAL_STATES:
            return existing
        status = {
            "experiment_id": self.experiment_id, "question_id": question_id,
            "state": "terminal_success" if success else "terminal_failed",
            "terminal_at_utc": utc_now(), **self.route_totals(question_id), **extra,
        }
        atomic_json(self.status_path(question_id), status)
        return status

    def reconcile_interrupted(self, ordered_question_ids: list[str]) -> dict[str, int]:
        charge_reconciliation = self.reconcile_attempt_charges()
        starts = read_jsonl(self.request_starts_path)
        responses = read_jsonl(self.provider_responses_path)
        ends = read_jsonl(self.attempt_ends_path)
        controllers = read_jsonl(self.controller_results_path)
        end_ids = {row.get("attempt_id") for row in ends}
        counts = {
            "terminal": 0, "never_started": 0, "interrupted_materialised": 0,
            **charge_reconciliation,
        }
        for question_id in ordered_question_ids:
            status = self.status(question_id)
            evidence = any(row.get("question_id") == question_id for row in starts + responses + ends + controllers)
            if status and status.get("state") in TERMINAL_STATES:
                counts["terminal"] += 1
            elif status or evidence:
                unmatched = [row["attempt_id"] for row in starts if row.get("question_id") == question_id and row.get("attempt_id") not in end_ids]
                self.terminal(question_id, success=False, extra={"failure_category": "interrupted_active_route", "prediction": None, "provider_outcome_unknown": bool(unmatched), "unmatched_request_attempt_ids": unmatched})
                counts["interrupted_materialised"] += 1
            else:
                counts["never_started"] += 1
        return counts
