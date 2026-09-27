from __future__ import annotations

import datetime
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class BudgetStateError(RuntimeError):
    pass


def _decimal(value: Decimal | str | int | float) -> Decimal:
    result = Decimal(str(value))
    if result < 0:
        raise ValueError("cost values must be non-negative")
    return result


class BudgetGuard:
    """Durable single-process cost reservations and hard-stop enforcement."""

    def __init__(self, ledger_path: Path, max_cost_usd: Decimal | str, hard_stop: bool = True):
        self.ledger_path = ledger_path
        self.max_cost_usd = _decimal(max_cost_usd)
        self.hard_stop = bool(hard_stop)

    def _append(self, row: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budget_schema": "hourvideo_api_budget_ledger_v1",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **row,
        }
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def rows(self) -> list[dict[str, Any]]:
        if not self.ledger_path.is_file():
            return []
        return [json.loads(line) for line in self.ledger_path.read_text(encoding="utf-8").splitlines() if line]

    def state(self) -> dict[str, Any]:
        reservations: dict[str, Decimal] = {}
        spent = Decimal("0")
        unknown_actual_cost_requests: set[str] = set()
        for row in self.rows():
            request_id = str(row["request_id"])
            if row["event"] == "reserve":
                if request_id in reservations:
                    raise BudgetStateError(f"duplicate active budget reservation: {request_id}")
                reservations[request_id] = _decimal(row["reserved_cost_usd"])
            elif row["event"] == "reconcile":
                if request_id not in reservations:
                    raise BudgetStateError(f"budget reconciliation without reservation: {request_id}")
                reservations.pop(request_id)
                actual = row.get("actual_cost_usd")
                if actual is None:
                    unknown_actual_cost_requests.add(request_id)
                else:
                    spent += _decimal(actual)
        return {
            "spent_usd": spent,
            "reserved_usd": sum(reservations.values(), Decimal("0")),
            "active_reservations": reservations,
            "unknown_actual_cost_requests": sorted(unknown_actual_cost_requests),
        }

    def reserve(self, request_id: str, worst_case_cost_usd: Decimal | str | None) -> None:
        state = self.state()
        if request_id in state["active_reservations"]:
            raise BudgetStateError(f"budget reservation already exists: {request_id}")
        if self.hard_stop and state["unknown_actual_cost_requests"]:
            raise BudgetExceeded("API budget hard stop: a prior request has unknown actual cost")
        if worst_case_cost_usd is None:
            if self.hard_stop:
                raise BudgetExceeded("API budget hard stop: next request has no approved pricing")
            reserve = Decimal("0")
        else:
            reserve = _decimal(worst_case_cost_usd)
        projected = state["spent_usd"] + state["reserved_usd"] + reserve
        if self.hard_stop and projected > self.max_cost_usd:
            raise BudgetExceeded("API budget hard stop: next request would exceed configured maximum")
        self._append({
            "event": "reserve",
            "request_id": request_id,
            "reserved_cost_usd": str(reserve),
            "max_cost_usd": str(self.max_cost_usd),
        })

    def reconcile(self, request_id: str, actual_cost_usd: Decimal | str | None) -> None:
        state = self.state()
        if request_id not in state["active_reservations"]:
            raise BudgetStateError(f"cannot reconcile unknown reservation: {request_id}")
        actual = None if actual_cost_usd is None else str(_decimal(actual_cost_usd))
        self._append({
            "event": "reconcile",
            "request_id": request_id,
            "actual_cost_usd": actual,
            "max_cost_usd": str(self.max_cost_usd),
        })
        updated = self.state()
        if self.hard_stop and updated["spent_usd"] > self.max_cost_usd:
            raise BudgetExceeded("API budget hard stop: reconciled cost exceeds configured maximum")
