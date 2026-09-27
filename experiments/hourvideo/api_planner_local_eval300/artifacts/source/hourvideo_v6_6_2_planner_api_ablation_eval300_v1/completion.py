from __future__ import annotations

import json
import time
from pathlib import Path


TERMINAL_STATUSES = {"normal_success", "downgraded_recovery", "failed"}


def route_completion_state(output_root: Path, question_id: str, route: str) -> str:
    """Return RUNNING until both durable terminal records agree on one route."""
    case = output_root / "cases" / question_id
    status_path = case / route / "route_status.json"
    events_path = case / "route_events.jsonl"
    if not status_path.is_file() or not events_path.is_file():
        return "RUNNING"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return "RUNNING"
    if status.get("side") != route or status.get("question_id") != question_id:
        return "RUNNING"
    if status.get("execution_status") not in TERMINAL_STATUSES:
        return "RUNNING"
    ends = [e for e in events if e.get("event") == "route_end" and e.get("route") == route and e.get("question_id") == question_id]
    return "TERMINAL" if ends else "RUNNING"


def wait_for_terminal_route(output_root: Path, question_id: str, route: str, *, timeout_sec: float = 7200, poll_sec: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while route_completion_state(output_root, question_id, route) != "TERMINAL":
        if time.monotonic() >= deadline:
            raise TimeoutError(f"route did not reach a durable terminal state: {question_id}/{route}")
        time.sleep(poll_sec)
