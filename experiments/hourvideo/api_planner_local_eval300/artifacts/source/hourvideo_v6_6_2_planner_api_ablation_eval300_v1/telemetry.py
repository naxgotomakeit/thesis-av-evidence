from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

from .credentials import sanitize_for_persistence


TELEMETRY_SCHEMA = "hourvideo_planner_api_attempt_telemetry_v1"


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class AttemptTelemetryJournal:
    """Append-only, fsync-backed request and attempt telemetry."""

    def __init__(self, root: Path):
        self.root = root
        self.starts_path = root / "api_request_starts.jsonl"
        self.attempts_path = root / "api_attempts.jsonl"

    @staticmethod
    def _append(path: Path, record: dict[str, Any], secrets: tuple[str, ...] = ()) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = sanitize_for_persistence(record, secrets)
        encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def record_start(self, record: dict[str, Any], secrets: tuple[str, ...] = ()) -> None:
        required = {
            "request_id", "logical_call_id", "role", "provider", "model",
            "attempt_index", "is_retry", "retry_reason", "pricing_version",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(f"request_start missing telemetry fields: {sorted(missing)}")
        self._append(self.starts_path, {
            "telemetry_schema": TELEMETRY_SCHEMA,
            "event": "request_start",
            "timestamp_utc": now_utc(),
            "input_tokens": None,
            "output_tokens": None,
            "latency_sec": None,
            "status": "request_started",
            "estimated_cost_usd": None,
            "sanitized_error_category": None,
            **record,
        }, secrets)

    def record_end(self, record: dict[str, Any], secrets: tuple[str, ...] = ()) -> None:
        required = {
            "request_id", "logical_call_id", "role", "provider", "model",
            "attempt_index", "is_retry", "retry_reason", "input_tokens",
            "output_tokens", "latency_sec", "status", "pricing_version",
            "estimated_cost_usd", "sanitized_error_category",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(f"attempt_end missing telemetry fields: {sorted(missing)}")
        self._append(self.attempts_path, {
            "telemetry_schema": TELEMETRY_SCHEMA,
            "event": "attempt_end",
            "timestamp_utc": now_utc(),
            **record,
        }, secrets)

    @staticmethod
    def load(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
