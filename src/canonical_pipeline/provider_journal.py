"""Durable paid-provider attempt journal for crash-safe canonical runs.

The journal contains request hashes and model output text/usage metadata, never
credentials or media bytes.  An unfinished attempt is deliberately blocked on
automatic resume so a possibly successful provider request is not duplicated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


JOURNAL_SCHEMA_VERSION = "canonical-provider-attempt-journal-v1"
ATTEMPT_STATES = {
    "pending",
    "sent",
    "response_saved",
    "validated",
    "checkpointed",
    "technical_failure",
}


class AmbiguousProviderAttemptError(RuntimeError):
    """Raised when resume could duplicate an unfinished paid provider call."""


def _safe_case_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError(f"Unsafe journal case/stage identifier: {value!r}")
    return value


def _hashable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _hashable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_hashable(item) for item in value]
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def request_fingerprint(value: Any) -> str:
    """Hash a provider request while replacing raw byte payloads with hashes."""
    encoded = json.dumps(_hashable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _usage_fields(usage: Any) -> dict[str, Any]:
    names = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "total_cached_tokens",
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
    )
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {name: usage.get(name) for name in names if name in usage}
    return {name: getattr(usage, name) for name in names if hasattr(usage, name)}


class ProviderAttemptJournal:
    """Persist and validate provider-attempt state transitions atomically."""

    def __init__(self, root: Path):
        self.root = root

    def _case_dir(self, case_id: str) -> Path:
        return self.root / _safe_case_id(case_id)

    def _path(self, case_id: str, stage: str, attempt_number: int) -> Path:
        if attempt_number < 1:
            raise ValueError("Provider attempt numbers are one-based")
        return self._case_dir(case_id) / f"{_safe_case_id(stage)}_{attempt_number:02d}.json"

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != JOURNAL_SCHEMA_VERSION:
            raise AmbiguousProviderAttemptError(f"Unknown provider journal schema: {path}")
        return value

    def records(self, case_id: str) -> list[dict[str, Any]]:
        """Return all durable attempt records for one case."""
        folder = self._case_dir(case_id)
        return [self._load(path) for path in sorted(folder.glob("*.json"))] if folder.is_dir() else []

    def assert_case_resumable(self, case_id: str) -> None:
        """Block automatic calls when any earlier attempt is not checkpointed."""
        unfinished = [record for record in self.records(case_id) if record.get("state") != "checkpointed"]
        if unfinished:
            summary = [
                {
                    "stage": item.get("stage"),
                    "attempt_number": item.get("attempt_number"),
                    "state": item.get("state"),
                }
                for item in unfinished
            ]
            raise AmbiguousProviderAttemptError(
                f"Automatic provider retry blocked for {case_id}; reconcile unfinished journal: {summary}"
            )

    def begin_attempt(
        self,
        case_id: str,
        stage: str,
        attempt_number: int,
        request_sha256: str,
    ) -> None:
        """Durably record pending then sent before crossing provider boundary."""
        path = self._path(case_id, stage, attempt_number)
        if path.exists():
            existing = self._load(path)
            raise AmbiguousProviderAttemptError(
                f"Provider attempt already exists: {case_id}/{stage}/{attempt_number}={existing.get('state')}"
            )
        record = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "case_id": case_id,
            "stage": stage,
            "attempt_number": attempt_number,
            "request_sha256": request_sha256,
            "state": "pending",
            "history": ["pending"],
            "response": None,
            "validation": None,
        }
        self._write(path, record)
        record["state"] = "sent"
        record["history"].append("sent")
        self._write(path, record)

    def response_saved(
        self,
        case_id: str,
        stage: str,
        attempt_number: int,
        response: dict[str, Any],
    ) -> None:
        """Persist a returned provider response before parsing/validation."""
        path = self._path(case_id, stage, attempt_number)
        record = self._load(path)
        if record.get("state") != "sent":
            raise AmbiguousProviderAttemptError(f"Cannot save response from state {record.get('state')}")
        record["response"] = response
        record["state"] = "response_saved"
        record["history"].append("response_saved")
        self._write(path, record)

    def technical_failure(
        self,
        case_id: str,
        stage: str,
        attempt_number: int,
        exc: Exception,
    ) -> None:
        """Record a returned exception without claiming the call was unbillable."""
        path = self._path(case_id, stage, attempt_number)
        record = self._load(path)
        record["state"] = "technical_failure"
        record["history"].append("technical_failure")
        record["validation"] = {
            "error_class": type(exc).__name__,
            "message": str(exc)[:240],
            "provider_success_ambiguous": True,
        }
        self._write(path, record)

    def validated(
        self,
        case_id: str,
        stage: str,
        attempt_number: int,
        validation: dict[str, Any],
    ) -> None:
        """Record local parsing/policy validation for a saved response."""
        path = self._path(case_id, stage, attempt_number)
        record = self._load(path)
        if record.get("state") != "response_saved":
            raise AmbiguousProviderAttemptError(f"Cannot validate response from state {record.get('state')}")
        record["validation"] = validation
        record["state"] = "validated"
        record["history"].append("validated")
        self._write(path, record)

    def mark_case_checkpointed(self, case_id: str, checkpoint_path: Path) -> None:
        """Close validated attempts only after the case checkpoint is durable."""
        records = self.records(case_id)
        for record in records:
            if record.get("state") not in {"validated", "technical_failure"}:
                if record.get("state") == "checkpointed":
                    continue
                raise AmbiguousProviderAttemptError(
                    f"Cannot checkpoint {case_id}; attempt remains {record.get('state')}"
                )
            path = self._path(case_id, str(record["stage"]), int(record["attempt_number"]))
            record["checkpoint"] = {
                "path": checkpoint_path.as_posix(),
                "exists": checkpoint_path.is_file(),
            }
            if not record["checkpoint"]["exists"]:
                raise FileNotFoundError(f"Case checkpoint is not durable: {checkpoint_path}")
            record["state"] = "checkpointed"
            record["history"].append("checkpointed")
            self._write(path, record)

    def planner_request(
        self,
        case_id: str,
        request: Callable[[str, str, int], Any],
    ) -> Callable[[str, str, int], Any]:
        """Wrap the Task5A provider callback with durable attempt states."""
        def wrapped(system: str, user: str, number: int) -> Any:
            self.begin_attempt(
                case_id,
                "question_planner",
                number,
                request_fingerprint({"system": system, "user": user, "attempt": number}),
            )
            try:
                reply = request(system, user, number)
            except Exception as exc:
                self.technical_failure(case_id, "question_planner", number, exc)
                raise
            self.response_saved(
                case_id,
                "question_planner",
                number,
                {
                    "text": str(reply.text),
                    "latency_sec": float(reply.latency_sec),
                    "input_tokens": int(reply.input_tokens),
                    "output_tokens": int(reply.output_tokens),
                },
            )
            return reply
        return wrapped

    def gemini_client(self, case_id: str, client: Any) -> Any:
        """Wrap google-genai interactions.create without changing its result."""
        journal = self

        class Interactions:
            def __init__(self, source: Any):
                self.source = source
                self.count = 0

            def create(self, **request: Any) -> Any:
                self.count += 1
                attempt = self.count
                journal.begin_attempt(
                    case_id,
                    "final_model_api",
                    attempt,
                    request_fingerprint(request),
                )
                try:
                    response = self.source.create(**request)
                except Exception as exc:
                    journal.technical_failure(case_id, "final_model_api", attempt, exc)
                    raise
                journal.response_saved(
                    case_id,
                    "final_model_api",
                    attempt,
                    {
                        "output_text": None if getattr(response, "output_text", None) is None else str(response.output_text),
                        "usage": _usage_fields(getattr(response, "usage", None)),
                    },
                )
                return response

        class Client:
            def __init__(self, source: Any):
                self.interactions = Interactions(source.interactions)

        return Client(client)
