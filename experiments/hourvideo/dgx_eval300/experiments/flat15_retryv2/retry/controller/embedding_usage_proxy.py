#!/usr/bin/env python3
"""Transparent OpenAI embedding-response usage logger.

The proxy delegates the unchanged embeddings.create call to the client made by
the frozen runtime.  It records only request metadata, hashes, response usage,
and timing; it never records credentials or plaintext query text.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_COUNT = 0
_INSTALLED = False


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _usage_int(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    ledger_raw = (os.getenv("EMBEDDING_USAGE_LEDGER") or "").strip()
    if not ledger_raw:
        raise RuntimeError("EMBEDDING_USAGE_LEDGER must be set before installing usage proxy")
    ledger = Path(ledger_raw)
    max_raw = (os.getenv("EMBEDDING_USAGE_MAX_REQUESTS") or "").strip()
    max_requests = int(max_raw) if max_raw else None
    price_per_million = float(os.getenv("EMBEDDING_INPUT_USD_PER_MILLION", "0.13"))

    from videoseal.utils.api import embeddings as embedding_module

    original_factory = embedding_module._mk_openai_client

    class _EmbeddingsProxy:
        def __init__(self, target: Any):
            self._target = target

        def create(self, *args: Any, **kwargs: Any) -> Any:
            global _COUNT
            with _LOCK:
                if max_requests is not None and _COUNT >= max_requests:
                    raise RuntimeError("embedding request cap reached before provider call")
                _COUNT += 1
                sequence = _COUNT
            values = kwargs.get("input")
            if values is None and len(args) >= 2:
                values = args[1]
            if isinstance(values, str):
                normalized = [values]
            else:
                normalized = list(values or [])
            canonical = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            started = time.monotonic()
            base = {
                "sequence": sequence,
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "pid": os.getpid(),
                "model": str(kwargs.get("model") or ""),
                "input_count": len(normalized),
                "input_sha256": hashlib.sha256(canonical).hexdigest(),
                "encoding_format": str(kwargs.get("encoding_format") or ""),
                "provider_request_reached": True,
            }
            try:
                response = self._target.create(*args, **kwargs)
                usage = getattr(response, "usage", None)
                prompt_tokens = _usage_int(usage, "prompt_tokens")
                total_tokens = _usage_int(usage, "total_tokens")
                record = dict(base)
                record.update({
                    "status": "success",
                    "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "elapsed_sec": round(time.monotonic() - started, 6),
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                    "input_cost_usd": (
                        round(prompt_tokens * price_per_million / 1_000_000.0, 12)
                        if prompt_tokens is not None else None
                    ),
                    "price_usd_per_million_input_tokens": price_per_million,
                    "response_request_id": str(getattr(response, "_request_id", "") or "") or None,
                })
                _append(ledger, record)
                return response
            except Exception as exc:
                record = dict(base)
                record.update({
                    "status": "failed",
                    "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "elapsed_sec": round(time.monotonic() - started, 6),
                    "prompt_tokens": None,
                    "total_tokens": None,
                    "input_cost_usd": None,
                    "price_usd_per_million_input_tokens": price_per_million,
                    "error_type": type(exc).__name__,
                })
                _append(ledger, record)
                raise

        def __getattr__(self, name: str) -> Any:
            return getattr(self._target, name)

    class _ClientProxy:
        def __init__(self, target: Any):
            self._target = target
            self.embeddings = _EmbeddingsProxy(target.embeddings)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._target, name)

    def factory(cfg: Any) -> Any:
        return _ClientProxy(original_factory(cfg))

    embedding_module._mk_openai_client = factory
    _INSTALLED = True

