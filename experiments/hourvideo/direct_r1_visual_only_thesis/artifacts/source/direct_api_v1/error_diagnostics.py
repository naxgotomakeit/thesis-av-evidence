"""Secret-safe exception-chain and traceback evidence for provider failures."""
from __future__ import annotations

import re
import traceback
from typing import Any


_REDACTIONS = (
    (re.compile(r"(?i)(x-api-key|api[_-]?key|authorization)(\s*[:=]\s*)([^\s,;\"'}]+)"), r"\1\2[REDACTED]"),
    (re.compile(r"(?i)(\"(?:x-api-key|api[_-]?key|authorization)\"\s*:\s*\")[^\"]*(\")"), r"\1[REDACTED]\2"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]+"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
)


def redact_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def exception_chain(error: BaseException) -> list[dict[str, Any]]:
    result = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append({
            "exception_type": type(current).__name__,
            "module": type(current).__module__,
            "message_redacted": redact_text(current),
            "http_status": int(getattr(current, "code")) if isinstance(getattr(current, "code", None), int) else None,
            "failure_stage": getattr(current, "failure_stage", None),
            "safe_detail": redact_text(getattr(current, "safe_detail", None)),
            "sanitized_error_body": redact_text(getattr(current, "sanitized_error_body", None)),
        })
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return result


def exception_diagnostic(error: BaseException) -> dict[str, Any]:
    formatted = "".join(traceback.TracebackException.from_exception(
        error, capture_locals=False).format(chain=True))
    return {
        "schema_version": "provider_exception_diagnostic_v1",
        "exception_chain": exception_chain(error),
        "traceback_redacted": redact_text(formatted),
        "credentials_or_headers_included": False,
    }
