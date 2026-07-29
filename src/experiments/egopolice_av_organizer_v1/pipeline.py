from __future__ import annotations

from typing import Any

from .validation import validate_evidence_map, validate_hashes, validate_presenter


def verify_frozen_baseline() -> dict[str, Any]:
    """Verify existing frozen artifacts without model or network calls."""
    checks = {
        "artifact_hashes": validate_hashes(),
        "organizer": validate_evidence_map(),
        "presenter": validate_presenter(),
    }
    return {
        "status": "passed" if all(x["valid"] for x in checks.values()) else "failed",
        "api_calls": 0,
        "checks": checks,
    }
