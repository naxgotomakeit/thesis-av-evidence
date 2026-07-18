"""Single canonical Task7A-to-Task7B v3 structured-output contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .task7b_v02 import FinalQAModelOutputV02


CANONICAL_OUTPUT_SCHEMA_VERSION = "task7b-v3-structured-output-v1"


def canonical_output_schema() -> dict[str, Any]:
    """Return the exact Pydantic schema supplied to Gemini by Task7B v3."""
    return FinalQAModelOutputV02.model_json_schema()


def canonical_output_schema_hash() -> str:
    """Hash the provider schema deterministically without runtime or secrets."""
    encoded = json.dumps(
        canonical_output_schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_required_output_schema(schema: dict[str, Any]) -> None:
    """Fail when Task7A embeds anything other than the Task7B v3 schema."""
    if schema != canonical_output_schema():
        raise ValueError("Task7A required_output_schema differs from canonical Task7B v3")
