"""Minimal, task-agnostic claim-level AV sufficiency contract V3."""

from .core import (
    build_legacy_adapter_view,
    build_v3_input,
    project_semantic_payload,
    semantic_schema,
    validate_semantic_payload,
)

__all__ = [
    "build_legacy_adapter_view",
    "build_v3_input",
    "project_semantic_payload",
    "semantic_schema",
    "validate_semantic_payload",
]
