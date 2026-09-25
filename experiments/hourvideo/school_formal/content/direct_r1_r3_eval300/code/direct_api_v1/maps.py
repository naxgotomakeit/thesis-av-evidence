"""Validated, no-gold map/question loading for a Direct v1 session."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from direct_api_prep.evidence import assert_safe_direct_evidence, sha256_file


class DirectInputError(ValueError):
    pass


FORBIDDEN_QUESTION_KEYS = {"correct_answer", "gold_answer", "answer_key", "winning_option"}


@dataclass(frozen=True)
class DirectInput:
    question: dict[str, Any]
    method: str
    map_document: dict[str, Any]
    map_path: str
    map_sha256: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DirectInputError(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise DirectInputError(f"JSON root must be an object: {path}")
    return value


def load_direct_input(*, question_path: Path, map_path: Path, method: str, expected_map_sha256: str | None = None) -> DirectInput:
    if method not in {"R1", "R3"}:
        raise DirectInputError("Direct map mode supports R1 or R3")
    question = _load_json(question_path)
    if set(question) & FORBIDDEN_QUESTION_KEYS:
        raise DirectInputError("gold/answer field cannot enter a Direct request")
    try:
        assert_safe_direct_evidence(question)
    except ValueError as error:
        raise DirectInputError("gold/staged content cannot enter a Direct request") from error
    options = question.get("answer_options")
    if not isinstance(question.get("question_id"), str) or not isinstance(question.get("question_text"), str):
        raise DirectInputError("question identity/text missing")
    if not isinstance(options, list) or len(options) != 5 or {row.get("option_id") for row in options if isinstance(row, dict)} != {"A", "B", "C", "D", "E"}:
        raise DirectInputError("Direct requires exactly options A/B/C/D/E")
    map_document = _load_json(map_path)
    assert_safe_direct_evidence(map_document)
    digest = sha256_file(map_path)
    if expected_map_sha256 is not None and digest != expected_map_sha256:
        raise DirectInputError("frozen map SHA mismatch")
    return DirectInput(question, method, map_document, str(map_path), digest)
