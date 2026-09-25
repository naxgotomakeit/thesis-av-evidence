"""Deterministic diagnostic classification for GenS answering outputs.

This module never extracts an answer choice from explanatory text.  Only the
frozen strict parser establishes ``valid_answer``.
"""
from __future__ import annotations

import re
from typing import Any

ABSTENTION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bi (?:can(?:not|'t)|am unable to) (?:accurately |reliably )?(?:determine|answer|choose|provide)",
    r"\b(?:cannot|can't|unable to) (?:accurately |reliably )?(?:determine|answer|choose|provide)",
    r"\b(?:insufficient|not enough|inadequate) (?:visual )?(?:evidence|information|context)",
    r"\bdo not have (?:enough|sufficient) (?:visual )?(?:evidence|information|context)",
    r"\bwithout (?:the |enough |sufficient )?(?:video|evidence|information|context).{0,100}\b(?:cannot|can't|unable)",
    r"\bno (?:reliable|confident) (?:answer|choice|determination)\b",
))


def classify_result(*, raw_response_text: str | None, parsed_prediction: str | None,
                    runtime_failure: bool = False) -> dict[str, Any]:
    if parsed_prediction in {"A", "B", "C", "D", "E"}:
        return {"result_class": "valid_answer", "manual_review_required": False,
                "matched_abstention_rule": None}
    if runtime_failure:
        return {"result_class": "runtime_failure", "manual_review_required": False,
                "matched_abstention_rule": None}
    text = raw_response_text or ""
    for index, pattern in enumerate(ABSTENTION_PATTERNS):
        if pattern.search(text):
            return {"result_class": "abstention", "manual_review_required": False,
                    "matched_abstention_rule": index}
    return {"result_class": "invalid_format", "manual_review_required": True,
            "matched_abstention_rule": None}
