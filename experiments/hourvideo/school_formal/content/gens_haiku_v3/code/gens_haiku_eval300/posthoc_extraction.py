"""Conservative, no-gold post-hoc extraction for frozen GenS responses.

This is deliberately separate from the formal strict parser.  It recognizes
only an explicit answer declaration/final choice or an answer-position label;
it never maps prose semantics to an option and never chooses among conflicts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .classification import ABSTENTION_PATTERNS

RULE_VERSION = "gens_haiku_posthoc_conservative_extraction_v1"
CHOICES = frozenset("ABCDE")


@dataclass(frozen=True)
class Candidate:
    answer: str
    rule: str
    start: int
    end: int


# These patterns require answer/choice language.  The captured letter must be
# an isolated option label, not merely the first A--E occurring in prose.
DECLARATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("explicit_answer_declaration", re.compile(
        r"\b(?:the\s+)?(?:correct\s+|best\s+|final\s+|most\s+(?:likely|reasonable)\s+)?"
        r"answer\s*(?:is|would\s+be|appears\s+to\s+be|:)\s*"
        r"(?:option\s*)?(?:\*\*)?\(?([A-E])(?=[\s\).,:*\-]|$)", re.I)),
    ("explicit_choice_declaration", re.compile(
        r"\b(?:i\s+(?:(?:would|will|must|have\s+to)\s+)?(?:choose|select)|"
        r"i\s+would\s+say|my\s+(?:final\s+)?choice\s+(?:is|would\s+be))\s*"
        r"(?:option\s*)?(?:\*\*)?\(?([A-E])(?=[\s\).,:*\-]|$)", re.I)),
    ("explicit_option_conclusion", re.compile(
        r"\b(?:the\s+)?(?:best|correct|most\s+(?:likely|reasonable|appropriate))\s+"
        r"(?:option|choice)\s*(?:is|would\s+be|:)\s*"
        r"(?:option\s*)?(?:\*\*)?\(?([A-E])(?=[\s\).,:*\-]|$)", re.I)),
)

PURE_LABEL_LINE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?\(?([A-E])\)?[\.:]?\s*(?:\*\*)?\s*$",
    re.I,
)
LEADING_ANSWER_LABEL = re.compile(
    r"\A\s*(?:#{1,6}\s*)?(?:answer\s*:\s*)?(?:\*\*)?\(?([A-E])\)?[\.:]"
    r"(?:\s|\*\*)+\S",
    re.I,
)


def _snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    lo, hi = max(0, start - radius), min(len(text), end + radius)
    value = text[lo:hi].replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def explicit_candidates(text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for rule, pattern in DECLARATION_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append(Candidate(match.group(1).upper(), rule, match.start(), match.end()))
    for match in PURE_LABEL_LINE.finditer(text):
        candidates.append(Candidate(match.group(1).upper(), "standalone_answer_label", match.start(), match.end()))
    match = LEADING_ANSWER_LABEL.search(text)
    if match:
        candidates.append(Candidate(match.group(1).upper(), "leading_answer_label", match.start(), match.end()))
    # Deduplicate overlapping detections of the same declaration while keeping
    # distinct textual evidence and any conflicting letters visible.
    unique: dict[tuple[str, int, int], Candidate] = {
        (item.answer, item.start, item.end): item for item in candidates
    }
    return sorted(unique.values(), key=lambda item: (item.start, item.end, item.rule))


def extract_posthoc(*, raw_response_text: str | None,
                    strict_prediction: str | None,
                    runtime_failure: bool = False) -> dict[str, Any]:
    text = raw_response_text or ""
    if strict_prediction in CHOICES:
        return {
            "new_class": "strict_valid",
            "extracted_answer": strict_prediction,
            "matching_rule": "frozen_strict_parser",
            "evidence_snippet": text.strip(),
            "manual_review_required": False,
            "candidate_answers": [strict_prediction],
        }
    if runtime_failure:
        return {
            "new_class": "ambiguous_or_unextractable",
            "extracted_answer": None,
            "matching_rule": "runtime_failure_no_response_extraction",
            "evidence_snippet": "",
            "manual_review_required": True,
            "candidate_answers": [],
        }

    candidates = explicit_candidates(text)
    answers = sorted({item.answer for item in candidates})
    if len(answers) == 1:
        evidence = next(item for item in candidates if item.answer == answers[0])
        return {
            "new_class": "explicit_answer_format_only",
            "extracted_answer": answers[0],
            "matching_rule": evidence.rule,
            "evidence_snippet": _snippet(text, evidence.start, evidence.end),
            "manual_review_required": False,
            "candidate_answers": answers,
        }
    if len(answers) > 1:
        return {
            "new_class": "ambiguous_or_unextractable",
            "extracted_answer": None,
            "matching_rule": "conflicting_explicit_candidates",
            "evidence_snippet": " | ".join(_snippet(text, c.start, c.end, 35) for c in candidates[:4]),
            "manual_review_required": True,
            "candidate_answers": answers,
        }

    for index, pattern in enumerate(ABSTENTION_PATTERNS):
        match = pattern.search(text)
        if match:
            return {
                "new_class": "abstention",
                "extracted_answer": None,
                "matching_rule": f"conservative_abstention_{index}",
                "evidence_snippet": _snippet(text, match.start(), match.end()),
                "manual_review_required": False,
                "candidate_answers": [],
            }
    return {
        "new_class": "ambiguous_or_unextractable",
        "extracted_answer": None,
        "matching_rule": "no_conservative_rule_match",
        "evidence_snippet": text.strip()[:240],
        "manual_review_required": True,
        "candidate_answers": [],
    }


def frozen_rule_spec() -> dict[str, Any]:
    return {
        "rule_version": RULE_VERSION,
        "precedence": [
            "frozen strict parser",
            "unique explicit answer declaration/final choice/answer-position label",
            "conflicting explicit candidates remain unextractable",
            "explicit abstention with no unique final choice",
            "otherwise ambiguous_or_unextractable and manual review",
        ],
        "declaration_patterns": [{"name": name, "pattern": pattern.pattern}
                                 for name, pattern in DECLARATION_PATTERNS],
        "standalone_answer_label_pattern": PURE_LABEL_LINE.pattern,
        "leading_answer_label_pattern": LEADING_ANSWER_LABEL.pattern,
        "abstention_patterns": [pattern.pattern for pattern in ABSTENTION_PATTERNS],
        "boundaries": {
            "never": ["first A-E in prose", "option discussion", "excluded option",
                      "semantic mapping", "gold-informed inference"],
            "accept": ["The answer is B.", "(B) on its own line", "B. followed by explanation",
                       "uncertainty plus an explicit unique final choice"],
            "reject": ["multiple conflicting final choices", "uncertainty without a choice",
                       "option discussion without a final declaration"],
        },
    }
