"""Extensible, stage-neutral metric registry for pilot evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .answer_metrics import normalize_answer, sentence_bleu, single_reference_cider


MetricFunction = Callable[[dict[str, Any], dict[str, Any]], Any]


@dataclass(frozen=True)
class MetricResult:
    """One metric value and its explicit interpretation boundary."""

    name: str
    value: Any
    available: bool
    category: str
    interpretation: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    category: str
    interpretation: str
    function: MetricFunction | None


class MetricRegistry:
    """Register metrics without coupling computation to HTML rendering."""

    def __init__(self) -> None:
        self._definitions: dict[str, MetricDefinition] = {}

    def register(self, definition: MetricDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Metric already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def evaluate(self, case: dict[str, Any], context: dict[str, Any] | None = None) -> list[MetricResult]:
        results: list[MetricResult] = []
        shared = context or {}
        for definition in self._definitions.values():
            if definition.function is None:
                results.append(MetricResult(definition.name, None, False, definition.category, definition.interpretation))
                continue
            try:
                value = definition.function(case, shared)
                results.append(MetricResult(definition.name, value, True, definition.category, definition.interpretation))
            except Exception as exc:  # metric failures must not fail a completed QA run
                results.append(MetricResult(definition.name, None, False, definition.category, definition.interpretation, type(exc).__name__))
        return results

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)


DIAGNOSTIC = "diagnostic_reference_overlap"


def baseline_metric_registry() -> MetricRegistry:
    """Return Baseline v1 lexical diagnostics plus future metric placeholders."""
    registry = MetricRegistry()
    registry.register(MetricDefinition("exact_match", DIAGNOSTIC, "Case-sensitive trimmed string identity; not authoritative open-ended QA accuracy.", lambda case, _: str(case.get("gold_reference_answer") or "").strip() == str(case.get("validated_prediction") or "").strip()))
    registry.register(MetricDefinition("normalized_exact_match", DIAGNOSTIC, "Case-folded punctuation-normalized identity; not semantic correctness.", lambda case, _: normalize_answer(case.get("gold_reference_answer")) == normalize_answer(case.get("validated_prediction"))))
    registry.register(MetricDefinition("bleu", DIAGNOSTIC, "Sentence BLEU-4 with method-1-style smoothing; diagnostic lexical overlap only.", lambda case, _: sentence_bleu(str(case.get("gold_reference_answer") or ""), str(case.get("validated_prediction") or ""))))
    registry.register(MetricDefinition("cider", DIAGNOSTIC, "Exploratory single-reference CIDEr-like score; diagnostic lexical overlap only.", lambda case, context: single_reference_cider(str(case.get("gold_reference_answer") or ""), str(case.get("validated_prediction") or ""), list(context["reference_corpus"]))))
    registry.register(MetricDefinition("semantic_similarity", "semantic_correctness", "Unavailable in Baseline v1; reserved for a future approved semantic metric.", None))
    registry.register(MetricDefinition("llm_judge", "human_or_judge", "Unavailable in Baseline v1; no LLM judge is called in this pilot scaffold.", None))
    registry.register(MetricDefinition("task_specific_metric", "task_specific", "Unavailable in Baseline v1; reserved for operation-specific evaluation.", None))
    return registry

