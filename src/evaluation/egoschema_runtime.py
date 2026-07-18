"""EgoSchema-only protocol adapters for the frozen canonical baseline.

The helpers in this module never rank, retain, drop, or otherwise select
evidence. They make dataset modality availability explicit and reveal the five
multiple-choice options only after Task 7A has finalized model-facing evidence.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from src.canonical_pipeline.state import CaseState

from .egoschema_adapter import EgoSchemaAdapterError, EgoSchemaRuntimeCase


MC_MARKER = re.compile(r"^\s*OPTION_INDEX\s*=\s*([0-4])\s*$", re.IGNORECASE)


def apply_dataset_modality_availability(state: CaseState) -> None:
    """Project a live planner result onto physically available modalities.

    The original plan is retained in the audit. No ranking, route score, or
    evidence budget is changed. For a visual-only dataset, unavailable audio
    resolver/anchor labels cannot create nonexistent audio branches.
    """
    if state.planner_output is None:
        raise ValueError("Planner output is required before availability projection")
    available = set(state.available_modalities)
    before = copy.deepcopy(state.planner_output)
    after = copy.deepcopy(before)
    requested = list(after.get("resolver_modalities", []))
    after["resolver_modalities"] = [item for item in requested if item in available]
    unavailable_requested = [item for item in requested if item not in available]

    # EgoSchema videos are visual-only. If an otherwise valid generic AV plan
    # asks exclusively for an unavailable channel, the dataset adapter exposes
    # visual as the only executable resolver. It does not alter visual routing,
    # local-refinement flags, ranking, or budget policy.
    availability_fallback_added = False
    if not after["resolver_modalities"] and "visual" in available:
        after["resolver_modalities"] = ["visual"]
        availability_fallback_added = True
    primary = after.get("primary_anchor_modality")
    if primary in {"visual", "speech", "acoustic"} and primary not in available:
        after["primary_anchor_modality"] = "visual" if "visual" in available else "unknown"

    state.planner_output = after
    state.usage["dataset_modality_availability"] = {
        "available_modalities": sorted(available),
        "planner_output_before_projection": before,
        "planner_output_after_projection": copy.deepcopy(after),
        "unavailable_requested_modalities": unavailable_requested,
        "availability_fallback_added_visual_resolver": availability_fallback_added,
        "selection_or_ranking_policy_changed": False,
    }
    state.record(
        "dataset_modality_availability",
        "unavailable_dataset_modalities_excluded_before_retrieval",
        available_modalities=sorted(available),
        unavailable_requested_modalities=unavailable_requested,
    )


def multiple_choice_question(case: EgoSchemaRuntimeCase) -> str:
    """Create the final-only MC instruction; gold is not an input."""
    options = "\n".join(
        f"Option {index}: {text}" for index, text in enumerate(case.options)
    )
    return (
        f"{case.question}\n\n"
        "Choose exactly one of the five answer options using only the supplied "
        "visual evidence. In the structured output `answer` field, return "
        "exactly OPTION_INDEX=n, where n is 0, 1, 2, 3, or 4. Do not answer "
        "with option text or any additional text in the `answer` field.\n\n"
        f"{options}"
    )


def reveal_options_for_final_qa(state: CaseState, case: EgoSchemaRuntimeCase) -> None:
    """Reveal options after Task 7A without changing its selected evidence."""
    if state.final_payload is None or state.preflight_result is None:
        raise ValueError("Task 7A payload/preflight must finish before options are revealed")
    if state.case_id != case.case_id:
        raise EgoSchemaAdapterError("Final-selection case identity mismatch")
    evidence_before = copy.deepcopy(state.final_payload["evidence_groups"])
    state.final_payload["question"] = multiple_choice_question(case)
    state.final_payload["multiple_choice_protocol"] = {
        "protocol": "protocol_b_question_only_retrieval",
        "options_revealed_after_task7a": True,
        "option_count": 5,
        "gold_available": False,
        "required_answer_marker": "OPTION_INDEX=n",
    }
    if state.final_payload["evidence_groups"] != evidence_before:
        raise RuntimeError("MC adapter changed Task 7A evidence")
    state.record(
        "egoschema_final_selection_adapter",
        "options_revealed_after_task7a",
        evidence_changed=False,
        gold_available=False,
    )


def validate_multiple_choice_answer(
    validated_answer: dict[str, Any], options: tuple[str, str, str, str, str]
) -> dict[str, Any]:
    """Map the frozen final-QA answer field to one option without semantics."""
    value = validated_answer.get("answer")
    status = validated_answer.get("answer_status")
    non_answer_statuses = {
        "insufficient_evidence",
        "query_or_premise_inconsistent",
    }
    if value is None:
        if status not in non_answer_statuses:
            raise EgoSchemaAdapterError(
                "A null MC answer is legal only for an existing non-answer status"
            )
        result = copy.deepcopy(validated_answer)
        result["selected_option_index"] = None
        result["selected_option_text"] = None
        result["predicted_option_index"] = None
        result["predicted_option_text"] = None
        result["mc_abstained"] = True
        result["mc_contract_valid"] = True
        return result
    match = MC_MARKER.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise EgoSchemaAdapterError(
            "Final answer did not satisfy the deterministic OPTION_INDEX=n contract"
        )
    index = int(match.group(1))
    result = copy.deepcopy(validated_answer)
    result["selected_option_index"] = index
    result["selected_option_text"] = options[index]
    result["predicted_option_index"] = index
    result["predicted_option_text"] = options[index]
    result["mc_abstained"] = False
    result["mc_contract_valid"] = True
    return result


def mc_accuracy_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Report standard and answered-only MC metrics without judging abstentions."""
    if not evaluations:
        raise ValueError("MC accuracy requires at least one evaluation")
    abstentions = sum(item.get("predicted_option_index") is None for item in evaluations)
    correct = sum(bool(item.get("correct")) for item in evaluations)
    answered = len(evaluations) - abstentions
    return {
        "case_count": len(evaluations),
        "correct": correct,
        "mc_accuracy": correct / len(evaluations),
        "abstention_count": abstentions,
        "abstention_rate": abstentions / len(evaluations),
        "answered_case_count": answered,
        "answered_only_accuracy": correct / answered if answered else None,
    }


class VisualOnlyFallback:
    """Represent an unavailable audio fallback without invoking a model."""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, state: CaseState, context: dict[str, Any]) -> dict[str, Any]:
        del context
        self.calls += 1
        if set(state.available_modalities) != {"visual"}:
            raise RuntimeError("VisualOnlyFallback may only serve a visual-only dataset")
        return {
            "triggered": True,
            "outcome": "dataset_audio_unavailable_no_model_call",
            "added_candidates": [],
            "model_calls": 0,
            "latency_sec": 0.0,
            "warnings": ["fallback_audio_modality_unavailable_in_dataset"],
            "provenance": {
                "dataset_modality_availability": ["visual"],
                "whisper_invoked": False,
            },
        }

    def lifecycle_audit(self) -> dict[str, Any]:
        return {
            "whisper_load_count": 0,
            "whisper_inference_calls": 0,
            "fallback_decisions_served": self.calls,
        }


def project_relative_or_absolute(path: Path, project_root: Path) -> str:
    """Return a stable project-relative asset path when possible."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
