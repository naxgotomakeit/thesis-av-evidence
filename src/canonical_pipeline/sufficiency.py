"""Canonical Task 5C v1.2 sufficiency and one-shot fallback.

Source lineage: final effective behavior developed through Task 5C v1/v1.1
and represented by v1.2. Historical versions are never runtime stages.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable

from src.retrieval.task5c_v1_1 import classify_v1_1, visual_accounting
from src.retrieval.task5c_v1_2 import (
    acoustic_evidence_role,
    broad_score_diagnostic,
    recompute_structural_status,
    timestamp_semantics,
)

from .state import CaseState, ExecutionMode


FallbackExecutor = Callable[[CaseState, dict[str, Any]], dict[str, Any]]


def _context(state: CaseState) -> dict[str, Any]:
    retrieval = state.retrieval_result or {}
    return {
        "case_id": state.case_id,
        "question": state.question,
        "selected_visual_evidence_frames": copy.deepcopy(retrieval.get("selected_visual_evidence_frames", [])),
        "anchor_resolution": copy.deepcopy(retrieval.get("anchor_resolution", {})),
        "local_visual_refinement": copy.deepcopy(retrieval.get("local_visual_refinement", {})),
    }


def _task5b_summary(state: CaseState) -> dict[str, Any]:
    retrieval = state.retrieval_result or {}
    return {
        "selected_candidates": copy.deepcopy(retrieval.get("selected_candidates", [])),
        "selected_visual_evidence_frames": copy.deepcopy(retrieval.get("selected_visual_evidence_frames", [])),
        "anchor_resolution": copy.deepcopy(retrieval.get("anchor_resolution", {})),
        "local_visual_refinement": copy.deepcopy(retrieval.get("local_visual_refinement", {})),
        "warnings": copy.deepcopy(retrieval.get("warnings", [])),
    }


def _diagnostics(state: CaseState, candidates: list[dict[str, Any]], frozen: dict[str, Any] | None) -> list[dict[str, Any]]:
    if frozen is not None:
        return copy.deepcopy(frozen.get("acoustic_evidence_diagnostics", []))
    plan = state.planner_output or {}
    semantics = timestamp_semantics(state.question, state.deterministic_cues, (state.retrieval_result or {}).get("anchor_resolution", {}))
    role = acoustic_evidence_role(plan)
    return [broad_score_diagnostic(candidate, role=role, semantics=semantics, local_audio_clip_reference=candidate.get("local_audio_clip_reference")) for candidate in candidates if candidate.get("modality") == "acoustic"]


def run_evidence_sufficiency(
    state: CaseState,
    *,
    frozen_v1_2: dict[str, Any] | None = None,
    fallback_executor: FallbackExecutor | None = None,
) -> CaseState:
    """Classify once and execute at most one fallback before final v1.2 status."""
    if state.retrieval_result is None or state.planner_output is None:
        raise ValueError("Retrieval and planner results are required before sufficiency")
    context = _context(state)
    pre_candidates = copy.deepcopy(state.retrieval_result.get("selected_candidates", []))
    pre_started = time.perf_counter()
    pre, _ = classify_v1_1(context, state.planner_output, state.deterministic_cues, pre_candidates)
    pre_duration = time.perf_counter() - pre_started
    decision_started = time.perf_counter()
    state.fallback_decision = {"required": bool(pre["fallback_required"]), "reason_codes": copy.deepcopy(pre["sufficiency_reason_codes"])}
    decision_duration = time.perf_counter() - decision_started
    fallback = {"triggered": False, "outcome": "not_triggered", "added_candidates": [], "model_calls": 0, "warnings": []}
    fallback_duration = None
    if pre["fallback_required"]:
        if state.fallback_execution_count:
            raise RuntimeError("Canonical Task 5C invariant violated: fallback attempted more than once")
        state.fallback_execution_count += 1
        if state.mode is ExecutionMode.REGRESSION_REPLAY:
            if frozen_v1_2 is None:
                raise ValueError("Regression fallback requires frozen Task 5C v1.2 evidence injection")
            fallback = copy.deepcopy(frozen_v1_2["source_task5c_v1_1"]["fallback"])
            fallback["canonical_execution"] = "frozen_fallback_evidence_injected_once"
        else:
            if fallback_executor is None:
                raise ValueError("Live fallback was required but no canonical fallback executor was supplied")
            fallback_started = time.perf_counter()
            fallback = fallback_executor(state, {"context": context, "pre_assessment": pre})
            fallback_duration = time.perf_counter() - fallback_started
            state.external_calls["fallback"] += int(fallback.get("model_calls", 0))
    post_candidates = pre_candidates + copy.deepcopy(fallback.get("added_candidates", []))
    post_started = time.perf_counter()
    post, _ = classify_v1_1(context, state.planner_output, state.deterministic_cues, post_candidates)
    diagnostics = _diagnostics(state, post_candidates, frozen_v1_2 if state.mode is ExecutionMode.REGRESSION_REPLAY else None)
    pre_final = recompute_structural_status(pre, diagnostics)
    post_final = recompute_structural_status(post, diagnostics)
    post_duration = time.perf_counter() - post_started
    additional = post_final["evidence_status"] == "insufficient"
    task5b_summary = _task5b_summary(state)
    result = {
        "case_id": state.case_id,
        "question": state.question,
        "execution_status": "complete",
        "task5a_plan_summary": copy.deepcopy(state.planner_output),
        "deterministic_question_cues": copy.deepcopy(state.deterministic_cues),
        "task5b_v1_1_input": task5b_summary,
        "source_task5c_v1_1": {"fallback": copy.deepcopy(fallback)},
        "pre_fallback_candidates": pre_candidates,
        "post_fallback_candidates": post_candidates,
        "pre_fallback_evidence_status": pre_final["evidence_status"],
        "pre_fallback_assessment": pre_final,
        "post_fallback_evidence_status": post_final["evidence_status"],
        "evidence_status": post_final["evidence_status"],
        "sufficiency_reason_codes": post_final["sufficiency_reason_codes"],
        "critical_missing_evidence": post_final["critical_missing_evidence"],
        "ambiguity_flags": post_final["ambiguity_flags"],
        "fallback_was_triggered": bool(fallback.get("triggered")),
        "fallback_recovered_evidence": bool(fallback.get("added_candidates")),
        "fallback_required": additional,
        "additional_fallback_required": additional,
        "questionable_followup_policy": "deferred_to_future_ablation",
        "automatic_additional_fallback_triggered": False,
        "acoustic_evidence_diagnostics": diagnostics,
        "local_audio_clips": copy.deepcopy((frozen_v1_2 or {}).get("local_audio_clips", [])),
        "visual_efficiency_accounting": visual_accounting(task5b_summary),
    }
    state.sufficiency_result = result
    state.fallback_result = fallback
    state.usage["task5c_v1_2_runtime"] = {
        "pre_classification_sec": pre_duration,
        "fallback_decision_sec": decision_duration,
        "fallback_execution_sec": fallback_duration,
        "fallback_local_asr_sec": fallback.get("latency_sec") if fallback.get("triggered") else None,
        "fallback_model_wall_clock_sec": (
            fallback.get("canonical_local_asr_total_sec", fallback.get("latency_sec"))
            if fallback.get("triggered") else None
        ),
        "post_fallback_processing_sec": post_duration,
        "fallback_model_calls": int(fallback.get("model_calls", 0)),
    }
    state.record("evidence_sufficiency", "task5c_v1_2_direct_final_behavior", fallback_execution_count=state.fallback_execution_count)
    return state
