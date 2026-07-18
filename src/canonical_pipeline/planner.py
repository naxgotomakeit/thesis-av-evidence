"""Canonical Task 5A v2 per-question boundary.

Source lineage: ``src.question_planner.task5a``. Regression accepts an exact
frozen plan; live callers inject one external request function.
"""

from __future__ import annotations

import copy
from typing import Callable

from src.question_planner.task5a import ModelReply, extract_cues, plan_with_retry

from .state import CaseState, ExecutionMode


def run_planner(
    state: CaseState,
    *,
    frozen_record: dict | None = None,
    request: Callable[[str, str, int], ModelReply] | None = None,
) -> CaseState:
    """Populate Task 5A v2 plan exactly once, with an explicit external boundary."""
    state.deterministic_cues = extract_cues(state.question)
    if state.mode is ExecutionMode.REGRESSION_REPLAY:
        if frozen_record is None:
            raise ValueError("Regression replay requires a frozen Task 5A v2 record")
        state.planner_output = copy.deepcopy(frozen_record["plan"])
        state.deterministic_cues = copy.deepcopy(frozen_record["deterministic_cues"])
        state.record("question_planner", "frozen_v2_plan_injected", external_calls=0)
        return state
    if request is None:
        raise ValueError("Live planner execution requires an injected Task 5A request function")
    outcome = plan_with_retry(state.question, state.deterministic_cues, request)
    if outcome["validation_status"] != "valid":
        raise RuntimeError("Task 5A v2 planner failed strict validation")
    state.planner_output = outcome["plan"]
    state.external_calls["planner"] += len(outcome["attempts"])
    state.usage["task5a_v2"] = {
        "attempts": outcome["attempts"],
        "api_calls": len(outcome["attempts"]),
        "retries": max(0, len(outcome["attempts"]) - 1),
        "planner_api_wall_clock_sec": sum(float(item["latency_sec"]) for item in outcome["attempts"]),
        "input_tokens": sum(int(item["input_tokens"]) for item in outcome["attempts"]),
        "output_tokens": sum(int(item["output_tokens"]) for item in outcome["attempts"]),
    }
    state.record("question_planner", "live_v2_plan_completed", external_calls=len(outcome["attempts"]))
    return state
