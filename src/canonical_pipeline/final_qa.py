"""Canonical Task 7B v3 external-call and validator boundary."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from src.final_qa.task7b_gemini import build_request, call_with_one_technical_retry
from src.final_qa.task7b_v02 import FinalQAModelOutputV02, active_pipeline_uncertainties, validate_v02
from src.instrumentation.efficiency import gemini_usage_metrics

from .state import CaseState, ExecutionMode


def run_final_qa(state: CaseState, project_root: Path, client: Any | None = None) -> CaseState:
    """Execute canonical Task 7B v3 once, or explicitly skip in regression."""
    if state.final_payload is None:
        raise ValueError("Task 7A payload is required before final QA")
    if state.mode is ExecutionMode.REGRESSION_REPLAY:
        state.record("final_model_api", "skipped_regression_replay", external_calls=0)
        return state
    if client is None:
        raise ValueError("Live Task 7B v3 requires an injected google-genai client")
    build_started = time.perf_counter()
    request, manifest, delivered = build_request(state.final_payload, project_root, "gemini-3.5-flash", "low", True, response_model=FinalQAModelOutputV02, uncertainties_override=active_pipeline_uncertainties(state.final_payload))
    request_build_sec = time.perf_counter() - build_started
    call_started = time.perf_counter()
    outcome = call_with_one_technical_retry(client, request, response_model=FinalQAModelOutputV02)
    call_total_sec = time.perf_counter() - call_started
    state.external_calls["final_qa"] += len(outcome.attempts)
    if not outcome.infrastructure_succeeded or outcome.parsed is None:
        raise RuntimeError("Task 7B v3 external call failed after the allowed technical retry")
    parse_started = time.perf_counter()
    reparsed = FinalQAModelOutputV02.model_validate_json(outcome.raw_text)
    parse_sec = time.perf_counter() - parse_started
    validate_started = time.perf_counter()
    validated, report = validate_v02(reparsed, state.final_payload, delivered)
    validation_sec = time.perf_counter() - validate_started
    usage = gemini_usage_metrics(getattr(outcome.interaction, "usage", None))
    provider_wall = sum(float(item.get("latency_sec", 0.0)) for item in outcome.attempts)
    state.final_model_output = {
        "raw_text": outcome.raw_text,
        "request_manifest": manifest,
        "attempts": outcome.attempts,
        "raw_attempt_outputs": outcome.raw_attempt_outputs,
    }
    state.validated_answer = validated
    state.usage["task7b_v3_validator"] = report
    state.usage["task7b_v3_runtime"] = {
        "request_build_sec": request_build_sec,
        "final_model_api_stage_sec": call_total_sec,
        "final_model_api_wall_clock_sec": provider_wall,
        "structured_output_parse_sec": parse_sec,
        "local_validation_sec": validation_sec,
        "initial_api_calls": 1,
        "retry_api_calls": max(0, len(outcome.attempts) - 1),
        "total_api_calls": len(outcome.attempts),
        "usage": usage,
    }
    state.record("final_model_api", "task7b_v3_completed", external_calls=len(outcome.attempts))
    return state
