from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.canonical_pipeline.state import CaseState, ExecutionMode
from src.evaluation.egoschema_adapter import EgoSchemaRuntimeCase
from src.evaluation.egoschema_runtime import (
    VisualOnlyFallback,
    apply_dataset_modality_availability,
    mc_accuracy_summary,
    reveal_options_for_final_qa,
    validate_multiple_choice_answer,
)


def _state() -> CaseState:
    state = CaseState(
        case_id="case_1",
        video_id="video_1",
        question="What happened?",
        video_duration_sec=180.0,
        mode=ExecutionMode.EXECUTE_LIVE,
        source_video_path=str(Path("video.mp4")),
        source_audio_path=None,
        available_modalities=("visual",),
    )
    state.planner_output = {
        "answer_requirement": {"operation": "describe_action", "description": "action"},
        "anchor_cues": [],
        "primary_anchor_modality": "speech",
        "resolver_modalities": ["speech", "visual"],
        "audio_role": "supporting_evidence",
        "temporal_relation": "none",
        "requires_local_visual_inspection": True,
        "visual_route": "coarse_then_local_refinement",
        "fallback_route": "existing policy",
        "planner_confidence": 0.8,
        "rationale": "test",
    }
    return state


def _case() -> EgoSchemaRuntimeCase:
    return EgoSchemaRuntimeCase(
        case_id="case_1",
        video_id="video_1",
        question="What happened?",
        options=("a", "b", "c", "d", "e"),
        video_path="sample_500/video_1.mp4",
        duration_sec=180.0,
        audio_available=False,
    )


def test_visual_only_projection_preserves_raw_plan_and_does_not_add_audio() -> None:
    state = _state()
    original = copy.deepcopy(state.planner_output)
    apply_dataset_modality_availability(state)
    assert state.planner_output["resolver_modalities"] == ["visual"]
    assert state.planner_output["primary_anchor_modality"] == "visual"
    assert state.usage["dataset_modality_availability"]["planner_output_before_projection"] == original
    assert state.planner_output["requires_local_visual_inspection"] is True
    assert state.planner_output["visual_route"] == "coarse_then_local_refinement"


def test_options_are_revealed_only_after_task7a_and_evidence_is_unchanged() -> None:
    state = _state()
    state.final_payload = {
        "case_id": "case_1",
        "question": "What happened?",
        "evidence_groups": [{"group_id": "g1", "visual_evidence": [{"evidence_id": "v1"}]}],
    }
    state.preflight_result = {"status": "ready"}
    evidence = copy.deepcopy(state.final_payload["evidence_groups"])
    assert "Option 0" not in state.final_payload["question"]
    reveal_options_for_final_qa(state, _case())
    assert "Option 0: a" in state.final_payload["question"]
    assert state.final_payload["evidence_groups"] == evidence
    assert state.final_payload["multiple_choice_protocol"]["gold_available"] is False


def test_mc_validation_requires_exact_provider_contract() -> None:
    validated = {"answer": "OPTION_INDEX=3", "answer_status": "answered"}
    result = validate_multiple_choice_answer(validated, _case().options)
    assert result["selected_option_index"] == 3
    assert result["selected_option_text"] == "d"
    assert result["predicted_option_index"] == 3
    assert result["predicted_option_text"] == "d"
    assert result["mc_abstained"] is False
    with pytest.raises(ValueError):
        validate_multiple_choice_answer({"answer": "probably option 3"}, _case().options)


def test_mc_legal_abstention_and_invalid_answered_null() -> None:
    result = validate_multiple_choice_answer(
        {
            "answer": None,
            "answer_status": "insufficient_evidence",
            "abstain": True,
        },
        _case().options,
    )
    assert result["predicted_option_index"] is None
    assert result["predicted_option_text"] is None
    assert result["mc_abstained"] is True
    with pytest.raises(ValueError):
        validate_multiple_choice_answer(
            {"answer": None, "answer_status": "answered", "abstain": False},
            _case().options,
        )


def test_mc_accuracy_counts_abstention_incorrect_and_reports_it_separately() -> None:
    summary = mc_accuracy_summary(
        [
            {"predicted_option_index": 1, "correct": True},
            {"predicted_option_index": 3, "correct": False},
            {"predicted_option_index": None, "correct": False},
        ]
    )
    assert summary["mc_accuracy"] == pytest.approx(1 / 3)
    assert summary["abstention_count"] == 1
    assert summary["abstention_rate"] == pytest.approx(1 / 3)
    assert summary["answered_only_accuracy"] == pytest.approx(1 / 2)


def test_visual_only_fallback_never_invokes_whisper_or_adds_evidence() -> None:
    fallback = VisualOnlyFallback()
    result = fallback.execute(_state(), {})
    assert result["model_calls"] == 0
    assert result["added_candidates"] == []
    assert result["provenance"]["whisper_invoked"] is False
    assert fallback.lifecycle_audit()["whisper_inference_calls"] == 0
