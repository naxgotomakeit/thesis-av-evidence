from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.canonical.run_new_case_smoke import (
    SMOKE_CASES,
    _load_posthoc,
    validate_smoke_cases,
)
from src.canonical_pipeline.query_scoring import FreshQueryScorer
from src.canonical_pipeline.smoke_trace import (
    build_initial_retrieval_trace,
    build_temporal_linking_trace,
    normalized_timings,
)
from src.canonical_pipeline.state import CaseState, ExecutionMode


def _state() -> CaseState:
    state = CaseState(
        case_id="new_case",
        video_id="video",
        question="What happened at 00:01 - 00:03?",
        video_duration_sec=10.0,
        mode=ExecutionMode.EXECUTE_LIVE,
    )
    state.retrieval_result = {
        "question_conditioned_scoring": {
            "modalities": {
                "acoustic": {
                    "query_text": state.question,
                    "model": "test-clap",
                    "historical_score_dependency": False,
                    "all_scores_finite": True,
                    "ranked_results": [
                        {"rank": 1, "acoustic_region_id": "outside", "start_time": 8.0, "end_time": 9.0, "similarity_score": 0.9},
                        {"rank": 2, "acoustic_region_id": "inside", "start_time": 1.0, "end_time": 3.0, "similarity_score": 0.8},
                    ],
                    "top_k_results": [],
                }
            }
        },
        "anchor_resolution": {"final_search_intervals": [{"start_sec": 1.0, "end_sec": 3.0}]},
        "acoustic_candidates": [{"acoustic_region_id": "inside"}],
        "speech_candidates": [],
        "coarse_visual_candidates": [],
        "linked_windows": [],
    }
    state.timings = [
        {"stage_name": "online_end_to_end_total", "executed": True, "skipped": False, "duration_sec": 2.0},
        {"stage_name": "visual_retrieval", "executed": False, "skipped": True, "skip_reason": "modality_not_required", "duration_sec": None},
    ]
    state.usage["timing_consistency"] = {"uninstrumented_overhead_sec": 0.1}
    return state


def test_smoke_case_allowlist_is_exact() -> None:
    assert SMOKE_CASES == ("00002_1", "00018_3", "00018_9")


def test_preload_loads_each_requested_encoder_once(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = object.__new__(FreshQueryScorer)
    calls: list[str] = []
    monkeypatch.setattr(scorer, "_load_visual", lambda: (object(), calls.append("visual") or 1.0))
    monkeypatch.setattr(scorer, "_load_speech", lambda video_id: (object(), calls.append(f"speech:{video_id}") or 2.0))
    monkeypatch.setattr(scorer, "_load_acoustic", lambda video_id: (object(), object(), calls.append(f"acoustic:{video_id}") or 3.0))
    assert scorer.preload("00002") == {"visual": 1.0, "speech": 2.0, "acoustic": 3.0}
    assert calls == ["visual", "speech:00002", "acoustic:00002"]


def test_initial_candidates_are_not_labelled_selected_evidence() -> None:
    trace = build_initial_retrieval_trace(_state())
    assert "initial_retrieval_candidates" in trace["modalities"]["acoustic"]
    assert "selected_evidence" not in trace["modalities"]["acoustic"]
    assert trace["modalities"]["acoustic"]["historical_task4_score_dependency"] is False


def test_temporal_trace_shows_semantic_top_one_dropped() -> None:
    rows = build_temporal_linking_trace(_state())["candidate_temporal_decisions"]["acoustic"]
    assert rows[0]["rank"] == 1
    assert rows[0]["decision"] == "dropped"
    assert rows[0]["decision_reason"] == "outside_executed_search_interval"
    assert rows[1]["decision"] == "kept"


def test_skipped_stage_duration_stays_null() -> None:
    rows = {item["stage_name"]: item for item in normalized_timings(_state())}
    assert rows["visual_retrieval"]["skipped"] is True
    assert rows["visual_retrieval"]["duration_sec"] is None


def test_gold_load_is_blocked_before_prediction_persistence() -> None:
    manifest = Path("outputs/pilot_20/baseline_v1/new_case_smoke_v0_1/test_manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest.write_text(json.dumps([{"case_id": "new_case", "answer": "gold"}]), encoding="utf-8")
        with pytest.raises(RuntimeError, match="Gold load blocked"):
            _load_posthoc(manifest, [{"case_id": "new_case", "validated_answer": None, "raw_model_output": None}])
    finally:
        manifest.unlink(missing_ok=True)


def test_validation_uses_manifest_cases_without_gold(monkeypatch: pytest.MonkeyPatch) -> None:
    class Runner:
        query_scorer = object()

        @staticmethod
        def validate_case(case_id: str) -> dict[str, object]:
            return {
                "case_id": case_id, "video_id": case_id.split("_")[0], "question": "q",
                "source_video_path": __file__, "source_audio_path": __file__,
            }

    rows = validate_smoke_cases(Runner())
    assert [row["case_id"] for row in rows] == list(SMOKE_CASES)
    assert all("answer" not in row and "gold" not in row for row in rows)
    assert all(row["historical_task4_execution_dependency"] is False for row in rows)
