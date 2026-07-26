from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.experiments.qaego4d_e1.core import (
    CONTEXT_SCOPE,
    CanonicalCase,
    build_prompt,
    deterministic_closed_options,
    _select_nearest_interval_candidate,
    decode_zero_decodable_interval_nearest_frame,
    load_cases,
    oracle_requested_timestamps,
    uniform_requested_timestamps,
    valid_result,
)
from src.experiments.qaego4d_e1.runner import _record
from src.experiments.qaego4d_e1.formal_core import CORE_CONDITIONS, _valid_core_record


def _case(*, task: str = "closed", start: float = 12.0, end: float = 16.0) -> CanonicalCase:
    return CanonicalCase(
        task=task,
        question_id="sample-1",
        clip_uid="clip", video_uid="video",
        clip_start_sec=10.0, clip_end_sec=20.0,
        parent_video_path=Path("/tmp/video.mp4"),
        question="Where is the cup?", answer="table",
        wrong_answers=("floor", "drawer", "shelf") if task == "closed" else (),
        evidence_start_sec=start, evidence_end_sec=end,
    )


def test_uniform_temporal_bin_centres_stay_strictly_inside_canonical_clip():
    timestamps = uniform_requested_timestamps(10.0, 20.0, budget=8, nominal_fps=30.0)
    assert len(timestamps) == 8
    assert timestamps == sorted(timestamps)
    assert all(10.0 <= value < 20.0 for value in timestamps)


def test_uniform_budget_reduces_to_available_nominal_frames():
    timestamps = uniform_requested_timestamps(10.0, 10.05, budget=32, nominal_fps=30.0)
    assert len(timestamps) == 1
    assert 10.0 <= timestamps[0] < 10.05


def test_interval_oracle_is_limited_to_evidence_and_clip():
    case = _case(start=12.0, end=16.0)
    timestamps, rule, bounds = oracle_requested_timestamps(case, budget=8, nominal_fps=30.0)
    assert rule == "interval_uniform_within_gt"
    assert bounds == (12.0, 16.0)
    assert 1 <= len(timestamps) <= 8
    assert all(12.0 <= value < 16.0 for value in timestamps)


def test_point_oracle_is_provisionally_retained_with_one_frame():
    case = _case(start=15.0, end=15.0)
    timestamps, rule, bounds = oracle_requested_timestamps(case, budget=8, nominal_fps=30.0)
    assert rule == "point_nearest_decodable_frame_provisional"
    assert timestamps == [15.0]
    # Point evidence has no non-empty interval; decoding is constrained to
    # the canonical clip while selecting the nearest real frame to the point.
    assert bounds == (10.0, 20.0)


def test_zero_decodable_interval_selects_exactly_one_nearest_candidate():
    before = SimpleNamespace(time=9.99)
    after = SimpleNamespace(time=10.04)
    selected = _select_nearest_interval_candidate([before, after], start_sec=10.0, end_sec=10.03)
    assert selected is before


def test_zero_decodable_interval_exact_tie_selects_earlier_candidate():
    before = SimpleNamespace(time=9.99)
    after = SimpleNamespace(time=10.03)
    selected = _select_nearest_interval_candidate([after, before], start_sec=10.0, end_sec=10.02)
    assert selected is before


def test_real_zero_decodable_open_case_uses_canonical_nearest_frame():
    repo = Path(__file__).resolve().parents[2]
    annotation_root = Path("/cs/student/project_msc/2025/rai/xinanx01/QaEgo4D/processed/data/unified")
    cases, _ = load_cases(
        task="open",
        annotation_root=annotation_root,
        manifest_path=repo / "configs/eval_manifests/qaego4d_open_eval_ids.json",
        mapping_path=repo / "configs/qaego4d/qaego4d_open_canonical_mapping.json",
    )
    case = next(item for item in cases if item.question_id == "c411d1a4-451d-41d2-b09d-792983808864_5")
    images, records, _, _ = decode_zero_decodable_interval_nearest_frame(case=case)
    try:
        assert len(images) == len(records) == 1
        assert abs(records[0]["actual_timestamp_sec"] - 2615.633333333333) < 1e-6
        assert case.clip_start_sec <= records[0]["actual_timestamp_sec"] < case.clip_end_sec
    finally:
        for image in images:
            image.close()


def test_open_and_closed_prompts_are_distinct_and_options_are_deterministic():
    closed = _case()
    open_case = _case(task="open")
    open_prompt, _, _ = build_prompt(open_case, open_template="OPEN:{question}", closed_template="CLOSED:{question}:{options}")
    closed_prompt, options, answer_index = build_prompt(closed, open_template="OPEN:{question}", closed_template="CLOSED:{question}:{options}")
    options_again, answer_index_again = deterministic_closed_options(closed)
    assert open_prompt.startswith("OPEN:")
    assert closed_prompt.startswith("CLOSED:")
    assert options == options_again
    assert answer_index == answer_index_again
    assert options[answer_index] == "table"


def test_complete_checkpoint_contract_is_strict():
    record = {
        "clip_uid": "clip", "video_uid": "video", "context_scope": CONTEXT_SCOPE,
        "video_decode_time_s": 0.0, "frames_decoded_online": 0, "frames_shown": 0,
            "answer_model_time_s": 1.0, "total_latency_s": 1.0, "model_calls": 1,
            "config_hash": "config", "manifest_hash": "manifest", "success": True,
            "cuda_memory": {},
    }
    assert valid_result(record, config_hash="config", manifest_hash="manifest")
    del record["frames_shown"]
    assert not valid_result(record, config_hash="config", manifest_hash="manifest")


def test_blind_record_exposes_frozen_online_logging_fields():
    class FakeModel:
        def infer(self, *, prompt, images, max_new_tokens):
            assert images == []
            return {
                "raw_output": "A", "answer_model_time_s": 0.2, "answer_preprocess_time_s": 0.1,
                "text_tokens": 5, "visual_tokens": 0, "total_input_tokens": 5,
                "output_tokens": 1, "peak_vram_gib": 1.0,
            }
    config = {
        "prompts": {"open": "OPEN:{question}", "closed": "CLOSED:{question}:{options}"},
        "conditions": {"blind": {"frame_budget": 0}},
        "decoding": {"open_max_new_tokens": 64, "closed_max_new_tokens": 8},
        "answer_models": {"engineering_dry_run_only": {"model_id": "test-2b"}},
    }
    record = _record(
        case=_case(), condition="blind", config=config, config_hash="config",
        manifest_hash="manifest", model=FakeModel(), run_id="run",
    )
    required = {
        "clip_uid", "video_uid", "context_scope", "video_decode_time_s", "frames_decoded_online",
        "frames_shown", "answer_model_time_s", "total_latency_s", "text_tokens", "visual_tokens",
        "model_calls", "api_calls", "peak_vram_gib", "config_hash", "manifest_hash",
    }
    assert required <= set(record)
    assert record["context_scope"] == CONTEXT_SCOPE
    assert record["frames_shown"] == record["frames_decoded_online"] == 0


def test_formal_core_excludes_uniform_32_and_accepts_only_amended_records():
    assert CORE_CONDITIONS == ("blind", "uniform_8", "oracle_leq8")
    record = {
        "question_id": "sample", "task": "open", "condition": "uniform_8",
        "clip_uid": "clip", "video_uid": "video", "context_scope": CONTEXT_SCOPE,
        "success": False, "config_hash": "config", "manifest_hash": "manifest",
        "formal_result": True, "run_kind": "formal_e1_core_headroom_gate",
        "protocol_amendment_hash": "amendment", "cuda_memory": {}, "model_calls": 1,
    }
    assert _valid_core_record(record, config_hash="config", manifest_hash="manifest", amendment_hash="amendment")
    record["condition"] = "uniform_32"
    assert not _valid_core_record(record, config_hash="config", manifest_hash="manifest", amendment_hash="amendment")
