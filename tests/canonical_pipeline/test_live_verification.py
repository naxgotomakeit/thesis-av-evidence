from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from scripts.canonical import run_live_verification
from src.canonical_pipeline.live_boundaries import LazyWhisperFallback, environment_presence
from src.canonical_pipeline.runner import CanonicalOnlineRunner
from src.canonical_pipeline.state import ExecutionMode
from src.canonical_pipeline.versions import load_canonical_config
from src.instrumentation.efficiency import gemini_usage_metrics
from src.instrumentation.timing import StageTimer, timing_consistency


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_canonical_config(ROOT)


def test_live_is_gated_by_execute_flag_and_cases_are_fixed():
    source = inspect.getsource(run_live_verification.main)
    assert "--execute-live" in source
    assert run_live_verification.CASES == ("00006_3", "00061_5")


def test_regression_replay_remains_zero_call():
    state = CanonicalOnlineRunner(CONFIG).run_case("00006_3", mode=ExecutionMode.REGRESSION_REPLAY)
    assert sum(state.external_calls.values()) == 0


def test_live_runner_does_not_accept_or_inject_frozen_planner_output():
    signature = inspect.signature(CanonicalOnlineRunner.run_live_case)
    assert "frozen_record" not in signature.parameters
    source = inspect.getsource(CanonicalOnlineRunner.run_live_case)
    assert "self.attempt_journal.planner_request(case_id, planner_request)" in source
    assert "frozen_record" not in source


def test_live_fallback_boundary_has_no_frozen_evidence_input():
    signature = inspect.signature(LazyWhisperFallback.execute)
    assert set(signature.parameters) == {"self", "state", "context"}
    source = inspect.getsource(LazyWhisperFallback.execute).casefold()
    assert "frozen_record" not in source and "frozen_v1_2" not in source


def test_fallback_at_most_once_invariant_remains_present():
    from src.canonical_pipeline import sufficiency
    assert "fallback attempted more than once" in inspect.getsource(sufficiency.run_evidence_sufficiency)


def test_gold_cannot_enter_runtime_state_before_prediction():
    from src.canonical_pipeline.state import CaseState
    fields = CaseState.__dataclass_fields__
    assert not {"gold", "answer", "reference", "weak_reference"}.intersection(fields)
    source = inspect.getsource(CanonicalOnlineRunner._safe_case)
    assert "source[\"answer\"]" not in source


def test_skipped_stage_has_null_duration():
    timer = StageTimer()
    timer.skip("fallback_execution", "fallback_not_triggered")
    row = timer.as_dicts()[0]
    assert row["skipped"] is True and row["executed"] is False and row["duration_sec"] is None


def test_timing_coverage_uses_nonoverlapping_top_levels():
    timer = StageTimer()
    with timer.stage("online_end_to_end_total"):
        with timer.stage("child", parent_stage="parent"):
            pass
        with timer.stage("parent", parent_stage="online_end_to_end_total"):
            pass
    result = timing_consistency(timer.as_dicts(), ["parent"])
    assert result["double_counting_avoided"] is True
    assert result["non_overlapping_top_level_duration_sum_sec"] >= 0


def test_missing_cache_field_remains_null_with_warning():
    result = gemini_usage_metrics({"total_input_tokens": 12, "total_output_tokens": 3, "total_tokens": 15})
    assert result["total_cached_tokens"] is None
    assert result["cache_hit"] is None
    assert result["provider_usage_warnings"]


def test_secret_presence_returns_booleans_only(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-log")
    monkeypatch.setenv("ANTHROPIC_MODEL", "model")
    monkeypatch.setenv("GEMINI_API_KEY", "do-not-log")
    assert environment_presence() == {"anthropic_key_present": True, "anthropic_model_present": True, "gemini_key_present": True}


def test_live_script_persists_prediction_before_loading_gold():
    source = inspect.getsource(run_live_verification.main)
    raw_pos = source.index('"live_raw_predictions.jsonl"')
    validated_pos = source.index('"live_validated_predictions.jsonl"')
    gold_pos = source.index('gold = {row["case_id"]')
    assert raw_pos < gold_pos and validated_pos < gold_pos


def test_frozen_inputs_hash_identical_during_dry_instrumentation():
    paths = [CONFIG.path(name) for name in ("planner_plans", "task5b_frozen", "task5c_frozen", "task6_frozen", "task7a_frozen")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    CanonicalOnlineRunner(CONFIG).run_case("00006_3")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after


def test_live_outputs_never_serialize_media_bytes_or_secret_values():
    source = inspect.getsource(run_live_verification)
    assert "base64" not in source
    assert "os.environ.get(\"GEMINI_API_KEY\")" not in source
    assert "os.environ.get(\"ANTHROPIC_API_KEY\")" not in source
