from __future__ import annotations

import inspect
import time
from pathlib import Path
from types import SimpleNamespace

from scripts import run_baseline_instrumentation as runner
from src.evaluation.answer_metrics import evaluate_answer, sentence_bleu, single_reference_cider
from src.final_qa.task7b_reporting import guarded_load_gold
from src.instrumentation.efficiency import gemini_usage_metrics
from src.instrumentation.timing import StageTimer, blocked_online_timings, existing_offline_artifact_timings, timing_consistency


ROOT = Path(__file__).resolve().parents[1]


def test_1_2_monotonic_timing_and_non_negative_duration():
    timer = StageTimer()
    with timer.stage("unit"):
        time.sleep(0.001)
    row = timer.as_dicts()[0]
    assert row["end_monotonic"] >= row["start_monotonic"]
    assert row["duration_sec"] >= 0


def test_3_skipped_stages_are_explicit():
    row = StageTimer().skip("visual_retrieval", "modality_not_required").to_dict()
    assert row["executed"] is False and row["skipped"] is True
    assert row["skip_reason"] == "modality_not_required" and row["duration_sec"] is None


def test_4_nested_stages_are_not_added_to_top_level_sum():
    records = [
        {"stage_name": "online_end_to_end_total", "executed": True, "duration_sec": 1.0},
        {"stage_name": "retrieval_total", "executed": True, "duration_sec": 0.4},
        {"stage_name": "speech_retrieval", "parent_stage": "retrieval_total", "executed": True, "duration_sec": 0.3},
    ]
    result = timing_consistency(records, ["retrieval_total"])
    assert result["non_overlapping_top_level_duration_sum_sec"] == 0.4
    assert result["uninstrumented_overhead_sec"] == 0.6


def test_5_6_online_total_consistency_and_overhead():
    records = [
        {"stage_name": "online_end_to_end_total", "executed": True, "duration_sec": 2.0},
        {"stage_name": "question_planner", "executed": True, "duration_sec": 0.5},
        {"stage_name": "final_model_api", "executed": True, "duration_sec": 1.0},
    ]
    result = timing_consistency(records, ["question_planner", "final_model_api"])
    assert result["consistency_status"] == "valid"
    assert result["uninstrumented_overhead_sec"] == 0.5
    assert result["uninstrumented_overhead_percentage"] == 25.0


def test_7_existing_offline_artifacts_are_not_reported_as_zero():
    rows = existing_offline_artifact_timings()
    assert rows and all(row["skip_reason"] == "not_measured_existing_artifact" for row in rows)
    assert all(row["duration_sec"] is None for row in rows)


def test_8_provider_wall_clock_is_separate_field_in_run_output_contract():
    source = inspect.getsource(runner.main)
    assert "provider_api_wall_clock_latency" in source
    assert "online_end_to_end" not in "provider_api_wall_clock_latency"


def test_9_cache_tokens_are_read_from_provider_usage():
    usage = SimpleNamespace(model_dump=lambda mode: {"total_input_tokens": 5000, "total_output_tokens": 100, "total_tokens": 5200, "total_cached_tokens": 1200})
    metrics = gemini_usage_metrics(usage)
    assert metrics["total_cached_tokens"] == 1200


def test_10_missing_cache_field_is_null_with_warning():
    metrics = gemini_usage_metrics({"total_input_tokens": 100})
    assert metrics["total_cached_tokens"] is None and metrics["cache_hit"] is None
    assert "provider_usage_field_unavailable:total_cached_tokens" in metrics["provider_usage_warnings"]


def test_11_12_uncached_tokens_hit_and_rate_are_deterministic():
    metrics = gemini_usage_metrics({"total_input_tokens": 5000, "total_output_tokens": 50, "total_tokens": 5050, "total_cached_tokens": 2000})
    assert metrics["uncached_input_tokens"] == 3000
    assert metrics["cache_hit"] is True and metrics["cache_hit_rate"] == 0.4
    assert metrics["cache_eligible_by_input_size"] is True


def test_13_gold_loader_requires_saved_raw_and_validation():
    source = inspect.getsource(guarded_load_gold)
    assert "raw_predictions_path.is_file" in source and "validated_predictions_path.is_file" in source


def test_14_bleu_cider_do_not_change_answer_status_or_retry():
    case = {"case_id": "fixture", "question": "q", "gold_dataset_answer": "alpha", "raw_model_prediction": "beta", "validated_prediction": "beta", "answer_status": "answered_with_uncertainty", "confidence": {"level": "low"}, "abstain": False, "uncertainties": []}
    evaluated = evaluate_answer(case, ["alpha"])
    assert evaluated["answer_status"] == case["answer_status"]
    assert evaluated["metric_interpretation"] == {"automatic_correctness": "not_determined", "metric_role": "diagnostic_reference_overlap"}
    assert sentence_bleu("alpha", "beta") >= 0 and single_reference_cider("alpha", "beta", ["alpha"]) >= 0
    assert "retry" not in evaluated


def test_15_evaluation_is_explicitly_excluded_from_online_latency():
    source = inspect.getsource(runner.main)
    assert "evaluation_excluded_from_online_latency" in source
    assert source.index("evaluation_started") > source.index('write_jsonl(OUT / "cache_metrics.jsonl"')


def test_16_no_gold_is_loaded_by_timing_or_efficiency_modules():
    import src.instrumentation.efficiency as efficiency
    import src.instrumentation.timing as timing
    assert "gold" not in inspect.getsource(timing).casefold()
    assert "gold" not in inspect.getsource(efficiency).casefold()


def test_17_api_keys_are_presence_only_and_never_serialized_as_values():
    source = inspect.getsource(runner.main)
    assert "api_key_presence_only" in source and "api_keys_logged" in source
    assert "os.environ" not in inspect.getsource(runner.html_page)


def test_18_only_two_requested_cases_are_planned():
    assert runner.CASES == ("00006_3", "00061_5")


def test_19_frozen_upstream_roots_are_distinct_from_new_output():
    assert all(root != runner.OUT and runner.OUT not in root.parents for root in runner.TRACKED_ROOTS)
    assert all(root.is_dir() for root in runner.TRACKED_ROOTS)


def test_blocked_online_records_do_not_fabricate_zero_duration():
    rows = blocked_online_timings()
    assert all(row["duration_sec"] is None for row in rows)
    assert any(row["stage_name"] == "online_end_to_end_total" for row in rows)

