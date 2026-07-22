from __future__ import annotations

import inspect

from src.baselines.egopolice_b0.core import evaluate_gt_exposure, uniform_timestamps
from src.diagnostics.b0 import runner


def test_gt_exposure_is_post_hoc_and_uses_half_open_interval():
    timestamps = uniform_timestamps(80.0, 8)

    hit = evaluate_gt_exposure(timestamps, [70, 80])
    miss = evaluate_gt_exposure(timestamps, [0, 1])

    assert hit == {
        "gt_interval_sec": [70.0, 80.0],
        "gt_interval_hit_at_8": 1,
        "number_of_frames_inside_gt_interval": 1,
        "nearest_sample_distance_to_gt_interval_seconds": 0.0,
    }
    assert miss["gt_interval_hit_at_8"] == 0
    assert miss["number_of_frames_inside_gt_interval"] == 0
    assert miss["nearest_sample_distance_to_gt_interval_seconds"] == 4.0


def test_batch_is_locked_to_exact_five_questions_and_frozen_budget():
    assert runner.QUESTION_IDS == ("1s_972", "10s_973", "10s_974", "60s_386", "1s_973")
    assert runner.NUM_FRAMES == 8
    assert runner.MAX_PIXELS == 262144
    assert runner.DTYPE == "bfloat16"
    assert runner.QUANTIZATION["mode"] == "none"


def test_sampler_call_cannot_receive_gt_interval():
    sampler_parameters = inspect.signature(runner.extract_uniform_frames).parameters

    assert "annotation_interval" not in sampler_parameters
    assert "start_sec" not in sampler_parameters
    assert "end_sec" not in sampler_parameters


def test_interpretation_rules_match_controlled_comparison_contract():
    assert runner._interpretation(True, False, 0) == "likely evidence sampling failure"
    assert "do not attribute purely to retrieval" in runner._interpretation(True, False, 1)
    assert "confound" in runner._interpretation(False, False, 0)
    assert "possible guessing" in runner._interpretation(True, True, 0)
