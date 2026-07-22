from __future__ import annotations

import pytest

from src.evaluation.egopolice_formal import aggregate_formal_b0, source_video_duration_bin


@pytest.mark.parametrize(
    ("duration", "expected"),
    [(300, "le_300"), (300.1, "301_600"), (600, "301_600"),
     (600.1, "601_1200"), (1200, "601_1200"), (1200.1, "gt_1200")],
)
def test_source_duration_bins_use_validated_full_video_duration(duration, expected):
    assert source_video_duration_bin(duration) == expected


def _record(question_class: str, video_bin: str, video_id: str, correct: bool) -> dict:
    return {
        "question_duration_class": question_class,
        "source_video_duration_bin": video_bin,
        "source_video_id": video_id,
        "correct": correct,
        "gt_interval_hit_at_8": 1,
        "nearest_sample_distance_to_gt_interval_seconds": 0,
        "total_per_query_latency_sec": 2,
        "frame_extraction_latency_sec": 1,
        "qwen_inference_latency_sec": 0.8,
        "text_input_tokens": 100,
        "visual_tokens_estimate": 2000,
        "model_facing_frames": 8,
        "model_calls": 1,
    }


def test_formal_aggregation_always_keeps_counts_and_both_duration_dimensions():
    result = aggregate_formal_b0([
        _record("1s", "le_300", "v1", True),
        _record("10s", "le_300", "v1", False),
        _record("60s", "gt_1200", "v2", True),
    ])

    assert result["overall"] == {
        "correct": 2, "total": 3, "n": 3, "accuracy": 2 / 3,
        "accuracy_percent": 200 / 3, "distinct_videos": 2,
    }
    assert result["by_question_duration_class"]["10s"]["total"] == 1
    assert result["by_source_video_duration"]["le_300"]["correct"] == 1
    assert result["by_source_video_duration"]["le_300"]["total"] == 2
    assert result["source_video_duration_x_question_duration"]["gt_1200"]["60s"]["n"] == 1
    diagnostics = result["by_source_video_duration"]["le_300"]["diagnostics"]
    assert diagnostics["mean_model_facing_frames"] == 8
    assert diagnostics["mean_model_calls"] == 1
