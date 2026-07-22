from __future__ import annotations

import json

import numpy as np

from src.diagnostics.cradio_v4.core import (
    load_frozen_questions,
    one_fps_timestamps,
    posthoc_gt_interval_metrics,
    rank_timestamps,
    timestamps_to_frame_indices,
)


def test_deterministic_one_fps_grid_and_frame_mapping() -> None:
    timestamps = one_fps_timestamps(3.2)
    assert timestamps.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert timestamps_to_frame_indices(timestamps, average_fps=30.0, frame_count=96).tolist() == [0, 30, 60, 90]


def test_ranking_is_annotation_independent_and_gt_metrics_are_posthoc() -> None:
    similarities = [0.1, 0.9, 0.4, 0.3]
    ranking = rank_timestamps(similarities)
    assert ranking.tolist() == [1, 2, 3, 0]
    metrics = posthoc_gt_interval_metrics(
        ranking, [0.0, 1.0, 2.0, 3.0], gt_start_sec=2.0, gt_end_sec=3.0
    )
    assert metrics["gt_interval_hit_at_k"] == {"1": False, "5": True, "8": True, "10": True}
    assert metrics["best_ranked_timestamp_distance_to_gt_interval_sec"] == 1.0
    assert metrics["rank_of_first_timestamp_inside_gt_interval"] == 2


def test_frozen_question_loader_reads_only_target_video(tmp_path) -> None:
    manifest = tmp_path / "questions.json"
    manifest.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": "10s_a",
                        "video_id": "pasadena/YKI08",
                        "duration_class": "10s",
                        "question": "What happens?",
                        "gt_interval_sec": [2.0, 12.0],
                        "options": ["unused"] * 5,
                        "ground_truth_index": 0,
                    },
                    {
                        "question_id": "1s_b",
                        "video_id": "pasadena/other",
                        "duration_class": "1s",
                        "question": "Ignored",
                        "gt_interval_sec": [0.0, 1.0],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = load_frozen_questions(manifest)
    assert [(row.question_id, row.question) for row in rows] == [("10s_a", "What happens?")]

