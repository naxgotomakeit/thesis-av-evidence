from __future__ import annotations

import inspect
import json

import pytest

from src.diagnostics.oracle import core
from src.diagnostics.oracle.core import (
    OracleCase,
    OracleInputError,
    build_oracle_messages,
    build_oracle_prompt,
    load_oracle_case,
    normalize_single_video_fps,
    parse_oracle_prediction,
)


def _case(tmp_path) -> OracleCase:
    video = tmp_path / "videos" / "pasadena" / "sample.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    rows = [
        {
            "id": "10s_test",
            "video": "pasadena/sample.mp4",
            "start second": 70,
            "end second": 80,
            "question": "What is visible?",
            "options": ["zero", "one", "two", "three", "four"],
            "answer": 1,
        }
    ]
    (tmp_path / "mcq_10s.json").write_text(json.dumps(rows), encoding="utf-8")
    return load_oracle_case(tmp_path, "10s", "10s_test")


def test_metadata_maps_to_source_video_and_exact_half_open_interval(tmp_path):
    case = _case(tmp_path)

    assert case.source_video_path == tmp_path / "videos" / "pasadena" / "sample.mp4"
    assert (case.start_sec, case.end_sec) == (70.0, 80.0)
    assert case.duration_class == "10s"
    assert case.ground_truth_index == 1


def test_mcq_prompt_has_all_options_and_strict_answer_contract(tmp_path):
    prompt = build_oracle_prompt(_case(tmp_path))

    assert "Question: What is visible?" in prompt
    assert "\n0. zero\n1. one\n2. two\n3. three\n4. four\n" in prompt
    assert prompt.endswith("Do not provide an explanation.")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("option 3", 3), ("B", 1), ("Option E.", 4)],
)
def test_prediction_parser_accepts_only_a_single_mcq_answer(raw, expected):
    assert parse_oracle_prediction(raw) == expected


def test_prediction_parser_rejects_explanations():
    with pytest.raises(OracleInputError):
        parse_oracle_prediction("1 because the officer touches the civilian")


def test_oracle_messages_use_exact_interval_and_do_not_import_b0_sampling(tmp_path):
    messages = build_oracle_messages(_case(tmp_path))
    video = messages[0]["content"][0]

    assert video["video_start"] == 70.0
    assert video["video_end"] == 80.0
    assert video["fps"] == 1.0
    assert video["max_pixels"] == 262144
    source = inspect.getsource(core)
    assert "uniform_timestamps" not in source
    assert "midpoint" not in source.lower()
    assert "src.baselines" not in source


def test_single_video_fps_list_is_normalized_for_transformers_5_14():
    normalized = normalize_single_video_fps([object()], {"fps": [1.0], "do_sample_frames": True})

    assert normalized == {"fps": 1.0, "do_sample_frames": True}

