from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.egopolice_b0.core import (
    BaselineInputError,
    build_mcq_prompt,
    load_mcq_cases,
    parse_prediction,
    resolve_video_path,
    uniform_timestamps,
)
from src.baselines.egopolice_b0.runner import _allowed_video_ids, _error_result, resolve_runtime_config
from src.baselines.egopolice_b0.model import validate_model_loading_settings


ROOT = Path(__file__).resolve().parents[2]
TEST_OUTPUT = ROOT / "outputs/baselines/egopolice_b0/test_fixture.json"


def test_uniform_sampling_uses_midpoints_of_full_video_bins() -> None:
    assert uniform_timestamps(80.0, 8) == [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]


def test_prompt_contains_question_options_but_no_gold_or_timestamp() -> None:
    prompt = build_mcq_prompt("What is visible?", ["a", "b", "c", "d", "e"], 8)
    assert "What is visible?" in prompt
    assert all(f"{index}. {value}" in prompt for index, value in enumerate("abcde"))
    assert "ground_truth" not in prompt
    assert "start second" not in prompt
    assert "end second" not in prompt


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("Option 3", 3), ("E", 4), ("b.", 1)])
def test_prediction_parser_accepts_one_answer(raw: str, expected: int) -> None:
    assert parse_prediction(raw) == expected


@pytest.mark.parametrize("raw", ["", "0 or 1", "The answer is 2", "5"])
def test_prediction_parser_rejects_non_contract_output(raw: str) -> None:
    with pytest.raises(BaselineInputError):
        parse_prediction(raw)


def test_loader_keeps_annotation_interval_outside_model_inputs() -> None:
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.write_text(
        json.dumps([
            {
                "id": "60s_1",
                "video": "youtube/channel/video_0_60.mp4",
                "start second": 10,
                "end second": 20,
                "question": "Q?",
                "options": ["a", "b", "c", "d", "e"],
                "answer": 2,
            }
        ]),
        encoding="utf-8",
    )
    case = load_mcq_cases(TEST_OUTPUT)[0]
    assert case["ground_truth_index"] == 2
    assert case["ignored_annotation_interval"] == [10, 20]
    prompt = build_mcq_prompt(case["question"], case["options"], 8)
    assert "10" not in prompt and "20" not in prompt


def test_video_path_preserves_dataset_subdirectories() -> None:
    root = Path("dataset-root")
    assert resolve_video_path(root, "copa/year/file.mp4") == root / "copa/year/file.mp4"


def test_portable_paths_resolve_from_environment_on_posix() -> None:
    config = json.loads((ROOT / "config/baselines/egopolice_b0.json").read_text(encoding="utf-8"))
    resolved, output = resolve_runtime_config(
        config,
        environ={
            "DATA_ROOT": "/home/student/thesis/data/EgoPolice_1.0.0",
            "MODEL_ROOT": "/home/student/thesis/models",
            "OUTPUT_ROOT": "/home/student/thesis/outputs",
            "FFMPEG_PATH": "/usr/bin/ffmpeg",
            "FFPROBE_PATH": "/usr/bin/ffprobe",
        },
        repo_root=Path("/repo"),
    )
    assert resolved["qa_path"].replace("\\", "/") == "/home/student/thesis/data/EgoPolice_1.0.0/mcq_60s.json"
    assert resolved["video_root"].replace("\\", "/") == "/home/student/thesis/data/EgoPolice_1.0.0/videos"
    assert resolved["model_path"].replace("\\", "/") == "/home/student/thesis/models/Qwen2.5-VL-7B-Instruct"
    assert resolved["ffmpeg_path"] == "/usr/bin/ffmpeg"
    assert output.as_posix() == "/home/student/thesis/outputs/baselines/egopolice_b0/results.jsonl"
    assert resolved["subset_manifest"].replace("\\", "/").endswith(
        "/config/data/egopolice_50videos.json"
    )


def test_cli_paths_override_environment_without_changing_b0_settings() -> None:
    config = json.loads((ROOT / "config/baselines/egopolice_b0.json").read_text(encoding="utf-8"))
    resolved, _ = resolve_runtime_config(
        config,
        overrides={"model_path": Path("/explicit/model"), "num_frames": 8, "max_pixels": 262144},
        environ={"MODEL_PATH": "/environment/model", "DATA_ROOT": "/data"},
    )
    assert resolved["model_path"].replace("\\", "/") == "/explicit/model"
    assert resolved["num_frames"] == 8
    assert resolved["max_pixels"] == 262144
    assert resolved["generation"] == config["generation"]


def test_bf16_unquantized_is_the_default_formal_loading_setting() -> None:
    config = json.loads((ROOT / "config/baselines/egopolice_b0.json").read_text(encoding="utf-8"))
    resolved, _ = resolve_runtime_config(config, environ={})
    assert resolved["dtype"] == "bfloat16"
    assert resolved["quantization"]["mode"] == "none"


def test_nf4_fallback_can_be_selected_without_changing_b0_budget() -> None:
    config = json.loads((ROOT / "config/baselines/egopolice_b0.json").read_text(encoding="utf-8"))
    resolved, _ = resolve_runtime_config(
        config,
        overrides={"dtype": "bfloat16", "quantization_mode": "nf4_4bit"},
    )
    assert resolved["dtype"] == "bfloat16"
    assert resolved["quantization"]["mode"] == "nf4_4bit"
    assert resolved["num_frames"] == 8
    assert resolved["max_pixels"] == 262144
    assert resolved["generation"] == config["generation"]


@pytest.mark.parametrize(
    ("dtype", "mode"),
    [("bfloat16", "none"), ("float16", "none"), ("bfloat16", "nf4_4bit")],
)
def test_supported_model_loading_settings(dtype: str, mode: str) -> None:
    validate_model_loading_settings(dtype, mode)


def test_unsupported_model_loading_setting_is_rejected() -> None:
    with pytest.raises(BaselineInputError):
        validate_model_loading_settings("int8", "none")
    with pytest.raises(BaselineInputError):
        validate_model_loading_settings("bfloat16", "automatic")


def test_result_metadata_records_requested_and_actual_loading_mode() -> None:
    config = json.loads((ROOT / "config/baselines/egopolice_b0.json").read_text(encoding="utf-8"))
    config["model_path"] = "/models/Qwen2.5-VL-7B-Instruct"
    case = {
        "video_id": "copa/video",
        "question_id": "60s_1",
        "question": "Q?",
        "options": ["a", "b", "c", "d", "e"],
        "ground_truth_index": 0,
        "ground_truth_text": "a",
        "ignored_annotation_interval": [1, 2],
    }
    record = _error_result(
        case,
        video_path=Path("video.mp4"),
        config=config,
        errors=["test"],
        error_type="runtime_error",
        actual_model_loading_mode="bfloat16_unquantized",
    )
    settings = record["inference_settings"]
    assert settings["dtype"] == "bfloat16"
    assert settings["quantization_mode"] == "none"
    assert settings["actual_model_loading_mode"] == "bfloat16_unquantized"


def test_b0_metadata_identifies_controlled_7b_checkpoint() -> None:
    config = json.loads((ROOT / "config/baselines/egopolice_b0.json").read_text(encoding="utf-8"))
    assert config["baseline_id"] == "egopolice_b0_uniform_qwen2_5_vl_7b_v0_1"
    assert config["model_directory_name"] == "Qwen2.5-VL-7B-Instruct"


def test_frozen_subset_filter_contains_exact_50_source_videos() -> None:
    allowed = _allowed_video_ids(ROOT / "config/data/egopolice_50videos.json")
    assert allowed is not None
    assert len(allowed) == 50
    assert "pasadena/bMMuC" in allowed
