from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.baselines.egopolice_b0.core import build_mcq_prompt, uniform_timestamps
from src.baselines.egopolice_formal.runner import (
    CONDITION_BLIND,
    CONDITION_UNIFORM8,
    build_blind_prompt,
    checkpoint_path,
    execute_formal,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0"
)
MODEL_PATH = Path(
    "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/Qwen2.5-VL-7B-Instruct"
)


class DummyImage:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DummyModel:
    def __init__(self, counters: dict[str, int], fail_at: int | None = None) -> None:
        counters["loads"] = counters.get("loads", 0) + 1
        self.counters = counters
        self.fail_at = fail_at
        self.model_load_latency_sec = 0.01
        self.actual_model_loading_mode = "bfloat16_unquantized"

    def infer(self, *, images, prompt, generation):
        self.counters["calls"] = self.counters.get("calls", 0) + 1
        if self.fail_at is not None and self.counters["calls"] == self.fail_at:
            raise RuntimeError("intentional mock interruption")
        prediction = self.counters["calls"] % 5
        return {
            "prediction_index": prediction,
            "raw_output": str(prediction),
            "preprocessing_latency_sec": 0.02,
            "inference_latency_sec": 0.03,
            "text_token_count": 64,
            "visual_token_count": 128 if images else 0,
            "total_input_token_count": 192 if images else 64,
            "output_token_count": 1,
            "peak_gpu_memory_bytes": 1024,
        }


def model_factory(counters: dict[str, int], fail_at: int | None = None):
    def factory(**_kwargs):
        return DummyModel(counters, fail_at=fail_at)

    return factory


def mock_environment(_model_path, *, dtype, quantization_mode):
    assert dtype == "bfloat16"
    assert quantization_mode == "none"
    return {
        "passed": True,
        "errors": [],
        "free_vram_bytes": 24 * 1024**3,
        "cuda_available": True,
        "gpu_name": "mock-gpu",
    }


def duration_by_path() -> dict[str, float]:
    payload = json.loads(
        (ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json").read_text()
    )
    return {
        row["actual_file_path"]: float(row["duration_sec"])
        for row in payload["videos"]
    }


def mock_frame_extractor(counters: dict[str, int]):
    durations = duration_by_path()

    def extract(*, video_path, ffmpeg_path, ffprobe_path, num_frames, max_pixels):
        assert ffmpeg_path == "ffmpeg"
        assert ffprobe_path == "ffprobe"
        assert num_frames == 8
        assert max_pixels == 262144
        counters["extractions"] = counters.get("extractions", 0) + 1
        duration = durations[str(video_path)]
        return (
            [DummyImage() for _ in range(8)],
            uniform_timestamps(duration, 8),
            duration,
            0.04,
        )

    return extract


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_prompts_preserve_uniform8_and_define_text_only_blind():
    options = ["a", "b", "c", "d", "e"]
    blind = build_blind_prompt("Q?", options)
    assert blind == (
        "Question: Q?\n\nOptions:\n0. a\n1. b\n2. c\n3. d\n4. e\n\n"
        "Return exactly one option index: 0, 1, 2, 3, or 4. "
        "Do not provide an explanation."
    )
    assert build_mcq_prompt("Q?", options, 8).startswith(
        "You are given 8 uniformly sampled frames from one full video "
        "in chronological order."
    )


@pytest.mark.parametrize("condition", [CONDITION_BLIND, CONDITION_UNIFORM8])
def test_dry_run_enumerates_exact_frozen_98_without_loading_model(tmp_path, condition):
    counters: dict[str, int] = {}
    output_dir = tmp_path / condition
    result = execute_formal(
        condition=condition,
        data_root=DATA_ROOT,
        model_path=MODEL_PATH,
        output_dir=output_dir,
        dry_run=True,
        model_factory=model_factory(counters),
        environment_checker=mock_environment,
    )
    assert result["formal_question_count"] == 98
    assert result["distinct_video_count"] == 20
    assert result["question_count_by_duration_class"] == {
        "1s": 39, "10s": 38, "60s": 21,
    }
    assert len(result["question_ids"]) == len(set(result["question_ids"])) == 98
    assert result["model_loaded"] is False
    assert result["model_calls"] == 0
    assert counters == {}
    assert not output_dir.exists()


def test_blind_atomic_checkpoints_explicit_na_and_invalid_only_recomputed(tmp_path):
    output_dir = tmp_path / "blind"
    first: dict[str, int] = {}
    result = execute_formal(
        condition=CONDITION_BLIND,
        data_root=DATA_ROOT,
        model_path=MODEL_PATH,
        output_dir=output_dir,
        model_factory=model_factory(first),
        environment_checker=mock_environment,
    )
    assert result["valid_completed_count"] == 98
    assert first == {"loads": 1, "calls": 98}
    checkpoints = sorted((output_dir / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 98
    record = json.loads(checkpoints[0].read_text())
    assert record["selected_timestamps_sec"] == []
    assert record["model_facing_frames"] == 0
    assert record["frame_extraction_latency_sec"] == 0.0
    assert record["gt_interval_hit_at_8"] is None
    assert record["number_of_frames_inside_gt_interval"] is None
    assert record["nearest_sample_distance_to_gt_interval_seconds"] is None
    assert record["visual_token_count"] == 0
    assert record["retrieval_metrics"]["applicable"] is False
    assert record["retrieval_metrics"]["retrieval_calls"] == 0
    assert (output_dir / "final_results.json").is_file()
    assert (output_dir / "REPORT.md").is_file()
    assert not list(output_dir.rglob("*.tmp"))

    untouched = {path.name: file_sha(path) for path in checkpoints[1:]}
    broken_path = checkpoints[0]
    broken = json.loads(broken_path.read_text())
    broken["model_calls"] = 2
    broken_path.write_text(json.dumps(broken))
    resumed: dict[str, int] = {}
    second = execute_formal(
        condition=CONDITION_BLIND,
        data_root=DATA_ROOT,
        model_path=MODEL_PATH,
        output_dir=output_dir,
        model_factory=model_factory(resumed),
        environment_checker=mock_environment,
    )
    assert second["valid_completed_count"] == 98
    assert resumed == {"loads": 1, "calls": 1}
    assert any((output_dir / "checkpoints").glob(f"{broken_path.stem}.invalid.*.json"))
    assert {path.name: file_sha(path) for path in checkpoints[1:]} == untouched

    no_load: dict[str, int] = {}
    third = execute_formal(
        condition=CONDITION_BLIND,
        data_root=DATA_ROOT,
        model_path=MODEL_PATH,
        output_dir=output_dir,
        model_factory=model_factory(no_load),
        environment_checker=mock_environment,
    )
    assert third["already_complete"] is True
    assert third["model_loaded"] is False
    assert no_load == {}


def test_uniform8_interruption_resumes_only_remaining_without_duplicates(tmp_path):
    output_dir = tmp_path / "uniform8"
    interrupted: dict[str, int] = {}
    extraction_first: dict[str, int] = {}
    with pytest.raises(RuntimeError, match="intentional mock interruption"):
        execute_formal(
            condition=CONDITION_UNIFORM8,
            data_root=DATA_ROOT,
            model_path=MODEL_PATH,
            output_dir=output_dir,
            model_factory=model_factory(interrupted, fail_at=3),
            environment_checker=mock_environment,
            frame_extractor=mock_frame_extractor(extraction_first),
        )
    assert interrupted == {"loads": 1, "calls": 3}
    assert extraction_first == {"extractions": 3}
    first_two = sorted((output_dir / "checkpoints").glob("*.json"))
    assert len(first_two) == 2
    first_hashes = {path.name: file_sha(path) for path in first_two}

    resumed: dict[str, int] = {}
    extraction_second: dict[str, int] = {}
    result = execute_formal(
        condition=CONDITION_UNIFORM8,
        data_root=DATA_ROOT,
        model_path=MODEL_PATH,
        output_dir=output_dir,
        model_factory=model_factory(resumed),
        environment_checker=mock_environment,
        frame_extractor=mock_frame_extractor(extraction_second),
    )
    assert result["valid_completed_count"] == 98
    assert resumed == {"loads": 1, "calls": 96}
    assert extraction_second == {"extractions": 96}
    checkpoints = sorted((output_dir / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 98
    assert len({json.loads(path.read_text())["question_id"] for path in checkpoints}) == 98
    assert {path.name: file_sha(path) for path in first_two} == first_hashes
    record = json.loads(checkpoints[-1].read_text())
    assert len(record["selected_timestamps_sec"]) == 8
    assert record["model_facing_frames"] == 8
    assert record["gt_interval_hit_at_8"] in (0, 1)
    assert record["number_of_frames_inside_gt_interval"] >= 0
    assert record["nearest_sample_distance_to_gt_interval_seconds"] >= 0
    assert record["retrieval_metrics"]["similarity_computations"] == 0
    assert json.loads((output_dir / "progress.json").read_text())["completed_count"] == 98
    assert (output_dir / "aggregate_summary.json").is_file()
    assert (output_dir / "video_length_x_question_class.csv").is_file()
