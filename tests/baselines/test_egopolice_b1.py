from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.baselines.egopolice_b1 import aggregation, prepare_retrieval, runner
from src.baselines.egopolice_b1.indexer import build_indexes
from src.baselines.egopolice_b1.retrieval import (
    build_raw_question_queries,
    cosine_ranking,
    select_top8_unique,
)
from src.baselines.egopolice_formal.runner import load_formal_contract


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0"
)
MODEL_PATH = Path(
    "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/Qwen2.5-VL-7B-Instruct"
)
VIDEO_MANIFEST = ROOT / "config/data/egopolice_ablation20_v1.json"
QUESTION_MANIFEST = ROOT / "config/data/egopolice_ablation_questions_v1.json"
READINESS = ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json"
CONFIG = ROOT / "config/baselines/egopolice_b0.json"


def contract():
    return load_formal_contract(
        data_root=DATA_ROOT,
        video_manifest_path=VIDEO_MANIFEST,
        question_manifest_path=QUESTION_MANIFEST,
        readiness_path=READINESS,
        config_path=CONFIG,
    )


class DummyImage:
    def close(self):
        pass


class DummyQwen:
    def __init__(self, counters, fail_at=None):
        counters["loads"] = counters.get("loads", 0) + 1
        self.counters = counters
        self.fail_at = fail_at
        self.model_load_latency_sec = 0.01
        self.actual_model_loading_mode = "bfloat16_unquantized"

    def infer(self, *, images, prompt, generation):
        assert len(images) == 8
        self.counters["calls"] = self.counters.get("calls", 0) + 1
        if self.fail_at == self.counters["calls"]:
            raise RuntimeError("mock interruption")
        prediction = self.counters["calls"] % 5
        return {
            "prediction_index": prediction,
            "raw_output": str(prediction),
            "preprocessing_latency_sec": 0.02,
            "inference_latency_sec": 0.03,
            "text_token_count": 64,
            "visual_token_count": 128,
            "total_input_token_count": 192,
            "output_token_count": 1,
            "peak_gpu_memory_bytes": 1024,
        }


def qwen_factory(counters, fail_at=None):
    return lambda **kwargs: DummyQwen(counters, fail_at)


def mock_environment(*args, **kwargs):
    return {"passed": True, "errors": [], "free_vram_bytes": 24 * 1024**3}


def mock_extract(**kwargs):
    assert len(kwargs["timestamps_sec"]) == len(set(kwargs["timestamps_sec"])) == 8
    return [DummyImage() for _ in range(8)], 0.04


def fake_dependencies(formal_contract):
    indexes = {
        video_id: {"index_fingerprint": f"index-{video_id}"}
        for video_id in formal_contract["videos"]
    }
    retrievals = {}
    for question in formal_contract["questions"]:
        top8 = [
            {
                "rank": index + 1,
                "index_row": index,
                "timestamp_sec": float(index),
                "source_frame_index": index,
                "cosine_similarity": 1.0 - index / 100.0,
            }
            for index in range(8)
        ]
        retrievals[question["question_id"]] = {
            "retrieval_fingerprint": f"retrieval-{question['question_id']}",
            "retrieval_query_text": question["question"],
            "query_text_embedding_time_sec": 0.01,
            "similarity_search_time_sec": 0.001,
            "peak_text_encoder_vram_bytes": 2048,
            "top_8_ranked": top8,
            "full_ranking_count": 10,
            "ranking_artifact_path": f"mock/{question['question_id']}.npz",
            "ranking_artifact_sha256": "0" * 64,
        }
    queries = build_raw_question_queries(formal_contract["questions"])
    return indexes, retrievals, queries


def test_manifest_dry_runs_enumerate_98_and_do_not_load_models(tmp_path):
    index = build_indexes(
        data_root=DATA_ROOT, index_root=tmp_path / "indexes",
        cache_root=tmp_path / "cache", video_manifest_path=VIDEO_MANIFEST,
        question_manifest_path=QUESTION_MANIFEST, readiness_path=READINESS,
        config_path=CONFIG, device_name="cuda:0", dry_run=True,
        reuse_yki08_diagnostic=False,
    )
    retrieval = prepare_retrieval.prepare_retrieval(
        data_root=DATA_ROOT, index_root=tmp_path / "indexes",
        output_dir=tmp_path / "outputs", cache_root=tmp_path / "cache",
        video_manifest_path=VIDEO_MANIFEST, question_manifest_path=QUESTION_MANIFEST,
        readiness_path=READINESS, config_path=CONFIG, device_name="cuda:0",
        dry_run=True,
    )
    qa = runner.run_b1(
        data_root=DATA_ROOT, model_path=MODEL_PATH,
        index_root=tmp_path / "indexes", output_dir=tmp_path / "outputs",
        video_manifest_path=VIDEO_MANIFEST, question_manifest_path=QUESTION_MANIFEST,
        readiness_path=READINESS, config_path=CONFIG, ffmpeg_path="ffmpeg",
        dry_run=True, execute_formal_inference=False,
    )
    assert index["target_video_count"] == 20
    assert index["model_loaded"] is False
    assert retrieval["question_count"] == 98
    assert retrieval["video_count"] == 20
    assert retrieval["retrieval_query_equals_raw_question_count"] == 98
    assert retrieval["answer_fields_used_for_retrieval"] is False
    assert retrieval["structurally_can_return_8_unique_1fps_frames_for_every_question"]
    assert retrieval["text_encoder_loaded"] is False
    assert qa["formal_question_count"] == 98
    assert qa["distinct_video_count"] == 20
    assert qa["structurally_exactly_8_unique_frames_per_question"]
    assert qa["qwen_loaded"] is False


def test_retrieval_module_exposes_only_raw_question_and_exact_top8():
    formal = contract()
    queries = build_raw_question_queries(formal["questions"])
    assert len(queries) == 98
    assert all(
        query.raw_question == question["question"]
        for query, question in zip(queries, formal["questions"])
    )
    source = (ROOT / "src/baselines/egopolice_b1/retrieval.py").read_text()
    assert "option" not in source.casefold()
    assert "ground_truth" not in source.casefold()
    assert "gt_interval" not in source.casefold()
    similarities, ranking = cosine_ranking(
        raw_question_embedding=np.array([1.0, 0.0], dtype=np.float32),
        normalized_visual_embeddings=np.array(
            [[0.0, 1.0], [1.0, 0.0], [0.8, 0.2], [0.6, 0.4],
             [0.4, 0.6], [0.2, 0.8], [-1.0, 0.0], [0.7, 0.3]],
            dtype=np.float32,
        ),
    )
    selected = select_top8_unique(
        ranking=ranking, timestamps_sec=np.arange(8, dtype=np.float64)
    )
    assert len(selected) == len(set(selected)) == 8
    assert similarities[selected[0]] == pytest.approx(similarities.max())


def test_b1_imports_no_forbidden_pipeline_components():
    forbidden = ("dinov", "segment", "caption", "audio", "planner", "router", "hierarch")
    for path in (ROOT / "src/baselines/egopolice_b1").glob("*.py"):
        tree = ast.parse(path.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        assert not [module for module in modules if any(word in module.casefold() for word in forbidden)]


def test_mock_retrieval_reuses_one_index_per_video_and_all_queries_are_raw(monkeypatch, tmp_path):
    formal = contract()
    load_counts = {}

    def valid_index(*, index_root, video, video_manifest_sha256):
        return True, {"index_fingerprint": f"index-{video['video_id']}"}, None

    def load_arrays(index_root, video_id):
        load_counts[video_id] = load_counts.get(video_id, 0) + 1
        visual = np.zeros((10, 1536), dtype=np.float32)
        visual[:, 0] = 1.0
        return {
            "embeddings": visual,
            "timestamps_sec": np.arange(10, dtype=np.float64),
            "frame_indices": np.arange(10, dtype=np.int64),
        }

    class Encoder:
        loads = 0
        queries = []

        def __init__(self, **kwargs):
            type(self).loads += 1
            self.model_load_latency_sec = 0.01
            self.actual_model_loading_mode = "mock"

        def encode(self, raw_question):
            type(self).queries.append(raw_question)
            value = np.zeros(1536, dtype=np.float32)
            value[0] = 1.0
            return {"embedding": value, "latency_sec": 0.01,
                    "peak_gpu_allocated_memory_bytes": 1}

        def close(self):
            pass

    monkeypatch.setattr(prepare_retrieval, "validate_index", valid_index)
    monkeypatch.setattr(prepare_retrieval, "load_index_arrays", load_arrays)
    result = prepare_retrieval.prepare_retrieval(
        data_root=DATA_ROOT, index_root=tmp_path / "indexes",
        output_dir=tmp_path / "outputs", cache_root=tmp_path / "cache",
        video_manifest_path=VIDEO_MANIFEST, question_manifest_path=QUESTION_MANIFEST,
        readiness_path=READINESS, config_path=CONFIG, device_name="cuda:0",
        dry_run=False, text_encoder_factory=Encoder,
    )
    assert result["valid_retrieval_count"] == 98
    assert Encoder.loads == 1
    assert Encoder.queries == [row["question"] for row in formal["questions"]]
    assert set(load_counts) == set(formal["videos"])
    assert set(load_counts.values()) == {1}
    records = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "outputs/retrieval_checkpoints").glob("*.json"))
    ]
    assert len(records) == 98
    assert all(row["retrieval_query_text"] == row["raw_question_text"] for row in records)
    assert all(row["answer_fields_used_for_retrieval"] is False for row in records)
    assert all(len({item["timestamp_sec"] for item in row["top_8_ranked"]}) == 8 for row in records)


def test_b1_interruption_resume_skips_only_valid_completed(monkeypatch, tmp_path):
    formal = contract()
    dependencies = fake_dependencies(formal)
    monkeypatch.setattr(runner, "_load_dependencies", lambda **kwargs: dependencies)
    output = tmp_path / "b1"
    first = {}
    with pytest.raises(RuntimeError, match="mock interruption"):
        runner.run_b1(
            data_root=DATA_ROOT, model_path=MODEL_PATH,
            index_root=tmp_path / "indexes", output_dir=output,
            video_manifest_path=VIDEO_MANIFEST, question_manifest_path=QUESTION_MANIFEST,
            readiness_path=READINESS, config_path=CONFIG, ffmpeg_path="ffmpeg",
            dry_run=False, execute_formal_inference=True,
            qwen_factory=qwen_factory(first, fail_at=3),
            environment_checker=mock_environment, frame_extractor=mock_extract,
            aggregate_when_complete=False,
        )
    assert first == {"loads": 1, "calls": 3}
    completed = sorted((output / "qa_checkpoints").glob("*.json"))
    assert len(completed) == 2
    saved_bytes = {path.name: path.read_bytes() for path in completed}

    second = {}
    result = runner.run_b1(
        data_root=DATA_ROOT, model_path=MODEL_PATH,
        index_root=tmp_path / "indexes", output_dir=output,
        video_manifest_path=VIDEO_MANIFEST, question_manifest_path=QUESTION_MANIFEST,
        readiness_path=READINESS, config_path=CONFIG, ffmpeg_path="ffmpeg",
        dry_run=False, execute_formal_inference=True,
        qwen_factory=qwen_factory(second), environment_checker=mock_environment,
        frame_extractor=mock_extract, aggregate_when_complete=False,
    )
    assert result["completed_count"] == 98
    assert second == {"loads": 1, "calls": 96}
    all_checkpoints = list((output / "qa_checkpoints").glob("*.json"))
    assert len(all_checkpoints) == 98
    assert len({json.loads(path.read_text())["question_id"] for path in all_checkpoints}) == 98
    records = [json.loads(path.read_text()) for path in all_checkpoints]
    summary = aggregation.aggregate_records(records)
    assert summary["overall"]["total"] == 98
    assert summary["overall"]["mean_model_facing_frames"] == 8
    assert summary["overall"]["total_qwen_calls"] == 98
    assert {
        name: row["total"]
        for name, row in summary["by_question_duration_class"].items()
    } == {"1s": 39, "10s": 38, "60s": 21}
    assert all(path.read_bytes() == saved_bytes[path.name] for path in completed)
    assert not list(output.rglob("*.tmp"))

    corrupted = sorted((output / "qa_checkpoints").glob("*.json"))[0]
    payload = json.loads(corrupted.read_text())
    payload["model_calls"] = 2
    corrupted.write_text(json.dumps(payload))
    repair = {}
    repaired = runner.run_b1(
        data_root=DATA_ROOT, model_path=MODEL_PATH,
        index_root=tmp_path / "indexes", output_dir=output,
        video_manifest_path=VIDEO_MANIFEST, question_manifest_path=QUESTION_MANIFEST,
        readiness_path=READINESS, config_path=CONFIG, ffmpeg_path="ffmpeg",
        dry_run=False, execute_formal_inference=True,
        qwen_factory=qwen_factory(repair), environment_checker=mock_environment,
        frame_extractor=mock_extract, aggregate_when_complete=False,
    )
    assert repaired["completed_count"] == 98
    assert repair == {"loads": 1, "calls": 1}
    assert list((output / "qa_checkpoints").glob(f"{corrupted.name}.invalid.*"))

    no_load = {}
    final = runner.run_b1(
        data_root=DATA_ROOT, model_path=MODEL_PATH,
        index_root=tmp_path / "indexes", output_dir=output,
        video_manifest_path=VIDEO_MANIFEST, question_manifest_path=QUESTION_MANIFEST,
        readiness_path=READINESS, config_path=CONFIG, ffmpeg_path="ffmpeg",
        dry_run=False, execute_formal_inference=True,
        qwen_factory=qwen_factory(no_load), environment_checker=mock_environment,
        frame_extractor=mock_extract, aggregate_when_complete=False,
    )
    assert final["already_complete"] is True
    assert no_load == {}
