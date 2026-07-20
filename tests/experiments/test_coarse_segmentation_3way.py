from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.experiments.prepare_coarse_segmentation_3way_manifest import build_manifest
from src.experiments.coarse_segmentation.comet_style import CometStyleConfig, segment as comet
from src.experiments.coarse_segmentation.dinov2_features import (
    frame_identity,
)
from src.experiments.coarse_segmentation.kts_style import (
    KTSConfig,
    _scatter_costs,
    segment as kts,
)
from src.experiments.coarse_segmentation.schema import make_segmentation, validate_segmentation


def synthetic_features() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    centers = np.eye(3, 8, dtype=np.float32)
    values = np.concatenate(
        [center + rng.normal(0, 0.01, (20, 8)) for center in centers], axis=0
    ).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values, np.arange(len(values), dtype=np.float64)


def test_normalized_schema_complete_ordered_and_deterministic() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    first = make_segmentation(
        video_id="v", method="current", video_duration=10.0,
        boundaries=[7.0, 3.0, 3.0], frame_timestamps=timestamps,
    )
    second = make_segmentation(
        video_id="v", method="current", video_duration=10.0,
        boundaries=[7.0, 3.0], frame_timestamps=timestamps,
    )
    assert first == second
    assert not validate_segmentation(first)
    assert first["segments"][0]["start"] == 0.0
    assert first["segments"][-1]["end"] == 10.0
    assert all(row["duration"] > 0 for row in first["segments"])


def test_comet_style_deterministic_and_no_duplicate_boundaries() -> None:
    features, timestamps = synthetic_features()
    first = comet(features, timestamps, CometStyleConfig())
    second = comet(features, timestamps, CometStyleConfig())
    assert first == second
    assert first["boundaries"] == sorted(set(first["boundaries"]))
    record = make_segmentation(
        video_id="v", method="comet_style_dinov2", video_duration=60.0,
        boundaries=first["boundaries"], frame_timestamps=timestamps,
    )
    assert not validate_segmentation(record)


def test_kts_published_dp_detects_piecewise_feature_changes() -> None:
    features, timestamps = synthetic_features()
    result = kts(
        features,
        timestamps,
        KTSConfig(max_change_points=8, minimum_segment_duration_sec=4.0),
    )
    assert result["boundaries"] == sorted(set(result["boundaries"]))
    assert result["diagnostics"]["selected_segment_count"] >= 2
    record = make_segmentation(
        video_id="v", method="kts_dinov2", video_duration=60.0,
        boundaries=result["boundaries"], frame_timestamps=timestamps,
    )
    assert not validate_segmentation(record)


def test_kts_kernel_scatter_matches_published_naive_definition() -> None:
    features, _ = synthetic_features()
    kernel = features @ features.T
    costs = _scatter_costs(kernel)
    for start, end in ((0, 7), (5, 23), (20, 40), (0, len(features))):
        block = kernel[start:end, start:end].astype(np.float64)
        expected = np.trace(block) - np.sum(block) / (end - start)
        assert np.isclose(costs[start, end], expected, atol=1e-8)


def test_comet_and_kts_share_same_immutable_dino_array() -> None:
    features, timestamps = synthetic_features()
    before = hashlib.sha256(features.tobytes()).hexdigest()
    comet(features, timestamps, CometStyleConfig())
    middle = hashlib.sha256(features.tobytes()).hexdigest()
    kts(features, timestamps, KTSConfig(max_change_points=8))
    after = hashlib.sha256(features.tobytes()).hexdigest()
    assert before == middle == after


def test_dino_cache_identity_is_reproducible() -> None:
    root = Path(__file__).resolve().parents[2]
    video = "994aecb6-ded3-4d5f-8f52-f0038a6dc057"
    frames = sorted((root / "outputs/visual_index" / video / "frames_1fps").glob("*.jpg"))[:3]
    assert len(frames) == 3
    assert frame_identity(frames) == frame_identity(list(frames))


def test_completed_run_reuses_one_exact_dino_cache_for_b_and_c() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/experiments/coarse_segmentation_3way_v0_1"
    run = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert run["method_b_and_c_cache_identity_equal"] is True
    assert len(run["shared_dinov2_cache_sha256_by_video"]) == 10
    for video_id, expected_sha in run["shared_dinov2_cache_sha256_by_video"].items():
        feature_path = out / "dinov2_cache" / video_id / "features.npy"
        actual_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        features = np.load(feature_path)
        assert actual_sha == expected_sha
        assert features.shape == (180, 384)
        assert np.isfinite(features).all()
        assert np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)


def test_all_completed_normalized_outputs_have_complete_coverage() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/experiments/coarse_segmentation_3way_v0_1"
    count = 0
    for name in ("current_segments.jsonl", "comet_segments.jsonl", "kts_segments.jsonl"):
        rows = [
            json.loads(line)
            for line in (out / name).read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert len(rows) == 10
        for row in rows:
            assert not validate_segmentation(row)
            count += 1
    assert count == 30


def test_manifest_determinism_and_no_gold_access() -> None:
    first = build_manifest()
    second = build_manifest()
    assert first == second
    assert len(first["videos"]) == 10
    assert first["qa_gold_used"] is False
    assert first["qa_correctness_used"] is False
    assert {row["group"] for row in first["videos"]} == {"problematic", "control"}


def test_experiment_source_has_no_paid_api_path() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "src/experiments/coarse_segmentation").glob("*.py")
    )
    assert "anthropic" not in sources.lower()
    assert "gemini" not in sources.lower()
    assert "GEMINI_API_KEY" not in sources
    assert "ANTHROPIC_API_KEY" not in sources
