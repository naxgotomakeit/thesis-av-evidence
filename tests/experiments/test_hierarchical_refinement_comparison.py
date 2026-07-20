from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.experiments.coarse_segmentation.comet_style import CometStyleConfig
from src.experiments.coarse_segmentation.retrieval_replay import replay
from src.experiments.hierarchical_refinement.pipeline import (
    local_comet_segmentation,
    merge_selected_kts_units,
    validate_local_coverage,
)


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "outputs/experiments/coarse_segmentation_3way_v0_1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition() -> list[dict]:
    return [
        {"segment_id": "k0", "start": 0.0, "end": 10.0, "duration": 10.0},
        {"segment_id": "k1", "start": 10.0, "end": 20.0, "duration": 10.0},
        {"segment_id": "k2", "start": 20.0, "end": 30.0, "duration": 10.0},
        {"segment_id": "k3", "start": 30.0, "end": 40.0, "duration": 10.0},
    ]


def selected(*ids: str) -> list[dict]:
    return [
        {"segment_id": identifier, "rank": rank, "score": 1.0 / rank}
        for rank, identifier in enumerate(ids, start=1)
    ]


def test_exact_frozen_manifest_and_configs_are_reused() -> None:
    manifest = ROOT / "data/manifests/coarse_segmentation_3way_10.json"
    run = load(PREVIOUS / "run_manifest.json")
    previous = load(PREVIOUS / "frozen_config.json")
    experiment = load(ROOT / "config/experiments/hierarchical_refinement_comparison_v0_1.json")
    assert sha256(manifest) == run["manifest_sha256"]
    assert experiment["source_manifest"] == "data/manifests/coarse_segmentation_3way_10.json"
    assert previous["kts"]["max_change_points"] == 30
    assert previous["comet_style"]["smoothing_sigma_frames"] == 1.0
    assert experiment["primary_kts_top_m"] == 3


def test_all_previous_dino_caches_match_run_manifest() -> None:
    run = load(PREVIOUS / "run_manifest.json")
    expected = run["shared_dinov2_cache_sha256_by_video"]
    assert len(expected) == 10
    for video_id, digest in expected.items():
        path = PREVIOUS / "dinov2_cache" / video_id / "features.npy"
        assert sha256(path) == digest


def test_stable_descending_top3_ranking_uses_shared_clip_scorer() -> None:
    segmentation = {
        "video_id": "v",
        "method": "kts_dinov2",
        "video_duration": 4.0,
        "segments": [
            {"segment_id": f"k{i}", "start": float(i), "end": float(i + 1), "duration": 1.0}
            for i in range(4)
        ],
    }
    regions = np.asarray([[1, 0], [1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
    result = replay(segmentation, regions, np.asarray([1, 0], dtype=np.float32), 3)
    assert [row["segment_id"] for row in result["selected_top_k"]] == ["k0", "k1", "k2"]


def test_adjacent_selected_units_merge() -> None:
    zones = merge_selected_kts_units(partition(), selected("k1", "k2", "k0"))
    assert len(zones) == 1
    assert (zones[0]["start"], zones[0]["end"]) == (0.0, 30.0)
    assert zones[0]["selected_kts_ids"] == ["k0", "k1", "k2"]


def test_non_adjacent_selected_units_remain_separate() -> None:
    zones = merge_selected_kts_units(partition(), selected("k0", "k2"))
    assert [(row["start"], row["end"]) for row in zones] == [(0.0, 10.0), (20.0, 30.0)]


def test_local_comet_stays_inside_zones_with_complete_coverage() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(40, 16)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    zones = merge_selected_kts_units(partition(), selected("k1", "k3"))
    record, diagnostics = local_comet_segmentation(
        video_id="v",
        video_duration=40.0,
        dino_features=features,
        frame_timestamps=np.arange(40, dtype=np.float64),
        candidate_zones=zones,
        config=CometStyleConfig(),
    )
    assert not validate_local_coverage(record)
    assert len(diagnostics) == 2
    for segment in record["segments"]:
        zone = next(row for row in zones if row["zone_id"] == segment["candidate_zone_id"])
        assert zone["start"] <= segment["start"] < segment["end"] <= zone["end"]


def test_experiment_has_no_paid_api_or_gold_data_path() -> None:
    source_paths = [
        ROOT / "src/experiments/hierarchical_refinement/pipeline.py",
        ROOT / "scripts/experiments/run_hierarchical_refinement_comparison.py",
    ]
    sources = "\n".join(path.read_text(encoding="utf-8") for path in source_paths).lower()
    assert "anthropic" not in sources
    assert "gemini" not in sources
    assert "api_key" not in sources
    config = load(ROOT / "config/experiments/hierarchical_refinement_comparison_v0_1.json")
    assert config["gold_accessed"] is False
    assert config["qa_correctness_accessed"] is False
    assert config["external_api_calls"] == 0


def test_completed_artifacts_preserve_reuse_and_local_coverage_contracts() -> None:
    out = ROOT / "outputs/experiments/hierarchical_refinement_comparison_v0_1"
    run = load(out / "run_manifest.json")
    assert run["case_count"] == 10
    assert run["dino_cache_exactly_reused"] is True
    assert run["frozen_kts_config_exactly_reused"] is True
    assert run["frozen_comet_config_exactly_reused"] is True
    assert run["shared_clip_scoring_regression_passed"] is True
    assert run["dino_inference_calls"] == 0
    assert run["paid_api_calls"] == 0
    assert run["gold_accessed"] is False
    rows = [
        json.loads(line)
        for line in (out / "kts_local_comet_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 10
    for row in rows:
        record = {
            "candidate_zones": row["candidate_zones"],
            "segments": row["local_comet_segments"],
        }
        assert not validate_local_coverage(record)
        assert row["gold_used"] is False
        assert row["qa_correctness_used"] is False


def test_review_html_renders_all_ten_cases_without_external_assets() -> None:
    out = ROOT / "outputs/experiments/hierarchical_refinement_comparison_v0_1"
    document = (out / "comparison.html").read_text(encoding="utf-8")
    manifest = load(ROOT / "data/manifests/coarse_segmentation_3way_10.json")
    assert all(f"id='{row['video_id']}'" in document for row in manifest["videos"])
    assert document.count("Manual review — intentionally unfilled") == 10
    assert document.count("data:image/jpeg;base64,") >= 200
    assert "src='http" not in document
