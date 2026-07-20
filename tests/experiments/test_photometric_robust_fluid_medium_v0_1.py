from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.experiments.adaptive_fluid_hierarchy_v0_1.experiment import fine_leaf_hash, topology_hash
from src.experiments.photometric_robust_fluid_medium_v0_1.experiment import (
    absolute_confidences,
    absolute_frontier,
    boundary_diagnostics,
    load_jsonl,
    validate_medium_frontier,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/experiments/photometric_robust_fluid_medium_v0_1.json"
SOURCE = ROOT / "outputs/experiments/long_video_hierarchy_stress_v0_2/safe_merge_hierarchies.jsonl"
OUT = ROOT / "outputs/experiments/photometric_robust_fluid_medium_v0_1"


def tiny_hierarchy() -> dict:
    leaf_base = {
        "node_type": "fine_leaf", "child_ids": [], "frame_count": 2,
        "internal_variability": 0.05, "internal_boundary_ids": [],
        "representative_frames": [{"kind": "temporal_50_percent", "timestamp": 0.5, "frame_path": "unused.jpg"}],
        "pooled_dinov2": [1.0, 0.0],
    }
    nodes = [
        {**leaf_base, "node_id": "F1", "parent_id": "L", "start": 0.0, "end": 1.0, "duration": 1.0, "leaf_ids": ["F1"]},
        {**leaf_base, "node_id": "F2", "parent_id": "L", "start": 1.0, "end": 2.0, "duration": 1.0, "leaf_ids": ["F2"]},
        {**leaf_base, "node_id": "F3", "parent_id": "R", "start": 2.0, "end": 3.0, "duration": 1.0, "leaf_ids": ["F3"]},
        {**leaf_base, "node_id": "F4", "parent_id": "R", "start": 3.0, "end": 4.0, "duration": 1.0, "leaf_ids": ["F4"]},
        {"node_id": "L", "node_type": "internal", "parent_id": "ROOT", "child_ids": ["F1", "F2"], "start": 0.0, "end": 2.0, "duration": 2.0, "leaf_ids": ["F1", "F2"], "frame_count": 4, "internal_variability": 0.06, "internal_boundary_ids": ["B1"], "representative_frames": [], "pooled_dinov2": [1.0, 0.0]},
        {"node_id": "R", "node_type": "internal", "parent_id": "ROOT", "child_ids": ["F3", "F4"], "start": 2.0, "end": 4.0, "duration": 2.0, "leaf_ids": ["F3", "F4"], "frame_count": 4, "internal_variability": 0.06, "internal_boundary_ids": ["B3"], "representative_frames": [], "pooled_dinov2": [1.0, 0.0]},
        {"node_id": "ROOT", "node_type": "internal", "parent_id": None, "child_ids": ["L", "R"], "start": 0.0, "end": 4.0, "duration": 4.0, "leaf_ids": ["F1", "F2", "F3", "F4"], "frame_count": 8, "internal_variability": 0.2, "internal_boundary_ids": ["B1", "B2", "B3"], "representative_frames": [], "pooled_dinov2": [1.0, 0.0]},
    ]
    boundaries = [
        {"boundary_id": "B1", "left_leaf_id": "F1", "right_leaf_id": "F2", "timestamp": 1.0},
        {"boundary_id": "B2", "left_leaf_id": "F2", "right_leaf_id": "F3", "timestamp": 2.0},
        {"boundary_id": "B3", "left_leaf_id": "F3", "right_leaf_id": "F4", "timestamp": 3.0},
    ]
    merges = [
        {"parent_id": "L", "child_ids": ["F1", "F2"], "merge_score": 0.1, "semantic_difference": 0.02, "inter_boundary_strength": 0.2, "merged_internal_variability": 0.06},
        {"parent_id": "R", "child_ids": ["F3", "F4"], "merge_score": 0.1, "semantic_difference": 0.02, "inter_boundary_strength": 0.2, "merged_internal_variability": 0.06},
        {"parent_id": "ROOT", "child_ids": ["L", "R"], "merge_score": 0.8, "semantic_difference": 0.4, "inter_boundary_strength": 0.8, "merged_internal_variability": 0.2},
    ]
    return {
        "video_id": "synthetic", "video_duration": 4.0, "root_id": "ROOT", "nodes": nodes,
        "accepted_merge_order": merges, "boundary_records": boundaries,
        "cuts": {"fine": {"node_ids": ["F1", "F2", "F3", "F4"]}},
    }


def diagnostic_map(*, photo: bool = False, persistent: bool = False, transient: bool = False) -> dict:
    return {
        boundary: {
            "boundary_id": boundary,
            "suspected_photometric_by_profile": {"strict": photo, "balanced": photo, "loose": photo},
            "persistent_state_change": persistent,
            "transient_or_reverting_change": transient,
        }
        for boundary in ("B1", "B2", "B3")
    }


def test_photometric_boundary_increases_merge_confidence_without_changing_tree() -> None:
    hierarchy = tiny_hierarchy()
    before = json.dumps(hierarchy, sort_keys=True)
    base, _ = absolute_confidences(hierarchy, diagnostic_map())
    photo, details = absolute_confidences(hierarchy, diagnostic_map(photo=True), photo_profile="balanced")
    assert photo["ROOT"] > base["ROOT"]
    assert details["ROOT"]["photometric_downweight_applied"] is True
    assert json.dumps(hierarchy, sort_keys=True) == before


def test_persistent_change_is_not_temporally_downweighted() -> None:
    hierarchy = tiny_hierarchy()
    photo, _ = absolute_confidences(
        hierarchy, diagnostic_map(photo=True, persistent=True, transient=True),
        photo_profile="balanced", use_temporal=True,
    )
    photo_only, _ = absolute_confidences(
        hierarchy, diagnostic_map(photo=True, persistent=True, transient=False),
        photo_profile="balanced", use_temporal=False,
    )
    assert photo == photo_only


def test_transient_reversion_is_further_downweighted() -> None:
    hierarchy = tiny_hierarchy()
    photo, _ = absolute_confidences(hierarchy, diagnostic_map(photo=True), photo_profile="balanced")
    temporal, details = absolute_confidences(
        hierarchy, diagnostic_map(photo=True, transient=True), photo_profile="balanced", use_temporal=True,
    )
    assert temporal["ROOT"] > photo["ROOT"]
    assert details["ROOT"]["temporal_downweight_applied"] is True


def test_mixed_branch_can_stop_at_different_depths() -> None:
    hierarchy = tiny_hierarchy()
    details = {key: {} for key in ("L", "R", "ROOT")}
    selected, _ = absolute_frontier(
        hierarchy, {"L": 0.95, "R": 0.7, "ROOT": 0.7}, details,
        minimum_confidence=0.85, maximum_local_drop=0.1, method="synthetic",
    )
    assert selected == ["L", "F3", "F4"]
    assert validate_medium_frontier(hierarchy, selected)["valid"]


def make_boundary_fixture(tmp_path: Path, state: str) -> tuple[dict, list[Path], np.ndarray, np.ndarray, np.ndarray]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    count = 22
    paths = []
    for index in range(count):
        value = 40 if index < 10 else 210
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        image[..., 0] = value
        image[..., 1] = value // 2
        image[..., 2] = 255 - value
        path = tmp_path / f"frame_{index:03d}.jpg"
        Image.fromarray(image).save(path)
        paths.append(path)
    a, b = np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
    original = np.tile(a, (count, 1))
    original[10] = b
    normalized = np.tile(a, (count, 1))
    if state == "persistent":
        normalized[11:] = b
    elif state == "transient":
        normalized[11:16] = b
    hierarchy = {
        "video_id": state,
        "boundary_records": [{
            "boundary_id": "B", "left_leaf_id": "F1", "right_leaf_id": "F2",
            "timestamp": 10.0, "signal_index": 9, "raw_change_strength": 0.2,
        }],
    }
    return hierarchy, paths, np.arange(count, dtype=np.float64), original, normalized


def test_pure_photometric_flicker_is_detected() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hierarchy, paths, timestamps, original, normalized = make_boundary_fixture(
        OUT / "test_fixture_images/flicker", "flicker"
    )
    rows = boundary_diagnostics(
        hierarchy, paths, timestamps, original, normalized,
        config["photometric_detection_profiles"], config["persistence"],
    )
    assert rows[0]["suspected_photometric_by_profile"]["balanced"] is True


def test_persistent_and_transient_state_classification() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    persistent = make_boundary_fixture(OUT / "test_fixture_images/persistent", "persistent")
    transient = make_boundary_fixture(OUT / "test_fixture_images/transient", "transient")
    persistent_row = boundary_diagnostics(
        *persistent, config["photometric_detection_profiles"], config["persistence"]
    )[0]
    transient_row = boundary_diagnostics(
        *transient, config["photometric_detection_profiles"], config["persistence"]
    )[0]
    assert persistent_row["persistent_state_change"] is True
    assert transient_row["persistent_state_change"] is False
    assert transient_row["transient_or_reverting_change"] is True


def test_frozen_config_and_source_identity() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["generate_coarse"] is False
    assert config["vlm_calls"] == config["external_api_calls"] == config["downloads"] == 0
    hierarchy = load_jsonl(SOURCE)[0]
    before = json.dumps(hierarchy, sort_keys=True)
    assert len(fine_leaf_hash(hierarchy)) == len(topology_hash(hierarchy)) == 64
    assert json.dumps(hierarchy, sort_keys=True) == before


@pytest.mark.skipif(not (OUT / "per_video_metrics.json").is_file(), reason="completed output not generated")
def test_completed_outputs_are_valid_and_zero_call() -> None:
    rows = json.loads((OUT / "per_video_metrics.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen_config.json").read_text(encoding="utf-8"))
    runtime = json.loads((OUT / "runtime_metrics.json").read_text(encoding="utf-8"))
    html_validation = json.loads((OUT / "html_validation.json").read_text(encoding="utf-8"))
    assert len(rows) == 14 * 10
    assert all(item["validity"]["valid"] for item in rows)
    assert all(item["validity"]["fine_leaf_preservation_rate"] == 1.0 for item in rows)
    assert frozen["source_files_unchanged"] and frozen["tree_identities_unchanged"]
    assert runtime["external_api_calls"] == runtime["vlm_calls"] == runtime["downloads"] == 0
    assert html_validation["video_count"] == 14
    assert html_validation["method_video_sections"] == 140
    assert html_validation["missing_image_reference_count"] == 0
    for filename in (
        "medium_comparison.html", "method_summary.json", "per_video_metrics.json",
        "flash_case_diagnostics.json", "photometric_boundary_diagnostics.json",
        "temporal_persistence_diagnostics.json", "fragmentation_metrics.json",
        "overmerge_risk_metrics.json", "complexity_alignment.json", "runtime_metrics.json",
        "frozen_config.json", "run_manifest.json", "README.md",
    ):
        assert (OUT / filename).is_file()


@pytest.mark.skipif(not (OUT / "photometric_boundary_diagnostics.json").is_file(), reason="completed output not generated")
def test_completed_absolute_frontier_is_deterministic() -> None:
    hierarchy = load_jsonl(SOURCE)[0]
    diagnostics = json.loads((OUT / "photometric_boundary_diagnostics.json").read_text(encoding="utf-8"))
    by_id = {
        item["boundary_id"]: item for item in diagnostics if item["video_id"] == hierarchy["video_id"]
    }
    confidence, details = absolute_confidences(
        hierarchy, by_id, photo_profile="balanced", use_temporal=True
    )
    first = absolute_frontier(
        hierarchy, confidence, details, minimum_confidence=0.85,
        maximum_local_drop=0.1, method="determinism",
    )
    second = absolute_frontier(
        hierarchy, confidence, details, minimum_confidence=0.85,
        maximum_local_drop=0.1, method="determinism",
    )
    assert first == second


def test_no_canonical_tracked_diff() -> None:
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    assert not any(path.startswith(("src/pipeline/", "config/canonical_pipeline", "tasks/")) for path in changed)
