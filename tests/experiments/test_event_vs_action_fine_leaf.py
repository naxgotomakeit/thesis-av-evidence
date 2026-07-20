from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.experiments.event_vs_action_fine_leaf.motion import (
    MotionProxyConfig,
    build_action_children,
    exact_rbf_change_points,
    extract_or_load_motion_proxy,
    motion_cache_compatible,
    validate_action_partition,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/experiments/event_vs_action_fine_leaf_v0_1.json"
SOURCE = ROOT / "outputs/experiments/coarse_segmentation_3way_v0_1/comet_segments.jsonl"
MANIFEST = ROOT / "data/manifests/coarse_segmentation_3way_10.json"
OUT = ROOT / "outputs/experiments/event_vs_action_fine_leaf_v0_1"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def boundary_signature(row: dict) -> list[tuple]:
    return [
        (
            segment["segment_id"], float(segment["start"]), float(segment["end"]),
            float(segment["duration"]), float(segment["representative_frame_timestamp"]),
        )
        for segment in row["segments"]
    ]


def test_frozen_manifest_and_event_source_are_exactly_reused() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert config["input_manifest"] == "data/manifests/coarse_segmentation_3way_10.json"
    assert len(manifest["videos"]) == 10
    source = load_jsonl(SOURCE)
    assert [row["video_id"] for row in source] == [row["video_id"] for row in manifest["videos"]]
    if (OUT / "event_level_segments.jsonl").exists():
        output = load_jsonl(OUT / "event_level_segments.jsonl")
        for left, right in zip(source, output):
            assert left["video_id"] == right["video_id"]
            assert boundary_signature(left) == boundary_signature(right)
            assert right["source_boundary_hash"] == right["output_boundary_hash"]


def test_rbf_change_points_are_deterministic_and_not_count_forced() -> None:
    values = np.vstack([
        np.zeros((6, 3), dtype=np.float64),
        np.full((6, 3), 5.0, dtype=np.float64),
    ])
    first = exact_rbf_change_points(values, minimum_size=3, penalty_multiplier=1.0)
    second = exact_rbf_change_points(values, minimum_size=3, penalty_multiplier=1.0)
    assert first == second
    assert first["change_point_indices"] == [6]
    assert exact_rbf_change_points(np.zeros((5, 2)), minimum_size=3, penalty_multiplier=1.0)["change_point_indices"] == []


def test_actions_cover_exactly_one_parent_without_gaps_or_overlaps() -> None:
    events = [
        {"segment_id": "event-0", "start": 0.0, "end": 12.0, "duration": 12.0},
        {"segment_id": "event-1", "start": 12.0, "end": 18.0, "duration": 6.0},
    ]
    starts = np.arange(18, dtype=np.float64)
    features = np.vstack([
        np.zeros((6, 2)), np.full((6, 2), 5.0), np.ones((6, 2)),
    ])
    motion = {
        "features": features,
        "interval_starts": starts,
        "interval_ends": starts + 1.0,
        "raw_motion": features[:, 0],
        "smoothed_motion": features[:, 0],
    }
    actions, decisions = build_action_children(
        video_id="synthetic", event_segments=events, motion=motion,
        config=MotionProxyConfig(minimum_action_duration_sec=3.0, penalty_multiplier=1.0),
    )
    assert validate_action_partition(events, actions)["valid"] is True
    assert all(action["parent_event_id"] in {"event-0", "event-1"} for action in actions)
    assert decisions[0]["change_points"][0]["timestamp"] == 6.0
    assert [row["start"] for row in actions if row["parent_event_id"] == "event-0"] == [0.0, 6.0]
    assert [row["end"] for row in actions if row["parent_event_id"] == "event-0"] == [6.0, 12.0]


def test_motion_cache_requires_exact_provenance_and_reuses_compatible_cache() -> None:
    cache_root = OUT / "_deterministic_test_cache"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)
    try:
        frames = []
        for index, value in enumerate((0, 64, 192)):
            path = cache_root / f"frame_{index:06d}.jpg"
            Image.fromarray(np.full((12, 16), value, dtype=np.uint8)).save(path)
            frames.append(path)
        npz = cache_root / "motion.npz"
        metadata = cache_root / "motion.json"
        cfg = MotionProxyConfig(image_width=16, image_height=12)
        first_arrays, first = extract_or_load_motion_proxy(
            video_id="cache-test", frame_paths=frames,
            timestamps=np.asarray([0.0, 1.0, 2.0]), video_duration=3.0,
            cache_npz=npz, cache_metadata=metadata, config=cfg,
        )
        assert first["cache_hit"] is False
        with patch(
            "src.experiments.event_vs_action_fine_leaf.motion.extract_motion_proxy",
            side_effect=AssertionError("compatible cache should avoid extraction"),
        ):
            second_arrays, second = extract_or_load_motion_proxy(
                video_id="cache-test", frame_paths=frames,
                timestamps=np.asarray([0.0, 1.0, 2.0]), video_duration=3.0,
                cache_npz=npz, cache_metadata=metadata, config=cfg,
            )
        assert second["cache_hit"] is True
        assert np.array_equal(first_arrays["features"], second_arrays["features"])
        stored = json.loads(metadata.read_text(encoding="utf-8"))
        assert motion_cache_compatible(stored, stored)
        mutated = {**stored, "frame_count": stored["frame_count"] + 1}
        assert not motion_cache_compatible(stored, mutated)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def test_experiment_is_zero_api_zero_gold_and_does_not_import_canonical() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["external_api_calls"] == 0
    assert config["canonical_pipeline_modified"] is False
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "src/experiments/event_vs_action_fine_leaf/motion.py",
            ROOT / "src/experiments/event_vs_action_fine_leaf/reporting.py",
            ROOT / "scripts/experiments/run_event_vs_action_fine_leaf.py",
        ]
    ).lower()
    assert "src.canonical_pipeline" not in sources
    assert "anthropic" not in sources
    assert "google.generativeai" not in sources
    assert "openai" not in sources
    assert "requests." not in sources
    assert "httpx" not in sources


def test_run_artifacts_validate_when_present() -> None:
    if not (OUT / "action_level_segments.jsonl").exists():
        return
    event_rows = load_jsonl(OUT / "event_level_segments.jsonl")
    action_rows = load_jsonl(OUT / "action_level_segments.jsonl")
    assert len(event_rows) == len(action_rows) == 10
    for event_row, action_row in zip(event_rows, action_rows):
        assert event_row["video_id"] == action_row["video_id"]
        events = event_row["segments"]
        actions = action_row["segments"]
        assert validate_action_partition(events, actions)["valid"] is True
        assert all(float(row["duration"]) > 0 for row in actions)
    run_manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["external_api_calls"] == 0
    assert run_manifest["gold_answers_used"] is False
    assert run_manifest["answer_options_used"] is False
    assert run_manifest["event_boundaries_exact"] is True
