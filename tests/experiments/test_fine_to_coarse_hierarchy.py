from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.fine_to_coarse_hierarchy.hierarchy import (
    build_boundary_records,
    build_fine_nodes,
    build_safe_hierarchy,
    build_ward_hierarchy,
    descendant_fine_ids,
    validate_hierarchy,
)


ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> tuple[list[dict], dict, np.ndarray, np.ndarray, list[str], list[dict]]:
    segments = [
        {"segment_id": f"C{i + 1}", "start": float(i * 2), "end": float((i + 1) * 2), "duration": 2.0}
        for i in range(4)
    ]
    features = np.asarray(
        [
            [1.0, 0.0], [0.99, 0.01],
            [0.98, 0.02], [0.99, 0.01],
            [-1.0, 0.0], [-0.99, 0.01],
            [-0.98, 0.02], [-0.99, 0.01],
        ],
        dtype=np.float32,
    )
    timestamps = np.arange(8, dtype=np.float64)
    paths = [f"outputs/visual_index/video/frames_1fps/frame_{i:06d}.jpg" for i in range(8)]
    # Fine boundaries at frame-grid indices 1, 3, 5. The middle boundary is deliberately strong.
    diagnostics = {
        "boundaries": [2.0, 4.0, 6.0],
        "raw_similarity": [0.99, 0.99, 0.98, 0.10, 0.98, 0.99, 0.99],
        "smoothed_similarity": [0.99, 0.98, 0.97, 0.10, 0.97, 0.98, 0.99],
        "diagnostics": {"boundary_indices": [1, 3, 5]},
    }
    boundaries = build_boundary_records(segments, diagnostics)
    fine = build_fine_nodes(
        segments=segments,
        features=features,
        timestamps=timestamps,
        frame_paths=paths,
        boundary_records=boundaries,
    )
    return segments, diagnostics, features, timestamps, paths, fine


def _build() -> tuple[dict, list[dict], dict, list[dict], list[dict], list[dict]]:
    segments, _, features, timestamps, paths, fine = _fixture()
    boundaries = build_boundary_records(
        segments,
        {
            "boundaries": [2.0, 4.0, 6.0],
            "raw_similarity": [0.99, 0.99, 0.98, 0.10, 0.98, 0.99, 0.99],
            "smoothed_similarity": [0.99, 0.98, 0.97, 0.10, 0.97, 0.98, 0.99],
            "diagnostics": {"boundary_indices": [1, 3, 5]},
        },
    )
    ward, ward_log = build_ward_hierarchy(
        video_id="video", video_duration=8.0, fine_nodes=fine,
        boundary_records=boundaries, timestamps=timestamps, frame_paths=paths,
        features=features,
    )
    safe, safe_log = build_safe_hierarchy(
        video_id="video", video_duration=8.0, fine_nodes=fine,
        boundary_records=boundaries, timestamps=timestamps, frame_paths=paths,
        features=features,
    )
    return ward, ward_log, safe, safe_log, fine, boundaries


def test_exact_frozen_ten_video_manifest_and_sources_are_reused() -> None:
    config = json.loads(
        (ROOT / "config/experiments/fine_to_coarse_hierarchy_v0_1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "data/manifests/coarse_segmentation_3way_10.json").read_text(encoding="utf-8")
    )
    assert config["source_manifest"] == "data/manifests/coarse_segmentation_3way_10.json"
    assert len(manifest["videos"]) == 10
    assert config["fine_source"] == "comet_segments.jsonl"
    assert config["kts_reference_source"] == "kts_segments.jsonl"
    assert config["dinov2_cache_source"] == "dinov2_cache"


def test_fine_boundaries_and_ids_are_immutable() -> None:
    segments, _, _, _, _, fine = _fixture()
    assert [(row["node_id"], row["start"], row["end"]) for row in fine] == [
        (row["segment_id"], row["start"], row["end"]) for row in segments
    ]
    assert all(row["node_type"] == "fine_leaf" for row in fine)
    assert all(len(row["pooled_dinov2"]) == 2 for row in fine)


def test_full_trees_preserve_all_fine_leaves_and_are_reversible() -> None:
    ward, _, safe, _, fine, _ = _build()
    for hierarchy in (ward, safe):
        assert hierarchy["invariants"]["valid"] is True
        assert hierarchy["invariants"]["fine_leaf_preservation_rate"] == 1.0
        assert hierarchy["invariants"]["reversible_to_original_fine_leaves"] is True
        nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
        assert descendant_fine_ids(hierarchy["root_id"], nodes) == [row["node_id"] for row in fine]
        assert len(nodes) == 2 * len(fine) - 1


def test_every_merge_is_temporally_adjacent_and_parent_exactly_covers_children() -> None:
    ward, _, safe, _, _, _ = _build()
    for hierarchy in (ward, safe):
        nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
        for node in nodes.values():
            if node["node_type"] == "fine_leaf":
                continue
            left, right = (nodes[identifier] for identifier in node["child_ids"])
            assert left["rightmost_leaf_index"] + 1 == right["leftmost_leaf_index"]
            assert left["end"] == right["start"]
            assert node["start"] == left["start"]
            assert node["end"] == right["end"]
            assert left["parent_id"] == node["node_id"]
            assert right["parent_id"] == node["node_id"]


def test_cuts_are_complete_ordered_partitions_without_gaps_or_overlaps() -> None:
    ward, _, safe, _, _, _ = _build()
    for hierarchy in (ward, safe):
        assert validate_hierarchy(hierarchy)["valid"] is True
        nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
        for cut in ("fine", "medium", "coarse"):
            rows = [nodes[node_id] for node_id in hierarchy["cuts"][cut]["node_ids"]]
            assert rows[0]["start"] == 0.0
            assert rows[-1]["end"] == 8.0
            assert all(left["end"] == right["start"] for left, right in zip(rows[:-1], rows[1:]))


def test_merge_order_is_deterministic() -> None:
    first = _build()
    second = _build()
    assert first[0]["accepted_merge_order"] == second[0]["accepted_merge_order"]
    assert first[2]["accepted_merge_order"] == second[2]["accepted_merge_order"]
    assert first[1] == second[1]
    assert first[3] == second[3]


def test_strong_boundary_veto_preserves_middle_boundary_at_safe_medium() -> None:
    _, _, safe, safe_log, _, boundaries = _build()
    assert boundaries[1]["very_strong_fine_to_medium_veto"] is True
    nodes = {row["node_id"]: row for row in safe["nodes"]}
    medium = [nodes[node_id] for node_id in safe["cuts"]["medium"]["node_ids"]]
    assert [node["leaf_ids"] for node in medium] == [["C1", "C2"], ["C3", "C4"]]
    assert any(
        not row["accepted"]
        and row["inter_boundary_id"] == boundaries[1]["boundary_id"]
        and row["reason"] == "very_strong_fine_boundary_veto_during_fine_to_medium"
        for row in safe_log
    )


def test_ward_uses_only_adjacent_candidate_pairs() -> None:
    _, ward_log, _, _, _, _ = _build()
    forbidden = {("C1", "C3"), ("C1", "C4"), ("C2", "C4")}
    assert not any((row["left_id"], row["right_id"]) in forbidden for row in ward_log)


def test_experiment_has_no_api_or_canonical_runtime_path() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "src/experiments/fine_to_coarse_hierarchy/hierarchy.py",
            ROOT / "scripts/experiments/run_fine_to_coarse_hierarchy.py",
        ]
    ).lower()
    for marker in ("import anthropic", "import google.genai", "messages.create", "generate_content"):
        assert marker not in sources
    assert "src.canonical_pipeline" not in sources
    assert "egoschema_comparison_pilot" not in sources
    assert "option0" not in sources


def test_configuration_explicitly_forbids_qa_inputs_and_new_inference() -> None:
    config = json.loads(
        (ROOT / "config/experiments/fine_to_coarse_hierarchy_v0_1.json").read_text(encoding="utf-8")
    )
    assert config["paid_api_calls"] == 0
    assert config["new_dinov2_inference"] == 0
    assert "gold answer" in config["prohibited_inputs"]
    assert "answer options" in config["prohibited_inputs"]
    assert config["canonical_pipeline_modified"] is False
