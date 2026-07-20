from __future__ import annotations

import json
from pathlib import Path

from src.experiments.medium_semantic_abstraction.experiment import (
    hierarchy_views,
    load_jsonl,
    prepare_medium_records,
    select_representative,
)
from src.experiments.medium_semantic_abstraction.qwen_local import resolve_xsym


ROOT = Path(__file__).resolve().parents[2]


def _hierarchies():
    return load_jsonl(ROOT / "outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl")


def test_exact_frozen_ten_video_hierarchy_reuse() -> None:
    manifest = json.loads((ROOT / "data/manifests/coarse_segmentation_3way_10.json").read_text())
    assert [row["video_id"] for row in _hierarchies()] == [row["video_id"] for row in manifest["videos"]]


def test_medium_and_coarse_cut_membership_complete() -> None:
    hierarchy = _hierarchies()[0]
    views = hierarchy_views(hierarchy)
    assert sum(len(view["medium"]) for view in views) == hierarchy["cuts"]["medium"]["node_count"]
    assert len(views) == hierarchy["cuts"]["coarse"]["node_count"]


def test_midpoint_selection_is_shared_and_deterministic() -> None:
    hierarchy = _hierarchies()[0]
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    medium = nodes[hierarchy["cuts"]["medium"]["node_ids"][0]]
    assert select_representative(medium, 0.5) == select_representative(medium, 0.5)
    records, _ = prepare_medium_records(
        hierarchies=[hierarchy], root=ROOT,
        output_dir=ROOT / "outputs/experiments/medium_semantic_abstraction_v0_1/_unit_test_assets",
        one_fractions=[0.5], three_fractions=[0.25, 0.5, 0.75],
    )
    first = records[0]
    one = first["conditions"]["one_frame"]["selected_images"][0]
    three = next(row for row in first["conditions"]["three_frame"]["selected_images"] if row["fraction"] == 0.5)
    assert one["source_image_sha256"] == three["source_image_sha256"]


def test_local_model_snapshot_pointer_resolves_without_download() -> None:
    config = json.loads((ROOT / "config/experiments/medium_semantic_abstraction_v0_1.json").read_text())
    snapshot = Path(config["local_model"]["snapshot_path"])
    resolved = resolve_xsym(snapshot / "config.json")
    assert resolved.is_file() and resolved.stat().st_size > 1000


def test_config_forbids_fine_captioning_questions_gold_and_external_api() -> None:
    config = json.loads((ROOT / "config/experiments/medium_semantic_abstraction_v0_1.json").read_text())
    assert config["caption_fine_nodes"] is False
    assert config["question_conditioning"] is False
    assert config["gold_or_options_accessed"] is False
    assert config["external_api_calls"] == 0
    assert config["canonical_pipeline_modified"] is False


def test_completed_local_run_is_complete_and_zero_call() -> None:
    output = ROOT / "outputs/experiments/medium_semantic_abstraction_v0_1"
    manifest = json.loads((output / "run_manifest.json").read_text())
    medium = json.loads((output / "per_medium_results.json").read_text())
    coarse = json.loads((output / "per_coarse_results.json").read_text())
    runtime = json.loads((output / "runtime_metrics.json").read_text())
    validation = json.loads((output / "html_validation.json").read_text())
    assert manifest["external_api_calls"] == 0
    assert len(medium) == 82 and len(coarse) == 43
    assert runtime["model_load_count"] == 1
    assert runtime["local_inference_call_count"] == 250
    assert validation == {
        "video_count": 10,
        "medium_count": 82,
        "coarse_count": 43,
        "image_reference_count": 328,
        "missing_image_count": 0,
    }
    assert all(
        row["conditions"][condition]["timing"]["image_count"] == 0
        for row in coarse for condition in ("one_frame", "three_frame")
    )
