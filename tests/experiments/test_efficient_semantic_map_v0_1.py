from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.efficient_semantic_map_v0_1.experiment import (
    group_video_keyframes,
    hierarchy_views,
    load_jsonl,
    parse_story,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/experiments/efficient_semantic_map_v0_1.json"
SOURCE = ROOT / "outputs/experiments/long_video_hierarchy_stress_v0_2"
OUT = ROOT / "outputs/experiments/efficient_semantic_map_v0_1"


def test_source_is_exact_frozen_four_video_hierarchy() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hierarchies = load_jsonl(ROOT / config["source_hierarchy"])
    manifest = json.loads((ROOT / config["source_manifest"]).read_text(encoding="utf-8"))
    assert len(hierarchies) == len(manifest["videos"]) == 4
    assert {row["video_id"] for row in hierarchies} == {row["video_id"] for row in manifest["videos"]}
    assert sum(len(row["cuts"]["medium"]["node_ids"]) for row in hierarchies) == 149
    assert sum(len(row["cuts"]["coarse"]["node_ids"]) for row in hierarchies) == 76


def test_hierarchy_views_preserve_exact_medium_and_coarse_membership() -> None:
    hierarchy = load_jsonl(SOURCE / "safe_merge_hierarchies.jsonl")[0]
    before = json.dumps(hierarchy, sort_keys=True)
    views = hierarchy_views(hierarchy)
    assert [row["coarse"]["node_id"] for row in views] == hierarchy["cuts"]["coarse"]["node_ids"]
    assert {node["node_id"] for view in views for node in view["medium"]} == set(
        hierarchy["cuts"]["medium"]["node_ids"]
    )
    assert json.dumps(hierarchy, sort_keys=True) == before


def test_complete_link_grouping_is_conservative_deterministic_and_metadata_only() -> None:
    records = [
        {
            "video_id": "v", "medium_id": f"M{i}", "start": float(i), "end": float(i + 1),
            "selected_keyframe": {"feature_index": i, "selection_score": 0.9 - i * 0.1},
        }
        for i in range(3)
    ]
    features = np.asarray([[1.0, 0.0], [0.99999, 0.004], [0.0, 1.0]], dtype=np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    first, _ = group_video_keyframes(records, features=features, threshold=0.995, maximum_group_size=3)
    second_records = json.loads(json.dumps([
        {
            "video_id": "v", "medium_id": f"M{i}", "start": float(i), "end": float(i + 1),
            "selected_keyframe": {"feature_index": i, "selection_score": 0.9 - i * 0.1},
        }
        for i in range(3)
    ]))
    second, _ = group_video_keyframes(second_records, features=features, threshold=0.995, maximum_group_size=3)
    assert [row["member_count"] for row in first] == [2, 1]
    assert first == second
    assert [row["medium_id"] for row in records] == ["M0", "M1", "M2"]


def test_story_parser_preserves_overall_and_order() -> None:
    overall, items = parse_story(
        "Overall story: An officer documents an extended roadside interaction.\n"
        "High-level storyline:\n1. The officer stands near a vehicle.\n2. Several people speak nearby."
    )
    assert overall.startswith("An officer")
    assert items == ["The officer stands near a vehicle.", "Several people speak nearby."]


def test_config_has_local_only_model_and_no_qa_or_structural_mutation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["local_model"]["model"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert config["local_model"]["local_files_only"] is True
    assert config["external_api_calls"] == 0
    assert config["question_conditioning"] is False
    assert config["gold_or_options_accessed"] is False
    assert all(config["structure_policy"][key] for key in (
        "fine_boundaries_frozen", "medium_boundaries_frozen", "coarse_boundaries_frozen"
    ))
    assert config["structure_policy"]["semantic_remerge_allowed"] is False


def test_completed_output_contract_and_source_hash() -> None:
    aggregate = json.loads((OUT / "aggregate_metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((OUT / "html_validation.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "frozen_config.json").read_text(encoding="utf-8"))
    medium = json.loads((OUT / "medium_captions.json").read_text(encoding="utf-8"))
    groups = json.loads((OUT / "semantic_groups.json").read_text(encoding="utf-8"))
    assert aggregate["video_count"] == 4
    assert aggregate["total_medium_nodes"] == len(medium) == 149
    assert aggregate["total_semantic_groups"] == len(groups)
    assert aggregate["actual_direct_vlm_image_calls"] == len(groups)
    assert {row["caption_source"] for row in medium} <= {"direct_vlm", "propagated_from_group"}
    assert validation["displayed_video_count"] == 4
    assert validation["displayed_medium_selected_image_count"] == 149
    assert validation["missing_image_reference_count"] == 0
    assert frozen["source_hierarchy_unchanged"] is True
    assert frozen["source_hierarchy_sha256_before"] == sha256_file(SOURCE / "safe_merge_hierarchies.jsonl")
    assert (OUT / "semantic_map_review.html").is_file()


def test_text_stages_are_frame_free_and_api_calls_zero() -> None:
    coarse = json.loads((OUT / "coarse_captions.json").read_text(encoding="utf-8"))
    stories = json.loads((OUT / "video_stories.json").read_text(encoding="utf-8"))
    runtime = json.loads((OUT / "runtime_metrics.json").read_text(encoding="utf-8"))
    assert all(row["input_uses_text_only"] for row in coarse)
    assert all(row["input_uses_coarse_text_only"] for row in stories)
    assert all(row["high_level_storyline"] for row in stories)
    assert all(row["high_level_storyline_source"] in {
        "parsed_model_output", "ordered_coarse_caption_fallback_due_format_noncompliance"
    } for row in stories)
    assert runtime["external_api_calls"] == 0
    source = (ROOT / "scripts/experiments/run_efficient_semantic_map_v0_1.py").read_text(encoding="utf-8").lower()
    assert "anthropic" not in source and "openai" not in source and "gemini" not in source
