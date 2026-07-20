from __future__ import annotations

import json
from pathlib import Path

from src.experiments.long_video_hierarchy_stress.experiment import (
    hierarchy_metrics,
    select_candidates,
    validate_cut_nesting,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_selection_is_deterministic_and_reports_duration_limit() -> None:
    config = json.loads((ROOT / "config/experiments/long_video_hierarchy_stress_v0_1.json").read_text())
    rows = [
        {"video_id": video_id, "duration_sec": 180.0, "source_path": f"D:/{video_id}.mp4"}
        for video_id in config["selection"]["selected_video_ids"]
    ]
    selected, inventory = select_candidates(
        rows, selected_ids=config["selection"]["selected_video_ids"], minimum_long_sec=300.0
    )
    assert [row["video_id"] for row in selected] == config["selection"]["selected_video_ids"]
    assert inventory["true_long_video_stress_possible"] is False


def test_existing_safe_hierarchy_has_exact_cut_nesting() -> None:
    path = ROOT / "outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl"
    hierarchy = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    result = validate_cut_nesting(hierarchy)
    assert result["valid"]
    assert result["fine_count"] == result["fine_assigned_to_medium"]
    assert result["medium_count"] == result["medium_assigned_to_coarse"]


def test_metrics_do_not_modify_hierarchy() -> None:
    path = ROOT / "outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl"
    hierarchy = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    before = json.dumps(hierarchy, sort_keys=True)
    result = hierarchy_metrics(hierarchy, chaining_ratio=0.85)
    assert result["fine_preservation_rate"] == 1.0
    assert json.dumps(hierarchy, sort_keys=True) == before


def test_config_prohibits_question_gold_and_api() -> None:
    config = json.loads((ROOT / "config/experiments/long_video_hierarchy_stress_v0_1.json").read_text())
    prohibited = " ".join(config["prohibited_inputs"]).lower()
    assert "question" in prohibited and "gold" in prohibited
    assert config["external_api_calls"] == 0
    assert config["canonical_pipeline_modified"] is False


def test_completed_output_has_frozen_five_and_valid_html() -> None:
    output = ROOT / "outputs/experiments/long_video_hierarchy_stress_v0_1"
    manifest = json.loads((output / "run_manifest.json").read_text())
    config = json.loads((ROOT / "config/experiments/long_video_hierarchy_stress_v0_1.json").read_text())
    validation = json.loads((output / "html_validation.json").read_text())
    assert [row["video_id"] for row in manifest["videos"]] == config["selection"]["selected_video_ids"]
    assert validation["video_count"] == 5
    assert validation["missing_image_count"] == 0
    assert (output / "hierarchy_review.html").is_file()
