from __future__ import annotations

import json
from pathlib import Path

from src.experiments.long_video_hierarchy_stress_v0_2.experiment import (
    hierarchy_metrics,
    parse_video_index,
    sanitize_sample_id,
    select_primary_candidates,
    validate_cut_nesting,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/experiments/long_video_hierarchy_stress_v0_2.json"
OUT = ROOT / "outputs/experiments/long_video_hierarchy_stress_v0_2"


def test_manifest_selection_is_deterministic_and_uses_distinct_sources() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = parse_video_index(Path(config["video_index_path"]))
    selected = select_primary_candidates(
        rows,
        targets=config["target_duration_sec"],
        preferred_prefixes=config["preferred_sample_prefixes"],
    )
    assert [row["sample_id"] for row in selected] == [
        "youtube/OfficialDCPolice/j42zsIXPci0_1_306",
        "youtube/SAPDMSVC/O79AmjbLMyU_5653_6253",
        "youtube/DallasPoliceDept/RHEOdb_u0QM_225_1456",
        "youtube/DallasPoliceDept/pbYRTN0NSGo_499_2302",
    ]
    assert len({row["source_url"] for row in selected}) == 4


def test_sanitized_sample_id_is_stable_and_path_safe() -> None:
    sample_id = "youtube/DallasPoliceDept/RHEOdb_u0QM_225_1456"
    value = sanitize_sample_id(sample_id)
    assert value == "youtube__DallasPoliceDept__RHEOdb_u0QM_225_1456"
    assert "/" not in value and "\\" not in value


def test_config_freezes_existing_methods_and_prohibits_qa_api_inputs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["hierarchy"]["method"] == "boundary_aware_safe_merge"
    assert config["comet_style"] == {
        "smoothing_sigma_frames": 1.0,
        "adaptive_mad_multiplier": 0.5,
        "minimum_prominence": 0.005,
        "minimum_segment_duration_sec": 4.0,
    }
    prohibited = " ".join(config["prohibited_inputs"]).lower()
    assert all(value in prohibited for value in ("question", "gold", "retrieval", "planner", "vlm"))
    assert config["external_model_api_calls"] == 0
    assert config["canonical_pipeline_modified"] is False


def test_existing_hierarchy_is_not_mutated_by_v02_validators() -> None:
    path = ROOT / "outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl"
    hierarchy = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    before = json.dumps(hierarchy, sort_keys=True)
    result = validate_cut_nesting(hierarchy)
    metrics = hierarchy_metrics(
        hierarchy, thresholds=[4.0, 20.0, 45.0, 90.0, 180.0], giant_ratio=0.5, chaining_ratio=0.85
    )
    assert result["valid"]
    assert metrics["fine_preservation_rate"] == 1.0
    assert "node_count_gt_180s" in metrics["coarse"]
    assert json.dumps(hierarchy, sort_keys=True) == before


def test_completed_outputs_have_four_valid_clips_and_no_missing_images() -> None:
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((OUT / "html_validation.json").read_text(encoding="utf-8"))
    runtimes = json.loads((OUT / "runtime_metrics.json").read_text(encoding="utf-8"))
    assert len(manifest["videos"]) == 4
    assert len({row["source_url"] for row in manifest["videos"]}) == 4
    assert all(row["decode_sanity"]["passed"] for row in manifest["videos"])
    assert validation["displayed_video_count"] == 4
    assert validation["image_reference_count"] > 0
    assert validation["missing_image_count"] == 0
    assert runtimes["external_api_calls"] == 0
    assert runtimes["model_load_count"] == 1
    assert (OUT / "hierarchy_review.html").is_file()


def test_runner_has_no_paid_api_or_question_path() -> None:
    source = (ROOT / "scripts/experiments/run_long_video_hierarchy_stress_v0_2.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "anthropic" not in lowered
    assert "gemini" not in lowered
    assert "openai" not in lowered
    assert "question_text" not in lowered
