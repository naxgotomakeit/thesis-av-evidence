from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.experiments.adaptive_fluid_hierarchy_v0_1.experiment import (
    build_tree_context,
    fine_leaf_hash,
    fluid_frontier,
    load_jsonl,
    topology_hash,
    validate_frontiers,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/experiments/adaptive_fluid_hierarchy_v0_1.json"
SOURCE = ROOT / "outputs/experiments/long_video_hierarchy_stress_v0_2/safe_merge_hierarchies.jsonl"
OUT = ROOT / "outputs/experiments/adaptive_fluid_hierarchy_v0_1"


def synthetic_context(kind: str) -> dict:
    nodes = {
        "F1": {"node_id": "F1", "child_ids": [], "start": 0.0, "end": 1.0, "duration": 1.0},
        "F2": {"node_id": "F2", "child_ids": [], "start": 1.0, "end": 2.0, "duration": 1.0},
        "F3": {"node_id": "F3", "child_ids": [], "start": 2.0, "end": 3.0, "duration": 1.0},
        "F4": {"node_id": "F4", "child_ids": [], "start": 3.0, "end": 4.0, "duration": 1.0},
        "L": {"node_id": "L", "child_ids": ["F1", "F2"], "start": 0.0, "end": 2.0, "duration": 2.0},
        "R": {"node_id": "R", "child_ids": ["F3", "F4"], "start": 2.0, "end": 4.0, "duration": 2.0},
        "ROOT": {"node_id": "ROOT", "child_ids": ["L", "R"], "start": 0.0, "end": 4.0, "duration": 4.0},
    }
    rank = {"ROOT": 0.9, "L": 0.9, "R": 0.9}
    drop = {"ROOT": 0.05, "L": 0.05, "R": 0.05}
    if kind == "complex":
        rank["ROOT"], drop["ROOT"] = 0.2, 0.5
    elif kind == "mixed":
        rank["ROOT"], drop["ROOT"] = 0.2, 0.5
        rank["R"], drop["R"] = 0.2, 0.5
    return {
        "nodes": nodes,
        "merge": {item: {"merge_score": 1.0 - rank[item]} for item in ("ROOT", "L", "R")},
        "quality": dict(rank), "q_rank": rank, "local_drop": drop,
        "child_quality_available": {"ROOT": True, "L": False, "R": False},
        "root_id": "ROOT", "fine_ids": ["F1", "F2", "F3", "F4"], "video_duration": 4.0,
    }


def test_uniform_repetitive_branch_collapses_deeply() -> None:
    selected, trace = fluid_frontier(
        synthetic_context("uniform"), minimum_q_rank=0.55, maximum_local_drop=0.2
    )
    assert selected == ["ROOT"]
    assert trace[0]["decision"] == "COLLAPSE"


def test_strong_boundary_complex_branch_stops_earlier() -> None:
    selected, trace = fluid_frontier(
        synthetic_context("complex"), minimum_q_rank=0.55, maximum_local_drop=0.2
    )
    assert selected == ["L", "R"]
    assert trace[0]["decision"] == "RECURSE"


def test_mixed_tree_stops_at_different_depths() -> None:
    selected, _ = fluid_frontier(
        synthetic_context("mixed"), minimum_q_rank=0.55, maximum_local_drop=0.2
    )
    assert selected == ["L", "F3", "F4"]


def test_frozen_long_tree_identity_and_determinism() -> None:
    hierarchy = load_jsonl(SOURCE)[0]
    before = json.dumps(hierarchy, sort_keys=True)
    context = build_tree_context(hierarchy)
    first, _ = fluid_frontier(context, minimum_q_rank=0.55, maximum_local_drop=0.2)
    second, _ = fluid_frontier(context, minimum_q_rank=0.55, maximum_local_drop=0.2)
    assert first == second
    assert len(fine_leaf_hash(hierarchy)) == len(topology_hash(hierarchy)) == 64
    assert json.dumps(hierarchy, sort_keys=True) == before


def test_config_freezes_requested_presets_and_prohibits_vlm_api_download() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["methods"]["fluid_strict"]["medium"] == {
        "minimum_q_rank": 0.7, "maximum_local_drop": 0.1
    }
    assert config["methods"]["fluid_balanced"]["medium"] == {
        "minimum_q_rank": 0.55, "maximum_local_drop": 0.2
    }
    assert config["methods"]["fluid_loose"]["medium"] == {
        "minimum_q_rank": 0.4, "maximum_local_drop": 0.3
    }
    assert config["rerun_vlm"] is False
    assert config["external_api_calls"] == config["downloads"] == 0
    assert config["tree_topology_modified"] is False
    assert config["fine_segmentation_modified"] is False


def test_completed_methods_preserve_fine_coverage_lineage_and_assets() -> None:
    rows = json.loads((OUT / "per_video_method_metrics.json").read_text(encoding="utf-8"))
    assert len(rows) == 14 * 7
    for row in rows:
        assert row["validity"]["valid"]
        assert row["validity"]["medium"]["fine_leaf_preservation_rate"] == 1.0
        assert row["validity"]["medium"]["gap_count"] == 0
        assert row["validity"]["medium"]["overlap_count"] == 0
        assert row["validity"]["lineage_violation_count"] == 0
        assert len(row["medium_to_coarse_membership"]) == row["medium"]["node_count"]
        for asset in row["representative_assets"].values():
            assert (OUT / asset["html_relative_path"]).is_file()


def test_completed_run_keeps_frozen_hashes_and_has_zero_calls() -> None:
    frozen = json.loads((OUT / "frozen_config.json").read_text(encoding="utf-8"))
    runtime = json.loads((OUT / "runtime_metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((OUT / "html_validation.json").read_text(encoding="utf-8"))
    assert frozen["source_files_unchanged"] is True
    assert frozen["external_api_calls"] == frozen["downloads"] == frozen["vlm_calls"] == 0
    assert runtime["external_api_calls"] == runtime["downloads"] == runtime["vlm_calls"] == 0
    assert validation["displayed_video_count"] == 14
    assert validation["primary_long_video_count"] == 4
    assert validation["displayed_method_video_sections"] == 98
    assert validation["missing_image_reference_count"] == 0


def test_no_canonical_tracked_diff() -> None:
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    assert not any(path.startswith(("src/pipeline/", "config/canonical_pipeline", "tasks/")) for path in changed)
