from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.thesis_av.visual.medium import build_tree_context, fluid_frontier, validate_frontiers


ROOT = Path(__file__).resolve().parents[1]
FROZEN_REF = "7c5c739"


def _git_show(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{FROZEN_REF}:{path}"], cwd=ROOT
    )


def _json_at(path: str) -> Any:
    return json.loads(_git_show(path).decode("utf-8"))


def _jsonl_at(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _git_show(path).decode("utf-8").splitlines()
        if line.strip()
    ]


def _normalized_source(data: bytes) -> str:
    return data.decode("utf-8").replace("\r\n", "\n").rstrip() + "\n"


def _function_ast(source: str, function_name: str) -> str:
    module = ast.parse(source)
    matches = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(matches) == 1, function_name
    return ast.dump(matches[0], include_attributes=False)


@pytest.mark.parametrize(
    ("frozen_path", "clean_path"),
    [
        ("src/experiments/coarse_segmentation/comet_style.py", "src/thesis_av/visual/comet_style.py"),
        ("src/experiments/coarse_segmentation/schema.py", "src/thesis_av/visual/segmentation_schema.py"),
        ("src/experiments/coarse_segmentation/dinov2_features.py", "src/thesis_av/visual/dinov2_features.py"),
        ("src/experiments/fine_to_coarse_hierarchy/hierarchy.py", "src/thesis_av/visual/hierarchy.py"),
        ("src/experiments/medium_semantic_abstraction/qwen_local.py", "src/thesis_av/visual/qwen_local.py"),
        ("src/experiments/semantic_coarse_v0_1/deterministic_coarse.py", "src/thesis_av/visual/deterministic_coarse.py"),
    ],
)
def test_mechanical_ports_match_frozen_source_exactly(
    frozen_path: str, clean_path: str
) -> None:
    assert _normalized_source((ROOT / clean_path).read_bytes()) == _normalized_source(
        _git_show(frozen_path)
    )


@pytest.mark.parametrize(
    ("frozen_path", "clean_path", "function_names"),
    [
        (
            "src/experiments/long_video_hierarchy_stress/experiment.py",
            "src/thesis_av/visual/sampling.py",
            ["extract_frames_1fps"],
        ),
        (
            "src/experiments/adaptive_fluid_hierarchy_v0_1/experiment.py",
            "src/thesis_av/visual/medium.py",
            [
                "sha256_file",
                "load_jsonl",
                "topology_hash",
                "fine_leaf_hash",
                "build_tree_context",
                "_ordered",
                "fluid_frontier",
                "validate_frontiers",
            ],
        ),
        (
            "src/experiments/efficient_semantic_map_v0_1/experiment.py",
            "src/thesis_av/visual/keyframes.py",
            [
                "sha256_file",
                "node_map",
                "hierarchy_views",
                "_minmax",
                "_quality",
                "_candidate_frames",
                "_nearest_index",
                "_copy_asset",
                "select_video_keyframes",
                "clean_caption",
                "parse_story",
            ],
        ),
        (
            "src/experiments/semantic_coarse_v0_1/experiment.py",
            "src/thesis_av/visual/semantic_helpers.py",
            [
                "sha256_file",
                "sha256_json",
                "load_jsonl",
                "write_json",
                "frozen_fluid_loose_frontiers",
                "posture_state_tokens",
                "validate_semantic_coarse",
            ],
        ),
    ],
)
def test_selected_extracted_functions_match_frozen_ast(
    frozen_path: str, clean_path: str, function_names: list[str]
) -> None:
    frozen = _git_show(frozen_path).decode("utf-8")
    clean = (ROOT / clean_path).read_text(encoding="utf-8")
    for function_name in function_names:
        assert _function_ast(clean, function_name) == _function_ast(frozen, function_name)


def test_production_has_no_experiment_import_or_rejected_medium_variant() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src/thesis_av/visual").glob("*.py"))
    )
    assert "src.experiments" not in production
    assert "from experiments" not in production
    for rejected in (
        "fluid_strict",
        "fluid_balanced",
        "fluid_guarded",
        "local_absolute",
        "photometric_robust",
        "photo_temporal",
        "global_elbow",
        "qwen_boundary_classifier",
    ):
        assert rejected not in production.lower()


def test_selected_config_locks_executed_models_thresholds_and_no_bypass() -> None:
    config = json.loads(
        (ROOT / "config/visual_pipeline_v1.json").read_text(encoding="utf-8")
    )
    long_config = _json_at("config/experiments/long_video_hierarchy_stress_v0_2.json")
    hierarchy_config = _json_at("config/experiments/fine_to_coarse_hierarchy_v0_1.json")
    adaptive_config = _json_at("config/experiments/adaptive_fluid_hierarchy_v0_1.json")
    semantic_config = _json_at("config/experiments/semantic_coarse_v0_1.json")
    assert config["sampling"] == {
        "fps": 1.0,
        "jpeg_quality": 88,
        "ffmpeg_filter": "fps=1",
        "timestamp_rule": "np.arange(round(duration))",
    }
    assert config["dinov2"]["model"] == "facebook/dinov2-small"
    assert config["dinov2"]["batch_size"] == long_config["dinov2"]["batch_size"]
    assert config["comet_style"] == long_config["comet_style"]
    assert config["hierarchy"]["representative_fractions"] == hierarchy_config["representative_frames"]
    assert config["hierarchy"]["include_dinov2_medoid"] == hierarchy_config["include_dinov2_medoid"]
    assert config["medium"] == {
        "method": "fluid_loose",
        **adaptive_config["methods"]["fluid_loose"]["medium"],
    }
    assert config["keyframe_selection"] == semantic_config["keyframe_selection"]
    assert config["prompts"] == {
        key: semantic_config["prompts"][key] for key in ("medium", "coarse", "story")
    }
    for key in (
        "family",
        "model",
        "snapshot_path",
        "local_files_only",
        "dtype",
        "device",
        "do_sample",
        "max_new_tokens_medium",
        "max_new_tokens_coarse",
        "max_new_tokens_story",
    ):
        assert config["local_model"][key] == semantic_config["local_model"][key]
    assert config["semantic_coarse"]["caption_continuity_cosine_min"] == 0.9
    assert config["semantic_coarse"]["strong_visual_transition_veto_cosine"] == 0.85
    assert config["semantic_coarse"]["caption_centroid_cosine_min"] == 0.9
    assert config["semantic_coarse"]["visual_centroid_cosine_min"] == 0.85
    assert config["semantic_coarse"]["photometric_bypass_executed"] is False
    assert config["semantic_coarse"]["qwen_boundary_calls"] == 0


def test_real_frozen_trees_replay_exact_fluid_loose_ids_and_lineage() -> None:
    trees = _jsonl_at(
        "outputs/experiments/long_video_hierarchy_stress_v0_2/"
        "safe_merge_hierarchies.jsonl"
    )
    metrics = _json_at(
        "outputs/experiments/adaptive_fluid_hierarchy_v0_1/"
        "per_video_method_metrics.json"
    )
    expected = {
        row["video_id"]: row
        for row in metrics
        if row.get("source_group") == "long_primary"
        and row.get("method") == "fluid_loose"
    }
    observed_counts = []
    for tree in trees:
        context = build_tree_context(tree)
        medium_ids, _ = fluid_frontier(
            context,
            minimum_q_rank=0.40,
            maximum_local_drop=0.30,
            level="medium",
        )
        assert medium_ids == expected[tree["video_id"]]["medium_ids"]
        observed_counts.append(len(medium_ids))
        validation = validate_frontiers(
            hierarchy=tree,
            medium_ids=medium_ids,
            coarse_ids=[tree["root_id"]],
        )
        assert validation["medium"]["fine_leaf_preservation_rate"] == 1.0
        assert validation["valid"] is True
        nodes = context["nodes"]
        for medium_id in medium_ids:
            node = nodes[medium_id]
            assert node["leaf_ids"]
            assert node["start"] < node["end"]
    assert observed_counts == [8, 12, 29, 55]


def test_real_cached_replay_matches_every_frozen_selected_stage() -> None:
    output = ROOT / "outputs/visual_pipeline_v1_fidelity_test"
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_visual_pipeline_v1_fidelity.py"),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    assert completed.returncode == 0
    report = json.loads(
        (output / "fidelity_report.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert [row["medium_count"] for row in report["structure"]] == [8, 12, 29, 55]
    assert [row["coarse_count"] for row in report["semantics"]] == [8, 12, 27, 18]
    assert sum(row["keyframe_total"] for row in report["structure"]) == 104
    assert all(not row["keyframe_mismatches"] for row in report["structure"])
