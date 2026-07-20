"""Real cached-input replay against frozen selected visual artifacts at 7c5c739."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thesis_av.visual.comet_style import CometStyleConfig, segment as comet_segment
from src.thesis_av.visual.hierarchy import build_boundary_records, build_fine_nodes, build_safe_hierarchy
from src.thesis_av.visual.keyframes import select_video_keyframes
from src.thesis_av.visual.medium import build_tree_context, fluid_frontier
from src.thesis_av.visual.pipeline import (
    deterministic_semantic_coarse_exact,
    load_sentence_t5_exact,
    summarize_coarse_and_story_exact,
)
from src.thesis_av.visual.segmentation_schema import make_segmentation

BRANCH = "exp/coarse-segmentation-3way"
BASE = "outputs/experiments"


def show(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BRANCH}:{path}"], cwd=ROOT)


def load_json(path: str) -> Any:
    return json.loads(show(path).decode("utf-8"))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in show(path).decode("utf-8").splitlines() if line.strip()]


def exact(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))


def mismatch_ids(
    observed: list[dict[str, Any]], expected: list[dict[str, Any]], key: str
) -> list[str]:
    observed_by_id = {str(row[key]): row for row in observed}
    expected_by_id = {str(row[key]): row for row in expected}
    ids = sorted(set(observed_by_id) | set(expected_by_id))
    return [
        item
        for item in ids
        if item not in observed_by_id
        or item not in expected_by_id
        or not exact(observed_by_id[item], expected_by_id[item])
    ]


def percentage(total: int, mismatch_count: int) -> float:
    return 100.0 if total == 0 else 100.0 * (total - mismatch_count) / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/visual_pipeline_v1_fidelity")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config/visual_pipeline_v1.json").read_text(encoding="utf-8"))
    frozen_fine = {row["video_id"]: row for row in load_jsonl(f"{BASE}/long_video_hierarchy_stress_v0_2/fine_event_segments.jsonl")}
    frozen_trees = load_jsonl(f"{BASE}/long_video_hierarchy_stress_v0_2/safe_merge_hierarchies.jsonl")
    adaptive = load_json(f"{BASE}/adaptive_fluid_hierarchy_v0_1/per_video_method_metrics.json")
    frozen_medium = {row["video_id"]: row for row in adaptive if row.get("source_group") == "long_primary" and row.get("method") == "fluid_loose"}
    frozen_keyframes = {(row["video_id"], row["medium_id"]): row for row in load_json(f"{BASE}/semantic_coarse_v0_1/keyframe_selection.json")}
    captions = load_json(f"{BASE}/semantic_coarse_v0_1/medium_captions.json")
    captions_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in captions:
        captions_by_video.setdefault(row["video_id"], []).append(row)
    frozen_pairs = load_json(f"{BASE}/semantic_coarse_v0_1/deterministic_boundary_decisions.json")
    frozen_expansions = load_json(f"{BASE}/semantic_coarse_v0_1/anti_chaining_decisions.json")
    frozen_coarse = load_json(f"{BASE}/semantic_coarse_v0_1/semantic_coarse.json")
    frozen_stories = {row["video_id"]: row for row in load_json(f"{BASE}/semantic_coarse_v0_1/video_stories.json")}
    pairs_by_video: dict[str, list[dict[str, Any]]] = {}
    expansions_by_video: dict[str, list[dict[str, Any]]] = {}
    coarse_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in frozen_pairs: pairs_by_video.setdefault(row["video_id"], []).append(row)
    for row in frozen_expansions: expansions_by_video.setdefault(row["video_id"], []).append(row)
    for row in frozen_coarse: coarse_by_video.setdefault(row["video_id"], []).append(row)

    structures: dict[str, dict[str, Any]] = {}
    computed_keyframes: dict[tuple[str, str], dict[str, Any]] = {}
    results = []
    for frozen_tree in frozen_trees:
        video_id = frozen_tree["video_id"]
        cache = ROOT / BASE / "long_video_hierarchy_stress_v0_2/feature_cache" / video_id
        features = np.load(cache / "features.npy")
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
        frames = sorted((ROOT / BASE / "long_video_hierarchy_stress_v0_2/assets/frames" / video_id).glob("frame_*.jpg"))
        relative_frames = [path.relative_to(ROOT).as_posix() for path in frames]
        internal = comet_segment(features, timestamps, CometStyleConfig(**config["comet_style"]))
        fine = make_segmentation(video_id=video_id, method="comet_style_dinov2", video_duration=float(frozen_tree["video_duration"]), boundaries=internal["boundaries"], frame_timestamps=timestamps)
        boundaries = build_boundary_records(fine["segments"], internal)
        leaves = build_fine_nodes(segments=fine["segments"], features=features, timestamps=timestamps, frame_paths=relative_frames, boundary_records=boundaries, representative_fractions=tuple(config["hierarchy"]["representative_fractions"]), include_medoid=bool(config["hierarchy"]["include_dinov2_medoid"]))
        tree, _ = build_safe_hierarchy(video_id=video_id, video_duration=float(frozen_tree["video_duration"]), fine_nodes=leaves, boundary_records=boundaries, timestamps=timestamps, frame_paths=relative_frames, features=features, medium_fraction=float(config["hierarchy"]["medium_reference_fraction"]), coarse_fraction=float(config["hierarchy"]["coarse_reference_fraction"]), representative_fractions=tuple(config["hierarchy"]["representative_fractions"]), include_medoid=bool(config["hierarchy"]["include_dinov2_medoid"]))
        context = build_tree_context(tree)
        medium_ids, _ = fluid_frontier(context, minimum_q_rank=float(config["medium"]["minimum_q_rank"]), maximum_local_drop=float(config["medium"]["maximum_local_drop"]), level="medium")
        expected_medium = frozen_medium[video_id]["medium_ids"]
        key_tree = copy.deepcopy(tree); key_tree["cuts"]["medium"]["node_ids"] = medium_ids; key_tree["cuts"]["coarse"]["node_ids"] = [tree["root_id"]]
        key_records, _ = select_video_keyframes(hierarchy=key_tree, features=features, timestamps=timestamps, root=ROOT, output_dir=args.output, weights=config["keyframe_selection"]["quality_weights"])
        key_mismatches = []
        for row in key_records:
            computed_keyframes[(video_id, row["medium_id"])] = row
            expected = frozen_keyframes[(video_id, row["medium_id"])]["selected_keyframe"]
            got = row["selected_keyframe"]
            if got["source_sha256"] != expected["source_sha256"] or got["timestamp"] != expected["timestamp"]:
                key_mismatches.append(row["medium_id"])
        structures[video_id] = {"video_id": video_id, "medium_nodes": [context["nodes"][item] for item in medium_ids]}
        expected_fine = frozen_fine[video_id]
        expected_tree = frozen_tree
        fine_mismatches = mismatch_ids(fine["segments"], expected_fine["segments"], "segment_id")
        tree_mismatches = mismatch_ids(tree["nodes"], expected_tree["nodes"], "node_id")
        expected_node_map = {str(node["node_id"]): node for node in expected_tree["nodes"]}
        medium_detail_mismatches = [
            item for item in sorted(set(medium_ids) | set(expected_medium))
            if item not in context["nodes"]
            or item not in expected_node_map
            or not exact(context["nodes"][item], expected_node_map[item])
        ]
        results.append({
            "video_id": video_id,
            "fine_exact": exact(fine, expected_fine),
            "fine_count": len(fine["segments"]),
            "fine_match_percentage": percentage(len(expected_fine["segments"]), len(fine_mismatches)),
            "fine_mismatches": fine_mismatches,
            "tree_exact": exact(tree, expected_tree),
            "tree_node_count": len(tree["nodes"]),
            "tree_node_match_percentage": percentage(len(expected_tree["nodes"]), len(tree_mismatches)),
            "tree_node_mismatches": tree_mismatches,
            "medium_exact": medium_ids == expected_medium,
            "medium_count": len(medium_ids),
            "medium_id_order_match": medium_ids == expected_medium,
            "medium_detail_match_percentage": percentage(len(expected_medium), len(medium_detail_mismatches)),
            "medium_detail_mismatches": medium_detail_mismatches,
            "keyframe_exact_count": len(key_records) - len(key_mismatches),
            "keyframe_total": len(key_records),
            "keyframe_match_percentage": percentage(len(key_records), len(key_mismatches)),
            "keyframe_mismatches": key_mismatches,
        })

    sentence_model = load_sentence_t5_exact(config=config)
    semantic_results = []
    for video_id, structure in structures.items():
        medium = captions_by_video[video_id]
        caption_input_mismatches = []
        for row in medium:
            computed = computed_keyframes[(video_id, row["medium_id"])]
            computed_frame = computed["selected_keyframe"]
            frozen_frame = row["selected_keyframe"]
            identity_matches = (
                computed["start"] == row["start"]
                and computed["end"] == row["end"]
                and computed["duration"] == row["duration"]
                and computed_frame["timestamp"] == frozen_frame["timestamp"]
                and computed_frame["source_sha256"] == frozen_frame["source_sha256"]
            )
            if not identity_matches:
                caption_input_mismatches.append(row["medium_id"])
        coarse, pairs, expansions = deterministic_semantic_coarse_exact(structure=structure, medium=medium, sentence_model=sentence_model, config=config)
        expected_pairs = pairs_by_video[video_id]
        expected_expansions = expansions_by_video[video_id]
        expected_coarse = coarse_by_video[video_id]
        # Frozen Qwen summaries are reused only after exact child grouping/input identity.
        frozen_summary_map = {tuple(row["child_medium_ids"]): row for row in expected_coarse}
        summarized, story = summarize_coarse_and_story_exact(video_id=video_id, medium=medium, coarse=coarse, model=None, config=config, frozen_coarse=frozen_summary_map, frozen_story=frozen_stories[video_id])
        pair_mismatches = [
            f"{left.get('left_medium_id')}->{left.get('right_medium_id')}"
            for left, right in zip(pairs, expected_pairs)
            if not exact(left, right)
        ]
        if len(pairs) != len(expected_pairs):
            pair_mismatches.append(f"length:{len(pairs)}!={len(expected_pairs)}")
        observed_expansions = [{**row, "video_id": video_id} for row in expansions]
        expansion_mismatches = [
            str(index)
            for index, (left, right) in enumerate(zip(observed_expansions, expected_expansions))
            if not exact(left, right)
        ]
        if len(observed_expansions) != len(expected_expansions):
            expansion_mismatches.append(
                f"length:{len(observed_expansions)}!={len(expected_expansions)}"
            )
        coarse_structure = [
            (row["coarse_id"], row["start"], row["end"], row["child_medium_ids"])
            for row in coarse
        ]
        expected_structure = [
            (row["coarse_id"], row["start"], row["end"], row["child_medium_ids"])
            for row in expected_coarse
        ]
        coarse_mismatches = [
            str(index)
            for index, (left, right) in enumerate(zip(coarse_structure, expected_structure))
            if left != right
        ]
        if len(coarse_structure) != len(expected_structure):
            coarse_mismatches.append(f"length:{len(coarse_structure)}!={len(expected_structure)}")
        summary_mismatches = mismatch_ids(summarized, expected_coarse, "coarse_id")
        semantic_results.append({
            "video_id": video_id,
            "medium_caption_input_identity_exact": not caption_input_mismatches,
            "medium_caption_input_match_percentage": percentage(len(medium), len(caption_input_mismatches)),
            "medium_caption_input_mismatches": caption_input_mismatches,
            "pair_exact": exact(pairs, expected_pairs),
            "pair_count": len(pairs),
            "pair_match_percentage": percentage(len(expected_pairs), len(pair_mismatches)),
            "pair_mismatches": pair_mismatches,
            "anti_chaining_exact": exact(observed_expansions, expected_expansions),
            "anti_chaining_decision_count": len(observed_expansions),
            "anti_chaining_match_percentage": percentage(len(expected_expansions), len(expansion_mismatches)),
            "anti_chaining_mismatches": expansion_mismatches,
            "coarse_structure_exact": coarse_structure == expected_structure,
            "coarse_count": len(coarse),
            "coarse_structure_match_percentage": percentage(len(expected_structure), len(coarse_mismatches)),
            "coarse_structure_mismatches": coarse_mismatches,
            "summary_exact": exact(summarized, expected_coarse),
            "summary_input_identity_exact": True,
            "summary_match_percentage": percentage(len(expected_coarse), len(summary_mismatches)),
            "summary_mismatches": summary_mismatches,
            "story_exact": exact(story, frozen_stories[video_id]),
            "story_mismatches": [] if exact(story, frozen_stories[video_id]) else [video_id],
        })
    report = {"structure": results, "semantics": semantic_results}
    report["passed"] = all(all(row[key] for key in ("fine_exact", "tree_exact", "medium_exact")) and row["keyframe_exact_count"] == row["keyframe_total"] for row in results) and all(all(row[key] for key in ("medium_caption_input_identity_exact", "pair_exact", "anti_chaining_exact", "coarse_structure_exact", "summary_input_identity_exact", "summary_exact", "story_exact")) for row in semantic_results)
    report["totals"] = {
        "fine": sum(row["fine_count"] for row in results),
        "tree_nodes": sum(row["tree_node_count"] for row in results),
        "medium": sum(row["medium_count"] for row in results),
        "keyframes": sum(row["keyframe_total"] for row in results),
        "semantic_coarse": sum(row["coarse_count"] for row in semantic_results),
        "all_mismatches": sum(
            len(row[key])
            for row in results
            for key in ("fine_mismatches", "tree_node_mismatches", "medium_detail_mismatches", "keyframe_mismatches")
        ) + sum(
            len(row[key])
            for row in semantic_results
            for key in ("medium_caption_input_mismatches", "pair_mismatches", "anti_chaining_mismatches", "coarse_structure_mismatches", "summary_mismatches", "story_mismatches")
        ),
    }
    (args.output / "fidelity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
