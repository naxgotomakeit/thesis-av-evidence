"""Build semantic Coarse and storylines from frozen Fluid Loose Medium captions."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.medium_semantic_abstraction.qwen_local import LocalQwen2VL, create_read_only_model_view
from src.experiments.semantic_coarse_v0_1.deterministic_coarse import (
    build_groups_with_anti_chaining, normalized, pair_decision,
)
from src.experiments.semantic_coarse_v0_1.experiment import (
    frozen_fluid_loose_frontiers, posture_state_tokens, sha256_file, sha256_json,
    validate_semantic_coarse, write_json,
)
from src.experiments.semantic_coarse_v0_1.reporting import render_html


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def main() -> int:
    started = time.perf_counter()
    config_path = ROOT / "config/experiments/semantic_coarse_v0_1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / "outputs/experiments/semantic_coarse_v0_1"
    captions_path = output / "medium_captions.json"
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    caption_map = {(row["video_id"], row["medium_id"]): row for row in captions}
    if len(captions) != 104 or len(caption_map) != 104:
        raise RuntimeError("Deterministic Semantic Coarse requires complete unique 104/104 Medium captions")
    source = config["source_policy"]
    frontiers, provenance = frozen_fluid_loose_frontiers(
        metrics_path=ROOT / source["adaptive_metrics"],
        hierarchy_path=ROOT / source["safe_merge_hierarchies"], expected_video_count=4,
    )
    frozen_identity = {
        "medium_captions_sha256": sha256_file(captions_path), "source": provenance,
        "frontiers": [{"video_id": row["video_id"], "frontier_sha256": row["frontier_sha256"]} for row in frontiers],
        "semantic_rule": config["semantic_coarse"], "coarse_prompt": config["prompts"]["coarse"],
        "story_prompt": config["prompts"]["story"], "model": config["local_model"],
    }
    fingerprint = sha256_json(frozen_identity)
    checkpoint_path = output / "deterministic_semantic_summary_checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing deterministic summary checkpoint fingerprint is incompatible")
    else:
        checkpoint = {"fingerprint": fingerprint, "coarse": {}, "stories": {}}

    sentence_load_started = time.perf_counter()
    sentence_model = SentenceTransformer(
        config["semantic_coarse"]["caption_semantic_encoder"], local_files_only=True, device="cuda"
    )
    sentence_load_sec = time.perf_counter() - sentence_load_started
    boundary_rows = []
    group_expansion_rows = []
    video_structures = []
    boundary_runtime = 0.0
    grouping_runtime = 0.0
    posture_vocabulary = list(config["semantic_coarse"]["posture_state_conflict_veto"]["tokens"])
    for frontier in frontiers:
        boundary_started = time.perf_counter()
        node_map = {str(node["node_id"]): node for node in frontier["medium_nodes"]}
        medium = [caption_map[(frontier["video_id"], medium_id)] for medium_id in frontier["medium_ids"]]
        caption_embeddings = sentence_model.encode(
            [row["caption"] for row in medium], normalize_embeddings=True, show_progress_bar=False
        )
        records = []
        for row, caption_embedding in zip(medium, caption_embeddings):
            visual_embedding = normalized(np.asarray(node_map[row["medium_id"]]["pooled_dinov2"], dtype=np.float32))
            records.append({
                **row, "caption_embedding": np.asarray(caption_embedding, dtype=np.float32),
                "visual_embedding": visual_embedding,
                "posture_tokens": posture_state_tokens(row["caption"], posture_vocabulary),
            })
        pairs = []
        for left, right in zip(records, records[1:]):
            caption_similarity = float(left["caption_embedding"] @ right["caption_embedding"])
            visual_similarity = float(left["visual_embedding"] @ right["visual_embedding"])
            decision, failures = pair_decision(
                caption_similarity=caption_similarity, visual_similarity=visual_similarity,
                left_postures=left["posture_tokens"], right_postures=right["posture_tokens"],
                caption_min=float(config["semantic_coarse"]["caption_continuity_cosine_min"]),
                visual_min=float(config["semantic_coarse"]["strong_visual_transition_veto_cosine"]),
            )
            pair = {
                "video_id": frontier["video_id"], "boundary_time": float(left["end"]),
                "left_medium_id": left["medium_id"], "right_medium_id": right["medium_id"],
                "left_caption": left["caption"], "right_caption": right["caption"],
                "caption_sentence_t5_similarity": caption_similarity,
                "dino_visual_similarity": visual_similarity,
                "left_posture_state_tokens": left["posture_tokens"],
                "right_posture_state_tokens": right["posture_tokens"],
                "pair_decision": decision, "failed_gates": failures,
                "qwen_boundary_calls": 0,
            }
            pairs.append(pair)
            boundary_rows.append(pair)
        boundary_runtime += time.perf_counter() - boundary_started
        grouping_started = time.perf_counter()
        groups, expansions = build_groups_with_anti_chaining(
            records=records, pair_rows=pairs,
            caption_centroid_min=float(config["semantic_coarse"]["anti_chaining"]["caption_centroid_cosine_min"]),
            visual_centroid_min=float(config["semantic_coarse"]["anti_chaining"]["visual_centroid_cosine_min"]),
        )
        for row in expansions:
            row["video_id"] = frontier["video_id"]
        group_expansion_rows.extend(expansions)
        coarse = []
        for index, group in enumerate(groups, start=1):
            children = [records[position] for position in group]
            coarse.append({
                "video_id": frontier["video_id"], "coarse_id": f"semantic_coarse_{index:04d}",
                "start": float(children[0]["start"]), "end": float(children[-1]["end"]),
                "duration": float(children[-1]["end"] - children[0]["start"]),
                "child_medium_ids": [row["medium_id"] for row in children],
                "child_count": len(children), "child_captions": [row["caption"] for row in children],
            })
        grouping_runtime += time.perf_counter() - grouping_started
        video_structures.append({"frontier": frontier, "medium": medium, "coarse": coarse, "expansions": expansions})

    # Boundary decisions are now frozen before any summary generation.
    write_json(output / "deterministic_boundary_decisions.json", boundary_rows)
    write_json(output / "anti_chaining_decisions.json", group_expansion_rows)
    mapping = [
        {"video_id": item["frontier"]["video_id"], "medium_id": medium_id, "coarse_id": coarse["coarse_id"]}
        for item in video_structures for coarse in item["coarse"] for medium_id in coarse["child_medium_ids"]
    ]
    write_json(output / "medium_to_coarse_mapping.json", mapping)

    model_view, model_files = create_read_only_model_view(
        Path(config["local_model"]["snapshot_path"]), output / "local_model_view"
    )
    qwen = LocalQwen2VL(model_view=model_view, seed=int(config["seed"]))
    coarse_results = []
    story_results = []
    video_results = []
    validation = []
    for item in video_structures:
        video_started = time.perf_counter()
        frontier, medium, coarse = item["frontier"], item["medium"], item["coarse"]
        summarized = []
        for group in coarse:
            key = f"{frontier['video_id']}|{'|'.join(group['child_medium_ids'])}"
            saved = checkpoint["coarse"].get(key)
            if saved is None:
                if group["child_count"] == 1:
                    saved = {
                        **group, "summary": group["child_captions"][0], "raw_output": group["child_captions"][0],
                        "summary_source": "singleton_copy_through",
                        "timing": {"image_count": 0, "image_decode_sec": 0.0, "image_preprocess_sec": 0.0,
                                   "model_inference_sec": 0.0, "total_semantic_generation_sec": 0.0,
                                   "input_tokens": 0, "output_tokens": 0, "peak_vram_bytes": 0},
                    }
                else:
                    descriptions = [
                        f"[{row['start']:.1f}-{row['end']:.1f}s] {row['caption']}"
                        for row in medium if row["medium_id"] in group["child_medium_ids"]
                    ]
                    raw, timing = qwen.summarize_text(
                        ordered_descriptions=descriptions, prompt=config["prompts"]["coarse"],
                        max_new_tokens=int(config["local_model"]["max_new_tokens_coarse"]),
                    )
                    saved = {**group, "summary": " ".join(raw.strip().split()), "raw_output": raw,
                             "summary_source": "local_qwen_text_only", "timing": timing}
                checkpoint["coarse"][key] = saved
                write_json(checkpoint_path, checkpoint)
            summarized.append(saved)
            coarse_results.append(saved)
        story_key = frontier["video_id"]
        story = checkpoint["stories"].get(story_key)
        if story is None:
            descriptions = [f"[{row['start']:.1f}-{row['end']:.1f}s] {row['summary']}" for row in summarized]
            raw, timing = qwen.summarize_text(
                ordered_descriptions=descriptions, prompt=config["prompts"]["story"],
                max_new_tokens=int(config["local_model"]["max_new_tokens_story"]),
            )
            story = {"video_id": frontier["video_id"], "coarse_inputs": descriptions,
                     "story": raw.strip(), "raw_output": raw, "timing": timing}
            checkpoint["stories"][story_key] = story
            write_json(checkpoint_path, checkpoint)
        story_results.append(story)
        check = validate_semantic_coarse(
            frontier=frontier, medium_records=medium, coarse_records=summarized
        )
        validation.append({"video_id": frontier["video_id"], **check})
        calls = {
            "medium_vlm": 0, "boundary_decision": 0, "boundary_sentence_t5_pair_encodes": max(0, len(medium) - 1),
            "coarse_summary": sum(row["summary_source"] == "local_qwen_text_only" for row in summarized),
            "singleton_copy_through": sum(row["summary_source"] == "singleton_copy_through" for row in summarized),
            "story": 1,
        }
        runtime = {
            "boundary_and_embedding_sec": 0.0,
            "coarse_summary_total_sec": sum(row["timing"]["total_semantic_generation_sec"] for row in summarized),
            "story_total_sec": story["timing"]["total_semantic_generation_sec"],
            "steady_state_total_sec": time.perf_counter() - video_started,
        }
        video_results.append({
            "video_id": frontier["video_id"], "video_duration": frontier["video_duration"],
            "fine_count": frontier["fine_count"], "medium_count": len(medium), "coarse_count": len(summarized),
            "medium": medium, "coarse": summarized, "story": story, "calls": calls,
            "runtime": runtime, "validation": check,
        })

    write_json(output / "semantic_coarse.json", coarse_results)
    write_json(output / "coarse_captions.json", coarse_results)
    write_json(output / "video_stories.json", story_results)
    write_json(output / "validation.json", validation)
    calls = {
        "medium_vlm": 0, "medium_vlm_this_stage": 0,
        "boundary_decision": 0, "qwen_boundary": 0,
        "sentence_t5_boundary_pair_encodes": len(boundary_rows),
        "coarse_summary": sum(row["summary_source"] == "local_qwen_text_only" for row in coarse_results),
        "singleton_copy_through": sum(row["summary_source"] == "singleton_copy_through" for row in coarse_results),
        "story": len(story_results), "external_api": 0,
    }
    aggregate = {
        "video_count": len(video_results), "medium_count": len(captions), "coarse_count": len(coarse_results),
        "per_video_counts": [{"video_id": row["video_id"], "medium": row["medium_count"], "coarse": row["coarse_count"]} for row in video_results],
        "calls": calls, "anti_chaining_prevented_merges": sum(row["anti_chaining_prevented_merge"] for row in group_expansion_rows),
        "qwen_model_load_sec": qwen.model_load_sec, "sentence_t5_model_load_sec": sentence_load_sec,
        "total_wall_sec": time.perf_counter() - started,
    }
    write_json(output / "aggregate_metrics.json", aggregate)
    runtime = {
        "sentence_t5_model_load_sec": sentence_load_sec, "boundary_and_caption_encoding_sec": boundary_runtime,
        "adaptive_grouping_sec": grouping_runtime, "qwen_model_load_sec": qwen.model_load_sec,
        "coarse_summary_sec": sum(row["timing"]["total_semantic_generation_sec"] for row in coarse_results),
        "storyline_sec": sum(row["timing"]["total_semantic_generation_sec"] for row in story_results),
        "total_wall_sec": aggregate["total_wall_sec"],
    }
    write_json(output / "runtime_metrics.json", runtime)
    manifest = {
        "schema_version": "semantic-coarse-deterministic-v1", "fingerprint": fingerprint,
        "medium_captions_sha256": sha256_file(captions_path), "source": provenance,
        "source_medium_method": "fluid_loose", "photometric_or_absolute_medium_variants_used": False,
        "fine_tree_or_medium_boundaries_modified": False, "qwen_boundary_calls": 0,
        "external_api_calls": 0, "model_files": model_files,
        "git_branch": git("branch", "--show-current"), "git_head": git("rev-parse", "HEAD"),
        "canonical_tracked_diff": bool(git("status", "--short", "--untracked-files=no")),
    }
    write_json(output / "run_manifest.json", manifest)
    html_validation = render_html(
        videos=video_results, aggregate=aggregate, output_path=output / "semantic_coarse_review.html"
    )
    write_json(output / "html_validation.json", html_validation)
    readme = (
        "# semantic_coarse_v0_1\n\n"
        "Frozen original Fluid Loose Medium captions are grouped by the deterministic Sentence-T5 + DINO + state-conflict rule. "
        "Group-centroid caption and visual checks prevent chaining. Qwen is used only for multi-Medium Coarse summaries and text-only storylines.\n\n"
        f"Medium: {len(captions)}; Coarse: {len(coarse_results)}; Qwen boundary calls: 0; external API calls: 0.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "runtime": runtime, "html": html_validation, "validation": validation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
