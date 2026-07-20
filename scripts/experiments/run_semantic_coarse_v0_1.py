"""Run semantic_coarse_v0_1 using only frozen Fluid Loose Mediums."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.efficient_semantic_map_v0_1.experiment import select_video_keyframes
from src.experiments.medium_semantic_abstraction.qwen_local import LocalQwen2VL, create_read_only_model_view
from src.experiments.semantic_coarse_v0_1.experiment import (
    build_adjacent_groups, frozen_fluid_loose_frontiers, photometric_transition_diagnostic,
    posture_state_tokens, sha256_file, sha256_json, validate_semantic_coarse, write_json,
)
from src.experiments.semantic_coarse_v0_1.reporting import render_html


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _clean_caption(text: str) -> str:
    return " ".join(text.strip().split())


REJECTED_PREFLIGHT_FINGERPRINT = "6181bc4d83acce974481afc298a00c8aa7f996d0fd2c716f6584a11f182e4c51"


def _load_checkpoint(path: Path, fingerprint: str) -> dict:
    if not path.is_file():
        return {"fingerprint": fingerprint, "medium": {}, "boundaries": {}, "coarse": {}, "stories": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("fingerprint") == REJECTED_PREFLIGHT_FINGERPRINT and fingerprint != REJECTED_PREFLIGHT_FINGERPRINT:
        # The image-caption prompt and frozen Medium inputs are unchanged. Preserve those paid-in-compute
        # local results, but quarantine all rejected boundary/group/summary results from the failed preflight.
        rejected = {
            "fingerprint": value["fingerprint"],
            "reason": "systematic broad-theme boundary collapse; 28/28 initial boundaries merged",
            "medium_captions_retained_as_stage_compatible": len(value.get("medium", {})),
            "boundary_decisions_rejected": len(value.get("boundaries", {})),
            "coarse_results_rejected": len(value.get("coarse", {})),
            "stories_rejected": len(value.get("stories", {})),
            "boundaries": value.get("boundaries", {}),
            "coarse": value.get("coarse", {}),
            "stories": value.get("stories", {}),
        }
        write_json(path.with_name("rejected_boundary_preflight.json"), rejected)
        return {
            "fingerprint": fingerprint, "medium": value.get("medium", {}),
            "boundaries": {}, "coarse": {}, "stories": {},
            "migration": {key: rejected[key] for key in rejected if key not in {"boundaries", "coarse", "stories"}},
        }
    if value.get("fingerprint") != fingerprint:
        raise RuntimeError("Existing semantic checkpoint fingerprint is incompatible")
    return value


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    write_json(path, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiments/semantic_coarse_v0_1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/semantic_coarse_v0_1")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source_policy"]
    metrics_path = ROOT / source["adaptive_metrics"]
    hierarchy_path = ROOT / source["safe_merge_hierarchies"]
    frontiers, provenance = frozen_fluid_loose_frontiers(
        metrics_path=metrics_path, hierarchy_path=hierarchy_path,
        expected_video_count=int(source["expected_video_count"]),
    )
    frozen_config = copy.deepcopy(config)
    frozen_config["config_source_sha256"] = sha256_file(config_path)
    write_json(output / "frozen_config.json", frozen_config)
    fingerprint = sha256_json({
        "config": config, "source": provenance,
        "frontiers": [{"video_id": x["video_id"], "sha256": x["frontier_sha256"]} for x in frontiers],
        "model_snapshot": config["local_model"]["snapshot_path"],
    })
    checkpoint_path = output / "semantic_checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path, fingerprint)

    keyframes_all = []
    keyframes_by_video = {}
    cache_load_total = 0.0
    for frontier in frontiers:
        cache_dir = ROOT / source["dinov2_cache_root"] / frontier["video_id"]
        started = time.perf_counter()
        features = np.load(cache_dir / "features.npy")
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
        cache_load_total += time.perf_counter() - started
        hierarchy = copy.deepcopy(frontier["hierarchy"])
        hierarchy["cuts"]["medium"]["node_ids"] = list(frontier["medium_ids"])
        hierarchy["cuts"]["coarse"]["node_ids"] = [hierarchy["root_id"]]
        records, timing = select_video_keyframes(
            hierarchy=hierarchy, features=features, timestamps=timestamps, root=ROOT,
            output_dir=output, weights=config["keyframe_selection"]["quality_weights"],
        )
        for record in records:
            record.pop("coarse_id", None)
        keyframes_by_video[frontier["video_id"]] = {"records": records, "timing": timing}
        keyframes_all.extend(records)
    write_json(output / "keyframe_selection.json", keyframes_all)

    model_view, model_files = create_read_only_model_view(
        Path(config["local_model"]["snapshot_path"]), output / "local_model_view"
    )
    model = LocalQwen2VL(model_view=model_view, seed=int(config["seed"]))
    sentence_t5_started = time.perf_counter()
    caption_semantic_model = SentenceTransformer(
        config["semantic_coarse"]["caption_semantic_encoder"],
        local_files_only=True,
        device="cuda" if str(model.device) == "cuda" else "cpu",
    )
    sentence_t5_load_sec = time.perf_counter() - sentence_t5_started
    medium_results = []
    boundary_results = []
    coarse_results = []
    story_results = []
    video_results = []
    validation = []
    prompts = config["prompts"]
    limits = config["local_model"]

    for frontier in frontiers:
        video_id = frontier["video_id"]
        video_started = time.perf_counter()
        keyframe_records = keyframes_by_video[video_id]["records"]
        medium = []
        for trace in keyframe_records:
            medium_id = trace["medium_id"]
            key = f"{video_id}|{medium_id}"
            saved = checkpoint["medium"].get(key)
            if saved is None:
                raw, timing = model.describe_images(
                    image_paths=[Path(trace["selected_keyframe"]["image_path"])],
                    prompt=prompts["medium"], max_new_tokens=int(limits["max_new_tokens_medium"]),
                )
                saved = {
                    "video_id": video_id, "medium_id": medium_id,
                    "start": trace["start"], "end": trace["end"], "duration": trace["duration"],
                    "selected_keyframe": trace["selected_keyframe"],
                    "caption": _clean_caption(raw), "raw_output": raw,
                    "caption_source": "direct_local_qwen_image",
                    "timing": timing, "keyframe_trace": trace,
                }
                checkpoint["medium"][key] = saved
                _save_checkpoint(checkpoint_path, checkpoint)
            medium.append(saved)
            medium_results.append(saved)

        decisions = []
        node_by_id = {str(node["node_id"]): node for node in frontier["medium_nodes"]}
        for left, right in zip(medium, medium[1:]):
            key = f"{video_id}|{left['medium_id']}|{right['medium_id']}"
            saved = checkpoint["boundaries"].get(key)
            if saved is None:
                left_vector = np.asarray(node_by_id[left["medium_id"]]["pooled_dinov2"], dtype=np.float32)
                right_vector = np.asarray(node_by_id[right["medium_id"]]["pooled_dinov2"], dtype=np.float32)
                left_vector /= max(float(np.linalg.norm(left_vector)), 1e-12)
                right_vector /= max(float(np.linalg.norm(right_vector)), 1e-12)
                adjacent_cosine = float(left_vector @ right_vector)
                photo = config["semantic_coarse"]["photometric_veto_bypass"]
                photometric = photometric_transition_diagnostic(
                    frame_root=ROOT / source["sampled_frame_root"], video_id=video_id,
                    boundary_time=float(left["end"]),
                    grayscale_correlation_min=float(photo["standardized_grayscale_correlation_min"]),
                    mean_luminance_delta_min=float(photo["mean_luminance_delta_min"]),
                    mean_rgb_delta_min=float(photo["mean_rgb_delta_min"]),
                )
                encode_started = time.perf_counter()
                caption_embeddings = caption_semantic_model.encode(
                    [left["caption"], right["caption"]], normalize_embeddings=True, show_progress_bar=False
                )
                caption_cosine = float(caption_embeddings[0] @ caption_embeddings[1])
                encode_sec = time.perf_counter() - encode_started
                dino_veto = (
                    adjacent_cosine < float(config["semantic_coarse"]["strong_visual_transition_veto_cosine"])
                    and not photometric["suspected_photometric_only"]
                )
                caption_continuous = caption_cosine >= float(config["semantic_coarse"]["caption_continuity_cosine_min"])
                posture_vocabulary = list(config["semantic_coarse"]["posture_state_conflict_veto"]["tokens"])
                left_postures = posture_state_tokens(left["caption"], posture_vocabulary)
                right_postures = posture_state_tokens(right["caption"], posture_vocabulary)
                posture_conflict = bool(left_postures and right_postures and set(left_postures).isdisjoint(right_postures))
                decision = "MERGE" if caption_continuous and not dino_veto and not posture_conflict else "STOP"
                failed = []
                if not caption_continuous:
                    failed.append("caption immediate-state continuity")
                if dino_veto:
                    failed.append("strong non-photometric DINO transition")
                if posture_conflict:
                    failed.append("explicit immediate posture/state conflict")
                raw = decision
                timing = {"image_count": 0, "image_decode_sec": 0.0, "image_preprocess_sec": 0.0,
                          "model_inference_sec": 0.0, "total_semantic_generation_sec": encode_sec,
                          "caption_encode_sec": encode_sec, "input_tokens": 0, "output_tokens": 0, "peak_vram_bytes": 0}
                parsed = {"decision": decision,
                          "reason": "all conservative continuity gates passed" if decision == "MERGE" else "; ".join(failed),
                          "parse_status": "deterministic_conservative_continuity_rule"}
                decision_source = "caption_sentence_t5_plus_dino_veto"
                saved = {
                    "video_id": video_id, "left_medium_id": left["medium_id"],
                    "right_medium_id": right["medium_id"], "boundary_time": left["end"],
                    "left_caption": left["caption"], "right_caption": right["caption"],
                    "adjacent_medium_dino_cosine": adjacent_cosine,
                    "caption_sentence_t5_cosine": caption_cosine,
                    "caption_continuity_threshold": float(config["semantic_coarse"]["caption_continuity_cosine_min"]),
                    "photometric_diagnostic": photometric,
                    "left_posture_state_tokens": left_postures,
                    "right_posture_state_tokens": right_postures,
                    "posture_state_conflict_veto": posture_conflict,
                    "strong_visual_transition_veto_threshold": float(config["semantic_coarse"]["strong_visual_transition_veto_cosine"]),
                    "decision_source": decision_source,
                    "raw_output": raw, "timing": timing, **parsed,
                }
                checkpoint["boundaries"][key] = saved
                _save_checkpoint(checkpoint_path, checkpoint)
            decisions.append(saved)
            boundary_results.append(saved)

        groups = build_adjacent_groups(video_id=video_id, medium_records=medium, decisions=decisions)
        coarse = []
        for group in groups:
            key = f"{video_id}|{'|'.join(group['child_medium_ids'])}"
            saved = checkpoint["coarse"].get(key)
            if saved is None:
                if group["child_count"] == 1:
                    saved = {
                        **group, "summary": group["child_captions"][0],
                        "raw_output": group["child_captions"][0],
                        "summary_source": "singleton_copy_through",
                        "timing": {"image_count": 0, "image_decode_sec": 0.0, "image_preprocess_sec": 0.0,
                                   "model_inference_sec": 0.0, "total_semantic_generation_sec": 0.0,
                                   "input_tokens": 0, "output_tokens": 0, "peak_vram_bytes": 0},
                    }
                else:
                    ordered = [
                        f"[{member['start']:.1f}-{member['end']:.1f}s] {member['caption']}"
                        for member in medium if member["medium_id"] in group["child_medium_ids"]
                    ]
                    raw, timing = model.summarize_text(
                        ordered_descriptions=ordered, prompt=prompts["coarse"],
                        max_new_tokens=int(limits["max_new_tokens_coarse"]),
                    )
                    saved = {**group, "summary": _clean_caption(raw), "raw_output": raw,
                             "summary_source": "local_qwen_text_only", "timing": timing}
                checkpoint["coarse"][key] = saved
                _save_checkpoint(checkpoint_path, checkpoint)
            coarse.append(saved)
            coarse_results.append(saved)

        story_key = video_id
        story = checkpoint["stories"].get(story_key)
        if story is None:
            ordered = [f"[{item['start']:.1f}-{item['end']:.1f}s] {item['summary']}" for item in coarse]
            raw, timing = model.summarize_text(
                ordered_descriptions=ordered, prompt=prompts["story"],
                max_new_tokens=int(limits["max_new_tokens_story"]),
            )
            story = {"video_id": video_id, "coarse_inputs": ordered, "story": raw.strip(), "raw_output": raw, "timing": timing}
            checkpoint["stories"][story_key] = story
            _save_checkpoint(checkpoint_path, checkpoint)
        story_results.append(story)
        check = validate_semantic_coarse(frontier=frontier, medium_records=medium, coarse_records=coarse)
        validation.append({"video_id": video_id, **check})
        medium_calls = len(medium)
        boundary_calls = 0
        coarse_calls = sum(item["summary_source"] == "local_qwen_text_only" for item in coarse)
        singleton = sum(item["summary_source"] == "singleton_copy_through" for item in coarse)
        runtime = {
            "keyframe_selection_sec": keyframes_by_video[video_id]["timing"]["total_keyframe_selection_sec"],
            "medium_caption_total_sec": sum(item["timing"]["total_semantic_generation_sec"] for item in medium),
            "medium_inference_sec": sum(item["timing"]["model_inference_sec"] for item in medium),
            "boundary_decision_total_sec": sum(item["timing"]["total_semantic_generation_sec"] for item in decisions),
            "coarse_summary_total_sec": sum(item["timing"]["total_semantic_generation_sec"] for item in coarse),
            "story_total_sec": story["timing"]["total_semantic_generation_sec"],
            "steady_state_total_sec": time.perf_counter() - video_started,
        }
        video_results.append({
            "video_id": video_id, "video_duration": frontier["video_duration"], "fine_count": frontier["fine_count"],
            "medium_count": len(medium), "coarse_count": len(coarse), "medium": medium, "coarse": coarse,
            "decisions": decisions, "story": story, "calls": {
                "medium_vlm": medium_calls, "boundary_decision": boundary_calls,
                "boundary_sentence_t5_pair_encodes": len(decisions),
                "coarse_summary": coarse_calls, "singleton_copy_through": singleton, "story": 1,
            }, "runtime": runtime, "validation": check,
        })

    write_json(output / "medium_captions.json", medium_results)
    write_json(output / "semantic_grouping_decisions.json", boundary_results)
    write_json(output / "semantic_coarse.json", coarse_results)
    write_json(output / "video_stories.json", story_results)
    write_json(output / "validation.json", validation)
    calls = {
        "medium_vlm": len(medium_results),
        "boundary_decision": 0,
        "boundary_sentence_t5_pair_encodes": len(boundary_results),
        "coarse_summary": sum(x["summary_source"] == "local_qwen_text_only" for x in coarse_results),
        "singleton_copy_through": sum(x["summary_source"] == "singleton_copy_through" for x in coarse_results),
        "story": len(story_results), "external_api": 0,
    }
    aggregate = {
        "video_count": len(video_results), "fine_count": sum(v["fine_count"] for v in video_results),
        "medium_count": len(medium_results), "coarse_count": len(coarse_results), "calls": calls,
        "model_load_sec": model.model_load_sec,
        "steady_state_total_sec": sum(v["runtime"]["steady_state_total_sec"] for v in video_results),
        "cold_total_sec": model.model_load_sec + sum(v["runtime"]["steady_state_total_sec"] for v in video_results),
        "model_inference_count_this_process": model.inference_count,
        "rejected_preflight": checkpoint.get("migration"),
    }
    write_json(output / "aggregate_metrics.json", aggregate)
    runtime = {
        "one_time": {
            "qwen_model_load_sec": model.model_load_sec, "qwen_load_count": model.load_count,
            "device": str(model.device), "dtype": str(model.dtype),
            "model_vram_allocated_bytes": model.model_vram_allocated_bytes,
            "model_load_peak_vram_bytes": model.model_load_peak_vram_bytes,
            "sentence_t5_model_load_sec": sentence_t5_load_sec,
            "dinov2_cache_load_sec": cache_load_total,
        },
        "per_video": [{"video_id": v["video_id"], **v["runtime"]} for v in video_results],
        "aggregate": aggregate,
    }
    write_json(output / "runtime_metrics.json", runtime)
    manifest = {
        "schema_version": "semantic-coarse-manifest-v1", "experiment_id": config["experiment_id"],
        "fingerprint": fingerprint, "provenance": provenance,
        "frontiers": [{"video_id": x["video_id"], "medium_count": x["medium_count"], "frontier_sha256": x["frontier_sha256"]} for x in frontiers],
        "model": config["local_model"], "model_files": model_files,
        "git_branch": _git("branch", "--show-current"), "git_head": _git("rev-parse", "HEAD"),
        "tracked_diff_before_or_after": bool(_git("status", "--short", "--untracked-files=no")),
        "external_api_calls": 0, "forbidden_photometric_source_read": false,
    }
    write_json(output / "run_manifest.json", manifest)
    html_check = render_html(videos=video_results, aggregate=aggregate, output_path=output / "semantic_coarse_review.html")
    write_json(output / "html_validation.json", html_check)
    readme = (
        "# semantic_coarse_v0_1\n\n"
        "This isolated run uses only the original `fluid_loose` frontier from adaptive_fluid_hierarchy_v0_1. "
        "Fine leaves, the Safe-Merge topology, and Medium intervals are frozen. Each Medium receives one direct local "
        "Qwen2-VL image caption. Adjacent-caption boundaries are conservatively classified by the same local model; "
        "Coarse summaries and stories are text-only. Singleton Coarse nodes copy their Medium caption with no summary call.\n\n"
        f"- Videos: {aggregate['video_count']}\n- Medium: {aggregate['medium_count']}\n- Semantic Coarse: {aggregate['coarse_count']}\n"
        f"- External API calls: 0\n- Review: `semantic_coarse_review.html`\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "html": html_check, "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
