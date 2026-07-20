"""Build a local-Qwen semantic map over the frozen EgoPolice hierarchy."""

from __future__ import annotations

import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.efficient_semantic_map_v0_1.experiment import (  # noqa: E402
    clean_caption,
    group_video_keyframes,
    hierarchy_views,
    load_jsonl,
    parse_story,
    select_video_keyframes,
    sha256_file,
)
from src.experiments.efficient_semantic_map_v0_1.reporting import render_semantic_map  # noqa: E402
from src.experiments.medium_semantic_abstraction.qwen_local import (  # noqa: E402
    LocalQwen2VL,
    create_read_only_model_view,
)


CONFIG_PATH = ROOT / "config/experiments/efficient_semantic_map_v0_1.json"
OUT = ROOT / "outputs/experiments/efficient_semantic_map_v0_1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_versions() -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for package in ("numpy", "torch", "transformers", "Pillow"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = None
    return output


def duration_label(start: float, end: float) -> str:
    return f"[{start:.1f}-{end:.1f}s]"


def main() -> int:
    wall_started = time.perf_counter()
    config = load_json(CONFIG_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    hierarchy_path = ROOT / config["source_hierarchy"]
    manifest_path = ROOT / config["source_manifest"]
    source_config_path = ROOT / config["source_experiment"] / "frozen_config.json"
    source_hash_before = sha256_file(hierarchy_path)
    hierarchies = load_jsonl(hierarchy_path)
    source_manifest = load_json(manifest_path)
    if len(hierarchies) != 4 or len(source_manifest["videos"]) != 4:
        raise RuntimeError("Expected exactly four frozen long-video hierarchies")
    if {row["video_id"] for row in hierarchies} != {row["video_id"] for row in source_manifest["videos"]}:
        raise RuntimeError("Hierarchy/manifest video identity mismatch")

    config_hash = sha256_file(CONFIG_PATH)
    fingerprint_payload = {
        "config_sha256": config_hash,
        "hierarchy_sha256": source_hash_before,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_frozen_config_sha256": sha256_file(source_config_path),
        "model_snapshot": config["local_model"]["snapshot_path"],
        "git_head": git_value("rev-parse", "HEAD"),
    }
    import hashlib
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_manifest = {
        "schema_version": "efficient-semantic-map-run-v1",
        "experiment_id": config["experiment_id"],
        "run_fingerprint": run_fingerprint,
        "fingerprint_components": fingerprint_payload,
        "source_videos": source_manifest["videos"],
        "video_ids": [row["video_id"] for row in hierarchies],
        "source_hierarchy_immutable": True,
        "question_conditioning": False,
        "gold_or_options_accessed": False,
        "external_api_calls": 0,
    }
    write_json(OUT / "run_manifest.json", run_manifest)

    keyframes: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    runtime_by_video: dict[str, dict[str, Any]] = {}
    feature_by_video: dict[str, np.ndarray] = {}
    timestamps_by_video: dict[str, np.ndarray] = {}
    hierarchy_by_video = {row["video_id"]: row for row in hierarchies}

    for index, hierarchy in enumerate(hierarchies, start=1):
        video_id = str(hierarchy["video_id"])
        print(f"[{index}/4] {video_id}: keyframe selection and semantic grouping", flush=True)
        feature_started = time.perf_counter()
        cache_dir = ROOT / config["source_dinov2_cache_root"] / video_id
        features = np.load(cache_dir / "features.npy")
        metadata = load_json(cache_dir / "metadata.json")
        if sha256_file(cache_dir / "features.npy") != metadata["features_sha256"]:
            raise RuntimeError(f"Frozen DINO cache hash mismatch: {video_id}")
        timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
        feature_load_sec = time.perf_counter() - feature_started
        feature_by_video[video_id] = features
        timestamps_by_video[video_id] = timestamps
        video_keyframes, keyframe_timing = select_video_keyframes(
            hierarchy=hierarchy, features=features, timestamps=timestamps, root=ROOT, output_dir=OUT,
            weights={key: float(value) for key, value in config["keyframe_selection"]["quality_weights"].items()},
        )
        video_groups, group_timing = group_video_keyframes(
            video_keyframes, features=features,
            threshold=float(config["semantic_grouping"]["cosine_similarity_threshold"]),
            maximum_group_size=int(config["semantic_grouping"]["maximum_group_size"]),
        )
        keyframes.extend(video_keyframes)
        groups.extend(video_groups)
        runtime_by_video[video_id] = {
            "video_id": video_id,
            "dinov2_feature_cache_load_sec": feature_load_sec,
            **keyframe_timing,
            **group_timing,
        }
        write_json(OUT / "keyframe_selection.json", keyframes)
        write_json(OUT / "semantic_groups.json", groups)

    model_view, model_files = create_read_only_model_view(
        Path(config["local_model"]["snapshot_path"]), OUT / "runtime_model_view"
    )
    print(
        f'Local model: {config["local_model"]["model"]} | snapshot={config["local_model"]["snapshot_path"]} '
        f'| planned image calls={len(groups)} | coarse text calls={sum(len(h["cuts"]["coarse"]["node_ids"]) for h in hierarchies)} '
        f'| story calls=4 | API calls=0',
        flush=True,
    )
    engine = LocalQwen2VL(model_view=model_view, seed=int(config["seed"]))
    print(f"Qwen loaded once on {engine.device}; load={engine.model_load_sec:.2f}s", flush=True)

    medium_results: list[dict[str, Any]] = []
    group_by_id = {row["semantic_group_id"]: row for row in groups}
    keyframe_by_medium = {(row["video_id"], row["medium_id"]): row for row in keyframes}
    existing_medium_path = OUT / "medium_captions.json"
    existing_medium = load_json(existing_medium_path) if existing_medium_path.is_file() else []
    if existing_medium and any(row.get("run_fingerprint") != run_fingerprint for row in existing_medium):
        raise RuntimeError("Incompatible Medium checkpoint fingerprint")
    completed_groups = {
        row["semantic_group_id"]: row for row in existing_medium if row["caption_source"] == "direct_vlm"
    }
    direct_by_group: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups, start=1):
        group_id = str(group["semantic_group_id"])
        if group_id in completed_groups:
            direct_by_group[group_id] = completed_groups[group_id]
            continue
        representative_id = str(group["representative_medium_id"])
        representative = keyframe_by_medium[(group["video_id"], representative_id)]
        raw, timing = engine.describe_images(
            image_paths=[Path(representative["selected_keyframe"]["image_path"])],
            prompt=str(config["medium_prompt"]),
            max_new_tokens=int(config["local_model"]["max_new_tokens_medium"]),
        )
        direct = {
            "run_fingerprint": run_fingerprint,
            "video_id": group["video_id"],
            "coarse_id": representative["coarse_id"],
            "medium_id": representative_id,
            "start": representative["start"],
            "end": representative["end"],
            "selected_keyframe": representative["selected_keyframe"],
            "semantic_group_id": group_id,
            "caption_source": "direct_vlm",
            "caption_source_medium_id": representative_id,
            "raw_vlm_output": raw,
            "cleaned_caption": clean_caption(raw),
            "inference_timing": timing,
        }
        direct_by_group[group_id] = direct
        checkpoint = [*completed_groups.values(), *direct_by_group.values()]
        write_json(existing_medium_path, checkpoint)
        if group_index % 10 == 0 or group_index == len(groups):
            print(f"  Medium semantic groups {group_index}/{len(groups)}", flush=True)

    for keyframe in keyframes:
        group_id = str(keyframe["semantic_group_id"])
        direct = direct_by_group[group_id]
        is_direct = keyframe["medium_id"] == direct["medium_id"]
        medium_results.append(
            {
                "run_fingerprint": run_fingerprint,
                "video_id": keyframe["video_id"],
                "coarse_id": keyframe["coarse_id"],
                "medium_id": keyframe["medium_id"],
                "start": keyframe["start"],
                "end": keyframe["end"],
                "selected_keyframe": keyframe["selected_keyframe"],
                "semantic_group_id": group_id,
                "similarity_to_group_representative": keyframe["similarity_to_group_representative"],
                "caption_source": "direct_vlm" if is_direct else "propagated_from_group",
                "caption_source_medium_id": direct["medium_id"],
                "raw_vlm_output": direct["raw_vlm_output"],
                "cleaned_caption": direct["cleaned_caption"],
                "inference_timing": direct["inference_timing"] if is_direct else {
                    "image_count": 0,
                    "image_decode_sec": 0.0,
                    "image_preprocess_sec": 0.0,
                    "model_inference_sec": 0.0,
                    "total_semantic_generation_sec": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "propagated_no_inference": True,
                },
            }
        )
    medium_results.sort(key=lambda row: (row["video_id"], row["start"], row["medium_id"]))
    write_json(existing_medium_path, medium_results)

    medium_by_id = {(row["video_id"], row["medium_id"]): row for row in medium_results}
    coarse_results: list[dict[str, Any]] = []
    existing_coarse_path = OUT / "coarse_captions.json"
    existing_coarse = load_json(existing_coarse_path) if existing_coarse_path.is_file() else []
    if existing_coarse and any(row.get("run_fingerprint") != run_fingerprint for row in existing_coarse):
        raise RuntimeError("Incompatible Coarse checkpoint fingerprint")
    completed_coarse = {(row["video_id"], row["coarse_id"]): row for row in existing_coarse}
    for hierarchy in hierarchies:
        video_id = str(hierarchy["video_id"])
        for view in hierarchy_views(hierarchy):
            coarse = view["coarse"]
            key = (video_id, str(coarse["node_id"]))
            if key in completed_coarse:
                coarse_results.append(completed_coarse[key])
                continue
            prep_started = time.perf_counter()
            ordered = [
                f'{duration_label(float(node["start"]), float(node["end"]))} '
                f'{medium_by_id[(video_id, str(node["node_id"]))]["cleaned_caption"]}'
                for node in view["medium"]
            ]
            prep_sec = time.perf_counter() - prep_started
            raw, timing = engine.summarize_text(
                ordered_descriptions=ordered,
                prompt=str(config["coarse_prompt"]),
                max_new_tokens=int(config["local_model"]["max_new_tokens_coarse"]),
            )
            record = {
                "run_fingerprint": run_fingerprint,
                "video_id": video_id,
                "coarse_id": str(coarse["node_id"]),
                "start": float(coarse["start"]),
                "end": float(coarse["end"]),
                "medium_ids": [str(node["node_id"]) for node in view["medium"]],
                "ordered_medium_caption_input": ordered,
                "raw_model_output": raw,
                "cleaned_caption": clean_caption(raw),
                "text_preparation_sec": prep_sec,
                "inference_timing": timing,
                "input_uses_text_only": True,
            }
            coarse_results.append(record)
            write_json(existing_coarse_path, [*completed_coarse.values(), *coarse_results])
    coarse_results.sort(key=lambda row: (row["video_id"], row["start"], row["coarse_id"]))
    write_json(existing_coarse_path, coarse_results)

    stories: list[dict[str, Any]] = []
    existing_story_path = OUT / "video_stories.json"
    existing_stories = load_json(existing_story_path) if existing_story_path.is_file() else []
    if existing_stories and any(row.get("run_fingerprint") != run_fingerprint for row in existing_stories):
        raise RuntimeError("Incompatible story checkpoint fingerprint")
    completed_story = {row["video_id"]: row for row in existing_stories}
    for hierarchy in hierarchies:
        video_id = str(hierarchy["video_id"])
        if video_id in completed_story:
            stories.append(completed_story[video_id])
            continue
        prep_started = time.perf_counter()
        ordered_coarse = [
            f'{duration_label(row["start"], row["end"])} {row["cleaned_caption"]}'
            for row in coarse_results if row["video_id"] == video_id
        ]
        prep_sec = time.perf_counter() - prep_started
        raw, timing = engine.summarize_text(
            ordered_descriptions=ordered_coarse,
            prompt=str(config["story_prompt"]),
            max_new_tokens=int(config["local_model"]["max_new_tokens_story"]),
        )
        overall, storyline = parse_story(raw)
        storyline_source = "parsed_model_output"
        if len(storyline) < 2:
            # Preserve a complete ordered global map when Qwen ignores the requested
            # numbered-list format. These are already generated Coarse captions,
            # not new visual inference or fabricated semantic content.
            storyline = list(ordered_coarse)
            storyline_source = "ordered_coarse_caption_fallback_due_format_noncompliance"
        record = {
            "run_fingerprint": run_fingerprint,
            "video_id": video_id,
            "ordered_coarse_caption_input": ordered_coarse,
            "raw_model_output": raw,
            "overall_story": overall,
            "high_level_storyline": storyline,
            "high_level_storyline_source": storyline_source,
            "text_preparation_sec": prep_sec,
            "inference_timing": timing,
            "input_uses_coarse_text_only": True,
        }
        stories.append(record)
        write_json(existing_story_path, [*completed_story.values(), *stories])
    stories.sort(key=lambda row: row["video_id"])
    write_json(existing_story_path, stories)

    group_video_counts = {video_id: 0 for video_id in hierarchy_by_video}
    for group in groups:
        group_video_counts[group["video_id"]] += 1
    coarse_video_counts = {video_id: 0 for video_id in hierarchy_by_video}
    for coarse in coarse_results:
        coarse_video_counts[coarse["video_id"]] += 1
    efficiency_rows: list[dict[str, Any]] = []
    for hierarchy in hierarchies:
        video_id = str(hierarchy["video_id"])
        medium_count = len(hierarchy["cuts"]["medium"]["node_ids"])
        group_count = group_video_counts[video_id]
        candidate_count = sum(row["candidate_count"] for row in keyframes if row["video_id"] == video_id)
        direct = [row for row in medium_results if row["video_id"] == video_id and row["caption_source"] == "direct_vlm"]
        avoided = medium_count - group_count
        efficiency_rows.append(
            {
                "video_id": video_id,
                "duration_sec": float(hierarchy["video_duration"]),
                "fine_count": len(hierarchy["cuts"]["fine"]["node_ids"]),
                "medium_count": medium_count,
                "coarse_count": coarse_video_counts[video_id],
                "candidate_frame_count": candidate_count,
                "selected_keyframe_count": medium_count,
                "semantic_group_count": group_count,
                "hypothetical_every_medium_vlm_calls": medium_count,
                "actual_direct_vlm_image_calls": len(direct),
                "propagated_caption_count": avoided,
                "vlm_calls_avoided": avoided,
                "vlm_call_reduction_ratio": avoided / medium_count,
                "total_images_processed_by_vlm": len(direct),
                "mean_direct_caption_total_sec": statistics.fmean(
                    row["inference_timing"]["total_semantic_generation_sec"] for row in direct
                ),
                "mean_caption_cost_per_medium_including_propagation_sec": sum(
                    row["inference_timing"]["total_semantic_generation_sec"] for row in direct
                ) / medium_count,
            }
        )
    write_json(OUT / "efficiency_metrics.json", efficiency_rows)

    for row in efficiency_rows:
        video_id = row["video_id"]
        runtime = runtime_by_video[video_id]
        direct = [item for item in medium_results if item["video_id"] == video_id and item["caption_source"] == "direct_vlm"]
        video_coarse = [item for item in coarse_results if item["video_id"] == video_id]
        story = next(item for item in stories if item["video_id"] == video_id)
        runtime.update(
            {
                "medium_image_decode_sec": sum(item["inference_timing"]["image_decode_sec"] for item in direct),
                "medium_image_preprocessing_sec": sum(item["inference_timing"]["image_preprocess_sec"] for item in direct),
                "medium_group_vlm_inference_sec": sum(item["inference_timing"]["model_inference_sec"] for item in direct),
                "medium_semantic_caption_total_sec": sum(item["inference_timing"]["total_semantic_generation_sec"] for item in direct),
                "coarse_text_preparation_sec": sum(item["text_preparation_sec"] for item in video_coarse),
                "coarse_llm_inference_sec": sum(item["inference_timing"]["model_inference_sec"] for item in video_coarse),
                "coarse_semantic_total_sec": sum(item["inference_timing"]["total_semantic_generation_sec"] for item in video_coarse),
                "whole_video_story_text_preparation_sec": story["text_preparation_sec"],
                "whole_video_story_inference_sec": story["inference_timing"]["model_inference_sec"],
                "whole_video_story_total_sec": story["inference_timing"]["total_semantic_generation_sec"],
            }
        )
        additive = (
            runtime["dinov2_feature_cache_load_sec"]
            + runtime["total_keyframe_selection_sec"]
            + runtime["total_semantic_deduplication_sec"]
            + runtime["medium_semantic_caption_total_sec"]
            + runtime["coarse_semantic_total_sec"]
            + runtime["whole_video_story_total_sec"]
        )
        runtime["steady_state_semantic_indexing_sec"] = additive
        runtime["semantic_indexing_sec_per_video_minute"] = additive / (row["duration_sec"] / 60.0)

    runtime_rows = [runtime_by_video[row["video_id"]] for row in efficiency_rows]
    peak_vram = max(
        [engine.model_load_peak_vram_bytes]
        + [row["inference_timing"]["peak_vram_bytes"] for row in medium_results if row["caption_source"] == "direct_vlm"]
        + [row["inference_timing"]["peak_vram_bytes"] for row in coarse_results]
        + [row["inference_timing"]["peak_vram_bytes"] for row in stories]
    )
    runtime_metrics = {
        "one_time": {
            "qwen_model_load_sec": engine.model_load_sec,
            "qwen_load_count": engine.load_count,
            "model_instance_id": engine.model_instance_id,
            "processor_instance_id": engine.processor_instance_id,
            "model_vram_allocated_bytes": engine.model_vram_allocated_bytes,
            "peak_vram_bytes": peak_vram,
            "dinov2_feature_cache_load_sec_total": sum(row["dinov2_feature_cache_load_sec"] for row in runtime_rows),
        },
        "per_video": runtime_rows,
        "steady_state_total_sec": sum(row["steady_state_semantic_indexing_sec"] for row in runtime_rows),
        "cold_total_sec": engine.model_load_sec + sum(row["steady_state_semantic_indexing_sec"] for row in runtime_rows),
        "local_inference_call_count": engine.inference_count,
        "external_api_calls": 0,
        "model_loading_excluded_from_per_video_cost": True,
        "wall_sec_before_html": time.perf_counter() - wall_started,
    }
    write_json(OUT / "runtime_metrics.json", runtime_metrics)

    aggregate = {
        "video_count": 4,
        "total_fine_nodes": sum(row["fine_count"] for row in efficiency_rows),
        "total_medium_nodes": sum(row["medium_count"] for row in efficiency_rows),
        "total_coarse_nodes": sum(row["coarse_count"] for row in efficiency_rows),
        "total_candidate_frames": sum(row["candidate_frame_count"] for row in efficiency_rows),
        "total_selected_keyframes": sum(row["selected_keyframe_count"] for row in efficiency_rows),
        "total_semantic_groups": sum(row["semantic_group_count"] for row in efficiency_rows),
        "hypothetical_every_medium_vlm_calls": sum(row["medium_count"] for row in efficiency_rows),
        "actual_direct_vlm_image_calls": sum(row["actual_direct_vlm_image_calls"] for row in efficiency_rows),
        "vlm_calls_avoided": sum(row["vlm_calls_avoided"] for row in efficiency_rows),
        "vlm_call_reduction_ratio": sum(row["vlm_calls_avoided"] for row in efficiency_rows) / sum(
            row["medium_count"] for row in efficiency_rows
        ),
        "mean_direct_caption_total_sec": statistics.fmean(
            row["inference_timing"]["total_semantic_generation_sec"]
            for row in medium_results if row["caption_source"] == "direct_vlm"
        ),
        "steady_state_total_sec": runtime_metrics["steady_state_total_sec"],
        "cold_total_sec": runtime_metrics["cold_total_sec"],
        "external_api_calls": 0,
    }
    write_json(OUT / "aggregate_metrics.json", aggregate)

    frozen_config = {
        **config,
        "run_fingerprint": run_fingerprint,
        "config_sha256": config_hash,
        "source_hierarchy_sha256_before": source_hash_before,
        "source_hierarchy_sha256_after": sha256_file(hierarchy_path),
        "source_hierarchy_unchanged": sha256_file(hierarchy_path) == source_hash_before,
        "model_files": model_files,
        "git_branch": git_value("branch", "--show-current"),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_at_execution": git_value("status", "--short"),
        "device": str(engine.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": package_versions(),
        "external_api_calls": 0,
    }
    write_json(OUT / "frozen_config.json", frozen_config)

    html_started = time.perf_counter()
    validation = render_semantic_map(
        output_path=OUT / "semantic_map_review.html", source_manifest=source_manifest,
        hierarchies=hierarchies, keyframes=keyframes, semantic_groups=groups,
        medium_captions=medium_results, coarse_captions=coarse_results, stories=stories,
        efficiency=efficiency_rows, runtimes=runtime_rows,
    )
    runtime_metrics["html_generation_sec"] = time.perf_counter() - html_started
    runtime_metrics["experiment_wall_sec"] = time.perf_counter() - wall_started
    write_json(OUT / "runtime_metrics.json", runtime_metrics)
    write_json(OUT / "html_validation.json", validation)
    (OUT / "README.md").write_text(
        "# Efficient semantic map v0.1\n\n"
        "A strictly local semantic layer over the frozen EgoPolice long-video Fine → Medium → Coarse hierarchy. "
        "Cheap quality and existing DINOv2 features select one Medium keyframe. Conservative complete-link "
        "groups may reuse a directly generated Qwen2-VL caption; all propagation is explicit. Coarse summaries "
        "read ordered Medium text only, and whole-video stories read ordered Coarse text only.\n\n"
        "No hierarchy boundary or membership is changed. No question, answer, retrieval, Planner, audio, QA, "
        "external API, or canonical pipeline path is used.\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate, "runtime": runtime_metrics, "html_validation": validation}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
