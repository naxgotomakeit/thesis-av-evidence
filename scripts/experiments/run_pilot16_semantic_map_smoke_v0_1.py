"""Run the pilot16 semantic-map chain for one video.

This adapter composes the existing frozen algorithms.  It does not implement
the 25/50/75 voting proposal and never falls back from the external runtime
volume to the internal disk.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import resource
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.adaptive_fluid_hierarchy_v0_1.experiment import (  # noqa: E402
    build_tree_context,
    fluid_frontier,
    validate_frontiers,
)
from src.experiments.coarse_segmentation.comet_style import CometStyleConfig, segment  # noqa: E402
from src.experiments.coarse_segmentation.dinov2_features import DINOv2FeatureExtractor  # noqa: E402
from src.experiments.coarse_segmentation.schema import make_segmentation  # noqa: E402
from src.experiments.efficient_semantic_map_v0_1.experiment import (  # noqa: E402
    clean_caption,
    group_video_keyframes,
    hierarchy_views,
    parse_story,
    select_video_keyframes,
)
from src.experiments.efficient_semantic_map_v0_1.reporting import render_semantic_map  # noqa: E402
from src.experiments.fine_to_coarse_hierarchy.hierarchy import (  # noqa: E402
    build_boundary_records,
    build_fine_nodes,
    build_safe_hierarchy,
)
from src.experiments.long_video_hierarchy_stress.experiment import extract_frames_1fps  # noqa: E402
from src.experiments.medium_semantic_abstraction.qwen_local import (  # noqa: E402
    LocalQwen2VL,
    create_read_only_model_view,
)

CONFIG = ROOT / "config/experiments/pilot16_semantic_map_smoke_v0_1.json"
SEMANTIC_CONFIG = ROOT / "config/experiments/efficient_semantic_map_v0_1.json"
HIERARCHY_CONFIG = ROOT / "config/experiments/long_video_hierarchy_stress_v0_2.json"
FLUID_CONFIG = ROOT / "config/experiments/adaptive_fluid_hierarchy_v0_1.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def mps_memory() -> dict[str, int | None]:
    if not hasattr(torch, "mps"):
        return {"current_allocated_bytes": None, "driver_allocated_bytes": None}
    current = getattr(torch.mps, "current_allocated_memory", None)
    driver = getattr(torch.mps, "driver_allocated_memory", None)
    return {
        "current_allocated_bytes": int(current()) if current else None,
        "driver_allocated_bytes": int(driver()) if driver else None,
    }


def safe_dataset_id(dataset_video_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", dataset_video_id.replace("/", "__"))
    value = value.strip("._-")
    if not value:
        raise RuntimeError(f"Unsafe empty dataset directory name for {dataset_video_id!r}")
    return value


def external_runtime_preflight(config: dict[str, Any]) -> dict[str, Any]:
    mount = Path(config["required_mount"])
    runtime_root = Path(config["runtime_root"])
    model_cache = Path(config["model_cache"])
    if not mount.is_dir():
        raise RuntimeError(f"Required external mount is not mounted: {mount}")
    if not runtime_root.is_relative_to(mount):
        raise RuntimeError(f"runtime_root must be below required mount: {runtime_root}")
    if not model_cache.is_dir():
        raise RuntimeError(f"Configured model_cache is unavailable: {model_cache}")
    runtime_root.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(runtime_root)
    free_bytes = stat.f_bavail * stat.f_frsize
    minimum_bytes = int(float(config["minimum_free_gib"]) * (1024 ** 3))
    if free_bytes < minimum_bytes:
        raise RuntimeError(f"Insufficient external free space: {free_bytes / 2**30:.2f} GiB")
    try:
        with tempfile.NamedTemporaryFile(prefix=".pilot16_runtime_probe_", dir=runtime_root, delete=True) as handle:
            handle.write(b"external-runtime-write-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeError(f"External runtime directory is not writable: {runtime_root}: {exc}") from exc
    return {
        "required_mount": str(mount), "runtime_root": str(runtime_root), "model_cache": str(model_cache),
        "free_bytes": free_bytes, "free_gib": free_bytes / 2**30,
        "minimum_free_gib": float(config["minimum_free_gib"]), "write_probe": "passed",
        "fallback_to_internal_disk": False,
    }


def probe(path: Path, ffprobe: Path) -> dict[str, Any]:
    command = [str(ffprobe), "-v", "error", "-show_entries",
               "stream=index,codec_type,codec_name,width,height,avg_frame_rate:format=duration",
               "-of", "json", str(path)]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return {"duration_sec": float(payload["format"]["duration"]), "streams": payload.get("streams", []),
            "audio_present": any(row.get("codec_type") == "audio" for row in payload.get("streams", []))}


def stage_record(started: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**(extra or {}), "elapsed_sec": time.perf_counter() - started,
            "peak_rss_bytes": max_rss_bytes(), "mps_memory": mps_memory()}


def external_html_links(output: Path, candidate_dir: Path, records: list[dict[str, Any]]) -> None:
    """Expose external images to the renderer through a lightweight symlink."""
    link_root = output / "external_assets"
    if link_root.exists() and not link_root.is_symlink():
        raise RuntimeError(f"Refusing to replace non-symlink renderer asset path: {link_root}")
    if link_root.is_symlink() and link_root.resolve() != candidate_dir.resolve():
        link_root.unlink()
    if not link_root.exists():
        link_root.symlink_to(candidate_dir, target_is_directory=True)
    for record in records:
        for item in record["candidates"]:
            item["html_relative_path"] = (Path("external_assets") / Path(item["image_path"]).relative_to(candidate_dir)).as_posix()
        record["selected_keyframe"]["html_relative_path"] = (Path("external_assets") / Path(record["selected_keyframe"]["image_path"]).relative_to(candidate_dir)).as_posix()


def add_external_image_fallback(review_path: Path, output: Path) -> None:
    """Keep relative links, with a local file URI fallback for symlink-hostile viewers."""
    text = review_path.read_text(encoding="utf-8")
    pattern = re.compile(r'<img src="(external_assets/[^"]+)"')

    def replace(match: re.Match[str]) -> str:
        relative = match.group(1)
        external = (output / relative).resolve()
        if not external.is_file():
            raise FileNotFoundError(external)
        uri = external.as_uri()
        return (f'<img src="{relative}" data-external-file="{uri}" '
                'onerror="this.onerror=null;this.src=this.dataset.externalFile"')

    updated = pattern.sub(replace, text)
    review_path.write_text(updated, encoding="utf-8")


def main() -> int:
    config_path = Path(os.environ.get("PILOT16_SMOKE_CONFIG", str(CONFIG)))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    semantic = json.loads(SEMANTIC_CONFIG.read_text(encoding="utf-8"))
    hierarchy_config = json.loads(HIERARCHY_CONFIG.read_text(encoding="utf-8"))
    fluid_config = json.loads(FLUID_CONFIG.read_text(encoding="utf-8"))
    preflight = external_runtime_preflight(config)
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    video = Path(config["video_path"])
    video_id = str(config["dataset_video_id"])
    safe_id = safe_dataset_id(video_id)
    runtime_video_root = Path(config["runtime_root"]) / safe_id
    frames_dir = runtime_video_root / "frames_1fps"
    candidate_dir = runtime_video_root / "candidate_frames"
    keyframes_dir = runtime_video_root / "keyframes"
    feature_cache_dir = runtime_video_root / str(config["dino_cache_dir_name"])
    qwen_view_dir = runtime_video_root / "qwen_model_view"
    for path in (runtime_video_root, frames_dir, candidate_dir, keyframes_dir, feature_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "status": "running", "experiment_id": config["experiment_id"], "dataset_video_id": video_id,
        "safe_dataset_id": safe_id, "video_path": video.as_posix(), "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "python": {"executable": sys.executable, "version": sys.version.split()[0], "machine": platform.machine()},
        "torch": {"version": torch.__version__, "mps_built": bool(torch.backends.mps.is_built()), "mps_available": bool(torch.backends.mps.is_available())},
        "external_runtime_preflight": preflight,
        "runtime_paths": {"video_root": runtime_video_root.as_posix(), "frames_1fps": frames_dir.as_posix(),
                           "candidate_frames": candidate_dir.as_posix(), "keyframes": keyframes_dir.as_posix(),
                           "feature_cache": feature_cache_dir.as_posix(), "qwen_model_view": qwen_view_dir.as_posix()},
        "stages": {}, "started_at_epoch_sec": time.time(),
    }
    write_json(output / "run_manifest.json", manifest)
    current_stage = "initialization"
    started_total = time.perf_counter()

    try:
        current_stage = "ffprobe"
        t = time.perf_counter(); media = probe(video, Path(config["ffprobe"]))
        manifest["media"] = media; manifest["stages"][current_stage] = stage_record(t, media); write_json(output / "run_manifest.json", manifest)

        current_stage = "extract_1fps"
        t = time.perf_counter()
        frames, timestamps, frame_meta = extract_frames_1fps(video, frames_dir, duration_sec=media["duration_sec"],
            jpeg_quality=int(config["jpeg_quality"]), ffmpeg_path=Path(config["ffmpeg"]))
        manifest["stages"][current_stage] = stage_record(t, {"frame_count": len(frames), "first_frame": str(frames[0]),
            "last_frame": str(frames[-1]), "timestamps_first_last": [float(timestamps[0]), float(timestamps[-1]),], "metadata": frame_meta})
        write_json(output / "run_manifest.json", manifest)

        current_stage = "dinov2_features"
        t = time.perf_counter(); device = "mps" if torch.backends.mps.is_available() else "cpu"
        extractor = DINOv2FeatureExtractor(device=device, batch_size=int(hierarchy_config["dinov2"]["batch_size"]))
        features, dino_meta = extractor.extract_or_load(video_id=video_id, frame_paths=frames, timestamps=timestamps, cache_dir=feature_cache_dir)
        manifest["stages"][current_stage] = stage_record(t, {"shape": list(features.shape), "device": device, "metadata": dino_meta})
        write_json(output / "run_manifest.json", manifest)

        current_stage = "fine_segmentation"
        t = time.perf_counter(); comet_cfg = hierarchy_config["comet_style"]
        comet = segment(features, timestamps, CometStyleConfig(**comet_cfg))
        fine = make_segmentation(video_id=video_id, method="comet_style_dinov2", video_duration=media["duration_sec"],
                                 boundaries=comet["boundaries"], frame_timestamps=timestamps)
        boundaries = build_boundary_records(fine["segments"], comet)
        fine_nodes = build_fine_nodes(segments=fine["segments"], features=features, timestamps=timestamps,
            frame_paths=[Path(item).as_posix() for item in frames], boundary_records=boundaries,
            representative_fractions=tuple(hierarchy_config["hierarchy"]["representative_fractions"]),
            include_medoid=bool(hierarchy_config["hierarchy"]["include_dinov2_medoid"]))
        manifest["stages"][current_stage] = stage_record(t, {"fine_count": len(fine_nodes), "boundary_count": len(boundaries), "comet_diagnostics": comet["diagnostics"]})
        write_json(output / "fine_events.json", {"segmentation": fine, "boundaries": boundaries, "fine_nodes": fine_nodes})
        write_json(output / "run_manifest.json", manifest)

        current_stage = "safe_merge_hierarchy"
        t = time.perf_counter(); hierarchy, merge_log = build_safe_hierarchy(video_id=video_id, video_duration=media["duration_sec"],
            fine_nodes=fine_nodes, boundary_records=boundaries, timestamps=timestamps,
            frame_paths=[Path(item).as_posix() for item in frames], features=features,
            medium_fraction=float(hierarchy_config["hierarchy"]["medium_reference_fraction"]),
            coarse_fraction=float(hierarchy_config["hierarchy"]["coarse_reference_fraction"]),
            representative_fractions=tuple(hierarchy_config["hierarchy"]["representative_fractions"]),
            include_medoid=bool(hierarchy_config["hierarchy"]["include_dinov2_medoid"]))
        manifest["stages"][current_stage] = stage_record(t, {"merge_log_count": len(merge_log)})
        manifest["fine_count"] = len(fine_nodes)
        write_json(output / "run_manifest.json", manifest)

        current_stage = "fluid_loose_frontier"
        t = time.perf_counter(); context = build_tree_context(hierarchy); loose = fluid_config["methods"]["fluid_loose"]
        medium_ids, medium_trace = fluid_frontier(context, minimum_q_rank=float(loose["medium"]["minimum_q_rank"]), maximum_local_drop=float(loose["medium"]["maximum_local_drop"]), level="medium")
        coarse_ids, coarse_trace = fluid_frontier(context, minimum_q_rank=float(loose["coarse"]["minimum_q_rank"]), maximum_local_drop=float(loose["coarse"]["maximum_local_drop"]), level="coarse")
        frontier_validation = validate_frontiers(hierarchy, medium_ids, coarse_ids)
        if not frontier_validation["valid"]:
            raise RuntimeError(f"Invalid Fluid Loose frontiers: {frontier_validation}")
        node_map = {str(row["node_id"]): row for row in hierarchy["nodes"]}
        manifest["stages"][current_stage] = stage_record(t, {"medium_count": len(medium_ids), "coarse_count": len(coarse_ids), "validation": frontier_validation})
        write_json(output / "medium_events.json", {"policy": "fluid_loose", "nodes": [node_map[item] for item in medium_ids], "frontier_trace": medium_trace, "coarse_nodes": [node_map[item] for item in coarse_ids], "coarse_frontier_trace": coarse_trace, "validation": frontier_validation})
        write_json(output / "run_manifest.json", manifest)

        current_stage = "weighted_keyframe_selection"
        t = time.perf_counter(); selection_hierarchy = copy.deepcopy(hierarchy)
        selection_hierarchy["cuts"]["medium"]["node_ids"] = medium_ids
        selection_hierarchy["cuts"]["coarse"]["node_ids"] = coarse_ids
        keyframes, selection_timing = select_video_keyframes(hierarchy=selection_hierarchy, features=features, timestamps=timestamps,
            root=ROOT, output_dir=candidate_dir, weights=semantic["keyframe_selection"]["quality_weights"])
        # Keep a separate selected-keyframe directory on the external volume.
        for record in keyframes:
            selected = record["selected_keyframe"]
            target = keyframes_dir / safe_id / f'{record["medium_id"]}_{Path(selected["image_path"]).name}'
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(selected["image_path"], target)
            selected["keyframes_external_path"] = target.as_posix()
        external_html_links(output, candidate_dir, keyframes)
        manifest["stages"][current_stage] = stage_record(t, {"keyframe_count": len(keyframes), "selection_timing": selection_timing})
        write_json(output / "keyframe_selection.json", {"records": keyframes, "selection_timing": selection_timing, "algorithm": semantic["keyframe_selection"]})
        write_json(output / "run_manifest.json", manifest)

        current_stage = "semantic_grouping"
        t = time.perf_counter(); groups, grouping_timing = group_video_keyframes(keyframes, features=features,
            threshold=float(semantic["semantic_grouping"]["cosine_similarity_threshold"]), maximum_group_size=int(semantic["semantic_grouping"]["maximum_group_size"]))
        manifest["stages"][current_stage] = stage_record(t, {"group_count": len(groups), "grouping_timing": grouping_timing})
        write_json(output / "keyframe_selection.json", {"records": keyframes, "semantic_groups": groups, "selection_timing": selection_timing, "grouping_timing": grouping_timing, "algorithm": semantic["keyframe_selection"]})
        write_json(output / "run_manifest.json", manifest)

        current_stage = "qwen_medium_captions"
        t = time.perf_counter(); model_view, model_files = create_read_only_model_view(Path(config["qwen_snapshot"]), qwen_view_dir)
        qwen = LocalQwen2VL(model_view=model_view, seed=int(config["seed"]))
        medium_prompt = semantic["medium_prompt"]; medium_captions: list[dict[str, Any]] = []; direct_by_group: dict[str, dict[str, Any]] = {}
        for group in groups:
            representative = next(row for row in keyframes if row["medium_id"] == group["representative_medium_id"])
            raw, timing = qwen.describe_images(image_paths=[Path(representative["selected_keyframe"]["image_path"])], prompt=medium_prompt, max_new_tokens=int(semantic["local_model"]["max_new_tokens_medium"]))
            direct_by_group[group["semantic_group_id"]] = {"raw": raw, "timing": timing}
        for record in sorted(keyframes, key=lambda row: (row["start"], row["medium_id"])):
            group_id = record["semantic_group_id"]; direct = direct_by_group[group_id]
            medium_captions.append({"video_id": video_id, "medium_id": record["medium_id"], "start": record["start"], "end": record["end"], "semantic_group_id": group_id,
                "caption_source": "direct_vlm" if record["medium_id"] == record["semantic_group_representative_medium_id"] else "propagated",
                "caption_source_medium_id": record["semantic_group_representative_medium_id"], "similarity_to_group_representative": record["similarity_to_group_representative"],
                "selected_keyframe": record["selected_keyframe"], "raw_vlm_output": direct["raw"], "cleaned_caption": clean_caption(direct["raw"]), "inference_timing": direct["timing"]})
        manifest["stages"][current_stage] = stage_record(t, {"medium_count": len(medium_captions), "direct_caption_count": len(direct_by_group), "propagated_caption_count": len(medium_captions) - len(direct_by_group), "model_load_sec": qwen.model_load_sec, "model_files": model_files})
        write_json(output / "medium_captions.json", medium_captions); write_json(output / "run_manifest.json", manifest)

        current_stage = "qwen_coarse_captions"
        t = time.perf_counter(); views = hierarchy_views(selection_hierarchy); medium_by_id = {row["medium_id"]: row for row in medium_captions}; coarse_captions: list[dict[str, Any]] = []
        for view in views:
            coarse = view["coarse"]; ordered = [medium_by_id[str(row["node_id"])] for row in view["medium"]]
            descriptions = [row["cleaned_caption"] for row in ordered]
            raw, timing = qwen.summarize_text(ordered_descriptions=descriptions, prompt=semantic["coarse_prompt"], max_new_tokens=int(semantic["local_model"]["max_new_tokens_coarse"]))
            coarse_captions.append({"video_id": video_id, "coarse_id": str(coarse["node_id"]), "start": coarse["start"], "end": coarse["end"], "medium_ids": [row["medium_id"] for row in ordered], "ordered_medium_caption_input": descriptions, "raw_model_output": raw, "cleaned_caption": clean_caption(raw), "inference_timing": timing, "text_only": True})
        manifest["stages"][current_stage] = stage_record(t, {"coarse_count": len(coarse_captions), "caption_success_count": len(coarse_captions)})
        write_json(output / "coarse_captions.json", coarse_captions); write_json(output / "run_manifest.json", manifest)

        current_stage = "chronological_storyline"
        t = time.perf_counter(); ordered_coarse = sorted(coarse_captions, key=lambda row: (row["start"], row["coarse_id"]))
        story_input = [f'{row["start"]:.1f}s–{row["end"]:.1f}s: {row["cleaned_caption"]}' for row in ordered_coarse]
        raw_story, story_timing = qwen.summarize_text(ordered_descriptions=story_input, prompt=semantic["story_prompt"], max_new_tokens=int(semantic["local_model"]["max_new_tokens_story"]))
        overall, storyline = parse_story(raw_story)
        story = {"video_id": video_id, "ordered_coarse_caption_input": story_input, "raw_model_output": raw_story, "overall_story": overall, "high_level_storyline": storyline, "high_level_storyline_source": "parsed_model_output", "inference_timing": story_timing}
        manifest["stages"][current_stage] = stage_record(t, {"story_success": bool(overall), "storyline_item_count": len(storyline)})
        write_json(output / "video_story.json", story); write_json(output / "run_manifest.json", manifest)

        current_stage = "review_render"
        t = time.perf_counter(); source_manifest = {"videos": [{"video_id": video_id, "sample_id": video_id, "downloaded_path": video.as_posix(), "actual_duration_sec": media["duration_sec"]}]}
        stats = {"video_id": video_id, "duration_sec": media["duration_sec"], "fine_count": len(fine_nodes), "medium_count": len(keyframes), "coarse_count": len(coarse_captions), "candidate_frame_count": sum(row["candidate_count"] for row in keyframes), "semantic_group_count": len(groups), "actual_direct_vlm_image_calls": len(direct_by_group), "vlm_calls_avoided": len(keyframes) - len(direct_by_group), "vlm_call_reduction_ratio": (len(keyframes) - len(direct_by_group)) / max(len(keyframes), 1), "selected_keyframe_count": len(keyframes)}
        runtime_stats = {"video_id": video_id, "steady_state_semantic_indexing_sec": manifest["stages"]["qwen_medium_captions"]["elapsed_sec"] + manifest["stages"]["qwen_coarse_captions"]["elapsed_sec"] + manifest["stages"]["chronological_storyline"]["elapsed_sec"]}
        render_info = render_semantic_map(output_path=output / "review.html", source_manifest=source_manifest, hierarchies=[selection_hierarchy], keyframes=keyframes, semantic_groups=groups, medium_captions=medium_captions, coarse_captions=coarse_captions, stories=[story], efficiency=[stats], runtimes=[runtime_stats])
        add_external_image_fallback(output / "review.html", output)
        manifest["stages"][current_stage] = stage_record(t, {"renderer": render_info})
        manifest["status"] = "completed"; manifest["finished_at_epoch_sec"] = time.time(); manifest["total_elapsed_sec"] = time.perf_counter() - started_total
        manifest["peak_rss_bytes"] = max_rss_bytes(); manifest["peak_rss_gib"] = manifest["peak_rss_bytes"] / 2**30; manifest["mps_memory_at_end"] = mps_memory()
        write_json(output / "run_manifest.json", manifest)
        report = ["# EgoPolice pilot16 semantic-map smoke test", "", f"- Status: **{manifest['status']}**", f"- Video: `{video_id}`", f"- Duration: {media['duration_sec']:.3f}s; 1 FPS frames: {len(frames)}", f"- Fine / Medium / Coarse: {len(fine_nodes)} / {len(keyframes)} / {len(coarse_captions)}", f"- Keyframes: {len(keyframes)}; direct Qwen image captions: {len(direct_by_group)}; propagated: {len(medium_captions)-len(direct_by_group)}", f"- Peak RSS: {manifest['peak_rss_gib']:.2f} GiB; device: {manifest['device']}", "", "All frames, candidate images, selected keyframes, feature cache, and Qwen model view are on the external runtime volume. No 25/50/75 voting was added.", ""]
        report += ["## Outputs", ""] + [f"- `{name}`" for name in ("run_manifest.json", "fine_events.json", "medium_events.json", "keyframe_selection.json", "medium_captions.json", "coarse_captions.json", "video_story.json", "review.html")]
        (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps({"status": "completed", "output_root": str(output), "runtime_video_root": str(runtime_video_root), "fine": len(fine_nodes), "medium": len(keyframes), "coarse": len(coarse_captions), "peak_rss_gib": manifest["peak_rss_gib"]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "failed"; manifest["failed_stage"] = current_stage; manifest["error"] = f"{type(exc).__name__}: {exc}"; manifest["finished_at_epoch_sec"] = time.time(); manifest["total_elapsed_sec"] = time.perf_counter() - started_total; manifest["peak_rss_bytes"] = max_rss_bytes(); manifest["mps_memory_at_failure"] = mps_memory()
        write_json(output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
