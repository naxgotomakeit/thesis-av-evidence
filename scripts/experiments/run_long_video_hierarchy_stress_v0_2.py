"""Download four explicit EgoPolice intervals and build the frozen visual hierarchy."""

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

from src.experiments.coarse_segmentation.comet_style import CometStyleConfig, segment as comet_segment  # noqa: E402
from src.experiments.coarse_segmentation.dinov2_features import DINOv2FeatureExtractor  # noqa: E402
from src.experiments.coarse_segmentation.schema import make_segmentation  # noqa: E402
from src.experiments.fine_to_coarse_hierarchy.hierarchy import (  # noqa: E402
    build_boundary_records,
    build_fine_nodes,
    build_safe_hierarchy,
)
from src.experiments.long_video_hierarchy_stress.experiment import extract_frames_1fps  # noqa: E402
from src.experiments.long_video_hierarchy_stress_v0_2.experiment import (  # noqa: E402
    candidate_ranking,
    decode_sanity,
    download_section,
    hierarchy_metrics,
    parse_video_index,
    probe_media,
    sanitize_sample_id,
    select_primary_candidates,
    sha256_file,
    validate_cut_nesting,
)
from src.experiments.long_video_hierarchy_stress_v0_2.reporting import render_hierarchy_review  # noqa: E402


CONFIG_PATH = ROOT / "config/experiments/long_video_hierarchy_stress_v0_2.json"
OUT = ROOT / "outputs/experiments/long_video_hierarchy_stress_v0_2"
THESIS_PYTHON = Path("C:/Users/72977/miniforge3/envs/thesis_av/python.exe")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "torch", "transformers", "Pillow"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def print_selection(rows: list[dict[str, Any]]) -> None:
    print("\nFrozen pre-download selection (metadata only)", flush=True)
    print("sample_id | source_url | start | end | duration | reason", flush=True)
    for row in rows:
        print(
            f'{row["sample_id"]} | {row["source_url"]} | {row["start_sec"]:.1f} | '
            f'{row["end_sec"]:.1f} | {row["target_clip_duration_sec"]:.1f}s | '
            f'{row["selection_reason"]}',
            flush=True,
        )


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"video_count": len(rows)}
    for level in ("fine", "medium", "coarse"):
        values = [row[level] for row in rows]
        output[level] = {
            "mean_node_count": statistics.fmean(row["node_count"] for row in values),
            "total_node_count": sum(row["node_count"] for row in values),
            "mean_segment_duration_sec": statistics.fmean(row["mean_duration_sec"] for row in values),
            "median_of_video_medians_sec": statistics.median(row["median_duration_sec"] for row in values),
            "mean_max_duration_sec": statistics.fmean(row["max_duration_sec"] for row in values),
            "maximum_duration_sec": max(row["max_duration_sec"] for row in values),
            "mean_largest_node_ratio": statistics.fmean(row["largest_node_duration_ratio"] for row in values),
            **{
                key: sum(row[key] for row in values)
                for key in (
                    "node_count_lt_4s", "node_count_gt_20s", "node_count_gt_45s",
                    "node_count_gt_90s", "node_count_gt_180s",
                )
            },
        }
    output.update(
        {
            "mean_hierarchy_depth": statistics.fmean(row["hierarchy_depth"] for row in rows),
            "giant_parent_case_count": sum(len(row["giant_parent_cases"]) for row in rows),
            "chaining_risk_case_count": sum(len(row["chaining_risk_cases"]) for row in rows),
            "fine_preservation_rate": min(row["fine_preservation_rate"] for row in rows),
        }
    )
    return output


def acquire_clips(config: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = select_primary_candidates(
        rows,
        targets=[float(value) for value in config["target_duration_sec"]],
        preferred_prefixes=list(config["preferred_sample_prefixes"]),
    )
    print_selection(selected)
    failures: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    used_urls: set[str] = set()
    video_dir = OUT / config["download"]["video_output_subdir"]
    for target_index, target in enumerate(config["target_duration_sec"], start=1):
        ranked = candidate_ranking(
            rows, target_sec=float(target), preferred_prefixes=list(config["preferred_sample_prefixes"])
        )
        attempts = 0
        chosen: dict[str, Any] | None = None
        for candidate in ranked:
            if candidate["source_url"] in used_urls:
                continue
            attempts += 1
            if attempts > int(config["download"]["maximum_attempts_per_target"]):
                break
            candidate = {
                **candidate,
                "duration_class_target_sec": float(target),
                "absolute_target_error_sec": abs(float(candidate["target_clip_duration_sec"]) - float(target)),
                "selection_reason": (
                    "closest downloadable explicit interval to target among preferred official police channels; "
                    "distinct source URL; stable sample_id tie-break"
                ),
            }
            print(f'[{target_index}/4] Download attempt {attempts}: {candidate["sample_id"]}', flush=True)
            try:
                download = download_section(
                    candidate,
                    python_path=THESIS_PYTHON,
                    yt_dlp_zipapp=ROOT / config["download"]["yt_dlp_zipapp"],
                    ffmpeg_path=Path(config["local_ffmpeg"]),
                    output_dir=video_dir,
                    format_selector=str(config["download"]["format"]),
                )
                probe = probe_media(download["path"], ffprobe_path=Path(config["local_ffprobe"]))
                sanity = decode_sanity(
                    download["path"], ffmpeg_path=Path(config["local_ffmpeg"]),
                    duration_sec=float(probe["actual_duration_sec"]),
                )
                if not sanity["passed"]:
                    raise RuntimeError(f'Decode sanity failed: {sanity["stderr_tail"]}')
                expected = float(candidate["target_clip_duration_sec"])
                tolerance = max(5.0, expected * 0.015)
                if abs(float(probe["actual_duration_sec"]) - expected) > tolerance:
                    raise RuntimeError(
                        f'Section duration mismatch: actual={probe["actual_duration_sec"]}, expected={expected}, tolerance={tolerance}'
                    )
                chosen = {
                    **candidate,
                    "video_id": sanitize_sample_id(str(candidate["sample_id"])),
                    "downloaded_path": download["path"].relative_to(ROOT).as_posix(),
                    "downloaded_sha256": sha256_file(download["path"]),
                    "downloaded_bytes": download["path"].stat().st_size,
                    "download_sec": download["download_sec"],
                    "download_cache_hit": download["cache_hit"],
                    "download_mode": download["download_mode"],
                    "yt_dlp_stdout_tail": download["stdout_tail"],
                    "yt_dlp_stderr_tail": download["stderr_tail"],
                    **probe,
                    "decode_sanity": sanity,
                    "download_attempt_rank": attempts,
                    "replacement_used": attempts > 1,
                }
                break
            except Exception as error:  # link failure is expected and must not abort all targets
                failures.append(
                    {
                        "target_duration_sec": float(target),
                        "sample_id": candidate["sample_id"],
                        "source_url": candidate["source_url"],
                        "attempt_rank": attempts,
                        "error": str(error),
                    }
                )
                print(f'  unavailable/invalid: {str(error)[-400:]}', flush=True)
        if chosen is None:
            raise RuntimeError(f"Unable to acquire a valid clip for target {target}s after {attempts} attempts")
        completed.append(chosen)
        used_urls.add(str(chosen["source_url"]))
    return completed, failures


def main() -> int:
    experiment_started = time.perf_counter()
    config = load_json(CONFIG_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    video_index = Path(config["video_index_path"])
    if not video_index.is_file():
        raise FileNotFoundError(video_index)
    for executable in (Path(config["local_ffmpeg"]), Path(config["local_ffprobe"]), THESIS_PYTHON):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    yt_dlp = ROOT / config["download"]["yt_dlp_zipapp"]
    if not yt_dlp.is_file():
        raise FileNotFoundError(yt_dlp)

    index_rows = parse_video_index(video_index)
    videos, download_failures = acquire_clips(config, index_rows)
    write_json(OUT / "selected_videos.json", {"videos": videos, "unavailable_attempts": download_failures})
    write_json(OUT / "download_failures.json", download_failures)

    manifest: dict[str, Any] = {
        "schema_version": "long-video-hierarchy-stress-manifest-v2",
        "experiment_id": config["experiment_id"],
        "video_index_path": video_index.as_posix(),
        "video_index_sha256": sha256_file(video_index),
        "selection_policy": config["selection_policy"],
        "selection_is_question_independent": True,
        "explicit_interval_rows_available": len(index_rows),
        "videos": videos,
        "unavailable_attempt_count": len(download_failures),
        "external_api_calls": 0,
    }
    write_json(OUT / "run_manifest.json", manifest)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_started = time.perf_counter()
    extractor = DINOv2FeatureExtractor(device=device, batch_size=int(config["dinov2"]["batch_size"]))
    model_load_wall = time.perf_counter() - model_started
    comet_config = CometStyleConfig(**config["comet_style"])
    hierarchies: list[dict[str, Any]] = []
    fine_segmentations: list[dict[str, Any]] = []
    merge_trace: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for index, video in enumerate(videos, start=1):
        video_id = str(video["video_id"])
        print(f"[{index}/4] {video_id}: 1 FPS decode → DINOv2 → CoMET Fine → Safe Merge", flush=True)
        video_started = time.perf_counter()
        frames, timestamps, frame_meta = extract_frames_1fps(
            ROOT / video["downloaded_path"], OUT / "assets/frames" / video_id,
            duration_sec=float(video["actual_duration_sec"]), jpeg_quality=int(config["jpeg_quality"]),
            ffmpeg_path=Path(config["local_ffmpeg"]),
        )
        dino, dino_meta = extractor.extract_or_load(
            video_id=video_id, frame_paths=frames, timestamps=timestamps,
            cache_dir=OUT / "feature_cache" / video_id,
        )
        persisted_dino_meta = load_json(OUT / "feature_cache" / video_id / "metadata.json")
        comet_started = time.perf_counter()
        comet_internal = comet_segment(dino, timestamps, comet_config)
        fine_segmentation = make_segmentation(
            video_id=video_id, method="comet_style_dinov2",
            video_duration=float(video["actual_duration_sec"]), boundaries=comet_internal["boundaries"],
            frame_timestamps=timestamps,
        )
        fine_segmentation_sec = time.perf_counter() - comet_started
        boundary_records = build_boundary_records(fine_segmentation["segments"], comet_internal)
        root_relative_frames = [path.relative_to(ROOT).as_posix() for path in frames]
        fine_nodes = build_fine_nodes(
            segments=fine_segmentation["segments"], features=dino, timestamps=timestamps,
            frame_paths=root_relative_frames, boundary_records=boundary_records,
            representative_fractions=tuple(config["hierarchy"]["representative_fractions"]),
            include_medoid=bool(config["hierarchy"]["include_dinov2_medoid"]),
        )
        hierarchy_started = time.perf_counter()
        hierarchy, decisions = build_safe_hierarchy(
            video_id=video_id, video_duration=float(video["actual_duration_sec"]),
            fine_nodes=fine_nodes, boundary_records=boundary_records, timestamps=timestamps,
            frame_paths=root_relative_frames, features=dino,
            medium_fraction=float(config["hierarchy"]["medium_reference_fraction"]),
            coarse_fraction=float(config["hierarchy"]["coarse_reference_fraction"]),
            representative_fractions=tuple(config["hierarchy"]["representative_fractions"]),
            include_medoid=bool(config["hierarchy"]["include_dinov2_medoid"]),
        )
        hierarchy_sec = time.perf_counter() - hierarchy_started
        nesting = validate_cut_nesting(hierarchy)
        if not hierarchy["invariants"]["valid"] or not nesting["valid"]:
            raise RuntimeError(f"Hierarchy invariant failure for {video_id}")
        metric = {
            "video_id": video_id,
            "sample_id": video["sample_id"],
            "duration_sec": float(video["actual_duration_sec"]),
            **hierarchy_metrics(
                hierarchy,
                thresholds=[float(value) for value in config["diagnostics"]["duration_thresholds_sec"]],
                giant_ratio=float(config["diagnostics"]["giant_parent_ratio"]),
                chaining_ratio=float(config["diagnostics"]["chaining_child_imbalance_ratio"]),
            ),
            "cut_nesting": nesting,
        }
        frame_cache_bytes = sum(path.stat().st_size for path in frames)
        runtime_rows.append(
            {
                "video_id": video_id,
                "download_and_trim_sec": float(video["download_sec"]),
                "download_cache_hit": bool(video["download_cache_hit"]),
                "video_decode_frame_sampling_sec": float(frame_meta["extraction_sec"]),
                "frame_sampling_cache_hit": bool(frame_meta["cache_hit"]),
                "sampled_frame_count": len(frames),
                "dinov2_feature_extraction_sec": float(dino_meta["extraction_sec"]),
                "dinov2_cache_hit": bool(dino_meta["cache_hit"]),
                "dinov2_original_extraction_sec": float(persisted_dino_meta["extraction_sec"]),
                "fine_segmentation_sec": fine_segmentation_sec,
                "safe_hierarchy_construction_sec": hierarchy_sec,
                "video_processing_wall_sec": time.perf_counter() - video_started,
                "downloaded_video_bytes": int(video["downloaded_bytes"]),
                "sampled_frames_bytes": frame_cache_bytes,
                "feature_cache_bytes": (OUT / "feature_cache" / video_id / "features.npy").stat().st_size,
            }
        )
        fine_segmentations.append(fine_segmentation)
        hierarchies.append(hierarchy)
        merge_trace.extend(decisions)
        diagnostics.append(
            {"video_id": video_id, "comet_internal": comet_internal, "boundary_records": boundary_records}
        )
        metrics.append(metric)
        write_jsonl(OUT / "fine_event_segments.jsonl", fine_segmentations)
        write_jsonl(OUT / "safe_merge_hierarchies.jsonl", hierarchies)
        write_jsonl(OUT / "merge_trace.jsonl", merge_trace)
        write_json(OUT / "per_video_metrics.json", metrics)
        write_json(OUT / "runtime_metrics.partial.json", {"per_video": runtime_rows})

    aggregate = aggregate_metrics(metrics)
    write_json(OUT / "per_video_metrics.json", metrics)
    write_json(OUT / "aggregate_metrics.json", aggregate)
    write_json(OUT / "segmentation_diagnostics.json", diagnostics)

    runtime: dict[str, Any] = {
        "one_time_model_load_sec": model_load_wall,
        "extractor_reported_model_load_sec": extractor.model_load_sec,
        "model_load_count": extractor.load_count,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "per_video": runtime_rows,
        "totals": {
            "download_and_trim_sec": sum(row["download_and_trim_sec"] for row in runtime_rows),
            "video_decode_frame_sampling_sec": sum(row["video_decode_frame_sampling_sec"] for row in runtime_rows),
            "cold_dinov2_feature_extraction_sec": sum(row["dinov2_original_extraction_sec"] for row in runtime_rows),
            "fine_segmentation_sec": sum(row["fine_segmentation_sec"] for row in runtime_rows),
            "safe_hierarchy_construction_sec": sum(row["safe_hierarchy_construction_sec"] for row in runtime_rows),
        },
        "external_api_calls": 0,
    }
    html_started = time.perf_counter()
    html_validation = render_hierarchy_review(
        output_path=OUT / "hierarchy_review.html", root=ROOT, manifest=manifest,
        hierarchies=hierarchies, metrics=metrics, runtimes=runtime_rows,
        previous_baseline=config["previous_three_minute_baseline"],
    )
    runtime["html_thumbnail_generation_sec"] = time.perf_counter() - html_started
    runtime["experiment_wall_sec"] = time.perf_counter() - experiment_started
    write_json(OUT / "runtime_metrics.json", runtime)
    write_json(OUT / "html_validation.json", html_validation)

    frozen = {
        **config,
        "config_sha256": sha256_file(CONFIG_PATH),
        "git_branch": git_value("branch", "--show-current"),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_at_execution": git_value("status", "--short"),
        "video_index_sha256": manifest["video_index_sha256"],
        "device": device,
        "gpu": runtime["gpu"],
        "package_versions": package_versions(),
        "external_api_calls": 0,
        "canonical_pipeline_modified": False,
    }
    write_json(OUT / "frozen_config.json", frozen)
    (OUT / "README.md").write_text(
        "# Genuine EgoPolice long-video hierarchy stress v0.2\n\n"
        "Four explicit intervals were selected deterministically from the user-provided EgoPolice video index, "
        "downloaded as interval-only clips, and processed cold through the existing frozen CoMET-style Fine "
        "segmentation and Boundary-aware Safe Merge hierarchy. Medium and Coarse are reference operating cuts, "
        "not claimed natural semantic levels.\n\n"
        "No question, retrieval, Planner, gold, options, VLM summary, external model API, or canonical pipeline "
        "modification is present. Downloaded videos and feature/frame caches are experiment-local ignored data.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "completed_videos": len(videos),
                "html": str(OUT / "hierarchy_review.html"),
                "html_validation": html_validation,
                "aggregate_metrics": aggregate,
                "api_calls": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
