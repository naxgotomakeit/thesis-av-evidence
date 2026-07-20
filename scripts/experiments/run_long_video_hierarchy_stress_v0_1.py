"""Run the isolated long-video hierarchy stress/generalization experiment."""

from __future__ import annotations

import hashlib
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
from src.experiments.long_video_hierarchy_stress.experiment import (  # noqa: E402
    extract_frames_1fps,
    hierarchy_metrics,
    inspect_local_videos,
    select_candidates,
    sha256_file,
    validate_cut_nesting,
)
from src.experiments.long_video_hierarchy_stress.reporting import render_hierarchy_review  # noqa: E402


CONFIG_PATH = ROOT / "config/experiments/long_video_hierarchy_stress_v0_1.json"
OUT = ROOT / "outputs/experiments/long_video_hierarchy_stress_v0_1"


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
    result = {}
    for name in ("numpy", "scipy", "opencv-python", "torch", "transformers", "Pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"video_count": len(rows)}
    for level in ("fine", "medium", "coarse"):
        selected = [row[level] for row in rows]
        output[level] = {
            "mean_node_count": statistics.fmean(row["node_count"] for row in selected),
            "mean_duration_sec": statistics.fmean(row["mean_duration_sec"] for row in selected),
            "median_of_video_median_duration_sec": statistics.median(row["median_duration_sec"] for row in selected),
            "mean_max_duration_sec": statistics.fmean(row["max_duration_sec"] for row in selected),
            "mean_largest_node_duration_ratio": statistics.fmean(row["largest_node_duration_ratio"] for row in selected),
            "total_very_short_count_lt_4s": sum(row["very_short_count_lt_4s"] for row in selected),
            "total_long_count_gt_20s": sum(row["long_count_gt_20s"] for row in selected),
            "total_long_count_gt_45s": sum(row["long_count_gt_45s"] for row in selected),
            "total_long_count_gt_90s": sum(row["long_count_gt_90s"] for row in selected),
        }
    output["mean_hierarchy_depth"] = statistics.fmean(row["hierarchy_depth"] for row in rows)
    output["giant_parent_case_count"] = sum(len(row["giant_parent_cases"]) for row in rows)
    output["chaining_risk_case_count"] = sum(len(row["chaining_risk_cases"]) for row in rows)
    return output


def main() -> int:
    started = time.perf_counter()
    config = load_json(CONFIG_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    roots = [Path(value) for value in config["selection"]["local_video_roots"]]
    inventory_started = time.perf_counter()
    ffprobe_path = Path(config["local_ffprobe"])
    ffmpeg_path = Path(config["local_ffmpeg"])
    if not ffprobe_path.is_file() or not ffmpeg_path.is_file():
        raise FileNotFoundError("Configured local ffmpeg/ffprobe binaries are unavailable")
    local_rows = inspect_local_videos(roots, ffprobe_path=ffprobe_path)
    selected, inventory = select_candidates(
        local_rows,
        selected_ids=[str(value) for value in config["selection"]["selected_video_ids"]],
        minimum_long_sec=float(config["selection"]["minimum_long_video_sec"]),
    )
    inventory["inventory_probe_sec"] = time.perf_counter() - inventory_started
    manifest = {
        "schema_version": "long-video-hierarchy-stress-manifest-v1",
        "experiment_id": config["experiment_id"],
        "selection_is_question_independent": True,
        "videos": [],
        "inventory": inventory,
    }
    for row in selected:
        source = Path(row["source_path"])
        manifest["videos"].append({**row, "source_sha256": sha256_file(source)})
    write_json(OUT / "run_manifest.json", manifest)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DINOv2FeatureExtractor(device=device, batch_size=int(config["dinov2"]["batch_size"]))
    comet_config = CometStyleConfig(**config["comet_style"])
    hierarchies: list[dict[str, Any]] = []
    fine_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest["videos"], start=1):
        video_id = str(row["video_id"])
        print(f"[{index}/{len(selected)}] {video_id}: frames/DINO/CoMET/Safe Merge", flush=True)
        per_video_started = time.perf_counter()
        frames, timestamps, frame_meta = extract_frames_1fps(
            Path(row["source_path"]), OUT / "assets/frames" / video_id,
            duration_sec=float(row["duration_sec"]), jpeg_quality=int(config["jpeg_quality"]),
            ffmpeg_path=ffmpeg_path,
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
            video_duration=float(row["duration_sec"]), boundaries=comet_internal["boundaries"],
            frame_timestamps=timestamps,
        )
        comet_sec = time.perf_counter() - comet_started
        boundary_records = build_boundary_records(fine_segmentation["segments"], comet_internal)
        frame_paths = [path.relative_to(ROOT).as_posix() for path in frames]
        fine_nodes = build_fine_nodes(
            segments=fine_segmentation["segments"], features=dino, timestamps=timestamps,
            frame_paths=frame_paths, boundary_records=boundary_records,
            representative_fractions=tuple(config["hierarchy"]["representative_fractions"]),
            include_medoid=bool(config["hierarchy"]["include_dinov2_medoid"]),
        )
        hierarchy_started = time.perf_counter()
        hierarchy, decisions = build_safe_hierarchy(
            video_id=video_id, video_duration=float(row["duration_sec"]),
            fine_nodes=fine_nodes, boundary_records=boundary_records,
            timestamps=timestamps, frame_paths=frame_paths, features=dino,
            medium_fraction=float(config["hierarchy"]["medium_reference_fraction"]),
            coarse_fraction=float(config["hierarchy"]["coarse_reference_fraction"]),
            representative_fractions=tuple(config["hierarchy"]["representative_fractions"]),
            include_medoid=bool(config["hierarchy"]["include_dinov2_medoid"]),
        )
        hierarchy_sec = time.perf_counter() - hierarchy_started
        nesting = validate_cut_nesting(hierarchy)
        if not hierarchy["invariants"]["valid"] or not nesting["valid"]:
            raise RuntimeError(f"Hierarchy invariant failed for {video_id}")
        metric = {
            "video_id": video_id,
            **hierarchy_metrics(
                hierarchy,
                chaining_ratio=float(config["diagnostics"]["chaining_child_imbalance_ratio"]),
            ),
            "cut_nesting": nesting,
        }
        fine_records.append(fine_segmentation)
        hierarchies.append(hierarchy)
        diagnostics.append({"video_id": video_id, "comet": comet_internal, "boundaries": boundary_records})
        merge_rows.extend(decisions)
        metrics.append(metric)
        runtime_rows.append(
            {
                "video_id": video_id,
                "frame_extraction_cache_hit": frame_meta["cache_hit"],
                "frame_extraction_sec": frame_meta["extraction_sec"],
                "dinov2_cache_hit": dino_meta["cache_hit"],
                "dinov2_feature_extraction_sec": dino_meta["extraction_sec"],
                "dinov2_cache_original_extraction_sec": persisted_dino_meta["extraction_sec"],
                "comet_segmentation_sec": comet_sec,
                "safe_hierarchy_construction_sec": hierarchy_sec,
                "video_wall_sec": time.perf_counter() - per_video_started,
                "feature_cache_bytes": (OUT / "feature_cache" / video_id / "features.npy").stat().st_size,
            }
        )
    aggregate_metrics = aggregate(metrics)
    write_jsonl(OUT / "fine_event_segments.jsonl", fine_records)
    write_jsonl(OUT / "safe_merge_hierarchies.jsonl", hierarchies)
    write_jsonl(OUT / "merge_trace.jsonl", merge_rows)
    write_json(OUT / "per_video_metrics.json", metrics)
    write_json(OUT / "aggregate_metrics.json", aggregate_metrics)
    write_json(OUT / "segmentation_diagnostics.json", diagnostics)
    frozen = {
        **config,
        "config_sha256": sha256_file(CONFIG_PATH),
        "git_branch": git_value("branch", "--show-current"),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_at_start": git_value("status", "--short"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": package_versions(),
        "external_api_calls": 0,
    }
    write_json(OUT / "frozen_config.json", frozen)
    runtime = {
        "model_load_count": extractor.load_count,
        "dinov2_model_load_sec": extractor.model_load_sec,
        "frame_extraction_sec_total": sum(row["frame_extraction_sec"] for row in runtime_rows),
        "dinov2_feature_extraction_sec_total": sum(row["dinov2_feature_extraction_sec"] for row in runtime_rows),
        "dinov2_cache_original_extraction_sec_total": sum(
            row["dinov2_cache_original_extraction_sec"] for row in runtime_rows
        ),
        "comet_segmentation_sec_total": sum(row["comet_segmentation_sec"] for row in runtime_rows),
        "safe_hierarchy_construction_sec_total": sum(row["safe_hierarchy_construction_sec"] for row in runtime_rows),
        "per_video": runtime_rows,
        "wall_sec_before_html": time.perf_counter() - started,
        "offline_reusable": True,
        "external_api_calls": 0,
    }
    write_json(OUT / "runtime_metrics.json", runtime)
    html_validation = render_hierarchy_review(
        output_path=OUT / "hierarchy_review.html", root=ROOT, manifest=manifest,
        hierarchies=hierarchies, metrics=metrics, inventory=inventory,
    )
    write_json(OUT / "html_validation.json", html_validation)
    (OUT / "README.md").write_text(
        "# Long-video hierarchy stress v0.1\n\n"
        "No local video longer than 180 seconds was available. This output is therefore a held-out "
        "180-second hierarchy generalization check, not the requested 5–20 minute duration stress test.\n\n"
        "It reuses the frozen CoMET-style configuration and Boundary-aware Safe Merge implementation. "
        "Questions, retrieval, gold, options, APIs, and the canonical pipeline are absent.\n",
        encoding="utf-8",
    )
    print(json.dumps({"videos": len(hierarchies), "inventory": inventory, "html": str(OUT / 'hierarchy_review.html'), "api_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
