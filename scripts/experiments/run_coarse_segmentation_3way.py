"""Run the isolated zero-API three-way coarse segmentation experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.coarse_segmentation.comet_style import (  # noqa: E402
    CometStyleConfig,
    segment as comet_segment,
)
from src.experiments.coarse_segmentation.dinov2_features import (  # noqa: E402
    DINOv2FeatureExtractor,
    MODEL_NAME,
)
from src.experiments.coarse_segmentation.kts_style import (  # noqa: E402
    KTSConfig,
    segment as kts_segment,
)
from src.experiments.coarse_segmentation.retrieval_replay import (  # noqa: E402
    pool_clip_regions,
    replay,
)
from src.experiments.coarse_segmentation.schema import (  # noqa: E402
    make_segmentation,
    segmentation_metrics,
)


CONFIG_PATH = ROOT / "config/experiments/coarse_segmentation_3way_v0_1.json"
MANIFEST_PATH = ROOT / "data/manifests/coarse_segmentation_3way_10.json"
PILOT_PATH = ROOT / "data/manifests/egoschema_comparison_pilot.json"
OUT = ROOT / "outputs/experiments/coarse_segmentation_3way_v0_1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def current_segmentation(
    video_id: str, duration: float, timestamps: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = ROOT / "outputs/visual_index" / video_id / "visual_state_regions.json"
    source = load(source_path)
    regions = source["visual_state_regions"]
    boundaries = [float(row["start_time"]) for row in regions[1:]]
    return make_segmentation(
        video_id=video_id,
        method="current",
        video_duration=duration,
        boundaries=boundaries,
        frame_timestamps=timestamps,
    ), {
        "source_path": source_path.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source_path),
        "canonical_region_ids": [row["region_id"] for row in regions],
        "canonical_representative_timestamps": [
            row["representative_frame_timestamp"] for row in regions
        ],
    }


def encode_clip_queries(questions: list[str], device: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import clip
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise
        # openai-clip imports only pkg_resources.packaging. New setuptools no
        # longer ships pkg_resources, so expose the already installed packaging
        # module without changing CLIP weights, preprocessing, or scoring.
        import packaging

        compatibility = types.ModuleType("pkg_resources")
        compatibility.packaging = packaging
        sys.modules["pkg_resources"] = compatibility
        import clip

    started = time.perf_counter()
    model, _ = clip.load(
        "ViT-B/32", device=device, download_root="C:/Users/72977/.cache/clip"
    )
    model.eval()
    load_sec = time.perf_counter() - started
    encode_started = time.perf_counter()
    with torch.inference_mode():
        tokens = clip.tokenize(questions, truncate=True).to(device)
        features = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
    if device == "cuda":
        torch.cuda.synchronize()
    return features.cpu().numpy().astype(np.float32), {
        "model": "OpenAI CLIP ViT-B/32",
        "model_load_sec": load_sec,
        "query_encode_sec": time.perf_counter() - encode_started,
        "query_count": len(questions),
    }


def mean_dict(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    return {
        field: statistics.fmean(float(row[field]) for row in rows if row[field] is not None)
        for field in fields
    }


def aggregates(
    metrics: list[dict[str, Any]],
    retrieval: list[dict[str, Any]],
    group_by_video: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_fields = [
        "number_of_segments",
        "mean_segment_duration",
        "median_segment_duration",
        "max_segment_duration",
        "largest_region_ratio",
        "short_segment_count_lt_2s",
        "short_segment_fraction_lt_2s",
        "short_segment_count_lt_4s",
        "short_segment_fraction_lt_4s",
        "boundaries_per_minute",
        "duration_coefficient_of_variation",
        "coverage_ratio",
    ]
    metric_output: dict[str, Any] = {}
    retrieval_output: dict[str, Any] = {}
    for group in ("all", "problematic", "control"):
        metric_output[group] = {}
        retrieval_output[group] = {}
        for method in ("current", "comet_style_dinov2", "kts_dinov2"):
            selected_metrics = [
                row
                for row in metrics
                if row["method"] == method
                and (group == "all" or group_by_video[row["video_id"]] == group)
            ]
            selected_retrieval = [
                row
                for row in retrieval
                if row["method"] == method
                and (group == "all" or group_by_video[row["video_id"]] == group)
            ]
            metric_output[group][method] = {
                "video_count": len(selected_metrics),
                **mean_dict(selected_metrics, metric_fields),
            }
            retrieval_output[group][method] = {
                "video_count": len(selected_retrieval),
                **mean_dict(
                    selected_retrieval,
                    [
                        "total_raw_selected_duration",
                        "unique_temporal_duration",
                        "refinement_search_space_seconds",
                        "refinement_search_space_ratio",
                        "top_k_overlap_redundancy_seconds",
                    ],
                ),
            }
    return metric_output, retrieval_output


def dependency_versions() -> dict[str, Any]:
    names = ["numpy", "scipy", "torch", "torchvision", "transformers", "Pillow"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    huggingface_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    dino_repository = huggingface_cache / "models--facebook--dinov2-small"
    revision_path = dino_repository / "refs/main"
    revision = revision_path.read_text(encoding="utf-8").strip() if revision_path.is_file() else None
    snapshot = dino_repository / "snapshots" / revision if revision else None
    model_files: dict[str, str] = {}
    if snapshot and snapshot.is_dir():
        for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
            path = snapshot / filename
            if path.is_file():
                model_files[filename] = sha256(path)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dinov2": {
            "model": MODEL_NAME,
            "architecture": "DINOv2 ViT-S/14",
            "huggingface_revision": revision,
            "local_files_only": True,
            "snapshot_file_sha256": model_files,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-dino", action="store_true")
    args = parser.parse_args()
    config = load(CONFIG_PATH)
    manifest = load(MANIFEST_PATH)
    pilot = load(PILOT_PATH)
    questions = {str(row["case_id"]): str(row["question"]) for row in pilot["cases"]}
    if manifest.get("qa_gold_used") or manifest.get("qa_correctness_used"):
        raise RuntimeError("Segmentation manifest is not gold-isolated")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "frozen_config.json", {**config, "source_config_sha256": sha256(CONFIG_PATH)})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DINOv2FeatureExtractor(
        device=device, batch_size=int(config["dinov2"]["batch_size"])
    )
    comet_config = CometStyleConfig(**{
        key: config["comet_style"][key]
        for key in (
            "smoothing_sigma_frames",
            "adaptive_mad_multiplier",
            "minimum_prominence",
            "minimum_segment_duration_sec",
        )
    })
    kts_config = KTSConfig(**{
        key: config["kts"][key]
        for key in (
            "max_change_points",
            "minimum_segment_duration_sec",
            "maximum_segment_duration_sec",
            "penalty_strength",
            "kernel",
        )
    })
    cases = manifest["videos"]
    query_features, clip_query_runtime = encode_clip_queries(
        [questions[str(row["case_id"])] for row in cases], device
    )
    segmentations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_diagnostics = []
    all_metrics = []
    retrieval_rows = []
    frame_grids = []
    indexing_runtime = []
    cache_shas = {}
    run_started = time.perf_counter()
    for case_index, row in enumerate(cases):
        video_id = str(row["video_id"])
        duration = float(row["duration"])
        visual_dir = ROOT / "outputs/visual_index" / video_id
        frame_paths = sorted((visual_dir / "frames_1fps").glob("frame_*.jpg"))
        if len(frame_paths) != 180:
            raise RuntimeError(f"Expected shared 180-frame 1 FPS grid for {video_id}, got {len(frame_paths)}")
        timestamps = np.arange(len(frame_paths), dtype=np.float64) / float(config["sampling_fps"])
        frame_grids.append(
            {
                "video_id": video_id,
                "frame_count": len(frame_paths),
                "actual_timestamps": timestamps.tolist(),
                "missing_frames": [],
                "effective_fps": 1.0,
            }
        )
        clip_frames = np.load(visual_dir / "frame_embeddings.npy")
        if len(clip_frames) != len(frame_paths):
            raise RuntimeError(f"CLIP frame index/grid mismatch for {video_id}")
        cache_dir = OUT / "dinov2_cache" / video_id
        if args.force_dino:
            for path in (cache_dir / "features.npy", cache_dir / "metadata.json"):
                if path.is_file():
                    path.unlink()
        dino, dino_meta = extractor.extract_or_load(
            video_id=video_id,
            frame_paths=frame_paths,
            timestamps=timestamps,
            cache_dir=cache_dir,
        )
        cache_shas[video_id] = dino_meta["features_sha256"]
        current_started = time.perf_counter()
        current, current_diag = current_segmentation(video_id, duration, timestamps)
        current_sec = time.perf_counter() - current_started
        comet_started = time.perf_counter()
        comet_internal = comet_segment(dino, timestamps, comet_config)
        comet = make_segmentation(
            video_id=video_id,
            method="comet_style_dinov2",
            video_duration=duration,
            boundaries=comet_internal["boundaries"],
            frame_timestamps=timestamps,
        )
        comet_sec = time.perf_counter() - comet_started
        kts_started = time.perf_counter()
        kts_internal = kts_segment(dino, timestamps, kts_config)
        kts = make_segmentation(
            video_id=video_id,
            method="kts_dinov2",
            video_duration=duration,
            boundaries=kts_internal["boundaries"],
            frame_timestamps=timestamps,
        )
        kts_sec = time.perf_counter() - kts_started
        for segmentation in (current, comet, kts):
            segmentations[segmentation["method"]].append(segmentation)
            all_metrics.append({**segmentation_metrics(segmentation), "group": row["group"]})
            pooled = pool_clip_regions(segmentation, clip_frames, timestamps)
            retrieval_rows.append(
                {
                    **replay(
                        segmentation,
                        pooled,
                        query_features[case_index],
                        int(config["retrieval_replay"]["top_k"]),
                    ),
                    "case_id": str(row["case_id"]),
                    "question": questions[str(row["case_id"])],
                    "group": row["group"],
                    "retrieval_encoder": "OpenAI CLIP ViT-B/32",
                    "answer_options_used": False,
                    "gold_used": False,
                }
            )
        method_diagnostics.append(
            {
                "video_id": video_id,
                "group": row["group"],
                "shared_dinov2_cache_sha256": dino_meta["features_sha256"],
                "comet_dinov2_cache_sha256": dino_meta["features_sha256"],
                "kts_dinov2_cache_sha256": dino_meta["features_sha256"],
                "current": current_diag,
                "comet": comet_internal,
                "kts": kts_internal,
            }
        )
        indexing_runtime.append(
            {
                "video_id": video_id,
                "dino_cache_hit": dino_meta["cache_hit"],
                "dino_feature_extraction_sec": dino_meta["extraction_sec"],
                "current_segmentation_sec": current_sec,
                "comet_segmentation_sec": comet_sec,
                "kts_segmentation_sec": kts_sec,
                "dino_cache_bytes": (cache_dir / "features.npy").stat().st_size
                + (cache_dir / "metadata.json").stat().st_size,
            }
        )
    for method, filename in (
        ("current", "current_segments.jsonl"),
        ("comet_style_dinov2", "comet_segments.jsonl"),
        ("kts_dinov2", "kts_segments.jsonl"),
    ):
        write_jsonl(OUT / filename, segmentations[method])
    write_jsonl(OUT / "retrieval_replay.jsonl", retrieval_rows)
    write_json(OUT / "per_video_metrics.json", all_metrics)
    write_json(OUT / "method_diagnostics.json", method_diagnostics)
    write_json(OUT / "shared_frame_grids.json", frame_grids)
    group_by_video = {str(row["video_id"]): str(row["group"]) for row in cases}
    aggregate_metrics, aggregate_retrieval = aggregates(
        all_metrics, retrieval_rows, group_by_video
    )
    runtime = {
        "device": device,
        "dinov2_model": MODEL_NAME,
        "dinov2_model_load_count": extractor.load_count,
        "dinov2_model_load_sec": extractor.model_load_sec,
        "dinov2_feature_extraction_sec_total": sum(
            row["dino_feature_extraction_sec"] for row in indexing_runtime
        ),
        "current_segmentation_sec_total": sum(
            row["current_segmentation_sec"] for row in indexing_runtime
        ),
        "comet_segmentation_sec_total": sum(
            row["comet_segmentation_sec"] for row in indexing_runtime
        ),
        "kts_segmentation_sec_total": sum(
            row["kts_segmentation_sec"] for row in indexing_runtime
        ),
        "clip_query_runtime": clip_query_runtime,
        "wall_clock_sec": time.perf_counter() - run_started,
        "per_video": indexing_runtime,
        "dino_cache_size_bytes": sum(row["dino_cache_bytes"] for row in indexing_runtime),
    }
    write_json(OUT / "aggregate_metrics.json", aggregate_metrics)
    write_json(OUT / "aggregate_retrieval.json", aggregate_retrieval)
    write_json(OUT / "runtime_metrics.json", runtime)
    write_json(OUT / "dependency_versions.json", dependency_versions())
    run_manifest = {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(CONFIG_PATH),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "video_count": len(cases),
        "video_ids": [row["video_id"] for row in cases],
        "shared_dinov2_cache_sha256_by_video": cache_shas,
        "method_b_and_c_cache_identity_equal": all(
            row["comet_dinov2_cache_sha256"] == row["kts_dinov2_cache_sha256"]
            for row in method_diagnostics
        ),
        "qa_gold_accessed": False,
        "qa_answers_or_options_accessed": False,
        "paid_api_calls": 0,
        "canonical_outputs_modified": False,
    }
    write_json(OUT / "run_manifest.json", run_manifest)
    readme = [
        "# Coarse segmentation 3-way v0.1",
        "",
        "Isolated comparison of current CLIP boundaries, CoMET-style DINOv2 local-minimum boundaries, and DINOv2+KTS boundaries.",
        "",
        "- DINOv2 cache is shared exactly between Methods B and C.",
        "- Retrieval replay pools the same existing CLIP frame features for every boundary set.",
        "- Questions only are used for retrieval; options, gold, and prior correctness are excluded.",
        "- No method is integrated into the canonical pipeline.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "videos": len(cases),
                "shared_cache": run_manifest["method_b_and_c_cache_identity_equal"],
                "api_calls": 0,
                "output": OUT.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
