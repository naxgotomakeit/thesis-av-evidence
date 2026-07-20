"""Run the zero-API KTS-gated local CoMET structural comparison."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_coarse_segmentation_3way import encode_clip_queries  # noqa: E402
from src.experiments.coarse_segmentation.comet_style import CometStyleConfig  # noqa: E402
from src.experiments.coarse_segmentation.retrieval_replay import (  # noqa: E402
    pool_clip_regions,
    replay,
    union_duration,
)
from src.experiments.hierarchical_refinement.pipeline import (  # noqa: E402
    local_comet_segmentation,
    merge_selected_kts_units,
    midpoint_containment,
    overlap_duration,
)


EXPERIMENT_CONFIG = ROOT / "config/experiments/hierarchical_refinement_comparison_v0_1.json"
PREVIOUS = ROOT / "outputs/experiments/coarse_segmentation_3way_v0_1"
MANIFEST = ROOT / "data/manifests/coarse_segmentation_3way_10.json"
OUT = ROOT / "outputs/experiments/hierarchical_refinement_comparison_v0_1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_fields(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    return {
        field: statistics.fmean(float(row[field]) for row in rows)
        for field in fields
    }


def selected_intervals(result: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(row["start"]), float(row["end"])) for row in result["selected_top_k"]]


def scoring_regression(
    fresh: dict[str, Any], historical: dict[str, Any], tolerance: float = 2e-6
) -> dict[str, Any]:
    fresh_rows, old_rows = fresh["ranked_regions"], historical["ranked_regions"]
    ids_equal = [row["segment_id"] for row in fresh_rows] == [
        row["segment_id"] for row in old_rows
    ]
    max_score_error = max(
        (abs(float(left["score"]) - float(right["score"])) for left, right in zip(fresh_rows, old_rows)),
        default=0.0,
    )
    selected_equal = [row["segment_id"] for row in fresh["selected_top_k"]] == [
        row["segment_id"] for row in historical["selected_top_k"]
    ]
    return {
        "ranking_ids_equal": ids_equal,
        "selected_ids_equal": selected_equal,
        "max_abs_score_error": max_score_error,
        "score_within_tolerance": max_score_error <= tolerance,
        "tolerance": tolerance,
        "passed": ids_equal and selected_equal and max_score_error <= tolerance,
    }


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty experiment output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    experiment = load(EXPERIMENT_CONFIG)
    manifest = load(MANIFEST)
    previous_config = load(PREVIOUS / "frozen_config.json")
    previous_run = load(PREVIOUS / "run_manifest.json")
    previous_runtime = load(PREVIOUS / "runtime_metrics.json")
    if sha256(MANIFEST) != previous_run["manifest_sha256"]:
        raise RuntimeError("Frozen 10-video manifest no longer matches previous experiment")
    if experiment["gold_accessed"] or experiment["qa_correctness_accessed"]:
        raise RuntimeError("Experiment configuration violates gold isolation")
    kts_records = {row["video_id"]: row for row in load_jsonl(PREVIOUS / "kts_segments.jsonl")}
    comet_records = {
        row["video_id"]: row for row in load_jsonl(PREVIOUS / "comet_segments.jsonl")
    }
    historical_retrieval = {
        (row["video_id"], row["method"]): row
        for row in load_jsonl(PREVIOUS / "retrieval_replay.jsonl")
    }
    cases = manifest["videos"]
    questions = [historical_retrieval[(row["video_id"], "kts_dinov2")]["question"] for row in cases]
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    query_features, query_runtime = encode_clip_queries(questions, device)
    comet_config = CometStyleConfig(
        **{
            key: previous_config["comet_style"][key]
            for key in (
                "smoothing_sigma_frames",
                "adaptive_mad_multiplier",
                "minimum_prominence",
                "minimum_segment_duration_sec",
            )
        }
    )
    primary_m = int(experiment["primary_kts_top_m"])
    final_top_k = int(experiment["final_top_k"])
    kts_only_results: list[dict[str, Any]] = []
    full_comet_results: list[dict[str, Any]] = []
    hybrid_results: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    containment_rows: list[dict[str, Any]] = []
    score_regressions: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cache_checks: dict[str, str] = {}
    run_started = time.perf_counter()
    for case_index, case in enumerate(cases):
        video_id = str(case["video_id"])
        duration = float(case["duration"])
        timestamps = np.arange(180, dtype=np.float64)
        visual_dir = ROOT / "outputs/visual_index" / video_id
        clip_frames = np.load(visual_dir / "frame_embeddings.npy")
        dino_path = PREVIOUS / "dinov2_cache" / video_id / "features.npy"
        dino_meta = load(dino_path.with_name("metadata.json"))
        if sha256(dino_path) != dino_meta["features_sha256"]:
            raise RuntimeError(f"DINO cache identity mismatch: {video_id}")
        if previous_run["shared_dinov2_cache_sha256_by_video"][video_id] != dino_meta["features_sha256"]:
            raise RuntimeError(f"DINO cache differs from previous run manifest: {video_id}")
        dino = np.load(dino_path, mmap_mode="r")
        cache_checks[video_id] = dino_meta["features_sha256"]
        kts_record, full_record = kts_records[video_id], comet_records[video_id]
        kts_pool_started = time.perf_counter()
        kts_pool = pool_clip_regions(kts_record, clip_frames, timestamps)
        kts_pool_sec = time.perf_counter() - kts_pool_started
        kts_score_started = time.perf_counter()
        kts_result = replay(kts_record, kts_pool, query_features[case_index], primary_m)
        kts_score_sec = time.perf_counter() - kts_score_started
        full_pool_started = time.perf_counter()
        full_pool = pool_clip_regions(full_record, clip_frames, timestamps)
        full_pool_sec = time.perf_counter() - full_pool_started
        full_score_started = time.perf_counter()
        full_result = replay(full_record, full_pool, query_features[case_index], final_top_k)
        full_score_sec = time.perf_counter() - full_score_started
        for method, fresh in (("kts_dinov2", kts_result), ("comet_style_dinov2", full_result)):
            check = scoring_regression(fresh, historical_retrieval[(video_id, method)])
            score_regressions.append({"video_id": video_id, "method": method, **check})
            if not check["passed"]:
                raise RuntimeError(f"Shared CLIP scoring regression mismatch: {video_id}/{method}: {check}")
        zone_started = time.perf_counter()
        zones = merge_selected_kts_units(kts_record["segments"], kts_result["selected_top_k"])
        zone_sec = time.perf_counter() - zone_started
        local_seg_started = time.perf_counter()
        local_record, local_diagnostics = local_comet_segmentation(
            video_id=video_id,
            video_duration=duration,
            dino_features=dino,
            frame_timestamps=timestamps,
            candidate_zones=zones,
            config=comet_config,
        )
        local_seg_sec = time.perf_counter() - local_seg_started
        local_pool_started = time.perf_counter()
        local_pool = pool_clip_regions(local_record, clip_frames, timestamps)
        local_pool_sec = time.perf_counter() - local_pool_started
        local_score_started = time.perf_counter()
        local_result = replay(local_record, local_pool, query_features[case_index], final_top_k)
        local_score_sec = time.perf_counter() - local_score_started
        zone_intervals = [(float(row["start"]), float(row["end"])) for row in zones]
        zone_duration = union_duration(zone_intervals)
        full_top = full_result["selected_top_k"]
        contained = midpoint_containment(full_top, zones)
        overlap = overlap_duration(selected_intervals(full_result), selected_intervals(local_result))
        full_top_duration = union_duration(selected_intervals(full_result))
        containment = {
            "case_id": str(case["case_id"]),
            "video_id": video_id,
            "group": case["group"],
            "diagnostic_proxy_only": True,
            "full_comet_top1_contained": bool(contained[0]) if contained else False,
            "full_comet_top3_containment_rate": sum(contained) / len(contained) if contained else 0.0,
            "full_comet_top3_midpoint_containment": contained,
            "temporal_overlap_with_full_comet_top3_seconds": overlap,
            "temporal_overlap_with_full_comet_top3_ratio": overlap / full_top_duration if full_top_duration else None,
        }
        containment_rows.append(containment)
        common = {
            "case_id": str(case["case_id"]),
            "video_id": video_id,
            "question": questions[case_index],
            "group": case["group"],
            "video_duration": duration,
            "answer_options_used": False,
            "gold_used": False,
            "qa_correctness_used": False,
        }
        kts_only_results.append(
            {
                **common,
                "method": "kts_only",
                "total_kts_units": len(kts_record["segments"]),
                "number_scored": len(kts_result["ranked_regions"]),
                "ranking": kts_result["ranked_regions"],
                "selected_top_k": kts_result["selected_top_k"],
                "selected_unique_duration": kts_result["unique_temporal_duration"],
                "selected_coverage_ratio": kts_result["refinement_search_space_ratio"],
            }
        )
        full_comet_results.append(
            {
                **common,
                "method": "full_comet",
                "total_full_comet_segments": len(full_record["segments"]),
                "number_scored": len(full_result["ranked_regions"]),
                "ranking": full_result["ranked_regions"],
                "selected_top_k": full_result["selected_top_k"],
                "selected_unique_duration": full_result["unique_temporal_duration"],
                "selected_coverage_ratio": full_result["refinement_search_space_ratio"],
                "refinement_input_seconds": duration,
                "refinement_input_ratio": 1.0,
                "source": "reused full-video CoMET segmentation from coarse_segmentation_3way_v0_1",
            }
        )
        hybrid_results.append(
            {
                **common,
                "method": "kts_local_comet",
                "kts_units_scored": len(kts_result["ranked_regions"]),
                "selected_kts_top_m": kts_result["selected_top_k"],
                "candidate_zones": zones,
                "candidate_zone_count": len(zones),
                "refinement_input_seconds": zone_duration,
                "refinement_input_ratio": zone_duration / duration,
                "local_comet_segments": local_record["segments"],
                "local_comet_diagnostics": local_diagnostics,
                "local_comet_candidate_count": len(local_record["segments"]),
                "local_comet_segments_scored": len(local_result["ranked_regions"]),
                "local_ranking": local_result["ranked_regions"],
                "selected_top_k": local_result["selected_top_k"],
                "selected_unique_duration": local_result["unique_temporal_duration"],
                "selected_coverage_ratio": local_result["refinement_search_space_ratio"],
                "fine_candidate_reduction": 1.0 - len(local_record["segments"]) / len(full_record["segments"]),
                "fine_temporal_processing_reduction": 1.0 - zone_duration / duration,
                "potential_filter_candidates_avoided": len(full_record["segments"]) - len(local_record["segments"]),
                "containment_proxy": containment,
                "dino_cache_sha256": dino_meta["features_sha256"],
            }
        )
        metric = {
            "case_id": str(case["case_id"]),
            "video_id": video_id,
            "group": case["group"],
            "kts_candidates_scored": len(kts_result["ranked_regions"]),
            "kts_selected_duration": kts_result["unique_temporal_duration"],
            "kts_selected_coverage_ratio": kts_result["refinement_search_space_ratio"],
            "full_comet_candidates_scored": len(full_result["ranked_regions"]),
            "full_comet_selected_duration": full_result["unique_temporal_duration"],
            "full_comet_selected_coverage_ratio": full_result["refinement_search_space_ratio"],
            "candidate_zone_count": len(zones),
            "refinement_input_seconds": zone_duration,
            "refinement_input_ratio": zone_duration / duration,
            "local_comet_candidates_generated": len(local_record["segments"]),
            "local_comet_candidates_scored": len(local_result["ranked_regions"]),
            "local_final_duration": local_result["unique_temporal_duration"],
            "local_final_coverage_ratio": local_result["refinement_search_space_ratio"],
            "fine_candidate_reduction": 1.0 - len(local_record["segments"]) / len(full_record["segments"]),
            "fine_temporal_processing_reduction": 1.0 - zone_duration / duration,
            "potential_filter_candidates_avoided": len(full_record["segments"]) - len(local_record["segments"]),
            "full_comet_top1_containment": float(containment["full_comet_top1_contained"]),
            "full_comet_top3_containment_rate": containment["full_comet_top3_containment_rate"],
            "temporal_overlap_with_full_comet_top3_seconds": overlap,
            "temporal_overlap_with_full_comet_top3_ratio": containment["temporal_overlap_with_full_comet_top3_ratio"],
        }
        metrics.append(metric)
        runtime_rows.append(
            {
                "video_id": video_id,
                "kts_region_pooling_sec": kts_pool_sec,
                "kts_scoring_sec": kts_score_sec,
                "full_comet_region_pooling_sec": full_pool_sec,
                "full_comet_scoring_sec": full_score_sec,
                "candidate_zone_construction_sec": zone_sec,
                "local_comet_segmentation_sec": local_seg_sec,
                "local_comet_region_pooling_sec": local_pool_sec,
                "local_comet_scoring_sec": local_score_sec,
            }
        )
        for top_m in experiment["sensitivity_top_m"]:
            selected = kts_result["ranked_regions"][: min(int(top_m), len(kts_result["ranked_regions"]))]
            top_zones = merge_selected_kts_units(kts_record["segments"], selected)
            top_record, _ = local_comet_segmentation(
                video_id=video_id,
                video_duration=duration,
                dino_features=dino,
                frame_timestamps=timestamps,
                candidate_zones=top_zones,
                config=comet_config,
            )
            top_contained = midpoint_containment(full_top, top_zones)
            top_seconds = union_duration([(float(z["start"]), float(z["end"])) for z in top_zones])
            sensitivity.append(
                {
                    "case_id": str(case["case_id"]),
                    "video_id": video_id,
                    "group": case["group"],
                    "top_m": int(top_m),
                    "candidate_zone_count": len(top_zones),
                    "candidate_zone_duration": top_seconds,
                    "candidate_zone_coverage_ratio": top_seconds / duration,
                    "local_comet_segment_count": len(top_record["segments"]),
                    "full_comet_top1_contained": bool(top_contained[0]) if top_contained else False,
                    "full_comet_top3_containment_rate": sum(top_contained) / len(top_contained) if top_contained else 0.0,
                    "diagnostic_proxy_only": True,
                }
            )
    if not all(row["passed"] for row in score_regressions):
        raise RuntimeError("Historical scorer equivalence did not pass")
    aggregate_fields = [key for key in metrics[0] if key not in {"case_id", "video_id", "group"}]
    aggregate: dict[str, Any] = {}
    for group in ("all", "problematic", "control"):
        rows = [row for row in metrics if group == "all" or row["group"] == group]
        aggregate[group] = {"case_count": len(rows), **mean_fields(rows, aggregate_fields)}
    sensitivity_aggregate: dict[str, Any] = {}
    sensitivity_fields = [
        "candidate_zone_count",
        "candidate_zone_duration",
        "candidate_zone_coverage_ratio",
        "local_comet_segment_count",
        "full_comet_top1_contained",
        "full_comet_top3_containment_rate",
    ]
    for group in ("all", "problematic", "control"):
        sensitivity_aggregate[group] = {}
        for top_m in experiment["sensitivity_top_m"]:
            rows = [
                row for row in sensitivity
                if row["top_m"] == top_m and (group == "all" or row["group"] == group)
            ]
            sensitivity_aggregate[group][str(top_m)] = {
                "case_count": len(rows), **mean_fields(rows, sensitivity_fields)
            }
    write_jsonl(OUT / "kts_only_results.jsonl", kts_only_results)
    write_jsonl(OUT / "full_comet_results.jsonl", full_comet_results)
    write_jsonl(OUT / "kts_local_comet_results.jsonl", hybrid_results)
    write_json(OUT / "per_case_metrics.json", metrics)
    write_json(OUT / "aggregate_metrics.json", {"primary_top_m": primary_m, "groups": aggregate, "top_m_sensitivity": sensitivity_aggregate})
    write_json(OUT / "top_m_sensitivity.json", {"per_case": sensitivity, "aggregate": sensitivity_aggregate})
    write_json(OUT / "containment_proxy.json", containment_rows)
    write_json(
        OUT / "runtime_metrics.json",
        {
            "offline_reused": {
                "dinov2_feature_extraction_sec_from_previous_run": previous_runtime["dinov2_feature_extraction_sec_total"],
                "kts_index_construction_sec_from_previous_run": previous_runtime["kts_segmentation_sec_total"],
                "dinov2_cache_bytes": previous_runtime["dino_cache_size_bytes"],
                "kts_index_bytes": (PREVIOUS / "kts_segments.jsonl").stat().st_size,
                "full_comet_index_bytes": (PREVIOUS / "comet_segments.jsonl").stat().st_size,
                "counted_as_online": False,
            },
            "online_shared": {"clip_query_encoder": query_runtime},
            "per_case": runtime_rows,
            "means": mean_fields(
                runtime_rows,
                [key for key in runtime_rows[0] if key != "video_id"],
            ),
            "wall_clock_sec_excluding_clip_model_and_query_load": time.perf_counter() - run_started,
            "dino_inference_calls": 0,
            "paid_api_calls": 0,
        },
    )
    previous_artifacts = [
        "frozen_config.json", "run_manifest.json", "kts_segments.jsonl",
        "comet_segments.jsonl", "retrieval_replay.jsonl", "shared_frame_grids.json",
    ]
    source_hashes = {name: sha256(PREVIOUS / name) for name in previous_artifacts}
    frozen = {
        **experiment,
        "source_experiment_config": {
            "dinov2": previous_config["dinov2"],
            "comet_style": previous_config["comet_style"],
            "kts": previous_config["kts"],
            "retrieval_replay": previous_config["retrieval_replay"],
        },
        "source_manifest_sha256": sha256(MANIFEST),
        "source_artifact_sha256": source_hashes,
        "config_sha256": sha256(EXPERIMENT_CONFIG),
    }
    write_json(OUT / "frozen_config.json", frozen)
    write_json(
        OUT / "run_manifest.json",
        {
            "experiment_id": experiment["experiment_id"],
            "case_count": len(cases),
            "case_ids": [str(row["case_id"]) for row in cases],
            "source_manifest_sha256": sha256(MANIFEST),
            "source_artifact_sha256": source_hashes,
            "dino_cache_sha256_by_video": cache_checks,
            "dino_cache_exactly_reused": cache_checks == previous_run["shared_dinov2_cache_sha256_by_video"],
            "frozen_kts_config_exactly_reused": frozen["source_experiment_config"]["kts"] == previous_config["kts"],
            "frozen_comet_config_exactly_reused": frozen["source_experiment_config"]["comet_style"] == previous_config["comet_style"],
            "shared_clip_scoring_regression": score_regressions,
            "shared_clip_scoring_regression_passed": all(row["passed"] for row in score_regressions),
            "gold_accessed": False,
            "qa_correctness_accessed": False,
            "answer_options_accessed": False,
            "dino_inference_calls": 0,
            "paid_api_calls": 0,
            "canonical_pipeline_modified": False,
        },
    )
    readme = """# Hierarchical refinement comparison v0.1

Zero-API structural comparison of KTS_ONLY, FULL_COMET, and KTS_LOCAL_COMET.

- Exactly reuses the frozen 10-video manifest, DINOv2 cache, KTS boundaries,
  full-video CoMET boundaries, CLIP frame embeddings, and question-only scorer.
- Primary hybrid always selects Top-3 KTS units, merges only consecutive selected
  units, and applies the frozen CoMET-style segmenter only inside those zones.
- FULL_COMET Top-3 containment is a diagnostic proxy, not ground-truth recall.
- No adaptive sufficiency, Task5C, Task6, final QA, answer options, gold, or paid API.
- No result is integrated into canonical Ours-v0.1.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "output": OUT.as_posix(),
                "source_reuse": True,
                "scoring_regressions": len(score_regressions),
                "paid_api_calls": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
