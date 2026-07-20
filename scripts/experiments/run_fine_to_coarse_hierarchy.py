"""Run the isolated, zero-API Fine→Medium→Coarse hierarchy experiment."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.fine_to_coarse_hierarchy.hierarchy import (  # noqa: E402
    build_boundary_records,
    build_fine_nodes,
    build_safe_hierarchy,
    build_ward_hierarchy,
    normalize,
)
from src.experiments.fine_to_coarse_hierarchy.reporting import render_comparison  # noqa: E402


CONFIG_PATH = ROOT / "config/experiments/fine_to_coarse_hierarchy_v0_1.json"
OUT = ROOT / "outputs/experiments/fine_to_coarse_hierarchy_v0_1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def duration_stats(intervals: list[dict[str, Any]], video_duration: float) -> dict[str, Any]:
    durations = [float(row["duration"]) for row in intervals]
    return {
        "node_count": len(intervals),
        "mean_duration_sec": statistics.fmean(durations),
        "median_duration_sec": statistics.median(durations),
        "max_duration_sec": max(durations),
        "largest_region_ratio": max(durations) / video_duration,
    }


def nodes_for_cut(hierarchy: dict[str, Any], cut: str) -> list[dict[str, Any]]:
    by_id = {row["node_id"]: row for row in hierarchy["nodes"]}
    return [by_id[node_id] for node_id in hierarchy["cuts"][cut]["node_ids"]]


def boundary_survival_for_hierarchy(
    hierarchy: dict[str, Any], boundary_records: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cut in ("medium", "coarse"):
        nodes = nodes_for_cut(hierarchy, cut)
        surviving_times = {round(float(node["end"]), 6) for node in nodes[:-1]}
        records = []
        for boundary in boundary_records:
            records.append(
                {
                    "boundary_id": boundary["boundary_id"],
                    "timestamp": boundary["timestamp"],
                    "strength_category": boundary["strength_category"],
                    "strength": boundary["smoothed_change_strength"],
                    "very_strong_veto": boundary["very_strong_fine_to_medium_veto"],
                    "survives": round(float(boundary["timestamp"]), 6) in surviving_times,
                }
            )
        by_category = {}
        for category in ("weak", "medium", "strong"):
            selected = [row for row in records if row["strength_category"] == category]
            survived = sum(row["survives"] for row in selected)
            by_category[category] = {
                "count": len(selected),
                "survived": survived,
                "survival_rate": survived / len(selected) if selected else None,
            }
        result[cut] = {
            "records": records,
            "by_strength": by_category,
            "strong_survival_rate": by_category["strong"]["survival_rate"] or 0.0,
            "all_survival_rate": sum(row["survives"] for row in records) / len(records) if records else 1.0,
        }
    return result


def fine_semantic_differences(fine_nodes: list[dict[str, Any]]) -> list[float]:
    return [
        float(1.0 - normalize(np.asarray(left["pooled_dinov2"])) @ normalize(np.asarray(right["pooled_dinov2"])))
        for left, right in zip(fine_nodes[:-1], fine_nodes[1:])
    ]


def cut_diagnostics(
    *,
    hierarchy: dict[str, Any],
    cut: str,
    fine_nodes: list[dict[str, Any]],
    boundary_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    nodes = nodes_for_cut(hierarchy, cut)
    stats = duration_stats(nodes, float(hierarchy["video_duration"]))
    stats["compression_ratio_vs_fine"] = len(nodes) / len(fine_nodes)
    stats["node_reduction_fraction"] = 1.0 - stats["compression_ratio_vs_fine"]
    variabilities = [float(node["internal_variability"]) for node in nodes]
    variance_threshold = float(np.quantile(variabilities, config["diagnostics"]["high_internal_variability_quantile"]))
    boundary_by_id = {row["boundary_id"]: row for row in boundary_records}
    semantic_differences = fine_semantic_differences(fine_nodes)
    endpoint_threshold = float(np.quantile(semantic_differences, config["diagnostics"]["chaining_endpoint_difference_quantile"]))
    fine_by_id = {row["node_id"]: row for row in fine_nodes}
    risk_nodes = []
    chaining_nodes = []
    stable_background_subtle_candidates = []
    semantic_low_threshold = float(np.quantile(semantic_differences, 0.25))
    for node in nodes:
        if len(node["leaf_ids"]) <= 1:
            continue
        crossed = [boundary_by_id[identifier] for identifier in node["internal_boundary_ids"]]
        crosses_strong = any(row["strength_category"] == "strong" for row in crossed)
        high_variance = float(node["internal_variability"]) >= variance_threshold
        large_fraction = float(node["duration"]) / hierarchy["video_duration"] >= config["diagnostics"]["large_node_video_ratio"]
        conditions = {
            "crosses_strong_fine_boundary": crosses_strong,
            "high_internal_variability": high_variance,
            "large_video_fraction": large_fraction,
        }
        if any(conditions.values()):
            risk_nodes.append(
                {
                    "node_id": node["node_id"], "start": node["start"], "end": node["end"],
                    "leaf_count": len(node["leaf_ids"]), "conditions": conditions,
                    "compound_risk": sum(conditions.values()) >= 2,
                }
            )
        first = fine_by_id[node["leaf_ids"][0]]
        last = fine_by_id[node["leaf_ids"][-1]]
        endpoint_difference = float(
            1.0 - normalize(np.asarray(first["pooled_dinov2"])) @ normalize(np.asarray(last["pooled_dinov2"]))
        )
        if len(node["leaf_ids"]) >= 3 and endpoint_difference >= endpoint_threshold:
            chaining_nodes.append(
                {
                    "node_id": node["node_id"], "leaf_count": len(node["leaf_ids"]),
                    "endpoint_semantic_difference": endpoint_difference,
                    "data_derived_threshold": endpoint_threshold,
                }
            )
        local_indices = [fine_nodes.index(fine_by_id[identifier]) for identifier in node["leaf_ids"]]
        adjacent_inside = [semantic_differences[index] for index in local_indices[:-1]]
        if len(node["leaf_ids"]) >= 3 and adjacent_inside and statistics.fmean(adjacent_inside) <= semantic_low_threshold:
            stable_background_subtle_candidates.append(
                {
                    "node_id": node["node_id"], "start": node["start"], "end": node["end"],
                    "fine_leaf_count": len(node["leaf_ids"]),
                    "mean_adjacent_semantic_difference": statistics.fmean(adjacent_inside),
                    "review_reason": "Many Fine boundaries occur within a low-DINO-change merged component; potential stable-background/subtle-manipulation case.",
                }
            )
    survival = boundary_survival_for_hierarchy(hierarchy, boundary_records)[cut]
    similar_threshold = float(np.quantile(semantic_differences, config["diagnostics"]["undermerge_similarity_quantile"]))
    similar_boundaries = [
        (boundary, semantic_differences[index])
        for index, boundary in enumerate(boundary_records)
        if semantic_differences[index] <= similar_threshold
    ]
    survival_by_id = {row["boundary_id"]: row["survives"] for row in survival["records"]}
    similar_surviving = sum(survival_by_id[boundary["boundary_id"]] for boundary, _ in similar_boundaries)
    undermerge_rate = similar_surviving / len(similar_boundaries) if similar_boundaries else 0.0
    stats.update(
        {
            "overmerge_risk_nodes": risk_nodes,
            "overmerge_risk_node_count": len(risk_nodes),
            "compound_overmerge_risk_node_count": sum(row["compound_risk"] for row in risk_nodes),
            "chaining_risk_nodes": chaining_nodes,
            "chaining_risk_node_count": len(chaining_nodes),
            "stable_background_subtle_candidates": stable_background_subtle_candidates,
            "stable_background_subtle_candidate_count": len(stable_background_subtle_candidates),
            "similar_fine_boundary_count": len(similar_boundaries),
            "similar_fine_boundaries_surviving": similar_surviving,
            "undermerge_survival_rate": undermerge_rate,
            "undermerge_flag": undermerge_rate >= config["diagnostics"]["undermerge_survival_fraction"],
            "variance_high_threshold": variance_threshold,
            "strong_boundary_survival_rate": survival["strong_survival_rate"],
        }
    )
    return stats


def fine_metrics(fine_nodes: list[dict[str, Any]], video_duration: float) -> dict[str, Any]:
    stats = duration_stats(fine_nodes, video_duration)
    stats.update(
        {
            "preservation_rate": 1.0,
            "original_interval_sha256": hashlib.sha256(
                json.dumps(
                    [(row["node_id"], row["start"], row["end"]) for row in fine_nodes],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    )
    return stats


def aggregate(per_video: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    group_by_video = {str(row["video_id"]): str(row["group"]) for row in manifest["videos"]}
    output: dict[str, Any] = {"schema_version": "fine-to-coarse-aggregate-metrics-v1", "video_count": len(per_video)}
    for group in ("all", "problematic", "control"):
        rows = [row for row in per_video if group == "all" or group_by_video[row["video_id"]] == group]
        section: dict[str, Any] = {
            "video_count": len(rows),
            "fine": {
                "total_nodes": sum(row["fine"]["node_count"] for row in rows),
                "mean_node_count": statistics.fmean(row["fine"]["node_count"] for row in rows),
                "mean_duration_sec": statistics.fmean(row["fine"]["mean_duration_sec"] for row in rows),
                "mean_max_duration_sec": statistics.fmean(row["fine"]["max_duration_sec"] for row in rows),
            },
            "kts_reference": {
                "mean_node_count": statistics.fmean(row["kts_reference"]["node_count"] for row in rows),
                "mean_largest_region_ratio": statistics.fmean(row["kts_reference"]["largest_region_ratio"] for row in rows),
            },
        }
        for method in ("ward", "safe"):
            section[method] = {}
            for cut in ("medium", "coarse"):
                cut_rows = [row[method][cut] for row in rows]
                section[method][cut] = {
                    "mean_node_count": statistics.fmean(row["node_count"] for row in cut_rows),
                    "mean_compression_ratio_vs_fine": statistics.fmean(row["compression_ratio_vs_fine"] for row in cut_rows),
                    "mean_node_reduction_fraction": statistics.fmean(row["node_reduction_fraction"] for row in cut_rows),
                    "mean_duration_sec": statistics.fmean(row["mean_duration_sec"] for row in cut_rows),
                    "mean_max_duration_sec": statistics.fmean(row["max_duration_sec"] for row in cut_rows),
                    "mean_largest_region_ratio": statistics.fmean(row["largest_region_ratio"] for row in cut_rows),
                    "strong_boundary_survival_rate": sum(
                        row["boundary_counts"][method][cut]["strong_survived"] for row in rows
                    ) / max(1, sum(row["boundary_counts"][method][cut]["strong_total"] for row in rows)),
                    "boundary_survival_by_strength": {
                        category: {
                            "total": sum(
                                row["boundary_counts"][method][cut][f"{category}_total"] for row in rows
                            ),
                            "survived": sum(
                                row["boundary_counts"][method][cut][f"{category}_survived"] for row in rows
                            ),
                            "survival_rate": (
                                sum(row["boundary_counts"][method][cut][f"{category}_survived"] for row in rows)
                                / max(1, sum(row["boundary_counts"][method][cut][f"{category}_total"] for row in rows))
                            ),
                        }
                        for category in ("weak", "medium", "strong")
                    },
                    "overmerge_risk_node_count": sum(row["overmerge_risk_node_count"] for row in cut_rows),
                    "compound_overmerge_risk_node_count": sum(row["compound_overmerge_risk_node_count"] for row in cut_rows),
                    "undermerge_flag_video_count": sum(row["undermerge_flag"] for row in cut_rows),
                    "chaining_risk_node_count": sum(row["chaining_risk_node_count"] for row in cut_rows),
                    "same_background_candidate_count": sum(row["stable_background_subtle_candidate_count"] for row in cut_rows),
                }
        output[group] = section
    # Cross-video scale variability is descriptive only.
    output["granularity_variability"] = {}
    for method in ("ward", "safe"):
        output["granularity_variability"][method] = {}
        for cut in ("medium", "coarse"):
            values = [row[method][cut]["mean_duration_sec"] for row in per_video]
            output["granularity_variability"][method][cut] = {
                "mean_node_duration_coefficient_of_variation": statistics.pstdev(values) / statistics.fmean(values),
                "unstable_granularity_flag_at_cv_gt_0_5": statistics.pstdev(values) / statistics.fmean(values) > 0.5,
            }
    return output


def main() -> int:
    started = time.perf_counter()
    config = load_json(CONFIG_PATH)
    source = ROOT / config["source_experiment"]
    manifest_path = ROOT / config["source_manifest"]
    manifest = load_json(manifest_path)
    fine_records = load_jsonl(source / config["fine_source"])
    kts_records = load_jsonl(source / config["kts_reference_source"])
    diagnostics = load_json(source / config["method_diagnostics_source"])
    frame_grids = load_json(source / config["frame_grid_source"])
    expected_ids = [str(row["video_id"]) for row in manifest["videos"]]
    for name, rows in (("Fine", fine_records), ("KTS", kts_records), ("diagnostics", diagnostics), ("frame grid", frame_grids)):
        if [str(row["video_id"]) for row in rows] != expected_ids:
            raise RuntimeError(f"{name} video identity/order differs from frozen 10-video manifest")
    fine_by_video = {row["video_id"]: row for row in fine_records}
    kts_by_video = {row["video_id"]: row for row in kts_records}
    diag_by_video = {row["video_id"]: row for row in diagnostics}
    grid_by_video = {row["video_id"]: row for row in frame_grids}
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "source_manifest": sha256(manifest_path),
        "source_frozen_config": sha256(source / "frozen_config.json"),
        "source_comet_segments": sha256(source / config["fine_source"]),
        "source_kts_segments": sha256(source / config["kts_reference_source"]),
        "source_method_diagnostics": sha256(source / config["method_diagnostics_source"]),
        "source_frame_grids": sha256(source / config["frame_grid_source"]),
    }
    ward_hierarchies: list[dict[str, Any]] = []
    safe_hierarchies: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    boundary_results: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    dino_hashes: dict[str, dict[str, str]] = {}
    for manifest_row in manifest["videos"]:
        video_id = str(manifest_row["video_id"])
        video_started = time.perf_counter()
        fine_source = fine_by_video[video_id]
        diag = diag_by_video[video_id]["comet"]
        grid = grid_by_video[video_id]
        timestamps = np.asarray(grid["actual_timestamps"], dtype=np.float64)
        visual_dir = ROOT / "outputs/visual_index" / video_id
        frame_paths_abs = sorted((visual_dir / "frames_1fps").glob("frame_*.jpg"))
        if len(frame_paths_abs) != len(timestamps):
            raise RuntimeError(f"Frame-grid mismatch for {video_id}")
        frame_paths = [path.relative_to(ROOT).as_posix() for path in frame_paths_abs]
        cache_dir = source / config["dinov2_cache_source"] / video_id
        features_path = cache_dir / "features.npy"
        metadata_path = cache_dir / "metadata.json"
        metadata = load_json(metadata_path)
        cache_load_started = time.perf_counter()
        features = np.load(features_path)
        cache_load_sec = time.perf_counter() - cache_load_started
        if len(features) != len(timestamps) or int(metadata["dimension"]) != features.shape[1]:
            raise RuntimeError(f"DINO cache schema mismatch for {video_id}")
        if metadata["normalization"].lower() != "l2" or metadata["model"] != "facebook/dinov2-small":
            raise RuntimeError(f"Unexpected DINO cache semantics for {video_id}")
        dino_hashes[video_id] = {
            "features_npy_sha256": sha256(features_path),
            "metadata_json_sha256": sha256(metadata_path),
            "declared_features_sha256": metadata["features_sha256"],
        }
        if dino_hashes[video_id]["features_npy_sha256"] != metadata["features_sha256"]:
            raise RuntimeError(f"DINO cache content hash differs from frozen metadata for {video_id}")
        boundary_records = build_boundary_records(fine_source["segments"], diag)
        fine_nodes = build_fine_nodes(
            segments=fine_source["segments"], features=features, timestamps=timestamps,
            frame_paths=frame_paths, boundary_records=boundary_records,
            representative_fractions=tuple(config["representative_frames"]),
            include_medoid=bool(config["include_dinov2_medoid"]),
        )
        source_intervals = [
            (str(row["segment_id"]), float(row["start"]), float(row["end"]))
            for row in fine_source["segments"]
        ]
        built_intervals = [(row["node_id"], row["start"], row["end"]) for row in fine_nodes]
        if source_intervals != built_intervals:
            raise RuntimeError(f"Immutable Fine interval invariant failed for {video_id}")
        ward_started = time.perf_counter()
        ward, ward_log = build_ward_hierarchy(
            video_id=video_id, video_duration=float(fine_source["video_duration"]),
            fine_nodes=fine_nodes, boundary_records=boundary_records, timestamps=timestamps,
            frame_paths=frame_paths, features=features,
            medium_fraction=float(config["medium_visualization_target_fraction"]),
            coarse_fraction=float(config["coarse_visualization_target_fraction"]),
            representative_fractions=tuple(config["representative_frames"]),
            include_medoid=bool(config["include_dinov2_medoid"]),
        )
        ward_sec = time.perf_counter() - ward_started
        safe_started = time.perf_counter()
        safe, safe_log = build_safe_hierarchy(
            video_id=video_id, video_duration=float(fine_source["video_duration"]),
            fine_nodes=fine_nodes, boundary_records=boundary_records, timestamps=timestamps,
            frame_paths=frame_paths, features=features,
            medium_fraction=float(config["medium_visualization_target_fraction"]),
            coarse_fraction=float(config["coarse_visualization_target_fraction"]),
            representative_fractions=tuple(config["representative_frames"]),
            include_medoid=bool(config["include_dinov2_medoid"]),
        )
        safe_sec = time.perf_counter() - safe_started
        if not ward["invariants"]["valid"] or not safe["invariants"]["valid"]:
            raise RuntimeError(f"Hierarchy invariant failure for {video_id}")
        ward_hierarchies.append(ward)
        safe_hierarchies.append(safe)
        decisions.extend(ward_log)
        decisions.extend(safe_log)
        ward_survival = boundary_survival_for_hierarchy(ward, boundary_records)
        safe_survival = boundary_survival_for_hierarchy(safe, boundary_records)
        boundary_results.append(
            {
                "video_id": video_id,
                "boundary_count": len(boundary_records),
                "boundary_strength_counts": dict(Counter(row["strength_category"] for row in boundary_records)),
                "ward": ward_survival,
                "safe": safe_survival,
            }
        )
        ward_metrics = {
            cut: cut_diagnostics(
                hierarchy=ward, cut=cut, fine_nodes=fine_nodes,
                boundary_records=boundary_records, config=config,
            )
            for cut in ("medium", "coarse")
        }
        safe_metrics = {
            cut: cut_diagnostics(
                hierarchy=safe, cut=cut, fine_nodes=fine_nodes,
                boundary_records=boundary_records, config=config,
            )
            for cut in ("medium", "coarse")
        }
        kts_stats = duration_stats(kts_by_video[video_id]["segments"], float(fine_source["video_duration"]))
        boundary_counts = defaultdict(lambda: defaultdict(dict))
        for method, survival in (("ward", ward_survival), ("safe", safe_survival)):
            for cut in ("medium", "coarse"):
                boundary_counts[method][cut] = {}
                for category in ("weak", "medium", "strong"):
                    category_rows = survival[cut]["by_strength"][category]
                    boundary_counts[method][cut][f"{category}_total"] = category_rows["count"]
                    boundary_counts[method][cut][f"{category}_survived"] = category_rows["survived"]
        failure_modes = {
            "ward_overmerge": ward_metrics["medium"]["compound_overmerge_risk_node_count"] > 0,
            "safe_overmerge": safe_metrics["medium"]["compound_overmerge_risk_node_count"] > 0,
            "ward_undermerge": ward_metrics["medium"]["undermerge_flag"],
            "safe_undermerge": safe_metrics["medium"]["undermerge_flag"],
            "ward_chaining": ward_metrics["medium"]["chaining_risk_node_count"] > 0,
            "safe_chaining": safe_metrics["medium"]["chaining_risk_node_count"] > 0,
            "ward_kts_like_collapse": (
                kts_stats["largest_region_ratio"] >= config["diagnostics"]["large_node_video_ratio"]
                and ward_metrics["medium"]["largest_region_ratio"] >= config["diagnostics"]["kts_like_ratio_tolerance"] * kts_stats["largest_region_ratio"]
            ),
            "safe_kts_like_collapse": (
                kts_stats["largest_region_ratio"] >= config["diagnostics"]["large_node_video_ratio"]
                and safe_metrics["medium"]["largest_region_ratio"] >= config["diagnostics"]["kts_like_ratio_tolerance"] * kts_stats["largest_region_ratio"]
            ),
            "useless_hierarchy_low_medium_compression": (
                ward_metrics["medium"]["node_reduction_fraction"] < 0.25
                and safe_metrics["medium"]["node_reduction_fraction"] < 0.25
            ),
            "same_background_video_risk": False,
        }
        per_video.append(
            {
                "video_id": video_id, "group": manifest_row["group"],
                "fine": fine_metrics(fine_nodes, float(fine_source["video_duration"])),
                "ward": ward_metrics, "safe": safe_metrics,
                "kts_reference": kts_stats,
                "boundary_counts": {method: dict(cuts) for method, cuts in boundary_counts.items()},
                "median_frame_adjacent_dino_change": statistics.median(
                    1.0 - float(value) for value in diag["raw_similarity"]
                ),
                "failure_mode_diagnostics": failure_modes,
            }
        )
        runtime_rows.append(
            {
                "video_id": video_id, "dino_cache_load_sec": cache_load_sec,
                "new_dinov2_inference_sec": 0.0, "ward_construction_sec": ward_sec,
                "safe_merge_construction_sec": safe_sec,
                "video_processing_wall_sec": time.perf_counter() - video_started,
                "fine_nodes": len(fine_nodes),
                "ward_tree_nodes": len(ward["nodes"]), "safe_tree_nodes": len(safe["nodes"]),
                "ward_parent_child_edges": ward["invariants"]["parent_child_edge_count"],
                "safe_parent_child_edges": safe["invariants"]["parent_child_edge_count"],
            }
        )
    # Cross-video same-background risk: low median appearance change plus high Fine fragmentation.
    changes = np.asarray([row["median_frame_adjacent_dino_change"] for row in per_video])
    counts = np.asarray([row["fine"]["node_count"] for row in per_video])
    low_change = float(np.quantile(changes, 0.25))
    high_fragmentation = float(np.quantile(counts, 0.75))
    for row in per_video:
        risk = (
            row["median_frame_adjacent_dino_change"] <= low_change
            and row["fine"]["node_count"] >= high_fragmentation
        )
        row["failure_mode_diagnostics"]["same_background_video_risk"] = bool(risk)
        row["same_background_risk_basis"] = {
            "median_change": row["median_frame_adjacent_dino_change"],
            "low_change_q25": low_change,
            "fine_count": row["fine"]["node_count"],
            "high_fragmentation_q75": high_fragmentation,
            "interpretation": "Potential stable-background/subtle-manipulation review case; not an action label.",
        }
    aggregate_metrics = aggregate(per_video, manifest)
    failure_summary = Counter(
        name for row in per_video for name, flagged in row["failure_mode_diagnostics"].items() if flagged
    )
    aggregate_metrics["failure_mode_video_counts"] = dict(sorted(failure_summary.items()))
    aggregate_metrics["fine_preservation_rate"] = 1.0
    write_jsonl(OUT / "ward_hierarchies.jsonl", ward_hierarchies)
    write_jsonl(OUT / "safe_merge_hierarchies.jsonl", safe_hierarchies)
    write_jsonl(OUT / "merge_decision_log.jsonl", decisions)
    write_json(OUT / "per_video_metrics.json", per_video)
    write_json(OUT / "aggregate_metrics.json", aggregate_metrics)
    write_json(OUT / "boundary_survival.json", boundary_results)
    frozen_config = {
        **config,
        "config_sha256": sha256(CONFIG_PATH),
        "source_artifact_hashes": source_hashes,
        "dinov2_cache_hashes_by_video": dino_hashes,
    }
    write_json(OUT / "frozen_config.json", frozen_config)
    run_manifest = {
        "schema_version": "fine-to-coarse-run-manifest-v1",
        "experiment_id": config["experiment_id"],
        "video_count": len(expected_ids), "video_ids": expected_ids,
        "source_manifest_sha256": source_hashes["source_manifest"],
        "source_fine_segments_sha256": source_hashes["source_comet_segments"],
        "source_kts_reference_sha256": source_hashes["source_kts_segments"],
        "fine_intervals_immutable": True,
        "fine_preservation_rate": 1.0,
        "dino_cache_reused": True, "new_dinov2_inference_calls": 0,
        "paid_api_calls": 0, "questions_accessed": False,
        "gold_answers_or_options_accessed": False,
        "canonical_pipeline_modified": False,
    }
    write_json(OUT / "run_manifest.json", run_manifest)
    # Runtime/storage is written after hierarchy files so measured metadata bytes are exact.
    metadata_files = [
        OUT / "ward_hierarchies.jsonl", OUT / "safe_merge_hierarchies.jsonl",
        OUT / "merge_decision_log.jsonl", OUT / "per_video_metrics.json",
        OUT / "aggregate_metrics.json", OUT / "boundary_survival.json",
    ]
    runtime = {
        "schema_version": "fine-to-coarse-runtime-v1",
        "offline_reusable_index_experiment": True,
        "new_dinov2_feature_extraction_calls": 0,
        "new_dinov2_feature_extraction_sec": 0.0,
        "dino_cache_load_sec_total": sum(row["dino_cache_load_sec"] for row in runtime_rows),
        "ward_construction_sec_total": sum(row["ward_construction_sec"] for row in runtime_rows),
        "safe_merge_construction_sec_total": sum(row["safe_merge_construction_sec"] for row in runtime_rows),
        "ward_construction_sec_mean_per_video": statistics.fmean(row["ward_construction_sec"] for row in runtime_rows),
        "safe_merge_construction_sec_mean_per_video": statistics.fmean(row["safe_merge_construction_sec"] for row in runtime_rows),
        "hierarchy_metadata_size_bytes": sum(path.stat().st_size for path in metadata_files),
        "reused_dino_cache_size_bytes": sum(
            (source / config["dinov2_cache_source"] / video_id / "features.npy").stat().st_size
            + (source / config["dinov2_cache_source"] / video_id / "metadata.json").stat().st_size
            for video_id in expected_ids
        ),
        "total_fine_nodes": sum(row["fine"]["node_count"] for row in per_video),
        "total_ward_tree_nodes": sum(row["ward_tree_nodes"] for row in runtime_rows),
        "total_safe_tree_nodes": sum(row["safe_tree_nodes"] for row in runtime_rows),
        "total_parent_child_edges": sum(
            row["ward_parent_child_edges"] + row["safe_parent_child_edges"] for row in runtime_rows
        ),
        "per_video": runtime_rows,
        "experiment_wall_sec_before_html": time.perf_counter() - started,
        "interpretation": "Hierarchy construction is offline. Runtime alone does not establish future online QA efficiency.",
    }
    write_json(OUT / "runtime_metrics.json", runtime)
    render_comparison(
        output_path=OUT / "comparison.html", manifest=manifest,
        ward_hierarchies=ward_hierarchies, safe_hierarchies=safe_hierarchies,
        kts_records=kts_records, per_video_metrics=per_video,
        aggregate_metrics=aggregate_metrics, boundary_survival=boundary_results,
    )
    readme = f"""# Fine to Coarse hierarchy v0.1

Offline proof-of-concept over {len(expected_ids)} frozen EgoSchema videos and {aggregate_metrics['all']['fine']['total_nodes']} immutable full-video CoMET-style Fine leaves.

Methods:

- Adjacent Ward: minimum adjacency-constrained Ward SSE increase.
- Boundary-aware Safe Merge: equal-weight data-derived semantic, boundary, variability, and size evidence with a Fine→Medium top-decile boundary veto.
- KTS is displayed only as a reference timeline and never constructs either hierarchy.

Medium≈50% and Coarse≈25% are transparent visualization cuts, not ground-truth event granularity. Every full binary tree is reversible to the exact source Fine intervals. No questions, gold, options, API, Planner, retrieval, Task5C/Task6, or final QA were used.

## Structural results

- Fine leaves: {aggregate_metrics['all']['fine']['total_nodes']} total, {aggregate_metrics['all']['fine']['mean_node_count']:.1f}/video; preservation 100%.
- Ward Medium: {aggregate_metrics['all']['ward']['medium']['mean_node_count']:.1f} nodes/video, {100*aggregate_metrics['all']['ward']['medium']['mean_node_reduction_fraction']:.1f}% reduction, {100*aggregate_metrics['all']['ward']['medium']['strong_boundary_survival_rate']:.1f}% strong-boundary survival.
- Safe Medium: {aggregate_metrics['all']['safe']['medium']['mean_node_count']:.1f} nodes/video, {100*aggregate_metrics['all']['safe']['medium']['mean_node_reduction_fraction']:.1f}% reduction, {100*aggregate_metrics['all']['safe']['medium']['strong_boundary_survival_rate']:.1f}% strong-boundary survival.
- Ward/Safe mean Medium largest-region ratio: {aggregate_metrics['all']['ward']['medium']['mean_largest_region_ratio']:.3f} / {aggregate_metrics['all']['safe']['medium']['mean_largest_region_ratio']:.3f}.
- Ward/Safe Medium compound over-merge risk nodes: {aggregate_metrics['all']['ward']['medium']['compound_overmerge_risk_node_count']} / {aggregate_metrics['all']['safe']['medium']['compound_overmerge_risk_node_count']}.

Safe Merge preserves strong boundaries more often and has fewer compound risk flags, but produces larger maximum regions by concentrating compression across weaker boundaries. Semantic usefulness therefore remains a human-review question rather than an automatic win.

Open `comparison.html` for aligned timelines, thumbnails, expandable Coarse→Medium→Fine lineage, and blank manual-review fields.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "videos": len(expected_ids), "fine_leaves": aggregate_metrics["all"]["fine"]["total_nodes"],
        "fine_preservation": aggregate_metrics["fine_preservation_rate"],
        "ward_medium_mean": aggregate_metrics["all"]["ward"]["medium"]["mean_node_count"],
        "safe_medium_mean": aggregate_metrics["all"]["safe"]["medium"]["mean_node_count"],
        "api_calls": 0, "output": OUT.as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
