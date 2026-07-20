"""Compare operating frontiers on unchanged frozen Boundary-aware Safe-Merge trees."""

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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.adaptive_fluid_hierarchy_v0_1.experiment import (  # noqa: E402
    build_tree_context,
    duration_metrics,
    elbow_diagnostics,
    fine_leaf_hash,
    fixed_target_frontier,
    fluid_frontier,
    load_jsonl,
    local_complexity_alignment,
    materialize_representative,
    overmerge_metrics,
    redundancy_metrics,
    sha256_file,
    threshold_frontier,
    topology_hash,
    validate_frontiers,
)
from src.experiments.adaptive_fluid_hierarchy_v0_1.reporting import render_comparison  # noqa: E402
from src.experiments.fine_to_coarse_hierarchy.hierarchy import build_boundary_records  # noqa: E402


CONFIG_PATH = ROOT / "config/experiments/adaptive_fluid_hierarchy_v0_1.json"
OUT = ROOT / "outputs/experiments/adaptive_fluid_hierarchy_v0_1"
METHOD_ORDER = [
    "fixed_reference", "fixed_coarser", "global_adaptive_elbow",
    "fluid_strict", "fluid_balanced", "fluid_loose", "fluid_balanced_guarded",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_versions() -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "Pillow"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = None
    return output


def source_bundle(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load primary four and optional ten without recomputing segmentation or topology."""
    long_cfg = config["primary_long_source"]
    long_path = ROOT / long_cfg["hierarchies"]
    long_hierarchies = load_jsonl(long_path)
    long_diag = {row["video_id"]: row for row in load_json(ROOT / long_cfg["diagnostics"])}
    rows = []
    for hierarchy in long_hierarchies:
        video_id = hierarchy["video_id"]
        rows.append(
            {
                "source_group": "long_primary", "hierarchy": hierarchy,
                "raw_similarity": long_diag[video_id]["comet_internal"]["raw_similarity"],
                "boundary_records": long_diag[video_id]["boundary_records"],
                "cache_dir": ROOT / long_cfg["dinov2_cache_root"] / video_id,
                "source_hierarchy_file": long_path,
            }
        )
    short_cfg = config["optional_short_source"]
    short_path = ROOT / short_cfg["hierarchies"]
    short_included = bool(short_cfg["enabled_if_available"] and short_path.is_file())
    if short_included:
        short_hierarchies = load_jsonl(short_path)
        segment_by_video = {row["video_id"]: row for row in load_jsonl(ROOT / short_cfg["comet_segments"])}
        diag_by_video = {row["video_id"]: row["comet"] for row in load_json(ROOT / short_cfg["diagnostics"])}
        for hierarchy in short_hierarchies:
            video_id = hierarchy["video_id"]
            diagnostic = diag_by_video[video_id]
            boundaries = build_boundary_records(segment_by_video[video_id]["segments"], diagnostic)
            rows.append(
                {
                    "source_group": "short_robustness", "hierarchy": hierarchy,
                    "raw_similarity": diagnostic["raw_similarity"], "boundary_records": boundaries,
                    "cache_dir": ROOT / short_cfg["dinov2_cache_root"] / video_id,
                    "source_hierarchy_file": short_path,
                }
            )
    provenance = {
        "long_hierarchy_path": long_path.relative_to(ROOT).as_posix(),
        "long_hierarchy_sha256": sha256_file(long_path),
        "long_video_count": len(long_hierarchies),
        "short_hierarchy_available_and_included": short_included,
        "short_hierarchy_path": short_path.relative_to(ROOT).as_posix() if short_included else None,
        "short_hierarchy_sha256": sha256_file(short_path) if short_included else None,
        "short_video_count": len(rows) - len(long_hierarchies),
    }
    return rows, provenance


def create_methods(
    hierarchy: dict[str, Any], context: dict[str, Any], config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, float]]:
    results: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    started = time.perf_counter()
    results["fixed_reference"] = {
        "medium_ids": list(hierarchy["cuts"]["medium"]["node_ids"]),
        "coarse_ids": list(hierarchy["cuts"]["coarse"]["node_ids"]),
        "medium_trace": [], "coarse_trace": [],
    }
    timings["fixed_reference"] = time.perf_counter() - started

    started = time.perf_counter()
    fixed_cfg = config["methods"]["fixed_coarser"]
    results["fixed_coarser"] = {
        "medium_ids": fixed_target_frontier(hierarchy, float(fixed_cfg["medium_fraction"])),
        "coarse_ids": fixed_target_frontier(hierarchy, float(fixed_cfg["coarse_fraction"])),
        "medium_trace": [], "coarse_trace": [],
    }
    timings["fixed_coarser"] = time.perf_counter() - started

    elbow_started = time.perf_counter()
    elbow = elbow_diagnostics(hierarchy)
    medium, medium_trace = threshold_frontier(context, maximum_merge_cost=elbow["medium_cost_threshold"])
    coarse, coarse_trace = threshold_frontier(context, maximum_merge_cost=elbow["coarse_cost_threshold"])
    results["global_adaptive_elbow"] = {
        "medium_ids": medium, "coarse_ids": coarse,
        "medium_trace": medium_trace, "coarse_trace": coarse_trace,
    }
    timings["global_adaptive_elbow"] = time.perf_counter() - elbow_started

    for method in ("fluid_strict", "fluid_balanced", "fluid_loose", "fluid_balanced_guarded"):
        started = time.perf_counter()
        method_cfg = config["methods"][method]
        medium_cfg, coarse_cfg = method_cfg["medium"], method_cfg["coarse"]
        medium, medium_trace = fluid_frontier(
            context,
            minimum_q_rank=float(medium_cfg["minimum_q_rank"]),
            maximum_local_drop=float(medium_cfg["maximum_local_drop"]),
            maximum_duration_sec=medium_cfg.get("maximum_duration_sec"),
            maximum_video_ratio=medium_cfg.get("maximum_video_ratio"), level="medium",
        )
        coarse, coarse_trace = fluid_frontier(
            context,
            minimum_q_rank=float(coarse_cfg["minimum_q_rank"]),
            maximum_local_drop=float(coarse_cfg["maximum_local_drop"]),
            maximum_duration_sec=coarse_cfg.get("maximum_duration_sec"),
            maximum_video_ratio=coarse_cfg.get("maximum_video_ratio"), level="coarse",
        )
        results[method] = {
            "medium_ids": medium, "coarse_ids": coarse,
            "medium_trace": medium_trace, "coarse_trace": coarse_trace,
        }
        timings[method] = time.perf_counter() - started
    return results, elbow, timings


def aggregate_method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scope, selected_rows in (
        ("all_14", rows),
        ("long_primary_4", [row for row in rows if row["source_group"] == "long_primary"]),
        ("short_robustness_10", [row for row in rows if row["source_group"] == "short_robustness"]),
    ):
        if not selected_rows:
            continue
        output[scope] = {}
        for method in METHOD_ORDER:
            values = [row for row in selected_rows if row["method"] == method]
            output[scope][method] = {
                "video_count": len(values),
                "total_fine_count": sum(row["fine_count"] for row in values),
                "total_medium_count": sum(row["medium"]["node_count"] for row in values),
                "total_coarse_count": sum(row["coarse"]["node_count"] for row in values),
                "median_medium_duration_sec": statistics.median(row["medium"]["median_duration_sec"] for row in values),
                "median_coarse_duration_sec": statistics.median(row["coarse"]["median_duration_sec"] for row in values),
                "maximum_parent_ratio": max(row["coarse"]["largest_node_ratio"] for row in values),
                "adjacent_similarity_median": statistics.median(
                    row["redundancy"]["similarity_distribution"]["median"]
                    for row in values if row["redundancy"]["similarity_distribution"]["median"] is not None
                ),
                "high_similarity_weak_boundary_pairs": sum(row["redundancy"]["high_similarity_weak_boundary_pair_count"] for row in values),
                "overmerge_chaining_flags": sum(
                    row["overmerge_medium"]["total_flagged_node_count"] + row["overmerge_coarse"]["total_flagged_node_count"]
                    for row in values
                ),
                "projected_vlm_calls": sum(row["medium"]["node_count"] for row in values),
                "adaptive_cut_compute_sec": sum(row["runtime"]["adaptive_cut_compute_sec"] for row in values),
                "metrics_compute_sec": sum(row["runtime"]["metrics_compute_sec"] for row in values),
                "all_structurally_valid": all(row["validity"]["valid"] for row in values),
            }
    return output


def main() -> int:
    experiment_started = time.perf_counter()
    config = load_json(CONFIG_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    bundles, source_provenance = source_bundle(config)
    frozen_file_hashes_before = {
        path: sha256_file(ROOT / path)
        for path in {bundle["source_hierarchy_file"].relative_to(ROOT).as_posix() for bundle in bundles}
    }
    hierarchy_load_sec = time.perf_counter() - load_started

    identity_records = [
        {
            "video_id": bundle["hierarchy"]["video_id"],
            "source_group": bundle["source_group"],
            "fine_leaf_sha256": fine_leaf_hash(bundle["hierarchy"]),
            "topology_sha256": topology_hash(bundle["hierarchy"]),
            "source_hierarchy_file": bundle["source_hierarchy_file"].relative_to(ROOT).as_posix(),
        }
        for bundle in bundles
    ]
    write_json(
        OUT / "run_manifest.json",
        {
            "schema_version": "adaptive-fluid-hierarchy-run-v1",
            "experiment_id": config["experiment_id"],
            "video_count": len(bundles), "primary_long_count": 4,
            "optional_short_count": len(bundles) - 4,
            "source_provenance": source_provenance,
            "tree_identities": identity_records,
            "method_order": METHOD_ORDER,
            "rerun_vlm": False, "external_api_calls": 0, "downloads": 0,
        },
    )

    per_method: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    elbow_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    overmerge_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    asset_by_video_node: dict[tuple[str, str], dict[str, Any]] = {}

    diag_cfg = config["diagnostics"]
    for video_index, bundle in enumerate(bundles, start=1):
        hierarchy = bundle["hierarchy"]
        video_id = str(hierarchy["video_id"])
        print(f"[{video_index}/{len(bundles)}] {video_id}: seven frozen-tree frontiers", flush=True)
        normalize_started = time.perf_counter()
        context = build_tree_context(hierarchy)
        normalization_sec = time.perf_counter() - normalize_started
        cache_started = time.perf_counter()
        features = np.load(bundle["cache_dir"] / "features.npy")
        cache_meta = load_json(bundle["cache_dir"] / "metadata.json")
        if sha256_file(bundle["cache_dir"] / "features.npy") != cache_meta["features_sha256"]:
            raise RuntimeError(f"Cached DINO feature hash mismatch: {video_id}")
        timestamps = np.asarray(cache_meta["timestamps"], dtype=np.float64)
        cache_load_sec = time.perf_counter() - cache_started
        boundary_by_pair = {
            (row["left_leaf_id"], row["right_leaf_id"]): row for row in bundle["boundary_records"]
        }
        boundary_by_id = {row["boundary_id"]: row for row in bundle["boundary_records"]}
        methods, elbow, cut_timings = create_methods(hierarchy, context, config)
        elbow_rows.append({"video_id": video_id, "source_group": bundle["source_group"], **elbow})
        baseline_calls = len(methods["fixed_reference"]["medium_ids"])
        method_medium_for_alignment: dict[str, list[str]] = {}

        for method in METHOD_ORDER:
            structural_started = time.perf_counter()
            frontiers = methods[method]
            validity = validate_frontiers(hierarchy, frontiers["medium_ids"], frontiers["coarse_ids"])
            if not validity["valid"]:
                raise RuntimeError(f"Invalid frontier: {video_id}/{method}")
            medium_metrics = duration_metrics(frontiers["medium_ids"], context)
            coarse_metrics = duration_metrics(frontiers["coarse_ids"], context)
            structural_metrics_sec = time.perf_counter() - structural_started
            rep_started = time.perf_counter()
            for node_id in set(frontiers["medium_ids"] + frontiers["coarse_ids"]):
                key = (video_id, node_id)
                if key not in asset_by_video_node:
                    asset_by_video_node[key] = materialize_representative(
                        context["nodes"][node_id], root=ROOT, output_dir=OUT, video_id=video_id
                    )
            representative_sec = time.perf_counter() - rep_started
            diagnostic_started = time.perf_counter()
            redundancy = redundancy_metrics(
                frontiers["medium_ids"], context=context, features=features, timestamps=timestamps,
                boundary_by_pair=boundary_by_pair,
                similarity_thresholds=[float(value) for value in diag_cfg["adjacent_similarity_thresholds"]],
                high_similarity=float(diag_cfg["high_similarity_threshold"]),
                weak_boundary_percentile=float(diag_cfg["weak_boundary_percentile_maximum"]),
            )
            over_medium = overmerge_metrics(
                frontiers["medium_ids"], context=context, boundary_by_id=boundary_by_id,
                high_boundary_percentile=float(diag_cfg["high_boundary_percentile_minimum"]),
                low_coherence_percentile=float(diag_cfg["low_coherence_percentile_maximum"]),
                giant_ratio=float(diag_cfg["giant_node_video_ratio"]),
                chaining_ratio=float(diag_cfg["chaining_child_imbalance_ratio"]),
            )
            over_coarse = overmerge_metrics(
                frontiers["coarse_ids"], context=context, boundary_by_id=boundary_by_id,
                high_boundary_percentile=float(diag_cfg["high_boundary_percentile_minimum"]),
                low_coherence_percentile=float(diag_cfg["low_coherence_percentile_maximum"]),
                giant_ratio=float(diag_cfg["giant_node_video_ratio"]),
                chaining_ratio=float(diag_cfg["chaining_child_imbalance_ratio"]),
            )
            diagnostic_metrics_sec = time.perf_counter() - diagnostic_started
            metrics_sec = structural_metrics_sec + diagnostic_metrics_sec
            projected = {
                "video_id": video_id, "source_group": bundle["source_group"], "method": method,
                "fixed_reference_calls": baseline_calls,
                "projected_vlm_calls": len(frontiers["medium_ids"]),
                "projected_reduction_count": baseline_calls - len(frontiers["medium_ids"]),
                "projected_reduction_ratio": (baseline_calls - len(frontiers["medium_ids"])) / baseline_calls,
                "actual_vlm_calls": 0,
            }
            record = {
                "video_id": video_id, "source_group": bundle["source_group"], "method": method,
                "video_duration": context["video_duration"], "fine_count": len(context["fine_ids"]),
                "medium_ids": frontiers["medium_ids"], "coarse_ids": frontiers["coarse_ids"],
                "medium_to_coarse_membership": validity["medium_to_coarse_membership"],
                "representative_assets": {
                    node_id: asset_by_video_node[(video_id, node_id)]
                    for node_id in set(frontiers["medium_ids"] + frontiers["coarse_ids"])
                },
                "medium": medium_metrics, "coarse": coarse_metrics, "validity": validity,
                "redundancy": redundancy, "overmerge_medium": over_medium, "overmerge_coarse": over_coarse,
                "projected_vlm": projected,
                "runtime": {
                    "adaptive_cut_compute_sec": cut_timings[method],
                    "representative_extraction_reuse_sec": representative_sec,
                    "structural_metrics_sec": structural_metrics_sec,
                    "redundancy_overmerge_metrics_sec": diagnostic_metrics_sec,
                    "metrics_compute_sec": metrics_sec,
                },
            }
            per_method.append(record)
            redundancy_rows.append(
                {"video_id": video_id, "source_group": bundle["source_group"], "method": method, **redundancy}
            )
            overmerge_rows.append(
                {"video_id": video_id, "source_group": bundle["source_group"], "method": method, "medium": over_medium, "coarse": over_coarse}
            )
            projected_rows.append(projected)
            method_medium_for_alignment[method] = frontiers["medium_ids"]
            if method.startswith("fluid_"):
                decisions.append(
                    {
                        "video_id": video_id, "source_group": bundle["source_group"], "method": method,
                        "medium_trace": frontiers["medium_trace"], "coarse_trace": frontiers["coarse_trace"],
                    }
                )
        alignment = local_complexity_alignment(
            hierarchy=hierarchy, method_medium_ids=method_medium_for_alignment,
            raw_similarity=bundle["raw_similarity"], window_sec=float(diag_cfg["window_duration_sec"]),
        )
        complexity_rows.append({"video_id": video_id, "source_group": bundle["source_group"], **alignment})
        runtime_rows.append(
            {
                "video_id": video_id, "source_group": bundle["source_group"],
                "score_normalization_sec": normalization_sec,
                "cached_embedding_load_sec": cache_load_sec,
                "global_elbow_detection_and_cut_sec": cut_timings["global_adaptive_elbow"],
                "local_fluid_pruning_sec_by_preset": {
                    method: cut_timings[method] for method in METHOD_ORDER if method.startswith("fluid_")
                },
            }
        )

    method_summary = aggregate_method_summary(per_method)
    write_json(OUT / "method_summary.json", method_summary)
    write_json(OUT / "per_video_method_metrics.json", per_method)
    write_json(OUT / "fluid_decisions.json", decisions)
    write_json(OUT / "global_elbow_diagnostics.json", elbow_rows)
    write_json(OUT / "redundancy_metrics.json", redundancy_rows)
    write_json(OUT / "overmerge_risk_metrics.json", overmerge_rows)
    write_json(OUT / "local_complexity_alignment.json", complexity_rows)
    write_json(OUT / "projected_vlm_savings.json", projected_rows)

    html_started = time.perf_counter()
    html_validation = render_comparison(
        output_path=OUT / "hierarchy_comparison.html", method_rows=per_method,
        method_summary=method_summary, decisions=decisions, elbows=elbow_rows,
        complexity=complexity_rows, method_order=METHOD_ORDER,
    )
    html_sec = time.perf_counter() - html_started
    runtime = {
        "cached_input_cost": {
            "hierarchy_tree_loading_sec": hierarchy_load_sec,
            "cached_embedding_loading_sec": sum(row["cached_embedding_load_sec"] for row in runtime_rows),
            "old_dinov2_extraction_recomputed": False,
        },
        "new_adaptive_cut_compute": {
            "score_normalization_sec": sum(row["score_normalization_sec"] for row in runtime_rows),
            "global_elbow_sec": sum(row["global_elbow_detection_and_cut_sec"] for row in runtime_rows),
            "local_fluid_sec_by_preset": {
                method: sum(row["local_fluid_pruning_sec_by_preset"][method] for row in runtime_rows)
                for method in METHOD_ORDER if method.startswith("fluid_")
            },
            "per_video": runtime_rows,
        },
        "representative_and_metrics_sec": sum(
            row["runtime"]["representative_extraction_reuse_sec"] + row["runtime"]["metrics_compute_sec"]
            for row in per_method
        ),
        "html_generation_sec": html_sec,
        "total_runtime_sec": time.perf_counter() - experiment_started,
        "external_api_calls": 0,
        "downloads": 0,
        "vlm_calls": 0,
    }
    write_json(OUT / "runtime_metrics.json", runtime)
    write_json(OUT / "html_validation.json", html_validation)

    frozen_file_hashes_after = {path: sha256_file(ROOT / path) for path in frozen_file_hashes_before}
    frozen_config = {
        **config,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_file_hashes_before": frozen_file_hashes_before,
        "source_file_hashes_after": frozen_file_hashes_after,
        "source_files_unchanged": frozen_file_hashes_before == frozen_file_hashes_after,
        "tree_identities_before_and_after": identity_records,
        "git_branch": git_value("branch", "--show-current"),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_at_execution": git_value("status", "--short"),
        "package_versions": package_versions(),
        "external_api_calls": 0, "downloads": 0, "vlm_calls": 0,
        "canonical_pipeline_modified": False,
    }
    write_json(OUT / "frozen_config.json", frozen_config)
    (OUT / "README.md").write_text(
        "# Adaptive fluid hierarchy v0.1\n\n"
        "Seven operating-frontier methods are compared on the exact frozen Safe-Merge trees. No Fine boundary, "
        "merge order, parent/child link, or topology is changed. Local fluid policies use the prespecified "
        "within-video merge-quality rank and local quality-drop thresholds. Medium/Coarse counts are outputs.\n\n"
        "The four long EgoPolice videos are primary; ten frozen short-video trees are included only for robustness. "
        "No VLM, API, download, question, retrieval, Planner, classifier, audio, QA, or canonical path is used.\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": method_summary["long_primary_4"], "runtime": runtime, "html": html_validation}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
