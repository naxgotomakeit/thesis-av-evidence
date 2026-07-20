"""Run the isolated photometric-robust Fine-to-Medium frontier experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
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
    fine_leaf_hash,
    fluid_frontier,
    sha256_file,
    topology_hash,
)
from src.experiments.photometric_robust_fluid_medium_v0_1.experiment import (  # noqa: E402
    NormalizedDINOExtractor,
    absolute_confidences,
    absolute_frontier,
    boundary_diagnostics,
    complexity_alignment,
    copy_representative_asset,
    load_jsonl,
    medium_metrics,
    validate_medium_frontier,
)
from src.experiments.photometric_robust_fluid_medium_v0_1.reporting import (  # noqa: E402
    METHOD_LABELS,
    build_html,
)


CONFIG_PATH = ROOT / "config/experiments/photometric_robust_fluid_medium_v0_1.json"
OUT = ROOT / "outputs/experiments/photometric_robust_fluid_medium_v0_1"
METHOD_ORDER = list(METHOD_LABELS)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def source_specifications(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    long_cfg = config["primary_source"]
    for hierarchy in load_jsonl(ROOT / long_cfg["hierarchies"]):
        video_id = hierarchy["video_id"]
        cache_dir = ROOT / long_cfg["dinov2_cache_root"] / video_id
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        specs.append({
            "hierarchy": hierarchy, "source_group": "long_primary",
            "hierarchy_path": ROOT / long_cfg["hierarchies"],
            "frame_paths": sorted((ROOT / long_cfg["frame_root"] / video_id).glob("*.jpg")),
            "original_cache_dir": cache_dir, "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
        })
    short_cfg = config["optional_short_source"]
    short_path = ROOT / short_cfg["hierarchies"]
    if short_cfg["enabled_if_available"] and short_path.is_file():
        for hierarchy in load_jsonl(short_path):
            video_id = hierarchy["video_id"]
            cache_dir = ROOT / short_cfg["dinov2_cache_root"] / video_id
            metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
            specs.append({
                "hierarchy": hierarchy, "source_group": "short_robustness",
                "hierarchy_path": short_path,
                "frame_paths": sorted((ROOT / short_cfg["frame_root"] / video_id / short_cfg["frame_subdirectory"]).glob("*.jpg")),
                "original_cache_dir": cache_dir, "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
            })
    for spec in specs:
        expected = len(spec["timestamps"])
        if len(spec["frame_paths"]) != expected:
            raise RuntimeError(f'{spec["hierarchy"]["video_id"]}: frames {len(spec["frame_paths"])} != timestamps {expected}')
        original = np.load(spec["original_cache_dir"] / "features.npy", mmap_mode="r")
        if original.shape != (expected, 384):
            raise RuntimeError(f'{spec["hierarchy"]["video_id"]}: original DINO cache incompatible: {original.shape}')
    return specs


def method_profile(method: str) -> str:
    for preset in ("strict", "balanced", "loose"):
        if method.endswith(preset):
            return preset
    return "loose"


def copy_diagnostic_frame(source: str, destination: Path) -> str:
    src = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or sha256_file(destination) != sha256_file(src):
        shutil.copy2(src, destination)
    return destination.relative_to(OUT).as_posix()


def overlap_count(ids: list[str], nodes: dict[str, dict[str, Any]], start: float, end: float) -> int:
    return sum(float(nodes[item]["end"]) > start and float(nodes[item]["start"]) < end for item in ids)


def aggregate_summary(per_video: list[dict[str, Any]], source_group: str | None) -> dict[str, Any]:
    selected = [item for item in per_video if source_group is None or item["source_group"] == source_group]
    output = {}
    for method in METHOD_ORDER:
        rows = [item for item in selected if item["method"] == method]
        output[method] = {
            "video_count": len(rows), "fine_count": sum(item["fine_count"] for item in rows),
            "medium_count": sum(item["metrics"]["medium_count"] for item in rows),
            "projected_vlm_calls": sum(item["metrics"]["medium_count"] for item in rows),
            "mean_projected_reduction_vs_current_ratio": float(np.mean([item["projected_reduction_vs_current_ratio"] for item in rows])),
            "median_medium_duration_sec": float(np.median([item["metrics"]["median_duration_sec"] for item in rows])),
            "photometric_crossed": sum(item["metrics"]["boundary_crossing"]["suspected_photometric_crossed"] for item in rows),
            "persistent_crossed": sum(item["metrics"]["boundary_crossing"]["persistent_crossed"] for item in rows),
            "transient_crossed": sum(item["metrics"]["boundary_crossing"]["transient_crossed"] for item in rows),
            "overmerge_risk_flags": sum(item["metrics"]["overmerge_risk_flag_count"] for item in rows),
        }
    return output


def main() -> None:
    run_started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    assets_root = OUT / "assets"
    inputs_started = time.perf_counter()
    specs = source_specifications(config)
    hierarchy_load_sec = time.perf_counter() - inputs_started
    primary = [item for item in specs if item["source_group"] == "long_primary"]
    if len(primary) != 4:
        raise RuntimeError(f"Expected four primary long trees, got {len(primary)}")

    source_files = sorted({item["hierarchy_path"] for item in specs})
    source_hashes_before = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in source_files}
    identities_before = {
        item["hierarchy"]["video_id"]: {
            "fine_leaf_sha256": fine_leaf_hash(item["hierarchy"]),
            "topology_sha256": topology_hash(item["hierarchy"]),
        }
        for item in specs
    }

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_cfg = config["normalized_dino"]
    extractor = NormalizedDINOExtractor(
        device=device, batch_size=int(dino_cfg["batch_size"]),
        lower=float(dino_cfg["lower_percentile"]), upper=float(dino_cfg["upper_percentile"]),
    )
    all_per_video: list[dict[str, Any]] = []
    all_photo_diagnostics: list[dict[str, Any]] = []
    all_persistence: list[dict[str, Any]] = []
    all_fragmentation: list[dict[str, Any]] = []
    all_risks: list[dict[str, Any]] = []
    all_complexity: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    hierarchies: dict[str, dict[str, Any]] = {}
    diagnostics_by_video: dict[str, list[dict[str, Any]]] = {}
    method_ids_by_video: dict[str, dict[str, list[str]]] = {}
    assets_by_video: dict[str, dict[str, dict[str, Any]]] = {}
    normalized_cache_records = []
    runtime_by_video = []

    for index, spec in enumerate(specs, start=1):
        hierarchy = spec["hierarchy"]
        video_id = hierarchy["video_id"]
        print(f"[{index}/{len(specs)}] {video_id}: normalized DINO + ten frozen-tree Medium frontiers", flush=True)
        hierarchies[video_id] = hierarchy
        original_features = np.load(spec["original_cache_dir"] / "features.npy")
        normalized_features, normalized_meta = extractor.extract_or_load(
            video_id=video_id, frame_paths=spec["frame_paths"], timestamps=spec["timestamps"],
            cache_dir=OUT / "normalized_dino_cache" / video_id,
        )
        normalized_cache_records.append({"video_id": video_id, **normalized_meta})
        started = time.perf_counter()
        diagnostics = boundary_diagnostics(
            hierarchy, spec["frame_paths"], spec["timestamps"], original_features, normalized_features,
            config["photometric_detection_profiles"], config["persistence"],
        )
        diagnostic_sec = time.perf_counter() - started
        diagnostics_by_video[video_id] = diagnostics
        all_photo_diagnostics.extend(diagnostics)
        all_persistence.extend({
            key: row[key] for key in (
                "video_id", "boundary_id", "timestamp", "pre_post_near_similarity", "pre_post_far_similarity",
                "post_near_post_far_similarity", "pre_post_near_change", "pre_post_far_change",
                "persistent_state_change", "transient_or_reverting_change", "window_frame_counts",
            )
        } for row in diagnostics)
        diagnostic_by_id = {item["boundary_id"]: item for item in diagnostics}
        context = build_tree_context(hierarchy)
        method_results: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
        pruning_started = time.perf_counter()
        method_results["current_fluid_loose"] = fluid_frontier(
            context,
            minimum_q_rank=float(config["adaptive_baseline"]["minimum_q_rank"]),
            maximum_local_drop=float(config["adaptive_baseline"]["maximum_local_drop"]),
            level="medium",
        )
        per_method_pruning = {}
        for family in ("local_absolute", "photo", "photo_temporal"):
            for preset, preset_cfg in config["absolute_merge_confidence"]["presets"].items():
                method = f"{family}_{preset}"
                item_started = time.perf_counter()
                confidence, details = absolute_confidences(
                    hierarchy, diagnostic_by_id,
                    photo_profile=preset if family != "local_absolute" else None,
                    use_temporal=family == "photo_temporal",
                    photo_downweight=float(config["photometric_boundary_downweight"]),
                    transient_downweight=float(config["persistence"]["transient_boundary_downweight"]),
                )
                method_results[method] = absolute_frontier(
                    hierarchy, confidence, details,
                    minimum_confidence=float(preset_cfg["minimum_confidence"]),
                    maximum_local_drop=float(preset_cfg["maximum_local_drop"]), method=method,
                )
                per_method_pruning[method] = time.perf_counter() - item_started
        pruning_sec = time.perf_counter() - pruning_started
        method_ids_by_video[video_id] = {method: result[0] for method, result in method_results.items()}
        current_count = len(method_results["current_fluid_loose"][0])
        metrics_started = time.perf_counter()
        assets_by_video[video_id] = {}
        nodes = context["nodes"]
        for method in METHOD_ORDER:
            medium_ids, trace = method_results[method]
            validity = validate_medium_frontier(hierarchy, medium_ids)
            if not validity["valid"]:
                raise RuntimeError(f"Invalid Medium frontier: {video_id} {method}: {validity}")
            profile = method_profile(method)
            metrics = medium_metrics(hierarchy, medium_ids, original_features, diagnostics, profile)
            for medium_id in medium_ids:
                if medium_id not in assets_by_video[video_id]:
                    assets_by_video[video_id][medium_id] = copy_representative_asset(
                        node=nodes[medium_id], root=ROOT, output_assets=assets_root, video_id=video_id,
                    )
            record = {
                "video_id": video_id, "source_group": spec["source_group"], "method": method,
                "video_duration": hierarchy["video_duration"], "fine_count": len(context["fine_ids"]),
                "medium_ids": medium_ids, "projected_vlm_calls": len(medium_ids),
                "projected_reduction_vs_current_count": current_count - len(medium_ids),
                "projected_reduction_vs_current_ratio": (current_count - len(medium_ids)) / current_count,
                "validity": validity, "metrics": metrics, "decision_trace": trace,
            }
            all_per_video.append(record)
            all_fragmentation.append({
                "video_id": video_id, "source_group": spec["source_group"], "method": method,
                **{key: metrics[key] for key in (
                    "adjacent_similarity_mean", "adjacent_similarity_median", "adjacent_similarity_counts",
                    "high_similarity_weak_boundary_pairs",
                )},
            })
            all_risks.append({
                "video_id": video_id, "source_group": spec["source_group"], "method": method,
                "flag_count": metrics["overmerge_risk_flag_count"], "rows": metrics["risk_rows"],
            })
            all_decisions.append({"video_id": video_id, "method": method, "trace": trace})
        metrics_sec = time.perf_counter() - metrics_started
        complexity_started = time.perf_counter()
        all_complexity.append(complexity_alignment(
            hierarchy, method_ids_by_video[video_id], diagnostics,
            float(config["diagnostics"]["window_duration_sec"]),
        ))
        complexity_sec = time.perf_counter() - complexity_started
        runtime_by_video.append({
            "video_id": video_id, "source_group": spec["source_group"],
            "normalized_dino_cache_hit": bool(normalized_meta["cache_hit"]),
            "normalized_grayscale_preprocessing_sec": float(normalized_meta["preprocessing_sec"]),
            "normalized_dino_inference_sec": float(normalized_meta["inference_sec"]),
            "photometric_and_temporal_diagnostics_sec": diagnostic_sec,
            "fluid_pruning_total_sec": pruning_sec, "fluid_pruning_by_variant_sec": per_method_pruning,
            "metrics_sec": metrics_sec, "complexity_alignment_sec": complexity_sec,
        })

    # Deterministic flashing-case selection uses only the frozen Fixed Reference cut.
    flash_start, flash_end = float(config["flashing_case"]["start_sec"]), float(config["flashing_case"]["end_sec"])
    flash_candidates = []
    for spec in primary:
        hierarchy = spec["hierarchy"]
        nodes = build_tree_context(hierarchy)["nodes"]
        ids = list(hierarchy["cuts"]["medium"]["node_ids"])
        flash_candidates.append((overlap_count(ids, nodes, flash_start, flash_end), hierarchy["video_id"]))
    _, flash_video_id = sorted(flash_candidates, key=lambda item: (-item[0], item[1]))[0]
    flash_hierarchy = hierarchies[flash_video_id]
    flash_nodes = build_tree_context(flash_hierarchy)["nodes"]
    flash_boundary_rows = [item for item in diagnostics_by_video[flash_video_id] if flash_start < item["timestamp"] < flash_end]
    flash_methods = []
    for record in [item for item in all_per_video if item["video_id"] == flash_video_id]:
        ids = [
            item for item in record["medium_ids"]
            if float(flash_nodes[item]["end"]) > flash_start and float(flash_nodes[item]["start"]) < flash_end
        ]
        crossed = record["metrics"]["boundary_crossing"]["rows"]
        inside_crossed = [item for item in crossed if flash_start < item["timestamp"] < flash_end]
        risks = [item for item in record["metrics"]["risk_rows"] if item["node_id"] in ids]
        similarities = []
        for left, right in zip(ids, ids[1:]):
            similarities.append(float(np.dot(
                np.asarray(flash_nodes[left]["pooled_dinov2"]), np.asarray(flash_nodes[right]["pooled_dinov2"])
            )))
        flash_methods.append({
            "method": record["method"], "medium_ids_in_interval": ids,
            "medium_count_in_interval": len(ids),
            "median_medium_duration_sec": float(np.median([flash_nodes[item]["duration"] for item in ids])),
            "adjacent_similarity_mean": float(np.mean(similarities)) if similarities else None,
            "suspected_photometric_boundaries_crossed": sum(item["suspected_photometric"] and item["crossed"] for item in inside_crossed),
            "persistent_boundaries_crossed": sum(item["persistent"] and item["crossed"] for item in inside_crossed),
            "risk_flags": sum(item["flagged"] for item in risks),
        })
    flash = {
        "selection_rule": config["flashing_case"]["video_selection_rule"],
        "selection_candidates_fixed_reference_overlap": [
            {"video_id": video_id, "overlap_count": count} for count, video_id in sorted(flash_candidates, reverse=True)
        ],
        "video_id": flash_video_id, "start_sec": flash_start, "end_sec": flash_end,
        "methods": flash_methods, "boundary_diagnostics": flash_boundary_rows,
    }

    # Structural-proxy transition panels: highest minimum(original, normalized) changes.
    primary_ids = {item["hierarchy"]["video_id"] for item in primary}
    primary_diagnostics = [item for item in all_photo_diagnostics if item["video_id"] in primary_ids]
    transition_rows = sorted(
        primary_diagnostics,
        key=lambda item: (-min(item["original_dino_change"], item["normalized_dino_change"]), item["video_id"], item["timestamp"]),
    )[: int(config["diagnostics"]["real_transition_count"])]
    real_transitions = []
    for index, item in enumerate(transition_rows):
        left_asset = copy_diagnostic_frame(item["left_frame_path"], assets_root / "diagnostic" / f"real_{index}_before.jpg")
        right_asset = copy_diagnostic_frame(item["right_frame_path"], assets_root / "diagnostic" / f"real_{index}_after.jpg")
        real_transitions.append({**item, "left_asset": left_asset, "right_asset": right_asset})

    # Stable-region panels use long frozen internal nodes with lowest normalized internal boundary change.
    stable_candidates = []
    for spec in primary:
        hierarchy = spec["hierarchy"]
        nodes = build_tree_context(hierarchy)["nodes"]
        by_id = {item["boundary_id"]: item for item in diagnostics_by_video[hierarchy["video_id"]]}
        for node in nodes.values():
            if not node["child_ids"] or float(node["duration"]) < 45.0:
                continue
            internal = [by_id[item] for item in node.get("internal_boundary_ids", []) if item in by_id]
            if not internal:
                continue
            stable_candidates.append({
                "video_id": hierarchy["video_id"], "node_id": node["node_id"], "start": node["start"], "end": node["end"],
                "duration": node["duration"], "normalized_mean": float(np.mean([item["normalized_dino_change"] for item in internal])),
                "persistent_rate": float(np.mean([item["persistent_state_change"] for item in internal])),
            })
    stable_candidates.sort(key=lambda item: (item["normalized_mean"], item["persistent_rate"], -item["duration"], item["video_id"]))
    selected_stable = []
    for candidate in stable_candidates:
        if any(
            candidate["video_id"] == old["video_id"]
            and max(0.0, min(candidate["end"], old["end"]) - max(candidate["start"], old["start"]))
            / min(candidate["duration"], old["duration"]) > 0.8
            for old in selected_stable
        ):
            continue
        selected_stable.append(candidate)
        if len(selected_stable) >= int(config["diagnostics"]["stable_long_region_count"]):
            break
    stable_regions = []
    for index, item in enumerate(selected_stable):
        spec = next(value for value in specs if value["hierarchy"]["video_id"] == item["video_id"])
        timestamps = spec["timestamps"]
        quarter = float(item["start"]) + 0.25 * float(item["duration"])
        three_quarter = float(item["start"]) + 0.75 * float(item["duration"])
        left_index = int(np.argmin(np.abs(timestamps - quarter)))
        right_index = int(np.argmin(np.abs(timestamps - three_quarter)))
        left_asset = copy_diagnostic_frame(spec["frame_paths"][left_index].as_posix(), assets_root / "diagnostic" / f"stable_{index}_left.jpg")
        right_asset = copy_diagnostic_frame(spec["frame_paths"][right_index].as_posix(), assets_root / "diagnostic" / f"stable_{index}_right.jpg")
        stable_regions.append({
            **item, "timestamp": 0.5 * (float(item["start"]) + float(item["end"])),
            "original_dino_change": None, "normalized_dino_change": item["normalized_mean"],
            "persistent_state_change": item["persistent_rate"], "left_asset": left_asset, "right_asset": right_asset,
        })

    primary_summary = aggregate_summary(all_per_video, "long_primary")
    all_summary = aggregate_summary(all_per_video, None)
    ablations = {}
    for preset in ("strict", "balanced", "loose"):
        a = primary_summary["current_fluid_loose"]["medium_count"]
        b = primary_summary[f"local_absolute_{preset}"]["medium_count"]
        c = primary_summary[f"photo_{preset}"]["medium_count"]
        d = primary_summary[f"photo_temporal_{preset}"]["medium_count"]
        ablations[preset] = {
            "A_to_B_remove_percentile_medium_delta": b - a,
            "B_to_C_add_photometric_medium_delta": c - b,
            "C_to_D_add_temporal_persistence_medium_delta": d - c,
            "A": a, "B": b, "C": c, "D": d,
        }

    runtime = {
        "one_time_auxiliary_index_cost": {
            "model_load_sec": extractor.load_sec,
            "normalized_grayscale_preprocessing_sec": sum(item["normalized_grayscale_preprocessing_sec"] for item in runtime_by_video),
            "normalized_dino_inference_sec": sum(item["normalized_dino_inference_sec"] for item in runtime_by_video),
            "cache_hits": sum(item["normalized_dino_cache_hit"] for item in runtime_by_video),
            "cache_misses": sum(not item["normalized_dino_cache_hit"] for item in runtime_by_video),
        },
        "cached_hierarchy_load_sec": hierarchy_load_sec,
        "photometric_and_temporal_diagnostics_sec": sum(item["photometric_and_temporal_diagnostics_sec"] for item in runtime_by_video),
        "fluid_pruning_sec": sum(item["fluid_pruning_total_sec"] for item in runtime_by_video),
        "metrics_and_complexity_sec": sum(item["metrics_sec"] + item["complexity_alignment_sec"] for item in runtime_by_video),
        "per_video": runtime_by_video,
        "html_generation_sec": None, "total_runtime_sec": None,
        "external_api_calls": 0, "vlm_calls": 0, "downloads": 0,
    }

    html_started = time.perf_counter()
    html_validation = build_html(
        output_path=OUT / "medium_comparison.html", per_video=all_per_video, hierarchies=hierarchies,
        assets=assets_by_video, flash=flash, real_transitions=real_transitions,
        stable_regions=stable_regions, method_order=METHOD_ORDER, runtime=runtime,
    )
    runtime["html_generation_sec"] = time.perf_counter() - html_started
    runtime["total_runtime_sec"] = time.perf_counter() - run_started

    source_hashes_after = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in source_files}
    identities_after = {
        item["hierarchy"]["video_id"]: {
            "fine_leaf_sha256": fine_leaf_hash(item["hierarchy"]),
            "topology_sha256": topology_hash(item["hierarchy"]),
        }
        for item in specs
    }
    if source_hashes_before != source_hashes_after or identities_before != identities_after:
        raise RuntimeError("Frozen Fine/tree identity changed during isolated experiment")
    if html_validation["missing_image_reference_count"]:
        raise RuntimeError(f"HTML has missing images: {html_validation}")

    method_summary = {
        "primary_long": primary_summary, "all_14": all_summary,
        "ablation_effects_primary_long": ablations,
        "interpretation": "structural diagnostics only; no winner selected",
    }
    frozen = {
        **config, "config_sha256": sha256_file(CONFIG_PATH),
        "source_file_hashes_before": source_hashes_before, "source_file_hashes_after": source_hashes_after,
        "source_files_unchanged": source_hashes_before == source_hashes_after,
        "tree_identities_before": identities_before, "tree_identities_after": identities_after,
        "tree_identities_unchanged": identities_before == identities_after,
        "model_runtime": {"device": device, "load_count": extractor.load_count, "load_sec": extractor.load_sec},
        "git_branch": git("branch", "--show-current"), "git_head": git("rev-parse", "HEAD"),
        "git_status_at_execution": git("status", "--short"),
        "canonical_tracked_diff": git("diff", "--name-only"),
    }
    manifest = {
        "schema_version": "photometric-robust-fluid-medium-run-v1",
        "experiment_id": config["experiment_id"], "video_count": len(specs),
        "primary_long_count": len(primary), "short_robustness_count": len(specs) - len(primary),
        "video_ids": [item["hierarchy"]["video_id"] for item in specs], "method_order": METHOD_ORDER,
        "fine_and_topology_identities": identities_before,
        "normalized_dino_cache": normalized_cache_records,
        "flashing_case_video_id": flash_video_id,
        "external_api_calls": 0, "vlm_calls": 0, "downloads": 0, "generate_coarse": False,
    }

    write_json(OUT / "method_summary.json", method_summary)
    write_json(OUT / "per_video_metrics.json", all_per_video)
    write_json(OUT / "flash_case_diagnostics.json", flash)
    write_json(OUT / "photometric_boundary_diagnostics.json", all_photo_diagnostics)
    write_json(OUT / "temporal_persistence_diagnostics.json", all_persistence)
    write_json(OUT / "fragmentation_metrics.json", all_fragmentation)
    write_json(OUT / "overmerge_risk_metrics.json", all_risks)
    write_json(OUT / "complexity_alignment.json", all_complexity)
    write_json(OUT / "fluid_decisions.json", all_decisions)
    write_json(OUT / "runtime_metrics.json", runtime)
    write_json(OUT / "frozen_config.json", frozen)
    write_json(OUT / "run_manifest.json", manifest)
    write_json(OUT / "html_validation.json", html_validation)
    (OUT / "README.md").write_text(
        "# Photometric-robust fluid Medium v0.1\n\n"
        "This isolated experiment compares ten Fine-to-Medium operating frontiers over exact frozen Safe-Merge trees. "
        "The same local DINOv2 ViT-S/14 supplies a grayscale/contrast-normalized auxiliary cache. Fine leaves and topology are unchanged; no Coarse, VLM, API, retrieval, audio, or QA is run. "
        "`medium_comparison.html` is the primary human-review artifact. No winner is selected automatically.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "primary_summary": primary_summary, "ablation": ablations, "flash": flash,
        "runtime": runtime, "html": html_validation,
    }, indent=2))


if __name__ == "__main__":
    main()
