"""Run the isolated, zero-API Event-level versus Action-level experiment."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.event_vs_action_fine_leaf.motion import (  # noqa: E402
    MotionProxyConfig,
    build_action_children,
    motion_cache_compatible,
)
from src.experiments.event_vs_action_fine_leaf import motion as motion_module  # noqa: E402
from src.experiments.event_vs_action_fine_leaf.reporting import render_comparison  # noqa: E402


CONFIG_PATH = ROOT / "config/experiments/event_vs_action_fine_leaf_v0_1.json"
OUT = ROOT / "outputs/experiments/event_vs_action_fine_leaf_v0_1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frame_paths(video_id: str) -> list[Path]:
    return sorted((ROOT / "outputs/visual_index" / video_id / "frames_1fps").glob("frame_*.jpg"))


def representative_frames(
    video_id: str, start: float, end: float, timestamps: np.ndarray,
    paths: list[Path], fractions: list[float],
) -> list[dict[str, Any]]:
    candidates = np.flatnonzero((timestamps >= start - 1e-8) & (timestamps < end - 1e-8))
    if not len(candidates):
        candidates = np.asarray([int(np.argmin(np.abs(timestamps - (start + end) / 2.0)))])
    rows = []
    for fraction in fractions:
        target = start + fraction * (end - start)
        index = int(candidates[int(np.argmin(np.abs(timestamps[candidates] - target)))])
        rows.append(
            {
                "kind": f"{int(fraction * 100)}%",
                "timestamp": float(timestamps[index]),
                "frame_path": paths[index].relative_to(ROOT).as_posix(),
            }
        )
    return rows


def decorate_segments(
    video_id: str, segments: list[dict[str, Any]], timestamps: np.ndarray,
    paths: list[Path], fractions: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            **segment,
            "representative_frames": representative_frames(
                video_id, float(segment["start"]), float(segment["end"]),
                timestamps, paths, fractions,
            ),
        }
        for segment in segments
    ]


def segment_stats(segments: list[dict[str, Any]], video_duration: float) -> dict[str, Any]:
    durations = [float(row["duration"]) for row in segments]
    count = len(durations)
    return {
        "segment_count": count,
        "mean_duration_sec": statistics.fmean(durations),
        "median_duration_sec": statistics.median(durations),
        "max_duration_sec": max(durations),
        "min_duration_sec": min(durations),
        "boundaries_per_minute": max(0, count - 1) / (video_duration / 60.0),
        "count_gt_20s": sum(value > 20.0 for value in durations),
        "fraction_gt_20s": sum(value > 20.0 for value in durations) / count,
        "count_gt_30s": sum(value > 30.0 for value in durations),
        "fraction_gt_30s": sum(value > 30.0 for value in durations) / count,
        "count_lt_2s": sum(value < 2.0 for value in durations),
        "fraction_lt_2s": sum(value < 2.0 for value in durations) / count,
        "count_lt_4s": sum(value < 4.0 for value in durations),
        "fraction_lt_4s": sum(value < 4.0 for value in durations) / count,
    }


def pooled_dino(
    segment: dict[str, Any], features: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    indices = np.flatnonzero(
        (timestamps >= float(segment["start"]) - 1e-8)
        & (timestamps < float(segment["end"]) - 1e-8)
    )
    if not len(indices):
        indices = np.asarray([int(np.argmin(np.abs(timestamps - (float(segment["start"]) + float(segment["end"])) / 2.0)))])
    vector = np.mean(features[indices], axis=0, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def appearance_change(
    event: dict[str, Any], features: np.ndarray, timestamps: np.ndarray
) -> float:
    indices = np.flatnonzero(
        (timestamps >= float(event["start"]) - 1e-8)
        & (timestamps < float(event["end"]) - 1e-8)
    )
    if len(indices) < 2:
        return 0.0
    local = features[indices]
    local = local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
    return float(np.mean(1.0 - np.sum(local[:-1] * local[1:], axis=1)))


def aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_durations = [duration for row in rows for duration in row["_event_durations"]]
    action_durations = [duration for row in rows for duration in row["_action_durations"]]
    event_total, action_total = len(event_durations), len(action_durations)

    def aggregate_level(durations: list[float], per_video_counts: list[int]) -> dict[str, Any]:
        return {
            "total_segments": len(durations),
            "mean_nodes_per_video": statistics.fmean(per_video_counts),
            "mean_duration_sec": statistics.fmean(durations),
            "median_duration_sec": statistics.median(durations),
            "max_duration_sec": max(durations),
            "min_duration_sec": min(durations),
            "fraction_gt_20s": sum(value > 20 for value in durations) / len(durations),
            "fraction_gt_30s": sum(value > 30 for value in durations) / len(durations),
            "fraction_lt_2s": sum(value < 2 for value in durations) / len(durations),
            "fraction_lt_4s": sum(value < 4 for value in durations) / len(durations),
            "mean_boundaries_per_minute": statistics.fmean(
                row["event_level" if durations is event_durations else "action_level"]["boundaries_per_minute"]
                for row in rows
            ),
        }

    return {
        "video_count": len(rows),
        "event_level": aggregate_level(event_durations, [row["event_level"]["segment_count"] for row in rows]),
        "action_level": aggregate_level(action_durations, [row["action_level"]["segment_count"] for row in rows]),
        "action_expansion_ratio": action_total / event_total,
        "parent_split_rate": sum(row["split_parent_count"] for row in rows) / event_total,
        "unchanged_event_rate": sum(row["unchanged_parent_count"] for row in rows) / event_total,
        "max_duration_reduction_sec": max(event_durations) - max(action_durations),
        "median_duration_reduction_sec": statistics.median(event_durations) - statistics.median(action_durations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute-motion-cache", action="store_true",
        help="Recompute and overwrite only this experiment's provenance-bound motion caches.",
    )
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    source_paths = {
        "manifest": ROOT / config["input_manifest"],
        "event_segments": ROOT / config["event_source"],
        "event_diagnostics": ROOT / config["event_diagnostics_source"],
        "frame_grids": ROOT / config["frame_grid_source"],
        "kts_reference": ROOT / config["kts_reference_source"],
        "coarse_config": ROOT / "outputs/experiments/coarse_segmentation_3way_v0_1/frozen_config.json",
    }
    for label, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact missing ({label}): {path}")
    manifest = load_json(source_paths["manifest"])
    if len(manifest["videos"]) != 10:
        raise ValueError("Experiment requires the exact frozen 10-video manifest")
    source_events = load_jsonl(source_paths["event_segments"])
    if len(source_events) != 10:
        raise ValueError("Frozen Event artifact must contain 10 videos")
    events_by_video = {row["video_id"]: row for row in source_events}
    if set(events_by_video) != {row["video_id"] for row in manifest["videos"]}:
        raise ValueError("Event source and frozen manifest video identities differ")
    grids = {row["video_id"]: row for row in load_json(source_paths["frame_grids"])}
    kts_rows = load_jsonl(source_paths["kts_reference"])
    fractions = [float(value) for value in config["representative_frames"]]
    action_cfg = config["action_level"]
    proxy_config = MotionProxyConfig(
        image_width=int(action_cfg["image_width"]),
        image_height=int(action_cfg["image_height"]),
        smoothing_sigma_frames=float(action_cfg["smoothing_sigma_frames"]),
        minimum_action_duration_sec=float(action_cfg["minimum_action_duration_sec"]),
        penalty_multiplier=float(action_cfg["penalty_multiplier"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    event_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    runtime_per_video: list[dict[str, Any]] = []
    diagnostic_event_records: list[dict[str, Any]] = []
    adjacency_records: list[dict[str, Any]] = []

    for manifest_item in manifest["videos"]:
        video_id = manifest_item["video_id"]
        event_source = events_by_video[video_id]
        duration = float(event_source["video_duration"])
        timestamps = np.asarray(grids[video_id]["actual_timestamps"], dtype=np.float64)
        paths = frame_paths(video_id)
        if len(paths) != len(timestamps):
            raise ValueError(f"Frame grid mismatch for {video_id}: {len(paths)} paths vs {len(timestamps)} timestamps")
        dino_path = ROOT / config["dinov2_cache_root"] / video_id / "features.npy"
        dino = np.load(dino_path)
        if len(dino) != len(timestamps):
            raise ValueError(f"DINO cache/grid mismatch for {video_id}")
        cache_npz = OUT / "motion_cache" / f"{video_id}.npz"
        cache_metadata = OUT / "motion_cache" / f"{video_id}.json"
        motion, motion_runtime = motion_module.extract_or_load_motion_proxy(
            video_id=video_id, frame_paths=paths, timestamps=timestamps,
            video_duration=duration, cache_npz=cache_npz,
            cache_metadata=cache_metadata, config=proxy_config,
            force_recompute=args.recompute_motion_cache,
        )
        # Exercise the provenance-gated reuse path without recomputing motion.
        reused_motion, reuse_runtime = motion_module.extract_or_load_motion_proxy(
            video_id=video_id, frame_paths=paths, timestamps=timestamps,
            video_duration=duration, cache_npz=cache_npz,
            cache_metadata=cache_metadata, config=proxy_config,
        )
        cache_reuse_verified = bool(reuse_runtime["cache_hit"]) and all(
            np.array_equal(motion[key], reused_motion[key]) for key in motion
        )
        if not cache_reuse_verified:
            raise RuntimeError(f"Motion cache reuse verification failed for {video_id}")
        split_started = time.perf_counter()
        actions, decisions = build_action_children(
            video_id=video_id, event_segments=event_source["segments"],
            motion=motion, config=proxy_config,
        )
        split_sec = time.perf_counter() - split_started
        decorated_events = decorate_segments(
            video_id, event_source["segments"], timestamps, paths, fractions
        )
        decorated_actions = decorate_segments(video_id, actions, timestamps, paths, fractions)
        event_rows.append(
            {
                "video_id": video_id,
                "method": "EVENT_LEVEL",
                "source_method": event_source["method"],
                "video_duration": duration,
                "segments": decorated_events,
                "source_boundary_hash": value_sha256(event_source["segments"]),
                "output_boundary_hash": value_sha256(
                    [{key: segment[key] for key in ("segment_id", "start", "end", "duration", "representative_frame_timestamp")} for segment in decorated_events]
                ),
            }
        )
        action_rows.append(
            {
                "video_id": video_id,
                "method": "ACTION_LEVEL",
                "motion_method": "LIGHTWEIGHT_MOTION_PROXY",
                "video_duration": duration,
                "segments": decorated_actions,
            }
        )
        mapping_rows.append(
            {
                "video_id": video_id,
                "events": [
                    {
                        "event_id": decision["parent_event_id"],
                        "action_child_ids": decision["action_child_ids"],
                    }
                    for decision in decisions
                ],
            }
        )
        motion_rows.append(
            {
                "video_id": video_id,
                "method": "LIGHTWEIGHT_MOTION_PROXY",
                "interval_starts": motion["interval_starts"].tolist(),
                "interval_ends": motion["interval_ends"].tolist(),
                "raw_motion": motion["raw_motion"].tolist(),
                "smoothed_motion": motion["smoothed_motion"].tolist(),
                "event_decisions": decisions,
            }
        )
        split_count = sum(decision["split"] for decision in decisions)
        event_stats = segment_stats(event_source["segments"], duration)
        action_stats = segment_stats(actions, duration)
        metric = {
            "video_id": video_id,
            "group": manifest_item["group"],
            "event_level": event_stats,
            "action_level": action_stats,
            "action_expansion_ratio": len(actions) / len(event_source["segments"]),
            "split_parent_count": split_count,
            "parent_split_rate": split_count / len(event_source["segments"]),
            "unchanged_parent_count": len(event_source["segments"]) - split_count,
            "unchanged_event_rate": 1.0 - split_count / len(event_source["segments"]),
            "max_duration_reduction_sec": event_stats["max_duration_sec"] - action_stats["max_duration_sec"],
            "median_duration_reduction_sec": event_stats["median_duration_sec"] - action_stats["median_duration_sec"],
            "_event_durations": [float(row["duration"]) for row in event_source["segments"]],
            "_action_durations": [float(row["duration"]) for row in actions],
        }
        metrics.append(metric)
        runtime_per_video.append(
            {
                "video_id": video_id,
                "device": "CPU",
                "motion_cache_hit": motion_runtime["cache_hit"],
                "motion_cache_reuse_verified": cache_reuse_verified,
                "cache_validation_sec": reuse_runtime["load_or_extract_sec"],
                "motion_load_or_extract_sec": motion_runtime["load_or_extract_sec"],
                "change_point_sec": split_sec,
                "total_action_index_sec": motion_runtime["load_or_extract_sec"] + split_sec,
                "motion_cache_size_bytes": motion_runtime["cache_size_bytes"],
            }
        )
        decision_by_event = {row["parent_event_id"]: row for row in decisions}
        action_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for action in decorated_actions:
            action_by_parent[action["parent_event_id"]].append(action)
        for event in decorated_events:
            decision = decision_by_event[event["segment_id"]]
            diagnostic_event_records.append(
                {
                    "video_id": video_id,
                    "group": manifest_item["group"],
                    "event_id": event["segment_id"],
                    "start": event["start"], "end": event["end"],
                    "appearance_change": appearance_change(event, dino, timestamps),
                    "max_motion_change_score": max(
                        [row["motion_change_score"] for row in decision["change_points"]],
                        default=0.0,
                    ),
                    "split": decision["split"],
                    "representative_frames": event["representative_frames"],
                }
            )
            children = action_by_parent[event["segment_id"]]
            for index, (left, right) in enumerate(zip(children[:-1], children[1:])):
                left_vector = pooled_dino(left, dino, timestamps)
                right_vector = pooled_dino(right, dino, timestamps)
                boundary = decision["change_points"][index]
                adjacency_records.append(
                    {
                        "video_id": video_id,
                        "parent_event_id": event["segment_id"],
                        "left_action_id": left["action_id"],
                        "right_action_id": right["action_id"],
                        "boundary_timestamp": boundary["timestamp"],
                        "dino_appearance_similarity": float(left_vector @ right_vector),
                        "motion_change_score": boundary["motion_change_score"],
                    }
                )

    diagnostic_cfg = config["diagnostics"]
    split_events = [row for row in diagnostic_event_records if row["split"]]
    appearance_threshold = float(np.quantile(
        [row["appearance_change"] for row in diagnostic_event_records],
        diagnostic_cfg["stable_appearance_quantile"],
    ))
    positive_scores = [row["max_motion_change_score"] for row in split_events]
    strong_motion_threshold = float(np.quantile(
        positive_scores, diagnostic_cfg["strong_motion_change_quantile"]
    )) if positive_scores else float("inf")
    subtle_candidates = [
        {
            **row,
            "classification": "stable-appearance / motion-change candidate",
            "appearance_change_threshold": appearance_threshold,
            "strong_motion_threshold": strong_motion_threshold,
            "semantic_action_claim": False,
        }
        for row in diagnostic_event_records
        if row["split"] and row["appearance_change"] <= appearance_threshold
        and row["max_motion_change_score"] >= strong_motion_threshold
    ]
    similarity_threshold = float(np.quantile(
        [row["dino_appearance_similarity"] for row in adjacency_records],
        diagnostic_cfg["similar_dino_quantile"],
    )) if adjacency_records else 1.0
    weak_motion_threshold = float(np.quantile(
        [row["motion_change_score"] for row in adjacency_records],
        diagnostic_cfg["weak_motion_quantile"],
    )) if adjacency_records else 0.0
    fragmentation_rows = []
    for metric, action_row, motion_row in zip(metrics, action_rows, motion_rows):
        video_id = metric["video_id"]
        internal_boundaries = sorted(
            point["timestamp"]
            for decision in motion_row["event_decisions"]
            for point in decision["change_points"]
        )
        nearby = [
            {"left": left, "right": right, "gap_sec": right - left}
            for left, right in zip(internal_boundaries[:-1], internal_boundaries[1:])
            if right - left < float(diagnostic_cfg["nearby_boundary_sec"])
        ]
        redundant = [
            {
                **row,
                "reason": "high adjacent DINO appearance similarity and weak motion-regime change",
                "similarity_threshold": similarity_threshold,
                "weak_motion_threshold": weak_motion_threshold,
            }
            for row in adjacency_records
            if row["video_id"] == video_id
            and row["dino_appearance_similarity"] >= similarity_threshold
            and row["motion_change_score"] <= weak_motion_threshold
        ]
        fragmentation_rows.append(
            {
                "video_id": video_id,
                "short_action_count_lt_2s": metric["action_level"]["count_lt_2s"],
                "short_action_fraction_lt_2s": metric["action_level"]["fraction_lt_2s"],
                "short_action_count_lt_4s": metric["action_level"]["count_lt_4s"],
                "short_action_fraction_lt_4s": metric["action_level"]["fraction_lt_4s"],
                "repeated_nearby_boundary_count": len(nearby),
                "repeated_nearby_boundaries": nearby,
                "possible_redundant_fragment_count": len(redundant),
                "possible_redundant_fragments": redundant,
                "automatic_flags_are_structural_not_semantic_judgments": True,
            }
        )
    fragmentation_by_video = {row["video_id"]: row for row in fragmentation_rows}
    for metric in metrics:
        metric["subtle_action_candidate_count"] = sum(
            row["video_id"] == metric["video_id"] for row in subtle_candidates
        )
        metric["fragmentation"] = fragmentation_by_video[metric["video_id"]]

    all_group = aggregate_group(metrics)
    problematic = aggregate_group([row for row in metrics if row["group"] == "problematic"])
    control = aggregate_group([row for row in metrics if row["group"] == "control"])
    aggregate = {
        "experiment_id": config["experiment_id"],
        "video_count": len(metrics),
        "motion_method": "LIGHTWEIGHT_MOTION_PROXY",
        "all": all_group,
        "problematic": problematic,
        "control": control,
        "subtle_action_candidate_count": len(subtle_candidates),
        "videos_with_subtle_action_candidates": sorted({row["video_id"] for row in subtle_candidates}),
        "fragmentation": {
            "total_repeated_nearby_boundaries": sum(row["repeated_nearby_boundary_count"] for row in fragmentation_rows),
            "total_possible_redundant_fragments": sum(row["possible_redundant_fragment_count"] for row in fragmentation_rows),
            "data_derived_similarity_threshold": similarity_threshold,
            "data_derived_weak_motion_threshold": weak_motion_threshold,
        },
        "diagnostic_thresholds": {
            "stable_appearance_change": appearance_threshold,
            "strong_motion_change": strong_motion_threshold,
        },
        "known_stable_background_case": {
            key: value for key, value in next(
            row for row in metrics
            if row["video_id"] == "afbc1ef9-bc2b-49f9-9009-3d7486563764"
            ).items() if not key.startswith("_")
        },
    }
    clean_metrics = []
    for row in metrics:
        clean_metrics.append({key: value for key, value in row.items() if not key.startswith("_")})

    source_hashes = {label: sha256(path) for label, path in source_paths.items()}
    event_boundary_equivalence = all(
        row["source_boundary_hash"] == row["output_boundary_hash"] for row in event_rows
    )
    frozen_config = {
        **config,
        "source_artifact_hashes": source_hashes,
        "event_boundary_equivalence_10_of_10": event_boundary_equivalence,
        "raft_environment_audit": {
            "torch": "2.5.1+cu121",
            "torchvision": "0.20.1+cpu",
            "raft_small_constructor_available": True,
            "raft_small_weights_cached": False,
            "raft_small_official_file_size_mb": 3.821,
            "raft_large_constructor_available": True,
            "raft_large_weights_cached": False,
            "raft_large_official_file_size_mb": 20.129,
            "download_or_install_performed": False,
            "decision": "Use explicitly labelled LIGHTWEIGHT_MOTION_PROXY",
        },
    }
    run_manifest = {
        "experiment_id": config["experiment_id"],
        "source_manifest": config["input_manifest"],
        "source_manifest_sha256": source_hashes["manifest"],
        "video_ids": [row["video_id"] for row in manifest["videos"]],
        "video_count": len(manifest["videos"]),
        "questions_used": False,
        "gold_answers_used": False,
        "answer_options_used": False,
        "qa_correctness_used": False,
        "external_api_calls": 0,
        "new_dinov2_inference_calls": 0,
        "event_source_sha256": source_hashes["event_segments"],
        "event_boundaries_exact": event_boundary_equivalence,
    }
    total_motion_cache_size = sum(row["motion_cache_size_bytes"] for row in runtime_per_video)
    runtime = {
        "offline_only": True,
        "device": "CPU",
        "new_dinov2_feature_extraction_sec": 0.0,
        "new_dinov2_inference_calls": 0,
        "per_video": runtime_per_video,
        "mean_motion_load_or_extract_sec_per_video": statistics.fmean(row["motion_load_or_extract_sec"] for row in runtime_per_video),
        "mean_change_point_sec_per_video": statistics.fmean(row["change_point_sec"] for row in runtime_per_video),
        "mean_total_action_index_sec_per_video": statistics.fmean(row["total_action_index_sec"] for row in runtime_per_video),
        "total_action_index_sec": sum(row["total_action_index_sec"] for row in runtime_per_video),
        "motion_cache_size_bytes": total_motion_cache_size,
        "motion_cache_size_mib": total_motion_cache_size / (1024 ** 2),
        "potential_online_cost_note": "Node counts are a structural proxy only; no online QA/retrieval was run.",
    }

    write_jsonl(OUT / "event_level_segments.jsonl", event_rows)
    write_jsonl(OUT / "action_level_segments.jsonl", action_rows)
    runtime["event_level_index_size_bytes"] = (OUT / "event_level_segments.jsonl").stat().st_size
    runtime["action_level_index_size_bytes"] = (OUT / "action_level_segments.jsonl").stat().st_size
    runtime["action_minus_event_index_size_bytes"] = (
        runtime["action_level_index_size_bytes"] - runtime["event_level_index_size_bytes"]
    )
    write_json(OUT / "parent_child_mapping.json", {"videos": mapping_rows})
    write_jsonl(OUT / "motion_signals.jsonl", motion_rows)
    write_json(OUT / "per_video_metrics.json", clean_metrics)
    write_json(OUT / "aggregate_metrics.json", aggregate)
    write_json(OUT / "subtle_action_candidates.json", subtle_candidates)
    write_json(OUT / "fragmentation_diagnostics.json", fragmentation_rows)
    write_json(OUT / "runtime_metrics.json", runtime)
    write_json(OUT / "frozen_config.json", frozen_config)
    write_json(OUT / "run_manifest.json", run_manifest)
    render_comparison(
        output_path=OUT / "comparison.html", manifest=manifest,
        event_rows=event_rows, action_rows=action_rows, kts_rows=kts_rows,
        motion_rows=motion_rows, per_video_metrics=clean_metrics,
        aggregate_metrics=aggregate, subtle_candidates=subtle_candidates,
        fragmentation=fragmentation_rows,
    )
    readme = f"""# Event vs Action Fine Leaf v0.1

Offline structural comparison on the exact frozen 10-video manifest.

- `EVENT_LEVEL` exactly reuses the prior CoMET-style DINOv2 boundaries.
- `ACTION_LEVEL` uses the explicitly labelled `LIGHTWEIGHT_MOTION_PROXY`: 1 FPS grayscale frame differences, edge changes, and spatial-grid changes, followed by Gaussian smoothing and exact penalized RBF change-point optimization inside each Event.
- It is not RAFT and not an exact CoMET Action-level reproduction.
- New DINO inference: 0. External/API calls: 0.
- Questions/options/gold/QA correctness used: no.
- Event boundary equality: {event_boundary_equivalence}.

Open `comparison.html` for aligned timelines, 25/50/75 thumbnails, per-split motion curves, and blank manual-review fields.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "output": OUT.as_posix(), "videos": len(metrics),
        "event_boundaries_exact": event_boundary_equivalence,
        "event_segments": all_group["event_level"]["total_segments"],
        "action_segments": all_group["action_level"]["total_segments"],
        "subtle_candidates": len(subtle_candidates),
        "external_api_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
