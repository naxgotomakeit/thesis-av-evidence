"""Local absolute and photometric-aware Fine-to-Medium frontier utilities."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from src.experiments.adaptive_fluid_hierarchy_v0_1.experiment import (
    build_tree_context,
    fine_leaf_hash,
    fluid_frontier,
    sha256_file,
    topology_hash,
)


TOLERANCE = 1e-6


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-12)


def robust_luminance_image(image: Image.Image, lower: float, upper: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    low, high = np.percentile(luminance, [lower, upper])
    if high - low > 1e-6:
        luminance = np.clip((luminance - low) * (255.0 / (high - low)), 0.0, 255.0)
    else:
        luminance = np.clip(luminance, 0.0, 255.0)
    array = np.repeat(luminance[..., None], 3, axis=2).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def frame_identity(frame_paths: list[Path]) -> str:
    payload = [
        {"path": path.as_posix(), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in frame_paths
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class NormalizedDINOExtractor:
    """Load the same frozen DINOv2 model once and cache grayscale-normalized CLS features."""

    def __init__(self, *, device: str, batch_size: int, lower: float, upper: float):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.lower = float(lower)
        self.upper = float(upper)
        started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-small", local_files_only=True, use_safetensors=True
        ).to(self.device)
        self.model.eval()
        self.load_sec = time.perf_counter() - started
        self.load_count = 1

    def extract_or_load(
        self, *, video_id: str, frame_paths: list[Path], timestamps: np.ndarray, cache_dir: Path
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        feature_path, metadata_path = cache_dir / "features.npy", cache_dir / "metadata.json"
        expected = {
            "schema_version": "photometric-normalized-dinov2-cache-v1",
            "video_id": video_id,
            "model": "facebook/dinov2-small",
            "feature": "last_hidden_state_cls_token",
            "normalization": "l2",
            "frame_count": len(frame_paths),
            "frame_identity_sha256": frame_identity(frame_paths),
            "timestamps": np.asarray(timestamps, dtype=np.float64).tolist(),
            "preprocessing": {
                "color": "ITU-R BT.601 luminance replicated to RGB",
                "contrast": "per-frame robust global percentile stretch",
                "lower_percentile": self.lower,
                "upper_percentile": self.upper,
            },
        }
        if feature_path.is_file() and metadata_path.is_file():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if {key: saved.get(key) for key in expected} == expected and saved.get("features_sha256") == sha256_file(feature_path):
                features = np.load(feature_path)
                if features.shape == (len(frame_paths), 384) and np.isfinite(features).all():
                    return features, {**saved, "cache_hit": True, "preprocessing_sec": 0.0, "inference_sec": 0.0}

        preprocess_sec = inference_sec = 0.0
        batches: list[np.ndarray] = []
        with self.torch.inference_mode():
            for offset in range(0, len(frame_paths), self.batch_size):
                started = time.perf_counter()
                images = []
                for path in frame_paths[offset : offset + self.batch_size]:
                    with Image.open(path) as image:
                        images.append(robust_luminance_image(image, self.lower, self.upper))
                preprocess_sec += time.perf_counter() - started
                started = time.perf_counter()
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.model(**inputs).last_hidden_state[:, 0, :]
                output = self.torch.nn.functional.normalize(output.float(), dim=-1)
                batches.append(output.cpu().numpy().astype(np.float32))
                if self.device.type == "cuda":
                    self.torch.cuda.synchronize()
                inference_sec += time.perf_counter() - started
        features = np.concatenate(batches, axis=0)
        np.save(feature_path, features)
        metadata = {
            **expected,
            "dtype": str(features.dtype),
            "dimension": int(features.shape[1]),
            "device": str(self.device),
            "batch_size": self.batch_size,
            "features_sha256": sha256_file(feature_path),
            "cache_hit": False,
            "preprocessing_sec": preprocess_sec,
            "inference_sec": inference_sec,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return features, metadata


def image_statistics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[..., channel], bins=16, range=(0.0, 1.0))
        hist = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
        histograms.append(hist)
    return {
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "rgb_histograms": np.asarray(histograms).tolist(),
    }


def _centroid(features: np.ndarray, indices: list[int]) -> np.ndarray | None:
    if not indices:
        return None
    return normalize(features[indices].mean(axis=0))


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    return float(np.clip(left @ right, -1.0, 1.0))


def boundary_diagnostics(
    hierarchy: dict[str, Any], frame_paths: list[Path], timestamps: np.ndarray,
    original_features: np.ndarray, normalized_features: np.ndarray,
    photo_profiles: dict[str, dict[str, float]], persistence_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    frame_stats = [image_statistics(path) for path in frame_paths]
    output = []
    for boundary in hierarchy["boundary_records"]:
        left_index = int(np.clip(int(boundary.get("signal_index", 0)), 0, len(frame_paths) - 2))
        right_index = left_index + 1
        original_change = float(1.0 - np.clip(original_features[left_index] @ original_features[right_index], -1.0, 1.0))
        normalized_change = float(1.0 - np.clip(normalized_features[left_index] @ normalized_features[right_index], -1.0, 1.0))
        left_stats, right_stats = frame_stats[left_index], frame_stats[right_index]
        mean_delta = abs(left_stats["luminance_mean"] - right_stats["luminance_mean"])
        std_delta = abs(left_stats["luminance_std"] - right_stats["luminance_std"])
        left_hist = np.asarray(left_stats["rgb_histograms"])
        right_hist = np.asarray(right_stats["rgb_histograms"])
        color_change = float(np.mean(0.5 * np.abs(left_hist - right_hist).sum(axis=1)))
        photometric_magnitude = max(mean_delta, std_delta, color_change)

        boundary_time = float(boundary["timestamp"])
        offsets = {
            "pre": persistence_cfg["pre_window_sec"],
            "post_near": persistence_cfg["post_near_window_sec"],
            "post_far": persistence_cfg["post_far_window_sec"],
        }
        groups: dict[str, list[int]] = {}
        for name, (lower, upper) in offsets.items():
            groups[name] = [
                index for index, timestamp in enumerate(timestamps)
                if boundary_time + float(lower) - TOLERANCE <= timestamp <= boundary_time + float(upper) + TOLERANCE
            ]
        pre, near, far = (_centroid(normalized_features, groups[name]) for name in ("pre", "post_near", "post_far"))
        sim_pre_near = _cosine(pre, near)
        sim_pre_far = _cosine(pre, far)
        sim_near_far = _cosine(near, far)
        change_pre_near = None if sim_pre_near is None else 1.0 - sim_pre_near
        change_pre_far = None if sim_pre_far is None else 1.0 - sim_pre_far
        persistent = bool(
            change_pre_near is not None and change_pre_far is not None and sim_near_far is not None
            and change_pre_near >= float(persistence_cfg["persistent_pre_near_change_min"])
            and change_pre_far >= float(persistence_cfg["persistent_pre_far_change_min"])
            and sim_near_far >= float(persistence_cfg["persistent_post_near_far_similarity_min"])
        )
        transient = bool(
            change_pre_near is not None and sim_pre_far is not None
            and change_pre_near >= float(persistence_cfg["transient_pre_near_change_min"])
            and sim_pre_far >= float(persistence_cfg["transient_pre_far_similarity_min"])
            and not persistent
        )
        photo_flags = {}
        for profile, values in photo_profiles.items():
            ratio = normalized_change / max(original_change, 1e-12)
            photo_flags[profile] = bool(
                original_change >= float(values["original_change_min"])
                and normalized_change <= float(values["normalized_change_max"])
                and photometric_magnitude >= float(values["photometric_magnitude_min"])
                and ratio <= float(values["normalized_to_original_ratio_max"])
            )
        output.append({
            "video_id": hierarchy["video_id"], "boundary_id": boundary["boundary_id"],
            "left_leaf_id": boundary["left_leaf_id"], "right_leaf_id": boundary["right_leaf_id"],
            "timestamp": boundary_time, "left_frame_path": frame_paths[left_index].as_posix(),
            "right_frame_path": frame_paths[right_index].as_posix(),
            "stored_raw_original_change": float(boundary["raw_change_strength"]),
            "original_dino_change": original_change, "normalized_dino_change": normalized_change,
            "normalized_to_original_ratio": normalized_change / max(original_change, 1e-12),
            "luminance_mean_change": mean_delta, "luminance_std_change": std_delta,
            "color_histogram_change": color_change, "photometric_magnitude": photometric_magnitude,
            "suspected_photometric_by_profile": photo_flags,
            "pre_post_near_similarity": sim_pre_near, "pre_post_far_similarity": sim_pre_far,
            "post_near_post_far_similarity": sim_near_far,
            "pre_post_near_change": change_pre_near, "pre_post_far_change": change_pre_far,
            "persistent_state_change": persistent, "transient_or_reverting_change": transient,
            "window_frame_counts": {name: len(indices) for name, indices in groups.items()},
        })
    return output


def native_merge_components(
    hierarchy: dict[str, Any], boundary_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    context = build_tree_context(hierarchy)
    nodes = context["nodes"]
    output = {}
    for merge in hierarchy["accepted_merge_order"]:
        node_id = str(merge["parent_id"])
        node = nodes[node_id]
        left, right = (nodes[str(item)] for item in node["child_ids"])
        combined_count = float(left["frame_count"] + right["frame_count"])
        before = (
            float(left["internal_variability"]) * float(left["frame_count"])
            + float(right["internal_variability"]) * float(right["frame_count"])
        ) / max(combined_count, 1.0)
        increase = max(0.0, float(merge["merged_internal_variability"]) - before)
        pressure = 0.5 * (
            float(node["duration"]) / float(hierarchy["video_duration"])
            + len(node["leaf_ids"]) / len(context["fine_ids"])
        )
        boundary_id = None
        left_leaf, right_leaf = str(left["leaf_ids"][-1]), str(right["leaf_ids"][0])
        for item in hierarchy["boundary_records"]:
            if item["left_leaf_id"] == left_leaf and item["right_leaf_id"] == right_leaf:
                boundary_id = str(item["boundary_id"])
                break
        if boundary_id is None or boundary_id not in boundary_by_id:
            raise RuntimeError(f"Missing frozen inter-boundary for {node_id}")
        output[node_id] = {
            "node_id": node_id, "boundary_id": boundary_id,
            "semantic_difference": float(merge["semantic_difference"]),
            "inter_boundary_strength": float(merge["inter_boundary_strength"]),
            "merged_internal_variability": float(merge["merged_internal_variability"]),
            "merged_variability_increase": increase,
            "duration_component_pressure": pressure,
        }
    return output


def absolute_confidences(
    hierarchy: dict[str, Any], boundary_diagnostics_by_id: dict[str, dict[str, Any]],
    *, photo_profile: str | None = None, use_temporal: bool = False,
    photo_downweight: float = 0.75, transient_downweight: float = 0.75,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    components = native_merge_components(hierarchy, boundary_diagnostics_by_id)
    confidence, details = {}, {}
    for node_id, item in components.items():
        boundary = boundary_diagnostics_by_id[item["boundary_id"]]
        photo = bool(photo_profile and boundary["suspected_photometric_by_profile"][photo_profile])
        transient = bool(use_temporal and boundary["transient_or_reverting_change"])
        persistent = bool(boundary["persistent_state_change"])
        effective_strength = float(item["inter_boundary_strength"])
        if photo:
            effective_strength *= 1.0 - float(photo_downweight)
        if transient and not persistent:
            effective_strength *= 1.0 - float(transient_downweight)
        values = {
            "semantic_similarity": float(np.clip(1.0 - item["semantic_difference"] / 2.0, 0.0, 1.0)),
            "boundary_weakness": float(np.clip(1.0 - effective_strength / 2.0, 0.0, 1.0)),
            "variability_stability": float(np.clip(1.0 - item["merged_variability_increase"], 0.0, 1.0)),
            "duration_component_safety": float(np.clip(1.0 - item["duration_component_pressure"], 0.0, 1.0)),
        }
        confidence[node_id] = statistics.fmean(values.values())
        details[node_id] = {
            **item, **values, "absolute_merge_confidence": confidence[node_id],
            "effective_boundary_strength": effective_strength,
            "photometric_downweight_applied": photo,
            "temporal_downweight_applied": transient and not persistent,
            "persistent_boundary": persistent,
        }
    return confidence, details


def absolute_frontier(
    hierarchy: dict[str, Any], confidences: dict[str, float], details: dict[str, dict[str, Any]],
    *, minimum_confidence: float, maximum_local_drop: float, method: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    context = build_tree_context(hierarchy)
    nodes = context["nodes"]
    drops = {}
    for node_id in confidences:
        children = [confidences[str(child)] for child in nodes[node_id]["child_ids"] if str(child) in confidences]
        drops[node_id] = max(children) - confidences[node_id] if children else 0.0
    selected, trace = [], []

    def visit(node_id: str) -> None:
        node = nodes[node_id]
        if not node["child_ids"]:
            selected.append(node_id)
            trace.append({"method": method, "node_id": node_id, "decision": "KEEP_LEAF", "reason": "leaf"})
            return
        quality_pass = confidences[node_id] >= minimum_confidence
        drop_pass = drops[node_id] <= maximum_local_drop
        collapse = quality_pass and drop_pass
        trace.append({
            "method": method, "node_id": node_id, "start": node["start"], "end": node["end"],
            "absolute_merge_confidence": confidences[node_id], "local_absolute_drop": drops[node_id],
            "minimum_confidence": minimum_confidence, "maximum_local_drop": maximum_local_drop,
            "decision": "COLLAPSE" if collapse else "RECURSE",
            "failed_conditions": [name for name, passed in (("confidence", quality_pass), ("local_drop", drop_pass)) if not passed],
            "components": details[node_id],
        })
        if collapse:
            selected.append(node_id)
        else:
            for child in node["child_ids"]:
                visit(str(child))

    visit(context["root_id"])
    selected.sort(key=lambda item: (float(nodes[item]["start"]), float(nodes[item]["end"]), item))
    return selected, trace


def validate_medium_frontier(hierarchy: dict[str, Any], medium_ids: list[str]) -> dict[str, Any]:
    context = build_tree_context(hierarchy)
    nodes, fine_ids = context["nodes"], set(context["fine_ids"])
    ordered = sorted(medium_ids, key=lambda item: (float(nodes[item]["start"]), item))
    owners: dict[str, str] = {}
    duplicate = 0
    for medium_id in ordered:
        for leaf in nodes[medium_id]["leaf_ids"]:
            duplicate += int(leaf in owners)
            owners[str(leaf)] = medium_id
    gaps = overlaps = 0
    for left, right in zip(ordered, ordered[1:]):
        delta = float(nodes[right]["start"]) - float(nodes[left]["end"])
        gaps += int(delta > TOLERANCE)
        overlaps += int(delta < -TOLERANCE)
    coverage = sum(float(nodes[item]["duration"]) for item in ordered) / float(hierarchy["video_duration"])
    valid = set(owners) == fine_ids and duplicate == gaps == overlaps == 0 and abs(coverage - 1.0) <= 1e-5
    return {
        "valid": valid, "fine_leaf_preservation_rate": len(set(owners) & fine_ids) / len(fine_ids),
        "missing_fine_count": len(fine_ids - set(owners)), "multi_parent_violations": duplicate,
        "gap_count": gaps, "overlap_count": overlaps, "temporal_coverage_ratio": coverage,
    }


def boundary_crossing_counts(
    hierarchy: dict[str, Any], medium_ids: list[str], diagnostics: list[dict[str, Any]], profile: str,
) -> dict[str, Any]:
    context = build_tree_context(hierarchy)
    owner = {}
    for medium_id in medium_ids:
        for leaf in context["nodes"][medium_id]["leaf_ids"]:
            owner[str(leaf)] = medium_id
    rows = []
    for boundary in diagnostics:
        crossed = owner[boundary["left_leaf_id"]] == owner[boundary["right_leaf_id"]]
        rows.append({
            "boundary_id": boundary["boundary_id"], "timestamp": boundary["timestamp"], "crossed": crossed,
            "suspected_photometric": boundary["suspected_photometric_by_profile"][profile],
            "persistent": boundary["persistent_state_change"],
            "transient": boundary["transient_or_reverting_change"],
        })
    return {
        "rows": rows,
        "suspected_photometric_total": sum(item["suspected_photometric"] for item in rows),
        "suspected_photometric_crossed": sum(item["suspected_photometric"] and item["crossed"] for item in rows),
        "suspected_photometric_retained": sum(item["suspected_photometric"] and not item["crossed"] for item in rows),
        "persistent_total": sum(item["persistent"] for item in rows),
        "persistent_crossed": sum(item["persistent"] and item["crossed"] for item in rows),
        "persistent_retained": sum(item["persistent"] and not item["crossed"] for item in rows),
        "transient_total": sum(item["transient"] for item in rows),
        "transient_crossed": sum(item["transient"] and item["crossed"] for item in rows),
        "transient_retained": sum(item["transient"] and not item["crossed"] for item in rows),
    }


def representative(node: dict[str, Any]) -> dict[str, Any]:
    for preferred in ("dinov2_medoid", "temporal_50_percent"):
        for item in node.get("representative_frames", []):
            if item["kind"] == preferred:
                return item
    return node["representative_frames"][0]


def medium_metrics(
    hierarchy: dict[str, Any], medium_ids: list[str], original_features: np.ndarray,
    diagnostics: list[dict[str, Any]], profile: str,
) -> dict[str, Any]:
    context = build_tree_context(hierarchy)
    nodes = context["nodes"]
    durations = [float(nodes[item]["duration"]) for item in medium_ids]
    reps = [normalize(np.asarray(nodes[item]["pooled_dinov2"], dtype=np.float32)) for item in medium_ids]
    similarities = [float(np.clip(left @ right, -1.0, 1.0)) for left, right in zip(reps, reps[1:])]
    crossing = boundary_crossing_counts(hierarchy, medium_ids, diagnostics, profile)
    boundary_by_id = {item["boundary_id"]: item for item in diagnostics}
    risks = []
    for medium_id in medium_ids:
        node = nodes[medium_id]
        internal = [boundary_by_id[item] for item in node.get("internal_boundary_ids", []) if item in boundary_by_id]
        strongest_original = max((item["original_dino_change"] for item in internal), default=0.0)
        strongest_normalized = max((item["normalized_dino_change"] for item in internal), default=0.0)
        children = [nodes[str(item)] for item in node["child_ids"]]
        imbalance = None
        if len(children) == 2:
            imbalance = max(float(item["duration"]) for item in children) / max(float(node["duration"]), 1e-12)
        low_coherence_giant = float(node["duration"]) / float(hierarchy["video_duration"]) >= 0.3 and float(node["internal_variability"]) >= 0.3
        chaining = imbalance is not None and imbalance >= 0.85 and len(node["leaf_ids"]) >= 4
        risks.append({
            "node_id": medium_id, "start": node["start"], "end": node["end"], "duration": node["duration"],
            "strongest_internal_original_boundary": strongest_original,
            "strongest_internal_normalized_boundary": strongest_normalized,
            "internal_visual_coherence": float(1.0 - node["internal_variability"]),
            "low_coherence_giant": low_coherence_giant, "chaining_risk": chaining,
            "flagged": low_coherence_giant or chaining,
        })
    high_sim_weak = 0
    ordered_boundaries = {round(float(item["timestamp"]), 6): item for item in hierarchy["boundary_records"]}
    for index, similarity in enumerate(similarities):
        boundary = ordered_boundaries.get(round(float(nodes[medium_ids[index]]["end"]), 6))
        if similarity > 0.9 and boundary and float(boundary["strength_empirical_percentile"]) <= 0.5:
            high_sim_weak += 1
    return {
        "medium_count": len(medium_ids), "median_duration_sec": statistics.median(durations),
        "p90_duration_sec": float(np.percentile(durations, 90)), "max_duration_sec": max(durations),
        "adjacent_similarity_mean": statistics.fmean(similarities) if similarities else None,
        "adjacent_similarity_median": statistics.median(similarities) if similarities else None,
        "adjacent_similarity_counts": {str(value): sum(item > value for item in similarities) for value in (0.85, 0.9, 0.95)},
        "high_similarity_weak_boundary_pairs": high_sim_weak,
        "boundary_crossing": crossing, "risk_rows": risks,
        "overmerge_risk_flag_count": sum(item["flagged"] for item in risks),
    }


def complexity_alignment(
    hierarchy: dict[str, Any], method_medium_ids: dict[str, list[str]], diagnostics: list[dict[str, Any]],
    window_sec: float,
) -> dict[str, Any]:
    context = build_tree_context(hierarchy)
    nodes = context["nodes"]
    windows = []
    for start in np.arange(0.0, float(hierarchy["video_duration"]), window_sec):
        end = min(float(hierarchy["video_duration"]), float(start + window_sec))
        duration_min = max((end - start) / 60.0, 1e-12)
        inside = [item for item in diagnostics if start < float(item["timestamp"]) < end]
        normalized_density = sum(item["normalized_dino_change"] for item in inside) / duration_min
        persistent_density = sum(item["persistent_state_change"] for item in inside) / duration_min
        medium_density = {
            method: sum(start <= 0.5 * (float(nodes[item]["start"]) + float(nodes[item]["end"])) < end for item in ids) / duration_min
            for method, ids in method_medium_ids.items()
        }
        windows.append({
            "start": float(start), "end": end, "normalized_change_density": normalized_density,
            "persistent_boundary_density": persistent_density, "medium_density_by_method": medium_density,
        })

    def rho(left: list[float], right: list[float]) -> float | None:
        if len(left) < 2 or np.ptp(left) <= 1e-12 or np.ptp(right) <= 1e-12:
            return None
        return float(spearmanr(left, right).statistic)

    return {
        "video_id": hierarchy["video_id"], "window_duration_sec": window_sec, "windows": windows,
        "correlations": {
            method: {
                "normalized_change_spearman": rho(
                    [item["normalized_change_density"] for item in windows],
                    [item["medium_density_by_method"][method] for item in windows],
                ),
                "persistent_boundary_spearman": rho(
                    [item["persistent_boundary_density"] for item in windows],
                    [item["medium_density_by_method"][method] for item in windows],
                ),
            }
            for method in method_medium_ids
        },
    }


def copy_representative_asset(
    *, node: dict[str, Any], root: Path, output_assets: Path, video_id: str,
) -> dict[str, Any]:
    item = representative(node)
    source = Path(item["frame_path"])
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_assets / video_id / f'{node["node_id"]}__{source.name}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or sha256_file(destination) != sha256_file(source):
        shutil.copy2(source, destination)
    return {
        "node_id": node["node_id"], "timestamp": item["timestamp"], "kind": item["kind"],
        "source_path": source.as_posix(), "html_relative_path": destination.relative_to(output_assets.parent).as_posix(),
    }
