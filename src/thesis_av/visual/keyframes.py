"""Exact frozen informative Medium keyframe selection and story parsing."""
from __future__ import annotations
import hashlib
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def node_map(hierarchy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["node_id"]): row for row in hierarchy["nodes"]}

def hierarchy_views(hierarchy: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose frozen reference-cut membership without modifying any node."""
    nodes = node_map(hierarchy)
    medium_ids = list(hierarchy["cuts"]["medium"]["node_ids"])
    coarse_ids = list(hierarchy["cuts"]["coarse"]["node_ids"])
    output: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for coarse_id in coarse_ids:
        coarse = nodes[coarse_id]
        coarse_leaves = set(coarse["leaf_ids"])
        medium = [nodes[item] for item in medium_ids if set(nodes[item]["leaf_ids"]) <= coarse_leaves]
        medium.sort(key=lambda row: (float(row["start"]), str(row["node_id"])))
        assigned.update(str(row["node_id"]) for row in medium)
        output.append({"coarse": coarse, "medium": medium})
    output.sort(key=lambda row: (float(row["coarse"]["start"]), str(row["coarse"]["node_id"])))
    if assigned != set(medium_ids):
        raise RuntimeError("Frozen Medium-to-Coarse membership is incomplete")
    return output

def _minmax(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]

def _quality(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    laplace = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    sharpness = float(np.var(laplace))
    mean_luma = float(gray.mean())
    clipped_fraction = float(((gray < 0.035) | (gray > 0.965)).mean())
    exposure = max(0.0, 1.0 - abs(mean_luma - 0.5) / 0.5) * (1.0 - clipped_fraction)
    histogram, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=False)
    probabilities = histogram.astype(np.float64) / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 5.0)
    return {
        "raw_sharpness_laplacian_variance": sharpness,
        "raw_mean_luminance": mean_luma,
        "raw_clipped_pixel_fraction": clipped_fraction,
        "raw_exposure_score": exposure,
        "raw_normalized_entropy": entropy,
    }

def _candidate_frames(node: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in node.get("representative_frames", []):
        path = str(item["frame_path"])
        if path in seen:
            continue
        seen.add(path)
        output.append(
            {"timestamp": float(item["timestamp"]), "kind": str(item["kind"]), "source_image_path": path}
        )
    if not output:
        raise RuntimeError(f"Medium node has no representative candidates: {node['node_id']}")
    return sorted(output, key=lambda row: (row["timestamp"], row["source_image_path"]))

def _nearest_index(timestamps: np.ndarray, timestamp: float) -> int:
    return int(np.argmin(np.abs(timestamps - float(timestamp))))

def _copy_asset(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = sha256_file(source)
    if destination.is_file():
        if sha256_file(destination) != identity:
            raise RuntimeError(f"Existing keyframe asset differs from source: {destination}")
    else:
        shutil.copy2(source, destination)
    return {"source_sha256": identity, "bytes": destination.stat().st_size}

def select_video_keyframes(
    *, hierarchy: dict[str, Any], features: np.ndarray, timestamps: np.ndarray,
    root: Path, output_dir: Path, weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Select one image per Medium using existing DINO and cheap image-quality signals."""
    video_id = str(hierarchy["video_id"])
    views = hierarchy_views(hierarchy)
    medium_nodes = [medium for view in views for medium in view["medium"]]
    pooled = np.asarray([node["pooled_dinov2"] for node in medium_nodes], dtype=np.float32)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    records: list[dict[str, Any]] = []
    prep_total = quality_total = representation_total = 0.0
    for medium_index, node in enumerate(medium_nodes):
        total_started = time.perf_counter()
        prep_started = time.perf_counter()
        candidates = _candidate_frames(node)
        for candidate in candidates:
            source = Path(candidate["source_image_path"])
            if not source.is_absolute():
                source = root / source
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = output_dir / "assets" / video_id / str(node["node_id"]) / source.name
            identity = _copy_asset(source, destination)
            candidate.update(
                {
                    "source_image_path": source.as_posix(),
                    "image_path": destination.as_posix(),
                    "html_relative_path": destination.relative_to(output_dir).as_posix(),
                    **identity,
                }
            )
        preparation_sec = time.perf_counter() - prep_started

        quality_started = time.perf_counter()
        qualities = [_quality(Path(candidate["image_path"])) for candidate in candidates]
        quality_sec = time.perf_counter() - quality_started

        representation_started = time.perf_counter()
        candidate_indices = [_nearest_index(timestamps, candidate["timestamp"]) for candidate in candidates]
        candidate_features = features[candidate_indices]
        node_pooled = pooled[medium_index]
        representative = [float(feature @ node_pooled) for feature in candidate_features]
        stability = []
        for frame_index, feature in zip(candidate_indices, candidate_features):
            neighbor_indices = [index for index in (frame_index - 1, frame_index + 1) if 0 <= index < len(features)]
            stability.append(
                float(np.mean([feature @ features[index] for index in neighbor_indices])) if neighbor_indices else 1.0
            )
        if len(medium_nodes) > 1:
            other = np.delete(pooled, medium_index, axis=0)
            discriminative = [1.0 - float(np.max(other @ feature)) for feature in candidate_features]
        else:
            discriminative = [1.0 for _ in candidates]
        components = {
            "representativeness": representative,
            "sharpness": [item["raw_sharpness_laplacian_variance"] for item in qualities],
            "exposure": [item["raw_exposure_score"] for item in qualities],
            "entropy": [item["raw_normalized_entropy"] for item in qualities],
            "temporal_stability": stability,
            "cross_medium_discriminativeness": discriminative,
        }
        normalized = {key: _minmax(values) for key, values in components.items()}
        for candidate_index, candidate in enumerate(candidates):
            candidate["feature_index"] = candidate_indices[candidate_index]
            candidate["raw_score_components"] = {
                "representativeness_cosine": representative[candidate_index],
                "temporal_stability_cosine": stability[candidate_index],
                "cross_medium_discriminativeness": discriminative[candidate_index],
                **qualities[candidate_index],
            }
            candidate["normalized_score_components"] = {
                key: values[candidate_index] for key, values in normalized.items()
            }
            candidate["selection_score"] = float(
                sum(weights[key] * normalized[key][candidate_index] for key in weights)
            )
        selected = min(
            candidates,
            key=lambda row: (-float(row["selection_score"]), float(row["timestamp"]), str(row["image_path"])),
        )
        representation_sec = time.perf_counter() - representation_started
        total_sec = time.perf_counter() - total_started
        prep_total += preparation_sec
        quality_total += quality_sec
        representation_total += representation_sec
        records.append(
            {
                "video_id": video_id,
                "coarse_id": next(
                    str(view["coarse"]["node_id"])
                    for view in views if str(node["node_id"]) in {str(item["node_id"]) for item in view["medium"]}
                ),
                "medium_id": str(node["node_id"]),
                "start": float(node["start"]),
                "end": float(node["end"]),
                "duration": float(node["duration"]),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selected_keyframe": dict(selected),
                "selection_rule": "weighted cheap quality + DINO representativeness/stability/discriminativeness",
                "selection_reason": (
                    "maximum frozen weighted score; VLM was not used; deterministic earlier-timestamp/path tie-break"
                ),
                "timing": {
                    "candidate_frame_preparation_sec": preparation_sec,
                    "keyframe_quality_scoring_sec": quality_sec,
                    "keyframe_representative_information_scoring_sec": representation_sec,
                    "total_keyframe_selection_sec": total_sec,
                },
            }
        )
    return records, {
        "candidate_frame_preparation_sec": prep_total,
        "keyframe_quality_scoring_sec": quality_total,
        "keyframe_representative_information_scoring_sec": representation_total,
        "total_keyframe_selection_sec": sum(item["timing"]["total_keyframe_selection_sec"] for item in records),
    }

def clean_caption(raw: str) -> str:
    return " ".join(raw.strip().split())

def parse_story(raw: str) -> tuple[str, list[str]]:
    """Parse requested headings even when the model emits both on one line."""
    text = raw.strip()
    split = re.split(r"high[- ]level storyline\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)
    overall_part = split[0].strip()
    storyline_part = split[1].strip() if len(split) == 2 else ""
    overall = re.sub(r"^overall story\s*:\s*", "", overall_part, flags=re.IGNORECASE).strip()
    storyline: list[str] = []
    if storyline_part:
        numbered = re.split(r"(?:^|\n|\s)(?=\d+[.)]\s+)", storyline_part)
        storyline = [re.sub(r"^\d+[.)]\s+", "", item.strip()).strip() for item in numbered if item.strip()]
        if not storyline:
            storyline = [clean_caption(storyline_part)]
    if not overall:
        overall = clean_caption(text)
    return clean_caption(overall), [clean_caption(item) for item in storyline if clean_caption(item)]

