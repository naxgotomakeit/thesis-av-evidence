from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def frame_timestamps(frame_count: int, fps: float, video_duration: float | None = None) -> list[float]:
    if frame_count < 0 or fps <= 0:
        raise ValueError("frame_count must be non-negative and fps must be positive")
    timestamps = [index / fps for index in range(frame_count)]
    if video_duration is not None:
        timestamps = [min(value, float(video_duration)) for value in timestamps]
    return timestamps


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def adjacent_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.empty((0,), dtype=np.float32)
    normalized = normalize_embeddings(np.asarray(embeddings, dtype=np.float32))
    return (1.0 - np.sum(normalized[:-1] * normalized[1:], axis=1)).astype(np.float32)


def segment_frame_indices(
    timestamps: list[float], distances: np.ndarray, threshold: float, min_region_duration_sec: float
) -> list[tuple[int, int]]:
    """Return inclusive frame-index ranges for temporally continuous stable regions."""
    count = len(timestamps)
    if count == 0:
        return []
    if count == 1:
        return [(0, 0)]
    if len(distances) != count - 1:
        raise ValueError("distances must contain one value per adjacent frame pair")
    starts = [0]
    for right_index, distance in enumerate(distances, start=1):
        elapsed = timestamps[right_index] - timestamps[starts[-1]]
        if float(distance) >= threshold and elapsed >= min_region_duration_sec:
            starts.append(right_index)
    ranges = [(start, starts[i + 1] - 1 if i + 1 < len(starts) else count - 1) for i, start in enumerate(starts)]
    if len(ranges) > 1:
        last_start, last_end = ranges[-1]
        last_duration = timestamps[last_end] - timestamps[last_start]
        if last_duration < min_region_duration_sec:
            ranges[-2] = (ranges[-2][0], last_end)
            ranges.pop()
    return ranges


def representative_frame_index(region_embeddings: np.ndarray, global_start_index: int = 0) -> int:
    if len(region_embeddings) == 0:
        raise ValueError("region_embeddings cannot be empty")
    normalized = normalize_embeddings(np.asarray(region_embeddings, dtype=np.float32))
    pooled = normalized.mean(axis=0)
    pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
    local_index = int(np.argmax(normalized @ pooled))
    return global_start_index + local_index


def build_regions(
    timestamps: list[float], embeddings: np.ndarray, frame_paths: list[Path], video_duration: float,
    threshold: float, min_region_duration_sec: float, output_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    distances = adjacent_cosine_distances(embeddings)
    ranges = segment_frame_indices(timestamps, distances, threshold, min_region_duration_sec)
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    pooled_embeddings = []
    regions = []
    frame_interval = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    for region_number, (start_index, end_index) in enumerate(ranges):
        portion = normalize_embeddings(embeddings[start_index : end_index + 1])
        pooled = portion.mean(axis=0)
        pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
        pooled_embeddings.append(pooled.astype(np.float32))
        representative = representative_frame_index(embeddings[start_index : end_index + 1], start_index)
        keyframe_path = keyframe_dir / f"region_{region_number:04d}_t{timestamps[representative]:08.3f}.jpg"
        shutil.copy2(frame_paths[representative], keyframe_path)
        internal_distances = distances[start_index:end_index]
        regions.append({
            "region_id": f"region_{region_number:04d}",
            "start_time": float(timestamps[start_index]),
            "end_time": float(min(video_duration, timestamps[end_index] + frame_interval)),
            "frame_timestamps": [float(x) for x in timestamps[start_index : end_index + 1]],
            "frame_indices": list(range(start_index, end_index + 1)),
            "region_pooled_embedding_path": "region_embeddings.npy",
            "region_embedding_index": region_number,
            "representative_keyframe_path": keyframe_path.relative_to(output_dir).as_posix(),
            "representative_frame_timestamp": float(timestamps[representative]),
            "mean_change_score": float(internal_distances.mean()) if len(internal_distances) else 0.0,
        })
    pooled_array = np.stack(pooled_embeddings) if pooled_embeddings else np.empty((0, embeddings.shape[1]), dtype=np.float32)
    return regions, pooled_array.astype(np.float32), distances


def validate_region_schema(region: dict[str, Any]) -> list[str]:
    required = {"region_id", "start_time", "end_time", "frame_timestamps", "frame_indices",
                "region_pooled_embedding_path", "region_embedding_index", "representative_keyframe_path",
                "representative_frame_timestamp", "mean_change_score"}
    errors = [f"missing:{field}" for field in sorted(required - region.keys())]
    if region.get("start_time", 0) > region.get("end_time", 0):
        errors.append("start_after_end")
    if not region.get("frame_timestamps"):
        errors.append("empty_frame_timestamps")
    return errors


def extract_frames(video_path: Path, frame_dir: Path, fps: float, ffmpeg_path: str) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink()
    pattern = frame_dir / "frame_%06d.jpg"
    command = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-i", str(video_path),
               "-vf", f"fps={fps}", "-q:v", "2", "-start_number", "0", str(pattern)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()[:1000]}")
    return sorted(frame_dir.glob("frame_*.jpg"))


def save_timeline(path: Path, distances: np.ndarray, regions: list[dict[str, Any]], duration: float,
                  threshold: float, weak_start: float | None, weak_end: float | None) -> None:
    width, height, margin = 1200, 300, 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_left, plot_right, plot_top, plot_bottom = margin, width - 20, 30, height - 50
    if weak_start is not None and weak_end is not None and duration > 0:
        x0 = plot_left + int((weak_start / duration) * (plot_right - plot_left))
        x1 = plot_left + int((weak_end / duration) * (plot_right - plot_left))
        draw.rectangle((x0, plot_top, x1, plot_bottom), fill=(255, 225, 235))
    maximum = max(float(distances.max()) if len(distances) else 0.0, threshold, 1e-6)
    threshold_y = plot_bottom - int((threshold / maximum) * (plot_bottom - plot_top))
    draw.line((plot_left, threshold_y, plot_right, threshold_y), fill=(220, 60, 60), width=2)
    points = []
    for index, value in enumerate(distances):
        time_value = index + 1
        x = plot_left + int((time_value / max(duration, 1.0)) * (plot_right - plot_left))
        y = plot_bottom - int((float(value) / maximum) * (plot_bottom - plot_top))
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(40, 90, 180), width=2)
    for region in regions[1:]:
        x = plot_left + int((region["start_time"] / max(duration, 1.0)) * (plot_right - plot_left))
        draw.line((x, plot_top, x, plot_bottom), fill=(20, 150, 80), width=2)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black")
    draw.text((plot_left, 4), "Adjacent-frame CLIP cosine distance", fill="black")
    draw.text((plot_left, 15), "Pink: Dataset-provided QA reference interval (not model prediction; not retrieval result; not gold boundary)", fill="black")
    draw.text((plot_left, plot_bottom + 12), "0s", fill="black")
    draw.text((plot_right - 70, plot_bottom + 12), f"{duration:.1f}s", fill="black")
    image.save(path)


def save_contact_sheet(path: Path, output_dir: Path, regions: list[dict[str, Any]]) -> None:
    thumb_w, thumb_h, label_h, columns = 240, 150, 34, 4
    rows = max(1, math.ceil(len(regions) / columns))
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, region in enumerate(regions):
        image = Image.open(output_dir / region["representative_keyframe_path"]).convert("RGB")
        thumb = ImageOps.fit(image, (thumb_w, thumb_h))
        x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + label_h)
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb_h + 3), f"{region['region_id']}  {region['start_time']:.1f}-{region['end_time']:.1f}s", fill="black")
    sheet.save(path, quality=90)


def write_regions_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
