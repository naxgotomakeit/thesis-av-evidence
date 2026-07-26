"""Offline Formal E2 index construction adapters.

No function in this module calls the 7B answer model or consumes QA ground
truth.  It prepares one shared canonical-clip Fine artifact and derives B1
Fine-only and B2 Fluid-Loose Medium candidate units from it.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.thesis_av.visual.comet_style import CometStyleConfig, segment as comet_segment
from src.thesis_av.visual.hierarchy import build_boundary_records, build_fine_nodes, build_safe_hierarchy
from src.thesis_av.visual.medium import build_tree_context, fluid_frontier
from src.thesis_av.visual.segmentation_schema import make_segmentation

from .core import build_event_records, build_b1_prime_groups


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonical_one_fps_requested_timestamps(*, clip_start_s: float, clip_end_s: float) -> np.ndarray:
    """Frozen relative 1FPS bins, represented in parent-video seconds.

    The temporal contract is the existing ``np.arange(round(duration))`` rule;
    values are offset into the parent only after that rule is fixed.  A clip
    shorter than 0.5 seconds is not silently densified and must be reported as
    an offline construction failure rather than using GT-aware sampling.
    """
    duration = float(clip_end_s) - float(clip_start_s)
    if duration <= 0:
        raise ValueError("Canonical clip duration must be positive")
    count = int(round(duration))
    if count < 1:
        raise ValueError("Frozen 1FPS np.arange(round(duration)) yields zero frames")
    return float(clip_start_s) + np.arange(count, dtype=np.float64)


def decode_canonical_one_fps_to_cache(
    *, parent_video_path: Path, clip_start_s: float, clip_end_s: float, output_dir: Path,
) -> tuple[list[Path], np.ndarray, np.ndarray, dict[str, Any]]:
    """Sequentially decode/cache the frozen canonical 1FPS sample set.

    This is the Amendment #4 behavior-preserving replacement for repeated
    random seeks.  It keeps the verified source-frame/timestamp/JPEG contract
    while making one canonical-start seek followed by an ordered decode pass.
    """
    import av
    from src.experiments.qaego4d_e1.core import _frame_index
    requested = canonical_one_fps_requested_timestamps(clip_start_s=clip_start_s, clip_end_s=clip_end_s)
    started = time.perf_counter()
    open_started = time.perf_counter()
    container = av.open(str(parent_video_path))
    parent_open_s = time.perf_counter() - open_started
    stream = container.streams.video[0]
    seek_started = time.perf_counter()
    container.seek(int(float(clip_start_s) / float(stream.time_base)), stream=stream, any_frame=False, backward=True)
    seek_s = time.perf_counter() - seek_started
    decode_s = selection_s = rgb_s = image_encode_s = storage_write_s = storage_metadata_s = 0.0
    images: list[Any] = []
    records: list[dict[str, Any]] = []
    previous = None
    target_index = 0
    decoded_count = 0
    seen: set[int] = set()
    try:
        iterator = container.decode(stream)
        while target_index < len(requested):
            decode_started = time.perf_counter()
            try:
                frame = next(iterator)
            except StopIteration:
                decode_s += time.perf_counter() - decode_started
                break
            decode_s += time.perf_counter() - decode_started
            decoded_count += 1
            select_started = time.perf_counter()
            if frame.time is None:
                selection_s += time.perf_counter() - select_started
                continue
            timestamp = float(frame.time)
            if timestamp < clip_start_s:
                selection_s += time.perf_counter() - select_started
                continue
            while target_index < len(requested) and timestamp >= float(requested[target_index]):
                target = float(requested[target_index])
                candidate = frame
                if previous is not None and abs(float(previous.time) - target) <= abs(timestamp - target):
                    candidate = previous
                pts = int(candidate.pts)
                if pts not in seen:
                    seen.add(pts)
                    convert_started = time.perf_counter()
                    image = candidate.to_image().convert("RGB")
                    rgb_s += time.perf_counter() - convert_started
                    images.append(image)
                    records.append({
                        "requested_timestamp_sec": target,
                        "actual_timestamp_sec": float(candidate.time),
                        "source_frame_index": _frame_index(candidate, stream),
                        "source_pts": pts,
                    })
                target_index += 1
            previous = frame
            selection_s += time.perf_counter() - select_started
            if timestamp >= clip_end_s and target_index >= len(requested):
                break
    finally:
        container.close()
    if target_index != len(requested):
        raise RuntimeError("Sequential canonical decoder did not resolve every frozen 1FPS target")
    if len(images) != len(requested):
        raise RuntimeError("Frozen 1FPS targets unexpectedly deduplicated; cannot silently change the timeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        for index, image in enumerate(images):
            path = output_dir / f"frame_{index:05d}.jpg"
            if not path.is_file():
                temporary = path.with_suffix(".tmp.jpg")
                encode_started = time.perf_counter()
                image.save(temporary, format="JPEG", quality=88)
                image_encode_s += time.perf_counter() - encode_started
                write_started = time.perf_counter()
                os.replace(temporary, path)
                storage_write_s += time.perf_counter() - write_started
            metadata_started = time.perf_counter()
            _ = path.stat().st_size
            storage_metadata_s += time.perf_counter() - metadata_started
            paths.append(path)
    finally:
        for image in images:
            image.close()
    timestamps = np.asarray([float(item["actual_timestamp_sec"]) for item in records], dtype=np.float64)
    frame_indices = np.asarray([int(item["source_frame_index"]) for item in records], dtype=np.int64)
    if not len(paths) or not np.all((timestamps >= clip_start_s) & (timestamps < clip_end_s)):
        raise RuntimeError("Canonical 1FPS cache escaped clip scope")
    return paths, timestamps, frame_indices, {
        "context_scope": "canonical_clip", "requested_timestamps_sec": requested.tolist(),
        "actual_timestamps_sec": timestamps.tolist(), "source_frame_indices": frame_indices.tolist(),
        "frames_decoded_offline": int(decoded_count),
        "offline_parent_open_time_s": parent_open_s,
        "offline_video_seek_time_s": seek_s,
        "offline_video_decode_time_s": decode_s + selection_s + rgb_s,
        "offline_cpu_decode_s": decode_s + selection_s + rgb_s,
        "offline_image_encode_s": image_encode_s,
        "offline_storage_write_s": storage_write_s,
        "offline_storage_metadata_s": storage_metadata_s,
        "offline_wall_latency_s": time.perf_counter() - started,
        "decoder": "sequential_one_seek_nearest_frame_tie_earlier",
        "cache_policy": "offline_cached_jpeg_1fps_frames; online_loads_selected_cached_representatives",
    }


def build_shared_fine_artifact(
    *, clip_uid: str, clip_start_s: float, clip_end_s: float, timestamps_parent_s: np.ndarray,
    frame_paths: list[Path], dino_features: np.ndarray, comet_config: dict[str, Any], hierarchy_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the one DINO Fine artifact consumed identically by B1 and B2."""
    parent_times = np.asarray(timestamps_parent_s, dtype=np.float64)
    if not len(parent_times) or len(parent_times) != len(frame_paths) or len(parent_times) != len(dino_features):
        raise ValueError("Shared Fine input timeline identity mismatch")
    local_times = parent_times - float(clip_start_s)
    duration = float(clip_end_s) - float(clip_start_s)
    internal = comet_segment(np.asarray(dino_features, dtype=np.float32), local_times, CometStyleConfig(**comet_config))
    fine = make_segmentation(
        video_id=clip_uid, method="comet_style_dinov2", video_duration=duration,
        boundaries=internal["boundaries"], frame_timestamps=local_times,
    )
    boundary_records = build_boundary_records(fine["segments"], internal)
    fine_nodes = build_fine_nodes(
        segments=fine["segments"], features=np.asarray(dino_features, dtype=np.float32), timestamps=local_times,
        frame_paths=[path.as_posix() for path in frame_paths], boundary_records=boundary_records,
        representative_fractions=tuple(hierarchy_config["representative_fractions"]),
        include_medoid=bool(hierarchy_config["include_dinov2_medoid"]),
    )
    # Keep DINO nodes local for hierarchy correctness but expose event bounds
    # in parent-video seconds to retrieval/diagnostics.
    return {
        "clip_uid": clip_uid, "context_scope": "canonical_clip", "clip_start_s": float(clip_start_s),
        "clip_end_s": float(clip_end_s), "timestamps_parent_s": parent_times,
        "timestamps_local_s": local_times, "frame_paths": frame_paths,
        "dino_features": np.asarray(dino_features, dtype=np.float32),
        "fine_segmentation": fine, "fine_nodes": fine_nodes, "boundary_records": boundary_records,
        "comet_internal": internal,
    }


def _node_units(nodes: Iterable[dict[str, Any]], *, clip_start_s: float, prefix: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for position, node in enumerate(nodes):
        units.append({
            "event_id": f"{prefix}_{position:04d}", "node_id": str(node["node_id"]),
            "start_s": float(clip_start_s) + float(node["start"]),
            "end_s": float(clip_start_s) + float(node["end"]),
            "leaf_ids": [str(value) for value in node.get("leaf_ids", [node["node_id"]])],
            "source_fine_segment_id": node.get("source_fine_segment_id"),
            "safe_merge_node_id": str(node["node_id"]) if node.get("node_type") == "internal" else None,
            "fluid_loose_node_id": str(node["node_id"]),
        })
    return units


def derive_b1_fine_units(shared_fine: dict[str, Any]) -> list[dict[str, Any]]:
    """Formal B1: stop at Fine; Safe-Merge/Fluid Loose are never invoked."""
    return _node_units(shared_fine["fine_nodes"], clip_start_s=float(shared_fine["clip_start_s"]), prefix="b1_fine")


def derive_b2_medium_units(*, shared_fine: dict[str, Any], hierarchy_config: dict[str, Any], medium_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Formal B2: exact shared Fine -> Safe-Merge -> Fluid Loose -> Medium."""
    duration = float(shared_fine["clip_end_s"]) - float(shared_fine["clip_start_s"])
    safe_merge_started = time.perf_counter()
    hierarchy, merge_trace = build_safe_hierarchy(
        video_id=str(shared_fine["clip_uid"]), video_duration=duration,
        fine_nodes=shared_fine["fine_nodes"], boundary_records=shared_fine["boundary_records"],
        timestamps=np.asarray(shared_fine["timestamps_local_s"]),
        frame_paths=[path.as_posix() for path in shared_fine["frame_paths"]],
        features=np.asarray(shared_fine["dino_features"], dtype=np.float32),
        medium_fraction=float(hierarchy_config["medium_reference_fraction"]),
        coarse_fraction=float(hierarchy_config["coarse_reference_fraction"]),
        representative_fractions=tuple(hierarchy_config["representative_fractions"]),
        include_medoid=bool(hierarchy_config["include_dinov2_medoid"]),
    )
    safe_merge_time_s = time.perf_counter() - safe_merge_started
    context = build_tree_context(hierarchy)
    fluid_loose_started = time.perf_counter()
    medium_ids, fluid_trace = fluid_frontier(
        context, minimum_q_rank=float(medium_config["minimum_q_rank"]),
        maximum_local_drop=float(medium_config["maximum_local_drop"]), level="medium",
    )
    fluid_loose_time_s = time.perf_counter() - fluid_loose_started
    medium_nodes = [context["nodes"][node_id] for node_id in medium_ids]
    medium_finalize_time_s = time.perf_counter() - fluid_loose_started - fluid_loose_time_s
    return _node_units(medium_nodes, clip_start_s=float(shared_fine["clip_start_s"]), prefix="b2_medium"), {
        "safe_merge_hierarchy": hierarchy, "safe_merge_trace": merge_trace,
        "fluid_loose_trace": fluid_trace, "medium_node_ids": medium_ids,
        "timing": {
            "offline_safe_merge_time_s": safe_merge_time_s,
            "offline_fluid_loose_time_s": fluid_loose_time_s,
            "offline_medium_finalize_time_s": medium_finalize_time_s,
        },
    }


def materialize_method_events(
    *, method: str, units: list[dict[str, Any]], shared_fine: dict[str, Any],
    source_frame_indices: np.ndarray, cradio_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    return build_event_records(
        units=units, timestamps=np.asarray(shared_fine["timestamps_parent_s"]),
        source_frame_indices=source_frame_indices, cradio_embeddings=cradio_embeddings, method=method,
    )


def conditional_b1_prime_units(*, b1_events: list[dict[str, Any]], b2_events: list[dict[str, Any]], clip_start_s: float, clip_end_s: float) -> tuple[bool, list[dict[str, Any]] | None]:
    """Return B1-prime groups only when Amendment #3's count gate is met."""
    if not b1_events:
        raise ValueError("B1 event list is empty")
    required = abs(len(b1_events) - len(b2_events)) / len(b1_events) > 0.25
    if not required:
        return False, None
    fine = [{"event_id": event["event_id"], "start_s": event["start_s"], "end_s": event["end_s"]} for event in b1_events]
    return True, build_b1_prime_groups(
        fine_events=fine, medium_count=len(b2_events), clip_start_s=clip_start_s, clip_end_s=clip_end_s,
    )
