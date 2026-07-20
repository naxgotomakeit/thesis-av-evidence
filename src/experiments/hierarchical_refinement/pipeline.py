"""Deterministic temporal-zone construction and local CoMET refinement."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.experiments.coarse_segmentation.comet_style import CometStyleConfig
from src.experiments.coarse_segmentation.comet_style import segment as comet_segment
from src.experiments.coarse_segmentation.retrieval_replay import union_duration


def merge_selected_kts_units(
    all_segments: list[dict[str, Any]],
    ranked_selection: list[dict[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Merge selected KTS units only when they are consecutive partition units."""
    index_by_id = {str(row["segment_id"]): index for index, row in enumerate(all_segments)}
    rank_by_id = {str(row["segment_id"]): int(row["rank"]) for row in ranked_selection}
    score_by_id = {str(row["segment_id"]): float(row["score"]) for row in ranked_selection}
    selected_ids = list(rank_by_id)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Duplicate selected KTS segment identity")
    unknown = [identifier for identifier in selected_ids if identifier not in index_by_id]
    if unknown:
        raise ValueError(f"Selected KTS IDs absent from partition: {unknown}")
    ordered = sorted(selected_ids, key=lambda identifier: index_by_id[identifier])
    zones: list[dict[str, Any]] = []
    for identifier in ordered:
        index = index_by_id[identifier]
        segment = all_segments[index]
        start, end = float(segment["start"]), float(segment["end"])
        if (
            zones
            and index == zones[-1]["source_indices"][-1] + 1
            and abs(start - float(zones[-1]["end"])) <= tolerance
        ):
            zones[-1]["end"] = end
            zones[-1]["duration"] = end - float(zones[-1]["start"])
            zones[-1]["selected_kts_ids"].append(identifier)
            zones[-1]["source_indices"].append(index)
            zones[-1]["selection_ranks"].append(rank_by_id[identifier])
            zones[-1]["selection_scores"].append(score_by_id[identifier])
            zones[-1]["merge_decisions"].append(
                {"left": zones[-1]["selected_kts_ids"][-2], "right": identifier, "merged": True}
            )
        else:
            zones.append(
                {
                    "zone_id": f"zone_{len(zones):02d}",
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "selected_kts_ids": [identifier],
                    "source_indices": [index],
                    "selection_ranks": [rank_by_id[identifier]],
                    "selection_scores": [score_by_id[identifier]],
                    "merge_decisions": [],
                }
            )
    return zones


def _nearest_timestamp(
    timestamps: np.ndarray, start: float, end: float, center: float
) -> float:
    available = timestamps[(timestamps >= start) & (timestamps < end)]
    if not len(available):
        available = timestamps[(timestamps >= start) & (timestamps <= end)]
    if not len(available):
        raise ValueError(f"No frame timestamp inside local segment [{start}, {end}]")
    return float(available[np.argmin(np.abs(available - center))])


def local_comet_segmentation(
    *,
    video_id: str,
    video_duration: float,
    dino_features: np.ndarray,
    frame_timestamps: np.ndarray,
    candidate_zones: list[dict[str, Any]],
    config: CometStyleConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply frozen CoMET segmentation independently inside selected KTS zones."""
    features = np.asarray(dino_features, dtype=np.float32)
    timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    if len(features) != len(timestamps):
        raise ValueError("DINO feature/timestamp length mismatch")
    segments: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for zone_index, zone in enumerate(candidate_zones):
        zone_start, zone_end = float(zone["start"]), float(zone["end"])
        mask = (timestamps >= zone_start) & (timestamps < zone_end)
        local_times = timestamps[mask]
        local_features = features[mask]
        if not len(local_times):
            raise ValueError(f"Candidate zone contains no shared frame: {zone}")
        result = comet_segment(local_features, local_times, config)
        boundaries = sorted(
            {
                float(value)
                for value in result["boundaries"]
                if zone_start < float(value) < zone_end
            }
        )
        edges = [zone_start, *boundaries, zone_end]
        zone_segments = []
        for local_index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            if end <= start:
                raise ValueError("Non-positive local CoMET segment")
            identifier = f"local_comet_z{zone_index:02d}_{local_index:04d}"
            segment = {
                "segment_id": identifier,
                "start": start,
                "end": end,
                "duration": end - start,
                "representative_frame_timestamp": _nearest_timestamp(
                    timestamps, start, end, (start + end) / 2.0
                ),
                "candidate_zone_id": str(zone["zone_id"]),
                "source_kts_ids": list(zone["selected_kts_ids"]),
            }
            segments.append(segment)
            zone_segments.append(identifier)
        diagnostics.append(
            {
                "zone_id": zone["zone_id"],
                "start": zone_start,
                "end": zone_end,
                "frame_count": int(mask.sum()),
                "boundaries": boundaries,
                "segment_ids": zone_segments,
                "raw_similarity": result["raw_similarity"],
                "smoothed_similarity": result["smoothed_similarity"],
                "comet_diagnostics": result["diagnostics"],
            }
        )
    record = {
        "video_id": str(video_id),
        "method": "kts_local_comet",
        "video_duration": float(video_duration),
        "candidate_zones": candidate_zones,
        "segments": segments,
    }
    errors = validate_local_coverage(record)
    if errors:
        raise ValueError(f"Invalid local CoMET coverage: {errors}")
    return record, diagnostics


def validate_local_coverage(record: dict[str, Any], tolerance: float = 1e-6) -> list[str]:
    """Validate complete, non-overlapping coverage within every candidate zone."""
    errors: list[str] = []
    segments = record.get("segments") or []
    by_zone: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        by_zone.setdefault(str(segment["candidate_zone_id"]), []).append(segment)
    identifiers: set[str] = set()
    for segment in segments:
        identifier = str(segment["segment_id"])
        if identifier in identifiers:
            errors.append(f"duplicate_segment_id:{identifier}")
        identifiers.add(identifier)
    for zone in record.get("candidate_zones") or []:
        zone_id = str(zone["zone_id"])
        rows = sorted(by_zone.get(zone_id, []), key=lambda row: float(row["start"]))
        if not rows:
            errors.append(f"empty_zone:{zone_id}")
            continue
        if abs(float(rows[0]["start"]) - float(zone["start"])) > tolerance:
            errors.append(f"zone_start_mismatch:{zone_id}")
        if abs(float(rows[-1]["end"]) - float(zone["end"])) > tolerance:
            errors.append(f"zone_end_mismatch:{zone_id}")
        for index, row in enumerate(rows):
            start, end = float(row["start"]), float(row["end"])
            if start < float(zone["start"]) - tolerance or end > float(zone["end"]) + tolerance:
                errors.append(f"outside_zone:{row['segment_id']}")
            if end <= start:
                errors.append(f"non_positive:{row['segment_id']}")
            if index:
                previous = float(rows[index - 1]["end"])
                if start > previous + tolerance:
                    errors.append(f"gap:{row['segment_id']}")
                if start < previous - tolerance:
                    errors.append(f"overlap:{row['segment_id']}")
    return errors


def midpoint_containment(
    reference_segments: list[dict[str, Any]], zones: list[dict[str, Any]]
) -> list[bool]:
    """Return whether each reference midpoint falls in any selected KTS zone."""
    return [
        any(
            float(zone["start"]) <= (float(row["start"]) + float(row["end"])) / 2.0
            < float(zone["end"])
            for zone in zones
        )
        for row in reference_segments
    ]


def overlap_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    """Return temporal intersection duration between two interval unions."""
    intersections = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start, end = max(left_start, right_start), min(left_end, right_end)
            if end > start:
                intersections.append((start, end))
    return union_duration(intersections)
