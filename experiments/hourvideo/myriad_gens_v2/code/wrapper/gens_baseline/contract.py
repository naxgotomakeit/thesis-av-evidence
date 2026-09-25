from __future__ import annotations

from typing import Any, Sequence


LATENCY_KEYS = (
    "candidate_extraction",
    "clip_text",
    "clip_images",
    "clip_sort",
    "gens_preprocess",
    "gens_generate",
    "gens_parse",
    "total",
)


def blank_latencies() -> dict[str, float]:
    return {key: 0.0 for key in LATENCY_KEYS}


def cap_gens_frames(mapped: Sequence[dict[str, Any]], cap: int = 16) -> tuple[list[dict], list[dict]]:
    if cap != 16:
        raise ValueError("frozen final cap is 16")
    relevance_order = sorted(
        (dict(item) for item in mapped),
        key=lambda item: (
            -item["gens_relevance_score"],
            item["gens_response_order"],
            item["gens_input_index"],
        ),
    )
    chosen = relevance_order[:cap]
    downstream = sorted(
        chosen, key=lambda item: (item["timestamp_sec"], item["frame_index"])
    )
    selected_frames: list[dict[str, Any]] = []
    for downstream_order, item in enumerate(downstream):
        selected_frames.append(
            {
                "downstream_order": downstream_order,
                "video_id": item["video_id"],
                "timestamp_sec": item["timestamp_sec"],
                "frame_index": item["frame_index"],
                "source_video_path": item["source_video_path"],
                "extracted_frame_path": item["extracted_frame_path"],
                "selection": {
                    "clip_rank": item["clip_rank"],
                    "clip_score": item["clip_score"],
                    "gens_input_index": item["gens_input_index"],
                    "gens_relevance_score": item["gens_relevance_score"],
                    "gens_span": item["gens_span"],
                    "gens_response_order": item["gens_response_order"],
                },
            }
        )
    return selected_frames, relevance_order


def build_package(
    *,
    profile: dict[str, Any],
    qa_uid: str,
    video_id: str,
    status: str,
    error: dict[str, Any] | None,
    selected_frames: list[dict[str, Any]],
    latencies: dict[str, float],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    merged_latencies = blank_latencies()
    merged_latencies.update({key: float(value) for key, value in latencies.items()})
    package = {
        "schema_version": 1,
        "qa_uid": qa_uid,
        "video_id": video_id,
        "method": profile["method"],
        "status": status,
        "error": error,
        "selected_frames": selected_frames,
        "actual_frame_count": len(selected_frames),
        "downstream_frame_order": "timestamp_sec_asc_then_frame_index_asc",
        "selection_latency_ms": merged_latencies,
        "provenance": provenance,
    }
    validate_package(package)
    return package


def validate_package(package: dict[str, Any], allow_shared_methods: bool = False) -> None:
    required = {
        "schema_version",
        "qa_uid",
        "video_id",
        "method",
        "status",
        "error",
        "selected_frames",
        "actual_frame_count",
        "downstream_frame_order",
        "selection_latency_ms",
        "provenance",
    }
    if set(package) != required:
        raise ValueError("downstream package top-level schema mismatch")
    methods = {"gens_hybrid_cap16"}
    if allow_shared_methods:
        methods.update({"uniform_16", "ours_cap16"})
    if package["method"] not in methods:
        raise ValueError("unexpected method")
    frames = package["selected_frames"]
    if not 0 <= len(frames) <= 16 or package["actual_frame_count"] != len(frames):
        raise ValueError("actual_frame_count must equal a 0-16 selected frame list")
    order = [(frame["timestamp_sec"], frame["frame_index"]) for frame in frames]
    if order != sorted(order):
        raise ValueError("downstream frames are not in frozen chronological order")
    if len({(frame["video_id"], frame["frame_index"]) for frame in frames}) != len(frames):
        raise ValueError("downstream frames are not unique")
    forbidden = {"gold", "gold_label", "answer", "prediction", "trajectory"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden:
                    raise ValueError(f"forbidden field in downstream package: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(package)
