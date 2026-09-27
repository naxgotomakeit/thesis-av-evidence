"""Pure request-shape scaffold; deliberately has no network transport."""
from __future__ import annotations

from typing import Any

from .evidence import EvidencePackage, MAX_PHYSICAL_IMAGES, assert_safe_direct_evidence
from direct_api_v1.policy import MAX_NEW_IMAGES_PER_TURN, MAX_UNIQUE_IMAGES_PER_QUESTION


def build_direct_request_shape(package: EvidencePackage, question: dict[str, Any]) -> dict[str, Any]:
    """Return a future transport-neutral shape after the physical-image guard."""
    if package.selection_state != "finalized":
        raise ValueError("cannot construct Direct request from non-finalized evidence")
    final_image_count = len(package.final_evidence)
    assert final_image_count <= MAX_PHYSICAL_IMAGES
    return {"question": question, "method": package.method, "map_text_evidence": package.map_text_evidence, "ordered_images": [{"chronological_order": index, "timestamp_sec": item.timestamp_sec, "frame_path": item.frame_path, "frame_sha256": item.frame_sha256} for index, item in enumerate(package.final_evidence)], "final_image_count": final_image_count, "network_transport": "intentionally_not_implemented"}


def build_direct_v1_session_shape(*, question: dict[str, Any], method: str, native_map: dict[str, Any], map_path: str, map_sha256: str, inspected_images: list[dict[str, Any]]) -> dict[str, Any]:
    """Transport-neutral same-agent session payload for future Direct v1 providers.

    This deliberately carries the native map and only frames the agent itself
    requested through the controller; it is not a precomputed retrieval set.
    """
    if method not in {"R1", "R3"}:
        raise ValueError("Direct v1 map session requires R1 or R3")
    if len(inspected_images) > MAX_UNIQUE_IMAGES_PER_QUESTION:
        raise ValueError("Direct v1 session exceeds physical-image hard cap")
    assert_safe_direct_evidence(question)
    assert_safe_direct_evidence(native_map)
    assert_safe_direct_evidence(inspected_images)
    return {
        "question": question, "method": method,
        "native_map": native_map,
        "map_identity": {"path": map_path, "sha256": map_sha256},
        "inspected_images": inspected_images,
        "frame_action_contract": {
            "actions": ["inspect_frames", "final_answer"],
            "max_new_images_per_turn": MAX_NEW_IMAGES_PER_TURN,
            "max_unique_physical_images_per_question": MAX_UNIQUE_IMAGES_PER_QUESTION,
        },
        "network_transport": "intentionally_not_implemented",
    }
