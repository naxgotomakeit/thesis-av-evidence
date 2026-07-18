from __future__ import annotations

import copy
import hashlib
import unicodedata
from typing import Any


ROLE_PRIORITY = {
    "trigger": 80, "temporal_anchor": 75, "direct_evidence": 70, "resolver": 65,
    "plausible_response": 60, "fallback_recovered": 55, "supporting": 30, "alternative": 25, "not_required": 0,
}


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_text(text: str) -> str:
    return " ".join("".join(char if char.isalnum() else " " for char in unicodedata.normalize("NFKC", str(text)).casefold()).split())


def overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, min(float(left["end_time"]), float(right["end_time"])) - max(float(left["start_time"]), float(right["start_time"])))


def temporal_gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, float(right["start_time"]) - float(left["end_time"]), float(left["start_time"]) - float(right["end_time"]))


def deduplicate_visual_frames(frames: list[dict[str, Any]], max_frames: int, anchor_time: float | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for frame in frames:
        key = (str(frame.get("video_id", "unknown_video")), int(frame.get("normalized_timestamp_ms", round(float(frame["timestamp"]) * 1000))))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = copy.deepcopy(frame)
        else:
            existing_dense = existing.get("provenance") in {"dense_frame", "both"}
            new_dense = frame.get("provenance") in {"dense_frame", "both"}
            if new_dense and not existing_dense:
                dropped.append({"candidate_id": f"frame_{key[0]}_{key[1]}", "reason": "duplicate_visual_timestamp_dense_provenance_preferred"})
                by_key[key] = copy.deepcopy(frame)
            else:
                dropped.append({"candidate_id": f"frame_{key[0]}_{key[1]}", "reason": "duplicate_visual_timestamp"})
    ordered = sorted(by_key.values(), key=lambda frame: (abs(float(frame["timestamp"]) - anchor_time) if anchor_time is not None else float(frame["timestamp"]), float(frame["timestamp"])))
    retained = ordered[:max_frames]
    for frame in ordered[max_frames:]:
        dropped.append({"candidate_id": f"frame_{frame.get('video_id','unknown')}_{frame.get('normalized_timestamp_ms', round(float(frame['timestamp'])*1000))}", "reason": "visual_frame_budget_anchor_nearest_selection"})
    return retained, dropped


def role_priority(candidate: dict[str, Any]) -> int:
    roles = candidate.get("roles", [])
    score = max((ROLE_PRIORITY.get(role, 0) for role in roles), default=0)
    if candidate.get("exact_phrase_match"):
        score += 12
    if candidate.get("local_audio_clip_reference"):
        score += 8
    if candidate.get("valid_asr_timestamps"):
        score += 5
    if candidate.get("dense_visual_provenance"):
        score += 4
    return score


def build_speech_relations(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triggers = [item for item in candidates if "trigger" in item.get("roles", [])]
    responses = [item for item in candidates if "plausible_response" in item.get("roles", [])]
    relations: list[dict[str, Any]] = []
    for trigger in triggers:
        for response in responses:
            if float(response["start_time"]) >= float(trigger["end_time"]):
                relations.append({"source_candidate_id": trigger["candidate_id"], "target_candidate_id": response["candidate_id"], "relation_type": "response_to", "temporal_gap_sec": round(float(response["start_time"]) - float(trigger["end_time"]), 6), "relation_confidence": "temporal_only", "relation_basis": "transcript_sequence"})
    for index, left in enumerate(responses):
        for right in responses[index + 1:]:
            relations.append({"source_candidate_id": left["candidate_id"], "target_candidate_id": right["candidate_id"], "relation_type": "alternative_to", "temporal_gap_sec": round(temporal_gap(left, right), 6), "relation_confidence": "temporal_only", "relation_basis": "transcript_sequence"})
    return relations


def temporal_relations(candidates: list[dict[str, Any]], operation: str) -> list[dict[str, Any]]:
    relations = build_speech_relations(candidates)
    seen = {(item["source_candidate_id"], item["target_candidate_id"], item["relation_type"]) for item in relations}
    anchors = [item for item in candidates if any(role in item.get("roles", []) for role in ("trigger", "temporal_anchor"))]
    resolvers = [item for item in candidates if "resolver" in item.get("roles", [])]
    for anchor in anchors:
        for resolver in resolvers:
            if anchor["candidate_id"] == resolver["candidate_id"]:
                continue
            overlap = overlap_seconds(anchor, resolver)
            relation_type = "resolves" if operation.startswith("identify") else ("overlaps" if overlap else "near")
            key = (anchor["candidate_id"], resolver["candidate_id"], relation_type)
            if key not in seen:
                relations.append({"source_candidate_id": anchor["candidate_id"], "target_candidate_id": resolver["candidate_id"], "relation_type": relation_type, "temporal_gap_sec": 0.0 if overlap else round(temporal_gap(anchor, resolver), 6), "relation_confidence": "temporal_only", "relation_basis": "temporal_overlap" if overlap else "anchor_distance"})
                seen.add(key)
    for candidate in candidates:
        if "fallback_recovered" in candidate.get("roles", []):
            relations.append({"source_candidate_id": candidate["candidate_id"], "target_candidate_id": "speech_branch_missing_evidence", "relation_type": "fallback_recovery_for", "temporal_gap_sec": 0.0, "relation_confidence": "provenance", "relation_basis": "fallback_provenance"})
    return relations


def apply_packet_budget(candidates: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    max_candidates = int(config["max_candidates_per_case"])
    max_supporting = int(config["max_supporting_candidates_per_case"])
    ordered = sorted(candidates, key=lambda item: (-role_priority(item), float(item.get("start_time", 0.0)), item["candidate_id"]))
    retained, dropped, supporting_count = [], [], 0
    protected_roles = {"trigger", "temporal_anchor", "direct_evidence", "resolver", "plausible_response", "fallback_recovered"}
    for candidate in ordered:
        roles = set(candidate.get("roles", []))
        protected = bool(roles & protected_roles)
        reason = None
        if "supporting" in roles and not protected and supporting_count >= max_supporting:
            reason = "redundant_supporting_acoustic_candidate"
        elif len(retained) >= max_candidates and not protected:
            reason = "lower_priority_same_role_candidate"
        if reason:
            dropped.append({"candidate_id": candidate["candidate_id"], "reason": reason})
        else:
            retained.append(candidate)
            if "supporting" in roles and not protected:
                supporting_count += 1
    violations = []
    if len(retained) > max_candidates:
        violations.append({"warning": "budget_exceeded_to_preserve_required_evidence", "limit": max_candidates, "actual": len(retained)})
    return retained, dropped, {"config": copy.deepcopy(config), "retained_candidate_count": len(retained), "dropped_candidate_count": len(dropped), "violations": violations}


def union_duration(candidates: list[dict[str, Any]], modality: str | None = None) -> float:
    spans = sorted((float(item["start_time"]), float(item["end_time"])) for item in candidates if modality is None or item.get("modality") == modality)
    total, stop = 0.0, float("-inf")
    for start, end in spans:
        if start > stop:
            total += end - start
        elif end > stop:
            total += end - stop
        stop = max(stop, end)
    return total
