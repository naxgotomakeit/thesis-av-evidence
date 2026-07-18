"""Canonical Task 7A v1 payload construction and deterministic preflight.

Source lineage: final provider-neutral Task 7A helpers. No semantic model is
called and gold/reference data is not accepted by this API.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.final_qa.task7a_preflight import (
    FINAL_ANSWER_POLICY,
    REQUIRED_OUTPUT_SCHEMA,
    build_pipeline_uncertainties,
    effective_bounds,
    path_is_file,
    payload_leakage_audit,
    validate_visual_frames,
)

from .state import CaseState


def _visual(candidate: dict[str, Any]) -> dict[str, Any]:
    return {"evidence_id": candidate["candidate_id"], "start_sec": float(candidate["start_time"]), "end_sec": float(candidate["end_time"]), "frames": [{"frame_path": frame["canonical_frame_path"], "timestamp_sec": float(frame["timestamp"]), "presentation_order": frame["presentation_order"], "selection_rank": frame["selection_rank"], "anchor_distance_sec": frame.get("anchor_distance_sec")} for frame in candidate["canonical_visual_frames"]], "roles": copy.deepcopy(candidate["roles"])}


def _speech(candidate: dict[str, Any]) -> dict[str, Any]:
    start, end = effective_bounds(candidate)
    return {"evidence_id": candidate["candidate_id"], "transcript": candidate.get("transcript_text", ""), "start_sec": start, "end_sec": end, "roles": copy.deepcopy(candidate["roles"]), "exact_phrase_match": candidate.get("exact_phrase_match"), "phrase_match_method": candidate.get("phrase_match_method"), "phrase_match_score": candidate.get("phrase_match_score"), "fallback_provenance": candidate.get("source") == "local_asr_fallback" or "fallback_recovered" in candidate.get("roles", []), "timestamp_validity": candidate.get("timestamp_validity", "valid"), "speaker_attribution_warning": candidate.get("speaker_attribution_warning"), "speaker_verified": False}


def _acoustic(candidate: dict[str, Any]) -> dict[str, Any]:
    return {"evidence_id": candidate["candidate_id"], "audio_clip_path": candidate.get("local_audio_clip_reference"), "start_sec": float(candidate["start_time"]), "end_sec": float(candidate["end_time"]), "roles": copy.deepcopy(candidate["roles"]), "acoustic_evidence_role": candidate.get("acoustic_evidence_role"), "semantic_interpretation_pending": bool(candidate.get("semantic_interpretation_pending", True)), "clap_retrieval_provenance": {"clap_similarity_score": candidate.get("clap_similarity_score"), "score_scope": candidate.get("score_scope"), "broad_source_warning": candidate.get("broad_source_warning"), "local_acoustic_score_recomputed": candidate.get("local_acoustic_score_recomputed", False), "not_semantic_verification": True}}


def build_final_payload(state: CaseState, project_root: Path) -> CaseState:
    """Build and validate the exact Task 7A v1 model-facing payload."""
    if state.evidence_packet is None:
        raise ValueError("Task 6 evidence packet is required before Task 7A")
    packet = state.evidence_packet
    candidates = {item["candidate_id"]: item for item in packet["retained_candidates"]}
    groups = []
    for source in packet["retained_evidence_groups"]:
        group = {"group_id": source["group_id"], "group_type": source["group_type"], "visual_evidence": [], "speech_evidence": [], "acoustic_evidence": [], "relations": copy.deepcopy(source["relations"]), "unresolved_ambiguities": copy.deepcopy(source["unresolved_ambiguities"]), "missing_information": copy.deepcopy(source["missing_information"])}
        for member in source["retained_candidates"]:
            candidate = candidates[member["candidate_id"]]
            target = {"visual": "visual_evidence", "speech": "speech_evidence", "acoustic": "acoustic_evidence"}[candidate["modality"]]
            group[target].append({"visual": _visual, "speech": _speech, "acoustic": _acoustic}[candidate["modality"]](candidate))
        groups.append(group)
    payload = {"case_id": state.case_id, "question": state.question, "operation": packet["operation"], "answer_required_modalities": copy.deepcopy(packet["answer_required_modalities"]), "supporting_modalities": copy.deepcopy(packet["supporting_modalities"]), "dataset_or_query_inconsistency_status": packet.get("dataset_or_query_inconsistency_status", "unknown"), "evidence_groups": groups, "pipeline_uncertainties": build_pipeline_uncertainties(packet), "final_answer_policy": copy.deepcopy(FINAL_ANSWER_POLICY), "required_output_schema": copy.deepcopy(REQUIRED_OUTPUT_SCHEMA)}
    visual = [item for group in groups for item in group["visual_evidence"]]
    speech = [item for group in groups for item in group["speech_evidence"]]
    acoustic = [item for group in groups for item in group["acoustic_evidence"]]
    visual_checks = [validate_visual_frames(item["frames"], project_root) for item in visual]
    invalid_visual = [path for check in visual_checks for path in check["invalid_paths"]]
    invalid_audio = [item.get("audio_clip_path") for item in acoustic if not path_is_file(project_root, item.get("audio_clip_path"))]
    available = set()
    if visual and not invalid_visual and all(check["chronological_visual_order"] for check in visual_checks):
        available.add("visual")
    if speech and all(float(item["start_sec"]) <= float(item["end_sec"]) and item["timestamp_validity"] != "excluded_outside_decode_interval" for item in speech):
        available.add("speech")
    if acoustic and not invalid_audio:
        available.add("acoustic")
    missing = sorted(set(payload["answer_required_modalities"]) - available)
    leakage = payload_leakage_audit(payload)
    status = "blocked" if missing or invalid_visual or invalid_audio or not leakage["leakage_check_passed"] else ("warning" if any(item["severity"] in {"material", "critical"} for item in payload["pipeline_uncertainties"]) else "ready")
    state.final_payload = payload
    state.preflight_result = {"case_id": state.case_id, "status": status, "required_modalities": payload["answer_required_modalities"], "available_modalities": sorted(available), "missing_assets": [f"required_modality:{item}" for item in missing], "invalid_paths": invalid_visual + [str(item) for item in invalid_audio if item], "leakage_check_passed": leakage["leakage_check_passed"], "leakage_audit": leakage}
    state.record("final_payload_build", "task7a_v1_payload_and_preflight", status=status)
    return state

