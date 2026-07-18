"""Canonical Task 6 v1.2 relation-aware packet construction.

This module directly composes final validated rules. Task 6 v1/v1.1/v1.2
historical scripts are source lineage only and are not runtime stages.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from src.retrieval.task6_relation_reranking import (
    apply_packet_budget,
    deduplicate_visual_frames,
    normalize_text,
    stable_id,
    temporal_relations,
)
from src.retrieval.task6_v1_1 import (
    classify_candidate_accounting,
    classify_modalities,
    corrected_relations,
)
from src.retrieval.task6_v1_2 import (
    candidate_count_consistent,
    chronological_frames,
    corrected_budget_accounting,
    correct_response_relations,
)

from .state import CaseState


def _quoted_phrases(cues: dict[str, Any]) -> list[str]:
    return [normalize_text(item["text"]) for item in cues.get("quoted_phrases", []) if item.get("text")]


def _exact_phrase(candidate: dict[str, Any], phrases: list[str]) -> bool:
    text = normalize_text(candidate.get("transcript_text", ""))
    return any(phrase and phrase in text for phrase in phrases)


def _normalize_nonvisual(record: dict[str, Any]) -> list[dict[str, Any]]:
    operation = record["task5a_plan_summary"]["answer_requirement"]["operation"]
    phrases = _quoted_phrases(record["deterministic_question_cues"])
    diagnostics = {item["candidate_id"]: item for item in record.get("acoustic_evidence_diagnostics", [])}
    source = [copy.deepcopy(item) for item in record["post_fallback_candidates"] if item.get("modality") != "visual"]
    trigger_end = None
    if operation == "measure_delay":
        triggers = [item for item in source if item.get("modality") == "speech" and _exact_phrase(item, phrases)]
        if triggers:
            trigger_end = min(float(item["end_time"]) for item in triggers)
    output = []
    for item in source:
        roles: list[str] = []
        if item.get("modality") == "acoustic":
            diagnostic = diagnostics.get(item["candidate_id"], {})
            roles.append(diagnostic.get("acoustic_evidence_role", "not_required"))
            item.update({key: copy.deepcopy(value) for key, value in diagnostic.items() if key not in {"candidate_id", "selected_acoustic_interval"}})
            item["local_audio_clip_reference"] = diagnostic.get("local_audio_clip_reference")
        else:
            phrase = _exact_phrase(item, phrases)
            if operation == "measure_delay":
                if phrase:
                    roles.append("trigger")
                elif trigger_end is not None and float(item["start_time"]) >= trigger_end:
                    roles.extend(["plausible_response", "alternative"])
            elif operation == "count_occurrences":
                roles.append("direct_evidence")
                if item.get("source") == "local_asr_fallback":
                    roles.append("fallback_recovered")
            elif phrase:
                roles.append("temporal_anchor")
            else:
                roles.append("supporting")
            item["exact_phrase_match"] = phrase
            item["valid_asr_timestamps"] = item.get("timestamp_validity", "valid") not in {"excluded_outside_decode_interval", "clipped_to_decode_interval"}
            if item.get("source") == "local_asr_fallback" and "fallback_recovered" not in roles:
                roles.append("fallback_recovered")
        item["roles"] = sorted(set(roles or ["not_required"]))
        item["provenance"] = {"original_candidate": copy.deepcopy(item)}
        output.append(item)
    return output


def _visual_packet(record: dict[str, Any], candidates: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    source = [copy.deepcopy(item) for item in record["post_fallback_candidates"] if item.get("modality") == "visual"]
    frames = copy.deepcopy(record["task5b_v1_1_input"].get("selected_visual_evidence_frames", []))
    if not source and not frames:
        return None, [], []
    anchors = [item for item in candidates if any(role in item.get("roles", []) for role in ("trigger", "temporal_anchor"))]
    anchor_time = None if not anchors else sum((float(item["start_time"]) + float(item["end_time"])) / 2 for item in anchors) / len(anchors)
    selected, dropped_frames = deduplicate_visual_frames(frames, int(config["max_visual_frames_per_evidence_group"]), anchor_time)
    starts = [float(item["start_time"]) for item in source] or [float(item["timestamp"]) for item in selected]
    ends = [float(item["end_time"]) for item in source] or [float(item["timestamp"]) for item in selected]
    source_ids = [item["candidate_id"] for item in source]
    packet = {"candidate_id": stable_id("visual_packet", record["case_id"], *source_ids), "candidate_type": "canonical_visual_evidence", "modality": "visual", "start_time": min(starts), "end_time": max(ends), "roles": ["resolver"], "canonical_visual_frames": selected, "dense_visual_provenance": any(frame.get("provenance") in {"dense_frame", "both"} for frame in selected), "source_candidate_ids": source_ids, "provenance": {"merged_visual_micro_windows": source, "canonical_frames_from_task5b_v1_1": selected}}
    dropped = [{"candidate_id": item["candidate_id"], "reason": "overlapping_visual_window_merged" if len(source) > 1 else "visual_micro_window_represented_by_canonical_frames"} for item in source] + dropped_frames
    return packet, dropped, selected


def _groups(record: dict[str, Any], candidates: list[dict[str, Any]], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operation = record["task5a_plan_summary"]["answer_requirement"]["operation"]
    if operation == "measure_delay":
        members = [item for item in candidates if any(role in item["roles"] for role in ("trigger", "plausible_response"))]
        group_type, reason = "trigger_response", "Preserves the trigger and plausible response alternatives without selecting a speaker or response as correct."
    elif any("resolver" in item["roles"] for item in candidates):
        members = [item for item in candidates if any(role in item["roles"] for role in ("temporal_anchor", "resolver"))]
        group_type, reason = "anchor_resolver", "Combines a temporal anchor with visual resolver evidence without inferring an answer."
    else:
        members = list(candidates)
        group_type = "fallback_recovery" if any("fallback_recovered" in item["roles"] for item in members) else "single_modality_evidence"
        reason = "Retains compact direct evidence; it is explicitly a single-modality group when no cross-modal relation is present."
    ids = {item["candidate_id"] for item in members}
    return [{"group_id": stable_id("group", record["case_id"], group_type), "operation": operation, "group_type": group_type, "primary_anchor": next((item["candidate_id"] for item in members if "trigger" in item["roles"] or "temporal_anchor" in item["roles"] or "fallback_recovered" in item["roles"]), None), "retained_candidates": members, "relations": [item for item in relations if item["source_candidate_id"] in ids or item["target_candidate_id"] in ids], "unresolved_ambiguities": copy.deepcopy(record["ambiguity_flags"]), "missing_information": [], "why_this_group_is_needed": reason}]


def build_evidence_packet(state: CaseState, config: dict[str, Any]) -> CaseState:
    """Execute final Task 6 v1.2 packet behavior once."""
    if state.sufficiency_result is None:
        raise ValueError("Task 5C result is required before Task 6")
    total_started = time.perf_counter()
    record = state.sufficiency_result
    before = copy.deepcopy(record["post_fallback_candidates"])
    normalized = _normalize_nonvisual(record)
    visual, visual_drops, selected_frames = _visual_packet(record, normalized, config)
    if visual:
        normalized.append(visual)
    protected = {"trigger", "temporal_anchor", "direct_evidence", "resolver", "plausible_response", "fallback_recovered"}
    required_roles = sorted({role for item in normalized for role in item.get("roles", []) if role in protected})
    required_modalities = sorted({item["modality"] for item in normalized if set(item.get("roles", [])) & protected})
    required_present = [item for item in normalized if set(item.get("roles", [])) & protected]
    supporting_drops, filtered = [], []
    for item in normalized:
        if item.get("roles") == ["supporting"] and required_present:
            supporting_drops.append({"candidate_id": item["candidate_id"], "reason": "redundant_supporting_acoustic_candidate"})
        else:
            filtered.append(item)
    retained, budget_drops, budget = apply_packet_budget(filtered, config)
    relation_started = time.perf_counter()
    initial_relations = temporal_relations(retained, record["task5a_plan_summary"]["answer_requirement"]["operation"])
    relation_construction_sec = time.perf_counter() - relation_started
    initial_groups = _groups(record, retained, initial_relations)
    base = {"case_id": state.case_id, "question": state.question, "operation": record["task5a_plan_summary"]["answer_requirement"]["operation"], "required_modalities": record["task5a_plan_summary"]["resolver_modalities"], "required_evidence_modalities": required_modalities, "required_roles": required_roles, "candidates_before_reranking": before, "retained_evidence_groups": initial_groups, "retained_candidates": retained, "dropped_candidates": visual_drops + supporting_drops + budget_drops, "candidate_drop_reasons": visual_drops + supporting_drops + budget_drops, "relations": initial_relations, "selected_visual_frames": selected_frames, "local_audio_clips": copy.deepcopy(record.get("local_audio_clips", [])), "speech_segments": [item for item in retained if item.get("modality") == "speech"], "unresolved_ambiguities": copy.deepcopy(record["ambiguity_flags"]), "source_unresolved_ambiguities": copy.deepcopy(record["ambiguity_flags"]), "missing_information": [], "structural_evidence_status": record["evidence_status"], "questionable_followup_policy": record["questionable_followup_policy"], "dataset_or_query_inconsistency_status": "unknown", "budget_accounting": budget, "provenance": {"task5a_plan": copy.deepcopy(record["task5a_plan_summary"]), "task5c_v1_2_acoustic_diagnostics": copy.deepcopy(record.get("acoustic_evidence_diagnostics", [])), "fallback_history": copy.deepcopy(record.get("source_task5c_v1_1", {}))}}
    accounting = classify_candidate_accounting(base)
    frames = chronological_frames(base["selected_visual_frames"])
    for candidate in retained:
        if candidate.get("candidate_type") == "canonical_visual_evidence":
            candidate["canonical_visual_frames"] = copy.deepcopy(frames)
    planner_requested = set(record["task5a_plan_summary"].get("resolver_modalities", []))
    primary = record["task5a_plan_summary"].get("primary_anchor_modality")
    if primary in {"speech", "acoustic", "visual"}:
        planner_requested.add(primary)
    modalities = classify_modalities(sorted(planner_requested), retained, {item["candidate_id"]: item for item in record.get("acoustic_evidence_diagnostics", [])}, accounting["actually_dropped_candidates"])
    rerank_started = time.perf_counter()
    relations = corrected_relations(retained, base["operation"], initial_relations)
    relations = correct_response_relations(relations, emit_trigger_of=True)
    base.pop("required_modalities")
    base.pop("required_evidence_modalities")
    base.update(modalities)
    base.update(accounting)
    base["selected_visual_frames"] = frames
    base["relations"] = relations
    base["retained_evidence_groups"] = _groups(record, retained, relations)
    base["budget_accounting"] = corrected_budget_accounting(base["budget_accounting"], base["actually_dropped_candidates"])
    base["consistency_checks"] = {"dropped_candidate_count_matches_actually_dropped": base["budget_accounting"]["dropped_candidate_count"] == len(base["actually_dropped_candidates"]), "candidate_count_equation": candidate_count_consistent(base)}
    if not all(base["consistency_checks"].values()):
        raise RuntimeError(f"Canonical Task 6 accounting inconsistency: {state.case_id}")
    relation_reranking_sec = time.perf_counter() - rerank_started
    state.evidence_packet = base
    total_sec = time.perf_counter() - total_started
    state.usage["task6_v1_2_runtime"] = {
        "relation_construction_sec": relation_construction_sec,
        "relation_reranking_sec": relation_reranking_sec,
        "evidence_packet_build_sec": max(0.0, total_sec - relation_construction_sec - relation_reranking_sec),
        "total_sec": total_sec,
    }
    state.record("relation_reranking", "task6_v1_2_direct_final_behavior", historical_runtime_stages=[])
    return state
