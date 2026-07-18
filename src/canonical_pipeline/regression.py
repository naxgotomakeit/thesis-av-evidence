"""Structural regression comparison against frozen canonical artifacts."""

from __future__ import annotations

import copy
from typing import Any

from .contracts import payload_evidence_ids, stabilize_relations
from .state import CaseState


def _ids(items: list[dict[str, Any]]) -> list[str]:
    return [item["candidate_id"] for item in items]


def _intervals(items: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    return [(item["candidate_id"], float(item["start_time"]), float(item["end_time"])) for item in items]


def _relations(items: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [(item["source_candidate_id"], item["target_candidate_id"], item["relation_type"], item.get("temporal_gap_sec"), item.get("relation_basis")) for item in items]


def _payload_entities(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index evidence entities without counting their frame/audio assets."""
    return {
        item["evidence_id"]: item
        for group in payload.get("evidence_groups", [])
        for field in ("visual_evidence", "speech_evidence", "acoustic_evidence")
        for item in group.get(field, [])
    }


def compare_state(state: CaseState, frozen: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare research-relevant semantics and classify every difference."""
    plan, retrieval, sufficiency, packet, payload = state.planner_output, state.retrieval_result, state.sufficiency_result, state.evidence_packet, state.final_payload
    frozen_plan, frozen_retrieval, frozen_sufficiency, frozen_packet, frozen_payload = (frozen[key] for key in ("planner", "retrieval", "sufficiency", "packet", "payload"))
    frozen_stable_relations, relation_audit = stabilize_relations(
        frozen_packet["relations"],
        (item["candidate_id"] for item in frozen_packet["retained_candidates"]),
    )
    payload_ids = payload_evidence_ids(payload or {})
    frozen_entities = _payload_entities(frozen_payload)
    current_entities = _payload_entities(payload or {})
    checks = {
        "planner.operation": (plan or {}).get("answer_requirement", {}).get("operation") == frozen_plan["plan"]["answer_requirement"]["operation"],
        "planner.resolver_modalities": (plan or {}).get("resolver_modalities") == frozen_plan["plan"]["resolver_modalities"],
        "planner.temporal_relation": (plan or {}).get("temporal_relation") == frozen_plan["plan"]["temporal_relation"],
        "planner.routing": all((plan or {}).get(key) == frozen_plan["plan"].get(key) for key in ("primary_anchor_modality", "audio_role", "requires_local_visual_inspection", "visual_route")),
        "retrieval.candidate_ids": _ids((retrieval or {}).get("all_candidates", [])) == _ids(frozen_retrieval["all_candidates"]),
        "retrieval.intervals": _intervals((retrieval or {}).get("all_candidates", [])) == _intervals(frozen_retrieval["all_candidates"]),
        "retrieval.selected_ids": _ids((retrieval or {}).get("selected_candidates", [])) == _ids(frozen_retrieval["selected_candidates"]),
        "retrieval.modalities": (retrieval or {}).get("executed_modalities") == frozen_retrieval["executed_modalities"],
        "retrieval.anchor_resolution": (retrieval or {}).get("anchor_resolution") == frozen_retrieval["anchor_resolution"],
        "sufficiency.status": (sufficiency or {}).get("evidence_status") == frozen_sufficiency["evidence_status"],
        "sufficiency.missing": (sufficiency or {}).get("critical_missing_evidence") == frozen_sufficiency["critical_missing_evidence"],
        "sufficiency.fallback_decision": (sufficiency or {}).get("fallback_was_triggered") == frozen_sufficiency["fallback_was_triggered"],
        "sufficiency.post_candidate_ids": _ids((sufficiency or {}).get("post_fallback_candidates", [])) == _ids(frozen_sufficiency["post_fallback_candidates"]),
        "fallback.at_most_once": state.fallback_execution_count <= 1,
        "packet.retained_ids": _ids((packet or {}).get("retained_candidates", [])) == _ids(frozen_packet["retained_candidates"]),
        "packet.actually_dropped": (packet or {}).get("actually_dropped_candidates") == frozen_packet["actually_dropped_candidates"],
        "packet.relations_semantically_stable": _relations((packet or {}).get("relations", [])) == _relations(frozen_stable_relations),
        "packet.visual_frames": (packet or {}).get("selected_visual_frames") == frozen_packet["selected_visual_frames"],
        "packet.budget": (packet or {}).get("budget_accounting") == frozen_packet["budget_accounting"],
        "packet.ambiguities": (packet or {}).get("unresolved_ambiguities") == frozen_packet["unresolved_ambiguities"],
        "payload.evidence_ids_complete": sorted(payload_ids) == sorted(_ids((packet or {}).get("retained_candidates", []))),
        "payload.preexisting_evidence_content": all(current_entities.get(key) == value for key, value in frozen_entities.items()),
        "payload.required_modalities": (payload or {}).get("answer_required_modalities") == frozen_payload["answer_required_modalities"],
        "payload.uncertainties": (payload or {}).get("pipeline_uncertainties") == frozen_payload["pipeline_uncertainties"],
        "payload.leakage": bool(state.preflight_result and state.preflight_result["leakage_check_passed"]),
        "external_calls.zero": sum(state.external_calls.values()) == 0,
    }
    differences = [{"field": key, "classification": "research-behavior mismatch", "category": 4} for key, passed in checks.items() if not passed]
    technical_changes = []
    frozen_payload_ids = payload_evidence_ids(frozen_payload)
    if sorted(payload_ids) != sorted(frozen_payload_ids):
        technical_changes.append({
            "field": "payload.evidence_ids",
            "classification": "technical serialization/contract change",
            "added_already_retained_ids": sorted(set(payload_ids) - set(frozen_payload_ids)),
        })
    if relation_audit["removed_invalid_relations"] or relation_audit["removed_duplicate_relations"]:
        technical_changes.append({
            "field": "packet.relations",
            "classification": "technical serialization/contract change",
            "audit": relation_audit,
        })
    if payload != frozen_payload:
        technical_changes.append({
            "field": "payload.schema_or_contract_metadata",
            "classification": "technical serialization/contract change",
        })
    return {"case_id": state.case_id, "checks": checks, "exact_check_count": sum(checks.values()), "check_count": len(checks), "differences": differences, "technical_changes": technical_changes, "overall_classification": "research-equivalent_with_technical_contract_changes" if not differences and technical_changes else ("exact match" if not differences else "research-behavior mismatch"), "live_safe_to_proceed": not differences, "canonical_state_audit": copy.deepcopy(state.audit), "fallback_execution_count": state.fallback_execution_count, "external_calls": copy.deepcopy(state.external_calls)}
