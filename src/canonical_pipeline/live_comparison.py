"""Live-to-frozen semantic comparison for the two-case verification."""

from __future__ import annotations

from typing import Any

from .regression import _ids, _intervals, _relations
from .state import CaseState


def compare_live_state(state: CaseState, frozen: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify research-relevant live differences without comparing answer wording."""
    plan = state.planner_output or {}
    retrieval = state.retrieval_result or {}
    sufficiency = state.sufficiency_result or {}
    packet = state.evidence_packet or {}
    payload = state.final_payload or {}
    frozen_plan = frozen["planner"]["plan"]
    frozen_retrieval = frozen["retrieval"]
    frozen_sufficiency = frozen["sufficiency"]
    frozen_packet = frozen["packet"]
    frozen_payload = frozen["payload"]
    checks: list[dict[str, Any]] = []

    def add(stage: str, field: str, live: Any, expected: Any, *, nondeterministic: bool = False) -> None:
        if live == expected:
            classification, category = "exact match", 1
        elif nondeterministic:
            classification, category = "expected nondeterministic variation", 3
        else:
            classification, category = "research-behavior mismatch", 4
        checks.append({"stage": stage, "field": field, "frozen": expected, "live": live, "classification": classification, "category": category})

    add("planner", "operation", plan.get("answer_requirement", {}).get("operation"), frozen_plan["answer_requirement"]["operation"])
    add("planner", "resolver_modalities", plan.get("resolver_modalities"), frozen_plan["resolver_modalities"])
    add("planner", "temporal_relation", plan.get("temporal_relation"), frozen_plan["temporal_relation"])
    add("planner", "routing", {key: plan.get(key) for key in ("primary_anchor_modality", "audio_role", "requires_local_visual_inspection", "visual_route")}, {key: frozen_plan.get(key) for key in ("primary_anchor_modality", "audio_role", "requires_local_visual_inspection", "visual_route")})
    add("retrieval", "executed_modalities", retrieval.get("executed_modalities"), frozen_retrieval["executed_modalities"])
    add("retrieval", "candidate_ids", _ids(retrieval.get("all_candidates", [])), _ids(frozen_retrieval["all_candidates"]))
    add("retrieval", "selected_ids", _ids(retrieval.get("selected_candidates", [])), _ids(frozen_retrieval["selected_candidates"]))
    add("retrieval", "intervals", _intervals(retrieval.get("all_candidates", [])), _intervals(frozen_retrieval["all_candidates"]))
    add("sufficiency", "status", sufficiency.get("evidence_status"), frozen_sufficiency["evidence_status"])
    add("sufficiency", "fallback_triggered", sufficiency.get("fallback_was_triggered"), frozen_sufficiency["fallback_was_triggered"])
    live_fallback_ids = _ids(sufficiency.get("post_fallback_candidates", []))
    frozen_fallback_ids = _ids(frozen_sufficiency["post_fallback_candidates"])
    fallback_variation = bool(state.fallback_execution_count and sufficiency.get("fallback_recovered_evidence"))
    add("fallback", "post_candidate_ids", live_fallback_ids, frozen_fallback_ids, nondeterministic=fallback_variation)
    add("fallback", "execution_count_at_most_once", state.fallback_execution_count <= 1, True)
    packet_nondeterministic = fallback_variation and live_fallback_ids != frozen_fallback_ids
    add("reranking", "retained_ids", _ids(packet.get("retained_candidates", [])), _ids(frozen_packet["retained_candidates"]), nondeterministic=packet_nondeterministic)
    add("reranking", "relations", _relations(packet.get("relations", [])), _relations(frozen_packet["relations"]), nondeterministic=packet_nondeterministic)
    add("reranking", "budget", packet.get("budget_accounting"), frozen_packet["budget_accounting"])
    add("reranking", "ambiguities", packet.get("unresolved_ambiguities"), frozen_packet["unresolved_ambiguities"])
    add("reranking", "visual_frame_order", [item.get("timestamp") for item in packet.get("selected_visual_frames", [])], [item.get("timestamp") for item in frozen_packet["selected_visual_frames"]])
    evidence_ids = sorted(item["evidence_id"] for group in payload.get("evidence_groups", []) for key in ("visual_evidence", "speech_evidence", "acoustic_evidence") for item in group[key])
    frozen_ids = sorted(item["evidence_id"] for group in frozen_payload["evidence_groups"] for key in ("visual_evidence", "speech_evidence", "acoustic_evidence") for item in group[key])
    add("payload", "evidence_ids", evidence_ids, frozen_ids, nondeterministic=packet_nondeterministic)
    add("payload", "answer_required_modalities", payload.get("answer_required_modalities"), frozen_payload["answer_required_modalities"])
    add("payload", "leakage_passed", bool(state.preflight_result and state.preflight_result.get("leakage_check_passed")), True)
    highest = max(item["category"] for item in checks)
    return {
        "case_id": state.case_id,
        "checks": checks,
        "overall_category": highest,
        "overall_classification": {1: "exact match", 2: "equivalent / harmless model-origin variation", 3: "expected nondeterministic variation", 4: "research-behavior mismatch"}[highest],
        "category_4_detected": highest == 4,
    }

