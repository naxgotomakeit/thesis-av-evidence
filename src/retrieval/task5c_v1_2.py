from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import soundfile as sf

from src.retrieval.task5c import read_local_audio


EPSILON = 1e-4


def has_explicit_time(cues: dict[str, Any]) -> bool:
    return any(item.get("start_sec") is not None for item in cues.get("time_cues", []))


def timestamp_semantics(question: str, cues: dict[str, Any], anchor_resolution: dict[str, Any]) -> dict[str, str]:
    lower = question.casefold()
    if has_explicit_time(cues):
        if re.search(r"\b(after|followed|following|before)\b", lower):
            return {"timestamp_semantics": "relative_search_interval", "anchor_timestamp_semantics": "event_anchor_interval"}
        return {"timestamp_semantics": "direct_target_interval", "anchor_timestamp_semantics": "direct_target_interval"}
    if re.search(r"\bat the (very )?start\b", lower):
        return {"timestamp_semantics": "approximate_temporal_phrase", "anchor_timestamp_semantics": "approximate_temporal_phrase"}
    if anchor_resolution.get("transcript_matches"):
        return {"timestamp_semantics": "speech_anchor_interval", "anchor_timestamp_semantics": "speech_anchor_interval"}
    if anchor_resolution.get("final_search_intervals"):
        return {"timestamp_semantics": "retrieval_discovered_interval", "anchor_timestamp_semantics": "unknown"}
    return {"timestamp_semantics": "unknown", "anchor_timestamp_semantics": "unknown"}


def acoustic_evidence_role(plan: dict[str, Any]) -> str:
    resolvers = set(plan.get("resolver_modalities", []))
    operation = plan.get("answer_requirement", {}).get("operation", "other")
    if "acoustic" not in resolvers:
        return "not_required"
    if operation == "describe_sound":
        return "direct_evidence"
    if operation == "measure_delay" and "speech" in resolvers:
        return "supporting"
    if plan.get("requires_local_visual_inspection") and operation in {"identify_object", "identify_source", "identify_person"}:
        return "temporal_anchor"
    if plan.get("audio_role") == "temporal_anchor":
        return "temporal_anchor"
    if plan.get("audio_role") == "direct_answer" and "visual" not in resolvers:
        return "direct_evidence"
    return "resolver"


def broad_score_diagnostic(
    candidate: dict[str, Any], *, role: str, semantics: dict[str, str], local_audio_clip_reference: str | None,
    broad_score_used_for_semantic_verification: bool = False, tolerance: float = EPSILON,
) -> dict[str, Any]:
    source_start, source_end = candidate.get("source_start_time"), candidate.get("source_end_time")
    selected_start, selected_end = float(candidate["start_time"]), float(candidate["end_time"])
    clipped = source_start is not None and source_end is not None and (abs(float(source_start) - selected_start) > tolerance or abs(float(source_end) - selected_end) > tolerance)
    local_score = any(candidate.get(key) is not None for key in ("local_acoustic_score", "local_acoustic_similarity_score", "locally_recomputed_acoustic_score"))
    scope = "local_selected_interval" if local_score else ("broader_source_region" if clipped and candidate.get("similarity_score") is not None else "none")
    local_available = bool(local_audio_clip_reference)
    independently_anchored = semantics["timestamp_semantics"] in {"direct_target_interval", "relative_search_interval", "speech_anchor_interval", "approximate_temporal_phrase"}
    broad_used_for_localization = scope == "broader_source_region" and not independently_anchored
    broad_used_for_semantic_verification = bool(broad_score_used_for_semantic_verification)
    warning = clipped and scope == "broader_source_region"
    affects = warning and role in {"direct_evidence", "temporal_anchor", "resolver"} and (broad_used_for_localization or not local_available or broad_used_for_semantic_verification)
    if not warning:
        effect = "no_broad_source_score_provenance"
    elif affects and broad_used_for_localization:
        effect = "selected_interval_is_not_independently_temporally_anchored"
    elif affects and not local_available:
        effect = "required_local_audio_evidence_unavailable"
    elif affects:
        effect = "broad_score_used_as_local_semantic_proof"
    else:
        effect = "broad_score_retained_as_retrieval_provenance_only_with_anchored_local_audio"
    warnings = ["broad_source_acoustic_region", "local_acoustic_semantics_not_verified"] if warning else []
    return {
        "candidate_id": candidate["candidate_id"],
        "source_acoustic_region_interval": {"start_sec": source_start, "end_sec": source_end},
        "selected_acoustic_interval": {"start_sec": selected_start, "end_sec": selected_end},
        "timestamp_semantics": semantics["timestamp_semantics"], "anchor_timestamp_semantics": semantics["anchor_timestamp_semantics"],
        "acoustic_evidence_role": role, "clap_similarity_score": candidate.get("similarity_score"), "score_scope": scope,
        "local_acoustic_score_recomputed": local_score, "local_audio_clip_reference": local_audio_clip_reference,
        "broad_score_used_for_localization": broad_used_for_localization, "broad_score_used_for_semantic_verification": broad_used_for_semantic_verification,
        "local_audio_evidence_available": local_available, "broad_source_warning": warning, "broad_source_affects_sufficiency": affects,
        "sufficiency_effect_reason": effect, "warnings": warnings,
        "structural_evidence_available": local_available if role in {"direct_evidence", "temporal_anchor", "resolver"} else False,
        "semantic_interpretation_pending": local_available,
    }


def recompute_structural_status(source_assessment: dict[str, Any], acoustic_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Remove broad-score-only penalties and add one only when generic conditions require it."""
    broad_codes = {"broad_source_region", "broad_source_acoustic_region", "local_acoustic_semantics_not_verified"}
    critical = copy.deepcopy(source_assessment.get("critical_missing_evidence", []))
    codes = set(source_assessment.get("sufficiency_reason_codes", [])) - broad_codes
    ambiguity = set(source_assessment.get("ambiguity_flags", [])) - broad_codes
    affecting = [item for item in acoustic_diagnostics if item["broad_source_affects_sufficiency"]]
    if affecting:
        codes.update({"broad_source_acoustic_region", "local_acoustic_semantics_not_verified"})
        ambiguity.update({"broad_source_acoustic_region", "local_acoustic_semantics_not_verified"})
    if critical:
        status = "insufficient"
    elif ambiguity:
        status = "questionable"
    else:
        status = "sufficient"
    return {"evidence_status": status, "sufficiency_reason_codes": sorted(codes), "critical_missing_evidence": critical, "ambiguity_flags": sorted(ambiguity), "broad_source_affecting_candidate_ids": [item["candidate_id"] for item in affecting], "structural_sufficiency_only": True, "interpretation": "Local raw audio is structural evidence; acoustic semantic interpretation remains pending."}


def export_local_clip(source_audio: Path, output_path: Path, start_sec: float, end_sec: float) -> dict[str, Any]:
    audio, sample_rate = read_local_audio(source_audio, start_sec, end_sec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return {"requested_selected_interval": {"start_sec": start_sec, "end_sec": end_sec}, "actual_extracted_interval": {"start_sec": start_sec, "end_sec": start_sec + len(audio) / sample_rate}, "source_audio_path": str(source_audio), "clip_path": str(output_path), "duration_sec": len(audio) / sample_rate, "sample_rate": sample_rate, "source_audio_unchanged": True}
