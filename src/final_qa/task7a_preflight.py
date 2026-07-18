"""Provider-neutral Task 7A final-QA payload construction and validation."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any


ANSWER_STATUSES = ["answered", "answered_with_uncertainty", "insufficient_evidence", "query_or_premise_inconsistent"]

FINAL_ANSWER_POLICY = {
    "allowed_answer_statuses": ANSWER_STATUSES,
    "rules": [
        {"if": "answer-required evidence is missing", "then": "insufficient_evidence"},
        {"if": "a critical unresolved ambiguity directly affects the answer", "then": "answered_with_uncertainty or abstain"},
        {"if": "observed evidence contradicts a required question premise", "then": "query_or_premise_inconsistent"},
        {"if": "evidence supports an answer but material uncertainty remains", "then": "answered_with_uncertainty"},
        {"if": "otherwise", "then": "answered"},
    ],
    "must_not_force_answer": True,
    "pipeline_uncertainties_must_be_preserved": True,
    "output_validation_rules": [
        "Every cited evidence_id must exist in the provided payload.",
        "The final model may not cite unseen evidence.",
        "answered cannot be used when a critical answer-affecting uncertainty remains.",
        "insufficient_evidence or query_or_premise_inconsistent may have a null answer.",
        "Confidence alone cannot override this deterministic policy.",
    ],
}

REQUIRED_OUTPUT_SCHEMA = {
    "answer_status": "answered | answered_with_uncertainty | insufficient_evidence | query_or_premise_inconsistent",
    "answer": "string or null",
    "reasoning_summary": "brief evidence-grounded explanation",
    "evidence_used": [{"evidence_id": "...", "modality": "visual | speech | acoustic", "start_sec": 0.0, "end_sec": 0.0, "supported_claim": "..."}],
    "uncertainties": [{"type": "...", "description": "...", "affected_claim": "...", "severity": "minor | material | critical", "source": "pipeline | final_model", "resolvable_with": "string or null"}],
    "missing_information": [],
    "confidence": {"level": "high | medium | low", "basis": "evidence-grounded explanation"},
    "abstain": False,
}

FORBIDDEN_FIELD_FRAGMENTS = (
    "ground_truth", "weak_reference", "reference_interval", "posthoc_evaluation",
    "human_review", "manual_listening", "benchmark_correctness", "expected_answer",
    "answer_options", "provided_timestamp",
)


def resolved_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def path_is_file(root: Path, value: str | None) -> bool:
    path = resolved_path(root, value)
    return bool(path and path.is_file())


def effective_bounds(candidate: dict[str, Any]) -> tuple[float, float]:
    return float(candidate.get("effective_start_time", candidate["start_time"])), float(candidate.get("effective_end_time", candidate["end_time"]))


def validate_visual_frames(frames: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    timestamps = [float(frame["timestamp_sec"]) for frame in frames]
    normalized = [round(timestamp * 1000) for timestamp in timestamps]
    orders = [frame["presentation_order"] for frame in frames]
    invalid = [frame["frame_path"] for frame in frames if not path_is_file(root, frame["frame_path"])]
    return {
        "chronological_visual_order": timestamps == sorted(timestamps),
        "presentation_order_sequential": orders == list(range(1, len(frames) + 1)),
        "no_duplicate_normalized_timestamps": len(normalized) == len(set(normalized)),
        "invalid_paths": invalid,
    }


def uncertainty(record_type: str, description: str, affected_claim: str, severity: str, resolvable_with: str) -> dict[str, Any]:
    return {"type": record_type, "description": description, "affected_claim": affected_claim, "severity": severity, "source": "pipeline", "resolvable_with": resolvable_with}


def build_pipeline_uncertainties(packet: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required = set(packet["answer_required_modalities"])
    for item in packet.get("unresolved_ambiguities", []):
        if item == "unresolved_speaker_attribution":
            records.append(uncertainty("speaker_attribution_uncertainty", "Speaker identity is not established by the retained transcript evidence.", "speaker identity", "material", "speaker diarization or verified speaker evidence"))
        elif item == "multiple_plausible_responses":
            records.append(uncertainty("multiple_plausible_answers", "Multiple plausible response segments are intentionally retained without choosing one.", "which response is the intended response", "material", "speaker-verified or additional contextual evidence"))
        else:
            records.append(uncertainty("semantic_uncertainty", f"Inherited unresolved ambiguity: {item}", "final evidence interpretation", "material", "additional grounded evidence"))
    for candidate in packet["retained_candidates"]:
        modality = candidate["modality"]
        if candidate.get("speaker_attribution_warning"):
            severity = "material" if modality in required and set(candidate.get("roles", [])) & {"direct_evidence", "trigger", "plausible_response"} else "minor"
            records.append(uncertainty("speaker_attribution_uncertainty", candidate["speaker_attribution_warning"], "speaker identity", severity, "speaker diarization or verified speaker evidence"))
        if candidate.get("semantic_interpretation_pending"):
            severity = "material" if modality in required else "minor"
            records.append(uncertainty("semantic_uncertainty", "Raw local audio is available, but no pipeline component has semantically interpreted it.", "acoustic interpretation", severity, "final grounded audio-capable model"))
        if "fallback_recovered" in candidate.get("roles", []):
            records.append(uncertainty("fallback_recovered_evidence", "Speech evidence was recovered by a local ASR fallback and retains its fallback provenance.", "phrase occurrence evidence", "minor", "independent transcription or verification"))
    for item in packet.get("missing_information", []):
        records.append(uncertainty("evidence_missing", str(item), "required evidence", "critical", "retrieve or provide the missing evidence"))
    if packet.get("structural_evidence_status") == "questionable":
        records.append(uncertainty("semantic_uncertainty", "The upstream structural evidence status is questionable; this does not assert an answer is incorrect.", "final answer confidence", "material", "independent verification or additional grounded evidence"))
    if packet.get("dataset_or_query_inconsistency_status", "unknown") == "unknown":
        records.append(uncertainty("dataset_or_query_inconsistency_unknown", "No machine-detectable dataset or query inconsistency was established; status remains unknown.", "question premise", "minor", "external independent audit"))
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for item in records:
        deduplicated.setdefault((item["type"], item["affected_claim"]), item)
    return list(deduplicated.values())


def payload_leakage_audit(payload: dict[str, Any]) -> dict[str, Any]:
    matches: list[str] = []

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                lowered = str(key).casefold()
                if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
                    matches.append(f"field:{location}.{key}")
                walk(nested, f"{location}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{location}[{index}]")
        elif isinstance(value, str):
            lowered = value.casefold()
            if any(fragment in lowered for fragment in ("weak reference", "ground-truth", "human review", "expected answer", "benchmark correctness")):
                matches.append(f"value:{location}")

    walk(payload, "payload")
    return {"leakage_check_passed": not matches, "forbidden_matches": matches}


def union_duration(items: list[dict[str, Any]]) -> float:
    spans = sorted((float(item["start_sec"]), float(item["end_sec"])) for item in items)
    total, stop = 0.0, float("-inf")
    for start, end in spans:
        if start > stop:
            total += end - start
        elif end > stop:
            total += end - stop
        stop = max(stop, end)
    return total


def count_types(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["type"] for item in items).items()))
