from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_3_local import core as _base
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import FINAL_SYSTEM, _gemini_call
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import _final_schema


preflight = _base.preflight
evaluate = _base.evaluate


def _downgrade_inconsistent_resolved_investigation(
    investigation: dict[str, Any], evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    downgraded = _base_downgrade(investigation, evidence)
    if downgraded["investigation_status"] != "resolved":
        return downgraded
    if downgraded["established_facts"].strip() and not downgraded["requested_coarse_ids"]:
        return downgraded
    downgraded = copy.deepcopy(downgraded)
    downgraded["investigation_status"] = "unresolved"
    downgraded["gap_reason"] = downgraded["gap_reason"].strip() or (
        "Model returned investigation_status=resolved while established_facts was empty or additional "
        "Coarse regions were still requested; deterministically downgraded to unresolved."
    )
    downgraded["established_facts"] = ""
    downgraded["atomic_facts"] = []
    return downgraded


def _indexed_batch_schema(indexes: list[int], requirement_ids: list[str]) -> dict[str, Any]:
    observation = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "fine_id": {"type": "integer", "enum": indexes},
            "finding": {"type": "string"},
            "visible_actions": {"type": "array", "items": {"type": "string"}},
            "visible_objects": {"type": "array", "items": {"type": "string"}},
            "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["fine_id", "finding", "visible_actions", "visible_objects", "uncertainty_notes"],
    }
    assessment = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "claim_status": {"type": "string", "enum": ["confirmed", "refuted", "inconclusive"]},
            "supporting_fine_ids": {"type": "array", "items": {"type": "integer", "enum": indexes}},
            "rationale": {"type": "string"},
        },
        "required": ["claim_status", "supporting_fine_ids", "rationale"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "observations": {
                "type": "array", "minItems": len(indexes), "maxItems": len(indexes), "items": observation,
            },
            "claim_assessments": {
                "type": "object", "additionalProperties": False,
                "properties": {rid: assessment for rid in requirement_ids}, "required": requirement_ids,
            },
        },
        "required": ["observations", "claim_assessments"],
    }


def _restore_indexed_fine_ids(value: dict[str, Any], fine_ids: list[str]) -> dict[str, Any]:
    restored = copy.deepcopy(value)
    for position, observation in enumerate(restored["observations"]):
        index = observation["fine_id"]
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise ValueError("Batch claim execution Fine integer coverage/order mismatch")
        observation["fine_id"] = fine_ids[index]
    for assessment in restored["claim_assessments"].values():
        canonical = []
        for index in assessment["supporting_fine_ids"]:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(fine_ids):
                raise ValueError("Batch claim execution cited invalid Fine integer")
            canonical.append(fine_ids[index])
        assessment["supporting_fine_ids"] = canonical
    return restored


def _fine_coverage_feedback(value: dict[str, Any], expected_indexes: list[int], error: ValueError) -> str:
    observations = value.get("observations", [])
    observed = [row.get("fine_id") for row in observations if isinstance(row, dict)]
    missing = [index for index in expected_indexes if index not in observed]
    duplicates = sorted({index for index in observed if observed.count(index) > 1 and isinstance(index, int)})
    return (
        "VALIDATOR ERROR FROM THE PREVIOUS RESPONSE: " + str(error) + "\n"
        f"Expected exactly {len(expected_indexes)} observations in this exact order: {expected_indexes}.\n"
        f"Previous response returned {len(observed)} observation IDs: {observed}.\n"
        f"Missing positions: {missing}. Duplicate positions: {duplicates}.\n"
        "Regenerate the entire JSON response. Return one observation for every position, in exact ascending "
        "order, with no omissions, duplicates, reordering, or early termination. Preserve the required "
        "claim_assessments object after all observations."
    )


def _execute_claims_batch_with_indexed_fines(
    cfg: dict[str, Any], question: dict[str, Any], claims_or_gaps_by_requirement: dict[str, str],
    unique_fines: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fine_ids = [row["fine_id"] for row in unique_fines]
    indexes = list(range(len(fine_ids)))
    requirement_ids = list(claims_or_gaps_by_requirement)
    payload = {
        "question_id": question["question_id"],
        "claims_or_gaps_to_investigate": claims_or_gaps_by_requirement,
        "ordered_images": [
            {"fine_id": index, "timestamp_sec": row["timestamp_sec"]}
            for index, row in enumerate(unique_fines)
        ],
    }
    inputs: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    for index, row in enumerate(unique_fines):
        image = Path(row["source_frame_path"])
        inputs.extend([
            {"type": "text", "text": f"FINE {index} timestamp={float(row['timestamp_sec']):.3f}s"},
            {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image.read_bytes()).decode("ascii")},
        ])
    schema = _indexed_batch_schema(indexes, requirement_ids)
    attempts = int(cfg.get("max_validation_retries", 2))
    last_error: ValueError | None = None
    retry_feedback: str | None = None
    for attempt in range(attempts):
        attempt_inputs = list(inputs)
        if retry_feedback is not None:
            attempt_inputs.append({"type": "text", "text": retry_feedback})
        result, usage, raw = _gemini_call(
            {"gemini": cfg["gemini"]}, _base.BATCH_CLAIM_EXECUTION_SYSTEM, attempt_inputs, schema,
        )
        try:
            restored = _restore_indexed_fine_ids(result, fine_ids)
            _base._validate_batch_claim_execution(restored, fine_ids, requirement_ids)
            usage["fine_id_adapter"] = "zero_based_position_to_canonical_fine_id_v1"
            usage["validator_feedback_retry"] = attempt > 0
            return restored, usage, raw
        except ValueError as error:
            last_error = error
            retry_feedback = _fine_coverage_feedback(result, indexes, error)
    raise last_error


def _final_schema_with_evidence_enum(
    question: dict[str, Any], requirements: list[dict[str, Any]], evidence: list[dict[str, Any]],
    allowed_option_ids: list[str] | None = None,
) -> dict[str, Any]:
    schema = _final_schema(question, requirements)
    evidence_ids = [row["evidence_id"] for row in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Final evidence payload contains duplicate evidence IDs")
    schema["properties"]["supporting_evidence_ids"]["items"]["enum"] = evidence_ids
    if allowed_option_ids is not None:
        schema["properties"]["selected_option_id"]["enum"] = allowed_option_ids
    return schema


def _allowed_final_option_ids(question: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    status_by_requirement = {
        row["requirement_id"]: row["status"] for row in resolved.get("assessments", [])
    }
    allowed = []
    for option in question["answer_options"]:
        option_id = option["option_id"]
        requirement_id = f"{question['question_id']}::option_{option_id.lower()}"
        if status_by_requirement.get(requirement_id) != "not_found":
            allowed.append(option_id)
    if not allowed:
        raise ValueError("Final answer has no non-refuted option; upstream option mapping is inconsistent")
    return allowed


def _validate_final_consistency(
    result: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    evidence: list[dict[str, Any]], allowed_option_ids: list[str],
) -> None:
    known_evidence = {row["evidence_id"] for row in evidence}
    known_requirements = {row["requirement_id"] for row in requirements}
    if (
        result["question_id"] != question["question_id"]
        or set(result["supporting_evidence_ids"]) - known_evidence
        or set(result["supporting_requirement_ids"]) - known_requirements
    ):
        raise ValueError("Final answer provenance invalid")
    if result["selected_option_id"] not in allowed_option_ids:
        raise ValueError(
            f"Final selected refuted option {result['selected_option_id']}; "
            f"allowed non-refuted options are {allowed_option_ids}"
        )


def _final_consistency_feedback(
    result: dict[str, Any], allowed_option_ids: list[str], error: ValueError,
) -> str:
    return (
        "VALIDATOR ERROR FROM THE PREVIOUS RESPONSE: " + str(error) + "\n"
        f"The previous response selected option {result.get('selected_option_id')!r}. "
        f"The only non-refuted option IDs are {allowed_option_ids}.\n"
        "Regenerate the entire JSON response. Select one of the allowed option IDs using the supplied "
        "resolved assessments and evidence. Do not select an option already marked not_found/refuted, "
        "and preserve valid requirement and evidence provenance."
    )


def _final_with_exact_evidence_ids(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    resolved: dict[str, Any], evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = {
        "question": question,
        "requirements": requirements,
        "resolved_assessments": resolved,
        "evidence": evidence,
    }
    allowed_option_ids = _allowed_final_option_ids(question, resolved)
    schema = _final_schema_with_evidence_enum(question, requirements, evidence, allowed_option_ids)
    inputs: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    attempts = int(cfg.get("max_validation_retries", 2))
    last_error: ValueError | None = None
    for attempt in range(attempts):
        result, usage, raw = _gemini_call({"gemini": cfg["gemini"]}, FINAL_SYSTEM, inputs, schema)
        try:
            _validate_final_consistency(result, question, requirements, evidence, allowed_option_ids)
            usage["final_evidence_id_adapter"] = "exact_payload_evidence_enum_v1"
            usage["allowed_evidence_id_count"] = len(evidence)
            usage["final_option_consistency_adapter"] = "exclude_refuted_option_enum_v1"
            usage["allowed_final_option_ids"] = allowed_option_ids
            usage["final_validator_feedback_retry"] = attempt > 0
            return result, usage, raw
        except ValueError as error:
            last_error = error
            inputs = [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                {"type": "text", "text": _final_consistency_feedback(result, allowed_option_ids, error)},
            ]
    raise last_error


def run_live(root, config_path, video_uid=None):
    original = _base._final
    original_downgrade = _base._downgrade_noncompliant_investigation
    original_execution = _base._execute_claims_batch
    global _base_downgrade
    _base_downgrade = original_downgrade
    _base._final = _final_with_exact_evidence_ids
    _base._downgrade_noncompliant_investigation = _downgrade_inconsistent_resolved_investigation
    _base._execute_claims_batch = _execute_claims_batch_with_indexed_fines
    try:
        return _base.run_live(root, config_path, video_uid=video_uid)
    finally:
        _base._final = original
        _base._downgrade_noncompliant_investigation = original_downgrade
        _base._execute_claims_batch = original_execution
