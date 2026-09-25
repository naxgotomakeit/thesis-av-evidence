from __future__ import annotations

import base64
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_3_local import core as _base
from experiments.hourvideo_r1_av_r3_2_single_video_smoke import live_runner as _live_runner
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import FINAL_SYSTEM
from experiments.hourvideo_v6_3_local_contract_fixes_v1 import core as _contract
from experiments.hourvideo_v6_3_local_final_evidence_enum_v1 import core as _enum


preflight = _contract.preflight
evaluate = _contract.evaluate

TEMPORAL_RELATIONS = ("earlier", "later", "unknown", "not_applicable")
_PAIRWISE_FIRST = re.compile(
    r"^\s*which\s+event\s+happened\s+first\s*:\s*(.+?)\s+or\s+(.+?)\s*\?\s*$",
    re.IGNORECASE,
)
_EARLIER_CUE = re.compile(r"\b(before|earlier|first|preceded)\b", re.IGNORECASE)
_LATER_CUE = re.compile(r"\b(after|later|second)\b|\bfollowed\s+by\b", re.IGNORECASE)

TEMPORAL_MAPPING_SYSTEM = """

The payload may contain temporal_operator.kind="pairwise_first". For that operator, the two named
candidate options must each report temporal_relation_to_other as "earlier", "later", or "unknown";
all other distractor options must report "not_applicable". Judge which candidate answers FIRST, not
merely whether its event occurred. The status and reason must agree with the relation: earlier means
supported and the reason must explicitly say that this option happened before/earlier than the other;
later means refuted and the reason must explicitly say after/later; unknown means unresolved. Either
return one earlier plus one later, or return unknown for both. Do not infer the answer from the option
order. Regenerate the full mapping if validator_feedback is present in the payload.
"""


class TemporalOperatorMappingError(ValueError):
    """Terminal mapping failure that preserves both model attempts for audit."""

    def __init__(self, errors: list[str], usages: list[dict[str, Any]]):
        self.errors = list(errors)
        self.usages = copy.deepcopy(usages)
        super().__init__(
            "Pairwise FIRST option mapping remained inconsistent after validator-feedback retry; "
            f"errors={self.errors}"
        )


class HomogeneousAllRefutedError(ValueError):
    """All options were rejected without option-specific distinguishing grounds."""

    def __init__(self, errors: list[str], usages: list[dict[str, Any]]):
        self.errors = list(errors)
        self.usages = copy.deepcopy(usages)
        super().__init__(
            "All-refuted option mapping remained non-distinguishing after validator-feedback retry; "
            f"errors={self.errors}"
        )


class EmptySchemaEnumError(ValueError):
    """Reject a schema locally before an empty enum can crash the vLLM grammar engine."""


def _validate_no_empty_schema_enums(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        if value.get("enum") == []:
            raise EmptySchemaEnumError(f"JSON Schema contains an empty enum at {path}.enum")
        for key, child in value.items():
            _validate_no_empty_schema_enums(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_empty_schema_enums(child, f"{path}[{index}]")


def _local_vllm_api_schema(value: Any) -> Any:
    """Keep probed xgrammar bounds while dropping unsupported provider keywords."""
    if isinstance(value, dict):
        unsupported = {"$schema", "uniqueItems", "minItems", "minLength", "minimum", "maximum"}
        return {
            key: _local_vllm_api_schema(child)
            for key, child in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_local_vllm_api_schema(child) for child in value]
    return value


def _bounded_indexed_batch_schema(
    indexes: list[int], requirement_ids: list[str],
) -> dict[str, Any]:
    """Bound only representation size; keep every observation field as free text."""
    schema = _enum._indexed_batch_schema(indexes, requirement_ids)
    observation = schema["properties"]["observations"]["items"]["properties"]
    observation["finding"]["maxLength"] = 512
    for field in ("visible_actions", "visible_objects", "uncertainty_notes"):
        observation[field]["maxItems"] = 8
        observation[field]["items"]["maxLength"] = 256
    for assessment in schema["properties"]["claim_assessments"]["properties"].values():
        properties = assessment["properties"]
        properties["supporting_fine_ids"]["maxItems"] = len(indexes)
        properties["rationale"]["maxLength"] = 1024
    _validate_no_empty_schema_enums(schema)
    return schema


def _claim_execution_retry_feedback(error: Exception, indexes: list[int]) -> str:
    return (
        "VALIDATOR ERROR FROM THE PREVIOUS RESPONSE: " + str(error) + "\n"
        "The previous structured response either reached max_tokens before closing or failed the "
        "Fine coverage/order contract. Regenerate the entire JSON response from the supplied images. "
        "Do not repeat phrases or object lists. Preserve freely chosen visual content, but express each "
        "observation once and continue through every Fine position in exact ascending order: "
        f"{indexes}. After the last observation, emit claim_assessments and close the object."
    )


def _execute_claims_batch_with_bounded_text_retry(
    cfg: dict[str, Any], question: dict[str, Any],
    claims_or_gaps_by_requirement: dict[str, str], unique_fines: list[dict[str, Any]],
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
    inputs: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
    ]
    for index, row in enumerate(unique_fines):
        image = Path(row["source_frame_path"])
        inputs.extend([
            {"type": "text", "text": f"FINE {index} timestamp={float(row['timestamp_sec']):.3f}s"},
            {
                "type": "image", "mime_type": "image/jpeg",
                "data": base64.b64encode(image.read_bytes()).decode("ascii"),
            },
        ])

    schema = _bounded_indexed_batch_schema(indexes, requirement_ids)
    attempts = max(1, int(cfg.get("max_validation_retries", 2)))
    errors: list[str] = []
    failed_attempts: list[dict[str, Any]] = []
    successful_usages: list[dict[str, Any]] = []
    attempt_telemetry: list[dict[str, Any]] = []
    raw_attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(attempts):
        attempt_inputs = list(inputs)
        if errors:
            attempt_inputs.append({
                "type": "text",
                "text": _claim_execution_retry_feedback(last_error or ValueError(errors[-1]), indexes),
            })
        started = time.perf_counter()
        try:
            result, usage, raw = _gemini_call_with_schema_preflight(
                {"gemini": cfg["gemini"]}, _base.BATCH_CLAIM_EXECUTION_SYSTEM,
                attempt_inputs, schema,
            )
        except RuntimeError as error:
            if "structured response reached max_tokens" not in str(error):
                raise
            last_error = error
            errors.append(str(error))
            failed = copy.deepcopy(getattr(error, "attempt_telemetry", {}))
            failed.update({
                "attempt": attempt + 1,
                "error": str(error),
                "latency_sec": failed.get("latency_sec", time.perf_counter() - started),
                "output_tokens": failed.get(
                    "output_tokens", int(cfg["gemini"]["max_output_tokens"]),
                ),
                "input_tokens": failed.get("input_tokens"),
            })
            failed_attempts.append(failed)
            attempt_telemetry.append({
                **copy.deepcopy(failed),
                "status": "max_tokens",
                "failure_reason": str(error),
                "retry_triggered": attempt + 1 < attempts,
            })
            continue

        successful_usages.append(usage)
        raw_attempts.append(raw)
        try:
            restored = _enum._restore_indexed_fine_ids(result, fine_ids)
            _base._validate_batch_claim_execution(restored, fine_ids, requirement_ids)
            for assessment in restored["claim_assessments"].values():
                cited = assessment["supporting_fine_ids"]
                if len(cited) != len(set(cited)):
                    raise ValueError("Batch claim execution supporting Fine IDs must be unique")
        except ValueError as error:
            last_error = error
            errors.append(str(error))
            failed = {
                **copy.deepcopy(usage),
                "attempt": attempt + 1,
                "error": str(error),
                "latency_sec": usage.get("latency_sec"),
                "output_tokens": usage.get("output_tokens"),
                "input_tokens": usage.get("input_tokens"),
            }
            failed_attempts.append(failed)
            attempt_telemetry.append({
                **copy.deepcopy(failed),
                "status": "validation_failed",
                "failure_reason": str(error),
                "retry_triggered": attempt + 1 < attempts,
            })
            continue

        attempt_telemetry.append({
            **copy.deepcopy(usage),
            "attempt": attempt + 1,
            "status": "success",
            "failure_reason": None,
            "retry_triggered": False,
        })
        combined = dict(usage)
        combined["latency_sec"] = sum(
            float(row.get("latency_sec") or 0.0)
            for row in attempt_telemetry
        )
        combined["output_tokens"] = sum(
            int(row.get("output_tokens") or 0)
            for row in attempt_telemetry
        )
        if all(row.get("input_tokens") is not None for row in attempt_telemetry):
            combined["input_tokens"] = sum(
                int(row.get("input_tokens") or 0)
                for row in attempt_telemetry
            )
        combined.update({
            "fine_id_adapter": "zero_based_position_to_canonical_fine_id_v1",
            "claim_execution_text_bounds_adapter": "free_text_generous_bounds_v1",
            "claim_execution_validator_feedback_retry": attempt > 0,
            "claim_execution_attempt_count": attempt + 1,
            "previous_validation_errors": errors,
            "failed_attempts": failed_attempts,
            "attempt_telemetry": attempt_telemetry,
            "attempt_telemetry_complete": all(
                row.get("input_tokens") is not None and row.get("output_tokens") is not None
                for row in attempt_telemetry
            ),
            "schema_projection": "preserve_xgrammar_max_items_max_length_v1",
        })
        return restored, combined, {
            "attempts": raw_attempts,
            "failed_attempts": failed_attempts,
            "restored_response": restored,
        }

    raise RuntimeError(
        "Claim execution remained invalid after bounded-output validator-feedback retry; "
        f"errors={errors}"
    ) from last_error


def _final_schema_with_indexed_evidence_guard(
    question: dict[str, Any], requirements: list[dict[str, Any]], evidence: list[dict[str, Any]],
    allowed_option_ids: list[str] | None = None,
) -> dict[str, Any]:
    schema = _ORIGINAL_FINAL_SCHEMA(
        question, requirements, evidence, allowed_option_ids=allowed_option_ids,
    )
    evidence_array = schema["properties"]["supporting_evidence_ids"]
    indexes = list(range(len(evidence)))
    evidence_array["items"] = (
        {"type": "integer", "enum": indexes} if indexes else {"type": "integer"}
    )
    evidence_array["maxItems"] = len(indexes)
    schema["properties"]["supporting_requirement_ids"]["maxItems"] = len(requirements)
    schema["properties"]["caveats"]["maxItems"] = 3
    _validate_no_empty_schema_enums(schema)
    return schema


def _gemini_call_with_schema_preflight(config, system, inputs, schema):
    _validate_no_empty_schema_enums(schema)
    original_api_schema = _live_runner._api_schema
    _live_runner._api_schema = _local_vllm_api_schema
    try:
        return _ORIGINAL_GEMINI_CALL(config, system, inputs, schema)
    finally:
        _live_runner._api_schema = original_api_schema


def _restore_final_evidence_indexes(
    result: dict[str, Any], evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    restored = copy.deepcopy(result)
    indexes = restored.get("supporting_evidence_ids", [])
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indexes):
        raise ValueError("Final supporting evidence indexes must be integers")
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"Final supporting evidence indexes must be unique; got {indexes}")
    if any(index < 0 or index >= len(evidence) for index in indexes):
        raise ValueError(
            f"Final supporting evidence index outside 0..{len(evidence) - 1}: {indexes}"
        )
    requirement_ids = restored.get("supporting_requirement_ids", [])
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("Final supporting requirement IDs must be unique")
    restored["supporting_evidence_ids"] = [evidence[index]["evidence_id"] for index in indexes]
    return restored


def _final_with_indexed_evidence_guard(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    resolved: dict[str, Any], evidence: list[dict[str, Any]],
):
    indexed_evidence = [
        {"evidence_index": index, **row} for index, row in enumerate(evidence)
    ]
    payload = {
        "question": question,
        "requirements": requirements,
        "resolved_assessments": resolved,
        "evidence": indexed_evidence,
        "evidence_index_contract": (
            "Return integer evidence_index values in supporting_evidence_ids. Each index may appear "
            "at most once. The client maps valid indexes back to canonical evidence IDs."
        ),
    }
    schema = _final_schema_with_indexed_evidence_guard(question, requirements, evidence)
    base_input = {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
    attempts = int(cfg.get("max_validation_retries", 2))
    refuted = _contract._refuted_option_ids(question, resolved)
    errors: list[str] = []
    raw_attempts: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    for attempt in range(attempts):
        inputs = [base_input]
        if errors:
            inputs.append({
                "type": "text",
                "text": (
                    "VALIDATOR ERROR FROM THE PREVIOUS RESPONSE: " + errors[-1] + "\n"
                    "Regenerate the complete compact JSON. supporting_evidence_ids must contain only "
                    "unique integer evidence_index values from the supplied evidence list. Do not repeat "
                    "an index. Cite only evidence actually used for the selected option."
                ),
            })
        result, usage, raw = _gemini_call_with_schema_preflight(
            {"gemini": cfg["gemini"]}, FINAL_SYSTEM, inputs, schema,
        )
        usages.append(usage)
        raw_attempts.append(raw)
        try:
            restored = _restore_final_evidence_indexes(result, evidence)
            _contract._validate_final_provenance(
                restored, question, requirements, evidence,
            )
            selected = restored["selected_option_id"]
            if selected in refuted:
                raise ValueError(
                    f"Final selected option {selected}, but option mapping marked it refuted/not_found"
                )
            combined = dict(usage)
            for field in ("input_tokens", "output_tokens", "latency_sec"):
                values = [row.get(field) for row in usages]
                if all(isinstance(value, (int, float)) for value in values):
                    combined[field] = sum(values)
            combined.update({
                "final_evidence_id_adapter": "bounded_integer_index_to_canonical_id_v1",
                "allowed_evidence_id_count": len(evidence),
                "final_option_consistency_adapter": "refuted_conflict_feedback_retry_v1",
                "refuted_final_option_ids": refuted,
                "final_validator_feedback_retry": attempt > 0,
                "previous_validation_errors": errors,
                "schema_preflight_adapter": "reject_recursive_empty_enum_before_provider_v1",
                "local_schema_projection": "preserve_xgrammar_max_items_max_length_v1",
            })
            return restored, combined, {
                "attempts": raw_attempts,
                "restored_response": restored,
            }
        except ValueError as error:
            errors.append(str(error))
    raise ValueError(
        "Final indexed-evidence contract remained invalid after validator-feedback retry; "
        f"errors={errors}"
    )


def _event_key(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    value = re.sub(r"^(?:the\s+)?camera\s+wearer\s+", "", value)
    return " ".join(value.split())


def _option_text(option: dict[str, Any]) -> str:
    return str(option.get("text", option.get("answer_text", "")))


def _pairwise_first_operator(question: dict[str, Any]) -> dict[str, Any] | None:
    match = _PAIRWISE_FIRST.match(str(question.get("question_text", "")))
    if not match:
        return None
    events = [match.group(1).strip(), match.group(2).strip()]
    option_by_key: dict[str, list[str]] = {}
    for option in question.get("answer_options", []):
        option_by_key.setdefault(_event_key(_option_text(option)), []).append(option["option_id"])
    option_ids: list[str] = []
    for event in events:
        matches = option_by_key.get(_event_key(event), [])
        if len(matches) != 1:
            return None
        option_ids.append(matches[0])
    if option_ids[0] == option_ids[1]:
        return None
    return {
        "kind": "pairwise_first",
        "left_event": events[0],
        "right_event": events[1],
        "left_option_id": option_ids[0],
        "right_option_id": option_ids[1],
    }


def _temporal_mapping_schema(
    requirements: list[dict[str, Any]], fact_texts: list[str], operator: dict[str, Any],
) -> dict[str, Any]:
    schema = copy.deepcopy(_base._option_mapping_schema(requirements, fact_texts))
    item = schema["properties"]["option_assessments"]["items"]
    item["properties"]["temporal_relation_to_other"] = {
        "type": "string", "enum": list(TEMPORAL_RELATIONS),
    }
    item["required"].append("temporal_relation_to_other")
    return schema


def _candidate_requirement_id(question_id: str, option_id: str) -> str:
    return f"{question_id}::option_{option_id.lower()}"


def _validate_temporal_mapping(
    value: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    fact_texts: list[str], operator: dict[str, Any],
) -> None:
    # Keep the base structural/provenance checks, but do not apply its unscoped phrase heuristic to
    # the two compared candidates. A correct D reason may explicitly say "C is refuted"; scanning the
    # entire D reason for the bare phrase "is refuted" falsely treats that statement about C as D
    # arguing against itself. The structured relation guard below is scoped to each candidate instead.
    req_ids = [row["requirement_id"] for row in requirements]
    known_facts = set(fact_texts)
    if value.get("question_id") != question["question_id"]:
        raise ValueError("Option mapping question mismatch")
    seen = [row.get("requirement_id") for row in value.get("option_assessments", [])]
    if sorted(seen) != sorted(req_ids):
        raise ValueError("Option mapping coverage/uniqueness invalid")
    candidate_rids = {
        _candidate_requirement_id(question["question_id"], operator["left_option_id"]),
        _candidate_requirement_id(question["question_id"], operator["right_option_id"]),
    }
    rows = {row["requirement_id"]: row for row in value["option_assessments"]}
    for rid, row in rows.items():
        if not row["reason"].strip():
            raise ValueError(f"Option mapping reason must not be empty: {rid}")
        if not set(row["used_facts"]) <= known_facts:
            raise ValueError(f"Option mapping used_facts references an unknown fact: {rid}")
        if _base._reason_is_closed_world_refutation(row["status"], row["reason"]):
            raise ValueError(f"Option mapping refuted from silence, not positive contradiction: {rid}")
        if row["status"] in ("supported", "refuted") and not row["used_facts"]:
            raise ValueError(f"supported/refuted mapping must cite used_facts: {rid}")
        relation = row["temporal_relation_to_other"]
        if rid not in candidate_rids:
            if _base._claim_text_contradicts_status(row["status"], row["reason"]):
                raise ValueError(f"Option mapping reason argues the opposite of its own status: {rid}")
            if relation != "not_applicable":
                raise ValueError(
                    f"Temporal operator applies only to the two named candidate events; "
                    f"{rid} must use temporal_relation_to_other=not_applicable"
                )
            continue
        reason = row["reason"]
        if relation == "not_applicable":
            raise ValueError(f"Named temporal candidate cannot be not_applicable: {rid}")
        expected_status = {"earlier": "supported", "later": "refuted", "unknown": "unresolved"}[relation]
        if row["status"] != expected_status:
            raise ValueError(
                f"Temporal operator contradiction for {rid}: relation={relation} requires "
                f"status={expected_status}, got status={row['status']}"
            )
        if relation == "earlier" and not _EARLIER_CUE.search(reason):
            raise ValueError(
                f"Temporal operator contradiction for {rid}: supported FIRST candidate reason must "
                "explicitly say it happened before/earlier than the other candidate"
            )
        if relation == "later" and not _LATER_CUE.search(reason):
            raise ValueError(
                f"Temporal operator contradiction for {rid}: refuted FIRST candidate reason must "
                "explicitly say it happened after/later than the other candidate"
            )
        if relation == "unknown" and (_EARLIER_CUE.search(reason) or _LATER_CUE.search(reason)):
            raise ValueError(
                f"Temporal operator contradiction for {rid}: relation=unknown but reason asserts an order"
            )
    pair_relations = {rows[rid]["temporal_relation_to_other"] for rid in candidate_rids}
    if pair_relations not in ({"earlier", "later"}, {"unknown"}):
        raise ValueError(
            "Pairwise FIRST mapping must return one earlier and one later, or unknown for both; "
            f"got {sorted(pair_relations)}"
        )


def _combined_usage(usages: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    combined = dict(usages[-1])
    for field in ("input_tokens", "output_tokens", "latency_sec"):
        values = [row.get(field) for row in usages]
        if all(isinstance(value, (int, float)) for value in values):
            combined[field] = sum(values)
    combined["option_mapping_operator_adapter"] = "pairwise_first_relation_guard_v1"
    combined["option_mapping_validator_feedback_retry"] = len(usages) > 1
    combined["option_mapping_attempt_count"] = len(usages)
    combined["previous_validation_errors"] = errors
    combined["option_mapping_attempt_traces"] = [
        {
            "attempt": index,
            "response_id": row.get("response_id"),
            "raw_text": row.get("raw_text"),
            "validator_error": errors[index - 1] if index <= len(errors) else None,
        }
        for index, row in enumerate(usages, 1)
    ]
    return combined


def _reason_signature(reason: str) -> set[str]:
    normalized = re.sub(r"\boption\s+[a-z]\b", "option", reason.lower())
    return set(re.findall(r"[a-z0-9]+", normalized))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _validate_no_homogeneous_all_refuted(mapping: dict[str, Any]) -> None:
    rows = mapping["option_assessments"]
    if len(rows) < 2 or any(row["status"] != "refuted" for row in rows):
        return
    fact_sets = {tuple(sorted(row.get("used_facts", []))) for row in rows}
    signatures = [_reason_signature(row.get("reason", "")) for row in rows]
    similarities = [
        _jaccard(signatures[left], signatures[right])
        for left in range(len(signatures))
        for right in range(left + 1, len(signatures))
    ]
    if len(fact_sets) == 1 and similarities and min(similarities) >= 0.9:
        raise ValueError(
            "All options were refuted using the same verified fact and effectively the same reason. "
            "The mapping did not evaluate the clauses that distinguish the answer options. A wording "
            "difference or a fact that fails to address those distinguishing clauses is not a positive "
            "contradiction; use unresolved where the verified facts cannot decide an option."
        )


def _retry_homogeneous_all_refuted_once(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    verified_facts: list[dict[str, Any]], first_mapping: dict[str, Any], first_usage: dict[str, Any],
    first_error: ValueError,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fact_texts = [fact["text"] for fact in verified_facts]
    payload = {
        "question": question,
        "requirements": requirements,
        "verified_facts": verified_facts,
        "previous_mapping": first_mapping,
        "validator_feedback": (
            "VALIDATOR ERROR FROM THE PREVIOUS RESPONSE: " + str(first_error) + "\n"
            "Regenerate the complete option mapping. Compare the option-specific clauses that actually "
            "distinguish the candidates. Do not refute every option from one shared wording difference. "
            "A refuted status requires a verified fact that positively contradicts that particular "
            "option; otherwise use unresolved."
        ),
    }
    schema = _base._option_mapping_schema(requirements, fact_texts)
    usages = [first_usage]
    errors = [str(first_error)]
    try:
        mapping, usage = _base._anthropic_call(
            cfg, _base.OPTION_MAPPING_SYSTEM, payload, schema,
            int(cfg["anthropic"]["claim_max_tokens"]),
        )
        usages.append(usage)
        _base._validate_option_mapping(mapping, question, requirements, fact_texts)
        _validate_no_homogeneous_all_refuted(mapping)
        combined = _combined_usage(usages, errors)
        combined["option_mapping_operator_adapter"] = "mapping_consistency_guard_v1"
        combined["all_refuted_homogeneous_feedback_retry"] = True
        return mapping, combined
    except (ValueError, RuntimeError) as error:
        errors.append(str(error))
        raise HomogeneousAllRefutedError(errors, usages) from error


def _call_option_mapping_with_temporal_guard(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    verified_facts: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    operator = _pairwise_first_operator(question)
    if operator is None:
        mapping, usage = _ORIGINAL_OPTION_MAPPING(cfg, question, requirements, verified_facts)
        try:
            _validate_no_homogeneous_all_refuted(mapping)
            return mapping, usage
        except ValueError as error:
            return _retry_homogeneous_all_refuted_once(
                cfg, question, requirements, verified_facts, mapping, usage, error,
            )
    fact_texts = [fact["text"] for fact in verified_facts]
    base_payload = {
        "question": question,
        "requirements": requirements,
        "verified_facts": verified_facts,
        "temporal_operator": operator,
    }
    schema = _temporal_mapping_schema(requirements, fact_texts, operator)
    attempts = int(cfg.get("max_validation_retries", 2))
    usages: list[dict[str, Any]] = []
    errors: list[str] = []
    last_error: Exception | None = None
    for attempt in range(attempts):
        payload = copy.deepcopy(base_payload)
        if errors:
            payload["validator_feedback"] = (
                "VALIDATOR ERROR FROM THE PREVIOUS RESPONSE: " + errors[-1] + "\n"
                "Regenerate the complete option mapping. Do not merely state that an event occurred; "
                "apply the FIRST operator and make relation, status, used facts, and reason agree."
            )
        try:
            mapping, usage = _base._anthropic_call(
                cfg, _base.OPTION_MAPPING_SYSTEM + TEMPORAL_MAPPING_SYSTEM, payload, schema,
                int(cfg["anthropic"]["claim_max_tokens"]),
            )
            usages.append(usage)
            _validate_temporal_mapping(mapping, question, requirements, fact_texts, operator)
            return mapping, _combined_usage(usages, errors)
        except (ValueError, RuntimeError) as error:
            last_error = error
            errors.append(str(error))
    raise TemporalOperatorMappingError(errors, usages) from last_error


_ORIGINAL_OPTION_MAPPING = _base._call_option_mapping
_ORIGINAL_FINAL_SCHEMA = _enum._final_schema_with_evidence_enum
_ORIGINAL_GEMINI_CALL = _base._gemini_call


def run_live(root, config_path, video_uid=None):
    original = _base._call_option_mapping
    original_base_gemini = _base._gemini_call
    original_contract_gemini = _contract._gemini_call
    original_enum_gemini = _enum._gemini_call
    original_final_schema = _enum._final_schema_with_evidence_enum
    original_contract_final = _contract._final_with_conflict_feedback
    original_indexed_execution = _enum._execute_claims_batch_with_indexed_fines
    _base._call_option_mapping = _call_option_mapping_with_temporal_guard
    _base._gemini_call = _gemini_call_with_schema_preflight
    _contract._gemini_call = _gemini_call_with_schema_preflight
    _enum._gemini_call = _gemini_call_with_schema_preflight
    _enum._final_schema_with_evidence_enum = _final_schema_with_indexed_evidence_guard
    _contract._final_with_conflict_feedback = _final_with_indexed_evidence_guard
    _enum._execute_claims_batch_with_indexed_fines = _execute_claims_batch_with_bounded_text_retry
    try:
        return _contract.run_live(root, config_path, video_uid=video_uid)
    finally:
        _base._call_option_mapping = original
        _base._gemini_call = original_base_gemini
        _contract._gemini_call = original_contract_gemini
        _enum._gemini_call = original_enum_gemini
        _enum._final_schema_with_evidence_enum = original_final_schema
        _contract._final_with_conflict_feedback = original_contract_final
        _enum._execute_claims_batch_with_indexed_fines = original_indexed_execution
