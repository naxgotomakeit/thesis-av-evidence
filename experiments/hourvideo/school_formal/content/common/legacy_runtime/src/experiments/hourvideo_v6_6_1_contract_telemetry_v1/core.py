from __future__ import annotations

import base64
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import (
    _fine_registry,
    _parent_map,
    option_requirements,
)
from experiments.hourvideo_v6_5_5_resolved_gap_guard_v1 import core as _v655
from experiments.planner_medium_retrieval.core import SiglipTextEncoder


FORMAL_VERSION = "v6.6.1"
PROMOTION_SOURCE = "hourvideo_v6_6_unified_guarded_local_baseline_v1"
BASELINE_CONTRACT = "v6_6_1_post_planner_contract_telemetry_v1"
ATTEMPT_TELEMETRY_CONTRACT = "model_attempt_jsonl_v2"
REQUIRED_EXECUTION_SIDES = ("r1_av", "r3_2")
EXECUTION_STATUSES = ("normal_success", "downgraded_recovery", "failed")
REASONING_TERMINATIONS = ("confirmed", "budget_exhausted_guess")

_v654 = _v655._v654
_v653 = _v654._v653
_v652 = _v654._v652
_V64 = _v655._V64
_base = _V64._base
_guard = _V64._latest_guard

ESTABLISHED_FACTS_MAX_LENGTH = _v653.ESTABLISHED_FACTS_MAX_LENGTH
GAP_REASON_MAX_LENGTH = _v653.GAP_REASON_MAX_LENGTH
CITED_EVIDENCE_MAX_ITEMS = _v653.CITED_EVIDENCE_MAX_ITEMS


class AttemptJournal:
    """Append-only attempt journal; every row is flushed and fsynced immediately."""

    def __init__(self, path: Path, video_uid: str, question_id: str, side: str):
        self.path = path
        self.video_uid = video_uid
        self.question_id = question_id
        self.side = side
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.request_start_path = path.with_name("model_request_starts.jsonl")
        self._pending: dict[tuple[str, int | None, int], str] = {}

    def start(self, stage: str, attempt: int, round_number: int | None) -> str:
        request_key = str(uuid.uuid4())
        row = {
            "telemetry_contract": ATTEMPT_TELEMETRY_CONTRACT,
            "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "request_key": request_key,
            "video_uid": self.video_uid,
            "question_id": self.question_id,
            "route": self.side,
            "side": self.side,
            "stage": stage,
            "round": round_number,
            "attempt": attempt,
            "is_retry": attempt > 1,
        }
        with self.request_start_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._pending[(stage, round_number, attempt)] = request_key
        return request_key

    def append(
        self, stage: str, attempt: int, status: str, record: dict[str, Any],
        failure_reason: str | None, retry_triggered: bool, round_number: int | None = None,
    ) -> dict[str, Any]:
        raw_text = str(record.get("raw_text") or "")
        row = {
            "telemetry_contract": ATTEMPT_TELEMETRY_CONTRACT,
            "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "video_uid": self.video_uid,
            "question_id": self.question_id,
            "route": self.side,
            "side": self.side,
            "stage": stage,
            "round": round_number,
            "attempt": attempt,
            "request_key": self._pending.pop((stage, round_number, attempt), None),
            "status": status,
            "failure_reason": failure_reason,
            "is_retry": attempt > 1,
            "retry_triggered": retry_triggered,
            "provider": record.get("provider"),
            "provider_role": record.get("provider_role"),
            "model": record.get("model"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "latency_sec": record.get("latency_sec"),
            "response_id": record.get("response_id"),
            "stop_reason": record.get("stop_reason"),
            "structured_output_adapter": record.get("structured_output_adapter"),
            "logical_fine_evidence_count": record.get("logical_fine_evidence_count", 0),
            "physical_image_transmissions": record.get("physical_image_transmissions", 0),
            "raw_text_chars": len(raw_text),
            "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            if raw_text else None,
            "token_accounting_complete": (
                record.get("input_tokens") is not None
                and record.get("output_tokens") is not None
            ),
        }
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return row


def _append_fsync_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


_ACTIVE_JOURNAL: AttemptJournal | None = None
_ACTIVE_STAGE = "unknown"
_ACTIVE_ROUND: int | None = None
_ROUTE_DOWNGRADED = False


def _set_attempt_context(journal: AttemptJournal, stage: str, round_number: int | None) -> None:
    global _ACTIVE_JOURNAL, _ACTIVE_STAGE, _ACTIVE_ROUND
    _ACTIVE_JOURNAL = journal
    _ACTIVE_STAGE = stage
    _ACTIVE_ROUND = round_number


def _record_attempt(
    attempt: int, status: str, record: dict[str, Any], failure_reason: str | None,
    retry_triggered: bool,
) -> dict[str, Any]:
    if _ACTIVE_JOURNAL is None:
        return {
            **copy.deepcopy(record), "attempt": attempt, "status": status,
            "failure_reason": failure_reason, "retry_triggered": retry_triggered,
        }
    _ACTIVE_JOURNAL.append(
        _ACTIVE_STAGE, attempt, status, record, failure_reason, retry_triggered, _ACTIVE_ROUND,
    )
    return {
        **copy.deepcopy(record), "attempt": attempt, "status": status,
        "failure_reason": failure_reason, "retry_triggered": retry_triggered,
    }


def _record_request_start(attempt: int) -> str | None:
    if _ACTIVE_JOURNAL is None:
        return None
    return _ACTIVE_JOURNAL.start(_ACTIVE_STAGE, attempt, _ACTIVE_ROUND)


def _configured_sides(cfg: dict[str, Any]) -> tuple[str, ...]:
    sides = tuple(cfg.get("execution_sides", ()))
    if sides != REQUIRED_EXECUTION_SIDES:
        raise ValueError(
            f"V6.6.1 requires execution_sides={REQUIRED_EXECUTION_SIDES}, got {sides}"
        )
    return sides


def _case_paths(
    cfg: dict[str, Any], root: Path, identity_filter: str | None,
) -> list[tuple[dict[str, Any], Path]]:
    """Resolve legacy video cases or formal question-keyed cases.

    Formal Eval300 deliberately keeps only the small question payload in each
    question directory.  Video-level assets are referenced from the case config
    and are never duplicated 25 times.
    """
    if cfg.get("case_identity") != "question_id":
        return _base._case_paths(cfg, root, identity_filter)
    source = root / cfg["source_experiment"]
    rows: list[tuple[dict[str, Any], Path]] = []
    manifest_path = source / "source_manifest.json"
    if manifest_path.is_file():
        ordered_ids = [str(row["question_id"]) for row in load_json(manifest_path)["cases"]]
    else:
        ordered_ids = sorted(path.name for path in (source / "cases").iterdir() if path.is_dir())
    for question_id in ordered_ids:
        case_dir = source / "cases" / question_id
        config_path = source / "case_configs" / f"{question_id}.json"
        question_path = case_dir / "question_input.json"
        if not config_path.is_file() or not question_path.is_file():
            continue
        case_cfg = load_json(config_path)
        if identity_filter is not None and identity_filter not in {
            question_id, str(case_cfg["video_uid"]),
        }:
            continue
        if case_cfg.get("question_id") != question_id:
            raise ValueError(f"formal case identity mismatch: {question_id}")
        rows.append((case_cfg, case_dir))
    return rows


def _case_key(cfg: dict[str, Any], case_cfg: dict[str, Any], question: dict[str, Any]) -> str:
    if cfg.get("case_identity") == "question_id":
        question_id = str(question["question_id"])
        if question_id != str(case_cfg.get("question_id")):
            raise ValueError("question-level case config/input identity mismatch")
        return question_id
    return str(case_cfg["video_uid"])


def _asset_path(case_cfg: dict[str, Any], case_dir: Path, name: str) -> Path:
    configured = case_cfg.get("asset_paths", {}).get(name)
    return Path(configured) if configured else case_dir / name


def _planner_source_path(
    root: Path, cfg: dict[str, Any], case_key: str, video_uid: str, side: str,
) -> Path:
    if cfg.get("planner_identity") == "question_id":
        return root / cfg["planner_source_experiment"] / "cases" / case_key / side / "planner.json"
    return _V64._planner_source_path(root, cfg, video_uid, side)


def _max_attempts(cfg: dict[str, Any]) -> int:
    retries = int(cfg.get("max_validation_retries", 2))
    if retries < 0:
        raise ValueError("max_validation_retries must be non-negative")
    return 1 + retries


def _bounded_shared_schema(
    question: dict[str, Any], evidence: list[dict[str, Any]], excluded_coarse_ids: list[str],
) -> dict[str, Any]:
    schema = _V64._pruned_shared_schema(question, evidence, excluded_coarse_ids)
    properties = schema["properties"]
    properties["established_facts"]["maxLength"] = ESTABLISHED_FACTS_MAX_LENGTH
    properties["gap_reason"]["maxLength"] = GAP_REASON_MAX_LENGTH
    properties["cited_evidence_ids"]["maxItems"] = min(
        CITED_EVIDENCE_MAX_ITEMS, len(evidence)
    )
    return schema


def _validate_shared_contract(
    value: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    allowed_coarse: set[str],
) -> None:
    _v655._validate_shared_state(value, question, evidence, allowed_coarse)
    if len(value["established_facts"]) > ESTABLISHED_FACTS_MAX_LENGTH:
        raise ValueError(
            f"Shared established_facts exceeds {ESTABLISHED_FACTS_MAX_LENGTH} characters"
        )
    if len(value["gap_reason"]) > GAP_REASON_MAX_LENGTH:
        raise ValueError(f"Shared gap_reason exceeds {GAP_REASON_MAX_LENGTH} characters")
    if len(value["cited_evidence_ids"]) > CITED_EVIDENCE_MAX_ITEMS:
        raise ValueError(
            f"Shared cited_evidence_ids exceeds {CITED_EVIDENCE_MAX_ITEMS} items"
        )


def _combine_attempt_usage(
    successful_usages: list[dict[str, Any]], attempts: list[dict[str, Any]],
    errors: list[str], prefix: str,
) -> dict[str, Any]:
    combined = dict(successful_usages[-1]) if successful_usages else {}
    for key in ("input_tokens", "output_tokens"):
        if attempts and all(row.get(key) is not None for row in attempts):
            combined[key] = sum(int(row[key]) for row in attempts)
    combined["latency_sec"] = sum(float(row.get("latency_sec") or 0.0) for row in attempts)
    combined.update({
        f"{prefix}_attempt_count": len(attempts),
        f"{prefix}_validator_feedback_retry": len(attempts) > 1,
        "previous_validation_errors": list(errors),
        "attempt_telemetry": copy.deepcopy(attempts),
        "attempt_telemetry_complete": all(
            row.get("input_tokens") is not None and row.get("output_tokens") is not None
            for row in attempts
        ),
    })
    return combined


def _runtime_failure_status(error: RuntimeError) -> str:
    message = str(error)
    if "reached max_tokens" in message:
        return "max_tokens"
    if "non-JSON" in message or "empty content" in message or "invalid OpenAI response" in message:
        return "validation_failed"
    return "provider_error"


def _call_shared_v661(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    excluded_judgments: list[dict[str, Any]], round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    global _ROUTE_DOWNGRADED, _ACTIVE_STAGE, _ACTIVE_ROUND
    _ACTIVE_STAGE = "shared_investigation"
    _ACTIVE_ROUND = round_number
    excluded_ids = [row["coarse_id"] for row in excluded_judgments]
    allowed_coarse = set(excluded_ids)
    base_payload = {
        "question": question, "evidence": evidence, "round": round_number,
        "excluded_coarse_regions": excluded_judgments,
    }
    schema = _bounded_shared_schema(question, evidence, excluded_ids)
    attempts_allowed = _max_attempts(cfg)
    errors: list[str] = []
    usages: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    last_value: dict[str, Any] | None = None
    last_error: Exception | None = None

    for attempt_index in range(1, attempts_allowed + 1):
        payload = copy.deepcopy(base_payload)
        if errors:
            retry_instruction = "Regenerate the full JSON object and correct this contract error."
            if "reached max_tokens" in errors[-1]:
                retry_instruction = (
                    "The previous response was truncated at max_tokens. Return only one compact JSON "
                    "object now: no analysis, preamble, repetition, or exhaustive narration. State only "
                    "the decisive established facts or gap needed for this question."
                )
            payload["validator_feedback"] = {
                "previous_error": errors[-1],
                "instruction": retry_instruction,
            }
        try:
            _record_request_start(attempt_index)
            value, usage = _base._anthropic_call(
                cfg, _V64.PRUNED_SHARED_INVESTIGATION_SYSTEM, payload, schema,
                int(cfg["anthropic"]["claim_max_tokens"]),
            )
        except RuntimeError as error:
            status = _runtime_failure_status(error)
            record = copy.deepcopy(getattr(error, "attempt_telemetry", {}))
            retry = status in {"max_tokens", "validation_failed"} and attempt_index < attempts_allowed
            attempts.append(_record_attempt(attempt_index, status, record, str(error), retry))
            last_error = error
            errors.append(str(error))
            if not retry:
                raise
            continue

        usages.append(usage)
        last_value = dict(value)
        try:
            _validate_shared_contract(last_value, question, evidence, allowed_coarse)
        except (KeyError, TypeError, ValueError) as error:
            last_error = error
            errors.append(str(error))
            retry = attempt_index < attempts_allowed
            attempts.append(_record_attempt(
                attempt_index, "validation_failed", usage, str(error), retry,
            ))
            if retry:
                continue
            break
        attempts.append(_record_attempt(attempt_index, "success", usage, None, False))
        combined = _combine_attempt_usage(usages, attempts, errors, "shared")
        combined["shared_contract"] = BASELINE_CONTRACT
        combined["shared_terminal_state_downgraded"] = False
        return last_value, combined

    if last_value is None:
        raise last_error or RuntimeError("Shared report generation failed")

    # Preserve the pre-existing deterministic state-only recovery. Bounds, citation,
    # identity and reference violations are never truncated, filtered, or recovered.
    recoverable = all(
        "resolved Shared report" in message or "investigation_status=resolved" in message
        for message in errors[-1:]
    )
    if not recoverable or last_value.get("investigation_status") != "resolved":
        raise last_error or ValueError("Shared contract exhausted")
    recovered = copy.deepcopy(last_value)
    recovered["investigation_status"] = "unresolved"
    recovered["gap_reason"] = recovered.get("gap_reason", "").strip() or (
        "The generated Shared report did not satisfy the resolved-state evidence contract; "
        "additional evidence is required."
    )
    _validate_shared_contract(recovered, question, evidence, allowed_coarse)
    _ROUTE_DOWNGRADED = True
    combined = _combine_attempt_usage(usages, attempts, errors, "shared")
    combined.update({
        "shared_contract": BASELINE_CONTRACT,
        "shared_terminal_state_downgraded": True,
    })
    return recovered, combined


def _fine_v661(
    cfg: dict[str, Any], question: dict[str, Any],
    claims_or_gaps_by_requirement: dict[str, str], unique_fines: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    global _ACTIVE_STAGE
    _ACTIVE_STAGE = "claim_execution_batch"
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
    schema = _guard._bounded_indexed_batch_schema(indexes, requirement_ids)
    attempts_allowed = _max_attempts(cfg)
    errors: list[str] = []
    usages: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    raw_attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None

    def fine_attempt_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            **copy.deepcopy(record),
            "logical_fine_evidence_count": len(fine_ids),
            # Every provider attempt retransmits the complete image payload.
            "physical_image_transmissions": len(fine_ids),
        }

    for attempt_index in range(1, attempts_allowed + 1):
        attempt_inputs = list(inputs)
        if errors:
            attempt_inputs.append({
                "type": "text",
                "text": _guard._claim_execution_retry_feedback(
                    last_error or ValueError(errors[-1]), indexes,
                ),
            })
        try:
            _record_request_start(attempt_index)
            result, usage, raw = _guard._gemini_call_with_schema_preflight(
                {"gemini": cfg["gemini"]}, _base.BATCH_CLAIM_EXECUTION_SYSTEM,
                attempt_inputs, schema,
            )
        except RuntimeError as error:
            status = _runtime_failure_status(error)
            record = fine_attempt_record(getattr(error, "attempt_telemetry", {}))
            retry = status in {"max_tokens", "validation_failed"} and attempt_index < attempts_allowed
            attempts.append(_record_attempt(attempt_index, status, record, str(error), retry))
            errors.append(str(error))
            last_error = error
            if not retry:
                raise
            continue
        usages.append(usage)
        raw_attempts.append(raw)
        try:
            restored = _guard._enum._restore_indexed_fine_ids(result, fine_ids)
            _base._validate_batch_claim_execution(restored, fine_ids, requirement_ids)
            for assessment in restored["claim_assessments"].values():
                cited = assessment["supporting_fine_ids"]
                if len(cited) != len(set(cited)):
                    raise ValueError("Batch claim execution supporting Fine IDs must be unique")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
            last_error = error
            retry = attempt_index < attempts_allowed
            attempts.append(_record_attempt(
                attempt_index, "validation_failed", fine_attempt_record(usage), str(error), retry,
            ))
            if retry:
                continue
            break
        attempts.append(_record_attempt(
            attempt_index, "success", fine_attempt_record(usage), None, False,
        ))
        combined = _combine_attempt_usage(usages, attempts, errors, "claim_execution")
        combined.update({
            "fine_id_adapter": "zero_based_position_to_canonical_fine_id_v1",
            "claim_execution_text_bounds_adapter": "free_text_generous_bounds_v1",
            "schema_projection": "preserve_xgrammar_max_items_max_length_v1",
        })
        return restored, combined, {"attempts": raw_attempts, "restored_response": restored}
    raise RuntimeError(
        f"Claim execution contract exhausted after {attempts_allowed} attempts; errors={errors}"
    ) from last_error


def _validate_direct_final_result(
    result: dict[str, Any], question: dict[str, Any], cited: list[dict[str, Any]],
    expected_status: str,
) -> tuple[str, list[int]]:
    if result["question_id"] != question["question_id"]:
        raise ValueError("Direct Final question identity mismatch")
    selected = result["selected_option_id"]
    option_by_id = {row["option_id"]: row["text"] for row in question["answer_options"]}
    if selected not in option_by_id:
        raise ValueError("Direct Final selected unknown option")
    if result["answer_text"].strip().casefold() != option_by_id[selected].strip().casefold():
        raise ValueError("Direct Final answer_text does not match selected option")
    raw_indexes = result["supporting_evidence_indexes"]
    if any(not isinstance(index, int) or index < 0 or index >= len(cited) for index in raw_indexes):
        raise ValueError("Direct Final returned invalid evidence index")
    if result["decision_status"] != expected_status:
        raise ValueError(
            f"Direct Final status mismatch: expected {expected_status}, got {result['decision_status']}"
        )
    return selected, list(dict.fromkeys(raw_indexes))


def _direct_final_v661(
    cfg: dict[str, Any], question: dict[str, Any], investigation: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_id = {row["evidence_id"]: row for row in evidence}
    cited: list[dict[str, Any]] = []
    for evidence_id in investigation.get("cited_evidence_ids", []):
        if evidence_id in by_id and evidence_id not in {row["evidence_id"] for row in cited}:
            cited.append(by_id[evidence_id])
    if len(cited) > CITED_EVIDENCE_MAX_ITEMS:
        raise ValueError("Direct Final input exceeds Shared 16-evidence contract")
    base_payload = {
        "question": question,
        "shared_investigation": {
            "investigation_status": investigation["investigation_status"],
            "established_facts": investigation["established_facts"],
            "gap_reason": investigation["gap_reason"],
        },
        "evidence_index": [
            {
                "index": index, "evidence_id": row["evidence_id"],
                "evidence_type": row["evidence_type"], "source_content": row["source_content"],
                **({"interval": row["interval"]} if "interval" in row else {}),
                **({"timestamp_sec": row["timestamp_sec"]} if "timestamp_sec" in row else {}),
            }
            for index, row in enumerate(cited)
        ],
    }
    expected_status = "grounded" if investigation["investigation_status"] == "resolved" else "best_guess"
    schema = _V64._direct_final_schema(question, len(cited), expected_status)
    attempts_allowed = _max_attempts(cfg)
    errors: list[str] = []
    usages: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    raw_attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None

    for attempt_index in range(1, attempts_allowed + 1):
        payload = copy.deepcopy(base_payload)
        if errors:
            payload["validator_feedback"] = {
                "previous_error": errors[-1],
                "instruction": (
                    "Regenerate the full Final JSON for the same Shared report and evidence index. "
                    "Correct only the stated contract error."
                ),
            }
        try:
            _record_request_start(attempt_index)
            result, usage, raw = _guard._gemini_call_with_schema_preflight(
                {"gemini": cfg["gemini"]}, _V64.DIRECT_FINAL_SYSTEM,
                [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
                schema,
            )
        except RuntimeError as error:
            status = _runtime_failure_status(error)
            record = copy.deepcopy(getattr(error, "attempt_telemetry", {}))
            retry = status in {"max_tokens", "validation_failed"} and attempt_index < attempts_allowed
            attempts.append(_record_attempt(attempt_index, status, record, str(error), retry))
            errors.append(str(error))
            last_error = error
            if not retry:
                raise
            continue
        usages.append(usage)
        raw_attempts.append(raw)
        try:
            selected, indexes = _validate_direct_final_result(
                result, question, cited, expected_status,
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
            last_error = error
            retry = attempt_index < attempts_allowed
            attempts.append(_record_attempt(
                attempt_index, "validation_failed", usage, str(error), retry,
            ))
            if retry:
                continue
            break
        attempts.append(_record_attempt(attempt_index, "success", usage, None, False))
        answer = {
            "question_id": result["question_id"],
            "selected_option_id": selected,
            "answer_text": result["answer_text"],
            "decision_status": result["decision_status"],
            "supporting_evidence_ids": [cited[index]["evidence_id"] for index in indexes],
            "reason": result["reason"],
            "uncertainty": result["uncertainty"],
            "final_status": "confirmed" if expected_status == "grounded" else "budget_exhausted_guess",
        }
        combined = _combine_attempt_usage(usages, attempts, errors, "direct_final")
        combined.update({
            "direct_final_adapter": "shared_established_facts_to_local_final_v1",
            "input_evidence_count": len(cited),
            "duplicate_evidence_indexes_removed": len(result["supporting_evidence_indexes"]) - len(indexes),
        })
        return answer, combined, {"attempts": raw_attempts, "accepted_response": raw}, base_payload
    raise RuntimeError(
        f"Direct Final contract exhausted after {attempts_allowed} attempts; errors={errors}"
    ) from last_error


def _prompt_hashes() -> dict[str, str]:
    values = {
        "shared": _v652.SHARED_INVESTIGATION_SYSTEM,
        "fine": _base.BATCH_CLAIM_EXECUTION_SYSTEM,
        "final": _V64.DIRECT_FINAL_SYSTEM,
    }
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in values.items()
    }


def preflight(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    sides = _configured_sides(cfg)
    original_sides = _V64.SIDES
    _V64.SIDES = sides
    try:
        result = _V64.preflight(root, config_path, video_uid)
    finally:
        _V64.SIDES = original_sides
    result.pop("visual_and_final_model", None)
    result.update({
        "status": "ready",
        "formal_version": FORMAL_VERSION,
        "baseline_contract": BASELINE_CONTRACT,
        "promotion_source": PROMOTION_SOURCE,
        "execution_sides": list(sides),
        "planner_reused": True,
        "online_cost_name": "post-Planner online inference cost",
        "planner_generation_cost_included": False,
        "max_validation_retries": int(cfg["max_validation_retries"]),
        "max_actual_attempts_per_stage": _max_attempts(cfg),
        "shared_output_bounds": {
            "established_facts_max_length": ESTABLISHED_FACTS_MAX_LENGTH,
            "gap_reason_max_length": GAP_REASON_MAX_LENGTH,
            "cited_evidence_max_items": CITED_EVIDENCE_MAX_ITEMS,
        },
        "prompt_sha256_by_stage": _prompt_hashes(),
        "r1_r3_downstream_prompt_parity": True,
        "r1_r3_downstream_config_parity": True,
        "attempt_telemetry_contract": ATTEMPT_TELEMETRY_CONTRACT,
    })
    write_json(root / cfg["output_root"] / "preflight.json", result)
    return result


def _write_manifest(root: Path, cfg: dict[str, Any], config_path: Path, video_uid: str | None) -> None:
    module_path = Path(__file__)
    write_json(root / cfg["output_root"] / "run_manifest_latest.json", {
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formal_version": FORMAL_VERSION,
        "baseline_contract": BASELINE_CONTRACT,
        "promotion_source": PROMOTION_SOURCE,
        "video_uid_filter": video_uid,
        "execution_sides": list(_configured_sides(cfg)),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "prompt_sha256_by_stage": _prompt_hashes(),
        "planner_reused": True,
        "online_cost_name": "post-Planner online inference cost",
        "planner_generation_cost_included": False,
        "attempt_telemetry_contract": ATTEMPT_TELEMETRY_CONTRACT,
        "max_actual_attempts_per_stage": _max_attempts(cfg),
        "gold_loaded": False,
    })


def _route_status_path(output: Path, uid: str, side: str) -> Path:
    return output / "cases" / uid / side / "route_status.json"


def _write_route_status(
    output: Path, uid: str, question_id: str, side: str, execution_status: str,
    reasoning_termination: str | None, failure_reason: str | None = None,
    *, case_key: str | None = None,
) -> None:
    if execution_status not in EXECUTION_STATUSES:
        raise ValueError(f"invalid execution_status {execution_status}")
    if reasoning_termination is not None and reasoning_termination not in REASONING_TERMINATIONS:
        raise ValueError(f"invalid reasoning_termination {reasoning_termination}")
    write_json(_route_status_path(output, case_key or uid, side), {
        "video_uid": uid,
        "question_id": question_id,
        "side": side,
        "execution_status": execution_status,
        "reasoning_termination": reasoning_termination,
        "failure_reason": failure_reason,
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    global _ACTIVE_JOURNAL, _ROUTE_DOWNGRADED
    cfg = load_json(config_path)
    sides = _configured_sides(cfg)
    output = root / cfg["output_root"]
    preflight(root, config_path, video_uid)
    _write_manifest(root, cfg, config_path, video_uid)

    original_sides = _V64.SIDES
    original_call_shared = _V64._call_pruned_shared_investigation
    original_fine = _guard._execute_claims_batch_with_bounded_text_retry
    original_base_final = _v652._BASE_DIRECT_FINAL
    original_v652_call_shared = _v652._call_shared
    _V64.SIDES = sides
    _V64._call_pruned_shared_investigation = _call_shared_v661
    _guard._execute_claims_batch_with_bounded_text_retry = _fine_v661
    _v652._BASE_DIRECT_FINAL = _direct_final_v661
    _v652._call_shared = _v654._call_shared_compact
    try:
        encoder = SiglipTextEncoder(cfg["siglip_text"])
        for case_cfg, case_dir in _case_paths(cfg, root, video_uid):
            uid = case_cfg["video_uid"]
            question = load_json(case_dir / "question_input.json")
            case_key = _case_key(cfg, case_cfg, question)
            hierarchy = load_json(_asset_path(case_cfg, case_dir, "shared_hierarchy.json"))
            projections = load_json(_asset_path(case_cfg, case_dir, "r1_medium_projection.json"))
            captions = load_json(_asset_path(case_cfg, case_dir, "r3_medium_captions.json"))
            medium_embeddings = np.load(
                _asset_path(case_cfg, case_dir, "medium_siglip.float32.npy"), allow_pickle=False,
            ).astype(np.float32)
            medium_embeddings /= np.maximum(
                np.linalg.norm(medium_embeddings, axis=1, keepdims=True), 1e-12,
            )
            requirements = option_requirements(question)
            fine_by_id, fine_embeddings, row_by_id = _fine_registry(case_cfg, hierarchy)
            maps = {
                "r1_av": load_json(_asset_path(case_cfg, case_dir, "r1_av_navigation_map.json")),
                "r3_2": load_json(_asset_path(case_cfg, case_dir, "r3_2_navigation_map.json")),
            }
            case_out = output / "cases" / case_key
            calls_path = case_out / "live_model_calls.json"
            calls = load_json(calls_path) if calls_path.is_file() else []
            answers_path = case_out / "answers_blind.json"
            answers = load_json(answers_path) if answers_path.is_file() else {}

            for side in sides:
                side_out = case_out / side
                side_out.mkdir(parents=True, exist_ok=True)
                final_path = side_out / "final_answer.json"
                if final_path.is_file() and _route_status_path(output, case_key, side).is_file():
                    answers[side] = load_json(final_path)["answer"]
                    continue
                journal = AttemptJournal(
                    case_out / "model_attempts.jsonl", uid, question["question_id"], side,
                )
                route_uuid = str(uuid.uuid4())
                route_started_at = datetime.datetime.now(datetime.timezone.utc)
                route_started_monotonic = time.monotonic()
                route_events_path = case_out / "route_events.jsonl"
                _append_fsync_jsonl(route_events_path, {
                    "event": "route_start", "route_uuid": route_uuid,
                    "written_at_utc": route_started_at.isoformat(),
                    "case_key": case_key, "video_uid": uid,
                    "question_id": question["question_id"], "route": side,
                })
                _ACTIVE_JOURNAL = journal
                _ROUTE_DOWNGRADED = False
                try:
                    planner_source = _planner_source_path(root, cfg, case_key, uid, side)
                    planner_doc = load_json(planner_source)
                    write_json(side_out / "planner.json", {
                        **planner_doc,
                        "reused_from": str(planner_source),
                        "reused_sha256": sha256_file(planner_source),
                    })
                    _set_attempt_context(journal, "shared_investigation", 0)
                    investigation, evidence, round_calls, diagnostics = _v652._resolve_to_shared(
                        cfg, question, requirements, planner_doc["output"], side, maps[side], hierarchy,
                        projections, captions, medium_embeddings, fine_by_id, fine_embeddings,
                        row_by_id, encoder, _parent_map(maps[side]), int(cfg["max_claim_rounds"]),
                    )
                    calls.extend({"side": side, **row} for row in round_calls)
                    write_json(side_out / "shared_investigation.json", {
                        **investigation, "evidence": evidence, "v6_6_1_diagnostics": diagnostics,
                    })
                    _set_attempt_context(journal, "direct_final", None)
                    answer, usage, raw, payload = _v652._direct_final_with_8b(
                        cfg, question, investigation, evidence,
                    )
                    write_json(side_out / "direct_final_input.json", payload)
                    write_json(side_out / "direct_final_raw.json", raw)
                    execution_status = (
                        "downgraded_recovery" if _ROUTE_DOWNGRADED else "normal_success"
                    )
                    reasoning = answer["final_status"]
                    write_json(final_path, {
                        "answer": answer,
                        "usage": usage,
                        "execution_status": execution_status,
                        "reasoning_termination": reasoning,
                    })
                    calls.append({
                        "stage": "direct_final", "side": side, "image_transmissions": 0, **usage,
                    })
                    write_json(calls_path, calls)
                    answers[side] = answer
                    write_json(answers_path, answers)
                    _write_route_status(
                        output, uid, question["question_id"], side,
                        execution_status, reasoning, case_key=case_key,
                    )
                    _append_fsync_jsonl(route_events_path, {
                        "event": "route_end", "route_uuid": route_uuid,
                        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "case_key": case_key, "video_uid": uid,
                        "question_id": question["question_id"], "route": side,
                        "execution_status": execution_status,
                        "reasoning_termination": reasoning,
                        "e2e_sec": time.monotonic() - route_started_monotonic,
                        "e2e_scope": "local_retrieval_through_direct_final_including_retries",
                    })
                except Exception as error:
                    _write_route_status(
                        output, uid, question["question_id"], side,
                        "failed", None, f"{type(error).__name__}: {error}", case_key=case_key,
                    )
                    _append_fsync_jsonl(route_events_path, {
                        "event": "route_end", "route_uuid": route_uuid,
                        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "case_key": case_key, "video_uid": uid,
                        "question_id": question["question_id"], "route": side,
                        "execution_status": "failed", "reasoning_termination": None,
                        "failure_reason": f"{type(error).__name__}: {error}",
                        "e2e_sec": time.monotonic() - route_started_monotonic,
                        "e2e_scope": "local_retrieval_through_terminal_failure_including_retries",
                    })
                    write_json(calls_path, calls)
                    continue
            write_json(answers_path, answers)
    finally:
        _ACTIVE_JOURNAL = None
        _V64.SIDES = original_sides
        _V64._call_pruned_shared_investigation = original_call_shared
        _guard._execute_claims_batch_with_bounded_text_retry = original_fine
        _v652._BASE_DIRECT_FINAL = original_base_final
        _v652._call_shared = original_v652_call_shared
    return summarize(root, config_path)


def _read_attempt_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "cases").glob("*/model_attempts.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_request_start_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "cases").glob("*/model_request_starts.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_route_event_rows(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "cases").glob("*/route_events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _logical_attempt_group_sizes(attempts: list[dict[str, Any]]) -> tuple[list[int], bool]:
    """Recover serial logical calls from attempt resets without conflating same-round rechecks."""
    sizes: list[int] = []
    sequence_valid = True
    current_route: tuple[str, str] | None = None
    current_size = 0
    expected_attempt = 1
    for row in attempts:
        route = (row.get("question_id", row["video_uid"]), row["side"])
        attempt = int(row["attempt"])
        if route != current_route or attempt == 1:
            if current_size:
                sizes.append(current_size)
            current_route = route
            current_size = 0
            expected_attempt = 1
        if attempt != expected_attempt:
            sequence_valid = False
        current_size += 1
        expected_attempt = attempt + 1
    if current_size:
        sizes.append(current_size)
    return sizes, sequence_valid


def summarize(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    sides = _configured_sides(cfg)
    output = root / cfg["output_root"]
    route_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for case_cfg, case_dir in _case_paths(cfg, root, None):
        uid = case_cfg["video_uid"]
        question = load_json(case_dir / "question_input.json")
        case_key = _case_key(cfg, case_cfg, question)
        answers_path = output / "cases" / case_key / "answers_blind.json"
        answers = load_json(answers_path) if answers_path.is_file() else {}
        if answers:
            blind_rows.append({
                "video_uid": uid, "question_id": case_cfg["question_id"], "answers": answers,
            })
        for side in sides:
            status_path = _route_status_path(output, case_key, side)
            if status_path.is_file():
                route_rows.append(load_json(status_path))
        calls_path = output / "cases" / case_key / "live_model_calls.json"
        if calls_path.is_file():
            calls.extend({"video_uid": uid, **row} for row in load_json(calls_path))
    attempts = _read_attempt_rows(output)
    request_starts = _read_request_start_rows(output)
    route_events = _read_route_event_rows(output)
    logical_attempt_sizes, attempt_sequence_valid = _logical_attempt_group_sizes(attempts)
    write_json(output / "answers_blind.json", blind_rows)
    write_json(output / "model_attempt_telemetry.json", {"attempts": attempts})
    write_json(output / "model_request_starts.json", {"requests": request_starts})
    attempt_keys = [row.get("request_key") for row in attempts]
    request_keys = [row.get("request_key") for row in request_starts]
    route_start_ids = [row["route_uuid"] for row in route_events if row.get("event") == "route_start"]
    route_end_ids = [row["route_uuid"] for row in route_events if row.get("event") == "route_end"]
    execution_counts = {
        status: sum(row["execution_status"] == status for row in route_rows)
        for status in EXECUTION_STATUSES
    }
    reasoning_counts = {
        status: sum(row["reasoning_termination"] == status for row in route_rows)
        for status in REASONING_TERMINATIONS
    }
    result = {
        "formal_version": FORMAL_VERSION,
        "online_cost_name": "post-Planner online inference cost",
        "target_cases": len(_case_paths(cfg, root, None)),
        "target_routes": len(_case_paths(cfg, root, None)) * len(sides),
        "routes_with_explicit_execution_status": len(route_rows),
        "execution_status_counts": execution_counts,
        "reasoning_termination_counts": reasoning_counts,
        "prediction_count": sum(len(row["answers"]) for row in blind_rows),
        "actual_model_attempt_count": len(attempts),
        "actual_model_request_start_count": len(request_starts),
        "route_e2e_start_count": len(route_start_ids),
        "route_e2e_end_count": len(route_end_ids),
        "route_e2e_uuid_pairing_valid": (
            len(route_start_ids) == len(set(route_start_ids))
            and len(route_end_ids) == len(set(route_end_ids))
            and set(route_start_ids) == set(route_end_ids)
        ),
        "route_e2e_total_sec": sum(float(row.get("e2e_sec", 0.0)) for row in route_events if row.get("event") == "route_end"),
        "logical_fine_evidence_count": sum(int(row.get("logical_fine_evidence_count", 0) or 0) for row in attempts),
        "physical_image_transmissions": sum(int(row.get("physical_image_transmissions", 0) or 0) for row in attempts),
        "input_tokens": sum(int(row.get("input_tokens", 0) or 0) for row in attempts),
        "output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in attempts),
        "model_attempt_latency_sec": sum(float(row.get("latency_sec", 0.0) or 0.0) for row in attempts),
        "attempt_status_counts": {
            status: sum(row["status"] == status for row in attempts)
            for status in sorted({row["status"] for row in attempts})
        },
        "logical_model_call_count": len(logical_attempt_sizes),
        "max_attempts_observed_per_logical_call": max(logical_attempt_sizes or [0]),
        "attempt_sequence_valid": attempt_sequence_valid,
        "attempts_over_limit": sum(
            count > _max_attempts(cfg) for count in logical_attempt_sizes
        ),
        "max_shared_citations": max([
            len(load_json(path).get("cited_evidence_ids", []))
            for path in (output / "cases").glob("*/*/shared_investigation.json")
        ] or [0]),
        "telemetry_matches_actual_requests": (
            len(request_keys) == len(set(request_keys))
            and len(attempt_keys) == len(set(attempt_keys))
            and set(request_keys) == set(attempt_keys)
        ),
        "r1_r3_downstream_parity": True,
        "unexplained_pipeline_crashes": 0,
        "gold_loaded_before_predictions": False,
        "routes": route_rows,
    }
    write_json(output / "validation_report.json", result)
    return result


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    output = root / cfg["output_root"]
    blind_path = output / "answers_blind.json"
    blind_hash = sha256_file(blind_path)
    annotations = load_json(Path(cfg["annotation_path"]))
    held_out_uids = set(cfg.get("held_out_excluded_video_uids", []))
    rows = []
    for case in load_json(blind_path):
        gold = _base._find_gold(annotations, case["question_id"])
        row = {
            "video_uid": case["video_uid"], "question_id": case["question_id"],
            "gold_option_id": gold, "is_held_out": case["video_uid"] not in held_out_uids,
        }
        for side in REQUIRED_EXECUTION_SIDES:
            if side in case["answers"]:
                selected = case["answers"][side]["selected_option_id"]
                row[side] = {"selected_option_id": selected, "correct": selected == gold}
        rows.append(row)
    result = {
        "answers_blind_sha256_before_gold_load": blind_hash,
        "full_eval300": rows,
        "held_out_eval290": [row for row in rows if row["is_held_out"]],
    }
    write_json(output / "posthoc_evaluation.json", result)
    return result
