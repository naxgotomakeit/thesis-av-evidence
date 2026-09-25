from __future__ import annotations

import copy
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_v6_6_1_contract_telemetry_v1 import core as _v661


FORMAL_VERSION = "v6.6.2"
PROMOTION_SOURCE = "hourvideo_v6_6_1_contract_telemetry_v1"
BASELINE_CONTRACT = "v6_6_2_shared_coarse_contract_v1"
ATTEMPT_TELEMETRY_CONTRACT = "model_attempt_jsonl_v3_full_shared_failure_audit"
COARSE_SELECTION_CONTRACT = "selectable_iff_unobserved_fine_exists_v1"
CITATION_CANONICALIZATION = "stable_first_occurrence_v1"

_v652 = _v661._v652
_v654 = _v661._v654
_V64 = _v661._V64
_base = _v661._base
_guard = _v661._guard
_BASE_PREFLIGHT = _v661.preflight

ESTABLISHED_FACTS_MAX_LENGTH = _v661.ESTABLISHED_FACTS_MAX_LENGTH
GAP_REASON_MAX_LENGTH = _v661.GAP_REASON_MAX_LENGTH
CITED_EVIDENCE_MAX_ITEMS = _v661.CITED_EVIDENCE_MAX_ITEMS


class AttemptJournal(_v661.AttemptJournal):
    """V6.6.2 journal retaining the complete Shared contract failure audit."""

    def append_detailed_shared(
        self,
        *,
        attempt: int,
        status: str,
        record: dict[str, Any],
        failure_reason: str | None,
        retry_triggered: bool,
        round_number: int,
        control_payload: dict[str, Any],
        full_payload_sha256: str,
        raw_model_response: str,
        parsed_response: dict[str, Any] | None,
        parsed_requested_coarse_ids: list[Any] | None,
        raw_citation_ids: list[Any] | None,
        deduplicated_citation_ids: list[Any] | None,
        duplicate_citation_ids: list[Any] | None,
        validator_error: str | None,
        retry_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        row = super().append(
            "shared_investigation", attempt, status, record, failure_reason,
            retry_triggered, round_number,
        )
        # super() has already flushed a compact row. Replace that last row atomically is
        # deliberately avoided: append-only crash safety wins. The complete audit is a
        # second linked event with the same request_key and an explicit event type.
        audit = {
            **row,
            "telemetry_contract": ATTEMPT_TELEMETRY_CONTRACT,
            "event_type": "shared_attempt_full_audit",
            "control_payload": copy.deepcopy(control_payload),
            "full_payload_sha256": full_payload_sha256,
            "raw_model_response": raw_model_response,
            "parsed_model_response": copy.deepcopy(parsed_response),
            "parsed_requested_coarse_ids": copy.deepcopy(parsed_requested_coarse_ids),
            "raw_citation_ids": copy.deepcopy(raw_citation_ids),
            "deduplicated_citation_ids": copy.deepcopy(deduplicated_citation_ids),
            "duplicate_citation_ids": copy.deepcopy(duplicate_citation_ids),
            "validator_error": validator_error,
            "retry_feedback": copy.deepcopy(retry_feedback),
        }
        encoded = json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.path.with_name("shared_attempt_audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return audit


def _stable_deduplicate(values: list[Any]) -> tuple[list[Any], list[Any]]:
    unique: list[Any] = []
    duplicates: list[Any] = []
    for value in values:
        if value in unique:
            if value not in duplicates:
                duplicates.append(value)
        else:
            unique.append(value)
    return unique, duplicates


def _coarse_to_fine_ids(
    map_doc: dict[str, Any], hierarchy: dict[str, Any],
) -> dict[str, list[str]]:
    medium_to_fines = {
        row["medium_id"]: list(row["source_fine_ids"])
        for row in hierarchy["medium_nodes"]
    }
    result: dict[str, list[str]] = {}
    for coarse in map_doc["coarse_regions"]:
        fine_ids: list[str] = []
        for medium_id in coarse["source_medium_ids"]:
            fine_ids.extend(medium_to_fines.get(medium_id, []))
        result[coarse["coarse_id"]] = list(dict.fromkeys(fine_ids))
    return result


def _coarse_controls(
    map_doc: dict[str, Any], hierarchy: dict[str, Any], observed_fine_ids: set[str],
    previously_selected_coarse_ids: set[str],
) -> dict[str, Any]:
    coarse_fines = _coarse_to_fine_ids(map_doc, hierarchy)
    ordered = [row["coarse_id"] for row in map_doc["coarse_regions"]]
    counts = {
        coarse_id: sum(fine_id not in observed_fine_ids for fine_id in coarse_fines[coarse_id])
        for coarse_id in ordered
    }
    selectable = [coarse_id for coarse_id in ordered if counts[coarse_id] > 0]
    exhausted = [coarse_id for coarse_id in ordered if counts[coarse_id] == 0]
    return {
        "selectable_coarse_ids": selectable,
        "exhausted_coarse_ids": exhausted,
        "previously_selected_coarse_ids": [
            coarse_id for coarse_id in ordered if coarse_id in previously_selected_coarse_ids
        ],
        "unobserved_fine_count_by_selectable_coarse": {
            coarse_id: counts[coarse_id] for coarse_id in selectable
        },
    }


def _explicit_shared_controls(controls: dict[str, Any]) -> dict[str, Any]:
    selectable = list(controls["selectable_coarse_ids"])
    return {
        **copy.deepcopy(controls),
        "max_cited_evidence_ids": CITED_EVIDENCE_MAX_ITEMS,
        "empty_selectable_rule": (
            "If selectable_coarse_ids is empty, requested_coarse_ids MUST be []."
        ),
        "coarse_selection_rule": (
            "requested_coarse_ids may contain only IDs from selectable_coarse_ids. "
            "A previously selected Coarse remains selectable while it has unobserved Fine nodes."
        ),
        "shared_text_limits": {
            "established_facts_max_chars": ESTABLISHED_FACTS_MAX_LENGTH,
            "gap_reason_max_chars": GAP_REASON_MAX_LENGTH,
        },
        "required_requested_coarse_ids_when_empty": [] if not selectable else None,
    }


def _bounded_shared_schema(
    question: dict[str, Any], evidence: list[dict[str, Any]], selectable: list[str],
) -> dict[str, Any]:
    schema = _V64._pruned_shared_schema(question, evidence, selectable)
    props = schema["properties"]
    props["established_facts"]["maxLength"] = ESTABLISHED_FACTS_MAX_LENGTH
    props["gap_reason"]["maxLength"] = GAP_REASON_MAX_LENGTH
    props["cited_evidence_ids"]["maxItems"] = min(CITED_EVIDENCE_MAX_ITEMS, len(evidence))
    return schema


def _citation_details(value: dict[str, Any] | None) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
    if not isinstance(value, dict) or not isinstance(value.get("cited_evidence_ids"), list):
        return None, None, None
    raw = list(value["cited_evidence_ids"])
    deduplicated, duplicates = _stable_deduplicate(raw)
    return raw, deduplicated, duplicates


def _shared_contract_error(
    value: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    controls: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = copy.deepcopy(value)
    raw_citations, dedup_citations, duplicates = _citation_details(canonical)
    requested = canonical.get("requested_coarse_ids")
    if not isinstance(requested, list):
        raise ValueError("Shared requested_coarse_ids must be an array")
    if raw_citations is None or dedup_citations is None or duplicates is None:
        raise ValueError("Shared cited_evidence_ids must be an array")
    canonical["cited_evidence_ids"] = dedup_citations

    all_ids = set(controls["selectable_coarse_ids"]) | set(controls["exhausted_coarse_ids"])
    nonexistent = [item for item in requested if item not in all_ids]
    exhausted = [item for item in requested if item in set(controls["exhausted_coarse_ids"])]
    invalid_selectable = [
        item for item in requested if item not in set(controls["selectable_coarse_ids"])
    ]
    details = {
        "actual_requested_coarse_ids": copy.deepcopy(requested),
        "nonexistent_coarse_ids": nonexistent,
        "exhausted_coarse_ids_returned": exhausted,
        "invalid_requested_coarse_ids": invalid_selectable,
        "selectable_coarse_ids": list(controls["selectable_coarse_ids"]),
        "raw_citation_count": len(raw_citations),
        "deduplicated_citation_count": len(dedup_citations),
        "duplicate_citation_ids": duplicates,
        "max_cited_evidence_ids": CITED_EVIDENCE_MAX_ITEMS,
    }
    if invalid_selectable:
        raise ValueError(
            "Shared requested invalid Coarse IDs: "
            f"returned={requested}; nonexistent={nonexistent}; exhausted={exhausted}; "
            f"selectable={controls['selectable_coarse_ids']}"
        )
    if len(dedup_citations) > CITED_EVIDENCE_MAX_ITEMS:
        raise ValueError(
            "Shared cited too many unique evidence IDs: "
            f"raw_count={len(raw_citations)}; deduplicated_count={len(dedup_citations)}; "
            f"duplicates={duplicates}; max={CITED_EVIDENCE_MAX_ITEMS}"
        )
    _v661._validate_shared_contract(
        canonical, question, evidence, set(controls["selectable_coarse_ids"]),
    )
    return canonical, details


def _retry_feedback(
    error: Exception, parsed: dict[str, Any] | None, controls: dict[str, Any],
) -> dict[str, Any]:
    raw, deduplicated, duplicates = _citation_details(parsed)
    requested = parsed.get("requested_coarse_ids") if isinstance(parsed, dict) else None
    requested_list = requested if isinstance(requested, list) else []
    all_ids = set(controls["selectable_coarse_ids"]) | set(controls["exhausted_coarse_ids"])
    nonexistent = [item for item in requested_list if item not in all_ids]
    exhausted = [
        item for item in requested_list if item in set(controls["exhausted_coarse_ids"])
    ]
    selectable = list(controls["selectable_coarse_ids"])
    return {
        "previous_validator_error": str(error),
        "instruction": "Regenerate the full JSON object for the same stage and same evidence input.",
        "coarse_contract": {
            "actual_requested_coarse_ids": copy.deepcopy(requested),
            "nonexistent_coarse_ids": nonexistent,
            "exhausted_coarse_ids": exhausted,
            "selectable_coarse_ids": selectable,
            "instruction": (
                "Return requested_coarse_ids=[] because no Coarse is selectable."
                if not selectable else
                "Return only IDs from the complete selectable_coarse_ids list; previously selected IDs are allowed if listed."
            ),
        },
        "citation_contract": {
            "raw_count": len(raw) if raw is not None else None,
            "deduplicated_count": len(deduplicated) if deduplicated is not None else None,
            "duplicate_ids": duplicates,
            "maximum_unique_citations": CITED_EVIDENCE_MAX_ITEMS,
            "instruction": "Select and return at most 16 unique cited_evidence_ids.",
        },
        "shared_text_limits": {
            "established_facts_max_chars": ESTABLISHED_FACTS_MAX_LENGTH,
            "gap_reason_max_chars": GAP_REASON_MAX_LENGTH,
        },
    }


def _record_shared_attempt(
    *, attempt: int, status: str, record: dict[str, Any], failure_reason: str | None,
    retry_triggered: bool, round_number: int, control_payload: dict[str, Any],
    full_payload: dict[str, Any], parsed: dict[str, Any] | None,
    validator_error: str | None, feedback: dict[str, Any] | None,
) -> dict[str, Any]:
    raw, deduplicated, duplicates = _citation_details(parsed)
    raw_text = str(record.get("raw_text") or "")
    summarized = {
        **copy.deepcopy(record), "attempt": attempt, "status": status,
        "failure_reason": failure_reason, "retry_triggered": retry_triggered,
    }
    journal = _v661._ACTIVE_JOURNAL
    if isinstance(journal, AttemptJournal):
        journal.append_detailed_shared(
            attempt=attempt, status=status, record=record,
            failure_reason=failure_reason, retry_triggered=retry_triggered,
            round_number=round_number, control_payload=control_payload,
            full_payload_sha256=hashlib.sha256(json.dumps(
                full_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
            raw_model_response=raw_text, parsed_response=parsed,
            parsed_requested_coarse_ids=(
                copy.deepcopy(parsed.get("requested_coarse_ids")) if isinstance(parsed, dict) else None
            ),
            raw_citation_ids=raw, deduplicated_citation_ids=deduplicated,
            duplicate_citation_ids=duplicates, validator_error=validator_error,
            retry_feedback=feedback,
        )
    else:
        _v661._record_attempt(attempt, status, record, failure_reason, retry_triggered)
    return summarized


def _safe_unresolved(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "investigation_status": "unresolved",
        "established_facts": "",
        "gap_reason": (
            "Shared output did not satisfy the local contract after all validation attempts; "
            "no invalid Shared output was admitted."
        ),
        "requested_coarse_ids": [],
        "cited_evidence_ids": [],
    }


def _call_shared_v662(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    controls: dict[str, Any], round_number: int,
    previous_valid_shared: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projected, short_to_canonical, projection = _v654._project_shared_evidence(evidence)
    explicit_controls = _explicit_shared_controls(controls)
    base_payload = {
        "question": question,
        "evidence": projected,
        "round": round_number,
        **copy.deepcopy(explicit_controls),
    }
    schema = _bounded_shared_schema(
        question, projected, list(controls["selectable_coarse_ids"]),
    )
    attempts_allowed = _v661._max_attempts(cfg)
    usages: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    feedback: dict[str, Any] | None = None

    _v661._ACTIVE_STAGE = "shared_investigation"
    _v661._ACTIVE_ROUND = round_number
    for attempt_index in range(1, attempts_allowed + 1):
        payload = copy.deepcopy(base_payload)
        if feedback is not None:
            payload["validator_feedback"] = copy.deepcopy(feedback)
        control_payload = {
            key: copy.deepcopy(value) for key, value in payload.items()
            if key not in {"question", "evidence"}
        }
        try:
            _v661._record_request_start(attempt_index)
            value, usage = _base._anthropic_call(
                cfg, _V64.PRUNED_SHARED_INVESTIGATION_SYSTEM, payload, schema,
                int(cfg["anthropic"]["claim_max_tokens"]),
            )
        except RuntimeError as error:
            status = _v661._runtime_failure_status(error)
            record = copy.deepcopy(getattr(error, "attempt_telemetry", {}))
            if status == "provider_error":
                _record_shared_attempt(
                    attempt=attempt_index, status=status, record=record,
                    failure_reason=str(error), retry_triggered=False,
                    round_number=round_number, control_payload=control_payload,
                    full_payload=payload, parsed=None, validator_error=str(error),
                    feedback=feedback,
                )
                raise
            retry = attempt_index < attempts_allowed
            next_feedback = _retry_feedback(error, None, controls)
            attempt_rows.append(_record_shared_attempt(
                attempt=attempt_index, status=status, record=record,
                failure_reason=str(error), retry_triggered=retry,
                round_number=round_number, control_payload=control_payload,
                full_payload=payload, parsed=None, validator_error=str(error),
                feedback=next_feedback,
            ))
            errors.append(str(error))
            feedback = next_feedback
            continue

        usages.append(usage)
        parsed = copy.deepcopy(value)
        try:
            canonical, details = _shared_contract_error(
                parsed, question, projected, controls,
            )
        except (KeyError, TypeError, ValueError) as error:
            retry = attempt_index < attempts_allowed
            next_feedback = _retry_feedback(error, parsed, controls)
            attempt_rows.append(_record_shared_attempt(
                attempt=attempt_index, status="validation_failed", record=usage,
                failure_reason=str(error), retry_triggered=retry,
                round_number=round_number, control_payload=control_payload,
                full_payload=payload, parsed=parsed, validator_error=str(error),
                feedback=next_feedback,
            ))
            errors.append(str(error))
            feedback = next_feedback
            continue
        attempt_rows.append(_record_shared_attempt(
            attempt=attempt_index, status="success", record=usage,
            failure_reason=None, retry_triggered=False,
            round_number=round_number, control_payload=control_payload,
            full_payload=payload, parsed=parsed, validator_error=None,
            feedback=feedback,
        ))
        canonical["cited_evidence_ids"] = [
            short_to_canonical.get(evidence_id, evidence_id)
            for evidence_id in canonical["cited_evidence_ids"]
        ]
        combined = _v661._combine_attempt_usage(usages, attempt_rows, errors, "shared")
        combined.update({
            "shared_contract": BASELINE_CONTRACT,
            "coarse_selection_contract": COARSE_SELECTION_CONTRACT,
            "citation_canonicalization": CITATION_CANONICALIZATION,
            "citation_canonicalization_details": details,
            "shared_terminal_state_downgraded": False,
            "shared_observation_projection": _v654.OBSERVATION_PROJECTION,
            "shared_observation_projection_metrics": projection,
            "shared_observation_short_id_map": short_to_canonical,
            "control_payload": explicit_controls,
        })
        return canonical, combined

    # Contract exhaustion is a recoverable Shared-stage failure. Invalid model output
    # is never admitted. Preserve the previous legal semantic state if available, but
    # clear its stale request so it cannot mutate current investigation state.
    recovered = copy.deepcopy(previous_valid_shared or _safe_unresolved(question))
    recovered["requested_coarse_ids"] = []
    if recovered["investigation_status"] == "resolved":
        recovered["investigation_status"] = "unresolved"
        recovered["gap_reason"] = (
            "The last valid Shared state was preserved after a later Shared contract failure; "
            "the investigation remains unresolved for safe recovery."
        )
    _v661._ROUTE_DOWNGRADED = True
    combined = _v661._combine_attempt_usage(usages, attempt_rows, errors, "shared")
    combined.update({
        "shared_contract": BASELINE_CONTRACT,
        "coarse_selection_contract": COARSE_SELECTION_CONTRACT,
        "citation_canonicalization": CITATION_CANONICALIZATION,
        "shared_terminal_state_downgraded": True,
        "logical_call_status": "downgraded_recovery",
        "invalid_output_admitted": False,
        "recovery_source": "previous_valid_shared" if previous_valid_shared else "local_safe_unresolved",
        "control_payload": explicit_controls,
    })
    return recovered, combined


def _filtered_retrieval_inputs(
    hierarchy: dict[str, Any], projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: np.ndarray, observed_fine_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    kept_mediums: list[dict[str, Any]] = []
    kept_indexes: list[int] = []
    for index, medium in enumerate(hierarchy["medium_nodes"]):
        available = [fine_id for fine_id in medium["source_fine_ids"] if fine_id not in observed_fine_ids]
        if available:
            kept_mediums.append({**copy.deepcopy(medium), "source_fine_ids": available})
            kept_indexes.append(index)
    filtered = {**copy.deepcopy(hierarchy), "medium_nodes": kept_mediums}
    return (
        filtered,
        [copy.deepcopy(projections[index]) for index in kept_indexes],
        [copy.deepcopy(captions[index]) for index in kept_indexes],
        medium_embeddings[kept_indexes],
    )


def _retrieve_unobserved_fines(
    cfg: dict[str, Any], side: str, question: dict[str, Any], plan: dict[str, Any],
    shared_id: str, selected_coarse_ids: set[str], map_doc: dict[str, Any],
    hierarchy: dict[str, Any], projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: np.ndarray, fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: np.ndarray, row_by_id: dict[str, int], encoder: Any,
    observed_fine_ids: set[str],
) -> list[dict[str, Any]]:
    filtered_hierarchy, filtered_projections, filtered_captions, filtered_embeddings = (
        _filtered_retrieval_inputs(
            hierarchy, projections, captions, medium_embeddings, observed_fine_ids,
        )
    )
    parent = _base._parent_map(map_doc)
    available_coarse = {
        parent[row["medium_id"]] for row in filtered_hierarchy["medium_nodes"]
    }
    scope = selected_coarse_ids & available_coarse
    if not scope:
        return []
    retrieval = _base._coarse_scoped_retrieval(
        cfg, side, question,
        {"requirement_plans": [_base._shared_search_plan(plan, shared_id)]},
        map_doc, filtered_hierarchy, filtered_projections, filtered_captions,
        filtered_embeddings, fine_by_id, fine_embeddings, row_by_id, encoder,
        {shared_id: scope},
    )
    rows = retrieval["requirements"][shared_id]["selected_fine_evidence"]
    if any(row["fine_id"] in observed_fine_ids for row in rows):
        raise AssertionError("V6.6.2 retrieval returned an already observed Fine ID")
    return rows


def _resolve_to_shared_v662(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    plan: dict[str, Any], side: str, map_doc: dict[str, Any], hierarchy: dict[str, Any],
    projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: np.ndarray, fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: np.ndarray, row_by_id: dict[str, int], encoder: Any,
    parent: dict[str, str], max_rounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    del requirements, parent
    calls: list[dict[str, Any]] = []
    shared_id = f"{question['question_id']}::shared"
    initial_by_requirement = {
        row["requirement_id"]: _base._locked_coarse_ids(row)
        for row in plan["requirement_plans"]
    }
    selected: set[str] = (
        set().union(*initial_by_requirement.values()) if initial_by_requirement else set()
    )
    previously_selected = set(selected)
    query_variants = [
        variant for row in plan["requirement_plans"] for variant in row["query_variants"]
    ]
    fallback = (
        _base._medium_lexical_fallback(
            side, question, query_variants, hierarchy, projections, captions,
        ) if not selected else []
    )
    evidence = _base._map_level_evidence(
        side, shared_id, selected, map_doc, hierarchy, projections, captions,
        _base._parent_map(map_doc),
    ) + fallback
    image_findings: dict[str, dict[str, Any]] = {}
    last_valid_shared: dict[str, Any] | None = None
    shared_contract_exhausted = False
    repeated_coarse_events: list[dict[str, Any]] = []

    def call_shared(round_number: int, stage: str) -> dict[str, Any]:
        nonlocal last_valid_shared, shared_contract_exhausted
        controls = _coarse_controls(
            map_doc, hierarchy, set(image_findings), previously_selected,
        )
        document, usage = _call_shared_v662(
            cfg, question, evidence, controls, round_number, last_valid_shared,
        )
        if not usage.get("shared_terminal_state_downgraded"):
            last_valid_shared = copy.deepcopy(document)
        else:
            shared_contract_exhausted = True
        calls.append({
            "stage": stage, "round": round_number, "requirement_id": shared_id,
            "reviewed_observations_available": len(image_findings), **usage,
        })
        return document

    investigation = call_shared(0, "shared_investigation")
    for round_number in range(1, max_rounds + 1):
        if shared_contract_exhausted:
            break
        if investigation["investigation_status"] != "unresolved":
            break
        requested = list(investigation["requested_coarse_ids"])
        if round_number > 1 and not requested:
            break
        repeated = [coarse_id for coarse_id in requested if coarse_id in previously_selected]
        if requested:
            repeated_coarse_events.append({
                "round": round_number, "requested_coarse_ids": requested,
                "repeated_coarse_ids": repeated,
            })
            selected |= set(requested)
            previously_selected |= set(requested)
            evidence = _base._map_level_evidence(
                side, shared_id, selected, map_doc, hierarchy, projections, captions,
                _base._parent_map(map_doc),
            ) + fallback + [
                _v652._reviewed_observation_as_evidence(
                    shared_id, 0, observation, fine_by_id,
                ) for observation in image_findings.values()
            ]
            investigation = call_shared(round_number, "shared_investigation_recheck")
            if shared_contract_exhausted:
                break
            if investigation["investigation_status"] != "unresolved":
                break
        if not selected:
            break
        observed_before = set(image_findings)
        fine_rows = _retrieve_unobserved_fines(
            cfg, side, question, plan, shared_id, selected, map_doc, hierarchy,
            projections, captions, medium_embeddings, fine_by_id, fine_embeddings,
            row_by_id, encoder, observed_before,
        )
        if fine_rows:
            unique, targets = _base._dedupe_fine_rows({shared_id: fine_rows})
            cap = int(cfg["ranking"].get("max_total_fine_evidence_per_batch", 16))
            unique = _base._allocate_fine_quota_by_requirement(unique, targets, cap)
            target = _v652._v65._v641._fine_gap_target(question, investigation)
            try:
                execution, usage, _ = _guard._execute_claims_batch_with_bounded_text_retry(
                    cfg, question, {shared_id: target}, unique,
                )
            except RuntimeError as error:
                if not str(error).startswith("Claim execution contract exhausted"):
                    raise
                _v661._ROUTE_DOWNGRADED = True
                calls.append({
                    "stage": "claim_execution_batch", "round": round_number,
                    "requirement_ids": [shared_id], "image_transmissions": len(unique),
                    "fine_ids": [row["fine_id"] for row in unique],
                    "observed_fine_ids_before": sorted(observed_before),
                    "only_new_fine_ids": True,
                    "logical_call_status": "downgraded_recovery",
                    "failure_reason": str(error),
                })
                evidence.append({
                    "evidence_id": f"claim_execution::{shared_id}::round{round_number}::contract_failure",
                    "evidence_type": "claim_execution_result",
                    "source_content": "Fine observation output failed its local contract; no invalid visual output was admitted.",
                    "claim_status": "inconclusive",
                    "supporting_fine_ids": [],
                    "retrieved_for_requirement_id": shared_id,
                })
                investigation = call_shared(round_number, "shared_investigation")
                continue
            calls.append({
                "stage": "claim_execution_batch", "round": round_number,
                "requirement_ids": [shared_id], "image_transmissions": len(unique),
                "fine_ids": [row["fine_id"] for row in unique],
                "observed_fine_ids_before": sorted(observed_before),
                "only_new_fine_ids": all(row["fine_id"] not in observed_before for row in unique),
                "reviewed_observations_returned": len(execution["observations"]), **usage,
            })
            for observation in execution["observations"]:
                image_findings[observation["fine_id"]] = observation
                evidence.append(_v652._reviewed_observation_as_evidence(
                    shared_id, round_number, observation, fine_by_id,
                ))
            evidence.append(_base._batch_assessment_as_evidence(
                shared_id, round_number, execution["claim_assessments"][shared_id],
            ))
        investigation = call_shared(round_number, "shared_investigation")

    diagnostics = {
        "formal_version": FORMAL_VERSION,
        "coarse_selection_contract": COARSE_SELECTION_CONTRACT,
        "citation_canonicalization": CITATION_CANONICALIZATION,
        "reviewed_visual_observation_count": len(image_findings),
        "repeated_coarse_events": repeated_coarse_events,
        "invalid_shared_admitted_to_history": False,
        "fine_retrieval_excludes_observed_before_ranking": True,
        "pruned_stages": [
            "atomic_coverage_reviewer", "fact_verification", "option_mapping",
            "mapping_consistency_guard", "mapping_based_final_consistency",
        ],
    }
    return investigation, evidence, calls, diagnostics


def _prompt_hashes() -> dict[str, str]:
    return _v661._prompt_hashes()


def preflight(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    if cfg.get("case_identity") == "question_id":
        from .formal_eval300 import formal_preflight
        result = formal_preflight(root, config_path, video_uid)
    else:
        result = _BASE_PREFLIGHT(root, config_path, video_uid)
    result.update({
        "formal_version": FORMAL_VERSION,
        "baseline_contract": BASELINE_CONTRACT,
        "promotion_source": PROMOTION_SOURCE,
        "coarse_selection_contract": COARSE_SELECTION_CONTRACT,
        "citation_canonicalization": CITATION_CANONICALIZATION,
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
        "execution_sides": list(_v661._configured_sides(cfg)),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "prompt_sha256_by_stage": _prompt_hashes(),
        "coarse_selection_contract": COARSE_SELECTION_CONTRACT,
        "citation_canonicalization": CITATION_CANONICALIZATION,
        "planner_reused": True,
        "online_cost_name": "post-Planner online inference cost",
        "planner_generation_cost_included": False,
        "attempt_telemetry_contract": ATTEMPT_TELEMETRY_CONTRACT,
        "max_actual_attempts_per_stage": _v661._max_attempts(cfg),
        "gold_loaded": False,
    })


def summarize(root: Path, config_path: Path) -> dict[str, Any]:
    result = _v661.summarize(root, config_path)
    cfg = load_json(config_path)
    result.update({
        "formal_version": FORMAL_VERSION,
        "baseline_contract": BASELINE_CONTRACT,
        "coarse_selection_contract": COARSE_SELECTION_CONTRACT,
        "citation_canonicalization": CITATION_CANONICALIZATION,
        "gold_loaded_before_predictions": False,
    })
    write_json(root / cfg["output_root"] / "validation_report.json", result)
    return result


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    original_preflight = _v661.preflight
    original_manifest = _v661._write_manifest
    original_journal = _v661.AttemptJournal
    original_resolver = _v652._resolve_to_shared
    original_summarize = _v661.summarize
    _v661.preflight = preflight
    _v661._write_manifest = _write_manifest
    _v661.AttemptJournal = AttemptJournal
    _v652._resolve_to_shared = _resolve_to_shared_v662
    # Prevent the base runner's terminal summarize from writing a V6.6.1-labelled report.
    _v661.summarize = lambda root_, config_: {"deferred_to": FORMAL_VERSION}
    try:
        _v661.run_live(root, config_path, video_uid=video_uid)
    finally:
        _v661.preflight = original_preflight
        _v661._write_manifest = original_manifest
        _v661.AttemptJournal = original_journal
        _v652._resolve_to_shared = original_resolver
        _v661.summarize = original_summarize
    cfg = load_json(config_path)
    # The inherited writer names this one envelope field after its source version.
    # Normalize only the version label; the contained diagnostics are already V6.6.2.
    for path in (root / cfg["output_root"] / "cases").glob("*/*/shared_investigation.json"):
        document = load_json(path)
        if "v6_6_1_diagnostics" in document:
            document["v6_6_2_diagnostics"] = document.pop("v6_6_1_diagnostics")
            write_json(path, document)
    return summarize(root, config_path)


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    """Post-hoc fixed-population scoring; never invoked during source/live execution."""
    cfg = load_json(config_path)
    if cfg.get("case_identity") != "question_id":
        return _v661.evaluate(root, config_path)
    from .formal_eval300 import score_fixed_population
    output = root / cfg["output_root"]
    uid_path = root / cfg["eval300_ordered_uid_path"]
    ordered = [row.strip() for row in uid_path.read_text(encoding="utf-8").splitlines() if row.strip()]
    annotations = load_json(Path(cfg["annotation_path"]))
    gold = {question_id: _base._find_gold(annotations, question_id) for question_id in ordered}
    predictions = []
    for question_id in ordered:
        for side in _v661._configured_sides(cfg):
            final_path = output / "cases" / question_id / side / "final_answer.json"
            status_path = output / "cases" / question_id / side / "route_status.json"
            status = load_json(status_path) if status_path.is_file() else {}
            row = {"question_id": question_id, "route": side, **status}
            if final_path.is_file():
                row["selected_option_id"] = load_json(final_path)["answer"]["selected_option_id"]
            predictions.append(row)
    result = {
        side: score_fixed_population(ordered, predictions, gold, set(cfg["pilot10_question_ids"]), side)
        for side in _v661._configured_sides(cfg)
    }
    write_json(output / "posthoc_evaluation.json", result)
    return result
