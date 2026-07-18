"""Task 7B v0.2 uncertainty, timestamp, usage, and reporting semantics."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.final_qa.task7b_validation import Confidence, UncertaintyAssessment, evidence_catalog, uncertainty_resolution_supported


TimestampSource = Literal["pipeline_validated", "model_estimated", "inherited_asr", "question_provided"]
AnswerStatus = Literal["answered", "answered_with_uncertainty", "insufficient_evidence", "query_or_premise_inconsistent"]


class EvidenceUsedV02(BaseModel):
    evidence_id: str
    modality: Literal["visual", "speech", "acoustic"]
    start_sec: float
    end_sec: float
    timestamp_source: TimestampSource
    container_evidence_start_sec: float
    container_evidence_end_sec: float
    supported_claim: str


class NewlyObservedUncertainty(BaseModel):
    type: str
    description: str
    affected_claim: str
    severity: Literal["minor", "material", "critical"]
    source: Literal["final_model"] = "final_model"
    resolvable_with: str | None = None


class FinalQAModelOutputV02(BaseModel):
    case_id: str
    answer_status: AnswerStatus
    answer: str | None
    reasoning_summary: str
    evidence_used: list[EvidenceUsedV02]
    uncertainty_assessments: list[UncertaintyAssessment]
    newly_observed_uncertainties: list[NewlyObservedUncertainty] = Field(default_factory=list)
    missing_information: list[str]
    confidence: Confidence
    abstain: bool


def dataset_audit_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("dataset_or_query_inconsistency_status", "unknown")
    mapped = "no_machine_detectable_issue" if status == "not_detected" else "not_evaluated"
    return {"status": mapped, "source": "pipeline", "affects_answer_policy": False, "notes": []}


def active_pipeline_uncertainties(payload: dict[str, Any]) -> list[dict[str, Any]]:
    active = []
    for index, record in enumerate(payload.get("pipeline_uncertainties", []), start=1):
        if record.get("type") == "dataset_or_query_inconsistency_unknown":
            continue
        active.append({"uncertainty_id": f"uncertainty_{index:03d}", **copy.deepcopy(record)})
    return active


def _final_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in record.items() if key != "uncertainty_id"}


def token_accounting(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = usage.get("provider_input_token_count", usage.get("input_tokens"))
    output_tokens = usage.get("provider_output_token_count", usage.get("output_tokens"))
    total_tokens = usage.get("provider_total_token_count", usage.get("total_tokens"))
    unattributed = None
    if all(isinstance(value, int) for value in (input_tokens, output_tokens, total_tokens)):
        unattributed = total_tokens - input_tokens - output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "unattributed_tokens": unattributed,
        "input_tokens_by_modality": copy.deepcopy(usage.get("input_tokens_by_modality")),
        "output_tokens_by_modality": copy.deepcopy(usage.get("output_tokens_by_modality")),
        "provider_usage_warnings": copy.deepcopy(usage.get("warnings", [])),
        "total_token_metric_source": "provider_total_token_count",
    }


def normalize_answer(text: str | None) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text or "").casefold()).split())


def posthoc_answer_record(case_id: str, question: str, gold: str | None, raw_answer: str | None, validated: dict[str, Any]) -> dict[str, Any]:
    exact = bool(gold is not None and normalize_answer(gold) == normalize_answer(validated.get("answer")))
    return {
        "case_id": case_id,
        "question": question,
        "gold_dataset_answer": gold,
        "raw_model_prediction": raw_answer,
        "validated_prediction": validated.get("answer"),
        "answer_status": validated.get("answer_status"),
        "abstain": validated.get("abstain"),
        "confidence": validated.get("confidence"),
        "normalized_exact_match_diagnostic": exact,
        "automatic_correctness": "not_determined",
        "human_review_status": "unreviewed",
        "human_correctness": None,
        "human_review_notes": None,
        "gold_quality_status": "unknown",
        "evaluation_stage": "post_hoc_after_prediction_and_validation",
    }


def _correct_evidence_used(items: list[dict[str, Any]], catalog: dict[str, dict[str, Any]], corrections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for item in items:
        evidence = catalog.get(item.get("evidence_id"))
        if evidence is None:
            corrections.append({"field": "evidence_used", "reason": "unseen_evidence_removed", "evidence_id": item.get("evidence_id")})
            continue
        container_start, container_end = float(evidence["start_sec"]), float(evidence["end_sec"])
        start, end = float(item.get("start_sec", container_start)), float(item.get("end_sec", container_end))
        if start < container_start - 1e-6 or end > container_end + 1e-6 or end < start:
            corrections.append({"field": "evidence_used", "reason": "model_timestamp_outside_container_removed", "evidence_id": item["evidence_id"]})
            continue
        source = item.get("timestamp_source")
        if source not in {"pipeline_validated", "model_estimated", "inherited_asr", "question_provided"}:
            source = "model_estimated" if (start, end) != (container_start, container_end) else "pipeline_validated"
            corrections.append({"field": "evidence_used", "reason": "timestamp_source_corrected", "evidence_id": item["evidence_id"], "after": source})
        if source == "pipeline_validated" and (abs(start - container_start) > 1e-6 or abs(end - container_end) > 1e-6):
            source = "model_estimated"
            corrections.append({"field": "evidence_used", "reason": "subinterval_marked_model_estimated", "evidence_id": item["evidence_id"]})
        output.append({
            "evidence_id": item["evidence_id"], "modality": evidence["modality"], "start_sec": start, "end_sec": end,
            "timestamp_source": source, "container_evidence_start_sec": container_start, "container_evidence_end_sec": container_end,
            "supported_claim": item.get("supported_claim", ""),
        })
    return output


def _apply_textual_model_estimates(result: dict[str, Any], evidence: list[dict[str, Any]], corrections: list[dict[str, Any]]) -> None:
    """Record an explicit model-stated absolute sub-interval without upgrading its provenance."""
    text = " ".join(str(result.get(key) or "") for key in ("answer", "reasoning_summary"))
    patterns = (
        r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
        r"\bfrom\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)?\s*(?:to|until|through)\s*(?:around|approximately|about)?\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
    )
    ranges: list[tuple[float, float]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = float(match.group(1)), float(match.group(2))
            if end >= start and (start, end) not in ranges:
                ranges.append((start, end))
    for start, end in ranges:
        compatible = [item for item in evidence if item["container_evidence_start_sec"] - 1e-6 <= start <= end <= item["container_evidence_end_sec"] + 1e-6]
        if len(compatible) != 1:
            continue
        item = compatible[0]
        if (start, end) == (item["start_sec"], item["end_sec"]):
            continue
        item["start_sec"], item["end_sec"] = start, end
        item["timestamp_source"] = "model_estimated"
        corrections.append({"field": "evidence_used", "reason": "textual_subinterval_recorded_as_model_estimated", "evidence_id": item["evidence_id"], "start_sec": start, "end_sec": end})


def validate_v02(model_output: FinalQAModelOutputV02 | dict[str, Any], payload: dict[str, Any], delivered_modalities: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = model_output.model_dump(mode="json") if isinstance(model_output, FinalQAModelOutputV02) else copy.deepcopy(model_output)
    corrections: list[dict[str, Any]] = []
    warnings: list[str] = []
    catalog = evidence_catalog(payload)
    immutable = active_pipeline_uncertainties(payload)
    immutable_by_id = {item["uncertainty_id"]: item for item in immutable}
    result["case_id"] = payload["case_id"]
    result["evidence_used"] = _correct_evidence_used(result.get("evidence_used", []), catalog, corrections)
    _apply_textual_model_estimates(result, result["evidence_used"], corrections)

    assessments: dict[str, dict[str, Any]] = {}
    for item in result.get("uncertainty_assessments", []):
        uncertainty_id = item.get("uncertainty_id")
        if uncertainty_id in immutable_by_id and uncertainty_id not in assessments:
            assessments[uncertainty_id] = copy.deepcopy(item)
        else:
            corrections.append({"field": "uncertainty_assessments", "reason": "unknown_or_duplicate_assessment_removed", "uncertainty_id": uncertainty_id})
    for uncertainty_id, record in immutable_by_id.items():
        if uncertainty_id not in assessments:
            assessments[uncertainty_id] = {"uncertainty_id": uncertainty_id, "status": "not_assessable", "assessment": "No assessment returned; uncertainty preserved locally.", "evidence_ids": []}
            corrections.append({"field": "uncertainty_assessments", "reason": "missing_assessment_added", "uncertainty_id": uncertainty_id})
        assessment = assessments[uncertainty_id]
        if assessment.get("status") == "resolved_by_final_evidence" and not uncertainty_resolution_supported(record, assessment, catalog):
            assessment["status"] = "unresolved"
            assessment["assessment"] = "Unsupported resolution rejected locally. " + assessment.get("assessment", "")
            corrections.append({"field": "uncertainty_assessments", "reason": "unsupported_resolution_rejected", "uncertainty_id": uncertainty_id})
    result["uncertainty_assessments"] = list(assessments.values())

    newly_observed = [NewlyObservedUncertainty.model_validate(item).model_dump(mode="json") for item in result.get("newly_observed_uncertainties", [])]
    final_uncertainties = [_final_record(immutable_by_id[item_id]) for item_id, assessment in assessments.items() if assessment["status"] != "resolved_by_final_evidence"]
    final_uncertainties.extend(copy.deepcopy(newly_observed))
    result["immutable_pipeline_uncertainties"] = immutable
    result["newly_observed_uncertainties"] = newly_observed
    result["final_uncertainties"] = final_uncertainties
    result["dataset_audit"] = dataset_audit_from_payload(payload)

    required = set(payload["answer_required_modalities"])
    if not required.issubset(delivered_modalities):
        result.update({"answer_status": "insufficient_evidence", "answer": None, "abstain": True})
        corrections.append({"field": "answer_status", "reason": "answer_required_modality_not_delivered", "missing": sorted(required - delivered_modalities)})
    critical = [item for item in final_uncertainties if item["severity"] == "critical"]
    if result.get("answer_status") == "answered" and critical:
        result["answer_status"] = "answered_with_uncertainty"
        corrections.append({"field": "answer_status", "reason": "critical_uncertainty_remains", "after": "answered_with_uncertainty"})
    if result.get("answer_status") == "answered_with_uncertainty" and not final_uncertainties:
        result["answer_status"] = "answered"
        corrections.append({"field": "answer_status", "reason": "no_active_final_uncertainty", "after": "answered"})
    should_abstain = result.get("answer_status") in {"insufficient_evidence", "query_or_premise_inconsistent"}
    if bool(result.get("abstain")) != should_abstain:
        result["abstain"] = should_abstain
        corrections.append({"field": "abstain", "reason": "aligned_with_answer_status", "after": should_abstain})

    validated = {
        "case_id": result["case_id"], "answer_status": result["answer_status"], "answer": result.get("answer"),
        "reasoning_summary": result.get("reasoning_summary", ""), "evidence_used": result["evidence_used"],
        "immutable_pipeline_uncertainties": immutable, "uncertainty_assessments": result["uncertainty_assessments"],
        "newly_observed_uncertainties": newly_observed, "final_uncertainties": final_uncertainties,
        "dataset_audit": result["dataset_audit"], "missing_information": result.get("missing_information", []),
        "confidence": result["confidence"], "abstain": result["abstain"],
    }
    report = {
        "case_id": payload["case_id"], "validation_passed": True, "corrections": corrections, "warnings": warnings,
        "supplied_evidence_ids": sorted(catalog), "cited_evidence_ids": [item["evidence_id"] for item in validated["evidence_used"]],
        "inherited_uncertainty_count": len(immutable), "resolved_uncertainty_count": sum(item["status"] == "resolved_by_final_evidence" for item in validated["uncertainty_assessments"]),
        "newly_observed_uncertainty_count": len(newly_observed), "final_uncertainty_count": len(final_uncertainties),
    }
    return validated, report


def reconstruct_v01_validated(old: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = evidence_catalog(payload)
    active = active_pipeline_uncertainties(payload)
    old_assessments = {item["uncertainty_id"]: item for item in old.get("uncertainty_assessments", [])}
    assessments = []
    for record in active:
        assessments.append(copy.deepcopy(old_assessments.get(record["uncertainty_id"], {"uncertainty_id": record["uncertainty_id"], "status": "not_assessable", "assessment": "Not assessed in v0.1.", "evidence_ids": []})))
    evidence = []
    for item in old.get("evidence_used", []):
        container = catalog[item["evidence_id"]]
        source = "model_estimated" if (float(item["start_sec"]), float(item["end_sec"])) != (float(container["start_sec"]), float(container["end_sec"])) else "pipeline_validated"
        evidence.append({**copy.deepcopy(item), "timestamp_source": source, "container_evidence_start_sec": float(container["start_sec"]), "container_evidence_end_sec": float(container["end_sec"])})
    estimate_corrections: list[dict[str, Any]] = []
    _apply_textual_model_estimates(old, evidence, estimate_corrections)
    final_uncertainties = [_final_record(record) for record in active if next(item for item in assessments if item["uncertainty_id"] == record["uncertainty_id"])["status"] != "resolved_by_final_evidence"]
    validated = {
        "case_id": old["case_id"], "answer_status": old["answer_status"], "answer": old.get("answer"), "reasoning_summary": old["reasoning_summary"],
        "evidence_used": evidence, "immutable_pipeline_uncertainties": active, "uncertainty_assessments": assessments,
        "newly_observed_uncertainties": [], "final_uncertainties": final_uncertainties, "dataset_audit": dataset_audit_from_payload(payload),
        "missing_information": old.get("missing_information", []), "confidence": old["confidence"], "abstain": old["abstain"],
    }
    report = {"case_id": old["case_id"], "validation_passed": True, "corrections": [{"field": "reporting", "reason": "dataset_audit_separated_from_answer_uncertainty"}, {"field": "schema", "reason": "immutable_pipeline_uncertainties_moved_to_local_audit"}, *estimate_corrections], "warnings": [], "supplied_evidence_ids": sorted(catalog), "cited_evidence_ids": [item["evidence_id"] for item in evidence], "inherited_uncertainty_count": len(active), "resolved_uncertainty_count": sum(item["status"] == "resolved_by_final_evidence" for item in assessments), "newly_observed_uncertainty_count": 0, "final_uncertainty_count": len(final_uncertainties)}
    return validated, report
