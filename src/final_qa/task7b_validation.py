"""Authoritative local validation for Task 7B final-QA pilot outputs."""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field


AnswerStatus = Literal["answered", "answered_with_uncertainty", "insufficient_evidence", "query_or_premise_inconsistent"]
UncertaintyStatus = Literal["unresolved", "resolved_by_final_evidence", "not_assessable"]
Modality = Literal["visual", "speech", "acoustic"]


class EvidenceUsed(BaseModel):
    evidence_id: str
    modality: Modality
    start_sec: float
    end_sec: float
    supported_claim: str


class InheritedUncertainty(BaseModel):
    uncertainty_id: str
    type: str
    description: str
    affected_claim: str
    severity: Literal["minor", "material", "critical"]
    source: Literal["pipeline"] = "pipeline"
    resolvable_with: str | None = None


class UncertaintyAssessment(BaseModel):
    uncertainty_id: str
    status: UncertaintyStatus
    assessment: str
    evidence_ids: list[str] = Field(default_factory=list)


class FinalUncertainty(BaseModel):
    type: str
    description: str
    affected_claim: str
    severity: Literal["minor", "material", "critical"]
    source: Literal["pipeline", "final_model"]
    resolvable_with: str | None = None


class Confidence(BaseModel):
    level: Literal["high", "medium", "low"]
    basis: str


class FinalQAResult(BaseModel):
    case_id: str
    answer_status: AnswerStatus
    answer: str | None
    reasoning_summary: str
    evidence_used: list[EvidenceUsed]
    inherited_pipeline_uncertainties: list[InheritedUncertainty]
    uncertainty_assessments: list[UncertaintyAssessment]
    final_uncertainties: list[FinalUncertainty]
    missing_information: list[str]
    confidence: Confidence
    abstain: bool


def with_uncertainty_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"uncertainty_id": f"uncertainty_{index:03d}", **copy.deepcopy(record)} for index, record in enumerate(records, start=1)]


def evidence_catalog(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for group in payload["evidence_groups"]:
        for modality, key in (("visual", "visual_evidence"), ("speech", "speech_evidence"), ("acoustic", "acoustic_evidence")):
            for item in group[key]:
                catalog[item["evidence_id"]] = {"modality": modality, **copy.deepcopy(item)}
    return catalog


def uncertainty_resolution_supported(uncertainty: dict[str, Any], assessment: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> bool:
    cited = [catalog[item] for item in assessment.get("evidence_ids", []) if item in catalog]
    if not cited or len(cited) != len(assessment.get("evidence_ids", [])):
        return False
    kind = uncertainty["type"]
    if kind == "speaker_attribution_uncertainty":
        return any(item.get("speaker_verified") is True for item in cited)
    if kind == "semantic_uncertainty":
        return any(item["modality"] in {"visual", "acoustic"} for item in cited)
    if kind == "fallback_recovered_evidence":
        return any(item["modality"] == "speech" and item.get("fallback_provenance") is True for item in cited)
    if kind == "dataset_or_query_inconsistency_unknown":
        return False
    return True


def validate_and_correct(model_result: FinalQAResult | dict[str, Any], payload: dict[str, Any], delivered_modalities: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = model_result.model_dump(mode="json") if isinstance(model_result, FinalQAResult) else copy.deepcopy(model_result)
    corrections: list[dict[str, Any]] = []
    warnings: list[str] = []
    catalog = evidence_catalog(payload)
    expected_uncertainties = with_uncertainty_ids(payload["pipeline_uncertainties"])
    expected_by_id = {item["uncertainty_id"]: item for item in expected_uncertainties}

    if result.get("case_id") != payload["case_id"]:
        corrections.append({"field": "case_id", "reason": "case_id_mismatch", "before": result.get("case_id"), "after": payload["case_id"]})
        result["case_id"] = payload["case_id"]

    if result.get("inherited_pipeline_uncertainties") != expected_uncertainties:
        corrections.append({"field": "inherited_pipeline_uncertainties", "reason": "immutable_pipeline_audit_restored"})
        result["inherited_pipeline_uncertainties"] = copy.deepcopy(expected_uncertainties)

    assessments_by_id: dict[str, dict[str, Any]] = {}
    for item in result.get("uncertainty_assessments", []):
        uncertainty_id = item.get("uncertainty_id")
        if uncertainty_id in expected_by_id and uncertainty_id not in assessments_by_id:
            assessments_by_id[uncertainty_id] = copy.deepcopy(item)
        elif uncertainty_id in assessments_by_id:
            corrections.append({"field": "uncertainty_assessments", "reason": "duplicate_assessment_removed", "uncertainty_id": uncertainty_id})
        else:
            corrections.append({"field": "uncertainty_assessments", "reason": "unknown_assessment_removed", "uncertainty_id": uncertainty_id})
    for uncertainty_id, inherited in expected_by_id.items():
        if uncertainty_id not in assessments_by_id:
            assessments_by_id[uncertainty_id] = {"uncertainty_id": uncertainty_id, "status": "not_assessable", "assessment": "The model did not provide the required assessment; local validation preserved the uncertainty.", "evidence_ids": []}
            corrections.append({"field": "uncertainty_assessments", "reason": "missing_assessment_added", "uncertainty_id": uncertainty_id})
        assessment = assessments_by_id[uncertainty_id]
        if assessment.get("status") == "resolved_by_final_evidence" and not uncertainty_resolution_supported(inherited, assessment, catalog):
            assessment["status"] = "unresolved"
            assessment["assessment"] = f"Unsupported resolution rejected locally. {assessment.get('assessment', '')}".strip()
            corrections.append({"field": "uncertainty_assessments", "reason": "unsupported_resolution_rejected", "uncertainty_id": uncertainty_id})
    result["uncertainty_assessments"] = list(assessments_by_id.values())

    valid_citations: list[dict[str, Any]] = []
    for citation in result.get("evidence_used", []):
        evidence = catalog.get(citation.get("evidence_id"))
        if evidence is None:
            corrections.append({"field": "evidence_used", "reason": "unseen_evidence_removed", "evidence_id": citation.get("evidence_id")})
            continue
        start, end = float(citation.get("start_sec", 0)), float(citation.get("end_sec", 0))
        if start < float(evidence["start_sec"]) - 1e-6 or end > float(evidence["end_sec"]) + 1e-6 or end < start:
            corrections.append({"field": "evidence_used", "reason": "citation_timestamp_outside_evidence_removed", "evidence_id": citation.get("evidence_id")})
            continue
        if citation.get("modality") != evidence["modality"]:
            corrections.append({"field": "evidence_used", "reason": "citation_modality_corrected", "evidence_id": citation["evidence_id"], "after": evidence["modality"]})
            citation["modality"] = evidence["modality"]
        claim = str(citation.get("supported_claim", ""))
        if "clap" in claim.casefold() and any(term in claim.casefold() for term in ("proves", "confirms", "identifies", "means")):
            corrections.append({"field": "evidence_used", "reason": "clap_semantic_claim_removed", "evidence_id": citation["evidence_id"]})
            continue
        valid_citations.append(citation)
    result["evidence_used"] = valid_citations

    required = set(payload["answer_required_modalities"])
    if not required.issubset(delivered_modalities):
        result["answer_status"] = "insufficient_evidence"
        result["answer"] = None
        result["abstain"] = True
        corrections.append({"field": "answer_status", "reason": "answer_required_modality_not_delivered", "missing": sorted(required - delivered_modalities)})

    unresolved_ids = {item["uncertainty_id"] for item in result["uncertainty_assessments"] if item["status"] != "resolved_by_final_evidence"}
    final_uncertainties = [copy.deepcopy(item) for item in result.get("final_uncertainties", [])]
    final_keys = {(item.get("type"), item.get("description"), item.get("affected_claim")) for item in final_uncertainties}
    for uncertainty_id in unresolved_ids:
        inherited = expected_by_id[uncertainty_id]
        key = (inherited["type"], inherited["description"], inherited["affected_claim"])
        if key not in final_keys:
            final_uncertainties.append({key: value for key, value in inherited.items() if key != "uncertainty_id"})
            final_keys.add(key)
            corrections.append({"field": "final_uncertainties", "reason": "unresolved_pipeline_uncertainty_restored", "uncertainty_id": uncertainty_id})
    result["final_uncertainties"] = final_uncertainties

    critical_unresolved = [expected_by_id[item] for item in unresolved_ids if expected_by_id[item]["severity"] == "critical"]
    if result.get("answer_status") == "answered" and critical_unresolved:
        result["answer_status"] = "answered_with_uncertainty"
        corrections.append({"field": "answer_status", "reason": "critical_answer_affecting_uncertainty_remains", "after": "answered_with_uncertainty"})
    if result.get("answer_status") == "answered_with_uncertainty" and not result["final_uncertainties"]:
        warnings.append("answered_with_uncertainty_without_final_uncertainty")
        result["answer_status"] = "answered"
        corrections.append({"field": "answer_status", "reason": "no_final_uncertainty_present", "after": "answered"})

    should_abstain = result.get("answer_status") in {"insufficient_evidence", "query_or_premise_inconsistent"}
    if bool(result.get("abstain")) != should_abstain:
        corrections.append({"field": "abstain", "reason": "aligned_with_answer_status", "before": result.get("abstain"), "after": should_abstain})
        result["abstain"] = should_abstain

    lowered_answer = str(result.get("answer") or "").casefold()
    if any(term in lowered_answer for term in ("the passenger is", "the speaker is", "the man is", "the woman is")) and not any(item.get("speaker_verified") is True for item in catalog.values()):
        warnings.append("speaker_identity_claim_without_verified_speaker_evidence")
        if result["answer_status"] == "answered":
            result["answer_status"] = "answered_with_uncertainty"
            corrections.append({"field": "answer_status", "reason": "speaker_identity_not_verified", "after": "answered_with_uncertainty"})

    validated = FinalQAResult.model_validate(result).model_dump(mode="json")
    report = {
        "case_id": payload["case_id"],
        "validation_passed": True,
        "corrections": corrections,
        "warnings": warnings,
        "supplied_evidence_ids": sorted(catalog),
        "cited_evidence_ids": [item["evidence_id"] for item in validated["evidence_used"]],
        "inherited_uncertainty_count": len(expected_uncertainties),
        "assessment_count": len(validated["uncertainty_assessments"]),
        "final_uncertainty_count": len(validated["final_uncertainties"]),
    }
    return validated, report
