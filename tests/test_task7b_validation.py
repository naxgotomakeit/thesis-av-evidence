from __future__ import annotations

import copy
import json
from pathlib import Path

from src.final_qa.task7b_validation import FinalQAResult, validate_and_correct, with_uncertainty_ids


ROOT = Path(__file__).resolve().parents[1]
PAYLOADS = ROOT / "outputs/final_qa_preflight/task7a_v1/task7a_payloads.jsonl"


def payload(case_id: str) -> dict:
    return next(json.loads(line) for line in PAYLOADS.read_text(encoding="utf-8").splitlines() if line and json.loads(line)["case_id"] == case_id)


def base_result(source: dict, status: str = "answered_with_uncertainty") -> dict:
    inherited = with_uncertainty_ids(source["pipeline_uncertainties"])
    return {
        "case_id": source["case_id"], "answer_status": status, "answer": "candidate answer", "reasoning_summary": "Brief evidence-grounded summary.",
        "evidence_used": [], "inherited_pipeline_uncertainties": copy.deepcopy(inherited),
        "uncertainty_assessments": [{"uncertainty_id": item["uncertainty_id"], "status": "unresolved", "assessment": "Remains unresolved.", "evidence_ids": []} for item in inherited],
        "final_uncertainties": [{key: value for key, value in item.items() if key != "uncertainty_id"} for item in inherited],
        "missing_information": [], "confidence": {"level": "medium", "basis": "Evidence is limited."}, "abstain": False,
    }


def test_h_unseen_evidence_citation_is_rejected():
    source = payload("00018_1")
    result = base_result(source)
    result["evidence_used"] = [{"evidence_id": "unseen", "modality": "acoustic", "start_sec": 0, "end_sec": 1, "supported_claim": "claim"}]
    validated, report = validate_and_correct(result, source, {"acoustic"})
    assert validated["evidence_used"] == []
    assert any(item["reason"] == "unseen_evidence_removed" for item in report["corrections"])


def test_i_pipeline_uncertainty_is_restored_immutably():
    source = payload("00018_1")
    result = base_result(source)
    result["inherited_pipeline_uncertainties"] = []
    validated, report = validate_and_correct(result, source, {"acoustic"})
    assert validated["inherited_pipeline_uncertainties"] == with_uncertainty_ids(source["pipeline_uncertainties"])
    assert any(item["reason"] == "immutable_pipeline_audit_restored" for item in report["corrections"])


def test_j_each_inherited_uncertainty_gets_exactly_one_assessment():
    source = payload("00018_1")
    result = base_result(source)
    result["uncertainty_assessments"] = []
    validated, _ = validate_and_correct(result, source, {"acoustic"})
    assert len(validated["uncertainty_assessments"]) == len(source["pipeline_uncertainties"])
    assert len({item["uncertainty_id"] for item in validated["uncertainty_assessments"]}) == len(source["pipeline_uncertainties"])


def test_k_unsupported_speaker_resolution_is_rejected():
    source = payload("00006_3")
    result = base_result(source)
    speaker = next(item for item in result["inherited_pipeline_uncertainties"] if item["type"] == "speaker_attribution_uncertainty")
    speech_id = next(item["evidence_id"] for group in source["evidence_groups"] for item in group["speech_evidence"])
    assessment = next(item for item in result["uncertainty_assessments"] if item["uncertainty_id"] == speaker["uncertainty_id"])
    assessment.update({"status": "resolved_by_final_evidence", "evidence_ids": [speech_id], "assessment": "Transcript resolves speaker."})
    validated, report = validate_and_correct(result, source, {"speech"})
    corrected = next(item for item in validated["uncertainty_assessments"] if item["uncertainty_id"] == speaker["uncertainty_id"])
    assert corrected["status"] == "unresolved"
    assert any(item["reason"] == "unsupported_resolution_rejected" for item in report["corrections"])


def test_l_critical_unresolved_uncertainty_prevents_plain_answered():
    source = payload("00018_1")
    source = copy.deepcopy(source)
    source["pipeline_uncertainties"].append({"type": "evidence_missing", "description": "Critical evidence gap.", "affected_claim": "answer", "severity": "critical", "source": "pipeline", "resolvable_with": "additional evidence"})
    result = base_result(source, status="answered")
    validated, report = validate_and_correct(result, source, {"acoustic"})
    assert validated["answer_status"] == "answered_with_uncertainty"
    assert any(item["reason"] == "critical_answer_affecting_uncertainty_remains" for item in report["corrections"])


def test_answer_required_modality_must_be_delivered():
    source = payload("00018_1")
    validated, report = validate_and_correct(base_result(source), source, set())
    assert validated["answer_status"] == "insufficient_evidence"
    assert validated["answer"] is None and validated["abstain"] is True


def test_p_task7a_and_task6_inputs_exist_for_hash_integrity_guard():
    paths = [
        ROOT / "outputs/final_qa_preflight/task7a_v1/task7a_payloads.jsonl",
        ROOT / "outputs/final_qa_preflight/task7a_v1/task7a_preflight_results.jsonl",
        ROOT / "outputs/relation_reranking/task6_v1_2/task6_evidence_packets.jsonl",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
