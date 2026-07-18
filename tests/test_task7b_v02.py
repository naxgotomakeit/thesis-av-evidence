from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts import run_task7b_v02_v03 as runner
from src.final_qa.task7b_gemini import build_request
from src.final_qa.task7b_reporting import guarded_load_gold, make_review_html
from src.final_qa.task7b_v02 import (
    FinalQAModelOutputV02,
    active_pipeline_uncertainties,
    dataset_audit_from_payload,
    token_accounting,
    validate_v02,
)


ROOT = Path(__file__).resolve().parents[1]
PAYLOADS = ROOT / "outputs/final_qa_preflight/task7a_v1/task7a_payloads.jsonl"


def payloads() -> dict[str, dict]:
    return {row["case_id"]: row for row in (json.loads(line) for line in PAYLOADS.read_text(encoding="utf-8").splitlines() if line)}


def model_output(payload: dict, evidence_id: str, modality: str, start: float, end: float) -> dict:
    active = active_pipeline_uncertainties(payload)
    return {
        "case_id": payload["case_id"], "answer_status": "answered_with_uncertainty", "answer": "fixture prediction",
        "reasoning_summary": "Brief evidence-grounded summary.",
        "evidence_used": [{"evidence_id": evidence_id, "modality": modality, "start_sec": start, "end_sec": end, "timestamp_source": "model_estimated", "container_evidence_start_sec": start, "container_evidence_end_sec": end, "supported_claim": "fixture"}],
        "uncertainty_assessments": [{"uncertainty_id": item["uncertainty_id"], "status": "unresolved", "assessment": "Unresolved.", "evidence_ids": []} for item in active],
        "newly_observed_uncertainties": [], "missing_information": [], "confidence": {"level": "medium", "basis": "Evidence limited."}, "abstain": False,
    }


def first_evidence(payload: dict) -> tuple[str, str, float, float]:
    for group in payload["evidence_groups"]:
        for key, modality in (("visual_evidence", "visual"), ("speech_evidence", "speech"), ("acoustic_evidence", "acoustic")):
            if group[key]:
                item = group[key][0]
                return item["evidence_id"], modality, float(item["start_sec"]), float(item["end_sec"])
    raise AssertionError("fixture has no evidence")


def test_a_gold_is_guarded_until_raw_and_validated_exist():
    tmp_path = ROOT / ".tmp_test/task7b_gold_guard"
    tmp_path.mkdir(parents=True, exist_ok=True)
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([{"case_id": "case", "answer": "secret"}]), encoding="utf-8")
    for name in ("raw.jsonl", "valid.jsonl"):
        path = tmp_path / name
        if path.exists():
            path.unlink()
    with pytest.raises(RuntimeError, match="requires_saved"):
        guarded_load_gold(gold, tmp_path / "raw.jsonl", tmp_path / "valid.jsonl", ["case"])
    (tmp_path / "raw.jsonl").write_text('{"case_id":"case"}\n', encoding="utf-8")
    (tmp_path / "valid.jsonl").write_text('{"case_id":"case"}\n', encoding="utf-8")
    assert guarded_load_gold(gold, tmp_path / "raw.jsonl", tmp_path / "valid.jsonl", ["case"])["case"] == "secret"


def test_b_c_d_e_dataset_audit_is_separate_and_model_does_not_copy_immutable_uncertainty():
    payload = payloads()["00002_7"]
    active = active_pipeline_uncertainties(payload)
    assert dataset_audit_from_payload(payload)["status"] in {"not_evaluated", "no_machine_detectable_issue"}
    assert all(item["type"] != "dataset_or_query_inconsistency_unknown" for item in active)
    schema = FinalQAModelOutputV02.model_json_schema()
    assert "inherited_pipeline_uncertainties" not in json.dumps(schema)
    request, manifest, _ = build_request(payload, ROOT, "gemini-3.5-flash", "low", False, response_model=FinalQAModelOutputV02, uncertainties_override=active)
    serialized = json.dumps({"request": request, "manifest": manifest}).casefold()
    assert "dataset_or_query_inconsistency_unknown" not in serialized
    assert "gold_dataset_answer" not in serialized and "expected_answer" not in serialized


def test_f_acoustic_request_explicitly_requests_interference_assessment():
    payload = payloads()["00002_7"]
    request, _, _ = build_request(payload, ROOT, "gemini-3.5-flash", "low", False, response_model=FinalQAModelOutputV02, uncertainties_override=active_pipeline_uncertainties(payload))
    text = json.dumps(request).casefold()
    for phrase in ("overlapping speech", "background noise", "reverberation", "signal_interference_uncertainty"):
        assert phrase in text


def test_g_model_estimated_timestamp_is_bounded_and_labelled():
    payload = payloads()["00002_7"]
    evidence_id, modality, start, end = first_evidence(payload)
    output = model_output(payload, evidence_id, modality, start, min(end, start + 0.4))
    validated, _ = validate_v02(output, payload, {modality})
    item = validated["evidence_used"][0]
    assert item["timestamp_source"] == "model_estimated"
    assert item["container_evidence_start_sec"] <= item["start_sec"] <= item["end_sec"] <= item["container_evidence_end_sec"]


def test_g_textual_absolute_subinterval_is_recorded_as_model_estimated():
    payload = payloads()["00002_7"]
    evidence_id, modality, start, end = first_evidence(payload)
    output = model_output(payload, evidence_id, modality, start, end)
    sub_start, sub_end = start + 0.2, min(end, start + 0.7)
    output["answer"] = f"The observed event occurs around {sub_start:.1f} to {sub_end:.1f} seconds."
    validated, report = validate_v02(output, payload, {modality})
    item = validated["evidence_used"][0]
    assert (item["start_sec"], item["end_sec"], item["timestamp_source"]) == (sub_start, sub_end, "model_estimated")
    assert any(correction["reason"] == "textual_subinterval_recorded_as_model_estimated" for correction in report["corrections"])


def test_h_i_unattributed_tokens_and_provider_total_are_authoritative():
    result = token_accounting({"provider_input_token_count": 1191, "provider_output_token_count": 675, "provider_total_token_count": 2074})
    assert result["unattributed_tokens"] == 208
    assert result["total_tokens"] == 2074
    assert result["total_token_metric_source"] == "provider_total_token_count"


def test_j_k_html_has_full_answers_and_readable_tables():
    payload = payloads()["00002_7"]
    case_id = payload["case_id"]
    validated = {case_id: {"answer_status": "answered", "answer": "full validated prediction", "confidence": {"level": "high"}, "abstain": False, "immutable_pipeline_uncertainties": [], "final_uncertainties": [], "dataset_audit": {"status": "not_evaluated"}, "evidence_used": []}}
    usage = {case_id: {"model": "gemini-3.5-flash", "thinking_level": "low", "input_tokens": 1, "output_tokens": 2, "unattributed_tokens": 3, "total_tokens": 6, "total_api_calls": 1}}
    posthoc = {case_id: {"gold_dataset_answer": "full gold answer", "raw_model_prediction": "raw prediction", "human_review_status": "unreviewed"}}
    page = make_review_html("report", None, [case_id], {case_id: payload}, {case_id: {}}, validated, usage, {case_id: {"corrections": []}}, {case_id: {}}, posthoc, {case_id: {"error": None}})
    assert "full gold answer" in page and "full validated prediction" in page
    assert "Gold Answer vs Prediction" in page and "Unattributed tokens" in page and "Modality token breakdown" in page


def test_l_stage_a_source_has_no_api_execution_path():
    source = inspect.getsource(runner.run_stage_a)
    assert "genai.Client" not in source
    assert "call_with_one_technical_retry" not in source
    assert "report_reconstruction_api_calls" in source


def test_m_n_remaining_live_run_is_exactly_four_and_excludes_v01_cases():
    assert runner.V03_CASES == ("00003_2", "00006_3", "00061_5", "00002_7")
    assert set(runner.V03_CASES).isdisjoint(runner.V02_CASES)


def test_o_no_gold_or_reference_is_available_to_request_builder():
    source = inspect.getsource(build_request).casefold()
    assert "gold" not in source
    assert "weak_reference" not in source


def test_p_upstream_inputs_exist_and_are_outside_new_output_directories():
    assert all(path.is_file() for path in runner.SOURCE_FILES)
    assert runner.V01 != runner.V02 != runner.V03
