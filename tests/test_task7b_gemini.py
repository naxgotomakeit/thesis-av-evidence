from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from src.final_qa import task7b_gemini as gemini
from src.final_qa.task7b_validation import FinalQAResult, with_uncertainty_ids


ROOT = Path(__file__).resolve().parents[1]
PAYLOADS = ROOT / "outputs/final_qa_preflight/task7a_v1/task7a_payloads.jsonl"


def payloads() -> dict[str, dict]:
    return {item["case_id"]: item for item in (json.loads(line) for line in PAYLOADS.read_text(encoding="utf-8").splitlines() if line)}


def valid_result(payload: dict) -> dict:
    inherited = with_uncertainty_ids(payload["pipeline_uncertainties"])
    return {
        "case_id": payload["case_id"], "answer_status": "answered_with_uncertainty", "answer": "pilot answer",
        "reasoning_summary": "Brief grounded summary.", "evidence_used": [],
        "inherited_pipeline_uncertainties": inherited,
        "uncertainty_assessments": [{"uncertainty_id": item["uncertainty_id"], "status": "unresolved", "assessment": "Not resolved.", "evidence_ids": []} for item in inherited],
        "final_uncertainties": [{key: value for key, value in item.items() if key != "uncertainty_id"} for item in inherited],
        "missing_information": [], "confidence": {"level": "low", "basis": "Uncertainty remains."}, "abstain": False,
    }


def test_a_b_00004_request_has_chronological_jpegs_and_one_wav():
    payload = payloads()["00004_1"]
    request, manifest, delivered = gemini.build_request(payload, ROOT, "gemini-3.5-flash", "low", True, media_loader=lambda _: b"fixture")
    images = [item for item in manifest["media"] if item["kind"] == "image"]
    audio = [item for item in manifest["media"] if item["kind"] == "audio"]
    assert [item["timestamp_sec"] for item in images] == [0.0, 0.5, 1.0, 1.5]
    assert len(audio) == 1
    assert all(item["mime_type"] == "image/jpeg" for item in images)
    assert audio[0]["mime_type"] == "audio/wav"
    assert {part["type"] for part in request["input"]} >= {"text", "image", "audio"}
    assert delivered == {"visual", "acoustic"}


def test_c_00018_request_has_audio_and_no_visual_evidence():
    payload = payloads()["00018_1"]
    request, manifest, delivered = gemini.build_request(payload, ROOT, "gemini-3.5-flash", "low", True, media_loader=lambda _: b"fixture")
    assert [item["kind"] for item in manifest["media"]] == ["audio"]
    assert not any(part["type"] == "image" for part in request["input"])
    assert delivered == {"acoustic"}


def test_d_e_store_false_and_low_thinking_are_always_explicit():
    request, manifest, _ = gemini.build_request(payloads()["00018_1"], ROOT, "gemini-3.5-flash", "low", False)
    assert request["store"] is False and manifest["store"] is False
    assert request["generation_config"] == {"thinking_level": "low"}
    assert "previous_interaction_id" not in request


def test_f_api_key_is_environment_only_and_not_part_of_manifest_or_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "never-log-this-value")
    request, manifest, _ = gemini.build_request(payloads()["00018_1"], ROOT, "gemini-3.5-flash", "low", False)
    assert gemini.gemini_key_available() is True
    serialized = json.dumps({"request": request, "manifest": manifest})
    assert "never-log-this-value" not in serialized
    assert "api_key" not in serialized


def test_g_prompt_has_no_gold_weak_reference_or_review_notes():
    request, _, _ = gemini.build_request(payloads()["00004_1"], ROOT, "gemini-3.5-flash", "low", False)
    text = json.dumps(request).casefold()
    for forbidden in ("weak_reference", "posthoc_evaluation", "ground_truth", "expected_answer", "human_review"):
        assert forbidden not in text


class FakeInteractions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def create(self, **request):
        value = self.outputs[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(output_text=value, usage=None)


def test_m_technical_retry_is_limited_to_one():
    interactions = FakeInteractions(["not json", "still not json"])
    outcome = gemini.call_with_one_technical_retry(SimpleNamespace(interactions=interactions), {"store": False})
    assert interactions.calls == 2
    assert outcome.infrastructure_succeeded is False
    assert len(outcome.attempts) == 2


def test_n_semantic_dissatisfaction_never_triggers_retry():
    payload = payloads()["00018_1"]
    raw = json.dumps(valid_result(payload))
    interactions = FakeInteractions([raw])
    outcome = gemini.call_with_one_technical_retry(SimpleNamespace(interactions=interactions), {"store": False})
    assert interactions.calls == 1
    assert outcome.infrastructure_succeeded is True
    assert outcome.parsed.confidence.level == "low"


def test_o_attempt_accounting_exposes_retry_count_without_hidden_calls():
    payload = payloads()["00018_1"]
    interactions = FakeInteractions([ValueError("empty_structured_output"), json.dumps(valid_result(payload))])
    outcome = gemini.call_with_one_technical_retry(SimpleNamespace(interactions=interactions), {"store": False})
    assert len(outcome.attempts) == interactions.calls == 2
    assert outcome.attempts[0]["success"] is False
    assert outcome.attempts[1]["success"] is True


def test_no_model_hidden_thought_summary_or_secret_logging_code():
    source = inspect.getsource(gemini).casefold()
    assert "thinking_summaries" not in source
    request, manifest, _ = gemini.build_request(payloads()["00018_1"], ROOT, "gemini-3.5-flash", "low", False)
    assert "previous_interaction_id" not in request
    assert manifest["previous_interaction_id_used"] is False
