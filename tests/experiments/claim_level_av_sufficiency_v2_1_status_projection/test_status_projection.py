import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments.claim_level_av_sufficiency_v2_1_status_projection import core


def claim(cid="c1", status="supported"):
    return {
        "claim_id": cid, "claim_text": "atomic content", "claim_type": "state",
        "status": status, "support_mode": [], "evidence_ids": [], "time_range": [],
        "reasoning_summary": "r", "confidence": "medium", "answer_critical": True,
        "reusable": True, "needs_raw_visual_review": False, "review_reason": None,
        "review_target": None, "remaining_uncertainty": [], "rejected_inferences": [],
    }


def result(rows):
    return {
        "question_id": "q", "claims": rows, "required_claims": [x["claim_id"] for x in rows],
        "supported_claims": [x["claim_id"] for x in rows if x["status"] == "supported"],
        "uncertain_claims": [x["claim_id"] for x in rows if x["status"] == "uncertain"],
        "conflicted_claims": [x["claim_id"] for x in rows if x["status"] == "conflicted"],
        "missing_claims": [x["claim_id"] for x in rows if x["status"] == "not_found"],
        "answerability_from_current_index": "partial", "visual_review_requests": [],
        "remaining_uncertainty": [],
    }


def test_consistent_output_is_unchanged():
    raw = result([claim("a", "supported"), claim("b", "uncertain")])
    projected, audit = core.project_status_lists(raw)
    assert projected == raw
    assert audit["fully_consistent_input"]
    assert audit["consistent_lists_byte_equivalent"]


def test_supported_in_uncertain_is_projected_without_status_change():
    raw = result([claim()])
    raw["supported_claims"] = []
    raw["uncertain_claims"] = ["c1"]
    projected, audit = core.project_status_lists(raw)
    assert projected["claims"][0]["status"] == "supported"
    assert projected["supported_claims"] == ["c1"]
    assert projected["uncertain_claims"] == []
    assert audit["structural_conflict_count"] == 1


def test_omitted_uncertain_is_restored():
    raw = result([claim("u", "uncertain")])
    raw["uncertain_claims"] = []
    projected, _ = core.project_status_lists(raw)
    assert projected["uncertain_claims"] == ["u"]


@pytest.mark.parametrize("rows,error", [
    ([claim("x"), claim("x")], "duplicate"),
    ([claim("x", "unknown")], "unknown status"),
    ([{**claim("x"), "claim_id": ""}], "claim_id"),
])
def test_invalid_atomic_claim_identity_fails(rows, error):
    with pytest.raises(ValueError, match=error):
        core.project_status_lists(result(rows))


def test_projection_is_deterministic_and_preserves_required_and_claim_hashes():
    raw = result([claim("a", "supported"), claim("b", "not_found")])
    raw["supported_claims"], raw["missing_claims"] = [], []
    first, audit1 = core.project_status_lists(raw)
    second, audit2 = core.project_status_lists(copy.deepcopy(raw))
    assert core.canonical_bytes(first) == core.canonical_bytes(second)
    assert audit1 == audit2
    assert first["required_claims"] == raw["required_claims"]
    assert audit1["claims_sha256_before"] == audit1["claims_sha256_after"]
    assert all(x["claim_object_sha256_before"] == x["claim_object_sha256_after"] for x in audit1["claims"])


def test_raw_envelope_is_persisted_before_truncation_failure(monkeypatch):
    writes = []
    class Usage:
        input_tokens = 1
        output_tokens = 2
    class Response:
        content = [type("Block", (), {"type": "text", "text": "{"})()]
        usage = Usage()
        stop_reason = "max_tokens"
        id = "r"
    class Messages:
        def create(self, **kwargs):
            return Response()
    class Client:
        messages = Messages()
    fake = SimpleNamespace(Anthropic=lambda **kwargs: Client())
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake)
    monkeypatch.setattr(core, "dump_json", lambda path, value: writes.append((path, value)))
    with pytest.raises(RuntimeError, match="truncated"):
        core.call_and_persist_raw(api_key="x", payload={"question_id": "q"}, question_dir=Path("unused"))
    assert writes and writes[0][0].name == "sufficiency_raw_model_response.json"
    assert writes[0][1]["raw_text"] == "{"


def test_projection_contains_no_event_or_question_specific_logic():
    source = inspect.getsource(core.project_status_lists).lower()
    for forbidden in ("weapon", "medical", "handcuff", "video_id", "question_id", "timestamp"):
        assert forbidden not in source


def test_pre_sufficiency_packets_are_complete_and_audio_is_not_reselected():
    repo = Path(__file__).resolve().parents[3]
    for kind in ("uniform", "selective", "dense"):
        report = core.validate_pre_sufficiency_packet(core._input_spec(repo, kind))
        assert report["valid"], report
        assert report["audio_selector_rerun"] is False
        assert all(x["selected_audio_ids_match"] for x in report["questions"])


def test_frozen_dense_packet_hash_is_unchanged():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "outputs/experiments/claim_level_av_sufficiency_v2/226/request_payloads.json"
    assert core.sha256(path) == "759e02084f88341cb31387711040cb02560d946801e3ec0ed0ae14e3a397c5c7"
