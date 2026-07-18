from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts.canonical import run_case
from src.canonical_pipeline.runner import CanonicalOnlineRunner
from src.canonical_pipeline.state import CaseState, ExecutionMode
from src.canonical_pipeline.sufficiency import run_evidence_sufficiency
from src.canonical_pipeline.versions import load_canonical_config
from src.final_qa.task7a_preflight import payload_leakage_audit


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_canonical_config(ROOT)


def runner() -> CanonicalOnlineRunner:
    return CanonicalOnlineRunner(CONFIG)


def test_1_canonical_version_map_is_explicit():
    assert CONFIG.versions == {"task5a": "v2", "task5b": "v1.1", "task5c": "v1.2", "task6": "v1.2", "task7a": "v1", "task7b": "v3"}


def test_2_version_selection_never_uses_mtime_or_filename_order():
    import src.canonical_pipeline.versions as versions
    source = inspect.getsource(versions).casefold()
    assert "getmtime" not in source and "stat().st_mtime" not in source
    assert "canonical_versions" in source


def test_3_task5c_fallback_executes_at_most_once():
    state = runner().run_case("00061_5")
    assert state.fallback_execution_count == 1
    frozen = next(json.loads(line) for line in CONFIG.path("task5c_frozen").read_text(encoding="utf-8").splitlines() if json.loads(line)["case_id"] == "00061_5")
    with pytest.raises(RuntimeError, match="more than once"):
        run_evidence_sufficiency(state, frozen_v1_2=frozen)


def test_4_task6_historical_versions_are_not_runtime_stages():
    state = runner().run_case("00006_3")
    stages = [item["stage"] for item in state.audit]
    assert "task6_v1" not in stages and "task6_v1_1" not in stages and "task6_v1_2_patch" not in stages
    assert stages.count("relation_reranking") == 1


def test_5_offline_indexes_are_reused_not_rebuilt():
    state = runner().run_case("00006_3")
    retrieval = next(item for item in state.audit if item["stage"] == "planner_guided_retrieval")
    assert retrieval["offline_indexes_reused"] is True
    assert all(item["skip_reason"] == "not_measured_existing_artifact" for item in state.timings if item["stage_name"] in {"visual_embedding", "global_asr", "acoustic_embedding_clap"})


def test_6_state_flows_through_all_deterministic_canonical_stages():
    state = runner().run_case("00006_3")
    assert state.planner_output and state.retrieval_result and state.sufficiency_result
    assert state.evidence_packet and state.final_payload and state.preflight_result


def test_7_regression_mode_makes_zero_external_calls():
    for case_id in ("00006_3", "00061_5"):
        assert sum(runner().run_case(case_id).external_calls.values()) == 0


def test_8_live_mode_is_explicitly_gated_by_cli_flag():
    source = inspect.getsource(run_case.main)
    assert "--execute-live" in source
    assert "Live execution is gated" in source


def test_9_normal_path_does_not_invent_fallback():
    state = runner().run_case("00006_3")
    assert state.fallback_execution_count == 0
    assert state.fallback_decision["required"] is False


def test_10_count_case_structurally_supports_one_fallback():
    state = runner().run_case("00061_5")
    assert state.fallback_execution_count == 1
    assert state.fallback_result["canonical_execution"] == "frozen_fallback_evidence_injected_once"
    assert [item["candidate_id"] for item in state.evidence_packet["retained_candidates"]] == ["task5c_local_asr_38c2885b94bc"]


def test_11_task5b_v1_1_corrected_behavior_is_used():
    state = runner().run_case("00006_3")
    assert state.retrieval_result["task5b_correction_version"] == "v1.1"
    assert "visual_evidence_accounting" in state.retrieval_result


def test_12_task6_v1_2_response_direction_and_accounting_are_preserved():
    packet = runner().run_case("00006_3").evidence_packet
    trigger = "speech_831130136e4a"
    responses = {"speech_22af56471fec", "speech_c5ab1c25358d"}
    assert {(item["source_candidate_id"], item["target_candidate_id"]) for item in packet["relations"] if item["relation_type"] == "response_to"} == {(response, trigger) for response in responses}
    assert packet["budget_accounting"]["dropped_candidate_count"] == len(packet["actually_dropped_candidates"])


def test_13_task7a_leakage_rules_are_preserved():
    state = runner().run_case("00061_5")
    assert payload_leakage_audit(state.final_payload)["leakage_check_passed"] is True
    assert state.preflight_result["leakage_check_passed"] is True


def test_14_task7b_logical_version_is_v3():
    assert CONFIG.versions["task7b"] == "v3"
    assert CONFIG.raw["historical_disk_mapping"]["task7b_v3"].endswith("task7b_v0_3")


def test_15_gold_reference_never_enters_online_state():
    state = runner().run_case("00006_3")
    serialized = json.dumps(state.__dict__, ensure_ascii=False).casefold()
    for forbidden in ("gold_dataset_answer", "provided_timestamp", "weak_reference", "expected_answer"):
        assert forbidden not in serialized


def test_16_frozen_outputs_remain_hash_identical_during_replay():
    paths = [CONFIG.path(name) for name in ("planner_plans", "task5b_frozen", "task5c_frozen", "task6_frozen", "task7a_frozen")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    runner().run_case("00006_3")
    runner().run_case("00061_5")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after


def test_canonical_source_has_no_machine_specific_absolute_paths():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/canonical_pipeline").glob("*.py"))
    assert "C:\\Users\\" not in source and "D:\\ThesisData" not in source

