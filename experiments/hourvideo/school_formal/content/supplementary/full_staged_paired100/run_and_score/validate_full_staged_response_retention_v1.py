#!/usr/bin/env python3
"""No-API equivalence check for provider-response retention instrumentation."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staged_api_v1.config import StagedConfig, sha256_file
from staged_api_v1.contracts import StageContractError, final_schema, tool_for
from staged_api_v1.provider import AnthropicStagedProvider
from staged_api_v1.store import StagedStore, atomic_json, read_jsonl

CONFIG = ROOT / "config/full_staged_api_r3_paired100_final_v2.json"
OUT = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
OLD_AUDIT = ROOT / "outputs/full_staged_api_audits/full_staged_api_r3_aligned_v2_pilot_validation_retry_v1/validation_retry_diagnostic.json"


class Client:
    def __init__(self, response): self.response = response
    def create(self, **payload): return self.response


class PreRetentionStore(StagedStore):
    """Counterfactual old instrumentation: no raw-response side journal."""
    def provider_response(self, row):
        return None


def response(blocks):
    return SimpleNamespace(
        id="synthetic", stop_reason="tool_use", content=blocks,
        usage=SimpleNamespace(input_tokens=10, cache_creation_input_tokens=0,
                              cache_read_input_tokens=0, output_tokens=2),
    )


def execute(store_class, native_response, root):
    cfg = StagedConfig.load(CONFIG)
    store = store_class(root, experiment_id="response-retention-test", hard_budget_usd=2, request_reserve_usd=.01)
    store.start_route("q", input_sha256="synthetic")
    provider = AnthropicStagedProvider(
        model=cfg.model, max_tokens_by_stage=cfg.max_tokens_by_stage,
        temperature=cfg.temperature, timeout_sec=1, max_transport_retries=0,
        pricing=cfg.pricing_object, store=store, client=Client(native_response),
    )
    question = {"question_id":"q", "answer_options":[{"option_id":c,"text":c} for c in "ABCDE"]}
    outcome = {"accepted": False, "payload": None, "exception": None}
    try:
        result = provider.call(question_id="q", stage="final", stage_call_index=1,
                               validation_retry_index=0, user_content=[],
                               schema=final_schema(question, 0, "best_guess"))
        outcome.update(accepted=True, payload=result.payload)
    except StageContractError as error:
        outcome["exception"] = str(error)
    outcome["attempt_end"] = read_jsonl(store.attempt_ends_path)
    outcome["controller_result"] = read_jsonl(store.controller_results_path)
    outcome["raw_response"] = read_jsonl(store.provider_responses_path)
    return outcome


def decision_projection(outcome):
    attempt = outcome["attempt_end"][0]
    controller = outcome["controller_result"][0]
    return {
        "accepted": outcome["accepted"], "payload": outcome["payload"],
        "exception": outcome["exception"],
        "attempt": {key: attempt.get(key) for key in (
            "provider_status", "provider_error_category", "ordinary_input_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens",
            "cache_aware_usd", "response_id", "response_metadata",
        )},
        "controller": {key: controller.get(key) for key in (
            "accepted", "rejection_category", "reason",
        )},
    }


def main():
    valid = {"question_id":"q","selected_option_id":"B","answer_text":"B",
             "decision_status":"best_guess","supporting_evidence_indexes":[],
             "reason":"basis","uncertainty":"gap"}
    semantic_invalid = {**valid, "selected_option_id":"Z"}
    name = tool_for("final", {})["name"]
    cases = {
        "valid": response([SimpleNamespace(type="tool_use", name=name, input=valid)]),
        "semantic_invalid_but_envelope_valid": response([SimpleNamespace(type="tool_use", name=name, input=semantic_invalid)]),
        "tool_plus_text": response([SimpleNamespace(type="tool_use", name=name, input=valid), SimpleNamespace(type="text", text="extra")]),
        "multiple_tools": response([SimpleNamespace(type="tool_use", name=name, input=valid), SimpleNamespace(type="tool_use", name=name, input=semantic_invalid)]),
        "malformed_input": response([SimpleNamespace(type="tool_use", name=name, input="bad")]),
    }
    comparisons = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, (case, native) in enumerate(cases.items()):
            before = execute(PreRetentionStore, native, root / f"before-{index}")
            after = execute(StagedStore, native, root / f"after-{index}")
            same = decision_projection(before) == decision_projection(after)
            comparisons.append({
                "case": case, "pre_post_decision_equal": same,
                "accepted_by_provider_envelope": after["accepted"],
                "parsed_payload_equal": before["payload"] == after["payload"],
                "exception": after["exception"],
                "raw_response_record_count_after": len(after["raw_response"]),
                "legacy_semantic_retry_input_unchanged": before["payload"] == after["payload"],
            })
    old = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    closure = old["source_closure"]
    historical = {
        name: {**record, "current_sha256": sha256_file(Path(record["path"])),
               "unchanged": sha256_file(Path(record["path"])) == record["sha256"]}
        for name, record in closure.items()
    }
    report = {
        "status": "PASS" if all(row["pre_post_decision_equal"] for row in comparisons) and all(row["unchanged"] for row in historical.values()) else "FAIL",
        "scope": "offline provider retention equivalence; no API/model/gold",
        "comparisons": comparisons,
        "historical_aligned_v2_pilot": {
            "critical_journals": historical,
            "modified": False,
            "unrecoverable_gap": "one historical rejected Fine response retained tool_use_count=2 but not its two individual tool payloads; they remain explicitly missing and were not fabricated",
        },
        "validation_layers": {
            "raw_provider_response": "journals/provider_responses.jsonl, durable before parsing",
            "provider_tool_envelope": "journals/controller_results.jsonl",
            "v6_6_2_semantic_validation_and_retry": "legacy_live/cases/<question_id>/model_attempts.jsonl and shared_attempt_audit.jsonl",
            "downgrade_and_terminal": "legacy_live/cases/<question_id>/r3_2/route_status.json",
        },
        "api_calls": 0, "model_calls": 0, "gold_loaded": False,
    }
    atomic_json(OUT / "response_retention_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
