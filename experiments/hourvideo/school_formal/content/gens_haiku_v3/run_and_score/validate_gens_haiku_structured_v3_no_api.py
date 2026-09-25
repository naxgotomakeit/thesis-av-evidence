#!/usr/bin/env python3
"""Deterministic mock/no-API validation for GenS structured-answer v3."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direct_api_v1.anthropic_provider import direct_action_tools  # noqa: E402
from gens_haiku_eval300.runtime import GensAnsweringStore, read_jsonl  # noqa: E402
from gens_haiku_eval300.structured_v2 import (  # noqa: E402
    CampaignBudgetLedger,
    StructuredProviderError,
    StructuredProviderOutcome,
)
from gens_haiku_eval300.structured_v3 import (  # noqa: E402
    StructuredV3Config,
    StructuredV3RouteRunner,
    build_structured_request,
    parse_structured_response,
    tool_schema,
)

DEFAULT_CONFIG = ROOT / "config/gens_haiku_structured_v3_direct_parser_aligned.json"


def tool(option="A", reason="The frames support option A.", *, name="final_answer", extra=False):
    value = {"selected_option_id": option, "reason": reason}
    if extra:
        value["extra"] = 1
    return {"type": "tool_use", "id": "tool-1", "name": name, "input": value}


def outcome(content, *, tokens=10):
    return StructuredProviderOutcome(
        status="response_received", raw_text=None, content=content,
        raw_response={"id": "m", "content": content}, input_tokens=tokens,
        output_tokens=2, response_id="m", response_model="claude-haiku-4-5-20251001",
        stop_reason="tool_use", latency_sec=0.01,
    )


class FakeProvider:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def call(self, payload):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = StructuredV3Config.load(config_path)
    rows = [json.loads(line) for line in Path(config.selector_path).read_text(encoding="utf-8").splitlines() if line]
    row = next(item for item in rows if item["selected_frame_count"] == 1)
    checks: list[dict] = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    check("schema exactly Direct final_answer", [tool_schema()] == direct_action_tools(0))
    check("only final_answer exposed", tool_schema()["name"] == "final_answer")
    check("tool choice any", config.tool_choice == {"type": "any"})
    schema = tool_schema()["input_schema"]
    check("required exact", schema["required"] == ["selected_option_id", "reason"])
    check("no extra properties", schema["additionalProperties"] is False)
    check("A-E enum", schema["properties"]["selected_option_id"]["enum"] == list("ABCDE"))
    check("reason bounds", schema["properties"]["reason"] == {"type": "string", "minLength": 1, "maxLength": 240})
    for option in "ABCDE":
        parsed = parse_structured_response([tool(option, f"Evidence supports {option}.")])
        check(f"valid {option}", parsed["result_class"] == "valid_answer" and parsed["prediction"] == option)
    cases = [
        ("null rejected", [tool(None)], "invalid_option"),
        ("invalid option", [tool("F")], "invalid_option"),
        ("empty reason", [tool("A", "")], "invalid_reason"),
        ("blank reason", [tool("A", "   ")], "invalid_reason"),
        ("missing fields", [{"type": "tool_use", "id": "x", "name": "final_answer", "input": {}}], "invalid_reason"),
        ("wrong tool", [tool("A", name="submit_answer")], "wrong_tool_name"),
        ("no tool", [{"type": "text", "text": "A"}], "missing_tool_call"),
        ("multiple tools", [tool("A"), tool("B")], "multiple_tool_calls"),
        ("tool plus text", [tool("A"), {"type": "text", "text": "A"}], "tool_plus_text_or_other_content"),
    ]
    for name, content, category in cases:
        parsed = parse_structured_response(content)
        check(name, parsed["result_class"] == "invalid_format" and parsed["prediction"] is None and parsed["failure_category"] == category)
    long_reason = "x" * 283
    long_parsed = parse_structured_response([tool("D", long_reason, extra=True)])
    check("Direct-aligned long reason accepted unchanged", long_parsed["prediction"] == "D" and long_parsed["reason"] == long_reason)
    converted = parse_structured_response([tool("A", 123)])
    check("Direct-aligned reason str conversion", converted["prediction"] == "A" and converted["reason"] == "123")
    payload, frames = build_structured_request(row, config, encode_images=False)
    check("request image count", len(frames) == 1 and len(payload["messages"][0]["content"]) == 2)
    check("images before question", payload["messages"][0]["content"][0]["type"] == "image" and payload["messages"][0]["content"][-1]["type"] == "text")
    check("no navigation tools", payload["tools"] == [tool_schema()])
    check("system prompt exact", payload["system"] == config.system_prompt)
    check("cache disabled", config.cache_policy.startswith("disabled_"))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        campaign = CampaignBudgetLedger(root / "campaign.json", 5.0)
        campaign.initialise()

        def execute(label, values):
            store = GensAnsweringStore(root / label, config.experiment_id, 5.0)
            store.initialise("fp")
            provider = FakeProvider(values)
            result = StructuredV3RouteRunner(config, store, campaign, lambda: provider).run_one(row, run_fingerprint="fp")
            return store, provider, result

        store, provider, result = execute("valid", [outcome([tool("D", "The frames support D.")])])
        check("valid terminal", result["prediction"] == "D" and store.status(row["qa_uid"])["state"] == "terminal_success")
        check("journals paired", len(read_jsonl(store.root / "journals/request_start.jsonl")) == len(read_jsonl(store.root / "journals/attempt_end.jsonl")) == 1)
        check("ledger charged", json.loads(store.ledger_path.read_text())["spent_usd"] > 0)
        again = StructuredV3RouteRunner(config, store, campaign, lambda: provider).run_one(row, run_fingerprint="fp")
        check("terminal skip", again["skipped_terminal"] and provider.calls == 1)
        invalid_store, invalid_provider, invalid = execute("invalid", [outcome([tool(None)])])
        check("invalid no semantic retry", invalid["result_class"] == "invalid_format" and invalid_provider.calls == 1 and invalid_store.status(row["qa_uid"])["state"] == "terminal_failed")
        retry_store, retry_provider, retry = execute("retry", [StructuredProviderError("timeout"), outcome([tool("E", "The frames support E.")])])
        check("one transport retry", retry["prediction"] == "E" and retry_provider.calls == 2 and retry["resource_totals"]["provider_attempts"] == 2)
        check("retry durable", len(read_jsonl(retry_store.root / "journals/attempt_end.jsonl")) == 2)
        interrupted = GensAnsweringStore(root / "interrupted", config.experiment_id, 5.0)
        interrupted.initialise("fp")
        interrupted.start(row["qa_uid"], "input")
        interrupted.append("request_start", {"question_id": row["qa_uid"], "attempt_id": "unknown", "retry_index": 0})
        check("unknown request conservative", interrupted.reconcile_interrupted([row["qa_uid"]]) == 1 and interrupted.status(row["qa_uid"])["state"] == "terminal_failed")
        tiny = CampaignBudgetLedger(root / "tiny.json", 0.1)
        tiny.initialise()
        tiny.charge_once("prior", 0.1, "prior")
        budget_store = GensAnsweringStore(root / "budget", config.experiment_id, 5.0)
        budget_store.initialise("fp")
        budget_provider = FakeProvider([outcome([tool("A")])])
        budget = StructuredV3RouteRunner(config, budget_store, tiny, lambda: budget_provider).run_one(row, run_fingerprint="fp")
        check("budget before provider", budget["failure_category"] == "budget_guard_refused_no_request" and budget_provider.calls == 0)

    print(json.dumps({
        "status": "PASS",
        "checks": len(checks),
        "details": checks,
        "api_calls": 0,
        "model_calls": 0,
        "gold_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
