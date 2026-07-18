from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google import genai  # noqa: E402

from src.final_qa.task7a_preflight import payload_leakage_audit  # noqa: E402
from src.final_qa.task7b_gemini import (  # noqa: E402
    build_request,
    call_with_one_technical_retry,
    gemini_key_available,
    usage_metadata,
)
from src.final_qa.task7b_validation import validate_and_correct  # noqa: E402


TASK7A = ROOT / "outputs/final_qa_preflight/task7a_v1"
PAYLOADS = TASK7A / "task7a_payloads.jsonl"
PREFLIGHT = TASK7A / "task7a_preflight_results.jsonl"
OUTPUT_SCHEMA = TASK7A / "task7_output_schema.json"
UNCERTAINTY_POLICY = TASK7A / "task7_uncertainty_policy.json"
TASK6_V12 = ROOT / "outputs/relation_reranking/task6_v1_2/task6_evidence_packets.jsonl"
OUT = ROOT / "outputs/final_qa/task7b_v0_1"
PILOT_CASES = ("00004_1", "00018_1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def payload_metrics(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    images = [item for item in manifest["media"] if item["kind"] == "image"]
    audio = [item for item in manifest["media"] if item["kind"] == "audio"]
    speech = [item for group in payload["evidence_groups"] for item in group["speech_evidence"]]
    return {
        "visual_frame_count": len(images),
        "visual_asset_bytes": sum(item["byte_count"] for item in images),
        "acoustic_clip_count": len(audio),
        "acoustic_duration_sec": round(sum(float(item["end_sec"]) - float(item["start_sec"]) for item in audio), 6),
        "acoustic_asset_bytes": sum(item["byte_count"] for item in audio),
        "speech_segment_count": len(speech),
        "transcript_character_count": sum(len(item.get("transcript", "")) for item in speech),
        "evidence_group_count": len(payload["evidence_groups"]),
        "uncertainty_count": len(payload["pipeline_uncertainties"]),
    }


def save_state(state: dict[str, list[dict[str, Any]]]) -> None:
    write_jsonl(OUT / "task7b_results.jsonl", state["results"])
    write_jsonl(OUT / "task7b_raw_model_outputs.jsonl", state["raw"])
    write_jsonl(OUT / "task7b_validated_outputs.jsonl", state["validated"])
    write_jsonl(OUT / "task7b_request_manifests.jsonl", state["manifests"])
    write_jsonl(OUT / "task7b_usage.jsonl", state["usage"])
    write_jsonl(OUT / "task7b_validator_report.jsonl", state["validator"])


def make_html(payload_by_case: dict[str, dict[str, Any]], state: dict[str, list[dict[str, Any]]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    raw = {item["case_id"]: item for item in state["raw"]}
    valid = {item["case_id"]: item for item in state["validated"]}
    usage = {item["case_id"]: item for item in state["usage"]}
    reports = {item["case_id"]: item for item in state["validator"]}
    results = {item["case_id"]: item for item in state["results"]}
    cards = []
    for case_id in summary["requested_cases"]:
        payload = payload_by_case[case_id]
        cards.append(f'''<article><h2>{case_id}</h2><h3>问题</h3><p>{html.escape(payload["question"])}</p>
<h3>精确提供的 evidence</h3>{pre(payload["evidence_groups"])}
<h3>图像时间戳与路径、Audio 区间与路径</h3>{pre(next((x for x in state["manifests"] if x["case_id"] == case_id), {}))}
<h3>继承 pipeline uncertainty</h3>{pre(payload["pipeline_uncertainties"])}
<h3>Raw model output</h3>{pre(raw.get(case_id, {}))}
<h3>Uncertainty assessments 与 final uncertainties</h3>{pre({"uncertainty_assessments": valid.get(case_id, {}).get("uncertainty_assessments"), "final_uncertainties": valid.get(case_id, {}).get("final_uncertainties")})}
<h3>验证后的 answer/status 与 evidence citations</h3>{pre(valid.get(case_id, {}))}
<h3>Validator corrections 与 leakage audit</h3>{pre(reports.get(case_id, {}))}
<h3>API calls、retries、tokens、bytes 与 latency</h3>{pre(usage.get(case_id, {}))}
<h3>Errors</h3>{pre(results.get(case_id, {}).get("error"))}</article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 7B v0.1 grounded QA pilot</title><style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:580px;overflow:auto}}</style><h1>Task 7B v0.1：两案 grounded multimodal final-QA pilot</h1><p>仅运行 00004_1 与 00018_1；未检索新证据，也未执行其余四案或 benchmark evaluation。</p>{pre(summary)}{''.join(cards)}'''


def build_summary(args: argparse.Namespace, state: dict[str, list[dict[str, Any]]], hashes: dict[str, Any], stopped_after_first: bool) -> dict[str, Any]:
    usage = state["usage"]
    return {
        "task": "Task 7B v0.1 two-case grounded multimodal final-QA pilot",
        "execution_mode": "live" if args.execute_live else "dry_run",
        "requested_cases": list(args.cases),
        "remaining_cases_not_run": [case for case in ("00002_7", "00003_2", "00006_3", "00061_5") if case not in args.cases],
        "model": args.model,
        "thinking_level": args.thinking_level,
        "store": False,
        "previous_interaction_id_used": False,
        "case_results": state["results"],
        "aggregate": {
            "completed_case_count": sum(item.get("infrastructure_succeeded", False) for item in state["results"]),
            "initial_api_calls": sum(item.get("initial_api_calls", 0) for item in usage),
            "retry_api_calls": sum(item.get("retry_api_calls", 0) for item in usage),
            "total_api_calls": sum(item.get("total_api_calls", 0) for item in usage),
            "input_tokens": sum(item.get("provider_input_token_count") or 0 for item in usage) or None,
            "output_tokens": sum(item.get("provider_output_token_count") or 0 for item in usage) or None,
            "total_tokens": sum(item.get("provider_total_token_count") or 0 for item in usage) or None,
            "total_latency_sec": round(sum(item.get("total_latency_sec", 0) for item in usage), 6),
            "stopped_after_first_case_infrastructure_failure": stopped_after_first,
            "estimated_cost": None,
        },
        "leakage_audits_passed": all(item.get("leakage_check_passed", False) for item in state["validator"]) if state["validator"] else True,
        "source_integrity": hashes,
        "model_api_calls_outside_requested_cases": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--thinking-level", default="low")
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args()
    if tuple(args.cases) != PILOT_CASES:
        raise SystemExit(f"Task 7B v0.1 must run exactly in order: {' '.join(PILOT_CASES)}")
    if args.thinking_level != "low":
        raise SystemExit("Task 7B v0.1 requires --thinking-level low")
    required_inputs = [PAYLOADS, PREFLIGHT, OUTPUT_SCHEMA, UNCERTAINTY_POLICY, TASK6_V12]
    for path in required_inputs:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    if args.execute_live and not gemini_key_available():
        raise SystemExit("GEMINI_API_KEY is unavailable; no API call was made")

    before_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in required_inputs}
    payload_by_case = {item["case_id"]: item for item in read_jsonl(PAYLOADS)}
    preflight_by_case = {item["case_id"]: item for item in read_jsonl(PREFLIGHT)}
    for case_id in args.cases:
        if case_id not in payload_by_case or preflight_by_case.get(case_id, {}).get("status") == "blocked":
            raise SystemExit(f"Case is unavailable or blocked by Task 7A: {case_id}")

    state: dict[str, list[dict[str, Any]]] = {key: [] for key in ("results", "raw", "validated", "manifests", "usage", "validator")}
    stopped_after_first = False
    client = None
    try:
        if args.execute_live:
            # The SDK reads GEMINI_API_KEY from the process environment.
            client = genai.Client()
        for index, case_id in enumerate(args.cases):
            payload = payload_by_case[case_id]
            request, manifest, delivered = build_request(payload, ROOT, args.model, args.thinking_level, include_media_data=args.execute_live)
            leakage = payload_leakage_audit(payload)
            manifest["task7a_preflight_status"] = preflight_by_case[case_id]["status"]
            manifest["leakage_check_passed"] = leakage["leakage_check_passed"]
            state["manifests"].append(manifest)
            save_state(state)  # sanitized manifest is persisted before the request
            metrics = payload_metrics(payload, manifest)
            if not args.execute_live:
                usage = {"case_id": case_id, "initial_api_calls": 0, "retry_api_calls": 0, "total_api_calls": 0, "model": args.model, "thinking_level": args.thinking_level, "request_latency_sec": 0.0, "retry_latency_sec": 0.0, "total_latency_sec": 0.0, **metrics, "provider_input_token_count": None, "provider_output_token_count": None, "provider_total_token_count": None, "input_tokens_by_modality": None, "output_tokens_by_modality": None, "schema_valid_on_first_call": None, "local_validation_corrections": [], "estimated_cost": None, "warnings": ["dry_run_no_api_call"]}
                state["usage"].append(usage)
                state["results"].append({"case_id": case_id, "status": "dry_run_ready", "infrastructure_succeeded": True, "answer_status": None, "answer": None, "error": None})
                state["validator"].append({"case_id": case_id, **leakage, "validation_status": "manifest_only"})
                save_state(state)
                continue

            outcome = call_with_one_technical_retry(client, request)
            attempts = outcome.attempts
            usage_values, usage_warnings = usage_metadata(outcome.interaction) if outcome.interaction is not None else ({"provider_input_token_count": None, "provider_output_token_count": None, "provider_total_token_count": None, "input_tokens_by_modality": None, "output_tokens_by_modality": None}, ["usage_unavailable_no_successful_interaction"])
            usage = {
                "case_id": case_id, "initial_api_calls": 1 if attempts else 0, "retry_api_calls": max(0, len(attempts) - 1), "total_api_calls": len(attempts),
                "model": args.model, "thinking_level": args.thinking_level,
                "request_latency_sec": attempts[0]["latency_sec"] if attempts else 0.0,
                "retry_latency_sec": sum(item["latency_sec"] for item in attempts[1:]), "total_latency_sec": sum(item["latency_sec"] for item in attempts),
                **metrics, **usage_values, "attempts": attempts, "schema_valid_on_first_call": bool(attempts and attempts[0]["success"]),
                "local_validation_corrections": [], "estimated_cost": None, "warnings": usage_warnings,
            }
            state["usage"].append(usage)
            state["raw"].append({"case_id": case_id, "attempt_outputs": outcome.raw_attempt_outputs, "raw_model_text": outcome.raw_text})
            if not outcome.infrastructure_succeeded or outcome.parsed is None:
                state["results"].append({"case_id": case_id, "status": "infrastructure_failed", "infrastructure_succeeded": False, "answer_status": None, "answer": None, "error": outcome.error})
                state["validator"].append({"case_id": case_id, **leakage, "validation_status": "not_run_due_to_infrastructure_failure", "error": outcome.error})
                save_state(state)
                if index == 0:
                    stopped_after_first = True
                    break
                continue
            validated, report = validate_and_correct(outcome.parsed, payload, delivered)
            report.update(leakage)
            usage["local_validation_corrections"] = copy.deepcopy(report["corrections"])
            state["validated"].append(validated)
            state["validator"].append(report)
            state["results"].append({"case_id": case_id, "status": "completed", "infrastructure_succeeded": True, "answer_status": validated["answer_status"], "answer": validated["answer"], "error": None})
            save_state(state)  # each case is fully saved before continuing
    finally:
        if client is not None:
            client.close()

    after_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in required_inputs}
    if before_hashes != after_hashes:
        raise RuntimeError("Task 7A or Task 6 input changed during Task 7B")
    hashes = {"before": before_hashes, "after": after_hashes, "unchanged": True}
    summary = build_summary(args, state, hashes, stopped_after_first)
    write_json(OUT / "task7b_summary.json", summary)
    lines = ["# Task 7B v0.1 summary", "", f"- Mode: {summary['execution_mode']}", f"- Cases: {', '.join(summary['requested_cases'])}", f"- Model / thinking / store: {args.model} / {args.thinking_level} / false", f"- Calls initial/retry/total: {summary['aggregate']['initial_api_calls']} / {summary['aggregate']['retry_api_calls']} / {summary['aggregate']['total_api_calls']}", f"- Tokens input/output/total: {summary['aggregate']['input_tokens']} / {summary['aggregate']['output_tokens']} / {summary['aggregate']['total_tokens']}", f"- Total latency: {summary['aggregate']['total_latency_sec']:.3f}s", f"- Leakage audits passed: {summary['leakage_audits_passed']}", f"- Upstream hashes unchanged: {hashes['unchanged']}"]
    for item in state["results"]:
        lines.append(f"- {item['case_id']}: {item['status']}; answer_status={item['answer_status']}; answer={item['answer']}")
    (OUT / "task7b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "task7b_human_review.html").write_text(make_html(payload_by_case, state, summary), encoding="utf-8")
    print(json.dumps({"execution_mode": summary["execution_mode"], "case_results": summary["case_results"], "aggregate": summary["aggregate"]}, ensure_ascii=False, indent=2))
    return 0 if not stopped_after_first else 2


if __name__ == "__main__":
    raise SystemExit(main())
