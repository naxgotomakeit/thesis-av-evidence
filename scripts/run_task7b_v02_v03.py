from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google import genai  # noqa: E402

from src.final_qa.task7a_preflight import payload_leakage_audit  # noqa: E402
from src.final_qa.task7b_gemini import build_request, call_with_one_technical_retry, gemini_key_available, usage_metadata  # noqa: E402
from src.final_qa.task7b_reporting import guarded_load_gold, make_review_html, read_jsonl, sha256, write_json, write_jsonl  # noqa: E402
from src.final_qa.task7b_v02 import (  # noqa: E402
    FinalQAModelOutputV02,
    active_pipeline_uncertainties,
    posthoc_answer_record,
    reconstruct_v01_validated,
    token_accounting,
    validate_v02,
)


TASK7A = ROOT / "outputs/final_qa_preflight/task7a_v1"
PAYLOADS = TASK7A / "task7a_payloads.jsonl"
PREFLIGHT = TASK7A / "task7a_preflight_results.jsonl"
TASK6 = ROOT / "outputs/relation_reranking/task6_v1_2/task6_evidence_packets.jsonl"
V01 = ROOT / "outputs/final_qa/task7b_v0_1"
V02 = ROOT / "outputs/final_qa/task7b_v0_2"
V03 = ROOT / "outputs/final_qa/task7b_v0_3"
GOLD = ROOT / "data/manifests/mvp_cases_6.json"
V02_CASES = ("00004_1", "00018_1")
V03_CASES = ("00003_2", "00006_3", "00061_5", "00002_7")
SOURCE_FILES = [PAYLOADS, PREFLIGHT, TASK6]


def source_hashes(extra: list[Path] | None = None) -> dict[str, str]:
    paths = SOURCE_FILES + (extra or [])
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}


def by_case(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in read_jsonl(path)}


def payload_metrics(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    images = [item for item in manifest["media"] if item["kind"] == "image"]
    audio = [item for item in manifest["media"] if item["kind"] == "audio"]
    speech = [item for group in payload["evidence_groups"] for item in group["speech_evidence"]]
    return {
        "visual_frame_count": len(images), "visual_asset_bytes": sum(item["byte_count"] for item in images),
        "acoustic_clip_count": len(audio), "acoustic_duration_sec": round(sum(float(item["end_sec"]) - float(item["start_sec"]) for item in audio), 6),
        "acoustic_asset_bytes": sum(item["byte_count"] for item in audio), "speech_segment_count": len(speech),
        "transcript_character_count": sum(len(item.get("transcript", "")) for item in speech),
        "evidence_group_count": len(payload["evidence_groups"]), "uncertainty_count": len(active_pipeline_uncertainties(payload)),
    }


def save_core(out: Path, state: dict[str, list[dict[str, Any]]]) -> None:
    for key, filename in (
        ("results", "task7b_results.jsonl"), ("raw", "task7b_raw_model_outputs.jsonl"),
        ("validated", "task7b_validated_outputs.jsonl"), ("manifests", "task7b_request_manifests.jsonl"),
        ("usage", "task7b_usage.jsonl"), ("validator", "task7b_validator_report.jsonl"),
        ("model_payloads", "task7b_model_payloads.jsonl"),
    ):
        write_jsonl(out / filename, state[key])


def raw_answer(row: dict[str, Any]) -> str | None:
    text = row.get("raw_model_text")
    if not text:
        return None
    try:
        return json.loads(text).get("answer")
    except (json.JSONDecodeError, AttributeError):
        return None


def finalize_reports(
    out: Path,
    task_name: str,
    cases: tuple[str, ...],
    payloads: dict[str, dict[str, Any]],
    state: dict[str, list[dict[str, Any]]],
    before_hashes: dict[str, str],
    reconstruction_note: str | None,
    execution_mode: str,
) -> None:
    # This is deliberately after raw and validated files have been persisted.
    gold = guarded_load_gold(GOLD, out / "task7b_raw_model_outputs.jsonl", out / "task7b_validated_outputs.jsonl", cases)
    raw_by = {row["case_id"]: row for row in state["raw"]}
    valid_by = {row["case_id"]: row for row in state["validated"]}
    posthoc = [posthoc_answer_record(case_id, payloads[case_id]["question"], gold[case_id], raw_answer(raw_by[case_id]), valid_by[case_id]) for case_id in cases]
    write_jsonl(out / "task7b_posthoc_evaluation.jsonl", posthoc)
    after_hashes = source_hashes([path for path in V01.glob("*") if path.is_file()] if "v0.2" in task_name else None)
    if before_hashes != after_hashes:
        changed = sorted(key for key in set(before_hashes) | set(after_hashes) if before_hashes.get(key) != after_hashes.get(key))
        raise RuntimeError(f"upstream_inputs_changed_during_task7b:{changed}")
    usage = state["usage"]
    summary = {
        "task": task_name, "execution_mode": execution_mode, "cases": list(cases), "model": "gemini-3.5-flash",
        "thinking_level": "low", "store": False, "dataset_answers_loaded_stage": "post_hoc_after_raw_and_validated_outputs_saved",
        "aggregate": {
            "case_count": len(cases), "initial_api_calls": sum(row.get("initial_api_calls", 0) for row in usage),
            "retry_api_calls": sum(row.get("retry_api_calls", 0) for row in usage), "total_api_calls": sum(row.get("total_api_calls", 0) for row in usage),
            "input_tokens": sum(row.get("input_tokens") or 0 for row in usage), "output_tokens": sum(row.get("output_tokens") or 0 for row in usage),
            "unattributed_tokens": sum(row.get("unattributed_tokens") or 0 for row in usage), "total_tokens": sum(row.get("total_tokens") or 0 for row in usage),
            "total_latency_sec": round(sum(float(row.get("total_latency_sec", 0)) for row in usage), 6),
            "estimated_cost": None,
        },
        "api_calls_during_this_stage": 0 if execution_mode == "zero_api_reconstruction" else sum(row.get("total_api_calls", 0) for row in usage),
        "historical_api_calls_preserved_for_reporting": sum(row.get("total_api_calls", 0) for row in usage) if execution_mode == "zero_api_reconstruction" else None,
        "case_results": state["results"], "leakage_audits_passed": all(row.get("leakage_check_passed", False) for row in state["validator"]),
        "source_integrity": {"before": before_hashes, "after": after_hashes, "unchanged": True},
        "reconstruction_note": reconstruction_note,
    }
    write_json(out / "task7b_summary.json", summary)
    lines = [f"# {task_name}", "", f"- Execution mode: {execution_mode}", f"- Cases: {', '.join(cases)}", f"- API calls during this stage: {summary['api_calls_during_this_stage']}", f"- Historical API calls initial/retry/total: {summary['aggregate']['initial_api_calls']} / {summary['aggregate']['retry_api_calls']} / {summary['aggregate']['total_api_calls']}", f"- Tokens input/output/unattributed/total: {summary['aggregate']['input_tokens']} / {summary['aggregate']['output_tokens']} / {summary['aggregate']['unattributed_tokens']} / {summary['aggregate']['total_tokens']}", f"- Total latency: {summary['aggregate']['total_latency_sec']:.3f}s", "- Gold/reference answers were loaded post-hoc only.", "- Automatic correctness: not_determined for open-ended QA."]
    if reconstruction_note:
        lines.append(f"- {reconstruction_note}")
    for row in state["results"]:
        lines.append(f"- {row['case_id']}: {row['answer_status']}; {row['answer']}")
    (out / "task7b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    maps = {key: {row["case_id"]: row for row in state[key]} for key in ("raw", "validated", "usage", "validator", "manifests", "results")}
    posthoc_by = {row["case_id"]: row for row in posthoc}
    (out / "task7b_human_review.html").write_text(make_review_html(task_name, reconstruction_note, list(cases), payloads, maps["raw"], maps["validated"], maps["usage"], maps["validator"], maps["manifests"], posthoc_by, maps["results"]), encoding="utf-8")


def run_stage_a() -> None:
    required_v01 = [V01 / name for name in ("task7b_results.jsonl", "task7b_raw_model_outputs.jsonl", "task7b_validated_outputs.jsonl", "task7b_request_manifests.jsonl", "task7b_usage.jsonl", "task7b_validator_report.jsonl")]
    for path in SOURCE_FILES + required_v01:
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")
    before = source_hashes([path for path in V01.glob("*") if path.is_file()])
    payloads = by_case(PAYLOADS)
    old = {name: by_case(V01 / filename) for name, filename in (
        ("results", "task7b_results.jsonl"), ("raw", "task7b_raw_model_outputs.jsonl"), ("validated", "task7b_validated_outputs.jsonl"),
        ("manifests", "task7b_request_manifests.jsonl"), ("usage", "task7b_usage.jsonl"), ("validator", "task7b_validator_report.jsonl"),
    )}
    state = {key: [] for key in ("results", "raw", "validated", "manifests", "usage", "validator", "model_payloads")}
    for case_id in V02_CASES:
        validated, report = reconstruct_v01_validated(old["validated"][case_id], payloads[case_id])
        leakage = payload_leakage_audit(payloads[case_id])
        report.update(leakage)
        usage = copy.deepcopy(old["usage"][case_id])
        usage.update(token_accounting(usage))
        usage["report_reconstruction_api_calls"] = 0
        usage["historical_calls_preserved_from_v0_1"] = True
        usage["local_validation_corrections"] = copy.deepcopy(report["corrections"])
        result = {"case_id": case_id, "status": "reconstructed_from_v0_1", "infrastructure_succeeded": True, "answer_status": validated["answer_status"], "answer": validated["answer"], "error": None}
        for key, value in (("results", result), ("raw", copy.deepcopy(old["raw"][case_id])), ("validated", validated), ("manifests", copy.deepcopy(old["manifests"][case_id])), ("usage", usage), ("validator", report), ("model_payloads", copy.deepcopy(payloads[case_id]))):
            state[key].append(value)
    save_core(V02, state)
    note = "Reconstructed from Task 7B v0.1 saved outputs; no model rerun."
    finalize_reports(V02, "Task 7B v0.2 corrected reporting", V02_CASES, payloads, state, before, note, "zero_api_reconstruction")
    print(json.dumps({"stage": "A", "cases": V02_CASES, "api_calls": 0, "output": str(V02)}, ensure_ascii=False))


def run_stage_b(execute_live: bool) -> None:
    for path in SOURCE_FILES:
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")
    if execute_live and not gemini_key_available():
        raise SystemExit("GEMINI_API_KEY is unavailable; no API call was made")
    before = source_hashes()
    payloads = by_case(PAYLOADS)
    preflight = by_case(PREFLIGHT)
    for case_id in V03_CASES:
        if case_id not in payloads or preflight.get(case_id, {}).get("status") == "blocked":
            raise SystemExit(f"Case unavailable or blocked: {case_id}")
    state = {key: [] for key in ("results", "raw", "validated", "manifests", "usage", "validator", "model_payloads")}
    client = genai.Client() if execute_live else None
    try:
        for case_id in V03_CASES:
            payload = payloads[case_id]
            active = active_pipeline_uncertainties(payload)
            request, manifest, delivered = build_request(payload, ROOT, "gemini-3.5-flash", "low", execute_live, response_model=FinalQAModelOutputV02, uncertainties_override=active)
            leakage = payload_leakage_audit(payload)
            manifest.update({"task7a_preflight_status": preflight[case_id]["status"], "leakage_check_passed": leakage["leakage_check_passed"], "dataset_gold_in_request": False})
            state["manifests"].append(manifest)
            state["model_payloads"].append({"case_id": case_id, "question": payload["question"], "operation": payload["operation"], "evidence_groups": payload["evidence_groups"], "active_pipeline_uncertainties": active, "dataset_audit": {"status": "not_evaluated", "source": "pipeline", "affects_answer_policy": False, "notes": []}, "final_answer_policy": payload["final_answer_policy"]})
            save_core(V03, state)  # sanitized payload and manifest are on disk before the call
            metrics = payload_metrics(payload, manifest)
            if not execute_live:
                usage = {"case_id": case_id, "initial_api_calls": 0, "retry_api_calls": 0, "total_api_calls": 0, "model": "gemini-3.5-flash", "thinking_level": "low", "request_latency_sec": 0.0, "retry_latency_sec": 0.0, "total_latency_sec": 0.0, **metrics, "provider_input_token_count": None, "provider_output_token_count": None, "provider_total_token_count": None, "input_tokens_by_modality": None, "output_tokens_by_modality": None, **token_accounting({}), "schema_valid_on_first_call": None, "local_validation_corrections": [], "estimated_cost": None, "warnings": ["dry_run_no_api_call"]}
                state["usage"].append(usage)
                state["results"].append({"case_id": case_id, "status": "dry_run_ready", "infrastructure_succeeded": True, "answer_status": None, "answer": None, "error": None})
                state["validator"].append({"case_id": case_id, **leakage, "validation_status": "manifest_only"})
                continue
            outcome = call_with_one_technical_retry(client, request, response_model=FinalQAModelOutputV02)
            attempts = outcome.attempts
            provider_usage, usage_warnings = usage_metadata(outcome.interaction) if outcome.interaction is not None else ({"provider_input_token_count": None, "provider_output_token_count": None, "provider_total_token_count": None, "input_tokens_by_modality": None, "output_tokens_by_modality": None}, ["usage_unavailable_no_successful_interaction"])
            usage = {"case_id": case_id, "initial_api_calls": 1 if attempts else 0, "retry_api_calls": max(0, len(attempts) - 1), "total_api_calls": len(attempts), "model": "gemini-3.5-flash", "thinking_level": "low", "request_latency_sec": attempts[0]["latency_sec"] if attempts else 0.0, "retry_latency_sec": sum(item["latency_sec"] for item in attempts[1:]), "total_latency_sec": sum(item["latency_sec"] for item in attempts), **metrics, **provider_usage, "attempts": attempts, "schema_valid_on_first_call": bool(attempts and attempts[0]["success"]), "estimated_cost": None, "warnings": usage_warnings}
            usage.update(token_accounting(usage))
            state["usage"].append(usage)
            state["raw"].append({"case_id": case_id, "attempt_outputs": outcome.raw_attempt_outputs, "raw_model_text": outcome.raw_text})
            if not outcome.infrastructure_succeeded or outcome.parsed is None:
                state["results"].append({"case_id": case_id, "status": "infrastructure_failed", "infrastructure_succeeded": False, "answer_status": None, "answer": None, "error": outcome.error})
                state["validator"].append({"case_id": case_id, **leakage, "validation_status": "not_run_due_to_infrastructure_failure", "error": outcome.error})
                save_core(V03, state)
                raise RuntimeError(f"Infrastructure failed for {case_id}; later cases were not run")
            validated, report = validate_v02(outcome.parsed, payload, delivered)
            report.update(leakage)
            usage["local_validation_corrections"] = copy.deepcopy(report["corrections"])
            state["validated"].append(validated)
            state["validator"].append(report)
            state["results"].append({"case_id": case_id, "status": "completed", "infrastructure_succeeded": True, "answer_status": validated["answer_status"], "answer": validated["answer"], "error": None})
            save_core(V03, state)  # raw and validated result saved immediately per case
    finally:
        if client is not None:
            client.close()
    if not execute_live:
        write_json(V03 / "task7b_dry_run_summary.json", {"cases": list(V03_CASES), "ready": True, "actual_api_calls": 0, "manifests": state["manifests"]})
        print(json.dumps({"stage": "B-dry-run", "cases": V03_CASES, "api_calls": 0, "ready": True}, ensure_ascii=False))
        return
    finalize_reports(V03, "Task 7B v0.3 remaining four-case grounded QA", V03_CASES, payloads, state, before, None, "live")
    print(json.dumps({"stage": "B-live", "results": state["results"], "aggregate_calls": sum(row["total_api_calls"] for row in state["usage"])}, ensure_ascii=False, indent=2))


def revalidate_saved_v03() -> None:
    """Apply local validator/reporting patches to saved raw results with zero API calls."""
    required = [V03 / name for name in ("task7b_raw_model_outputs.jsonl", "task7b_request_manifests.jsonl", "task7b_usage.jsonl")]
    if not all(path.is_file() for path in required):
        raise SystemExit("Saved v0.3 live artifacts are incomplete")
    before = source_hashes()
    payloads = by_case(PAYLOADS)
    raw_by, manifest_by, usage_by = (by_case(path) for path in required)
    state = {key: [] for key in ("results", "raw", "validated", "manifests", "usage", "validator", "model_payloads")}
    for case_id in V03_CASES:
        raw = raw_by[case_id]
        parsed = FinalQAModelOutputV02.model_validate_json(raw["raw_model_text"])
        payload = payloads[case_id]
        manifest = manifest_by[case_id]
        delivered = {item["kind"] for item in manifest["media"]}
        delivered = {"visual" if item == "image" else "acoustic" if item == "audio" else item for item in delivered}
        if any(group["speech_evidence"] for group in payload["evidence_groups"]):
            delivered.add("speech")
        validated, report = validate_v02(parsed, payload, delivered)
        leakage = payload_leakage_audit(payload)
        report.update(leakage)
        usage = copy.deepcopy(usage_by[case_id])
        usage["local_validation_corrections"] = copy.deepcopy(report["corrections"])
        state["raw"].append(copy.deepcopy(raw))
        state["validated"].append(validated)
        state["manifests"].append(copy.deepcopy(manifest))
        state["usage"].append(usage)
        state["validator"].append(report)
        state["results"].append({"case_id": case_id, "status": "completed", "infrastructure_succeeded": True, "answer_status": validated["answer_status"], "answer": validated["answer"], "error": None})
        state["model_payloads"].append({"case_id": case_id, "question": payload["question"], "operation": payload["operation"], "evidence_groups": payload["evidence_groups"], "active_pipeline_uncertainties": active_pipeline_uncertainties(payload), "dataset_audit": validated["dataset_audit"], "final_answer_policy": payload["final_answer_policy"]})
    save_core(V03, state)
    finalize_reports(V03, "Task 7B v0.3 remaining four-case grounded QA", V03_CASES, payloads, state, before, None, "live_results_locally_revalidated_zero_api")
    print(json.dumps({"stage": "B-local-revalidation", "cases": V03_CASES, "new_api_calls": 0}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--revalidate-saved", action="store_true")
    args = parser.parse_args()
    if args.stage == "a":
        if args.execute_live:
            raise SystemExit("Stage A is zero-call only")
        run_stage_a()
    else:
        if args.revalidate_saved:
            if args.execute_live:
                raise SystemExit("Saved-output revalidation is zero-call only")
            revalidate_saved_v03()
        else:
            run_stage_b(args.execute_live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
