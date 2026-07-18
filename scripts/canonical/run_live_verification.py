"""Run the authorized two-case canonical live verification and reports."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.canonical_pipeline.live_boundaries import (  # noqa: E402
    LazyWhisperFallback,
    anthropic_requester,
    environment_presence,
)
from src.canonical_pipeline.live_comparison import compare_live_state  # noqa: E402
from src.canonical_pipeline.runner import CanonicalOnlineRunner  # noqa: E402
from src.canonical_pipeline.state import CaseState  # noqa: E402
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402


CASES = ("00006_3", "00061_5")
OUT = ROOT / "outputs/canonical_pipeline/live_v0_1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen(config: Any, case_id: str) -> dict[str, dict[str, Any]]:
    mapping = {"planner": "planner_plans", "retrieval": "task5b_frozen", "sufficiency": "task5c_frozen", "packet": "task6_frozen", "payload": "task7a_frozen"}
    return {key: next(row for row in _jsonl(config.path(name)) if row["case_id"] == case_id) for key, name in mapping.items()}


def _timing(state: CaseState, name: str) -> float | None:
    item = next((row for row in state.timings if row["stage_name"] == name), None)
    return None if item is None else item.get("duration_sec")


def _media_metrics(state: CaseState) -> dict[str, Any]:
    manifest = (state.final_model_output or {}).get("request_manifest", {})
    media = manifest.get("media", [])
    payload = state.final_payload or {}
    return {
        "visual_frame_count": sum(item.get("kind") == "image" for item in media),
        "visual_asset_bytes": sum(int(item.get("byte_count", 0)) for item in media if item.get("kind") == "image"),
        "audio_clip_count": sum(item.get("kind") == "audio" for item in media),
        "audio_duration_sec": sum(float(item.get("end_sec", 0)) - float(item.get("start_sec", 0)) for item in media if item.get("kind") == "audio"),
        "audio_asset_bytes": sum(int(item.get("byte_count", 0)) for item in media if item.get("kind") == "audio"),
        "speech_segment_count": sum(len(group["speech_evidence"]) for group in payload.get("evidence_groups", [])),
        "evidence_group_count": len(payload.get("evidence_groups", [])),
        "relation_count": sum(len(group["relations"]) for group in payload.get("evidence_groups", [])),
    }


def _safe_state(state: CaseState) -> dict[str, Any]:
    return asdict(state)


def _report_html(summary: dict[str, Any], results: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    by_case = {row["case_id"]: row for row in results}
    def f(value: Any) -> str:
        return "—" if value is None else (f"{value:.4f}" if isinstance(value, float) else html.escape(str(value)))
    latency_rows = "".join(f"<tr><td>{cid}</td>" + "".join(f"<td>{f(by_case[cid]['timing_breakdown'].get(key))}</td>" for key in ("online_total", "planner", "retrieval", "refinement", "sufficiency", "fallback", "reranking", "payload", "gemini", "parse_validation", "overhead", "coverage")) + "</tr>" for cid in by_case)
    api_rows = "".join(f"<tr><td>{cid}</td><td>{f(row['planner_api_wall_clock_sec'])}</td><td>{f(row['fallback_model_wall_clock_sec'])}</td><td>{f(row['final_model_api_wall_clock_sec'])}</td><td>{f(row['total_api_wall_clock_sec'])}</td><td>{f(row['api_percent_online'])}</td></tr>" for cid, row in by_case.items())
    eff_rows = "".join(f"<tr><td>{cid}</td><td>{row['efficiency']['visual_frame_count']}</td><td>{f(row['efficiency']['audio_duration_sec'])}</td><td>{row['efficiency']['speech_segment_count']}</td><td>{f(row['gemini_usage'].get('input_tokens'))}</td><td>{f(row['gemini_usage'].get('total_cached_tokens'))}</td><td>{f(row['gemini_usage'].get('uncached_input_tokens'))}</td><td>{f(row['gemini_usage'].get('output_tokens'))}</td><td>{f(row['gemini_usage'].get('total_tokens'))}</td><td>{row['total_api_calls']}</td><td>{row['retries']}</td></tr>" for cid, row in by_case.items())
    cache_rows = "".join(f"<tr><td>{cid}</td><td>{f(row['gemini_usage'].get('input_tokens'))}</td><td>{f(row['gemini_usage'].get('total_cached_tokens'))}</td><td>{f(row['gemini_usage'].get('cache_hit'))}</td><td>{f(row['gemini_usage'].get('cache_hit_rate'))}</td><td>{not bool(row['gemini_usage'].get('provider_usage_warnings'))}</td></tr>" for cid, row in by_case.items())
    comp_rows = "".join(f"<tr><td>{html.escape(item['stage'])}</td><td><pre>{html.escape(json.dumps(item['frozen'], ensure_ascii=False))}</pre></td><td><pre>{html.escape(json.dumps(item['live'], ensure_ascii=False))}</pre></td><td>{html.escape(item['classification'])}</td></tr>" for comp in comparisons for item in comp["checks"])
    answer_rows = "".join(f"<tr><td>{cid}</td><td>{html.escape(str(row.get('gold_answer')))}</td><td>{html.escape(str(row['validated_answer'].get('answer')))}</td><td>{html.escape(str(row.get('frozen_prediction')))}</td><td>{html.escape(str(row['validated_answer'].get('answer_status')))}</td></tr>" for cid, row in by_case.items())
    details = "".join(f"<section><h2>{cid}</h2><pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></section>" for cid, row in by_case.items())
    return f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>Canonical live v0.1</title><style>body{{font-family:Arial;margin:2rem}}table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border:1px solid #bbb;padding:.4rem;vertical-align:top}}pre{{white-space:pre-wrap;max-width:80rem}}</style><h1>Canonical 首次 LIVE 验证</h1><p>仅运行 00006_3 与 00061_5；gold 在预测保存与验证后才加载。</p><h2>TABLE 1 — End-to-end latency</h2><table><tr><th>Case</th><th>Online total</th><th>Planner</th><th>Retrieval</th><th>Refinement</th><th>Sufficiency</th><th>Fallback</th><th>Reranking</th><th>Payload</th><th>Gemini</th><th>Parse + Validation</th><th>Overhead</th><th>Timing coverage</th></tr>{latency_rows}</table><h2>TABLE 2 — Provider/API latency</h2><table><tr><th>Case</th><th>Planner API</th><th>Fallback model</th><th>Gemini API</th><th>Total API wall-clock</th><th>% online</th></tr>{api_rows}</table><h2>TABLE 3 — Efficiency</h2><table><tr><th>Case</th><th>Frames</th><th>Audio sec</th><th>Speech segs</th><th>Input tok</th><th>Cached tok</th><th>Uncached tok</th><th>Output tok</th><th>Total tok</th><th>API calls</th><th>Retries</th></tr>{eff_rows}</table><h2>TABLE 4 — Caching</h2><table><tr><th>Case</th><th>Input tok</th><th>Cached tok</th><th>Cache hit</th><th>Cache hit rate</th><th>Field available</th></tr>{cache_rows}</table><h2>TABLE 5 — Live behavior consistency</h2><table><tr><th>Stage</th><th>Frozen</th><th>Live</th><th>Classification</th></tr>{comp_rows}</table><h2>TABLE 6 — Gold vs Prediction</h2><table><tr><th>Case</th><th>Gold</th><th>Live prediction</th><th>Frozen prediction</th><th>Status</th></tr>{answer_rows}</table>{details}</html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args()
    if tuple(args.cases) != CASES:
        raise SystemExit("This verification is restricted to 00006_3 then 00061_5")
    load_dotenv(ROOT / ".env", override=False)
    presence = environment_presence()
    config = load_canonical_config(ROOT)
    frozen_paths = [config.path(name) for name in ("planner_plans", "task5b_frozen", "task5c_frozen", "task6_frozen", "task7a_frozen")]
    before_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in frozen_paths}
    manifest = {
        "mode": "execute_live" if args.execute_live else "dry_run",
        "cases": list(CASES), "execution_order": list(CASES), "environment_presence": presence,
        "planned_external_operations": {"00006_3": {"planner": 1, "fallback_local_asr": "only_if_triggered", "gemini": 1}, "00061_5": {"planner": 1, "fallback_local_asr": "only_if_triggered", "gemini": 1}},
        "model_settings": {"gemini": "gemini-3.5-flash", "thinking_level": "low", "store": False},
        "upstream_hashes_before": before_hashes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "live_run_manifest.json", manifest)
    if not args.execute_live:
        print(json.dumps({"dry_run": True, "cases": list(CASES), "required_environment_present": all(presence.values()), "planned_external_operations": manifest["planned_external_operations"]}, ensure_ascii=False, indent=2))
        return 0
    if not all(presence.values()):
        raise SystemExit("Required live environment configuration is unavailable; no API calls were made")

    from google import genai

    planner_request, planner_model = anthropic_requester()
    final_client = genai.Client()
    whisper = LazyWhisperFallback(ROOT)
    runner = CanonicalOnlineRunner(config)
    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    for case_id in CASES:
        state = runner.run_live_case(case_id, planner_request=planner_request, fallback_executor=whisper.execute, final_client=final_client)
        raw_rows.append({"case_id": case_id, "raw_model_text": state.final_model_output["raw_text"]})
        _write_jsonl(OUT / "live_raw_predictions.jsonl", raw_rows)
        validated_rows.append({"case_id": case_id, **state.validated_answer})
        _write_jsonl(OUT / "live_validated_predictions.jsonl", validated_rows)
        comparison = compare_live_state(state, _frozen(config, case_id))
        comparisons.append(comparison)
        _write_jsonl(OUT / "live_behavior_comparison.jsonl", comparisons)
        c5 = state.usage["task5c_v1_2_runtime"]
        c7 = state.usage["task7b_v3_runtime"]
        cache = c7["usage"]
        consistency = state.usage["timing_consistency"]
        online = consistency["online_end_to_end_duration_sec"]
        planner_wall = state.usage["task5a_v2"]["planner_api_wall_clock_sec"]
        fallback_wall = (state.fallback_result or {}).get("canonical_local_asr_total_sec", c5.get("fallback_model_wall_clock_sec"))
        gemini_wall = c7["final_model_api_wall_clock_sec"]
        api_total = planner_wall + gemini_wall
        external_or_model_total = api_total + (fallback_wall or 0.0)
        efficiency = _media_metrics(state)
        frozen_final_dir = ROOT / config.raw["historical_disk_mapping"]["task7b_v3"]
        frozen_prediction = next((row.get("answer") for row in _jsonl(frozen_final_dir / "task7b_validated_outputs.jsonl") if row["case_id"] == case_id), None)
        result = {
            "case_id": case_id, "question": state.question, "validated_answer": state.validated_answer,
            "planner_output": state.planner_output, "retained_evidence_ids": [item["candidate_id"] for item in state.evidence_packet["retained_candidates"]],
            "fallback_execution_count": state.fallback_execution_count, "fallback_result": state.fallback_result,
            "behavior_comparison": comparison, "preflight": state.preflight_result, "efficiency": efficiency,
            "gemini_usage": cache, "planner_model": planner_model, "planner_api_wall_clock_sec": planner_wall,
            "fallback_model_wall_clock_sec": fallback_wall,
            "fallback_decoder_wall_clock_sec": (state.fallback_result or {}).get("latency_sec") if state.fallback_execution_count else None,
            "fallback_model_load_latency_sec": (state.fallback_result or {}).get("model_load_latency_sec") if state.fallback_execution_count else None,
            "final_model_api_wall_clock_sec": gemini_wall,
            "total_api_wall_clock_sec": api_total,
            "total_external_or_model_wall_clock_sec": external_or_model_total,
            "api_percent_online": api_total / online * 100 if online else None,
            "total_api_calls": state.external_calls["planner"] + state.external_calls["final_qa"],
            "retries": state.usage["task5a_v2"]["retries"] + c7["retry_api_calls"],
            "timing_breakdown": {"online_total": online, "planner": _timing(state, "question_planner"), "retrieval": _timing(state, "retrieval_total"), "refinement": _timing(state, "refinement_total"), "sufficiency": _timing(state, "evidence_sufficiency"), "fallback": _timing(state, "fallback_execution"), "reranking": _timing(state, "reranking_and_packet"), "payload": _timing(state, "final_payload_build"), "gemini": _timing(state, "final_model_api"), "parse_validation": (_timing(state, "structured_output_parse") or 0) + (_timing(state, "local_validation") or 0), "overhead": consistency["uninstrumented_overhead_sec"], "coverage": consistency["timing_coverage"]},
            "frozen_prediction": frozen_prediction,
            "fallback_breakdown": {"pre_fallback_online_time": _timing(state, "question_planner") + _timing(state, "retrieval_total") + c5["pre_classification_sec"], "fallback_decision_time": c5["fallback_decision_sec"], "fallback_execution_time": c5["fallback_execution_sec"], "fallback_local_asr_time": (state.fallback_result or {}).get("canonical_local_asr_total_sec"), "post_fallback_processing_time": c5["post_fallback_processing_sec"], "final_model_time": gemini_wall, "total_online_time": online, "fallback_fraction": (c5["fallback_execution_sec"] / online) if c5["fallback_execution_sec"] and online else None},
        }
        results.append(result)
        timings.extend({"case_id": case_id, **row} for row in state.timings)
        efficiency_rows.append({"case_id": case_id, **efficiency})
        usage_rows.append({"case_id": case_id, "planner": state.usage["task5a_v2"], "fallback": c5, "final_qa": c7})
        cache_rows.append({"case_id": case_id, **cache})
        _write_jsonl(OUT / "live_results.jsonl", results)
        _write_jsonl(OUT / "live_stage_timings.jsonl", timings)
        _write_jsonl(OUT / "live_efficiency.jsonl", efficiency_rows)
        _write_jsonl(OUT / "live_model_usage.jsonl", usage_rows)
        _write_jsonl(OUT / "live_cache_metrics.jsonl", cache_rows)
        if comparison["category_4_detected"]:
            break

    # Gold is deliberately loaded only after raw and validated prediction files exist.
    if not (OUT / "live_raw_predictions.jsonl").is_file() or not (OUT / "live_validated_predictions.jsonl").is_file():
        raise RuntimeError("Prediction persistence guard failed before post-hoc gold loading")
    manifest_rows = json.loads(config.path("case_manifest").read_text(encoding="utf-8"))
    gold = {row["case_id"]: row["answer"] for row in manifest_rows if row["case_id"] in {item["case_id"] for item in results}}
    for result in results:
        result["gold_answer"] = gold[result["case_id"]]
    _write_jsonl(OUT / "live_results.jsonl", results)
    after_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in frozen_paths}
    manifest["upstream_hashes_after"] = after_hashes
    manifest["upstream_hashes_unchanged"] = before_hashes == after_hashes
    manifest["completed_cases"] = [row["case_id"] for row in results]
    _write_json(OUT / "live_run_manifest.json", manifest)
    summary = {"case_count": len(results), "cases": results, "category_4_detected": any(item["category_4_detected"] for item in comparisons), "upstream_hashes_unchanged": before_hashes == after_hashes}
    _write_json(OUT / "live_summary.json", summary)
    md_lines = ["# Canonical live v0.1", "", "First live verification for exactly two frozen development cases.", ""]
    for row in results:
        timing = row["timing_breakdown"]
        usage = row["gemini_usage"]
        md_lines.extend([
            f"## {row['case_id']}", "",
            f"- Answer status: `{row['validated_answer']['answer_status']}`",
            f"- Answer: {row['validated_answer'].get('answer')}",
            f"- Retained evidence: `{row['retained_evidence_ids']}`",
            f"- Fallback executions: {row['fallback_execution_count']}",
            f"- Behavior comparison: `{row['behavior_comparison']['overall_classification']}`",
            f"- Online total: {timing['online_total']:.6f}s",
            f"- Timing coverage: {timing['coverage']:.6%}",
            f"- Uninstrumented overhead: {timing['overhead']:.6f}s",
            f"- Planner/Gemini wall clock: {row['planner_api_wall_clock_sec']:.6f}s / {row['final_model_api_wall_clock_sec']:.6f}s",
            f"- Fallback local-ASR wall clock: {row['fallback_model_wall_clock_sec']}",
            f"- Gemini input/output/total tokens: {usage.get('input_tokens')}/{usage.get('output_tokens')}/{usage.get('total_tokens')}",
            f"- Gemini cached tokens/cache hit: {usage.get('total_cached_tokens')}/{usage.get('cache_hit')}", "",
        ])
    md_lines.extend(["## Invariants", "", f"- Upstream hashes unchanged: `{before_hashes == after_hashes}`", f"- Category-4 mismatch detected: `{summary['category_4_detected']}`", "- No prompt, routing, retrieval, fallback, reranking, budget, uncertainty, or answer-policy changes were made.", ""])
    (OUT / "live_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    (OUT / "live_review.html").write_text(_report_html(summary, results, comparisons), encoding="utf-8")
    print(json.dumps({"completed_cases": manifest["completed_cases"], "category_4_detected": summary["category_4_detected"], "upstream_hashes_unchanged": manifest["upstream_hashes_unchanged"]}, ensure_ascii=False, indent=2))
    return 2 if summary["category_4_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
