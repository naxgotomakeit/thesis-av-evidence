from __future__ import annotations

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

from src.evaluation.answer_metrics import evaluate_answer  # noqa: E402
from src.instrumentation.efficiency import gemini_usage_metrics  # noqa: E402
from src.instrumentation.timing import blocked_online_timings, existing_offline_artifact_timings, timing_consistency  # noqa: E402


CASES = ("00006_3", "00061_5")
OUT = ROOT / "outputs/baseline_instrumentation/v0_1"
TASK7A = ROOT / "outputs/final_qa_preflight/task7a_v1/task7a_payloads.jsonl"
TASK7B = ROOT / "outputs/final_qa/task7b_v0_3"
TRACKED_ROOTS = [
    ROOT / "outputs/question_planner/v2", ROOT / "outputs/planner_guided_retrieval/v1_1",
    ROOT / "outputs/evidence_sufficiency/task5c_v1_2", ROOT / "outputs/relation_reranking/task6_v1_2",
    ROOT / "outputs/final_qa_preflight/task7a_v1", ROOT / "outputs/final_qa/task7b_v0_1",
    ROOT / "outputs/final_qa/task7b_v0_2", ROOT / "outputs/final_qa/task7b_v0_3",
]
NON_OVERLAPPING = [
    "question_planner", "modality_routing", "retrieval_total", "relation_construction",
    "relation_reranking", "local_visual_refinement", "local_audio_refinement",
    "evidence_sufficiency", "fallback_decision", "fallback_execution",
    "evidence_packet_build", "final_payload_build", "final_model_api", "local_validation",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def hash_files() -> dict[str, str]:
    paths = sorted({path for root in TRACKED_ROOTS for path in root.rglob("*") if path.is_file()})
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def payload_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    visual = [item for group in payload["evidence_groups"] for item in group["visual_evidence"]]
    speech = [item for group in payload["evidence_groups"] for item in group["speech_evidence"]]
    acoustic = [item for group in payload["evidence_groups"] for item in group["acoustic_evidence"]]
    frames = [frame for item in visual for frame in item["frames"]]
    audio_paths = [ROOT / item["audio_clip_path"] for item in acoustic if item.get("audio_clip_path")]
    frame_paths = [ROOT / frame["frame_path"] for frame in frames]
    return {
        "visual_frame_count": len(frames), "visual_asset_bytes": sum(path.stat().st_size for path in frame_paths if path.is_file()),
        "audio_clip_count": len(acoustic), "audio_duration_sec": sum(float(item["end_sec"]) - float(item["start_sec"]) for item in acoustic),
        "audio_asset_bytes": sum(path.stat().st_size for path in audio_paths if path.is_file()),
        "speech_segment_count": len(speech), "transcript_character_count": sum(len(item.get("transcript", "")) for item in speech),
        "evidence_group_count": len(payload["evidence_groups"]), "relation_count": sum(len(group["relations"]) for group in payload["evidence_groups"]),
    }


def stage_rows(case_id: str) -> list[dict[str, Any]]:
    rows = existing_offline_artifact_timings() + blocked_online_timings()
    for row in rows:
        row["case_id"] = case_id
        if case_id == "00006_3" and row["stage_name"] in {"visual_retrieval", "acoustic_retrieval", "local_visual_refinement", "local_audio_refinement"}:
            row["skip_reason"] = "modality_not_required"
        if case_id == "00006_3" and row["stage_name"] in {"fallback_execution", "fallback_local_asr"}:
            row["skip_reason"] = "fallback_not_triggered"
        if case_id == "00061_5" and row["stage_name"] in {"visual_retrieval", "acoustic_retrieval", "local_visual_refinement", "local_audio_refinement"}:
            row["skip_reason"] = "modality_not_required"
        if case_id == "00061_5" and row["stage_name"] in {"fallback_decision", "fallback_execution", "fallback_local_asr"}:
            row["notes"].append("planned_fallback_path_but_not_executed_because_clean_replay_is_blocked")
    return rows


def html_page(summary: dict[str, Any], timings: list[dict[str, Any]], efficiency: list[dict[str, Any]], caches: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return html.escape("N/A" if value is None else str(value))
    def table(headers: list[str], rows: list[list[Any]]) -> str:
        return "<table><tr>" + "".join(f"<th>{cell(x)}</th>" for x in headers) + "</tr>" + "".join("<tr>" + "".join(f"<td>{cell(x)}</td>" for x in row) + "</tr>" for row in rows) + "</table>"
    timing_by = {case: {row["stage_name"]: row for row in timings if row["case_id"] == case} for case in CASES}
    eval_by, eff_by, cache_by = ({row["case_id"]: row for row in items} for items in (evaluations, efficiency, caches))
    latency = [[case, None, None, None, None, None, None, None, None, None, None] for case in CASES]
    percentages = [[case, row["stage_name"], row["duration_sec"], None] for case in CASES for row in timings if row["case_id"] == case and row["stage_name"] in NON_OVERLAPPING]
    eff = [[case, eff_by[case]["visual_frame_count"], eff_by[case]["audio_duration_sec"], eff_by[case]["speech_segment_count"], None, None, None, None, None, 0, 0] for case in CASES]
    caching = [[case, cache_by[case]["input_tokens"], cache_by[case]["cache_eligible_by_input_size"], cache_by[case]["total_cached_tokens"], cache_by[case]["cache_hit"], cache_by[case]["cache_hit_rate"]] for case in CASES]
    comparison = [[case, eval_by[case]["gold_reference_answer"], eval_by[case]["validated_prediction"], eval_by[case]["answer_status"], eval_by[case]["reference_metrics"]["normalized_exact_match"], round(eval_by[case]["reference_metrics"]["bleu"], 6), round(eval_by[case]["reference_metrics"]["cider"], 6)] for case in CASES]
    details = "".join(f"<h2>{case}</h2><h3>Exact stage timeline</h3><pre>{html.escape(json.dumps([row for row in timings if row['case_id']==case],ensure_ascii=False,indent=2))}</pre><h3>Efficiency / model / cache</h3><pre>{html.escape(json.dumps({'efficiency':eff_by[case],'cache':cache_by[case]},ensure_ascii=False,indent=2))}</pre><h3>Post-hoc evaluation scaffold</h3><pre>{html.escape(json.dumps(eval_by[case],ensure_ascii=False,indent=2))}</pre>" for case in CASES)
    return f"""<!doctype html><meta charset=utf-8><title>Baseline instrumentation v0.1</title><style>body{{font:15px system-ui;margin:2rem;max-width:1500px}}table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border:1px solid #bbb;padding:.45rem;vertical-align:top}}th{{background:#eef3f8}}pre{{white-space:pre-wrap;background:#f5f7f9;padding:1rem}}</style><h1>Baseline instrumentation v0.1 — dry-run blocked audit</h1><p><b>未执行 live pipeline。</b> 当前批处理架构不能无损、单案、从离线索引完整重放；所有 online duration 均保持 N/A，未用旧文件时间戳或 0 秒冒充测量。</p><h2>TABLE 1 — End-to-end latency</h2>{table(['Case','Online total','Planner','Retrieval','Reranking','Refinement','Sufficiency','Fallback','Final Gemini','Validation','Uninstrumented overhead'],latency)}<h2>TABLE 2 — Stage percentage</h2>{table(['Case','Stage','Duration','% of online total'],percentages)}<h2>TABLE 3 — Efficiency</h2>{table(['Case','Frames','Audio sec','Speech segs','Input tok','Cached tok','Uncached tok','Output tok','Total tok','API calls','Retries'],eff)}<h2>TABLE 4 — Caching</h2>{table(['Case','Input tokens','Cache eligible','Cached tokens','Cache hit','Cache hit rate'],caching)}<h2>TABLE 5 — Gold vs Prediction</h2>{table(['Case','Gold','Prediction','Status','Exact Match','BLEU','CIDEr'],comparison)}<p><b>BLEU/CIDEr are diagnostic reference-overlap metrics, NOT final correctness judgments.</b></p><h2>架构审计</h2><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>{details}"""


def main() -> int:
    before = hash_files()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        dotenv_available = True
    except ImportError:
        dotenv_available = False
    key_presence = {"ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")), "ANTHROPIC_MODEL": bool(os.environ.get("ANTHROPIC_MODEL")), "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY"))}
    payloads = {row["case_id"]: row for row in read_jsonl(TASK7A)}
    audit_started = time.perf_counter()
    limitations = [
        "Task 5A runner is a six-case batch writer targeting frozen outputs/question_planner/v2 and has no isolated per-case output contract.",
        "Task 5B is a monolithic script coupled to serialized Task 5A/Task 4 artifacts and fixed output/media paths.",
        "Task 5C v1.1 consumes Task 5C v1 serialized output and reruns corrected fallback; composing v1 then v1.1 would double-execute fallback rather than replay one production path.",
        "Task 5C v1.2, Task 6 v1/v1.1/v1.2, and Task 7A are versioned transformations over saved files rather than one transactional per-question entry point.",
        "Consequently, a complete online total and fallback overhead cannot be measured without first defining a canonical replay entry point, which is outside this instrumentation-only task.",
    ]
    audit_duration = time.perf_counter() - audit_started
    timings = [row for case in CASES for row in stage_rows(case)]
    consistency = {case: timing_consistency([row for row in timings if row["case_id"] == case], NON_OVERLAPPING) for case in CASES}
    efficiency = [{"case_id": case, **payload_metrics(payloads[case]), "planner_api_calls": 0, "final_model_api_calls": 0, "fallback_model_calls": 0, "total_api_calls": 0, "retries": 0, "model_names": {"planner": os.environ.get("ANTHROPIC_MODEL") if key_presence["ANTHROPIC_MODEL"] else None, "final": "gemini-3.5-flash"}, "thinking_level": "low", "measurement_status": "planned_payload_only_online_replay_blocked"} for case in CASES]
    model_usage = [{"case_id": case, "actual_external_calls": 0, "planner_usage": None, "final_model_usage": None, "fallback_model_usage": None, "provider_api_wall_clock_latency": None, "warnings": ["no_live_calls_due_to_blocked_clean_online_replay"]} for case in CASES]
    caches = [{"case_id": case, **gemini_usage_metrics(None)} for case in CASES]
    write_jsonl(OUT / "stage_timings.jsonl", timings)
    write_jsonl(OUT / "efficiency_metrics.jsonl", efficiency)
    write_jsonl(OUT / "model_usage.jsonl", model_usage)
    write_jsonl(OUT / "cache_metrics.jsonl", caches)
    # Evaluation is a scaffold sanity check over already-saved frozen v0.3 predictions; no model is rerun.
    raw_path, valid_path, posthoc_path = (TASK7B / name for name in ("task7b_raw_model_outputs.jsonl", "task7b_validated_outputs.jsonl", "task7b_posthoc_evaluation.jsonl"))
    if not raw_path.is_file() or not valid_path.is_file():
        raise RuntimeError("saved_prediction_and_validation_required_before_posthoc_evaluation")
    valid = {row["case_id"]: row for row in read_jsonl(valid_path)}
    posthoc = {row["case_id"]: row for row in read_jsonl(posthoc_path)}
    eval_inputs = [{"case_id": case, "question": payloads[case]["question"], "gold_dataset_answer": posthoc[case]["gold_dataset_answer"], "raw_model_prediction": posthoc[case]["raw_model_prediction"], "validated_prediction": valid[case]["answer"], "answer_status": valid[case]["answer_status"], "confidence": valid[case]["confidence"], "abstain": valid[case]["abstain"], "uncertainties": valid[case]["final_uncertainties"]} for case in CASES]
    corpus = [str(row["gold_dataset_answer"] or "") for row in eval_inputs]
    evaluation_started = time.perf_counter()
    evaluations = [evaluate_answer(row, corpus) for row in eval_inputs]
    evaluation_duration = time.perf_counter() - evaluation_started
    write_jsonl(OUT / "evaluation_diagnostics.jsonl", evaluations)
    manifest = {
        "task": "baseline measurement infrastructure dry-run", "requested_cases": list(CASES), "live_execution_performed": False,
        "stop_reason": "complete_online_replay_not_cleanly_available", "api_key_presence_only": key_presence,
        "dotenv_loader_available": dotenv_available, "api_keys_logged": False, "planned_external_calls_if_replay_existed": {"planner": 2, "final_gemini": 2, "fallback_whisper": "indeterminate because current v1/v1.1 composition duplicates fallback"},
        "actual_external_calls": 0, "store": False, "previous_interaction_id_used": False,
        "research_behavior_changed": False, "architectural_entry_point_map": {"planner": "scripts/run_task5a_question_planner.py -> src.question_planner.task5a.plan_with_retry", "retrieval": "scripts/run_task5b_retrieval.py::build_case", "sufficiency_v1": "scripts/run_task5c_evidence_sufficiency.py::process_record", "sufficiency_v1_1": "scripts/run_task5c_v1_1.py::process", "acoustic_semantics_v1_2": "scripts/run_task5c_v1_2.py::process", "reranking": "scripts/run_task6_relation_reranking.py::process plus v1.1/v1.2 serialization patches", "payload": "scripts/run_task7a_preflight.py::build_payload", "final": "src.final_qa.task7b_gemini.build_request/call_with_one_technical_retry"},
        "limitations": limitations,
    }
    write_json(OUT / "run_manifest.json", manifest)
    after = hash_files()
    if before != after:
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise RuntimeError(f"frozen_upstream_changed:{changed}")
    summary = {"status": "blocked_before_live_calls", "requested_cases": list(CASES), "real_online_replay_possible": False, "timing_trustworthiness": "instrumentation utility tested; no online timings claimed", "stage_consistency": consistency, "architectural_audit_duration_sec": audit_duration, "evaluation_overhead_sec": evaluation_duration, "evaluation_excluded_from_online_latency": True, "cache_observation": "not_observed_no_live_calls", "offline_indexing": "not_measured_existing_artifact", "actual_api_calls": 0, "upstream_hashes": {"before": before, "after": after, "unchanged": True}, "limitations": limitations, "larger_pilot_ready": False}
    write_json(OUT / "instrumentation_summary.json", summary)
    markdown = ["# Baseline instrumentation v0.1", "", "Status: **blocked before live calls**. The current versioned batch architecture cannot replay the full per-question online path without duplicating fallback or overwriting frozen outputs.", "", "- Actual API/model calls: 0", "- Online timings: not measured; never reported as zero", "- Offline indexing timings: not_measured_existing_artifact", f"- Architecture audit overhead: {audit_duration:.6f}s (excluded from online latency)", f"- Evaluation scaffold overhead: {evaluation_duration:.6f}s (excluded from online latency)", "- Cache behavior: not observed; total_cached_tokens is null with an explicit warning", "- BLEU/CIDEr: diagnostic reference overlap only, never correctness", "- Larger pilot ready: no", "", "## Limitations", *[f"- {item}" for item in limitations]]
    (OUT / "instrumentation_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (OUT / "instrumentation_review.html").write_text(html_page(summary, timings, efficiency, caches, evaluations), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "cases": list(CASES), "actual_external_calls": 0, "output": str(OUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
