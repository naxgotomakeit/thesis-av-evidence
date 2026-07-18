from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import whisper
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.task5c import local_audio_path  # noqa:E402
from src.retrieval.task5c_v1_1 import (  # noqa:E402
    FALLBACK_MODEL, classify_v1_1, evaluate_candidates, staged_local_asr_fallback, visual_accounting,
)


V1_RESULTS = ROOT / "outputs/evidence_sufficiency/task5c_v1/task5c_results.jsonl"
TASK5B_V11 = ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl"
TASK5B_V11_SUMMARY = ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_summary.json"
TASK5B_BASE = ROOT / "outputs/planner_guided_retrieval/task5b_candidates.jsonl"
TASK5A = ROOT / "outputs/question_planner/v2/task5a_plans.jsonl"
MANIFEST = ROOT / "data/manifests/mvp_cases_6.json"
CONFIG = ROOT / "configs/audio_mvp.yaml"
OUT = ROOT / "outputs/evidence_sufficiency/task5c_v1_1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_results(records: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "task5c_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_v1_without_posthoc_reference(path: Path) -> list[dict[str, Any]]:
    """Drop the v1 post-hoc field before parsing; it is loaded only after v1.1 evidence is saved."""
    rows = []
    marker = ', "posthoc_weak_reference_evaluation":'
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index = line.rfind(marker)
        clean = line[:index] + "}" if index >= 0 else line
        rows.append(json.loads(clean))
    return rows


def phrases(cues: dict[str, Any]) -> list[str]:
    return [str(item["text"]) for item in cues.get("quoted_phrases", []) if item.get("text")]


def hard_and_decode_intervals(record: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    cues = record["deterministic_question_cues"]
    for cue in cues.get("time_cues", []):
        if cue.get("start_sec") is not None:
            hard = float(cue["start_sec"]), float(cue.get("end_sec", cue["start_sec"]))
            return hard, hard
    fallback = record.get("fallback", {})
    search = fallback.get("search_interval") or fallback.get("decode_interval")
    if search and search.get("start_sec") is not None:
        decode = float(search["start_sec"]), float(search["end_sec"])
        return decode, decode
    anchor = record.get("task5b_v1_1_input", {}).get("anchor_resolution", {}).get("final_search_intervals", [])
    if anchor:
        decode = float(anchor[0]["start_sec"]), float(anchor[0]["end_sec"])
        return decode, decode
    return None


def video_id(record: dict[str, Any]) -> str:
    files = record["task5b_v1_1_input"].get("selected_candidates", [])
    for candidate in files:
        if candidate.get("video_id"):
            return str(candidate["video_id"])
    paths = record["task5b_v1_1_input"].get("local_visual_refinement", {}).get("micro_windows", [])
    for candidate in paths:
        if candidate.get("video_id"):
            return str(candidate["video_id"])
    source = record["task5b_v1_1_input"].get("source_candidates", "")
    # The case id is not used for routing; audio/video index paths are unavailable in v1 summary for non-visual cases.
    return str(record["case_id"]).split("_", 1)[0]


class LazyWhisper:
    def __init__(self, config: dict[str, Any]):
        self.config, self.transcriber = config, None
        self.model_name, self.device, self.load_count, self.load_latency_sec = str(config.get("whisper_model", FALLBACK_MODEL)), "not_loaded", 0, 0.0

    def get(self) -> Callable:
        if self.transcriber is not None:
            return self.transcriber
        started = time.perf_counter()
        requested = str(self.config.get("whisper_device", "cpu"))
        self.device = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        model = whisper.load_model(self.model_name, device=self.device, download_root=str(self.config.get("whisper_cache_dir", "")) or None)
        self.load_latency_sec, self.load_count = time.perf_counter() - started, 1

        def transcribe(audio, sample_rate):
            if sample_rate != 16000:
                raise ValueError("Task 5C v1.1 local Whisper input must be 16 kHz")
            return model.transcribe(audio, fp16=self.device == "cuda", word_timestamps=False, condition_on_previous_text=False, verbose=None)
        self.transcriber = transcribe
        return self.transcriber


def no_fallback() -> dict[str, Any]:
    return {"triggered": False, "outcome": "not_triggered", "model": f"whisper-{FALLBACK_MODEL}", "model_calls": 0, "local_audio_duration_sec": 0.0, "total_decoding_audio_duration_sec": 0.0, "valid_transcript_segments": [], "excluded_transcript_segments": [], "phrase_matches": [], "added_candidates": [], "early_stop_after_complete_pass": False, "chunk_fallback_triggered": False, "chunk_fallback_reason": None, "warnings": [], "latency_sec": 0.0}


def record_context(record: dict[str, Any]) -> dict[str, Any]:
    source = record["task5b_v1_1_input"]
    return {"case_id": record["case_id"], "question": record["question"], "selected_visual_evidence_frames": copy.deepcopy(source.get("selected_visual_evidence_frames", [])), "anchor_resolution": copy.deepcopy(source.get("anchor_resolution", {})), "local_visual_refinement": copy.deepcopy(source.get("local_visual_refinement", {}))}


def process(record: dict[str, Any], whisper_model: LazyWhisper) -> dict[str, Any]:
    started = time.perf_counter()
    plan, cues = copy.deepcopy(record["task5a_plan_summary"]), copy.deepcopy(record["deterministic_question_cues"])
    context = record_context(record)
    pre_candidates = copy.deepcopy(record["fallback"]["pre_fallback_candidates"])
    pre, pre_acoustic = classify_v1_1(context, plan, cues, pre_candidates)
    fallback = no_fallback()
    intervals = hard_and_decode_intervals(record)
    if pre["fallback_required"] and intervals:
        hard, decode = intervals
        vid = video_id(record)
        source_mp4 = next((item.get("source_mp4_path") for item in context["local_visual_refinement"].get("dense_frames", []) if item.get("source_mp4_path")), "")
        audio_path = local_audio_path(vid, source_mp4)
        try:
            before_load = whisper_model.load_count
            fallback = staged_local_asr_fallback(case_id=record["case_id"], video_id=vid, search_interval=decode, hard_question_interval=hard, phrases=phrases(cues), source_audio=audio_path, transcriber=whisper_model.get(), model_name=whisper_model.model_name)
            fallback["model_device"] = whisper_model.device
            fallback["model_loads_for_case"] = whisper_model.load_count - before_load
            fallback["model_load_latency_sec"] = whisper_model.load_latency_sec if fallback["model_loads_for_case"] else 0.0
        except Exception as exc:
            fallback = no_fallback()
            fallback.update({"triggered": True, "outcome": "failed_with_error", "warnings": [f"local_asr_model_initialization_error:{type(exc).__name__}:{str(exc)[:240]}"], "hard_question_interval": {"start_sec": hard[0], "end_sec": hard[1]}, "decode_interval": {"start_sec": decode[0], "end_sec": decode[1]}})
    post_candidates = pre_candidates + copy.deepcopy(fallback["added_candidates"])
    post, post_acoustic = classify_v1_1(context, plan, cues, post_candidates)
    fallback_was_triggered = bool(fallback["triggered"])
    recovered = bool(fallback["added_candidates"])
    additional_required = post["evidence_status"] == "insufficient"
    result = {
        "case_id": record["case_id"], "question": record["question"], "execution_status": record["execution_status"],
        "task5a_plan_summary": plan, "deterministic_question_cues": cues,
        "task5b_v1_1_input": copy.deepcopy(record["task5b_v1_1_input"]),
        "source_task5c_v1_fallback_history": copy.deepcopy(record["fallback"]), "source_task5c_v1_fallback_required": record["fallback_required"],
        "pre_fallback_evidence_status": pre["evidence_status"], "pre_fallback_assessment": pre,
        "fallback": fallback, "fallback_was_triggered": fallback_was_triggered, "fallback_recovered_evidence": recovered,
        "post_fallback_evidence_status": post["evidence_status"], "evidence_status": post["evidence_status"],
        "sufficiency_reason_codes": post["sufficiency_reason_codes"], "critical_missing_evidence": post["critical_missing_evidence"], "ambiguity_flags": post["ambiguity_flags"],
        "fallback_required": additional_required, "additional_fallback_required": additional_required,
        "pre_fallback_candidates": pre_candidates, "post_fallback_candidates": post_candidates,
        "broad_acoustic_source_diagnostics_pre": pre_acoustic, "broad_acoustic_source_diagnostics_post": post_acoustic,
        "visual_efficiency_accounting": visual_accounting(record["task5b_v1_1_input"]),
        "recommended_human_check": "Review semantic relevance, speaker attribution, broad acoustic source provenance, and any clipped/excluded ASR timestamps. Structural sufficiency is not a final-answer claim.",
        "runtime": {"task5c_llm_api_calls": 0, "task5c_vlm_calls": 0, "old_task5c_v1_local_whisper_model_calls": record["runtime"].get("local_whisper_model_calls", 0), "local_whisper_model_calls": fallback["model_calls"], "old_task5c_v1_decoding_audio_duration_sec": record["fallback"].get("total_decoding_audio_duration_sec", 0.0), "local_audio_duration_processed_sec": fallback["total_decoding_audio_duration_sec"], "unique_local_audio_duration_sec": fallback["local_audio_duration_sec"], "fallback_latency_sec": fallback["latency_sec"], "total_task5c_v1_1_latency_sec": time.perf_counter() - started, "candidates_before_fallback": len(pre_candidates), "candidates_after_fallback": len(post_candidates)},
    }
    return result


def status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {status: sum(record[field] == status for record in records) for status in ("sufficient", "questionable", "insufficient")}


def html_page(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    cards = []
    for record in records:
        cards.append(f'''<article><h2>{record["case_id"]}</h2><h3>原始问题</h3><p>{html.escape(record["question"])}</p>
<h3>执行状态</h3>{pre({"execution_status":record["execution_status"],"pre_fallback_evidence_status":record["pre_fallback_evidence_status"],"post_fallback_evidence_status":record["post_fallback_evidence_status"],"fallback_was_triggered":record["fallback_was_triggered"],"fallback_recovered_evidence":record["fallback_recovered_evidence"],"fallback_required":record["fallback_required"],"additional_fallback_required":record["additional_fallback_required"]})}
<h3>Fallback 历史与本次分阶段执行</h3>{pre({"source_task5c_v1_fallback_history":record["source_task5c_v1_fallback_history"],"task5c_v1_1_fallback":record["fallback"]})}
<h3>有效与排除的 ASR 转写片段</h3>{pre({"valid_transcript_segments":record["fallback"].get("valid_transcript_segments",[]),"excluded_transcript_segments":record["fallback"].get("excluded_transcript_segments",[])})}
<h3>Broad acoustic-source 诊断</h3>{pre(record["broad_acoustic_source_diagnostics_post"])}
<h3>修正后的视觉帧效率统计</h3>{pre(record["visual_efficiency_accounting"])}
<h3>Fallback 前弱参考评估</h3>{pre(record.get("pre_fallback_reference_evaluation",{}))}<h3>Fallback 后弱参考评估</h3>{pre(record.get("post_fallback_reference_evaluation",{}))}
<h3>成本与延迟</h3>{pre(record["runtime"])}<h3>人工检查建议</h3><p>{html.escape(record["recommended_human_check"])}</p></article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 5C v1.1 证据状态与 ASR 时间戳修正</title><style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:480px;overflow:auto}}</style><h1>Task 5C v1.1：证据充分性、时间戳与效率统计修正</h1><p>弱参考仅在证据结果首次保存后加载；所有状态均为结构性判断，不生成最终答案。</p>{pre(summary["aggregate"])}{''.join(cards)}'''


def main() -> int:
    argparse.ArgumentParser().parse_args()
    required = [V1_RESULTS, TASK5B_V11, TASK5B_V11_SUMMARY, TASK5B_BASE, TASK5A, MANIFEST, CONFIG]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    tracked = [V1_RESULTS, TASK5B_V11, TASK5B_BASE, TASK5A]
    before = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    records_v1 = load_v1_without_posthoc_reference(V1_RESULTS)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    whisper_model = LazyWhisper(config)
    records = [process(record, whisper_model) for record in records_v1]
    # The first write contains no weak-reference evaluation.
    write_results(records)
    # Post-hoc stage: now load weak timestamps and legacy visual accounting only.
    manifest = load(MANIFEST)
    references = {row["case_id"]: (float(row["provided_timestamp_start"]), float(row["provided_timestamp_end"])) for row in manifest}
    legacy = load(TASK5B_V11_SUMMARY).get("visual_frame_accounting", {}).get("per_case", {})
    for record in records:
        record["pre_fallback_reference_evaluation"] = evaluate_candidates(record["pre_fallback_candidates"], references[record["case_id"]])
        record["post_fallback_reference_evaluation"] = evaluate_candidates(record["post_fallback_candidates"], references[record["case_id"]])
        record["visual_efficiency_accounting"] = visual_accounting(record["task5b_v1_1_input"], legacy.get(record["case_id"], {}).get("old_task5b_selected_visual_frame_count"))
    write_results(records)
    after = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    if before != after:
        raise RuntimeError("Frozen Task 5A/5B/5C or global-index source changed during Task 5C v1.1")
    aggregate = {"case_count": len(records), "pre_fallback_evidence_status_counts": status_counts(records, "pre_fallback_evidence_status"), "post_fallback_evidence_status_counts": status_counts(records, "post_fallback_evidence_status"), "fallback_triggered_case_count": sum(record["fallback_was_triggered"] for record in records), "fallback_recovery_count": sum(record["fallback_recovered_evidence"] for record in records), "additional_fallback_required_count": sum(record["additional_fallback_required"] for record in records), "old_task5c_v1_whisper_calls": sum(record["runtime"]["old_task5c_v1_local_whisper_model_calls"] for record in records), "new_task5c_v1_1_whisper_calls": sum(record["runtime"]["local_whisper_model_calls"] for record in records), "old_task5c_v1_decoding_audio_duration_sec": sum(record["runtime"]["old_task5c_v1_decoding_audio_duration_sec"] for record in records), "new_task5c_v1_1_decoding_audio_duration_sec": sum(record["runtime"]["local_audio_duration_processed_sec"] for record in records), "selected_visual_frames": sum(record["visual_efficiency_accounting"]["selected_visual_frames"] for record in records), "task5c_llm_api_calls": 0, "task5c_vlm_calls": 0, "invalid_asr_segment_count": sum(len(record["fallback"].get("excluded_transcript_segments", [])) for record in records), "clipped_asr_segment_count": sum(sum(segment["timestamp_validity"] == "clipped_to_decode_interval" for segment in record["fallback"].get("valid_transcript_segments", [])) for record in records), "broad_acoustic_ambiguity_case_count": sum(any(item["ambiguity_codes"] for item in record["broad_acoustic_source_diagnostics_post"]) for record in records)}
    summary = {"task": "Task 5C v1.1 generic correctness and accounting fix", "evidence_results_saved_before_reference_load": True, "weak_reference_role": "post-hoc only; never used for classification, fallback, timestamp validation, or candidate generation", "no_llm_or_vlm_calls": True, "source_inputs": {"task5c_v1": str(V1_RESULTS.relative_to(ROOT)).replace("\\", "/"), "task5b_v1_1": str(TASK5B_V11.relative_to(ROOT)).replace("\\", "/")}, "source_integrity": {"before": before, "after": after, "unchanged": True}, "aggregate": aggregate, "warnings": sorted({warning for record in records for warning in record["fallback"].get("warnings", [])})}
    write_json(OUT / "task5c_summary.json", summary)
    markdown = ["# Task 5C v1.1 summary", "", "Generic reporting, timestamp-validation, staged-ASR, acoustic-provenance, and efficiency-accounting correction only.", "", f"- Pre/post evidence status: {aggregate['pre_fallback_evidence_status_counts']} / {aggregate['post_fallback_evidence_status_counts']}", f"- Old/new Whisper calls and decode audio: {aggregate['old_task5c_v1_whisper_calls']} / {aggregate['new_task5c_v1_1_whisper_calls']}; {aggregate['old_task5c_v1_decoding_audio_duration_sec']:.3f}s / {aggregate['new_task5c_v1_1_decoding_audio_duration_sec']:.3f}s", f"- Invalid/clipped ASR segments: {aggregate['invalid_asr_segment_count']} / {aggregate['clipped_asr_segment_count']}", f"- Broad acoustic ambiguity cases: {aggregate['broad_acoustic_ambiguity_case_count']}", "- No LLM/VLM calls. Weak references were loaded only after the evidence file was first saved.", "", "| case | execution | pre | post | fallback required now | visual frames |", "|---|---|---|---|---|---:|"]
    for record in records:
        markdown.append(f"| {record['case_id']} | {record['execution_status']} | {record['pre_fallback_evidence_status']} | {record['post_fallback_evidence_status']} | {record['fallback_required']} | {record['visual_efficiency_accounting']['selected_visual_frames']} |")
    (OUT / "task5c_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (OUT / "task5c_human_review.html").write_text(html_page(records, summary), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
