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

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.task5c import (  # noqa:E402
    FALLBACK_MODEL,
    classify_evidence,
    local_audio_path,
    no_fallback,
    run_local_asr_fallback,
)


SOURCE_CANDIDATES = ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl"
SOURCE_SUMMARY = ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_summary.json"
PLAN_PATH = ROOT / "outputs/question_planner/v2/task5a_plans.jsonl"
OUTPUT = ROOT / "outputs/evidence_sufficiency/task5c_v1"
CONFIG_PATH = ROOT / "configs/audio_mvp.yaml"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_results(records: list[dict[str, Any]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "task5c_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_id_from_record(record: dict[str, Any]) -> str:
    for key in ("visual_micro_index", "coarse_visual_index", "speech_transcripts"):
        value = record.get("input_files", {}).get(key)
        if value:
            return Path(value).parent.name
    raise ValueError("unable_to_resolve_video_id_from_task5b_input_files")


class LazyWhisper:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.transcriber: Callable | None = None
        self.model_name = str(config.get("whisper_model", FALLBACK_MODEL))
        self.device = "not_loaded"
        self.load_latency_sec = 0.0
        self.load_count = 0

    def get(self) -> Callable:
        if self.transcriber is not None:
            return self.transcriber
        started = time.perf_counter()
        import torch
        import whisper

        requested = str(self.config.get("whisper_device", "cpu"))
        self.device = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        model = whisper.load_model(self.model_name, device=self.device, download_root=str(self.config.get("whisper_cache_dir", "")) or None)

        def transcribe(audio, sample_rate):
            if sample_rate != 16000:
                raise ValueError("Task 5C local Whisper input must be 16 kHz")
            return model.transcribe(audio, fp16=self.device == "cuda", word_timestamps=False, condition_on_previous_text=False, verbose=None)

        self.transcriber = transcribe
        self.load_latency_sec = time.perf_counter() - started
        self.load_count = 1
        return self.transcriber


def execution_status(record: dict[str, Any]) -> str:
    return str(record.get("codex_diagnostic", {}).get("execution_status", "complete"))


def process_record(base: dict[str, Any], task5a: dict[str, Any], whisper: LazyWhisper) -> dict[str, Any]:
    started = time.perf_counter()
    plan = copy.deepcopy(task5a["plan"])
    cues = copy.deepcopy(task5a["deterministic_cues"])
    pre_candidates = copy.deepcopy(base.get("selected_candidates", []))
    pre = classify_evidence(base, plan, cues, pre_candidates)
    fallback = no_fallback()
    if pre["fallback_required"]:
        video_id = video_id_from_record(base)
        source_audio = local_audio_path(video_id, str(base.get("input_files", {}).get("source_mp4", "")))
        try:
            before_loads = whisper.load_count
            transcriber = whisper.get()
            fallback = run_local_asr_fallback(case_id=base["case_id"], video_id=video_id, record=base, cues=cues, source_audio=source_audio, transcriber=transcriber, model_name=whisper.model_name)
            fallback["model_device"] = whisper.device
            fallback["model_loads_for_case"] = whisper.load_count - before_loads
            fallback["model_load_latency_sec"] = whisper.load_latency_sec if fallback["model_loads_for_case"] else 0.0
            fallback["trigger_reasons"] = [item["type"] for item in pre["critical_missing_evidence"] if "speech" in item["type"] or "phrase" in item["type"] or "count" in item["type"]]
        except Exception as exc:
            fallback = no_fallback()
            fallback.update({"triggered": True, "trigger_reasons": ["critical_speech_evidence_missing"], "source_audio_path": str(source_audio), "outcome": "failed_with_error", "warnings": [f"local_asr_model_initialization_error:{type(exc).__name__}:{str(exc)[:240]}"]})
    post_candidates = pre_candidates + copy.deepcopy(fallback["added_candidates"])
    post = classify_evidence(base, plan, cues, post_candidates)
    fallback["pre_fallback_evidence_status"] = pre["evidence_status"]
    fallback["post_fallback_evidence_status"] = post["evidence_status"]
    fallback["pre_fallback_candidates"] = pre_candidates
    fallback["post_fallback_candidates"] = post_candidates
    result = {
        "case_id": base["case_id"],
        "question": base["question"],
        "execution_status": execution_status(base),
        "task5a_plan_summary": plan,
        "deterministic_question_cues": cues,
        "task5b_v1_1_input": {
            "source_candidates": str(SOURCE_CANDIDATES.relative_to(ROOT)).replace("\\", "/"),
            "selected_candidates": copy.deepcopy(base.get("selected_candidates", [])),
            "selected_visual_evidence_frames": copy.deepcopy(base.get("selected_visual_evidence_frames", [])),
            "anchor_resolution": copy.deepcopy(base.get("anchor_resolution", {})),
            "local_visual_refinement": copy.deepcopy(base.get("local_visual_refinement", {})),
            "warnings": copy.deepcopy(base.get("warnings", [])),
        },
        "pre_fallback_evidence_status": pre["evidence_status"],
        "pre_fallback_assessment": pre,
        "fallback": fallback,
        "post_fallback_evidence_status": post["evidence_status"],
        "evidence_status": post["evidence_status"],
        "sufficiency_reason_codes": post["sufficiency_reason_codes"],
        "critical_missing_evidence": post["critical_missing_evidence"],
        "ambiguity_flags": post["ambiguity_flags"],
        "fallback_required": pre["fallback_required"],
        "post_fallback_candidates": post_candidates,
        "recommended_human_check": "Verify semantic relevance and any speaker identity manually. Structural sufficiency does not prove the final answer is correct.",
        "runtime": {
            "task5c_llm_api_calls": 0,
            "task5c_vlm_calls": 0,
            "local_whisper_model_calls": fallback["model_calls"],
            "unique_local_audio_duration_sec": fallback["local_audio_duration_sec"],
            "local_audio_duration_processed_sec": fallback["total_decoding_audio_duration_sec"],
            "fallback_latency_sec": fallback["latency_sec"],
            "total_task5c_latency_sec": time.perf_counter() - started,
            "candidates_before_fallback": len(pre_candidates),
            "candidates_after_fallback": len(post_candidates),
        },
    }
    return result


def status_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {status: sum(record[key] == status for record in records) for status in ("sufficient", "questionable", "insufficient")}


def make_html(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    cards = []
    for record in records:
        cards.append(f'''<article><h2>{record["case_id"]}</h2>
<h3>原始问题</h3><p>{html.escape(record["question"])}</p>
<h3>执行完成 与 证据充分 的区别</h3><p>execution_status = <b>{record["execution_status"]}</b>；evidence_status = <b>{record["evidence_status"]}</b>。后者仅表示结构性证据充分性，不代表最终答案正确。</p>
<h3>Task 5A 计划摘要</h3>{pre(record["task5a_plan_summary"])}
<h3>Task 5B v1.1 已选证据</h3>{pre(record["task5b_v1_1_input"])}
<h3>Fallback 前证据状态</h3>{pre(record["pre_fallback_assessment"])}
<h3>局部 ASR fallback</h3>{pre(record["fallback"])}
<h3>Fallback 后证据状态</h3>{pre({key:record[key] for key in ("post_fallback_evidence_status", "evidence_status", "sufficiency_reason_codes", "critical_missing_evidence", "ambiguity_flags", "fallback_required")})}
<h3>新增成本与延迟</h3>{pre(record["runtime"])}
<h3>Post-hoc 弱参考评估</h3>{pre(record.get("posthoc_weak_reference_evaluation", {"status":"loaded only after Task 5C retrieval/fallback results were saved"}))}
<h3>人工检查建议</h3><p>{html.escape(record["recommended_human_check"])}</p><textarea placeholder="人工备注（不写回源数据）"></textarea></article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 5C v1 证据充分性与局部 ASR fallback</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:480px;overflow:auto}}textarea{{width:100%;height:75px}}</style>
<h1>Task 5C v1：确定性证据充分性检测与局部 Speech fallback</h1><h2>实验概览</h2><p>无 LLM、无 VLM、无 reranking、无最终问答。弱参考仅在候选与 fallback 结果保存后才加载，且不是精确金标准边界。</p>{pre(summary["aggregate"])}{''.join(cards)}'''


def main() -> int:
    argparse.ArgumentParser().parse_args()
    for path in (SOURCE_CANDIDATES, SOURCE_SUMMARY, PLAN_PATH, CONFIG_PATH):
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    source_hash_before, plans_hash_before = sha256(SOURCE_CANDIDATES), sha256(PLAN_PATH)
    base_records = load_jsonl(SOURCE_CANDIDATES)
    task5a = {row["case_id"]: row for row in load_jsonl(PLAN_PATH)}
    missing_plans = sorted({record["case_id"] for record in base_records} - set(task5a))
    if missing_plans:
        raise SystemExit(f"Missing frozen Task 5A v2 plans for: {missing_plans}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    whisper = LazyWhisper(config)
    records = [process_record(base, task5a[base["case_id"]], whisper) for base in base_records]
    # Retrieval and any local fallback are saved before loading post-hoc weak-reference material.
    write_results(records)
    source_summary = load_json(SOURCE_SUMMARY)
    evaluations = source_summary.get("posthoc_evaluation", {})
    for record in records:
        record["posthoc_weak_reference_evaluation"] = copy.deepcopy(evaluations.get(record["case_id"], {}))
    write_results(records)
    source_hash_after, plans_hash_after = sha256(SOURCE_CANDIDATES), sha256(PLAN_PATH)
    if source_hash_before != source_hash_after or plans_hash_before != plans_hash_after:
        raise RuntimeError("Frozen Task 5A/Task 5B v1.1 inputs changed during Task 5C execution")
    aggregate = {
        "case_count": len(records),
        "pre_fallback_evidence_status_counts": status_counts(records, "pre_fallback_evidence_status"),
        "post_fallback_evidence_status_counts": status_counts(records, "post_fallback_evidence_status"),
        "fallback_triggered_case_count": sum(record["fallback"]["triggered"] for record in records),
        "fallback_recovery_count": sum(record["fallback"]["outcome"] == "recovered_evidence" for record in records),
        "still_insufficient_count": sum(record["evidence_status"] == "insufficient" for record in records),
        "total_unique_local_asr_audio_duration_sec": sum(record["runtime"]["unique_local_audio_duration_sec"] for record in records),
        "total_local_asr_audio_duration_sec": sum(record["runtime"]["local_audio_duration_processed_sec"] for record in records),
        "total_local_whisper_model_calls": sum(record["runtime"]["local_whisper_model_calls"] for record in records),
        "total_additional_latency_sec": sum(record["runtime"]["total_task5c_latency_sec"] for record in records),
        "task5c_llm_api_calls": 0,
        "task5c_vlm_calls": 0,
    }
    summary = {
        "task": "Task 5C v1 deterministic evidence sufficiency detection with local speech fallback",
        "retrieval_and_fallback_saved_before_reference_load": True,
        "weak_reference_role": "post-hoc only; copied after results were saved; not a precise gold boundary",
        "no_llm_or_vlm_calls": True,
        "source_inputs": {"task5b_v1_1_candidates": str(SOURCE_CANDIDATES.relative_to(ROOT)).replace("\\", "/"), "task5a_v2_plans": str(PLAN_PATH.relative_to(ROOT)).replace("\\", "/")},
        "input_integrity": {"task5b_v1_1_sha256_before": source_hash_before, "task5b_v1_1_sha256_after": source_hash_after, "task5a_v2_sha256_before": plans_hash_before, "task5a_v2_sha256_after": plans_hash_after, "unchanged": True},
        "aggregate": aggregate,
        "warnings": sorted({warning for record in records for warning in record["fallback"].get("warnings", [])}),
    }
    write_json(OUTPUT / "task5c_summary.json", summary)
    markdown = ["# Task 5C v1 summary", "", "Deterministic structural evidence sufficiency only; this does not claim semantic correctness or produce final answers.", "", f"- Cases: {aggregate['case_count']}", f"- Pre/post evidence status: {aggregate['pre_fallback_evidence_status_counts']} / {aggregate['post_fallback_evidence_status_counts']}", f"- Fallback triggered/recovered/still insufficient: {aggregate['fallback_triggered_case_count']} / {aggregate['fallback_recovery_count']} / {aggregate['still_insufficient_count']}", f"- Local ASR unique crop / decoding-pass audio / model calls / additional latency: {aggregate['total_unique_local_asr_audio_duration_sec']:.3f}s / {aggregate['total_local_asr_audio_duration_sec']:.3f}s / {aggregate['total_local_whisper_model_calls']} / {aggregate['total_additional_latency_sec']:.3f}s", "- Task 5C LLM/VLM calls: 0 / 0", "- References were loaded only after retrieval/fallback records were saved and remain weak post-hoc evaluation.", "- Frozen Task 5A and Task 5B v1.1 input hashes were unchanged.", "", "| case | execution | pre | post | fallback |", "|---|---|---|---|---|"]
    for record in records:
        markdown.append(f"| {record['case_id']} | {record['execution_status']} | {record['pre_fallback_evidence_status']} | {record['post_fallback_evidence_status']} | {record['fallback']['outcome']} |")
    (OUTPUT / "task5c_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (OUTPUT / "task5c_human_review.html").write_text(make_html(records, summary), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
