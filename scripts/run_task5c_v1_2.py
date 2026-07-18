from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.task5c_v1_2 import (  # noqa:E402
    acoustic_evidence_role, broad_score_diagnostic, export_local_clip, recompute_structural_status, timestamp_semantics,
)


V11_RESULTS = ROOT / "outputs/evidence_sufficiency/task5c_v1_1/task5c_results.jsonl"
TASK5B_V11 = ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl"
TASK5B_BASE = ROOT / "outputs/planner_guided_retrieval/task5b_candidates.jsonl"
TASK5A = ROOT / "outputs/question_planner/v2/task5a_plans.jsonl"
TASK5C_V1 = ROOT / "outputs/evidence_sufficiency/task5c_v1/task5c_results.jsonl"
OUT = ROOT / "outputs/evidence_sufficiency/task5c_v1_2"
AUDIO_ROOT = Path("D:/ThesisData/EgoSound/data/Ego4d/audios")


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


def load_without_references(path: Path) -> list[dict[str, Any]]:
    rows, marker = [], ', "pre_fallback_reference_evaluation":'
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index = line.rfind(marker)
        rows.append(json.loads(line[:index] + "}" if index >= 0 else line))
    return rows


def video_ids_by_case() -> dict[str, str]:
    mapping = {}
    for row in (json.loads(line) for line in TASK5B_V11.read_text(encoding="utf-8").splitlines() if line.strip()):
        files = row.get("input_files", {})
        index_path = files.get("coarse_visual_index") or files.get("speech_transcripts") or ""
        video_id = Path(index_path).parent.name if index_path else Path(str(files.get("source_mp4", ""))).stem
        if not video_id:
            raise ValueError(f"Unable to resolve video ID for {row['case_id']}")
        mapping[row["case_id"]] = video_id
    return mapping


def acoustic_candidates(record: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    candidates = record["pre_fallback_candidates"] if stage == "pre" else record["post_fallback_candidates"]
    return [item for item in candidates if item.get("modality") == "acoustic"]


def local_clip_for(candidate: dict[str, Any], case_id: str, video_id: str, role: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    if role not in {"direct_evidence", "temporal_anchor", "resolver"}:
        return None, None, None
    source = AUDIO_ROOT / f"{video_id}.wav"
    start, end = float(candidate["start_time"]), float(candidate["end_time"])
    name = f"{case_id}_{start:.3f}_{end:.3f}_{candidate['candidate_id']}.wav"
    path = OUT / "local_audio_clips" / case_id / name
    try:
        metadata = export_local_clip(source, path, start, end)
        metadata["clip_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        return metadata["clip_path"], metadata, None
    except Exception as exc:
        return None, None, f"local_acoustic_clip_export_error:{type(exc).__name__}:{str(exc)[:240]}"


def decorate_acoustic(record: dict[str, Any], video_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan, question = record["task5a_plan_summary"], record["question"]
    semantics = timestamp_semantics(question, record["deterministic_question_cues"], record["task5b_v1_1_input"].get("anchor_resolution", {}))
    role = acoustic_evidence_role(plan)
    diagnostics, clips = [], []
    for candidate in acoustic_candidates(record, "post"):
        clip_path, clip, warning = local_clip_for(candidate, record["case_id"], video_id, role)
        diagnostic = broad_score_diagnostic(candidate, role=role, semantics=semantics, local_audio_clip_reference=clip_path)
        if warning:
            diagnostic.setdefault("clip_warnings", []).append(warning)
        diagnostics.append(diagnostic)
        if clip:
            clips.append(clip)
    return diagnostics, clips


def source_post_assessment(record: dict[str, Any]) -> dict[str, Any]:
    return {"evidence_status": record["post_fallback_evidence_status"], "sufficiency_reason_codes": record["sufficiency_reason_codes"], "critical_missing_evidence": record["critical_missing_evidence"], "ambiguity_flags": record["ambiguity_flags"]}


def process(record: dict[str, Any], video_id: str) -> dict[str, Any]:
    diagnostics, clips = decorate_acoustic(record, video_id)
    pre = recompute_structural_status(record["pre_fallback_assessment"], diagnostics)
    post = recompute_structural_status(source_post_assessment(record), diagnostics)
    additional = post["evidence_status"] == "insufficient"
    return {
        "case_id": record["case_id"], "question": record["question"], "execution_status": record["execution_status"],
        "task5a_plan_summary": copy.deepcopy(record["task5a_plan_summary"]), "deterministic_question_cues": copy.deepcopy(record["deterministic_question_cues"]),
        "task5b_v1_1_input": copy.deepcopy(record["task5b_v1_1_input"]), "source_task5c_v1_1": {"pre_fallback_evidence_status": record["pre_fallback_evidence_status"], "post_fallback_evidence_status": record["post_fallback_evidence_status"], "fallback": copy.deepcopy(record["fallback"]), "runtime": copy.deepcopy(record["runtime"])},
        "pre_fallback_candidates": copy.deepcopy(record["pre_fallback_candidates"]), "post_fallback_candidates": copy.deepcopy(record["post_fallback_candidates"]),
        "pre_fallback_evidence_status": pre["evidence_status"], "pre_fallback_assessment": pre,
        "post_fallback_evidence_status": post["evidence_status"], "evidence_status": post["evidence_status"], "sufficiency_reason_codes": post["sufficiency_reason_codes"], "critical_missing_evidence": post["critical_missing_evidence"], "ambiguity_flags": post["ambiguity_flags"],
        "fallback_was_triggered": record["fallback_was_triggered"], "fallback_recovered_evidence": record["fallback_recovered_evidence"], "fallback_required": additional, "additional_fallback_required": additional,
        "questionable_followup_policy": "deferred_to_future_ablation", "automatic_additional_fallback_triggered": False,
        "acoustic_evidence_diagnostics": diagnostics, "local_audio_clips": clips,
        "visual_efficiency_accounting": copy.deepcopy(record["visual_efficiency_accounting"]),
        "runtime": {"task5c_llm_api_calls": 0, "task5c_vlm_calls": 0, "new_whisper_model_calls": 0, "new_clap_inference_calls": 0, "local_audio_clip_count": len(clips), "local_audio_clip_duration_sec": sum(item["duration_sec"] for item in clips), "source_task5c_v1_1_runtime": copy.deepcopy(record["runtime"])},
        "recommended_human_check": "Listen to any exported local raw-audio clip for semantic interpretation. Broad CLAP provenance is retained as retrieval history only; speaker and semantic questions remain pending human or future-model review.",
    }


def status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {status: sum(record[field] == status for record in records) for status in ("sufficient", "questionable", "insufficient")}


def make_html(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    cards = []
    for record in records:
        cards.append(f'''<article><h2>{record["case_id"]}</h2><h3>原始问题</h3><p>{html.escape(record["question"])}</p>
<h3>结构性证据状态</h3>{pre({key:record[key] for key in ("execution_status","pre_fallback_evidence_status","post_fallback_evidence_status","fallback_required","additional_fallback_required","questionable_followup_policy","automatic_additional_fallback_triggered")})}
<h3>Acoustic evidence roles、时间语义与 CLAP provenance</h3>{pre(record["acoustic_evidence_diagnostics"])}
<h3>导出的局部原始音频证据</h3>{pre(record["local_audio_clips"])}
<h3>结构性与语义性区分</h3><p>local_audio_evidence_available / structural_evidence_available 只表示局部原始音频可供后续检查；semantic_interpretation_pending 表示尚未进行声音语义判读。</p>
<h3>保留的 Task 5C v1.1 fallback</h3>{pre(record["source_task5c_v1_1"])}
<h3>修正后的视觉效率统计</h3>{pre(record["visual_efficiency_accounting"])}
<h3>Fallback 前弱参考评估</h3>{pre(record.get("pre_fallback_reference_evaluation",{}))}<h3>Fallback 后弱参考评估</h3>{pre(record.get("post_fallback_reference_evaluation",{}))}
<h3>成本与人工检查</h3>{pre(record["runtime"])}<p>{html.escape(record["recommended_human_check"])}</p></article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 5C v1.2 acoustic evidence semantic correction</title><style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:500px;overflow:auto}}</style><h1>Task 5C v1.2：Acoustic evidence 结构性语义修正</h1><p>不重新计算 CLAP；broad-source score 只保留为检索 provenance。弱参考仅在证据结果首次保存后加载。</p>{pre(summary["aggregate"])}{''.join(cards)}'''


def main() -> int:
    argparse.ArgumentParser().parse_args()
    required = [V11_RESULTS, TASK5B_V11, TASK5B_BASE, TASK5A, TASK5C_V1]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    tracked = [V11_RESULTS, TASK5B_V11, TASK5B_BASE, TASK5A, TASK5C_V1]
    before = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    videos = video_ids_by_case()
    records = [process(record, videos[record["case_id"]]) for record in load_without_references(V11_RESULTS)]
    # First evidence write intentionally excludes all weak-reference evaluation fields.
    write_results(records)
    # Post-hoc stage: copy the frozen v1.1 pre/post weak-reference evaluations after evidence is saved.
    source_with_references = {record["case_id"]: record for record in (json.loads(line) for line in V11_RESULTS.read_text(encoding="utf-8").splitlines() if line.strip())}
    for record in records:
        source = source_with_references[record["case_id"]]
        record["pre_fallback_reference_evaluation"] = copy.deepcopy(source["pre_fallback_reference_evaluation"])
        record["post_fallback_reference_evaluation"] = copy.deepcopy(source["post_fallback_reference_evaluation"])
    write_results(records)
    after = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    if before != after:
        raise RuntimeError("Frozen Task 5A/5B/5C input changed during Task 5C v1.2")
    aggregate = {"case_count": len(records), "pre_fallback_evidence_status_counts": status_counts(records, "pre_fallback_evidence_status"), "post_fallback_evidence_status_counts": status_counts(records, "post_fallback_evidence_status"), "local_audio_clip_count": sum(record["runtime"]["local_audio_clip_count"] for record in records), "local_audio_clip_duration_sec": sum(record["runtime"]["local_audio_clip_duration_sec"] for record in records), "broad_source_warning_count": sum(item["broad_source_warning"] for record in records for item in record["acoustic_evidence_diagnostics"]), "broad_source_affects_sufficiency_count": sum(item["broad_source_affects_sufficiency"] for record in records for item in record["acoustic_evidence_diagnostics"]), "new_whisper_model_calls": 0, "new_clap_inference_calls": 0, "task5c_llm_api_calls": 0, "task5c_vlm_calls": 0}
    summary = {"task": "Task 5C v1.2 narrow acoustic evidence semantic correction", "evidence_results_saved_before_reference_load": True, "weak_reference_role": "post-hoc only; never used for roles, timestamp semantics, clip extraction, status, or candidate generation", "no_llm_vlm_whisper_or_new_clap_calls": True, "source_inputs": {"task5c_v1_1": str(V11_RESULTS.relative_to(ROOT)).replace("\\", "/"), "task5b_v1_1": str(TASK5B_V11.relative_to(ROOT)).replace("\\", "/")}, "source_integrity": {"before": before, "after": after, "unchanged": True}, "aggregate": aggregate, "warnings": sorted({warning for record in records for item in record["acoustic_evidence_diagnostics"] for warning in item.get("clip_warnings", [])})}
    write_json(OUT / "task5c_summary.json", summary)
    lines = ["# Task 5C v1.2 summary", "", "Narrow acoustic structural-semantics correction. Broad CLAP provenance remains visible but does not alone downgrade anchored local raw-audio evidence.", "", f"- Pre/post statuses: {aggregate['pre_fallback_evidence_status_counts']} / {aggregate['post_fallback_evidence_status_counts']}", f"- Local clips: {aggregate['local_audio_clip_count']} ({aggregate['local_audio_clip_duration_sec']:.3f}s)", f"- Broad warnings / broad penalties: {aggregate['broad_source_warning_count']} / {aggregate['broad_source_affects_sufficiency_count']}", "- New Whisper/CLAP/LLM/VLM calls: 0 / 0 / 0 / 0", "- Questionable-case follow-up policy: `deferred_to_future_ablation`.", "", "| case | pre | post | clips | broad penalty |", "|---|---|---|---:|---:|"]
    for record in records:
        lines.append(f"| {record['case_id']} | {record['pre_fallback_evidence_status']} | {record['post_fallback_evidence_status']} | {len(record['local_audio_clips'])} | {sum(item['broad_source_affects_sufficiency'] for item in record['acoustic_evidence_diagnostics'])} |")
    (OUT / "task5c_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "task5c_human_review.html").write_text(make_html(records, summary), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
