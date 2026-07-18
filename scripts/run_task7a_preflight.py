from __future__ import annotations

import copy
import hashlib
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_qa.task7a_preflight import (  # noqa: E402
    FINAL_ANSWER_POLICY,
    REQUIRED_OUTPUT_SCHEMA,
    build_pipeline_uncertainties,
    count_types,
    effective_bounds,
    path_is_file,
    payload_leakage_audit,
    resolved_path,
    union_duration,
    validate_visual_frames,
)


INPUT = ROOT / "outputs/relation_reranking/task6_v1_2/task6_evidence_packets.jsonl"
OUT = ROOT / "outputs/final_qa_preflight/task7a_v1"
FROZEN_PRIOR_OUTPUTS = [
    ROOT / "outputs/question_planner/v2/task5a_plans.jsonl",
    ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl",
    ROOT / "outputs/evidence_sufficiency/task5c_v1/task5c_results.jsonl",
    ROOT / "outputs/evidence_sufficiency/task5c_v1_1/task5c_results.jsonl",
    ROOT / "outputs/evidence_sufficiency/task5c_v1_2/task5c_results.jsonl",
    ROOT / "outputs/relation_reranking/task6_v1/task6_evidence_packets.jsonl",
    ROOT / "outputs/relation_reranking/task6_v1_1/task6_evidence_packets.jsonl",
    INPUT,
]

MODEL_REQUIREMENTS = {
    "text_input": True,
    "multi_image_input": True,
    "raw_audio_input": True,
    "structured_json_output": True,
    "single_request_per_question_preferred": True,
    "supports_mixed_text_image_audio_request": True,
    "modality_specific_exceptions": [
        "Speech-only cases may use transcript text without raw audio.",
        "Visual plus speech cases require images and transcript text.",
        "Acoustic direct-evidence cases require raw local audio.",
        "No model may answer an acoustic-quality question from CLAP score alone.",
    ],
}


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


def load_packets_without_posthoc() -> list[dict[str, Any]]:
    packets = []
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            packet = json.loads(line)
            packet.pop("posthoc_evaluation", None)
            packets.append(packet)
    return packets


def visual_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    frames = [
        {
            "frame_path": frame["canonical_frame_path"],
            "timestamp_sec": float(frame["timestamp"]),
            "presentation_order": frame["presentation_order"],
            "selection_rank": frame["selection_rank"],
            "anchor_distance_sec": frame.get("anchor_distance_sec"),
        }
        for frame in candidate["canonical_visual_frames"]
    ]
    return {
        "evidence_id": candidate["candidate_id"], "start_sec": float(candidate["start_time"]), "end_sec": float(candidate["end_time"]),
        "frames": frames, "roles": copy.deepcopy(candidate["roles"]),
    }


def speech_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    start, end = effective_bounds(candidate)
    return {
        "evidence_id": candidate["candidate_id"], "transcript": candidate.get("transcript_text", ""), "start_sec": start, "end_sec": end,
        "roles": copy.deepcopy(candidate["roles"]), "exact_phrase_match": candidate.get("exact_phrase_match"),
        "phrase_match_method": candidate.get("phrase_match_method"), "phrase_match_score": candidate.get("phrase_match_score"),
        "fallback_provenance": candidate.get("source") == "local_asr_fallback" or "fallback_recovered" in candidate.get("roles", []),
        "timestamp_validity": candidate.get("timestamp_validity", "valid"),
        "speaker_attribution_warning": candidate.get("speaker_attribution_warning"), "speaker_verified": False,
    }


def acoustic_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": candidate["candidate_id"], "audio_clip_path": candidate.get("local_audio_clip_reference"),
        "start_sec": float(candidate["start_time"]), "end_sec": float(candidate["end_time"]), "roles": copy.deepcopy(candidate["roles"]),
        "acoustic_evidence_role": candidate.get("acoustic_evidence_role"), "semantic_interpretation_pending": bool(candidate.get("semantic_interpretation_pending", True)),
        "clap_retrieval_provenance": {
            "clap_similarity_score": candidate.get("clap_similarity_score"), "score_scope": candidate.get("score_scope"),
            "broad_source_warning": candidate.get("broad_source_warning"), "local_acoustic_score_recomputed": candidate.get("local_acoustic_score_recomputed", False),
            "not_semantic_verification": True,
        },
    }


def build_payload(packet: dict[str, Any]) -> dict[str, Any]:
    candidates = {candidate["candidate_id"]: candidate for candidate in packet["retained_candidates"]}
    groups = []
    for source_group in packet["retained_evidence_groups"]:
        group = {"group_id": source_group["group_id"], "group_type": source_group["group_type"], "visual_evidence": [], "speech_evidence": [], "acoustic_evidence": [], "relations": copy.deepcopy(source_group["relations"]), "unresolved_ambiguities": copy.deepcopy(source_group["unresolved_ambiguities"]), "missing_information": copy.deepcopy(source_group["missing_information"])}
        for source_candidate in source_group["retained_candidates"]:
            candidate = candidates[source_candidate["candidate_id"]]
            if candidate["modality"] == "visual":
                group["visual_evidence"].append(visual_evidence(candidate))
            elif candidate["modality"] == "speech":
                group["speech_evidence"].append(speech_evidence(candidate))
            elif candidate["modality"] == "acoustic":
                group["acoustic_evidence"].append(acoustic_evidence(candidate))
        groups.append(group)
    payload = {
        "case_id": packet["case_id"], "question": packet["question"], "operation": packet["operation"],
        "answer_required_modalities": copy.deepcopy(packet["answer_required_modalities"]), "supporting_modalities": copy.deepcopy(packet["supporting_modalities"]),
        "dataset_or_query_inconsistency_status": packet.get("dataset_or_query_inconsistency_status", "unknown"),
        "evidence_groups": groups, "pipeline_uncertainties": build_pipeline_uncertainties(packet),
        "final_answer_policy": copy.deepcopy(FINAL_ANSWER_POLICY), "required_output_schema": copy.deepcopy(REQUIRED_OUTPUT_SCHEMA),
    }
    return payload


def flatten(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [item for group in payload["evidence_groups"] for item in group[key]]


def preflight(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    visual = flatten(payload, "visual_evidence")
    frames = [frame for item in visual for frame in item["frames"]]
    per_visual_validation = [validate_visual_frames(item["frames"], ROOT) for item in visual]
    visual_validation = {
        "chronological_visual_order": all(item["chronological_visual_order"] for item in per_visual_validation),
        "presentation_order_sequential": all(item["presentation_order_sequential"] for item in per_visual_validation),
        "no_duplicate_normalized_timestamps": all(item["no_duplicate_normalized_timestamps"] for item in per_visual_validation),
        "invalid_paths": [path for item in per_visual_validation for path in item["invalid_paths"]],
    }
    speech = flatten(payload, "speech_evidence")
    acoustic = flatten(payload, "acoustic_evidence")
    invalid_audio = [item.get("audio_clip_path") for item in acoustic if not path_is_file(ROOT, item.get("audio_clip_path"))]
    available = set()
    if visual and not visual_validation["invalid_paths"] and visual_validation["chronological_visual_order"]:
        available.add("visual")
    if speech and all(float(item["start_sec"]) <= float(item["end_sec"]) and item.get("timestamp_validity") != "excluded_outside_decode_interval" for item in speech):
        available.add("speech")
    if acoustic and not invalid_audio:
        available.add("acoustic")
    required = set(payload["answer_required_modalities"])
    missing_modalities = sorted(required - available)
    evidence_ids = [item["evidence_id"] for item in visual + speech + acoustic]
    leakage = payload_leakage_audit(payload)
    invalid_paths = visual_validation["invalid_paths"] + [str(path) for path in invalid_audio if path]
    missing_assets = [f"required_modality:{item}" for item in missing_modalities]
    critical = [item for item in payload["pipeline_uncertainties"] if item["severity"] == "critical"]
    material = [item for item in payload["pipeline_uncertainties"] if item["severity"] == "material"]
    if missing_assets or invalid_paths or not leakage["leakage_check_passed"] or len(evidence_ids) != len(set(evidence_ids)) or not visual_validation["presentation_order_sequential"] or not visual_validation["no_duplicate_normalized_timestamps"]:
        status = "blocked"
    elif material or critical:
        status = "warning"
    else:
        status = "ready"
    asset_paths = {frame["frame_path"] for frame in frames} | {item["audio_clip_path"] for item in acoustic if item.get("audio_clip_path")}
    asset_bytes = sum(resolved_path(ROOT, path).stat().st_size for path in asset_paths if resolved_path(ROOT, path) and resolved_path(ROOT, path).is_file())
    metrics = {
        "visual_frame_count": len(frames), "visual_evidence_duration_sec": round(union_duration(visual), 6),
        "acoustic_clip_count": len(acoustic), "unique_acoustic_duration_sec": round(union_duration(acoustic), 6),
        "speech_segment_count": len(speech), "transcript_character_count": sum(len(item["transcript"]) for item in speech),
        "evidence_group_count": len(payload["evidence_groups"]), "relation_count": sum(len(group["relations"]) for group in payload["evidence_groups"]),
        "uncertainty_count": len(payload["pipeline_uncertainties"]), "total_local_asset_bytes": asset_bytes,
        "expected_final_model_api_calls": 1, "actual_final_model_api_calls": 0,
    }
    result = {
        "case_id": payload["case_id"], "status": status, "required_modalities": payload["answer_required_modalities"], "available_modalities": sorted(available),
        "missing_assets": missing_assets, "invalid_paths": invalid_paths, "chronological_visual_order": visual_validation["chronological_visual_order"],
        "audio_clip_available": not invalid_audio, "speech_timestamps_valid": "speech" in available or not speech,
        "critical_uncertainties": critical, "leakage_check_passed": leakage["leakage_check_passed"], "notes": [],
        "consistency": {"unique_evidence_ids": len(evidence_ids) == len(set(evidence_ids)), **visual_validation}, "planned_input_size": metrics,
    }
    if material:
        result["notes"].append("Material pipeline uncertainty is intentionally preserved for the final model.")
    if critical:
        result["notes"].append("Critical uncertainty is present; final answer policy must not force an answer.")
    return result, leakage


def make_html(payloads: list[dict[str, Any]], results: list[dict[str, Any]], audits: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    by_case = {item["case_id"]: item for item in results}
    audit_by_case = {item["case_id"]: item for item in audits}
    cards = []
    for payload in payloads:
        result = by_case[payload["case_id"]]
        cards.append(f'''<article><h2>{payload["case_id"]}</h2><h3>问题与 operation</h3><p>{html.escape(payload["question"])}</p>
<h3>必要与支持模态</h3>{pre({"answer_required_modalities": payload["answer_required_modalities"], "supporting_modalities": payload["supporting_modalities"]})}
<h3>精确模型输入：视觉帧、Speech、局部 Audio</h3>{pre(payload["evidence_groups"])}
<h3>继承的不确定性与可能 answer statuses</h3>{pre({"pipeline_uncertainties": payload["pipeline_uncertainties"], "final_answer_policy": payload["final_answer_policy"]})}
<h3>资产验证与预检状态</h3>{pre(result)}
<h3>泄漏审计</h3>{pre(audit_by_case[payload["case_id"]])}
<h3>计划输入规模</h3>{pre(result["planned_input_size"])}
</article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 7A final QA preflight</title><style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:560px;overflow:auto}}</style><h1>Task 7A：最终多模态问答预检与载荷验证</h1><p>这是干运行：仅验证精确模型输入，不调用模型，不生成答案，不进行检索或基准评估。</p>{pre(summary["aggregate"])}{''.join(cards)}'''


def main() -> int:
    for path in FROZEN_PRIOR_OUTPUTS:
        if not path.is_file():
            raise SystemExit(f"Missing required frozen input: {path}")
    before_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in FROZEN_PRIOR_OUTPUTS}
    packets = load_packets_without_posthoc()
    payloads, results, audits = [], [], []
    for packet in packets:
        payload = build_payload(packet)
        result, audit = preflight(payload)
        if not audit["leakage_check_passed"]:
            result["status"] = "blocked"
        payloads.append(payload); results.append(result); audits.append({"case_id": payload["case_id"], **audit})
    # Model-facing payloads are saved before any post-hoc reference stage.  Task 7A has no such stage.
    write_jsonl(OUT / "task7a_payloads.jsonl", payloads)
    write_jsonl(OUT / "task7a_preflight_results.jsonl", results)
    write_jsonl(OUT / "task7_leakage_audit.jsonl", audits)
    after_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in FROZEN_PRIOR_OUTPUTS}
    if before_hashes != after_hashes:
        raise RuntimeError("A frozen Task 5/6 input changed during Task 7A")
    status_counts = count_types([{"type": result["status"]} for result in results])
    modality_combinations = {result["case_id"]: "+".join(result["required_modalities"]) for result in results}
    aggregate = {
        "case_count": len(payloads), "status_counts": status_counts, "modality_combinations": modality_combinations,
        "total_visual_frames": sum(item["planned_input_size"]["visual_frame_count"] for item in results),
        "total_unique_acoustic_duration_sec": round(sum(item["planned_input_size"]["unique_acoustic_duration_sec"] for item in results), 6),
        "total_speech_segments": sum(item["planned_input_size"]["speech_segment_count"] for item in results),
        "uncertainty_type_counts": count_types([uncertainty for payload in payloads for uncertainty in payload["pipeline_uncertainties"]]),
        "missing_asset_count": sum(len(item["missing_assets"]) for item in results), "invalid_path_count": sum(len(item["invalid_paths"]) for item in results),
        "leakage_checks_passed": all(item["leakage_check_passed"] for item in results),
        "expected_final_model_api_calls": len(payloads), "actual_final_model_api_calls": 0,
        "llm_api_calls": 0, "vlm_calls": 0, "whisper_calls": 0, "clap_calls": 0, "retrieval_calls": 0, "media_decoding_calls": 0,
    }
    summary = {"task": "Task 7A final multimodal QA preflight and payload validation", "payloads_saved_without_weak_references": True, "source_integrity": {"before": before_hashes, "after": after_hashes, "unchanged": True}, "aggregate": aggregate}
    write_json(OUT / "task7_model_requirements.json", MODEL_REQUIREMENTS)
    write_json(OUT / "task7_output_schema.json", REQUIRED_OUTPUT_SCHEMA)
    write_json(OUT / "task7_uncertainty_policy.json", {"pipeline_uncertainty_types": ["temporal_uncertainty", "semantic_uncertainty", "speaker_attribution_uncertainty", "source_identity_uncertainty", "evidence_missing", "multiple_plausible_answers", "query_premise_uncertainty", "fallback_recovered_evidence", "dataset_or_query_inconsistency_unknown"], "final_answer_policy": FINAL_ANSWER_POLICY})
    write_json(OUT / "task7a_summary.json", summary)
    markdown = ["# Task 7A preflight summary", "", "Dry-run payload construction only; no model, retrieval, media decoding, or benchmark evaluation.", "", f"- Ready/warning/blocked: {aggregate['status_counts']}", f"- Planned frames/audio duration/speech segments: {aggregate['total_visual_frames']} / {aggregate['total_unique_acoustic_duration_sec']:.3f}s / {aggregate['total_speech_segments']}", f"- Leakage audits passed: {aggregate['leakage_checks_passed']}; missing/invalid assets: {aggregate['missing_asset_count']} / {aggregate['invalid_path_count']}", f"- Expected final calls: {aggregate['expected_final_model_api_calls']}; actual calls: 0", f"- Pipeline uncertainty types: {aggregate['uncertainty_type_counts']}"]
    (OUT / "task7a_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (OUT / "task7a_human_review.html").write_text(make_html(payloads, results, audits, summary), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
