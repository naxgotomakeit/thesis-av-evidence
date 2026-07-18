from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.task6_relation_reranking import (  # noqa:E402
    apply_packet_budget, deduplicate_visual_frames, normalize_text, overlap_seconds, stable_id, temporal_gap, temporal_relations, union_duration,
)


INPUT = ROOT / "outputs/evidence_sufficiency/task5c_v1_2/task5c_results.jsonl"
TASK5B_V11 = ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl"
TASK5A = ROOT / "outputs/question_planner/v2/task5a_plans.jsonl"
MANIFEST = ROOT / "data/manifests/mvp_cases_6.json"
CONFIG = ROOT / "configs/task6_relation_reranking.yaml"
OUT = ROOT / "outputs/relation_reranking/task6_v1"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_without_references(path: Path) -> list[dict[str, Any]]:
    rows, marker = [], ', "pre_fallback_reference_evaluation":'
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index = line.rfind(marker)
        rows.append(json.loads(line[:index] + "}" if index >= 0 else line))
    return rows


def quoted_phrases(cues: dict[str, Any]) -> list[str]:
    return [normalize_text(item["text"]) for item in cues.get("quoted_phrases", []) if item.get("text")]


def exact_phrase(candidate: dict[str, Any], phrases: list[str]) -> bool:
    text = normalize_text(candidate.get("transcript_text", ""))
    return any(phrase and phrase in text for phrase in phrases)


def visual_anchor_time(candidates: list[dict[str, Any]]) -> float | None:
    anchors = [item for item in candidates if any(role in item.get("roles", []) for role in ("trigger", "temporal_anchor"))]
    if not anchors:
        return None
    return sum((float(item["start_time"]) + float(item["end_time"])) / 2 for item in anchors) / len(anchors)


def normalize_nonvisual(record: dict[str, Any]) -> list[dict[str, Any]]:
    operation = record["task5a_plan_summary"]["answer_requirement"]["operation"]
    phrases = quoted_phrases(record["deterministic_question_cues"])
    diagnostics = {item["candidate_id"]: item for item in record.get("acoustic_evidence_diagnostics", [])}
    source = [copy.deepcopy(item) for item in record["post_fallback_candidates"] if item.get("modality") != "visual"]
    trigger_end = None
    if operation == "measure_delay":
        triggers = [item for item in source if item.get("modality") == "speech" and exact_phrase(item, phrases)]
        if triggers:
            trigger_end = min(float(item["end_time"]) for item in triggers)
    output = []
    for item in source:
        roles: list[str] = []
        if item.get("modality") == "acoustic":
            diagnostic = diagnostics.get(item["candidate_id"], {})
            roles.append(diagnostic.get("acoustic_evidence_role", "not_required"))
            item.update({key: copy.deepcopy(value) for key, value in diagnostic.items() if key not in {"candidate_id", "selected_acoustic_interval"}})
            item["local_audio_clip_reference"] = diagnostic.get("local_audio_clip_reference")
        elif item.get("modality") == "speech":
            phrase = exact_phrase(item, phrases)
            if operation == "measure_delay":
                if phrase:
                    roles.append("trigger")
                elif trigger_end is not None and float(item["start_time"]) >= trigger_end:
                    roles.extend(["plausible_response", "alternative"])
            elif operation == "count_occurrences":
                roles.append("direct_evidence")
                if item.get("source") == "local_asr_fallback":
                    roles.append("fallback_recovered")
            elif phrase:
                roles.append("temporal_anchor")
            else:
                roles.append("supporting")
            item["exact_phrase_match"] = phrase
            item["valid_asr_timestamps"] = item.get("timestamp_validity", "valid") not in {"excluded_outside_decode_interval", "clipped_to_decode_interval"}
            if item.get("source") == "local_asr_fallback" and "fallback_recovered" not in roles:
                roles.append("fallback_recovered")
        item["roles"] = sorted(set(roles or ["not_required"]))
        item["provenance"] = {"original_candidate": copy.deepcopy(item)}
        output.append(item)
    return output


def merge_visual(record: dict[str, Any], candidates: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    source_visual = [copy.deepcopy(item) for item in record["post_fallback_candidates"] if item.get("modality") == "visual"]
    frames = copy.deepcopy(record["task5b_v1_1_input"].get("selected_visual_evidence_frames", []))
    if not source_visual and not frames:
        return None, [], []
    selected_frames, dropped_frames = deduplicate_visual_frames(frames, int(config["max_visual_frames_per_evidence_group"]), visual_anchor_time(candidates))
    starts = [float(item["start_time"]) for item in source_visual] or [float(item["timestamp"]) for item in selected_frames]
    ends = [float(item["end_time"]) for item in source_visual] or [float(item["timestamp"]) for item in selected_frames]
    source_ids = [item["candidate_id"] for item in source_visual]
    merged = {"candidate_id": stable_id("visual_packet", record["case_id"], *source_ids), "candidate_type": "canonical_visual_evidence", "modality": "visual", "start_time": min(starts), "end_time": max(ends), "roles": ["resolver"], "canonical_visual_frames": selected_frames, "dense_visual_provenance": any(frame.get("provenance") in {"dense_frame", "both"} for frame in selected_frames), "source_candidate_ids": source_ids, "provenance": {"merged_visual_micro_windows": source_visual, "canonical_frames_from_task5b_v1_1": selected_frames}}
    dropped = [{"candidate_id": item["candidate_id"], "reason": "overlapping_visual_window_merged" if len(source_visual) > 1 else "visual_micro_window_represented_by_canonical_frames"} for item in source_visual]
    dropped.extend(dropped_frames)
    return merged, dropped, selected_frames


def remove_redundant_supporting(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = [item for item in candidates if any(role in item.get("roles", []) for role in ("trigger", "temporal_anchor", "direct_evidence", "resolver", "plausible_response", "fallback_recovered"))]
    retained, dropped = [], []
    for item in candidates:
        if item.get("roles") == ["supporting"] and required:
            dropped.append({"candidate_id": item["candidate_id"], "reason": "redundant_supporting_acoustic_candidate"})
        else:
            retained.append(item)
    return retained, dropped


def build_groups(record: dict[str, Any], candidates: list[dict[str, Any]], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operation = record["task5a_plan_summary"]["answer_requirement"]["operation"]
    groups = []
    if operation == "measure_delay":
        members = [item for item in candidates if any(role in item["roles"] for role in ("trigger", "plausible_response"))]
        if members:
            groups.append({"group_id": stable_id("group", record["case_id"], "trigger_response"), "operation": operation, "group_type": "trigger_response", "primary_anchor": next((item["candidate_id"] for item in members if "trigger" in item["roles"]), None), "retained_candidates": members, "relations": [relation for relation in relations if relation["source_candidate_id"] in {item["candidate_id"] for item in members} or relation["target_candidate_id"] in {item["candidate_id"] for item in members}], "unresolved_ambiguities": record["ambiguity_flags"], "missing_information": [], "why_this_group_is_needed": "Preserves the trigger and plausible response alternatives without selecting a speaker or response as correct."})
    elif any("resolver" in item["roles"] for item in candidates):
        members = [item for item in candidates if any(role in item["roles"] for role in ("temporal_anchor", "resolver"))]
        groups.append({"group_id": stable_id("group", record["case_id"], "anchor_resolver"), "operation": operation, "group_type": "anchor_resolver", "primary_anchor": next((item["candidate_id"] for item in members if "temporal_anchor" in item["roles"]), None), "retained_candidates": members, "relations": [relation for relation in relations if relation["source_candidate_id"] in {item["candidate_id"] for item in members} or relation["target_candidate_id"] in {item["candidate_id"] for item in members}], "unresolved_ambiguities": record["ambiguity_flags"], "missing_information": [], "why_this_group_is_needed": "Combines a temporal anchor with visual resolver evidence without inferring an answer."})
    else:
        members = list(candidates)
        group_type = "fallback_recovery" if any("fallback_recovered" in item["roles"] for item in members) else "single_modality_evidence"
        groups.append({"group_id": stable_id("group", record["case_id"], group_type), "operation": operation, "group_type": group_type, "primary_anchor": next((item["candidate_id"] for item in members if "fallback_recovered" in item["roles"]), None), "retained_candidates": members, "relations": relations, "unresolved_ambiguities": record["ambiguity_flags"], "missing_information": [], "why_this_group_is_needed": "Retains compact direct evidence; it is explicitly a single-modality group when no cross-modal relation is present."})
    return groups


def evaluate(candidates: list[dict[str, Any]], reference: tuple[float, float]) -> dict[str, Any]:
    overlaps = [max(0.0, min(float(item["end_time"]), reference[1]) - max(float(item["start_time"]), reference[0])) for item in candidates]
    distances = [max(0.0, reference[0] - float(item["end_time"]), float(item["start_time"]) - reference[1]) for item in candidates]
    hit = any(value > 0 for value in overlaps)
    return {"reference_role": "post-hoc weak reference only; not used for reranking or selection", "reference_interval_hit": hit, "maximum_temporal_overlap_sec": max(overlaps, default=0.0), "minimum_temporal_distance_sec": min(distances, default=None), "weak_temporal_recall_preserved": hit, "candidate_count": len(candidates)}


def duration_for_modalities(candidates: list[dict[str, Any]], modalities: set[str]) -> float:
    """Temporal union for all candidates of the requested modalities."""
    return union_duration([item for item in candidates if item.get("modality") in modalities])


def process(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    before = [copy.deepcopy(item) for item in record["post_fallback_candidates"]]
    normalized = normalize_nonvisual(record)
    visual, visual_drops, selected_frames = merge_visual(record, normalized, config)
    if visual is not None:
        normalized.append(visual)
    protected_roles = {"trigger", "temporal_anchor", "direct_evidence", "resolver", "plausible_response", "fallback_recovered"}
    required_roles = sorted({role for item in normalized for role in item.get("roles", []) if role in protected_roles})
    required_evidence_modalities = sorted({item["modality"] for item in normalized if set(item.get("roles", [])) & protected_roles})
    normalized, supporting_drops = remove_redundant_supporting(normalized)
    retained, budget_drops, budget = apply_packet_budget(normalized, config)
    relations = temporal_relations(retained, record["task5a_plan_summary"]["answer_requirement"]["operation"])
    groups = build_groups(record, retained, relations)
    all_drops = visual_drops + supporting_drops + budget_drops
    return {"case_id": record["case_id"], "question": record["question"], "operation": record["task5a_plan_summary"]["answer_requirement"]["operation"], "required_modalities": record["task5a_plan_summary"]["resolver_modalities"], "required_evidence_modalities": required_evidence_modalities, "required_roles": required_roles, "candidates_before_reranking": before, "retained_evidence_groups": groups, "retained_candidates": retained, "dropped_candidates": all_drops, "candidate_drop_reasons": all_drops, "relations": relations, "selected_visual_frames": selected_frames, "local_audio_clips": copy.deepcopy(record.get("local_audio_clips", [])), "speech_segments": [item for item in retained if item.get("modality") == "speech"], "unresolved_ambiguities": copy.deepcopy(record["ambiguity_flags"]), "source_unresolved_ambiguities": copy.deepcopy(record["ambiguity_flags"]), "missing_information": [], "structural_evidence_status": record["evidence_status"], "questionable_followup_policy": record["questionable_followup_policy"], "dataset_or_query_inconsistency_status": "unknown", "budget_accounting": budget, "provenance": {"task5c_v1_2_source": str(INPUT.relative_to(ROOT)).replace("\\", "/"), "task5a_plan": copy.deepcopy(record["task5a_plan_summary"]), "task5c_v1_2_acoustic_diagnostics": copy.deepcopy(record.get("acoustic_evidence_diagnostics", [])), "fallback_history": copy.deepcopy(record.get("source_task5c_v1_1", {}))}, "runtime": {"reranking_latency_sec": time.perf_counter() - started, "llm_api_calls": 0, "vlm_calls": 0, "whisper_calls": 0, "clap_calls": 0}}


def write_packets(packets: list[dict[str, Any]]) -> None:
    write_jsonl(OUT / "task6_evidence_packets.jsonl", packets)
    write_jsonl(OUT / "task6_dropped_candidates.jsonl", [{"case_id": packet["case_id"], **drop} for packet in packets for drop in packet["dropped_candidates"]])
    write_jsonl(OUT / "task6_relations.jsonl", [{"case_id": packet["case_id"], **relation} for packet in packets for relation in packet["relations"]])


def make_html(packets: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    cards = []
    for packet in packets:
        cards.append(f'''<article><h2>{packet["case_id"]}</h2><h3>问题与 operation</h3><p>{html.escape(packet["question"])}</p>{pre({"operation":packet["operation"],"required_modalities":packet["required_modalities"]})}
<h3>Reranking 前候选</h3>{pre(packet["candidates_before_reranking"])}<h3>保留的 evidence groups</h3>{pre(packet["retained_evidence_groups"])}<h3>显式 relations</h3>{pre(packet["relations"])}<h3>被移除候选与原因</h3>{pre(packet["dropped_candidates"])}
<h3>视觉帧缩减</h3>{pre(packet["selected_visual_frames"])}<h3>保留的 Audio / Speech evidence</h3>{pre({"local_audio_clips":packet["local_audio_clips"],"speech_segments":packet["speech_segments"]})}
<h3>未解决歧义与预算</h3>{pre({"unresolved_ambiguities":packet["unresolved_ambiguities"],"budget_accounting":packet["budget_accounting"]})}<h3>弱参考后验比较</h3>{pre(packet.get("posthoc_evaluation",{}))}<h3>延迟与调用</h3>{pre(packet["runtime"])} </article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 6 relation-aware evidence packets</title><style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:500px;overflow:auto}}</style><h1>Task 6：关系感知的紧凑证据包</h1><p>仅重新组织现有候选；不检索、不调用模型、不生成最终答案。弱参考在证据包首次保存后才用于后验比较。</p>{pre(summary["aggregate"])}{''.join(cards)}'''


def make_html_v2(packets: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Chinese review page; unicode escapes avoid host-console encoding changes."""
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"

    cards = []
    for packet in packets:
        cards.append(
            f'''<article><h2>{packet["case_id"]}</h2>
<h3>\u95ee\u9898\u4e0e operation</h3><p>{html.escape(packet["question"])}</p>{pre({"operation": packet["operation"], "required_modalities": packet["required_modalities"], "required_roles": packet["required_roles"]})}
<h3>Reranking \u524d\u5019\u9009</h3>{pre(packet["candidates_before_reranking"])}
<h3>\u4fdd\u7559\u7684 evidence groups</h3>{pre(packet["retained_evidence_groups"])}
<h3>\u663e\u5f0f relations</h3>{pre(packet["relations"])}
<h3>\u88ab\u79fb\u9664\u5019\u9009\u4e0e\u539f\u56e0</h3>{pre(packet["dropped_candidates"])}
<h3>\u89c6\u89c9\u5e27\u7f29\u51cf</h3>{pre(packet["selected_visual_frames"])}
<h3>\u4fdd\u7559\u7684 Audio / Speech evidence</h3>{pre({"local_audio_clips": packet["local_audio_clips"], "speech_segments": packet["speech_segments"]})}
<h3>\u672a\u89e3\u51b3\u6b67\u4e49\u4e0e\u9884\u7b97</h3>{pre({"unresolved_ambiguities": packet["unresolved_ambiguities"], "budget_accounting": packet["budget_accounting"]})}
<h3>\u5f31\u53c2\u8003\u540e\u9a8c\u6bd4\u8f83</h3>{pre(packet.get("posthoc_evaluation", {}))}
<h3>\u5ef6\u8fdf\u4e0e\u8c03\u7528</h3>{pre(packet["runtime"])}
</article>'''
        )
    return f'''<!doctype html><meta charset=utf-8><title>Task 6 relation-aware evidence packets</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:500px;overflow:auto}}</style>
<h1>Task 6\uff1a\u5173\u7cfb\u611f\u77e5\u7684\u7d27\u51d1\u8bc1\u636e\u5305</h1>
<p>\u4ec5\u91cd\u65b0\u7ec4\u7ec7\u73b0\u6709\u5019\u9009\uff1b\u4e0d\u68c0\u7d22\u3001\u4e0d\u8c03\u7528\u6a21\u578b\u3001\u4e0d\u751f\u6210\u6700\u7ec8\u7b54\u6848\u3002\u5f31\u53c2\u8003\u5728\u8bc1\u636e\u5305\u9996\u6b21\u4fdd\u5b58\u540e\u624d\u7528\u4e8e\u540e\u9a8c\u6bd4\u8f83\u3002</p>
{pre(summary["aggregate"])}{''.join(cards)}'''


def main() -> int:
    argparse.ArgumentParser().parse_args()
    for path in (INPUT, TASK5B_V11, TASK5A, MANIFEST, CONFIG):
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    tracked = [INPUT, TASK5B_V11, TASK5A]
    before_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    packets = [process(record, config) for record in load_without_references(INPUT)]
    # First packet write contains no weak-reference field.
    write_packets(packets)
    # Post-hoc weak-reference stage begins only after packets are saved.
    references = {row["case_id"]: (float(row["provided_timestamp_start"]), float(row["provided_timestamp_end"])) for row in load(MANIFEST)}
    for packet in packets:
        packet["posthoc_evaluation"] = {"before_task5c_v1_2": evaluate(packet["candidates_before_reranking"], references[packet["case_id"]]), "after_task6_packet": evaluate(packet["retained_candidates"], references[packet["case_id"]])}
    write_packets(packets)
    after_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    if before_hashes != after_hashes:
        raise RuntimeError("Frozen Task 5A/5B/5C input changed during Task 6")
    aggregate = {
        "case_count": len(packets),
        "before_candidate_count": sum(len(packet["candidates_before_reranking"]) for packet in packets),
        "after_candidate_count": sum(len(packet["retained_candidates"]) for packet in packets),
        "before_modality_count": sum(len({item.get("modality") for item in packet["candidates_before_reranking"]}) for packet in packets),
        "after_modality_count": sum(len({item.get("modality") for item in packet["retained_candidates"]}) for packet in packets),
        "before_unique_audio_duration_sec": round(sum(duration_for_modalities(packet["candidates_before_reranking"], {"speech", "acoustic"}) for packet in packets), 6),
        "after_unique_audio_duration_sec": round(sum(duration_for_modalities(packet["retained_candidates"], {"speech", "acoustic"}) for packet in packets), 6),
        "before_unique_visual_duration_sec": round(sum(duration_for_modalities(packet["candidates_before_reranking"], {"visual"}) for packet in packets), 6),
        "after_unique_visual_duration_sec": round(sum(duration_for_modalities(packet["retained_candidates"], {"visual"}) for packet in packets), 6),
        "before_unique_evidence_duration_sec": round(sum(union_duration(packet["candidates_before_reranking"]) for packet in packets), 6),
        "after_unique_evidence_duration_sec": round(sum(union_duration(packet["retained_candidates"]) for packet in packets), 6),
        "before_visual_frame_count": sum(len(packet["provenance"].get("task5c_v1_2_acoustic_diagnostics", [])) * 0 for packet in packets) + sum(len(record.get("task5b_v1_1_input", {}).get("selected_visual_evidence_frames", [])) for record in load_without_references(INPUT)),
        "after_visual_frame_count": sum(len(packet["selected_visual_frames"]) for packet in packets),
        "before_speech_segment_count": sum(sum(item.get("modality") == "speech" for item in packet["candidates_before_reranking"]) for packet in packets),
        "after_speech_segment_count": sum(len(packet["speech_segments"]) for packet in packets),
        "before_local_audio_clip_count": sum(len(record.get("local_audio_clips", [])) for record in load_without_references(INPUT)),
        "after_local_audio_clip_count": sum(len(packet["local_audio_clips"]) for packet in packets),
        "required_modality_retention": all(set(packet["required_evidence_modalities"]) <= {item.get("modality") for item in packet["retained_candidates"]} for packet in packets),
        "required_role_retention": all(set(packet["required_roles"]) <= {role for item in packet["retained_candidates"] for role in item.get("roles", [])} for packet in packets),
        "ambiguity_retention": all(set(packet["source_unresolved_ambiguities"]).issubset(set(packet["unresolved_ambiguities"])) for packet in packets),
        "fallback_evidence_retention": all(not any(item.get("source") == "local_asr_fallback" for item in packet["candidates_before_reranking"]) or any(item.get("source") == "local_asr_fallback" for item in packet["retained_candidates"]) for packet in packets),
        "budget_violations": [violation for packet in packets for violation in packet["budget_accounting"]["violations"]],
        "reranking_latency_sec": sum(packet["runtime"]["reranking_latency_sec"] for packet in packets),
        "llm_api_calls": 0, "vlm_calls": 0, "whisper_calls": 0, "clap_calls": 0,
    }
    summary = {"task": "Task 6 v1 deterministic relation-aware reranking and compact evidence packets", "packets_saved_before_reference_load": True, "weak_reference_role": "post-hoc only; never used to rank, retain, drop, or restore evidence", "source_integrity": {"before": before_hashes, "after": after_hashes, "unchanged": True}, "config": config, "aggregate": aggregate}
    write_json(OUT / "task6_summary.json", summary)
    lines = ["# Task 6 v1 summary", "", "Deterministic packet construction only; no new retrieval or model call.", "", f"- Candidates before/after: {aggregate['before_candidate_count']} / {aggregate['after_candidate_count']}", f"- Visual frames before/after: {aggregate['before_visual_frame_count']} / {aggregate['after_visual_frame_count']}", f"- Unique audio duration before/after: {aggregate['before_unique_audio_duration_sec']:.3f}s / {aggregate['after_unique_audio_duration_sec']:.3f}s", f"- Unique visual duration before/after: {aggregate['before_unique_visual_duration_sec']:.3f}s / {aggregate['after_unique_visual_duration_sec']:.3f}s", f"- Unique all-modality evidence duration before/after: {aggregate['before_unique_evidence_duration_sec']:.3f}s / {aggregate['after_unique_evidence_duration_sec']:.3f}s", f"- Required modality / role retention: {aggregate['required_modality_retention']} / {aggregate['required_role_retention']}", f"- Reranking latency: {aggregate['reranking_latency_sec']:.6f}s; model/API calls: 0", f"- Budget violations: {aggregate['budget_violations'] or 'none'}", "", "| case | before candidates | retained | groups | dropped |", "|---|---:|---:|---:|---:|"]
    for packet in packets:
        lines.append(f"| {packet['case_id']} | {len(packet['candidates_before_reranking'])} | {len(packet['retained_candidates'])} | {len(packet['retained_evidence_groups'])} | {len(packet['dropped_candidates'])} |")
    (OUT / "task6_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "task6_human_review.html").write_text(make_html(packets, summary), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
