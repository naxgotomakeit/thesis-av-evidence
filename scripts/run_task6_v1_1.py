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

from src.retrieval.task6_v1_1 import (  # noqa: E402
    classify_candidate_accounting,
    classify_modalities,
    corrected_relations,
    order_visual_frames,
    relation_counts,
    union_duration,
)


INPUT = ROOT / "outputs/relation_reranking/task6_v1/task6_evidence_packets.jsonl"
TASK5C = ROOT / "outputs/evidence_sufficiency/task5c_v1_2/task5c_results.jsonl"
OUT = ROOT / "outputs/relation_reranking/task6_v1_1"
FROZEN_PRIOR_OUTPUTS = [
    ROOT / "outputs/question_planner/v2/task5a_plans.jsonl",
    ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl",
    ROOT / "outputs/evidence_sufficiency/task5c_v1/task5c_results.jsonl",
    ROOT / "outputs/evidence_sufficiency/task5c_v1_1/task5c_results.jsonl",
    TASK5C,
    INPUT,
]


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


def load_v1_without_reference() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove v1's already-computed weak-reference values before correction."""
    packets, posthoc = [], {}
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        packet = json.loads(line)
        posthoc[packet["case_id"]] = packet.pop("posthoc_evaluation", None)
        packets.append(packet)
    return packets, posthoc


def planner_requested_modalities(packet: dict[str, Any]) -> list[str]:
    plan = packet["provenance"]["task5a_plan"]
    requested = set(plan.get("resolver_modalities", []))
    primary = plan.get("primary_anchor_modality")
    if primary in {"speech", "acoustic", "visual"}:
        requested.add(primary)
    return sorted(requested)


def acoustic_diagnostics(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["candidate_id"]: copy.deepcopy(item) for item in packet["provenance"].get("task5c_v1_2_acoustic_diagnostics", [])}


def rebuild_groups(packet: dict[str, Any], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = copy.deepcopy(packet["retained_evidence_groups"])
    candidates = {item["candidate_id"] for item in packet["retained_candidates"]}
    for group in groups:
        group["relations"] = [
            relation for relation in relations
            if relation["source_candidate_id"] in candidates or relation["target_candidate_id"] in candidates
        ]
    return groups


def process(packet: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = copy.deepcopy(packet)
    accounting = classify_candidate_accounting(result)
    ordered_frames = order_visual_frames(result.get("selected_visual_frames", []), result["retained_candidates"])
    result["selected_visual_frames"] = ordered_frames
    for candidate in result["retained_candidates"]:
        if candidate.get("candidate_type") == "canonical_visual_evidence":
            candidate["canonical_visual_frames"] = copy.deepcopy(ordered_frames)

    diagnostics = acoustic_diagnostics(result)
    modality_fields = classify_modalities(
        planner_requested_modalities(result), result["retained_candidates"], diagnostics, accounting["actually_dropped_candidates"]
    )
    relations = corrected_relations(result["retained_candidates"], result["operation"], result["relations"])
    result.pop("required_modalities", None)
    result.pop("required_evidence_modalities", None)
    result.update(modality_fields)
    result.update(accounting)
    result["relations"] = relations
    result["retained_evidence_groups"] = rebuild_groups(result, relations)
    result["runtime"] = {
        "relation_correction_latency_sec": time.perf_counter() - started,
        "llm_api_calls": 0,
        "vlm_calls": 0,
        "whisper_calls": 0,
        "clap_calls": 0,
        "retrieval_calls": 0,
        "media_decoding_calls": 0,
    }
    return result


def make_html(packets: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"

    cards = []
    for packet in packets:
        cards.append(f'''<article><h2>{packet["case_id"]}</h2>
<h3>问题与 operation</h3><p>{html.escape(packet["question"])}</p>
<h3>模态分类</h3>{pre({key: packet[key] for key in ["planner_requested_modalities", "answer_required_modalities", "supporting_modalities", "retained_modalities", "dropped_supporting_modalities"]})}
<h3>保留的 evidence groups</h3>{pre(packet["retained_evidence_groups"])}
<h3>时间关系</h3>{pre([x for x in packet["relations"] if x["relation_basis"] in {"temporal_overlap", "anchor_distance", "transcript_sequence"}])}
<h3>功能性 resolver relations</h3>{pre([x for x in packet["relations"] if x["relation_type"] == "resolves"])}
<h3>合并与转换候选</h3>{pre({"merged_source_candidates": packet["merged_source_candidates"], "transformed_candidates": packet["transformed_candidates"]})}
<h3>真正移除的候选</h3>{pre(packet["actually_dropped_candidates"])}
<h3>移除的视觉帧</h3>{pre(packet["dropped_visual_frames"])}
<h3>视觉帧：selection_rank 与 chronological presentation_order</h3>{pre(packet["selected_visual_frames"])}
<h3>未解决歧义与预算</h3>{pre({"unresolved_ambiguities": packet["unresolved_ambiguities"], "budget_accounting": packet["budget_accounting"]})}
<h3>弱参考后验比较</h3>{pre(packet.get("posthoc_evaluation", {}))}</article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 6 v1.1 evidence packets</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:500px;overflow:auto}}</style>
<h1>Task 6 v1.1：关系与报告正确性修正</h1><p>仅重新表达既有 Task 6 v1 证据；没有检索、媒体读取、模型调用或最终问答。弱参考仅在初始包保存后用于后验展示。</p>{pre(summary["aggregate"])}{''.join(cards)}'''


def main() -> int:
    for path in FROZEN_PRIOR_OUTPUTS:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    tracked = FROZEN_PRIOR_OUTPUTS
    before_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    source_packets, posthoc_by_case = load_v1_without_reference()
    task5c_cases = {json.loads(line)["case_id"] for line in TASK5C.read_text(encoding="utf-8").splitlines() if line.strip()}
    if {packet["case_id"] for packet in source_packets} != task5c_cases:
        raise RuntimeError("Task 6 v1 and Task 5C v1.2 case sets differ")

    packets = [process(packet) for packet in source_packets]
    # Save the corrected packet before weak-reference values are restored.
    write_jsonl(OUT / "task6_evidence_packets.jsonl", packets)
    for packet in packets:
        packet["posthoc_evaluation"] = copy.deepcopy(posthoc_by_case[packet["case_id"]])

    write_jsonl(OUT / "task6_evidence_packets.jsonl", packets)
    write_jsonl(OUT / "task6_merged_candidates.jsonl", [{"case_id": packet["case_id"], **item} for packet in packets for item in packet["merged_source_candidates"] + packet["transformed_candidates"]])
    write_jsonl(OUT / "task6_dropped_candidates.jsonl", [{"case_id": packet["case_id"], **item} for packet in packets for item in packet["actually_dropped_candidates"]])
    write_jsonl(OUT / "task6_dropped_visual_frames.jsonl", [{"case_id": packet["case_id"], **item} for packet in packets for item in packet["dropped_visual_frames"]])
    write_jsonl(OUT / "task6_relations.jsonl", [{"case_id": packet["case_id"], **item} for packet in packets for item in packet["relations"]])

    after_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    if before_hashes != after_hashes:
        raise RuntimeError("Task 6 v1 or Task 5C v1.2 input changed during Task 6 v1.1")
    aggregate = {
        "case_count": len(packets),
        "before_candidate_count": sum(len(packet["candidates_before_reranking"]) for packet in packets),
        "after_candidate_count": sum(len(packet["retained_candidates"]) for packet in packets),
        "merged_candidate_count": sum(len(packet["merged_source_candidates"]) for packet in packets),
        "merged_packet_count": sum(bool(packet["merged_source_candidates"]) for packet in packets),
        "merge_reduction_count": sum(max(0, len(packet["merged_source_candidates"]) - bool(packet["merged_source_candidates"])) for packet in packets),
        "transformed_candidate_count": sum(len(packet["transformed_candidates"]) for packet in packets),
        "actually_dropped_candidate_count": sum(len(packet["actually_dropped_candidates"]) for packet in packets),
        "dropped_visual_frame_count": sum(len(packet["dropped_visual_frames"]) for packet in packets),
        "planner_requested_modalities": sorted({modality for packet in packets for modality in packet["planner_requested_modalities"]}),
        "answer_required_modalities": sorted({modality for packet in packets for modality in packet["answer_required_modalities"]}),
        "retained_modalities": sorted({modality for packet in packets for modality in packet["retained_modalities"]}),
        "dropped_supporting_modalities": sorted({modality for packet in packets for modality in packet["dropped_supporting_modalities"]}),
        "required_modality_retention": all(set(packet["answer_required_modalities"]).issubset(set(packet["retained_modalities"])) for packet in packets),
        "required_role_retention": all(set(packet["required_roles"]).issubset({role for candidate in packet["retained_candidates"] for role in candidate.get("roles", [])}) for packet in packets),
        "ambiguity_retention": all(set(packet["source_unresolved_ambiguities"]).issubset(set(packet["unresolved_ambiguities"])) for packet in packets),
        "fallback_evidence_retention": all(not any(item.get("source") == "local_asr_fallback" for item in packet["candidates_before_reranking"]) or any(item.get("source") == "local_asr_fallback" for item in packet["retained_candidates"]) for packet in packets),
        "before_unique_audio_duration_sec": round(sum(union_duration(packet["candidates_before_reranking"], {"speech", "acoustic"}) for packet in packets), 6),
        "after_unique_audio_duration_sec": round(sum(union_duration(packet["retained_candidates"], {"speech", "acoustic"}) for packet in packets), 6),
        "before_unique_visual_duration_sec": round(sum(union_duration(packet["candidates_before_reranking"], {"visual"}) for packet in packets), 6),
        "after_unique_visual_duration_sec": round(sum(union_duration(packet["retained_candidates"], {"visual"}) for packet in packets), 6),
        "before_visual_frame_count": sum(len(packet.get("selected_visual_frames", [])) + len(packet["dropped_visual_frames"]) for packet in packets),
        "after_visual_frame_count": sum(len(packet["selected_visual_frames"]) for packet in packets),
        "relation_counts": relation_counts([relation for packet in packets for relation in packet["relations"]]),
        "budget_violations": [item for packet in packets for item in packet["budget_accounting"].get("violations", [])],
        "runtime_sec": round(sum(packet["runtime"]["relation_correction_latency_sec"] for packet in packets), 6),
        "llm_api_calls": 0, "vlm_calls": 0, "whisper_calls": 0, "clap_calls": 0, "retrieval_calls": 0, "media_decoding_calls": 0,
    }
    summary = {
        "task": "Task 6 v1.1 isolated relation and reporting correction",
        "packets_saved_before_reference_restore": True,
        "weak_reference_role": "post-hoc display only; never used for relation construction, accounting, selection, or budget",
        "source_integrity": {"before": before_hashes, "after": after_hashes, "unchanged": True},
        "candidate_count_reconciliation": "before = after + actually_dropped + merge_reduction; transformed representations preserve candidate count",
        "aggregate": aggregate,
    }
    write_json(OUT / "task6_summary.json", summary)
    markdown = [
        "# Task 6 v1.1 summary", "", "Isolated semantic/reporting correction; no retrieval, media decode, or model call.", "",
        f"- Candidates before/after: {aggregate['before_candidate_count']} / {aggregate['after_candidate_count']}",
        f"- Merged source / transformed / actually dropped: {aggregate['merged_candidate_count']} / {aggregate['transformed_candidate_count']} / {aggregate['actually_dropped_candidate_count']}",
        f"- Dropped visual frames: {aggregate['dropped_visual_frame_count']}; visual frames before/after: {aggregate['before_visual_frame_count']} / {aggregate['after_visual_frame_count']}",
        f"- Required modality / role retention: {aggregate['required_modality_retention']} / {aggregate['required_role_retention']}",
        f"- Relation counts: {aggregate['relation_counts']}",
        f"- Runtime: {aggregate['runtime_sec']:.6f}s; model/API/retrieval/media calls: 0",
    ]
    (OUT / "task6_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (OUT / "task6_human_review.html").write_text(make_html(packets, summary), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
