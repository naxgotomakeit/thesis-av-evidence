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

from src.retrieval.task6_v1_2 import (  # noqa: E402
    candidate_count_consistent,
    chronological_frames,
    correct_response_relations,
    corrected_budget_accounting,
    merge_reduction_count,
    relation_counts,
)


INPUT = ROOT / "outputs/relation_reranking/task6_v1_1/task6_evidence_packets.jsonl"
TASK6_V1 = ROOT / "outputs/relation_reranking/task6_v1/task6_evidence_packets.jsonl"
OUT = ROOT / "outputs/relation_reranking/task6_v1_2"


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


def load_without_posthoc() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packets, posthoc = [], {}
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            packet = json.loads(line)
            posthoc[packet["case_id"]] = packet.pop("posthoc_evaluation", None)
            packets.append(packet)
    return packets, posthoc


def candidate_identity(packet: dict[str, Any]) -> list[tuple[str, float, float]]:
    return [(item["candidate_id"], float(item["start_time"]), float(item["end_time"])) for item in packet["retained_candidates"]]


def frame_identity(packet: dict[str, Any]) -> list[tuple[float, str | None, int | None, float | None]]:
    return sorted((float(item["timestamp"]), item.get("canonical_frame_path"), item.get("selection_rank"), item.get("anchor_distance_sec")) for item in packet.get("selected_visual_frames", []))


def update_groups(packet: dict[str, Any]) -> None:
    candidate_by_id = {item["candidate_id"]: item for item in packet["retained_candidates"]}
    for group in packet["retained_evidence_groups"]:
        member_ids = {item["candidate_id"] for item in group.get("retained_candidates", [])}
        for member in group.get("retained_candidates", []):
            updated = candidate_by_id.get(member["candidate_id"])
            if updated and updated.get("candidate_type") == "canonical_visual_evidence":
                member["canonical_visual_frames"] = copy.deepcopy(updated["canonical_visual_frames"])
        group["relations"] = [
            copy.deepcopy(relation) for relation in packet["relations"]
            if relation["source_candidate_id"] in member_ids or relation["target_candidate_id"] in member_ids
        ]


def process(source: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    packet = copy.deepcopy(source)
    frames = chronological_frames(packet.get("selected_visual_frames", []))
    packet["selected_visual_frames"] = frames
    for candidate in packet["retained_candidates"]:
        if candidate.get("candidate_type") == "canonical_visual_evidence":
            candidate["canonical_visual_frames"] = copy.deepcopy(frames)
    packet["relations"] = correct_response_relations(packet["relations"], emit_trigger_of=True)
    packet["budget_accounting"] = corrected_budget_accounting(packet["budget_accounting"], packet["actually_dropped_candidates"])
    packet["consistency_checks"] = {
        "dropped_candidate_count_matches_actually_dropped": packet["budget_accounting"]["dropped_candidate_count"] == len(packet["actually_dropped_candidates"]),
        "candidate_count_equation": candidate_count_consistent(packet),
    }
    if not all(packet["consistency_checks"].values()):
        raise AssertionError(f"Task 6 v1.2 accounting inconsistency for {packet['case_id']}")
    update_groups(packet)
    packet["runtime"] = {
        "serialization_patch_latency_sec": time.perf_counter() - started,
        "llm_api_calls": 0, "vlm_calls": 0, "whisper_calls": 0, "clap_calls": 0,
        "retrieval_calls": 0, "media_decoding_calls": 0,
    }
    return packet


def make_html(packets: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    cards = []
    for packet in packets:
        responses = [item for item in packet["relations"] if item["relation_type"] in {"response_to", "trigger_of", "alternative_to"}]
        cards.append(f'''<article><h2>{packet["case_id"]}</h2><h3>实际序列化的视觉帧顺序</h3>{pre(packet["selected_visual_frames"])}
<h3>selection_rank 与 presentation_order</h3><p>帧按时间戳展示；selection_rank 保留原选择优先级。</p>
<h3>trigger_of / response_to 方向</h3>{pre(responses)}
<h3>每案真正删除候选计数</h3>{pre({"dropped_candidate_count": packet["budget_accounting"]["dropped_candidate_count"], "actually_dropped_candidates": packet["actually_dropped_candidates"], "consistency_checks": packet["consistency_checks"]})}
<h3>保留的 evidence groups</h3>{pre(packet["retained_evidence_groups"])}
<h3>弱参考后验比较</h3>{pre(packet.get("posthoc_evaluation", {}))}</article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 6 v1.2 patch review</title><style>body{{font:15px system-ui;margin:2rem;max-width:1450px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:2.5rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:500px;overflow:auto}}</style><h1>Task 6 v1.2：序列化与关系方向修复</h1><p>仅修复既有包的帧序列化、关系方向和删除计数；没有检索、媒体读取、模型调用或新证据选择。</p>{pre(summary["aggregate"])}{''.join(cards)}'''


def main() -> int:
    tracked = [TASK6_V1, INPUT]
    for path in tracked:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")
    before_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    sources, posthoc = load_without_posthoc()
    packets = [process(source) for source in sources]
    for source, packet in zip(sources, packets, strict=True):
        invariants = {
            "retained_candidate_ids_and_intervals_unchanged": candidate_identity(source) == candidate_identity(packet),
            "selected_frame_identity_unchanged": frame_identity(source) == frame_identity(packet),
            "local_audio_clips_unchanged": source["local_audio_clips"] == packet["local_audio_clips"],
            "ambiguity_fields_unchanged": source["unresolved_ambiguities"] == packet["unresolved_ambiguities"] and source["source_unresolved_ambiguities"] == packet["source_unresolved_ambiguities"],
            "modality_classifications_unchanged": all(source[key] == packet[key] for key in ("planner_requested_modalities", "answer_required_modalities", "supporting_modalities", "retained_modalities", "dropped_supporting_modalities")),
        }
        if not all(invariants.values()):
            raise AssertionError(f"Task 6 v1.2 invariant failed for {packet['case_id']}: {invariants}")
        packet["serialization_patch_invariants"] = invariants
    # Save before restoring v1.1's already-computed post-hoc values.
    write_jsonl(OUT / "task6_evidence_packets.jsonl", packets)
    for packet in packets:
        packet["posthoc_evaluation"] = copy.deepcopy(posthoc[packet["case_id"]])
    write_jsonl(OUT / "task6_evidence_packets.jsonl", packets)
    write_jsonl(OUT / "task6_relations.jsonl", [{"case_id": packet["case_id"], **relation} for packet in packets for relation in packet["relations"]])
    write_jsonl(OUT / "task6_dropped_candidates.jsonl", [{"case_id": packet["case_id"], **item} for packet in packets for item in packet["actually_dropped_candidates"]])
    after_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked}
    if before_hashes != after_hashes:
        raise RuntimeError("Task 6 v1 or v1.1 input changed during Task 6 v1.2")
    aggregate = {
        "case_count": len(packets),
        "before_candidate_count": sum(len(packet["candidates_before_reranking"]) for packet in packets),
        "after_candidate_count": sum(len(packet["retained_candidates"]) for packet in packets),
        "per_case_dropped_candidate_counts": {packet["case_id"]: packet["budget_accounting"]["dropped_candidate_count"] for packet in packets},
        "all_consistency_checks_passed": all(all(packet["consistency_checks"].values()) and all(packet["serialization_patch_invariants"].values()) for packet in packets),
        "relation_counts": relation_counts([relation for packet in packets for relation in packet["relations"]]),
        "runtime_sec": round(sum(packet["runtime"]["serialization_patch_latency_sec"] for packet in packets), 6),
        "llm_api_calls": 0, "vlm_calls": 0, "whisper_calls": 0, "clap_calls": 0, "retrieval_calls": 0, "media_decoding_calls": 0,
    }
    summary = {
        "task": "Task 6 v1.2 minimal serialization and relation-direction patch",
        "packets_saved_before_reference_restore": True,
        "weak_reference_role": "post-hoc only; not used by this patch",
        "source_integrity": {"before": before_hashes, "after": after_hashes, "unchanged": True},
        "aggregate": aggregate,
    }
    write_json(OUT / "task6_summary.json", summary)
    markdown = [
        "# Task 6 v1.2 summary", "", "Serialization and relation-direction patch only; no retrieval or model call.", "",
        f"- Candidates before/after: {aggregate['before_candidate_count']} / {aggregate['after_candidate_count']}",
        f"- Per-case genuine drop counts: {aggregate['per_case_dropped_candidate_counts']}",
        f"- Relation counts: {aggregate['relation_counts']}",
        f"- All consistency checks: {aggregate['all_consistency_checks_passed']}",
        f"- Runtime: {aggregate['runtime_sec']:.6f}s; model/API/retrieval/media calls: 0",
    ]
    (OUT / "task6_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (OUT / "task6_human_review.html").write_text(make_html(packets, summary), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
