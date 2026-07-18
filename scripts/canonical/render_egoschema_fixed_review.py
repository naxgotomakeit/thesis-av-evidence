"""Render the completed fixed EgoSchema pilot as a human visual audit.

This module reads durable artifacts only. It does not call models, mutate the
historical invalid run, or alter canonical research behavior.
"""

from __future__ import annotations

import html
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/egoschema_comparison_pilot.json"
OLD = ROOT / "outputs/egoschema/ours_v0/protocol_b_pilot_25"
OUT = ROOT / "outputs/egoschema/ours_v0/protocol_b_pilot_25_fixed_v0_1"
PAID = OUT / "smoke/per_case"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def h(value: Any) -> str:
    return html.escape("N/A" if value is None else str(value))


def asset_path(raw: str, *, case_id: str, coarse: bool = False) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if coarse and not raw.startswith("outputs/"):
        return ROOT / "outputs/visual_index" / case_id / path
    return ROOT / path


def img(raw: str | None, label: str, case_id: str, *, coarse: bool = False) -> str:
    if not raw:
        return ""
    path = asset_path(raw, case_id=case_id, coarse=coarse)
    if not path.is_file():
        return f"<figure class='missing'><div>missing</div><figcaption>{h(label)}</figcaption></figure>"
    rel = os.path.relpath(path, OUT).replace("\\", "/")
    return (
        f"<figure><img loading='lazy' src='{h(rel)}' alt='{h(label)}'>"
        f"<figcaption>{h(label)}</figcaption></figure>"
    )


def interval(item: dict[str, Any]) -> tuple[float, float]:
    start = float(item.get("start_time", item.get("start_sec", 0.0)))
    end = float(item.get("end_time", item.get("end_sec", start)))
    return start, end


def bar(start: float, end: float, duration: float, css: str, title: str) -> str:
    left = max(0.0, min(100.0, 100.0 * start / duration))
    width = max(0.28, 100.0 * max(0.0, end - start) / duration)
    return (
        f"<i class='mark {css}' style='left:{left:.4f}%;width:{width:.4f}%' "
        f"title='{h(title)}'></i>"
    )


def point(timestamp: float, duration: float, css: str, title: str) -> str:
    left = max(0.0, min(99.7, 100.0 * timestamp / duration))
    return f"<i class='point {css}' style='left:{left:.4f}%' title='{h(title)}'></i>"


def timeline_row(label: str, marks: str) -> str:
    return (
        "<div class='timeline-row'><div class='timeline-label'>"
        f"{h(label)}</div><div class='track'>{marks}</div></div>"
    )


def timing_value(runtime: dict[str, Any], key: str) -> float:
    return float((runtime.get("model_usage") or {}).get(key) or 0.0)


def old_stage(case_id: str, stage_name: str) -> float:
    record = load(OLD / "per_case" / case_id / "runtime_result.json")
    for item in record.get("timings", []):
        if item.get("stage_name") == stage_name and item.get("executed"):
            return float(item.get("duration_sec") or 0.0)
    return 0.0


def composed_latency(
    case_id: str, trace: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, float]:
    """Compose measured, non-overlapping stages under safe fixed-run reuse."""
    planner = old_stage(case_id, "question_planner")
    query_encode = old_stage(case_id, "visual_query_encode")
    similarity = old_stage(case_id, "visual_similarity_search")
    deterministic = float(
        trace.get("zero_call_replay", {}).get(
            "deterministic_task7a_replay_latency_sec", 0.0
        )
    )
    request = timing_value(runtime, "request_build_sec")
    gemini = timing_value(runtime, "final_model_api_stage_sec")
    parse = timing_value(runtime, "structured_output_parse_sec")
    validation = timing_value(runtime, "local_validation_sec")
    total = sum(
        (planner, query_encode, similarity, deterministic, request, gemini, parse, validation)
    )
    return {
        "planner_sec": planner,
        "visual_query_encode_sec": query_encode,
        "visual_similarity_search_sec": similarity,
        "corrected_deterministic_task7a_replay_sec": deterministic,
        "final_request_build_sec": request,
        "final_gemini_sec": gemini,
        "structured_parse_sec": parse,
        "local_validation_sec": validation,
        "composed_online_latency_sec": total,
    }


def semantic_alignment(trace: dict[str, Any]) -> dict[str, Any]:
    initial = trace["initial_retrieval"]["modalities"]["visual"][
        "initial_retrieval_candidates"
    ]
    top = initial[0] if initial else None
    micro = trace["refinement"]["local_visual"].get("micro_windows", [])
    final = [
        frame
        for evidence in trace.get("final_selected_evidence", [])
        for frame in evidence.get("frames", [])
    ]
    anchor = trace.get("task6_visual_frame_selection_anchor") or {}
    top_overlap = bool(
        top
        and any(
            max(float(top["start_time"]), interval(item)[0])
            < min(float(top["end_time"]), interval(item)[1])
            for item in micro
        )
    )
    anchor_time = anchor.get("timestamp_sec")
    final_aligned = bool(
        anchor_time is not None
        and final
        and max(abs(float(frame["timestamp_sec"]) - float(anchor_time)) for frame in final)
        <= 1.0
    )
    top_survives = bool(
        top
        and any(
            float(top["start_time"])
            <= float(frame["timestamp_sec"])
            <= float(top["end_time"])
            for frame in final
        )
    )
    return {
        "semantic_top1_micro_overlap": top_overlap,
        "final_frames_align_task6_anchor": final_aligned,
        "semantic_top1_survives_to_final": top_survives,
        "correct_evidence_present_upstream": "not_determined_no_temporal_ground_truth",
    }


def diagnose(
    trace: dict[str, Any], evaluation: dict[str, Any], question_type: str
) -> dict[str, Any]:
    alignment = semantic_alignment(trace)
    if evaluation["correct"]:
        return {
            "stage": "none",
            "explanation": "预测正确；未分配失败阶段。",
            **alignment,
        }
    answer_status = evaluation.get("answer_status")
    initial = trace["initial_retrieval"]["modalities"]["visual"].get(
        "initial_retrieval_candidates", []
    )
    micro = trace["refinement"]["local_visual"].get("micro_windows", [])
    dense = trace["refinement"]["local_visual"].get("dense_frames", [])
    sufficiency = trace.get("sufficiency_fallback", {})
    if not initial:
        stage = "coarse semantic retrieval"
        reason = "没有产生可用的初始视觉候选。"
    elif not micro or not alignment["semantic_top1_micro_overlap"]:
        stage = "micro-window refinement"
        reason = "micro-window 未保留语义 Top-1 区域。"
    elif not dense:
        stage = "local dense refinement"
        reason = "局部 dense refinement 没有产生可用帧。"
    elif sufficiency.get("post_status") != "sufficient":
        stage = "Task5C sufficiency"
        reason = (
            f"Task5C post_status={sufficiency.get('post_status')}；证据状态本身不充分。"
        )
    elif not alignment["final_frames_align_task6_anchor"]:
        stage = "Task6 compaction / 4-frame budget"
        reason = "最终帧未与 Task6 语义 anchor 对齐。"
    elif answer_status != "answered":
        stage = "ambiguous evidence"
        reason = "上游时间传播对齐，但最终策略选择 abstain/非回答状态。"
    elif question_type in {"global_summary", "temporal_or_change"}:
        stage = "Task6 compaction / 4-frame budget"
        reason = (
            "这是全局/时序问题；最终仅 4 帧（约 1.5 秒跨度）可能无法表达完整过程。"
            "该判断是结构诊断，不证明上游候选包含正确答案证据。"
        )
    else:
        stage = "unclear"
        reason = (
            "语义 Top-1、micro-window 与最终 anchor 均对齐；仅凭选项标签无法证明"
            "正确证据是否在上游，因此不自动归咎 Gemini。"
        )
    return {"stage": stage, "explanation": reason, **alignment}


def load_complete() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load(MANIFEST)
    records = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        trace_path = OUT / "per_case" / case_id / "task7a_replay.json"
        runtime_path = PAID / case_id / "runtime_result.json"
        evaluation_path = PAID / case_id / "posthoc_evaluation.json"
        missing = [
            str(path)
            for path in (trace_path, runtime_path, evaluation_path)
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(f"Incomplete fixed case {case_id}: {missing}")
        trace = load(trace_path)
        runtime = load(runtime_path)
        evaluation = load(evaluation_path)
        failure = diagnose(trace, evaluation, case["question_type"])
        latency = composed_latency(case_id, trace, runtime)
        records.append(
            {
                "case": case,
                "trace": trace,
                "runtime": runtime,
                "evaluation": evaluation,
                "failure": failure,
                "latency": latency,
            }
        )
    if len(records) != 25:
        raise RuntimeError(f"Expected 25 complete cases, got {len(records)}")
    return records, manifest


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(item["evaluation"]["correct"]) for item in records)
    abstain = sum(
        item["evaluation"].get("predicted_option_index") is None for item in records
    )
    wrong = len(records) - correct - abstain
    answered = len(records) - abstain
    latencies = [item["latency"]["composed_online_latency_sec"] for item in records]
    frame_counts = [
        sum(
            len(evidence.get("frames", []))
            for evidence in item["trace"].get("final_selected_evidence", [])
        )
        for item in records
    ]
    by_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item["case"]["question_type"]].append(item)
    for question_type, items in sorted(grouped.items()):
        type_correct = sum(row["evaluation"]["correct"] for row in items)
        by_type[question_type] = {
            "cases": len(items),
            "correct": type_correct,
            "accuracy": type_correct / len(items),
        }
    journal_files = list((OUT / "provider_attempt_journal").glob("*/final_model_api_*.json"))
    journal_records = [load(path) for path in journal_files]
    technical_failures = sum(
        any(str(state).startswith("technical_failure") for state in row.get("history", []))
        for row in journal_records
    )
    failures = Counter(
        item["failure"]["stage"]
        for item in records
        if not item["evaluation"]["correct"]
    )
    summary = {
        "evaluation_validity": "valid_fixed_temporal_evidence_propagation",
        "cases_completed": len(records),
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "mc_accuracy": correct / len(records),
        "answered_only_accuracy": correct / answered if answered else None,
        "answer_status_distribution": dict(
            Counter(item["evaluation"].get("answer_status") for item in records)
        ),
        "accuracy_by_question_type": by_type,
        "online_latency": {
            "definition": (
                "composed measured stages under safe reuse: unchanged saved planner/query-"
                "encode/search timings + corrected deterministic Task7A replay + new Task7B"
            ),
            "mean_sec": statistics.fmean(latencies),
            "median_sec": statistics.median(latencies),
            "p95_sec": percentile(latencies, 0.95),
            "min_sec": min(latencies),
            "max_sec": max(latencies),
        },
        "mean_model_facing_frames": statistics.fmean(frame_counts),
        "planner_calls_fixed_run": 0,
        "gemini_calls": len(journal_records),
        "technical_failure_attempts": technical_failures,
        "dominant_failure_stages": dict(failures.most_common()),
        "semantic_alignment": {
            "final_frames_align_anchor": sum(
                item["failure"]["final_frames_align_task6_anchor"] for item in records
            ),
            "semantic_top1_survives_to_final": sum(
                item["failure"]["semantic_top1_survives_to_final"] for item in records
            ),
            "correct_evidence_upstream_presence": (
                "not directly verifiable: EgoSchema supplies answer-option labels, not temporal evidence annotations"
            ),
        },
    }
    return summary


def evidence_rows(items: list[dict[str, Any]], decision: str) -> str:
    if not items:
        return "<tr><td colspan='7'>无记录</td></tr>"
    rows = []
    for item in items:
        start, end = interval(item)
        roles = ", ".join(item.get("roles", [])) or item.get("candidate_type", "")
        score = item.get("similarity_score", item.get("query_rank", "N/A"))
        reason = item.get("drop_reason") or item.get("reason") or decision
        rows.append(
            "<tr>"
            f"<td>{h(item.get('candidate_id', item.get('evidence_id')))}</td>"
            f"<td>{h(item.get('modality'))}</td><td>{start:.1f}–{end:.1f}s</td>"
            f"<td>{h(roles)}</td><td>{h(score)}</td><td>{h(decision)}</td>"
            f"<td>{h(reason)}</td></tr>"
        )
    return "".join(rows)


def render(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    navigation = []
    overview = []
    sections = []
    for record in records:
        case = record["case"]
        trace = record["trace"]
        runtime = record["runtime"]
        evaluation = record["evaluation"]
        failure = record["failure"]
        case_id = str(case["case_id"])
        duration = float(case.get("duration_sec") or 180.0)
        initial = trace["initial_retrieval"]["modalities"]["visual"][
            "initial_retrieval_candidates"
        ]
        micro = trace["refinement"]["local_visual"].get("micro_windows", [])
        dense = trace["refinement"]["local_visual"].get("dense_frames", [])
        task6 = trace.get("reranking", {})
        before = task6.get("before", [])
        retained = task6.get("retained", [])
        dropped = task6.get("dropped", [])
        anchor = trace.get("task6_visual_frame_selection_anchor") or {}
        final = [
            frame
            for evidence in trace.get("final_selected_evidence", [])
            for frame in evidence.get("frames", [])
        ]
        predicted = evaluation.get("predicted_option_index")
        result_class = "correct" if evaluation["correct"] else ("abstain" if predicted is None else "wrong")
        result_label = "CORRECT" if evaluation["correct"] else ("ABSTAIN" if predicted is None else "WRONG")
        confidence = (runtime.get("validated_answer") or {}).get("confidence") or {}
        navigation.append(f"<a class='{result_class}' href='#{case_id}'>{h(case_id[:8])}</a>")
        overview.append(
            f"<tr class='{result_class}'><td><a href='#{case_id}'>{h(case_id)}</a></td>"
            f"<td>{h(case['question_type'])}</td><td>{h(result_label)}</td>"
            f"<td>{h(failure['stage'])}</td></tr>"
        )

        coarse_marks = "".join(
            bar(
                float(item["start_time"]),
                float(item["end_time"]),
                duration,
                "coarse top" if int(item["rank"]) <= 3 else "coarse",
                f"R{item['rank']} score={float(item['similarity_score']):.4f} {item['start_time']}-{item['end_time']}s",
            )
            for item in initial
        )
        semantic_marks = "".join(
            bar(
                float(item["start_time"]),
                float(item["end_time"]),
                duration,
                f"semantic r{item['rank']}",
                f"semantic R{item['rank']} score={float(item['similarity_score']):.4f}",
            )
            for item in initial[:3]
        )
        micro_marks = "".join(
            bar(*interval(item), duration, "micro", f"{item.get('candidate_id')} score={item.get('similarity_score')}")
            for item in micro
        )
        dense_marks = ""
        if dense:
            dense_marks += bar(
                float(dense[0]["timestamp"]),
                float(dense[-1]["timestamp"]),
                duration,
                "dense",
                f"dense coverage {dense[0]['timestamp']}-{dense[-1]['timestamp']}s ({len(dense)} frames)",
            )
        task5_marks = "".join(
            bar(*interval(item), duration, "task5", str(item.get("candidate_id")))
            for item in before
        )
        task6_marks = "".join(
            bar(*interval(item), duration, "task6", str(item.get("candidate_id")))
            for item in retained
        )
        final_marks = "".join(
            point(float(frame["timestamp_sec"]), duration, "final", f"final order {frame.get('presentation_order')}")
            for frame in final
        )
        if anchor.get("timestamp_sec") is not None:
            final_marks += point(float(anchor["timestamp_sec"]), duration, "anchor", f"anchor={anchor.get('source')}")
        timeline = (
            "<div class='timeline'><div class='ticks'><span>0s</span><span>30</span><span>60</span><span>90</span><span>120</span><span>150</span><span>180s</span></div>"
            + timeline_row("Coarse regions", coarse_marks)
            + timeline_row("Semantic Top-K", semantic_marks)
            + timeline_row("Micro-windows", micro_marks)
            + timeline_row("Dense coverage", dense_marks)
            + timeline_row("Task5C evidence", task5_marks)
            + timeline_row("Task6 retained", task6_marks)
            + timeline_row("Final frames", final_marks)
            + "</div>"
        )

        coarse_gallery = "".join(
            img(
                item.get("representative_keyframe_path"),
                f"R{item['rank']} {item['start_time']}-{item['end_time']}s score={float(item['similarity_score']):.4f}",
                case_id,
                coarse=True,
            )
            for item in initial[:5]
        )
        micro_gallery = "".join(
            img(
                item.get("representative_frame_path"),
                f"{item.get('candidate_id')} {interval(item)[0]}-{interval(item)[1]}s parent={','.join(item.get('coarse_region_provenance', []))}",
                case_id,
            )
            for item in micro
        )
        dense_gallery = ""
        if dense:
            sample_indexes = sorted(
                {round(i * (len(dense) - 1) / 5) for i in range(6)}
                | {
                    min(
                        range(len(dense)),
                        key=lambda index: abs(
                            float(dense[index]["timestamp"]) - float(frame["timestamp_sec"])
                        ),
                    )
                    for frame in final
                }
            )
            final_times = {float(frame["timestamp_sec"]) for frame in final}
            dense_gallery = "".join(
                img(
                    dense[index].get("frame_path"),
                    f"dense {dense[index]['timestamp']}s"
                    + (" — RETAINED BY TASK6" if float(dense[index]["timestamp"]) in final_times else ""),
                    case_id,
                )
                for index in sample_indexes
            )
        final_gallery = "".join(
            img(
                frame.get("frame_path"),
                f"order={frame.get('presentation_order')} t={frame['timestamp_sec']}s selection_rank={frame.get('selection_rank')}",
                case_id,
            )
            for frame in sorted(final, key=lambda value: value.get("presentation_order", 0))
        )
        options = "".join(
            f"<li class='{'predicted-option' if predicted == index else ''} {'gold-option' if evaluation['gold_option_index'] == index else ''}'>"
            f"<b>{index}</b> {h(option)}</li>"
            for index, option in enumerate(case["options"])
        )
        suff = trace.get("sufficiency_fallback", {})
        suff_table = evidence_rows(before, "entering Task5C")
        task6_before = evidence_rows(before, "entering Task6")
        task6_after = evidence_rows(retained, "retained") + evidence_rows(dropped, "dropped")
        sections.append(
            f"<section id='{case_id}' class='case {result_class}' data-result='{result_class}' data-type='{h(case['question_type'])}' data-failure='{h(failure['stage'])}'>"
            f"<header><div><h2>{h(case_id)}</h2><span class='badge {result_class}'>{result_label}</span>"
            f"<span class='badge neutral'>{h(case['question_type'])}</span></div><a href='#top'>↑ Summary</a></header>"
            f"<h3>Question</h3><p class='question'>{h(case['question'])}</p><ol start='0' class='options'>{options}</ol>"
            f"<div class='answer-grid'><div><b>Prediction</b><p>{h(evaluation.get('predicted_option_text') if predicted is not None else 'ABSTAIN')}</p></div>"
            f"<div><b>Gold (post-hoc only)</b><p>{h(evaluation['gold_option_text'])}</p></div>"
            f"<div><b>Status / confidence</b><p>{h(evaluation.get('answer_status'))} / {h(confidence.get('level'))}</p></div></div>"
            f"<h3>Aligned 0–{duration:.0f}s evidence flow</h3>{timeline}"
            f"<h3>Coarse semantic retrieval</h3><div class='gallery'>{coarse_gallery}</div>"
            f"<h3>Micro-window refinement</h3><p>Top-1 overlap={h(record['failure']['semantic_top1_micro_overlap'])}</p><div class='gallery'>{micro_gallery}</div>"
            f"<h3>Dense local refinement</h3><p>{len(dense)} dense frames; shown as six range samples plus every final retained frame.</p><div class='gallery'>{dense_gallery}</div>"
            f"<h3>Task5C sufficiency</h3><div class='decision'><b>{h(suff.get('post_status', '')).upper()}</b> · fallback={h(suff.get('fallback_decision', {}).get('required'))} · reasons={h(suff.get('reason_codes'))} · uncertainty={h(suff.get('ambiguity_flags'))}</div>"
            f"<table><thead><tr><th>ID</th><th>Modality</th><th>Time</th><th>Role</th><th>Score/rank</th><th>State</th><th>Reason</th></tr></thead><tbody>{suff_table}</tbody></table>"
            f"<h3>Task6 before / after</h3><div class='anchor-card'><b>Anchor source:</b> {h(anchor.get('source'))} · <b>timestamp:</b> {h(anchor.get('timestamp_sec'))}s · <b>final budget:</b> 4 frames</div>"
            f"<div class='split'><div><h4>BEFORE TASK6</h4><table><tbody>{task6_before}</tbody></table></div><div><h4>AFTER TASK6 / DROPPED</h4><table><tbody>{task6_after}</tbody></table></div></div>"
            f"<h3 class='prominent'>Exact final Gemini visual input</h3><div class='gallery final-gallery'>{final_gallery}</div>"
            f"<h3>Prediction & failure analysis</h3><div class='failure-panel'><b>FIRST SUSPECTED FAILURE STAGE:</b> {h(failure['stage'])}<p>{h(failure['explanation'])}</p>"
            f"<p>final↔anchor aligned={h(failure['final_frames_align_task6_anchor'])}; semantic Top-1 survives={h(failure['semantic_top1_survives_to_final'])}; upstream correct-evidence presence={h(failure['correct_evidence_present_upstream'])}</p></div>"
            f"<h3>Latency</h3><table><tbody>{''.join(f'<tr><th>{h(k)}</th><td>{v:.4f}s</td></tr>' for k,v in record['latency'].items())}</tbody></table>"
            f"<details><summary>Raw saved trace (debug only)</summary><pre>{h(json.dumps({'planner':trace.get('planner'),'sufficiency':suff,'task6':task6,'validated_answer':runtime.get('validated_answer')},ensure_ascii=False,indent=2))}</pre></details>"
            "</section>"
        )

    latency = summary["online_latency"]
    filters = sorted({item["case"]["question_type"] for item in records})
    failure_filters = sorted(
        {item["failure"]["stage"] for item in records if item["failure"]["stage"] != "none"}
    )
    filter_buttons = "".join(
        f"<button onclick=\"filterCases('type','{h(value)}')\">{h(value)}</button>"
        for value in filters
    ) + "".join(
        f"<button onclick=\"filterCases('failure','{h(value)}')\">⚠ {h(value)}</button>"
        for value in failure_filters
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Fixed Ours-v0 EgoSchema 25-case review</title><style>
:root{{--correct:#137333;--wrong:#b3261e;--abstain:#8a4b08;--ink:#172033;--paper:#f4f6fa}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:20px}}.sticky{{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #ccd3df;padding:8px 16px;display:flex;gap:6px;overflow:auto}}.sticky a{{padding:4px 7px;border-radius:5px;text-decoration:none;color:white}}.sticky a.correct,.badge.correct{{background:var(--correct)}}.sticky a.wrong,.badge.wrong{{background:var(--wrong)}}.sticky a.abstain,.badge.abstain{{background:var(--abstain)}}.summary,.case{{background:white;border:1px solid #d8deea;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 2px 8px #0000000b}}.case.correct{{border-left:7px solid var(--correct)}}.case.wrong{{border-left:7px solid var(--wrong)}}.case.abstain{{border-left:7px solid var(--abstain)}}header{{display:flex;justify-content:space-between;align-items:center}}header h2{{display:inline;margin-right:10px}}.badge{{color:white;padding:4px 8px;border-radius:999px;margin-right:5px;font-weight:700}}.badge.neutral{{background:#526071}}.question{{font-size:1.25rem;font-weight:650}}.options{{list-style-position:inside;padding:0}}.options li{{padding:7px;border-bottom:1px solid #e6e9ef}}.predicted-option{{outline:2px solid #3b82f6}}.gold-option{{background:#e8f5e9}}.answer-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.answer-grid>div,.decision,.anchor-card,.failure-panel{{background:#eef3f9;padding:12px;border-radius:8px}}.ticks{{margin-left:150px;display:flex;justify-content:space-between;color:#657083;font-size:12px}}.timeline-row{{display:grid;grid-template-columns:140px 1fr;gap:10px;align-items:center;margin:7px 0}}.timeline-label{{text-align:right;font-weight:650}}.track{{height:28px;position:relative;background:repeating-linear-gradient(to right,#eef1f5 0,#eef1f5 calc(16.666% - 1px),#c9d1dd calc(16.666% - 1px),#c9d1dd 16.666%)}}.mark{{position:absolute;height:18px;top:5px;border-radius:3px;opacity:.78}}.coarse{{background:#90a4ae}}.coarse.top{{background:#1565c0}}.semantic.r1{{background:#0d47a1}}.semantic.r2{{background:#1976d2}}.semantic.r3{{background:#64b5f6}}.micro{{background:#ef6c00}}.dense{{background:#9575cd;opacity:.42}}.task5{{background:#00897b}}.task6{{background:#7b1fa2}}.point{{position:absolute;width:5px;height:24px;top:2px;background:#111}}.point.anchor{{width:2px;background:#f44336;height:28px;top:0}}.gallery{{display:flex;gap:10px;overflow:auto;padding:8px 0}}figure{{margin:0;min-width:155px}}figure img{{width:155px;height:105px;object-fit:cover;border-radius:6px;background:#ddd}}figcaption{{font-size:11px;max-width:155px}}.final-gallery figure img{{width:240px;height:160px;border:4px solid #111}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #cbd2dc;padding:6px;vertical-align:top}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.prominent{{background:#111;color:#fff;padding:8px}}pre{{white-space:pre-wrap;max-height:500px;overflow:auto}}button{{padding:6px 9px;margin:3px;border:1px solid #9aa5b5;background:white;border-radius:6px;cursor:pointer}}@media(max-width:900px){{.answer-grid,.split{{grid-template-columns:1fr}}.timeline-row{{grid-template-columns:100px 1fr}}.ticks{{margin-left:110px}}}}
</style></head><body id='top'><nav class='sticky'>{''.join(navigation)}</nav><main><section class='summary'><h1>Ours-v0 EgoSchema Protocol-B fixed_v0_1 — 25-case human review</h1><p><b>MC accuracy:</b> {summary['correct']}/25 = {summary['mc_accuracy']:.1%} · <b>wrong:</b> {summary['wrong']} · <b>abstain:</b> {summary['abstain']} · <b>answered-only:</b> {summary['answered_only_accuracy']:.1%}</p><p><b>Mean final frames:</b> {summary['mean_model_facing_frames']:.2f} · <b>Composed online latency:</b> mean {latency['mean_sec']:.2f}s / median {latency['median_sec']:.2f}s / p95 {latency['p95_sec']:.2f}s</p><p class='note'>Latency is an auditable composed stage metric under safe reuse, not a single uninterrupted wall-clock: unchanged saved planner/query timings + corrected deterministic replay + new Gemini.</p><div><button onclick=\"filterCases('all','all')\">All</button><button onclick=\"filterCases('result','correct')\">Correct</button><button onclick=\"filterCases('result','wrong')\">Wrong</button><button onclick=\"filterCases('result','abstain')\">Abstain</button>{filter_buttons}</div><h2>Case overview</h2><table><thead><tr><th>Case</th><th>Question type</th><th>Result</th><th>Suspected failure</th></tr></thead><tbody>{''.join(overview)}</tbody></table></section>{''.join(sections)}</main><script>
function filterCases(kind,value){{document.querySelectorAll('section.case').forEach(s=>{{let show=kind==='all'||s.dataset[kind]===value;s.style.display=show?'block':'none';}});}}
</script></body></html>"""
    (OUT / "ours_v0_egoschema_review.html").write_text(document, encoding="utf-8")


def main() -> int:
    records, _ = load_complete()
    summary = aggregate(records)
    write_json(OUT / "aggregate_metrics.json", summary)
    write_json(
        OUT / "failure_analysis.json",
        [
            {
                "case_id": item["case"]["case_id"],
                "correct": item["evaluation"]["correct"],
                **item["failure"],
            }
            for item in records
            if not item["evaluation"]["correct"]
        ],
    )
    write_json(
        OUT / "per_case_metrics.json",
        [
            {
                "case_id": item["case"]["case_id"],
                "question_type": item["case"]["question_type"],
                "evaluation": item["evaluation"],
                "latency": item["latency"],
                "failure": item["failure"],
            }
            for item in records
        ],
    )
    render(records, summary)
    lines = [
        "# Ours-v0 EgoSchema Protocol-B fixed_v0_1",
        "",
        f"- Completed: {summary['cases_completed']}/25",
        f"- MC accuracy: {summary['correct']}/25 ({summary['mc_accuracy']:.1%})",
        f"- Wrong / abstain: {summary['wrong']} / {summary['abstain']}",
        f"- Answered-only accuracy: {summary['answered_only_accuracy']:.1%}",
        f"- Mean model-facing frames: {summary['mean_model_facing_frames']:.2f}",
        f"- Mean composed online latency: {summary['online_latency']['mean_sec']:.3f}s",
        f"- Planner / Gemini calls: {summary['planner_calls_fixed_run']} / {summary['gemini_calls']}",
        f"- Technical failure attempts: {summary['technical_failure_attempts']}",
        f"- Dominant failure stages: {summary['dominant_failure_stages']}",
        "",
        "Latency is composed from unchanged saved planner/query timing, corrected local replay, and new final-answer timing.",
    ]
    (OUT / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
