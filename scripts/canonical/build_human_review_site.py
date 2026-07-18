"""Build the complete frozen Baseline v1 human-review website.

This script is report-generation only. It reads the durable 20-case pilot
checkpoint and never imports or executes planner, retrieval, media, or model
boundaries.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CASE_IDS = [
    "00002_7",
    "00004_1",
    "00018_1",
    "00003_2",
    "00006_3",
    "00061_5",
    "00002_1",
    "00004_4",
    "00018_5",
    "00003_1",
    "00018_3",
    "00003_6",
    "00018_7",
    "00018_9",
    "00006_4",
    "00002_2",
    "00018_2",
    "00061_4",
    "00004_2",
    "00006_7",
]

MODALITIES = ("visual", "speech", "acoustic")


def _h(value: Any) -> str:
    return html.escape("N/A" if value is None else str(value), quote=True)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value)) if value else "—"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _candidate_id(item: dict[str, Any]) -> str:
    for field in (
        "candidate_id",
        "evidence_id",
        "region_id",
        "transcript_segment_id",
        "acoustic_region_id",
        "microclip_id",
    ):
        if item.get(field):
            return str(item[field])
    return "N/A"


def _interval(item: dict[str, Any]) -> str:
    start = item.get("start_time", item.get("start_sec", item.get("effective_start_sec")))
    end = item.get("end_time", item.get("end_sec", item.get("effective_end_sec")))
    if start is None and end is None:
        return "N/A"
    return f"{_fmt(start)}–{_fmt(end)} s"


def _score(item: dict[str, Any]) -> Any:
    return item.get("similarity_score", item.get("score", item.get("clap_similarity_score")))


def _roles(item: dict[str, Any]) -> str:
    value = item.get("roles", item.get("role", []))
    return _fmt(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]], css_class: str = "") -> str:
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_h(cell)}</td>" for cell in row) + "</tr>")
    if not body:
        body.append(f'<tr><td colspan="{len(headers)}" class="muted">无记录 / No records</td></tr>')
    head = "".join(f"<th>{_h(item)}</th>" for item in headers)
    return f'<div class="table-wrap"><table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _raw_prediction(case: dict[str, Any]) -> Any:
    raw = case.get("raw_model_output")
    if not isinstance(raw, dict):
        return None
    raw_text = raw.get("raw_text")
    if isinstance(raw_text, str):
        try:
            parsed = json.loads(raw_text)
            return parsed.get("answer")
        except json.JSONDecodeError:
            return raw_text
    return raw.get("answer")


def _relation_text(relations: list[dict[str, Any]], evidence_id: str) -> str:
    parts = []
    for relation in relations:
        source = relation.get("source_candidate_id")
        target = relation.get("target_candidate_id")
        if evidence_id in {source, target}:
            parts.append(f"{source} --{relation.get('relation_type')}--> {target}")
    return "; ".join(parts) or "—"


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for index, item in enumerate(candidates, start=1):
        transcript = item.get("transcript", item.get("text", item.get("transcript_text", "")))
        visual = item.get("representative_keyframe_path", item.get("representative_frame_path", ""))
        rows.append([
            item.get("rank", index),
            _candidate_id(item),
            _fmt(_score(item), 6),
            _interval(item),
            transcript or visual or "—",
        ])
    return rows


def _planner_sentence(plan: dict[str, Any]) -> str:
    operation = plan.get("answer_requirement", {}).get("operation", "未记录")
    modalities = _fmt(plan.get("resolver_modalities", []))
    anchor = plan.get("primary_anchor_modality", "未记录")
    relation = plan.get("temporal_relation", "未记录")
    return (
        f"Planner 的意思是：以 {anchor} 作为主要锚点，按 {relation} 时间关系搜索 "
        f"{modalities} 证据，用于执行 {operation}；这里仅复述已保存的 planner trace。"
    )


def _timing_value(timings: list[dict[str, Any]], names: set[str]) -> float | None:
    values = [
        float(item["duration_sec"])
        for item in timings
        if item.get("stage_name") in names
        and item.get("executed")
        and item.get("duration_sec") is not None
    ]
    return sum(values) if values else None


def _model_usage(case: dict[str, Any]) -> dict[str, Any]:
    usage = case.get("model_usage", {})
    planner = usage.get("planner", {})
    gemini_runtime = usage.get("gemini", {})
    gemini_usage = gemini_runtime.get("usage", {}) if isinstance(gemini_runtime, dict) else {}
    return {
        "planner_tokens": (
            f"in={_fmt(planner.get('input_tokens'))}, out={_fmt(planner.get('output_tokens'))}"
        ),
        "gemini_tokens": (
            f"in={_fmt(gemini_usage.get('input_tokens'))}, "
            f"out={_fmt(gemini_usage.get('output_tokens'))}, "
            f"total={_fmt(gemini_usage.get('total_tokens'))}"
        ),
        "api_calls": usage.get("external_calls", {}),
    }


def _visual_frames(final_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        frame
        for group in final_payload.get("evidence_groups", [])
        for evidence in group.get("visual_evidence", [])
        for frame in evidence.get("frames", [])
    ]


def _final_payload_section(payload: dict[str, Any]) -> str:
    visual_rows: list[list[Any]] = []
    speech_rows: list[list[Any]] = []
    audio_rows: list[list[Any]] = []
    relations: list[dict[str, Any]] = []
    for group in payload.get("evidence_groups", []):
        relations.extend(group.get("relations", []))
        for item in group.get("visual_evidence", []):
            timestamps = [frame.get("timestamp_sec") for frame in item.get("frames", [])]
            visual_rows.append([item.get("evidence_id"), len(timestamps), _fmt(timestamps), _interval(item)])
        for item in group.get("speech_evidence", []):
            speech_rows.append([item.get("evidence_id"), _interval(item), item.get("transcript", "")])
        for item in group.get("acoustic_evidence", []):
            audio_rows.append([
                item.get("evidence_id"),
                _interval(item),
                item.get("audio_clip_path", "N/A"),
                _fmt(float(item.get("end_sec", 0)) - float(item.get("start_sec", 0))),
            ])
    relation_rows = [
        [item.get("source_candidate_id"), item.get("relation_type"), item.get("target_candidate_id"), item.get("temporal_gap_sec")]
        for item in relations
    ]
    uncertainty_rows = [
        [item.get("type"), item.get("severity"), item.get("affected_claim"), item.get("description")]
        for item in payload.get("pipeline_uncertainties", [])
    ]
    return (
        '<h4>Visual</h4>'
        + _table(["Evidence ID", "Frames", "Timestamps", "Interval"], visual_rows)
        + '<h4>Speech</h4>'
        + _table(["Evidence ID", "Interval", "Transcript"], speech_rows)
        + '<h4>Audio</h4>'
        + _table(["Evidence ID", "Interval", "WAV path", "Duration s"], audio_rows)
        + '<h4>Relations</h4>'
        + _table(["Source", "Type", "Target", "Gap s"], relation_rows)
        + '<h4>Inherited uncertainties</h4>'
        + _table(["Type", "Severity", "Affected claim", "Description"], uncertainty_rows)
    )


def _render_case(
    case: dict[str, Any],
    index: int,
    missing: dict[str, list[str]],
) -> str:
    case_id = case["case_id"]
    posthoc = case.get("posthoc_evaluation", {})
    plan = case.get("planner", {}).get("structured_output", {})
    raw_prediction = _raw_prediction(case)
    raw_record = case.get("raw_model_output")
    raw_prediction_available = isinstance(raw_record, dict) and isinstance(raw_record.get("raw_text"), str)
    raw_prediction_display = "null (model abstention)" if raw_prediction_available and raw_prediction is None else raw_prediction
    validated = case.get("validated_answer") or {}
    validated_prediction_display = (
        "null (validated abstention)"
        if "answer" in validated and validated.get("answer") is None
        else validated.get("answer")
    )
    gold = posthoc.get("gold_answer")
    if not case.get("planner"):
        missing[case_id].append("planner")
    if not raw_prediction_available:
        missing[case_id].append("raw_prediction (final QA raw output was not saved/executed)")
    if not case.get("final_payload"):
        missing[case_id].append("final_payload")

    previous_id = EXPECTED_CASE_IDS[index - 1] if index else None
    next_id = EXPECTED_CASE_IDS[index + 1] if index + 1 < len(EXPECTED_CASE_IDS) else None
    nav = (
        f'<a href="#case-{previous_id}">← Previous Case</a>' if previous_id else '<span class="muted">← Previous Case</span>'
    )
    nav += ' <a href="#summary">Back to summary</a> '
    nav += f'<a href="#case-{next_id}">Next Case →</a>' if next_id else '<span class="muted">Next Case →</span>'

    # Initial retrieval candidates.
    modality_cards = []
    modalities = case.get("initial_retrieval", {}).get("modalities", {})
    for modality in MODALITIES:
        record = modalities.get(modality, {})
        candidates = record.get("initial_retrieval_candidates", [])
        state = "executed" if record.get("executed") else f"skipped — {record.get('skip_reason', 'reason unavailable')}"
        modality_cards.append(
            f'<div class="channel"><h4>{modality.upper()}</h4>'
            f'<p><b>Routing:</b> {_h(state)}<br><b>Query:</b> {_h(record.get("query", "N/A"))}</p>'
            + _table(["Rank", "ID", "Score", "Interval", "Transcript / visual path"], _candidate_rows(candidates))
            + "</div>"
        )

    # Temporal filtering decisions and linked records.
    temporal = case.get("temporal_linking", {})
    temporal_rows = []
    decisions = temporal.get("candidate_temporal_decisions", {})
    for modality in MODALITIES:
        for item in decisions.get(modality, []) if isinstance(decisions, dict) else []:
            temporal_rows.append([
                modality,
                _candidate_id(item),
                item.get("rank"),
                item.get("decision", "kept" if item.get("temporally_valid") else "dropped"),
                item.get("decision_reason", "N/A"),
                _fmt(item.get("temporal_overlap_sec")),
                _fmt(item.get("temporal_distance_sec")),
                _fmt(item.get("anchor_relation", item.get("relation_type"))),
            ])
    linked_rows = []
    for item in temporal.get("linked_windows", []):
        linked_rows.append([
            item.get("anchor_candidate_id", item.get("source_candidate_id", _candidate_id(item))),
            item.get("resolver_candidate_id", item.get("target_candidate_id", "N/A")),
            item.get("relation_type", item.get("temporal_relation", "N/A")),
            _fmt(item.get("temporal_gap_sec", item.get("anchor_distance_sec"))),
            _interval(item),
        ])

    # Refinement.
    refinement = case.get("refinement", {})
    visual_refinement = refinement.get("local_visual", {})
    visual_rows = []
    for item in visual_refinement.get("micro_windows", []):
        visual_rows.append([
            _candidate_id(item),
            _interval(item),
            _fmt(item.get("selected_frame_timestamps", item.get("frame_timestamps", []))),
            _fmt(item.get("selected_frame_paths", item.get("frame_paths", []))),
        ])
    audio_rows = []
    for item in refinement.get("local_audio_clips", []):
        audio_rows.append([
            _candidate_id(item),
            _interval(item),
            item.get("local_audio_clip_reference", item.get("output_path", "N/A")),
            _fmt(item.get("duration_sec", item.get("duration"))),
        ])

    # Task5C.
    sufficiency = case.get("sufficiency_fallback", {})
    coverage = case.get("coverage_audit", {})
    fallback = sufficiency.get("fallback_result", {})
    fallback_rows = [[
        _fmt(fallback.get("triggered", False)),
        fallback.get("strategy", fallback.get("outcome", "N/A")),
        _fmt(fallback.get("decode_interval", fallback.get("search_interval"))),
        _fmt([_candidate_id(item) for item in fallback.get("added_candidates", [])]),
        fallback.get("model_calls", 0),
    ]]

    # Task6 before/after and diagnostics.
    reranking = case.get("reranking", {})
    before = reranking.get("before", [])
    retained = reranking.get("retained", [])
    dropped = reranking.get("actually_dropped", [])
    relations = reranking.get("relations", [])
    retained_ids = {_candidate_id(item) for item in retained}
    dropped_by_id = {_candidate_id(item): item for item in dropped}
    before_rows = []
    for item in before:
        evidence_id = _candidate_id(item)
        before_rows.append([
            evidence_id,
            item.get("modality"),
            _roles(item),
            _interval(item),
            _fmt(item.get("selection_rank", item.get("rank", _score(item)))),
            _relation_text(relations, evidence_id),
            "retained" if evidence_id in retained_ids else "dropped / transformed / merged",
        ])
    after_rows = []
    for item in retained:
        evidence_id = _candidate_id(item)
        after_rows.append([
            evidence_id,
            item.get("modality"),
            _roles(item),
            _interval(item),
            _fmt(item.get("selection_rank", item.get("rank", _score(item)))),
            _relation_text(relations, evidence_id),
            "retained",
        ])
    dropped_rows = []
    for evidence_id, item in dropped_by_id.items():
        dropped_rows.append([
            evidence_id,
            item.get("modality"),
            item.get("reason", item.get("drop_reason", item.get("reason_code", "reason not recorded"))),
        ])
    task6_diag = posthoc.get("task6_diagnostic", {})

    timings = case.get("timings", [])
    timing_rows = []
    online = _timing_value(timings, {"online_end_to_end_total"})
    for item in timings:
        duration = item.get("duration_sec") if item.get("executed") else None
        share = None if duration is None or not online else 100.0 * float(duration) / online
        timing_rows.append([
            item.get("stage_name"),
            _fmt(item.get("executed")),
            _fmt(duration),
            _fmt(share),
            item.get("skip_reason") or "—",
        ])
    usage = _model_usage(case)
    summary_timing_rows = [
        ["Planner", _fmt(_timing_value(timings, {"question_planner"}))],
        ["Retrieval", _fmt(_timing_value(timings, {"visual_retrieval", "speech_retrieval", "acoustic_retrieval"}))],
        ["Refinement", _fmt(_timing_value(timings, {"local_visual_refinement", "local_audio_refinement"}))],
        ["Fallback", _fmt(_timing_value(timings, {"fallback_execution"}))],
        ["Task6", _fmt(_timing_value(timings, {"relation_construction", "relation_reranking", "evidence_packet_build"}))],
        ["Gemini", _fmt(_timing_value(timings, {"final_gemini_api"}))],
        ["Online total", _fmt(online)],
        ["Planner tokens", usage["planner_tokens"]],
        ["Gemini tokens", usage["gemini_tokens"]],
        ["API calls", _fmt(usage["api_calls"])],
    ]

    confidence = validated.get("confidence", {})
    uncertainties = validated.get("final_uncertainties", case.get("final_payload", {}).get("pipeline_uncertainties", []))
    status = validated.get("answer_status", case.get("preflight", {}).get("status", "unavailable"))
    automatic_diagnosis = posthoc.get("failure_localization", {})
    return f"""
    <article class="case-card status-{_h(status)}" id="case-{_h(case_id)}">
      <div class="case-nav">{nav}</div>
      <section class="hero">
        <div><span class="eyebrow">1 · QUESTION</span><h2>{_h(case_id)} — {_h(case.get('question'))}</h2></div>
        <div class="answer-grid">
          <div><b>Question type</b><br>{_h(posthoc.get('question_type'))}</div>
          <div class="gold"><b>Gold answer</b><br><span class="warning">POST-HOC ONLY — NOT AVAILABLE TO RUNTIME</span><br>{_h(gold)}</div>
          <div><b>Raw prediction</b><br>{_h(raw_prediction_display)}</div>
          <div><b>Validated prediction</b><br>{_h(validated_prediction_display)}</div>
          <div><b>Answer status</b><br>{_h(status)}</div>
          <div><b>Confidence</b><br>{_h(confidence.get('level'))} — {_h(confidence.get('basis'))}</div>
          <div class="wide"><b>Final uncertainty</b><br>{_h(_fmt(uncertainties))}</div>
        </div>
      </section>

      <section><span class="eyebrow">2 · PLANNER</span><h3>问题规划 / Question Planner</h3>
        {_table(["Field", "Saved value"], [
            ["operation", plan.get('answer_requirement', {}).get('operation')],
            ["requested modalities", _fmt(plan.get('resolver_modalities', []))],
            ["anchor modality", plan.get('primary_anchor_modality')],
            ["resolver modalities", _fmt(plan.get('resolver_modalities', []))],
            ["temporal relation", plan.get('temporal_relation')],
            ["question-derived temporal constraint", _fmt(temporal.get('question_derived_temporal_cues', []))],
            ["local visual refinement", plan.get('requires_local_visual_inspection')],
            ["visual route", plan.get('visual_route')],
        ])}
        <p class="explain">{_h(_planner_sentence(plan))}</p>
        <details><summary>Saved structured planner output</summary><pre>{_h(json.dumps(plan, ensure_ascii=False, indent=2))}</pre></details>
      </section>

      <section><span class="eyebrow">3 · INITIAL RETRIEVAL CANDIDATES</span><h3>过滤前的初始检索</h3>
        <p class="notice">这些是 INITIAL RETRIEVAL CANDIDATES，不是 final selected evidence。</p>
        <div class="channels">{''.join(modality_cards)}</div>
      </section>

      <section><span class="eyebrow">4 · TEMPORAL FILTERING / LINKING</span><h3>时间过滤与跨模态链接</h3>
        <p class="explain">“为什么这一条被留下/删除？”由 saved decision_reason、overlap 和 distance 直接说明。</p>
        {_table(["Modality", "Candidate", "Raw rank", "Kept / dropped", "Why", "Overlap s", "Distance s", "Anchor relation"], temporal_rows)}
        <h4>Linked evidence</h4>
        {_table(["Anchor / source", "Resolver / target", "Relation", "Gap / anchor distance s", "Interval"], linked_rows)}
      </section>

      <section><span class="eyebrow">5 · LOCAL REFINEMENT</span><h3>局部细化</h3>
        <p><b>Visual:</b> {_h('executed' if visual_refinement.get('executed') else 'skipped')} — {_h(visual_refinement.get('reason'))}</p>
        {_table(["Evidence", "Refined interval", "Retained timestamps", "Frame paths"], visual_rows)}
        <p><b>Audio:</b> {_h('executed' if audio_rows else 'skipped / no local clip')}.</p>
        {_table(["Evidence", "Local WAV interval", "WAV path", "Duration s"], audio_rows)}
      </section>

      <section><span class="eyebrow">6 · TASK5C SUFFICIENCY</span><h3>证据充分性与 fallback</h3>
        <div class="decision {_h(sufficiency.get('post_status', 'unknown'))}">{_h(str(sufficiency.get('post_status', 'unknown')).upper())}</div>
        {_table(["Field", "Value"], [
            ["Evidence entering Task5C", _fmt([_candidate_id(item) for item in before])],
            ["Pre-status", sufficiency.get('pre_status')],
            ["Post-status", sufficiency.get('post_status')],
            ["Why / reason codes", _fmt(sufficiency.get('reason_codes', []))],
            ["Missing information", _fmt(sufficiency.get('critical_missing_evidence', []))],
            ["Required modalities", _fmt(plan.get('resolver_modalities', []))],
            ["Uncertainty", _fmt(sufficiency.get('ambiguity_flags', []))],
            ["Fallback required", sufficiency.get('fallback_decision', {}).get('required')],
            ["Fallback execution count", sufficiency.get('fallback_execution_count')],
            ["Requested domain", _fmt(coverage.get('requested_interval'))],
            ["Coverage ratio", _fmt(coverage.get('coverage_fraction'))],
            ["Uncovered spans", _fmt(coverage.get('uncovered_intervals', []))],
        ])}
        <h4>Fallback result</h4>{_table(["Triggered", "Type / outcome", "Interval", "Recovered evidence", "Model calls"], fallback_rows)}
      </section>

      <section class="task6"><span class="eyebrow">7 · TASK6 — KEY REVIEW</span><h3>Task6 前后对照</h3>
        <div class="task6-grid"><div><h4>LEFT — BEFORE TASK6</h4>{_table(["Evidence ID", "Modality", "Role", "Interval", "Priority / rank", "Relation", "Decision"], before_rows)}</div>
        <div><h4>RIGHT — AFTER TASK6</h4>{_table(["Evidence ID", "Modality", "Role", "Interval", "Priority / rank", "Relation", "Decision"], after_rows)}</div></div>
        <h4>TASK6 DROPPED</h4>{_table(["Evidence ID", "Modality", "Recorded reason"], dropped_rows)}
        <div class="flags">
          <div><b>required modalities before Task6</b><br>{_h(_fmt(task6_diag.get('required_modalities_before_task6', [])))}</div>
          <div><b>required modalities after Task6</b><br>{_h(_fmt(task6_diag.get('required_modalities_after_task6', [])))}</div>
          <div><b>required_modality_lost</b><br>{_h(_fmt(task6_diag.get('required_modality_lost')))}</div>
          <div><b>cross_modal_pair_broken</b><br>{_h(_fmt(task6_diag.get('cross_modal_pair_broken')))}</div>
          <div><b>relation_endpoint_dropped</b><br>{_h(_fmt(task6_diag.get('relation_endpoint_dropped')))}</div>
        </div>
      </section>

      <section><span class="eyebrow">8 · FINAL MODEL INPUT</span><h3>Gemini 实际收到的输入</h3>
        <p class="notice">仅展示保存的 model-facing metadata，不展示 base64 或 media bytes。</p>
        {_final_payload_section(case.get('final_payload', {}))}
      </section>

      <section><span class="eyebrow">9 · FINAL ANSWER ANALYSIS</span><h3>人工复核面板</h3>
        {_table(["Field", "Value"], [
            ["Question", case.get('question')],
            ["Gold (post-hoc)", gold],
            ["Prediction", validated.get('answer')],
            ["Semantic correctness", "unreviewed — likely_correct / partially_correct / likely_wrong / appropriate_abstention / inappropriate_abstention / insufficient_to_judge"],
            ["First likely failure point", "unreviewed — planner / initial retrieval / temporal filtering / cross-modal linking / local refinement / Task5C sufficiency / fallback / Task6 compaction / final Gemini reasoning / dataset/query issue / no obvious failure"],
            ["automatic_structural_diagnosis (not human truth)", _fmt(automatic_diagnosis)],
        ])}
      </section>

      <section><span class="eyebrow">10 · EFFICIENCY</span><h3>每案效率（次要信息）</h3>
        {_table(["Component", "Time / usage"], summary_timing_rows)}
        <details><summary>Full saved stage timing</summary>{_table(["Stage", "Executed", "Latency s", "% online total", "Skip reason"], timing_rows)}</details>
      </section>
      <div class="case-nav bottom">{nav}</div>
    </article>
    """


def _styles() -> str:
    return """
    :root{--ink:#172033;--muted:#667085;--line:#d8dee9;--paper:#fff;--bg:#f4f7fb;--blue:#2357a5;--amber:#9a6700;--red:#b42318;--green:#067647}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,Segoe UI,Arial,sans-serif}
    a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.layout{display:grid;grid-template-columns:220px minmax(0,1fr);max-width:1900px;margin:auto}
    aside{position:sticky;top:0;height:100vh;overflow:auto;padding:18px;background:#152238;color:#fff}aside a{color:#dbeafe;display:block;padding:4px 0}.main{min-width:0;padding:26px}
    .page-title,.case-card{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:0 3px 14px #17203310}.page-title{padding:24px;margin-bottom:22px}.case-card{margin:28px 0;overflow:hidden;scroll-margin-top:12px}
    section{padding:22px 24px;border-top:1px solid var(--line)}section:first-of-type{border-top:0}.hero{background:linear-gradient(135deg,#f8fbff,#fff)}h1,h2,h3,h4{line-height:1.25}h2{font-size:25px;margin:8px 0 16px}h3{font-size:19px;margin:7px 0 14px}.eyebrow{font-weight:800;letter-spacing:.08em;color:var(--blue);font-size:12px}
    .table-wrap{overflow:auto;margin:10px 0 15px}table{width:100%;border-collapse:collapse;background:#fff}th,td{border:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top;max-width:600px;word-break:break-word}th{background:#edf3fb;position:sticky;top:0}.summary-table td:nth-child(1){font-weight:700;white-space:nowrap}
    .answer-grid,.flags{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}.answer-grid>div,.flags>div{padding:12px;background:#f8fafc;border:1px solid var(--line);border-radius:8px}.answer-grid .wide{grid-column:1/-1}.gold{border-left:4px solid var(--amber)!important}.warning{font-size:11px;font-weight:800;color:var(--amber)}
    .channels{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}.channel{min-width:0;padding:12px;border:1px solid var(--line);border-radius:9px;background:#fbfdff}.task6{background:#fffdf7}.task6-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}.task6-grid>div{min-width:0}
    .notice,.explain{padding:10px 12px;border-left:4px solid var(--blue);background:#eef5ff}.decision{display:inline-block;padding:7px 13px;border-radius:999px;font-weight:800;margin-bottom:10px}.decision.sufficient{background:#d1fadf;color:var(--green)}.decision.questionable{background:#fef0c7;color:var(--amber)}.decision.insufficient{background:#fee4e2;color:var(--red)}
    .case-nav{display:flex;justify-content:space-between;gap:12px;padding:11px 24px;background:#eaf1fb;position:sticky;top:0;z-index:4}.case-nav.bottom{position:static}.muted{color:var(--muted)}details{margin:10px 0}pre{max-height:440px;overflow:auto;padding:12px;background:#101828;color:#e6edf7;border-radius:8px;white-space:pre-wrap}.completeness{padding:14px;background:#ecfdf3;border:1px solid #a6f4c5;border-radius:8px}
    .status-answered{border-top:5px solid var(--green)}.status-answered_with_uncertainty{border-top:5px solid var(--amber)}.status-insufficient_evidence{border-top:5px solid var(--red)}
    @media(max-width:1000px){.layout{display:block}aside{position:relative;height:auto}.task6-grid{grid-template-columns:1fr}.main{padding:12px}.channels{grid-template-columns:1fr}}
    @media print{aside,.case-nav{display:none}.layout{display:block}.main{padding:0}.case-card{break-before:page;box-shadow:none}}
    """


def build_site(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Render exactly the approved 20 cases and return a completeness audit."""
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["case_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate case IDs in frozen pilot checkpoint")
    expected = set(EXPECTED_CASE_IDS)
    missing_ids = sorted(expected - set(by_id))
    if missing_ids:
        raise ValueError(f"Cannot render partial review; missing cases: {missing_ids}")
    selected = [by_id[case_id] for case_id in EXPECTED_CASE_IDS]
    if len(selected) != 20:
        raise AssertionError("expected_case_count=20 but selected count differs")

    missing_fields: dict[str, list[str]] = defaultdict(list)
    summary_rows = []
    case_sections = []
    for index, case in enumerate(selected):
        posthoc = case.get("posthoc_evaluation", {})
        validated = case.get("validated_answer") or {}
        status = validated.get("answer_status", case.get("preflight", {}).get("status", "unavailable"))
        prediction = (
            "null (validated abstention)"
            if "answer" in validated and validated.get("answer") is None
            else validated.get("answer")
        )
        summary_rows.append([
            f'<a href="#case-{_h(case["case_id"])}">{_h(case["case_id"])}</a>',
            posthoc.get("question_type"),
            case.get("question"),
            posthoc.get("gold_answer"),
            prediction,
            status,
            posthoc.get("failure_localization", {}).get("category", "unavailable"),
            "unreviewed",
        ])
        case_sections.append(_render_case(case, index, missing_fields))

    rendered_ids = [case["case_id"] for case in selected]
    if rendered_ids != EXPECTED_CASE_IDS:
        raise AssertionError("Rendered case order/identity differs from approved review list")
    # The first column intentionally contains safe anchors, so render this one
    # table without escaping that already-created link cell.
    summary_body = "".join(
        "<tr>" + "".join(
            f"<td>{cell if col == 0 else _h(cell)}</td>" for col, cell in enumerate(row)
        ) + "</tr>"
        for row in summary_rows
    )
    summary_head = "".join(
        f"<th>{_h(value)}</th>" for value in ["Case", "Question Type", "Question", "Gold", "Prediction", "Status", "Main suspected failure", "Review status"]
    )
    nav_links = "".join(f'<a href="#case-{_h(case_id)}">{_h(case_id)}</a>' for case_id in EXPECTED_CASE_IDS)
    missing_rows = [[case_id, field] for case_id, fields in sorted(missing_fields.items()) for field in fields]
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Canonical Baseline v1 — 20-case Human Review</title><style>{_styles()}</style></head>
    <body><div class="layout"><aside><h3>20-case review</h3><a href="#summary">Summary</a>{nav_links}</aside><main class="main">
    <header class="page-title" id="summary"><span class="eyebrow">CANONICAL BASELINE V1 · FROZEN SAVED ARTIFACTS</span><h1>20-Case Human Review</h1>
      <p>本页面仅由已保存的完整 pilot checkpoints 生成；未重跑 case，未调用 Claude、Gemini、Whisper 或外部 API。Gold 始终是 post-hoc。</p>
      <div class="completeness"><b>expected_case_count = 20</b> · <b>rendered_case_count = 20</b><br>Rendered IDs: {_h(', '.join(rendered_ids))}</div>
      <div class="table-wrap"><table class="summary-table"><thead><tr>{summary_head}</tr></thead><tbody>{summary_body}</tbody></table></div>
      <details><summary>Saved-field availability audit</summary>{_table(["Case", "Unavailable saved field"], missing_rows)}</details>
    </header>
    {''.join(case_sections)}
    </main></div></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    audit = {
        "expected_case_count": 20,
        "rendered_case_count": len(rendered_ids),
        "rendered_case_ids": rendered_ids,
        "complete": len(rendered_ids) == 20 and rendered_ids == EXPECTED_CASE_IDS,
        "missing_data_fields": dict(missing_fields),
        "source": input_path.as_posix(),
        "api_calls": 0,
    }
    audit_path = output_path.with_name("human_review_generation_audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/pilot_20/baseline_v1/full_pilot_v0_1/case_results.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pilot_20/baseline_v1/human_review_v0_1/baseline_v1_20_case_human_review.html"),
    )
    args = parser.parse_args()
    audit = build_site(args.input, args.output)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
