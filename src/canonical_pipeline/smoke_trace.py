"""Diagnostic traces and reports for the three-case new-question smoke run.

This module observes canonical state after execution.  It never selects,
scores, filters, or mutates evidence and never accepts gold/reference data in
its runtime trace builders.
"""

from __future__ import annotations

import html
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .state import CaseState


SMOKE_CASES = ("00002_1", "00018_3", "00018_9")
MODALITY_ID_KEYS = {
    "visual": "region_id",
    "speech": "transcript_segment_id",
    "acoustic": "acoustic_region_id",
}
REQUIRED_STAGE_NAMES = (
    "question_planner", "modality_routing",
    "visual_query_encode", "visual_similarity_search", "visual_retrieval",
    "speech_query_encode", "speech_similarity_search", "speech_retrieval",
    "acoustic_query_encode", "acoustic_similarity_search", "acoustic_retrieval",
    "temporal_linking", "local_visual_refinement", "local_audio_refinement",
    "evidence_sufficiency", "fallback_decision", "fallback_execution",
    "fallback_local_asr", "relation_construction", "relation_reranking",
    "evidence_packet_build", "local_wav_materialization", "final_payload_build",
    "final_gemini_api", "structured_parse", "local_validation",
    "online_end_to_end_total", "uninstrumented_overhead",
)


def _overlap(start: float, end: float, left: float, right: float) -> float:
    return max(0.0, min(end, right) - max(start, left))


def _distance(start: float, end: float, intervals: list[dict[str, Any]]) -> float | None:
    if not intervals:
        return None
    distances = []
    for interval in intervals:
        left, right = float(interval["start_sec"]), float(interval["end_sec"])
        distances.append(max(left - end, start - right, 0.0))
    return min(distances)


def _candidate_source_ids(retrieval: dict[str, Any], modality: str) -> set[str]:
    if modality == "visual":
        return {str(item["region_id"]) for item in retrieval.get("coarse_visual_candidates", [])}
    if modality == "speech":
        return {str(item["transcript_segment_id"]) for item in retrieval.get("speech_candidates", [])}
    return {str(item["acoustic_region_id"]) for item in retrieval.get("acoustic_candidates", [])}


def build_initial_retrieval_trace(state: CaseState) -> dict[str, Any]:
    """Expose raw fresh scores without calling them selected evidence."""
    retrieval = state.retrieval_result or {}
    scored = retrieval.get("question_conditioned_scoring", {}).get("modalities", {})
    modalities: dict[str, Any] = {}
    for modality in ("visual", "speech", "acoustic"):
        audit = scored.get(modality)
        if audit is None:
            modalities[modality] = {
                "executed": False,
                "skip_reason": "modality_not_required",
                "query": None,
                "initial_retrieval_candidates": [],
            }
            continue
        modalities[modality] = {
            "executed": True,
            "skip_reason": None,
            "query": audit.get("query_text"),
            "encoder": audit.get("model"),
            "historical_task4_score_dependency": bool(audit.get("historical_score_dependency")),
            "initial_retrieval_candidates": audit.get("ranked_results", []),
            "top_scored_candidates": audit.get("top_k_results", []),
            "all_scores_finite": audit.get("all_scores_finite"),
        }
    return {"case_id": state.case_id, "terminology": {
        "initial_retrieval_candidates": "Fresh semantic-score output before temporal constraints.",
        "final_selected_evidence": "Only evidence retained after canonical Task 5B-Task 7A processing.",
    }, "modalities": modalities}


def build_temporal_linking_trace(state: CaseState) -> dict[str, Any]:
    """Audit how raw score candidates relate to executed temporal intervals."""
    retrieval = state.retrieval_result or {}
    intervals = retrieval.get("anchor_resolution", {}).get("final_search_intervals", [])
    scored = retrieval.get("question_conditioned_scoring", {}).get("modalities", {})
    cue_intervals = [
        {"start_sec": float(item["start_sec"]), "end_sec": float(item.get("end_sec", item["start_sec"]))}
        for item in state.deterministic_cues.get("time_cues", []) if item.get("start_sec") is not None
    ]
    rows: dict[str, list[dict[str, Any]]] = {}
    for modality in ("visual", "speech", "acoustic"):
        source_ids = _candidate_source_ids(retrieval, modality)
        id_key = MODALITY_ID_KEYS[modality]
        traced = []
        for item in scored.get(modality, {}).get("ranked_results", []):
            start, end = float(item["start_time"]), float(item["end_time"])
            overlap = sum(_overlap(start, end, float(x["start_sec"]), float(x["end_sec"])) for x in intervals)
            cue_overlap = sum(_overlap(start, end, float(x["start_sec"]), float(x["end_sec"])) for x in cue_intervals)
            source_id = str(item[id_key])
            retained = source_id in source_ids
            traced.append({
                **item,
                "temporal_overlap_sec": overlap,
                "temporal_distance_sec": _distance(start, end, intervals),
                "temporally_valid": overlap > 0.0,
                "question_constraint_overlap_sec": cue_overlap if cue_intervals else None,
                "question_constraint_distance_sec": _distance(start, end, cue_intervals),
                "valid_against_explicit_question_constraint": cue_overlap > 0.0 if cue_intervals else None,
                "retained_by_task5b": retained,
                "decision": "kept" if retained else "dropped",
                "decision_reason": (
                    "overlaps_executed_search_interval"
                    if retained else
                    "outside_executed_search_interval"
                    if overlap <= 0 else
                    "not_retained_by_task5b_candidate_construction"
                ),
            })
        rows[modality] = traced
    return {
        "case_id": state.case_id,
        "question_derived_temporal_cues": state.deterministic_cues.get("time_cues", []),
        "planner_temporal_relation": (state.planner_output or {}).get("temporal_relation"),
        "executed_search_intervals": intervals,
        "anchor_resolution": retrieval.get("anchor_resolution"),
        "candidate_temporal_decisions": rows,
        "linked_windows": retrieval.get("linked_windows", []),
    }


def augment_checkpoint_temporal_audit(case: dict[str, Any]) -> None:
    """Add explicit-question constraint distances to an existing checkpoint."""
    cues = [
        {"start_sec": float(item["start_sec"]), "end_sec": float(item.get("end_sec", item["start_sec"]))}
        for item in case["planner"]["deterministic_cues"].get("time_cues", []) if item.get("start_sec") is not None
    ]
    for rows in case["temporal_linking"]["candidate_temporal_decisions"].values():
        for item in rows:
            start, end = float(item["start_time"]), float(item["end_time"])
            cue_overlap = sum(_overlap(start, end, float(x["start_sec"]), float(x["end_sec"])) for x in cues)
            item["question_constraint_overlap_sec"] = cue_overlap if cues else None
            item["question_constraint_distance_sec"] = _distance(start, end, cues)
            item["valid_against_explicit_question_constraint"] = cue_overlap > 0.0 if cues else None


def _payload_evidence(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in (payload or {}).get("evidence_groups", []):
        for modality, key in (("visual", "visual_evidence"), ("speech", "speech_evidence"), ("acoustic", "acoustic_evidence")):
            for item in group.get(key, []):
                output.append({"modality": modality, **item})
    return output


def build_evidence_funnel(state: CaseState, temporal: dict[str, Any]) -> dict[str, Any]:
    """Count evidence representations at each named stage."""
    retrieval = state.retrieval_result or {}
    packet = state.evidence_packet or {}
    payload = _payload_evidence(state.final_payload)
    raw = {modality: len(rows) for modality, rows in temporal["candidate_temporal_decisions"].items()}
    valid = {modality: sum(bool(item["temporally_valid"]) for item in rows) for modality, rows in temporal["candidate_temporal_decisions"].items()}
    task5b = {
        "visual": len(retrieval.get("all_candidates", [])) - len(retrieval.get("speech_candidates", [])) - len(retrieval.get("acoustic_candidates", [])),
        "speech": len(retrieval.get("speech_candidates", [])),
        "acoustic": len(retrieval.get("acoustic_candidates", [])),
    }
    post = (state.sufficiency_result or {}).get("post_fallback_candidates", [])
    post_counts = {modality: sum(item.get("modality") == modality for item in post) for modality in raw}
    retained = packet.get("retained_candidates", [])
    retained_counts = {modality: sum(item.get("modality") == modality for item in retained) for modality in raw}
    payload_counts = {modality: sum(item.get("modality") == modality for item in payload) for modality in raw}
    return {
        "case_id": state.case_id,
        "initial_retrieval_candidates": raw,
        "temporally_valid_initial_candidates": valid,
        "task5b_candidates": task5b,
        "linked_windows": len(retrieval.get("linked_windows", [])),
        "post_sufficiency_fallback_evidence": post_counts,
        "task6_retained_evidence": retained_counts,
        "final_model_facing_evidence": payload_counts,
        "overall": {
            "initial": sum(raw.values()), "temporally_valid": sum(valid.values()),
            "task5b": sum(task5b.values()), "post_sufficiency": len(post),
            "task6_retained": len(retained), "final_payload": len(payload),
        },
    }


def build_coverage_audit(state: CaseState) -> dict[str, Any]:
    """Describe selected temporal coverage without changing count policy."""
    cues = [item for item in state.deterministic_cues.get("time_cues", []) if item.get("start_sec") is not None]
    requested = None
    if cues:
        requested = [float(cues[0]["start_sec"]), float(cues[0].get("end_sec", cues[0]["start_sec"]))]
    elif (state.retrieval_result or {}).get("anchor_resolution", {}).get("final_search_intervals"):
        intervals = state.retrieval_result["anchor_resolution"]["final_search_intervals"]
        requested = [min(float(item["start_sec"]) for item in intervals), max(float(item["end_sec"]) for item in intervals)]
    evidence = sorted((float(item["start_sec"]), float(item["end_sec"])) for item in _payload_evidence(state.final_payload))
    merged: list[list[float]] = []
    for start, end in evidence:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps: list[list[float]] = []
    coverage = None
    if requested:
        cursor = requested[0]
        covered = 0.0
        for start, end in merged:
            left, right = max(start, requested[0]), min(end, requested[1])
            if right <= left:
                continue
            if left > cursor:
                gaps.append([cursor, left])
            covered += max(0.0, right - max(left, cursor))
            cursor = max(cursor, right)
        if cursor < requested[1]:
            gaps.append([cursor, requested[1]])
        width = max(0.0, requested[1] - requested[0])
        coverage = covered / width if width else 1.0
    return {
        "operation": (state.planner_output or {}).get("answer_requirement", {}).get("operation"),
        "requested_interval": requested,
        "final_evidence_intervals": [list(item) for item in evidence],
        "merged_covered_intervals": merged,
        "uncovered_intervals": gaps,
        "coverage_fraction": coverage,
        "complete_temporal_coverage": coverage is not None and coverage >= 1.0 - 1e-9,
        "detected_occurrences": [item for item in (state.sufficiency_result or {}).get("post_fallback_candidates", []) if item.get("modality") == "speech" and item.get("phrase_match_score") is not None],
        "interpretation": "descriptive_only_existing_policy_unchanged",
    }


def reconstruct_evidence_funnel(case: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a missing derived funnel from a durable runtime checkpoint."""
    decisions = case["temporal_linking"]["candidate_temporal_decisions"]
    raw = {modality: len(rows) for modality, rows in decisions.items()}
    valid = {modality: sum(bool(item.get("temporally_valid")) for item in rows) for modality, rows in decisions.items()}
    task5b = {
        "visual": len((case.get("refinement", {}).get("local_visual") or {}).get("micro_windows", [])),
        "speech": sum(bool(item.get("retained_by_task5b")) for item in decisions.get("speech", [])),
        "acoustic": sum(bool(item.get("retained_by_task5b")) for item in decisions.get("acoustic", [])),
    }
    before = case.get("reranking", {}).get("before", [])
    retained = case.get("reranking", {}).get("retained", [])
    payload = case.get("final_selected_evidence", [])
    post_counts = {modality: sum(item.get("modality") == modality for item in before) for modality in raw}
    retained_counts = {modality: sum(item.get("modality") == modality for item in retained) for modality in raw}
    payload_counts = {modality: sum(item.get("modality") == modality for item in payload) for modality in raw}
    return {
        "case_id": case["case_id"],
        "initial_retrieval_candidates": raw,
        "temporally_valid_initial_candidates": valid,
        "task5b_candidates": task5b,
        "linked_windows": len(case["temporal_linking"].get("linked_windows", [])),
        "post_sufficiency_fallback_evidence": post_counts,
        "task6_retained_evidence": retained_counts,
        "final_model_facing_evidence": payload_counts,
        "overall": {
            "initial": sum(raw.values()), "temporally_valid": sum(valid.values()),
            "task5b": sum(task5b.values()), "post_sufficiency": len(before),
            "task6_retained": len(retained), "final_payload": len(payload),
        },
        "reconstructed_from_durable_checkpoint": True,
    }


def normalized_timings(state: CaseState) -> list[dict[str, Any]]:
    """Return required smoke-stage names, with skipped durations kept null."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in state.timings:
        by_name.setdefault(item["stage_name"], []).append(dict(item))
    aliases = {
        "final_gemini_api": "final_model_api",
        "structured_parse": "structured_output_parse",
    }
    output = []
    for stage in REQUIRED_STAGE_NAMES:
        if stage == "uninstrumented_overhead":
            consistency = state.usage.get("timing_consistency", {})
            output.append({"stage_name": stage, "executed": True, "skipped": False, "skip_reason": None, "duration_sec": consistency.get("uninstrumented_overhead_sec")})
            continue
        source = aliases.get(stage, stage)
        occurrences = by_name.get(source, [])
        executed = [item for item in occurrences if item.get("executed") and item.get("duration_sec") is not None]
        if executed:
            item = dict(executed[0])
            item["duration_sec"] = sum(float(value["duration_sec"]) for value in executed)
            item["occurrence_count"] = len(executed)
            item["notes"] = list(item.get("notes", [])) + ["same_name_occurrences_aggregated_without_overwrite"] if len(executed) > 1 else list(item.get("notes", []))
        elif occurrences:
            item = dict(occurrences[0])
            item["occurrence_count"] = len(occurrences)
        else:
            item = {"stage_name": source, "executed": False, "skipped": True, "skip_reason": "stage_not_present", "duration_sec": None, "occurrence_count": 0}
        item["stage_name"] = stage
        if item.get("skipped"):
            item["duration_sec"] = None
        output.append(item)
    return output


def case_runtime_trace(state: CaseState) -> dict[str, Any]:
    """Build a JSON-safe runtime-only checkpoint; it contains no gold."""
    initial = build_initial_retrieval_trace(state)
    temporal = build_temporal_linking_trace(state)
    funnel = build_evidence_funnel(state, temporal)
    retrieval = state.retrieval_result or {}
    packet = state.evidence_packet or {}
    c5 = state.usage.get("task5c_v1_2_runtime", {})
    c7 = state.usage.get("task7b_v3_runtime", {})
    payload_evidence = _payload_evidence(state.final_payload)
    return {
        "case_id": state.case_id,
        "video_id": state.video_id,
        "question": state.question,
        "planner": {
            "structured_output": state.planner_output,
            "deterministic_cues": state.deterministic_cues,
            "usage": state.usage.get("task5a_v2"),
        },
        "routing": {
            "executed_modalities": retrieval.get("executed_modalities", []),
            "skipped_modalities": retrieval.get("skipped_modalities", []),
            "routing_trace": retrieval.get("routing_trace", []),
        },
        "initial_retrieval": initial,
        "temporal_linking": temporal,
        "refinement": {
            "local_visual": retrieval.get("local_visual_refinement"),
            "local_audio_clips": (state.sufficiency_result or {}).get("local_audio_clips", []),
        },
        "sufficiency_fallback": {
            "pre_status": (state.sufficiency_result or {}).get("pre_fallback_evidence_status"),
            "post_status": (state.sufficiency_result or {}).get("evidence_status"),
            "reason_codes": (state.sufficiency_result or {}).get("sufficiency_reason_codes", []),
            "critical_missing_evidence": (state.sufficiency_result or {}).get("critical_missing_evidence", []),
            "ambiguity_flags": (state.sufficiency_result or {}).get("ambiguity_flags", []),
            "fallback_decision": state.fallback_decision,
            "fallback_result": state.fallback_result,
            "fallback_execution_count": state.fallback_execution_count,
            "runtime": c5,
        },
        "reranking": {
            "before": packet.get("candidates_before_reranking", []),
            "retained": packet.get("retained_candidates", []),
            "actually_dropped": packet.get("actually_dropped_candidates", []),
            "merged": packet.get("merged_source_candidates", []),
            "transformed": packet.get("transformed_candidates", []),
            "dropped_visual_frames": packet.get("dropped_visual_frames", []),
            "relations": packet.get("relations", []),
            "budget_accounting": packet.get("budget_accounting", {}),
        },
        "evidence_funnel": funnel,
        "coverage_audit": build_coverage_audit(state),
        "final_selected_evidence": payload_evidence,
        "final_payload": state.final_payload,
        "preflight": state.preflight_result,
        "raw_model_output": state.final_model_output,
        "validated_answer": state.validated_answer,
        "timings": normalized_timings(state),
        "timing_consistency": state.usage.get("timing_consistency", {}),
        "model_usage": {
            "planner": state.usage.get("task5a_v2"),
            "fallback": c5,
            "gemini": c7,
            "external_calls": state.external_calls,
        },
        "audit": state.audit,
        "runtime_gold_loaded": False,
    }


def stage_summary(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(cases)
    output = []
    for stage in REQUIRED_STAGE_NAMES:
        durations = []
        for case in rows:
            item = next((row for row in case["timings"] if row["stage_name"] == stage), None)
            if item and item.get("executed") and item.get("duration_sec") is not None:
                durations.append(float(item["duration_sec"]))
        output.append({
            "stage": stage, "executed_count": len(durations), "case_count": len(rows),
            "executed_mean_sec": statistics.fmean(durations) if durations else None,
            "amortized_mean_sec": sum(durations) / len(rows) if rows else None,
            "median_sec": statistics.median(durations) if durations else None,
            "min_sec": min(durations) if durations else None,
            "max_sec": max(durations) if durations else None,
            "total_sec": sum(durations),
        })
    return output


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _json(value: Any) -> str:
    return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"


def _candidate_table(rows: list[dict[str, Any]], modality: str) -> str:
    id_key = MODALITY_ID_KEYS[modality]
    return _table(["Rank", "Candidate ID", "Interval", "Score", "Question-cue distance", "Question-cue overlap", "Valid vs explicit cue", "Executed-search distance", "Executed-search overlap", "Decision", "Reason"], [
        [item.get("rank"), item.get(id_key), f"{_fmt(item.get('start_time'))}-{_fmt(item.get('end_time'))}", item.get("similarity_score"), item.get("question_constraint_distance_sec"), item.get("question_constraint_overlap_sec"), item.get("valid_against_explicit_question_constraint"), item.get("temporal_distance_sec"), item.get("temporal_overlap_sec"), item.get("decision"), item.get("decision_reason")]
        for item in rows
    ])


def render_smoke_html(
    cases: list[dict[str, Any]],
    aggregate: dict[str, Any],
    posthoc: dict[str, dict[str, Any]],
) -> str:
    """Render a structured human review; media bytes and secrets are excluded."""
    stage_rows = [[row["stage"], f"{row['executed_count']}/{row['case_count']}", row["executed_mean_sec"], row["amortized_mean_sec"], row["median_sec"], row["min_sec"], row["max_sec"], row["total_sec"]] for row in aggregate["stage_latency"]]
    cold_rows = [[name, values.get("cold_load_sec"), values.get("mean_warm_query_encode_sec"), values.get("mean_similarity_search_sec"), values.get("amortized_cost_per_question_sec")] for name, values in aggregate["cold_vs_warm"].items()]
    top_rows = [[name, values["mean_sec_per_question"], values["percent_mean_online"]] for name, values in aggregate["top_level_decomposition"].items()]
    details = []
    for case in cases:
        cid = case["case_id"]
        evaluation = posthoc.get(cid, {})
        planner = case["planner"]
        temporal = case["temporal_linking"]
        rerank = case["reranking"]
        sf = case["sufficiency_fallback"]
        validated = case.get("validated_answer") or {}
        metric_map = {item["name"]: item for item in evaluation.get("metrics", [])}
        supplied = case.get("final_selected_evidence", [])
        evidence_rows = [[item.get("evidence_id"), item.get("modality"), f"{_fmt(item.get('start_sec'))}-{_fmt(item.get('end_sec'))}", item.get("roles"), item.get("transcript") or item.get("audio_clip_path") or f"{len(item.get('frames', []))} frames"] for item in supplied]
        relation_rows = [[item.get("source_candidate_id"), item.get("relation_type"), item.get("target_candidate_id"), item.get("temporal_gap_sec"), item.get("relation_basis")] for item in rerank.get("relations", [])]
        timing_rows = [[item["stage_name"], item.get("executed"), item.get("duration_sec"), (float(item["duration_sec"]) / next((float(x["duration_sec"]) for x in case["timings"] if x["stage_name"] == "online_end_to_end_total"), 1.0) * 100 if item.get("duration_sec") is not None and item["stage_name"] != "online_end_to_end_total" else None), item.get("skip_reason")] for item in case["timings"]]
        raw_answer = None
        try:
            raw_answer = json.loads((case.get("raw_model_output") or {}).get("raw_text", "{}")).get("answer")
        except (json.JSONDecodeError, TypeError):
            raw_answer = (case.get("raw_model_output") or {}).get("raw_text")
        metric_rows = [[name, metric_map.get(name, {}).get("value"), metric_map.get(name, {}).get("category"), metric_map.get(name, {}).get("interpretation")] for name in ("exact_match", "normalized_exact_match", "bleu", "cider", "semantic_similarity", "llm_judge", "task_specific_metric")]
        modality_sections = "".join(f"<h5>{modality.upper()} — initial_retrieval_candidates</h5>{_candidate_table(temporal['candidate_temporal_decisions'].get(modality, []), modality)}" for modality in ("visual", "speech", "acoustic"))
        details.append(f"""
<section id='{cid}'><h2>{cid}</h2>
<h3>STEP 0 — INPUT</h3><p><strong>Question:</strong> {html.escape(case['question'])}</p>
{_table(['Field','Value'], [['Video ID',case['video_id']],['Question type',evaluation.get('question_type')],['temporal_hint_strength',evaluation.get('temporal_hint_strength')],['provided_timestamp (post-hoc audit only)',evaluation.get('provided_timestamp')],['provided_timestamp available to runtime',False]])}
<h3>STEP 1 — QUESTION PLANNER</h3><p>规划器决定搜索什么：operation、modalities、anchor/resolver 与 temporal constraints 如下。</p>{_json(planner)}
<h3>STEP 2 — MODALITY ROUTING</h3>{_json(case['routing'])}
<h3>STEP 3–4 — INITIAL RETRIEVAL / TEMPORAL LINKING</h3><p><code>initial_retrieval_candidates</code> 仅表示原始语义得分候选，不是 <code>final_selected_evidence</code>。</p>{modality_sections}
<h4>Executed temporal constraint and links</h4>{_json({'question_derived_cues':temporal['question_derived_temporal_cues'],'executed_search_intervals':temporal['executed_search_intervals'],'linked_windows':temporal['linked_windows']})}
<h3>STEP 5 — LOCAL REFINEMENT</h3>{_json(case['refinement'])}
<h3>STEP 6–7 — EVIDENCE SUFFICIENCY / FALLBACK</h3>{_json(sf)}
<h3>STEP 8 — RELATION-AWARE RERANKING</h3>
{_table(['Before','Retained','Actually dropped','Merged','Dropped frames'], [[len(rerank['before']),len(rerank['retained']),len(rerank['actually_dropped']),len(rerank['merged']),len(rerank['dropped_visual_frames'])]])}
{_table(['Source','Relation','Target','Gap sec','Basis'], relation_rows)}<details><summary>完整 reranking 记录</summary>{_json(rerank)}</details>
<h3>STEP 9 — EVIDENCE FUNNEL / FINAL PACKET</h3>{_json(case['evidence_funnel'])}
<h4>Count / temporal coverage audit</h4>{_json(case['coverage_audit'])}
<h3>STEP 10 — FINAL MODEL PAYLOAD</h3>{_table(['Evidence ID','Modality','Time','Roles','Model-facing content'], evidence_rows)}<details><summary>完整 provider-neutral payload</summary>{_json(case['final_payload'])}</details>
<h3>STEP 11 — FINAL ANSWER</h3>{_table(['Field','Value'], [['Gold (post-hoc only)',evaluation.get('gold_answer')],['Raw Prediction',raw_answer],['Validated Prediction',validated.get('answer')],['Answer Status',validated.get('answer_status')],['Confidence',validated.get('confidence')],['Final Uncertainties',validated.get('final_uncertainties')],['Abstain',validated.get('abstain')]])}
<h3>STEP 12 — METRICS</h3><p class='warning'>Exact Match / BLEU / CIDEr 均为 diagnostic_reference_overlap，不是开放式问答的权威正确性判断。</p>{_table(['Metric','Value','Category','Interpretation'], metric_rows)}
<h3>STEP 13 — EFFICIENCY</h3>{_table(['Stage','Executed','Latency sec','% Online','Skip reason'], timing_rows)}
<h3>FAILURE LOCALIZATION</h3>{_json(evaluation.get('failure_localization'))}
<h3>LEAKAGE / AUDIT</h3>{_json({'runtime_gold_loaded':case['runtime_gold_loaded'],'preflight':case['preflight'],'audit':case['audit']})}</section>
""")
    return f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>New-question smoke v0.1</title><style>
body{{font-family:Inter,Arial,sans-serif;margin:2rem;color:#172033;background:#f7f9fc}}h1,h2,h3{{color:#153d66}}section{{background:white;padding:1.2rem;margin:1.2rem 0;border:1px solid #d9e2ef;border-radius:10px}}table{{border-collapse:collapse;width:100%;margin:.7rem 0 1.2rem;background:white}}th,td{{border:1px solid #cbd5e1;padding:.45rem;vertical-align:top;text-align:left}}th{{background:#eaf1f8}}pre{{white-space:pre-wrap;overflow:auto;background:#f3f6fa;padding:.7rem;border-radius:6px}}code{{background:#edf2f7;padding:.1rem .25rem}}.warning{{border-left:4px solid #d97706;background:#fff7ed;padding:.7rem}}
</style></head><body><h1>Canonical Baseline v1 — 三个新问题 LIVE Smoke Test</h1>
<p>三个查询在一个进程内运行；CLIP、Sentence-T5、CLAP 仅冷启动一次。Gold 仅在 raw prediction 与 validated prediction 保存后加载。</p>
<h2>Stage latency summary</h2>{_table(['Stage','Executed N/3','Executed mean','Amortized mean','Median','Min','Max','Total'],stage_rows)}
<h2>Top-level latency decomposition</h2>{_table(['Component','Mean sec/question','% mean online latency'],top_rows)}
<h2>Cold vs warm encoder performance</h2>{_table(['Component','Cold load','Mean warm query encode','Mean similarity','Amortized cost/question'],cold_rows)}
<h2>Aggregate execution</h2>{_json({key:value for key,value in aggregate.items() if key not in {'stage_latency','top_level_decomposition','cold_vs_warm'}})}
{''.join(details)}</body></html>"""


def write_html(path: Path, cases: list[dict[str, Any]], aggregate: dict[str, Any], posthoc: dict[str, dict[str, Any]]) -> None:
    """Write the smoke review without embedding media bytes."""
    path.write_text(render_smoke_html(cases, aggregate, posthoc), encoding="utf-8")
