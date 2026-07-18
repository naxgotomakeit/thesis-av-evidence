"""Post-hoc diagnostics and HTML for the frozen 20-case Baseline v1 pilot.

All functions operate on already validated runtime traces.  They do not alter
planner, retrieval, sufficiency, reranking, payload, or answer behavior.
Gold/reference fields are accepted only by the explicitly post-hoc functions.
"""

from __future__ import annotations

import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.canonical_pipeline.smoke_trace import MODALITY_ID_KEYS, REQUIRED_STAGE_NAMES
from src.evaluation.registry import baseline_metric_registry


def percentile(values: list[float], fraction: float = 0.95) -> float | None:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def timing_value(case: dict[str, Any], name: str) -> float | None:
    item = next((row for row in case.get("timings", []) if row["stage_name"] == name), None)
    return None if item is None else item.get("duration_sec")


def enrich_initial_candidates(case: dict[str, Any], project_root: Path) -> None:
    """Attach offline-index display metadata absent from older reused traces."""
    video_id = case["video_id"]
    sources = {
        "visual": (project_root / "outputs/visual_index" / video_id / "visual_state_regions.json", "visual_state_regions"),
        "speech": (project_root / "outputs/audio_index" / video_id / "transcript_embedding_index.json", "rows"),
        "acoustic": (project_root / "outputs/audio_index" / video_id / "acoustic_embedding_index.json", "rows"),
    }
    for modality, (path, collection) in sources.items():
        if not path.is_file():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))[collection]
        key = MODALITY_ID_KEYS[modality]
        metadata = {str(item[key]): item for item in rows}
        for container in (
            case["initial_retrieval"]["modalities"].get(modality, {}).get("initial_retrieval_candidates", []),
            case["initial_retrieval"]["modalities"].get(modality, {}).get("top_scored_candidates", []),
            case["temporal_linking"]["candidate_temporal_decisions"].get(modality, []),
        ):
            for item in container:
                source = metadata.get(str(item.get(key)), {})
                if modality == "visual":
                    item.setdefault("representative_keyframe_path", source.get("representative_keyframe_path"))
                    item.setdefault("representative_frame_timestamp", source.get("representative_frame_timestamp"))
                elif modality == "speech":
                    item.setdefault("transcript_text", source.get("text", ""))
                    item.setdefault("language", source.get("language"))
                else:
                    item.setdefault("low_information_or_silence", source.get("low_information_or_silence"))
                    item.setdefault("speech_overlap_ratio", source.get("speech_overlap_ratio"))


def task6_diagnostic(case: dict[str, Any]) -> dict[str, Any]:
    """Measure modality/relation preservation without changing Task6 output."""
    planner = case["planner"]["structured_output"] or {}
    requested = set(planner.get("resolver_modalities", []))
    primary = planner.get("primary_anchor_modality")
    if primary in {"visual", "speech", "acoustic"}:
        requested.add(primary)
    before = case["reranking"].get("before", [])
    retained = case["reranking"].get("retained", [])
    final = case.get("final_selected_evidence", [])
    before_modalities = {item.get("modality") for item in before}
    after_modalities = {item.get("modality") for item in retained}
    final_modalities = {item.get("modality") for item in final}
    required_before = sorted(requested)
    required_after = sorted(requested & after_modalities)
    lost = sorted(requested - after_modalities)
    pre_pair = requested & before_modalities
    post_pair = requested & after_modalities
    linked_endpoints = {
        candidate_id
        for window in case["temporal_linking"].get("linked_windows", [])
        for candidate_id in window.get("source_candidate_ids", [])
    }
    actually_dropped = {item.get("candidate_id") for item in case["reranking"].get("actually_dropped", [])}
    dropped_relation_endpoints = sorted(linked_endpoints & actually_dropped)
    return {
        "case_id": case["case_id"],
        "operation": planner.get("answer_requirement", {}).get("operation"),
        "required_modalities_before_task6": required_before,
        "required_modalities_after_task6": required_after,
        "modalities_entering_task6": sorted(item for item in before_modalities if item),
        "modalities_retained_by_task6": sorted(item for item in after_modalities if item),
        "final_payload_modalities": sorted(item for item in final_modalities if item),
        "required_modality_lost": bool(lost),
        "lost_required_modalities": lost,
        "cross_modal_pair_broken": len(pre_pair) >= 2 and len(post_pair) < len(pre_pair),
        "relation_endpoint_dropped": bool(dropped_relation_endpoints),
        "dropped_relation_endpoints": dropped_relation_endpoints,
        "before_task6_count": len(before),
        "after_task6_count": len(retained),
        "actually_dropped_count": len(case["reranking"].get("actually_dropped", [])),
        "compression_ratio": len(retained) / len(before) if before else None,
        "answer_status": (case.get("validated_answer") or {}).get("answer_status"),
        "diagnostic_association_only": True,
    }


def counting_diagnostic(case: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    operation = (case["planner"]["structured_output"] or {}).get("answer_requirement", {}).get("operation")
    if operation != "count_occurrences" and manifest.get("question_type") != "Counting":
        return None
    coverage = case.get("coverage_audit", {})
    requested = coverage.get("requested_interval")
    merged = coverage.get("merged_covered_intervals", [])
    covered_duration = sum(max(0.0, float(end) - float(start)) for start, end in merged)
    answer = (case.get("validated_answer") or {}).get("answer")
    exact_numeric = bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", str(answer or "")))
    insufficient = case["sufficiency_fallback"].get("post_status") == "insufficient"
    incomplete = coverage.get("complete_temporal_coverage") is False
    return {
        "case_id": case["case_id"], "operation": operation,
        "runtime_requested_domain": requested,
        "dataset_provided_timestamp_posthoc": manifest.get("provided_timestamp"),
        "observed_evidence_intervals": coverage.get("final_evidence_intervals", []),
        "covered_duration_sec": covered_duration,
        "coverage_ratio": coverage.get("coverage_fraction"),
        "uncovered_intervals": coverage.get("uncovered_intervals", []),
        "detected_occurrence_count": len(coverage.get("detected_occurrences", [])),
        "sufficiency_status": case["sufficiency_fallback"].get("post_status"),
        "fallback_execution_count": case["sufficiency_fallback"].get("fallback_execution_count"),
        "prediction": answer, "gold_answer": manifest.get("answer"),
        "incomplete_coverage": incomplete,
        "exact_numeric_answer": exact_numeric,
        "unsupported_exact_count": bool(exact_numeric and (incomplete or insufficient)),
        "diagnostic_only": True,
    }


def crossmodal_diagnostic(case: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    expected = set(manifest.get("expected_modality_stratum", []))
    planner = case["planner"]["structured_output"] or {}
    planned = set(planner.get("resolver_modalities", []))
    if len(expected) < 2 and len(planned) < 2:
        return None
    initial = {modality for modality, data in case["initial_retrieval"]["modalities"].items() if data.get("executed")}
    before = {item.get("modality") for item in case["reranking"].get("before", [])}
    after = {item.get("modality") for item in case["reranking"].get("retained", [])}
    final = {item.get("modality") for item in case.get("final_selected_evidence", [])}
    required = planned or expected
    return {
        "case_id": case["case_id"], "expected_modalities_posthoc": sorted(expected),
        "planner_required_modalities": sorted(required),
        "initially_available_modalities": sorted(item for item in initial if item),
        "pre_task6_modalities": sorted(item for item in before if item),
        "post_task6_modalities": sorted(item for item in after if item),
        "final_payload_modalities": sorted(item for item in final if item),
        "cross_modal_completeness_preserved": required <= final,
        "missing_final_modalities": sorted(required - final),
        "diagnostic_only": True,
    }


def failure_localization(case: dict[str, Any], task6: dict[str, Any], count: dict[str, Any] | None) -> dict[str, Any]:
    """Assign a structural category without semantic correctness inference."""
    status = (case.get("validated_answer") or {}).get("answer_status")
    sf = case.get("sufficiency_fallback", {})
    if case.get("technical_failure"):
        category, evidence = "technical failure", case["technical_failure"]
    elif task6["required_modality_lost"] or task6["cross_modal_pair_broken"]:
        category, evidence = "Task6 compaction/reranking issue", {"lost": task6["lost_required_modalities"], "pair_broken": task6["cross_modal_pair_broken"]}
    elif count and count["incomplete_coverage"]:
        category, evidence = "evidence coverage insufficiency", {"coverage_ratio": count["coverage_ratio"], "unsupported_exact_count": count["unsupported_exact_count"]}
    elif sf.get("fallback_execution_count") and sf.get("post_status") == "insufficient":
        category, evidence = "fallback failure", sf.get("fallback_result")
    elif sf.get("post_status") == "insufficient" or status == "insufficient_evidence":
        category, evidence = "evidence sufficiency failure", {"status": sf.get("post_status"), "missing": sf.get("critical_missing_evidence")}
    elif status == "query_or_premise_inconsistent":
        category, evidence = "query/premise inconsistency", (case.get("validated_answer") or {}).get("reasoning_summary")
    else:
        category, evidence = "unclear / human review required", "No automatic semantic judge or task-specific correctness metric was used."
    return {"case_id": case["case_id"], "category": category, "evidence": evidence, "automatic_causal_claim": False}


def evaluate_posthoc(cases: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach gold diagnostics only after every runtime result is durable."""
    if not all(case.get("validated_answer") or case.get("technical_failure") for case in cases):
        raise RuntimeError("Post-hoc evaluation blocked until all runtime results are saved")
    by_id = {row["case_id"]: row for row in manifest_rows}
    corpus = [str(by_id[case["case_id"]].get("answer", "")) for case in cases]
    registry = baseline_metric_registry()
    posthoc: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    task6_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for case in cases:
        source = by_id[case["case_id"]]
        validated = case.get("validated_answer") or {}
        metrics = [item.to_dict() for item in registry.evaluate({"gold_reference_answer": source.get("answer"), "validated_prediction": validated.get("answer")}, {"reference_corpus": corpus})]
        t6 = task6_diagnostic(case)
        count = counting_diagnostic(case, source)
        cross = crossmodal_diagnostic(case, source)
        failure = failure_localization(case, t6, count)
        task6_rows.append(t6)
        if count:
            count_rows.append(count)
        if cross:
            cross_rows.append(cross)
        record = {
            "case_id": case["case_id"], "question_type": source.get("question_type"),
            "temporal_hint_strength": source.get("temporal_hint_strength"),
            "provided_timestamp_available": source.get("provided_timestamp_available"),
            "provided_timestamp": source.get("provided_timestamp"),
            "provided_timestamp_runtime_input": False,
            "runtime_question_temporal_cues": case["planner"]["deterministic_cues"].get("time_cues", []),
            "gold_answer": source.get("answer"), "metrics": metrics,
            "human_review_status": "unreviewed", "human_review_notes": None,
            "task6_diagnostic": t6, "counting_diagnostic": count,
            "crossmodal_diagnostic": cross, "failure_localization": failure,
            "gold_loaded_posthoc": True,
        }
        posthoc[case["case_id"]] = record
        metric_rows.append({"case_id": case["case_id"], "gold_answer": source.get("answer"), "raw_prediction": _raw_answer(case), "validated_prediction": validated.get("answer"), "answer_status": validated.get("answer_status"), "confidence": validated.get("confidence"), "final_uncertainties": validated.get("final_uncertainties", []), "human_review_status": "unreviewed", "metrics": metrics, "metric_role": "diagnostic_reference_overlap"})
    return posthoc, metric_rows, task6_rows, count_rows, cross_rows


def _raw_answer(case: dict[str, Any]) -> Any:
    raw = (case.get("raw_model_output") or {}).get("raw_text")
    try:
        return json.loads(raw or "{}").get("answer")
    except (json.JSONDecodeError, TypeError):
        return raw


def stage_aggregates(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for stage in REQUIRED_STAGE_NAMES:
        values = [float(value) for case in cases if (value := timing_value(case, stage)) is not None]
        output.append({
            "stage": stage, "executed_count": len(values), "case_count": len(cases),
            "mean_executed_sec": statistics.fmean(values) if values else None,
            "amortized_mean_sec": sum(values) / len(cases),
            "median_sec": statistics.median(values) if values else None,
            "p95_sec": percentile(values), "min_sec": min(values) if values else None,
            "max_sec": max(values) if values else None, "total_sec": sum(values),
        })
    return output


def case_efficiency(case: dict[str, Any]) -> dict[str, Any]:
    evidence = case.get("final_selected_evidence", [])
    frames = [frame for item in evidence if item.get("modality") == "visual" for frame in item.get("frames", [])]
    audio = [item for item in evidence if item.get("modality") == "acoustic"]
    speech = [item for item in evidence if item.get("modality") == "speech"]
    usage = case.get("model_usage", {})
    planner = usage.get("planner") or {}
    gemini_runtime = usage.get("gemini") or {}
    gemini = gemini_runtime.get("usage", {})
    return {
        "case_id": case["case_id"], "visual_frame_count": len(frames),
        "audio_clip_count": len(audio),
        "audio_duration_sec": sum(max(0.0, float(item["end_sec"]) - float(item["start_sec"])) for item in audio),
        "speech_segment_count": len(speech),
        "transcript_character_count": sum(len(str(item.get("transcript", ""))) for item in speech),
        "final_evidence_count": len(evidence),
        "initial_candidate_count": case["evidence_funnel"]["overall"]["initial"],
        "temporally_valid_count": case["evidence_funnel"]["overall"]["temporally_valid"],
        "task5b_count": case["evidence_funnel"]["overall"]["task5b"],
        "post_sufficiency_count": case["evidence_funnel"]["overall"]["post_sufficiency"],
        "task6_retained_count": case["evidence_funnel"]["overall"]["task6_retained"],
        "evidence_compression_ratio": len(evidence) / case["evidence_funnel"]["overall"]["initial"] if case["evidence_funnel"]["overall"]["initial"] else None,
        "planner_calls": planner.get("api_calls"), "planner_retries": planner.get("retries"),
        "planner_input_tokens": planner.get("input_tokens"), "planner_output_tokens": planner.get("output_tokens"),
        "planner_api_wall_clock_sec": planner.get("planner_api_wall_clock_sec"),
        "gemini_calls": gemini_runtime.get("total_api_calls"), "gemini_retries": gemini_runtime.get("retry_api_calls"),
        "gemini_input_tokens": gemini.get("input_tokens"), "gemini_output_tokens": gemini.get("output_tokens"),
        "gemini_total_tokens": gemini.get("total_tokens"), "gemini_cached_tokens": gemini.get("total_cached_tokens"),
        "gemini_uncached_input_tokens": gemini.get("uncached_input_tokens"), "gemini_cache_hit": gemini.get("cache_hit"),
        "gemini_tokens_by_modality": gemini.get("tokens_by_modality"),
        "gemini_api_wall_clock_sec": gemini_runtime.get("final_model_api_wall_clock_sec"),
        "fallback_model_calls": (usage.get("fallback") or {}).get("fallback_model_calls", 0),
    }


def _metric_value(posthoc: dict[str, Any], name: str) -> float | None:
    item = next((metric for metric in posthoc.get("metrics", []) if metric["name"] == name), None)
    if not item or not item.get("available"):
        return None
    value = item.get("value")
    return float(value) if isinstance(value, (int, float, bool)) else None


def _group_breakdown(cases: list[dict[str, Any]], posthoc: dict[str, dict[str, Any]], efficiency: dict[str, dict[str, Any]], groups: dict[str, list[str]]) -> dict[str, Any]:
    output = {}
    for label, case_ids in groups.items():
        subset = [case for case in cases if case["case_id"] in case_ids]
        if not subset:
            continue
        eff = [efficiency[case["case_id"]] for case in subset]
        output[label] = {
            "case_count": len(subset),
            "mean_online_latency_sec": statistics.fmean(float(timing_value(case, "online_end_to_end_total") or 0.0) for case in subset),
            "mean_retrieval_latency_sec": statistics.fmean(sum(float(timing_value(case, name) or 0.0) for name in ("visual_retrieval", "speech_retrieval", "acoustic_retrieval")) for case in subset),
            "fallback_rate": sum(case["sufficiency_fallback"].get("fallback_execution_count", 0) > 0 for case in subset) / len(subset),
            "mean_initial_candidates": statistics.fmean(item["initial_candidate_count"] for item in eff),
            "mean_final_evidence": statistics.fmean(item["final_evidence_count"] for item in eff),
            "mean_gemini_tokens": statistics.fmean(float(item["gemini_total_tokens"] or 0) for item in eff),
            "insufficient_evidence_count": sum((case.get("validated_answer") or {}).get("answer_status") == "insufficient_evidence" for case in subset),
            "diagnostic_metrics": {name: statistics.fmean(value for case in subset if (value := _metric_value(posthoc[case["case_id"]], name)) is not None) for name in ("exact_match", "normalized_exact_match", "bleu", "cider")},
        }
    return output


def aggregate_all(cases: list[dict[str, Any]], posthoc: dict[str, dict[str, Any]], task6_rows: list[dict[str, Any]], count_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]], lifecycle: dict[str, Any]) -> dict[str, Any]:
    """Build aggregate latency, efficiency, breakdown and diagnostic records."""
    efficiency_rows = [case_efficiency(case) for case in cases]
    efficiency = {item["case_id"]: item for item in efficiency_rows}
    online = [float(timing_value(case, "online_end_to_end_total") or 0.0) for case in cases]
    status = Counter((case.get("validated_answer") or {}).get("answer_status", "technical_failure") for case in cases)
    means = {name: statistics.fmean(item[name] for item in efficiency_rows) for name in ("initial_candidate_count", "temporally_valid_count", "task5b_count", "post_sufficiency_count", "task6_retained_count", "final_evidence_count", "visual_frame_count", "audio_duration_sec", "speech_segment_count")}
    stage_rows = stage_aggregates(cases)
    components: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        visual_refine = float(timing_value(case, "local_visual_refinement") or 0)
        local_wav = float(timing_value(case, "local_wav_materialization") or 0)
        retrieval = sum(float(timing_value(case, name) or 0) for name in ("visual_retrieval", "speech_retrieval", "acoustic_retrieval"))
        row = {
            "Planner": float(timing_value(case, "question_planner") or 0),
            "Fresh retrieval/query scoring": max(0.0, retrieval - visual_refine),
            "Temporal/refinement": float(timing_value(case, "temporal_linking") or 0) + visual_refine,
            "Sufficiency/fallback": float(timing_value(case, "evidence_sufficiency") or 0) + float(timing_value(case, "fallback_decision") or 0) + float(timing_value(case, "fallback_execution") or 0),
            "Task6 reranking/packet": sum(float(timing_value(case, name) or 0) for name in ("relation_construction", "relation_reranking", "evidence_packet_build")),
            "Media/payload": local_wav + float(timing_value(case, "final_payload_build") or 0),
            "Final Gemini": float(timing_value(case, "final_gemini_api") or 0),
            "Parse/validation": float(timing_value(case, "structured_parse") or 0) + float(timing_value(case, "local_validation") or 0),
        }
        row["Other overhead"] = float(timing_value(case, "online_end_to_end_total") or 0) - sum(row.values())
        for key, value in row.items():
            components[key].append(value)
    mean_online = statistics.fmean(online)
    temporal_groups: dict[str, list[str]] = defaultdict(list)
    type_groups: dict[str, list[str]] = defaultdict(list)
    modality_groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        meta = posthoc[case["case_id"]]
        temporal_groups[meta["temporal_hint_strength"]].append(case["case_id"])
        type_groups[meta["question_type"]].append(case["case_id"])
        expected = next((row for row in cross_rows if row["case_id"] == case["case_id"]), None)
        modalities = set(expected["expected_modalities_posthoc"] if expected else (case["planner"]["structured_output"] or {}).get("resolver_modalities", []))
        for modality in modalities:
            modality_groups[modality].append(case["case_id"])
        if len(modalities) > 1:
            modality_groups["cross-modal"].append(case["case_id"])
    metric_means = {name: statistics.fmean(value for case in cases if (value := _metric_value(posthoc[case["case_id"]], name)) is not None) for name in ("exact_match", "normalized_exact_match", "bleu", "cider")}
    planner_lat = [float(item["planner_api_wall_clock_sec"]) for item in efficiency_rows if int(item["planner_calls"] or 0) > 0 and item["planner_api_wall_clock_sec"] is not None]
    gemini_lat = [float(item["gemini_api_wall_clock_sec"]) for item in efficiency_rows if int(item["gemini_calls"] or 0) > 0 and item["gemini_api_wall_clock_sec"] is not None]
    fallback_values = [float(timing_value(case, "fallback_execution")) for case in cases if timing_value(case, "fallback_execution") is not None]
    return {
        "case_count": len(cases), "answer_status_distribution": dict(status),
        "online_latency": {"mean": mean_online, "median": statistics.median(online), "p95": percentile(online), "min": min(online), "max": max(online)},
        "stage_latency": stage_rows,
        "top_level_decomposition": {name: {"mean_sec_per_question": statistics.fmean(values), "percent_mean_online": statistics.fmean(values) / mean_online * 100 if mean_online else None} for name, values in components.items()},
        "mean_timing_coverage": statistics.fmean(float(case.get("timing_consistency", {}).get("timing_coverage") or 0) for case in cases),
        "metric_means": metric_means,
        "efficiency_means": means,
        "evidence_compression_ratio": sum(item["final_evidence_count"] for item in efficiency_rows) / sum(item["initial_candidate_count"] for item in efficiency_rows),
        "planner_api": {"total_calls": sum(int(item["planner_calls"] or 0) for item in efficiency_rows), "calls_per_question": statistics.fmean(float(item["planner_calls"] or 0) for item in efficiency_rows), "input_tokens_total": sum(int(item["planner_input_tokens"] or 0) for item in efficiency_rows), "output_tokens_total": sum(int(item["planner_output_tokens"] or 0) for item in efficiency_rows), "executed_case_count": len(planner_lat), "wall_clock_mean_executed": statistics.fmean(planner_lat) if planner_lat else None, "wall_clock_amortized_mean_per_question": sum(planner_lat) / len(cases), "wall_clock_mean": statistics.fmean(planner_lat) if planner_lat else None, "wall_clock_mean_semantics": "executed_calls_only", "wall_clock_median_executed": statistics.median(planner_lat) if planner_lat else None, "wall_clock_p95_executed": percentile(planner_lat)},
        "gemini_api": {"total_calls": sum(int(item["gemini_calls"] or 0) for item in efficiency_rows), "calls_per_question": statistics.fmean(float(item["gemini_calls"] or 0) for item in efficiency_rows), "input_tokens_total": sum(int(item["gemini_input_tokens"] or 0) for item in efficiency_rows), "output_tokens_total": sum(int(item["gemini_output_tokens"] or 0) for item in efficiency_rows), "total_tokens": sum(int(item["gemini_total_tokens"] or 0) for item in efficiency_rows), "cached_tokens": sum(int(item["gemini_cached_tokens"] or 0) for item in efficiency_rows), "uncached_input_tokens": sum(int(item["gemini_uncached_input_tokens"] or 0) for item in efficiency_rows), "cache_hit_cases": sum(item["gemini_cache_hit"] is True for item in efficiency_rows), "executed_case_count": len(gemini_lat), "wall_clock_mean_executed": statistics.fmean(gemini_lat) if gemini_lat else None, "wall_clock_amortized_mean_per_question": sum(gemini_lat) / len(cases), "wall_clock_mean": statistics.fmean(gemini_lat) if gemini_lat else None, "wall_clock_mean_semantics": "executed_calls_only", "wall_clock_median_executed": statistics.median(gemini_lat) if gemini_lat else None, "wall_clock_p95_executed": percentile(gemini_lat)},
        "fallback": {"case_count": len(fallback_values), "rate": len(fallback_values) / len(cases), "mean_executed_sec": statistics.fmean(fallback_values) if fallback_values else None, "amortized_sec_per_question": sum(fallback_values) / len(cases), "model_calls": sum(int(item["fallback_model_calls"] or 0) for item in efficiency_rows)},
        "task6": {"mean_entering": statistics.fmean(row["before_task6_count"] for row in task6_rows), "mean_retained": statistics.fmean(row["after_task6_count"] for row in task6_rows), "mean_actually_dropped": statistics.fmean(row["actually_dropped_count"] for row in task6_rows), "mean_compression_ratio": statistics.fmean(row["compression_ratio"] for row in task6_rows if row["compression_ratio"] is not None), "required_modality_loss_count": sum(row["required_modality_lost"] for row in task6_rows), "cross_modal_pair_broken_count": sum(row["cross_modal_pair_broken"] for row in task6_rows), "relation_endpoint_dropped_count": sum(row["relation_endpoint_dropped"] for row in task6_rows)},
        "counting": {"case_count": len(count_rows), "incomplete_coverage_count": sum(row["incomplete_coverage"] for row in count_rows), "unsupported_exact_count_count": sum(row["unsupported_exact_count"] for row in count_rows)},
        "crossmodal": {"case_count": len(cross_rows), "completeness_preserved_count": sum(row["cross_modal_completeness_preserved"] for row in cross_rows)},
        "temporal_hint_breakdown": _group_breakdown(cases, posthoc, efficiency, temporal_groups),
        "question_type_breakdown": _group_breakdown(cases, posthoc, efficiency, type_groups),
        "modality_breakdown": _group_breakdown(cases, posthoc, efficiency, modality_groups),
        "efficiency_per_case": efficiency_rows, "model_lifecycle": lifecycle,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]], *, raw_columns: set[int] | None = None) -> str:
    raw_columns = raw_columns or set()
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            rendered = str(value) if index in raw_columns else html.escape(_fmt(value))
            cells.append(f"<td>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _json(value: Any) -> str:
    return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"


def render_html(cases: list[dict[str, Any]], posthoc: dict[str, dict[str, Any]], aggregate: dict[str, Any], task6_rows: list[dict[str, Any]], count_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    """Render dashboard plus complete human-readable per-case pipeline traces."""
    t6 = {row["case_id"]: row for row in task6_rows}
    counts = {row["case_id"]: row for row in count_rows}
    cross = {row["case_id"]: row for row in cross_rows}
    failure_map = {row["case_id"]: row for row in failures}
    stage_table = _table(["Stage", "Executed N/20", "Mean Executed", "Amortized/Q", "Median", "P95", "Min", "Max", "Total"], [[row["stage"], f"{row['executed_count']}/{row['case_count']}", row["mean_executed_sec"], row["amortized_mean_sec"], row["median_sec"], row["p95_sec"], row["min_sec"], row["max_sec"], row["total_sec"]] for row in aggregate["stage_latency"]])
    decomposition = _table(["Component", "Mean sec/question", "% Mean Online"], [[key, value["mean_sec_per_question"], value["percent_mean_online"]] for key, value in aggregate["top_level_decomposition"].items()])
    overview = []
    for case in cases:
        meta = posthoc[case["case_id"]]
        metrics = {item["name"]: item.get("value") for item in meta["metrics"]}
        answer = case.get("validated_answer") or {}
        overview.append([f"<a href='#{case['case_id']}'>{case['case_id']}</a>", case["question"], meta["gold_answer"], answer.get("answer"), answer.get("answer_status"), answer.get("confidence", {}).get("level"), metrics.get("exact_match"), metrics.get("bleu"), metrics.get("cider"), meta["human_review_status"]])
    t6_table = _table(["Case", "Operation", "Required modalities", "Before", "After", "Required lost", "Pair broken", "Endpoint dropped", "Outcome"], [[row["case_id"], row["operation"], row["required_modalities_before_task6"], row["before_task6_count"], row["after_task6_count"], row["required_modality_lost"], row["cross_modal_pair_broken"], row["relation_endpoint_dropped"], row["answer_status"]] for row in task6_rows])
    count_table = _table(["Case", "Runtime domain", "Coverage", "Occurrences", "Sufficiency", "Fallback", "Prediction", "Gold", "Unsupported exact count"], [[row["case_id"], row["runtime_requested_domain"], row["coverage_ratio"], row["detected_occurrence_count"], row["sufficiency_status"], row["fallback_execution_count"], row["prediction"], row["gold_answer"], row["unsupported_exact_count"]] for row in count_rows])
    cross_table = _table(["Case", "Required", "Initial", "Pre-Task6", "Post-Task6", "Final", "Complete"], [[row["case_id"], row["planner_required_modalities"], row["initially_available_modalities"], row["pre_task6_modalities"], row["post_task6_modalities"], row["final_payload_modalities"], row["cross_modal_completeness_preserved"]] for row in cross_rows])
    breakdowns = "".join(f"<h3>{html.escape(title)}</h3>" + _table(["Group", "N", "Online", "Retrieval", "Fallback rate", "Initial", "Final", "Gemini tok", "Insufficient", "Diagnostics"], [[name, values["case_count"], values["mean_online_latency_sec"], values["mean_retrieval_latency_sec"], values["fallback_rate"], values["mean_initial_candidates"], values["mean_final_evidence"], values["mean_gemini_tokens"], values["insufficient_evidence_count"], values["diagnostic_metrics"]] for name, values in aggregate[key].items()]) for title, key in (("Temporal hint breakdown", "temporal_hint_breakdown"), ("Question-type breakdown", "question_type_breakdown"), ("Modality breakdown (non-exclusive)", "modality_breakdown")))
    case_sections = []
    for case in cases:
        cid = case["case_id"]
        meta = posthoc[cid]
        planner = case["planner"]
        temporal = case["temporal_linking"]
        answer = case.get("validated_answer") or {}
        metric_map = {item["name"]: item for item in meta["metrics"]}
        candidate_sections = []
        for modality in ("visual", "speech", "acoustic"):
            key = MODALITY_ID_KEYS[modality]
            rows = temporal["candidate_temporal_decisions"].get(modality, [])
            candidate_sections.append(f"<h5>{modality.upper()}</h5>" + _table(["Rank", "ID", "Score", "Interval", "Transcript/keyframe", "Cue distance", "Cue overlap", "Executed overlap", "Decision", "Reason"], [[item.get("rank"), item.get(key), item.get("similarity_score"), f"{_fmt(item.get('start_time'))}-{_fmt(item.get('end_time'))}", item.get("transcript_text") or item.get("representative_keyframe_path"), item.get("question_constraint_distance_sec"), item.get("question_constraint_overlap_sec"), item.get("temporal_overlap_sec"), item.get("decision"), item.get("decision_reason")] for item in rows]))
        before_rows = [[item.get("candidate_id"), item.get("modality"), f"{_fmt(item.get('start_time'))}-{_fmt(item.get('end_time'))}", item.get("roles"), item.get("anchor_distance_sec")] for item in case["reranking"].get("before", [])]
        retained_rows = [[item.get("candidate_id"), item.get("modality"), f"{_fmt(item.get('start_time'))}-{_fmt(item.get('end_time'))}", item.get("roles"), item.get("anchor_distance_sec")] for item in case["reranking"].get("retained", [])]
        dropped_rows = [[item.get("candidate_id"), item.get("reason")] for item in case["reranking"].get("actually_dropped", [])]
        relations = [[item.get("source_candidate_id"), item.get("relation_type"), item.get("target_candidate_id"), item.get("temporal_gap_sec"), item.get("relation_basis")] for item in case["reranking"].get("relations", [])]
        evidence_rows = [[item.get("evidence_id"), item.get("modality"), f"{_fmt(item.get('start_sec'))}-{_fmt(item.get('end_sec'))}", item.get("roles"), item.get("transcript") or item.get("audio_clip_path") or [frame.get("timestamp_sec") for frame in item.get("frames", [])]] for item in case.get("final_selected_evidence", [])]
        timings = [[item["stage_name"], item.get("executed"), item.get("duration_sec"), item.get("skip_reason")] for item in case.get("timings", [])]
        raw = _raw_answer(case)
        case_sections.append(f"""
<section id='{cid}'><h2>{cid}</h2><p class='source'>source_run = {html.escape(case.get('source_run','full_pilot_v0_1'))}</p>
<h3>STEP 0 — INPUT</h3>{_table(['Field','Value'],[['Video ID',case['video_id']],['Question',case['question']],['Question type',meta['question_type']],['temporal_hint_strength',meta['temporal_hint_strength']],['raw question temporal cues',meta['runtime_question_temporal_cues']],['dataset provided_timestamp — POST-HOC / NOT AVAILABLE TO ONLINE RUNTIME',meta['provided_timestamp']]])}
<h3>STEP 1 — PLANNER</h3><p>规划器决定系统应搜索的 operation、anchor、resolver modalities 与 temporal relation。</p>{_table(['Latency','Input tokens','Output tokens'],[[timing_value(case,'question_planner'),(case['model_usage'].get('planner') or {}).get('input_tokens'),(case['model_usage'].get('planner') or {}).get('output_tokens')]])}{_json(planner['structured_output'])}
<h3>STEP 2 — MODALITY ROUTING</h3>{_json(case['routing'])}
<h3>STEP 3–4 — FRESH INITIAL RETRIEVAL / TEMPORAL FILTERING</h3><p class='warning'>INITIAL RETRIEVAL CANDIDATES 不是 final selected evidence。表格对比 semantic score 与 temporal validity。</p>{''.join(candidate_sections)}
<h3>STEP 5 — CROSS-MODAL / RELATION LINKING</h3>{_json(temporal.get('linked_windows',[]))}
<h3>STEP 6 — LOCAL REFINEMENT</h3>{_json(case['refinement'])}
<h3>STEP 7 — TASK5C SUFFICIENCY</h3>{_json(case['sufficiency_fallback'])}<h4>Coverage diagnostic</h4>{_json(case.get('coverage_audit'))}
<h3>STEP 8 — FALLBACK</h3><p>{'Executed once.' if case['sufficiency_fallback'].get('fallback_execution_count') else 'Fallback skipped — see fallback_decision and reason codes above.'}</p>
<h3>STEP 9 — TASK6 BEFORE / DECISION / AFTER</h3><h4>Before Task6</h4>{_table(['ID','Modality','Interval','Roles','Anchor distance'],before_rows)}<h4>Relations</h4>{_table(['Source','Relation','Target','Gap','Basis'],relations)}<h4>Retained</h4>{_table(['ID','Modality','Interval','Roles','Anchor distance'],retained_rows)}<h4>Actually dropped</h4>{_table(['ID','Reason'],dropped_rows)}<h4>Diagnostic</h4>{_json(t6[cid])}
<h3>STEP 10 — EVIDENCE FUNNEL / FINAL PAYLOAD</h3>{_json(case['evidence_funnel'])}{_table(['Evidence ID','Modality','Interval','Roles','Model-facing content'],evidence_rows)}<details><summary>Provider-neutral Task7A payload</summary>{_json(case['final_payload'])}</details>
<h3>STEP 11 — FINAL ANSWER</h3>{_table(['Field','Value'],[['QUESTION',case['question']],['GOLD ANSWER',meta['gold_answer']],['RAW PREDICTION',raw],['VALIDATED PREDICTION',answer.get('answer')],['ANSWER STATUS',answer.get('answer_status')],['CONFIDENCE',answer.get('confidence')],['FINAL UNCERTAINTIES',answer.get('final_uncertainties')],['HUMAN REVIEW',meta['human_review_status']]])}
<h3>STEP 12 — METRICS</h3><p class='warning'>Lexical/reference-overlap metrics are diagnostic_reference_overlap and are NOT authoritative open-ended QA accuracy.</p>{_table(['Metric','Value','Available','Category','Interpretation'],[[name,metric_map[name].get('value'),metric_map[name].get('available'),metric_map[name].get('category'),metric_map[name].get('interpretation')] for name in metric_map])}
<h3>STEP 13 — EFFICIENCY</h3>{_table(['Stage','Executed','Latency sec','Skip reason'],timings)}
<h3>FAILURE LOCALIZATION</h3>{_json(failure_map[cid])}</section>""")
    return f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Baseline v1 Pilot 20</title><style>body{{font-family:Inter,Arial,sans-serif;margin:2rem;background:#f5f7fb;color:#172033}}h1,h2,h3{{color:#153d66}}section{{background:white;border:1px solid #d7e0ec;border-radius:10px;padding:1.2rem;margin:1.2rem 0}}table{{border-collapse:collapse;width:100%;background:white;margin:.7rem 0 1.3rem}}th,td{{border:1px solid #cbd5e1;padding:.4rem;vertical-align:top;text-align:left}}th{{background:#e8f0f8;position:sticky;top:0}}pre{{white-space:pre-wrap;overflow:auto;background:#f1f5f9;padding:.7rem;border-radius:6px}}.warning{{border-left:4px solid #d97706;background:#fff7ed;padding:.7rem}}.source{{color:#475569}}a{{color:#075985}}</style></head><body>
<h1>Canonical Baseline v1 — EgoSound 20-case Pilot</h1><p class='warning'>Wrong answers, insufficient evidence and reranking loss are preserved baseline measurements. Gold was loaded only after runtime predictions and validation were saved.</p>
<h2>A. Pilot Summary</h2>{_json({key:aggregate[key] for key in ('case_count','answer_status_distribution','online_latency','metric_means','mean_timing_coverage')})}
<h2>B. Question / Gold / Prediction</h2>{_table(['Case','Question','Gold','Prediction','Status','Confidence','EM','BLEU','CIDEr','Human review'],overview,raw_columns={0})}
<h2>C. Answer Metrics</h2><p class='warning'>These lexical/reference-overlap metrics are diagnostic and are NOT treated as authoritative open-ended QA accuracy.</p>{_json(aggregate['metric_means'])}
<h2>D. Full Pipeline Stage Latency</h2>{stage_table}
<h2>E. Latency Decomposition</h2>{decomposition}
<h2>F. API / Token / Cache / Lifecycle</h2>{_json({'planner':aggregate['planner_api'],'gemini':aggregate['gemini_api'],'fallback':aggregate['fallback'],'model_lifecycle':aggregate['model_lifecycle']})}
<h2>G. Evidence Funnel</h2>{_json({'means':aggregate['efficiency_means'],'compression_ratio':aggregate['evidence_compression_ratio']})}
<h2>H. Task6 Diagnostics</h2><p>Associations are diagnostic; causality is not asserted.</p>{t6_table}
<h2>I. Counting / Global Coverage</h2>{count_table}
<h2>J. Cross-modal Completeness</h2>{cross_table}
<h2>K–L. Breakdowns</h2>{breakdowns}
<h2>M. Failure Taxonomy</h2>{_table(['Case','Category','Evidence'],[[row['case_id'],row['category'],row['evidence']] for row in failures])}
<h1>Full Per-case Reviews</h1>{''.join(case_sections)}</body></html>"""


def write_html(path: Path, *args: Any) -> None:
    path.write_text(render_html(*args), encoding="utf-8")
