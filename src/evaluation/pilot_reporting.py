"""Provider-neutral pilot report schema and safe HTML primitives."""

from __future__ import annotations

import html
from typing import Any


def display_metric(metric: dict[str, Any]) -> str:
    """Render unavailable metrics explicitly rather than as numeric zero."""
    if not metric.get("available"):
        return "unavailable"
    value = metric.get("value")
    return "unavailable" if value is None else str(value)


def pilot_report_schema() -> dict[str, Any]:
    return {
        "sections": ["pilot_summary", "answer_metrics_summary", "efficiency_summary", "temporal_hint_analysis", "case_overview", "per_case_review", "failure_analysis"],
        "metric_categories": ["diagnostic_reference_overlap", "semantic_correctness", "task_specific", "human_or_judge", "efficiency"],
        "required_case_fields": ["case_id", "video_id", "question", "gold_answer_posthoc", "raw_prediction", "validated_prediction", "answer_status", "confidence", "final_uncertainties", "selected_evidence", "efficiency", "temporal_localization", "audit"],
        "selected_evidence_fields": ["evidence_id", "modality", "start_sec", "end_sec", "relation", "selection_rank", "anchor_distance_sec", "fallback_provenance", "ambiguities"],
        "retrieval_and_sufficiency_fields": ["planner_operation", "requested_modalities", "executed_modalities", "candidate_count_by_modality", "retained_evidence_count", "fallback_triggered", "fallback_outcome", "fallback_execution_count", "sufficiency_status", "missing_evidence_types", "ambiguity_types", "provided_timestamp_overlap", "provided_timestamp_distance_sec", "evidence_window_coverage"],
        "efficiency_fields": ["online_end_to_end_total", "planner_latency", "retrieval_latency", "refinement_latency", "sufficiency_latency", "fallback_latency", "reranking_latency", "payload_latency", "final_model_api_wall_clock", "parse_and_validation_latency", "uninstrumented_overhead", "timing_coverage", "planner_api_calls", "gemini_api_calls", "fallback_model_calls", "retries", "planner_input_tokens", "planner_output_tokens", "gemini_input_tokens", "gemini_output_tokens", "gemini_total_tokens", "cached_tokens", "uncached_input_tokens", "cache_hit", "cache_hit_rate", "visual_frame_count", "local_audio_seconds", "speech_segment_count"],
        "aggregate_statistics": ["mean", "median", "min", "max", "p95"],
        "breakdowns": ["question_type", "answer_required_modality", "temporal_hint_strength", "fallback_status", "modality_weight", "count_or_aggregation_vs_local_event"],
        "charts": ["latency_distribution", "planner_gemini_fallback_contribution", "fallback_vs_nonfallback_latency", "token_distribution", "temporal_hint_strength_breakdown"],
        "failure_categories": ["retrieval_miss", "insufficient_temporal_coverage", "count_or_aggregation_insufficiency", "source_attribution_uncertainty", "premise_inconsistency", "audio_interference", "planner_or_routing_issue", "final_reasoning_error", "evidence_sufficient_but_answer_wrong", "technical_failure"],
        "lexical_metric_warning": "These lexical/reference-overlap metrics are diagnostic and are not treated as authoritative open-ended QA accuracy.",
        "gold_loading_order": ["save_raw_prediction", "save_validated_prediction", "load_gold_posthoc", "compute_metrics"],
    }


def selection_html(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(record.get(key, '')))}</td>" for key in ("case_id", "video_id", "question_type", "temporal_hint_strength", "provided_timestamp", "expected_modalities", "selection_tags")) + "</tr>"
        for record in records
    )
    return f"""<!doctype html><html lang='en'><meta charset='utf-8'><title>EgoSound pilot Phase 1</title><style>body{{font-family:Arial;margin:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;vertical-align:top}}.note{{background:#fff5cc;padding:1rem}}</style><h1>EgoSound Pilot 20 — Phase 1 Selection</h1><p class='note'>Selection used question metadata and timestamps only. Gold answers and prior prediction correctness were not selection inputs. Live API/model calls: 0.</p><h2>Summary</h2><ul><li>Cases: {summary['case_count']}</li><li>Existing/new: {summary['existing_case_count']}/{summary['new_case_count']}</li><li>Indexed videos reused: {html.escape(', '.join(summary['indexed_video_ids']))}</li></ul><h2>Selected cases</h2><table><tr><th>Case</th><th>Video</th><th>Question type</th><th>Hint</th><th>Provided timestamp</th><th>Expected modalities</th><th>Selection tags</th></tr>{rows}</table><h2>Future report contract</h2><p>{html.escape(pilot_report_schema()['lexical_metric_warning'])}</p></html>"""
