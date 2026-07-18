from __future__ import annotations

import json
import inspect
from pathlib import Path

from scripts.canonical.run_full_pilot import (
    MANIFEST,
    REUSED_CASES,
    safe_manifest_rows,
    smoke_compatibility,
)
from src.canonical_pipeline.query_scoring import FreshQueryScorer
from src.canonical_pipeline.runner import CanonicalOnlineRunner
from src.evaluation.full_pilot_reporting import (
    counting_diagnostic,
    stage_aggregates,
    task6_diagnostic,
)


def _case() -> dict:
    return {
        "case_id": "case",
        "planner": {"structured_output": {"resolver_modalities": ["speech", "visual"], "primary_anchor_modality": "speech", "answer_requirement": {"operation": "compare_events"}}},
        "reranking": {
            "before": [{"candidate_id": "speech", "modality": "speech"}, {"candidate_id": "visual", "modality": "visual"}],
            "retained": [{"candidate_id": "visual_packet", "modality": "visual"}],
            "actually_dropped": [{"candidate_id": "speech", "reason": "supporting"}],
        },
        "temporal_linking": {"linked_windows": [{"source_candidate_ids": ["speech", "visual"]}]},
        "final_selected_evidence": [{"evidence_id": "visual_packet", "modality": "visual"}],
        "validated_answer": {"answer_status": "insufficient_evidence", "answer": None},
        "sufficiency_fallback": {"post_status": "sufficient", "fallback_execution_count": 0},
        "timings": [{"stage_name": "online_end_to_end_total", "duration_sec": 2.0, "executed": True, "skipped": False}],
    }


def test_manifest_safe_view_has_20_and_excludes_gold() -> None:
    rows = safe_manifest_rows(MANIFEST)
    assert len(rows) == 20
    assert all("answer" not in row and "provided_context" not in row for row in rows)


def test_smoke_reuse_is_exact_and_compatible() -> None:
    audit = smoke_compatibility()
    assert audit["compatible"] is True
    assert REUSED_CASES == ("00002_1", "00018_3", "00018_9")


def test_task6_diagnostic_detects_broken_cross_modal_pair() -> None:
    result = task6_diagnostic(_case())
    assert result["required_modality_lost"] is True
    assert result["lost_required_modalities"] == ["speech"]
    assert result["cross_modal_pair_broken"] is True
    assert result["relation_endpoint_dropped"] is True


def test_counting_diagnostic_flags_unsupported_exact_count() -> None:
    case = _case()
    case["planner"]["structured_output"]["answer_requirement"]["operation"] = "count_occurrences"
    case["coverage_audit"] = {"requested_interval": [0, 100], "merged_covered_intervals": [[0, 5]], "final_evidence_intervals": [[0, 5]], "coverage_fraction": 0.05, "uncovered_intervals": [[5, 100]], "complete_temporal_coverage": False, "detected_occurrences": []}
    case["sufficiency_fallback"]["post_status"] = "insufficient"
    case["validated_answer"] = {"answer_status": "answered", "answer": "0"}
    result = counting_diagnostic(case, {"question_type": "Counting", "answer": "4"})
    assert result is not None
    assert result["unsupported_exact_count"] is True


def test_skipped_stage_is_not_zero_in_aggregate() -> None:
    case = _case()
    rows = stage_aggregates([case])
    fallback = next(row for row in rows if row["stage"] == "fallback_execution")
    assert fallback["executed_count"] == 0
    assert fallback["mean_executed_sec"] is None
    assert fallback["amortized_mean_sec"] == 0.0


def test_lifecycle_audit_reports_process_identity_without_secrets() -> None:
    scorer = object.__new__(FreshQueryScorer)
    scorer.device = "cpu"
    scorer._models = {"visual": object(), "speech": object(), "acoustic": object()}
    scorer._load_counts = {"visual": 1, "speech": 1, "acoustic": 1}
    scorer._query_counts = {"visual": 10, "speech": 4, "acoustic": 6}
    scorer._instance_ids = {key: id(value) for key, value in scorer._models.items()}
    scorer._load_times = {"visual": 1.0, "speech": 2.0, "acoustic": 3.0}
    audit = scorer.lifecycle_audit()
    assert audit["persistent_reuse_invariant"] is True
    assert audit["models"]["visual"]["load_count"] == 1
    assert audit["models"]["visual"]["queries_served"] == 10
    assert "key" not in json.dumps(audit).casefold()


def test_preflight_blocked_return_is_opt_in_for_measurement_only() -> None:
    parameter = inspect.signature(CanonicalOnlineRunner.run_live_case).parameters[
        "return_preflight_blocked"
    ]
    assert parameter.default is False
    script = (Path(__file__).parents[2] / "scripts/canonical/run_full_pilot.py").read_text(
        encoding="utf-8"
    )
    assert "return_preflight_blocked=True" in script
