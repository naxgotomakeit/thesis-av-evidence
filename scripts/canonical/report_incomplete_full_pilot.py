"""Build a post-hoc report after a mandated technical-stop condition.

This command performs no model, retrieval, or media calls. It reports only
durable validated checkpoints and marks the intended pilot as incomplete.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.canonical.run_full_pilot import (  # noqa: E402
    CHECKPOINTS,
    MANIFEST,
    OUT,
    REUSED_CASES,
    _source_lifecycle_from_smoke,
    _write_json,
    _write_jsonl,
    frozen_hashes,
)
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402
from src.evaluation.full_pilot_reporting import (  # noqa: E402
    aggregate_all,
    evaluate_posthoc,
    write_html,
)
from src.evaluation.pilot_store import CaseCheckpointStore  # noqa: E402


BLOCKER = {
    "case_id": "00018_2",
    "stage": "task6_candidate_accounting_invariant",
    "error_class": "RuntimeError",
    "safe_error": "Canonical Task 6 accounting inconsistency: 00018_2",
    "research_logic_modified": False,
}


def main() -> int:
    """Generate an explicitly incomplete dashboard from durable checkpoints."""
    manifest_rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    intended_ids = [row["case_id"] for row in manifest_rows]
    store = CaseCheckpointStore(CHECKPOINTS)
    cases = [case for case_id in intended_ids if (case := store.load(case_id)) is not None]
    completed_ids = [case["case_id"] for case in cases]
    incomplete_ids = [case_id for case_id in intended_ids if case_id not in completed_ids]
    if len(cases) != 16 or incomplete_ids != ["00018_2", "00061_4", "00004_2", "00006_7"]:
        raise RuntimeError(
            f"Unexpected partial-run boundary: completed={len(cases)}, incomplete={incomplete_ids}"
        )

    posthoc, metric_rows, task6_rows, count_rows, cross_rows = evaluate_posthoc(
        cases, manifest_rows
    )
    failures = [posthoc[case_id]["failure_localization"] for case_id in completed_ids]
    failures.append(
        {
            "case_id": "00018_2",
            "category": "technical failure",
            "evidence": BLOCKER,
            "automatic_causal_claim": False,
        }
    )

    run_manifest_path = OUT / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    new_ids = [case["case_id"] for case in cases if case.get("source_run") == "full_pilot_v0_1"]
    lifecycle = {
        "reused_source_process": _source_lifecycle_from_smoke(),
        "current_execution_segments": [
            {
                "source_run": "full_pilot_v0_1",
                "case_count": len(new_ids),
                "encoder_audit": run_manifest.get("current_process_lifecycle"),
                "whisper_audit": run_manifest.get("whisper_lifecycle"),
                "cold_start_sec": run_manifest.get("current_process_cold_start_sec"),
            }
        ],
        "reuse_boundary_note": (
            "Three cases came from the compatible earlier smoke process; "
            "13 durable new cases came from one persistent process."
        ),
    }
    aggregate = aggregate_all(cases, posthoc, task6_rows, count_rows, cross_rows, lifecycle)
    aborted_path = OUT / "aborted_attempts.json"
    aborted = json.loads(aborted_path.read_text(encoding="utf-8"))
    aggregate.update(
        {
            "pilot_status": "incomplete_technical_stop",
            "intended_case_count": 20,
            "completed_case_count": len(cases),
            "technical_failure_count": 1,
            "not_run_case_count": 3,
            "incomplete_case_ids": incomplete_ids,
            "blocking_failure": BLOCKER,
            "aborted_attempts": aborted,
        }
    )
    aggregate["planner_api"]["aborted_calls_with_usage_unavailable"] = sum(
        int(item.get("planner_calls") or 0) for item in aborted
    )
    aggregate["planner_api"]["total_calls_including_aborted"] = (
        aggregate["planner_api"]["total_calls"]
        + aggregate["planner_api"]["aborted_calls_with_usage_unavailable"]
    )

    smoke_cold = lifecycle["reused_source_process"].get("cold_start_sec") or {}
    current_cold = run_manifest.get("current_process_cold_start_sec") or {}
    combined_cold = sum(float(value or 0) for value in smoke_cold.values()) + sum(
        float(value or 0) for value in current_cold.values()
    )
    warm_mean = sum(
        sum(
            float(item.get("duration_sec") or 0)
            for item in case.get("timings", [])
            if item.get("stage_name")
            in {
                "visual_query_encode",
                "visual_similarity_search",
                "speech_query_encode",
                "speech_similarity_search",
                "acoustic_query_encode",
                "acoustic_similarity_search",
            }
        )
        for case in cases
    ) / len(cases)
    aggregate["encoder_cost"] = {
        "reused_source_cold_start_sec": smoke_cold,
        "current_process_cold_start_sec": current_cold,
        "combined_two_source_run_cold_start_sec": combined_cold,
        "warm_query_scoring_mean_per_completed_question_sec": warm_mean,
        "amortized_combined_encoder_cost_per_completed_question_sec": (
            combined_cold / len(cases) + warm_mean
        ),
    }

    final_rows = []
    for case in cases:
        row = copy.deepcopy(case)
        row["posthoc_evaluation"] = posthoc[case["case_id"]]
        row["human_review_status"] = "unreviewed"
        final_rows.append(row)
    _write_jsonl(OUT / "case_results.jsonl", final_rows)
    _write_jsonl(OUT / "metrics.jsonl", metric_rows)
    _write_jsonl(
        OUT / "task6_before_after.jsonl",
        [
            {
                "case_id": row["case_id"],
                "diagnostic": row,
                "before": next(case for case in cases if case["case_id"] == row["case_id"])["reranking"]["before"],
                "retained": next(case for case in cases if case["case_id"] == row["case_id"])["reranking"]["retained"],
                "actually_dropped": next(case for case in cases if case["case_id"] == row["case_id"])["reranking"]["actually_dropped"],
            }
            for row in task6_rows
        ],
    )
    _write_jsonl(OUT / "failure_localization.jsonl", failures)
    _write_json(
        OUT / "aggregate_latency.json",
        {
            key: aggregate[key]
            for key in (
                "pilot_status",
                "completed_case_count",
                "incomplete_case_ids",
                "online_latency",
                "stage_latency",
                "top_level_decomposition",
                "mean_timing_coverage",
                "encoder_cost",
            )
        },
    )
    _write_json(
        OUT / "aggregate_efficiency.json",
        {
            key: aggregate[key]
            for key in (
                "pilot_status",
                "efficiency_means",
                "evidence_compression_ratio",
                "planner_api",
                "gemini_api",
                "fallback",
                "efficiency_per_case",
                "model_lifecycle",
            )
        },
    )
    _write_json(OUT / "aggregate_task6_diagnostics.json", {"summary": aggregate["task6"], "cases": task6_rows})
    _write_json(OUT / "aggregate_counting_diagnostics.json", {"summary": aggregate["counting"], "cases": count_rows})
    _write_json(OUT / "aggregate_crossmodal_diagnostics.json", {"summary": aggregate["crossmodal"], "cases": cross_rows})
    _write_json(
        OUT / "aggregate_metrics.json",
        {
            "pilot_status": aggregate["pilot_status"],
            "completed_case_count": len(cases),
            "summary": aggregate["metric_means"],
            "answer_status_distribution": aggregate["answer_status_distribution"],
            "temporal_hint_breakdown": aggregate["temporal_hint_breakdown"],
            "question_type_breakdown": aggregate["question_type_breakdown"],
            "modality_breakdown": aggregate["modality_breakdown"],
        },
    )
    _write_json(OUT / "partial_pilot_status.json", aggregate)

    config = load_canonical_config(ROOT)
    after = frozen_hashes(config)
    before = run_manifest.get("frozen_hashes_before", {})
    run_manifest.update(
        {
            "pilot_status": "incomplete_technical_stop",
            "completed_cases": completed_ids,
            "incomplete_cases": incomplete_ids,
            "blocking_failure": BLOCKER,
            "newly_executed_cases": new_ids,
            "frozen_hashes_after": after,
            "frozen_hashes_unchanged": before == after,
            "gold_loaded_posthoc_only_for_durable_completed_cases": True,
            "lifecycle": lifecycle,
        }
    )
    _write_json(run_manifest_path, run_manifest)

    summary = [
        "# Canonical Baseline v1 — EgoSound pilot 20 (INCOMPLETE)",
        "",
        "> Technical stop: 16/20 durable cases. No Task6 or research behavior was changed.",
        "",
        f"- Blocking case/stage: `{BLOCKER['case_id']}` / `{BLOCKER['stage']}`",
        f"- Completed: {len(cases)} (3 reused + {len(new_ids)} new)",
        f"- Not completed: {', '.join(incomplete_ids)}",
        f"- Answer statuses among completed cases: `{aggregate['answer_status_distribution']}`",
        f"- Online latency mean/median/p95 (completed only): {aggregate['online_latency']['mean']:.3f}/{aggregate['online_latency']['median']:.3f}/{aggregate['online_latency']['p95']:.3f}s",
        f"- Diagnostic reference-overlap metrics (completed only): `{aggregate['metric_means']}`",
        f"- Frozen hashes unchanged: `{before == after}`",
        "",
        "Lexical metrics are diagnostic_reference_overlap only. This is not a completed 20-case baseline result.",
    ]
    (OUT / "baseline_v1_pilot_20_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    html_path = OUT / "baseline_v1_pilot_20_review.html"
    write_html(html_path, cases, posthoc, aggregate, task6_rows, count_rows, cross_rows, failures)
    html_text = html_path.read_text(encoding="utf-8")
    html_text = html_text.replace(
        "<h1>Canonical Baseline v1",
        "<div class='warning'><strong>INCOMPLETE TECHNICAL STOP: 16/20 durable cases.</strong> "
        "Task6 accounting invariant failed for 00018_2; 00061_4, 00004_2 and 00006_7 were not run. "
        "No research logic was changed.</div><h1>Canonical Baseline v1",
        1,
    ).replace("Executed N/20", "Executed N/16 (partial)")
    html_path.write_text(html_text, encoding="utf-8")
    print(json.dumps({"status": "incomplete_technical_stop", "completed": len(cases), "incomplete": incomplete_ids, "external_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
