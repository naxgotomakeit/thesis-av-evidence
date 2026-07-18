"""Execute the frozen 20-case EgoSound Baseline v1 pilot resume-safely."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_pipeline.live_boundaries import LazyWhisperFallback, anthropic_requester, environment_presence  # noqa: E402
from src.canonical_pipeline.fingerprint import build_run_fingerprint  # noqa: E402
from src.canonical_pipeline.query_scoring import FreshQueryScorer  # noqa: E402
from src.canonical_pipeline.runner import CanonicalOnlineRunner  # noqa: E402
from src.canonical_pipeline.smoke_trace import augment_checkpoint_temporal_audit, case_runtime_trace, reconstruct_evidence_funnel  # noqa: E402
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402
from src.evaluation.full_pilot_reporting import (  # noqa: E402
    aggregate_all, enrich_initial_candidates, evaluate_posthoc, write_html,
)
from src.evaluation.pilot_store import CaseCheckpointStore  # noqa: E402


MANIFEST = ROOT / "data/manifests/egosound_pilot_20.json"
SMOKE_ROOT = ROOT / "outputs/pilot_20/baseline_v1/new_case_smoke_v0_1"
OUT = ROOT / "outputs/pilot_20/baseline_v1/full_pilot_v0_1"
CHECKPOINTS = OUT / "checkpoints"
REUSED_CASES = ("00002_1", "00018_3", "00018_9")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_hashes(config: Any) -> dict[str, str]:
    paths = [config.path(name) for name in ("planner_plans", "task5b_frozen", "task5c_frozen", "task6_frozen", "task7a_frozen")]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def safe_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Load only fields needed before prediction; omit answer/reference fields."""
    allowed = {
        "case_id", "video_id", "question", "question_type", "temporal_hint_strength",
        "expected_modality_stratum", "provided_timestamp_available", "provided_timestamp",
        "video_duration", "video_path", "audio_path",
    }
    return [{key: value for key, value in row.items() if key in allowed} for row in json.loads(path.read_text(encoding="utf-8"))]


def smoke_compatibility() -> dict[str, Any]:
    """Verify that reused results match the current frozen execution contract."""
    run = json.loads((SMOKE_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "all_three_completed": run.get("completed_cases") == list(REUSED_CASES),
        "frozen_hashes_unchanged": run.get("frozen_hashes_unchanged") is True,
        "fresh_scoring": run.get("historical_task4_execution_dependency") is False,
        "shared_encoder_process": run.get("shared_encoder_process") is True,
        "gemini_model": run.get("model_settings", {}).get("gemini") == "gemini-3.5-flash",
        "thinking_level": run.get("model_settings", {}).get("thinking_level") == "low",
        "store_false": run.get("model_settings", {}).get("store") is False,
    }
    traces = []
    for case_id in REUSED_CASES:
        path = SMOKE_ROOT / "checkpoints" / f"{case_id}.json"
        row = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        traces.append(bool(row.get("validated_answer") and row.get("timings") and row.get("final_payload") and row.get("runtime_gold_loaded") is False))
    checks["trace_schema_compatible"] = all(traces)
    return {"compatible": all(checks.values()), "checks": checks, "source_manifest": run}


def seed_reused_cases(store: CaseCheckpointStore, compatibility: dict[str, Any]) -> None:
    if not compatibility["compatible"]:
        raise RuntimeError(f"Generalized smoke results are not reusable: {compatibility['checks']}")
    for case_id in REUSED_CASES:
        if store.load(case_id) is not None:
            continue
        source = json.loads((SMOKE_ROOT / "checkpoints" / f"{case_id}.json").read_text(encoding="utf-8"))
        source["source_run"] = "new_case_smoke_v0_1"
        source["reused_without_api_call"] = True
        store.save(case_id, source)


def validate_cases(runner: CanonicalOnlineRunner, safe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in safe_rows:
        safe = runner.validate_case(source["case_id"])
        output.append({
            "case_id": source["case_id"], "accepted": True, "video_id": safe["video_id"],
            "question_present": bool(safe["question"]),
            "indexes_and_source_media_available": Path(str(safe["source_video_path"])).is_file() and Path(str(safe["source_audio_path"])).is_file(),
            "fresh_scoring_enabled": runner.query_scorer is not None,
            "historical_task4_execution_dependency": False,
            "task6_contract": "producer_validated_before_task7a",
            "local_wav_capability": True,
        })
    return output


def _ordered_cases(store: CaseCheckpointStore, case_ids: list[str]) -> list[dict[str, Any]]:
    return [row for case_id in case_ids if (row := store.load(case_id)) is not None]


def write_runtime_outputs(cases: list[dict[str, Any]]) -> None:
    """Write gold-free incremental runtime artifacts after each case."""
    _write_jsonl(OUT / "case_results.jsonl", cases)
    _write_jsonl(OUT / "planner_traces.jsonl", [{"case_id": row["case_id"], "source_run": row.get("source_run"), **row["planner"]} for row in cases])
    _write_jsonl(OUT / "initial_retrieval_candidates.jsonl", [{"case_id": row["case_id"], "source_run": row.get("source_run"), **row["initial_retrieval"]} for row in cases])
    _write_jsonl(OUT / "temporal_filtering_traces.jsonl", [row["temporal_linking"] for row in cases])
    _write_jsonl(OUT / "relation_linking_traces.jsonl", [{"case_id": row["case_id"], "task5b_linked_windows": row["temporal_linking"].get("linked_windows", []), "task6_relations": row["reranking"].get("relations", [])} for row in cases])
    _write_jsonl(OUT / "refinement_traces.jsonl", [{"case_id": row["case_id"], **row["refinement"]} for row in cases])
    _write_jsonl(OUT / "sufficiency_fallback_traces.jsonl", [{"case_id": row["case_id"], **row["sufficiency_fallback"]} for row in cases])
    _write_jsonl(OUT / "evidence_funnels.jsonl", [row["evidence_funnel"] for row in cases])
    _write_jsonl(OUT / "final_payloads.jsonl", [{"case_id": row["case_id"], "payload": row["final_payload"], "preflight": row["preflight"]} for row in cases])
    _write_jsonl(OUT / "timings.jsonl", [{"case_id": row["case_id"], "source_run": row.get("source_run"), **item} for row in cases for item in row["timings"]])
    _write_jsonl(OUT / "model_usage.jsonl", [{"case_id": row["case_id"], "source_run": row.get("source_run"), **row["model_usage"]} for row in cases])


def _source_lifecycle_from_smoke() -> dict[str, Any]:
    run = json.loads((SMOKE_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    traces = [json.loads((SMOKE_ROOT / "checkpoints" / f"{case_id}.json").read_text(encoding="utf-8")) for case_id in REUSED_CASES]
    query_counts = {
        modality: sum(case["initial_retrieval"]["modalities"].get(modality, {}).get("executed") is True for case in traces)
        for modality in ("visual", "speech", "acoustic")
    }
    return {
        "source_run": "new_case_smoke_v0_1", "case_count": 3,
        "load_count": {modality: 1 for modality in ("visual", "speech", "acoustic")},
        "queries_served": query_counts,
        "instance_ids": "not_recorded_in_source_run",
        "proof": "single FreshQueryScorer, explicit preload once, shared_encoder_process=true, and zero per-case model-load timings",
        "cold_start_sec": run.get("encoder_cold_start_sec"),
    }


def _summary_markdown(cases: list[dict[str, Any]], aggregate: dict[str, Any], posthoc: dict[str, dict[str, Any]], reused: list[str], newly_run: list[str], hashes_unchanged: bool) -> str:
    lines = ["# Canonical Baseline v1 — EgoSound pilot 20", "", "Frozen baseline measurement; no research behavior was changed.", "", "## Execution", "", f"- Cases represented: {len(cases)}", f"- Reused generalized smoke cases: {', '.join(reused)}", f"- Newly executed cases: {len(newly_run)}", f"- Answer statuses: `{aggregate['answer_status_distribution']}`", f"- Frozen hashes unchanged: `{hashes_unchanged}`", "", "## Aggregate", "", f"- Online latency mean/median/p95: {aggregate['online_latency']['mean']:.3f}/{aggregate['online_latency']['median']:.3f}/{aggregate['online_latency']['p95']:.3f}s", f"- Diagnostic metrics: `{aggregate['metric_means']}`", f"- Fallback: `{aggregate['fallback']}`", f"- Task6 diagnostics: `{aggregate['task6']}`", f"- Counting diagnostics: `{aggregate['counting']}`", f"- Evidence means: `{aggregate['efficiency_means']}`", "", "## Cases", ""]
    for case in cases:
        answer = case.get("validated_answer") or {}
        meta = posthoc[case["case_id"]]
        lines.extend([f"### {case['case_id']}", "", f"- Question: {case['question']}", f"- Gold: {meta['gold_answer']}", f"- Prediction: {answer.get('answer')}", f"- Status: `{answer.get('answer_status')}`", f"- Human review: `unreviewed`", f"- Structural failure category: `{meta['failure_localization']['category']}`", ""])
    lines.extend(["Lexical metrics are diagnostic_reference_overlap only and are not authoritative semantic correctness.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    safe_rows = safe_manifest_rows(manifest_path)
    case_ids = [row["case_id"] for row in safe_rows]
    if len(case_ids) != 20 or len(set(case_ids)) != 20:
        raise SystemExit("Approved pilot manifest must contain exactly 20 unique cases")

    load_dotenv(ROOT / ".env", override=False)
    config = load_canonical_config(ROOT)
    scorer = FreshQueryScorer(ROOT)
    runner = CanonicalOnlineRunner(config, manifest_path=manifest_path, data_root=args.data_root, query_scorer=scorer, materialization_root=OUT / "runtime_media")
    compatibility = smoke_compatibility()
    readiness = validate_cases(runner, safe_rows)
    presence = environment_presence()
    before = frozen_hashes(config)
    OUT.mkdir(parents=True, exist_ok=True)
    store = CaseCheckpointStore(CHECKPOINTS)
    seed_reused_cases(store, compatibility)
    existing_before = set(store.completed_case_ids())
    incomplete = [case_id for case_id in case_ids if case_id not in existing_before]
    prior_manifest_path = OUT / "run_manifest.json"
    prior_manifest = (
        json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        if prior_manifest_path.is_file()
        else {}
    )
    partial_status_path = OUT / "partial_pilot_status.json"
    prior_partial_status = (
        json.loads(partial_status_path.read_text(encoding="utf-8"))
        if partial_status_path.is_file()
        else {}
    )
    prior_lifecycle_segments = copy.deepcopy(
        prior_manifest.get("lifecycle", {}).get("current_execution_segments")
        or prior_partial_status.get("model_lifecycle", {}).get("current_execution_segments")
        or []
    )
    prior_newly_executed = list(prior_manifest.get("newly_executed_cases", []))
    if not prior_newly_executed:
        prior_newly_executed = [
            case_id
            for case_id in case_ids
            if (checkpoint := store.load(case_id)) is not None
            and checkpoint.get("source_run") == "full_pilot_v0_1"
        ]
    run_manifest = {
        "canonical_versions": config.versions, "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "case_ids": case_ids, "reused_cases": list(REUSED_CASES), "incomplete_before_run": incomplete,
        "mode": "execute_live" if args.execute_live else "dry_run", "readiness": readiness,
        "smoke_compatibility": compatibility["checks"], "environment_presence": presence,
        "historical_task4_execution_dependency": False, "gold_runtime_access": False,
        "persistent_encoder_requirement": True,
        "planned_external_calls": {case_id: {"planner": 0 if case_id in existing_before else 1, "gemini": 0 if case_id in existing_before else 1, "fallback": "only_if_triggered"} for case_id in case_ids},
        "model_settings": {"gemini": "gemini-3.5-flash", "thinking_level": "low", "store": False},
        "frozen_hashes_before": before,
    }
    run_manifest["run_fingerprint"] = build_run_fingerprint(
        ROOT,
        config,
        manifest_path,
        (row["video_id"] for row in safe_rows),
        planner_model=os.environ.get("ANTHROPIC_MODEL"),
    )
    run_manifest["reused_result_fingerprint_policy"] = (
        "required_for_future_results; pre-fingerprint completed pilot retained as historical artifact"
    )
    if not incomplete:
        for key in (
            "planner_model",
            "current_process_cold_start_sec",
            "current_process_model_instance_ids",
            "current_process_lifecycle",
            "whisper_lifecycle",
            "newly_executed_cases",
        ):
            if key in prior_manifest:
                run_manifest[key] = prior_manifest[key]
    _write_json(
        OUT / ("run_manifest.json" if args.execute_live else "dry_run_manifest.json"),
        run_manifest,
    )
    write_runtime_outputs(_ordered_cases(store, case_ids))
    if not args.execute_live:
        print(json.dumps({"dry_run": True, "cases": len(case_ids), "reused": list(REUSED_CASES), "incomplete": incomplete, "readiness": sum(row["accepted"] for row in readiness), "environment_presence": presence, "external_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    if not all(presence.values()):
        raise SystemExit("Required live environment configuration is unavailable; no new API calls were made")

    lifecycle_segments = prior_lifecycle_segments
    newly_executed: list[str] = prior_newly_executed
    resumed_this_process: list[str] = []
    whisper = LazyWhisperFallback(ROOT)
    if incomplete:
        cold = scorer.preload(safe_rows[0]["video_id"], ("visual", "speech", "acoustic"))
        initial_lifecycle = scorer.lifecycle_audit()
        if not initial_lifecycle["persistent_reuse_invariant"] or any(item["load_count"] != 1 for item in initial_lifecycle["models"].values()):
            raise RuntimeError("Persistent encoder preload invariant failed before live cases")
        stable_ids = {name: item["instance_id"] for name, item in initial_lifecycle["models"].items()}
        from google import genai

        planner_request, planner_model = anthropic_requester()
        final_client = genai.Client()
        run_manifest["planner_model"] = planner_model
        run_manifest["current_process_cold_start_sec"] = cold
        run_manifest["current_process_model_instance_ids"] = stable_ids
        _write_json(OUT / "run_manifest.json", run_manifest)
        for case_id in incomplete:
            state = runner.run_live_case(case_id, planner_request=planner_request, fallback_executor=whisper.execute, final_client=final_client, return_preflight_blocked=True)
            if state.fallback_execution_count > 1:
                raise RuntimeError(f"Fallback executed more than once: {case_id}")
            trace = case_runtime_trace(state)
            trace["source_run"] = "full_pilot_v0_1"
            trace["reused_without_api_call"] = False
            trace["run_fingerprint_sha256"] = run_manifest["run_fingerprint"]["fingerprint_sha256"]
            enrich_initial_candidates(trace, ROOT)
            store.save(case_id, trace)
            if case_id not in newly_executed:
                newly_executed.append(case_id)
            resumed_this_process.append(case_id)
            audit = scorer.lifecycle_audit()
            if not audit["persistent_reuse_invariant"] or any(item["load_count"] != 1 for item in audit["models"].values()):
                raise RuntimeError("Query encoder reloaded inside the per-case loop")
            if {name: item["instance_id"] for name, item in audit["models"].items()} != stable_ids:
                raise RuntimeError("Query encoder instance identity changed inside the per-case loop")
            run_manifest["current_process_lifecycle"] = audit
            run_manifest["whisper_lifecycle"] = whisper.lifecycle_audit()
            run_manifest["completed_cases"] = store.completed_case_ids()
            _write_json(OUT / "run_manifest.json", run_manifest)
            write_runtime_outputs(_ordered_cases(store, case_ids))
        final_lifecycle = scorer.lifecycle_audit()
        expected_queries = {
            modality: sum((store.load(case_id) or {})["initial_retrieval"]["modalities"].get(modality, {}).get("executed") is True for case_id in resumed_this_process)
            for modality in ("visual", "speech", "acoustic")
        }
        observed_queries = {name: item["queries_served"] for name, item in final_lifecycle["models"].items()}
        if observed_queries != expected_queries:
            raise RuntimeError(f"Encoder query lifecycle mismatch: observed={observed_queries}, expected={expected_queries}")
        lifecycle_segments.append({"source_run": "full_pilot_v0_1_resume", "case_count": len(resumed_this_process), "case_ids": resumed_this_process, "encoder_audit": final_lifecycle, "whisper_audit": whisper.lifecycle_audit(), "cold_start_sec": cold})
        run_manifest["newly_executed_cases"] = newly_executed
    scorer.close()

    cases = _ordered_cases(store, case_ids)
    if len(cases) != 20:
        raise RuntimeError(f"Pilot stopped with {len(cases)}/20 durable results")
    for case in cases:
        augment_checkpoint_temporal_audit(case)
        if case.get("evidence_funnel") is None:
            case["evidence_funnel"] = reconstruct_evidence_funnel(case)
        enrich_initial_candidates(case, ROOT)
    write_runtime_outputs(cases)

    # Reopen the manifest for gold only after all 20 runtime outputs are durable.
    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    posthoc, metric_rows, task6_rows, count_rows, cross_rows = evaluate_posthoc(cases, manifest_rows)
    failures = [posthoc[case_id]["failure_localization"] for case_id in case_ids]
    lifecycle = {
        "reused_source_process": _source_lifecycle_from_smoke(),
        "current_execution_segments": lifecycle_segments,
        "reuse_boundary_note": "Three completed cases came from an earlier compatible persistent process. The 17 new cases span persistent process segments separated by the mandated technical stop; every segment reports its own load counts and instance identities rather than merging them.",
    }
    aggregate = aggregate_all(cases, posthoc, task6_rows, count_rows, cross_rows, lifecycle)
    aborted_path = OUT / "aborted_attempts.json"
    aborted_attempts = json.loads(aborted_path.read_text(encoding="utf-8")) if aborted_path.is_file() else []
    aggregate["aborted_attempts"] = aborted_attempts
    aggregate["planner_api"]["aborted_calls_with_usage_unavailable"] = sum(int(item.get("planner_calls", 0)) for item in aborted_attempts)
    aggregate["planner_api"]["total_calls_including_aborted"] = aggregate["planner_api"]["total_calls"] + aggregate["planner_api"]["aborted_calls_with_usage_unavailable"]
    smoke_cold = compatibility["source_manifest"].get("encoder_cold_start_sec", {})
    current_cold = run_manifest.get("current_process_cold_start_sec", {})
    completed_process_cold = [segment.get("cold_start_sec", {}) or {} for segment in lifecycle_segments]
    combined_cold = sum(float(value or 0) for value in smoke_cold.values()) + sum(
        float(value or 0)
        for segment in completed_process_cold
        for value in segment.values()
    )
    warm_mean = sum(sum(float((next((item for item in case["timings"] if item["stage_name"] == stage), {}) or {}).get("duration_sec") or 0) for stage in ("visual_query_encode", "visual_similarity_search", "speech_query_encode", "speech_similarity_search", "acoustic_query_encode", "acoustic_similarity_search")) for case in cases) / len(cases)
    aggregate["encoder_cost"] = {"reused_source_cold_start_sec": smoke_cold, "completed_live_process_cold_start_segments": completed_process_cold, "latest_process_cold_start_sec": current_cold, "combined_all_source_process_cold_start_sec": combined_cold, "warm_query_scoring_mean_per_question_sec": warm_mean, "amortized_combined_encoder_cost_per_question_sec": combined_cold / len(cases) + warm_mean}

    _write_jsonl(OUT / "metrics.jsonl", metric_rows)
    _write_jsonl(OUT / "task6_before_after.jsonl", [{"case_id": row["case_id"], "diagnostic": row, "before": next(case for case in cases if case["case_id"] == row["case_id"])["reranking"]["before"], "retained": next(case for case in cases if case["case_id"] == row["case_id"])["reranking"]["retained"], "actually_dropped": next(case for case in cases if case["case_id"] == row["case_id"])["reranking"]["actually_dropped"]} for row in task6_rows])
    _write_jsonl(OUT / "failure_localization.jsonl", failures)
    _write_json(OUT / "aggregate_latency.json", {key: aggregate[key] for key in ("online_latency", "stage_latency", "top_level_decomposition", "mean_timing_coverage", "encoder_cost")})
    _write_json(OUT / "aggregate_efficiency.json", {key: aggregate[key] for key in ("efficiency_means", "evidence_compression_ratio", "planner_api", "gemini_api", "fallback", "efficiency_per_case", "model_lifecycle")})
    _write_json(OUT / "aggregate_task6_diagnostics.json", {"summary": aggregate["task6"], "cases": task6_rows})
    _write_json(OUT / "aggregate_counting_diagnostics.json", {"summary": aggregate["counting"], "cases": count_rows})
    _write_json(OUT / "aggregate_crossmodal_diagnostics.json", {"summary": aggregate["crossmodal"], "cases": cross_rows})
    _write_json(OUT / "aggregate_metrics.json", {"summary": aggregate["metric_means"], "answer_status_distribution": aggregate["answer_status_distribution"], "temporal_hint_breakdown": aggregate["temporal_hint_breakdown"], "question_type_breakdown": aggregate["question_type_breakdown"], "modality_breakdown": aggregate["modality_breakdown"]})

    final_case_rows = []
    for case in cases:
        row = copy.deepcopy(case)
        row["posthoc_evaluation"] = posthoc[case["case_id"]]
        row["human_review_status"] = "unreviewed"
        final_case_rows.append(row)
    _write_jsonl(OUT / "case_results.jsonl", final_case_rows)
    after = frozen_hashes(config)
    run_manifest.update({"newly_executed_cases": newly_executed, "completed_cases": case_ids, "frozen_hashes_after": after, "frozen_hashes_unchanged": before == after, "gold_loaded_posthoc_only": True, "lifecycle": lifecycle})
    _write_json(OUT / "run_manifest.json", run_manifest)
    (OUT / "baseline_v1_pilot_20_summary.md").write_text(_summary_markdown(cases, aggregate, posthoc, list(REUSED_CASES), newly_executed, before == after), encoding="utf-8")
    write_html(OUT / "baseline_v1_pilot_20_review.html", cases, posthoc, aggregate, task6_rows, count_rows, cross_rows, failures)
    print(json.dumps({"completed": len(cases), "reused": list(REUSED_CASES), "newly_executed": newly_executed, "frozen_hashes_unchanged": before == after, "output": str(OUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
