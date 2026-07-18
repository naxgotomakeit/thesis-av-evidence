"""Run the fixed three-case generalized canonical Baseline v1 smoke test.

Without ``--execute-live`` this command performs validation and writes a
zero-call manifest.  Live mode runs the three allowlisted cases sequentially
with one shared FreshQueryScorer and resume-safe per-case checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_pipeline.live_boundaries import (  # noqa: E402
    LazyWhisperFallback,
    anthropic_requester,
    environment_presence,
)
from src.canonical_pipeline.fingerprint import build_run_fingerprint  # noqa: E402
from src.canonical_pipeline.query_scoring import FreshQueryScorer  # noqa: E402
from src.canonical_pipeline.runner import CanonicalOnlineRunner  # noqa: E402
from src.canonical_pipeline.smoke_trace import (  # noqa: E402
    SMOKE_CASES,
    augment_checkpoint_temporal_audit,
    case_runtime_trace,
    reconstruct_evidence_funnel,
    stage_summary,
    write_html,
)
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402
from src.evaluation.pilot_store import CaseCheckpointStore  # noqa: E402
from src.evaluation.registry import baseline_metric_registry  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data/manifests/egosound_pilot_20.json"
OUT = ROOT / "outputs/pilot_20/baseline_v1/new_case_smoke_v0_1"
CHECKPOINTS = OUT / "checkpoints"


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


def frozen_artifact_hashes(config: Any) -> dict[str, str]:
    """Hash frozen stage outputs that must remain read-only."""
    paths = [config.path(name) for name in ("planner_plans", "task5b_frozen", "task5c_frozen", "task6_frozen", "task7a_frozen")]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def validate_smoke_cases(runner: CanonicalOnlineRunner) -> list[dict[str, Any]]:
    """Validate the exact allowlist without retaining gold/reference fields."""
    rows = []
    for case_id in SMOKE_CASES:
        safe = runner.validate_case(case_id)
        rows.append({
            "case_id": case_id,
            "accepted": True,
            "video_id": safe["video_id"],
            "question_present": bool(safe["question"]),
            "source_video_available": Path(str(safe["source_video_path"])).is_file(),
            "source_audio_available": Path(str(safe["source_audio_path"])).is_file(),
            "fresh_scoring_enabled": runner.query_scorer is not None,
            "historical_task4_execution_dependency": False,
            "local_wav_capability": True,
            "task6_serialization_contract": "validated_before_task7a",
        })
    return rows


def _split_rows(cases: list[dict[str, Any]]) -> None:
    """Persist every required runtime trace after each completed case."""
    _write_jsonl(OUT / "case_results.jsonl", cases)
    _write_jsonl(OUT / "planner_traces.jsonl", [{"case_id": row["case_id"], **row["planner"]} for row in cases])
    _write_jsonl(OUT / "initial_retrieval_candidates.jsonl", [{"case_id": row["case_id"], **row["initial_retrieval"]} for row in cases])
    _write_jsonl(OUT / "temporal_linking_traces.jsonl", [row["temporal_linking"] for row in cases])
    _write_jsonl(OUT / "refinement_traces.jsonl", [{"case_id": row["case_id"], **row["refinement"]} for row in cases])
    _write_jsonl(OUT / "sufficiency_fallback_traces.jsonl", [{"case_id": row["case_id"], **row["sufficiency_fallback"]} for row in cases])
    _write_jsonl(OUT / "reranking_traces.jsonl", [{"case_id": row["case_id"], **row["reranking"]} for row in cases])
    _write_jsonl(OUT / "evidence_funnels.jsonl", [row["evidence_funnel"] for row in cases])
    _write_jsonl(OUT / "final_payloads.jsonl", [{"case_id": row["case_id"], "payload": row["final_payload"], "preflight": row["preflight"]} for row in cases])
    _write_jsonl(OUT / "timings.jsonl", [{"case_id": row["case_id"], **item} for row in cases for item in row["timings"]])
    _write_jsonl(OUT / "model_usage.jsonl", [{"case_id": row["case_id"], **row["model_usage"]} for row in cases])


def _media_efficiency(case: dict[str, Any]) -> dict[str, Any]:
    evidence = case["final_selected_evidence"]
    frames = [frame for item in evidence if item["modality"] == "visual" for frame in item.get("frames", [])]
    audio = [item for item in evidence if item["modality"] == "acoustic"]
    speech = [item for item in evidence if item["modality"] == "speech"]
    usage = case["model_usage"]
    planner = usage.get("planner") or {}
    gemini = (usage.get("gemini") or {}).get("usage", {})
    return {
        "case_id": case["case_id"],
        "visual_frame_count": len(frames),
        "audio_clip_count": len(audio),
        "audio_duration_sec": sum(max(0.0, float(item["end_sec"]) - float(item["start_sec"])) for item in audio),
        "speech_segment_count": len(speech),
        "transcript_character_count": sum(len(str(item.get("transcript", ""))) for item in speech),
        "evidence_group_count": len((case["final_payload"] or {}).get("evidence_groups", [])),
        "final_evidence_count": len(evidence),
        "planner_api_calls": planner.get("api_calls"),
        "planner_input_tokens": planner.get("input_tokens"),
        "planner_output_tokens": planner.get("output_tokens"),
        "gemini_api_calls": (usage.get("gemini") or {}).get("total_api_calls"),
        "gemini_input_tokens": gemini.get("input_tokens"),
        "gemini_output_tokens": gemini.get("output_tokens"),
        "gemini_total_tokens": gemini.get("total_tokens"),
        "gemini_cached_tokens": gemini.get("total_cached_tokens"),
        "gemini_uncached_input_tokens": gemini.get("uncached_input_tokens"),
        "gemini_cache_hit": gemini.get("cache_hit"),
        "gemini_cache_hit_rate": gemini.get("cache_hit_rate"),
        "fallback_model_calls": (usage.get("fallback") or {}).get("fallback_model_calls", 0),
        "total_external_api_calls": sum(int(value or 0) for value in (planner.get("api_calls"), (usage.get("gemini") or {}).get("total_api_calls"))),
    }


def _timing(case: dict[str, Any], name: str) -> float | None:
    item = next((row for row in case["timings"] if row["stage_name"] == name), None)
    return None if not item else item.get("duration_sec")


def _top_level_decomposition(cases: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in (
        "Planner", "Fresh retrieval/query scoring", "Temporal/refinement",
        "Sufficiency/fallback", "Reranking", "Payload/media", "Gemini",
        "Validation", "Overhead",
    )}
    for case in cases:
        retrieval = float(_timing(case, "visual_retrieval") or 0) + float(_timing(case, "speech_retrieval") or 0) + float(_timing(case, "acoustic_retrieval") or 0)
        # Child retrieval timings include query encode/search and branch work;
        # local visual refinement is subtracted and reported separately.
        visual_refine = float(_timing(case, "local_visual_refinement") or 0)
        local_wav = float(_timing(case, "local_wav_materialization") or 0)
        components = {
            "Planner": float(_timing(case, "question_planner") or 0),
            "Fresh retrieval/query scoring": max(0.0, retrieval - visual_refine),
            "Temporal/refinement": float(_timing(case, "temporal_linking") or 0) + visual_refine,
            "Sufficiency/fallback": max(0.0, float(next((row["duration_sec"] for row in case["timings"] if row["stage_name"] == "evidence_sufficiency" and row.get("duration_sec") is not None), 0.0)) + float(_timing(case, "fallback_decision") or 0) + float(_timing(case, "fallback_execution") or 0)),
            "Reranking": float(_timing(case, "relation_construction") or 0) + float(_timing(case, "relation_reranking") or 0) + float(_timing(case, "evidence_packet_build") or 0),
            "Payload/media": float(_timing(case, "final_payload_build") or 0) + local_wav,
            "Gemini": float(_timing(case, "final_gemini_api") or 0),
            "Validation": float(_timing(case, "structured_parse") or 0) + float(_timing(case, "local_validation") or 0),
        }
        online = float(_timing(case, "online_end_to_end_total") or 0)
        components["Overhead"] = online - sum(components.values())
        for name, value in components.items():
            values[name].append(value)
    mean_online = statistics.fmean(float(_timing(case, "online_end_to_end_total") or 0) for case in cases)
    return {name: {"mean_sec_per_question": statistics.fmean(rows), "percent_mean_online": statistics.fmean(rows) / mean_online * 100 if mean_online else None} for name, rows in values.items()}


def _cold_warm(cases: list[dict[str, Any]], cold: dict[str, float]) -> dict[str, Any]:
    result = {}
    for modality, label in (("visual", "CLIP"), ("speech", "Sentence-T5"), ("acoustic", "CLAP")):
        encodes = [float(_timing(case, f"{modality}_query_encode")) for case in cases if _timing(case, f"{modality}_query_encode") is not None]
        searches = [float(_timing(case, f"{modality}_similarity_search")) for case in cases if _timing(case, f"{modality}_similarity_search") is not None]
        cold_sec = float(cold.get(modality, 0.0))
        result[label] = {
            "cold_load_sec": cold_sec,
            "mean_warm_query_encode_sec": statistics.fmean(encodes) if encodes else None,
            "mean_similarity_search_sec": statistics.fmean(searches) if searches else None,
            "amortized_cost_per_question_sec": cold_sec / len(SMOKE_CASES) + (sum(encodes) + sum(searches)) / len(SMOKE_CASES),
            "executed_query_count": len(encodes),
        }
    return result


def aggregate_results(cases: list[dict[str, Any]], cold: dict[str, float]) -> dict[str, Any]:
    efficiency = [_media_efficiency(case) for case in cases]
    online = [float(_timing(case, "online_end_to_end_total") or 0) for case in cases]
    warm_per_case = [sum(float(_timing(case, stage) or 0) for stage in (
        "visual_query_encode", "visual_similarity_search", "speech_query_encode",
        "speech_similarity_search", "acoustic_query_encode", "acoustic_similarity_search",
    )) for case in cases]
    funnels = [case["evidence_funnel"]["overall"] for case in cases]
    return {
        "case_count": len(cases),
        "completed_case_ids": [case["case_id"] for case in cases],
        "online_latency": {"mean_sec": statistics.fmean(online), "median_sec": statistics.median(online), "min_sec": min(online), "max_sec": max(online)},
        "stage_latency": stage_summary(cases),
        "top_level_decomposition": _top_level_decomposition(cases),
        "cold_vs_warm": _cold_warm(cases, cold),
        "cold_start_total_sec": sum(cold.values()),
        "warm_mean_query_scoring_sec_per_question": statistics.fmean(warm_per_case),
        "amortized_encoder_cost_per_question_sec": sum(cold.values()) / len(SMOKE_CASES) + statistics.fmean(warm_per_case),
        "fallback": {
            "case_count": sum(case["sufficiency_fallback"]["fallback_execution_count"] > 0 for case in cases),
            "rate": sum(case["sufficiency_fallback"]["fallback_execution_count"] > 0 for case in cases) / len(cases),
            "total_model_calls": sum(int(item["fallback_model_calls"] or 0) for item in efficiency),
            "executed_mean_sec": statistics.fmean([float(_timing(case, "fallback_execution")) for case in cases if _timing(case, "fallback_execution") is not None]) if any(_timing(case, "fallback_execution") is not None for case in cases) else None,
            "amortized_sec_per_question": sum(float(_timing(case, "fallback_execution") or 0) for case in cases) / len(cases),
        },
        "evidence_funnel_means": {key: statistics.fmean(float(item[key]) for item in funnels) for key in funnels[0]},
        "media_means": {
            "frames": statistics.fmean(item["visual_frame_count"] for item in efficiency),
            "audio_sec": statistics.fmean(item["audio_duration_sec"] for item in efficiency),
            "speech_segments": statistics.fmean(item["speech_segment_count"] for item in efficiency),
        },
        "api": {
            "total_calls": sum(item["total_external_api_calls"] for item in efficiency),
            "calls_per_question": statistics.fmean(item["total_external_api_calls"] for item in efficiency),
            "planner_calls": sum(int(item["planner_api_calls"] or 0) for item in efficiency),
            "gemini_calls": sum(int(item["gemini_api_calls"] or 0) for item in efficiency),
        },
        "tokens": {
            "planner_input_total": sum(int(item["planner_input_tokens"] or 0) for item in efficiency),
            "planner_output_total": sum(int(item["planner_output_tokens"] or 0) for item in efficiency),
            "gemini_input_total": sum(int(item["gemini_input_tokens"] or 0) for item in efficiency),
            "gemini_output_total": sum(int(item["gemini_output_tokens"] or 0) for item in efficiency),
            "gemini_total": sum(int(item["gemini_total_tokens"] or 0) for item in efficiency),
        },
        "cache": {
            "hit_cases": sum(item["gemini_cache_hit"] is True for item in efficiency),
            "field_unavailable_cases": sum(item["gemini_cached_tokens"] is None for item in efficiency),
            "cached_tokens_total": sum(int(item["gemini_cached_tokens"] or 0) for item in efficiency),
            "uncached_input_tokens_total": sum(int(item["gemini_uncached_input_tokens"] or 0) for item in efficiency),
        },
        "efficiency_per_case": efficiency,
    }


def _load_posthoc(manifest_path: Path, cases: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load gold only after runtime checkpoints and predictions are durable."""
    if not all(case.get("validated_answer") and case.get("raw_model_output") for case in cases):
        raise RuntimeError("Gold load blocked until every prediction is saved and validated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = {case["case_id"] for case in cases}
    rows = {row["case_id"]: row for row in manifest if row["case_id"] in wanted}
    corpus = [str(rows[case_id]["answer"]) for case_id in SMOKE_CASES if case_id in rows]
    registry = baseline_metric_registry()
    posthoc: dict[str, dict[str, Any]] = {}
    metrics_rows = []
    for case in cases:
        row = rows[case["case_id"]]
        validated = case["validated_answer"] or {}
        metric_case = {"case_id": case["case_id"], "gold_reference_answer": row["answer"], "validated_prediction": validated.get("answer")}
        metrics = [item.to_dict() for item in registry.evaluate(metric_case, {"reference_corpus": corpus})]
        metric_map = {item["name"]: item for item in metrics}
        operation = (case["planner"]["structured_output"] or {}).get("answer_requirement", {}).get("operation")
        status = validated.get("answer_status")
        ambiguities = case["sufficiency_fallback"].get("ambiguity_flags", [])
        before_modalities = {item.get("modality") for item in case["reranking"].get("before", [])}
        final_modalities = {item.get("modality") for item in case.get("final_selected_evidence", [])}
        requested_modalities = set((case["planner"]["structured_output"] or {}).get("resolver_modalities", []))
        lost_requested_modalities = sorted((before_modalities & requested_modalities) - final_modalities)
        if status == "query_or_premise_inconsistent":
            first_point, basis = "query/premise issue", "final status reports evidence-observed premise inconsistency"
        elif status == "insufficient_evidence" and lost_requested_modalities:
            first_point, basis = "reranking/evidence compaction", f"retrieved requested modalities were absent from the final payload: {lost_requested_modalities}"
        elif status == "insufficient_evidence":
            first_point, basis = "evidence coverage", "answer-required evidence remained structurally insufficient"
        elif operation == "count_occurrences" and not case["coverage_audit"].get("complete_temporal_coverage"):
            first_point, basis = "evidence coverage", "count operation has uncovered requested intervals"
        elif any("speaker" in str(item).casefold() for item in ambiguities):
            first_point, basis = "source attribution", "pipeline retained speaker-attribution ambiguity"
        elif status == "answered_with_uncertainty":
            first_point, basis = "human_review_required", "material final uncertainty remains; no semantic judge was used"
        elif metric_map["normalized_exact_match"]["value"] is False:
            first_point, basis = "human_review_required", "lexical mismatch alone cannot localize an open-ended QA error"
        else:
            first_point, basis = "not_detected", "no structural failure was deterministically identified"
        posthoc[case["case_id"]] = {
            "case_id": case["case_id"], "question_type": row.get("question_type"),
            "temporal_hint_strength": row.get("temporal_hint_strength"),
            "provided_timestamp": row.get("provided_timestamp"),
            "provided_timestamp_available": row.get("provided_timestamp_available"),
            "provided_timestamp_runtime_input": False,
            "gold_answer": row["answer"], "metrics": metrics,
            "failure_localization": {"likely_first_failure_point": first_point, "basis": basis, "diagnostic_only": True},
            "gold_loaded_posthoc": True,
        }
        metrics_rows.append({"case_id": case["case_id"], "gold_answer": row["answer"], "validated_prediction": validated.get("answer"), "answer_status": status, "metrics": metrics, "metric_role": "diagnostic_reference_overlap"})
    return posthoc, metrics_rows


def _markdown(cases: list[dict[str, Any]], aggregate: dict[str, Any], posthoc: dict[str, dict[str, Any]], hashes_unchanged: bool) -> str:
    lines = ["# New-question live smoke v0.1", "", "Exactly three new pilot questions were executed in one process with shared warm query encoders.", "", "## Results", ""]
    for case in cases:
        evaluation = posthoc[case["case_id"]]
        answer = case["validated_answer"] or {}
        lines.extend([
            f"### {case['case_id']}", "",
            f"- Question: {case['question']}",
            f"- Gold (post-hoc): {evaluation['gold_answer']}",
            f"- Validated prediction: {answer.get('answer')}",
            f"- Status: `{answer.get('answer_status')}`",
            f"- Final evidence IDs: `{[item.get('evidence_id') for item in case['final_selected_evidence']]}`",
            f"- Fallback executions: {case['sufficiency_fallback']['fallback_execution_count']}",
            f"- Online latency: {_timing(case, 'online_end_to_end_total'):.6f}s",
            f"- Likely first failure point: `{evaluation['failure_localization']['likely_first_failure_point']}`", "",
        ])
    lines.extend([
        "## Aggregate", "",
        f"- Completed: {len(cases)}/3",
        f"- Mean online latency: {aggregate['online_latency']['mean_sec']:.6f}s",
        f"- Cold encoder startup: {aggregate['cold_start_total_sec']:.6f}s",
        f"- Warm query scoring mean/question: {aggregate['warm_mean_query_scoring_sec_per_question']:.6f}s",
        f"- Amortized encoder cost/question: {aggregate['amortized_encoder_cost_per_question_sec']:.6f}s",
        f"- Fallback rate: {aggregate['fallback']['rate']:.2%}",
        f"- Total external API calls: {aggregate['api']['total_calls']}",
        f"- Cache hit cases: {aggregate['cache']['hit_cases']}",
        f"- Frozen artifact hashes unchanged: `{hashes_unchanged}`", "",
        "Lexical metrics are diagnostic_reference_overlap only; semantic correctness remains for human review.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(SMOKE_CASES))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--execute-live", action="store_true")
    args = parser.parse_args()
    if tuple(args.cases) != SMOKE_CASES:
        raise SystemExit("This smoke test is restricted to 00002_1, 00018_3, and 00018_9 in that order")
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    prior_manifest_path = OUT / "run_manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8")) if prior_manifest_path.is_file() else {}

    load_dotenv(ROOT / ".env", override=False)
    config = load_canonical_config(ROOT)
    scorer = FreshQueryScorer(ROOT)
    runner = CanonicalOnlineRunner(
        config, manifest_path=manifest_path, data_root=args.data_root,
        query_scorer=scorer, materialization_root=OUT / "runtime_media",
    )
    readiness = validate_smoke_cases(runner)
    presence = environment_presence()
    before_hashes = frozen_artifact_hashes(config)
    manifest = {
        "run": "new_case_smoke_v0_1", "cases": list(SMOKE_CASES),
        "mode": "execute_live" if args.execute_live else "dry_run",
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "environment_presence": presence,
        "readiness": readiness,
        "historical_task4_execution_dependency": False,
        "gold_runtime_access": False,
        "planned_external_operations_per_case": {case_id: {"planner_initial_calls": 1, "gemini_initial_calls": 1, "fallback_local_model": "only_if_canonical_policy_triggers"} for case_id in SMOKE_CASES},
        "shared_encoder_process": True,
        "model_settings": {"gemini": "gemini-3.5-flash", "thinking_level": "low", "store": False},
        "frozen_hashes_before": before_hashes,
    }
    manifest["run_fingerprint"] = build_run_fingerprint(
        ROOT,
        config,
        manifest_path,
        (
            row["video_id"]
            for row in json.loads(manifest_path.read_text(encoding="utf-8"))
            if row["case_id"] in SMOKE_CASES
        ),
        planner_model=os.environ.get("ANTHROPIC_MODEL"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "run_manifest.json", manifest)
    if not args.execute_live:
        print(json.dumps({"dry_run": True, "cases": list(SMOKE_CASES), "readiness": readiness, "environment_presence": presence, "external_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    if not all(presence.values()):
        raise SystemExit("Required live environment configuration is unavailable; no API call was made")

    store = CaseCheckpointStore(CHECKPOINTS)
    completed = [store.load(case_id) for case_id in SMOKE_CASES]
    cases = [item for item in completed if item is not None]
    _split_rows(cases)

    # Cold model startup is intentionally outside every per-question online
    # timer and happens once for the persistent scorer.
    all_already_complete = all(store.load(case_id) is not None for case_id in SMOKE_CASES)
    prior_cold = prior_manifest.get("encoder_cold_start_sec")
    cold = prior_cold if all_already_complete and isinstance(prior_cold, dict) else scorer.preload("00002", ("visual", "speech", "acoustic"))
    manifest["encoder_cold_start_sec"] = cold
    manifest["cold_start_total_sec"] = sum(cold.values())
    _write_json(OUT / "run_manifest.json", manifest)

    try:
        if not all_already_complete:
            from google import genai

            planner_request, planner_model = anthropic_requester()
            final_client = genai.Client()
            whisper = LazyWhisperFallback(ROOT)
            manifest["planner_model"] = planner_model
            _write_json(OUT / "run_manifest.json", manifest)
        for case_id in SMOKE_CASES:
            if store.load(case_id) is not None:
                continue
            state = runner.run_live_case(
                case_id, planner_request=planner_request,
                fallback_executor=whisper.execute, final_client=final_client,
            )
            if state.fallback_execution_count > 1:
                raise RuntimeError(f"Fallback executed more than once: {case_id}")
            trace = case_runtime_trace(state)
            trace["run_fingerprint_sha256"] = manifest["run_fingerprint"]["fingerprint_sha256"]
            store.save(case_id, trace)
            cases = [item for item in (store.load(cid) for cid in SMOKE_CASES) if item is not None]
            _split_rows(cases)
            manifest["completed_cases"] = [item["case_id"] for item in cases]
            _write_json(OUT / "run_manifest.json", manifest)
    finally:
        scorer.close()

    cases = [store.load(case_id) for case_id in SMOKE_CASES]
    if any(item is None for item in cases):
        raise RuntimeError("Smoke run stopped before all three durable checkpoints were written")
    completed_cases = [item for item in cases if item is not None]
    for case in completed_cases:
        augment_checkpoint_temporal_audit(case)
        if case.get("evidence_funnel") is None:
            case["evidence_funnel"] = reconstruct_evidence_funnel(case)
        store.save(case["case_id"], case, overwrite=True)
    _split_rows(completed_cases)

    # Strict ordering: raw and validated results already exist in checkpoints
    # and case_results.jsonl before the manifest is reopened for gold.
    posthoc, metric_rows = _load_posthoc(manifest_path, completed_cases)
    _write_jsonl(OUT / "metrics.jsonl", metric_rows)

    aggregate = aggregate_results(completed_cases, cold)
    failure_rows = [{"case_id": case_id, **row["failure_localization"]} for case_id, row in posthoc.items()]
    _write_json(OUT / "aggregate_latency.json", {"stage_latency": aggregate["stage_latency"], "top_level_decomposition": aggregate["top_level_decomposition"], "online_latency": aggregate["online_latency"], "cold_vs_warm": aggregate["cold_vs_warm"]})
    _write_json(OUT / "aggregate_efficiency.json", {key: aggregate[key] for key in ("evidence_funnel_means", "media_means", "api", "tokens", "cache", "fallback", "efficiency_per_case")})
    _write_json(OUT / "failure_localization.json", failure_rows)

    after_hashes = frozen_artifact_hashes(config)
    manifest["frozen_hashes_after"] = after_hashes
    manifest["frozen_hashes_unchanged"] = before_hashes == after_hashes
    manifest["completed_cases"] = [item["case_id"] for item in completed_cases]
    manifest["external_api_calls"] = aggregate["api"]["total_calls"]
    manifest["gold_loaded_posthoc_only"] = True
    _write_json(OUT / "run_manifest.json", manifest)
    _write_json(OUT / "aggregate_results.json", {"aggregate": aggregate, "posthoc": posthoc, "frozen_hashes_unchanged": before_hashes == after_hashes})
    (OUT / "new_case_smoke_summary.md").write_text(_markdown(completed_cases, aggregate, posthoc, before_hashes == after_hashes), encoding="utf-8")
    write_html(OUT / "new_case_smoke_review.html", completed_cases, aggregate, posthoc)
    print(json.dumps({
        "completed_cases": manifest["completed_cases"],
        "frozen_hashes_unchanged": manifest["frozen_hashes_unchanged"],
        "external_api_calls": manifest["external_api_calls"],
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
