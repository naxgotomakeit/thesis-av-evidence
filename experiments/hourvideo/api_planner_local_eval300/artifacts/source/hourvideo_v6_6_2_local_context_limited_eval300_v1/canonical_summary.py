from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.canonical_question_route_summary_v1.core import (
    canonical_json_sha256,
    distribution,
    freeze_tree,
    load_json,
    load_jsonl,
    now_utc,
    ordered_identity_sha256,
    score_after_structural_pass,
    sha256_file,
    summarize_attempts,
    usage_summary,
    validate_fixed_population,
    write_json,
    write_jsonl,
)


VERSION = "hourvideo_v6_6_2_local_context_limited_eval300_canonical_summary_v1"
SIDES = ("r1_av", "r3_2")


def _root_paths(root: Path, config_path: Path) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    cfg = load_json(config_path)
    experiment = root / "outputs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1"
    live = root / cfg["output_root"]
    canonical = experiment / "canonical_summary_v1"
    eligibility = root / cfg["eligibility_root"]
    return cfg, experiment, live, canonical, eligibility


def freeze_raw_outputs(root: Path, config_path: Path) -> dict[str, Any]:
    cfg, experiment, live, canonical, _ = _root_paths(root, config_path)
    canonical.mkdir(parents=True, exist_ok=True)
    result = freeze_tree(experiment, excluded_roots=[canonical])
    result.update({
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "live_root": str(live),
        "old_validation_report_path": str(live / "validation_report.json"),
        "old_validation_report_sha256": sha256_file(live / "validation_report.json"),
        "freeze_scope": "entire existing formal experiment output tree excluding versioned canonical_summary_v1",
    })
    write_json(canonical / "raw_formal_output_freeze_manifest.json", result)
    write_json(canonical / "OLD_VALIDATION_REPORT_INVALID_AGGREGATION_DO_NOT_USE.json", {
        "status": "INVALID_AGGREGATION_DO_NOT_USE",
        "created_at_utc": now_utc(),
        "invalid_artifact": str(live / "validation_report.json"),
        "invalid_artifact_sha256": result["old_validation_report_sha256"],
        "reason": "legacy top-level report was overwritten by the last route pass and mixed cross-route/live-gate cumulative telemetry",
        "artifact_modified_or_deleted": False,
        "replacement": str(canonical / "canonical_validation_report.json"),
    })
    return result


def _attempt_rows(case_dir: Path, qid: str, side: str) -> list[dict[str, Any]]:
    return [
        row for row in load_jsonl(case_dir / "model_attempts.jsonl")
        if row.get("question_id") == qid and (row.get("route") or row.get("side")) == side
    ]


def _request_rows(case_dir: Path, qid: str, side: str) -> list[dict[str, Any]]:
    return [
        row for row in load_jsonl(case_dir / "model_request_starts.jsonl")
        if row.get("question_id") == qid and (row.get("route") or row.get("side")) == side
    ]


def _event_rows(case_dir: Path, qid: str, side: str) -> list[dict[str, Any]]:
    return [
        row for row in load_jsonl(case_dir / "route_events.jsonl")
        if row.get("question_id") == qid and row.get("route") == side
    ]


def _selected_option(final_path: Path) -> str | None:
    if not final_path.is_file():
        return None
    answer = load_json(final_path).get("answer") or {}
    value = answer.get("selected_option_id")
    return str(value).strip().upper() if value is not None else None


def _failure_kind(status: dict[str, Any]) -> str | None:
    if status.get("formal_status") == "context_overflow_pre_model" or status.get("failure_kind") == "context_overflow_pre_model":
        return "context_overflow_pre_model"
    explicit = status.get("failure_kind")
    if explicit:
        return str(explicit)
    if status.get("execution_status") != "failed":
        return None
    reason = str(status.get("failure_reason") or "").lower()
    if "direct final contract exhausted" in reason:
        return "direct_final_contract_exhausted"
    if "maximum context length" in reason or "context_overflow" in reason:
        return "runtime_context_overflow"
    if "visual structured response reached max_tokens" in reason:
        return "fine_or_final_visual_max_tokens"
    if "timeout" in reason:
        return "timeout"
    if "provider" in reason or "http 4" in reason or "http 5" in reason:
        return "provider_error"
    return "unclassified_route_failure"


def _canonical_route(
    live: Path, capacity: dict[str, Any], gate_qid: str,
) -> tuple[dict[str, Any], list[str]]:
    qid, side = capacity["question_id"], capacity["route"]
    case_dir = live / "cases" / qid
    side_dir = case_dir / side
    errors: list[str] = []
    status_path = side_dir / "route_status.json"
    if not status_path.is_file():
        status: dict[str, Any] = {
            "question_id": qid, "side": side, "execution_status": "failed",
            "failure_kind": "missing_route_status", "prediction_present": False,
        }
        errors.append(f"missing route_status: {qid}::{side}")
    else:
        status = load_json(status_path)
    if status.get("question_id") != qid or (status.get("side") or status.get("route")) != side:
        errors.append(f"route_status identity mismatch: {qid}::{side}")
    attempts = _attempt_rows(case_dir, qid, side)
    requests = _request_rows(case_dir, qid, side)
    events = _event_rows(case_dir, qid, side)
    starts = [row for row in events if row.get("event") == "route_start"]
    ends = [row for row in events if row.get("event") == "route_end"]
    overflow = not bool(capacity["eligible"])
    if overflow:
        if attempts or requests or starts or ends:
            errors.append(f"context overflow route has live telemetry: {qid}::{side}")
        e2e = None
    else:
        if len(starts) != 1 or len(ends) != 1:
            errors.append(f"route start/end count is {len(starts)}/{len(ends)}, expected 1/1: {qid}::{side}")
            e2e = None
        else:
            if starts[0].get("route_uuid") != ends[0].get("route_uuid"):
                errors.append(f"route UUID mismatch: {qid}::{side}")
            e2e = float(ends[0].get("e2e_sec")) if ends[0].get("e2e_sec") is not None else None
        request_keys = [row.get("request_key") for row in requests]
        attempt_keys = [row.get("request_key") for row in attempts]
        if Counter(request_keys) != Counter(attempt_keys):
            errors.append(f"request/attempt UUID mismatch: {qid}::{side}")
    final_path = side_dir / "final_answer.json"
    selected = _selected_option(final_path)
    row = {
        "question_id": qid,
        "video_uid": capacity["video_uid"],
        "route": side,
        "route_key": f"{qid}::{side}",
        "eligible": bool(capacity["eligible"]),
        "formal_status": status.get("formal_status") or status.get("execution_status") or "failed",
        "execution_status": status.get("execution_status") or "failed",
        "reasoning_termination": status.get("reasoning_termination"),
        "failure_kind": _failure_kind(status),
        "failure_reason": status.get("failure_reason"),
        "prediction_present": selected is not None,
        "selected_option_id": selected,
        "counts_in_fixed_denominator": True,
        "e2e_sec": e2e,
        "e2e_in_latency_distribution": e2e is not None and not overflow,
        "post_planner_cost": summarize_attempts(attempts),
        "request_start_count": len(requests),
        "attempt_end_count": len(attempts),
        "request_attempt_pairing_valid": Counter(row.get("request_key") for row in requests) == Counter(row.get("request_key") for row in attempts),
        "route_event_start_count": len(starts),
        "route_event_end_count": len(ends),
        "source_phase": "live_gate_promoted_single_canonical_route" if qid == gate_qid else "formal_batch",
        "live_gate_counted_as_extra_route": False,
        "raw_artifacts": {
            "route_status": str(status_path),
            "final_answer": str(final_path) if final_path.is_file() else None,
            "attempt_jsonl": str(case_dir / "model_attempts.jsonl") if (case_dir / "model_attempts.jsonl").is_file() else None,
            "request_start_jsonl": str(case_dir / "model_request_starts.jsonl") if (case_dir / "model_request_starts.jsonl").is_file() else None,
            "route_event_jsonl": str(case_dir / "route_events.jsonl") if (case_dir / "route_events.jsonl").is_file() else None,
        },
    }
    if overflow:
        row["formal_status"] = "context_overflow_pre_model"
        row["execution_status"] = "failed"
        row["failure_kind"] = "context_overflow_pre_model"
    return row, errors


def _planner_summary(
    experiment: Path, capacities: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    planner_root = experiment / "frozen_planner"
    all_attempts = load_jsonl(planner_root / "planner_attempts.jsonl")
    errors: list[str] = []
    rows = []
    for capacity in capacities:
        if not capacity["eligible"]:
            continue
        qid, side = capacity["question_id"], capacity["route"]
        planner_path = planner_root / "cases" / qid / side / "planner.json"
        route_attempts = [
            row for row in all_attempts
            if row.get("event") == "attempt_end" and row.get("question_id") == qid and row.get("route") == side
        ]
        if not planner_path.is_file():
            errors.append(f"missing Frozen Planner: {qid}::{side}")
            accepted_usage = []
        else:
            planner = load_json(planner_path)
            if planner.get("question_id") != qid or planner.get("route") != side:
                errors.append(f"Frozen Planner identity mismatch: {qid}::{side}")
            usage = planner.get("usage") or {}
            accepted_usage = [{"stage": "planner", "status": "accepted", **usage}]
        rows.append({
            "question_id": qid, "route": side, "route_key": f"{qid}::{side}",
            "planner_present": planner_path.is_file(),
            "selected_accepted_planner": usage_summary(accepted_usage),
            "actual_all_planner_attempts_including_retries": usage_summary(route_attempts),
        })
    selected = []
    actual = []
    for row in rows:
        selected.append(row["selected_accepted_planner"])
        actual.append(row["actual_all_planner_attempts_including_retries"])
    def combine(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "requests": sum(row["requests"] for row in items),
            "input_tokens": sum(row["input_tokens"] for row in items),
            "output_tokens": sum(row["output_tokens"] for row in items),
            "model_latency_sec": sum(row["model_latency_sec"] for row in items),
        }
    return {
        "scope": "Frozen Planner generation; excluded from post-Planner online E2E",
        "eligible_route_count": len(rows),
        "route_rows": rows,
        "selected_accepted_planner_cost": combine(selected),
        "actual_all_planner_attempt_cost_including_retries": combine(actual),
    }, errors


def build_structural(root: Path, config_path: Path) -> dict[str, Any]:
    cfg, experiment, live, canonical, eligibility = _root_paths(root, config_path)
    freeze_path = canonical / "raw_formal_output_freeze_manifest.json"
    if not freeze_path.is_file():
        freeze_raw_outputs(root, config_path)
    ordered = [row for row in Path(cfg["eval300_ordered_uid_path"]).read_text(encoding="utf-8").splitlines() if row]
    paired = [row for row in (eligibility / "common_eligible_question_ids.txt").read_text(encoding="utf-8").splitlines() if row]
    capacities = load_jsonl(eligibility / "route_capacity.jsonl")
    by_identity = {(row["question_id"], row["route"]): row for row in capacities}
    gate_qid = load_json(experiment / "preflight/live_gate_report.json")["gate_question_id"]
    routes = []
    errors: list[str] = []
    for qid in ordered:
        for side in SIDES:
            capacity = by_identity.get((qid, side))
            if capacity is None:
                errors.append(f"missing capacity identity: {qid}::{side}")
                continue
            row, row_errors = _canonical_route(live, capacity, gate_qid)
            routes.append(row)
            errors.extend(row_errors)
    report = validate_fixed_population(routes, ordered, paired)
    report["errors"] = errors + report["errors"]
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    planner, planner_errors = _planner_summary(experiment, capacities)
    report["errors"].extend(planner_errors)
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report.update({
        "schema_version": VERSION + "_structural_validation_v1",
        "created_at_utc": now_utc(),
        "gold_loaded": False,
        "ordered_eval300_sha256": ordered_identity_sha256(ordered),
        "paired_context_feasible_sha256": ordered_identity_sha256(paired),
        "canonical_route_identity_sha256": ordered_identity_sha256(row["route_key"] for row in routes),
        "gate_policy": {
            "gate_question_id": gate_qid,
            "extra_gate_records_counted": 0,
            "promoted_identity_count": 2,
            "reason": "the formal resume policy reused the sole durable gate trajectories; each identity appears exactly once",
        },
        "old_validation_report_status": "INVALID_AGGREGATION_DO_NOT_USE",
    })
    write_jsonl(canonical / "canonical_routes.jsonl", routes)
    write_json(canonical / "planner_cost.json", planner)
    write_json(canonical / "structural_validation.json", report)
    write_json(canonical / "population_manifests.json", {
        "eval300": {"count": 300, "ordered_question_ids": ordered, "sha256": ordered_identity_sha256(ordered)},
        "eval297": {"count": 297, "ordered_question_ids": [qid for qid in ordered if qid not in set(cfg["pilot10_question_ids"])], "excluded_pilot_question_ids": cfg["pilot10_question_ids"]},
        "paired_context_feasible_150": {"count": 150, "ordered_question_ids": paired, "sha256": ordered_identity_sha256(paired), "frozen_before_execution": True},
    })
    return report


def _aggregate_route_cost(routes: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row["post_planner_cost"]["selected_successful_attempts"]
        for row in routes if row["prediction_present"]
    ]
    actual = [row["post_planner_cost"]["actual_all_attempts_including_retries"] for row in routes]
    fields = ("requests", "input_tokens", "output_tokens", "model_latency_sec", "logical_fine_evidence", "physical_image_transmissions")
    return {
        "selected_final_path_cost": {field: sum(row[field] for row in selected) for field in fields},
        "actual_total_cost_including_retries": {field: sum(row[field] for row in actual) for field in fields},
        "selected_final_path_definition": "accepted/success attempts only, and only for routes with a durable valid prediction",
        "actual_total_definition": "all actual attempts on every executed route, including failed routes and retries",
    }


def _completion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "denominator": len(rows),
        "prediction_count": sum(row["prediction_present"] for row in rows),
        "missing_prediction_count": sum(not row["prediction_present"] for row in rows),
        "strict_completion": sum(row["prediction_present"] for row in rows) / len(rows),
        "execution_status_counts": dict(sorted(Counter(row["execution_status"] for row in rows).items())),
        "formal_status_counts": dict(sorted(Counter(row["formal_status"] for row in rows).items())),
        "failure_kind_counts": dict(sorted(Counter(str(row.get("failure_kind") or "none") for row in rows).items())),
    }


def build_scored(root: Path, config_path: Path) -> dict[str, Any]:
    # Imported only after the structural no-gold gate passes.  This keeps the
    # freeze/structural phases independent from the evaluator dependency graph.
    from experiments.hourvideo_v6_6_2_shared_coarse_contract_v1 import formal_eval300
    from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import _find_gold
    cfg, experiment, live, canonical, eligibility = _root_paths(root, config_path)
    structural = load_json(canonical / "structural_validation.json")
    routes = load_jsonl(canonical / "canonical_routes.jsonl")
    populations = load_json(canonical / "population_manifests.json")
    ordered = populations["eval300"]["ordered_question_ids"]
    paired = populations["paired_context_feasible_150"]["ordered_question_ids"]

    def load_gold() -> dict[str, str]:
        annotations = load_json(Path(cfg["annotation_path"]))
        return {qid: _find_gold(annotations, qid) for qid in ordered}

    def score(gold: dict[str, str]) -> dict[str, Any]:
        predictions = [{
            "question_id": row["question_id"], "route": row["route"],
            "selected_option_id": row["selected_option_id"],
            "execution_status": row["execution_status"], "failure_kind": row["failure_kind"],
            "reasoning_termination": row["reasoning_termination"],
        } for row in routes]
        route_scores = {
            side: formal_eval300.score_fixed_population(
                ordered, predictions, gold, set(cfg["pilot10_question_ids"]), side
            )
            for side in SIDES
        }
        paired_set = set(paired)
        paired_scores = {}
        for side in SIDES:
            full_rows = route_scores[side]["full_eval300"]["rows"]
            selected_rows = [row for row in full_rows if row["question_id"] in paired_set]
            if [row["question_id"] for row in selected_rows] != paired:
                raise RuntimeError(f"paired-150 order/identity drift for {side}")
            paired_scores[side] = {
                "denominator": 150,
                "correct": sum(row["correct"] for row in selected_rows),
                "accuracy": sum(row["correct"] for row in selected_rows) / 150.0,
                "rows": selected_rows,
                "population_selected_before_execution": True,
            }
        return {"routes": route_scores, "paired_context_feasible_150": paired_scores}

    scores = score_after_structural_pass(structural, load_gold, score)
    by_side = {side: [row for row in routes if row["route"] == side] for side in SIDES}
    paired_set = set(paired)
    subsets = {
        "r1_eval300": by_side["r1_av"],
        "r3_capacity_aware_eval300": by_side["r3_2"],
        "r1_eval297": [row for row in by_side["r1_av"] if row["question_id"] not in set(cfg["pilot10_question_ids"])],
        "r3_eval297": [row for row in by_side["r3_2"] if row["question_id"] not in set(cfg["pilot10_question_ids"])],
        "r1_paired_context_feasible_150": [row for row in by_side["r1_av"] if row["question_id"] in paired_set],
        "r3_paired_context_feasible_150": [row for row in by_side["r3_2"] if row["question_id"] in paired_set],
    }
    latency = {
        name: {
            "post_planner_route_e2e_sec": distribution(row["e2e_sec"] for row in rows if row["e2e_in_latency_distribution"]),
            "denominator": len(rows),
            "excluded_context_overflow": sum(row["formal_status"] == "context_overflow_pre_model" for row in rows),
            "missing_e2e_not_imputed": sum(row["e2e_sec"] is None and row["formal_status"] != "context_overflow_pre_model" for row in rows),
        } for name, rows in subsets.items()
    }
    cost = {name: {"denominator": len(rows), **_aggregate_route_cost(rows)} for name, rows in subsets.items()}
    completion = {name: _completion(rows) for name, rows in subsets.items()}
    failures = {
        name: {
            "denominator": len(rows),
            "failure_kind_counts": dict(sorted(Counter(str(row.get("failure_kind") or "none") for row in rows).items())),
            "failed_route_keys": [row["route_key"] for row in rows if row["execution_status"] == "failed"],
            "missing_prediction_route_keys": [row["route_key"] for row in rows if not row["prediction_present"]],
        } for name, rows in subsets.items()
    }
    write_json(canonical / "accuracy.json", {"schema_version": VERSION + "_accuracy_v1", "public_evaluator": "score_fixed_population", **scores})
    write_json(canonical / "completion.json", completion)
    write_json(canonical / "latency.json", latency)
    write_json(canonical / "calls_tokens_images.json", cost)
    write_json(canonical / "failure_classification.json", failures)
    report = {
        "schema_version": VERSION + "_validation_v1",
        "status": "PASS",
        "created_at_utc": now_utc(),
        "structural_validation_status": structural["status"],
        "gold_loaded_only_after_structural_pass": True,
        "old_validation_report_status": "INVALID_AGGREGATION_DO_NOT_USE",
        "canonical_route_count": len(routes),
        "accuracy_artifact": str(canonical / "accuracy.json"),
        "completion_artifact": str(canonical / "completion.json"),
        "latency_artifact": str(canonical / "latency.json"),
        "cost_artifact": str(canonical / "calls_tokens_images.json"),
        "failure_artifact": str(canonical / "failure_classification.json"),
    }
    write_json(canonical / "canonical_validation_report.json", report)
    return report


def build_manifest(root: Path, config_path: Path) -> dict[str, Any]:
    _, _, _, canonical, _ = _root_paths(root, config_path)
    manifest_path = canonical / "canonical_manifest.json"
    files = []
    for path in sorted(canonical.iterdir()):
        if not path.is_file() or path == manifest_path:
            continue
        files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    tree_sha = hashlib.sha256("".join(f"{row['sha256']}  {row['path']}\n" for row in files).encode()).hexdigest()
    code_paths = [
        Path(__file__),
        root / "src/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/formal_runtime.py",
        root / "src/experiments/canonical_question_route_summary_v1/core.py",
        root / "scripts/experiments/build_hourvideo_v6_6_2_local_context_limited_canonical_summary_v1.py",
        root / "tests/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/test_canonical_summary.py",
        root / "tests/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/test_formal_contract.py",
        config_path,
    ]
    result = {
        "schema_version": VERSION + "_manifest_v1",
        "created_at_utc": now_utc(),
        "status": "CANONICAL_COMPLETE",
        "source_tree_frozen_before_aggregation": True,
        "no_model_or_api_calls": True,
        "canonical_tree_sha256": tree_sha,
        "files": files,
        "code_and_config": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in code_paths
        ],
    }
    write_json(manifest_path, result)
    return result


def run_all(root: Path, config_path: Path) -> dict[str, Any]:
    freeze = freeze_raw_outputs(root, config_path)
    structural = build_structural(root, config_path)
    if structural["status"] != "PASS":
        return {"status": "BLOCKED_STRUCTURAL_VALIDATION", "freeze": freeze, "structural": structural}
    scored = build_scored(root, config_path)
    manifest = build_manifest(root, config_path)
    return {"status": "CANONICAL_COMPLETE", "freeze": freeze, "structural": structural, "scored": scored, "manifest": manifest}
