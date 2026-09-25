#!/usr/bin/env python3
"""Additive, offline finalization for the Flat-15 Eval300 formal run.

The ``gold-free`` command has no dataset/parquet argument and cannot read gold.
The ``score`` command is a separate invocation and refuses to run unless the
gold-free 300-UID prediction closure passes its recorded SHA-256 check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BASE = Path("/home/naxucl/data/HourVideo/videoseal_original/videoseal_flat15_eval300_formal_v1_20260910T184300Z")
OUT = Path("/home/naxucl/data/HourVideo/videoseal_original/videoseal_flat15_eval300_finalization_v2_20260914T003140Z")
RETRY_RAW = BASE / "retry_raw_v2_20260913T124100Z"
RETRY_CONTROL = BASE / "retry_control_v2_20260913T124100Z"
RETRY_PROTOCOL = BASE / "retry_protocol_fix_v1_20260913T041000Z"
FIRST_AUDIT = BASE / "protocol_integrity_audit_20260913T005805Z"
RECOVERY_AUDIT = BASE / "retry_infrastructure_audit_20260913T115931Z"
RECOVERY_SMOKE = BASE / "retry_recovery_smoke_20260913T124048Z"
RUNTIME = Path("/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/videoseal_flat15_eval300_preflight_v3_20260910T183640Z/runtime_overlay_v2")
PARQUET = Path("/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_dev_v1.0_videoseal_dgx.parquet")
COMPARISON_REPORT = Path("/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z/analysis/h15_vs_flat_data_archive_20260829T152745Z/FINAL_H15_VS_FLAT_DATA_ARCHIVE.md")

LEGAL = set("ABCDE")
METRIC_KEYS = {
    "uid", "status", "error_type", "elapsed_sec", "steps", "planner_calls",
    "visual_retrieve_calls", "visual_inspect_calls", "planner_tokens",
    "visual_tokens", "sent_images", "unique_sent_timestamps",
    "planner_token_usage", "visual_token_usage",
}
TERMINAL = {"success", "timeout", "failed", "incomplete"}
INFRA_NEEDLES = (
    "missing required env var", "unauthorized", "authentication",
    "invalid api key", "connection refused", "connection error",
    "name or service not known", "index not found",
    "embedding response missing", "no such file or directory",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha_manifest(path: Path, files: Iterable[Path], relative_to: Path | None = None) -> None:
    unique = sorted({p.resolve() for p in files}, key=lambda p: str(p))
    lines = []
    for source in unique:
        if not source.is_file():
            raise FileNotFoundError(source)
        label = str(source.relative_to(relative_to.resolve())) if relative_to else str(source)
        lines.append(f"{sha256_file(source)}  {label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_sha_manifest(path: Path, relative_to: Path | None = None) -> list[str]:
    results = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, label = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid SHA manifest line {number}: {path}") from exc
        target = (relative_to / label).resolve() if relative_to else Path(label)
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {target}: expected={expected} actual={actual}")
        results.append(f"{label}: OK")
    return results


def one_by_uid(paths: Iterable[Path], kind: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        if kind == "trajectory":
            uid = str(load_json(path).get("uid") or "")
        else:
            uid = str(load_json(path).get("uid") or path.stem)
        if not uid:
            raise RuntimeError(f"{kind} missing uid: {path}")
        if uid in result:
            raise RuntimeError(f"duplicate {kind} for uid={uid}: {result[uid]} and {path}")
        result[uid] = path
    return result


def last_model_response(trajectory: dict[str, Any]) -> str:
    for step in reversed(trajectory.get("steps") or []):
        if isinstance(step, dict) and isinstance(step.get("model_response"), str):
            return step["model_response"]
    return ""


def final_tags(text: str) -> list[str]:
    return [x.strip() for x in re.findall(r"<final>\s*(.*?)\s*</final>", text, flags=re.I | re.S)]


def infrastructure_findings(trajectory: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tool_errors = []
    infra = []
    for index, step in enumerate(trajectory.get("steps") or []):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict) or observation.get("ok") is not False:
            continue
        error = str(observation.get("error") or "tool returned ok=false")
        row = {"step_index": index, "error": error}
        tool_errors.append(row)
        if any(needle in error.lower() for needle in INFRA_NEEDLES):
            infra.append(row)
    return tool_errors, infra


def number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


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


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    planner_prompt = sum(number(m.get("planner_token_usage", {}).get("prompt_tokens")) for m in metrics)
    planner_completion = sum(number(m.get("planner_token_usage", {}).get("completion_tokens")) for m in metrics)
    visual_prompt = sum(number(m.get("visual_token_usage", {}).get("prompt_tokens")) for m in metrics)
    visual_completion = sum(number(m.get("visual_token_usage", {}).get("completion_tokens")) for m in metrics)
    elapsed = [number(m.get("elapsed_sec")) for m in metrics]
    return {
        "attempts": len(metrics),
        "status_counts": dict(sorted(Counter(str(m.get("status") or "missing") for m in metrics).items())),
        "elapsed_sec_sum": sum(elapsed),
        "elapsed_hours_sum": sum(elapsed) / 3600,
        "planner_calls": int(sum(number(m.get("planner_calls")) for m in metrics)),
        "visual_retrieve_calls": int(sum(number(m.get("visual_retrieve_calls")) for m in metrics)),
        "visual_inspect_calls": int(sum(number(m.get("visual_inspect_calls")) for m in metrics)),
        "planner_tokens": {
            "prompt": int(planner_prompt),
            "completion": int(planner_completion),
            "total": int(sum(number(m.get("planner_tokens")) for m in metrics)),
        },
        "visual_tokens": {
            "prompt": int(visual_prompt),
            "completion": int(visual_completion),
            "total": int(sum(number(m.get("visual_tokens")) for m in metrics)),
        },
        "sent_images": int(sum(number(m.get("sent_images")) for m in metrics)),
        "unique_sent_timestamps": int(sum(number(m.get("unique_sent_timestamps")) for m in metrics)),
    }


def parse_controller_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def gold_free() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "state.txt").exists():
        raise RuntimeError("finalization state already exists; refusing to overwrite")

    # Freeze/hash the completed retry-v2 source closure before any merge.
    raw_files = [p for p in RETRY_RAW.rglob("*") if p.is_file()]
    control_files = [p for p in RETRY_CONTROL.rglob("*") if p.is_file()]
    snapshot_files = raw_files + control_files
    snapshot_manifest = OUT / "retry_v2_source_snapshot.sha256"
    write_sha_manifest(snapshot_manifest, snapshot_files)
    snapshot_check = verify_sha_manifest(snapshot_manifest)
    (OUT / "retry_v2_source_snapshot.verify.txt").write_text("\n".join(snapshot_check) + "\n", encoding="utf-8")

    retry_uids_path = RETRY_PROTOCOL / "retry_uids.txt"
    frozen_uids_path = BASE / "frozen_ordered_eval300_uids.txt"
    retry_uids = [x.strip() for x in retry_uids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    frozen_uids = [x.strip() for x in frozen_uids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(retry_uids) != 57 or len(set(retry_uids)) != 57:
        raise RuntimeError("retry manifest is not exactly 57 unique UIDs")
    if len(frozen_uids) != 300 or len(set(frozen_uids)) != 300:
        raise RuntimeError("frozen evaluation manifest is not exactly 300 unique UIDs")

    retry_manifest = load_json(RETRY_PROTOCOL / "retry_manifest.json")
    if sha256_file(retry_uids_path) != retry_manifest.get("uid_sha256"):
        raise RuntimeError("retry UID file hash does not match retry_manifest.json")
    experiment_config = load_json(BASE / "experiment_config.json")
    if sha256_file(frozen_uids_path) != experiment_config["uid_file"]["sha256"]:
        raise RuntimeError("frozen 300-UID file hash does not match experiment_config.json")
    if (RETRY_CONTROL / "runner_exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("retry runner did not exit zero")
    if (RETRY_CONTROL / "state.txt").read_text(encoding="utf-8").strip() != "RETRY_RAW_COMPLETE_PENDING_FREEZE":
        raise RuntimeError("retry controller is not at raw-complete pending-freeze state")
    if (RETRY_RAW / "INFRASTRUCTURE_STOP.json").exists():
        raise RuntimeError("retry raw closure contains INFRASTRUCTURE_STOP.json")

    strict_rows = load_jsonl(FIRST_AUDIT / "strict_first_pass_per_uid.jsonl")
    strict_map = {str(row["uid"]): row for row in strict_rows}
    if len(strict_rows) != 300 or len(strict_map) != 300 or set(strict_map) != set(frozen_uids):
        raise RuntimeError("first-pass strict reconstruction does not cover frozen 300-UID set")
    first_complete = {uid for uid, row in strict_map.items() if row.get("strict_first_pass_complete") is True}
    first_failed = set(frozen_uids) - first_complete
    if len(first_complete) != 243 or len(first_failed) != 57:
        raise RuntimeError("first-pass strict 243/57 split mismatch")
    if set(retry_uids) != first_failed:
        raise RuntimeError("57-UID retry manifest differs from strict first-pass failure set")
    if set(retry_uids) & first_complete:
        raise RuntimeError("retry manifest overlaps strict first-pass successes")

    metric_paths = one_by_uid(RETRY_RAW.glob("*/metrics/*.json"), "metric")
    trajectory_paths = one_by_uid(RETRY_RAW.glob("*/*/trajectory.json"), "trajectory")
    prediction_paths = one_by_uid(RETRY_RAW.glob("*/preds/*.json"), "prediction")
    if set(metric_paths) != set(retry_uids):
        raise RuntimeError("retry metrics do not exactly cover 57-UID manifest")
    if set(trajectory_paths) != set(retry_uids):
        raise RuntimeError("retry trajectories do not exactly cover 57-UID manifest")
    if len(prediction_paths) != 24 or not set(prediction_paths).issubset(set(retry_uids)):
        raise RuntimeError("retry prediction file set is not the expected 24-UID subset")

    controller_rows = parse_controller_records(RETRY_CONTROL / "controller.log")
    controller_map = {str(row.get("uid") or ""): row for row in controller_rows}
    if len(controller_rows) != 57 or len(controller_map) != 57 or set(controller_map) != set(retry_uids):
        raise RuntimeError("controller terminal records do not exactly cover retry manifest")

    sys.path.insert(0, str(RUNTIME))
    from videoseal.utils.lvbench_io import parse_choice_letter, parse_choice_letter_smart

    validation_rows = []
    all_tool_errors = []
    all_infra = []
    success_raw_final_agreement = 0
    groundtruth_populated = 0
    for uid in retry_uids:
        metric_path = metric_paths[uid]
        trajectory_path = trajectory_paths[uid]
        metric = load_json(metric_path)
        trajectory = load_json(trajectory_path)
        prediction_path = prediction_paths.get(uid)
        prediction = load_json(prediction_path) if prediction_path else {}
        video_id = uid.rsplit("_", 2)[0]
        answer = str(trajectory.get("answer") or "").strip()
        question = str(trajectory.get("question") or "")
        parsed = ""
        if answer:
            parsed = str(parse_choice_letter_smart(answer, question) or parse_choice_letter(answer) or "").strip().upper()
        parsed = parsed if parsed in LEGAL else ""
        pred_value = str(prediction.get("pred") or "").strip().upper()
        response = last_model_response(trajectory)
        tags = final_tags(response)
        tool_errors, infra = infrastructure_findings(trajectory)
        for row in tool_errors:
            all_tool_errors.append({"uid": uid, **row})
        for row in infra:
            all_infra.append({"uid": uid, **row})
        populated_gt = trajectory.get("groundtruth") not in (None, "") or prediction.get("gt") not in (None, "")
        groundtruth_populated += int(populated_gt)
        schema_complete = METRIC_KEYS.issubset(metric)
        strict_success = (
            metric.get("status") == "success"
            and schema_complete
            and bool(trajectory.get("finished_at"))
            and bool(answer)
            and parsed in LEGAL
        )
        if str(controller_map[uid].get("status")) != str(metric.get("status")):
            raise RuntimeError(f"controller/metric status mismatch for uid={uid}")
        if str(metric.get("uid")) != uid or str(trajectory.get("uid")) != uid:
            raise RuntimeError(f"UID mismatch inside retry artifact for uid={uid}")
        if str(trajectory.get("video_id")) != video_id:
            raise RuntimeError(f"video_id mismatch for uid={uid}")
        if prediction_path and str(prediction.get("uid")) != uid:
            raise RuntimeError(f"prediction UID mismatch for uid={uid}")
        if any(key in prediction for key in ("gt", "groundtruth")):
            raise RuntimeError(f"retry prediction unexpectedly contains gold key for uid={uid}")
        if populated_gt:
            raise RuntimeError(f"retry artifact contains populated groundtruth for uid={uid}")
        status = str(metric.get("status"))
        if status not in TERMINAL:
            raise RuntimeError(f"non-terminal retry metric for uid={uid}: {status}")
        if status == "success":
            if not strict_success or prediction_path is None or pred_value != parsed:
                raise RuntimeError(f"metric success fails strict completion for uid={uid}")
            if tags and tags[-1].strip().upper() == answer.upper():
                success_raw_final_agreement += 1
        elif status == "incomplete":
            if prediction_path is None or pred_value:
                raise RuntimeError(f"incomplete prediction-file contract mismatch for uid={uid}")
        elif status == "timeout" and prediction_path is not None:
            raise RuntimeError(f"timeout unexpectedly has a prediction file for uid={uid}")

        if strict_success:
            failure_class = None
        elif status == "timeout":
            failure_class = "timeout"
        elif status == "incomplete" and not answer:
            failure_class = "incomplete_empty_answer"
        elif status == "incomplete":
            failure_class = "incomplete_no_unique_legal_choice"
        else:
            failure_class = status
        validation_rows.append({
            "uid": uid,
            "video_id": video_id,
            "metric_path": str(metric_path),
            "trajectory_path": str(trajectory_path),
            "prediction_path": str(prediction_path) if prediction_path else None,
            "metric_status": status,
            "metric_schema_complete": schema_complete,
            "trajectory_finished": bool(trajectory.get("finished_at")),
            "trajectory_answer": answer,
            "frozen_parser_prediction": parsed or None,
            "prediction_file_value": pred_value or None,
            "strict_retry_success": strict_success,
            "failure_class": failure_class,
            "terminal_model_response_present": bool(response),
            "terminal_model_response_bytes": len(response.encode("utf-8")),
            "terminal_model_response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "terminal_final_tags": tags,
            "tool_error_count": len(tool_errors),
            "infrastructure_error_count": len(infra),
            "groundtruth_populated": populated_gt,
        })

    status_counts = Counter(row["metric_status"] for row in validation_rows)
    strict_retry = {row["uid"] for row in validation_rows if row["strict_retry_success"]}
    if status_counts != Counter({"timeout": 33, "success": 17, "incomplete": 7}):
        raise RuntimeError(f"unexpected retry terminal status counts: {status_counts}")
    if len(strict_retry) != 17:
        raise RuntimeError("strict retry success count is not 17")
    if all_infra:
        raise RuntimeError(f"infrastructure errors found in retry-v2: {all_infra}")
    if groundtruth_populated:
        raise RuntimeError("retry-v2 has populated groundtruth values")

    write_jsonl(OUT / "retry_validation_per_uid.jsonl", validation_rows)
    retry_validation = {
        "generated_utc": utc_now(),
        "status": "PASS",
        "gold_accessed": False,
        "frozen_uid_count": 300,
        "first_pass_strict_complete": 243,
        "retry_manifest_count": 57,
        "retry_manifest_unique": 57,
        "retry_uid_set_equals_first_pass_strict_failure_set": True,
        "retry_metric_count": len(metric_paths),
        "retry_trajectory_count": len(trajectory_paths),
        "retry_prediction_file_count": len(prediction_paths),
        "retry_status_counts": dict(sorted(status_counts.items())),
        "strict_retry_success": len(strict_retry),
        "all_metric_successes_pass_strict_rule": True,
        "success_terminal_final_tag_matches_trajectory_answer": success_raw_final_agreement,
        "prediction_file_explanation": "run_one writes a prediction file whenever the agent returns normally; 17 contain legal A-E and are status=success, while 7 contain an empty pred and are status=incomplete; 33 killed timeout workers never reach prediction writing",
        "prediction_files_equal_success_plus_incomplete": len(prediction_paths) == status_counts["success"] + status_counts["incomplete"],
        "tool_observation_error_count": len(all_tool_errors),
        "tool_observation_errors": all_tool_errors,
        "infrastructure_error_count": len(all_infra),
        "infrastructure_stop_sidecar_present": False,
        "populated_groundtruth_count": groundtruth_populated,
        "controller_record_count": len(controller_rows),
        "runner_exit_code": 0,
        "source_state": "RETRY_RAW_COMPLETE_PENDING_FREEZE",
        "retry_source_snapshot_entries": len(snapshot_check),
        "retry_source_snapshot_manifest_sha256": sha256_file(snapshot_manifest),
    }
    write_json(OUT / "retry_validation.json", retry_validation)

    validation_map = {row["uid"]: row for row in validation_rows}
    final_rows = []
    selected_metric_paths = []
    for uid in frozen_uids:
        first = strict_map[uid]
        if uid in first_complete:
            prediction = str(first["frozen_strict_prediction"]).upper()
            if prediction not in LEGAL:
                raise RuntimeError(f"invalid strict first-pass prediction for uid={uid}")
            final = {
                "uid": uid,
                "video_id": str(first["video_id"]),
                "selected_source": "first_pass_strict",
                "selected_metric_status": str(first["metric_status"]),
                "strict_complete": True,
                "prediction": prediction,
                "failure_class": None,
                "metric_path": str(first["metric_path"]),
                "trajectory_path": str(first["trajectory_path"]),
            }
            selected_metric_paths.append(Path(first["metric_path"]))
        else:
            retry = validation_map[uid]
            final = {
                "uid": uid,
                "video_id": retry["video_id"],
                "selected_source": "retry_v2",
                "selected_metric_status": retry["metric_status"],
                "strict_complete": retry["strict_retry_success"],
                "prediction": retry["frozen_parser_prediction"] if retry["strict_retry_success"] else None,
                "failure_class": retry["failure_class"],
                "metric_path": retry["metric_path"],
                "trajectory_path": retry["trajectory_path"],
            }
            selected_metric_paths.append(Path(retry["metric_path"]))
        if "retry_raw_v1" in final["metric_path"] or "retry_raw_v1" in final["trajectory_path"]:
            raise RuntimeError(f"retry-v1 contamination for uid={uid}")
        final_rows.append(final)

    if len(final_rows) != 300 or len({row["uid"] for row in final_rows}) != 300:
        raise RuntimeError("final merged closure does not contain 300 unique UIDs")
    if [row["uid"] for row in final_rows] != frozen_uids:
        raise RuntimeError("final merged closure order differs from frozen manifest")
    strict_completed = sum(row["strict_complete"] for row in final_rows)
    if strict_completed != 260:
        raise RuntimeError(f"expected 260 final strict completions, got {strict_completed}")
    if Counter(row["selected_source"] for row in final_rows) != Counter({"first_pass_strict": 243, "retry_v2": 57}):
        raise RuntimeError("final selected-source counts mismatch")

    final_path = OUT / "final_predictions_gold_free.jsonl"
    write_jsonl(final_path, final_rows)
    final_sha = sha256_file(final_path)
    (OUT / "final_predictions_gold_free.sha256").write_text(f"{final_sha}  final_predictions_gold_free.jsonl\n", encoding="utf-8")
    final_check = verify_sha_manifest(OUT / "final_predictions_gold_free.sha256", relative_to=OUT)
    (OUT / "final_predictions_gold_free.verify.txt").write_text("\n".join(final_check) + "\n", encoding="utf-8")
    os.chmod(final_path, 0o444)

    merge_summary = {
        "generated_utc": utc_now(),
        "status": "GOLD_FREE_FINAL_PREDICTION_CLOSURE_FROZEN",
        "gold_accessed": False,
        "frozen_uid_count": 300,
        "final_record_count": 300,
        "final_unique_uid_count": 300,
        "final_uid_order_equals_frozen_manifest": True,
        "selected_from_first_pass_strict": 243,
        "selected_from_retry_v2": 57,
        "retry_v2_strict_success": 17,
        "final_strict_completed": 260,
        "final_failure_count": 40,
        "final_failure_class_counts": dict(sorted(Counter(row["failure_class"] for row in final_rows if row["failure_class"]).items())),
        "retry_v1_selected_records": 0,
        "backfill_run": False,
        "postprocessing_predictions_used": False,
        "prediction_sha256": final_sha,
    }
    write_json(OUT / "gold_free_merge_summary.json", merge_summary)

    first_metric_paths = sorted(BASE.glob("*/metrics/*.json"))
    if len(first_metric_paths) != 300:
        raise RuntimeError(f"expected 300 first-pass metrics, got {len(first_metric_paths)}")
    first_metrics = [load_json(path) for path in first_metric_paths]
    retry_metrics = [load_json(metric_paths[uid]) for uid in retry_uids]
    selected_metrics = [load_json(path) for path in selected_metric_paths]
    completed_metric_paths = [Path(row["metric_path"]) for row in final_rows if row["strict_complete"]]
    completed_metrics = [load_json(path) for path in completed_metric_paths]
    completed_elapsed = [number(metric.get("elapsed_sec")) for metric in completed_metrics]

    embed_rows = load_jsonl(RETRY_CONTROL / "embedding_usage.jsonl")
    embed_success = [row for row in embed_rows if row.get("status") == "success" and row.get("provider_request_reached") is True]
    retry_agg = aggregate_metrics(retry_metrics)
    if len(embed_rows) != len(embed_success):
        raise RuntimeError("retry embedding ledger contains unsuccessful/non-provider entries")
    # Each retry UID ran in its own child and the controller dispatched UIDs
    # sequentially.  Provider-ledger PIDs ordered by their first request can
    # therefore be aligned exactly with the frozen retry order.  A hard timeout
    # can kill a child after its embedding request returns but before the full
    # tool step is persisted.  Keep these two accounting scopes separate.
    ledger_by_pid: dict[int, list[dict[str, Any]]] = {}
    for row in embed_success:
        ledger_by_pid.setdefault(int(row["pid"]), []).append(row)
    ordered_pids = sorted(
        ledger_by_pid,
        key=lambda pid: min(str(row["started_at_utc"]) for row in ledger_by_pid[pid]),
    )
    if len(ordered_pids) != len(retry_uids):
        raise RuntimeError("embedding ledger PID population cannot be aligned to 57 sequential retry UIDs")
    embedding_reconciliation = []
    for uid, pid in zip(retry_uids, ordered_pids):
        metric = load_json(metric_paths[uid])
        persisted = int(metric.get("visual_retrieve_calls") or 0)
        provider = len(ledger_by_pid[pid])
        delta = provider - persisted
        if delta < 0:
            raise RuntimeError(f"provider ledger is missing persisted retrieval requests for uid={uid}")
        if delta and (metric.get("status") != "timeout" or delta != 1):
            raise RuntimeError(f"unexpected embedding/persisted-action delta for uid={uid}: {delta}")
        embedding_reconciliation.append({
            "uid": uid,
            "status": metric.get("status"),
            "child_pid": pid,
            "persisted_visual_retrieve_actions": persisted,
            "successful_provider_embedding_requests": provider,
            "unpersisted_provider_requests_at_timeout": delta,
            "last_provider_request_finished_utc": max(str(row["finished_at_utc"]) for row in ledger_by_pid[pid]),
        })
    unpersisted = sum(row["unpersisted_provider_requests_at_timeout"] for row in embedding_reconciliation)
    mismatches = [row for row in embedding_reconciliation if row["unpersisted_provider_requests_at_timeout"]]
    if len(embed_success) != retry_agg["visual_retrieve_calls"] + unpersisted:
        raise RuntimeError("embedding reconciliation does not close")
    if unpersisted != 6 or len(mismatches) != 6:
        raise RuntimeError(f"expected six single-request timeout-boundary deltas, got {unpersisted} across {len(mismatches)} UIDs")
    write_json(OUT / "retry_embedding_reconciliation.json", {
        "status": "PASS",
        "mapping_rule": "57 unique child PIDs ordered by first provider-request time aligned to the sequential frozen 57-UID manifest",
        "persisted_visual_retrieve_actions": retry_agg["visual_retrieve_calls"],
        "successful_provider_embedding_requests": len(embed_success),
        "unpersisted_provider_requests_at_hard_timeout": unpersisted,
        "mismatch_uid_count": len(mismatches),
        "mismatch_rows": mismatches,
        "interpretation": "provider requests are actual billable calls; persisted trajectory actions are the reproducible metric scope. Six timeout workers were killed after an embedding response but before committing the corresponding tool step.",
    })
    smoke_cost = load_json(RECOVERY_SMOKE / "cost_ledger.json")
    anomalous_cost = load_json(RECOVERY_AUDIT / "anomalous_cost_ledger.json")
    first_agg = aggregate_metrics(first_metrics)
    final_selected_agg = aggregate_metrics(selected_metrics)
    actual_formal_agg = aggregate_metrics(first_metrics + retry_metrics)
    first_attempt = experiment_config["runner_timing"]["attempts"][0]
    first_wall_start = datetime.fromisoformat(first_attempt["started_at_utc"])
    first_wall_end = datetime.fromisoformat(first_attempt["finished_at_utc"])
    retry_created = [
        datetime.fromisoformat(str(load_json(path)["created_at"]))
        for path in trajectory_paths.values()
    ]
    retry_wall_start = min(retry_created)
    retry_wall_end = datetime.fromtimestamp((RETRY_CONTROL / "state.txt").stat().st_mtime, tz=timezone.utc)
    first_wall_sec = (first_wall_end - first_wall_start).total_seconds()
    retry_wall_sec = (retry_wall_end - retry_wall_start).total_seconds()
    cost_summary = {
        "generated_utc": utc_now(),
        "gold_accessed": False,
        "definitions": {
            "actual_formal": "all 300 first-pass attempts plus all 57 formal retry-v2 attempts",
            "final_selected": "one selected terminal attempt per UID: 243 retained first-pass strict successes plus all 57 retry-v2 terminal attempts",
            "completed_latency": "only the 260 final-selected strict-completed attempts",
            "elapsed": "sum of recorded metric.elapsed_sec; sequential wall-clock duration is reported separately where recoverable",
            "timeout_metric_limit": "metrics are reconstructed from the last persisted trajectory; a killed in-flight step and the remainder to the 1000-second wrapper timeout are not represented, so all-attempt metric sums are lower bounds on consumed wall time",
        },
        "first_pass_actual": first_agg,
        "formal_retry_v2_actual": retry_agg,
        "actual_formal_total": actual_formal_agg,
        "final_selected_300": final_selected_agg,
        "final_selected_strict_completed_latency": {
            "count": len(completed_elapsed),
            "sum_sec": sum(completed_elapsed),
            "sum_hours": sum(completed_elapsed) / 3600,
            "mean_sec": statistics.mean(completed_elapsed),
            "median_sec": statistics.median(completed_elapsed),
            "p90_sec": percentile(completed_elapsed, 0.90),
            "p95_sec": percentile(completed_elapsed, 0.95),
        },
        "sequential_batch_wall_clock": {
            "first_pass_started_utc": first_wall_start.isoformat(),
            "first_pass_finished_utc": first_wall_end.isoformat(),
            "first_pass_wall_sec": first_wall_sec,
            "first_pass_wall_hours": first_wall_sec / 3600,
            "formal_retry_v2_started_utc_earliest_trajectory": retry_wall_start.isoformat(),
            "formal_retry_v2_finished_utc_state_mtime": retry_wall_end.isoformat(),
            "formal_retry_v2_wall_sec": retry_wall_sec,
            "formal_retry_v2_wall_hours": retry_wall_sec / 3600,
            "executed_batch_wall_hours_sum": (first_wall_sec + retry_wall_sec) / 3600,
            "scope_note": "sum of sequential first-pass and retry-v2 batch wall durations; excludes the calendar gap, smoke, retry-v1 diagnostic, and idle model-server lifetime",
        },
        "embedding_cost": {
            "first_pass": {
                "provider_request_count": None,
                "provider_reported_input_tokens": None,
                "input_cost_usd": None,
                "status": "unknown: no historical first-pass embedding usage ledger was found; not imputed as zero",
            },
            "formal_retry_v2": {
                "provider_request_count": len(embed_success),
                "unique_response_request_ids": len({row.get("response_request_id") for row in embed_success}),
                "provider_reported_input_tokens": int(sum(number(row.get("prompt_tokens")) for row in embed_success)),
                "input_cost_usd": sum(number(row.get("input_cost_usd")) for row in embed_success),
                "price_usd_per_million_input_tokens": 0.13,
                "ledger_status_counts": dict(sorted(Counter(str(row.get("status")) for row in embed_rows).items())),
                "persisted_visual_retrieve_actions": retry_agg["visual_retrieve_calls"],
                "unpersisted_provider_requests_at_hard_timeout": unpersisted,
                "reconciliation_status": "PASS",
            },
            "actual_formal_total_input_cost_usd": None,
            "actual_formal_known_retry_component_usd": sum(number(row.get("input_cost_usd")) for row in embed_success),
            "actual_formal_total_status": "unknown because the first-pass embedding component is unknown",
        },
        "smoke_separate": smoke_cost,
        "retry_v1_anomalous_separate_excluded_from_merge": anomalous_cost,
    }
    write_json(OUT / "cost_summary.json", cost_summary)

    source_inputs = [
        frozen_uids_path,
        BASE / "experiment_config.json",
        FIRST_AUDIT / "strict_first_pass_per_uid.jsonl",
        FIRST_AUDIT / "strict_reconstruction_summary.json",
        FIRST_AUDIT / "raw_artifacts_snapshot.sha256",
        retry_uids_path,
        RETRY_PROTOCOL / "retry_manifest.json",
        RETRY_PROTOCOL / "PROTOCOL_AMENDMENT.md",
        RECOVERY_AUDIT / "RECOVERY_PLAN.md",
        RECOVERY_AUDIT / "anomalous_cost_ledger.json",
        RECOVERY_SMOKE / "cost_ledger.json",
        COMPARISON_REPORT,
    ]
    write_sha_manifest(OUT / "SOURCE_MANIFEST.sha256", source_inputs)
    source_check = verify_sha_manifest(OUT / "SOURCE_MANIFEST.sha256")
    (OUT / "SOURCE_MANIFEST.verify.txt").write_text("\n".join(source_check) + "\n", encoding="utf-8")

    (OUT / "gold_free_stage.txt").write_text("COMPLETE / VERIFIED / NO GOLD ACCESSED\n", encoding="utf-8")
    print(json.dumps({
        "status": "GOLD_FREE_STAGE_COMPLETE",
        "retry_strict_success": len(strict_retry),
        "final_strict_completed": strict_completed,
        "prediction_sha256": final_sha,
    }, sort_keys=True))


def normalize_gold(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in LEGAL:
        return text
    match = re.fullmatch(r"(?:OPTION|ANSWER)?\s*[:\-]?\s*([A-E])(?:[\s.)]*)", text)
    return match.group(1) if match else ""


def score() -> None:
    stage = (OUT / "gold_free_stage.txt").read_text(encoding="utf-8").strip()
    if stage != "COMPLETE / VERIFIED / NO GOLD ACCESSED":
        raise RuntimeError("gold-free stage is not complete")
    final_manifest = OUT / "final_predictions_gold_free.sha256"
    final_check = verify_sha_manifest(final_manifest, relative_to=OUT)
    if final_check != ["final_predictions_gold_free.jsonl: OK"]:
        raise RuntimeError("final prediction closure verification failed")
    merge = load_json(OUT / "gold_free_merge_summary.json")
    retry_validation = load_json(OUT / "retry_validation.json")
    if merge.get("status") != "GOLD_FREE_FINAL_PREDICTION_CLOSURE_FROZEN":
        raise RuntimeError("final prediction closure is not frozen")
    final_rows = load_jsonl(OUT / "final_predictions_gold_free.jsonl")
    if len(final_rows) != 300 or len({row["uid"] for row in final_rows}) != 300:
        raise RuntimeError("frozen prediction closure is not 300 unique UIDs")

    # This is the first stage in this program that imports/opens the gold source.
    import pyarrow.parquet as pq
    table = pq.read_table(PARQUET, columns=["extra_info.qa_uid", "extra_info.ground_truth"])
    columns = {name: table[name].to_pylist() for name in table.column_names}
    gold_map: dict[str, str] = {}
    for uid_value, gold_value in zip(columns["qa_uid"], columns["ground_truth"]):
        uid = str(uid_value or "").strip()
        if uid not in {row["uid"] for row in final_rows}:
            continue
        gold = normalize_gold(gold_value)
        if not gold:
            raise RuntimeError(f"non-legal gold for frozen uid={uid}")
        if uid in gold_map:
            raise RuntimeError(f"duplicate gold uid={uid}")
        gold_map[uid] = gold
    if set(gold_map) != {row["uid"] for row in final_rows}:
        raise RuntimeError("gold UID set does not equal frozen prediction UID set")

    scored_rows = []
    for row in final_rows:
        prediction = row.get("prediction") if row.get("strict_complete") else None
        correct = bool(row.get("strict_complete") and prediction == gold_map[row["uid"]])
        scored_rows.append({**row, "gold": gold_map[row["uid"]], "correct": correct})
    write_jsonl(OUT / "scored_results.jsonl", scored_rows)

    completed = [row for row in scored_rows if row["strict_complete"]]
    correct = sum(row["correct"] for row in scored_rows)
    by_source = {}
    for source in ("first_pass_strict", "retry_v2"):
        rows = [row for row in scored_rows if row["selected_source"] == source]
        done = [row for row in rows if row["strict_complete"]]
        by_source[source] = {
            "selected_records": len(rows),
            "strict_completed": len(done),
            "correct": sum(row["correct"] for row in rows),
            "correct_over_selected": sum(row["correct"] for row in rows) / len(rows),
            "correct_over_strict_completed": sum(row["correct"] for row in done) / len(done) if done else None,
        }
    failure_counts = dict(sorted(Counter(row["failure_class"] for row in scored_rows if not row["strict_complete"]).items()))
    score_summary = {
        "generated_utc": utc_now(),
        "status": "COMPLETE / VALID",
        "gold_loaded_only_after_frozen_prediction_sha256_verified": True,
        "prediction_sha256_verified_before_gold_load": merge["prediction_sha256"],
        "gold_source": str(PARQUET),
        "gold_source_sha256": sha256_file(PARQUET),
        "denominator": 300,
        "strict_completed": len(completed),
        "strict_completion_rate": len(completed) / 300,
        "correct": correct,
        "correct_over_300": correct / 300,
        "correct_over_strict_completed": correct / len(completed),
        "final_failure_count": 300 - len(completed),
        "final_failure_class_counts": failure_counts,
        "by_selected_source": by_source,
    }
    write_json(OUT / "score_summary.json", score_summary)

    historical = {
        "scope": "latest archived final-selected strict comparison table; fixed denominator 300; incomplete is not correct",
        "source": str(COMPARISON_REPORT),
        "source_lines": "311-315 for result metrics; 328 onward for completed-only E2E; 338 onward for final-selected calls",
        "methods": {
            "Flat-30": {"strict_completed": 254, "correct": 82, "timeout": 36, "success_strict_invalid": 10, "completed_accuracy": 82 / 254, "completed_e2e_mean_sec": 318.99, "completed_e2e_median_sec": 252.53, "planner_calls": 1300, "retrieval_calls": 566, "inspector_calls": 354, "sent_images": 21492, "final_selected_elapsed_hours": 32.62},
            "H-8": {"strict_completed": 270, "correct": 78, "timeout": 29, "success_strict_invalid": 1, "completed_accuracy": 78 / 270, "completed_e2e_mean_sec": 247.03, "completed_e2e_median_sec": 205.99, "planner_calls": 1251, "retrieval_calls": 442, "inspector_calls": 220, "sent_images": 13348, "final_selected_elapsed_hours": 26.19},
            "H-15": {"strict_completed": 270, "correct": 81, "timeout": 26, "success_strict_invalid": 4, "completed_accuracy": 81 / 270, "completed_e2e_mean_sec": 250.27, "completed_e2e_median_sec": 219.25, "planner_calls": 1154, "retrieval_calls": 486, "inspector_calls": 302, "sent_images": 18336, "final_selected_elapsed_hours": 26.10},
            "H-30": {"strict_completed": 249, "correct": 78, "timeout": 45, "success_strict_invalid": 6, "completed_accuracy": 78 / 249, "completed_e2e_mean_sec": 305.70, "completed_e2e_median_sec": 260.24, "planner_calls": 1307, "retrieval_calls": 413, "inspector_calls": 245, "sent_images": 14892, "final_selected_elapsed_hours": 33.45},
        },
    }
    cost = load_json(OUT / "cost_summary.json")
    current_selected = cost["final_selected_300"]
    current_latency = cost["final_selected_strict_completed_latency"]
    historical["methods"]["Flat-15 (this run)"] = {
        "strict_completed": len(completed),
        "correct": correct,
        "timeout": failure_counts.get("timeout", 0),
        "incomplete": sum(v for k, v in failure_counts.items() if str(k).startswith("incomplete")),
        "completed_accuracy": correct / len(completed),
        "completed_e2e_mean_sec": current_latency["mean_sec"],
        "completed_e2e_median_sec": current_latency["median_sec"],
        "planner_calls": current_selected["planner_calls"],
        "retrieval_calls": current_selected["visual_retrieve_calls"],
        "inspector_calls": current_selected["visual_inspect_calls"],
        "sent_images": current_selected["sent_images"],
        "final_selected_elapsed_hours": current_selected["elapsed_hours_sum"],
    }
    write_json(OUT / "comparison_summary.json", historical)

    fp = cost["first_pass_actual"]
    rt = cost["formal_retry_v2_actual"]
    total = cost["actual_formal_total"]
    sel = cost["final_selected_300"]
    emb = cost["embedding_cost"]
    smoke = cost["smoke_separate"]
    abnormal = cost["retry_v1_anomalous_separate_excluded_from_merge"]["deduplicated_totals"]
    methods = historical["methods"]
    wall = cost["sequential_batch_wall_clock"]
    report = f"""# Flat-15 Eval300 canonical final report

## Final status

**COMPLETE / VALID**

This additive finalization did not rerun a model, retry, backfill, index build, or prediction parser, and it made no writes to the historical formal-run roots. The data-bearing metrics, trajectories, and predictions used by the merge still match their pre-retry hashes. Two shared service logs are an explicit exception to the broader pre-retry snapshot: the approved retry-v2 reused the same Planner and Visual services and had already appended `logs/planner.log` and `logs/visual.log` before finalization. `FIRST_PASS_PRE_RETRY_SNAPSHOT_RECHECK.json` records the 924-pass/2-expected-difference result.

An earlier additive finalization attempt (`videoseal_flat15_eval300_finalization_v1_20260914T002232Z`) stopped before gold access when it incorrectly required provider embedding requests to equal persisted retrieval actions. It is diagnostic only and is excluded from this closure.

## 1. Retry-v2 acceptance and freeze

- Frozen retry manifest: exactly 57 unique UIDs; its set equals the 57 strict first-pass failures and has zero overlap with the retained 243 strict first-pass successes.
- Terminal retry-v2 artifacts: 57 metrics and 57 trajectories, covering the manifest exactly; runner exit code 0; source state `RETRY_RAW_COMPLETE_PENDING_FREEZE`.
- Terminal statuses: 17 success, 33 timeout, 7 incomplete.
- All 17 metric successes satisfy the strict rule: complete metric schema, finished trajectory, non-empty answer, and the frozen parser yields exactly one legal A-E prediction. No timeout or incomplete was promoted.
- There are 24 prediction files because `run_one` writes one whenever the agent returns normally: 17 contain legal A-E predictions and are success; 7 contain an empty prediction and are incomplete. The 33 hard-timeout workers terminate before that write.
- Infrastructure-stop sidecar: absent. Populated retry gold fields: 0. Infrastructure-error matches in failed tool observations: 0.
- Retry-v2 raw/control snapshot entries: {retry_validation['retry_source_snapshot_entries']}; snapshot manifest SHA-256: `{retry_validation['retry_source_snapshot_manifest_sha256']}`. Immediate readback and final readback both pass.
- The older 926-entry first-pass pre-retry snapshot now has 924 matches and two expected shared-log differences (`logs/planner.log`, `logs/visual.log`) caused by the later approved retry service activity. No metric, trajectory, prediction, configuration, or frozen UID entry differs.

## 2. Gold-free merge and prediction freeze

- Frozen population: 300 unique UIDs in frozen order.
- Selected records: 243 strict first-pass successes plus all 57 retry-v2 terminal outcomes.
- Retry-v2 recoveries: 17 strict successes; final strict completion: **260/300 ({len(completed)/3:.2f}%)**.
- Preserved failures: 33 timeout and 7 incomplete; retry-v1 selected records: 0; backfill: not run; post-processing/backfilled predictions: not used.
- Frozen gold-free prediction SHA-256: `{merge['prediction_sha256']}`.
- Gold was loaded only in the separate scoring invocation after this hash was independently read back and verified.

## 3. Final score

| Metric | Result |
|---|---:|
| Correct / 300 | **{correct}/300 ({100*correct/300:.2f}%)** |
| Strict completed / 300 | **{len(completed)}/300 ({100*len(completed)/300:.2f}%)** |
| Correct / strict completed | **{correct}/{len(completed)} ({100*correct/len(completed):.2f}%)** |
| Final timeout | {failure_counts.get('timeout', 0)} |
| Final incomplete | {sum(v for k, v in failure_counts.items() if str(k).startswith('incomplete'))} |

`Correct / 300` is the primary fixed-denominator result; every retained failure is not correct. Completed-only accuracy is conditional on the 260 completed-item subset and is not interchangeable with fixed-300 accuracy.

## 4. Formal calls, tokens, images, and elapsed time

All table values below are sums of per-attempt metric fields. “Actual formal” includes superseded first-pass failures plus their retry attempts; “final-selected” contains one selected attempt per UID. For killed timeouts, metrics are reconstructed from the last persisted trajectory, so an in-flight call and the remainder to the 1000-second wrapper timeout are absent: all-attempt calls/tokens/images/metric-elapsed values are reproducible **recorded lower bounds**, not exact consumed totals.

| Scope | Attempts | Planner calls | Retrieval calls | Inspector calls | Planner tokens | Visual tokens | Images | Elapsed hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| First pass, actual | {fp['attempts']} | {fp['planner_calls']} | {fp['visual_retrieve_calls']} | {fp['visual_inspect_calls']} | {fp['planner_tokens']['total']} | {fp['visual_tokens']['total']} | {fp['sent_images']} | {fp['elapsed_hours_sum']:.2f} |
| Formal retry-v2, actual | {rt['attempts']} | {rt['planner_calls']} | {rt['visual_retrieve_calls']} | {rt['visual_inspect_calls']} | {rt['planner_tokens']['total']} | {rt['visual_tokens']['total']} | {rt['sent_images']} | {rt['elapsed_hours_sum']:.2f} |
| Actual formal total | {total['attempts']} | {total['planner_calls']} | {total['visual_retrieve_calls']} | {total['visual_inspect_calls']} | {total['planner_tokens']['total']} | {total['visual_tokens']['total']} | {total['sent_images']} | {total['elapsed_hours_sum']:.2f} |
| Final-selected 300 | {sel['attempts']} | {sel['planner_calls']} | {sel['visual_retrieve_calls']} | {sel['visual_inspect_calls']} | {sel['planner_tokens']['total']} | {sel['visual_tokens']['total']} | {sel['sent_images']} | {sel['elapsed_hours_sum']:.2f} |

Final-selected strict-completed latency (n=260): mean {current_latency['mean_sec']:.2f}s, median {current_latency['median_sec']:.2f}s, P90 {current_latency['p90_sec']:.2f}s, P95 {current_latency['p95_sec']:.2f}s; sum {current_latency['sum_hours']:.2f}h. Timeout/incomplete attempts are excluded from this distribution but retained in final-selected and actual-formal totals.

Sequential batch wall time, which includes the hard-timeout remainder omitted by terminal trajectory metrics, was {wall['first_pass_wall_hours']:.2f}h for first pass and {wall['formal_retry_v2_wall_hours']:.2f}h for formal retry-v2 ({wall['executed_batch_wall_hours_sum']:.2f}h combined, excluding the calendar gap, smoke, retry-v1 diagnostics, and idle server lifetime).

Embedding cost:

- First pass: **unknown**, because no historical provider-usage ledger exists; it is not reported as zero.
- Formal retry-v2: {emb['formal_retry_v2']['provider_request_count']} successful provider requests, {emb['formal_retry_v2']['provider_reported_input_tokens']} provider-reported input tokens, **USD {emb['formal_retry_v2']['input_cost_usd']:.8f}**. This is six more than the 178 persisted retrieval actions: six timeout UIDs each completed one embedding request before the worker was killed, but never committed the enclosing tool step. `retry_embedding_reconciliation.json` closes this boundary per UID.
- Actual formal total embedding cost: **unknown** because the first-pass component is unknown; the known retry-v2 component is USD {emb['actual_formal_known_retry_component_usd']:.8f}.

Separate non-formal-method overhead:

- Recovery smoke: 1 embedding request / 6 input tokens / USD {smoke['embedding']['input_cost_usd']:.8f}; 1 local Summarizer call / {smoke['local_summarizer']['total_tokens']} tokens; 0 Planner and 0 Inspector calls. Excluded from formal totals.
- Invalid retry-v1 diagnostic: {abnormal['planner_calls']} Planner calls / {abnormal['planner_total_tokens']} tokens; {abnormal['inspector_calls']} Inspector calls / {abnormal['inspector_total_tokens']} tokens / {abnormal['inspector_images']} images; {abnormal['retrieval_attempts']} failed retrieval attempts, 0 provider requests, recorded checkpoint elapsed {abnormal['elapsed_sec_sum_of_recorded_attempt_checkpoints']:.2f}s. Preserved as anomalous investment and excluded from merge/formal-method efficiency.

## 5. Same-definition comparison

The result rows use each method's latest archived final-selected 300-UID view: strict completion, fixed-denominator Correct/300, and completed-only accuracy. The efficiency columns use one selected attempt per UID, while completed E2E mean uses only strict-completed items. Historical Flat-30/H values come from `{COMPARISON_REPORT}` (result table lines 311-315 and efficiency tables beginning at line 328).

| Method | Strict completed | Correct / 300 | Correct / completed | Timeout | Other strict failure | Completed E2E mean | Planner | Retrieval | Inspector | Images | Selected elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat-15 (this run) | {len(completed)} ({100*len(completed)/300:.2f}%) | {correct} ({100*correct/300:.2f}%) | {100*correct/len(completed):.2f}% | {failure_counts.get('timeout', 0)} | 7 incomplete | {current_latency['mean_sec']:.2f}s | {sel['planner_calls']} | {sel['visual_retrieve_calls']} | {sel['visual_inspect_calls']} | {sel['sent_images']} | {sel['elapsed_hours_sum']:.2f}h |
| Flat-30 | 254 (84.67%) | 82 (27.33%) | 32.28% | 36 | 10 strict-invalid | 318.99s | 1300 | 566 | 354 | 21492 | 32.62h |
| H-8 | 270 (90.00%) | 78 (26.00%) | 28.89% | 29 | 1 strict-invalid | 247.03s | 1251 | 442 | 220 | 13348 | 26.19h |
| H-15 | 270 (90.00%) | 81 (27.00%) | 30.00% | 26 | 4 strict-invalid | 250.27s | 1154 | 486 | 302 | 18336 | 26.10h |
| H-30 | 249 (83.00%) | 78 (26.00%) | 31.33% | 45 | 6 strict-invalid | 305.70s | 1307 | 413 | 245 | 14892 | 33.45h |

The “other strict failure” labels differ by runner outcome: this Flat-15 run records 7 explicit `incomplete` outcomes, while the archived comparison reports `status=success` but strict-invalid counts for the other methods. They are all excluded from strict completion and Correct/300, but are not claimed to be the same failure mechanism. Historical token totals and embedding fees are not present in this comparison table and are therefore not imputed.

## 6. Artifacts and integrity

- `retry_validation_per_uid.jsonl`: raw-response hashes, trajectory answers, frozen parser results, statuses, and infrastructure checks for all 57 retry UIDs.
- `final_predictions_gold_free.jsonl`: frozen 300-UID prediction/status closure with failures retained and no gold fields.
- `scored_results.jsonl`: separate post-freeze scored view containing gold.
- `cost_summary.json`: actual-formal, final-selected, smoke, anomalous, and embedding-cost scopes.
- `FIRST_PASS_PRE_RETRY_SNAPSHOT_RECHECK.json`: exact expected/current hashes for the two shared service logs and confirmation that the other 924 pre-retry snapshot entries still match.
- `retry_v2_source_snapshot.sha256`, `final_predictions_gold_free.sha256`, `SOURCE_MANIFEST.sha256`, and `MANIFEST.sha256`: source and finalization integrity manifests. Their corresponding verification files record successful readback.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "state.txt").write_text("COMPLETE / VALID\n", encoding="utf-8")
    (OUT / "gold_access_log.json").write_text(json.dumps({
        "loaded_utc": utc_now(),
        "loaded_in_separate_score_invocation": True,
        "prediction_sha256_verified_before_load": merge["prediction_sha256"],
        "gold_source": str(PARQUET),
        "gold_source_sha256": sha256_file(PARQUET),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Recheck source closure after report construction.
    retry_recheck = verify_sha_manifest(OUT / "retry_v2_source_snapshot.sha256")
    (OUT / "retry_v2_source_snapshot.final_verify.txt").write_text("\n".join(retry_recheck) + "\n", encoding="utf-8")
    source_recheck = verify_sha_manifest(OUT / "SOURCE_MANIFEST.sha256")
    (OUT / "SOURCE_MANIFEST.final_verify.txt").write_text("\n".join(source_recheck) + "\n", encoding="utf-8")

    excluded = {"MANIFEST.sha256", "MANIFEST.verify.txt"}
    bundle_files = [path for path in OUT.iterdir() if path.is_file() and path.name not in excluded]
    write_sha_manifest(OUT / "MANIFEST.sha256", bundle_files, relative_to=OUT)
    bundle_check = verify_sha_manifest(OUT / "MANIFEST.sha256", relative_to=OUT)
    (OUT / "MANIFEST.verify.txt").write_text("\n".join(bundle_check) + "\n", encoding="utf-8")
    for path in OUT.iterdir():
        if path.is_file():
            os.chmod(path, 0o444)
    print(json.dumps({
        "status": "COMPLETE / VALID",
        "correct": correct,
        "strict_completed": len(completed),
        "completed_accuracy": correct / len(completed),
        "bundle_manifest_entries": len(bundle_check),
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("gold-free", "score"))
    args = parser.parse_args()
    if args.command == "gold-free":
        gold_free()
    else:
        score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
