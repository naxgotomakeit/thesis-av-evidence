"""Resumable QaEgo4D E1 Core headroom-gate runner.

This module intentionally contains only the approved E1 core conditions:
Blind, Uniform-8, and Oracle <=8.  Uniform-32 is recorded as a separately
approved auxiliary diagnostic and is never scheduled here.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import socket
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .core import (
    CONTEXT_SCOPE,
    TASKS,
    CanonicalCase,
    E1ProtocolError,
    atomic_write_json,
    load_json,
    sha256_file,
    stable_hash,
)
from .model import LocalQwenVLEngineeringModel
from .runner import DEFAULT_ANNOTATION_ROOT, DEFAULT_CONFIG, _checkpoint_path, _load_all_cases, _record


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AMENDMENT = ROOT / "config/experiments/qaego4d_e1_core_amendment_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/qaego4d_e1_core_v1"
CORE_CONDITIONS = ("blind", "uniform_8", "oracle_leq8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _core_checkpoint_path(output_root: Path, case: CanonicalCase, condition: str) -> Path:
    return _checkpoint_path(output_root, case, condition)


def _valid_core_record(record: Any, *, config_hash: str, manifest_hash: str, amendment_hash: str) -> bool:
    required = {
        "question_id", "task", "condition", "clip_uid", "video_uid", "context_scope",
        "success", "config_hash", "manifest_hash", "formal_result", "run_kind",
        "protocol_amendment_hash", "cuda_memory", "model_calls",
    }
    return bool(
        isinstance(record, dict)
        and required <= set(record)
        and record.get("config_hash") == config_hash
        and record.get("manifest_hash") == manifest_hash
        and record.get("protocol_amendment_hash") == amendment_hash
        and record.get("formal_result") is True
        and record.get("run_kind") == "formal_e1_core_headroom_gate"
        and record.get("context_scope") == CONTEXT_SCOPE
        and record.get("condition") in CORE_CONDITIONS
    )


def _load_cached(
    *, output_root: Path, cases: dict[str, list[CanonicalCase]], config_hash: str,
    manifest_hashes: dict[str, str], amendment_hash: str,
) -> tuple[list[dict[str, Any]], list[tuple[CanonicalCase, str]]]:
    cached: list[dict[str, Any]] = []
    pending: list[tuple[CanonicalCase, str]] = []
    for task in TASKS:
        for case in cases[task]:
            for condition in CORE_CONDITIONS:
                path = _core_checkpoint_path(output_root, case, condition)
                record: Any = None
                if path.is_file():
                    try:
                        record = load_json(path)
                    except (json.JSONDecodeError, OSError):
                        corrupt = path.with_suffix(path.suffix + f".corrupt-{int(time.time())}")
                        path.replace(corrupt)
                if _valid_core_record(
                    record, config_hash=config_hash, manifest_hash=manifest_hashes[task],
                    amendment_hash=amendment_hash,
                ):
                    cached.append(record)
                else:
                    pending.append((case, condition))
    return cached, pending


def _closed_correct(record: dict[str, Any]) -> bool | None:
    if record.get("task") != "closed" or not record.get("success"):
        return None
    predicted = record.get("parsed_closed_option")
    truth = record.get("ground_truth_option_letter_after_prediction")
    return bool(predicted is not None and predicted == truth)


def _exact_two_sided_binomial(b: int, c: int) -> float | None:
    """Exact McNemar p-value for discordant paired binary outcomes."""
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, index) for index in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _condition_stats(records: list[dict[str, Any]], *, task: str, condition: str) -> dict[str, Any]:
    rows = [row for row in records if row["task"] == task and row["condition"] == condition]
    successful = [row for row in rows if row.get("success")]
    result: dict[str, Any] = {
        "records": len(rows), "successful": len(successful), "failed": len(rows) - len(successful),
        "mean_answer_model_time_s": mean([float(row["answer_model_time_s"]) for row in successful]) if successful else None,
        "median_answer_model_time_s": median([float(row["answer_model_time_s"]) for row in successful]) if successful else None,
    }
    if task == "closed":
        correct = sum(bool(_closed_correct(row)) for row in successful)
        result.update({"correct": correct, "accuracy": correct / len(successful) if successful else None})
    else:
        # V2.2 requires an independently frozen Open evaluator.  Do not invent
        # an exact-match proxy and accidentally turn it into a formal score.
        result.update({"open_quality_score": None, "open_quality_status": "pending_independent_frozen_evaluator"})
    return result


def _aggregate(
    *, output_root: Path, records: list[dict[str, Any]], metadata: dict[str, Any], expected_records: int,
) -> dict[str, Any]:
    records = sorted(records, key=lambda row: (row["task"], row["question_id"], row["condition"]))
    path = output_root / "per_question_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in records) + "\n", encoding="utf-8")
    by_task_condition = {
        task: {condition: _condition_stats(records, task=task, condition=condition) for condition in CORE_CONDITIONS}
        for task in TASKS
    }
    closed_by_key = {
        (row["question_id"], row["condition"]): row
        for row in records if row["task"] == "closed" and row.get("success")
    }
    paired_rows = []
    b = c = concordant_correct = concordant_wrong = 0
    for case_id in sorted({key[0] for key in closed_by_key}):
        u8 = closed_by_key.get((case_id, "uniform_8"))
        oracle = closed_by_key.get((case_id, "oracle_leq8"))
        if not u8 or not oracle:
            continue
        u8_correct, oracle_correct = bool(_closed_correct(u8)), bool(_closed_correct(oracle))
        if u8_correct and not oracle_correct:
            b += 1
        elif not u8_correct and oracle_correct:
            c += 1
        elif u8_correct:
            concordant_correct += 1
        else:
            concordant_wrong += 1
        paired_rows.append({"question_id": case_id, "uniform_8_correct": u8_correct, "oracle_leq8_correct": oracle_correct})
    closed_u8 = by_task_condition["closed"]["uniform_8"].get("accuracy")
    closed_oracle = by_task_condition["closed"]["oracle_leq8"].get("accuracy")
    paired = {
        "task": "closed", "paired_successful_questions": len(paired_rows),
        "uniform_8_correct_oracle_wrong": b,
        "uniform_8_wrong_oracle_correct": c,
        "concordant_correct": concordant_correct, "concordant_wrong": concordant_wrong,
        "accuracy_gap_oracle_minus_uniform_8": (
            closed_oracle - closed_u8 if closed_u8 is not None and closed_oracle is not None else None
        ),
        "mcnemar_exact_two_sided_p": _exact_two_sided_binomial(b, c),
    }
    successful = [row for row in records if row.get("success")]
    answer_times = [float(row["answer_model_time_s"]) for row in successful if row.get("answer_model_time_s") is not None]
    total_latency = [float(row["total_latency_s"]) for row in successful if row.get("total_latency_s") is not None]
    peaks = [float(row["peak_vram_gib"]) for row in successful if row.get("peak_vram_gib") is not None]
    output = {
        **metadata,
        "expected_records": expected_records,
        "completed_records": len(records),
        "complete": len(records) == expected_records,
        "core_conditions": list(CORE_CONDITIONS),
        "condition_counts": dict(Counter(row["condition"] for row in records)),
        "task_counts": dict(Counter(row["task"] for row in records)),
        "all_canonical_clip_scope": all(row["context_scope"] == CONTEXT_SCOPE for row in records),
        "all_selected_frames_inside_canonical_clip": all(
            all(row["clip_start_sec"] <= frame["actual_timestamp_sec"] < row["clip_end_sec"] for frame in row["selected_frames"])
            for row in records
        ),
        "results_by_task_and_condition": by_task_condition,
        "paired_uniform_8_vs_oracle_leq8_closed": paired,
        "open_quality_gate_status": "inconclusive_pending_independent_frozen_open_evaluator",
        "failures": [
            {"task": row["task"], "question_id": row["question_id"], "condition": row["condition"], "error": row.get("error")}
            for row in records if not row.get("success")
        ],
        "timing": {
            "total_answer_model_time_s": sum(answer_times),
            "mean_answer_model_time_s": mean(answer_times) if answer_times else None,
            "median_answer_model_time_s": median(answer_times) if answer_times else None,
            "total_online_latency_s": sum(total_latency),
            "peak_vram_gib_max": max(peaks) if peaks else None,
        },
    }
    atomic_write_json(output_root / "aggregate_summary.json", output)
    return output


def _protocol_metadata(
    *, config: dict[str, Any], config_hash: str, manifest_hashes: dict[str, str], amendment: dict[str, Any],
    amendment_hash: str, expected_records: int,
) -> dict[str, Any]:
    return {
        "run_id": f"qaego4d-e1-core-{config_hash[:12]}-{amendment_hash[:12]}",
        "created_at": _utc_now(), "run_kind": "formal_e1_core_headroom_gate",
        "config_path": str(DEFAULT_CONFIG), "config_hash": config_hash,
        "config_file_sha256": sha256_file(DEFAULT_CONFIG), "manifest_hashes": manifest_hashes,
        "protocol_amendment": amendment, "protocol_amendment_hash": amendment_hash,
        "expected_question_counts": config["expected_question_counts"], "expected_records": expected_records,
        "answer_model_profile": "formal_candidate", "answer_model_spec": config["answer_models"]["formal_candidate"],
        "open_prompt_hash": stable_hash(config["prompts"]["open"]),
        "closed_prompt_hash": stable_hash(config["prompts"]["closed"]),
        "uniform_8_rule_hash": stable_hash(config["conditions"]["uniform_8"]),
        "oracle_rule_hash": stable_hash(config["conditions"]["oracle_leq8"]),
        "decoding_hash": stable_hash(config["decoding"]),
        "host": socket.gethostname(), "python": platform.python_version(),
        "instrumentation": {"version": "cuda-memory-telemetry-v1", "empty_cache_after_call": True},
    }


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    amendment = load_json(args.amendment)
    if config.get("experiment_id") != "E1" or tuple(amendment.get("core_conditions", ())) != CORE_CONDITIONS:
        raise E1ProtocolError("Formal E1 core protocol/amendment contract mismatch")
    if sha256_file(args.config) != amendment.get("parent_protocol_sha256"):
        raise E1ProtocolError("Frozen parent E1 protocol SHA256 does not match the approved amendment")
    config_hash, amendment_hash = stable_hash(config), stable_hash(amendment)
    cases, manifest_hashes = _load_all_cases(config, args.annotation_root)
    expected_records = sum(len(cases[task]) for task in TASKS) * len(CORE_CONDITIONS)
    metadata = _protocol_metadata(
        config=config, config_hash=config_hash, manifest_hashes=manifest_hashes, amendment=amendment,
        amendment_hash=amendment_hash, expected_records=expected_records,
    )
    if args.plan_only:
        print(json.dumps({
            "planned_question_counts": {task: len(cases[task]) for task in TASKS},
            "planned_condition_counts": {condition: sum(len(cases[task]) for task in TASKS) for condition in CORE_CONDITIONS},
            "planned_records": expected_records, "conditions": list(CORE_CONDITIONS),
            "uniform_32_scheduled": False, "config_hash": config_hash, "amendment_hash": amendment_hash,
        }, ensure_ascii=False))
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "protocol_snapshot.json", metadata)
    cached, pending = _load_cached(
        output_root=args.output_root, cases=cases, config_hash=config_hash,
        manifest_hashes=manifest_hashes, amendment_hash=amendment_hash,
    )
    print(json.dumps({
        "planned_question_counts": {task: len(cases[task]) for task in TASKS},
        "planned_condition_counts": {condition: sum(len(cases[task]) for task in TASKS) for condition in CORE_CONDITIONS},
        "planned_records": expected_records, "cache_hits": len(cached), "pending": len(pending),
        "uniform_32_scheduled": False,
    }, ensure_ascii=False), flush=True)
    if args.plan_only:
        return 0
    session_started = time.perf_counter()
    model: LocalQwenVLEngineeringModel | None = None
    created: list[dict[str, Any]] = []
    load_metadata: dict[str, Any] = {"model_loads": 0, "model_load_time_s": None, "pre_model_load_memory": None, "post_model_load_memory": None}
    try:
        if pending:
            spec = config["answer_models"]["formal_candidate"]
            model = LocalQwenVLEngineeringModel(
                model_path=Path(spec["local_path"]), model_id=str(spec["model_id"]),
                max_pixels=int(config["decoding"]["max_pixels"]), seed=int(config["decoding"]["seed"]),
                minimum_free_vram_gib=float(args.minimum_free_vram_gib),
            )
            load_metadata = {
                "model_loads": 1, "model_load_time_s": model.model_load_time_s,
                "pre_model_load_memory": model.pre_model_load_memory,
                "post_model_load_memory": model.post_model_load_memory,
            }
            for ordinal, (case, condition) in enumerate(pending, start=1):
                record = _record(
                    case=case, condition=condition, config=config, config_hash=config_hash,
                    manifest_hash=manifest_hashes[case.task], model=model, run_id=metadata["run_id"],
                    model_profile="formal_candidate", formal_result=True,
                    run_kind="formal_e1_core_headroom_gate", protocol_amendment_hash=amendment_hash,
                )
                atomic_write_json(_core_checkpoint_path(args.output_root, case, condition), record)
                created.append(record)
                atomic_write_json(args.output_root / "progress.json", {
                    "updated_at": _utc_now(), "expected_records": expected_records,
                    "cache_hits_at_session_start": len(cached), "created_this_session": len(created),
                    "completed_so_far": len(cached) + len(created), "pending_at_session_start": len(pending),
                    "last_completed": {"task": case.task, "question_id": case.question_id, "condition": condition,
                                       "success": record["success"]},
                })
                if ordinal % max(1, int(args.progress_every)) == 0:
                    print(json.dumps({"completed": len(cached) + len(created), "expected": expected_records,
                                      "last": [case.task, case.question_id, condition], "success": record["success"]}, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            model.close()
            del model
        gc.collect()
    all_records = sorted(cached + created, key=lambda row: (row["task"], row["question_id"], row["condition"]))
    result = _aggregate(
        output_root=args.output_root, records=all_records,
        metadata={**metadata, **load_metadata, "session_wall_time_s": time.perf_counter() - session_started,
                  "cache_hits_this_session": len(cached), "created_this_session": len(created)},
        expected_records=expected_records,
    )
    print(json.dumps({"completed": result["completed_records"], "expected": expected_records,
                      "complete": result["complete"], "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)
    return 0 if result["complete"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run formal QaEgo4D E1 Core headroom gate (no Uniform-32)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--plan-only", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
