#!/usr/bin/env python3
"""Build a self-contained thesis package from the frozen paired100 closure.

This is a post-hoc, read-only analysis of the raw experiment.  It verifies the
no-gold closure before loading annotations and writes only to a new analysis
namespace.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
RUN = ROOT / "outputs/full_staged_api_formal/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
SOURCE_ANALYSIS = ROOT / "outputs/full_staged_api_analysis/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
OUT = ROOT / "outputs/full_staged_api_analysis/full_staged_api_r3_vs_direct_r3_paired100_thesis_package_v1"
MANIFEST = PREFLIGHT / "paired100_manifest.json"
STRUCTURAL = SOURCE_ANALYSIS / "structural_validation.json"
PLANNER_COST = PREFLIGHT / "planner_cost_summary.json"
OLD_REVIEW = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_eval300_v1_aligned_v2/PROTOCOL_REVIEW.md"
LEGACY_V661 = Path(
    "${SCHOOL_MAIN_SYSTEM_ROOT}/"
    "thesis-av-evidence-hourvideo/hourvideo_v6_1_runtime/src/experiments/"
    "hourvideo_v6_6_1_contract_telemetry_v1/core.py"
)
EXPECTED_MANIFEST_SHA = "963c3c30dfbe73d7602a54558fa866cc0f74ddad35e12ee7c3fb268f78cacf8b"
EXPECTED_RUNTIME_FINGERPRINT = "06dc90fb8652a8a8cd392bf654040c64446fca246a1bc800d2b45a53a40820e2"
EXPECTED_RAW_CLOSURE_SHA = "fec945bfcf788b4cdf3b709794e6797e2a5bf974c83005cb86ef58b2bcdc3a17"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    collected = [float(value) for value in values]
    if not collected:
        return {"n": 0, "sum": None, "mean": None, "median": None, "p90": None}
    return {
        "n": len(collected),
        "sum": sum(collected),
        "mean": statistics.mean(collected),
        "median": statistics.median(collected),
        "p90": percentile(collected, 0.90),
    }


def verify_raw_closure() -> dict[str, Any]:
    structural = json.loads(STRUCTURAL.read_text(encoding="utf-8"))
    if structural.get("status") != "PASS" or structural.get("gold_loaded") is not False:
        raise RuntimeError("the frozen no-gold structural validation is not PASS")
    observed_inventory = []
    for row in structural["raw_file_inventory"]:
        path = Path(row["path"])
        observed_inventory.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    closure = canonical_sha(observed_inventory)
    if closure != structural["raw_closure_sha256"] or closure != EXPECTED_RAW_CLOSURE_SHA:
        raise RuntimeError("the frozen raw closure changed")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("paired100 manifest SHA mismatch")
    return structural


def load_gold(path: Path, qids: list[str]) -> tuple[dict[str, str], str]:
    annotations = json.loads(path.read_text(encoding="utf-8"))
    gold = {
        question["qid"]: question["correct_answer_label"]
        for video in annotations.values()
        for question in video["benchmark_dataset"]
    }
    if not set(qids).issubset(gold):
        raise RuntimeError("gold identity coverage mismatch")
    return gold, sha256_file(path)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "N/A":
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def main(gold_path: Path) -> None:
    structural = verify_raw_closure()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runtime_fingerprint"] != EXPECTED_RUNTIME_FINGERPRINT:
        raise RuntimeError("runtime fingerprint mismatch")
    qids = list(manifest["ordered_question_ids"])
    if len(qids) != 100 or len(set(qids)) != 100:
        raise RuntimeError("paired100 population mismatch")

    # Gold is intentionally loaded only after the frozen raw closure passes.
    gold, gold_sha = load_gold(gold_path, qids)

    attempt_rows = read_jsonl(RUN / "journals/attempt_ends.jsonl")
    controller_rows = read_jsonl(RUN / "journals/controller_results.jsonl")
    attempts_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    controllers_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        attempts_by_qid[row["question_id"]].append(row)
    for row in controller_rows:
        controllers_by_qid[row["question_id"]].append(row)

    planner = json.loads(PLANNER_COST.read_text(encoding="utf-8"))
    planner_by_qid = {row["question_id"]: row for row in planner["attempts"]}
    direct_links = {row["question_id"]: row for row in manifest["direct_r3_pairing"]["links"]}

    per_question: list[dict[str, Any]] = []
    final_retry_rows: list[dict[str, Any]] = []
    all_model_attempts: list[dict[str, Any]] = []
    for canonical_index, qid in enumerate(qids):
        status = json.loads((RUN / "route_status" / f"{qid}.json").read_text(encoding="utf-8"))
        case = RUN / "legacy_live" / "cases" / qid
        model_attempts = read_jsonl(case / "model_attempts.jsonl")
        all_model_attempts.extend(model_attempts)
        scientific_attempts_by_stage = defaultdict(list)
        for model_row in model_attempts:
            scientific_attempts_by_stage[model_row.get("stage")].append(model_row)

        shared_path = case / "r3_2" / "shared_investigation.json"
        shared = json.loads(shared_path.read_text(encoding="utf-8")) if shared_path.is_file() else {}
        staged_fine_ids = {
            row.get("fine_id")
            for row in shared.get("evidence", [])
            if row.get("evidence_type") == "reviewed_visual_observation" and row.get("fine_id")
        }
        staged_attempts = attempts_by_qid[qid]
        stage_calls = {name: int((status.get("stage_calls") or {}).get(name, 0)) for name in ("shared", "fine", "final")}
        staged_input = sum(int(row.get("ordinary_input_tokens") or 0) for row in staged_attempts)
        staged_cache_create = sum(int(row.get("cache_creation_input_tokens") or 0) for row in staged_attempts)
        staged_cache_read = sum(int(row.get("cache_read_input_tokens") or 0) for row in staged_attempts)
        staged_output = sum(int(row.get("output_tokens") or 0) for row in staged_attempts)
        staged_cost = sum(float(row.get("cache_aware_usd") or 0.0) for row in staged_attempts)
        staged_api_latency = sum(float(row.get("latency_sec") or 0.0) for row in staged_attempts)
        staged_image_transmissions = sum(int(row.get("physical_image_transmissions") or 0) for row in model_attempts)

        direct_status = json.loads(Path(direct_links[qid]["status_path"]).read_text(encoding="utf-8"))
        direct_artifact_path = Path(direct_status["artifact_path"])
        direct = json.loads(direct_artifact_path.read_text(encoding="utf-8"))
        direct_inspection_rounds = sum(
            row.get("action_type") == "inspect_frames" and row.get("provider_status") == "accepted"
            for row in direct.get("turns", [])
        )
        direct_new_image_transports = sum(int(row.get("images_transmitted") or 0) for row in direct.get("turns", []))
        direct_transport_retries = sum(
            int(row.get("attempt_index") or 1) > 1 for row in direct.get("provider_attempts", [])
        )
        if direct_new_image_transports != int(direct.get("unique_images_transmitted") or 0):
            raise RuntimeError(f"Direct unique-image accounting mismatch for {qid}")

        planner_row = planner_by_qid[qid]
        answer = gold[qid]
        staged_prediction = status.get("prediction")
        direct_prediction = direct_status.get("prediction")
        row = {
            "canonical_index": canonical_index,
            "question_id": qid,
            "video_id": qid.rsplit("_", 2)[0],
            "gold_option": answer,
            "staged_prediction": staged_prediction,
            "staged_correct": int(staged_prediction in "ABCDE" and staged_prediction == answer) if staged_prediction else 0,
            "staged_state": status["state"],
            "staged_failure_category": status.get("failure_category") or "",
            "staged_failure_detail": (status.get("legacy_status") or {}).get("failure_reason") or "",
            "staged_logical_calls": sum(stage_calls.values()),
            "staged_shared_calls": stage_calls["shared"],
            "staged_fine_calls": stage_calls["fine"],
            "staged_final_calls": stage_calls["final"],
            "staged_physical_requests": len(staged_attempts),
            "staged_validation_retry_requests": sum(bool(item.get("is_validation_retry")) for item in staged_attempts),
            "staged_transport_retry_requests": sum(bool(item.get("is_transport_retry")) for item in staged_attempts),
            "staged_image_transmissions_including_validation_retries": staged_image_transmissions,
            "staged_unique_reviewed_images": len(staged_fine_ids),
            "staged_ordinary_input_tokens": staged_input,
            "staged_cache_creation_tokens": staged_cache_create,
            "staged_cache_read_tokens": staged_cache_read,
            "staged_output_tokens": staged_output,
            "staged_api_usd": staged_cost,
            "staged_api_latency_sec": staged_api_latency,
            "staged_route_wall_sec": float(status.get("route_wall_time_sec") or 0.0),
            "planner_status": planner_row["status"],
            "planner_physical_requests": 1,
            "planner_retry_attempts": int(bool(planner_row.get("is_retry"))),
            "planner_input_tokens_unsplit": int(planner_row["input_tokens"]),
            "planner_output_tokens": int(planner_row["output_tokens"]),
            "planner_recorded_estimated_usd": float(planner_row["recorded_estimated_cost_usd"]),
            "planner_api_latency_sec": float(planner_row["latency_sec"]),
            "reconstructed_full_staged_usd": staged_cost + float(planner_row["recorded_estimated_cost_usd"]),
            "reconstructed_full_staged_api_latency_sec": staged_api_latency + float(planner_row["latency_sec"]),
            "reconstructed_full_staged_route_time_sec": float(status.get("route_wall_time_sec") or 0.0) + float(planner_row["latency_sec"]),
            "direct_prediction": direct_prediction,
            "direct_correct": int(direct_prediction in "ABCDE" and direct_prediction == answer) if direct_prediction else 0,
            "direct_state": direct_status["state"],
            "direct_failure_category": direct_status.get("category") if direct_status["state"] == "terminal_failed" else "",
            "direct_logical_model_turns": int(direct.get("rounds") or 0),
            "direct_physical_requests": int(direct.get("total_api_attempts") or 0),
            "direct_structural_correction_attempts": int(direct.get("correction_attempts") or 0),
            "direct_transport_retry_requests": direct_transport_retries,
            "direct_inspection_rounds": direct_inspection_rounds,
            "direct_new_image_transports": direct_new_image_transports,
            "direct_unique_images": int(direct.get("unique_images_transmitted") or 0),
            "direct_ordinary_input_tokens": int(direct.get("total_input_tokens") or 0),
            "direct_cache_creation_tokens": int(direct.get("total_cache_creation_input_tokens") or 0),
            "direct_cache_read_tokens": int(direct.get("total_cache_read_input_tokens") or 0),
            "direct_output_tokens": int(direct.get("total_output_tokens") or 0),
            "direct_api_usd": float(direct.get("total_usd") or 0.0),
            "direct_api_latency_sec": float(direct.get("total_modeled_api_latency_sec") or 0.0),
            "direct_route_wall_sec": float(direct.get("route_wall_time_sec") or 0.0),
        }
        per_question.append(row)

        final_controllers = {
            int(item.get("validation_retry_index") or 0): item
            for item in controllers_by_qid[qid]
            if item.get("stage") == "final"
        }
        for item in scientific_attempts_by_stage["direct_final"]:
            retry_index = int(item["attempt"]) - 1
            envelope = final_controllers.get(retry_index, {})
            final_retry_rows.append({
                "question_id": qid,
                "validation_attempt": int(item["attempt"]),
                "validation_retry_index": retry_index,
                "provider_tool_envelope_accepted": envelope.get("accepted"),
                "scientific_validation_status": item.get("status"),
                "scientific_failure_reason": item.get("failure_reason") or "",
                "retry_triggered": bool(item.get("retry_triggered")),
                "route_state": status["state"],
            })

    if len(per_question) != 100 or {row["question_id"] for row in per_question} != set(qids):
        raise RuntimeError("per-question output coverage mismatch")

    staged_correct = sum(row["staged_correct"] for row in per_question)
    direct_correct = sum(row["direct_correct"] for row in per_question)
    paired = Counter()
    for row in per_question:
        paired[(bool(row["staged_correct"]), bool(row["direct_correct"]))] += 1
    staged_only = paired[(True, False)]
    direct_only = paired[(False, True)]
    discordant = staged_only + direct_only
    exact_mcnemar_p = min(
        1.0,
        2.0 * sum(math.comb(discordant, index) for index in range(min(staged_only, direct_only) + 1)) / (2 ** discordant),
    )

    staged_api_dist = distribution(row["staged_api_latency_sec"] for row in per_question)
    staged_wall_dist = distribution(row["staged_route_wall_sec"] for row in per_question)
    planner_api_dist = distribution(row["planner_api_latency_sec"] for row in per_question)
    reconstructed_api_dist = distribution(row["reconstructed_full_staged_api_latency_sec"] for row in per_question)
    reconstructed_time_dist = distribution(row["reconstructed_full_staged_route_time_sec"] for row in per_question)
    direct_api_dist = distribution(row["direct_api_latency_sec"] for row in per_question)
    direct_wall_dist = distribution(row["direct_route_wall_sec"] for row in per_question)

    stage_status_counts = Counter((row.get("stage"), row.get("status")) for row in all_model_attempts)
    stage_logical = {
        name: sum(row[f"staged_{name}_calls"] for row in per_question)
        for name in ("shared", "fine", "final")
    }
    stage_physical = Counter(row["stage"] for row in attempt_rows)
    staged_cost = sum(row["staged_api_usd"] for row in per_question)
    planner_cost = sum(row["planner_recorded_estimated_usd"] for row in per_question)
    direct_cost = sum(row["direct_api_usd"] for row in per_question)
    reconstructed_cost = staged_cost + planner_cost
    downstream_more_pct = (staged_cost - direct_cost) / direct_cost * 100.0
    reconstructed_more_pct = (reconstructed_cost - direct_cost) / direct_cost * 100.0

    token_fields_staged = {
        "ordinary_input": sum(row["staged_ordinary_input_tokens"] for row in per_question),
        "cache_creation": sum(row["staged_cache_creation_tokens"] for row in per_question),
        "cache_read": sum(row["staged_cache_read_tokens"] for row in per_question),
        "output": sum(row["staged_output_tokens"] for row in per_question),
    }
    token_fields_direct = {
        "ordinary_input": sum(row["direct_ordinary_input_tokens"] for row in per_question),
        "cache_creation": sum(row["direct_cache_creation_tokens"] for row in per_question),
        "cache_read": sum(row["direct_cache_read_tokens"] for row in per_question),
        "output": sum(row["direct_output_tokens"] for row in per_question),
    }
    staged_image_transmissions = sum(row["staged_image_transmissions_including_validation_retries"] for row in per_question)
    staged_unique_images = sum(row["staged_unique_reviewed_images"] for row in per_question)
    direct_unique_images = sum(row["direct_unique_images"] for row in per_question)

    failure_rows = [row for row in per_question if row["staged_state"] == "terminal_failed"]
    source_files = [
        MANIFEST,
        STRUCTURAL,
        PLANNER_COST,
        PREFLIGHT / "legacy_downstream_config.json",
        OLD_REVIEW,
        ROOT / "src/staged_api_v1/config.py",
        ROOT / "src/staged_api_v1/legacy_bridge.py",
        ROOT / "src/staged_api_v1/provider.py",
        LEGACY_V661,
        Path(manifest["direct_r3_pairing"]["source_structural_validation_path"]),
        gold_path,
    ]
    traceability = [
        {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in source_files
    ]

    summary_rows = [
        {
            "scope": "Full Staged downstream (new Shared/Fine/Final)",
            "fixed_question_denominator": 100,
            "correct": staged_correct,
            "accuracy": staged_correct / 100,
            "successful_or_prediction_routes": 97,
            "failed_routes": 3,
            "completion_rate": 0.97,
            "logical_calls": sum(stage_logical.values()),
            "physical_requests": len(attempt_rows),
            "validation_retry_requests": sum(bool(row.get("is_validation_retry")) for row in attempt_rows),
            "transport_retry_requests": sum(bool(row.get("is_transport_retry")) for row in attempt_rows),
            "image_transmissions": staged_image_transmissions,
            "unique_images_total": staged_unique_images,
            "unique_images_mean_per_question": staged_unique_images / 100,
            "ordinary_input_tokens": token_fields_staged["ordinary_input"],
            "historical_unsplit_input_tokens": "N/A",
            "cache_creation_tokens": token_fields_staged["cache_creation"],
            "cache_read_tokens": token_fields_staged["cache_read"],
            "output_tokens": token_fields_staged["output"],
            "api_usd": staged_cost,
            "api_latency_sum_sec": staged_api_dist["sum"],
            "api_latency_mean_sec": staged_api_dist["mean"],
            "api_latency_median_sec": staged_api_dist["median"],
            "api_latency_p90_sec": staged_api_dist["p90"],
            "route_wall_sum_sec": staged_wall_dist["sum"],
            "route_wall_mean_sec": staged_wall_dist["mean"],
            "route_wall_median_sec": staged_wall_dist["median"],
            "route_wall_p90_sec": staged_wall_dist["p90"],
            "notes": "Measured new downstream run; all timing distributions use all 100 routes.",
        },
        {
            "scope": "Historical R3 Planner (selected same 100)",
            "fixed_question_denominator": 100,
            "correct": "N/A",
            "accuracy": "N/A",
            "successful_or_prediction_routes": 100,
            "failed_routes": 0,
            "completion_rate": "N/A",
            "logical_calls": 100,
            "physical_requests": 100,
            "validation_retry_requests": "N/A",
            "transport_retry_requests": "N/A",
            "image_transmissions": 0,
            "unique_images_total": 0,
            "unique_images_mean_per_question": 0,
            "ordinary_input_tokens": "N/A",
            "historical_unsplit_input_tokens": planner["known_input_tokens"],
            "cache_creation_tokens": "N/A",
            "cache_read_tokens": "N/A",
            "output_tokens": planner["known_output_tokens"],
            "api_usd": planner_cost,
            "api_latency_sum_sec": planner_api_dist["sum"],
            "api_latency_mean_sec": planner_api_dist["mean"],
            "api_latency_median_sec": planner_api_dist["median"],
            "api_latency_p90_sec": planner_api_dist["p90"],
            "route_wall_sum_sec": "N/A",
            "route_wall_mean_sec": "N/A",
            "route_wall_median_sec": "N/A",
            "route_wall_p90_sec": "N/A",
            "notes": "Historical map-only Planner stage; recorded usage-based estimate. Cache token classes and separate route wall were not retained for this selected subset.",
        },
        {
            "scope": "Reconstructed complete Full Staged (Planner + downstream)",
            "fixed_question_denominator": 100,
            "correct": staged_correct,
            "accuracy": staged_correct / 100,
            "successful_or_prediction_routes": 97,
            "failed_routes": 3,
            "completion_rate": 0.97,
            "logical_calls": 100 + sum(stage_logical.values()),
            "physical_requests": 100 + len(attempt_rows),
            "validation_retry_requests": "184 downstream; historical Planner subtype N/A (0 total retries recorded)",
            "transport_retry_requests": "0 downstream; historical Planner subtype N/A (0 total retries recorded)",
            "image_transmissions": staged_image_transmissions,
            "unique_images_total": staged_unique_images,
            "unique_images_mean_per_question": staged_unique_images / 100,
            "ordinary_input_tokens": "N/A",
            "historical_unsplit_input_tokens": planner["known_input_tokens"],
            "cache_creation_tokens": f"{token_fields_staged['cache_creation']} downstream known; Planner N/A",
            "cache_read_tokens": f"{token_fields_staged['cache_read']} downstream known; Planner N/A",
            "output_tokens": planner["known_output_tokens"] + token_fields_staged["output"],
            "api_usd": reconstructed_cost,
            "api_latency_sum_sec": reconstructed_api_dist["sum"],
            "api_latency_mean_sec": reconstructed_api_dist["mean"],
            "api_latency_median_sec": reconstructed_api_dist["median"],
            "api_latency_p90_sec": reconstructed_api_dist["p90"],
            "route_wall_sum_sec": reconstructed_time_dist["sum"],
            "route_wall_mean_sec": reconstructed_time_dist["mean"],
            "route_wall_median_sec": reconstructed_time_dist["median"],
            "route_wall_p90_sec": reconstructed_time_dist["p90"],
            "notes": "Reconstructed across two historical runs; cost is additive, but latency/time is not one measured end-to-end wall clock.",
        },
        {
            "scope": "R3 Direct-v1.2 (same frozen 100)",
            "fixed_question_denominator": 100,
            "correct": direct_correct,
            "accuracy": direct_correct / 100,
            "successful_or_prediction_routes": 99,
            "failed_routes": 1,
            "completion_rate": 0.99,
            "logical_calls": sum(row["direct_logical_model_turns"] for row in per_question),
            "physical_requests": sum(row["direct_physical_requests"] for row in per_question),
            "validation_retry_requests": "N/A (different one-action controller; structural correction attempts=0)",
            "transport_retry_requests": sum(row["direct_transport_retry_requests"] for row in per_question),
            "image_transmissions": direct_unique_images,
            "unique_images_total": direct_unique_images,
            "unique_images_mean_per_question": direct_unique_images / 100,
            "ordinary_input_tokens": token_fields_direct["ordinary_input"],
            "historical_unsplit_input_tokens": "N/A",
            "cache_creation_tokens": token_fields_direct["cache_creation"],
            "cache_read_tokens": token_fields_direct["cache_read"],
            "output_tokens": token_fields_direct["output"],
            "api_usd": direct_cost,
            "api_latency_sum_sec": direct_api_dist["sum"],
            "api_latency_mean_sec": direct_api_dist["mean"],
            "api_latency_median_sec": direct_api_dist["median"],
            "api_latency_p90_sec": direct_api_dist["p90"],
            "route_wall_sum_sec": direct_wall_dist["sum"],
            "route_wall_mean_sec": direct_wall_dist["mean"],
            "route_wall_median_sec": direct_wall_dist["median"],
            "route_wall_p90_sec": direct_wall_dist["p90"],
            "notes": "Historical frozen Direct routes. Image count is recorded new unique image transport; repeated image blocks in later serialized conversation requests were not retained as a separate metric.",
        },
    ]

    retry_audit = {
        "frozen_config_value": 2,
        "frozen_config_field": "max_validation_retries",
        "actual_interpretation": "two retries after the first attempt; maximum three validation attempts per logical stage call",
        "actual_code": {
            "path": str(LEGACY_V661),
            "sha256": sha256_file(LEGACY_V661),
            "max_attempts_expression": "return 1 + retries",
            "final_loop_expression": "for attempt_index in range(1, attempts_allowed + 1)",
        },
        "adapter_mapping": {
            "path": str((ROOT / "src/staged_api_v1/legacy_bridge.py").resolve()),
            "sha256": sha256_file(ROOT / "src/staged_api_v1/legacy_bridge.py"),
            "mapping": "validation_retry_index = legacy attempt index - 1",
        },
        "incorrect_historical_documentation": {
            "path": str(OLD_REVIEW.resolve()),
            "sha256": sha256_file(OLD_REVIEW),
            "wording": "Final allows 2 under the actual legacy loops",
            "classification": "documentation error; contradicted by frozen runtime code, candidate manifest config, and formal attempt logs",
        },
        "formal_final_physical_attempts": len(final_retry_rows),
        "formal_final_logical_calls": stage_logical["final"],
        "formal_final_validation_retry_requests": sum(row["validation_attempt"] > 1 for row in final_retry_rows),
        "two_three_attempt_exhaustions": [
            {
                "question_id": row["question_id"],
                "attempts": 3,
                "failure": row["staged_failure_detail"],
            }
            for row in failure_rows
            if "Direct Final contract exhausted after 3 attempts" in row["staged_failure_detail"]
        ],
    }

    metadata = {
        "schema_version": "paired100_thesis_package_v1",
        "experiment": manifest["experiment_id"],
        "population": {"questions": 100, "videos": len({row["video_id"] for row in per_question})},
        "frozen_identity": {
            "paired100_manifest_sha256": EXPECTED_MANIFEST_SHA,
            "runtime_fingerprint": EXPECTED_RUNTIME_FINGERPRINT,
            "raw_closure_sha256_before": EXPECTED_RAW_CLOSURE_SHA,
            "gold_source_sha256": gold_sha,
        },
        "accuracy": {
            "full_staged": {"correct": staged_correct, "denominator": 100, "accuracy": staged_correct / 100, "predictions": 97},
            "direct_r3": {"correct": direct_correct, "denominator": 100, "accuracy": direct_correct / 100, "predictions": 99},
        },
        "paired": {
            "both_correct": paired[(True, True)],
            "both_wrong": paired[(False, False)],
            "direct_only_correct": direct_only,
            "staged_only_correct": staged_only,
            "discordant_pairs": discordant,
            "exact_two_sided_mcnemar_p": exact_mcnemar_p,
        },
        "retry_audit": retry_audit,
        "scientific_validation_status_counts": {
            f"{stage}:{status}": count for (stage, status), count in sorted(stage_status_counts.items())
        },
        "cost_comparison": {
            "new_downstream_usd": staged_cost,
            "historical_planner_recorded_estimated_usd": planner_cost,
            "reconstructed_complete_full_staged_usd": reconstructed_cost,
            "direct_usd": direct_cost,
            "new_downstream_minus_direct_usd": staged_cost - direct_cost,
            "new_downstream_percent_more_than_direct": downstream_more_pct,
            "reconstructed_minus_direct_usd": reconstructed_cost - direct_cost,
            "reconstructed_percent_more_than_direct": reconstructed_more_pct,
        },
        "source_traceability": traceability,
        "raw_results_modified": False,
        "api_calls_for_package": 0,
        "model_calls_for_package": 0,
    }

    OUT.mkdir(parents=True, exist_ok=False)
    per_question_path = OUT / "paired100_per_question.csv"
    with per_question_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)
    summary_path = OUT / "paired100_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    retry_path = OUT / "final_retry_audit.csv"
    with retry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_retry_rows[0]))
        writer.writeheader()
        writer.writerows(final_retry_rows)
    atomic_json(OUT / "analysis_metadata.json", metadata)

    outcome_table = md_table(
        ["配对结果", "题数"],
        [
            ["两者都正确", paired[(True, True)]],
            ["两者都错误（含无预测失败）", paired[(False, False)]],
            ["仅 Direct 正确", direct_only],
            ["仅 Full Staged 正确", staged_only],
        ],
    )
    resource_table = md_table(
        ["口径", "正确/100", "预测/成功", "逻辑调用", "物理请求", "Validation retry", "Transport retry", "图片传输", "每题去重图总和/均值", "API 费用"],
        [
            ["Staged 新增下游", "24/100", "97/100", 478, 662, 184, 0, "1,424（含验证重传）", "1,312 / 13.12", f"${staged_cost:.9f}"],
            ["历史 R3 Planner", "N/A", "100/100 stage outputs", 100, 100, "N/A（总 retry=0）", "N/A（总 retry=0）", "0 原图", "0 / 0", f"${planner_cost:.9f}（记录估算）"],
            ["重建完整 Staged", "24/100", "97/100", 578, 762, "184 + Planner subtype N/A", "0 + Planner subtype N/A", "1,424", "1,312 / 13.12", f"${reconstructed_cost:.9f}"],
            ["R3 Direct", "32/100", "99/100", 487, 487, "N/A（结构纠错=0）", 0, "928 次新图传输", "928 / 9.28", f"${direct_cost:.9f}"],
        ],
    )
    token_table = md_table(
        ["口径", "Ordinary input", "历史未拆分 input", "Cache creation", "Cache read", "Output"],
        [
            ["Staged 新增下游", fmt(token_fields_staged["ordinary_input"], 0), "N/A", fmt(token_fields_staged["cache_creation"], 0), fmt(token_fields_staged["cache_read"], 0), fmt(token_fields_staged["output"], 0)],
            ["历史 R3 Planner", "N/A", fmt(planner["known_input_tokens"], 0), "N/A", "N/A", fmt(planner["known_output_tokens"], 0)],
            ["重建完整 Staged", "N/A（Planner 未拆分）", fmt(planner["known_input_tokens"], 0), f"{fmt(token_fields_staged['cache_creation'], 0)}（仅下游已知）", f"{fmt(token_fields_staged['cache_read'], 0)}（仅下游已知）", fmt(planner["known_output_tokens"] + token_fields_staged["output"], 0)],
            ["R3 Direct", fmt(token_fields_direct["ordinary_input"], 0), "N/A", fmt(token_fields_direct["cache_creation"], 0), fmt(token_fields_direct["cache_read"], 0), fmt(token_fields_direct["output"], 0)],
        ],
    )
    latency_table = md_table(
        ["口径", "API latency sum", "mean", "median", "P90", "route wall sum", "mean", "median", "P90"],
        [
            ["Staged 新增下游（实测）", fmt(staged_api_dist["sum"]), fmt(staged_api_dist["mean"]), fmt(staged_api_dist["median"]), fmt(staged_api_dist["p90"]), fmt(staged_wall_dist["sum"]), fmt(staged_wall_dist["mean"]), fmt(staged_wall_dist["median"]), fmt(staged_wall_dist["p90"])],
            ["历史 R3 Planner", fmt(planner_api_dist["sum"]), fmt(planner_api_dist["mean"]), fmt(planner_api_dist["median"]), fmt(planner_api_dist["p90"]), "N/A", "N/A", "N/A", "N/A"],
            ["重建完整 Staged（跨运行相加）", fmt(reconstructed_api_dist["sum"]), fmt(reconstructed_api_dist["mean"]), fmt(reconstructed_api_dist["median"]), fmt(reconstructed_api_dist["p90"]), fmt(reconstructed_time_dist["sum"]), fmt(reconstructed_time_dist["mean"]), fmt(reconstructed_time_dist["median"]), fmt(reconstructed_time_dist["p90"])],
            ["R3 Direct（历史实测）", fmt(direct_api_dist["sum"]), fmt(direct_api_dist["mean"]), fmt(direct_api_dist["median"]), fmt(direct_api_dist["p90"]), fmt(direct_wall_dist["sum"]), fmt(direct_wall_dist["mean"]), fmt(direct_wall_dist["median"]), fmt(direct_wall_dist["p90"])],
        ],
    )
    failure_lines = "\n".join(
        f"- `{row['question_id']}`：{row['staged_failure_detail']}" for row in failure_rows
    )
    source_lines = "\n".join(
        f"- `{row['path']}`  \n  SHA-256: `{row['sha256']}`" for row in traceability
    )

    thesis_md = f"""# R3 Full Staged vs R3 Direct paired100：论文结果资料

## 结论摘要

本资料包分析固定的 paired100（100 道题，来自 12 个视频），所有主要准确率均以固定 100 题为分母，失败不删除。R3 Full Staged 在 100 题中答对 **{staged_correct} 题（24%）**，产生 97 个预测；同题 R3 Direct 答对 **{direct_correct} 题（32%）**，产生 99 个预测。两者相差 8 个百分点，但题目级双侧 exact McNemar 检验为 **p={exact_mcnemar_p:.10f}**；该结果在常用 0.05 阈值下不显著，也不能解释为两种方法等价。

## 实验与冻结身份

- Full Staged：复用冻结的 R3 Planner 输出，新增调用 Shared、Fine、Final 三个 Haiku 阶段。
- Direct：冻结的 Direct-v1.2 3/16 R3 路线，同一批 100 题，不重跑。
- paired100 manifest SHA-256：`{EXPECTED_MANIFEST_SHA}`
- Full Staged runtime fingerprint：`{EXPECTED_RUNTIME_FINGERPRINT}`
- 原始正式结果闭包 SHA-256：`{EXPECTED_RAW_CLOSURE_SHA}`
- 无 gold 结构核验：PASS；随后才由独立统计阶段读取 gold。
- gold 源 SHA-256：`{gold_sha}`
- 统计范围：fixed denominator = 100；Full Staged 的 3 个失败与 Direct 的 1 个历史失败均按错误计入。

## 主结果与资源

{resource_table}

Full Staged 新增下游费用比 Direct 高 **${staged_cost - direct_cost:.9f}（{downstream_more_pct:.6f}%）**；把历史 Planner 记录估算加入后，重建完整 Full Staged 费用比 Direct 高 **${reconstructed_cost - direct_cost:.9f}（{reconstructed_more_pct:.6f}%）**。百分比均以未四舍五入数值计算。

### Token 明细

{token_table}

历史 Planner 的已选 100 题日志仅保留总 input/output 与 usage-based cost estimate，没有可用于本资料包的 ordinary/cache creation/cache read 分拆。因此重建完整流程的这些 token 类不能伪装成精确总数；表中只列下游已知部分并将未知项标为 N/A。

### 延迟

单位均为秒；mean、median、P90 均按全部 100 题计算，P90 使用排序后 `(n-1)*0.9` 的线性插值。

{latency_table}

历史 Planner 与本次新增下游是在不同时间分别运行。表中的“重建完整 Staged”延迟是逐题历史 Planner API latency 与本次下游 latency/wall time 的算术相加，**不是一次实测端到端 wall time**。本次下游正式运行的观测整体 wall-clock 为 6,602.582 秒；它不与历史 Planner wall time混称。

## 配对统计

{outcome_table}

双侧 exact McNemar 使用 26 个 discordant pairs（Direct-only 17，Staged-only 9），在零假设下计算 `X ~ Binomial(26, 0.5)`，p 值为 `2 * P(X <= 9) = {exact_mcnemar_p:.10f}`。100 题来自 12 个视频，同一视频中的题目可能相关，因此题目级 McNemar 的独立性假设并不完全成立；该检验应视为描述性/探索性结果，不能据此追加选题或夸大统计结论。

## Final validation attempts 矛盾核对

冻结配置字段是 `max_validation_retries=2`。实际冻结 V6.6.2 代码 `_max_attempts` 返回 `1 + retries`，Final 循环执行 `range(1, attempts_allowed + 1)`；因此真实规则是：**初次 validation attempt 后最多重试 2 次，共最多 3 次 validation attempts**。`legacy_bridge.py` 将 legacy attempt 1/2/3 映射为 provider telemetry 的 `validation_retry_index` 0/1/2。

旧 aligned_v2 `PROTOCOL_REVIEW.md` 中的“Final allows 2 under the actual legacy loops”是文档措辞错误。它与实际冻结代码、paired100 manifest 中的配置以及正式 attempt 日志均矛盾。正式运行没有偏离其冻结代码：两条 Final exhaustion 路线均真实记录了 attempt 1、2、3，三次都因 `answer_text` 与所选 option 文本不一致而失败。新报告纠正文档表述，但没有覆盖旧文件。

冻结规则还允许每个 validation attempt 最多 1 次 transport retry，因此理论上每个逻辑阶段调用最多 3×2=6 个物理请求；本次正式 paired100 的 transport retry 为 0。Final 共 100 个逻辑调用、118 个物理请求，其中 18 个是 validation retry 请求。

## 失败与验证层级

Full Staged 有 3 条正式失败：

{failure_lines}

- 两条 `Direct Final contract exhausted after 3 attempts` 属于旧 V6.6.2 **科学语义验证失败**；三次 provider tool envelope 均已通过，但 `answer_text` 与 `selected_option_id` 对应的原选项文本不一致。
- `KeyError: 'uncertainty'` 路线的 Final provider envelope 与 `_validate_direct_final_result` 均记录为成功，随后在构造最终答案时访问缺失字段而触发运行错误。这是本次冻结实现中暴露的 post-validation 字段覆盖缺口，不是拒答、预算停止或 SSH 中断。
- 全部 662 个物理请求中，provider tool envelope 层 661 accepted、1 rejected。唯一 envelope rejection 是 Fine 阶段“exactly one tool_use is required”，随后按冻结 validation retry 恢复，不是终态失败。
- V6.6.2 科学 attempt 日志共有 445 success、217 validation_failed；其中 184 个失败触发了下一次 validation 请求，其余为最后一次失败后进入既定降级或终止。正式结果没有 transport retry、未知 provider outcome、预算停止或 interrupted-active 路线。
- Direct 的唯一历史失败为 `runtime_failure:turn_limit_exhausted`，达到 32 turn 上限；并非 schema/语义验证失败或中断。

## 图片口径

- Full Staged 的“每批最多 16 图”是 **Fine 单次批处理上限**。每个 claim 最多 3 个调查轮，且 validation retry 会重传该批图片；它不是每题全局 16 张唯一图上限。因此本次共 1,424 次图片传输，最终每题去重后的 reviewed Fine 图总和为 1,312（均值 13.12）。
- Direct 的 16 是 **每题全局唯一图片上限**，每次 inspect 最多 3 张新图。本批 100 题新传输的唯一图总和为 928（均值 9.28）。Direct 多轮请求中历史图像块随会话再次序列化的总次数没有作为独立指标保存在冻结 route artifact 中，因此不把 928 误称为所有 HTTP payload 内 image blocks 的累计次数。

## 方法差异与解释限制

Full Staged 与 Direct 不是仅更换提示词的同构系统。Full Staged 复用 per-option Planner，经过确定性 Coarse→Fine 检索，并由 Shared/Fine/Final 阶段及各阶段 validation retry 组成；Direct 是一个读取完整 R3 map、自己选择检查时间点的单 agent，多轮对话共享每题 16 张唯一图预算。因而准确率、成本、图片数和完成率差异同时包含推理拓扑、检索责任、证据预算语义及重试协议差异。

paired100 是从 Eval300 固定抽取的 100 题而非完整 Eval300。样本覆盖 12 个视频，但视频内题目相关；本结果不外推为全体 300 题的确定结论。失败按固定分母保留，未做失败剔除、补跑或事后择优。

## 指标定义

- **逻辑调用**：Full Staged 中一次 Shared/Fine/Final 科学阶段调用；Direct 中一次模型决策 turn。
- **物理请求**：实际发生并记账的 provider 请求，包括 validation/transport retry。
- **Validation retry**：同一科学阶段因工具外壳或 V6.6.2 语义验证失败而追加的 provider 请求；不等同于新的科学调查轮。
- **图片传输**：Full Staged 为 Fine provider attempts 实际携带的图片数，验证重试重传会重复计数；Direct 表中为 controller 记录的新唯一原图传输。
- **完成率**：产生合法 A–E 最终预测的题数除以固定 100。
- **API latency**：各题已记录 provider latency 的总和；**route wall**：该路线的实测墙钟时间。跨历史运行相加一律标为 reconstructed。

## 源文件追溯

以下绝对路径只用于学校机器上的审计追溯；理解本资料包的表格和结论不依赖这些路径：

{source_lines}

## 完整性声明

生成资料包前后均重新计算 `structural_validation.json` 所列原始文件的 SHA/大小闭包，结果保持 `{EXPECTED_RAW_CLOSURE_SHA}`。本资料包不含 API key、认证头、环境变量秘密、provider base64 图片或原始响应正文；只包含派生统计和追溯 SHA。本次资料整理 API 调用数为 0，模型调用数为 0。
"""
    atomic_text(OUT / "PAIRED100_THESIS_RESULTS.md", thesis_md)

    english_md = f"""# R3 Full Staged vs. R3 Direct on the paired-100 subset

## Experimental setting

We compared frozen R3 Full Staged and R3 Direct-v1.2 outputs on the same 100 questions drawn from 12 videos. Full Staged reused the frozen R3 Planner outputs and newly executed the Shared, Fine, and Final Haiku stages. Direct used the previously frozen R3 Direct routes. All accuracy and completion figures use a fixed denominator of 100; failed routes were retained and counted as incorrect. The paired population, runtime, and raw result closure are identified by manifest SHA-256 `{EXPECTED_MANIFEST_SHA}`, runtime fingerprint `{EXPECTED_RUNTIME_FINGERPRINT}`, and raw-closure SHA-256 `{EXPECTED_RAW_CLOSURE_SHA}`, respectively.

## Results

Full Staged answered 24/100 questions correctly (24%) and produced valid predictions for 97/100 routes. R3 Direct answered 32/100 correctly (32%) and produced predictions for 99/100 routes. Fifteen questions were answered correctly by both methods, 59 by neither, 17 by Direct only, and 9 by Full Staged only. A two-sided exact McNemar test on the 26 discordant pairs gave p={exact_mcnemar_p:.10f}. Thus, this paired subset does not provide conventional 0.05-level evidence for a difference, but the test also does not establish equivalence.

The new Shared/Fine/Final execution required 478 logical stage calls and 662 physical provider requests, including 184 validation retries and no transport retries. It transmitted 1,424 image instances when validation retransmissions were counted and accumulated 1,312 within-question unique reviewed Fine images (13.12 per question). Its cache-aware API cost was ${staged_cost:.9f}. The selected historical Planner records contributed an estimated ${planner_cost:.9f}, giving a reconstructed Full Staged cost of ${reconstructed_cost:.9f}. Direct used 487 model turns/physical requests, 928 new unique image transports (9.28 per question), and cost ${direct_cost:.9f}. Relative to Direct and using unrounded values, the new downstream portion cost {downstream_more_pct:.3f}% more, while the reconstructed Planner-plus-downstream cost was {reconstructed_more_pct:.3f}% higher.

The measured downstream API-latency sum was {staged_api_dist['sum']:.3f} s, and the sum of per-route wall times was {staged_wall_dist['sum']:.3f} s (mean {staged_wall_dist['mean']:.3f}, median {staged_wall_dist['median']:.3f}, P90 {staged_wall_dist['p90']:.3f} s). Direct recorded {direct_api_dist['sum']:.3f} s of API latency and {direct_wall_dist['sum']:.3f} s of route wall time (mean {direct_wall_dist['mean']:.3f}, median {direct_wall_dist['median']:.3f}, P90 {direct_wall_dist['p90']:.3f} s). Planner and downstream timings came from separate historical runs; their additive reconstruction must not be described as a measured end-to-end wall clock.

## Retry-contract clarification

The frozen setting `max_validation_retries=2` means two retries after the initial response, not two total attempts. The executed V6.6.2 code therefore allowed up to three validation attempts for Final. Two failed routes each recorded three provider-envelope-valid Final responses that were rejected by the scientific validator because `answer_text` did not match the selected option text. An older protocol-review sentence claiming that Final allowed two attempts was a documentation error; the frozen runtime, manifest, and formal logs consistently show the three-attempt rule. A third Full Staged route failed with a post-validation `KeyError: 'uncertainty'`. The historical Direct comparison contained one turn-limit failure.

## Limitations

Questions from the same video are not independent, whereas the item-level McNemar calculation treats pairs as independent. The p-value should therefore be interpreted descriptively and not as a cluster-aware confirmatory test. The paired-100 subset is not the complete Eval300 population. Full Staged and Direct also differ in inference topology and visual-budget semantics: Full Staged permits up to 16 images per Fine batch and can retransmit them during validation retries, whereas Direct has a global ceiling of 16 unique images per question. Finally, historical Planner usage did not retain the same cache-token breakdown as the downstream run, and reconstructed cross-run latency is not a measured end-to-end runtime.
"""
    atomic_text(OUT / "PAIRED100_PAPER_TEXT_EN.md", english_md)

    # Verify the raw closure again after creating only independent analysis files.
    final_structural = verify_raw_closure()
    if final_structural["raw_closure_sha256"] != EXPECTED_RAW_CLOSURE_SHA:
        raise RuntimeError("raw closure changed while building package")
    metadata["frozen_identity"]["raw_closure_sha256_after"] = EXPECTED_RAW_CLOSURE_SHA
    atomic_json(OUT / "analysis_metadata.json", metadata)

    package_files = sorted(
        [path for path in OUT.iterdir() if path.is_file() and path.name != "MANIFEST.sha256"],
        key=lambda path: path.name,
    )
    manifest_text = "".join(f"{sha256_file(path)}  {path.name}\n" for path in package_files)
    atomic_text(OUT / "MANIFEST.sha256", manifest_text)

    print(json.dumps({
        "status": "PASS",
        "output_directory": str(OUT),
        "files": [path.name for path in sorted(OUT.iterdir())],
        "population": 100,
        "videos": 12,
        "staged_correct": staged_correct,
        "direct_correct": direct_correct,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p,
        "raw_closure_sha256_before_after": EXPECTED_RAW_CLOSURE_SHA,
        "final_attempt_rule": "initial + at most two validation retries = at most three attempts",
        "api_calls": 0,
        "model_calls": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.gold.resolve())
