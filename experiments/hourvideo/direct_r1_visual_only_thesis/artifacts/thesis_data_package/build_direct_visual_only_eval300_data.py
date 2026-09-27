#!/usr/bin/env python3
"""Build a read-only-derived thesis data package from frozen v13 assets."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
V13 = Path("${PROJECT_MSC_ROOT}/direct_r1_visual_only_correction_v13_unified_recovery")
ORIGINAL = Path("${PROJECT_MSC_ROOT}/msc_thesis/main_system/isolated_workspaces/hourvideo_direct_api_eval300_v1/outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1")
GOLD = Path("${PROJECT_MSC_ROOT}/HourVideo/benchmark/v1.0_release/json/dev_v1.0_annotations.json")
FINAL_REPORT_SHA = "6aaada06175b6945063ef90c3ddd47c134fc31317b9cb8b488d3cdb0b10818e6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_sum(rows, field):
    return sum(float(row[field]) for row in rows if row.get(field) is not None)


def route_metrics(artifact: dict) -> dict:
    attempts = artifact.get("provider_attempts", [])
    confirmed = [row for row in attempts if row.get("response_received") is True]
    unknown = [row for row in attempts if row.get("response_received") is not True]
    retries = [row for row in attempts if int(row.get("attempt_index") or 1) > 1]
    return {
        "provider_attempts_recorded": len(attempts),
        "confirmed_responses": len(confirmed),
        "retry_sends": len(retries),
        "unknown_requests": len(unknown),
        "input_tokens_confirmed": int(numeric_sum(confirmed, "input_tokens")),
        "cache_creation_input_tokens_confirmed": int(numeric_sum(confirmed, "cache_creation_input_tokens")),
        "cache_read_input_tokens_confirmed": int(numeric_sum(confirmed, "cache_read_input_tokens")),
        "output_tokens_confirmed": int(numeric_sum(confirmed, "output_tokens")),
        "unique_images_transmitted": int(artifact.get("unique_images_transmitted") or 0),
        "recorded_api_latency_sec": numeric_sum(confirmed, "api_latency_sec"),
        "route_wall_time_sec": float(artifact.get("route_wall_time_sec") or 0),
        "known_route_cost_usd": float(artifact.get("total_usd") or 0),
        "structural_correction_attempts": int(artifact.get("correction_attempts") or 0),
    }


def aggregate(per_route: list[dict]) -> dict:
    def vals(field):
        return [row[field] for row in per_route]

    output = {
        "routes": len(per_route),
        "provider_attempts_recorded": sum(vals("provider_attempts_recorded")),
        "confirmed_responses": sum(vals("confirmed_responses")),
        "retry_sends": sum(vals("retry_sends")),
        "unknown_requests": sum(vals("unknown_requests")),
        "structural_correction_attempts": sum(vals("structural_correction_attempts")),
        "input_tokens_confirmed": sum(vals("input_tokens_confirmed")),
        "cache_creation_input_tokens_confirmed": sum(vals("cache_creation_input_tokens_confirmed")),
        "cache_read_input_tokens_confirmed": sum(vals("cache_read_input_tokens_confirmed")),
        "output_tokens_confirmed": sum(vals("output_tokens_confirmed")),
        "unique_images_transmitted": sum(vals("unique_images_transmitted")),
        "recorded_api_latency_sec": sum(vals("recorded_api_latency_sec")),
        "route_wall_time_sec_sum_not_continuous_wall_clock": sum(vals("route_wall_time_sec")),
        "known_route_cost_usd": sum(vals("known_route_cost_usd")),
    }
    for field in ("provider_attempts_recorded", "confirmed_responses", "unique_images_transmitted", "recorded_api_latency_sec", "route_wall_time_sec", "known_route_cost_usd"):
        name = "route_wall_time_sec_sum_not_continuous_wall_clock" if field == "route_wall_time_sec" else field
        output[f"mean_{field}_per_route"] = statistics.mean(vals(field))
        output[f"median_{field}_per_route"] = statistics.median(vals(field))
    output["known_route_cost_usd_mean_per_route"] = output.pop("mean_known_route_cost_usd_per_route")
    output["known_route_cost_usd_median_per_route"] = output.pop("median_known_route_cost_usd_per_route")
    output["coverage"] = {
        "route_denominator_for_means_and_medians": len(per_route),
        "token_cost_and_api_latency_response_denominator": output["confirmed_responses"],
        "recorded_send_denominator": output["provider_attempts_recorded"],
        "unknown_request_values_not_zero_filled": True,
        "route_wall_time_is_sum_of_cross_run_route_durations_not_single_run_wall_clock": True,
    }
    return output


def main() -> None:
    final_report = V13 / "R1_DIRECT_VISUAL_ONLY_RECOVERED_EVAL300_FINAL_REPORT.md"
    if sha(final_report) != FINAL_REPORT_SHA:
        raise RuntimeError("v13 final report SHA mismatch")

    # Verify the already-frozen final checklist before deriving new data.
    checklist = V13 / "FINAL_AUDIT_SHA256.txt"
    for line in checklist.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        if sha(V13 / relative.strip()) != expected:
            raise RuntimeError(f"v13 checklist mismatch: {relative}")

    merged_path = V13 / "outputs/r1_visual_only_recovered_eval300_v1/merged_results_no_gold.json"
    scored_path = V13 / "outputs/r1_visual_only_recovered_eval300_v1/scored_results_with_gold.json"
    structural_path = V13 / "outputs/r1_visual_only_recovered_eval300_v1/structural_summary.json"
    merged = load(merged_path)
    scored = load(scored_path)
    structural = load(structural_path)
    reuse_manifest_path = V13 / "manifests/route_reuse_manifest_v8.json"
    reuse_manifest = load(reuse_manifest_path)
    v8_manifest_path = V13 / "runs/direct_r1_visual_only_correction_v8_formal_175/formal_manifest_visual_only_correction_v8.json"
    v13_manifest_path = V13 / "runs/direct_r1_visual_only_correction_v13_unified_recovery_remaining_12/recovery_manifest_v13.json"
    v8_manifest = load(v8_manifest_path)
    v13_manifest = load(v13_manifest_path)
    ledger_path = V13 / "runs/direct_r1_visual_only_correction_v13_unified_recovery_remaining_12/budget_ledger.json"
    ledger = load(ledger_path)

    annotations = load(GOLD)
    gold = {q["qid"]: q["correct_answer_label"] for video in annotations.values() for q in video["benchmark_dataset"]}
    merged_by = {(row["method"], row["question_id"]): row for row in merged["results"]}
    scored_by = {(row["method"], row["question_id"]): row for row in scored["per_question"]}
    qids_r1 = {qid for method, qid in merged_by if method == "R1"}
    qids_r3 = {qid for method, qid in merged_by if method == "R3"}
    if len(merged_by) != 600 or len(qids_r1) != 300 or qids_r1 != qids_r3:
        raise RuntimeError("merged R1/R3 population is not an exact paired 300")

    rerun = {row["question_id"]: row for row in reuse_manifest["rerun"]["routes"]}
    reused = {row["question_id"]: row for row in reuse_manifest["reuse_r1"]["routes"]}
    if len(rerun) != 175 or len(reused) != 125 or set(rerun) & set(reused) or set(rerun) | set(reused) != qids_r1:
        raise RuntimeError("175+125 partition mismatch")

    per_question = []
    aggregate_inputs = {"R1": [], "R3": []}
    transitions = Counter()
    transitions_175 = Counter()
    changed = changed_175 = 0
    reuse_identity_failures = []
    new_failures = []
    for qid in sorted(qids_r1):
        video_id = qid.split("_")[0]
        row1, row3 = merged_by[("R1", qid)], merged_by[("R3", qid)]
        score1, score3 = scored_by[("R1", qid)], scored_by[("R3", qid)]
        art1_path, art3_path = Path(row1["source_artifact"]), Path(row3["source_artifact"])
        if sha(art1_path) != row1["source_artifact_sha256"] or sha(art3_path) != row3["source_artifact_sha256"]:
            raise RuntimeError(f"source artifact SHA mismatch for {qid}")
        art1, art3 = load(art1_path), load(art3_path)
        m1, m3 = route_metrics(art1), route_metrics(art3)
        aggregate_inputs["R1"].append(m1)
        aggregate_inputs["R3"].append(m3)

        old_path = ORIGINAL / "route_artifacts" / f"R1:{qid}.json"
        old = load(old_path)
        old_sha = sha(old_path)
        old_valid = old.get("terminal_status") == "final_answer" and old.get("final_prediction") in {"A", "B", "C", "D", "E"}
        old_correct = bool(old_valid and old.get("final_prediction") == gold[qid])
        new_correct = bool(score1["correct"])
        transition = ("correct" if old_correct else "wrong") + "->" + ("correct" if new_correct else "wrong")
        transitions[transition] += 1
        if qid in rerun:
            transitions_175[transition] += 1
        answer_changed = old.get("final_prediction") != row1.get("final_prediction")
        changed += answer_changed
        changed_175 += answer_changed and qid in rerun
        is_new_failure = row1.get("terminal_status") != "final_answer" and old.get("terminal_status") == "final_answer"
        if is_new_failure:
            new_failures.append(qid)

        if qid in reused:
            expected = reused[qid]
            if str(art1_path) != expected["source_artifact"] or old_sha != expected["source_artifact_sha256"] or old.get("final_prediction") != row1.get("final_prediction"):
                reuse_identity_failures.append(qid)

        out = {
            "question_id": qid,
            "video_id": video_id,
            "gold": gold[qid],
            "r1_partition": "rerun_175" if qid in rerun else "reused_125",
            "old_r1_source_path": str(old_path),
            "old_r1_source_sha256": old_sha,
            "old_r1_status": old.get("terminal_status"),
            "old_r1_answer": old.get("final_prediction"),
            "old_r1_correct": old_correct,
            "new_r1_source_kind": row1["source_kind"],
            "new_r1_source_path": str(art1_path),
            "new_r1_source_sha256": row1["source_artifact_sha256"],
            "new_r1_status": row1.get("terminal_status"),
            "new_r1_answer": row1.get("final_prediction"),
            "new_r1_correct": new_correct,
            "r1_old_to_new": transition,
            "r1_answer_changed": answer_changed,
            "r1_new_failure": is_new_failure,
            "new_r1_unknown_reserved_usd": float(row1.get("unknown_reserved_usd") or 0),
            "r3_source_path": str(art3_path),
            "r3_source_sha256": row3["source_artifact_sha256"],
            "r3_status": row3.get("terminal_status"),
            "r3_answer": row3.get("final_prediction"),
            "r3_correct": bool(score3["correct"]),
        }
        for prefix, metrics in (("new_r1", m1), ("r3", m3)):
            out.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        per_question.append(out)

    if reuse_identity_failures:
        raise RuntimeError(f"reused R1 identity mismatch: {reuse_identity_failures[:3]}")
    if Counter(row["new_r1_correct"] for row in per_question)[True] != 85 or Counter(row["r3_correct"] for row in per_question)[True] != 103:
        raise RuntimeError("independent main-result recomputation differs from frozen score")

    csv_path = OUT / "direct_visual_only_eval300_per_question.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)

    methods = {method: aggregate(routes) for method, routes in aggregate_inputs.items()}
    result = {
        "schema_version": "direct_visual_only_eval300_thesis_data_v1",
        "verification": {
            "v13_final_report_sha256": sha(final_report),
            "v13_final_checklist_sha256": sha(checklist),
            "v13_final_checklist_all_ok": True,
            "source_route_artifact_sha_checked": 600,
            "gold_sha256": sha(GOLD),
            "paired_unique_300": True,
            "rerun_175_reuse_125_partition": True,
            "reuse_125_identity_failures": 0,
        },
        "identity": {
            "original_direct_candidate_path": str(ORIGINAL / "formal_manifest_final_candidate_no_api_v3.json"),
            "original_direct_candidate_sha256": sha(ORIGINAL / "formal_manifest_final_candidate_no_api_v3.json"),
            "original_direct_launch_lock_path": str(ORIGINAL / "formal_launch_lock_v3.json"),
            "original_direct_launch_lock_sha256": sha(ORIGINAL / "formal_launch_lock_v3.json"),
            "v8_formal_manifest_path": str(v8_manifest_path),
            "v8_formal_manifest_sha256": sha(v8_manifest_path),
            "v13_recovery_manifest_path": str(v13_manifest_path),
            "v13_recovery_manifest_sha256": sha(v13_manifest_path),
            "map_transform_manifest_path": str(V13 / "manifests/map_transform_manifest.json"),
            "map_transform_manifest_sha256": sha(V13 / "manifests/map_transform_manifest.json"),
            "reuse_manifest_path": str(reuse_manifest_path),
            "reuse_manifest_sha256": sha(reuse_manifest_path),
            "code_freeze_manifest_path": str(V13 / "manifests/code_freeze_manifest.json"),
            "code_freeze_manifest_sha256": sha(V13 / "manifests/code_freeze_manifest.json"),
            "merged_no_gold_path": str(merged_path),
            "merged_no_gold_sha256": sha(merged_path),
            "scored_results_path": str(scored_path),
            "scored_results_sha256": sha(scored_path),
            "protocol_identity": v13_manifest["protocol_identity"],
        },
        "main_results": scored["methods"],
        "paired_r1_r3": scored["paired_comparison"],
        "efficiency": methods,
        "old_to_new_r1": {
            "old_correct": sum(row["old_r1_correct"] for row in per_question),
            "new_correct": sum(row["new_r1_correct"] for row in per_question),
            "net_correct_change": sum(row["new_r1_correct"] for row in per_question) - sum(row["old_r1_correct"] for row in per_question),
            "all_300_transitions": dict(sorted(transitions.items())),
            "rerun_175_transitions": dict(sorted(transitions_175.items())),
            "answer_changed_all_300": changed,
            "answer_changed_rerun_175": changed_175,
            "answer_unchanged_reused_125": 125,
            "new_failure_count": len(new_failures),
            "new_failure_qids": new_failures,
        },
        "cost_scopes": {
            "A_method_evaluation": {
                "r1_visual_only_300_known_route_cost_usd": methods["R1"]["known_route_cost_usd"],
                "r1_formal_unknown_request_count": methods["R1"]["unknown_requests"],
                "r1_formal_unknown_reserved_usd_not_spend": structural["cost"]["formal_unknown_reserved_usd"],
                "r3_300_known_route_cost_usd": methods["R3"]["known_route_cost_usd"],
                "r3_unknown_request_count": methods["R3"]["unknown_requests"],
            },
            "B_correction_project": {
                "ledger_cap_usd": ledger["cap_usd"],
                "settled_usd_including_smoke": ledger["spent_usd"],
                "unknown_reserved_usd_not_spend": ledger["reserved_usd"],
                "available_usd": ledger["cap_usd"] - ledger["spent_usd"] - ledger["reserved_usd"],
                "corrected_175_known_route_cost_usd": structural["cost"]["corrected_175_known_route_cost_usd"],
                "settled_smoke_cost_usd": ledger["spent_usd"] - structural["cost"]["corrected_175_known_route_cost_usd"],
                "formal_unknown_reserved_usd_not_spend": structural["cost"]["formal_unknown_reserved_usd"],
                "historical_smoke_unknown_reserved_usd_not_spend": ledger["reserved_usd"] - structural["cost"]["formal_unknown_reserved_usd"],
            },
        },
        "limitations": [
            "Five R1 formal request outcomes are unknown; their tokens, billed cost, response and latency are unavailable and were not zero-filled.",
            "Route-duration sums combine routes executed across multiple run segments and are not a single continuous Eval300 wall-clock.",
            "The frame manifests are current frozen snapshots; a complete historical hash freeze for every original frame was unavailable.",
            "Cross-time server-side model identity cannot be proven byte-identical.",
            "Input isolation established removal of audio_channel content, but observed answer changes cannot all be causally attributed to ASR removal because reruns occurred at a later time and used recovery-management changes.",
        ],
    }
    (OUT / "direct_visual_only_eval300_statistics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "r1_correct": result["main_results"]["R1"]["correct"],
        "r3_correct": result["main_results"]["R3"]["correct"],
        "transitions": result["old_to_new_r1"],
        "efficiency": methods,
        "cost_scopes": result["cost_scopes"],
        "csv_rows": len(per_question),
    }, indent=2))


if __name__ == "__main__":
    main()
