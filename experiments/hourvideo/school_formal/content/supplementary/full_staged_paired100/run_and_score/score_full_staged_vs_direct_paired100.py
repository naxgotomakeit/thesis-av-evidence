#!/usr/bin/env python3
"""Independent fixed-denominator scoring for frozen paired100 results."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staged_api_v1.config import canonical_sha, sha256_file
from staged_api_v1.store import atomic_json, atomic_text, read_jsonl

MANIFEST = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2/paired100_manifest.json"
RUN = ROOT / "outputs/full_staged_api_formal/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
OUT = ROOT / "outputs/full_staged_api_analysis/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
STRUCTURAL = OUT / "structural_validation.json"


def percentile(values, p):
    if not values: return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    structural = json.loads(STRUCTURAL.read_text(encoding="utf-8"))
    if structural.get("status") != "PASS" or structural.get("gold_loaded") is not False:
        raise RuntimeError("scoring requires a frozen PASS no-gold closure")
    inventory = [{"path": row["path"], "sha256": sha256_file(Path(row["path"])),
                  "bytes": Path(row["path"]).stat().st_size} for row in structural["raw_file_inventory"]]
    if canonical_sha(inventory) != structural["raw_closure_sha256"]:
        raise RuntimeError("raw paired100 closure changed before scoring")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    qids = list(manifest["ordered_question_ids"])
    annotations = json.loads(args.gold.read_text(encoding="utf-8"))
    gold = {question["qid"]: question["correct_answer_label"]
            for video in annotations.values() for question in video["benchmark_dataset"]}
    if not set(qids).issubset(gold):
        raise RuntimeError("gold identity coverage mismatch")

    direct_links = {row["question_id"]: row for row in manifest["direct_r3_pairing"]["links"]}
    rows = []
    staged_statuses = []
    direct_statuses = []
    direct_artifacts = []
    for qid in qids:
        staged = json.loads((RUN / "route_status" / f"{qid}.json").read_text(encoding="utf-8"))
        direct = json.loads(Path(direct_links[qid]["status_path"]).read_text(encoding="utf-8"))
        staged_statuses.append(staged); direct_statuses.append(direct)
        if direct.get("artifact_path") and Path(direct["artifact_path"]).is_file():
            direct_artifacts.append(json.loads(Path(direct["artifact_path"]).read_text(encoding="utf-8")))
        sp, dp, answer = staged.get("prediction"), direct.get("prediction"), gold[qid]
        rows.append({
            "question_id": qid, "video_id": qid.rsplit("_", 2)[0], "gold": answer,
            "staged_state": staged["state"], "staged_prediction": sp,
            "staged_correct": sp in list("ABCDE") and sp == answer,
            "staged_failure_category": staged.get("failure_category"),
            "direct_state": direct["state"], "direct_prediction": dp,
            "direct_correct": dp in list("ABCDE") and dp == answer,
            "direct_failure_category": direct.get("category") if direct["state"] == "terminal_failed" else None,
        })

    attempts = read_jsonl(RUN / "journals/attempt_ends.jsonl")
    starts = read_jsonl(RUN / "journals/request_starts.jsonl")
    controllers = read_jsonl(RUN / "journals/controller_results.jsonl")
    token_fields = ("ordinary_input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
    cost_fields = ("ordinary_input_usd", "cache_creation_usd", "cache_read_usd", "output_usd")
    staged_resources = {field: sum(int(row.get(field) or 0) for row in attempts) for field in token_fields}
    for field in cost_fields:
        staged_resources[field] = sum(float(row.get("cost_breakdown", {}).get(field) or 0) for row in attempts)
    staged_resources.update({
        "cache_aware_usd": sum(float(row.get("cache_aware_usd") or 0) for row in attempts),
        "physical_provider_requests": len(attempts),
        "validation_retry_requests": sum(bool(row.get("is_validation_retry")) for row in attempts),
        "transport_retry_requests": sum(bool(row.get("is_transport_retry")) for row in attempts),
        "provider_envelope_rejections": sum(row.get("accepted") is False for row in controllers),
        "total_api_latency_sec": sum(float(row.get("latency_sec") or 0) for row in attempts),
        "route_wall_time_sum_sec": sum(float(row.get("route_wall_time_sec") or 0) for row in staged_statuses),
        **structural["visual"],
    })
    logical = Counter()
    for row in staged_statuses:
        logical.update({key: int(value) for key, value in (row.get("stage_calls") or {}).items()})
    staged_resources["logical_stage_calls"] = dict(logical)
    staged_resources["logical_stage_calls_total"] = sum(logical.values())
    stage_physical = Counter(row["stage"] for row in attempts)
    staged_resources["physical_requests_by_stage"] = dict(stage_physical)
    route_walls = [float(row.get("route_wall_time_sec") or 0) for row in staged_statuses]
    staged_resources["mean_route_wall_sec_all_100"] = statistics.mean(route_walls)
    staged_resources["median_route_wall_sec_all_100"] = statistics.median(route_walls)
    staged_resources["mean_api_latency_sec_per_route_all_100"] = staged_resources["total_api_latency_sec"] / 100
    if starts:
        first = min(datetime.fromisoformat(row["timestamp_utc"]) for row in starts)
        last = max(datetime.fromisoformat(row["terminal_at_utc"]) for row in staged_statuses)
        staged_resources["observed_formal_wall_clock_sec"] = (last - first).total_seconds()

    direct_resources = {
        "provider_attempts": sum(int(row.get("total_api_attempts") or 0) for row in direct_artifacts),
        "unique_images": sum(int(row.get("unique_images_transmitted") or 0) for row in direct_artifacts),
        "model_turns": sum(int(row.get("rounds") or 0) for row in direct_artifacts),
        "inspection_rounds": sum(
            turn.get("action_type") == "inspect_frames" and turn.get("provider_status") == "accepted"
            for row in direct_artifacts for turn in row.get("turns", [])
        ),
        "ordinary_input_tokens": sum(int(row.get("total_input_tokens") or 0) for row in direct_artifacts),
        "cache_creation_input_tokens": sum(int(row.get("total_cache_creation_input_tokens") or 0) for row in direct_artifacts),
        "cache_read_input_tokens": sum(int(row.get("total_cache_read_input_tokens") or 0) for row in direct_artifacts),
        "output_tokens": sum(int(row.get("total_output_tokens") or 0) for row in direct_artifacts),
        "cache_aware_usd": sum(float(row.get("total_usd") or 0) for row in direct_artifacts),
        "total_api_latency_sec": sum(float(row.get("total_modeled_api_latency_sec") or 0) for row in direct_artifacts),
        "route_wall_time_sum_sec": sum(float(row.get("route_wall_time_sec") or 0) for row in direct_artifacts),
    }
    direct_walls = [float(row.get("route_wall_time_sec") or 0) for row in direct_artifacts]
    direct_resources["route_artifact_count"] = len(direct_artifacts)
    direct_resources["mean_route_wall_sec_all_100"] = statistics.mean(direct_walls)
    direct_resources["median_route_wall_sec_all_100"] = statistics.median(direct_walls)

    staged_correct = sum(row["staged_correct"] for row in rows)
    direct_correct = sum(row["direct_correct"] for row in rows)
    paired = Counter()
    for row in rows:
        paired[("correct" if row["staged_correct"] else "incorrect",
                "correct" if row["direct_correct"] else "incorrect")] += 1
    planner_cost = float(manifest["historical_r3_planner_cost"]["known_recorded_estimated_cost_usd"])
    planner_latency = float(manifest["historical_r3_planner_cost"]["known_latency_sec"])
    report = {
        "schema_version": "full_staged_r3_vs_direct_r3_paired100_score_v1",
        "population": 100,
        "fixed_denominator": 100,
        "staged": {
            "correct": staged_correct, "accuracy": staged_correct / 100,
            "terminal_routes": 100, "prediction_count": sum(row["staged_prediction"] in list("ABCDE") for row in rows),
            "completion_rate": sum(row["staged_prediction"] in list("ABCDE") for row in rows) / 100,
            "failed_routes": 100 - sum(row["staged_prediction"] in list("ABCDE") for row in rows),
            "failure_categories": dict(Counter(row["staged_failure_category"] for row in rows if row["staged_failure_category"])),
            "resources": staged_resources,
        },
        "direct_r3": {
            "correct": direct_correct, "accuracy": direct_correct / 100,
            "terminal_routes": 100, "prediction_count": sum(row["direct_prediction"] in list("ABCDE") for row in rows),
            "completion_rate": sum(row["direct_prediction"] in list("ABCDE") for row in rows) / 100,
            "failed_routes": 100 - sum(row["direct_prediction"] in list("ABCDE") for row in rows),
            "failure_categories": dict(Counter(row["direct_failure_category"] for row in rows if row["direct_failure_category"])),
            "resources": direct_resources,
        },
        "paired_outcomes_staged_then_direct": {f"{left}_{right}": count for (left, right), count in paired.items()},
        "cost_scope": {
            "new_full_staged_downstream_usd": staged_resources["cache_aware_usd"],
            "historical_planner_recorded_estimated_usd": planner_cost,
            "reconstructed_complete_full_staged_cost_usd": planner_cost + staged_resources["cache_aware_usd"],
            "direct_r3_historical_route_cost_usd": direct_resources["cache_aware_usd"],
            "latency_warning": "historical Planner latency plus new downstream latency is a reconstructed additive quantity, not one measured end-to-end wall clock",
            "historical_planner_latency_sec": planner_latency,
            "reconstructed_planner_plus_downstream_route_wall_sum_sec": planner_latency + staged_resources["route_wall_time_sum_sec"],
        },
        "raw_closure_sha256": structural["raw_closure_sha256"],
        "gold_source_path": str(args.gold.resolve()),
        "gold_source_sha256": sha256_file(args.gold),
        "raw_results_modified": False,
        "per_question": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "paired100_score.json", report)
    with (OUT / "paired100_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = [
        "# R3 Full Staged vs R3 Direct paired100", "",
        f"- Full Staged: **{staged_correct}/100 = {staged_correct}%**, predictions {report['staged']['prediction_count']}/100.",
        f"- R3 Direct: **{direct_correct}/100 = {direct_correct}%**, predictions {report['direct_r3']['prediction_count']}/100.",
        "- Fixed denominator: 100 for both methods; failed routes count as incorrect.",
        f"- Paired outcomes (Staged/Direct): both correct {paired[('correct','correct')]}, Staged-only correct {paired[('correct','incorrect')]}, Direct-only correct {paired[('incorrect','correct')]}, neither correct {paired[('incorrect','incorrect')] }.",
        "",
        "## Full Staged resources", "",
        f"- Logical calls: {staged_resources['logical_stage_calls_total']} ({dict(logical)}).",
        f"- Physical requests: {len(attempts)}; validation retries {staged_resources['validation_retry_requests']}; transport retries {staged_resources['transport_retry_requests']}.",
        f"- Images: {staged_resources['physical_image_transmissions_including_retries']} physical transmissions; {staged_resources['unique_reviewed_fine_images']} unique reviewed fine images.",
        f"- Tokens: ordinary input {staged_resources['ordinary_input_tokens']}; cache write {staged_resources['cache_creation_input_tokens']}; cache read {staged_resources['cache_read_input_tokens']}; output {staged_resources['output_tokens']}.",
        f"- API latency sum {staged_resources['total_api_latency_sec']:.3f}s; observed formal wall clock {staged_resources['observed_formal_wall_clock_sec']:.3f}s.",
        "- Failed routes: `4572b198-2c1c-4920-bcf0-95fcebe12261_4_3`, `819c8af7-851f-434f-ab32-318285bc54b1_7_17`, `db3f7933-dfa0-4678-9d4f-393b628ded45_11_2`.",
        "",
        "## Cost scopes", "",
        f"- New Full Staged downstream cost: **${staged_resources['cache_aware_usd']:.6f}**.",
        f"- Historical selected Planner cost: **${planner_cost:.6f}** (recorded usage-based estimate).",
        f"- Reconstructed complete Full Staged cost: **${planner_cost + staged_resources['cache_aware_usd']:.6f}**.",
        f"- Direct R3 historical cost on the same 100: **${direct_resources['cache_aware_usd']:.6f}**.",
        f"- Raw closure SHA: `{structural['raw_closure_sha256']}`.", "",
        "Planner and downstream were run at different times; their summed latency is not reported as a measured end-to-end wall time.",
    ]
    atomic_text(OUT / "PAIRED100_REPORT.md", "\n".join(summary) + "\n")
    archive_files = [MANIFEST, STRUCTURAL, OUT / "raw_file_inventory.json", OUT / "paired100_score.json",
                     OUT / "paired100_comparison.csv", OUT / "PAIRED100_REPORT.md"]
    archive = {"files": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in archive_files],
               "raw_closure_sha256": structural["raw_closure_sha256"]}
    archive["archive_identity_sha256"] = canonical_sha(archive)
    atomic_json(OUT / "archive_manifest.json", archive)
    print(json.dumps({key: value for key, value in report.items() if key != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
