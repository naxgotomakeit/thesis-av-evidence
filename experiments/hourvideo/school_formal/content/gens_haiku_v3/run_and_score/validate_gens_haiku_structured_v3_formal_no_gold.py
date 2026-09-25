#!/usr/bin/env python3
"""No-gold structural closure for a completed GenS structured-answer v3 run."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gens_haiku_eval300.runtime import atomic_json, canonical_sha, read_jsonl, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--campaign-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    qids = manifest["question_ids"]
    run_fp = canonical_sha(manifest)
    routes = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in (args.namespace / "routes").glob("*.json")}
    statuses = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in (args.namespace / "route_status").glob("*.json")}
    starts = read_jsonl(args.namespace / "journals/request_start.jsonl")
    ends = read_jsonl(args.namespace / "journals/attempt_end.jsonl")
    ledger = json.loads((args.namespace / "budget_ledger.json").read_text(encoding="utf-8"))
    campaign = json.loads(args.campaign_ledger.read_text(encoding="utf-8"))
    selector = {
        json.loads(line)["qa_uid"]: json.loads(line)
        for line in Path(manifest["config"]["selector_path"]).read_text(encoding="utf-8").splitlines()
        if line
    }
    failures: list[str] = []
    started_order = list(dict.fromkeys(str(row["question_id"]) for row in starts))
    if len(qids) != len(set(qids)) or len(qids) != 300:
        failures.append("manifest population mismatch")
    if set(routes) != set(qids) or set(statuses) != set(qids):
        failures.append("route/status population mismatch")
    if started_order != qids:
        failures.append("request-start identity/order mismatch")
    if any(not status["state"].startswith("terminal_") for status in statuses.values()):
        failures.append("nonterminal route")
    start_ids = [row["attempt_id"] for row in starts]
    end_ids = [row["attempt_id"] for row in ends]
    if len(start_ids) != len(set(start_ids)) or len(end_ids) != len(set(end_ids)):
        failures.append("duplicate attempt IDs")
    if set(start_ids) != set(end_ids) or set(end_ids) != set(ledger["charges"]):
        failures.append("request/attempt/ledger mismatch")
    if abs(sum(float(value) for value in ledger["charges"].values()) - float(ledger["spent_usd"])) > 1e-12:
        failures.append("formal ledger total mismatch")
    campaign_formal = {
        attempt_id: value
        for attempt_id, value in campaign["charges"].items()
        if value["namespace"] == str(args.namespace)
    }
    if set(campaign_formal) != set(end_ids):
        failures.append("campaign/formal attempt mismatch")
    if abs(sum(float(value["usd"]) for value in campaign_formal.values()) - float(ledger["spent_usd"])) > 1e-12:
        failures.append("campaign/formal cost mismatch")

    result_counts = Counter()
    frame_refs = 0
    for qid in qids:
        route = routes.get(qid)
        if route is None:
            continue
        result_class = route.get("result_class")
        result_counts[result_class] += 1
        if route.get("run_fingerprint") != run_fp:
            failures.append(f"fingerprint mismatch:{qid}")
        if result_class == "valid_answer":
            if route.get("prediction") not in list("ABCDE"):
                failures.append(f"valid answer mismatch:{qid}")
            if route.get("response_model") != manifest["config"]["model"]:
                failures.append(f"model mismatch:{qid}")
            if route.get("tool_use_count") != 1 or route.get("content_block_types") != ["tool_use"]:
                failures.append(f"tool structure mismatch:{qid}")
            if route.get("stop_reason") != "tool_use":
                failures.append(f"stop reason mismatch:{qid}")
            tool_input = route.get("normalised_tool_input")
            if not isinstance(tool_input, dict):
                failures.append(f"tool input mismatch:{qid}")
            else:
                reason = str(tool_input.get("reason", ""))
                if not reason.strip() or route.get("reason") != reason:
                    failures.append(f"reason mismatch:{qid}")
        elif result_class not in {"invalid_format", "runtime_failure"}:
            failures.append(f"unknown result class:{qid}")
        source = selector[qid]
        if route.get("video_id") != source["video_id"] or route.get("selected_frame_count") != source["selected_frame_count"]:
            failures.append(f"source identity mismatch:{qid}")
        if len(route.get("selected_frames", [])) != source["selected_frame_count"]:
            failures.append(f"frame count mismatch:{qid}")
        for frame, expected in zip(route.get("selected_frames", []), source["selected_frames"]):
            if (
                frame["chronological_order"] != expected["chronological_order"]
                or frame["expected_sha256"] != expected["frame_sha256"]
                or frame["observed_sha256"] != expected["frame_sha256"]
                or sha256_file(Path(frame["path"])) != expected["frame_sha256"]
            ):
                failures.append(f"frame identity/SHA mismatch:{qid}")
        frame_refs += route.get("selected_frame_count", 0)

    raw_files = sorted([
        *(args.namespace / "routes").glob("*.json"),
        *(args.namespace / "route_status").glob("*.json"),
        args.namespace / "journals/request_start.jsonl",
        args.namespace / "journals/attempt_end.jsonl",
        args.namespace / "budget_ledger.json",
        args.namespace / "run_fingerprint.json",
    ])
    inventory = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in raw_files]
    route_latencies = [float(routes[qid]["resource_totals"]["api_latency_sec"]) for qid in qids if qid in routes]
    wall_times = [float(routes[qid]["route_wall_time_sec"]) for qid in qids if qid in routes]
    observed_wall = None
    if starts and statuses:
        first_ts = min(datetime.fromisoformat(row["timestamp"]) for row in starts)
        last_ts = max(datetime.fromisoformat(status["terminal_at"]) for status in statuses.values())
        observed_wall = (last_ts - first_ts).total_seconds()
    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "population": 300,
        "unique_question_ids": len(set(qids)),
        "canonical_order_exact": started_order == qids,
        "terminal_routes": len(statuses),
        "result_class_counts": dict(result_counts),
        "request_starts": len(starts),
        "attempt_ends": len(ends),
        "transport_retries": len(ends) - len(routes),
        "ledger_charges": len(ledger["charges"]),
        "formal_spent_usd": ledger["spent_usd"],
        "campaign_spent_usd_including_smoke": campaign["spent_usd"],
        "frame_references_verified": frame_refs,
        "run_fingerprint": run_fp,
        "raw_closure_sha256": canonical_sha(inventory),
        "raw_file_inventory": inventory,
        "campaign_ledger_path": str(args.campaign_ledger),
        "campaign_ledger_sha256_at_closure": sha256_file(args.campaign_ledger),
        "latency": {
            "total_api_latency_sec_all_300": sum(route_latencies),
            "mean_api_latency_sec_all_300": statistics.mean(route_latencies) if route_latencies else None,
            "median_api_latency_sec_all_300": statistics.median(route_latencies) if route_latencies else None,
            "total_route_wall_time_sec_all_300": sum(wall_times),
            "mean_route_wall_time_sec_all_300": statistics.mean(wall_times) if wall_times else None,
            "median_route_wall_time_sec_all_300": statistics.median(wall_times) if wall_times else None,
            "observed_experiment_wall_clock_sec": observed_wall,
            "denominator": 300,
        },
        "gold_loaded": False,
        "v1_results_imported": False,
        "v2_results_imported": False,
        "smoke_results_imported": False,
    }
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "raw_file_inventory"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
