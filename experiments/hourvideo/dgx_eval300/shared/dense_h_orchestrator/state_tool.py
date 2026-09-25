#!/usr/bin/env python3
"""Strict completion/attempt state for resumable V7.4 Eval300 orchestration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

LEGAL = set("ABCDE")
METRIC_KEYS = {
    "uid", "status", "error_type", "elapsed_sec", "steps", "planner_calls",
    "visual_retrieve_calls", "visual_inspect_calls", "planner_tokens",
    "visual_tokens", "sent_images", "unique_sent_timestamps",
    "planner_token_usage", "visual_token_usage",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def uids(path: Path) -> list[str]:
    rows = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if len(rows) != len(set(rows)):
        raise SystemExit(f"duplicate UID in {path}")
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def video_id(uid: str) -> str:
    return uid.rsplit("_", 2)[0]


def metric_path(root: Path, uid: str) -> Path:
    return root / video_id(uid) / "metrics" / f"{uid}.json"


def pred_path(root: Path, uid: str) -> Path:
    return root / video_id(uid) / "preds" / f"{uid}.json"


def trajectories(root: Path, uid: str) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in (root / video_id(uid)).glob("*/trajectory.json"):
        data = read_json(path)
        if str(data.get("uid") or "") == uid:
            found.append((path, data))
    return found


def attempted(root: Path, uid: str) -> bool:
    return read_json(metric_path(root, uid)).get("status") in {"success", "timeout", "failed", "incomplete"}


def complete(root: Path, uid: str) -> tuple[bool, dict[str, Any]]:
    metric = read_json(metric_path(root, uid))
    pred = read_json(pred_path(root, uid))
    letter = str(pred.get("pred") or "").strip().upper()
    candidates = [(p, d) for p, d in trajectories(root, uid) if d.get("finished_at") and str(d.get("answer") or "").strip()]
    metrics_complete = METRIC_KEYS.issubset(metric)
    ok = metric.get("status") == "success" and metrics_complete and letter in LEGAL and bool(candidates)
    latest = max(candidates, key=lambda x: x[0].stat().st_mtime_ns) if candidates else (None, {})
    return ok, {"metric": metric, "prediction": pred, "trajectory_path": str(latest[0]) if latest[0] else None, "trajectory": latest[1]}


def preflight(args: argparse.Namespace) -> int:
    uid_path, index, parquet = Path(args.uids), Path(args.index), Path(args.parquet)
    rows = uids(uid_path)
    actual_sha = sha256(uid_path)
    if len(rows) != 300 or actual_sha != args.expected_sha:
        raise SystemExit(f"frozen UID mismatch count={len(rows)} sha256={actual_sha}")
    from videoseal.runner.per_question_runner import load_tasks_from_parquet
    tasks = load_tasks_from_parquet(parquet, uids=set(rows))
    by_uid = {str(x["uid"]): x for x in tasks}
    if set(by_uid) != set(rows) or len(tasks) != 300:
        raise SystemExit(f"parquet coverage mismatch tasks={len(tasks)} unique={len(by_uid)}")
    vids = sorted({str(x["video_id"]) for x in tasks})
    missing_index = [v for v in vids if not (index / "cases" / v / "shared_hierarchy.json").is_file()]
    missing_video = [str(x["video_path"]) for x in tasks if not Path(str(x["video_path"])).is_file()]
    result = {"uid_count": 300, "unique_uid_count": 300, "uid_sha256": actual_sha, "video_count": len(vids),
              "index_coverage": 12 - len(missing_index), "video_path_coverage": 300 - len(missing_video),
              "missing_index": missing_index, "missing_video_paths": missing_video}
    if len(vids) != 12 or missing_index or missing_video:
        raise SystemExit(json.dumps(result, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def pending(args: argparse.Namespace) -> int:
    rows = uids(Path(args.uids))
    phase = Path(args.phase)
    source = Path(args.source) if args.source else None
    if args.kind == "first":
        selected = [u for u in rows if not attempted(phase, u)]
    else:
        if source is None:
            raise SystemExit("retry pending requires --source")
        selected = [u for u in rows if not complete(source, u)[0] and not attempted(phase, u)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{u}\n" for u in selected), encoding="utf-8")
    print(json.dumps({"kind": args.kind, "pending": len(selected), "output": str(out)}, sort_keys=True))
    return 0


def coverage(args: argparse.Namespace) -> int:
    rows = uids(Path(args.uids)); phase = Path(args.phase)
    attempted_rows = [u for u in rows if attempted(phase, u)]
    completed_rows = [u for u in rows if complete(phase, u)[0]]
    result = {"attempted": len(attempted_rows), "complete": len(completed_rows), "not_attempted": len(rows)-len(attempted_rows)}
    print(json.dumps(result, sort_keys=True))
    if args.require_attempted is not None and len(attempted_rows) != args.require_attempted:
        return 4
    return 0


def infra_gate(args: argparse.Namespace) -> int:
    rows = uids(Path(args.uids)); phase = Path(args.phase)
    streak = maximum = 0
    for uid in rows:
        metric = read_json(metric_path(phase, uid))
        if metric.get("error_type") == "http":
            streak += 1; maximum = max(maximum, streak)
        else:
            streak = 0
    result = {"maximum_consecutive_http_failures": maximum, "threshold": args.threshold}
    print(json.dumps(result, sort_keys=True))
    return 8 if maximum >= args.threshold else 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--uids", required=True); p.add_argument("--expected-sha", required=True); p.add_argument("--index", required=True); p.add_argument("--parquet", required=True); p.set_defaults(func=preflight)
    p = sub.add_parser("pending"); p.add_argument("--uids", required=True); p.add_argument("--phase", required=True); p.add_argument("--kind", choices=("first", "retry"), required=True); p.add_argument("--source"); p.add_argument("--output", required=True); p.set_defaults(func=pending)
    p = sub.add_parser("coverage"); p.add_argument("--uids", required=True); p.add_argument("--phase", required=True); p.add_argument("--require-attempted", type=int); p.set_defaults(func=coverage)
    p = sub.add_parser("infra-gate"); p.add_argument("--uids", required=True); p.add_argument("--phase", required=True); p.add_argument("--threshold", type=int, default=3); p.set_defaults(func=infra_gate)
    ns = ap.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
