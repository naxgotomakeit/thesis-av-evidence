#!/usr/bin/env python3
"""Validate retry outputs without treating backfilled/placeholder predictions as resolved."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LEGAL_ANSWER = re.compile(r"^[A-E](?:[\s,]*[A-E])*$")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uids", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    expected = [line.strip() for line in args.uids.read_text().splitlines() if line.strip()]
    expected_set = set(expected)
    metrics: dict[str, dict] = {}
    for path in args.run_root.glob("*/metrics/*.json"):
        try:
            item = load_json(path)
        except Exception:
            continue
        uid = str(item.get("uid") or "")
        if uid in expected_set:
            metrics[uid] = item

    trajectories: dict[str, list[tuple[Path, dict]]] = {}
    for path in args.run_root.glob("*/*/trajectory.json"):
        try:
            item = load_json(path)
        except Exception:
            continue
        uid = str(item.get("uid") or "")
        if uid in expected_set:
            trajectories.setdefault(uid, []).append((path, item))

    resolved: list[dict] = []
    unresolved: list[dict] = []
    for uid in expected:
        metric = metrics.get(uid, {})
        candidates = sorted(trajectories.get(uid, []), key=lambda pair: str(pair[1].get("finished_at") or ""))
        valid = []
        for path, traj in candidates:
            answer = str(traj.get("answer") or "").strip().upper()
            complete = bool(traj.get("finished_at")) and isinstance(traj.get("steps"), list) and bool(traj.get("steps"))
            if complete and LEGAL_ANSWER.fullmatch(answer):
                valid.append((path, traj, answer))
        if metric.get("status") == "success" and valid:
            path, traj, answer = valid[-1]
            resolved.append(
                {
                    "uid": uid,
                    "answer": answer,
                    "trajectory": str(path),
                    "elapsed_sec": metric.get("elapsed_sec", traj.get("elapsed_sec")),
                    "steps": len(traj.get("steps") or []),
                    "note": traj.get("note"),
                }
            )
        else:
            unresolved.append(
                {
                    "uid": uid,
                    "status": metric.get("status", "missing"),
                    "error_type": metric.get("error_type", "missing"),
                    "valid_complete_trajectory": bool(valid),
                }
            )

    report = {
        "expected": len(expected),
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "resolved_items": resolved,
        "unresolved_items": unresolved,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("expected", "resolved", "unresolved")}, sort_keys=True))
    if args.require_all and unresolved:
        return 40
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
