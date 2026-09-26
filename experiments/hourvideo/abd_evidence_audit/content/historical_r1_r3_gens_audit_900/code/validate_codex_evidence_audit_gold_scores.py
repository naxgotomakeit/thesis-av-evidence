#!/usr/bin/env python3
"""Independently validate linked correctness after the evidence audit is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_root", type=Path)
    ap.add_argument("gold", type=Path)
    args = ap.parse_args()
    root = args.audit_root.resolve()
    freeze = json.loads((root / "evidence_audit_freeze.json").read_text())
    frozen = root / "evidence_audit_frozen.jsonl"
    if sha(frozen) != freeze["evidence_audit_sha256"]:
        raise SystemExit("audit must be frozen before gold validation")
    rows = [json.loads(x) for x in (root / "scored_evidence_audit.jsonl").read_text().splitlines() if x]
    source = json.loads(args.gold.read_text())
    gold = {
        item["qid"]: item["correct_answer_label"]
        for video in source.values()
        for item in video["benchmark_dataset"]
    }
    missing = sorted({row["question_id"] for row in rows} - set(gold))
    mismatches = []
    counts = Counter()
    for row in rows:
        prediction = row["prediction"]
        recomputed = isinstance(prediction, str) and prediction in "ABCDE" and prediction == gold.get(row["question_id"])
        counts[row["source_group"]] += int(recomputed)
        if recomputed != row["correct"]:
            mismatches.append({"neutral_id": row["neutral_id"], "question_id": row["question_id"], "method": row["source_group"]})
    report = {
        "status": "PASS" if not missing and not mismatches and counts == Counter({"R1": 88, "R3": 103, "GENS": 93}) else "FAIL",
        "audit_frozen_before_gold": True,
        "frozen_audit_sha256": freeze["evidence_audit_sha256"],
        "gold_source_path": str(args.gold.resolve()),
        "gold_source_sha256": sha(args.gold),
        "gold_question_count": len(gold),
        "audited_route_count": len(rows),
        "missing_gold_question_ids": missing,
        "linked_correctness_mismatches": mismatches,
        "independently_recomputed_correct": dict(counts),
        "raw_results_modified": False,
    }
    out = root / "independent_gold_validation.json"
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
