#!/usr/bin/env python3
"""Independent full-denominator ABD scorer; never imported by generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score(manifest: dict[str, Any], artifact_root: Path, gold: dict[str, str]) -> dict[str, Any]:
    question_ids = list(dict.fromkeys(task["question_id"] for task in manifest["tasks"]))
    if len(question_ids) != 300 or set(gold) != set(question_ids):
        raise RuntimeError("scorer requires exactly the frozen 300-question gold mapping")
    correct: dict[str, dict[str, bool]] = {variant: {} for variant in "ABD"}
    missing: dict[str, int] = {variant: 0 for variant in "ABD"}
    for task in manifest["tasks"]:
        path = artifact_root / f"{task['task_id'].replace(':', '__')}.json"
        prediction = None
        if path.is_file():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if artifact.get("result_class") == "valid_answer":
                prediction = artifact.get("prediction")
        if prediction is None:
            missing[task["variant"]] += 1
        correct[task["variant"]][task["question_id"]] = prediction == gold[task["question_id"]]
    arms = {
        variant: {
            "denominator": 300,
            "correct": sum(correct[variant].values()),
            "accuracy": sum(correct[variant].values()) / 300,
            "failure_or_missing_counted_incorrect": missing[variant],
        }
        for variant in "ABD"
    }
    def paired(left: str, right: str) -> dict[str, Any]:
        return {
            "comparison": f"{left}_vs_{right}",
            "left_better": sum(correct[left][qid] and not correct[right][qid] for qid in question_ids),
            "right_better": sum(correct[right][qid] and not correct[left][qid] for qid in question_ids),
            "both_same_correctness": sum(correct[left][qid] == correct[right][qid] for qid in question_ids),
        }
    return {
        "arms": arms,
        "primary_comparison": paired("D", "B"),
        "supplementary_comparison": paired("D", "A"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.artifact_root,
        json.loads(args.gold.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
