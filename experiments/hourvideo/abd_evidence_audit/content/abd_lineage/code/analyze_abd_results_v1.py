#!/usr/bin/env python3
"""Independent ABD scorer and reporting package built after a no-gold closure."""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.store import atomic_json, read_jsonl  # noqa: E402
from gens_haiku_eval300.runtime import canonical_sha, sha256_file  # noqa: E402


MANIFEST = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/formal_candidate_manifest.json"
RUN = ROOT / "outputs/abd_eval300_formal_v1"
OUT = ROOT / "outputs/abd_eval300_analysis_v1"
CLOSURE = OUT / "structural_closure_report.json"
GOLD = ROOT / "../../../../HourVideo/benchmark/v1.0_release/json/dev_v1.0_annotations.json"


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows), "total": sum(rows),
        "mean": statistics.fmean(rows) if rows else None,
        "median": statistics.median(rows) if rows else None,
        "p95": percentile(rows, 0.95),
        "min": min(rows) if rows else None, "max": max(rows) if rows else None,
    }


def paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_only = sum(row[f"{left}_correct"] and not row[f"{right}_correct"] for row in rows)
    right_only = sum(row[f"{right}_correct"] and not row[f"{left}_correct"] for row in rows)
    both = sum(row[f"{left}_correct"] and row[f"{right}_correct"] for row in rows)
    neither = len(rows) - left_only - right_only - both
    return {
        "comparison": f"{left}_vs_{right}", "denominator": len(rows),
        "both_correct": both, f"{left}_only_correct": left_only,
        f"{right}_only_correct": right_only, "neither_correct": neither,
        "accuracy_difference_left_minus_right": (
            sum(row[f"{left}_correct"] for row in rows)
            - sum(row[f"{right}_correct"] for row in rows)
        ) / len(rows),
    }


def verify_closure() -> dict[str, Any]:
    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    if closure.get("status") != "PASS" or closure.get("gold_loaded") is not False:
        raise RuntimeError("scoring requires PASS no-gold ABD closure")
    observed = [{
        "path": row["path"], "sha256": sha256_file(Path(row["path"])),
        "bytes": Path(row["path"]).stat().st_size,
    } for row in closure["raw_file_inventory"]]
    if canonical_sha(observed) != closure["raw_closure_sha256"]:
        raise RuntimeError("ABD raw closure changed before scoring")
    return closure


def main() -> None:
    closure = verify_closure()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    annotations = json.loads(GOLD.resolve().read_text(encoding="utf-8"))
    annotation_rows = [question for video in annotations.values() for question in video["benchmark_dataset"]]
    gold = {row["qid"]: row["correct_answer_label"] for row in annotation_rows}
    question_type = {row["qid"]: row.get("task", "unknown") for row in annotation_rows}
    qids = list(dict.fromkeys(task["question_id"] for task in manifest["tasks"]))
    if len(qids) != 300 or any(qid not in gold for qid in qids):
        raise RuntimeError("gold coverage differs from frozen 300 questions")

    starts = {row["task_id"]: row for row in read_jsonl(RUN / "journals/request_starts.jsonl")}
    ends = {row["task_id"]: row for row in read_jsonl(RUN / "journals/attempt_ends.jsonl")}
    statuses = {
        json.loads(path.read_text(encoding="utf-8"))["task_id"]: json.loads(path.read_text(encoding="utf-8"))
        for path in (RUN / "task_status").glob("*.json")
    }
    artifacts = {
        json.loads(path.read_text(encoding="utf-8"))["task_id"]: json.loads(path.read_text(encoding="utf-8"))
        for path in (RUN / "task_artifacts").glob("*.json")
    }
    tasks_by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for task in manifest["tasks"]:
        tasks_by_qid[task["question_id"]][task["variant"]] = task

    per_question: list[dict[str, Any]] = []
    for qid in qids:
        row: dict[str, Any] = {
            "question_id": qid,
            "video_id": tasks_by_qid[qid]["A"]["video_id"],
            "question_type": question_type[qid],
            "gold": gold[qid],
        }
        for variant in "ABD":
            task = tasks_by_qid[qid][variant]
            task_id = task["task_id"]
            artifact, end, start, status = artifacts[task_id], ends[task_id], starts[task_id], statuses[task_id]
            prediction = artifact.get("prediction") if artifact.get("result_class") == "valid_answer" else None
            route_wall = (
                datetime.fromisoformat(status["terminal_at_utc"])
                - datetime.fromisoformat(start["timestamp_utc"])
            ).total_seconds()
            row.update({
                f"{variant}_task_id": task_id,
                f"{variant}_prediction": prediction,
                f"{variant}_correct": prediction == gold[qid],
                f"{variant}_state": status["state"],
                f"{variant}_result_class": artifact["result_class"],
                f"{variant}_failure_category": artifact.get("failure_category"),
                f"{variant}_input_tokens": end["ordinary_input_tokens"],
                f"{variant}_output_tokens": end["output_tokens"],
                f"{variant}_cache_creation_tokens": end["cache_creation_input_tokens"],
                f"{variant}_cache_read_tokens": end["cache_read_input_tokens"],
                f"{variant}_cost_usd": end["cache_aware_usd"],
                f"{variant}_api_latency_sec": end["latency_sec"],
                f"{variant}_route_wall_sec": route_wall,
                f"{variant}_image_count": start["image_count"],
                f"{variant}_map_present": start["map_present"],
            })
        per_question.append(row)

    arms: dict[str, Any] = {}
    for variant in "ABD":
        correct = sum(row[f"{variant}_correct"] for row in per_question)
        states = Counter(row[f"{variant}_state"] for row in per_question)
        result_classes = Counter(row[f"{variant}_result_class"] for row in per_question)
        failures = Counter(row[f"{variant}_failure_category"] or "none" for row in per_question)
        arms[variant] = {
            "denominator": 300, "completed": sum(states.values()),
            "completion_rate": sum(states.values()) / 300,
            "correct": correct, "accuracy": correct / 300,
            "state_counts": dict(states), "result_class_counts": dict(result_classes),
            "failure_category_counts": dict(failures),
            "resources": {
                "provider_requests": 300,
                "ordinary_input_tokens": sum(row[f"{variant}_input_tokens"] for row in per_question),
                "output_tokens": sum(row[f"{variant}_output_tokens"] for row in per_question),
                "cache_creation_input_tokens": sum(row[f"{variant}_cache_creation_tokens"] for row in per_question),
                "cache_read_input_tokens": sum(row[f"{variant}_cache_read_tokens"] for row in per_question),
                "cost_usd": sum(row[f"{variant}_cost_usd"] for row in per_question),
                "image_transmissions": sum(row[f"{variant}_image_count"] for row in per_question),
            },
            "api_latency_sec": distribution(row[f"{variant}_api_latency_sec"] for row in per_question),
            "route_wall_sec": distribution(row[f"{variant}_route_wall_sec"] for row in per_question),
        }

    def grouped(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_question:
            groups[str(row[field])].append(row)
        return {
            name: {
                "questions": len(rows),
                "arms": {variant: {
                    "correct": sum(row[f"{variant}_correct"] for row in rows),
                    "accuracy": sum(row[f"{variant}_correct"] for row in rows) / len(rows),
                } for variant in "ABD"},
                "D_vs_B": paired(rows, "D", "B"),
                "D_vs_A": paired(rows, "D", "A"),
            }
            for name, rows in sorted(groups.items())
        }

    by_video = grouped("video_id")
    by_question_type = grouped("question_type")
    report = {
        "schema_version": "abd_eval300_independent_score_v1",
        "population": {"questions": 300, "tasks": 900, "fixed_denominator_per_arm": 300},
        "arms": arms,
        "primary_comparison": paired(per_question, "D", "B"),
        "supplementary_comparison": paired(per_question, "D", "A"),
        "by_video": by_video,
        "by_original_question_type": by_question_type,
        "raw_closure_sha256": closure["raw_closure_sha256"],
        "gold_source_path": str(GOLD.resolve()),
        "gold_source_sha256": sha256_file(GOLD.resolve()),
        "judge_calls": 0,
        "raw_results_modified": False,
        "per_question": per_question,
    }
    atomic_json(OUT / "abd_score_report.json", report)
    with (OUT / "abd_per_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)
    with (OUT / "abd_arm_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "denominator", "completed", "correct", "accuracy", "input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens", "cost_usd", "mean_api_latency_sec", "p95_api_latency_sec"])
        for variant in "ABD":
            arm, resource = arms[variant], arms[variant]["resources"]
            writer.writerow([variant, 300, arm["completed"], arm["correct"], arm["accuracy"], resource["ordinary_input_tokens"], resource["output_tokens"], resource["cache_creation_input_tokens"], resource["cache_read_input_tokens"], resource["cost_usd"], arm["api_latency_sec"]["mean"], arm["api_latency_sec"]["p95"]])

    def write_group_csv(path: Path, groups: dict[str, Any], group_label: str) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                group_label, "questions", "A_correct", "A_accuracy", "B_correct", "B_accuracy",
                "D_correct", "D_accuracy", "DB_both", "D_only_vs_B", "B_only_vs_D",
                "DB_neither", "D_minus_B", "DA_both", "D_only_vs_A", "A_only_vs_D",
                "DA_neither", "D_minus_A",
            ])
            for name, group in groups.items():
                db, da = group["D_vs_B"], group["D_vs_A"]
                writer.writerow([
                    name, group["questions"], group["arms"]["A"]["correct"], group["arms"]["A"]["accuracy"],
                    group["arms"]["B"]["correct"], group["arms"]["B"]["accuracy"],
                    group["arms"]["D"]["correct"], group["arms"]["D"]["accuracy"],
                    db["both_correct"], db["D_only_correct"], db["B_only_correct"], db["neither_correct"],
                    db["accuracy_difference_left_minus_right"], da["both_correct"], da["D_only_correct"],
                    da["A_only_correct"], da["neither_correct"], da["accuracy_difference_left_minus_right"],
                ])

    write_group_csv(OUT / "abd_by_video.csv", by_video, "video_id")
    write_group_csv(OUT / "abd_by_question_type.csv", by_question_type, "question_type")

    lines = [
        "# ABD Eval300 final report", "",
        f"Raw closure SHA-256: `{closure['raw_closure_sha256']}`.", "",
        "| Arm | Complete | Correct | Accuracy | Input tokens | Output tokens | Cache create/read | Cost | API mean / p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in "ABD":
        arm, resource = arms[variant], arms[variant]["resources"]
        lines.append(f"| {variant} | {arm['completed']}/300 | {arm['correct']} | {arm['accuracy']:.2%} | {resource['ordinary_input_tokens']:,} | {resource['output_tokens']:,} | {resource['cache_creation_input_tokens']:,}/{resource['cache_read_input_tokens']:,} | ${resource['cost_usd']:.6f} | {arm['api_latency_sec']['mean']:.3f}s / {arm['api_latency_sec']['p95']:.3f}s |")
    primary, supplementary = report["primary_comparison"], report["supplementary_comparison"]
    lines += [
        "", "## Paired comparisons", "",
        f"Primary D vs B: both correct {primary['both_correct']}; D-only {primary['D_only_correct']}; B-only {primary['B_only_correct']}; neither {primary['neither_correct']}; accuracy difference {primary['accuracy_difference_left_minus_right']:+.2%}.",
        f"Supplementary D vs A: both correct {supplementary['both_correct']}; D-only {supplementary['D_only_correct']}; A-only {supplementary['A_only_correct']}; neither {supplementary['neither_correct']}; accuracy difference {supplementary['accuracy_difference_left_minus_right']:+.2%}.",
        "", "All arm accuracies use 300 as the denominator. Missing and failed tasks would count as incorrect; this run has no missing tasks. No judge calls were made.",
    ]
    (OUT / "ABD_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    package = OUT / "ABD_RESULTS_PACKAGE.tar.gz"
    analysis_files = [
        OUT / "raw_file_inventory.json", CLOSURE, OUT / "abd_score_report.json",
        OUT / "abd_per_question.csv", OUT / "abd_arm_summary.csv", OUT / "abd_by_video.csv",
        OUT / "abd_by_question_type.csv", OUT / "ABD_FINAL_REPORT.md",
    ]
    identity_files = [
        MANIFEST,
        ROOT / "config/abd_formal_candidate_v1.json",
        ROOT / "config/abd_draft_v1.json",
        ROOT / "config/abd_common_system_prompt_v1.txt",
        ROOT / "drafts/abd_direct_eval300_v1/limited_batch_v1/execution_identity_mapping.json",
        ROOT / "drafts/abd_direct_eval300_v1/limited_batch_v1/batch_control_manifest.json",
        ROOT / "drafts/abd_direct_eval300_v1/limited_batch_v1/authorization_APPROVED_FIRST_QUESTION_ABD.json",
        ROOT / "drafts/abd_direct_eval300_v1/continuation_v1/execution_identity_mapping.json",
        ROOT / "drafts/abd_direct_eval300_v1/continuation_v1/continuation_control_manifest.json",
        ROOT / "drafts/abd_direct_eval300_v1/continuation_v1/authorization_APPROVED_REMAINING_897.json",
        ROOT / "scripts/freeze_abd_raw_closure_v1.py",
        ROOT / "scripts/analyze_abd_results_v1.py",
    ]
    raw_files = [Path(row["path"]) for row in closure["raw_file_inventory"]]
    with tarfile.open(package, "w:gz") as archive:
        for path in [*raw_files, *analysis_files, *identity_files]:
            archive.add(path, arcname=str(path.resolve().relative_to(ROOT.resolve())))
    archive_manifest = {
        "schema_version": "abd_eval300_downloadable_archive_v1",
        "archive_path": str(package.resolve()),
        "archive_sha256": sha256_file(package),
        "archive_bytes": package.stat().st_size,
        "raw_closure_sha256": closure["raw_closure_sha256"],
        "analysis_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in analysis_files],
        "identity_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in identity_files],
    }
    atomic_json(OUT / "archive_manifest.json", archive_manifest)
    print(json.dumps({
        "arms": arms, "primary_comparison": primary,
        "supplementary_comparison": supplementary,
        "raw_closure_sha256": closure["raw_closure_sha256"],
        "archive": archive_manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
