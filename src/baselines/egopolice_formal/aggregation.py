from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.baselines.egopolice_b0.core import BaselineInputError
from src.evaluation.egopolice_formal import (
    QUESTION_DURATION_CLASSES,
    VIDEO_DURATION_BINS,
)

from .runner import (
    CONDITION_BLIND,
    CONDITION_UNIFORM8,
    FORMAL_CONDITIONS,
    ROOT,
    atomic_write_json,
    atomic_write_text,
    build_run_spec,
    default_output_dir,
    inspect_checkpoints,
    load_formal_contract,
)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _median(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def _group(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    total = len(rows)
    result = {
        "correct": correct,
        "total": total,
        "n": total,
        "accuracy": correct / total if total else None,
        "accuracy_percent": 100 * correct / total if total else None,
        "distinct_videos": len({row["source_video_id"] for row in rows}),
        "mean_total_per_query_latency_sec": _mean(rows, "total_per_query_latency_sec"),
        "median_total_per_query_latency_sec": _median(rows, "total_per_query_latency_sec"),
        "mean_frame_extraction_latency_sec": _mean(rows, "frame_extraction_latency_sec"),
        "mean_preprocessing_latency_sec": _mean(rows, "preprocessing_latency_sec"),
        "mean_inference_latency_sec": _mean(rows, "qwen_inference_latency_sec"),
        "mean_model_facing_frames": _mean(rows, "model_facing_frames"),
        "total_model_calls": sum(int(row["model_calls"]) for row in rows),
        "mean_model_calls": _mean(rows, "model_calls"),
        "mean_text_input_tokens": _mean(rows, "text_input_tokens"),
        "mean_visual_tokens": _mean(rows, "visual_token_count"),
        "mean_total_input_tokens": _mean(rows, "total_input_tokens"),
        "mean_output_tokens": _mean(rows, "output_tokens"),
        "peak_gpu_allocated_memory_bytes": max(
            (int(row["peak_gpu_allocated_memory_bytes"]) for row in rows), default=0
        ),
    }
    if condition == CONDITION_UNIFORM8:
        hits = sum(int(row["gt_interval_hit_at_8"]) for row in rows)
        result["gt_interval_diagnostics"] = {
            "applicable": True,
            "hits": hits,
            "total": total,
            "hit_at_8_rate": hits / total if total else None,
            "mean_frames_inside_gt_interval": _mean(
                rows, "number_of_frames_inside_gt_interval"
            ),
            "mean_nearest_frame_distance_to_gt_interval_sec": _mean(
                rows, "nearest_sample_distance_to_gt_interval_seconds"
            ),
        }
    else:
        result["gt_interval_diagnostics"] = {
            "applicable": False,
            "reason": "Blind has no selected frames",
            "hits": None,
            "total": None,
            "hit_at_8_rate": None,
            "mean_frames_inside_gt_interval": None,
            "mean_nearest_frame_distance_to_gt_interval_sec": None,
        }
    return result


def aggregate_records(
    records: Iterable[dict[str, Any]], *, condition: str
) -> dict[str, Any]:
    rows = list(records)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_class = str(row["question_duration_class"])
        video_bin = str(row["source_video_duration_bin"])
        if question_class not in QUESTION_DURATION_CLASSES:
            raise BaselineInputError(f"Invalid question duration class: {question_class}")
        if video_bin not in VIDEO_DURATION_BINS:
            raise BaselineInputError(f"Invalid source duration bin: {video_bin}")
        by_question[question_class].append(row)
        by_video[video_bin].append(row)
        cross[(video_bin, question_class)].append(row)
    return {
        "condition": condition,
        "overall": _group(rows, condition),
        "by_question_duration_class": {
            name: _group(by_question[name], condition)
            for name in QUESTION_DURATION_CLASSES
        },
        "by_source_video_duration": {
            name: _group(by_video[name], condition) for name in VIDEO_DURATION_BINS
        },
        "source_video_duration_x_question_duration": {
            video_bin: {
                question_class: _group(cross[(video_bin, question_class)], condition)
                for question_class in QUESTION_DURATION_CLASSES
            }
            for video_bin in VIDEO_DURATION_BINS
        },
    }


def _csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _flat_group(dimension: str, name: str, group: dict[str, Any]) -> dict[str, Any]:
    gt = group["gt_interval_diagnostics"]
    return {
        "dimension": dimension,
        "group": name,
        "correct": group["correct"],
        "total": group["total"],
        "n": group["n"],
        "accuracy": group["accuracy"],
        "accuracy_percent": group["accuracy_percent"],
        "distinct_videos": group["distinct_videos"],
        "gt_interval_hit_at_8_applicable": gt["applicable"],
        "gt_interval_hit_at_8_hits": gt["hits"],
        "gt_interval_hit_at_8_rate": gt["hit_at_8_rate"],
        "mean_nearest_frame_distance_sec": gt["mean_nearest_frame_distance_to_gt_interval_sec"],
        "mean_total_latency_sec": group["mean_total_per_query_latency_sec"],
        "median_total_latency_sec": group["median_total_per_query_latency_sec"],
        "mean_extraction_latency_sec": group["mean_frame_extraction_latency_sec"],
        "mean_preprocessing_latency_sec": group["mean_preprocessing_latency_sec"],
        "mean_inference_latency_sec": group["mean_inference_latency_sec"],
        "mean_frames": group["mean_model_facing_frames"],
        "total_model_calls": group["total_model_calls"],
        "peak_gpu_allocated_memory_bytes": group["peak_gpu_allocated_memory_bytes"],
    }


def _report_markdown(
    *, condition: str, aggregate: dict[str, Any], run_spec: dict[str, Any]
) -> str:
    overall = aggregate["overall"]
    gt = overall["gt_interval_diagnostics"]
    lines = [
        f"# EgoPolice formal {condition} ablation98 v1",
        "",
        "This report is generated only after all 98 frozen question checkpoints pass validation.",
        "",
        f"- Run fingerprint: `{run_spec['run_fingerprint']}`",
        f"- Model: `{run_spec['model_path']}`",
        f"- Loading: `{run_spec['dtype']}`, quantization `{run_spec['quantization_mode']}`",
        f"- Accuracy: {overall['correct']}/{overall['total']} ({overall['accuracy_percent']:.2f}%)",
        f"- Distinct videos: {overall['distinct_videos']}",
        f"- Mean/median query latency: {overall['mean_total_per_query_latency_sec']:.3f} / {overall['median_total_per_query_latency_sec']:.3f} s",
        f"- Frames/question: {overall['mean_model_facing_frames']:.2f}",
        f"- Total model calls: {overall['total_model_calls']}",
    ]
    if gt["applicable"]:
        lines.append(
            f"- GT Interval Hit@8: {gt['hits']}/{gt['total']} ({100 * gt['hit_at_8_rate']:.2f}%)"
        )
    else:
        lines.append("- GT Interval Hit@8: N/A (Blind has no selected frames)")
    lines += [
        "",
        "## By GT-duration class",
        "",
        "| Class | Correct | Total | Accuracy | GT Interval Hit@8 | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in QUESTION_DURATION_CLASSES:
        group = aggregate["by_question_duration_class"][name]
        group_gt = group["gt_interval_diagnostics"]
        hit = (
            f"{group_gt['hits']}/{group_gt['total']}"
            if group_gt["applicable"] else "N/A"
        )
        lines.append(
            f"| {name} | {group['correct']} | {group['total']} | "
            f"{group['accuracy_percent']:.2f}% | {hit} | "
            f"{group['mean_total_per_query_latency_sec']:.3f} |"
        )
    lines += [
        "",
        "## By full source-video duration",
        "",
        "| Bin | Videos | Correct | Total | Accuracy | GT Interval Hit@8 | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VIDEO_DURATION_BINS:
        group = aggregate["by_source_video_duration"][name]
        group_gt = group["gt_interval_diagnostics"]
        hit = (
            f"{group_gt['hits']}/{group_gt['total']}"
            if group_gt["applicable"] else "N/A"
        )
        lines.append(
            f"| {name} | {group['distinct_videos']} | {group['correct']} | "
            f"{group['total']} | {group['accuracy_percent']:.2f}% | {hit} | "
            f"{group['mean_total_per_query_latency_sec']:.3f} |"
        )
    lines += [
        "",
        "GT Interval Hit@8 is only a coarse temporal interval-exposure diagnostic; it does not prove decisive visual evidence was visible.",
        "",
    ]
    return "\n".join(lines)


def write_final_artifacts(
    *, condition: str, output_dir: Path, contract: dict[str, Any],
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    valid, pending, invalid = inspect_checkpoints(
        output_dir=output_dir, contract=contract, run_spec=run_spec,
        quarantine_invalid=False,
    )
    if len(valid) != 98 or pending or invalid:
        raise BaselineInputError(
            f"Final aggregation requires 98 valid checkpoints: valid={len(valid)}, "
            f"pending={len(pending)}, invalid={len(invalid)}"
        )
    by_id = {row["question_id"]: row for row in valid}
    ordered = [by_id[row["question_id"]] for row in contract["questions"]]
    aggregate = aggregate_records(ordered, condition=condition)
    sessions = []
    for path in sorted((output_dir / "sessions").glob("*.json")):
        sessions.append(json.loads(path.read_text(encoding="utf-8")))
    aggregate["session_summary"] = {
        "session_count": len(sessions),
        "model_load_count": sum(int(row.get("model_load_count") or 0) for row in sessions),
        "attempted_model_calls": sum(int(row.get("model_calls") or 0) for row in sessions),
        "model_load_latency_sec_by_session": [
            row.get("model_load_latency_sec") for row in sessions
            if row.get("model_load_latency_sec") is not None
        ],
    }
    final = {
        "schema_version": "egopolice-formal-final-results-v1",
        "formal_result": True,
        "condition": condition,
        "run_fingerprint": run_spec["run_fingerprint"],
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "completed_question_count": len(ordered),
        "sessions": sessions,
        "records": ordered,
        "aggregate": aggregate,
    }
    atomic_write_json(output_dir / "final_results.json", final)
    atomic_write_json(output_dir / "aggregate_summary.json", aggregate)
    atomic_write_text(
        output_dir / "per_question_results.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
    )

    flat_fields = list(_flat_group("overall", "all", aggregate["overall"]))
    all_rows = [_flat_group("overall", "all", aggregate["overall"])]
    question_rows = [
        _flat_group("question_duration_class", name,
                    aggregate["by_question_duration_class"][name])
        for name in QUESTION_DURATION_CLASSES
    ]
    video_rows = [
        _flat_group("source_video_duration", name,
                    aggregate["by_source_video_duration"][name])
        for name in VIDEO_DURATION_BINS
    ]
    cross_rows = [
        _flat_group(
            "source_video_duration_x_question_duration",
            f"{video_bin}__{question_class}",
            aggregate["source_video_duration_x_question_duration"][video_bin][question_class],
        )
        for video_bin in VIDEO_DURATION_BINS
        for question_class in QUESTION_DURATION_CLASSES
    ]
    atomic_write_text(
        output_dir / "aggregate_summary.csv",
        _csv_text(flat_fields, [*all_rows, *question_rows, *video_rows, *cross_rows]),
    )
    atomic_write_text(
        output_dir / "by_question_class.csv", _csv_text(flat_fields, question_rows)
    )
    atomic_write_text(
        output_dir / "by_video_length.csv", _csv_text(flat_fields, video_rows)
    )
    atomic_write_text(
        output_dir / "video_length_x_question_class.csv",
        _csv_text(flat_fields, cross_rows),
    )
    report_path = output_dir / "REPORT.md"
    atomic_write_text(
        report_path,
        _report_markdown(condition=condition, aggregate=aggregate, run_spec=run_spec),
    )
    return {
        "condition": condition,
        "completed_question_count": len(ordered),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one complete formal EgoPolice control run"
    )
    parser.add_argument("--condition", choices=FORMAL_CONDITIONS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--video-manifest", type=Path,
        default=ROOT / "config/data/egopolice_ablation20_v1.json",
    )
    parser.add_argument(
        "--question-manifest", type=Path,
        default=ROOT / "config/data/egopolice_ablation_questions_v1.json",
    )
    parser.add_argument(
        "--readiness", type=Path,
        default=ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json",
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/baselines/egopolice_b0.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_formal_contract(
        data_root=args.data_root, video_manifest_path=args.video_manifest,
        question_manifest_path=args.question_manifest, readiness_path=args.readiness,
        config_path=args.config,
    )
    run_spec = build_run_spec(
        condition=args.condition, model_path=args.model_path, contract=contract
    )
    result = write_final_artifacts(
        condition=args.condition,
        output_dir=args.output_dir or default_output_dir(args.condition),
        contract=contract, run_spec=run_spec,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
