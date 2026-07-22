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
from src.baselines.egopolice_formal.runner import (
    ROOT,
    atomic_write_json,
    atomic_write_text,
    load_formal_contract,
)
from src.evaluation.egopolice_formal import (
    QUESTION_DURATION_CLASSES,
    VIDEO_DURATION_BINS,
)

from .indexer import DEFAULT_INDEX_ROOT, validate_index
from .prepare_retrieval import DEFAULT_OUTPUT_DIR, validate_retrieval_checkpoint
from .retrieval import build_raw_question_queries
from .runner import build_b1_run_spec, inspect_qa_checkpoints


DEFAULT_B0_RESULTS = ROOT / "outputs/experiments/B0/formal_ablation98_v1/final_results.json"


def _mean(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _median(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    hits = sum(int(row["gt_interval_hit_at_8"]) for row in rows)
    return {
        "correct": correct,
        "total": total,
        "n": total,
        "accuracy": correct / total if total else None,
        "accuracy_percent": 100 * correct / total if total else None,
        "distinct_videos": len({row["source_video_id"] for row in rows}),
        "gt_interval_hit_at_8_hits": hits,
        "gt_interval_hit_at_8_rate": hits / total if total else None,
        "mean_frames_inside_gt_interval": _mean(
            rows, "number_of_frames_inside_gt_interval"
        ),
        "mean_nearest_frame_distance_to_gt_interval_sec": _mean(
            rows, "nearest_sample_distance_to_gt_interval_seconds"
        ),
        "mean_total_online_query_latency_sec": _mean(
            rows, "total_online_query_latency_sec"
        ),
        "median_total_online_query_latency_sec": _median(
            rows, "total_online_query_latency_sec"
        ),
        "mean_query_text_embedding_time_sec": _mean(
            rows, "query_text_embedding_time_sec"
        ),
        "mean_similarity_search_time_sec": _mean(rows, "similarity_search_time_sec"),
        "mean_frame_extraction_latency_sec": _mean(rows, "frame_extraction_latency_sec"),
        "mean_qwen_preprocessing_latency_sec": _mean(rows, "preprocessing_latency_sec"),
        "mean_qwen_inference_latency_sec": _mean(rows, "qwen_inference_latency_sec"),
        "mean_model_facing_frames": _mean(rows, "model_facing_frames"),
        "total_qwen_calls": sum(int(row["model_calls"]) for row in rows),
        "mean_qwen_calls": _mean(rows, "model_calls"),
        "mean_text_input_tokens": _mean(rows, "text_input_tokens"),
        "mean_visual_tokens": _mean(rows, "visual_token_count"),
        "mean_total_input_tokens": _mean(rows, "total_input_tokens"),
        "mean_output_tokens": _mean(rows, "output_tokens"),
        "peak_qwen_gpu_allocated_memory_bytes": max(
            (int(row["peak_qwen_gpu_allocated_memory_bytes"]) for row in rows), default=0
        ),
        "offline_indexing_included_in_online_latency": False,
    }


def aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
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
            raise BaselineInputError(f"Invalid video duration bin: {video_bin}")
        by_question[question_class].append(row)
        by_video[video_bin].append(row)
        cross[(video_bin, question_class)].append(row)
    return {
        "overall": _group(rows),
        "by_question_duration_class": {
            name: _group(by_question[name]) for name in QUESTION_DURATION_CLASSES
        },
        "by_source_video_duration": {
            name: _group(by_video[name]) for name in VIDEO_DURATION_BINS
        },
        "source_video_duration_x_question_duration": {
            video_bin: {
                question_class: _group(cross[(video_bin, question_class)])
                for question_class in QUESTION_DURATION_CLASSES
            }
            for video_bin in VIDEO_DURATION_BINS
        },
    }


def _offline_summary(
    *, index_root: Path, contract: dict[str, Any],
) -> dict[str, Any]:
    question_counts: dict[str, int] = defaultdict(int)
    for question in contract["questions"]:
        question_counts[question["video_id"]] += 1
    rows = []
    for video_id, video in contract["videos"].items():
        valid, metadata, reason = validate_index(
            index_root=index_root, video=video,
            video_manifest_sha256=contract["video_manifest_sha256"],
        )
        if not valid or metadata is None:
            raise BaselineInputError(f"Invalid B1 index during aggregation: {video_id}: {reason}")
        count = question_counts[video_id]
        rows.append({
            "video_id": video_id,
            "formal_question_count": count,
            "video_duration_sec": metadata["video_duration_sec"],
            "embedded_frame_count": metadata["embedded_frame_count"],
            "decode_time_sec": metadata["decode_time_sec"],
            "preprocess_time_sec": metadata["preprocess_time_sec"],
            "visual_embedding_inference_time_sec": metadata[
                "visual_embedding_inference_time_sec"
            ],
            "total_indexing_time_sec_excluding_shared_model_load": metadata[
                "total_indexing_time_sec"
            ],
            "amortized_indexing_time_sec_per_question_excluding_shared_model_load": (
                float(metadata["total_indexing_time_sec"]) / count
            ),
            "peak_gpu_allocated_memory_bytes": metadata[
                "peak_gpu_allocated_memory_bytes"
            ],
            "embedding_dimension": metadata["embedding_dimension"],
            "embedding_dtype": metadata["embedding_dtype"],
            "index_size_bytes": metadata["artifact_size_bytes"],
        })
    sessions = []
    for path in sorted((index_root / "sessions").glob("*.json")):
        session = json.loads(path.read_text(encoding="utf-8"))
        if session.get("video_manifest_sha256") == contract["video_manifest_sha256"]:
            sessions.append(session)
    shared_load = sum(float(row.get("model_load_time_sec") or 0) for row in sessions)
    per_video_total = sum(float(row["total_indexing_time_sec_excluding_shared_model_load"]) for row in rows)
    total_frames = sum(int(row["embedded_frame_count"]) for row in rows)
    for row in rows:
        allocated_load = (
            shared_load * int(row["embedded_frame_count"]) / total_frames
            if total_frames else 0.0
        )
        row["allocated_shared_model_load_by_frame_share_sec"] = allocated_load
        row["amortized_indexing_time_sec_per_question_including_allocated_model_load"] = (
            float(row["total_indexing_time_sec_excluding_shared_model_load"])
            + allocated_load
        ) / int(row["formal_question_count"])
    total = shared_load + per_video_total
    return {
        "video_count": len(rows),
        "total_embedded_frames": total_frames,
        "total_index_size_bytes": sum(int(row["index_size_bytes"]) for row in rows),
        "total_indexing_time_sec_excluding_shared_model_load": per_video_total,
        "shared_model_load_time_sec": shared_load,
        "shared_model_load_count": sum(int(row.get("model_load_count") or 0) for row in sessions),
        "total_indexing_wall_cost_sec_including_recorded_shared_model_loads": total,
        "overall_amortized_indexing_cost_sec_per_question": total / 98,
        "per_video_shared_model_load_allocation_rule": (
            "recorded shared model-load time allocated in proportion to embedded-frame count"
        ),
        "peak_gpu_allocated_memory_bytes": max(
            (int(row["peak_gpu_allocated_memory_bytes"]) for row in rows), default=0
        ),
        "per_video": rows,
        "sessions": sessions,
    }


def _b0_comparison(
    *, b1: dict[str, Any], offline: dict[str, Any], b0_results_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    b0 = json.loads(b0_results_path.read_text(encoding="utf-8"))
    if (
        b0.get("completed_question_count") != 98
        or b0.get("video_manifest_sha256") != contract["video_manifest_sha256"]
        or b0.get("question_manifest_sha256") != contract["question_manifest_sha256"]
    ):
        raise BaselineInputError("B0 comparison artifact is not the paired frozen run")
    b0_overall = b0["aggregate"]["overall"]
    b1_overall = b1["overall"]
    return {
        "paired_question_count": 98,
        "b0_uniform8": {
            "accuracy": b0_overall["accuracy"],
            "gt_interval_hit_at_8_rate": b0_overall["gt_interval_diagnostics"]["hit_at_8_rate"],
            "frames_per_question": b0_overall["mean_model_facing_frames"],
            "qwen_calls_per_question": b0_overall["mean_model_calls"],
            "mean_online_latency_sec": b0_overall["mean_total_per_query_latency_sec"],
            "offline_indexing_cost_sec": 0,
            "amortized_indexing_cost_sec_per_question": 0,
        },
        "b1_raw_question_cradio_top8": {
            "accuracy": b1_overall["accuracy"],
            "gt_interval_hit_at_8_rate": b1_overall["gt_interval_hit_at_8_rate"],
            "frames_per_question": b1_overall["mean_model_facing_frames"],
            "qwen_calls_per_question": b1_overall["mean_qwen_calls"],
            "mean_online_latency_sec": b1_overall["mean_total_online_query_latency_sec"],
            "offline_indexing_cost_sec": offline[
                "total_indexing_wall_cost_sec_including_recorded_shared_model_loads"
            ],
            "amortized_indexing_cost_sec_per_question": offline[
                "overall_amortized_indexing_cost_sec_per_question"
            ],
        },
    }


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _flat(dimension: str, group_name: str, group: dict[str, Any]) -> dict[str, Any]:
    return {"dimension": dimension, "group": group_name, **group}


def write_b1_final_artifacts(
    *, output_dir: Path, index_root: Path, contract: dict[str, Any],
    run_spec: dict[str, Any], retrieval_records: dict[str, dict[str, Any]],
    b0_results_path: Path = DEFAULT_B0_RESULTS,
) -> dict[str, Any]:
    valid, pending, invalid = inspect_qa_checkpoints(
        output_dir=output_dir, contract=contract,
        retrieval_records=retrieval_records, run_spec=run_spec,
        quarantine_invalid=False,
    )
    if len(valid) != 98 or pending or invalid:
        raise BaselineInputError(
            f"B1 aggregation requires 98 valid results: valid={len(valid)}, "
            f"pending={len(pending)}, invalid={len(invalid)}"
        )
    by_id = {row["question_id"]: row for row in valid}
    ordered = [by_id[row["question_id"]] for row in contract["questions"]]
    aggregate = aggregate_records(ordered)
    offline = _offline_summary(index_root=index_root, contract=contract)
    comparison = _b0_comparison(
        b1=aggregate, offline=offline, b0_results_path=b0_results_path,
        contract=contract,
    )
    qwen_sessions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "qwen_sessions").glob("*.json"))
    ]
    retrieval_sessions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "retrieval_sessions").glob("*.json"))
    ]
    aggregate["offline_indexing"] = offline
    aggregate["qwen_session_summary"] = {
        "session_count": len(qwen_sessions),
        "model_load_count": sum(int(row.get("model_load_count") or 0) for row in qwen_sessions),
        "model_load_latency_sec_by_session": [
            row["model_load_latency_sec"] for row in qwen_sessions
            if row.get("model_load_latency_sec") is not None
        ],
        "attempted_model_calls": sum(int(row.get("model_calls") or 0) for row in qwen_sessions),
    }
    aggregate["retrieval_session_summary"] = {
        "session_count": len(retrieval_sessions),
        "text_encoder_load_count": sum(int(row.get("model_load_count") or 0) for row in retrieval_sessions),
        "text_encoder_load_latency_sec_by_session": [
            row["model_load_latency_sec"] for row in retrieval_sessions
            if row.get("model_load_latency_sec") is not None
        ],
    }
    final = {
        "schema_version": "egopolice-b1-formal-final-results-v1",
        "formal_result": True,
        "baseline_id": run_spec["baseline_id"],
        "run_fingerprint": run_spec["run_fingerprint"],
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "completed_question_count": len(ordered),
        "records": ordered,
        "aggregate": aggregate,
        "b0_vs_b1": comparison,
    }
    atomic_write_json(output_dir / "final_results.json", final)
    atomic_write_json(output_dir / "aggregate_summary.json", aggregate)
    atomic_write_json(output_dir / "b0_vs_b1_comparison.json", comparison)
    atomic_write_text(
        output_dir / "per_question_results.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
    )
    question_rows = [
        _flat("question_duration_class", name, aggregate["by_question_duration_class"][name])
        for name in QUESTION_DURATION_CLASSES
    ]
    video_rows = [
        _flat("source_video_duration", name, aggregate["by_source_video_duration"][name])
        for name in VIDEO_DURATION_BINS
    ]
    cross_rows = [
        _flat(
            "source_video_duration_x_question_duration",
            f"{video_bin}__{question_class}",
            aggregate["source_video_duration_x_question_duration"][video_bin][question_class],
        )
        for video_bin in VIDEO_DURATION_BINS
        for question_class in QUESTION_DURATION_CLASSES
    ]
    atomic_write_text(output_dir / "aggregate_summary.csv", _csv_text([
        _flat("overall", "all", aggregate["overall"]), *question_rows, *video_rows, *cross_rows
    ]))
    atomic_write_text(output_dir / "by_question_class.csv", _csv_text(question_rows))
    atomic_write_text(output_dir / "by_video_length.csv", _csv_text(video_rows))
    atomic_write_text(output_dir / "video_length_x_question_class.csv", _csv_text(cross_rows))
    atomic_write_text(output_dir / "offline_indexing_by_video.csv", _csv_text(offline["per_video"]))
    report = (
        "# EgoPolice formal B1 ablation98 v1\n\n"
        "Generated only after all 98 frozen B1 checkpoints validate. B1 uses the exact raw "
        "question for retrieval, eight C-RADIO-ranked 1 FPS frames, and one frozen Qwen call.\n\n"
        f"- Run fingerprint: `{run_spec['run_fingerprint']}`\n"
        f"- Accuracy: {aggregate['overall']['correct']}/98 "
        f"({aggregate['overall']['accuracy_percent']:.2f}%)\n"
        f"- GT Interval Hit@8: {aggregate['overall']['gt_interval_hit_at_8_hits']}/98\n"
        f"- Mean/median online latency: "
        f"{aggregate['overall']['mean_total_online_query_latency_sec']:.3f}/"
        f"{aggregate['overall']['median_total_online_query_latency_sec']:.3f} s\n"
        f"- Offline index size: {offline['total_index_size_bytes']} bytes\n"
        f"- Offline indexing cost: "
        f"{offline['total_indexing_wall_cost_sec_including_recorded_shared_model_loads']:.3f} s\n\n"
        "GT Interval Hit@8 is a coarse interval-exposure diagnostic and does not prove "
        "that decisive visual evidence is visible. No interpretation is made here.\n"
    )
    atomic_write_text(output_dir / "REPORT.md", report)
    return {"completed_question_count": 98, "output_dir": str(output_dir)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate complete formal EgoPolice B1")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--b0-results", type=Path, default=DEFAULT_B0_RESULTS)
    parser.add_argument("--video-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation20_v1.json")
    parser.add_argument("--question-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation_questions_v1.json")
    parser.add_argument("--readiness", type=Path, default=ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines/egopolice_b0.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_formal_contract(
        data_root=args.data_root, video_manifest_path=args.video_manifest,
        question_manifest_path=args.question_manifest, readiness_path=args.readiness,
        config_path=args.config,
    )
    queries = build_raw_question_queries(contract["questions"])
    index_metadata: dict[str, dict[str, Any]] = {}
    for video_id, video in contract["videos"].items():
        valid, metadata, reason = validate_index(
            index_root=args.index_root, video=video,
            video_manifest_sha256=contract["video_manifest_sha256"],
        )
        if not valid or metadata is None:
            raise BaselineInputError(f"Invalid B1 index {video_id}: {reason}")
        index_metadata[video_id] = metadata
    retrieval_records: dict[str, dict[str, Any]] = {}
    for query in queries:
        valid, record, reason = validate_retrieval_checkpoint(
            output_dir=args.output_dir, query=query,
            index_metadata=index_metadata[query.video_id],
            question_manifest_sha256=contract["question_manifest_sha256"],
        )
        if not valid or record is None:
            raise BaselineInputError(f"Invalid B1 retrieval {query.question_id}: {reason}")
        retrieval_records[query.question_id] = record
    run_spec = build_b1_run_spec(
        model_path=args.model_path, contract=contract,
        index_metadata=index_metadata, retrieval_records=retrieval_records,
    )
    result = write_b1_final_artifacts(
        output_dir=args.output_dir, index_root=args.index_root,
        contract=contract, run_spec=run_spec,
        retrieval_records=retrieval_records, b0_results_path=args.b0_results,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
