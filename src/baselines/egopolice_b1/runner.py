from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from src.baselines.egopolice_b0.core import (
    BaselineInputError,
    _limit_pixels,
    build_mcq_prompt,
    evaluate_gt_exposure,
)
from src.baselines.egopolice_b0.model import Qwen25VL7BBaseline, preflight_environment
from src.baselines.egopolice_formal.runner import (
    ROOT,
    atomic_write_json,
    load_formal_contract,
    sha256_file,
    stable_sha256,
    utc_now,
)

from .constants import (
    B1_BASELINE_ID,
    B1_RESULT_SCHEMA_VERSION,
    B1_RUN_SCHEMA_VERSION,
    QWEN_MINIMUM_FREE_VRAM_BYTES,
    cradio_frozen_configuration,
)
from .indexer import DEFAULT_CACHE_ROOT, DEFAULT_INDEX_ROOT, validate_index
from .prepare_retrieval import (
    DEFAULT_OUTPUT_DIR,
    prepare_retrieval,
    retrieval_paths,
    validate_retrieval_checkpoint,
)
from .retrieval import build_raw_question_queries


QwenFactory = Callable[..., Any]
EnvironmentChecker = Callable[..., dict[str, Any]]
FrameExtractor = Callable[..., tuple[list[Image.Image], float]]


def extract_frames_at_timestamps(
    *, video_path: Path, timestamps_sec: Sequence[float], ffmpeg_path: str,
    max_pixels: int,
) -> tuple[list[Image.Image], float]:
    if len(timestamps_sec) != 8 or len(set(float(value) for value in timestamps_sec)) != 8:
        raise BaselineInputError("B1 requires exactly eight unique frame timestamps")
    started = time.perf_counter()
    images: list[Image.Image] = []
    try:
        for timestamp in timestamps_sec:
            command = [
                ffmpeg_path, "-hide_banner", "-loglevel", "error",
                "-ss", f"{float(timestamp):.6f}", "-i", str(video_path),
                "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            ]
            completed = subprocess.run(command, capture_output=True, check=False)
            if completed.returncode != 0 or not completed.stdout:
                message = completed.stderr.decode("utf-8", errors="replace").strip()
                raise BaselineInputError(
                    message or f"ffmpeg failed at {float(timestamp):.3f}s"
                )
            with Image.open(BytesIO(completed.stdout)) as decoded:
                image = decoded.convert("RGB").copy()
            limited = _limit_pixels(image, max_pixels)
            if limited is not image:
                image.close()
            images.append(limited)
    except Exception:
        for image in images:
            image.close()
        raise
    return images, time.perf_counter() - started


def build_b1_run_spec(
    *, model_path: Path, contract: dict[str, Any],
    index_metadata: dict[str, dict[str, Any]],
    retrieval_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    config = contract["config"]
    implementation_paths = [
        Path(__file__),
        ROOT / "src/baselines/egopolice_b1/constants.py",
        ROOT / "src/baselines/egopolice_b1/indexer.py",
        ROOT / "src/baselines/egopolice_b1/retrieval.py",
        ROOT / "src/baselines/egopolice_b1/prepare_retrieval.py",
        ROOT / "src/baselines/egopolice_b0/core.py",
        ROOT / "src/baselines/egopolice_b0/model.py",
    ]
    payload = {
        "schema_version": B1_RUN_SCHEMA_VERSION,
        "baseline_id": B1_BASELINE_ID,
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "model_path": str(model_path.resolve()),
        "dtype": config["dtype"],
        "quantization_mode": config["quantization"]["mode"],
        "seed": config["seed"],
        "max_pixels": config["max_pixels"],
        "num_frames": 8,
        "generation": config["generation"],
        "prompt_version": config["prompt_version"],
        "retrieval_query": "raw_frozen_question_exactly",
        "retrieval_top_k": 8,
        "retrieval_allocation": "stable_top8_unique_no_nms_no_diversity_no_reranking",
        "model_input_order": "chronological_after_top8_allocation",
        "cradio_configuration": cradio_frozen_configuration(),
        "index_fingerprints": {
            video_id: row["index_fingerprint"]
            for video_id, row in sorted(index_metadata.items())
        },
        "retrieval_fingerprints": {
            question_id: row["retrieval_fingerprint"]
            for question_id, row in sorted(retrieval_records.items())
        },
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in implementation_paths
        },
    }
    return {**payload, "run_fingerprint": stable_sha256(payload)}


def qa_checkpoint_path(output_dir: Path, question_id: str) -> Path:
    if not question_id or any(char in question_id for char in "/\\"):
        raise ValueError(f"Unsafe question ID: {question_id!r}")
    return output_dir / "qa_checkpoints" / f"{question_id}.json"


def _same_float(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
    except (TypeError, ValueError):
        return False


def validate_b1_result(
    *, record: dict[str, Any], question: dict[str, Any], video: dict[str, Any],
    retrieval: dict[str, Any], run_spec: dict[str, Any],
) -> tuple[bool, str | None]:
    try:
        exact = {
            "schema_version": B1_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "formal_result": True,
            "baseline_id": B1_BASELINE_ID,
            "run_fingerprint": run_spec["run_fingerprint"],
            "question_id": question["question_id"],
            "source_video_id": question["video_id"],
            "question_duration_class": question["duration_class"],
            "source_video_duration_bin": video["source_video_duration_bin"],
            "raw_question_text": question["question"],
            "retrieval_query_text": question["question"],
            "retrieval_query_equals_raw_question": True,
            "answer_fields_used_for_retrieval": False,
            "gt_used_for_ranking": False,
            "question": question["question"],
            "options": question["options"],
            "ground_truth_index": question["ground_truth_index"],
            "ground_truth_text": question["ground_truth_text"],
            "gt_interval_sec": [float(value) for value in question["gt_interval_sec"]],
            "model_calls": 1,
            "model_facing_frames": 8,
            "frame_count": 8,
            "dtype": "bfloat16",
            "quantization_mode": "none",
            "retrieval_fingerprint": retrieval["retrieval_fingerprint"],
        }
        for name, expected in exact.items():
            if record.get(name) != expected:
                return False, f"field mismatch: {name}"
        prediction = record.get("prediction_index")
        if not isinstance(prediction, int) or not 0 <= prediction < 5:
            return False, "invalid prediction"
        if record.get("prediction_text") != question["options"][prediction]:
            return False, "prediction text mismatch"
        if record.get("correct") != (prediction == question["ground_truth_index"]):
            return False, "correctness mismatch"
        if not _same_float(record.get("full_source_video_duration_sec"), video["duration_sec"]):
            return False, "source duration mismatch"
        ranked = retrieval["top_8_ranked"]
        ranked_timestamps = [float(row["timestamp_sec"]) for row in ranked]
        chronological = sorted(ranked_timestamps)
        if record.get("selected_timestamps_ranked_sec") != ranked_timestamps:
            return False, "ranked timestamp mismatch"
        if record.get("selected_timestamps_sec") != chronological:
            return False, "chronological model-input timestamp mismatch"
        if record.get("selected_similarity_scores_ranked") != [
            float(row["cosine_similarity"]) for row in ranked
        ]:
            return False, "selected similarity mismatch"
        exposure = evaluate_gt_exposure(
            chronological, list(question["gt_interval_sec"])
        )
        for name in (
            "gt_interval_hit_at_8", "number_of_frames_inside_gt_interval",
            "nearest_sample_distance_to_gt_interval_seconds",
        ):
            if not _same_float(record.get(name), exposure[name]):
                return False, f"GT exposure mismatch: {name}"
        for name in (
            "query_text_embedding_time_sec", "similarity_search_time_sec",
            "frame_extraction_latency_sec", "preprocessing_latency_sec",
            "qwen_inference_latency_sec", "total_online_query_latency_sec",
            "peak_qwen_gpu_allocated_memory_bytes",
        ):
            value = record.get(name)
            if not isinstance(value, (int, float)) or value < 0:
                return False, f"invalid nonnegative metric: {name}"
        expected_online = sum(float(record[name]) for name in (
            "query_text_embedding_time_sec", "similarity_search_time_sec",
            "frame_extraction_latency_sec", "preprocessing_latency_sec",
            "qwen_inference_latency_sec",
        ))
        if not _same_float(record["total_online_query_latency_sec"], expected_online):
            return False, "online latency accounting mismatch"
        if not isinstance(record.get("actual_model_loading_mode"), str):
            return False, "missing loading mode"
        if record.get("offline_indexing_included_in_online_latency") is not False:
            return False, "offline/online separation flag missing"
        forbidden = record.get("forbidden_mechanism_calls")
        if not isinstance(forbidden, dict) or any(value != 0 for value in forbidden.values()):
            return False, "forbidden mechanism fixed-zero contract mismatch"
    except Exception as exc:
        return False, repr(exc)
    return True, None


def inspect_qa_checkpoints(
    *, output_dir: Path, contract: dict[str, Any],
    retrieval_records: dict[str, dict[str, Any]], run_spec: dict[str, Any],
    quarantine_invalid: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    valid = []
    pending = []
    invalid = []
    for question in contract["questions"]:
        path = qa_checkpoint_path(output_dir, question["question_id"])
        if not path.is_file():
            pending.append(question)
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            passed, reason = validate_b1_result(
                record=record, question=question,
                video=contract["videos"][question["video_id"]],
                retrieval=retrieval_records[question["question_id"]],
                run_spec=run_spec,
            )
        except Exception as exc:
            passed, reason = False, repr(exc)
        if passed:
            valid.append(record)
            continue
        invalid.append({"question_id": question["question_id"], "reason": str(reason)})
        if quarantine_invalid:
            suffix = f".invalid.{int(time.time())}.{uuid.uuid4().hex[:8]}"
            destination = path.with_name(path.name + suffix)
            os.replace(path, destination)
            atomic_write_json(
                output_dir / "qa_quarantine" / f"{question['question_id']}{suffix}.json",
                {"quarantined_at": utc_now(), "reason": reason,
                 "moved": str(destination)},
            )
        pending.append(question)
    return valid, pending, invalid


def _write_progress(
    *, output_dir: Path, run_spec: dict[str, Any], valid: list[dict[str, Any]],
    pending: list[dict[str, Any]], invalid: list[dict[str, str]],
) -> None:
    atomic_write_json(output_dir / "progress.json", {
        "schema_version": "egopolice-b1-progress-v1",
        "updated_at": utc_now(),
        "run_fingerprint": run_spec["run_fingerprint"],
        "completed_count": len(valid),
        "pending_count": len(pending),
        "completed_question_ids": [row["question_id"] for row in valid],
        "pending_question_ids": [row["question_id"] for row in pending],
        "invalid_checkpoints_quarantined": invalid,
    })


def _load_dependencies(
    *, contract: dict[str, Any], index_root: Path, output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    queries = build_raw_question_queries(contract["questions"])
    index_metadata: dict[str, dict[str, Any]] = {}
    for video in contract["videos"].values():
        passed, metadata, reason = validate_index(
            index_root=index_root, video=video,
            video_manifest_sha256=contract["video_manifest_sha256"],
        )
        if not passed:
            raise BaselineInputError(f"Missing/invalid B1 index {video['video_id']}: {reason}")
        index_metadata[video["video_id"]] = metadata or {}
    retrieval_records: dict[str, dict[str, Any]] = {}
    for query in queries:
        passed, record, reason = validate_retrieval_checkpoint(
            output_dir=output_dir, query=query,
            index_metadata=index_metadata[query.video_id],
            question_manifest_sha256=contract["question_manifest_sha256"],
        )
        if not passed:
            raise BaselineInputError(
                f"Missing/invalid B1 retrieval {query.question_id}: {reason}"
            )
        retrieval_records[query.question_id] = record or {}
    return index_metadata, retrieval_records, queries


def run_b1(
    *, data_root: Path, model_path: Path, index_root: Path, output_dir: Path,
    video_manifest_path: Path, question_manifest_path: Path,
    readiness_path: Path, config_path: Path, ffmpeg_path: str,
    dry_run: bool, execute_formal_inference: bool,
    qwen_factory: QwenFactory = Qwen25VL7BBaseline,
    environment_checker: EnvironmentChecker = preflight_environment,
    frame_extractor: FrameExtractor = extract_frames_at_timestamps,
    aggregate_when_complete: bool = True,
) -> dict[str, Any]:
    contract = load_formal_contract(
        data_root=data_root, video_manifest_path=video_manifest_path,
        question_manifest_path=question_manifest_path,
        readiness_path=readiness_path, config_path=config_path,
    )
    queries = build_raw_question_queries(contract["questions"])
    structural = {
        "formal_question_count": len(queries),
        "distinct_video_count": len({row.video_id for row in queries}),
        "query_equals_raw_question_count": sum(
            query.raw_question == question["question"]
            for query, question in zip(queries, contract["questions"])
        ),
        "answer_fields_in_retrieval_query_objects": False,
        "structurally_exactly_8_unique_frames_per_question": all(
            math.ceil(contract["videos"][query.video_id]["duration_sec"]) >= 8
            for query in queries
        ),
        "question_count_by_duration_class": dict(Counter(
            row["duration_class"] for row in contract["questions"]
        )),
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "dry_run": dry_run,
        "qwen_loaded": False,
        "qwen_calls": 0,
    }
    index_metadata = {}
    retrieval_records = {}
    dependency_errors = []
    try:
        index_metadata, retrieval_records, _ = _load_dependencies(
            contract=contract, index_root=index_root, output_dir=output_dir
        )
    except BaselineInputError as exc:
        dependency_errors.append(str(exc))
    structural["valid_index_count"] = len(index_metadata)
    structural["valid_retrieval_count"] = len(retrieval_records)
    structural["dependency_errors"] = dependency_errors
    if dry_run:
        return structural
    if not execute_formal_inference:
        raise BaselineInputError(
            "Refusing formal B1 inference without --execute-formal-inference"
        )
    if dependency_errors:
        raise BaselineInputError("; ".join(dependency_errors))

    run_spec = build_b1_run_spec(
        model_path=model_path, contract=contract,
        index_metadata=index_metadata, retrieval_records=retrieval_records,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != run_spec["run_fingerprint"]:
            raise BaselineInputError("B1 output directory belongs to a different run")
    else:
        atomic_write_json(run_config_path, {
            **run_spec, "created_at": utc_now(), "output_dir": str(output_dir),
        })
    valid, pending, invalid = inspect_qa_checkpoints(
        output_dir=output_dir, contract=contract,
        retrieval_records=retrieval_records, run_spec=run_spec,
        quarantine_invalid=True,
    )
    _write_progress(
        output_dir=output_dir, run_spec=run_spec, valid=valid,
        pending=pending, invalid=invalid,
    )
    if not pending:
        if aggregate_when_complete:
            from .aggregation import write_b1_final_artifacts
            write_b1_final_artifacts(
                output_dir=output_dir, index_root=index_root,
                contract=contract, run_spec=run_spec,
                retrieval_records=retrieval_records,
            )
        return {**structural, "already_complete": True, "completed_count": 98}

    environment = environment_checker(
        model_path, dtype=run_spec["dtype"],
        quantization_mode=run_spec["quantization_mode"],
    )
    if not environment.get("passed"):
        raise BaselineInputError(f"Qwen preflight failed: {environment.get('errors')}")
    if int(environment.get("free_vram_bytes") or 0) < QWEN_MINIMUM_FREE_VRAM_BYTES:
        raise BaselineInputError("Formal B1 Qwen phase requires at least 20 GiB free VRAM")
    atomic_write_json(output_dir / "qwen_environment.json", {
        "captured_at": utc_now(), "run_fingerprint": run_spec["run_fingerprint"],
        "environment": environment,
    })
    session = {
        "schema_version": "egopolice-b1-qwen-session-v1",
        "session_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "started_at": utc_now(), "run_fingerprint": run_spec["run_fingerprint"],
        "starting_completed_count": len(valid), "starting_pending_count": len(pending),
        "model_load_count": 0, "model_calls": 0,
        "completed_question_ids": [], "status": "starting",
    }
    session_path = output_dir / "qwen_sessions" / f"{session['session_id']}.json"
    atomic_write_json(session_path, session)
    model = None
    config = contract["config"]
    try:
        model = qwen_factory(
            model_path=model_path, max_pixels=int(config["max_pixels"]),
            seed=int(config["seed"]), dtype=str(config["dtype"]),
            quantization=dict(config["quantization"]),
        )
        session.update({
            "status": "running", "model_load_count": 1,
            "model_load_latency_sec": float(model.model_load_latency_sec),
            "actual_model_loading_mode": str(model.actual_model_loading_mode),
        })
        atomic_write_json(session_path, session)
        for question in list(pending):
            attempt_started = time.perf_counter()
            images: list[Image.Image] = []
            model_call_started = False
            try:
                retrieval = retrieval_records[question["question_id"]]
                ranked_timestamps = [
                    float(row["timestamp_sec"]) for row in retrieval["top_8_ranked"]
                ]
                chronological = sorted(ranked_timestamps)
                video = contract["videos"][question["video_id"]]
                images, extraction_sec = frame_extractor(
                    video_path=Path(video["path"]), timestamps_sec=chronological,
                    ffmpeg_path=ffmpeg_path, max_pixels=int(config["max_pixels"]),
                )
                if len(images) != 8:
                    raise BaselineInputError("B1 runtime did not extract exactly eight frames")
                prompt = build_mcq_prompt(
                    str(question["question"]), list(question["options"]), 8
                )
                model_call_started = True
                session["model_calls"] += 1
                inference = model.infer(
                    images=images, prompt=prompt, generation=dict(config["generation"])
                )
                prediction = int(inference["prediction_index"])
                # GT is evaluated only after retrieval, frame allocation and the
                # one final Qwen call have completed.
                exposure = evaluate_gt_exposure(
                    chronological, list(question["gt_interval_sec"])
                )
                component_online = (
                    float(retrieval["query_text_embedding_time_sec"])
                    + float(retrieval["similarity_search_time_sec"])
                    + float(extraction_sec)
                    + float(inference["preprocessing_latency_sec"])
                    + float(inference["inference_latency_sec"])
                )
                record = {
                    "schema_version": B1_RESULT_SCHEMA_VERSION,
                    "status": "completed", "formal_result": True,
                    "baseline_id": B1_BASELINE_ID,
                    "run_fingerprint": run_spec["run_fingerprint"],
                    "session_id": session["session_id"], "completed_at": utc_now(),
                    "question_id": question["question_id"],
                    "source_video_id": question["video_id"],
                    "source_video_path": video["path"],
                    "question_duration_class": question["duration_class"],
                    "source_video_duration_bin": video["source_video_duration_bin"],
                    "full_source_video_duration_sec": video["duration_sec"],
                    "raw_question_text": question["question"],
                    "retrieval_query_text": retrieval["retrieval_query_text"],
                    "retrieval_query_equals_raw_question": True,
                    "answer_fields_used_for_retrieval": False,
                    "gt_used_for_ranking": False,
                    "question": question["question"], "options": question["options"],
                    "prediction_index": prediction,
                    "prediction_text": question["options"][prediction],
                    "ground_truth_index": question["ground_truth_index"],
                    "ground_truth_text": question["ground_truth_text"],
                    "correct": prediction == int(question["ground_truth_index"]),
                    "raw_model_output": inference.get("raw_output"),
                    "gt_interval_sec": exposure["gt_interval_sec"],
                    "selected_timestamps_ranked_sec": ranked_timestamps,
                    "selected_timestamps_sec": chronological,
                    "selected_similarity_scores_ranked": [
                        float(row["cosine_similarity"])
                        for row in retrieval["top_8_ranked"]
                    ],
                    "top_8_ranked_metadata": retrieval["top_8_ranked"],
                    "model_input_order": "chronological",
                    "retrieval_fingerprint": retrieval["retrieval_fingerprint"],
                    "full_ranking_count": retrieval["full_ranking_count"],
                    "full_ranking_artifact_path": retrieval["ranking_artifact_path"],
                    "full_ranking_artifact_sha256": retrieval["ranking_artifact_sha256"],
                    **exposure,
                    "query_text_embedding_time_sec": retrieval["query_text_embedding_time_sec"],
                    "similarity_search_time_sec": retrieval["similarity_search_time_sec"],
                    "frame_extraction_latency_sec": float(extraction_sec),
                    "preprocessing_latency_sec": inference["preprocessing_latency_sec"],
                    "qwen_inference_latency_sec": inference["inference_latency_sec"],
                    "qa_stage_wall_latency_sec": time.perf_counter() - attempt_started,
                    "total_online_query_latency_sec": component_online,
                    "offline_indexing_included_in_online_latency": False,
                    "model_calls": 1, "model_facing_frames": 8, "frame_count": 8,
                    "text_input_tokens": inference.get("text_token_count"),
                    "visual_token_count": inference.get("visual_token_count"),
                    "total_input_tokens": inference.get("total_input_token_count"),
                    "output_tokens": inference.get("output_token_count"),
                    "peak_qwen_gpu_allocated_memory_bytes": int(inference["peak_gpu_memory_bytes"]),
                    "peak_text_encoder_vram_bytes": retrieval["peak_text_encoder_vram_bytes"],
                    "dtype": run_spec["dtype"],
                    "quantization_mode": run_spec["quantization_mode"],
                    "actual_model_loading_mode": str(model.actual_model_loading_mode),
                    "forbidden_mechanism_calls": {
                        "answer_option_retrieval": 0, "gt_guided_retrieval": 0,
                        "query_rewriting": 0, "captioning": 0, "dinov2": 0,
                        "segmentation": 0, "hierarchy": 0, "audio": 0,
                        "router": 0, "planner": 0, "fallback": 0,
                        "local_densification": 0, "temporal_nms": 0,
                        "diversity_reranking": 0,
                    },
                    "errors": [], "warnings": [], "oom": False,
                }
                passed, reason = validate_b1_result(
                    record=record, question=question, video=video,
                    retrieval=retrieval, run_spec=run_spec,
                )
                if not passed:
                    raise BaselineInputError(f"Refusing invalid B1 checkpoint: {reason}")
                atomic_write_json(
                    qa_checkpoint_path(output_dir, question["question_id"]), record
                )
                valid.append(record)
                pending = [
                    row for row in pending if row["question_id"] != question["question_id"]
                ]
                session["completed_question_ids"].append(question["question_id"])
                _write_progress(
                    output_dir=output_dir, run_spec=run_spec, valid=valid,
                    pending=pending, invalid=invalid,
                )
                atomic_write_json(session_path, session)
            except Exception as exc:
                failure = {
                    "schema_version": "egopolice-b1-qa-attempt-failure-v1",
                    "question_id": question["question_id"],
                    "session_id": session["session_id"], "failed_at": utc_now(),
                    "model_call_started": model_call_started,
                    "error_type": type(exc).__name__, "error_message": str(exc),
                    "attempt_latency_sec": time.perf_counter() - attempt_started,
                }
                atomic_write_json(
                    output_dir / "qa_failures" / question["question_id"]
                    / f"{session['session_id']}.json", failure,
                )
                session["status"] = "failed"; session["failure"] = failure
                atomic_write_json(session_path, session)
                raise
            finally:
                for image in images:
                    image.close()
        session["status"] = "completed"; session["ended_at"] = utc_now()
        atomic_write_json(session_path, session)
    except Exception as exc:
        if session.get("status") != "failed":
            session["status"] = "failed"
            session["ended_at"] = utc_now()
            session["failure"] = {
                "error_type": type(exc).__name__, "error_message": str(exc)
            }
            atomic_write_json(session_path, session)
        raise
    finally:
        if model is not None:
            del model
        gc.collect()

    valid, pending, invalid_after = inspect_qa_checkpoints(
        output_dir=output_dir, contract=contract,
        retrieval_records=retrieval_records, run_spec=run_spec,
        quarantine_invalid=False,
    )
    if len(valid) != 98 or pending or invalid_after:
        raise BaselineInputError("B1 ended without 98 valid QA checkpoints")
    _write_progress(
        output_dir=output_dir, run_spec=run_spec, valid=valid,
        pending=pending, invalid=[*invalid, *invalid_after],
    )
    if aggregate_when_complete:
        from .aggregation import write_b1_final_artifacts
        write_b1_final_artifacts(
            output_dir=output_dir, index_root=index_root, contract=contract,
            run_spec=run_spec, retrieval_records=retrieval_records,
        )
    return {
        **structural, "completed_count": 98, "pending_count": 0,
        "qwen_loaded": True, "qwen_load_count": 1,
        "qwen_calls": session["model_calls"], "session_path": str(session_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal EgoPolice B1 raw-question Top-8 QA")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cradio-device", default="cuda:0")
    parser.add_argument("--ffmpeg-path", default=os.environ.get("FFMPEG_PATH", "ffmpeg"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-formal-inference", action="store_true")
    parser.add_argument("--video-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation20_v1.json")
    parser.add_argument("--question-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation_questions_v1.json")
    parser.add_argument("--readiness", type=Path, default=ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines/egopolice_b0.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and not args.execute_formal_inference:
        raise BaselineInputError(
            "Refusing retrieval/model execution without --execute-formal-inference; "
            "use --dry-run for a no-model validation"
        )
    # Retrieval is a separately checkpointed online stage. The CLI prepares or
    # resumes it before Qwen, so one command is sufficient while the pure
    # run_b1 function remains injectable for dry-run and checkpoint tests.
    retrieval_result = prepare_retrieval(
        data_root=args.data_root, index_root=args.index_root,
        output_dir=args.output_dir, cache_root=args.cache_root,
        video_manifest_path=args.video_manifest,
        question_manifest_path=args.question_manifest,
        readiness_path=args.readiness, config_path=args.config,
        device_name=args.cradio_device, dry_run=args.dry_run,
    )
    result = run_b1(
        data_root=args.data_root, model_path=args.model_path,
        index_root=args.index_root, output_dir=args.output_dir,
        video_manifest_path=args.video_manifest,
        question_manifest_path=args.question_manifest,
        readiness_path=args.readiness, config_path=args.config,
        ffmpeg_path=args.ffmpeg_path, dry_run=args.dry_run,
        execute_formal_inference=args.execute_formal_inference,
    )
    print(json.dumps({"retrieval": retrieval_result, "qa": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
