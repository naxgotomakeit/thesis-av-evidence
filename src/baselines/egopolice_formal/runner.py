from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src.baselines.egopolice_b0.core import (
    BaselineInputError,
    build_mcq_prompt,
    evaluate_gt_exposure,
    extract_uniform_frames,
    uniform_timestamps,
)
from src.baselines.egopolice_b0.model import Qwen25VL7BBaseline, preflight_environment
from src.evaluation.egopolice_formal import source_video_duration_bin


ROOT = Path(__file__).resolve().parents[3]
CONDITION_BLIND = "blind"
CONDITION_UNIFORM8 = "uniform8"
FORMAL_CONDITIONS = (CONDITION_BLIND, CONDITION_UNIFORM8)
EXPECTED_VIDEO_MANIFEST_SHA256 = "0cb8d55962634d900d440f943a7d91ca5fe5473f3f5d0c096c826685acd3e1c5"
EXPECTED_QUESTION_MANIFEST_SHA256 = "fca734b9764ed132483ba3858db34243b50384ba2f1e115197c15762adcb23a5"
EXPECTED_QUESTION_COUNTS = Counter({"1s": 39, "10s": 38, "60s": 21})
MINIMUM_FREE_VRAM_BYTES = 20 * 1024**3
CHECKPOINT_SCHEMA_VERSION = "egopolice-formal-question-result-v1"
RUN_SCHEMA_VERSION = "egopolice-formal-run-v1"
SAFE_QUESTION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

ModelFactory = Callable[..., Any]
EnvironmentChecker = Callable[..., dict[str, Any]]
FrameExtractor = Callable[..., tuple[list[Any], list[float], float, float]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Durably replace one file without exposing a partial new version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def build_blind_prompt(question: str, options: list[str]) -> str:
    if len(options) != 5:
        raise BaselineInputError("Blind control requires exactly five options")
    option_lines = "\n".join(f"{index}. {text}" for index, text in enumerate(options))
    return (
        f"Question: {question}\n\nOptions:\n{option_lines}\n\n"
        "Return exactly one option index: 0, 1, 2, 3, or 4. "
        "Do not provide an explanation."
    )


def default_output_dir(condition: str) -> Path:
    if condition == CONDITION_BLIND:
        return ROOT / "outputs/experiments/Blind/formal_ablation98_v1"
    if condition == CONDITION_UNIFORM8:
        return ROOT / "outputs/experiments/B0/formal_ablation98_v1"
    raise BaselineInputError(f"Unsupported formal condition: {condition}")


def _validate_b0_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "num_frames": 8,
        "max_pixels": 262144,
        "seed": 0,
        "dtype": "bfloat16",
        "prompt_version": "egopolice-b0-mcq-v1",
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise BaselineInputError(
                f"Frozen B0 config mismatch for {name}: {config.get(name)!r}"
            )
    if config.get("quantization", {}).get("mode") != "none":
        raise BaselineInputError("Formal controls require no quantization")
    generation = config.get("generation", {})
    if generation != {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 8,
        "use_cache": True,
    }:
        raise BaselineInputError(f"Frozen deterministic decoding changed: {generation}")
    return config


def load_formal_contract(
    *, data_root: Path, video_manifest_path: Path, question_manifest_path: Path,
    readiness_path: Path, config_path: Path,
) -> dict[str, Any]:
    video_sha = sha256_file(video_manifest_path)
    question_sha = sha256_file(question_manifest_path)
    if video_sha != EXPECTED_VIDEO_MANIFEST_SHA256:
        raise BaselineInputError(f"Frozen video manifest SHA256 mismatch: {video_sha}")
    if question_sha != EXPECTED_QUESTION_MANIFEST_SHA256:
        raise BaselineInputError(f"Frozen question manifest SHA256 mismatch: {question_sha}")

    video_payload = json.loads(video_manifest_path.read_text(encoding="utf-8"))
    question_payload = json.loads(question_manifest_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    videos = list(video_payload.get("videos") or [])
    questions = list(question_payload.get("questions") or [])
    if len(videos) != 20 or len({row.get("video_id") for row in videos}) != 20:
        raise BaselineInputError("Formal video manifest must contain 20 unique videos")
    if len(questions) != 98 or len({row.get("question_id") for row in questions}) != 98:
        raise BaselineInputError("Formal question manifest must contain 98 unique questions")
    counts = Counter(str(row.get("duration_class")) for row in questions)
    if counts != EXPECTED_QUESTION_COUNTS:
        raise BaselineInputError(f"Unexpected formal question distribution: {counts}")
    if (
        readiness.get("video_manifest_sha256") != video_sha
        or readiness.get("question_manifest_sha256") != question_sha
        or readiness.get("summary", {}).get("ready") != 20
        or readiness.get("summary", {}).get("failed") != 0
        or readiness.get("summary", {}).get("missing") != 0
    ):
        raise BaselineInputError("Strict formal readiness artifact is not 20/20 for these hashes")

    manifest_videos = {str(row["video_id"]): row for row in videos}
    readiness_videos = {
        str(row["manifest_video_id"]): row for row in readiness.get("videos", [])
    }
    if set(manifest_videos) != set(readiness_videos):
        raise BaselineInputError("Readiness/video-manifest ID mismatch")
    video_contract: dict[str, dict[str, Any]] = {}
    for video_id, manifest_row in manifest_videos.items():
        audit = readiness_videos[video_id]
        path = data_root.joinpath(*Path(str(manifest_row["relative_video_path"])).parts)
        duration = float(audit.get("duration_sec") or 0)
        duration_bin = str(audit.get("source_video_duration_bin") or "")
        if (
            audit.get("status") != "ready"
            or audit.get("readable_valid") is not True
            or audit.get("full_packet_scan_passed") is not True
            or audit.get("mapping_unambiguous") is not True
            or audit.get("error") is not None
            or not path.is_file()
            or path.stat().st_size <= 0
            or duration <= 0
            or duration_bin != source_video_duration_bin(duration)
        ):
            raise BaselineInputError(f"Formal video failed strict runtime readiness: {video_id}")
        video_contract[video_id] = {
            "video_id": video_id,
            "path": str(path),
            "duration_sec": duration,
            "source_video_duration_bin": duration_bin,
            "file_size_bytes": int(path.stat().st_size),
        }

    for question in questions:
        question_id = str(question.get("question_id") or "")
        video_id = str(question.get("video_id") or "")
        options = question.get("options")
        ground_truth = question.get("ground_truth_index")
        interval = question.get("gt_interval_sec")
        if not SAFE_QUESTION_ID.fullmatch(question_id):
            raise BaselineInputError(f"Unsafe or empty question ID: {question_id!r}")
        if video_id not in video_contract:
            raise BaselineInputError(f"Question maps outside formal videos: {question_id}")
        if not isinstance(options, list) or len(options) != 5:
            raise BaselineInputError(f"Question does not have five options: {question_id}")
        if not isinstance(ground_truth, int) or not 0 <= ground_truth < 5:
            raise BaselineInputError(f"Invalid GT answer: {question_id}")
        if question.get("ground_truth_text") != options[ground_truth]:
            raise BaselineInputError(f"GT answer text mismatch: {question_id}")
        if not isinstance(interval, list) or len(interval) != 2:
            raise BaselineInputError(f"Invalid GT interval: {question_id}")
        start, end = (float(value) for value in interval)
        if start < 0 or end <= start or end > video_contract[video_id]["duration_sec"]:
            raise BaselineInputError(f"GT interval outside source video: {question_id}")

    config = _validate_b0_config(config_path)
    stable_videos = [
        {
            "video_id": row["video_id"],
            "duration_sec": row["duration_sec"],
            "source_video_duration_bin": row["source_video_duration_bin"],
            "file_size_bytes": row["file_size_bytes"],
        }
        for row in video_contract.values()
    ]
    stable_videos.sort(key=lambda row: row["video_id"])
    return {
        "video_manifest_path": str(video_manifest_path),
        "video_manifest_sha256": video_sha,
        "question_manifest_path": str(question_manifest_path),
        "question_manifest_sha256": question_sha,
        "readiness_path": str(readiness_path),
        "config_path": str(config_path),
        "config": config,
        "questions": questions,
        "videos": video_contract,
        "stable_video_contract": stable_videos,
    }


def build_run_spec(
    *, condition: str, model_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    if condition not in FORMAL_CONDITIONS:
        raise BaselineInputError(f"Unsupported formal condition: {condition}")
    config = contract["config"]
    implementation_paths = [
        Path(__file__),
        ROOT / "src/baselines/egopolice_b0/core.py",
        ROOT / "src/baselines/egopolice_b0/model.py",
    ]
    run_contract = {
        "schema_version": RUN_SCHEMA_VERSION,
        "condition": condition,
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "model_path": str(model_path.resolve()),
        "dtype": config["dtype"],
        "quantization_mode": config["quantization"]["mode"],
        "seed": config["seed"],
        "max_pixels": config["max_pixels"],
        "num_frames": 0 if condition == CONDITION_BLIND else config["num_frames"],
        "generation": config["generation"],
        "prompt_version": (
            "egopolice-blind-mcq-v1"
            if condition == CONDITION_BLIND else config["prompt_version"]
        ),
        "stable_video_contract": contract["stable_video_contract"],
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in implementation_paths
        },
    }
    return {**run_contract, "run_fingerprint": stable_sha256(run_contract)}


def checkpoint_path(output_dir: Path, question_id: str) -> Path:
    if not SAFE_QUESTION_ID.fullmatch(question_id):
        raise BaselineInputError(f"Unsafe question ID for checkpoint: {question_id!r}")
    return output_dir / "checkpoints" / f"{question_id}.json"


def _same_float(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def validate_completed_record(
    record: dict[str, Any], *, question: dict[str, Any], video: dict[str, Any],
    run_spec: dict[str, Any],
) -> tuple[bool, str | None]:
    try:
        prediction = record.get("prediction_index")
        expected_pairs = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "completed",
            "condition": run_spec["condition"],
            "run_fingerprint": run_spec["run_fingerprint"],
            "question_id": question["question_id"],
            "source_video_id": question["video_id"],
            "question": question["question"],
            "options": question["options"],
            "question_duration_class": question["duration_class"],
            "source_video_duration_bin": video["source_video_duration_bin"],
            "gt_interval_sec": [float(value) for value in question["gt_interval_sec"]],
            "ground_truth_index": question["ground_truth_index"],
            "ground_truth_text": question["ground_truth_text"],
            "model_calls": 1,
            "dtype": "bfloat16",
            "quantization_mode": "none",
        }
        for name, expected in expected_pairs.items():
            if record.get(name) != expected:
                return False, f"field mismatch: {name}"
        if not isinstance(prediction, int) or not 0 <= prediction < 5:
            return False, "invalid prediction index"
        if record.get("prediction_text") != question["options"][prediction]:
            return False, "prediction text mismatch"
        if record.get("correct") is not (prediction == question["ground_truth_index"]):
            return False, "correctness mismatch"
        if not _same_float(record.get("full_source_video_duration_sec"), video["duration_sec"]):
            return False, "source duration mismatch"
        for name in (
            "frame_extraction_latency_sec", "preprocessing_latency_sec",
            "qwen_inference_latency_sec", "total_per_query_latency_sec",
            "peak_gpu_allocated_memory_bytes",
        ):
            value = record.get(name)
            if not isinstance(value, (int, float)) or value < 0:
                return False, f"invalid nonnegative metric: {name}"
        for name in ("text_input_tokens", "total_input_tokens", "output_tokens"):
            value = record.get(name)
            if value is not None and (not isinstance(value, int) or value < 0):
                return False, f"invalid token metric: {name}"
        if not isinstance(record.get("actual_model_loading_mode"), str):
            return False, "missing actual model loading mode"
        timestamps = record.get("selected_timestamps_sec")
        if run_spec["condition"] == CONDITION_BLIND:
            if timestamps != [] or record.get("model_facing_frames") != 0:
                return False, "Blind visual-input contract mismatch"
            if record.get("frame_extraction_latency_sec") != 0.0:
                return False, "Blind extraction latency must be zero"
            if any(record.get(name) is not None for name in (
                "gt_interval_hit_at_8", "number_of_frames_inside_gt_interval",
                "nearest_sample_distance_to_gt_interval_seconds",
            )):
                return False, "Blind GT visual diagnostics must be N/A"
            if record.get("visual_token_count") != 0:
                return False, "Blind visual tokens must be zero"
        else:
            expected_timestamps = uniform_timestamps(video["duration_sec"], 8)
            if (
                not isinstance(timestamps, list) or len(timestamps) != 8
                or not all(_same_float(a, b) for a, b in zip(timestamps, expected_timestamps))
                or record.get("model_facing_frames") != 8
            ):
                return False, "Uniform-8 timestamp/frame contract mismatch"
            exposure = evaluate_gt_exposure(
                expected_timestamps, list(question["gt_interval_sec"])
            )
            for name in (
                "gt_interval_hit_at_8", "number_of_frames_inside_gt_interval",
                "nearest_sample_distance_to_gt_interval_seconds",
            ):
                if not _same_float(record.get(name), exposure[name]):
                    return False, f"GT exposure mismatch: {name}"
        retrieval = record.get("retrieval_metrics")
        if not isinstance(retrieval, dict) or retrieval.get("applicable") is not False:
            return False, "retrieval N/A contract missing"
        if any(retrieval.get(name) != 0 for name in (
            "retrieval_calls", "embedding_computations", "similarity_computations",
            "reusable_index_size_bytes",
        )):
            return False, "retrieval fixed-zero contract mismatch"
    except Exception as exc:
        return False, f"checkpoint validation exception: {exc!r}"
    return True, None


def _quarantine_invalid_checkpoint(path: Path, reason: str) -> Path:
    destination = path.with_name(
        f"{path.stem}.invalid.{int(time.time())}.{uuid.uuid4().hex[:8]}.json"
    )
    os.replace(path, destination)
    atomic_write_json(
        destination.with_suffix(".reason.json"),
        {"quarantined_at": utc_now(), "original_path": str(path), "reason": reason},
    )
    return destination


def inspect_checkpoints(
    *, output_dir: Path, contract: dict[str, Any], run_spec: dict[str, Any],
    quarantine_invalid: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    valid: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    videos = contract["videos"]
    for question in contract["questions"]:
        path = checkpoint_path(output_dir, str(question["question_id"]))
        if not path.is_file():
            pending.append(question)
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            record = {}
            reason = f"unreadable checkpoint: {exc!r}"
        else:
            _, reason = validate_completed_record(
                record, question=question, video=videos[question["video_id"]],
                run_spec=run_spec,
            )
        if reason is None:
            valid.append(record)
            continue
        invalid.append({"question_id": question["question_id"], "reason": reason})
        if quarantine_invalid:
            _quarantine_invalid_checkpoint(path, reason)
        pending.append(question)
    return valid, pending, invalid


def _write_progress(
    output_dir: Path, *, run_spec: dict[str, Any], valid: list[dict[str, Any]],
    pending: list[dict[str, Any]], invalid: list[dict[str, str]],
) -> None:
    atomic_write_json(output_dir / "progress.json", {
        "schema_version": "egopolice-formal-progress-v1",
        "updated_at": utc_now(),
        "condition": run_spec["condition"],
        "run_fingerprint": run_spec["run_fingerprint"],
        "completed_count": len(valid),
        "pending_count": len(pending),
        "total_count": len(valid) + len(pending),
        "completed_question_ids": [row["question_id"] for row in valid],
        "pending_question_ids": [row["question_id"] for row in pending],
        "invalid_checkpoints_quarantined": invalid,
    })


def _ensure_run_config(output_dir: Path, payload: dict[str, Any]) -> None:
    path = output_dir / "run_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != payload["run_fingerprint"]:
            raise BaselineInputError(
                f"Output directory belongs to a different formal run: {path}"
            )
        return
    atomic_write_json(path, {**payload, "created_at": utc_now()})


def _make_record(
    *, question: dict[str, Any], video: dict[str, Any], run_spec: dict[str, Any],
    inference: dict[str, Any], prediction: int, timestamps: list[float],
    frame_count: int, extraction_latency: float, total_latency: float,
    session_id: str,
) -> dict[str, Any]:
    condition = run_spec["condition"]
    if condition == CONDITION_UNIFORM8:
        exposure = evaluate_gt_exposure(timestamps, list(question["gt_interval_sec"]))
        visual_applicability = "applicable"
    else:
        exposure = {
            "gt_interval_sec": [float(value) for value in question["gt_interval_sec"]],
            "gt_interval_hit_at_8": None,
            "number_of_frames_inside_gt_interval": None,
            "nearest_sample_distance_to_gt_interval_seconds": None,
        }
        visual_applicability = "not_applicable_no_visual_input"
    options = list(question["options"])
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "completed",
        "formal_result": True,
        "condition": condition,
        "run_fingerprint": run_spec["run_fingerprint"],
        "session_id": session_id,
        "completed_at": utc_now(),
        "question_id": question["question_id"],
        "source_video_id": question["video_id"],
        "source_video_path": video["path"],
        "question_duration_class": question["duration_class"],
        "source_video_duration_bin": video["source_video_duration_bin"],
        "full_source_video_duration_sec": video["duration_sec"],
        "question": question["question"],
        "options": options,
        "prediction_index": prediction,
        "prediction_text": options[prediction],
        "ground_truth_index": question["ground_truth_index"],
        "ground_truth_text": question["ground_truth_text"],
        "correct": prediction == int(question["ground_truth_index"]),
        "raw_model_output": inference.get("raw_output"),
        "gt_interval_sec": exposure["gt_interval_sec"],
        "selected_timestamps_sec": timestamps,
        "sampled_frame_timestamps_sec": timestamps,
        "sampling_rule": (
            "not_applicable_no_visual_input"
            if condition == CONDITION_BLIND
            else "full_duration_8_equal_bins_exact_midpoint"
        ),
        "sampling_question_independent": condition == CONDITION_UNIFORM8,
        "visual_metrics_applicability": visual_applicability,
        "gt_interval_hit_at_8": exposure["gt_interval_hit_at_8"],
        "number_of_frames_inside_gt_interval": exposure["number_of_frames_inside_gt_interval"],
        "nearest_sample_distance_to_gt_interval_seconds": exposure["nearest_sample_distance_to_gt_interval_seconds"],
        "frame_extraction_latency_sec": extraction_latency,
        "preprocessing_latency_sec": float(inference["preprocessing_latency_sec"]),
        "qwen_inference_latency_sec": float(inference["inference_latency_sec"]),
        "total_per_query_latency_sec": total_latency,
        "model_calls": 1,
        "model_facing_frames": frame_count,
        "frame_count": frame_count,
        "text_input_tokens": inference.get("text_token_count"),
        "visual_token_count": int(inference.get("visual_token_count") or 0),
        "visual_tokens_estimate": int(inference.get("visual_token_count") or 0),
        "visual_tokens_are_estimated": condition == CONDITION_UNIFORM8,
        "total_input_tokens": inference.get("total_input_token_count"),
        "output_tokens": inference.get("output_token_count"),
        "peak_gpu_allocated_memory_bytes": int(inference["peak_gpu_memory_bytes"]),
        "dtype": run_spec["dtype"],
        "quantization_mode": run_spec["quantization_mode"],
        "actual_model_loading_mode": inference["actual_model_loading_mode"],
        "model_checkpoint": run_spec["model_path"],
        "model_calls_contract": "exactly_one",
        "retrieval_metrics": {
            "applicable": False,
            "reason": "Blind/Uniform-8 controls contain no retrieval",
            "retrieval_calls": 0,
            "embedding_computations": 0,
            "similarity_computations": 0,
            "reusable_index_size_bytes": 0,
        },
        "fixed_zero_costs": {
            "offline_reusable_preprocessing": 0,
            "reusable_index_size_bytes": 0,
            "embedding_computations": 0,
            "similarity_computations": 0,
            "caption_generation_calls": 0,
            "audio_calls": 0,
            "router_calls": 0,
            "planner_calls": 0,
        },
        "errors": [],
        "warnings": [],
        "oom": False,
    }


def execute_formal(
    *, condition: str, data_root: Path, model_path: Path, output_dir: Path,
    video_manifest_path: Path = ROOT / "config/data/egopolice_ablation20_v1.json",
    question_manifest_path: Path = ROOT / "config/data/egopolice_ablation_questions_v1.json",
    readiness_path: Path = ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json",
    config_path: Path = ROOT / "config/baselines/egopolice_b0.json",
    ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe",
    dry_run: bool = False, model_factory: ModelFactory = Qwen25VL7BBaseline,
    environment_checker: EnvironmentChecker = preflight_environment,
    frame_extractor: FrameExtractor = extract_uniform_frames,
    aggregate_when_complete: bool = True,
) -> dict[str, Any]:
    contract = load_formal_contract(
        data_root=data_root, video_manifest_path=video_manifest_path,
        question_manifest_path=question_manifest_path, readiness_path=readiness_path,
        config_path=config_path,
    )
    run_spec = build_run_spec(
        condition=condition, model_path=model_path, contract=contract
    )
    valid, pending, invalid = inspect_checkpoints(
        output_dir=output_dir, contract=contract, run_spec=run_spec,
        quarantine_invalid=not dry_run,
    )
    enumeration = {
        "condition": condition,
        "formal_question_count": len(contract["questions"]),
        "distinct_video_count": len(contract["videos"]),
        "question_count_by_duration_class": dict(
            Counter(row["duration_class"] for row in contract["questions"])
        ),
        "valid_completed_count": len(valid),
        "pending_count": len(pending),
        "invalid_checkpoint_count": len(invalid),
        "question_ids": [row["question_id"] for row in contract["questions"]],
        "run_fingerprint": run_spec["run_fingerprint"],
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "dry_run": dry_run,
        "model_loaded": False,
        "model_calls": 0,
    }
    if dry_run:
        return enumeration

    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_run_config(output_dir, {
        **run_spec,
        "video_manifest_path": contract["video_manifest_path"],
        "question_manifest_path": contract["question_manifest_path"],
        "readiness_path": contract["readiness_path"],
        "output_dir": str(output_dir),
        "question_count": len(contract["questions"]),
    })
    _write_progress(
        output_dir, run_spec=run_spec, valid=valid, pending=pending, invalid=invalid
    )
    if not pending:
        if aggregate_when_complete:
            from .aggregation import write_final_artifacts
            write_final_artifacts(
                condition=condition, output_dir=output_dir, contract=contract,
                run_spec=run_spec,
            )
        return {**enumeration, "already_complete": True, "model_loaded": False}

    environment = environment_checker(
        model_path,
        dtype=str(run_spec["dtype"]),
        quantization_mode=str(run_spec["quantization_mode"]),
    )
    if not environment.get("passed"):
        raise BaselineInputError(f"Formal model preflight failed: {environment.get('errors')}")
    if int(environment.get("free_vram_bytes") or 0) < MINIMUM_FREE_VRAM_BYTES:
        raise BaselineInputError(
            "Formal BF16 run requires at least 20 GiB free VRAM; "
            f"observed {environment.get('free_vram_bytes')} bytes"
        )
    atomic_write_json(output_dir / "environment.json", {
        "captured_at": utc_now(),
        "condition": condition,
        "run_fingerprint": run_spec["run_fingerprint"],
        "minimum_free_vram_bytes": MINIMUM_FREE_VRAM_BYTES,
        "environment": environment,
    })

    session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    session_path = output_dir / "sessions" / f"{session_id}.json"
    session = {
        "schema_version": "egopolice-formal-session-v1",
        "session_id": session_id,
        "condition": condition,
        "run_fingerprint": run_spec["run_fingerprint"],
        "started_at": utc_now(),
        "starting_completed_count": len(valid),
        "starting_pending_count": len(pending),
        "environment": environment,
        "model_load_count": 0,
        "model_calls": 0,
        "completed_this_session": [],
        "status": "starting",
    }
    atomic_write_json(session_path, session)
    model = None
    config = contract["config"]
    try:
        try:
            model = model_factory(
                model_path=model_path,
                max_pixels=int(config["max_pixels"]),
                seed=int(config["seed"]),
                dtype=str(config["dtype"]),
                quantization=dict(config["quantization"]),
            )
        except Exception as exc:
            session["status"] = "model_load_failed"
            session["ended_at"] = utc_now()
            session["failure"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            atomic_write_json(session_path, session)
            raise
        session["model_load_count"] = 1
        session["model_load_latency_sec"] = float(model.model_load_latency_sec)
        session["actual_model_loading_mode"] = str(model.actual_model_loading_mode)
        session["status"] = "running"
        atomic_write_json(session_path, session)

        for question in list(pending):
            query_started = time.perf_counter()
            images: list[Any] = []
            model_call_started = False
            try:
                video = contract["videos"][question["video_id"]]
                if condition == CONDITION_UNIFORM8:
                    images, timestamps, decoded_duration, extraction_latency = frame_extractor(
                        video_path=Path(video["path"]), ffmpeg_path=ffmpeg_path,
                        ffprobe_path=ffprobe_path, num_frames=8,
                        max_pixels=int(config["max_pixels"]),
                    )
                    expected_timestamps = uniform_timestamps(video["duration_sec"], 8)
                    if (
                        len(images) != 8
                        or not _same_float(decoded_duration, video["duration_sec"])
                        or len(timestamps) != 8
                        or not all(_same_float(a, b) for a, b in zip(timestamps, expected_timestamps))
                    ):
                        raise BaselineInputError(
                            f"Runtime Uniform-8 sampling contract failed: {question['question_id']}"
                        )
                    prompt = build_mcq_prompt(
                        str(question["question"]), list(question["options"]), 8
                    )
                else:
                    timestamps = []
                    extraction_latency = 0.0
                    prompt = build_blind_prompt(
                        str(question["question"]), list(question["options"])
                    )
                model_call_started = True
                session["model_calls"] += 1
                inference = model.infer(
                    images=images, prompt=prompt, generation=dict(config["generation"])
                )
                prediction = int(inference["prediction_index"])
                inference["actual_model_loading_mode"] = str(model.actual_model_loading_mode)
                # GT fields are evaluated only after the model call. They never
                # influence Blind input or Uniform-8 timestamp selection.
                record = _make_record(
                    question=question, video=video, run_spec=run_spec,
                    inference=inference, prediction=prediction,
                    timestamps=list(timestamps), frame_count=len(images),
                    extraction_latency=float(extraction_latency),
                    total_latency=time.perf_counter() - query_started,
                    session_id=session_id,
                )
                valid_record, reason = validate_completed_record(
                    record, question=question, video=video, run_spec=run_spec
                )
                if not valid_record:
                    raise BaselineInputError(f"Refusing invalid result checkpoint: {reason}")
                atomic_write_json(
                    checkpoint_path(output_dir, str(question["question_id"])), record
                )
                valid.append(record)
                pending = [
                    row for row in pending
                    if row["question_id"] != question["question_id"]
                ]
                session["completed_this_session"].append(question["question_id"])
                _write_progress(
                    output_dir, run_spec=run_spec, valid=valid,
                    pending=pending, invalid=invalid,
                )
                atomic_write_json(session_path, session)
            except Exception as exc:
                failure = {
                    "schema_version": "egopolice-formal-attempt-failure-v1",
                    "status": "technical_failure",
                    "condition": condition,
                    "run_fingerprint": run_spec["run_fingerprint"],
                    "session_id": session_id,
                    "question_id": question["question_id"],
                    "failed_at": utc_now(),
                    "model_call_started": model_call_started,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "total_attempt_latency_sec": time.perf_counter() - query_started,
                }
                atomic_write_json(
                    output_dir / "failures" / str(question["question_id"])
                    / f"{session_id}.json",
                    failure,
                )
                session["status"] = "failed"
                session["failure"] = failure
                atomic_write_json(session_path, session)
                raise
            finally:
                for image in images:
                    close = getattr(image, "close", None)
                    if callable(close):
                        close()

        session["status"] = "completed"
        session["ended_at"] = utc_now()
        atomic_write_json(session_path, session)
    finally:
        if model is not None:
            del model
        gc.collect()

    valid, pending, invalid_after = inspect_checkpoints(
        output_dir=output_dir, contract=contract, run_spec=run_spec,
        quarantine_invalid=False,
    )
    _write_progress(
        output_dir, run_spec=run_spec, valid=valid, pending=pending,
        invalid=[*invalid, *invalid_after],
    )
    if pending or invalid_after or len(valid) != 98:
        raise BaselineInputError(
            f"Formal run ended without 98 valid results: valid={len(valid)}, pending={len(pending)}"
        )
    if aggregate_when_complete:
        from .aggregation import write_final_artifacts
        write_final_artifacts(
            condition=condition, output_dir=output_dir, contract=contract,
            run_spec=run_spec,
        )
    return {
        **enumeration,
        "valid_completed_count": len(valid),
        "pending_count": 0,
        "model_loaded": True,
        "model_load_count": 1,
        "model_calls": session["model_calls"],
        "session_id": session_id,
        "output_dir": str(output_dir),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a production-safe frozen EgoPolice formal control"
    )
    parser.add_argument("--condition", choices=FORMAL_CONDITIONS, required=True)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--model-path", type=Path, default=os.environ.get("MODEL_PATH"))
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
    parser.add_argument("--ffmpeg-path", default=os.environ.get("FFMPEG_PATH", "ffmpeg"))
    parser.add_argument("--ffprobe-path", default=os.environ.get("FFPROBE_PATH", "ffprobe"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data_root is None or args.model_path is None:
        raise BaselineInputError(
            "Set --data-root/--model-path or DATA_ROOT/MODEL_PATH"
        )
    output_dir = args.output_dir or default_output_dir(args.condition)
    result = execute_formal(
        condition=args.condition, data_root=args.data_root,
        model_path=args.model_path, output_dir=output_dir,
        video_manifest_path=args.video_manifest,
        question_manifest_path=args.question_manifest,
        readiness_path=args.readiness, config_path=args.config,
        ffmpeg_path=args.ffmpeg_path, ffprobe_path=args.ffprobe_path,
        dry_run=args.dry_run,
    )
    printable = {
        key: value for key, value in result.items()
        if key != "question_ids"
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
