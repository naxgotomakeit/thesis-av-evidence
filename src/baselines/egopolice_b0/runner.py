from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from .core import (
    BaselineInputError,
    build_mcq_prompt,
    extract_uniform_frames,
    load_mcq_cases,
    resolve_video_path,
)
from .model import Qwen25VL7BBaseline, preflight_environment


ROOT = Path(__file__).resolve().parents[3]


def _path_value(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def resolve_runtime_config(
    config: dict[str, Any], *, overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None, repo_root: Path = ROOT,
) -> tuple[dict[str, Any], Path]:
    """Resolve portable paths without changing any B0 inference setting.

    Precedence is CLI override, environment variable, then frozen config.
    Derived metadata/video/model paths are constructed only after their roots
    have been resolved, so the same config works on Windows and POSIX.
    """
    values = dict(config)
    cli = overrides or {}
    env = dict(os.environ if environ is None else environ)

    def choose(cli_name: str, env_name: str, config_name: str) -> Any:
        return cli.get(cli_name) or env.get(env_name) or values.get(config_name)

    data_root = _path_value(choose("data_root", "DATA_ROOT", "data_root"))
    model_root = _path_value(choose("model_root", "MODEL_ROOT", "model_root"))
    output_root = _path_value(choose("output_root", "OUTPUT_ROOT", "output_root"))
    model_path = _path_value(choose("model_path", "MODEL_PATH", "model_path"))
    qa_path = _path_value(choose("qa_path", "QA_PATH", "qa_path"))
    video_root = _path_value(choose("video_root", "VIDEO_ROOT", "video_root"))
    subset_manifest = _path_value(choose("subset_manifest", "EGOPOLICE_SUBSET_MANIFEST", "subset_manifest"))
    output_path = _path_value(cli.get("output"))

    if model_path is None and model_root is not None:
        model_path = model_root / str(values.get("model_directory_name", "Qwen2.5-VL-7B-Instruct"))
    if qa_path is None and data_root is not None:
        qa_path = data_root / str(values.get("qa_metadata_file", "mcq_60s.json"))
    if video_root is None and data_root is not None:
        video_root = data_root / str(values.get("video_subdir", "videos"))
    if output_path is None:
        root = output_root or (repo_root / "outputs")
        output_path = root / str(values.get("output_relative_path", "baselines/egopolice_b0/results.jsonl"))
    if subset_manifest is not None and not subset_manifest.is_absolute():
        subset_manifest = repo_root / subset_manifest

    values["data_root"] = str(data_root) if data_root else None
    values["model_root"] = str(model_root) if model_root else None
    values["output_root"] = str(output_root) if output_root else None
    values["model_path"] = str(model_path) if model_path else None
    values["qa_path"] = str(qa_path) if qa_path else None
    values["video_root"] = str(video_root) if video_root else None
    values["subset_manifest"] = str(subset_manifest) if subset_manifest else None
    values["ffmpeg_path"] = str(choose("ffmpeg_path", "FFMPEG_PATH", "ffmpeg_path") or "ffmpeg")
    values["ffprobe_path"] = str(choose("ffprobe_path", "FFPROBE_PATH", "ffprobe_path") or "ffprobe")
    for name in ("num_frames", "max_pixels"):
        if cli.get(name) is not None:
            values[name] = int(cli[name])
    values["dtype"] = str(choose("dtype", "MODEL_DTYPE", "dtype") or "bfloat16")
    quantization = dict(values.get("quantization") or {})
    quantization["mode"] = str(
        cli.get("quantization_mode")
        or env.get("QUANTIZATION_MODE")
        or quantization.get("mode", "none")
    )
    values["quantization"] = quantization
    return values, output_path


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _error_result(
    case: dict[str, Any], *, video_path: Path, config: dict[str, Any],
    errors: list[str], error_type: str, actual_model_loading_mode: str | None = None,
) -> dict[str, Any]:
    return {
        "baseline_id": config["baseline_id"],
        "video_id": case["video_id"],
        "question_id": case["question_id"],
        "question": case["question"],
        "options": case["options"],
        "video_path": str(video_path),
        "ground_truth_index": case["ground_truth_index"],
        "ground_truth_text": case["ground_truth_text"],
        "prediction_index": None,
        "prediction_text": None,
        "correct": None,
        "sampled_frame_timestamps_sec": [],
        "frame_count": 0,
        "frame_extraction_latency_sec": 0.0,
        "model_inference_latency_sec": 0.0,
        "total_per_question_latency_sec": 0.0,
        "model_calls": 0,
        "text_token_count": None,
        "visual_token_count": None,
        "total_input_token_count": None,
        "output_token_count": None,
        "peak_gpu_memory_bytes": 0,
        "model_checkpoint": config["model_path"],
        "inference_settings": {
            "num_frames": config["num_frames"],
            "max_pixels": config["max_pixels"],
            "dtype": config["dtype"],
            "quantization_mode": config["quantization"]["mode"],
            "actual_model_loading_mode": actual_model_loading_mode,
            "generation": config["generation"],
            "prompt_version": config["prompt_version"],
        },
        "ignored_annotation_interval": case["ignored_annotation_interval"],
        "status": "technical_failure",
        "error_type": error_type,
        "error_message": ";".join(errors),
        "oom": False,
        "external_api_calls": 0,
    }


def _allowed_video_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("videos")
    if not isinstance(rows, list) or not rows:
        raise BaselineInputError(f"Invalid frozen subset manifest: {path}")
    return {str(row["video_id"]).removesuffix(".mp4") for row in rows}


def run(config: dict[str, Any], *, output_path: Path, limit: int, case_id: str | None) -> dict[str, Any]:
    if limit < 1 or limit > 5:
        raise BaselineInputError("B0 safety limit must be between 1 and 5")
    cases = load_mcq_cases(Path(config["qa_path"]))
    subset_path = Path(config["subset_manifest"]) if config.get("subset_manifest") else None
    allowed = _allowed_video_ids(subset_path)
    if allowed is not None:
        cases = [case for case in cases if case["video_id"] in allowed]
    if case_id:
        cases = [case for case in cases if case["question_id"] == case_id]
        if not cases:
            raise BaselineInputError(f"Question not found: {case_id}")
    selected = cases[:limit]
    model_path = Path(config["model_path"])
    video_root = Path(config["video_root"])
    environment = preflight_environment(
        model_path,
        dtype=str(config["dtype"]),
        quantization_mode=str(config["quantization"]["mode"]),
    )
    case_paths = [(case, resolve_video_path(video_root, case["video_relative_path"])) for case in selected]
    missing_videos = [str(path) for _, path in case_paths if not path.is_file()]
    preflight_errors = list(environment["errors"])
    if missing_videos:
        preflight_errors.append("missing_video_files:" + "|".join(missing_videos))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    if preflight_errors:
        for case, video_path in case_paths:
            record = _error_result(
                case,
                video_path=video_path,
                config=config,
                errors=preflight_errors,
                error_type="preflight_failed",
            )
            _append_jsonl(output_path, record)
            records.append(record)
        return {"passed": False, "preflight": environment, "records": records, "model_load_latency_sec": 0.0}

    model = Qwen25VL7BBaseline(
        model_path=model_path,
        max_pixels=int(config["max_pixels"]),
        seed=int(config["seed"]),
        dtype=str(config["dtype"]),
        quantization=config["quantization"],
    )
    for case, video_path in case_paths:
        question_started = time.perf_counter()
        images = []
        try:
            images, timestamps, _, frame_latency = extract_uniform_frames(
                video_path=video_path,
                ffmpeg_path=str(config["ffmpeg_path"]),
                ffprobe_path=str(config["ffprobe_path"]),
                num_frames=int(config["num_frames"]),
                max_pixels=int(config["max_pixels"]),
            )
            prompt = build_mcq_prompt(case["question"], case["options"], len(images))
            inference = model.infer(images=images, prompt=prompt, generation=config["generation"])
            prediction = int(inference["prediction_index"])
            record = {
                "baseline_id": config["baseline_id"],
                "video_id": case["video_id"],
                "question_id": case["question_id"],
                "question": case["question"],
                "options": case["options"],
                "video_path": str(video_path),
                "ground_truth_index": case["ground_truth_index"],
                "ground_truth_text": case["ground_truth_text"],
                "prediction_index": prediction,
                "prediction_text": case["options"][prediction],
                "correct": prediction == case["ground_truth_index"],
                "raw_model_output": inference["raw_output"],
                "sampled_frame_timestamps_sec": timestamps,
                "frame_count": len(images),
                "frame_extraction_latency_sec": frame_latency,
                "model_inference_latency_sec": inference["inference_latency_sec"],
                "total_per_question_latency_sec": time.perf_counter() - question_started,
                "model_calls": 1,
                "text_token_count": inference["text_token_count"],
                "visual_token_count": inference["visual_token_count"],
                "total_input_token_count": inference["total_input_token_count"],
                "output_token_count": inference["output_token_count"],
                "peak_gpu_memory_bytes": inference["peak_gpu_memory_bytes"],
                "model_checkpoint": str(model_path),
                "inference_settings": {
                    "num_frames": config["num_frames"],
                    "max_pixels": config["max_pixels"],
                    "dtype": config["dtype"],
                    "quantization_mode": config["quantization"]["mode"],
                    "actual_model_loading_mode": model.actual_model_loading_mode,
                    "generation": config["generation"],
                    "prompt_version": config["prompt_version"],
                },
                "ignored_annotation_interval": case["ignored_annotation_interval"],
                "status": "completed",
                "error_type": None,
                "error_message": None,
                "oom": False,
                "external_api_calls": 0,
            }
        except Exception as exc:
            import torch

            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            record = _error_result(
                case,
                video_path=video_path,
                config=config,
                errors=[repr(exc)],
                error_type="cuda_oom" if oom else "runtime_error",
                actual_model_loading_mode=model.actual_model_loading_mode,
            )
            record["total_per_question_latency_sec"] = time.perf_counter() - question_started
            record["oom"] = oom
        finally:
            for image in images:
                image.close()
        _append_jsonl(output_path, record)
        records.append(record)
        if record["status"] != "completed":
            break
    return {
        "passed": bool(records) and all(row["status"] == "completed" for row in records),
        "preflight": environment,
        "records": records,
        "model_load_latency_sec": model.model_load_latency_sec,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines/egopolice_b0.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--qa-path", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--case-id")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--quantization-mode", choices=("none", "nf4_4bit"))
    parser.add_argument("--ffmpeg-path")
    parser.add_argument("--ffprobe-path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    overrides = {
        key: getattr(args, key)
        for key in (
            "data_root", "model_root", "model_path", "qa_path", "video_root", "output_root",
            "output", "subset_manifest", "num_frames", "max_pixels", "dtype",
            "quantization_mode", "ffmpeg_path", "ffprobe_path",
        )
    }
    config, output_path = resolve_runtime_config(config, overrides=overrides)
    missing = [key for key in ("model_path", "qa_path", "video_root") if not config.get(key)]
    if missing:
        raise BaselineInputError(
            "Required portable path values missing: " + ",".join(missing)
            + ". Set CLI paths or DATA_ROOT and MODEL_PATH/MODEL_ROOT."
        )
    summary = run(config, output_path=output_path, limit=args.limit, case_id=args.case_id)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2
