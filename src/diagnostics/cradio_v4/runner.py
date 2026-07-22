from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    TARGET_VIDEO_ID,
    load_frozen_questions,
    one_fps_timestamps,
    posthoc_gt_interval_metrics,
    rank_timestamps,
    timestamps_to_frame_indices,
)


RADIO_GITHUB_REVISION = "c0f37017930e9dda53f93424cf4bf39fc51f287e"
RADIO_HF_REPO = "nvidia/C-RADIOv4-SO400M"
RADIO_HF_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"
RADIO_CHECKPOINT_FILENAME = "c-radio_v4-so400m_half.pth.tar"
RADIO_MODEL_ALIAS = "c-radio_v4-so400m"
RADIO_ADAPTOR = "siglip2-g"
SIGLIP2_TEXT_MODEL = "google/siglip2-giant-opt-patch16-384"
OFFICIAL_CARD_PARAMETER_COUNT = 431_237_240
INPUT_RESOLUTION = (512, 512)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    thesis_root = project_root.parents[1]
    parser = argparse.ArgumentParser(description="Isolated C-RADIOv4 1-FPS representation smoke")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=thesis_root / "data" / "EgoPolice_1.0.0",
    )
    parser.add_argument(
        "--question-manifest",
        type=Path,
        default=project_root / "config" / "data" / "egopolice_ablation_questions_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "diagnostics" / "cradio_v4",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=thesis_root / "models" / "diagnostic_caches" / "cradio_v4",
        help="Large model cache outside the quota-limited user home directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prior-error", action="append", default=[])
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(video_path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    video_streams = [stream for stream in payload["streams"] if stream.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise RuntimeError(f"Expected exactly one video stream in {video_path}")
    stream = video_streams[0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    frame_count = int(stream.get("nb_frames") or round(float(payload["format"]["duration"]) * fps))
    return {
        "duration_sec": float(payload["format"]["duration"]),
        "file_size_bytes": int(payload["format"]["size"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "average_fps": fps,
        "frame_count": frame_count,
        "video_codec": stream["codec_name"],
    }


def _to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    raise TypeError(f"Cannot move tokenizer output to {device}: {type(batch)!r}")


def _package_versions() -> dict[str, str]:
    from importlib.metadata import version

    packages = ["torch", "torchvision", "transformers", "decord", "timm", "einops"]
    return {name: version(name) for name in packages}


def _cached_snapshot_revision(hf_home: Path, repo_id: str) -> str | None:
    snapshot_root = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    revisions = sorted(path.name for path in snapshot_root.glob("*") if path.is_dir())
    return revisions[0] if len(revisions) == 1 else None


def _load_model(device: Any, checkpoint_path: Path) -> tuple[Any, dict[str, Any], list[str], float]:
    import torch

    caught: list[str] = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        model, checkpoint = torch.hub.load(
            f"NVlabs/RADIO:{RADIO_GITHUB_REVISION}",
            "radio_model",
            version=str(checkpoint_path),
            adaptor_names=[RADIO_ADAPTOR],
            progress=True,
            skip_validation=True,
            return_checkpoint=True,
            trust_repo=True,
        )
        model.to(device).eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        caught.extend(f"{item.category.__name__}: {item.message}" for item in records)
    return model, checkpoint, caught, time.perf_counter() - started


def _decode_and_embed(
    *, model: Any, video_path: Path, timestamps: np.ndarray, frame_indices: np.ndarray,
    batch_size: int, device: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    import decord
    import torch
    from torch.nn import functional as F

    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=4)
    chunks: list[np.ndarray] = []
    decode_sec = 0.0
    preprocess_sec = 0.0
    inference_sec = 0.0
    visual_forward_calls = 0
    generation_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for offset in range(0, len(frame_indices), batch_size):
            indices = frame_indices[offset : offset + batch_size]
            started = time.perf_counter()
            rgb = reader.get_batch(indices.tolist()).asnumpy()
            decode_sec += time.perf_counter() - started

            started = time.perf_counter()
            frames = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=False).div_(255.0)
            frames = F.interpolate(frames, size=INPUT_RESOLUTION, mode="bilinear", align_corners=False)
            preprocess_sec += time.perf_counter() - started

            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(frames)
                summaries = output[RADIO_ADAPTOR][0]
            summaries = F.normalize(summaries.float(), dim=-1)
            torch.cuda.synchronize(device)
            inference_sec += time.perf_counter() - started
            if summaries.ndim != 2 or summaries.shape[0] != len(indices):
                raise RuntimeError(f"Unexpected siglip2-g summary shape: {tuple(summaries.shape)}")
            chunks.append(summaries.to(device="cpu", dtype=torch.float16).numpy())
            visual_forward_calls += 1

    embeddings = np.concatenate(chunks, axis=0)
    if embeddings.shape[0] != timestamps.size:
        raise RuntimeError("Embedding/timestamp count mismatch")
    generation_wall_sec = time.perf_counter() - generation_started
    return embeddings, {
        "generation_wall_time_sec": generation_wall_sec,
        "decode_time_sec": decode_sec,
        "preprocess_time_sec": preprocess_sec,
        "visual_inference_time_sec": inference_sec,
        "visual_forward_calls": visual_forward_calls,
        "peak_gpu_allocated_bytes_embedding_phase": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    args.cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_root / "huggingface")
    os.environ["TORCH_HOME"] = str(args.cache_root / "torch")

    import torch
    from huggingface_hub import hf_hub_download
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    device = torch.device(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < 12 * 1024**3:
        raise RuntimeError(f"Insufficient free GPU memory: {free_bytes} bytes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.data_root / "videos" / "pasadena" / "YKI08.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    probe = _ffprobe(video_path)
    questions = load_frozen_questions(args.question_manifest)
    timestamps = one_fps_timestamps(probe["duration_sec"])
    frame_indices = timestamps_to_frame_indices(
        timestamps, average_fps=probe["average_fps"], frame_count=probe["frame_count"]
    )

    checkpoint_download_started = time.perf_counter()
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=RADIO_HF_REPO,
            filename=RADIO_CHECKPOINT_FILENAME,
            revision=RADIO_HF_REVISION,
        )
    )
    checkpoint_download_or_cache_sec = time.perf_counter() - checkpoint_download_started
    checkpoint_sha256 = _sha256(checkpoint_path)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    overall_started = time.perf_counter()
    model, checkpoint, caught_warnings, model_loading_sec = _load_model(device, checkpoint_path)
    peak_after_load = int(torch.cuda.max_memory_allocated(device))

    runtime_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    text_parameter_count = sum(
        parameter.numel() for parameter in model.adaptors[RADIO_ADAPTOR].text_model.parameters()
    )
    backbone_parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    text_config = model.adaptors[RADIO_ADAPTOR].text_model.config
    siglip_revision = getattr(text_config, "_commit_hash", None) or _cached_snapshot_revision(
        args.cache_root / "huggingface", SIGLIP2_TEXT_MODEL
    )

    embeddings, timing = _decode_and_embed(
        model=model,
        video_path=video_path,
        timestamps=timestamps,
        frame_indices=frame_indices,
        batch_size=args.batch_size,
        device=device,
    )
    peak_after_embedding = timing["peak_gpu_allocated_bytes_embedding_phase"]

    artifact_path = args.output_dir / "YKI08_1fps_siglip2g_embeddings.npz"
    serialization_started = time.perf_counter()
    np.savez_compressed(
        artifact_path,
        embeddings=embeddings,
        timestamps_sec=timestamps,
        frame_indices=frame_indices,
    )
    serialization_sec = time.perf_counter() - serialization_started
    artifact_size = artifact_path.stat().st_size

    text_started = time.perf_counter()
    tokenizer_output = model.adaptors[RADIO_ADAPTOR].tokenizer([row.question for row in questions])
    tokenizer_output = _to_device(tokenizer_output, device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        text_embeddings = model.adaptors[RADIO_ADAPTOR].encode_text(tokenizer_output, normalize=True)
    torch.cuda.synchronize(device)
    text_encoding_sec = time.perf_counter() - text_started

    visual_embeddings = torch.from_numpy(embeddings.astype(np.float32)).to(device)
    visual_embeddings = F.normalize(visual_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
    similarity_matrix = text_embeddings @ visual_embeddings.T
    similarity_matrix = similarity_matrix.cpu().numpy()

    retrieval_rows: list[dict[str, Any]] = []
    for question, similarities in zip(questions, similarity_matrix, strict=True):
        ranking = rank_timestamps(similarities)
        metrics = posthoc_gt_interval_metrics(
            ranking,
            timestamps,
            gt_start_sec=question.gt_start_sec,
            gt_end_sec=question.gt_end_sec,
        )
        top10 = [
            {
                "rank": rank + 1,
                "timestamp_sec": float(timestamps[index]),
                "source_frame_index": int(frame_indices[index]),
                "cosine_similarity": float(similarities[index]),
            }
            for rank, index in enumerate(ranking[:10])
        ]
        retrieval_rows.append(
            {
                "question_id": question.question_id,
                "duration_class": question.duration_class,
                "question_text_only": question.question,
                "answer_options_used_for_retrieval": False,
                "gt_used_for_ranking": False,
                **metrics,
                "top_10": top10,
            }
        )

    peak_full = max(
        peak_after_load,
        peak_after_embedding,
        int(torch.cuda.max_memory_allocated(device)),
    )
    unique_question_texts = len({row.question for row in questions})
    aggregate_hits = {
        str(k): {
            "hits": sum(row["gt_interval_hit_at_k"][str(k)] for row in retrieval_rows),
            "total": len(retrieval_rows),
            "rate": sum(row["gt_interval_hit_at_k"][str(k)] for row in retrieval_rows) / len(retrieval_rows),
        }
        for k in (1, 5, 8, 10)
    }

    result_path = args.output_dir / "YKI08_cradio_v4_so400m_1fps_results.json"
    result = {
        "schema_version": "cradio-v4-representation-smoke-v1",
        "diagnostic_label": "C-RADIOv4 representation smoke — exploratory, not a formal B0–B4 result.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_qwen_calls": True,
        "frozen_experiment_definitions_modified": False,
        "model": {
            "official_model_id": RADIO_HF_REPO,
            "checkpoint_filename": RADIO_CHECKPOINT_FILENAME,
            "checkpoint_revision": RADIO_HF_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "official_implementation": "NVlabs/RADIO",
            "implementation_revision": RADIO_GITHUB_REVISION,
            "adaptor": RADIO_ADAPTOR,
            "adaptor_text_model": SIGLIP2_TEXT_MODEL,
            "adaptor_text_model_revision_if_available": siglip_revision,
            "compute_dtype": "bfloat16 (CUDA autocast)",
            "stored_embedding_dtype": str(embeddings.dtype),
            "official_card_parameter_count": OFFICIAL_CARD_PARAMETER_COUNT,
            "runtime_backbone_parameter_count": backbone_parameter_count,
            "runtime_text_model_parameter_count": text_parameter_count,
            "runtime_total_loaded_parameter_count": runtime_parameter_count,
            "input_resolution_height_width": list(INPUT_RESOLUTION),
            "model_loading_time_sec": model_loading_sec,
            "model_loading_time_note": (
                "Cold-cache load of the C-RADIO checkpoint and official siglip2-g adaptor; "
                "includes the adaptor's first download and construction of "
                "google/siglip2-giant-opt-patch16-384, but excludes the separately timed "
                "C-RADIO checkpoint cache resolution."
            ),
            "checkpoint_download_or_cache_resolution_time_sec": checkpoint_download_or_cache_sec,
            "checkpoint_training_dtype": str(getattr(checkpoint.get("args"), "dtype", "unknown")),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_device": torch.cuda.get_device_name(device),
            "gpu_total_bytes_before_load": total_bytes,
            "gpu_free_bytes_before_load": free_bytes,
            "batch_size": args.batch_size,
            "cache_root": str(args.cache_root),
            "peak_gpu_allocated_bytes": peak_full,
            "warnings": caught_warnings,
            "errors_encountered_and_fixed_before_success": args.prior_error,
            "errors_in_successful_run": [],
        },
        "video": {
            "video_id": TARGET_VIDEO_ID,
            "path": str(video_path),
            **probe,
            "sampling": "deterministic timestamps 0, 1, ..., ceil(duration)-1 seconds (t < duration)",
            "sampling_fps": 1.0,
            "embedded_frame_count": int(timestamps.size),
            "embedding_dimension": int(embeddings.shape[1]),
        },
        "offline_embedding": {
            **timing,
            "serialization_time_sec": serialization_sec,
            "frames_per_second": timestamps.size / timing["generation_wall_time_sec"],
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "artifact_size_bytes": artifact_size,
            "bytes_per_embedded_frame": artifact_size / timestamps.size,
        },
        "retrieval": {
            "question_manifest": str(args.question_manifest),
            "question_count": len(questions),
            "unique_question_text_count": unique_question_texts,
            "identical_text_ranking_expected": unique_question_texts == 1,
            "query_definition": "question text only; options excluded",
            "text_encoding_time_sec_for_all_questions": text_encoding_sec,
            "text_encoder_forward_calls": 1,
            "aggregate_gt_interval_hit_at_k": aggregate_hits,
            "metric_limitation": (
                "GT Interval metrics are post-hoc temporal exposure diagnostics, not true evidence recall."
            ),
            "questions": retrieval_rows,
        },
        "call_accounting": {
            "qwen_model_calls": 0,
            "cradio_visual_forward_calls": timing["visual_forward_calls"],
            "siglip2_text_encoder_forward_calls": 1,
        },
        "control": {
            "dinov2_comparison_run": False,
            "reason": (
                "The frozen DINOv2 component is vision-only and has no existing question-compatible "
                "text-aligned similarity method; no custom mapping was invented."
            ),
        },
        "total_diagnostic_wall_time_after_checkpoint_available_sec": time.perf_counter() - overall_started,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result_path": str(result_path), "aggregate": aggregate_hits}, indent=2))

    del similarity_matrix, visual_embeddings, text_embeddings, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
