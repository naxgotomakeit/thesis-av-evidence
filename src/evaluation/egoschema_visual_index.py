"""Build Ours-v0's frozen visual indexes for EgoSchema videos.

This is offline, question-independent processing. It reuses the exact
1-FPS/CLIP/state-region and 4-second/2-second-stride micro-window primitives
already used by the canonical baseline.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image

from src.visual.micro_clips import build_micro_clips
from src.visual.state_regions import (
    build_regions,
    extract_frames,
    frame_timestamps,
    validate_region_schema,
    write_regions_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _encode_frames(
    frame_paths: list[Path], model: Any, preprocess: Any, device: str, batch_size: int
) -> np.ndarray:
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frame_paths), batch_size):
            tensors = [
                preprocess(Image.open(path).convert("RGB"))
                for path in frame_paths[start : start + batch_size]
            ]
            features = model.encode_image(torch.stack(tensors).to(device)).float()
            features = features / features.norm(dim=-1, keepdim=True)
            batches.append(features.cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def _clip_module() -> Any:
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import packaging
        import packaging.version
        import types

        shim = types.ModuleType("pkg_resources")
        shim.packaging = packaging
        sys.modules["pkg_resources"] = shim
    import clip

    return clip


def _valid_existing(visual_dir: Path, micro_dir: Path, video_id: str) -> bool:
    required = (
        visual_dir / "visual_state_regions.json",
        visual_dir / "frame_embeddings.npy",
        visual_dir / "region_embeddings.npy",
        micro_dir / "microclip_index.json",
        micro_dir / "microclip_embeddings.npy",
    )
    if not all(path.is_file() for path in required):
        return False
    visual = json.loads(required[0].read_text(encoding="utf-8"))
    micro = json.loads(required[3].read_text(encoding="utf-8"))
    return (
        str(visual.get("video_id")) == video_id
        and str(micro.get("video_id")) == video_id
        and visual.get("encoder") == "openai_clip_vit_b_32"
        and float(visual.get("sampling_fps", 0)) == 1.0
        and int(micro.get("window_sec", 0)) == 4
        and int(micro.get("stride_sec", 0)) == 2
    )


def build_visual_indexes(
    *,
    project_root: Path,
    cases: Iterable[dict[str, Any]],
    data_root: Path,
    summary_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Build or safely reuse question-independent visual indexes."""
    config = yaml.safe_load((project_root / "configs/visual_mvp.yaml").read_text(encoding="utf-8"))
    micro_config = yaml.safe_load((project_root / "configs/visual_microclips.yaml").read_text(encoding="utf-8"))
    fps = float(config["fps"])
    if fps != 1.0 or int(micro_config["window_sec"]) != 4 or int(micro_config["stride_sec"]) != 2:
        raise RuntimeError("EgoSchema indexing requires the unchanged Ours-v0 visual configuration")
    device = "cuda" if str(config.get("device", "cuda")).startswith("cuda") and torch.cuda.is_available() else "cpu"
    ffmpeg = Path(str(config["ffmpeg_path"]))
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"Configured ffmpeg is unavailable: {ffmpeg}")

    rows = list(cases)
    video_ids = [str(row["video_id"]) for row in rows]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("Offline EgoSchema index input must contain unique videos")

    needed = []
    for row in rows:
        video_id = str(row["video_id"])
        visual_dir = project_root / "outputs/visual_index" / video_id
        micro_dir = project_root / "outputs/visual_micro_index" / video_id
        if force or not _valid_existing(visual_dir, micro_dir, video_id):
            needed.append(row)

    clip = _clip_module()
    load_started = time.perf_counter()
    model, preprocess = clip.load(
        "ViT-B/32",
        device=device,
        download_root=str(Path(str(config["clip_download_root"]))),
    )
    model.eval()
    model_load_sec = time.perf_counter() - load_started
    results: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for row in rows:
        video_id = str(row["video_id"])
        source = data_root / str(row["video_path"])
        if not source.is_file():
            raise FileNotFoundError(f"EgoSchema source video is unavailable: {source}")
        visual_dir = project_root / "outputs/visual_index" / video_id
        micro_dir = project_root / "outputs/visual_micro_index" / video_id
        if not force and _valid_existing(visual_dir, micro_dir, video_id):
            results.append({
                "case_id": row["case_id"],
                "video_id": video_id,
                "reused_existing": True,
                "frame_sampling_sec": 0.0,
                "clip_image_encoding_sec": 0.0,
                "temporal_region_construction_sec": 0.0,
                "micro_index_construction_sec": 0.0,
                "total_offline_indexing_sec": 0.0,
                "sampled_frame_count": len(list((visual_dir / "frames_1fps").glob("frame_*.jpg"))),
                "visual_region_count": len(json.loads((visual_dir / "visual_state_regions.json").read_text(encoding="utf-8"))["visual_state_regions"]),
                "index_size_bytes": sum(path.stat().st_size for path in (visual_dir, micro_dir) for path in path.rglob("*") if path.is_file()),
            })
            continue

        total_started = time.perf_counter()
        visual_dir.mkdir(parents=True, exist_ok=True)
        micro_dir.mkdir(parents=True, exist_ok=True)

        sampling_started = time.perf_counter()
        frame_paths = extract_frames(source, visual_dir / "frames_1fps", fps, str(ffmpeg))
        frame_sampling_sec = time.perf_counter() - sampling_started
        if not frame_paths:
            raise RuntimeError(f"No frames extracted for {video_id}")
        duration = float(row["duration_sec"])
        timestamps = frame_timestamps(len(frame_paths), fps, duration)

        encoding_started = time.perf_counter()
        embeddings = _encode_frames(
            frame_paths, model, preprocess, device, int(config.get("batch_size", 32))
        )
        if device == "cuda":
            torch.cuda.synchronize()
        clip_image_encoding_sec = time.perf_counter() - encoding_started
        np.save(visual_dir / "frame_embeddings.npy", embeddings)

        regions_started = time.perf_counter()
        regions, region_embeddings, _ = build_regions(
            timestamps,
            embeddings,
            frame_paths,
            duration,
            float(config["cosine_distance_threshold"]),
            float(config["min_region_duration_sec"]),
            visual_dir,
        )
        errors = [error for region in regions for error in validate_region_schema(region)]
        if errors:
            raise RuntimeError(f"Visual region schema failed for {video_id}: {errors[:5]}")
        np.save(visual_dir / "region_embeddings.npy", region_embeddings)
        write_regions_json(
            visual_dir / "visual_state_regions.json",
            {
                "video_id": video_id,
                "case_ids": [row["case_id"]],
                "video_path": str(row["video_path"]),
                "encoder": "openai_clip_vit_b_32",
                "sampling_fps": fps,
                "cosine_distance_threshold": float(config["cosine_distance_threshold"]),
                "min_region_duration_sec": float(config["min_region_duration_sec"]),
                "task_2_question_independent": True,
                "visual_index_inputs": ["video_frames"],
                "excluded_from_visual_index_computation": [
                    "question", "answer", "answer_options", "gold", "reference_timestamp"
                ],
                "visual_state_region_definition": "A temporally continuous visually stable segment; not a semantic event.",
                "frame_embeddings_path": "frame_embeddings.npy",
                "region_embeddings_path": "region_embeddings.npy",
                "visual_state_regions": regions,
            },
        )
        temporal_region_construction_sec = time.perf_counter() - regions_started

        micro_started = time.perf_counter()
        micro_rows, micro_embeddings = build_micro_clips(
            embeddings,
            frame_paths,
            duration,
            int(micro_config["window_sec"]),
            int(micro_config["stride_sec"]),
        )
        representatives = micro_dir / "representative_frames"
        representatives.mkdir(parents=True, exist_ok=True)
        normalized_rows = []
        for item in micro_rows:
            number = int(item.pop("embedding_index"))
            item["microclip_id"] = item.pop("micro_clip_id").replace("micro_clip_", "microclip_")
            item["embedding_row_index"] = number
            item["frame_paths"] = [
                Path(path).resolve().relative_to(project_root.resolve()).as_posix()
                for path in item["frame_paths"]
            ]
            source_frame = Path(item["representative_frame_path"])
            target = representatives / f"{item['microclip_id']}_t{float(item['representative_frame_timestamp']):07.3f}.jpg"
            shutil.copy2(source_frame, target)
            item["representative_frame_path"] = target.relative_to(project_root).as_posix()
            normalized_rows.append(item)
        np.save(micro_dir / "microclip_embeddings.npy", micro_embeddings)
        _write_json(
            micro_dir / "microclip_index.json",
            {
                "video_id": video_id,
                "construction_inputs": ["frame_embeddings", "1fps_frames", "video_duration"],
                "excluded_inputs": [
                    "question", "answer", "answer_options", "gold", "annotation_context"
                ],
                "window_sec": int(micro_config["window_sec"]),
                "stride_sec": int(micro_config["stride_sec"]),
                "embedding_model": "OpenAI CLIP ViT-B/32",
                "embedding_dtype": str(micro_embeddings.dtype),
                "embedding_dimension": int(micro_embeddings.shape[1]),
                "warnings": [],
                "microclips": normalized_rows,
            },
        )
        micro_index_construction_sec = time.perf_counter() - micro_started
        total_sec = time.perf_counter() - total_started
        results.append({
            "case_id": row["case_id"],
            "video_id": video_id,
            "source_video_sha256": _sha256(source),
            "reused_existing": False,
            "frame_sampling_sec": frame_sampling_sec,
            "clip_image_encoding_sec": clip_image_encoding_sec,
            "temporal_region_construction_sec": temporal_region_construction_sec,
            "micro_index_construction_sec": micro_index_construction_sec,
            "total_offline_indexing_sec": total_sec,
            "sampled_frame_count": len(frame_paths),
            "visual_region_count": len(regions),
            "microclip_count": len(normalized_rows),
            "index_size_bytes": sum(path.stat().st_size for path in (visual_dir, micro_dir) for path in path.rglob("*") if path.is_file()),
        })
        _write_json(
            summary_path,
            {
                "status": "in_progress",
                "model": "OpenAI CLIP ViT-B/32",
                "device": device,
                "model_load_sec_once": model_load_sec,
                "completed_videos": len(results),
                "videos": results,
            },
        )

    summary = {
        "status": "complete",
        "question_independent": True,
        "question_or_options_used": False,
        "gold_used": False,
        "model": "OpenAI CLIP ViT-B/32",
        "model_load_count": 1,
        "device": device,
        "model_load_sec_once": model_load_sec,
        "wall_clock_sec": time.perf_counter() - run_started,
        "frame_sampling_sec_total": sum(float(item["frame_sampling_sec"]) for item in results),
        "clip_image_encoding_sec_total": sum(float(item["clip_image_encoding_sec"]) for item in results),
        "temporal_region_construction_sec_total": sum(float(item["temporal_region_construction_sec"]) for item in results),
        "micro_index_construction_sec_total": sum(float(item["micro_index_construction_sec"]) for item in results),
        "total_offline_indexing_sec_sum": sum(float(item["total_offline_indexing_sec"]) for item in results),
        "sampled_frame_count_total": sum(int(item["sampled_frame_count"]) for item in results),
        "index_size_bytes_total": sum(int(item["index_size_bytes"]) for item in results),
        "videos": results,
    }
    _write_json(summary_path, summary)
    return summary
