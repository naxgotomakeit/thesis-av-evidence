from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.thesis_av.visual.comet_style import CometStyleConfig, segment as comet_segment
from src.thesis_av.visual.hierarchy import build_boundary_records, build_fine_nodes, build_safe_hierarchy
from src.thesis_av.visual.medium import build_tree_context, fluid_frontier
from src.thesis_av.visual.segmentation_schema import make_segmentation, segmentation_metrics


VIDEO_ID = "pasadena/YKI08"
DINO_MODEL_ID = "facebook/dinov2-small"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_mapping(value: Any) -> Any:
    """Convert Transformers mapping wrappers (for example SizeDict) to JSON."""
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain_mapping(item) for key, item in asdict(value).items()}
    return dict(value) if hasattr(value, "items") else value


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected a non-empty finite one-dimensional distribution")
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "std": float(array.std()),
        "mad": float(np.median(np.abs(array - np.median(array)))),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    }


def boundary_agreement(
    left: Sequence[float], right: Sequence[float], tolerance_sec: float,
) -> dict[str, Any]:
    """One-to-one nearest matching with deterministic distance/index ties."""
    a = [float(value) for value in left]
    b = [float(value) for value in right]
    candidates = sorted(
        (
            (abs(x - y), i, j)
            for i, x in enumerate(a)
            for j, y in enumerate(b)
            if abs(x - y) <= tolerance_sec
        ),
        key=lambda row: (row[0], row[1], row[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches = []
    for distance, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append({"left": a[i], "right": b[j], "distance_sec": float(distance)})
    precision = len(matches) / len(b) if b else None
    recall = len(matches) / len(a) if a else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall else 0.0
    )
    return {
        "tolerance_sec": tolerance_sec,
        "matched_count": len(matches),
        "left_count": len(a),
        "right_count": len(b),
        "right_precision_vs_left": precision,
        "right_recall_vs_left": recall,
        "f1": f1,
        "matches": matches,
        "unmatched_left": [a[index] for index in range(len(a)) if index not in used_a],
        "unmatched_right": [b[index] for index in range(len(b)) if index not in used_b],
    }


def _duration_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min_sec": float(array.min()),
        "mean_sec": float(array.mean()),
        "median_sec": float(np.median(array)),
        "max_sec": float(array.max()),
        "lt_2_sec_count": int((array < 2.0).sum()),
        "lt_4_sec_count": int((array < 4.0).sum()),
        "lt_5_sec_count": int((array < 5.0).sum()),
        "lt_10_sec_count": int((array < 10.0).sum()),
    }


def replay_frozen_structure(
    *, label: str, features: np.ndarray, timestamps: np.ndarray,
    video_duration: float, config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the exact frozen CoMET/Safe-Merge/Fluid-Loose logic in isolation."""
    comet_config = CometStyleConfig(**config["comet_style"])
    internal = comet_segment(features, timestamps, comet_config)
    fine = make_segmentation(
        video_id=VIDEO_ID,
        method="comet_style_dinov2",  # frozen schema name retained for exact replay
        video_duration=video_duration,
        boundaries=internal["boundaries"],
        frame_timestamps=timestamps,
    )
    boundary_records = build_boundary_records(fine["segments"], internal)
    virtual_paths = [f"in_memory_1fps/frame_{index:05d}.jpg" for index in range(len(timestamps))]
    hierarchy_config = config["hierarchy"]
    leaves = build_fine_nodes(
        segments=fine["segments"],
        features=features,
        timestamps=timestamps,
        frame_paths=virtual_paths,
        boundary_records=boundary_records,
        representative_fractions=tuple(hierarchy_config["representative_fractions"]),
        include_medoid=bool(hierarchy_config["include_dinov2_medoid"]),
    )
    tree, merge_trace = build_safe_hierarchy(
        video_id=VIDEO_ID,
        video_duration=video_duration,
        fine_nodes=leaves,
        boundary_records=boundary_records,
        timestamps=timestamps,
        frame_paths=virtual_paths,
        features=features,
        medium_fraction=float(hierarchy_config["medium_reference_fraction"]),
        coarse_fraction=float(hierarchy_config["coarse_reference_fraction"]),
        representative_fractions=tuple(hierarchy_config["representative_fractions"]),
        include_medoid=bool(hierarchy_config["include_dinov2_medoid"]),
    )
    context = build_tree_context(tree)
    medium_ids, medium_trace = fluid_frontier(
        context,
        minimum_q_rank=float(config["medium"]["minimum_q_rank"]),
        maximum_local_drop=float(config["medium"]["maximum_local_drop"]),
        level="medium",
    )
    medium_segments = [
        {
            "node_id": node_id,
            "start": float(context["nodes"][node_id]["start"]),
            "end": float(context["nodes"][node_id]["end"]),
            "duration": float(context["nodes"][node_id]["duration"]),
            "fine_leaf_count": len(context["nodes"][node_id]["leaf_ids"]),
        }
        for node_id in medium_ids
    ]
    fine_metrics = segmentation_metrics(fine)
    raw_similarity = np.asarray(internal["raw_similarity"], dtype=np.float64)
    smooth_similarity = np.asarray(internal["smoothed_similarity"], dtype=np.float64)
    replay = {
        "representation": label,
        "frozen_logic_reused_unchanged": True,
        "legacy_schema_note": (
            "The exact frozen schema/method label and pooled_dinov2 field names are retained in-memory "
            "for replay compatibility; they do not claim that C-RADIO features are DINOv2."
        ),
        "settings": {
            "comet_style": config["comet_style"],
            "hierarchy": config["hierarchy"],
            "medium": config["medium"],
        },
        "adjacent_similarity": {
            "raw": raw_similarity.tolist(),
            "smoothed": smooth_similarity.tolist(),
            "raw_distribution": distribution_summary(raw_similarity),
            "smoothed_distribution": distribution_summary(smooth_similarity),
        },
        "comet_diagnostics": internal["diagnostics"],
        "fine_segmentation": fine,
        "fine_metrics": fine_metrics,
        "boundary_records": boundary_records,
        "fine_duration_summary": _duration_summary([row["duration"] for row in fine["segments"]]),
        "safe_merge": {
            "tree_valid": tree["invariants"],
            "accepted_merge_count": len(tree["accepted_merge_order"]),
            "merge_trace_row_count": len(merge_trace),
        },
        "medium_segments": medium_segments,
        "medium_duration_summary": _duration_summary([row["duration"] for row in medium_segments]),
        "medium_trace": medium_trace,
    }
    internal_objects = {"tree": tree, "context": context}
    return replay, internal_objects


def _extract_dinov2(
    *, video_path: Path, timestamps: np.ndarray, frame_indices: np.ndarray,
    output_dir: Path, cache_root: Path, batch_size: int, device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    import decord
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("An uncontended CUDA GPU is required for the comparable DINOv2 benchmark")
    torch.cuda.set_device(device)
    gpu_line = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[device.index or 0]
    used_mib, free_mib, utilization = [float(value.strip()) for value in gpu_line.split(",")]
    if utilization > 10.0 or used_mib > 1024.0:
        raise RuntimeError(
            "GPU is materially occupied; refusing a scientifically invalid contended timing run: "
            f"used={used_mib:.0f} MiB free={free_mib:.0f} MiB util={utilization:.0f}%"
        )
    download_started = time.perf_counter()
    snapshot = Path(snapshot_download(repo_id=DINO_MODEL_ID, cache_dir=cache_root / "huggingface/hub"))
    checkpoint_resolution_sec = time.perf_counter() - download_started
    revision = snapshot.name
    model_file = snapshot / "model.safetensors"

    torch.cuda.empty_cache()
    # PyTorch 2.13's CUDA memory-stat bindings reject a ``torch.device``
    # argument on this host.  The intended device is current, so use the
    # current-device form that is also compatible with older PyTorch builds.
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True).to(device).eval()
    torch.cuda.synchronize(device)
    model_load_sec = time.perf_counter() - load_started
    peak_load = int(torch.cuda.max_memory_allocated())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})

    torch.cuda.reset_peak_memory_stats()
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=4)
    batches = []
    decode_sec = preprocess_sec = inference_sec = 0.0
    generation_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(frame_indices), batch_size):
            indices = frame_indices[offset:offset + batch_size]
            started = time.perf_counter()
            arrays = reader.get_batch(indices.tolist()).asnumpy()
            decode_sec += time.perf_counter() - started

            started = time.perf_counter()
            images = [Image.fromarray(array, mode="RGB") for array in arrays]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            preprocess_sec += time.perf_counter() - started

            started = time.perf_counter()
            output = model(**inputs).last_hidden_state[:, 0, :]
            output = torch.nn.functional.normalize(output.float(), dim=-1)
            torch.cuda.synchronize(device)
            inference_sec += time.perf_counter() - started
            batches.append(output.cpu().numpy().astype(np.float32))
    generation_sec = time.perf_counter() - generation_started
    features = np.concatenate(batches, axis=0)
    peak_embedding = int(torch.cuda.max_memory_allocated())

    artifact_path = output_dir / "YKI08_1fps_dinov2_embeddings.npz"
    serialization_started = time.perf_counter()
    np.savez_compressed(
        artifact_path,
        embeddings=features,
        timestamps_sec=timestamps,
        frame_indices=frame_indices,
    )
    serialization_sec = time.perf_counter() - serialization_started
    metadata = {
        "model_id": DINO_MODEL_ID,
        "checkpoint_revision": revision,
        "checkpoint_file": str(model_file),
        "checkpoint_sha256": sha256_file(model_file),
        "parameter_count": parameter_count,
        "parameter_dtypes": parameter_dtypes,
        "feature": "last_hidden_state_cls_token",
        "normalization": "l2",
        "dimension": int(features.shape[1]),
        "stored_dtype": str(features.dtype),
        "batch_size": batch_size,
        "gpu_preflight": {
            "used_mib": used_mib,
            "free_mib": free_mib,
            "utilization_percent": utilization,
            "uncontended_requirement_passed": True,
        },
        "processor_size": _plain_mapping(getattr(processor, "size", None)),
        "processor_crop_size": _plain_mapping(getattr(processor, "crop_size", None)),
        "checkpoint_download_or_cache_resolution_sec": checkpoint_resolution_sec,
        "model_load_sec": model_load_sec,
        "generation_wall_time_sec": generation_sec,
        "decode_sec": decode_sec,
        "preprocess_sec": preprocess_sec,
        "inference_sec": inference_sec,
        "serialization_sec": serialization_sec,
        "frames_per_second": len(features) / generation_sec,
        "peak_gpu_allocated_bytes_load": peak_load,
        "peak_gpu_allocated_bytes_embedding": peak_embedding,
        "peak_gpu_allocated_bytes": max(peak_load, peak_embedding),
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "artifact_bytes_per_frame": artifact_path.stat().st_size / len(features),
        "raw_embedding_payload_bytes": features.nbytes,
        "raw_embedding_bytes_per_frame": features.nbytes / len(features),
    }
    (output_dir / "dinov2_embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    del model, inputs
    torch.cuda.empty_cache()
    return features, metadata


def _rank_matched_boundaries(replay: dict[str, Any], count: int) -> list[float]:
    records = sorted(
        replay["boundary_records"],
        key=lambda row: (-float(row["smoothed_change_strength"]), float(row["timestamp"])),
    )
    return sorted(float(row["timestamp"]) for row in records[:count])


def _nearest_distance(values: Sequence[float], target: float) -> float | None:
    return min((abs(float(value) - target) for value in values), default=None)


def _boundary_examples(dino: dict[str, Any], cradio: dict[str, Any]) -> list[dict[str, Any]]:
    d_records = sorted(dino["boundary_records"], key=lambda row: -float(row["smoothed_change_strength"]))
    c_records = sorted(cradio["boundary_records"], key=lambda row: -float(row["smoothed_change_strength"]))
    examples: list[dict[str, Any]] = []

    def add(label: str, timestamp: float, source: str, strength: float) -> None:
        if any(abs(row["timestamp_sec"] - timestamp) < 0.5 for row in examples):
            return
        examples.append({
            "example_type": label,
            "timestamp_sec": timestamp,
            "source_representation": source,
            "source_smoothed_change_strength": strength,
            "nearest_dinov2_boundary_distance_sec": _nearest_distance(
                [row["timestamp"] for row in dino["boundary_records"]], timestamp
            ),
            "nearest_cradio_boundary_distance_sec": _nearest_distance(
                [row["timestamp"] for row in cradio["boundary_records"]], timestamp
            ),
        })

    if d_records:
        add("strongest_dinov2_boundary", float(d_records[0]["timestamp"]), "dinov2", float(d_records[0]["smoothed_change_strength"]))
    if c_records:
        add("strongest_cradio_boundary", float(c_records[0]["timestamp"]), "cradio", float(c_records[0]["smoothed_change_strength"]))
    for row in d_records:
        timestamp = float(row["timestamp"])
        if (_nearest_distance([item["timestamp"] for item in c_records], timestamp) or 0.0) <= 2.0:
            add("strong_boundary_agreement_within_2s", timestamp, "dinov2", float(row["smoothed_change_strength"]))
            break
    for row in d_records:
        timestamp = float(row["timestamp"])
        distance = _nearest_distance([item["timestamp"] for item in c_records], timestamp)
        if distance is None or distance > 5.0:
            add("strong_dinov2_only_boundary_gt5s", timestamp, "dinov2", float(row["smoothed_change_strength"]))
            break
    for row in c_records:
        timestamp = float(row["timestamp"])
        distance = _nearest_distance([item["timestamp"] for item in d_records], timestamp)
        if distance is None or distance > 5.0:
            add("strong_cradio_only_boundary_gt5s", timestamp, "cradio", float(row["smoothed_change_strength"]))
            break
    return examples


def _write_contact_sheets(video_path: Path, examples: list[dict[str, Any]], output_dir: Path) -> None:
    import decord
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=2)
    fps = float(reader.get_avg_fps())
    for index, example in enumerate(examples, 1):
        center = float(example["timestamp_sec"])
        times = [max(0.0, center - 1.0), center, min(center + 1.0, (len(reader) - 1) / fps)]
        indices = [min(len(reader) - 1, int(np.floor(value * fps))) for value in times]
        arrays = reader.get_batch(indices).asnumpy()
        thumbs = []
        for array, timestamp in zip(arrays, times):
            image = Image.fromarray(array, mode="RGB")
            image.thumbnail((384, 216))
            canvas = Image.new("RGB", (384, 246), "white")
            canvas.paste(image, ((384 - image.width) // 2, 0))
            ImageDraw.Draw(canvas).text((8, 222), f"{timestamp:.1f}s", fill="black")
            thumbs.append(canvas)
        sheet = Image.new("RGB", (1152, 276), "white")
        for position, image in enumerate(thumbs):
            sheet.paste(image, (position * 384, 30))
        ImageDraw.Draw(sheet).text(
            (8, 8), f"{example['example_type']} @ {center:.1f}s", fill="black"
        )
        path = output_dir / f"{index:02d}_{example['example_type']}_{int(round(center)):05d}s.jpg"
        sheet.save(path, quality=90)
        example["contact_sheet_path"] = str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    *, config_path: Path, video_path: Path, cradio_embedding_path: Path,
    cradio_result_path: Path, output_dir: Path, cache_root: Path,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with np.load(cradio_embedding_path) as artifact:
        cradio = np.asarray(artifact["embeddings"], dtype=np.float32)
        timestamps = np.asarray(artifact["timestamps_sec"], dtype=np.float64)
        frame_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
    cradio_audit = json.loads(cradio_result_path.read_text(encoding="utf-8"))
    duration = float(cradio_audit["video"]["duration_sec"])
    if not np.array_equal(timestamps, np.arange(round(duration), dtype=np.float64)):
        raise RuntimeError("C-RADIO timestamps do not match the frozen 1 FPS contract")

    dino, dino_metadata = _extract_dinov2(
        video_path=video_path,
        timestamps=timestamps,
        frame_indices=frame_indices,
        output_dir=output_dir,
        cache_root=cache_root,
        batch_size=int(config["dinov2"]["batch_size"]),
        device_name=device_name,
    )
    dino_replay, _ = replay_frozen_structure(
        label="DINOv2", features=dino, timestamps=timestamps,
        video_duration=duration, config=config,
    )
    cradio_replay, _ = replay_frozen_structure(
        label="C-RADIOv4 siglip2-g visual summary", features=cradio,
        timestamps=timestamps, video_duration=duration, config=config,
    )
    dino_boundaries = dino_replay["fine_segmentation"]["segments"][:-1]
    dino_boundary_times = [float(row["end"]) for row in dino_boundaries]
    cradio_boundaries = cradio_replay["fine_segmentation"]["segments"][:-1]
    cradio_boundary_times = [float(row["end"]) for row in cradio_boundaries]
    agreement = {
        str(tolerance): boundary_agreement(dino_boundary_times, cradio_boundary_times, tolerance)
        for tolerance in (0.0, 1.0, 2.0, 5.0)
    }
    nearest_d_to_c = [
        _nearest_distance(cradio_boundary_times, timestamp) for timestamp in dino_boundary_times
    ]
    nearest_c_to_d = [
        _nearest_distance(dino_boundary_times, timestamp) for timestamp in cradio_boundary_times
    ]
    rank_count = min(len(dino_boundary_times), len(cradio_boundary_times))
    rank_matched_dino = _rank_matched_boundaries(dino_replay, rank_count)
    rank_matched_cradio = _rank_matched_boundaries(cradio_replay, rank_count)
    rank_matched = {
        str(tolerance): boundary_agreement(rank_matched_dino, rank_matched_cradio, tolerance)
        for tolerance in (1.0, 2.0, 5.0)
    }

    dino_mad = float(dino_replay["comet_diagnostics"]["mad_smoothed_similarity"])
    cradio_mad = float(cradio_replay["comet_diagnostics"]["mad_smoothed_similarity"])
    dino_prominence = float(dino_replay["comet_diagnostics"]["adaptive_prominence"])
    cradio_prominence = float(cradio_replay["comet_diagnostics"]["adaptive_prominence"])
    mad_ratio = max(dino_mad, cradio_mad) / max(min(dino_mad, cradio_mad), 1e-12)
    absolute_scale_compatibility = {
        "same_l2_normalized_cosine_domain": True,
        "same_frozen_thresholds_applied": True,
        "dinov2_smoothed_mad": dino_mad,
        "cradio_smoothed_mad": cradio_mad,
        "mad_scale_ratio_larger_over_smaller": mad_ratio,
        "dinov2_adaptive_prominence": dino_prominence,
        "cradio_adaptive_prominence": cradio_prominence,
        "material_scale_difference_flag": mad_ratio > 2.0,
        "interpretation": (
            "The identical frozen settings are mathematically applicable because both inputs are L2-normalized "
            "cosine representations. A >2x MAD scale ratio is flagged because the shared absolute 0.005 "
            "prominence floor can have different effective stringency; no threshold was retuned."
        ),
    }

    examples = _boundary_examples(dino_replay, cradio_replay)
    _write_contact_sheets(video_path, examples, output_dir / "qualitative_boundaries")
    cradio_offline = cradio_audit["offline_embedding"]
    efficiency = [
        {
            "representation": "DINOv2-small CLS",
            "model": DINO_MODEL_ID,
            "dimension": dino_metadata["dimension"],
            "compute_dtype": ",".join(dino_metadata["parameter_dtypes"]),
            "stored_dtype": dino_metadata["stored_dtype"],
            "generation_wall_time_sec": dino_metadata["generation_wall_time_sec"],
            "frames_per_second": dino_metadata["frames_per_second"],
            "peak_gpu_allocated_bytes": dino_metadata["peak_gpu_allocated_bytes"],
            "artifact_size_bytes": dino_metadata["artifact_size_bytes"],
            "artifact_bytes_per_frame": dino_metadata["artifact_bytes_per_frame"],
            "raw_embedding_bytes_per_frame": dino_metadata["raw_embedding_bytes_per_frame"],
        },
        {
            "representation": "C-RADIOv4 siglip2-g summary",
            "model": cradio_audit["model"]["official_model_id"],
            "dimension": cradio_audit["video"]["embedding_dimension"],
            "compute_dtype": cradio_audit["model"]["compute_dtype"],
            "stored_dtype": cradio_audit["model"]["stored_embedding_dtype"],
            "generation_wall_time_sec": cradio_offline["generation_wall_time_sec"],
            "frames_per_second": cradio_offline["frames_per_second"],
            "peak_gpu_allocated_bytes": cradio_audit["runtime"]["peak_gpu_allocated_bytes"],
            "artifact_size_bytes": cradio_offline["artifact_size_bytes"],
            "artifact_bytes_per_frame": cradio_offline["bytes_per_embedded_frame"],
            "raw_embedding_bytes_per_frame": cradio_audit["video"]["embedding_dimension"] * 2,
        },
    ]

    result = {
        "schema_version": "cradio-v4-dinov2-control-v1",
        "diagnostic_label": "C-RADIOv4 vs DINOv2 exploratory diagnostic — not formal B0-B4",
        "qwen_calls": 0,
        "frozen_pipeline_outputs_overwritten": False,
        "input": {
            "video_id": VIDEO_ID,
            "video_path": str(video_path),
            "duration_sec": duration,
            "timestamps_identical": True,
            "timestamp_count": len(timestamps),
            "timestamp_rule": "frozen np.arange(round(duration)) 1 FPS grid",
            "same_source_frame_indices": True,
            "preprocessing_fairness": (
                "Both consume the same decoded RGB source frames and timestamps. DINOv2 uses its official "
                "AutoImageProcessor; C-RADIO uses the previously audited official-policy 512x512 [0,1] tensor input."
            ),
        },
        "dinov2_embedding": dino_metadata,
        "cradio_embedding_source": {
            "artifact": str(cradio_embedding_path),
            "artifact_sha256": cradio_offline["artifact_sha256"],
            "reused": True,
        },
        "absolute_frozen_threshold_compatibility": absolute_scale_compatibility,
        "representations": {
            "dinov2": dino_replay,
            "cradio": cradio_replay,
        },
        "boundary_agreement": {
            "exact_frozen_threshold_outputs": agreement,
            "median_nearest_distance_dinov2_to_cradio_sec": float(statistics.median(nearest_d_to_c)) if nearest_d_to_c else None,
            "median_nearest_distance_cradio_to_dinov2_sec": float(statistics.median(nearest_c_to_d)) if nearest_c_to_d else None,
            "rank_matched_secondary": {
                "label": "Secondary rank-matched comparison; no frozen threshold retuning.",
                "boundary_count_per_representation": rank_count,
                "agreement": rank_matched,
            },
        },
        "efficiency": efficiency,
        "qualitative_boundary_examples": examples,
    }
    result_path = output_dir / "dino_comparison_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "dinov2_replay.json").write_text(json.dumps(dino_replay, indent=2) + "\n", encoding="utf-8")
    (output_dir / "cradio_replay.json").write_text(json.dumps(cradio_replay, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "representation_efficiency.csv", efficiency)
    _write_csv(output_dir / "segmentation_summary.csv", [
        {
            "representation": label,
            "fine_segments": replay["fine_metrics"]["number_of_segments"],
            "fine_mean_duration_sec": replay["fine_metrics"]["mean_segment_duration"],
            "fine_median_duration_sec": replay["fine_metrics"]["median_segment_duration"],
            "fine_lt4_count": replay["fine_metrics"]["short_segment_count_lt_4s"],
            "fine_boundaries_per_minute": replay["fine_metrics"]["boundaries_per_minute"],
            "medium_segments": replay["medium_duration_summary"]["count"],
            "medium_mean_duration_sec": replay["medium_duration_summary"]["mean_sec"],
            "medium_median_duration_sec": replay["medium_duration_summary"]["median_sec"],
        }
        for label, replay in (("DINOv2", dino_replay), ("C-RADIOv4", cradio_replay))
    ])
    _write_csv(output_dir / "adjacent_similarity_summary.csv", [
        {"representation": label, **replay["adjacent_similarity"]["raw_distribution"]}
        for label, replay in (("DINOv2", dino_replay), ("C-RADIOv4", cradio_replay))
    ])
    _write_csv(output_dir / "boundary_agreement.csv", [
        {
            "tolerance_sec": tolerance,
            "matched_count": values["matched_count"],
            "dinov2_boundary_count": values["left_count"],
            "cradio_boundary_count": values["right_count"],
            "precision": values["right_precision_vs_left"],
            "recall": values["right_recall_vs_left"],
            "f1": values["f1"],
        }
        for tolerance, values in agreement.items()
    ])
    _write_csv(output_dir / "qualitative_boundary_examples.csv", examples)
    return result
