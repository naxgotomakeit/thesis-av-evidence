from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines.egopolice_formal.runner import (
    ROOT,
    atomic_write_json,
    load_formal_contract,
    sha256_file,
    stable_sha256,
    utc_now,
)
from src.diagnostics.cradio_v4.core import one_fps_timestamps, timestamps_to_frame_indices
from src.diagnostics.cradio_v4.runner import (
    RADIO_ADAPTOR,
    RADIO_CHECKPOINT_FILENAME,
    RADIO_HF_REPO,
    RADIO_HF_REVISION,
    _decode_and_embed,
    _ffprobe,
    _load_model,
)

from .constants import (
    CRADIO_INDEX_SCHEMA_VERSION,
    INDEX_BATCH_SIZE,
    INDEX_EMBEDDING_DIMENSION,
    INDEX_MINIMUM_FREE_VRAM_BYTES,
    cradio_frozen_configuration,
)


THESIS_ROOT = ROOT.parents[1]
DEFAULT_INDEX_ROOT = (
    ROOT / "outputs/indexes/cradio_v4_so400m_siglip2g_1fps/egopolice_ablation20_v1"
)
DEFAULT_CACHE_ROOT = THESIS_ROOT / "models/diagnostic_caches/cradio_v4"
DEFAULT_YKI08_DIAGNOSTIC_INDEX = (
    ROOT / "outputs/diagnostics/cradio_v4/YKI08_1fps_siglip2g_embeddings.npz"
)
DEFAULT_YKI08_DIAGNOSTIC_RESULT = (
    ROOT / "outputs/diagnostics/cradio_v4/YKI08_cradio_v4_so400m_1fps_results.json"
)


def index_directory(index_root: Path, video_id: str) -> Path:
    parts = video_id.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"Unsafe video ID: {video_id!r}")
    return index_root.joinpath(*parts)


def index_paths(index_root: Path, video_id: str) -> tuple[Path, Path]:
    directory = index_directory(index_root, video_id)
    return directory / "index.npz", directory / "metadata.json"


def build_index_fingerprint(
    *, video: dict[str, Any], video_manifest_sha256: str,
) -> str:
    return stable_sha256({
        "schema_version": CRADIO_INDEX_SCHEMA_VERSION,
        "video_manifest_sha256": video_manifest_sha256,
        "video_id": video["video_id"],
        "duration_sec": video["duration_sec"],
        "file_size_bytes": video["file_size_bytes"],
        "configuration": cradio_frozen_configuration(),
    })


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_contract(
    *, arrays: Any, duration_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.asarray(arrays["embeddings"])
    timestamps = np.asarray(arrays["timestamps_sec"], dtype=np.float64)
    frame_indices = np.asarray(arrays["frame_indices"], dtype=np.int64)
    expected_timestamps = one_fps_timestamps(duration_sec)
    if embeddings.shape != (expected_timestamps.size, INDEX_EMBEDDING_DIMENSION):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    if embeddings.dtype != np.float16:
        raise ValueError(f"Unexpected embedding dtype: {embeddings.dtype}")
    if not np.array_equal(timestamps, expected_timestamps):
        raise ValueError("Index timestamps do not match the frozen 1 FPS grid")
    if frame_indices.shape != timestamps.shape or np.any(np.diff(frame_indices) < 0):
        raise ValueError("Invalid source frame-index mapping")
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite C-RADIO embeddings")
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3, rtol=0.0):
        raise ValueError("Stored visual embeddings are not L2 normalized")
    return embeddings, timestamps, frame_indices


def validate_index(
    *, index_root: Path, video: dict[str, Any], video_manifest_sha256: str,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    artifact_path, metadata_path = index_paths(index_root, video["video_id"])
    if not artifact_path.is_file() or not metadata_path.is_file():
        return False, None, "index artifact or metadata missing"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_fingerprint = build_index_fingerprint(
            video=video, video_manifest_sha256=video_manifest_sha256
        )
        if metadata.get("schema_version") != CRADIO_INDEX_SCHEMA_VERSION:
            raise ValueError("index schema mismatch")
        if metadata.get("index_fingerprint") != expected_fingerprint:
            raise ValueError("index fingerprint mismatch")
        if metadata.get("artifact_sha256") != sha256_file(artifact_path):
            raise ValueError("index artifact SHA256 mismatch")
        if int(metadata.get("artifact_size_bytes") or -1) != artifact_path.stat().st_size:
            raise ValueError("index artifact size mismatch")
        with np.load(artifact_path, allow_pickle=False) as arrays:
            embeddings, timestamps, _ = _array_contract(
                arrays=arrays, duration_sec=float(video["duration_sec"])
            )
        if int(metadata.get("embedded_frame_count") or -1) != timestamps.size:
            raise ValueError("metadata frame count mismatch")
        if int(metadata.get("embedding_dimension") or -1) != embeddings.shape[1]:
            raise ValueError("metadata embedding dimension mismatch")
    except Exception as exc:
        return False, None, str(exc)
    return True, metadata, None


def _quarantine_index(index_root: Path, video_id: str, reason: str) -> None:
    artifact_path, metadata_path = index_paths(index_root, video_id)
    suffix = f".invalid.{int(time.time())}.{uuid.uuid4().hex[:8]}"
    moved = []
    for path in (artifact_path, metadata_path):
        if path.exists():
            destination = path.with_name(path.name + suffix)
            os.replace(path, destination)
            moved.append(str(destination))
    atomic_write_json(
        index_directory(index_root, video_id) / f"quarantine{suffix}.json",
        {"quarantined_at": utc_now(), "reason": reason, "moved": moved},
    )


def _write_index(
    *, index_root: Path, video: dict[str, Any], video_manifest_sha256: str,
    embeddings: np.ndarray, timestamps: np.ndarray, frame_indices: np.ndarray,
    timing: dict[str, Any], source: dict[str, Any], peak_vram_bytes: int,
) -> dict[str, Any]:
    artifact_path, metadata_path = index_paths(index_root, video["video_id"])
    serialization_started = time.perf_counter()
    _atomic_npz(
        artifact_path, embeddings=embeddings, timestamps_sec=timestamps,
        frame_indices=frame_indices,
    )
    serialization_sec = time.perf_counter() - serialization_started
    metadata = {
        "schema_version": CRADIO_INDEX_SCHEMA_VERSION,
        "created_at": utc_now(),
        "video_id": video["video_id"],
        "video_path": video["path"],
        "video_manifest_sha256": video_manifest_sha256,
        "index_fingerprint": build_index_fingerprint(
            video=video, video_manifest_sha256=video_manifest_sha256
        ),
        "configuration": cradio_frozen_configuration(),
        "video_duration_sec": video["duration_sec"],
        "embedded_frame_count": int(timestamps.size),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "decode_time_sec": float(timing["decode_time_sec"]),
        "preprocess_time_sec": float(timing["preprocess_time_sec"]),
        "visual_embedding_inference_time_sec": float(timing["visual_inference_time_sec"]),
        "embedding_generation_wall_time_sec": float(timing["generation_wall_time_sec"]),
        "serialization_time_sec": serialization_sec,
        "total_indexing_time_sec": float(timing["generation_wall_time_sec"]) + serialization_sec,
        "peak_gpu_allocated_memory_bytes": int(peak_vram_bytes),
        "visual_forward_calls": int(timing["visual_forward_calls"]),
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "bytes_per_embedded_frame": artifact_path.stat().st_size / timestamps.size,
        "source": source,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def import_yki08_diagnostic_index(
    *, index_root: Path, video: dict[str, Any], video_manifest_sha256: str,
    diagnostic_index: Path, diagnostic_result: Path,
) -> dict[str, Any]:
    if video["video_id"] != "pasadena/YKI08":
        raise ValueError("The compatibility import is only defined for pasadena/YKI08")
    result = json.loads(diagnostic_result.read_text(encoding="utf-8"))
    model = result["model"]
    expected = cradio_frozen_configuration()
    checks = {
        "official_model_id": model["official_model_id"],
        "checkpoint_filename": model["checkpoint_filename"],
        "checkpoint_revision": model["checkpoint_revision"],
        "implementation_revision": model["implementation_revision"],
        "adaptor": model["adaptor"],
        "adaptor_text_model": model["adaptor_text_model"],
        "stored_embedding_dtype": model["stored_embedding_dtype"],
        "input_resolution_height_width": model["input_resolution_height_width"],
    }
    for name, observed in checks.items():
        if observed != expected[name]:
            raise ValueError(f"Incompatible YKI08 diagnostic setting: {name}")
    if result["runtime"]["batch_size"] != INDEX_BATCH_SIZE:
        raise ValueError("Incompatible YKI08 diagnostic batch size")
    if result["offline_embedding"]["artifact_sha256"] != sha256_file(diagnostic_index):
        raise ValueError("YKI08 diagnostic artifact hash mismatch")
    with np.load(diagnostic_index, allow_pickle=False) as arrays:
        embeddings, timestamps, frame_indices = _array_contract(
            arrays=arrays, duration_sec=float(video["duration_sec"])
        )
        embeddings = embeddings.copy()
        timestamps = timestamps.copy()
        frame_indices = frame_indices.copy()
    timing = {
        "decode_time_sec": result["offline_embedding"]["decode_time_sec"],
        "preprocess_time_sec": result["offline_embedding"]["preprocess_time_sec"],
        "visual_inference_time_sec": result["offline_embedding"]["visual_inference_time_sec"],
        "generation_wall_time_sec": result["offline_embedding"]["generation_wall_time_sec"],
        "visual_forward_calls": result["offline_embedding"]["visual_forward_calls"],
    }
    return _write_index(
        index_root=index_root, video=video,
        video_manifest_sha256=video_manifest_sha256,
        embeddings=embeddings, timestamps=timestamps, frame_indices=frame_indices,
        timing=timing,
        source={
            "type": "reused_validated_yki08_diagnostic",
            "diagnostic_index": str(diagnostic_index),
            "diagnostic_result": str(diagnostic_result),
            "original_artifact_sha256": sha256_file(diagnostic_index),
            "visual_reencoding_performed": False,
        },
        peak_vram_bytes=int(result["runtime"]["peak_gpu_allocated_bytes"]),
    )


def estimate_indexing(contract: dict[str, Any]) -> dict[str, Any]:
    diagnostic = json.loads(DEFAULT_YKI08_DIAGNOSTIC_RESULT.read_text(encoding="utf-8"))
    throughput = float(diagnostic["offline_embedding"]["frames_per_second"])
    bytes_per_frame = float(diagnostic["offline_embedding"]["bytes_per_embedded_frame"])
    load_sec = float(diagnostic["model"]["model_loading_time_sec"])
    serialization_per_frame = (
        float(diagnostic["offline_embedding"]["serialization_time_sec"])
        / int(diagnostic["video"]["embedded_frame_count"])
    )
    rows = []
    for video in contract["videos"].values():
        frames = int(math.ceil(float(video["duration_sec"])))
        generation_sec = frames / throughput
        serialization_sec = frames * serialization_per_frame
        rows.append({
            "video_id": video["video_id"],
            "duration_sec": video["duration_sec"],
            "estimated_1fps_frames": frames,
            "estimated_generation_time_sec": generation_sec,
            "estimated_serialization_time_sec": serialization_sec,
            "estimated_artifact_size_bytes": int(math.ceil(frames * bytes_per_frame)),
        })
    return {
        "basis": "linear projection from validated YKI08 diagnostic; estimate only",
        "validated_throughput_frames_per_sec": throughput,
        "validated_bytes_per_frame": bytes_per_frame,
        "one_time_model_load_sec": load_sec,
        "total_1fps_frames": sum(row["estimated_1fps_frames"] for row in rows),
        "total_generation_time_sec": sum(row["estimated_generation_time_sec"] for row in rows),
        "total_serialization_time_sec": sum(row["estimated_serialization_time_sec"] for row in rows),
        "total_wall_time_including_one_model_load_sec": (
            load_sec + sum(row["estimated_generation_time_sec"] + row["estimated_serialization_time_sec"] for row in rows)
        ),
        "total_artifact_size_bytes": sum(row["estimated_artifact_size_bytes"] for row in rows),
        "peak_gpu_allocated_bytes_basis": diagnostic["runtime"]["peak_gpu_allocated_bytes"],
        "per_video": rows,
    }


def write_index_manifest(
    *, index_root: Path, contract: dict[str, Any], metadata_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Write the validated reusable-index inventory without changing any index."""
    by_id = {str(row["video_id"]): row for row in metadata_rows}
    expected_ids = set(contract["videos"])
    if set(by_id) != expected_ids:
        raise ValueError("Index manifest requires exactly the frozen 20 videos")
    ordered = [by_id[video_id] for video_id in contract["videos"]]
    question_counts = {
        video_id: sum(
            question["video_id"] == video_id for question in contract["questions"]
        )
        for video_id in contract["videos"]
    }
    rows = []
    for metadata in ordered:
        count = question_counts[metadata["video_id"]]
        rows.append({
            **metadata,
            "formal_question_count": count,
            "amortized_indexing_time_sec_per_formal_question_excluding_shared_model_load": (
                float(metadata["total_indexing_time_sec"]) / count
            ),
        })
    payload = {
        "schema_version": "egopolice-b1-cradio-index-manifest-v1",
        "created_at": utc_now(),
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "configuration": cradio_frozen_configuration(),
        "video_count": len(rows),
        "formal_question_count": len(contract["questions"]),
        "total_embedded_frames": sum(int(row["embedded_frame_count"]) for row in rows),
        "total_artifact_size_bytes": sum(int(row["artifact_size_bytes"]) for row in rows),
        "total_per_video_indexing_time_sec_excluding_shared_model_load": sum(
            float(row["total_indexing_time_sec"]) for row in rows
        ),
        "per_video": rows,
    }
    atomic_write_json(index_root / "index_manifest.json", payload)
    return payload


def build_indexes(
    *, data_root: Path, index_root: Path, cache_root: Path,
    video_manifest_path: Path, question_manifest_path: Path,
    readiness_path: Path, config_path: Path, device_name: str,
    dry_run: bool, reuse_yki08_diagnostic: bool,
) -> dict[str, Any]:
    contract = load_formal_contract(
        data_root=data_root, video_manifest_path=video_manifest_path,
        question_manifest_path=question_manifest_path,
        readiness_path=readiness_path, config_path=config_path,
    )
    valid: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for video in contract["videos"].values():
        passed, metadata, reason = validate_index(
            index_root=index_root, video=video,
            video_manifest_sha256=contract["video_manifest_sha256"],
        )
        if passed:
            valid.append(metadata or {})
        else:
            if reason != "index artifact or metadata missing":
                invalid.append({"video_id": video["video_id"], "reason": str(reason)})
            pending.append(video)
    result = {
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "target_video_count": 20,
        "valid_index_count": len(valid),
        "pending_index_count": len(pending),
        "invalid_index_count": len(invalid),
        "configuration": cradio_frozen_configuration(),
        "estimate": estimate_indexing(contract),
        "dry_run": dry_run,
        "model_loaded": False,
        "visual_encoding_calls": 0,
    }
    if dry_run:
        return result

    index_root.mkdir(parents=True, exist_ok=True)
    for row in invalid:
        _quarantine_index(index_root, row["video_id"], row["reason"])
    if reuse_yki08_diagnostic:
        yki = next((row for row in pending if row["video_id"] == "pasadena/YKI08"), None)
        if yki is not None:
            imported = import_yki08_diagnostic_index(
                index_root=index_root, video=yki,
                video_manifest_sha256=contract["video_manifest_sha256"],
                diagnostic_index=DEFAULT_YKI08_DIAGNOSTIC_INDEX,
                diagnostic_result=DEFAULT_YKI08_DIAGNOSTIC_RESULT,
            )
            valid.append(imported)
            pending.remove(yki)
    if not pending:
        manifest = write_index_manifest(
            index_root=index_root, contract=contract, metadata_rows=valid
        )
        return {
            **result, "valid_index_count": 20, "pending_index_count": 0,
            "index_manifest_path": str(index_root / "index_manifest.json"),
            "index_manifest": manifest,
        }

    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    import torch
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal C-RADIO indexing")
    device = torch.device(device_name)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < INDEX_MINIMUM_FREE_VRAM_BYTES:
        raise RuntimeError(
            f"C-RADIO indexing requires 12 GiB free VRAM; observed {free_bytes}"
        )
    checkpoint_path = Path(hf_hub_download(
        repo_id=RADIO_HF_REPO, filename=RADIO_CHECKPOINT_FILENAME,
        revision=RADIO_HF_REVISION,
    ))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = None
    session = {
        "schema_version": "egopolice-b1-cradio-index-session-v1",
        "started_at": utc_now(),
        "configuration": cradio_frozen_configuration(),
        "video_manifest_sha256": contract["video_manifest_sha256"],
        "starting_pending_video_ids": [row["video_id"] for row in pending],
        "gpu": {"device": str(device), "free_bytes": free_bytes, "total_bytes": total_bytes},
        "completed_video_ids": [],
        "visual_encoding_calls": 0,
        "status": "starting",
    }
    session_path = index_root / "sessions" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}.json"
    atomic_write_json(session_path, session)
    try:
        model, _, warnings, model_load_sec = _load_model(device, checkpoint_path)
        session.update({
            "status": "running", "model_load_count": 1,
            "model_load_time_sec": model_load_sec, "warnings": warnings,
        })
        atomic_write_json(session_path, session)
        for video in pending:
            probe = _ffprobe(Path(video["path"]))
            if not math.isclose(probe["duration_sec"], video["duration_sec"], abs_tol=1e-6):
                raise RuntimeError(f"Video duration changed: {video['video_id']}")
            timestamps = one_fps_timestamps(probe["duration_sec"])
            frame_indices = timestamps_to_frame_indices(
                timestamps, average_fps=probe["average_fps"],
                frame_count=probe["frame_count"],
            )
            embeddings, timing = _decode_and_embed(
                model=model, video_path=Path(video["path"]), timestamps=timestamps,
                frame_indices=frame_indices, batch_size=INDEX_BATCH_SIZE, device=device,
            )
            _array_contract(
                arrays={"embeddings": embeddings, "timestamps_sec": timestamps,
                        "frame_indices": frame_indices},
                duration_sec=video["duration_sec"],
            )
            metadata = _write_index(
                index_root=index_root, video=video,
                video_manifest_sha256=contract["video_manifest_sha256"],
                embeddings=embeddings, timestamps=timestamps,
                frame_indices=frame_indices, timing=timing,
                source={"type": "formal_visual_encoding", "visual_reencoding_performed": True},
                peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)),
            )
            valid.append(metadata)
            session["completed_video_ids"].append(video["video_id"])
            session["visual_encoding_calls"] += int(timing["visual_forward_calls"])
            atomic_write_json(session_path, session)
        session["status"] = "completed"
        session["ended_at"] = utc_now()
        session["peak_gpu_allocated_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        atomic_write_json(session_path, session)
    except Exception as exc:
        session["status"] = "failed"
        session["ended_at"] = utc_now()
        session["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(session_path, session)
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

    passed = []
    for video in contract["videos"].values():
        ok, metadata, reason = validate_index(
            index_root=index_root, video=video,
            video_manifest_sha256=contract["video_manifest_sha256"],
        )
        if not ok:
            raise RuntimeError(f"Post-build index validation failed for {video['video_id']}: {reason}")
        passed.append(metadata)
    manifest = write_index_manifest(
        index_root=index_root, contract=contract,
        metadata_rows=[row for row in passed if row is not None],
    )
    return {
        **result, "valid_index_count": len(passed), "pending_index_count": 0,
        "model_loaded": True, "model_load_count": 1,
        "visual_encoding_calls": session["visual_encoding_calls"],
        "session_path": str(session_path),
        "index_manifest_path": str(index_root / "index_manifest.json"),
        "index_manifest": manifest,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable frozen formal EgoPolice C-RADIO 1 FPS indexes"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-compatible-yki08-diagnostic", action="store_true")
    parser.add_argument("--video-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation20_v1.json")
    parser.add_argument("--question-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation_questions_v1.json")
    parser.add_argument("--readiness", type=Path, default=ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines/egopolice_b0.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_indexes(
        data_root=args.data_root, index_root=args.index_root,
        cache_root=args.cache_root, video_manifest_path=args.video_manifest,
        question_manifest_path=args.question_manifest,
        readiness_path=args.readiness, config_path=args.config,
        device_name=args.device, dry_run=args.dry_run,
        reuse_yki08_diagnostic=args.reuse_compatible_yki08_diagnostic,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
