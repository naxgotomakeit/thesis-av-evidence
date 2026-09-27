from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import load_json, sha256_file, write_json


def _question(manifest_path: Path, qid: str) -> dict[str, Any]:
    doc = load_json(manifest_path)
    matches = [row for row in doc["questions"] if row["question_id"] == qid]
    if len(matches) != 1:
        raise RuntimeError(f"Question does not resolve uniquely: {qid}")
    return matches[0]


def prepare_sources(root: Path, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    frame_dir = Path(cfg["frame_dir"])
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if len(frames) != 1595 or [p.name for p in frames] != [f"frame_{i:05d}.jpg" for i in range(1595)]:
        raise RuntimeError("The frozen 1-fps frame cache is incomplete or reordered")
    for path in frames:
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Unreadable empty frame: {path}")

    sig_path, dino_path = Path(cfg["siglip_npz"]), Path(cfg["dino_npz"])
    sig_meta, dino_meta = Path(cfg["siglip_metadata"]), Path(cfg["dino_metadata"])
    for path in (Path(cfg["video_path"]), sig_path, dino_path, sig_meta, dino_meta):
        if not path.is_file():
            raise FileNotFoundError(path)
    sig = np.load(sig_path, allow_pickle=False)
    dino = np.load(dino_path, allow_pickle=False)
    if sig["frame_ids"].tolist() != dino["frame_ids"].tolist() or sig["timestamps_sec"].tolist() != dino["timestamps_sec"].tolist():
        raise RuntimeError("SigLIP and DINO frame identity differs")
    if sig["embedding"].shape != (1595, 768) or dino["embedding"].shape != (1595, 384):
        raise RuntimeError("Frozen embedding shape mismatch")

    duration = float(len(frames))
    fine_width = int(cfg["fine_duration_sec"])
    medium_width = int(cfg["medium_duration_sec"])
    fine_nodes = []
    for index, start in enumerate(range(0, len(frames), fine_width), 1):
        end = min(len(frames), start + fine_width)
        center = min(len(frames) - 1, (start + end - 1) // 2)
        fine_nodes.append({
            "fine_id": f"F{index:03d}", "start_sec": float(start), "end_sec": float(end),
            "timestamp_sec": float(center), "source_frame_path": str(frames[center]),
            "frame_index": center,
        })
    medium_nodes, pooled = [], []
    for index, start in enumerate(range(0, len(frames), medium_width), 1):
        end = min(len(frames), start + medium_width)
        child = [row for row in fine_nodes if row["start_sec"] >= start and row["start_sec"] < end]
        medium_id = f"M{index:03d}"
        vector = sig["embedding"][start:end].astype(np.float32).mean(axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        pooled.append(vector)
        medium_nodes.append({
            "medium_id": medium_id, "start_sec": float(start), "end_sec": float(end),
            "duration_sec": float(end - start), "source_fine_ids": [row["fine_id"] for row in child],
            "representative_frame_paths": [row["source_frame_path"] for row in child],
            "embedding_ref": {"artifact": "medium_siglip.float32.npy", "row_index": index - 1, "dimension": 768},
        })
        for row in child:
            row["parent_medium_id"] = medium_id
    np.save(out / "medium_siglip.float32.npy", np.stack(pooled).astype(np.float32), allow_pickle=False)
    question = _question(root / cfg["safe_question_manifest"], cfg["question_id"])
    hierarchy = {
        "schema_version": "hourvideo-single-video-smoke-hierarchy-v1",
        "video_uid": cfg["video_uid"], "duration_sec": duration,
        "segmentation_policy": {"fine_fixed_sec": fine_width, "medium_fixed_sec": medium_width, "question_independent": True},
        "fine_nodes": fine_nodes, "medium_nodes": medium_nodes,
    }
    source_audit = {
        "video_uid": cfg["video_uid"], "frame_count": len(frames), "fine_count": len(fine_nodes), "medium_count": len(medium_nodes),
        "question": question,
        "sources": {
            "video": {"path": cfg["video_path"], "sha256": sha256_file(Path(cfg["video_path"]))},
            "siglip_npz": {"path": cfg["siglip_npz"], "sha256": sha256_file(sig_path)},
            "siglip_metadata": {"path": cfg["siglip_metadata"], "sha256": sha256_file(sig_meta)},
            "dino_npz": {"path": cfg["dino_npz"], "sha256": sha256_file(dino_path)},
            "dino_metadata": {"path": cfg["dino_metadata"], "sha256": sha256_file(dino_meta)},
            "safe_question_manifest": {"path": cfg["safe_question_manifest"], "sha256": sha256_file(root / cfg["safe_question_manifest"])},
        },
        "gold_or_reference_loaded": False,
    }
    write_json(out / "source_artifact_audit.json", source_audit)
    write_json(out / "input_manifest.json", {
        "experiment": cfg["experiment"], "video_uid": cfg["video_uid"], "question_id": cfg["question_id"],
        "video_count": 1, "question_count": 1, "gold_loaded_during_construction": False,
        "local_stages": ["shared hierarchy", "Whisper ASR", "YOLOv8x-OIV7 + BoT-SORT", "local Qwen dense captions"],
        "live_call_budget": cfg["planned_live_calls"], "source_hashes": source_audit["sources"],
    })
    write_json(out / "question_input.json", question)
    write_json(out / "shared_hierarchy.json", hierarchy)
    write_json(out / "detector_job.json", {
        "video_uid": cfg["video_uid"], "frames": [str(p) for p in frames],
        "medium_intervals": [{k: row[k] for k in ("medium_id", "start_sec", "end_sec")} for row in medium_nodes],
        "detector": cfg["detector"],
    })
    return source_audit


def run_audio(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    import soundfile as sf
    import torch
    import whisper

    wav = out / "audio_16khz_mono.wav"
    command = [cfg["audio"]["ffmpeg"], "-y", "-v", "error", "-i", cfg["video_path"], "-vn", "-ac", "1", "-ar", "16000", str(wav)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0 or not wav.is_file():
        raise RuntimeError(f"ffmpeg audio extraction failed: {completed.stderr[-1000:]}")
    extraction_sec = time.perf_counter() - started
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_started = time.perf_counter()
    model = whisper.load_model(cfg["audio"]["whisper_model"], device=device, download_root=cfg["audio"]["whisper_cache"])
    model_load_sec = time.perf_counter() - model_started
    infer_started = time.perf_counter()
    waveform, sample_rate = sf.read(wav, dtype="float32", always_2d=False)
    if sample_rate != 16000 or waveform.ndim != 1:
        raise RuntimeError(f"Extracted audio contract mismatch: sr={sample_rate}, shape={waveform.shape}")
    result = model.transcribe(waveform, language=cfg["audio"]["language"], fp16=device == "cuda", word_timestamps=False, condition_on_previous_text=False, verbose=False)
    inference_sec = time.perf_counter() - infer_started
    segments = []
    for index, row in enumerate(result.get("segments", [])):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        segments.append({
            "audio_id": f"A{index + 1:04d}", "start_sec": round(float(row["start"]), 3),
            "end_sec": round(float(row["end"]), 3), "exact_transcript": text,
            "source_type": "audio_asr", "language": result.get("language"),
        })
    audit = {
        "model": f"openai-whisper-{cfg['audio']['whisper_model']}", "device": device,
        "audio_sha256": sha256_file(wav), "segment_count": len(segments),
        "ffmpeg_extraction_sec": extraction_sec, "model_load_sec": model_load_sec,
        "inference_sec": inference_sec, "api_calls": 0,
    }
    write_json(out / "audio_asr.json", {"video_uid": cfg["video_uid"], "segments": segments})
    write_json(out / "audio_cost.json", audit)
    return audit


def attach_audio(mediums: list[dict[str, Any]], audio: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        row["medium_id"]: [a for a in audio if float(a["start_sec"]) < float(row["end_sec"]) and float(a["end_sec"]) > float(row["start_sec"])]
        for row in mediums
    }
