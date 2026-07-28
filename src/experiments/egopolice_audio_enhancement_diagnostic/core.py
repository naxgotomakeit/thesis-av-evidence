from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value.replace("/", "__")).strip("._-")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_checked(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def extract_audio(video: Path, output: Path, ffmpeg: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_checked([str(ffmpeg), "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)])


def decode_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    import soundfile as sf
    info = sf.info(path)
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        raise RuntimeError(f"Expected {sample_rate} Hz audio, got {sr} Hz")
    return mono, {"sample_rate": int(sr), "channels": int(data.shape[1]), "duration_sec": float(info.duration), "frames": int(info.frames)}


def load_vad_and_regions(waveform: np.ndarray, sample_rate: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    model = load_silero_vad(onnx=False)
    raw = get_speech_timestamps(torch.from_numpy(waveform), model, threshold=float(cfg["vad_threshold"]), sampling_rate=sample_rate, min_speech_duration_ms=int(cfg["minimum_speech_duration_ms"]), speech_pad_ms=int(cfg["speech_pad_ms"]), return_seconds=False)
    intervals = [{"start_sec": float(x["start"])/sample_rate, "end_sec": min(len(waveform)/sample_rate, float(x["end"])/sample_rate)} for x in raw]
    merged: list[dict[str, Any]] = []
    for item in intervals:
        if not merged or item["start_sec"] - merged[-1]["end_sec"] > float(cfg["speech_merge_gap_sec"]):
            merged.append(dict(item))
        else:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], item["end_sec"])
    return [{"segment_id": f"vad_{i:04d}", "start_sec": round(x["start_sec"], 3), "end_sec": round(x["end_sec"], 3), "duration_sec": round(x["end_sec"]-x["start_sec"], 3)} for i, x in enumerate(merged)]


def transcribe(model: Any, waveform: np.ndarray, sample_rate: int, regions: list[dict[str, Any]], cfg: dict[str, Any], *, language: str | None, task: str | None) -> list[dict[str, Any]]:
    records = []
    for region in regions:
        start = max(0, int(region["start_sec"] * sample_rate)); end = min(len(waveform), int(region["end_sec"] * sample_rate))
        clip = waveform[start:end]
        result: dict[str, Any] = {}
        status = "ok"
        try:
            kwargs = {"fp16": False, "word_timestamps": bool(cfg["whisper_word_timestamps"]), "condition_on_previous_text": False, "verbose": None}
            if language: kwargs["language"] = language
            if task: kwargs["task"] = task
            result = model.transcribe(np.asarray(clip, dtype=np.float32), **kwargs)
        except Exception as exc:
            status = f"asr_error:{type(exc).__name__}"
        text = str(result.get("text", "")).strip()
        seg_meta = result.get("segments", []) or []
        metadata = seg_meta[0] if seg_meta else {}
        record = {"segment_id": region["segment_id"], "start_sec": region["start_sec"], "end_sec": region["end_sec"], "raw_transcript": text, "detected_language": result.get("language"), "language_probability": result.get("language_probability"), "no_speech_probability": metadata.get("no_speech_prob"), "avg_logprob": metadata.get("avg_logprob"), "compression_ratio": metadata.get("compression_ratio"), "asr_status": status if not text else "ok", "fallback_used": False, "warnings": []}
        if not text: record["warnings"].append("empty_transcript")
        if result.get("language") and result.get("language") != "en": record["warnings"].append("non_english_detected")
        if len(text) < 2 and region["duration_sec"] < 0.35: record["warnings"].append("short_or_truncated")
        records.append(record)
    return records


def needs_fallback(record: dict[str, Any]) -> bool:
    text = record.get("raw_transcript", "")
    repeated = bool(text) and len(set(re.findall(r"\b\w+\b", text.lower()))) <= 2 and len(re.findall(r"\b\w+\b", text.lower())) >= 5
    low = record.get("avg_logprob") is not None and float(record["avg_logprob"]) < -1.0
    return (not text) or record.get("detected_language") not in (None, "en") or repeated or low or bool(record.get("warnings"))


def fallback_regions(records: list[dict[str, Any]], duration: float, context: float) -> list[dict[str, Any]]:
    result = []
    for r in records:
        if not needs_fallback(r):
            continue
        start = round(max(0.0, r["start_sec"] - context), 3)
        end = round(min(duration, r["end_sec"] + context), 3)
        result.append({**r, "start_sec": start, "end_sec": end, "duration_sec": round(end - start, 3), "fallback_source_start_sec": r["start_sec"], "fallback_source_end_sec": r["end_sec"]})
    return result


def target_scores(labels: list[str], scores: list[float], targets: dict[str, list[str]]) -> dict[str, float]:
    result = {}
    for name, needles in targets.items():
        hits = [score for label, score in zip(labels, scores) if any(n.lower() in label.lower() for n in needles)]
        result[name] = round(max(hits, default=0.0), 6)
    return result


def run_ast(waveform: np.ndarray, sample_rate: int, cfg: dict[str, Any], model_name: str, cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    processor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = ASTForAudioClassification.from_pretrained(model_name, cache_dir=str(cache_dir)).eval().to("cpu")
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    window = int(float(cfg["sed_window_sec"])*sample_rate); hop = int(float(cfg["sed_hop_sec"])*sample_rate)
    count = max(1, 1 + math.ceil(max(0, len(waveform)-window)/hop)); windows = []
    for i in range(count):
        start = i*hop; end = min(len(waveform), start+window); clip = waveform[start:end]
        if len(clip) < window: clip = np.pad(clip, (0, window-len(clip)))
        inputs = processor(clip, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad(): scores = torch.sigmoid(model(**inputs).logits[0]).cpu().numpy().tolist()
        top = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)[:int(cfg["sed_top_k"])]
        targets = target_scores(labels, scores, cfg["target_labels"])
        candidates = [name for name, score in targets.items() if score >= float(cfg["sed_target_threshold"])]
        if "gunshot_like_candidate" in candidates: candidates = ["gunshot_like_candidate"] + [x for x in candidates if x != "gunshot_like_candidate"]
        windows.append({"window_id": f"window_{i:06d}", "start_sec": round(start/sample_rate, 3), "end_sec": round(min(len(waveform)/sample_rate, (start+window)/sample_rate), 3), "top_labels": [{"label": l, "score": round(float(s), 6)} for l,s in top], "target_event_scores": targets, "candidate_types": candidates, "model_name": model_name, "window_size_sec": float(cfg["sed_window_sec"]), "hop_size_sec": float(cfg["sed_hop_sec"])})
    return windows, {"model_name": model_name, "num_labels": len(labels), "window_count": len(windows), "device": "cpu"}


def export_clip(wav: Path, output: Path, start: float, end: float, ffmpeg: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_checked([str(ffmpeg), "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(wav), "-c:a", "pcm_s16le", str(output)])
