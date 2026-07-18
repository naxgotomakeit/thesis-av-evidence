from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import get_window


REFERENCE_LABEL = (
    "Dataset-provided QA reference interval\n"
    "(not model prediction; not retrieval result; not gold boundary)"
)


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return float(samples) / float(sample_rate)


def decode_wav(path: str | Path, target_sample_rate: int = 16000) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    info = sf.info(path)
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    channels = int(waveform.shape[1])
    mono = waveform.mean(axis=1)
    warnings: list[str] = []
    if sample_rate != target_sample_rate:
        import librosa

        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_sample_rate)
        warnings.append(f"resampled_from_{sample_rate}_hz")
    metadata = {
        "audio_path": path.as_posix(),
        "original_sample_rate": int(sample_rate),
        "analysis_sample_rate": int(target_sample_rate),
        "channels": channels,
        "frames": int(info.frames),
        "duration": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
        "readable": True,
        "warnings": warnings,
    }
    return np.asarray(mono, dtype=np.float32), metadata


def merge_intervals(intervals: Iterable[dict[str, Any]], max_gap_sec: float) -> list[dict[str, Any]]:
    ordered = sorted(intervals, key=lambda x: (x["start_time"], x["end_time"]))
    merged: list[dict[str, Any]] = []
    for interval in ordered:
        current = dict(interval)
        if not merged or current["start_time"] - merged[-1]["end_time"] > max_gap_sec:
            merged.append(current)
        else:
            merged[-1]["end_time"] = max(merged[-1]["end_time"], current["end_time"])
            merged[-1].setdefault("source_vad_segments", []).extend(
                current.get("source_vad_segments", [])
            )
    return merged


def interval_overlap_ratio(start: float, end: float, intervals: Iterable[dict[str, Any]]) -> float:
    duration = max(0.0, end - start)
    if duration == 0:
        return 0.0
    overlap = sum(
        max(0.0, min(end, float(item["end_time"])) - max(start, float(item["start_time"])))
        for item in intervals
    )
    return float(min(1.0, overlap / duration))


def _frame_signal(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if waveform.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))
    count = 1 + int(math.ceil((waveform.size - frame_length) / hop_length))
    padded_length = (count - 1) * hop_length + frame_length
    padded = np.pad(waveform, (0, max(0, padded_length - waveform.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    return np.asarray(frames[:count], dtype=np.float32)


def compute_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    smoothing_sec: float,
) -> dict[str, np.ndarray]:
    frame_length = max(32, int(round(window_sec * sample_rate)))
    hop_length = max(1, int(round(hop_sec * sample_rate)))
    frames = _frame_signal(waveform, frame_length, hop_length)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    window = get_window("hann", frame_length, fftbins=True).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    normalized = magnitude / (magnitude.sum(axis=1, keepdims=True) + 1e-10)
    log_magnitude = np.log1p(100.0 * magnitude)
    flux = np.zeros(len(frames), dtype=np.float64)
    log_change = np.zeros(len(frames), dtype=np.float64)
    rms_change = np.zeros(len(frames), dtype=np.float64)
    if len(frames) > 1:
        flux[1:] = np.sqrt(np.sum(np.maximum(normalized[1:] - normalized[:-1], 0.0) ** 2, axis=1))
        log_change[1:] = np.mean(np.abs(log_magnitude[1:] - log_magnitude[:-1]), axis=1)
        rms_change[1:] = np.abs(np.diff(np.log(rms + 1e-8)))

    def robust_scale(values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return np.maximum(0.0, (values - median) / (1.4826 * mad + 1e-8))

    raw_change = 0.4 * robust_scale(flux) + 0.4 * robust_scale(log_change) + 0.2 * robust_scale(rms_change)
    smooth_frames = min(len(raw_change), max(1, int(round(smoothing_sec / hop_sec))))
    kernel = np.ones(smooth_frames, dtype=np.float64) / smooth_frames
    smoothed = np.convolve(raw_change, kernel, mode="same")
    times = np.minimum(
        np.arange(len(frames), dtype=np.float64) * hop_sec + window_sec / 2,
        waveform.size / sample_rate,
    )
    return {
        "times": times,
        "rms": rms,
        "spectral_flux": flux,
        "log_spectral_change": log_change,
        "raw_change": raw_change,
        "smoothed_change": smoothed,
    }


def propose_acoustic_boundaries(
    times: np.ndarray,
    change: np.ndarray,
    duration: float,
    minimum_region_duration_sec: float,
    method: str = "median_mad",
    mad_multiplier: float = 2.75,
) -> tuple[list[float], float]:
    if method != "median_mad":
        raise ValueError(f"unsupported boundary threshold method: {method}")
    if len(change) < 3 or duration <= minimum_region_duration_sec:
        return [0.0, float(duration)], float("inf")
    median = float(np.median(change))
    mad = float(np.median(np.abs(change - median)))
    threshold = median + mad_multiplier * 1.4826 * mad
    peaks = [
        float(times[i])
        for i in range(1, len(change) - 1)
        if change[i] >= threshold and change[i] >= change[i - 1] and change[i] > change[i + 1]
    ]
    boundaries = [0.0]
    for candidate in sorted(peaks):
        if candidate - boundaries[-1] >= minimum_region_duration_sec and duration - candidate >= minimum_region_duration_sec:
            boundaries.append(candidate)
    boundaries.append(float(duration))
    return boundaries, threshold


def build_acoustic_regions(
    features: dict[str, np.ndarray],
    boundaries: list[float],
    speech_regions: list[dict[str, Any]],
    low_information_quantile: float,
) -> list[dict[str, Any]]:
    times = features["times"]
    rms = features["rms"]
    global_cutoff = float(np.quantile(rms, low_information_quantile)) if len(rms) else 0.0
    regions = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (times >= start) & ((times < end) if index < len(boundaries) - 2 else (times <= end))
        local_rms = rms[mask] if np.any(mask) else np.array([0.0])
        local_change = features["smoothed_change"][mask] if np.any(mask) else np.array([0.0])
        mean_rms = float(np.mean(local_rms))
        regions.append(
            {
                "acoustic_region_id": f"acoustic_{index:04d}",
                "start_time": round(float(start), 3),
                "end_time": round(float(end), 3),
                "duration": round(float(end - start), 3),
                "mean_rms": mean_rms,
                "peak_rms": float(np.max(local_rms)),
                "mean_spectral_change_score": float(np.mean(local_change)),
                "speech_overlap_ratio": round(interval_overlap_ratio(start, end, speech_regions), 4),
                "low_information_or_silence": bool(mean_rms <= global_cutoff),
                "warnings": [],
            }
        )
    return regions


def validate_speech_region(region: dict[str, Any]) -> bool:
    required = {"speech_region_id", "start_time", "end_time", "duration", "transcript", "asr_language", "vad_probability_summary", "warnings"}
    return required.issubset(region) and region["end_time"] >= region["start_time"]


def validate_transcript_segment(segment: dict[str, Any]) -> bool:
    required = {"transcript_segment_id", "speech_region_id", "start_time", "end_time", "text", "language", "whisper_metadata", "warnings"}
    return required.issubset(segment) and segment["end_time"] >= segment["start_time"]


def validate_acoustic_region(region: dict[str, Any]) -> bool:
    required = {"acoustic_region_id", "start_time", "end_time", "duration", "mean_rms", "peak_rms", "mean_spectral_change_score", "speech_overlap_ratio", "low_information_or_silence", "warnings"}
    return required.issubset(region) and region["end_time"] >= region["start_time"]


def validate_complete_coverage(regions: list[dict[str, Any]], duration: float, tolerance: float = 1e-3) -> bool:
    if not regions:
        return duration == 0
    if abs(float(regions[0]["start_time"])) > tolerance or abs(float(regions[-1]["end_time"]) - duration) > tolerance:
        return False
    return all(abs(float(a["end_time"]) - float(b["start_time"])) <= tolerance for a, b in zip(regions[:-1], regions[1:]))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
