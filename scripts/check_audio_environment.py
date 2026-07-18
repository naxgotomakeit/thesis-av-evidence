#!/usr/bin/env python3
"""Task 3.0: model-free full-audio decoding sanity check for case 00002_7."""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_PATH = Path(r"D:\ThesisData\EgoSound\data\Ego4d\audios\00002.wav")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audio_index" / "sanity_check_00002"
REFERENCE_START = 13.0
REFERENCE_END = 14.0
REFERENCE_LABEL = "Dataset-provided QA reference interval"
REFERENCE_DISCLAIMER = "not model prediction; not retrieval result; not gold boundary"


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM, got sample width {sample_width}")
        raw = wav.readframes(frame_count)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels)
    mono = samples.mean(axis=1) / 32768.0
    return mono, sample_rate, channels, sample_width


def canvas(title: str, width: int = 1280, height: int = 420):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 8), title, fill="black")
    draw.text((55, 21), REFERENCE_LABEL, fill=(150, 25, 80))
    draw.text((55, 34), f"({REFERENCE_DISCLAIMER})", fill=(150, 25, 80))
    return image, draw, (55, 55, width - 25, height - 45)


def x_for_time(seconds: float, duration: float, bounds) -> int:
    left, _, right, _ = bounds
    return left + round(max(0.0, min(duration, seconds)) / duration * (right - left))


def add_reference_overlay(draw: ImageDraw.ImageDraw, bounds, duration: float):
    left, top, right, bottom = bounds
    x0, x1 = x_for_time(REFERENCE_START, duration, bounds), x_for_time(REFERENCE_END, duration, bounds)
    draw.rectangle((x0, top, x1, bottom), fill=(255, 220, 235))
    draw.rectangle((left, top, right, bottom), outline="black")
    draw.text((left, bottom + 8), "0s", fill="black")
    draw.text((right - 50, bottom + 8), f"{duration:.1f}s", fill="black")


def save_waveform(samples: np.ndarray, sample_rate: int, path: Path):
    duration = len(samples) / sample_rate
    image, draw, bounds = canvas("Full-audio waveform (question-independent analysis)")
    add_reference_overlay(draw, bounds, duration)
    left, top, right, bottom = bounds
    width = right - left
    usable = samples[: (len(samples) // width) * width]
    envelope = usable.reshape(width, -1)
    lows, highs = envelope.min(axis=1), envelope.max(axis=1)
    middle, scale = (top + bottom) / 2, (bottom - top) * 0.46
    for x, (low, high) in enumerate(zip(lows, highs)):
        draw.line((left + x, middle - high * scale, left + x, middle - low * scale), fill=(30, 80, 170))
    image.save(path)


def frame_rms(samples: np.ndarray, sample_rate: int, frame_ms=25, hop_ms=10):
    frame = round(sample_rate * frame_ms / 1000)
    hop = round(sample_rate * hop_ms / 1000)
    count = 1 + max(0, (len(samples) - frame) // hop)
    values = np.empty(count, dtype=np.float32)
    for index in range(count):
        chunk = samples[index * hop:index * hop + frame]
        values[index] = math.sqrt(float(np.mean(chunk * chunk)))
    times = (np.arange(count) * hop + frame / 2) / sample_rate
    return times, values


def save_curve(times, values, duration, path):
    image, draw, bounds = canvas("Full-audio short-time RMS (25 ms frame, 10 ms hop)")
    add_reference_overlay(draw, bounds, duration)
    left, top, right, bottom = bounds
    maximum = max(float(values.max()), 1e-9)
    points = [(x_for_time(float(t), duration, bounds), bottom - int(float(v) / maximum * (bottom - top))) for t, v in zip(times, values)]
    if len(points) > 1:
        draw.line(points, fill=(25, 125, 65), width=1)
    image.save(path)


def save_spectrogram(samples, sample_rate, duration, path):
    frequencies, times, power = signal.spectrogram(samples, fs=sample_rate, window="hann", nperseg=512,
                                                   noverlap=384, detrend=False, scaling="spectrum", mode="psd")
    log_power = 10.0 * np.log10(np.maximum(power, 1e-12))
    low, high = np.percentile(log_power, [5, 99])
    normalized = np.clip((log_power - low) / max(high - low, 1e-9), 0, 1)
    rgb = np.empty((normalized.shape[0], normalized.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (255 * np.clip(1.7 * normalized - 0.35, 0, 1)).astype(np.uint8)
    rgb[..., 1] = (255 * np.clip(1.8 - np.abs(normalized - 0.55) * 3.0, 0, 1)).astype(np.uint8)
    rgb[..., 2] = (255 * np.clip(1.2 - 1.4 * normalized, 0, 1)).astype(np.uint8)
    spec = Image.fromarray(np.flipud(rgb), "RGB")
    image, draw, bounds = canvas("Full-audio standard spectrogram (Hann 32 ms, 8 ms hop; 0-8 kHz)")
    add_reference_overlay(draw, bounds, duration)
    left, top, right, bottom = bounds
    spec = spec.resize((right - left, bottom - top), Image.Resampling.BILINEAR)
    image.paste(spec, (left, top))
    # Reapply translucent-looking reference using an outline and hatch so spectrogram remains visible.
    x0, x1 = x_for_time(REFERENCE_START, duration, bounds), x_for_time(REFERENCE_END, duration, bounds)
    draw.rectangle((x0, top, x1, bottom), outline=(255, 80, 150), width=3)
    for x in range(x0, x1, 8):
        draw.line((x, top, x, bottom), fill=(255, 150, 195), width=1)
    draw.rectangle((left, top, right, bottom), outline="black")
    draw.text((4, top), "8 kHz", fill="black")
    draw.text((12, bottom - 10), "0", fill="black")
    image.save(path)
    return frequencies, times, log_power


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, sample_rate, channels, sample_width = read_pcm16_mono(AUDIO_PATH)
    duration = len(samples) / sample_rate
    rms_times, rms = frame_rms(samples, sample_rate)
    save_waveform(samples, sample_rate, OUTPUT_DIR / "waveform.png")
    frequencies, spectrogram_times, log_power = save_spectrogram(samples, sample_rate, duration, OUTPUT_DIR / "spectrogram.png")
    save_curve(rms_times, rms, duration, OUTPUT_DIR / "rms_curve.png")
    metadata = {
        "case_id": "00002_7", "video_id": "00002", "audio_path": "data/Ego4d/audios/00002.wav",
        "analysis_scope": "full_audio_question_independent", "duration_seconds": duration,
        "sample_rate_hz": sample_rate, "channels": channels, "sample_width_bytes": sample_width,
        "codec": "pcm_s16le", "sample_count": len(samples),
        "rms_analysis": {"frame_ms": 25, "hop_ms": 10, "frame_count": len(rms),
                         "minimum": float(rms.min()), "maximum": float(rms.max()), "mean": float(rms.mean())},
        "spectrogram_analysis": {"window": "hann", "nperseg": 512, "noverlap": 384,
                                 "frequency_bins": len(frequencies), "time_bins": len(spectrogram_times),
                                 "frequency_range_hz": [0.0, sample_rate / 2]},
        "dataset_provided_qa_reference_interval": {
            "start": REFERENCE_START, "end": REFERENCE_END, "role": "visualization_overlay_only",
            "used_to_crop_or_construct_analysis": False,
            "label": f"{REFERENCE_LABEL} ({REFERENCE_DISCLAIMER})"
        },
        "semantic_interpretation": "None. RMS and spectrogram statistics are low-level acoustic signals only."
    }
    (OUTPUT_DIR / "audio_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "duration": duration, "sample_rate": sample_rate,
                      "channels": channels, "rms_frames": len(rms), "spectrogram_shape": list(log_power.shape)}, indent=2))


if __name__ == "__main__":
    main()
