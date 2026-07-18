from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import wave
from array import array
from pathlib import Path, PurePosixPath
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
REQUIRED_MANIFEST_FIELDS = {
    "case_id", "source_subset", "video_id", "video_path", "audio_path",
    "question", "answer", "question_type", "provided_timestamp_start",
    "provided_timestamp_end", "provided_context", "evidence_source",
    "video_duration", "audio_duration", "video_has_embedded_audio",
    "media_valid", "warnings",
}


def _time_to_seconds(value: str) -> float | None:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def parse_provided_timestamp(value: Any) -> tuple[float | None, float | None, list[str]]:
    """Parse an annotation timestamp without treating it as verified evidence."""
    if value is None or not str(value).strip():
        return None, None, []
    text = str(value).strip()
    match = re.fullmatch(r"\s*(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(?:-|–|—|to)\s*(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*", text, re.IGNORECASE)
    if not match:
        return None, None, [f"unparsed_provided_timestamp:{text}"]
    start, end = _time_to_seconds(match.group(1)), _time_to_seconds(match.group(2))
    if start is None or end is None or end < start:
        return None, None, [f"invalid_provided_timestamp:{text}"]
    return start, end, []


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("annotations", "data", "questions", "items"):
            if isinstance(payload.get(key), list):
                entries = payload[key]
                break
        else:
            raise ValueError(f"No annotation list found in {path}")
    else:
        raise ValueError(f"Unsupported annotation root {type(payload).__name__} in {path}")
    return [item for item in entries if isinstance(item, dict)]


def video_id_from_annotation(entry: dict[str, Any]) -> str | None:
    raw_path = entry.get("video_path")
    if isinstance(raw_path, str) and raw_path.strip():
        return PurePosixPath(raw_path.replace("\\", "/")).stem
    qid = entry.get("question_id")
    if isinstance(qid, str):
        match = re.match(r"^(.+?)_\d+$", qid)
        return match.group(1) if match else None
    return None


def index_by_stem(paths: list[Path]) -> tuple[dict[str, Path], dict[str, list[str]]]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(paths):
        if path.stem in index:
            duplicates.setdefault(path.stem, [str(index[path.stem])]).append(str(path))
        else:
            index[path.stem] = path
    return index, duplicates


def match_media(entry: dict[str, Any], videos: dict[str, Path], audios: dict[str, Path]) -> tuple[str | None, Path | None, Path | None, list[str]]:
    warnings: list[str] = []
    video_id = video_id_from_annotation(entry)
    if not video_id:
        return None, None, None, ["cannot_determine_video_id"]
    video = videos.get(video_id)
    audio = audios.get(video_id)
    if video is None:
        warnings.append("missing_video")
    if audio is None:
        warnings.append("missing_external_audio")
    qid = entry.get("question_id")
    if isinstance(qid, str) and not qid.startswith(video_id + "_"):
        warnings.append("question_id_video_id_mismatch")
    return video_id, video, audio, warnings


def _ratio(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None


def ffprobe_media(path: Path, ffprobe: str) -> tuple[dict[str, Any], list[str]]:
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate,sample_rate,channels", "-of", "json", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, [f"ffprobe_failed:{type(exc).__name__}:{exc}"]
    if result.returncode != 0:
        detail = (result.stderr or "unknown ffprobe error").strip().replace("\n", " ")[:500]
        return {}, [f"ffprobe_failed:{detail}"]
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {}, [f"ffprobe_invalid_json:{exc}"]
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    try:
        duration = float(payload.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    metadata = {
        "duration": duration,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "fps": _ratio((video or {}).get("avg_frame_rate")) or _ratio((video or {}).get("r_frame_rate")),
        "width": (video or {}).get("width"),
        "height": (video or {}).get("height"),
        "sample_rate": int(audio["sample_rate"]) if audio and str(audio.get("sample_rate", "")).isdigit() else None,
        "channels": (audio or {}).get("channels"),
    }
    warnings = []
    if duration is None or duration <= 0:
        warnings.append("invalid_or_missing_duration")
    expected = "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "audio"
    if expected == "video" and video is None:
        warnings.append("missing_video_stream")
    if expected == "audio" and audio is None:
        warnings.append("missing_audio_stream")
    return metadata, warnings


def wav_signal_activity(path: Path, rms_threshold: float = 0.001) -> tuple[dict[str, Any], list[str]]:
    """Estimate signal-active fraction from PCM WAV frames; this is not speech detection."""
    try:
        with wave.open(str(path), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            if width not in (1, 2, 4) or rate <= 0 or frames <= 0:
                return {}, ["signal_activity_unavailable:unsupported_or_empty_wav"]
            window_frames = max(1, rate)
            active = total = 0
            rms_values: list[float] = []
            typecode = {1: "B", 2: "h", 4: "i"}[width]
            scale = float({1: 128, 2: 32768, 4: 2147483648}[width])
            while True:
                raw = wav.readframes(window_frames)
                if not raw:
                    break
                samples = array(typecode)
                samples.frombytes(raw)
                if width > 1 and __import__("sys").byteorder != "little":
                    samples.byteswap()
                values = [(x - 128) / scale if width == 1 else x / scale for x in samples]
                rms = math.sqrt(sum(x * x for x in values) / max(1, len(values)))
                rms_values.append(rms)
                total += 1
                active += int(rms >= rms_threshold)
            fraction = active / total if total else 0.0
            return {"signal_active_fraction": fraction, "mean_window_rms": statistics.fmean(rms_values) if rms_values else 0.0, "analysis_window_seconds": 1.0}, []
    except (OSError, EOFError, wave.Error) as exc:
        return {}, [f"signal_activity_unavailable:{type(exc).__name__}:{exc}"]


def relative_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), "min": min(values) if values else None, "p25": percentile(values, .25), "median": percentile(values, .5), "p75": percentile(values, .75), "p95": percentile(values, .95), "max": max(values) if values else None, "mean": statistics.fmean(values) if values else None}


def validate_manifest_row(row: dict[str, Any]) -> list[str]:
    errors = [f"missing_field:{name}" for name in sorted(REQUIRED_MANIFEST_FIELDS - row.keys())]
    if row.get("evidence_source") is not None:
        errors.append("unverified_evidence_source")
    if not isinstance(row.get("warnings"), list):
        errors.append("warnings_not_list")
    if not isinstance(row.get("media_valid"), bool):
        errors.append("media_valid_not_bool")
    return errors
