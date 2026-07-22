from __future__ import annotations

import json
import math
import re
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


class BaselineInputError(RuntimeError):
    """Raised when B0 cannot construct a valid local input."""


def load_mcq_cases(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise BaselineInputError("EgoPolice MCQ input must be a JSON list")
    output: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BaselineInputError(f"MCQ row {ordinal} is not an object")
        question_id = str(row.get("id") or row.get("question_id") or "")
        video = str(row.get("video") or row.get("video_id") or "")
        question = str(row.get("question") or "")
        options = list(row.get("options") or [])
        answer = row.get("answer")
        if not question_id or not video or not question or len(options) != 5:
            raise BaselineInputError(f"Invalid EgoPolice MCQ row: {ordinal}")
        if not isinstance(answer, int) or not 0 <= answer <= 4:
            raise BaselineInputError(f"Invalid answer index for {question_id}")
        output.append(
            {
                "question_id": question_id,
                "video_id": video.removesuffix(".mp4"),
                "video_relative_path": video,
                "question": question,
                "options": [str(item) for item in options],
                "ground_truth_index": answer,
                "ground_truth_text": str(options[answer]),
                # Dataset clip annotations are retained for audit only. B0 never
                # supplies them to frame sampling or to the model.
                "ignored_annotation_interval": [
                    row.get("start second"),
                    row.get("end second"),
                ],
            }
        )
    return output


def uniform_timestamps(duration_sec: float, num_frames: int) -> list[float]:
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise BaselineInputError("Video duration must be positive and finite")
    if num_frames <= 0:
        raise BaselineInputError("num_frames must be positive")
    bin_width = duration_sec / num_frames
    return [float((index + 0.5) * bin_width) for index in range(num_frames)]


def evaluate_gt_exposure(
    timestamps: list[float], annotation_interval: list[Any],
) -> dict[str, Any]:
    """Evaluate temporal exposure after question-independent B0 sampling.

    This metric helper does not select or alter frames. Callers must first
    finish sampling from the full video duration, then pass the frozen
    timestamps here for post-hoc comparison with the annotation interval.
    """
    if len(annotation_interval) != 2:
        raise BaselineInputError("GT interval must contain [start, end)")
    try:
        start_sec, end_sec = (float(value) for value in annotation_interval)
    except (TypeError, ValueError) as exc:
        raise BaselineInputError("GT interval values must be numeric") from exc
    if start_sec < 0 or end_sec <= start_sec:
        raise BaselineInputError(f"Invalid GT interval: [{start_sec}, {end_sec})")
    inside = [timestamp for timestamp in timestamps if start_sec <= timestamp < end_sec]
    if inside:
        nearest_distance = 0.0
    else:
        nearest_distance = min(
            min(abs(timestamp - start_sec), abs(timestamp - end_sec))
            for timestamp in timestamps
        )
    return {
        "gt_interval_sec": [start_sec, end_sec],
        "gt_interval_hit_at_8": int(bool(inside)),
        "number_of_frames_inside_gt_interval": len(inside),
        "nearest_sample_distance_to_gt_interval_seconds": float(nearest_distance),
    }


def build_mcq_prompt(question: str, options: list[str], frame_count: int) -> str:
    if len(options) != 5:
        raise BaselineInputError("B0 requires exactly five options")
    option_lines = "\n".join(f"{index}. {text}" for index, text in enumerate(options))
    return (
        f"You are given {frame_count} uniformly sampled frames from one full video "
        "in chronological order. Answer the multiple-choice question using only "
        "those frames.\n\n"
        f"Question: {question}\n\nOptions:\n{option_lines}\n\n"
        "Return exactly one option index: 0, 1, 2, 3, or 4. "
        "Do not provide an explanation."
    )


def parse_prediction(raw_text: str) -> int:
    text = raw_text.strip()
    exact = re.fullmatch(r"(?:option\s*)?([0-4])\.?", text, flags=re.IGNORECASE)
    if exact:
        return int(exact.group(1))
    letter = re.fullmatch(r"(?:option\s*)?([A-E])\.?", text, flags=re.IGNORECASE)
    if letter:
        return ord(letter.group(1).upper()) - ord("A")
    raise BaselineInputError(f"Model did not return one unambiguous MCQ index: {raw_text!r}")


def resolve_video_path(video_root: Path, relative_path: str) -> Path:
    # Explicit components preserve nested dataset paths on Windows and POSIX.
    return video_root.joinpath(*relative_path.replace("\\", "/").split("/"))


def probe_duration(video_path: Path, ffprobe_path: str) -> float:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    if completed.returncode != 0:
        raise BaselineInputError(completed.stderr.strip() or "ffprobe failed")
    try:
        return float(completed.stdout.strip())
    except ValueError as exc:
        raise BaselineInputError("ffprobe returned an invalid duration") from exc


def _limit_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        raise BaselineInputError("max_pixels must be positive")
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / float(width * height))
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(resized, Image.Resampling.LANCZOS)


def extract_uniform_frames(
    *, video_path: Path, ffmpeg_path: str, ffprobe_path: str,
    num_frames: int, max_pixels: int,
) -> tuple[list[Image.Image], list[float], float, float]:
    started = time.perf_counter()
    duration = probe_duration(video_path, ffprobe_path)
    timestamps = uniform_timestamps(duration, num_frames)
    images: list[Image.Image] = []
    try:
        for timestamp in timestamps:
            command = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ]
            completed = subprocess.run(command, capture_output=True, check=False)
            if completed.returncode != 0 or not completed.stdout:
                message = completed.stderr.decode("utf-8", errors="replace").strip()
                raise BaselineInputError(message or f"ffmpeg failed at {timestamp:.3f}s")
            with Image.open(BytesIO(completed.stdout)) as decoded:
                image = decoded.convert("RGB").copy()
            limited = _limit_pixels(image, max_pixels)
            if limited is not image:
                image.close()
            images.append(limited)
    except Exception:
        for image in images:
            image.close()
        raise
    return images, timestamps, duration, time.perf_counter() - started
