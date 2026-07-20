"""Exact frozen ffmpeg 1 FPS sampling contract."""
from __future__ import annotations
import subprocess
import time
from pathlib import Path
from typing import Any
import numpy as np

def extract_frames_1fps(
    video_path: Path, output_dir: Path, *, duration_sec: float, jpeg_quality: int,
    ffmpeg_path: Path,
) -> tuple[list[Path], np.ndarray, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg's 1 FPS filter emits one frame per complete/nearest one-second bin;
    # round() matches its deterministic count for fractional container durations.
    expected_count = int(round(duration_sec))
    frame_paths = [output_dir / f"frame_{index:05d}.jpg" for index in range(expected_count)]
    if all(path.is_file() and path.stat().st_size > 0 for path in frame_paths):
        return frame_paths, np.arange(expected_count, dtype=np.float64), {
            "cache_hit": True,
            "extraction_sec": 0.0,
            "frame_count": expected_count,
            "sampling_fps": 1.0,
        }
    started = time.perf_counter()
    quality = max(2, min(31, int(round((100 - int(jpeg_quality)) / 3.2)) + 2))
    completed = subprocess.run(
        [
            str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
            "-vf", "fps=1", "-q:v", str(quality), "-start_number", "0",
            str(output_dir / "frame_%05d.jpg"),
        ],
        check=True, capture_output=True, text=True,
    )
    written = sorted(output_dir.glob("frame_*.jpg"))
    if len(written) != expected_count:
        raise RuntimeError(
            f"1 FPS extraction incomplete: {len(written)}/{expected_count}; ffmpeg={completed.stderr[-500:]}"
        )
    return frame_paths, np.arange(expected_count, dtype=np.float64), {
        "cache_hit": False,
        "extraction_sec": time.perf_counter() - started,
        "frame_count": expected_count,
        "sampling_fps": 1.0,
    }

