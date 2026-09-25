from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from .access import AccessPolicy


@dataclass(frozen=True)
class SourceFrame:
    frame_index: int
    pts: int
    time_base_numerator: int
    time_base_denominator: int

    @classmethod
    def from_pts(
        cls, frame_index: int, pts: int, time_base_numerator: int, time_base_denominator: int
    ) -> "SourceFrame":
        return cls(frame_index, pts, time_base_numerator, time_base_denominator)

    @classmethod
    def from_seconds(cls, frame_index: int, seconds: str | int | float | Fraction) -> "SourceFrame":
        timestamp = _seconds_fraction(seconds)
        return cls(frame_index, timestamp.numerator, 1, timestamp.denominator)

    @property
    def timestamp(self) -> Fraction:
        return Fraction(self.pts * self.time_base_numerator, self.time_base_denominator)

    @property
    def timestamp_sec(self) -> float:
        return float(self.timestamp)


@dataclass(frozen=True)
class CandidateFrame:
    video_id: str
    target_timestamp_sec: int
    timestamp_sec: float
    frame_index: int
    source_pts: int
    source_time_base_numerator: int
    source_time_base_denominator: int
    source_video_path: str
    extracted_frame_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _seconds_fraction(value: str | int | float | Fraction) -> Fraction:
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("time value must be finite") from exc
    if result.denominator == 0:
        raise ValueError("time value must be finite")
    return result


def integer_second_targets(duration_sec: str | int | float | Fraction) -> list[int]:
    duration = _seconds_fraction(duration_sec)
    if duration <= 0:
        raise ValueError("duration_sec must be finite and positive")
    target_count = (duration.numerator + duration.denominator - 1) // duration.denominator
    return list(range(target_count))


def choose_one_fps_source_frames(
    frames: Iterable[SourceFrame], duration_sec: str | int | float | Fraction
) -> list[tuple[int, SourceFrame]]:
    ordered = list(frames)
    if not ordered:
        raise ValueError("video produced no timestamped frames")
    if any(frame.frame_index < 0 or frame.timestamp < 0 for frame in ordered):
        raise ValueError("frame indices and timestamps must be non-negative")
    if any(
        (left.timestamp, left.frame_index) > (right.timestamp, right.frame_index)
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("source frames must be in presentation-time order")

    duration = _seconds_fraction(duration_sec)
    eligible = [frame for frame in ordered if frame.timestamp < duration]
    if not eligible:
        raise ValueError("video produced no in-duration timestamped frames")
    timestamps = [frame.timestamp for frame in eligible]
    selected: list[tuple[int, SourceFrame]] = []
    seen_indices: set[int] = set()
    for target in integer_second_targets(duration_sec):
        target_timestamp = Fraction(target, 1)
        insertion = bisect.bisect_left(timestamps, target_timestamp)
        choices = []
        if insertion < len(eligible):
            choices.append(eligible[insertion])
        if insertion > 0:
            choices.append(eligible[insertion - 1])
        chosen = min(
            choices,
            key=lambda frame: (
                abs(frame.timestamp - target_timestamp),
                frame.timestamp,
                frame.frame_index,
            ),
        )
        if chosen.frame_index not in seen_indices:
            selected.append((target, chosen))
            seen_indices.add(chosen.frame_index)
    return selected


def extract_one_fps_candidates(
    video_id: str,
    video_path: str,
    duration_sec: float,
    output_dir: str | Path,
    policy: AccessPolicy,
) -> list[dict]:
    """Decode one video only; this function is never used by offline tests."""
    import av

    source_path = policy.assert_read_allowed(video_path)
    destination = Path(output_dir).absolute()
    policy.assert_write_allowed(destination)
    destination.mkdir(parents=True, exist_ok=True)

    frame_meta: list[SourceFrame] = []
    with av.open(source_path) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                continue
            time_base = frame.time_base
            metadata = SourceFrame.from_pts(
                frame_index=frame_index,
                pts=int(frame.pts),
                time_base_numerator=int(time_base.numerator),
                time_base_denominator=int(time_base.denominator),
            )
            if metadata.timestamp < 0:
                continue
            frame_meta.append(metadata)

    chosen = choose_one_fps_source_frames(frame_meta, duration_sec)
    chosen_by_index = {frame.frame_index: (target, frame) for target, frame in chosen}
    candidates: list[CandidateFrame] = []
    with av.open(source_path) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in chosen_by_index:
                continue
            target, metadata = chosen_by_index[frame_index]
            filename = f"frame_{frame_index:09d}_t{metadata.timestamp_sec:.6f}.png"
            image_path = destination / filename
            policy.assert_write_allowed(image_path)
            frame.to_image().save(image_path, format="PNG", compress_level=6)
            candidates.append(
                CandidateFrame(
                    video_id=video_id,
                    target_timestamp_sec=target,
                    timestamp_sec=metadata.timestamp_sec,
                    frame_index=frame_index,
                    source_pts=metadata.pts,
                    source_time_base_numerator=metadata.time_base_numerator,
                    source_time_base_denominator=metadata.time_base_denominator,
                    source_video_path=source_path,
                    extracted_frame_path=str(image_path),
                )
            )

    candidates.sort(key=lambda item: (item.target_timestamp_sec, item.frame_index))
    if len({item.frame_index for item in candidates}) != len(candidates):
        raise RuntimeError("1 FPS extraction produced duplicate source frame indices")
    if any(item.video_id != video_id for item in candidates):
        raise RuntimeError("cross-video candidate contamination detected")
    return [item.to_dict() for item in candidates]
