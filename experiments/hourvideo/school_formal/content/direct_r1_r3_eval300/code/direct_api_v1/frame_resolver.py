"""Read-only projection from requested seconds to original HourVideo 1-fps frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any

from direct_api_prep.evidence import sha256_file


class FrameResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedFrame:
    requested_timestamp_sec: float
    resolved_timestamp_sec: float
    frame_index: int
    frame_path: str
    expected_sha256: str
    observed_sha256: str
    duplicate_of_seen_frame: bool = False
    duplicate_of_turn_frame: bool = False

    @property
    def physical_identity(self) -> str:
        return self.frame_path


@dataclass(frozen=True)
class FrozenFrameResolver:
    video_uid: str
    duration_sec: float
    frame_directory: Path
    filename_width: int = 5

    @classmethod
    def from_hierarchy(cls, hierarchy: dict[str, Any]) -> "FrozenFrameResolver":
        duration = hierarchy.get("duration_sec")
        nodes = hierarchy.get("fine_nodes")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) < 0:
            raise FrameResolutionError("hierarchy has no valid duration_sec")
        if not isinstance(nodes, list) or not nodes:
            raise FrameResolutionError("hierarchy has no Fine-node frame projection")
        first = Path(str(nodes[0].get("source_frame_path", "")))
        match = re.fullmatch(r"frame_(\d+)\.jpg", first.name)
        if not match or not first.parent.is_dir():
            raise FrameResolutionError("cannot infer original 1-fps frame directory from hierarchy")
        return cls(str(hierarchy.get("video_uid", "")), float(duration), first.parent, len(match.group(1)))

    def _candidate_indices(self, requested: float) -> list[int]:
        if not math.isfinite(requested) or requested < 0 or requested > self.duration_sec:
            raise FrameResolutionError(f"requested timestamp outside video duration: {requested}")
        floor = int(math.floor(requested))
        ceil = int(math.ceil(requested))
        return sorted({floor, ceil}, key=lambda index: (abs(index - requested), index))

    def resolve(self, requested_timestamp_sec: float) -> ResolvedFrame:
        candidates = self._candidate_indices(float(requested_timestamp_sec))
        limit = int(math.floor(self.duration_sec))
        ordered = list(candidates)
        for distance in range(1, limit + 1):
            for index in (candidates[0] - distance, candidates[-1] + distance):
                if 0 <= index <= limit and index not in ordered:
                    ordered.append(index)
        for index in ordered:
            path = self.frame_directory / f"frame_{index:0{self.filename_width}d}.jpg"
            if path.is_file():
                expected = sha256_file(path)
                observed = sha256_file(path)
                if observed != expected:
                    raise FrameResolutionError(f"frame SHA mismatch: {path}")
                return ResolvedFrame(float(requested_timestamp_sec), float(index), index, str(path), expected, observed)
        raise FrameResolutionError(f"no valid original 1-fps frame near {requested_timestamp_sec}")
