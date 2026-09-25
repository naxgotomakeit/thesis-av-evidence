"""Strict, provider-neutral actions accepted by the Direct v1 controller."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Union


class DirectActionError(ValueError):
    pass


@dataclass(frozen=True)
class InspectFramesAction:
    timestamps_sec: tuple[float, ...]
    reason: str
    action: str = "inspect_frames"

    def __post_init__(self) -> None:
        if not self.timestamps_sec:
            raise DirectActionError("inspect_frames requires at least one timestamp")
        if not self.reason.strip():
            raise DirectActionError("inspect_frames requires a reason")
        if any(not math.isfinite(value) for value in self.timestamps_sec):
            raise DirectActionError("frame timestamps must be finite")


@dataclass(frozen=True)
class FinalAnswerAction:
    selected_option_id: str
    reason: str
    action: str = "final_answer"

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise DirectActionError("final_answer requires a reason")


DirectAction = Union[InspectFramesAction, FinalAnswerAction]


def parse_action(value: DirectAction | dict[str, Any]) -> DirectAction:
    if isinstance(value, (InspectFramesAction, FinalAnswerAction)):
        return value
    if not isinstance(value, dict):
        raise DirectActionError("Direct action must be an object")
    action = value.get("action")
    if action == "inspect_frames":
        timestamps = value.get("timestamps_sec")
        if not isinstance(timestamps, list):
            raise DirectActionError("inspect_frames.timestamps_sec must be an array")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in timestamps):
            raise DirectActionError("frame timestamps must be numeric")
        return InspectFramesAction(tuple(float(item) for item in timestamps), str(value.get("reason", "")))
    if action == "final_answer":
        return FinalAnswerAction(str(value.get("selected_option_id", "")), str(value.get("reason", "")))
    raise DirectActionError(f"unsupported Direct action: {action!r}")
