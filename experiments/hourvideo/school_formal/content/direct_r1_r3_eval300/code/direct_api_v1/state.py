"""Session state used by the controller; no provider implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .frame_resolver import ResolvedFrame
from .maps import DirectInput


class SessionMode(str, Enum):
    NAVIGATION_AND_INSPECTION_ALLOWED = "navigation_and_inspection_allowed"
    ANSWER_ONLY_BUDGET_EXHAUSTED = "answer_only_budget_exhausted"
    TERMINAL = "terminal"


@dataclass
class DirectSessionState:
    direct_input: DirectInput
    mode: SessionMode = SessionMode.NAVIGATION_AND_INSPECTION_ALLOWED
    rounds: int = 0
    seen_frame_paths: set[str] = field(default_factory=set)
    inspected_frames: list[ResolvedFrame] = field(default_factory=list)
    final_prediction: str | None = None
    terminal_status: str | None = None
    correction_attempts: int = 0
    turn_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def question_id(self) -> str:
        return self.direct_input.question["question_id"]

    @property
    def unique_image_count(self) -> int:
        return len(self.seen_frame_paths)

    @property
    def is_terminal(self) -> bool:
        return self.mode is SessionMode.TERMINAL
