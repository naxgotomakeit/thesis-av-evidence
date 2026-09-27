"""Deterministic scripted agent used only for framework tests/smoke plumbing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import DirectAction, parse_action
from .state import DirectSessionState


@dataclass
class FakeDirectAgent:
    scripted_actions: list[DirectAction | dict[str, Any]]
    calls: int = 0
    received_states: list[DirectSessionState] = field(default_factory=list)

    def next_action(self, session_state: DirectSessionState) -> DirectAction:
        if self.calls >= len(self.scripted_actions):
            raise RuntimeError("fake agent script exhausted")
        self.received_states.append(session_state)
        action = parse_action(self.scripted_actions[self.calls])
        self.calls += 1
        return action
