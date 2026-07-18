"""Typed per-question state shared by canonical online stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    REGRESSION_REPLAY = "regression_replay"
    EXECUTE_LIVE = "execute_live"


@dataclass
class CaseState:
    """Canonical runtime state; gold/reference information is deliberately absent."""

    case_id: str
    video_id: str
    question: str
    video_duration_sec: float
    mode: ExecutionMode
    source_video_path: str | None = None
    source_audio_path: str | None = None
    available_modalities: tuple[str, ...] = ("visual", "speech", "acoustic")
    planner_output: dict[str, Any] | None = None
    deterministic_cues: dict[str, Any] = field(default_factory=dict)
    retrieval_result: dict[str, Any] | None = None
    sufficiency_result: dict[str, Any] | None = None
    fallback_decision: dict[str, Any] | None = None
    fallback_result: dict[str, Any] | None = None
    fallback_execution_count: int = 0
    evidence_packet: dict[str, Any] | None = None
    final_payload: dict[str, Any] | None = None
    preflight_result: dict[str, Any] | None = None
    final_model_output: dict[str, Any] | None = None
    validated_answer: dict[str, Any] | None = None
    timings: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)
    external_calls: dict[str, int] = field(default_factory=lambda: {"planner": 0, "fallback": 0, "final_qa": 0})

    def record(self, stage: str, detail: str, **values: Any) -> None:
        self.audit.append({"stage": stage, "detail": detail, **values})
