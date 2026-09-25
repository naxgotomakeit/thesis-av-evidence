"""Cost-neutral telemetry records for Direct v1 routes and agent turns."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnTelemetry:
    turn_index: int
    action_type: str
    requested_timestamps_sec: list[float]
    resolved_frames: list[dict[str, Any]]
    images_transmitted: int
    duplicate_requests: int
    provider_status: str
    action_reason: str | None = None
    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens: int | None = None
    api_latency_sec: float | None = None
    api_cost_usd: float | None = None
    ordinary_input_cost_usd: float | None = None
    cache_creation_cost_usd: float | None = None
    cache_read_cost_usd: float | None = None
    output_cost_usd: float | None = None
    standard_rate_equivalent_usd: float | None = None


@dataclass
class ProviderAttemptTelemetry:
    """Sanitised immutable-accounting record, persisted before action validation."""
    turn_index: int
    attempt_index: int
    provider: str
    model: str
    provider_status: str
    response_received: bool
    response_timestamp_utc: str | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    parsed_action: dict[str, Any] | None = None
    action_type: str | None = None
    action_reason: str | None = None
    requested_timestamps_sec: list[Any] = field(default_factory=list)
    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens: int | None = None
    ordinary_input_cost_usd: float | None = None
    cache_creation_cost_usd: float | None = None
    cache_read_cost_usd: float | None = None
    output_cost_usd: float | None = None
    cache_aware_usd: float | None = None
    api_latency_sec: float | None = None
    controller_validation: str | None = None
    structural_correction: bool = False
    correction_mode: str | None = None
    malformed_response_category: str | None = None


@dataclass
class RouteTelemetry:
    question_id: str
    method: str
    started_at_utc: str = field(default_factory=utc_now)
    ended_at_utc: str | None = None
    terminal_status: str | None = None
    final_prediction: str | None = None
    rounds: int = 0
    unique_images_transmitted: int = 0
    total_api_attempts: int = 0
    correction_attempts: int = 0
    total_input_tokens: int = 0
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    total_output_tokens: int = 0
    total_usd: float = 0.0
    total_ordinary_input_usd: float = 0.0
    total_cache_creation_usd: float = 0.0
    total_cache_read_usd: float = 0.0
    total_output_usd: float = 0.0
    total_standard_rate_equivalent_usd: float = 0.0
    total_modeled_api_latency_sec: float = 0.0
    route_wall_time_sec: float | None = None
    turns: list[TurnTelemetry] = field(default_factory=list)
    provider_attempts: list[ProviderAttemptTelemetry] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
