"""Single-agent Direct v1 loop with explicit, code-enforced visual budgets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time
from typing import Any, Protocol

from .actions import DirectAction, DirectActionError, FinalAnswerAction, InspectFramesAction, parse_action
from .frame_resolver import FrameResolutionError, FrozenFrameResolver, ResolvedFrame
from .state import DirectSessionState, SessionMode
from .telemetry import ProviderAttemptTelemetry, RouteTelemetry, TurnTelemetry
from .policy import MAX_NEW_IMAGES_PER_TURN, MAX_UNIQUE_IMAGES_PER_QUESTION


class DirectStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentTurnResult:
    action: DirectAction | dict
    provider_status: str = "accepted"
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
    api_attempts: int = 1
    provider_attempt_records: tuple[ProviderAttemptTelemetry, ...] = ()


class DirectAgentClient(Protocol):
    """Future providers implement this one session-preserving action method."""

    def next_action(self, session_state: DirectSessionState) -> DirectAction | dict: ...


class DirectController:
    def __init__(self, *, resolver: FrozenFrameResolver, max_turns: int = 32, enable_action_correction: bool = False, lifecycle: Any | None = None) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.resolver = resolver
        self.max_turns = max_turns
        self.enable_action_correction = enable_action_correction
        self.lifecycle = lifecycle

    def _record_validation(self, state: DirectSessionState, validation: str, action: Any = None,
                           metadata: AgentTurnResult | None = None) -> None:
        """Persist controller disposition before any corresponding state mutation."""
        if self.lifecycle is not None:
            self.lifecycle.controller_validation(
                validation=validation, state=state, action=action,
                provider_attempts=() if metadata is None else metadata.provider_attempt_records,
            )

    @staticmethod
    def _resolved_dict(frame: ResolvedFrame) -> dict:
        return asdict(frame)

    def _close(self, state: DirectSessionState, telemetry: RouteTelemetry, status: str) -> None:
        state.mode = SessionMode.TERMINAL
        state.terminal_status = status
        telemetry.terminal_status = status
        telemetry.final_prediction = state.final_prediction
        telemetry.rounds = state.rounds
        telemetry.unique_images_transmitted = state.unique_image_count
        telemetry.correction_attempts = state.correction_attempts
        telemetry.ended_at_utc = datetime.now(timezone.utc).isoformat()

    def _append_turn(self, state: DirectSessionState, telemetry: RouteTelemetry, turn: TurnTelemetry) -> None:
        telemetry.turns.append(turn)
        telemetry.total_input_tokens += int(turn.input_tokens or 0)
        telemetry.total_cache_creation_input_tokens += int(turn.cache_creation_input_tokens or 0)
        telemetry.total_cache_read_input_tokens += int(turn.cache_read_input_tokens or 0)
        telemetry.total_output_tokens += int(turn.output_tokens or 0)
        telemetry.total_modeled_api_latency_sec += float(turn.api_latency_sec or 0.0)
        telemetry.total_usd += float(turn.api_cost_usd or 0.0)
        telemetry.total_ordinary_input_usd += float(turn.ordinary_input_cost_usd or 0.0)
        telemetry.total_cache_creation_usd += float(turn.cache_creation_cost_usd or 0.0)
        telemetry.total_cache_read_usd += float(turn.cache_read_cost_usd or 0.0)
        telemetry.total_output_usd += float(turn.output_cost_usd or 0.0)
        telemetry.total_standard_rate_equivalent_usd += float(turn.standard_rate_equivalent_usd or 0.0)
        state.turn_history.append({
            "turn_index": turn.turn_index, "action_type": turn.action_type,
            "requested_timestamps_sec": turn.requested_timestamps_sec,
            "resolved_frames": turn.resolved_frames,
            "images_transmitted": turn.images_transmitted,
            "duplicate_requests": turn.duplicate_requests,
            "action_reason": turn.action_reason,
            "provider_status": turn.provider_status,
        })

    def _persist_provider_attempts(self, telemetry: RouteTelemetry, records: tuple[ProviderAttemptTelemetry, ...], validation: str | None = None) -> None:
        for record in records:
            if validation is not None:
                record.controller_validation = validation
            telemetry.provider_attempts.append(record)

    def _reject(self, state: DirectSessionState, telemetry: RouteTelemetry, action_type: str, requested: list[float], reason: str, metadata: AgentTurnResult | None = None, action_reason: str | None = None, terminal: bool = True, correction_exhausted: bool = False) -> None:
        self._record_validation(state, f"rejected:{reason}", {"action": action_type, "timestamps_sec": requested, "reason": action_reason}, metadata)
        state.rounds += 1
        self._append_turn(state, telemetry, self._turn_from_metadata(
            state.rounds, action_type, action_reason or "", requested, [], 0, 0, metadata,
            provider_status_override=f"rejected:{reason}",
        ))
        if metadata is not None:
            self._persist_provider_attempts(telemetry, metadata.provider_attempt_records, f"rejected:{reason}")
        if terminal:
            status = f"structural_action_correction_exhausted:{reason}" if correction_exhausted else f"invalid_action:{reason}"
            self._close(state, telemetry, status)

    def apply_action(self, state: DirectSessionState, telemetry: RouteTelemetry, value: DirectAction | dict, metadata: AgentTurnResult | None = None, correction_allowed: bool = False, correction_exhausted: bool = False) -> str | None:
        """Apply one action. Invalid actions terminally reject before image transport."""
        if state.is_terminal:
            raise DirectStateError("action after terminal Direct session")
        try:
            action = parse_action(value)
        except DirectActionError as error:
            raw = value if isinstance(value, dict) else {}
            requested = raw.get("timestamps_sec", []) if isinstance(raw.get("timestamps_sec", []), list) else []
            reason = str(raw.get("_invalid_category") or error)
            self._reject(state, telemetry, "invalid", requested, reason, metadata, str(raw.get("reason", "")), terminal=not correction_allowed, correction_exhausted=correction_exhausted); return reason if correction_allowed else None
        if isinstance(action, FinalAnswerAction):
            options = {row["option_id"] for row in state.direct_input.question["answer_options"]}
            if action.selected_option_id not in options:
                self._reject(state, telemetry, action.action, [], "invalid_option", metadata, action.reason, terminal=not correction_allowed, correction_exhausted=correction_exhausted); return "invalid_option" if correction_allowed else None
            self._record_validation(state, "accepted", action, metadata)
            state.rounds += 1
            state.final_prediction = action.selected_option_id
            self._append_turn(state, telemetry, self._turn_from_metadata(state.rounds, action.action, action.reason, [], [], 0, 0, metadata))
            if metadata is not None:
                self._persist_provider_attempts(telemetry, metadata.provider_attempt_records, "accepted")
            self._close(state, telemetry, "final_answer")
            return None

        requested = list(action.timestamps_sec)
        if state.mode is SessionMode.ANSWER_ONLY_BUDGET_EXHAUSTED:
            self._reject(state, telemetry, action.action, requested, "answer_only_budget_exhausted", metadata, action.reason, terminal=not correction_allowed, correction_exhausted=correction_exhausted); return "answer_only_budget_exhausted" if correction_allowed else None
        if len(requested) > MAX_NEW_IMAGES_PER_TURN:
            self._reject(state, telemetry, action.action, requested, "max_new_images_per_turn", metadata, action.reason, terminal=not correction_allowed, correction_exhausted=correction_exhausted); return "max_new_images_per_turn" if correction_allowed else None
        try:
            resolved = [self.resolver.resolve(timestamp) for timestamp in requested]
        except FrameResolutionError as error:
            self._reject(state, telemetry, action.action, requested, f"frame_resolution:{error}", metadata, action.reason, terminal=not correction_allowed, correction_exhausted=correction_exhausted); return f"frame_resolution:{error}" if correction_allowed else None

        turn_paths: set[str] = set()
        annotated: list[ResolvedFrame] = []
        new_frames: list[ResolvedFrame] = []
        duplicates = 0
        for frame in resolved:
            duplicate_turn = frame.frame_path in turn_paths
            duplicate_seen = frame.frame_path in state.seen_frame_paths
            annotated_frame = ResolvedFrame(
                **{**asdict(frame), "duplicate_of_seen_frame": duplicate_seen, "duplicate_of_turn_frame": duplicate_turn}
            )
            annotated.append(annotated_frame)
            turn_paths.add(frame.frame_path)
            if duplicate_seen or duplicate_turn:
                duplicates += 1
            else:
                new_frames.append(annotated_frame)
        if state.unique_image_count + len(new_frames) > MAX_UNIQUE_IMAGES_PER_QUESTION:
            self._reject(state, telemetry, action.action, requested, "global_unique_image_budget_exceeded", metadata, action.reason, terminal=not correction_allowed, correction_exhausted=correction_exhausted); return "global_unique_image_budget_exceeded" if correction_allowed else None

        self._record_validation(state, "accepted", action, metadata)
        state.rounds += 1
        # The same agent retains its logical session, but visual presentation is
        # chronological regardless of the order in which it asked for timestamps.
        annotated.sort(key=lambda frame: (frame.resolved_timestamp_sec, frame.requested_timestamp_sec, frame.frame_path))
        new_frames.sort(key=lambda frame: (frame.resolved_timestamp_sec, frame.requested_timestamp_sec, frame.frame_path))
        for frame in new_frames:
            state.seen_frame_paths.add(frame.frame_path)
            state.inspected_frames.append(frame)
        self._append_turn(state, telemetry, self._turn_from_metadata(
            state.rounds, action.action, action.reason, requested, [self._resolved_dict(frame) for frame in annotated],
            len(new_frames), duplicates, metadata,
        ))
        if metadata is not None:
            self._persist_provider_attempts(telemetry, metadata.provider_attempt_records, "accepted")
        if state.unique_image_count == MAX_UNIQUE_IMAGES_PER_QUESTION:
            state.mode = SessionMode.ANSWER_ONLY_BUDGET_EXHAUSTED
        return None

    @staticmethod
    def _turn_from_metadata(round_index: int, action_type: str, reason: str, requested: list[float], resolved: list[dict], images: int, duplicates: int, metadata: AgentTurnResult | None, provider_status_override: str | None = None) -> TurnTelemetry:
        if metadata is None:
            return TurnTelemetry(round_index, action_type, requested, resolved, images, duplicates, "accepted", reason)
        return TurnTelemetry(
            round_index, action_type, requested, resolved, images, duplicates, provider_status_override or metadata.provider_status, reason,
            metadata.input_tokens, metadata.cache_creation_input_tokens, metadata.cache_read_input_tokens,
            metadata.output_tokens, metadata.api_latency_sec, metadata.api_cost_usd,
            metadata.ordinary_input_cost_usd, metadata.cache_creation_cost_usd,
            metadata.cache_read_cost_usd, metadata.output_cost_usd,
            metadata.standard_rate_equivalent_usd,
        )

    def run(self, *, state: DirectSessionState, agent: DirectAgentClient) -> RouteTelemetry:
        """Run a same-session fake/future-provider loop. The controller never ranks frames."""
        started = time.perf_counter()
        telemetry = RouteTelemetry(question_id=state.question_id, method=state.direct_input.method)
        while not state.is_terminal:
            if state.rounds >= self.max_turns:
                self._record_validation(state, "runtime_failure:turn_limit_exhausted")
                self._close(state, telemetry, "runtime_failure:turn_limit_exhausted")
                break
            try:
                result = agent.next_action(state)
            except Exception as error:  # A provider adapter will surface its own classified status later.
                records = tuple(getattr(error, "attempt_records", ()))
                if records:
                    telemetry.total_api_attempts += len(records)
                    self._persist_provider_attempts(telemetry, records, "provider_failure")
                self._record_validation(state, "provider_failure", metadata=None)
                self._close(state, telemetry, f"runtime_failure:agent:{type(error).__name__}")
                break
            if isinstance(result, AgentTurnResult):
                telemetry.total_api_attempts += result.api_attempts
                invalid = self.apply_action(state, telemetry, result.action, result, correction_allowed=self.enable_action_correction and state.correction_attempts == 0)
                if invalid and not state.is_terminal:
                    state.correction_attempts += 1
                    telemetry.correction_attempts = state.correction_attempts
                    remaining = MAX_UNIQUE_IMAGES_PER_QUESTION - state.unique_image_count
                    message = ("Visual inspection budget is exhausted. Return final_answer with exactly one option A, B, C, D, or E." if remaining == 0 else f"Invalid action. Remaining visual budget: {remaining} unique images. Return exactly one valid action: inspect_frames with at most {min(MAX_NEW_IMAGES_PER_TURN, remaining)} timestamps, or final_answer.")
                    try:
                        corrected = agent.next_action(state, correction_message=message)
                        if isinstance(corrected, AgentTurnResult):
                            telemetry.total_api_attempts += corrected.api_attempts
                            self.apply_action(state, telemetry, corrected.action, corrected, correction_allowed=False, correction_exhausted=True)
                        else: self.apply_action(state, telemetry, corrected, correction_exhausted=True)
                    except Exception as error:
                        records = tuple(getattr(error, "attempt_records", ()))
                        if records:
                            telemetry.total_api_attempts += len(records)
                            self._persist_provider_attempts(telemetry, records, "provider_failure:correction")
                        self._record_validation(state, "provider_failure:correction", metadata=None)
                        self._close(state, telemetry, f"runtime_failure:correction:{type(error).__name__}")
            else:
                self.apply_action(state, telemetry, result)
        telemetry.route_wall_time_sec = time.perf_counter() - started
        return telemetry
