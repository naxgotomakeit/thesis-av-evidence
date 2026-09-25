"""Anthropic Messages adapter for the single-agent Direct v1 loop."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import time
from typing import Any
from types import SimpleNamespace
from collections.abc import Mapping
import urllib.error
import urllib.request

from .controller import AgentTurnResult
from .policy import MAX_NEW_IMAGES_PER_TURN, MAX_UNIQUE_IMAGES_PER_QUESTION, SMOKE_MAX_TOTAL_API_USD
from .prompt import DIRECT_V1_SYSTEM_PROMPT
from .pricing import AnthropicCacheAwarePricing
from .state import DirectSessionState
from .telemetry import ProviderAttemptTelemetry, utc_now


class DirectProviderError(RuntimeError):
    def __init__(self, message: str, *, attempt_records: tuple[ProviderAttemptTelemetry, ...] = ()) -> None:
        super().__init__(message)
        self.attempt_records = attempt_records


class _HttpAnthropicMessages:
    """Minimal SDK-free Messages client for the project environment's Python."""
    def __init__(self, *, api_key: str, timeout: float) -> None:
        self.api_key, self.timeout = api_key, timeout

    def create(self, **payload: Any) -> Any:
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={
                "content-type": "application/json", "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"), object_hook=lambda value: SimpleNamespace(**value))
        except urllib.error.HTTPError as error:
            raise DirectProviderError(f"Anthropic HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise DirectProviderError("Anthropic network failure") from error


def load_existing_anthropic_credentials(path: Path) -> str:
    """Load the established private env file without exposing its value."""
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ.setdefault("ANTHROPIC_API_KEY", line.split("=", 1)[1].strip().strip('"').strip("'"))
                break
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise DirectProviderError("ANTHROPIC_API_KEY unavailable through the existing private mechanism")
    return key


def direct_action_tools(max_new_images: int = 3) -> list[dict[str, Any]]:
    """Native Anthropic client tools: one schema-bounded Direct action per turn."""
    short_reason = {"type": "string", "minLength": 1, "maxLength": 240}
    tools=[]
    if max_new_images > 0:
      tools.append({"name": "inspect_frames", "description": "Request only the allowed number of new, valid video timestamps for the most informative next visual check.",
         "input_schema": {"type": "object", "additionalProperties": False,
                          "properties": {"timestamps_sec": {"type": "array", "minItems": 1, "maxItems": max_new_images, "items": {"type": "number"}}, "reason": short_reason},
                          "required": ["timestamps_sec", "reason"]}})
    tools += [
        {"name": "final_answer", "description": "Finish with exactly one answer option.",
         "input_schema": {"type": "object", "additionalProperties": False,
                          "properties": {"selected_option_id": {"type": "string", "enum": ["A", "B", "C", "D", "E"]}, "reason": short_reason},
                          "required": ["selected_option_id", "reason"]}},
    ]
    return tools


def normalise_tool_input(value: Any) -> dict[str, Any] | None:
    """Accept only native mappings or this module's HTTP fallback object shape."""
    if isinstance(value, Mapping):
        return dict(value)
    # `_HttpAnthropicMessages` uses this exact `object_hook` representation for
    # every nested JSON object, including `tool_use.input`.
    if isinstance(value, SimpleNamespace):
        return dict(vars(value))
    return None


class AnthropicDirectAgent:
    """Stateful same-conversation client; no retrieval policy exists in this adapter."""
    def __init__(self, *, credential_env_path: Path, model: str, pricing: AnthropicCacheAwarePricing | dict[str, float | str | None], max_output_tokens: int = 512, timeout_sec: float = 120.0, max_retries: int = 1, max_total_usd: float = SMOKE_MAX_TOTAL_API_USD, client: Any | None = None, lifecycle: Any | None = None) -> None:
        self.model = model
        self.pricing = pricing if isinstance(pricing, AnthropicCacheAwarePricing) else AnthropicCacheAwarePricing.from_mapping(pricing)
        self.max_output_tokens = max_output_tokens
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.max_total_usd = max_total_usd
        self.total_cache_aware_usd = 0.0
        self._history: list[dict[str, Any]] = []
        self._sent_frame_count = 0
        self._last_controller_round = 0
        self._pending_tool_use_id: str | None = None
        self._pending_correction_mode: str | None = None
        self._initialized = False
        self.lifecycle = lifecycle
        if client is None:
            key = load_existing_anthropic_credentials(credential_env_path)
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=key, timeout=timeout_sec)
            except ModuleNotFoundError:
                client = SimpleNamespace(messages=_HttpAnthropicMessages(api_key=key, timeout=timeout_sec))
        self.client = client

    @staticmethod
    def _text(response: Any) -> str:
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")

    def _system(self, state: DirectSessionState) -> list[dict[str, Any]]:
        # Map bytes are re-read verbatim and remain identically ordered across every turn.
        raw_map = Path(state.direct_input.map_path).read_text(encoding="utf-8")
        return [
            {"type": "text", "text": DIRECT_V1_SYSTEM_PROMPT},
            {"type": "text", "text": "VIDEO MAP (frozen native representation):\n" + raw_map, "cache_control": {"type": "ephemeral"}},
        ]

    def _initial_message(self, state: DirectSessionState) -> dict[str, Any]:
        question = state.direct_input.question
        options = "\n".join(f"{row['option_id']}. {row['text']}" for row in question["answer_options"])
        return {"role": "user", "content": [{"type": "text", "text": f"Question: {question['question_text']}\nOptions:\n{options}\n{self._budget_state(state)}\nReturn the next Direct action."}]}

    @staticmethod
    def _budget_state(state: DirectSessionState) -> str:
        remaining = max(0, MAX_UNIQUE_IMAGES_PER_QUESTION - state.unique_image_count)
        if not remaining:
            return f"Visual inspection state: {MAX_UNIQUE_IMAGES_PER_QUESTION} inspected; 0 remaining. No inspection is allowed; return final_answer."
        return f"Visual inspection state: {state.unique_image_count} inspected; {remaining} remaining; maximum new images this turn: {min(MAX_NEW_IMAGES_PER_TURN, remaining)}."

    def _inspection_message(self, state: DirectSessionState) -> dict[str, Any]:
        if not self._pending_tool_use_id:
            raise DirectProviderError("inspection continuation lacks prior tool-use identity")
        fresh = state.inspected_frames[self._sent_frame_count:]
        if fresh:
            lead = "New chronologically ordered original video frames follow. "
        else:
            lead = "The requested timestamps resolved only to already inspected physical frames, so no new image was transmitted. "
        content: list[dict[str, Any]] = [{"type": "text", "text": lead + self._budget_state(state) + " Reassess and return the next Direct action."}]
        for frame in fresh:
            content.append({"type": "text", "text": f"Frame timestamp={frame.resolved_timestamp_sec:.3f}s (requested {frame.requested_timestamp_sec:.3f}s)."})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(Path(frame.frame_path).read_bytes()).decode("ascii")}})
        self._sent_frame_count = len(state.inspected_frames)
        self._last_controller_round = state.rounds
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": self._pending_tool_use_id, "content": content}]}

    @staticmethod
    def _tool_action(response: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        blocks = list(response.content)
        tool_blocks = [block for block in blocks if getattr(block, "type", None) == "tool_use"]
        non_tool = [getattr(block, "type", None) for block in blocks if getattr(block, "type", None) != "tool_use"]
        tool_metadata = []
        for block in tool_blocks:
            provider_input = getattr(block, "input", None)
            normalised = normalise_tool_input(provider_input)
            tool_metadata.append({
                "id": getattr(block, "id", None), "name": getattr(block, "name", None),
                "input_runtime_type": type(provider_input).__name__,
                "input_normalised": normalised is not None,
                "normalised_input": normalised,
            })
        metadata = {
            "content_block_types": [getattr(block, "type", None) for block in blocks],
            "tool_use_count": len(tool_blocks), "tool_metadata": tool_metadata,
        }
        if len(tool_blocks) == 0:
            metadata.update({"malformed_response_category": "no_tool_use", "correction_mode": "user_message"})
            return {"_invalid_category": "no_tool_use"}, metadata, None
        if len(tool_blocks) > 1:
            metadata.update({"malformed_response_category": "multiple_tool_use", "correction_mode": "user_message"})
            return {"_invalid_category": "multiple_tool_use"}, metadata, None
        if non_tool:
            metadata.update({"malformed_response_category": "tool_plus_text", "correction_mode": "user_message"})
            return {"_invalid_category": "tool_plus_text"}, metadata, None
        tool = tool_blocks[0]
        name, provider_input = getattr(tool, "name", None), getattr(tool, "input", None)
        tool_input = normalise_tool_input(provider_input)
        metadata["tool_input_runtime_type"] = type(provider_input).__name__
        metadata["tool_input_normalised"] = tool_input is not None and not isinstance(provider_input, dict)
        if tool_input is None:
            metadata.update({"malformed_response_category": "unnormalisable_tool_input", "correction_mode": "user_message"})
            return {"_invalid_category": "unnormalisable_tool_input"}, metadata, None
        tool_identity = {"id": getattr(tool, "id", None), "name": name, "input": tool_input}
        if not isinstance(tool_identity["id"], str) or not tool_identity["id"]:
            metadata.update({"malformed_response_category": "missing_tool_use_id", "correction_mode": "user_message"})
            return {"_invalid_category": "missing_tool_use_id"}, metadata, None
        metadata["correction_mode"] = "tool_result"
        if name == "inspect_frames":
            return {"action": "inspect_frames", "timestamps_sec": tool_input.get("timestamps_sec"), "reason": tool_input.get("reason", ""), "selected_option_id": ""}, metadata, tool_identity
        if name == "final_answer":
            return {"action": "final_answer", "timestamps_sec": [], "selected_option_id": tool_input.get("selected_option_id", ""), "reason": tool_input.get("reason", "")}, metadata, tool_identity
        return {"_invalid_tool_name": name}, metadata, tool_identity

    def _estimated_next_upper_bound(self, state: DirectSessionState) -> float:
        # Count every map byte as a cache-write token, conservatively, for the first
        # request. Subsequent calls retain this safe upper bound.
        map_upper_tokens = Path(state.direct_input.map_path).stat().st_size
        return (map_upper_tokens * self.pricing.cache_write_usd_per_million + self.max_output_tokens * self.pricing.output_usd_per_million) / 1_000_000

    def _cost(self, usage: Any) -> tuple[dict[str, int | None], dict[str, float], float]:
        def optional(name: str) -> int | None:
            value = getattr(usage, name, None)
            return None if value is None else int(value)
        values = {name: optional(name) for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
        breakdown = self.pricing.breakdown(
            ordinary_input_tokens=int(values["input_tokens"] or 0),
            cache_creation_input_tokens=int(values["cache_creation_input_tokens"] or 0),
            cache_read_input_tokens=int(values["cache_read_input_tokens"] or 0),
            output_tokens=int(values["output_tokens"] or 0),
        )
        # Audit-only legacy comparison; never use it as Direct route cost.
        standard = ((int(values["input_tokens"] or 0) + int(values["cache_creation_input_tokens"] or 0) + int(values["cache_read_input_tokens"] or 0)) * self.pricing.input_usd_per_million + int(values["output_tokens"] or 0) * self.pricing.output_usd_per_million) / 1_000_000
        return values, breakdown, standard

    def next_action(self, state: DirectSessionState, correction_message: str | None = None) -> AgentTurnResult:
        if self.total_cache_aware_usd + self._estimated_next_upper_bound(state) > self.max_total_usd:
            raise DirectProviderError("hard_api_budget_would_be_exceeded")
        is_correction = correction_message is not None
        correction_mode: str | None = None
        if not self._initialized:
            self._history.append(self._initial_message(state)); self._initialized = True
        elif correction_message is not None:
            correction_mode = self._pending_correction_mode
            if correction_mode == "tool_result":
                if not self._pending_tool_use_id:
                    raise DirectProviderError("tool-linked correction lacks prior tool-use identity")
                self._history.append({"role":"user","content":[{"type":"tool_result","tool_use_id":self._pending_tool_use_id,"is_error":True,"content":[{"type":"text","text":correction_message}]}]})
            elif correction_mode == "user_message":
                self._history.append({"role": "user", "content": [{"type": "text", "text": correction_message}]})
            else:
                raise DirectProviderError("correction transport mode unavailable")
        elif state.rounds > self._last_controller_round:
            self._history.append(self._inspection_message(state))
        else:
            raise DirectProviderError("provider called without newly available controller state")
        last_error: Exception | None = None
        attempts: list[ProviderAttemptTelemetry] = []
        for attempt in range(self.max_retries + 1):
            payload = dict(
                model=self.model, max_tokens=self.max_output_tokens, temperature=0.0,
                system=self._system(state), messages=self._history,
                tools=direct_action_tools(min(MAX_NEW_IMAGES_PER_TURN, max(0, MAX_UNIQUE_IMAGES_PER_QUESTION - state.unique_image_count))), tool_choice={"type": "any"},
            )
            attempt_id = None
            if self.lifecycle is not None:
                attempt_id = self.lifecycle.provider_request_start(
                    turn_index=state.rounds + 1, attempt_index=attempt + 1,
                    provider="anthropic", model=self.model,
                    estimated_upper_bound_usd=self._estimated_next_upper_bound(state),
                )
            started = time.perf_counter()
            try:
                response = self.client.messages.create(**payload)
                if self.lifecycle is not None:
                    self.lifecycle.provider_response_received()
                values, breakdown, standard = self._cost(response.usage)
                cost = breakdown["total_cache_aware_usd"]
                self.total_cache_aware_usd += cost
                if self.total_cache_aware_usd > self.max_total_usd:
                    raise DirectProviderError("hard_api_budget_exceeded_after_response")
                action, response_metadata, tool_identity = self._tool_action(response)
                malformed_category = response_metadata.get("malformed_response_category")
                latency = time.perf_counter() - started
                attempt_record = ProviderAttemptTelemetry(
                    turn_index=state.rounds + 1, attempt_index=attempt + 1,
                    provider="anthropic", model=self.model,
                    provider_status="response_received" if attempt == 0 else f"response_received_after_retry:{attempt}",
                    response_received=True, response_timestamp_utc=utc_now(),
                    response_metadata={**response_metadata, "message_id": getattr(response, "id", None)},
                    parsed_action=action, action_type=action.get("action") if isinstance(action.get("action"), str) else None,
                    action_reason=action.get("reason") if isinstance(action.get("reason"), str) else None,
                    requested_timestamps_sec=list(action.get("timestamps_sec", [])) if isinstance(action.get("timestamps_sec", []), list) else [],
                    input_tokens=values["input_tokens"], cache_creation_input_tokens=values["cache_creation_input_tokens"],
                    cache_read_input_tokens=values["cache_read_input_tokens"], output_tokens=values["output_tokens"],
                    ordinary_input_cost_usd=breakdown["ordinary_input_usd"], cache_creation_cost_usd=breakdown["cache_creation_usd"],
                    cache_read_cost_usd=breakdown["cache_read_usd"], output_cost_usd=breakdown["output_usd"],
                    cache_aware_usd=cost, api_latency_sec=latency,
                    structural_correction=is_correction, correction_mode=correction_mode,
                    malformed_response_category=malformed_category,
                )
                attempts.append(attempt_record)
                if self.lifecycle is not None:
                    self.lifecycle.provider_attempt_end(attempt_id, attempt_record)
                if tool_identity is not None:
                    self._history.append({"role": "assistant", "content": [{"type": "tool_use", **tool_identity}]})
                    self._pending_tool_use_id = tool_identity["id"]
                    self._pending_correction_mode = "tool_result"
                elif response_metadata.get("correction_mode") == "user_message":
                    # A fixed marker retains the same conversation without replaying
                    # unsafe/free-form model content or ambiguous tool calls.
                    self._history.append({"role": "assistant", "content": [{"type": "text", "text":
                        f"[Structurally invalid Direct response: {malformed_category}.]"}]})
                    self._pending_tool_use_id = None
                    self._pending_correction_mode = "user_message"
                return AgentTurnResult(
                    action=action, provider_status="accepted" if attempt == 0 else f"accepted_after_retry:{attempt}",
                    input_tokens=values["input_tokens"], cache_creation_input_tokens=values["cache_creation_input_tokens"],
                    cache_read_input_tokens=values["cache_read_input_tokens"], output_tokens=values["output_tokens"],
                    api_latency_sec=latency, api_cost_usd=cost,
                    ordinary_input_cost_usd=breakdown["ordinary_input_usd"], cache_creation_cost_usd=breakdown["cache_creation_usd"],
                    cache_read_cost_usd=breakdown["cache_read_usd"], output_cost_usd=breakdown["output_usd"],
                    standard_rate_equivalent_usd=standard,
                    api_attempts=attempt + 1, provider_attempt_records=tuple(attempts),
                )
            except Exception as error:
                last_error = error
                if isinstance(error, DirectProviderError) and error.attempt_records:
                    attempts.extend(error.attempt_records)
                else:
                    failed_record = ProviderAttemptTelemetry(
                        turn_index=state.rounds + 1, attempt_index=attempt + 1, provider="anthropic", model=self.model,
                        provider_status=f"provider_error:{type(error).__name__}", response_received=False,
                        response_timestamp_utc=utc_now(), api_latency_sec=time.perf_counter() - started,
                        structural_correction=is_correction, correction_mode=correction_mode,
                    )
                    attempts.append(failed_record)
                    if self.lifecycle is not None:
                        self.lifecycle.provider_attempt_end(attempt_id, failed_record)
        raise DirectProviderError(f"provider_failure_after_{self.max_retries + 1}_attempts:{type(last_error).__name__}", attempt_records=tuple(attempts)) from last_error
