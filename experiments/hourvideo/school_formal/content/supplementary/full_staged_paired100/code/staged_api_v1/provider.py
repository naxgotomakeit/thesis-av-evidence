"""Native Anthropic tool adapter for the three post-Planner stages."""
from __future__ import annotations

import json
import hashlib
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from direct_api_v1.anthropic_provider import load_existing_anthropic_credentials, normalise_tool_input
from direct_api_v1.pricing import AnthropicCacheAwarePricing
from .contracts import PROMPTS, TOOL_NAMES, StageContractError, tool_for
from .store import StagedStore


@dataclass(frozen=True)
class StageResult:
    stage: str
    payload: dict[str, Any]
    attempt_id: str
    response_id: str | None
    usage: dict[str, int]
    cost: dict[str, float]
    latency_sec: float
    raw_metadata: dict[str, Any]


class StageProviderError(RuntimeError):
    pass


class _HttpMessages:
    def __init__(self, api_key: str, timeout_sec: float) -> None:
        self.api_key, self.timeout_sec = api_key, timeout_sec

    def create(self, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "anthropic-beta": "prompt-caching-2024-07-31"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"), object_hook=lambda value: SimpleNamespace(**value))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


_SENSITIVE_RESPONSE_KEYS = {
    "api_key", "authorization", "credential", "credentials", "headers",
    "request_headers", "token", "x-api-key",
}


def _provider_response_tree(value: Any) -> Any:
    """Convert only known provider/fallback response shapes to JSON data.

    Anthropic SDK messages expose ``model_dump`` and the SDK-free fallback
    recursively uses ``SimpleNamespace``.  Mapping/list/scalar nodes cover the
    resulting trees.  Unknown arbitrary objects are rejected instead of being
    introspected or silently stringified.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            result[name] = "[REDACTED]" if name.lower() in _SENSITIVE_RESPONSE_KEYS else _provider_response_tree(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_provider_response_tree(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return _provider_response_tree(vars(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _provider_response_tree(model_dump(mode="json"))
        except TypeError:
            return _provider_response_tree(model_dump())
    raise StageProviderError(f"unsupported provider response node: {type(value).__name__}")


def serialise_provider_response(response: Any) -> tuple[Any, str]:
    tree = _provider_response_tree(response)
    encoded = json.dumps(tree, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return tree, hashlib.sha256(encoded).hexdigest()


class AnthropicStagedProvider:
    def __init__(self, *, model: str, max_tokens_by_stage: Mapping[str, int], temperature: float, timeout_sec: float,
                 max_transport_retries: int, pricing: AnthropicCacheAwarePricing, store: StagedStore,
                 credential_env_path: Path | None = None, client: Any | None = None) -> None:
        self.model = model
        self.max_tokens = dict(max_tokens_by_stage)
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.max_transport_retries = max_transport_retries
        self.pricing = pricing
        self.store = store
        if client is None:
            if credential_env_path is None:
                raise StageProviderError("credential path required for real provider")
            key = load_existing_anthropic_credentials(credential_env_path)
            try:
                import anthropic
                # All physical retries belong to the outer, journalled loop
                # below.  The SDK must never make an invisible retry.
                client = anthropic.Anthropic(
                    api_key=key, timeout=timeout_sec, max_retries=0,
                ).messages
            except ModuleNotFoundError:
                client = _HttpMessages(key, timeout_sec)
        self.client = client

    @staticmethod
    def _content(response: Any) -> list[Any]:
        return list(_field(response, "content", []) or [])

    def _request(self, *, stage: str, user_content: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model, "max_tokens": int(self.max_tokens[stage]), "temperature": self.temperature,
            "system": [{"type": "text", "text": PROMPTS[stage], "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_content}],
            "tools": [tool_for(stage, schema)], "tool_choice": {"type": "tool", "name": TOOL_NAMES[stage]},
        }

    def call(self, *, question_id: str, stage: str, stage_call_index: int, validation_retry_index: int,
             user_content: list[dict[str, Any]], schema: dict[str, Any]) -> StageResult:
        if stage not in PROMPTS:
            raise StageProviderError(f"unknown stage: {stage}")
        request = self._request(stage=stage, user_content=user_content, schema=schema)
        last_error: Exception | None = None
        for transport_index in range(self.max_transport_retries + 1):
            attempt_id = str(uuid.uuid4())
            common = {
                "question_id": question_id, "stage": stage, "stage_call_index": stage_call_index,
                "validation_retry_index": validation_retry_index, "transport_retry_index": transport_index,
                "is_validation_retry": validation_retry_index > 0, "is_transport_retry": transport_index > 0,
                "attempt_id": attempt_id, "model": self.model,
            }
            self.store.request_start(common)
            started = time.perf_counter()
            try:
                if isinstance(self.client, _HttpMessages):
                    response = self.client.create(request)
                else:
                    response = self.client.create(**request)
            except Exception as error:
                latency = time.perf_counter() - started
                self.store.attempt_end({**common, "provider_status": "transport_error", "provider_error_category": type(error).__name__,
                    "ordinary_input_tokens": None, "cache_creation_input_tokens": None, "cache_read_input_tokens": None, "output_tokens": None,
                    "cache_aware_usd": None, "authoritative_usage_available": False,
                    "provider_outcome_unknown": True, "latency_sec": latency, "response_id": None})
                last_error = error
                if transport_index < self.max_transport_retries:
                    continue
                raise StageProviderError(f"provider transport exhausted: {type(error).__name__}") from error

            latency = time.perf_counter() - started
            # Critical ordering invariant: preserve the complete native
            # response before reading usage, normalising tool input, or
            # applying either envelope or frozen V6.6.2 semantic validation.
            response_tree, response_sha256 = serialise_provider_response(response)
            self.store.provider_response({
                **common,
                "provider_status": "response_received",
                "response_runtime_type": type(response).__name__,
                "response_sha256": response_sha256,
                "response": response_tree,
                "latency_sec_at_receipt": latency,
            })
            usage_obj = _field(response, "usage", {})
            usage = {
                "ordinary_input_tokens": int(_field(usage_obj, "input_tokens", 0) or 0),
                "cache_creation_input_tokens": int(_field(usage_obj, "cache_creation_input_tokens", 0) or 0),
                "cache_read_input_tokens": int(_field(usage_obj, "cache_read_input_tokens", 0) or 0),
                "output_tokens": int(_field(usage_obj, "output_tokens", 0) or 0),
            }
            cost = self.pricing.breakdown(**usage)
            blocks = self._content(response)
            tool_blocks = [block for block in blocks if _field(block, "type") == "tool_use"]
            non_tools = [block for block in blocks if _field(block, "type") != "tool_use"]
            native_input = _field(tool_blocks[0], "input") if len(tool_blocks) == 1 else None
            normalized = normalise_tool_input(native_input)
            raw_metadata = {
                "content_block_types": [_field(block, "type") for block in blocks],
                "tool_use_count": len(tool_blocks), "tool_name": _field(tool_blocks[0], "name") if len(tool_blocks) == 1 else None,
                "tool_input_runtime_type": type(native_input).__name__, "tool_input_normalised": normalized is not None,
                "normalised_tool_input": normalized, "stop_reason": _field(response, "stop_reason"),
            }
            # Critical invariant: provider result and authoritative cost are durable before local validation.
            self.store.attempt_end({**common, "provider_status": "response_received", "provider_error_category": None,
                **usage, "cache_aware_usd": cost["total_cache_aware_usd"], "cost_breakdown": cost,
                "latency_sec": latency, "response_id": str(_field(response, "id", "")) or None, "response_metadata": raw_metadata})
            try:
                if len(tool_blocks) != 1: raise StageContractError("exactly one tool_use is required")
                if non_tools: raise StageContractError("tool plus text/other content is forbidden")
                if _field(tool_blocks[0], "name") != TOOL_NAMES[stage]: raise StageContractError("wrong stage tool")
                if normalized is None: raise StageContractError("tool input cannot be normalized")
                # The adapter validates only the Anthropic action envelope.
                # Scientific payload normalisation and validation remain the
                # frozen V6.6.2 backend's responsibility in legacy_bridge.
                parsed = dict(normalized)
            except StageContractError as error:
                self.store.controller_result({**common, "accepted": False, "rejection_category": "stage_contract_invalid", "reason": str(error)})
                raise
            self.store.controller_result({**common, "accepted": True, "rejection_category": None})
            return StageResult(stage, parsed, attempt_id, str(_field(response, "id", "")) or None, usage, cost, latency, raw_metadata)
        raise StageProviderError("unreachable provider state") from last_error
