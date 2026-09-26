"""Real Anthropic SDK transport boundary for ABD, with no hidden retries."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from direct_api_v1.anthropic_provider import load_existing_anthropic_credentials


SENSITIVE_KEYS = frozenset({
    "api_key", "authorization", "credential", "credentials", "headers",
    "request_headers", "token", "x-api-key",
})


class AbdProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderReceipt:
    response: Any
    latency_sec: float


def field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def provider_tree(value: Any) -> Any:
    """Convert supported response objects without introspecting secrets."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else provider_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [provider_tree(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return provider_tree(vars(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return provider_tree(model_dump(mode="json"))
        except TypeError:
            return provider_tree(model_dump())
    raise AbdProviderError(f"unsupported provider response node: {type(value).__name__}")


def serialise_provider_response(response: Any) -> tuple[Any, str]:
    tree = provider_tree(response)
    encoded = json.dumps(tree, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return tree, hashlib.sha256(encoded).hexdigest()


def authoritative_usage(response_tree: Any) -> dict[str, int] | None:
    usage = field(response_tree, "usage")
    if usage is None:
        return None
    input_tokens = field(usage, "input_tokens")
    output_tokens = field(usage, "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    try:
        values = {
            "ordinary_input_tokens": int(input_tokens),
            "cache_creation_input_tokens": int(field(usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(field(usage, "cache_read_input_tokens", 0) or 0),
            "output_tokens": int(output_tokens),
        }
    except (TypeError, ValueError):
        return None
    if any(value < 0 for value in values.values()):
        return None
    return values


class AnthropicAbdProvider:
    """One physical SDK call per ``send``; retries are explicitly disabled."""

    def __init__(self, messages_client: Any) -> None:
        self.messages_client = messages_client

    @classmethod
    def from_credentials(cls, *, credential_env_path: Path, timeout_sec: float) -> "AnthropicAbdProvider":
        key = load_existing_anthropic_credentials(credential_env_path)
        try:
            import anthropic
        except ModuleNotFoundError as error:
            raise AbdProviderError("Anthropic SDK is missing from the frozen live interpreter") from error
        # Every physical request must correspond to one durable request_start.
        # The SDK is not allowed to perform unjournalled retries.
        client = anthropic.Anthropic(api_key=key, timeout=timeout_sec, max_retries=0)
        return cls(client.messages)

    def send(self, payload: dict[str, Any]) -> ProviderReceipt:
        started = time.perf_counter()
        try:
            response = self.messages_client.create(**payload)
        except Exception as error:
            raise AbdProviderError(f"provider outcome unknown after {type(error).__name__}") from error
        return ProviderReceipt(response=response, latency_sec=time.perf_counter() - started)
