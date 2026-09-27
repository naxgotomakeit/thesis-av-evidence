from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from .credentials import SecretCredential
from .providers import ProviderAttemptError, ProviderRequest, TransportResponse


UNSUPPORTED_SCHEMA_KEYS = {
    "$schema", "uniqueItems", "minItems", "maxItems", "minLength", "maxLength",
    "minimum", "maximum",
}


def project_anthropic_schema(value: Any) -> Any:
    """Match the frozen provider schema projection without changing the internal validator."""
    if isinstance(value, dict):
        return {
            key: project_anthropic_schema(item)
            for key, item in value.items()
            if key not in UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [project_anthropic_schema(item) for item in value]
    return value


def anthropic_sdk_preflight() -> dict[str, Any]:
    """Import and signature inspection only. This does not instantiate a client or send a request."""
    import anthropic

    signature = inspect.signature(anthropic.resources.messages.Messages.create)
    if "output_config" not in signature.parameters:
        raise RuntimeError("installed Anthropic SDK does not support output_config")
    return {
        "provider": "anthropic",
        "sdk_version": str(anthropic.__version__),
        "messages_create_available": True,
        "structured_output_parameter": "output_config",
        "network_request_made": False,
    }


class AnthropicPlannerTransport:
    """One Anthropic Messages request per invocation; SDK retries are explicitly disabled."""

    def __init__(
        self,
        *,
        timeout_sec: float,
        client_factory: Callable[..., Any] | None = None,
    ):
        self.timeout_sec = float(timeout_sec)
        self.client_factory = client_factory

    def _client(self, credential: SecretCredential) -> Any:
        if self.client_factory is not None:
            return self.client_factory(
                api_key=credential.reveal_for_request(),
                timeout=self.timeout_sec,
                max_retries=0,
            )
        import anthropic

        return anthropic.Anthropic(
            api_key=credential.reveal_for_request(),
            timeout=self.timeout_sec,
            max_retries=0,
        )

    def __call__(self, request: ProviderRequest, credential: SecretCredential) -> TransportResponse:
        import anthropic

        client = self._client(credential)
        try:
            response = client.messages.create(
                model=request.model,
                max_tokens=request.max_output_tokens,
                temperature=request.temperature,
                system=request.system_prompt,
                messages=[{
                    "role": "user",
                    "content": json.dumps(
                        request.payload, ensure_ascii=False, separators=(",", ":"),
                    ),
                }],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": project_anthropic_schema(request.schema),
                    }
                },
            )
        except anthropic.APITimeoutError as error:
            raise ProviderAttemptError("timeout") from error
        except anthropic.RateLimitError as error:
            raise ProviderAttemptError("rate_limited") from error
        except anthropic.APIConnectionError as error:
            raise ProviderAttemptError("transport_error") from error
        except anthropic.APIStatusError as error:
            raise ProviderAttemptError("provider_error") from error

        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        response_id = str(response.id)
        stop_reason = str(response.stop_reason or "")
        if stop_reason == "max_tokens":
            raise ProviderAttemptError(
                "max_tokens", input_tokens=input_tokens, output_tokens=output_tokens,
                provider_request_id=response_id,
            )
        raw_text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        if not raw_text.strip():
            raise ProviderAttemptError(
                "empty_response", input_tokens=input_tokens, output_tokens=output_tokens,
                provider_request_id=response_id,
            )
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ProviderAttemptError(
                "invalid_response", input_tokens=input_tokens, output_tokens=output_tokens,
                provider_request_id=response_id,
            ) from error
        if not isinstance(parsed, dict):
            raise ProviderAttemptError(
                "invalid_response", input_tokens=input_tokens, output_tokens=output_tokens,
                provider_request_id=response_id,
            )
        return TransportResponse(
            parsed_output=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_request_id=response_id,
            stop_reason=stop_reason,
        )
