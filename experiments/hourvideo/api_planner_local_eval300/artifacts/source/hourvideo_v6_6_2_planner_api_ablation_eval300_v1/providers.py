from __future__ import annotations

import time
import uuid
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from .budget import BudgetGuard
from .credentials import SecretCredential, load_credential
from .pricing import CostEstimate, PricingCatalog
from .telemetry import AttemptTelemetryJournal


class UnsupportedProviderError(RuntimeError):
    pass


class ProviderCallFailed(RuntimeError):
    pass


class ProviderAttemptError(RuntimeError):
    """Sanitized transport failure with optional billable usage."""

    def __init__(
        self,
        category: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        provider_request_id: str | None = None,
    ):
        safe_category = category if category in {
            "timeout", "transport_error", "provider_error", "rate_limited",
            "max_tokens", "invalid_response", "empty_response",
        } else "provider_error"
        super().__init__(f"planner provider attempt failed: {safe_category}")
        self.category = safe_category
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.provider_request_id = provider_request_id


@dataclass(frozen=True)
class ProviderRequest:
    logical_call_id: str
    role: str
    provider: str
    model: str
    system_prompt: str
    payload: dict[str, Any]
    schema: dict[str, Any]
    max_output_tokens: int
    estimated_input_tokens: int
    temperature: float = 0.0
    retry_feedback_instruction: str | None = None


@dataclass(frozen=True)
class TransportResponse:
    parsed_output: dict[str, Any]
    input_tokens: int
    output_tokens: int
    provider_request_id: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class ProviderResult:
    parsed_output: dict[str, Any]
    role: str
    provider: str
    model: str
    request_id: str
    logical_call_id: str
    attempt_index: int
    input_tokens: int
    output_tokens: int
    latency_sec: float
    status: str
    pricing_version: str
    estimated_cost_usd: Decimal | None
    provider_request_id: str | None


class ProviderTransport(Protocol):
    def __call__(self, request: ProviderRequest, credential: SecretCredential) -> TransportResponse:
        ...


SchemaValidator = Callable[[dict[str, Any], dict[str, Any]], None]


class PlannerProviderAdapter:
    """Planner-only boundary; the caller must explicitly inject an approved transport."""

    def __init__(
        self,
        *,
        transports: Mapping[str, ProviderTransport],
        credential_env_by_provider: Mapping[str, str],
        pricing: PricingCatalog,
        telemetry: AttemptTelemetryJournal,
        budget: BudgetGuard,
        environ: Mapping[str, str] | None = None,
    ):
        self._transports = dict(transports)
        self._credential_env_by_provider = dict(credential_env_by_provider)
        self._pricing = pricing
        self._telemetry = telemetry
        self._budget = budget
        self._environ = environ

    def _transport_and_credential(self, provider: str) -> tuple[ProviderTransport, SecretCredential]:
        if provider not in self._transports or provider not in self._credential_env_by_provider:
            raise UnsupportedProviderError(f"unsupported Planner provider: {provider}")
        credential = load_credential(self._credential_env_by_provider[provider], self._environ)
        return self._transports[provider], credential

    def call(
        self,
        request: ProviderRequest,
        *,
        max_attempts: int,
        schema_validator: SchemaValidator,
    ) -> ProviderResult:
        if request.role != "planner":
            raise UnsupportedProviderError("provider adapter permits only the Planner role")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        transport, credential = self._transport_and_credential(request.provider)
        secret = credential.reveal_for_request()
        last_category = "unknown"
        last_feedback = "unknown"

        for attempt_index in range(1, max_attempts + 1):
            request_id = str(uuid.uuid4())
            is_retry = attempt_index > 1
            retry_reason = last_category if is_retry else None
            reservation = self._pricing.estimate(
                request.provider, request.model,
                request.estimated_input_tokens, request.max_output_tokens,
            )
            self._budget.reserve(request_id, reservation.estimated_cost_usd)
            common = {
                "request_id": request_id,
                "logical_call_id": request.logical_call_id,
                "role": request.role,
                "provider": request.provider,
                "model": request.model,
                "attempt_index": attempt_index,
                "is_retry": is_retry,
                "retry_reason": retry_reason,
                "pricing_version": self._pricing.pricing_version,
            }
            self._telemetry.record_start(common, (secret,))
            started = time.perf_counter()
            attempt_request = request
            if is_retry and request.retry_feedback_instruction:
                retry_payload = json.loads(json.dumps(request.payload))
                retry_payload["validator_feedback"] = {
                    "previous_error": last_feedback,
                    "instruction": request.retry_feedback_instruction,
                }
                attempt_request = ProviderRequest(
                    **{
                        **request.__dict__,
                        "payload": retry_payload,
                    }
                )
            response: TransportResponse | None = None
            failure_provider_request_id: str | None = None
            failure_stop_reason: str | None = None
            status = "provider_error"
            error_category: str | None = None
            try:
                response = transport(attempt_request, credential)
                schema_validator(response.parsed_output, request.schema)
                status = "success"
            except ProviderAttemptError as error:
                error_category = error.category
                last_category = error.category
                last_feedback = f"ProviderAttemptError: {error.category}"
                failure_provider_request_id = error.provider_request_id
                failure_stop_reason = error.category
                if error.input_tokens is not None and error.output_tokens is not None:
                    response = TransportResponse(
                        parsed_output={}, input_tokens=error.input_tokens,
                        output_tokens=error.output_tokens,
                        provider_request_id=error.provider_request_id,
                        stop_reason=error.category,
                    )
            except (KeyError, TypeError, ValueError) as error:
                status = "schema_validation_failed"
                error_category = "schema_validation_failed"
                last_category = error_category
                last_feedback = f"{type(error).__name__}: {error}"
            except Exception:
                error_category = "unexpected_provider_error"
                last_category = error_category
                last_feedback = "RuntimeError: unexpected_provider_error"

            latency = time.perf_counter() - started
            input_tokens = response.input_tokens if response is not None else None
            output_tokens = response.output_tokens if response is not None else None
            actual_cost: CostEstimate | None = None
            if input_tokens is not None and output_tokens is not None:
                actual_cost = self._pricing.estimate(
                    request.provider, request.model, input_tokens, output_tokens,
                )
            if actual_cost is not None:
                cost_value = actual_cost.estimated_cost_usd
                cost_basis = "reported_usage"
            else:
                # A transport-level failure may not expose billable usage. Charge the
                # already-approved worst-case reservation so failures remain in cost
                # accounting and the frozen outer retry rule can continue safely.
                cost_value = reservation.estimated_cost_usd
                cost_basis = "worst_case_reservation"
            self._telemetry.record_end({
                **common,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_sec": latency,
                "status": status,
                "estimated_cost_usd": str(cost_value) if cost_value is not None else None,
                "estimated_cost_basis": cost_basis,
                "sanitized_error_category": error_category,
                "provider_request_id": (
                    response.provider_request_id if response else failure_provider_request_id
                ),
                "stop_reason": response.stop_reason if response else failure_stop_reason,
                "retry_triggered": status != "success" and attempt_index < max_attempts,
            }, (secret,))
            self._budget.reconcile(request_id, cost_value)

            if status == "success" and response is not None:
                return ProviderResult(
                    parsed_output=response.parsed_output,
                    role=request.role,
                    provider=request.provider,
                    model=request.model,
                    request_id=request_id,
                    logical_call_id=request.logical_call_id,
                    attempt_index=attempt_index,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_sec=latency,
                    status=status,
                    pricing_version=self._pricing.pricing_version,
                    estimated_cost_usd=cost_value,
                    provider_request_id=response.provider_request_id,
                )

        raise ProviderCallFailed(
            f"Planner provider contract exhausted after {max_attempts} attempts; category={last_category}"
        )
