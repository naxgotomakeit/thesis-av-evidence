"""Independent GenS + Haiku structured-answer v2 runtime.

The frozen GenS frames are passed unchanged.  This module adds one and only one
answer-submission tool; it contains no selector, retrieval, map, or gold path.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from direct_api_v1.pricing import AnthropicCacheAwarePricing
from .runtime import (
    GensAnsweringStore, ProviderOutcome, aggregate_attempts, answer_prompt,
    atomic_json, canonical_sha, resolve_frames, sha256_file, utc_now,
    validate_input_row,
)

CHOICES = frozenset("ABCDE")
SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the single final multiple-choice result, or null when the selected frames are insufficient.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {
                "anyOf": [
                    {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                    {"type": "null"},
                ]
            }
        },
        "required": ["answer"],
    },
}


@dataclass(frozen=True)
class StructuredV2Config:
    schema_version: str
    experiment_id: str
    provider: str
    model: str
    max_output_tokens: int
    temperature: float
    timeout_sec: float
    max_retries: int
    concurrency: int
    campaign_hard_api_budget_usd: float
    per_request_budget_reserve_usd: float
    api_enabled_by_default: bool
    system_prompt: str | None
    message_layout: str
    image_transport: str
    cache_policy: str
    prompt_template_path: str
    tool_name: str
    tool_choice: dict[str, Any]
    pricing: dict[str, Any]
    direct_reference_config: str
    selector_path: str
    uid_order_path: str
    frame_root: str
    canonical_population_manifest: str

    @classmethod
    def load(cls, path: Path) -> "StructuredV2Config":
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        direct = json.loads(Path(config.direct_reference_config).read_text(encoding="utf-8"))
        for field in ("model", "max_output_tokens", "timeout_sec", "max_retries"):
            if getattr(config, field) != direct[field]:
                raise ValueError(f"GenS v2/Direct mismatch for {field}")
        if config.provider != direct["provider"] or config.temperature != 0.0:
            raise ValueError("provider/temperature mismatch")
        if config.api_enabled_by_default or config.concurrency != 1:
            raise ValueError("unsafe execution defaults")
        if config.tool_name != "submit_answer" or config.tool_choice != {"type": "tool", "name": "submit_answer"}:
            raise ValueError("unexpected structured-answer tool contract")
        if config.campaign_hard_api_budget_usd != 5.0 or config.per_request_budget_reserve_usd <= 0:
            raise ValueError("unexpected campaign budget configuration")
        return config

    @property
    def pricing_object(self) -> AnthropicCacheAwarePricing:
        return AnthropicCacheAwarePricing.from_mapping(self.pricing)


def tool_schema() -> dict[str, Any]:
    return json.loads(json.dumps(SUBMIT_ANSWER_TOOL))


def build_structured_request(row: dict[str, Any], config: StructuredV2Config,
                             *, encode_images: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_input_row(row)
    frames = resolve_frames(row, Path(config.frame_root), verify_sha=True)
    content: list[dict[str, Any]] = []
    for frame in frames:
        data = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii") if encode_images else "<validated-jpeg-bytes>"
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    template = Path(config.prompt_template_path).read_text(encoding="utf-8")
    content.append({"type": "text", "text": answer_prompt(row, template)})
    payload: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_output_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": content}],
        "tools": [tool_schema()],
        "tool_choice": dict(config.tool_choice),
    }
    if config.system_prompt:
        payload["system"] = config.system_prompt
    return payload, frames


def normalise_tool_input(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, SimpleNamespace):
        return dict(vars(value))
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def plain_provider_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): plain_provider_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_provider_value(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return {str(k): plain_provider_value(v) for k, v in vars(value).items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Anthropic SDK response models expose a controlled JSON serializer.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return plain_provider_value(dump())
    return {"unsupported_runtime_type": type(value).__name__}


def parse_structured_response(content: Any) -> dict[str, Any]:
    blocks = list(content or [])
    tool_blocks = [b for b in blocks if _field(b, "type") == "tool_use"]
    text_blocks = [str(_field(b, "text", "")) for b in blocks if _field(b, "type") == "text"]
    base = {
        "tool_use_count": len(tool_blocks),
        "content_block_types": [_field(b, "type") for b in blocks],
        "accompanying_text": "".join(text_blocks),
        "tool_name": None,
        "tool_use_id": None,
        "tool_input_runtime_type": None,
        "normalised_tool_input": None,
        "prediction": None,
    }
    if len(tool_blocks) == 0:
        return {**base, "result_class": "invalid_format", "failure_category": "missing_tool_call"}
    if len(tool_blocks) != 1:
        return {**base, "result_class": "invalid_format", "failure_category": "multiple_tool_calls"}
    block = tool_blocks[0]
    name, identifier, native_input = _field(block, "name"), _field(block, "id"), _field(block, "input")
    normalized = normalise_tool_input(native_input)
    base.update({"tool_name": name, "tool_use_id": identifier,
                 "tool_input_runtime_type": type(native_input).__name__,
                 "normalised_tool_input": normalized})
    if name != "submit_answer":
        return {**base, "result_class": "invalid_format", "failure_category": "wrong_tool_name"}
    if normalized is None:
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_tool_input_type"}
    if set(normalized) != {"answer"}:
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_tool_input_fields"}
    answer = normalized["answer"]
    if answer is None:
        return {**base, "result_class": "abstention", "failure_category": None}
    if isinstance(answer, str) and answer in CHOICES:
        return {**base, "result_class": "valid_answer", "prediction": answer, "failure_category": None}
    return {**base, "result_class": "invalid_format", "failure_category": "invalid_answer_value"}


@dataclass
class StructuredProviderOutcome(ProviderOutcome):
    content: Any = None
    raw_response: Any = None


class StructuredProviderError(RuntimeError):
    pass


class _HttpMessages:
    def __init__(self, api_key: str, timeout: float) -> None:
        self.api_key, self.timeout = api_key, timeout

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class StructuredHaikuProvider:
    def __init__(self, config: StructuredV2Config, *, api_key: str, client: Any | None = None) -> None:
        if not api_key:
            raise StructuredProviderError("explicit API key required")
        self.config = config
        self.client = client or _HttpMessages(api_key, config.timeout_sec)

    def call(self, payload: dict[str, Any]) -> StructuredProviderOutcome:
        started = time.perf_counter()
        try:
            response = self.client.create(payload)
        except (TimeoutError, urllib.error.URLError) as error:
            raise StructuredProviderError(type(error).__name__) from error
        usage = _field(response, "usage", {})
        content = _field(response, "content", [])
        return StructuredProviderOutcome(
            status="response_received", raw_text=None, content=content,
            raw_response=plain_provider_value(response),
            input_tokens=int(_field(usage, "input_tokens", 0)),
            cache_creation_input_tokens=int(_field(usage, "cache_creation_input_tokens", 0)),
            cache_read_input_tokens=int(_field(usage, "cache_read_input_tokens", 0)),
            output_tokens=int(_field(usage, "output_tokens", 0)),
            latency_sec=time.perf_counter() - started,
            response_id=_field(response, "id"), response_model=_field(response, "model"),
            stop_reason=_field(response, "stop_reason"), stop_sequence=_field(response, "stop_sequence"),
        )


class CampaignBudgetLedger:
    """One $5 authorization ledger shared by isolated v2 smoke/formal stores."""
    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path, self.cap_usd = path, cap_usd

    def initialise(self) -> None:
        if not self.path.exists():
            atomic_json(self.path, {"cap_usd": self.cap_usd, "spent_usd": 0.0,
                                    "charges": {}, "created_at": utc_now()})
        ledger = json.loads(self.path.read_text(encoding="utf-8"))
        if float(ledger["cap_usd"]) != self.cap_usd:
            raise RuntimeError("campaign budget cap mismatch")

    def assert_budget(self, reserve_usd: float) -> None:
        ledger = json.loads(self.path.read_text(encoding="utf-8"))
        if float(ledger["spent_usd"]) + reserve_usd > self.cap_usd + 1e-12:
            raise RuntimeError("campaign hard API budget would be exceeded")

    def charge_once(self, attempt_id: str, usd: float, namespace: str) -> bool:
        ledger = json.loads(self.path.read_text(encoding="utf-8"))
        if attempt_id in ledger["charges"]:
            return False
        ledger["charges"][attempt_id] = {"usd": usd, "namespace": namespace}
        ledger["spent_usd"] = float(ledger["spent_usd"]) + usd
        atomic_json(self.path, ledger)
        return True

    def reconcile(self, attempts: list[dict[str, Any]], namespace: str) -> int:
        added = 0
        for row in attempts:
            if self.charge_once(str(row["attempt_id"]), float(row.get("cache_aware_usd", 0.0)), namespace):
                added += 1
        return added


def attempt_cost(outcome: ProviderOutcome, pricing: AnthropicCacheAwarePricing) -> dict[str, float]:
    return pricing.breakdown(
        ordinary_input_tokens=outcome.input_tokens,
        cache_creation_input_tokens=outcome.cache_creation_input_tokens,
        cache_read_input_tokens=outcome.cache_read_input_tokens,
        output_tokens=outcome.output_tokens,
    )


class StructuredV2RouteRunner:
    def __init__(self, config: StructuredV2Config, store: GensAnsweringStore,
                 campaign_ledger: CampaignBudgetLedger, provider_factory: Callable[[], Any]) -> None:
        self.config, self.store = config, store
        self.campaign_ledger, self.provider_factory = campaign_ledger, provider_factory

    def run_one(self, row: dict[str, Any], *, run_fingerprint: str) -> dict[str, Any]:
        qid = str(row["qa_uid"])
        prior = self.store.status(qid)
        if prior and prior["state"].startswith("terminal_"):
            return {"question_id": qid, "skipped_terminal": True}
        started = time.perf_counter()
        self.store.start(qid, canonical_sha(row))
        attempts: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []
        try:
            payload, frames = build_structured_request(row, self.config, encode_images=True)
            provider = self.provider_factory()
        except Exception as error:
            return self._terminal_failure(row, frames, attempts, run_fingerprint, started,
                                          f"input_or_provider_initialisation_failure:{type(error).__name__}")
        for retry_index in range(self.config.max_retries + 1):
            attempt_id = f"v2:{qid}:{uuid.uuid4()}"
            try:
                self.campaign_ledger.assert_budget(self.config.per_request_budget_reserve_usd)
                self.store.assert_budget(self.config.per_request_budget_reserve_usd)
            except RuntimeError:
                return self._terminal_failure(row, frames, attempts, run_fingerprint, started,
                                              "budget_guard_refused_no_request")
            self.store.append("request_start", {
                "question_id": qid, "attempt_id": attempt_id, "retry_index": retry_index,
                "model": self.config.model, "image_count": len(frames), "request_kind": "structured_submit_answer",
            })
            call_started = time.perf_counter()
            try:
                outcome = provider.call(payload)
            except StructuredProviderError as error:
                outcome = StructuredProviderOutcome(status="provider_error", raw_text=None,
                    latency_sec=time.perf_counter() - call_started, error_category=type(error).__name__)
            costs = attempt_cost(outcome, self.config.pricing_object)
            parsed = parse_structured_response(outcome.content) if outcome.status == "response_received" else {
                "result_class": "runtime_failure", "prediction": None,
                "failure_category": outcome.error_category or "provider_error",
                "tool_use_count": 0, "accompanying_text": "", "normalised_tool_input": None,
            }
            record = {
                "question_id": qid, "attempt_id": attempt_id, "retry_index": retry_index,
                "provider_status": outcome.status, "response_id": outcome.response_id,
                "response_model": outcome.response_model, "stop_reason": outcome.stop_reason,
                "stop_sequence": outcome.stop_sequence, "raw_provider_response": outcome.raw_response,
                **parsed, "error_category": outcome.error_category,
                "input_tokens": outcome.input_tokens,
                "cache_creation_input_tokens": outcome.cache_creation_input_tokens,
                "cache_read_input_tokens": outcome.cache_read_input_tokens,
                "output_tokens": outcome.output_tokens, "latency_sec": outcome.latency_sec,
                **costs, "cache_aware_usd": costs["total_cache_aware_usd"],
            }
            # Provider attempt is durable and charged before result classification.
            self.store.append("attempt_end", record)
            self.store.charge_once(attempt_id, record["cache_aware_usd"])
            self.campaign_ledger.charge_once(attempt_id, record["cache_aware_usd"], str(self.store.root))
            attempts.append(record)
            if outcome.status == "response_received":
                return self._terminal_response(row, frames, attempts, run_fingerprint, started, record)
        return self._terminal_failure(row, frames, attempts, run_fingerprint, started, "provider_retry_exhausted")

    def _terminal_response(self, row: dict[str, Any], frames: list[dict[str, Any]], attempts: list[dict[str, Any]],
                           run_fingerprint: str, started: float, record: dict[str, Any]) -> dict[str, Any]:
        result_class = record["result_class"]
        success = result_class in {"valid_answer", "abstention"}
        artifact = {
            "question_id": row["qa_uid"], "video_id": row["video_id"],
            "prediction": record.get("prediction"), "result_class": result_class,
            "terminal_status": "structured_response" if success else "invalid_answer",
            "failure_category": record.get("failure_category"),
            "raw_provider_response": record.get("raw_provider_response"),
            "tool_use_count": record.get("tool_use_count"),
            "normalised_tool_input": record.get("normalised_tool_input"),
            "accompanying_text": record.get("accompanying_text"),
            "response_model": record.get("response_model"), "stop_reason": record.get("stop_reason"),
            "stop_sequence": record.get("stop_sequence"),
            "selected_frame_count": len(frames), "selected_frames": frames,
            "resource_totals": aggregate_attempts(attempts),
            "route_wall_time_sec": time.perf_counter() - started, "run_fingerprint": run_fingerprint,
        }
        self.store.terminal(str(row["qa_uid"]), success, artifact)
        return artifact

    def _terminal_failure(self, row: dict[str, Any], frames: list[dict[str, Any]], attempts: list[dict[str, Any]],
                          run_fingerprint: str, started: float, category: str) -> dict[str, Any]:
        artifact = {
            "question_id": row["qa_uid"], "video_id": row.get("video_id"), "prediction": None,
            "result_class": "runtime_failure", "terminal_status": category, "failure_category": category,
            "selected_frame_count": len(frames), "selected_frames": frames,
            "resource_totals": aggregate_attempts(attempts),
            "route_wall_time_sec": time.perf_counter() - started, "run_fingerprint": run_fingerprint,
        }
        self.store.terminal(str(row["qa_uid"]), False, artifact)
        return artifact


def runtime_fingerprint(config_path: Path, manifest: dict[str, Any], runner_path: Path,
                        preflight_path: Path) -> str:
    config = StructuredV2Config.load(config_path)
    actual = {
        "config_sha256": sha256_file(config_path),
        "runtime_sha256": sha256_file(Path(__file__)),
        "shared_runtime_sha256": sha256_file(Path(__file__).with_name("runtime.py")),
        "prompt_template_sha256": sha256_file(Path(config.prompt_template_path)),
        "runner_sha256": sha256_file(runner_path),
        "preflight_sha256": sha256_file(preflight_path),
        "selector_sha256": sha256_file(Path(config.selector_path)),
        "uid_order_sha256": sha256_file(Path(config.uid_order_path)),
        "canonical_population_manifest_sha256": sha256_file(Path(config.canonical_population_manifest)),
        "tool_schema_sha256": canonical_sha(tool_schema()),
        "prompt_spec_sha256": canonical_sha(manifest["prompt_spec"]),
    }
    if actual != manifest["fingerprints"]:
        raise RuntimeError("v2 frozen manifest/runtime fingerprint mismatch")
    if manifest["question_count"] != 300 or manifest.get("gold_loaded") is not False:
        raise RuntimeError("invalid v2 formal population or gold isolation")
    return canonical_sha(manifest)
