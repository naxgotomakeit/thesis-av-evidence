"""GenS + Haiku structured-answer v3: Direct-aligned forced A-E answering.

Frozen GenS-selected original JPEGs are supplied in chronological order.  This
module exposes only Direct-v1.2's ``final_answer`` tool.  It contains no map,
selector, retrieval, inspection tool, abstention value, or answer-repair turn.
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from direct_api_v1.pricing import AnthropicCacheAwarePricing
from .runtime import (
    GensAnsweringStore,
    ProviderOutcome,
    aggregate_attempts,
    canonical_sha,
    resolve_frames,
    sha256_file,
)
from .structured_v2 import (
    CampaignBudgetLedger,
    StructuredHaikuProvider,
    StructuredProviderError,
    StructuredProviderOutcome,
)


CHOICES = frozenset("ABCDE")

# Byte-for-byte semantic copy of direct_api_v1.anthropic_provider.direct_action_tools(0)[0].
FINAL_ANSWER_TOOL = {
    "name": "final_answer",
    "description": "Finish with exactly one answer option.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_option_id": {
                "type": "string",
                "enum": ["A", "B", "C", "D", "E"],
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
        },
        "required": ["selected_option_id", "reason"],
    },
}


@dataclass(frozen=True)
class StructuredV3Config:
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
    system_prompt_path: str
    message_layout: str
    image_transport: str
    cache_policy: str
    tool_name: str
    tool_choice: dict[str, Any]
    pricing: dict[str, Any]
    direct_reference_config: str
    selector_path: str
    uid_order_path: str
    frame_root: str
    canonical_population_manifest: str

    @classmethod
    def load(cls, path: Path) -> "StructuredV3Config":
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        direct = json.loads(Path(config.direct_reference_config).read_text(encoding="utf-8"))
        for field in ("model", "max_output_tokens", "timeout_sec", "max_retries"):
            if getattr(config, field) != direct[field]:
                raise ValueError(f"GenS v3/Direct mismatch for {field}")
        if config.provider != direct["provider"] or config.temperature != 0.0:
            raise ValueError("provider/temperature mismatch")
        if config.api_enabled_by_default or config.concurrency != 1:
            raise ValueError("unsafe execution defaults")
        if config.tool_name != "final_answer" or config.tool_choice != {"type": "any"}:
            raise ValueError("unexpected Direct-aligned final-answer tool contract")
        if config.campaign_hard_api_budget_usd != 5.0 or config.per_request_budget_reserve_usd <= 0:
            raise ValueError("unexpected campaign budget configuration")
        if not Path(config.system_prompt_path).is_file():
            raise ValueError("missing frozen v3 system prompt")
        return config

    @property
    def pricing_object(self) -> AnthropicCacheAwarePricing:
        return AnthropicCacheAwarePricing.from_mapping(self.pricing)

    @property
    def system_prompt(self) -> str:
        return Path(self.system_prompt_path).read_text(encoding="utf-8").rstrip("\n")


def tool_schema() -> dict[str, Any]:
    return json.loads(json.dumps(FINAL_ANSWER_TOOL))


def user_question_text(row: dict[str, Any]) -> str:
    options = row["options"]
    return (
        f"Question: {row['question']}\n"
        "Options:\n"
        + "\n".join(f"{choice}. {options[choice]}" for choice in "ABCDE")
        + "\nReturn the final_answer action."
    )


def build_structured_request(
    row: dict[str, Any], config: StructuredV3Config, *, encode_images: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = resolve_frames(row, Path(config.frame_root), verify_sha=True)
    content: list[dict[str, Any]] = []
    for frame in frames:
        data = (
            base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii")
            if encode_images
            else "<validated-jpeg-bytes>"
        )
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        })
    content.append({"type": "text", "text": user_question_text(row)})
    return {
        "model": config.model,
        "max_tokens": config.max_output_tokens,
        "temperature": config.temperature,
        "system": config.system_prompt,
        "messages": [{"role": "user", "content": content}],
        "tools": [tool_schema()],
        "tool_choice": dict(config.tool_choice),
    }, frames


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def normalise_tool_input(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, SimpleNamespace):
        return dict(vars(value))
    return None


def parse_structured_response(content: Any) -> dict[str, Any]:
    """Apply Direct's actual final-answer parsing; never infer from free text.

    The provider tool schema retains its ``maxLength=240`` guidance, but the
    frozen Direct-v1.2 Python path does not revalidate that upper bound.  It
    converts option/reason values with ``str()``, rejects a blank/whitespace
    reason, ignores additional tool-input keys, and lets the controller enforce
    the A--E option set.  GenS v3 mirrors those effective semantics here while
    retaining its single-turn/no-correction protocol.
    """
    blocks = list(content or [])
    tool_blocks = [block for block in blocks if _field(block, "type") == "tool_use"]
    non_tool_blocks = [block for block in blocks if _field(block, "type") != "tool_use"]
    text = "".join(str(_field(block, "text", "")) for block in non_tool_blocks if _field(block, "type") == "text")
    base = {
        "tool_use_count": len(tool_blocks),
        "content_block_types": [_field(block, "type") for block in blocks],
        "accompanying_text": text,
        "tool_name": None,
        "tool_use_id": None,
        "tool_input_runtime_type": None,
        "normalised_tool_input": None,
        "prediction": None,
        "reason": None,
    }
    if not tool_blocks:
        return {**base, "result_class": "invalid_format", "failure_category": "missing_tool_call"}
    if len(tool_blocks) != 1:
        return {**base, "result_class": "invalid_format", "failure_category": "multiple_tool_calls"}
    if non_tool_blocks:
        return {**base, "result_class": "invalid_format", "failure_category": "tool_plus_text_or_other_content"}
    block = tool_blocks[0]
    name = _field(block, "name")
    identifier = _field(block, "id")
    native_input = _field(block, "input")
    normalized = normalise_tool_input(native_input)
    base.update({
        "tool_name": name,
        "tool_use_id": identifier,
        "tool_input_runtime_type": type(native_input).__name__,
        "normalised_tool_input": normalized,
    })
    if name != "final_answer":
        return {**base, "result_class": "invalid_format", "failure_category": "wrong_tool_name"}
    if not isinstance(identifier, str) or not identifier:
        return {**base, "result_class": "invalid_format", "failure_category": "missing_tool_use_id"}
    if normalized is None:
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_tool_input_type"}
    option = str(normalized.get("selected_option_id", ""))
    reason = str(normalized.get("reason", ""))
    base["reason"] = reason
    # Direct's FinalAnswerAction validates the reason before the controller
    # validates the option, so preserve that order for mixed-invalid inputs.
    if not reason.strip():
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_reason"}
    if option not in CHOICES:
        return {**base, "result_class": "invalid_format", "failure_category": "invalid_option"}
    return {**base, "prediction": option, "result_class": "valid_answer", "failure_category": None}


class StructuredV3HaikuProvider(StructuredHaikuProvider):
    """The validated v2 transport reused with the independent v3 request contract."""


class StructuredV3RouteRunner:
    def __init__(
        self,
        config: StructuredV3Config,
        store: GensAnsweringStore,
        campaign_ledger: CampaignBudgetLedger,
        provider_factory: Callable[[], Any],
    ) -> None:
        self.config = config
        self.store = store
        self.campaign_ledger = campaign_ledger
        self.provider_factory = provider_factory

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
            return self._terminal_failure(
                row, frames, attempts, run_fingerprint, started,
                f"input_or_provider_initialisation_failure:{type(error).__name__}",
            )
        for retry_index in range(self.config.max_retries + 1):
            attempt_id = f"v3:{qid}:{uuid.uuid4()}"
            try:
                self.campaign_ledger.assert_budget(self.config.per_request_budget_reserve_usd)
                self.store.assert_budget(self.config.per_request_budget_reserve_usd)
            except RuntimeError:
                return self._terminal_failure(
                    row, frames, attempts, run_fingerprint, started, "budget_guard_refused_no_request"
                )
            self.store.append("request_start", {
                "question_id": qid,
                "attempt_id": attempt_id,
                "retry_index": retry_index,
                "model": self.config.model,
                "image_count": len(frames),
                "request_kind": "structured_final_answer_v3",
            })
            call_started = time.perf_counter()
            try:
                outcome = provider.call(payload)
            except StructuredProviderError as error:
                outcome = StructuredProviderOutcome(
                    status="provider_error",
                    raw_text=None,
                    latency_sec=time.perf_counter() - call_started,
                    error_category=type(error).__name__,
                )
            costs = self.config.pricing_object.breakdown(
                ordinary_input_tokens=outcome.input_tokens,
                cache_creation_input_tokens=outcome.cache_creation_input_tokens,
                cache_read_input_tokens=outcome.cache_read_input_tokens,
                output_tokens=outcome.output_tokens,
            )
            parsed = (
                parse_structured_response(outcome.content)
                if outcome.status == "response_received"
                else {
                    "result_class": "runtime_failure",
                    "prediction": None,
                    "reason": None,
                    "failure_category": outcome.error_category or "provider_error",
                    "tool_use_count": 0,
                    "content_block_types": [],
                    "accompanying_text": "",
                    "normalised_tool_input": None,
                }
            )
            record = {
                "question_id": qid,
                "attempt_id": attempt_id,
                "retry_index": retry_index,
                "provider_status": outcome.status,
                "response_id": outcome.response_id,
                "response_model": outcome.response_model,
                "stop_reason": outcome.stop_reason,
                "stop_sequence": outcome.stop_sequence,
                "raw_provider_response": outcome.raw_response,
                **parsed,
                "error_category": outcome.error_category,
                "input_tokens": outcome.input_tokens,
                "cache_creation_input_tokens": outcome.cache_creation_input_tokens,
                "cache_read_input_tokens": outcome.cache_read_input_tokens,
                "output_tokens": outcome.output_tokens,
                "latency_sec": outcome.latency_sec,
                **costs,
                "cache_aware_usd": costs["total_cache_aware_usd"],
            }
            # Durability invariant: response/usage is journaled and charged before classification.
            self.store.append("attempt_end", record)
            self.store.charge_once(attempt_id, record["cache_aware_usd"])
            self.campaign_ledger.charge_once(
                attempt_id, record["cache_aware_usd"], str(self.store.root)
            )
            attempts.append(record)
            if outcome.status == "response_received":
                return self._terminal_response(row, frames, attempts, run_fingerprint, started, record)
        return self._terminal_failure(
            row, frames, attempts, run_fingerprint, started, "provider_retry_exhausted"
        )

    def _terminal_response(
        self,
        row: dict[str, Any],
        frames: list[dict[str, Any]],
        attempts: list[dict[str, Any]],
        run_fingerprint: str,
        started: float,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        valid = record["result_class"] == "valid_answer"
        artifact = {
            "question_id": row["qa_uid"],
            "video_id": row["video_id"],
            "prediction": record.get("prediction"),
            "reason": record.get("reason"),
            "result_class": record["result_class"],
            "terminal_status": "final_answer" if valid else "protocol_failure",
            "failure_category": record.get("failure_category"),
            "raw_provider_response": record.get("raw_provider_response"),
            "tool_use_count": record.get("tool_use_count"),
            "content_block_types": record.get("content_block_types"),
            "normalised_tool_input": record.get("normalised_tool_input"),
            "accompanying_text": record.get("accompanying_text"),
            "response_model": record.get("response_model"),
            "stop_reason": record.get("stop_reason"),
            "stop_sequence": record.get("stop_sequence"),
            "selected_frame_count": len(frames),
            "selected_frames": frames,
            "resource_totals": aggregate_attempts(attempts),
            "route_wall_time_sec": time.perf_counter() - started,
            "run_fingerprint": run_fingerprint,
        }
        self.store.terminal(str(row["qa_uid"]), valid, artifact)
        return artifact

    def _terminal_failure(
        self,
        row: dict[str, Any],
        frames: list[dict[str, Any]],
        attempts: list[dict[str, Any]],
        run_fingerprint: str,
        started: float,
        category: str,
    ) -> dict[str, Any]:
        artifact = {
            "question_id": row["qa_uid"],
            "video_id": row.get("video_id"),
            "prediction": None,
            "reason": None,
            "result_class": "runtime_failure",
            "terminal_status": category,
            "failure_category": category,
            "selected_frame_count": len(frames),
            "selected_frames": frames,
            "resource_totals": aggregate_attempts(attempts),
            "route_wall_time_sec": time.perf_counter() - started,
            "run_fingerprint": run_fingerprint,
        }
        self.store.terminal(str(row["qa_uid"]), False, artifact)
        return artifact


def runtime_fingerprint(
    config_path: Path,
    manifest: dict[str, Any],
    runner_path: Path,
    preflight_path: Path,
) -> str:
    config = StructuredV3Config.load(config_path)
    actual = {
        "config_sha256": sha256_file(config_path),
        "runtime_sha256": sha256_file(Path(__file__)),
        "shared_runtime_sha256": sha256_file(Path(__file__).with_name("runtime.py")),
        "provider_base_sha256": sha256_file(Path(__file__).with_name("structured_v2.py")),
        "system_prompt_sha256": sha256_file(Path(config.system_prompt_path)),
        "runner_sha256": sha256_file(runner_path),
        "preflight_sha256": sha256_file(preflight_path),
        "selector_sha256": sha256_file(Path(config.selector_path)),
        "uid_order_sha256": sha256_file(Path(config.uid_order_path)),
        "canonical_population_manifest_sha256": sha256_file(
            Path(config.canonical_population_manifest)
        ),
        "tool_schema_sha256": canonical_sha(tool_schema()),
        "prompt_spec_sha256": canonical_sha(manifest["prompt_spec"]),
    }
    if actual != manifest["fingerprints"]:
        raise RuntimeError("v3 frozen manifest/runtime fingerprint mismatch")
    if manifest["question_count"] != 300 or manifest.get("gold_loaded") is not False:
        raise RuntimeError("invalid v3 formal population or gold isolation")
    return canonical_sha(manifest)
