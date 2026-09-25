"""Lazy bridge to the frozen V6.6.2 R3 scientific workflow.

Imports are deliberately delayed: offline preflight never imports torch,
Pydantic, SigLIP, or any local generation stack.  Live execution delegates
map projection, retrieval, evidence updates and stopping to V6.6.2; only its
three provider functions are process-locally substituted.
"""
from __future__ import annotations

import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import StagedConfig, canonical_sha, sha256_file
from .contracts import FINE_SYSTEM_PROMPT, FINAL_SYSTEM_PROMPT, StageContractError
from .provider import AnthropicStagedProvider
from .store import StagedStore


def _anthropic_content(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for item in inputs:
        if item["type"] == "text":
            content.append({"type": "text", "text": str(item["text"])})
        elif item["type"] == "image":
            content.append({"type": "image", "source": {"type": "base64", "media_type": str(item.get("mime_type", "image/jpeg")), "data": item["data"]}})
        else:
            raise ValueError(f"unsupported legacy input type: {item['type']}")
    return content


class LegacyV662R3Executor:
    def __init__(self, cfg: StagedConfig, provider: AnthropicStagedProvider, store: StagedStore, config_path: Path) -> None:
        self.cfg, self.provider, self.store, self.config_path = cfg, provider, store, config_path
        self._active_question_id: str | None = None
        self._stage_calls = {"shared": 0, "fine": 0, "final": 0}
        self._legacy_validation_attempt = 1

    def _usage(self, result: Any, role: str) -> dict[str, Any]:
        return {
            "provider": "anthropic", "provider_role": role, "model": self.cfg.model,
            "input_tokens": result.usage["ordinary_input_tokens"],
            "cache_creation_input_tokens": result.usage["cache_creation_input_tokens"],
            "cache_read_input_tokens": result.usage["cache_read_input_tokens"],
            "output_tokens": result.usage["output_tokens"], "latency_sec": result.latency_sec,
            "response_id": result.response_id, "stop_reason": result.raw_metadata.get("stop_reason"),
            "raw_text": json.dumps(result.payload, ensure_ascii=False, separators=(",", ":")),
            "cache_aware_usd": result.cost["total_cache_aware_usd"],
            "structured_output_adapter": "anthropic_native_tool_use_v1",
        }

    def _scientific_trace_index(self, question_id: str) -> dict[str, Any]:
        """Index frozen semantic/retry/downgrade journals without changing them."""
        case = Path(self.cfg.output_root) / "legacy_live" / "cases" / question_id
        candidates = {
            "v6_6_2_semantic_validation_and_retry": case / "model_attempts.jsonl",
            "v6_6_2_complete_shared_validation": case / "shared_attempt_audit.jsonl",
            "v6_6_2_route_downgrade_and_terminal": case / "r3_2" / "route_status.json",
        }
        records: dict[str, Any] = {}
        for purpose, path in candidates.items():
            if not path.is_file():
                records[purpose] = {"path": str(path), "present": False}
                continue
            record: dict[str, Any] = {
                "path": str(path), "present": True, "sha256": sha256_file(path),
            }
            if path.suffix == ".jsonl":
                record["record_count"] = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
            records[purpose] = record
        return {
            "provider_raw_response_layer": "journals/provider_responses.jsonl",
            "provider_tool_envelope_layer": "journals/controller_results.jsonl",
            "frozen_scientific_layers": records,
            "semantics": "index only; frozen V6.6.2 validation, retry, and downgrade decisions are unchanged",
        }

    def _semantic_call(self, cfg: dict[str, Any], system: str, payload: dict[str, Any], schema: dict[str, Any], max_tokens: int):
        del cfg, system, max_tokens
        qid = self._active_question_id or str(payload.get("question", {}).get("question_id", payload.get("question_id", "unknown")))
        if self._legacy_validation_attempt == 1: self._stage_calls["shared"] += 1
        try:
            result = self.provider.call(question_id=qid, stage="shared", stage_call_index=self._stage_calls["shared"], validation_retry_index=self._legacy_validation_attempt - 1,
                user_content=[{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}], schema=schema)
        except StageContractError as error:
            raise RuntimeError(f"non-JSON or invalid tool response: {error}") from error
        return result.payload, self._usage(result, "semantic_shared")

    def _visual_call(self, cfg: dict[str, Any], system: str, inputs: list[dict[str, Any]], schema: dict[str, Any]):
        del cfg
        stage = "fine" if system == FINE_SYSTEM_PROMPT else "final" if system == FINAL_SYSTEM_PROMPT else None
        if stage is None:
            raise RuntimeError("unexpected legacy visual/final prompt identity")
        qid = self._active_question_id or "unknown"
        if self._legacy_validation_attempt == 1: self._stage_calls[stage] += 1
        try:
            result = self.provider.call(question_id=qid, stage=stage, stage_call_index=self._stage_calls[stage], validation_retry_index=self._legacy_validation_attempt - 1,
                user_content=_anthropic_content(inputs), schema=schema)
        except StageContractError as error:
            raise RuntimeError(f"non-JSON or invalid tool response: {error}") from error
        usage = self._usage(result, "visual_fine" if stage == "fine" else "semantic_final")
        raw = {"id": result.response_id, "model": self.cfg.model, "stop_reason": result.raw_metadata.get("stop_reason"), "tool_input": result.payload}
        return result.payload, usage, raw

    @contextmanager
    def _patched(self) -> Iterator[tuple[Any, Any]]:
        legacy_root = Path(self.cfg.legacy_runtime_root)
        original_sys_path = list(sys.path)
        # The frozen backend contains both ``experiments...`` and
        # ``src.experiments...`` imports, so both roots are required.  Put the
        # legacy repository itself first to prevent this workspace's unrelated
        # ``src`` directory from shadowing its package.
        sys.path.insert(0, str(legacy_root / "src"))
        sys.path.insert(0, str(legacy_root))
        from experiments.hourvideo_v6_6_2_shared_coarse_contract_v1 import core as v662
        v661 = v662._v661
        base, guard = v661._base, v661._guard
        originals = {
            "configured_sides": v661._configured_sides, "case_paths": v661._case_paths,
            "semantic": base._anthropic_call, "visual": guard._gemini_call_with_schema_preflight,
            "summarize": v662.summarize,
            "preflight": v662.preflight,
            "request_start": v661._record_request_start,
        }
        qid = self._active_question_id
        v661._configured_sides = lambda cfg: ("r3_2",)
        v661._case_paths = lambda cfg, root, identity_filter: [row for row in originals["case_paths"](cfg, root, qid) if row[0].get("question_id") == qid]
        base._anthropic_call = self._semantic_call
        guard._gemini_call_with_schema_preflight = self._visual_call
        def capture_request_start(attempt: int):
            self._legacy_validation_attempt = int(attempt)
            return originals["request_start"](attempt)
        v661._record_request_start = capture_request_start
        v662.summarize = lambda root, config_path: {"status": "route_complete"}
        v662.preflight = lambda root, config_path, video_uid=None: {"status": "frozen_preflight_already_verified", "gold_loaded": False}
        try:
            yield v662, v661
        finally:
            v661._configured_sides = originals["configured_sides"]
            v661._case_paths = originals["case_paths"]
            base._anthropic_call = originals["semantic"]
            guard._gemini_call_with_schema_preflight = originals["visual"]
            v661._record_request_start = originals["request_start"]
            v662.summarize = originals["summarize"]
            v662.preflight = originals["preflight"]
            sys.path[:] = original_sys_path

    def run_one(self, question_id: str) -> dict[str, Any]:
        self._active_question_id = question_id
        legacy_root = Path(self.cfg.legacy_runtime_root)
        try:
            with self._patched() as (v662, _):
                v662.run_live(legacy_root, self.config_path, video_uid=question_id)
            final_path = Path(self.cfg.output_root) / "legacy_live" / "cases" / question_id / "r3_2" / "final_answer.json"
            status_path = final_path.with_name("route_status.json")
            if final_path.is_file():
                document = json.loads(final_path.read_text(encoding="utf-8"))
                return {"success": True, "prediction": document["answer"]["selected_option_id"], "legacy_final_path": str(final_path), "stage_calls": copy.deepcopy(self._stage_calls), "scientific_trace_index": self._scientific_trace_index(question_id)}
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            return {"success": False, "prediction": None, "failure_category": "legacy_route_failed", "legacy_status": status, "stage_calls": copy.deepcopy(self._stage_calls), "scientific_trace_index": self._scientific_trace_index(question_id)}
        finally:
            self._active_question_id = None


def route_input_sha(question_id: str, planner_sha: str, config_fingerprint: str) -> str:
    return canonical_sha({"question_id": question_id, "route": "r3_2", "planner_sha256": planner_sha, "config_fingerprint": config_fingerprint})
