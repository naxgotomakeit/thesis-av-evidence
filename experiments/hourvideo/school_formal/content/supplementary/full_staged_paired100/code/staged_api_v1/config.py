"""Configuration and fingerprint gate for Full Staged API."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from direct_api_v1.pricing import AnthropicCacheAwarePricing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class StagedConfig:
    experiment_id: str
    api_enabled_by_default: bool
    model: str
    temperature: float
    timeout_sec: float
    max_transport_retries: int
    max_validation_retries: int
    max_claim_rounds: int
    max_fine_images_per_batch: int
    max_total_fine_evidence_per_claim: int
    max_fine_per_medium: int
    minimum_fine_gap_sec: float
    hard_api_budget_usd: float
    per_request_budget_reserve_usd: float
    max_tokens_by_stage: dict[str, int]
    cache_policy: dict[str, Any]
    pricing: dict[str, Any]
    legacy_runtime_root: str
    source_experiment: str
    planner_source_experiment: str
    eval300_manifest: str
    eval300_uid_order: str
    output_root: str
    credential_env_path: str
    live_python: str
    siglip_text: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "StagedConfig":
        cfg = cls(**json.loads(path.read_text(encoding="utf-8")))
        if cfg.api_enabled_by_default:
            raise ValueError("Full Staged API must default to no API")
        if cfg.model != "claude-haiku-4-5-20251001" or cfg.temperature != 0.0:
            raise ValueError("unexpected frozen provider configuration")
        if cfg.max_transport_retries != 1 or cfg.max_validation_retries != 2:
            raise ValueError("unexpected retry contract")
        if cfg.max_claim_rounds != 3 or cfg.max_fine_images_per_batch != 16:
            raise ValueError("unexpected staged scientific budget")
        if set(cfg.max_tokens_by_stage) != {"shared", "fine", "final"}:
            raise ValueError("stage token configuration incomplete")
        if cfg.max_tokens_by_stage["shared"] != 2400:
            raise ValueError("Shared max_tokens must match frozen Planner-only downstream (2400)")
        if cfg.hard_api_budget_usd <= 0 or cfg.per_request_budget_reserve_usd <= 0:
            raise ValueError("unsafe budget configuration")
        if not Path(cfg.live_python).is_file():
            raise ValueError("configured Full Staged live Python is unavailable")
        return cfg

    @property
    def pricing_object(self) -> AnthropicCacheAwarePricing:
        return AnthropicCacheAwarePricing.from_mapping(self.pricing)
