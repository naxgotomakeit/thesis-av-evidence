"""Provider-neutral efficiency and Gemini cache accounting."""

from __future__ import annotations

from typing import Any


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return {}


def gemini_usage_metrics(usage: Any, model: str = "gemini-3.5-flash") -> dict[str, Any]:
    data = _usage_dict(usage)
    input_tokens = data.get("total_input_tokens", data.get("input_tokens"))
    output_tokens = data.get("total_output_tokens", data.get("output_tokens"))
    total_tokens = data.get("total_tokens")
    cached_present = "total_cached_tokens" in data
    cached_tokens = data.get("total_cached_tokens") if cached_present else None
    warnings: list[str] = []
    if not cached_present:
        warnings.append("provider_usage_field_unavailable:total_cached_tokens")
    unattributed = total_tokens - input_tokens - output_tokens if all(isinstance(value, int) for value in (input_tokens, output_tokens, total_tokens)) else None
    uncached = max(input_tokens - cached_tokens, 0) if isinstance(input_tokens, int) and isinstance(cached_tokens, int) else None
    hit = cached_tokens > 0 if isinstance(cached_tokens, int) else None
    hit_rate = cached_tokens / input_tokens if isinstance(cached_tokens, int) and isinstance(input_tokens, int) and input_tokens > 0 else None
    return {
        "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "unattributed_tokens": unattributed,
        "tokens_by_modality": data.get("input_tokens_by_modality"),
        "output_tokens_by_modality": data.get("output_tokens_by_modality"),
        "total_cached_tokens": cached_tokens, "uncached_input_tokens": uncached,
        "cache_hit": hit, "cache_hit_rate": hit_rate,
        "cache_eligible_by_input_size": input_tokens >= 4096 if isinstance(input_tokens, int) else None,
        "cache_eligibility_rule": "input_tokens >= 4096 for Gemini 3.5 Flash",
        "provider_usage_warnings": warnings,
    }

