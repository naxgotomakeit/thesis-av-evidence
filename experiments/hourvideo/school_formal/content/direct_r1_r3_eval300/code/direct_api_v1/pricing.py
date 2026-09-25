"""Authoritative cache-aware Anthropic token accounting for Direct v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AnthropicCacheAwarePricing:
    """USD per million tokens; cache writes are selected explicitly by TTL."""

    input_usd_per_million: float
    output_usd_per_million: float
    cache_write_5m_usd_per_million: float
    cache_write_1h_usd_per_million: float | None
    cache_read_usd_per_million: float
    cache_write_ttl: str = "5m"

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | str | None]) -> "AnthropicCacheAwarePricing":
        # Retain legacy input/output aliases only for callers that have not yet
        # migrated; cache fields remain mandatory to avoid a silent estimate.
        return cls(
            input_usd_per_million=float(values.get("ordinary_input", values.get("input", 0.0))),
            output_usd_per_million=float(values.get("output", 0.0)),
            cache_write_5m_usd_per_million=float(values["cache_write_5m"]),
            cache_write_1h_usd_per_million=None if values.get("cache_write_1h") is None else float(values["cache_write_1h"]),
            cache_read_usd_per_million=float(values["cache_read"]),
            cache_write_ttl=str(values.get("cache_write_ttl", "5m")),
        )

    @property
    def cache_write_usd_per_million(self) -> float:
        if self.cache_write_ttl == "5m":
            return self.cache_write_5m_usd_per_million
        if self.cache_write_ttl == "1h":
            if self.cache_write_1h_usd_per_million is None:
                raise ValueError("1h cache write pricing has not been configured")
            return self.cache_write_1h_usd_per_million
        raise ValueError(f"unsupported cache_write_ttl: {self.cache_write_ttl}")

    def breakdown(self, *, ordinary_input_tokens: int = 0, cache_creation_input_tokens: int = 0,
                  cache_read_input_tokens: int = 0, output_tokens: int = 0) -> dict[str, float]:
        ordinary = ordinary_input_tokens * self.input_usd_per_million / 1_000_000
        creation = cache_creation_input_tokens * self.cache_write_usd_per_million / 1_000_000
        read = cache_read_input_tokens * self.cache_read_usd_per_million / 1_000_000
        output = output_tokens * self.output_usd_per_million / 1_000_000
        return {
            "ordinary_input_usd": ordinary,
            "cache_creation_usd": creation,
            "cache_read_usd": read,
            "output_usd": output,
            "total_cache_aware_usd": ordinary + creation + read + output,
        }
