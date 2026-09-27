from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class CostEstimate:
    pricing_version: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal | None
    priced: bool

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_record(self) -> dict[str, Any]:
        return {
            "pricing_version": self.pricing_version,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": (
                str(self.estimated_cost_usd) if self.estimated_cost_usd is not None else None
            ),
            "priced": self.priced,
        }


class PricingCatalog:
    def __init__(self, pricing_version: str, entries: list[dict[str, Any]]):
        self.pricing_version = pricing_version
        self._entries = {(row["provider"], row["model"]): dict(row) for row in entries}
        if len(self._entries) != len(entries):
            raise ValueError("pricing catalog contains duplicate provider/model entries")

    @classmethod
    def load(cls, path: Path) -> "PricingCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(str(payload["pricing_version"]), list(payload.get("entries", [])))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PricingCatalog":
        return cls(str(payload["pricing_version"]), list(payload.get("entries", [])))

    def estimate(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        row = self._entries.get((provider, model))
        if row is None:
            return CostEstimate(
                self.pricing_version, provider, model, input_tokens, output_tokens, None, False,
            )
        input_rate = Decimal(str(row["input_usd_per_million_tokens"]))
        output_rate = Decimal(str(row["output_usd_per_million_tokens"]))
        cost = (Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate) / MILLION
        return CostEstimate(
            self.pricing_version, provider, model, input_tokens, output_tokens, cost, True,
        )
