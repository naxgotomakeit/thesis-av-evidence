from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable


ROLES = ("planner", "shared", "fine", "final")


def summarize_attempt_costs(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate actual attempts, including failures and retries, without inventing unknown cost."""
    buckets = {role: [] for role in ROLES}
    unknown_roles: set[str] = set()
    for row in rows:
        role = str(row.get("role"))
        if role not in buckets:
            unknown_roles.add(role)
            continue
        buckets[role].append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        known_cost = Decimal("0")
        unpriced = 0
        for row in items:
            value = row.get("estimated_cost_usd")
            if value is None:
                unpriced += 1
            else:
                known_cost += Decimal(str(value))
        return {
            "api_calls": len(items),
            "input_tokens": sum(int(row.get("input_tokens") or 0) for row in items),
            "output_tokens": sum(int(row.get("output_tokens") or 0) for row in items),
            "total_tokens": sum(
                int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)
                for row in items
            ),
            "estimated_cost_usd_known": str(known_cost),
            "unpriced_or_unknown_cost_attempts": unpriced,
            "failed_or_retry_attempts": sum(
                row.get("status") != "success" or bool(row.get("is_retry")) for row in items
            ),
        }

    by_role = {role: summarize(items) for role, items in buckets.items()}
    total = summarize([row for items in buckets.values() for row in items])
    return {
        "schema_version": "hourvideo_api_attempt_cost_summary_v1",
        "scope": "actual attempts including failures and retries",
        "by_role": by_role,
        "total": total,
        "unknown_roles": sorted(unknown_roles),
    }
