from __future__ import annotations

import datetime
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def ordered_identity_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    """Linear interpolation, matching NumPy's default percentile definition."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "sum": sum(rows),
        "mean": mean(rows) if rows else None,
        "median": median(rows) if rows else None,
        "p90": percentile(rows, 0.90),
        "p95": percentile(rows, 0.95),
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
        "overflow_zero_imputed": False,
    }


def usage_summary(attempts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(attempts)
    return {
        "requests": len(rows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "model_latency_sec": sum(float(row.get("latency_sec") or 0.0) for row in rows),
        "logical_fine_evidence": sum(int(row.get("logical_fine_evidence_count") or 0) for row in rows),
        "physical_image_transmissions": sum(int(row.get("physical_image_transmissions") or 0) for row in rows),
        "status_counts": dict(sorted(Counter(str(row.get("status") or "unknown") for row in rows).items())),
    }


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in attempts if row.get("status") in {"success", "accepted"}]
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_stage[str(row.get("stage") or "unknown")].append(row)
    return {
        "selected_successful_attempts": usage_summary(accepted),
        "actual_all_attempts_including_retries": usage_summary(attempts),
        "by_stage_actual_all_attempts": {
            stage: usage_summary(rows) for stage, rows in sorted(by_stage.items())
        },
        "selected_cost_definition": "attempts with accepted/success status on the canonical route",
        "actual_cost_definition": "all actual attempts, including validation failures, truncations, provider errors and retries",
    }


def freeze_tree(source_root: Path, *, excluded_roots: Iterable[Path] = ()) -> dict[str, Any]:
    source_root = source_root.resolve()
    excluded = [path.resolve() for path in excluded_roots]
    rows: list[dict[str, Any]] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in excluded):
            continue
        rows.append({
            "path": str(path.relative_to(source_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    tree_sha = hashlib.sha256(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode()
    ).hexdigest()
    return {
        "schema_version": "immutable_tree_freeze_manifest_v1",
        "created_at_utc": now_utc(),
        "source_root": str(source_root),
        "excluded_roots": [str(path) for path in excluded],
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": tree_sha,
        "files": rows,
    }


def validate_fixed_population(
    routes: list[dict[str, Any]],
    ordered_question_ids: list[str],
    paired_question_ids: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    identities = [(row.get("question_id"), row.get("route")) for row in routes]
    expected = [(qid, side) for qid in ordered_question_ids for side in ("r1_av", "r3_2")]
    if len(ordered_question_ids) != 300 or len(set(ordered_question_ids)) != 300:
        errors.append("ordered Eval300 is not 300 unique question IDs")
    if identities != expected:
        errors.append("canonical route identity/order differs from ordered Eval300 x [r1_av,r3_2]")
    if len(identities) != len(set(identities)):
        errors.append("duplicate question_id + route identity")
    r1 = [row for row in routes if row.get("route") == "r1_av"]
    r3 = [row for row in routes if row.get("route") == "r3_2"]
    r3_overflow = [row for row in r3 if row.get("formal_status") == "context_overflow_pre_model"]
    r3_executed = [row for row in r3 if row.get("formal_status") != "context_overflow_pre_model"]
    if len(r1) != 300:
        errors.append(f"R1 population is {len(r1)}, expected 300")
    if len(r3) != 300 or len(r3_executed) != 150 or len(r3_overflow) != 150:
        errors.append(
            f"R3 population/executed/overflow is {len(r3)}/{len(r3_executed)}/{len(r3_overflow)}, expected 300/150/150"
        )
    if len(paired_question_ids) != 150 or len(set(paired_question_ids)) != 150:
        errors.append("paired context-feasible list is not 150 unique IDs")
    if any(qid not in ordered_question_ids for qid in paired_question_ids):
        errors.append("paired context-feasible list contains a non-Eval300 ID")
    for row in routes:
        if row.get("formal_status") == "context_overflow_pre_model":
            if row.get("prediction_present") or row["post_planner_cost"]["actual_all_attempts_including_retries"]["requests"]:
                errors.append(f"overflow route has prediction or model request: {row['route_key']}")
            if row.get("e2e_sec") is not None:
                errors.append(f"overflow route has imputed E2E: {row['route_key']}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "gold_loaded": False,
        "errors": errors,
        "counts": {
            "routes": len(routes), "unique_routes": len(set(identities)),
            "r1": len(r1), "r3": len(r3),
            "r3_executed": len(r3_executed), "r3_context_overflow_pre_model": len(r3_overflow),
            "paired_context_feasible": len(paired_question_ids),
        },
    }


def score_after_structural_pass(
    structural_report: dict[str, Any],
    load_gold: Callable[[], Any],
    scorer: Callable[[Any], Any],
) -> Any:
    """Hard boundary proving gold is inaccessible before structural PASS."""
    if structural_report.get("status") != "PASS":
        raise RuntimeError("structural validation failed; refusing to load gold")
    gold = load_gold()
    return scorer(gold)

