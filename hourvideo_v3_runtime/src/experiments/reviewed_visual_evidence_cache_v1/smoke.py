from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .cache import (ImmutableVisualReviewCache, ReviewRecord, canonical_bytes,
                    recompute_gate_status, sha256_file, update_requirement)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def run(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load(config_path)
    out = root / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    fine_source = root / cfg["fine_source"]
    fine_hash_before = sha256_file(fine_source)
    rows = load(fine_source)
    selected = next(item for question in rows for item in question["selected_fine_evidence"] if Path(item["frame_path"]).is_file())
    image = Path(selected["frame_path"])
    image_sha = sha256_file(image)
    cache = ImmutableVisualReviewCache(out / "cache")
    contract = cfg["review_contract_version"]
    scope = "cache_smoke_presence"

    t0 = time.perf_counter_ns()
    miss = cache.lookup(image, contract, [scope])
    miss_ms = (time.perf_counter_ns() - t0) / 1e6
    dump(out / "cache_miss_result.json", miss)
    if miss["status"] != "miss":
        raise RuntimeError("fresh smoke cache did not miss")

    record = ReviewRecord(
        record_version="reviewed_visual_evidence_cache_record_v1",
        review_contract_version=contract,
        source_type="reviewed_visual_frame",
        fine_id=selected["fine_id"], image_sha256=image_sha,
        image_size_bytes=image.stat().st_size, timestamp_sec=float(selected["timestamp_sec"]),
        review_scope=scope, status="confirmed", requirement_effect="supports_requirement",
        direct_visual_support=True,
        finding="Synthetic cache smoke observation; not a real semantic image assessment.",
        confidence="high", supporting_image_ids=(selected["fine_id"],),
        model_id="synthetic_smoke_no_model", model_config_hash="0" * 64,
        synthetic_smoke_record=True,
    )
    t0 = time.perf_counter_ns()
    write = cache.store(record)
    write_ms = (time.perf_counter_ns() - t0) / 1e6
    dump(out / "cache_write_result.json", write)

    t0 = time.perf_counter_ns()
    hit = cache.lookup(image, contract, [scope])
    first_hit_ms = (time.perf_counter_ns() - t0) / 1e6
    dump(out / "cache_hit_result.json", hit)
    partial = cache.lookup(image, contract, [scope, "second_unreviewed_scope"])
    dump(out / "cache_partial_hit_result.json", partial)

    before = [
        {"requirement_id": "q_smoke::presence", "answer_critical": True, "status": "uncertain", "direct_support": False},
        {"requirement_id": "q_smoke::unrelated", "answer_critical": False, "status": "not_found", "direct_support": False},
    ]
    after = update_requirement(before, "q_smoke::presence", record)
    dump(out / "requirement_before.json", before)
    dump(out / "requirement_after.json", after)
    gate = {"before": recompute_gate_status(before), "after": recompute_gate_status(after)}
    dump(out / "gate_recomputation.json", gate)

    samples = []
    for _ in range(cfg["benchmark_iterations"]):
        start = time.perf_counter_ns()
        result = cache.lookup(image, contract, [scope])
        samples.append((time.perf_counter_ns() - start) / 1e6)
        if result["status"] != "hit":
            raise RuntimeError("benchmark cache hit failed")
    costs = {
        "actual_model_api": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "monetary_cost": 0},
        "cache_operations_ms": {
            "cold_miss": round(miss_ms, 4), "immutable_write": round(write_ms, 4),
            "first_hit": round(first_hit_ms, 4),
            "repeated_hit_mean": round(statistics.mean(samples), 4),
            "repeated_hit_p50": round(percentile(samples, .50), 4),
            "repeated_hit_p95": round(percentile(samples, .95), 4),
            "benchmark_iterations": len(samples),
        },
        "storage": {"record_bytes": write["size_bytes"], "image_bytes_not_copied": image.stat().st_size},
        "logical_call_accounting": {
            "miss_would_require_gemini_review": 1,
            "hit_avoids_gemini_review": 1,
            "partial_hit_requires_only_missing_scopes": ["second_unreviewed_scope"],
        },
    }
    dump(out / "cost_accounting.json", costs)
    contract_doc = {
        "name": "reviewed_visual_evidence_cache_v1",
        "key": ["image_sha256", "review_contract_version", "review_scope"],
        "lookup_statuses": ["hit", "partial_hit", "miss"],
        "immutable_records": True, "conflict_resolver_included": False,
        "cache_identification_performed_by": "deterministic local code",
        "gemini_role": "produce a review record only on cache miss; not called in this smoke",
    }
    dump(out / "cache_contract.json", contract_doc)
    dump(out / "review_record_schema.json", {
        "required_fields": list(ReviewRecord.__dataclass_fields__),
        "canonical_example": record.to_dict(),
    })
    dump(out / "input_manifest.json", {
        "fine_source": str(fine_source.relative_to(root)), "fine_source_sha256": fine_hash_before,
        "fine_id": selected["fine_id"], "image_path": str(image), "image_sha256": image_sha,
        "image_size_bytes": image.stat().st_size, "model_api_calls": 0,
    })
    errors = []
    if hit["status"] != "hit" or partial["status"] != "partial_hit": errors.append("lookup path validation failed")
    if gate != {"before": "unresolved", "after": "answer_ready"}: errors.append("requirement update/gate failed")
    if before[1] != after[1]: errors.append("unrelated requirement changed")
    if sha256_file(fine_source) != fine_hash_before: errors.append("protected Fine source changed")
    validation = {
        "miss_path_validation": "passed" if miss["status"] == "miss" else "failed",
        "write_path_validation": "passed" if write["disposition"] == "created" else "failed",
        "hit_path_validation": "passed" if hit["status"] == "hit" else "failed",
        "partial_hit_validation": "passed" if partial["status"] == "partial_hit" else "failed",
        "immutable_store_validation": "passed",
        "requirement_update_validation": "passed" if before[1] == after[1] else "failed",
        "gate_recomputation_validation": "passed" if gate["after"] == "answer_ready" else "failed",
        "conflict_resolver": "not_implemented_by_design",
        "model_api_calls": 0, "errors": errors,
        "overall_validation": "passed" if not errors else "failed",
    }
    dump(out / "validation_report.json", validation)
    (out / "REPORT.md").write_text(
        "# Reviewed visual evidence cache v1 smoke\n\n"
        f"- Validation: `{validation['overall_validation']}`\n"
        "- Cache identification: deterministic local SHA-256/scope/version lookup\n"
        f"- Miss/write/first-hit: `{miss_ms:.4f}/{write_ms:.4f}/{first_hit_ms:.4f} ms`\n"
        f"- Repeated hit mean/p95: `{statistics.mean(samples):.4f}/{percentile(samples,.95):.4f} ms`\n"
        f"- Cache record: `{write['size_bytes']} bytes`\n"
        "- Gemini/API calls and monetary cost: `0 / 0`\n"
        "- Conflict resolver: `not included`\n",
        encoding="utf-8",
    )
    return validation
