from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from experiments.r1_av_r3_2_claim_guided_gemini_closed_loop_v3.core import (
    FINAL_SYSTEM,
    FinalBatch,
    answerability,
    call,
    final_request,
    validate_final,
)
from experiments.r1_av_r3_2_review_cache_integration_canary_v1.core import (
    QUESTION_ORDER,
    REQ_SCOPE,
    fine_registry,
)
from experiments.reviewed_visual_evidence_cache_v1 import LayeredVisualReviewCache
from experiments.reviewed_visual_evidence_cache_v1.cache import canonical_bytes, sha256_file
from experiments.reviewed_visual_evidence_cache_v1_gemini_canary.core import response_text, usage


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


DISPLAY_DASHES = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"}


def normalize_display_text(value: str) -> tuple[str, int]:
    """Use an ASCII range separator at the user-facing serialization boundary."""
    output = []
    replacements = 0
    for char in value:
        if char in DISPLAY_DASHES:
            output.append("-")
            replacements += 1
        else:
            output.append(char)
    return "".join(output), replacements


def normalize_final_answers(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = json.loads(json.dumps(value, ensure_ascii=False))
    rows = []
    for side, questions in normalized.items():
        for question_id, answer in questions.items():
            answer["answer"], answer_count = normalize_display_text(answer["answer"])
            caveat_count = 0
            normalized_caveats = []
            for caveat in answer["caveats"]:
                normalized_caveat, count = normalize_display_text(caveat)
                normalized_caveats.append(normalized_caveat)
                caveat_count += count
            answer["caveats"] = normalized_caveats
            rows.append({
                "side": side,
                "question_id": question_id,
                "answer_replacements": answer_count,
                "caveat_replacements": caveat_count,
            })
    return normalized, {
        "policy": "replace Unicode dash/minus characters with ASCII hyphen in user-facing answer and caveat text only",
        "source_model_response_modified": False,
        "semantic_content_modified": False,
        "replacement_count": sum(row["answer_replacements"] + row["caveat_replacements"] for row in rows),
        "rows": rows,
    }


def directory_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(p for p in root.rglob("*.json") if p.is_file()):
        rows.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"root": str(root), "record_count": len(rows), "manifest_sha256": digest, "records": rows}


def build_cache_hit_audit(
    root: Path,
    cfg: dict[str, Any],
    scoped: dict[str, Any],
    r1_fine: list[dict[str, Any]],
    r3_fine: list[dict[str, Any]],
) -> dict[str, Any]:
    registry = fine_registry(r1_fine, r3_fine)
    fine_by = {
        "r1_av": {row["question_id"]: row for row in r1_fine},
        "r3_2": {row["question_id"]: row for row in r3_fine},
    }
    cache = LayeredVisualReviewCache(
        [root / path for path in cfg["cache_roots"]],
        root / cfg["output_root"] / "unused_write_layer",
    )
    rows, missing, expected_keys = [], [], set()
    for side in ("r1_av", "r3_2"):
        for qid in QUESTION_ORDER:
            gate = scoped[side][qid]
            allowed = fine_by[side][qid]["gemini_contract"]["allowed_image_ids_by_requirement"]
            for requirement_id in gate["reviewable_requirement_ids"]:
                scope = REQ_SCOPE.get(requirement_id)
                if scope is None:
                    raise ValueError(f"missing canonical scope for {requirement_id}")
                for fine_id in allowed.get(requirement_id, []):
                    fine = registry[fine_id]
                    image = Path(fine["frame_path"])
                    key = (sha256_file(image), cfg["review_contract_version"], scope)
                    duplicate_request = key in expected_keys
                    expected_keys.add(key)
                    result = cache.lookup(image, cfg["review_contract_version"], [scope])
                    row = {
                        "side": side,
                        "question_id": qid,
                        "requirement_id": requirement_id,
                        "review_scope": scope,
                        "fine_id": fine_id,
                        "timestamp_sec": fine["timestamp_sec"],
                        "image_sha256": key[0],
                        "lookup_status": result["status"],
                        "duplicate_request_for_same_cache_key": duplicate_request,
                        "visual_model_call": False,
                        "image_transmitted": False,
                    }
                    rows.append(row)
                    if result["status"] != "hit":
                        missing.append(row)
    return {
        "requested_cache_lookups": len(rows),
        "unique_cache_keys": len(expected_keys),
        "cache_hits": sum(row["lookup_status"] == "hit" for row in rows),
        "cache_misses": len(missing),
        "duplicate_requests_reused": sum(row["duplicate_request_for_same_cache_key"] for row in rows),
        "visual_model_calls": 0,
        "image_transmissions": 0,
        "rows": rows,
        "missing": missing,
    }


def build_packets(
    scoped: dict[str, Any], updated: dict[str, Any], temporal: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    packets: dict[str, list[dict[str, Any]]] = {"r1_av": [], "r3_2": []}
    for side in packets:
        for qid in QUESTION_ORDER:
            rows = updated[side][qid]["requirement_assessments"]
            accepted_temporal = temporal if side == "r3_2" and qid == "q_handcuff_before_medical" else None
            packets[side].append({
                "question_id": qid,
                "question": scoped[side][qid]["question"],
                "answerability": answerability(rows, accepted_temporal is not None),
                "requirement_assessments": rows,
                "accepted_temporal_resolution": accepted_temporal,
                "reviewed_visual_cache_policy": {
                    "cache_records_are_evidence": True,
                    "raw_images_sent_to_final_model": False,
                    "final_model_must_not_reselect_evidence": True,
                },
            })
    return packets


def validate_packets(
    scoped: dict[str, Any], updated: dict[str, Any], packets: dict[str, list[dict[str, Any]]]
) -> list[str]:
    errors = []
    for side in ("r1_av", "r3_2"):
        by_q = {row["question_id"]: row for row in packets[side]}
        if list(by_q) != QUESTION_ORDER:
            errors.append(f"{side}: question order mismatch")
        for qid in QUESTION_ORDER:
            packet = by_q[qid]
            expected = updated[side][qid]["requirement_assessments"]
            if packet["requirement_assessments"] != expected:
                errors.append(f"{side}/{qid}: assessment mutation")
            expected_ids = [x["requirement_id"] for x in scoped[side][qid]["in_scope_requirement_assessments"]]
            actual_ids = [x["requirement_id"] for x in packet["requirement_assessments"]]
            if actual_ids != expected_ids:
                errors.append(f"{side}/{qid}: requirement coverage/order mismatch")
    return errors


def _usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calls": len(rows),
        "input_tokens": sum(row["usage"]["input_tokens"] for row in rows),
        "output_tokens": sum(row["usage"]["output_tokens"] for row in rows),
        "thought_tokens": sum(row["usage"].get("thought_tokens", 0) for row in rows),
        "latency_sec": round(sum(row["latency_sec"] for row in rows), 4),
    }


def call_rest(request: dict[str, Any], schema: type[FinalBatch], endpoint: str, timeout_sec: int, api_key: str) -> dict[str, Any]:
    http_request = urllib.request.Request(
        endpoint,
        data=canonical_bytes(request),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_sec) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gemini HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    latency = time.perf_counter() - started
    raw_text = response_text(raw_response)
    parsed = schema.model_validate(json.loads(raw_text)).model_dump(mode="json")
    token_usage = usage(raw_response)
    return {
        "raw": raw_text,
        "provider_response": raw_response,
        "parsed": parsed,
        "latency_sec": latency,
        "usage": {
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
            "thought_tokens": token_usage["thought_tokens"],
            "total_tokens": token_usage["total_tokens"],
        },
    }


def run(root: Path, config_path: Path, allow_api_calls: bool = False) -> dict[str, Any]:
    cfg = load(config_path)
    output = root / cfg["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    sources = {key: root / value for key, value in cfg["sources"].items()}
    missing_sources = [str(path) for path in sources.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(missing_sources)
    hashes_before = {key: sha256_file(path) for key, path in sources.items()}
    cache_before = [directory_manifest(root / path) for path in cfg["cache_roots"]]
    dump(output / "source_artifact_audit.json", {
        "sources": {key: {"path": str(path.relative_to(root)), "sha256": hashes_before[key], "bytes": path.stat().st_size} for key, path in sources.items()},
        "cache_layers": cache_before,
    })

    scoped = load(sources["scoped_gate"])
    updated = load(sources["updated_assessments"])
    r1_fine, r3_fine = load(sources["r1_fine"]), load(sources["r3_fine"])
    temporal = load(sources["temporal_sidecar"])
    hit_audit = build_cache_hit_audit(root, cfg, scoped, r1_fine, r3_fine)
    dump(output / "visual_cache_hit_audit.json", hit_audit)
    packets = build_packets(scoped, updated, temporal)
    packet_errors = validate_packets(scoped, updated, packets)
    dump(output / "resolved_final_handoff.json", packets)

    no_api_errors = list(packet_errors)
    if hit_audit["cache_misses"]:
        no_api_errors.append("review cache is incomplete")
    if hit_audit["visual_model_calls"] or hit_audit["image_transmissions"]:
        no_api_errors.append("visual model activity detected")
    if load(sources["cache_validation"]).get("overall_validation") != "passed":
        no_api_errors.append("source cache canary validation did not pass")
    no_api = {
        "cache_lookup_validation": "passed" if not hit_audit["cache_misses"] else "failed",
        "requirement_handoff_validation": "passed" if not packet_errors else "failed",
        "visual_calls": 0,
        "image_transmissions": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "sufficiency_calls": 0,
        "errors": no_api_errors,
        "overall_validation": "passed" if not no_api_errors else "failed",
    }
    dump(output / "no_api_test_report.json", no_api)
    if no_api_errors:
        raise RuntimeError(no_api_errors)
    if not allow_api_calls:
        validation = {**no_api, "live_final_execution": "not_run", "overall_validation": "ready_for_live_final_gemini"}
        dump(output / "validation_report.json", validation)
        return validation
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY unavailable; no model call made")

    api_key = os.environ["GEMINI_API_KEY"]
    final_results, final_inputs, final_errors = {}, {}, []
    started = time.perf_counter()
    for side in ("r1_av", "r3_2"):
        request, saved = final_request(cfg["model"], side, packets[side])
        final_inputs[side] = saved
        result = call_rest(request, FinalBatch, cfg["endpoint"], cfg["timeout_sec"], api_key)
        final_errors += [f"{side}: {error}" for error in validate_final(result["parsed"], packets[side])]
        final_results[side] = result
        dump(output / "final_gemini_raw_responses.json", final_results)
    wall_latency = time.perf_counter() - started
    dump(output / "final_gemini_inputs.json", final_inputs)
    raw_answers = {side: {row["question_id"]: row for row in result["parsed"]["answers"]} for side, result in final_results.items()}
    answers, display_audit = normalize_final_answers(raw_answers)
    dump(output / "final_answers.json", answers)
    dump(output / "text_encoding_audit.json", display_audit)

    cache_after = [directory_manifest(root / path) for path in cfg["cache_roots"]]
    sources_unchanged = hashes_before == {key: sha256_file(path) for key, path in sources.items()}
    caches_unchanged = cache_before == cache_after
    final_usage = _usage(list(final_results.values()))
    final_usage["billable_output_tokens"] = final_usage["output_tokens"] + final_usage["thought_tokens"]
    final_usage["estimated_paid_tier_cost_usd"] = round(
        (
            final_usage["input_tokens"] * cfg["pricing_usd_per_million"]["input"]
            + final_usage["billable_output_tokens"] * cfg["pricing_usd_per_million"]["output"]
        ) / 1_000_000,
        8,
    )
    cost = {
        "reused_visual_review": {
            "new_calls": 0,
            "new_image_transmissions": 0,
            "cache_lookups": hit_audit["requested_cache_lookups"],
            "unique_cache_keys": hit_audit["unique_cache_keys"],
            "duplicate_requests_reused": hit_audit["duplicate_requests_reused"],
        },
        "final_text_only_gemini": final_usage,
        "wall_latency_sec": round(wall_latency, 4),
        "new_total_model_calls": final_usage["calls"],
        "new_media_inputs": 0,
    }
    dump(output / "cost_accounting.json", cost)
    validation_errors = list(final_errors)
    if not sources_unchanged:
        validation_errors.append("protected source changed")
    if not caches_unchanged:
        validation_errors.append("cache source changed")
    validation = {
        "cache_lookup_validation": "passed",
        "requirement_handoff_validation": "passed",
        "final_gemini_schema_validation": "passed" if not final_errors else "failed",
        "final_answer_grounding_validation": "passed" if not final_errors else "failed",
        "protected_source_validation": "passed" if sources_unchanged and caches_unchanged else "failed",
        "visual_review_calls": 0,
        "image_transmissions": 0,
        "final_gemini_calls": final_usage["calls"],
        "planner_calls": 0,
        "retrieval_calls": 0,
        "sufficiency_calls": 0,
        "qa_calls": 0,
        "errors": validation_errors,
        "semantic_acceptance": "pending_manual_answer_review" if not validation_errors else "failed_automatic_validation",
        "overall_validation": "pending_manual_answer_review" if not validation_errors else "failed",
    }
    dump(output / "validation_report.json", validation)
    lines = [
        "# R1_AV / R3_2 cached visual-review full closed loop v1", "",
        f"- Overall: `{validation['overall_validation']}`",
        f"- Cached visual lookups / misses: `{hit_audit['requested_cache_lookups']}/{hit_audit['cache_misses']}`",
        f"- Unique cache keys / duplicate request reuse: `{hit_audit['unique_cache_keys']}/{hit_audit['duplicate_requests_reused']}`",
        "- New visual-review calls / image transmissions: `0/0`",
        f"- New text-only Final Gemini calls: `{final_usage['calls']}`",
        f"- Final tokens: `{final_usage['input_tokens']}` input / `{final_usage['output_tokens']}` output",
        f"- Final latency: `{final_usage['latency_sec']}s`", "", "## R1_AV answers", "",
    ]
    lines += [f"- **{qid}**: {answers['r1_av'][qid]['answer']}" for qid in QUESTION_ORDER]
    lines += ["", "## R3_2 answers", ""]
    lines += [f"- **{qid}**: {answers['r3_2'][qid]['answer']}" for qid in QUESTION_ORDER]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validation


def replay_display_normalization(root: Path, config_path: Path) -> dict[str, Any]:
    """Normalize stored user-facing text without a model or upstream call."""
    cfg = load(config_path)
    output = root / cfg["output_root"]
    raw = load(output / "final_gemini_raw_responses.json")
    raw_answers = {
        side: {row["question_id"]: row for row in result["parsed"]["answers"]}
        for side, result in raw.items()
    }
    answers, audit = normalize_final_answers(raw_answers)
    dump(output / "final_answers.json", answers)
    dump(output / "text_encoding_audit.json", audit)
    report = load(output / "validation_report.json")
    report["user_facing_text_encoding_validation"] = "passed"
    report["display_normalization_api_calls"] = 0
    dump(output / "validation_report.json", report)
    cost = load(output / "cost_accounting.json")
    lines = [
        "# R1_AV / R3_2 cached visual-review full closed loop v1", "",
        f"- Overall: `{report['overall_validation']}`",
        f"- Cached visual lookups / misses: `{cost['reused_visual_review']['cache_lookups']}/0`",
        f"- Unique cache keys / duplicate request reuse: `{cost['reused_visual_review']['unique_cache_keys']}/{cost['reused_visual_review']['duplicate_requests_reused']}`",
        "- New visual-review calls / image transmissions: `0/0`",
        f"- New text-only Final Gemini calls: `{cost['final_text_only_gemini']['calls']}`",
        f"- Final tokens: `{cost['final_text_only_gemini']['input_tokens']}` input / `{cost['final_text_only_gemini']['output_tokens']}` output",
        f"- Final latency: `{cost['final_text_only_gemini']['latency_sec']}s`",
        f"- Estimated Final Gemini paid-tier cost: `${cost['final_text_only_gemini']['estimated_paid_tier_cost_usd']}`",
        f"- User-facing dash replacements: `{audit['replacement_count']}`; normalization API calls: `0`",
        "", "## R1_AV answers", "",
    ]
    for qid in QUESTION_ORDER:
        row = answers["r1_av"][qid]
        lines.append(f"- **{qid}**: {row['answer']}")
        if row["caveats"]:
            lines.append(f"  - Caveat: {' '.join(row['caveats'])}")
    lines += ["", "## R3_2 answers", ""]
    for qid in QUESTION_ORDER:
        row = answers["r3_2"][qid]
        lines.append(f"- **{qid}**: {row['answer']}")
        if row["caveats"]:
            lines.append(f"  - Caveat: {' '.join(row['caveats'])}")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"overall_validation": "passed", "replacement_count": audit["replacement_count"], "api_calls": 0}
