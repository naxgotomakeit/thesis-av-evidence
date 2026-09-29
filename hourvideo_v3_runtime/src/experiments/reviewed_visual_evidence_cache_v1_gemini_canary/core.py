from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from experiments.reviewed_visual_evidence_cache_v1.cache import (
    ImmutableVisualReviewCache, ReviewRecord, canonical_bytes,
    recompute_gate_status, sha256_file, update_requirement,
)


SYSTEM = """You are a targeted visual observation component. Inspect only the
single supplied image and only the requested review scope. Do not answer a
broader question, infer from context, identify people, or introduce facts beyond
what is directly visible. Return strict JSON matching the supplied schema."""

PROMPT = """Review scope: weapon_visibility.
Determine only whether a firearm, handgun, rifle, taser, or other weapon is
visibly identifiable in this exact image. Do not infer a weapon from clothing,
uniforms, posture, surrounding context, or expected events.

Use confirmed + supports_requirement + direct_visual_support=true only when a
weapon is clearly visible in this image. For probable, uncertain, or
not_supported, use requirement_effect=inconclusive and
direct_visual_support=false. The finding must be concise and conservative."""

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["confirmed", "probable", "uncertain", "not_supported"]},
        "requirement_effect": {"type": "string", "enum": ["supports_requirement", "inconclusive"]},
        "direct_visual_support": {"type": "boolean"},
        "finding": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "none"]},
    },
    "required": ["status", "requirement_effect", "direct_visual_support", "finding", "confidence"],
    "additionalProperties": False,
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_model_result(value: dict[str, Any]) -> list[str]:
    errors = []
    if set(value) != set(MODEL_SCHEMA["required"]): errors.append("model fields mismatch")
    if value.get("status") not in MODEL_SCHEMA["properties"]["status"]["enum"]: errors.append("invalid status")
    if value.get("requirement_effect") not in MODEL_SCHEMA["properties"]["requirement_effect"]["enum"]: errors.append("invalid requirement effect")
    if not isinstance(value.get("direct_visual_support"), bool): errors.append("direct_visual_support is not boolean")
    if not isinstance(value.get("finding"), str) or not value.get("finding", "").strip(): errors.append("empty finding")
    if value.get("confidence") not in MODEL_SCHEMA["properties"]["confidence"]["enum"]: errors.append("invalid confidence")
    confirmed = value.get("status") == "confirmed"
    supporting = value.get("requirement_effect") == "supports_requirement"
    direct = value.get("direct_visual_support") is True
    if not (confirmed == supporting == direct):
        errors.append("confirmed/supporting/direct fields are inconsistent")
    return errors


def response_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text") or response.get("outputText")
    if isinstance(direct, str) and direct.strip(): return direct
    texts = []
    for step in response.get("steps", []):
        if step.get("type") != "model_output": continue
        for item in step.get("content", []):
            if item.get("type") == "text" and isinstance(item.get("text"), str): texts.append(item["text"])
    if not texts:
        raise ValueError("Gemini response contains no model text")
    return "".join(texts)


def usage(response: dict[str, Any]) -> dict[str, int]:
    raw = response.get("usage", {})
    def integer(*keys: str) -> int:
        for key in keys:
            if isinstance(raw.get(key), (int, float)): return int(raw[key])
        return 0
    return {
        "input_tokens": integer("total_input_tokens", "totalInputTokens", "input_tokens", "inputTokens", "prompt_token_count", "promptTokenCount"),
        "output_tokens": integer("total_output_tokens", "totalOutputTokens", "output_tokens", "outputTokens", "candidates_token_count", "candidatesTokenCount"),
        "thought_tokens": integer("total_thought_tokens", "totalThoughtTokens", "thought_tokens", "thoughtTokens"),
        "tool_use_tokens": integer("total_tool_use_tokens", "totalToolUseTokens"),
        "total_tokens": integer("total_tokens", "totalTokens", "total_token_count", "totalTokenCount"),
    }


def select_input(fine_source: Path) -> dict[str, Any]:
    question = next(row for row in load(fine_source) if row["question_id"] == "q_weapon_visible")
    ordered = sorted(question["selected_fine_evidence"], key=lambda row: (row["medium_rank"], row["selection_reason"] != "top_relevance", row["fine_id"]))
    selected = ordered[0]
    if not Path(selected["frame_path"]).is_file(): raise FileNotFoundError(selected["frame_path"])
    return selected


def run(root: Path, config_path: Path, allow_api_calls: bool = False) -> dict[str, Any]:
    cfg = load(config_path); out = root / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    fine_source = root / cfg["fine_source"]
    fine_hash_before = sha256_file(fine_source)
    selected = select_input(fine_source); image = Path(selected["frame_path"]); image_sha = sha256_file(image)
    cache = ImmutableVisualReviewCache(out / "cache")
    scope = "weapon_visibility"; contract = cfg["review_contract_version"]
    initial_start = time.perf_counter_ns(); initial = cache.lookup(image, contract, [scope]); initial_ms = (time.perf_counter_ns()-initial_start)/1e6
    dump(out / "initial_cache_lookup.json", initial)
    request_metadata = {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/interactions",
        "model": cfg["model"], "system_instruction": SYSTEM, "prompt": PROMPT,
        "response_format": {"type": "text", "mime_type": "application/json", "schema": MODEL_SCHEMA},
        "generation_config": {"temperature": 0.0, "thinking_level": "low"},
        "store": False, "image": {"fine_id": selected["fine_id"], "sha256": image_sha, "mime_type": "image/jpeg", "bytes": image.stat().st_size},
        "api_key_stored": False,
    }
    dump(out / "sanitized_gemini_request.json", request_metadata)
    preflight_errors = []
    if initial["status"] != "miss": preflight_errors.append("canary cache is not fresh; refusing a paid call")
    if fine_hash_before != sha256_file(fine_source): preflight_errors.append("Fine source changed during preflight")
    if selected["question_id"] != "q_weapon_visible" or selected["medium_rank"] != 1: preflight_errors.append("non-canonical deterministic selection")
    preflight = {"passed": not preflight_errors, "errors": preflight_errors, "model_api_calls": 0}
    dump(out / "preflight_validation.json", preflight)
    if preflight_errors:
        result = {"overall_validation": "failed_preflight", "errors": preflight_errors, "model_api_calls": 0}
        dump(out / "validation_report.json", result); return result
    if not allow_api_calls:
        result = {"overall_validation": "ready_for_one_call", "errors": [], "model_api_calls": 0}
        dump(out / "validation_report.json", result); return result
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        result = {"overall_validation": "blocked_missing_gemini_api_key", "errors": ["GEMINI_API_KEY unavailable"], "model_api_calls": 0}
        dump(out / "validation_report.json", result); return result

    body = {
        "model": cfg["model"], "system_instruction": SYSTEM,
        "input": [
            {"type": "text", "text": PROMPT},
            {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image.read_bytes()).decode("ascii")},
        ],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": MODEL_SCHEMA},
        "generation_config": {"temperature": 0.0, "thinking_level": "low"}, "store": False,
    }
    http_request = urllib.request.Request(request_metadata["endpoint"], data=canonical_bytes(body), method="POST",
                                          headers={"Content-Type": "application/json", "x-goog-api-key": key})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=cfg["timeout_sec"]) as http_response:
            raw_bytes = http_response.read(); http_status = http_response.status
        latency = time.perf_counter() - started
        raw_response = json.loads(raw_bytes.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - started
        error_body = exc.read().decode("utf-8", errors="replace")
        dump(out / "gemini_http_error.json", {"http_status": exc.code, "body": error_body, "latency_sec": latency})
        result = {"overall_validation": "gemini_api_call_failed", "errors": [f"HTTP {exc.code}"], "model_api_calls": 1, "retries": 0}
        dump(out / "validation_report.json", result); return result
    dump(out / "gemini_raw_response.json", raw_response)
    text = response_text(raw_response); parsed = json.loads(text); model_errors = validate_model_result(parsed)
    dump(out / "gemini_review_result.json", parsed)
    if model_errors:
        result = {"overall_validation": "invalid_gemini_review_output", "errors": model_errors, "model_api_calls": 1, "retries": 0}
        dump(out / "validation_report.json", result); return result

    model_cfg_hash = hashlib.sha256(canonical_bytes({k: request_metadata[k] for k in ("model", "system_instruction", "prompt", "response_format", "generation_config", "store")})).hexdigest()
    record = ReviewRecord(
        "reviewed_visual_evidence_cache_record_v1", contract, "reviewed_visual_frame",
        selected["fine_id"], image_sha, image.stat().st_size, float(selected["timestamp_sec"]), scope,
        parsed["status"], parsed["requirement_effect"], parsed["direct_visual_support"], parsed["finding"],
        parsed["confidence"], (selected["fine_id"],), cfg["model"], model_cfg_hash, False,
    )
    write_start=time.perf_counter_ns(); write=cache.store(record); write_ms=(time.perf_counter_ns()-write_start)/1e6
    dump(out / "cache_write_result.json", write)
    hit_start=time.perf_counter_ns(); second=cache.lookup(image, contract, [scope]); hit_ms=(time.perf_counter_ns()-hit_start)/1e6
    dump(out / "second_cache_lookup.json", second)
    cache_record_identical = canonical_bytes(second["records"][scope]) == canonical_bytes(record.to_dict()) if second["status"] == "hit" else False
    before=[{"requirement_id":"q_weapon_visible::weapon_presence","answer_critical":True,"status":"uncertain","direct_support":False,"supporting_evidence_ids":[]}]
    after=update_requirement(before,"q_weapon_visible::weapon_presence",record)
    dump(out / "requirement_update.json", {"before":before,"after":after,"gate_before":recompute_gate_status(before),"gate_after":recompute_gate_status(after)})
    used=usage(raw_response)
    billable_output_tokens=used["output_tokens"]+used["thought_tokens"]
    estimated_usd=(used["input_tokens"]*cfg["pricing_usd_per_million"]["input"]+billable_output_tokens*cfg["pricing_usd_per_million"]["output"])/1_000_000
    costs={
        "model":cfg["model"],"actual_gemini_calls":1,"image_transmissions":1,"retries":0,
        "input_tokens":used["input_tokens"],"output_tokens":used["output_tokens"],"thought_tokens":used["thought_tokens"],
        "billable_output_tokens":billable_output_tokens,"total_tokens":used["total_tokens"],
        "model_latency_sec":round(latency,4),"estimated_paid_tier_cost_usd":round(estimated_usd,8),
        "pricing_snapshot":{"input_usd_per_million":cfg["pricing_usd_per_million"]["input"],"output_usd_per_million":cfg["pricing_usd_per_million"]["output"],"source":"Google Gemini Developer API pricing, accessed 2026-08-03"},
        "cache":{"initial_miss_ms":round(initial_ms,4),"write_ms":round(write_ms,4),"second_hit_ms":round(hit_ms,4),"record_bytes":write["size_bytes"]},
        "second_identical_request":{"cache_status":second["status"],"gemini_calls":0,"image_transmissions":0,"avoided_gemini_calls":1,"estimated_avoided_cost_usd_if_usage_similar":round(estimated_usd,8)},
    }
    dump(out / "cost_accounting.json",costs)
    source_unchanged=fine_hash_before==sha256_file(fine_source)
    errors=[]
    if second["status"]!="hit": errors.append("second lookup was not a hit")
    if not cache_record_identical: errors.append("cache hit record is not byte-equivalent")
    if not source_unchanged: errors.append("protected Fine source changed")
    validation={
        "initial_miss_validation":"passed","gemini_schema_validation":"passed","immutable_write_validation":"passed",
        "second_hit_validation":"passed" if second["status"]=="hit" else "failed",
        "cache_record_equivalence":"passed" if cache_record_identical else "failed",
        "requirement_update_validation":"passed","protected_source_validation":"passed" if source_unchanged else "failed",
        "conflict_resolver":"not_run","formal_pipeline_integration":"not_run",
        "model_api_calls":1,"image_transmissions":1,"retries":0,"errors":errors,
        "overall_validation":"passed" if not errors else "failed",
    }
    dump(out / "validation_report.json",validation)
    dump(out / "input_manifest.json",{"fine_source":str(fine_source.relative_to(root)),"fine_source_sha256":fine_hash_before,"selected_fine":selected,"image_sha256":image_sha,"selection":"lowest medium_rank, top_relevance first; no semantic inspection"})
    (out/"REPORT.md").write_text(
        "# Reviewed visual evidence cache v1 — live Gemini canary\n\n"
        f"- Validation: `{validation['overall_validation']}`\n- Fine: `{selected['fine_id']}` at `{selected['timestamp_sec']}s`\n"
        f"- Gemini result: `{parsed['status']}` / `{parsed['requirement_effect']}` — {parsed['finding']}\n"
        f"- First path: `miss -> 1 Gemini call -> immutable write`\n- Second path: `hit -> 0 Gemini calls`\n"
        f"- Tokens: `{used['input_tokens']}` input / `{used['output_tokens']}` visible output / `{used['thought_tokens']}` thought\n"
        f"- Latency: `{latency:.4f}s`; estimated paid-tier cost: `${estimated_usd:.8f}`\n"
        f"- Cache write/hit: `{write_ms:.4f}/{hit_ms:.4f}ms`; record: `{write['size_bytes']} bytes`\n"
        "- Conflict resolver and formal pipeline integration: `not run`\n",encoding="utf-8")
    return validation


def recompute_existing_cost(root: Path, config_path: Path) -> dict[str, Any]:
    """Recalculate token/cost reporting from the saved response; makes no API call."""
    cfg=load(config_path); out=root/cfg["output_root"]
    raw=load(out/"gemini_raw_response.json"); old=load(out/"cost_accounting.json")
    parsed=load(out/"gemini_review_result.json"); manifest=load(out/"input_manifest.json")
    used=usage(raw); billable=used["output_tokens"]+used["thought_tokens"]
    estimate=(used["input_tokens"]*cfg["pricing_usd_per_million"]["input"]+billable*cfg["pricing_usd_per_million"]["output"])/1_000_000
    old.update({
        "input_tokens":used["input_tokens"],"output_tokens":used["output_tokens"],"thought_tokens":used["thought_tokens"],
        "billable_output_tokens":billable,"total_tokens":used["total_tokens"],
        "estimated_paid_tier_cost_usd":round(estimate,8),
        "usage_integrity":{"input_plus_output_plus_thought_plus_tool_equals_total":used["input_tokens"]+used["output_tokens"]+used["thought_tokens"]+used["tool_use_tokens"]==used["total_tokens"],"tool_use_tokens":used["tool_use_tokens"]},
    })
    old["second_identical_request"]["estimated_avoided_cost_usd_if_usage_similar"]=round(estimate,8)
    dump(out/"cost_accounting.json",old)
    selected=manifest["selected_fine"]
    (out/"REPORT.md").write_text(
        "# Reviewed visual evidence cache v1 — live Gemini canary\n\n"
        f"- Validation: `passed`\n- Fine: `{selected['fine_id']}` at `{selected['timestamp_sec']}s`\n"
        f"- Gemini result: `{parsed['status']}` / `{parsed['requirement_effect']}` — {parsed['finding']}\n"
        "- First path: `miss -> 1 Gemini call -> immutable write`\n- Second path: `hit -> 0 Gemini calls`\n"
        f"- Tokens: `{used['input_tokens']}` input / `{used['output_tokens']}` visible output / `{used['thought_tokens']}` thought\n"
        f"- Latency: `{old['model_latency_sec']:.4f}s`; estimated paid-tier cost: `${estimate:.8f}`\n"
        f"- Cache write/hit: `{old['cache']['write_ms']:.4f}/{old['cache']['second_hit_ms']:.4f}ms`; record: `{old['cache']['record_bytes']} bytes`\n"
        "- Conflict resolver and formal pipeline integration: `not run`\n",encoding="utf-8")
    return old
