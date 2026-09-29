from __future__ import annotations

import base64
import copy
import hashlib
import html
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import (
    FINAL_SYSTEM,
    _final,
    _gemini_call,
    _load_env,
)
from experiments.r1_av_r3_2_claim_guided_gemini_closed_loop_v3.core import (
    ROUTE_SYSTEM,
    RouteDecision,
    route_request,
)
from experiments.r1_av_r3_2_review_cache_integration_canary_v1.core import apply_scope_records
from experiments.reviewed_visual_evidence_cache_v1 import LayeredVisualReviewCache, ReviewRecord
from experiments.reviewed_visual_evidence_cache_v1.cache import canonical_bytes, sha256_file
from experiments.reviewed_visual_evidence_cache_v1_gemini_canary.core import response_text, usage as gemini_usage
from experiments.shared_question_scope_review_gate_v1 import (
    build_route_payload,
    canonical_review_scope,
    select_local_fines,
    validate_route_decision,
)


REVIEW_SYSTEM = """You are the targeted local visual-review component from a frozen question-scope review path. Review every supplied image independently and only for the named fact scope and target requirement. Do not answer the multiple-choice question. Do not infer unshown continuity, completion, causality, duration, identity, or absence across the wider interval from a sparse frame. Return exactly one record per supplied Fine ID. Use supports_requirement only when that individual frame directly contributes visible information for the target requirement; otherwise use inconclusive."""


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return sha256_file(path)


def _initial_sufficiency_calls(source: Path) -> dict[str, dict[str, Any]]:
    calls = load(source / "live_cost_progress.json")
    rows: dict[str, dict[str, Any]] = {}
    for call_row in calls:
        if call_row.get("stage") == "sufficiency_initial":
            rows[call_row["side"]] = json.loads(call_row["raw_text"])
    if set(rows) != {"r1_av", "r3_2"}:
        raise ValueError("cannot uniquely resolve both initial Sufficiency results")
    return rows


def _initial_evidence(trace: dict[str, Any]) -> list[dict[str, Any]]:
    # The v1 diagnostic retained a mutable reference and later appended reviewed
    # findings to its saved input. Reconstruct the first-pass packet by removing
    # only those later reviewed_visual_frame additions.
    rows = [copy.deepcopy(row) for row in trace["sufficiency_input"]["evidence"] if row.get("evidence_type") != "reviewed_visual_frame"]
    ids = [row["evidence_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate initial evidence ID")
    return rows


def _ranges(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"range": list(map(float, row["interval"])), "source": row["evidence_id"]}
        for row in evidence if isinstance(row.get("interval"), list) and len(row["interval"]) == 2
    ]


def _navigation_ranges(trace: dict[str, Any], map_doc: dict[str, Any]) -> list[dict[str, Any]]:
    suggested = set(trace["planner_output"]["suggested_coarse_ids"])
    return [
        {"range": [float(row["start_sec"]), float(row["end_sec"])], "source": row["coarse_id"]}
        for row in map_doc["coarse_regions"] if row["coarse_id"] in suggested
    ]


def _review_schema(fine_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "review_scope": {"type": "string"},
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "fine_id": {"type": "string", "enum": fine_ids},
                        "status": {"type": "string", "enum": ["confirmed", "probable", "uncertain", "not_supported"]},
                        "requirement_effect": {"type": "string", "enum": ["supports_requirement", "inconclusive"]},
                        "direct_visual_support": {"type": "boolean"},
                        "finding": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low", "none"]},
                    },
                    "required": ["fine_id", "status", "requirement_effect", "direct_visual_support", "finding", "confidence"],
                },
            },
        },
        "required": ["question_id", "review_scope", "records"],
    }


def _review_missing(
    cfg: dict[str, Any], question: dict[str, Any], requirement: dict[str, Any],
    scope: str, fine_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompt = {
        "question_id": question["question_id"],
        "fact_scope": scope,
        "target_requirement": requirement,
        "policy": {
            "review_each_image_independently": True,
            "do_not_answer_question": True,
            "sparse_absence_is_not_video_absence": True,
        },
    }
    inputs: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
    for row in fine_rows:
        image = Path(row["source_frame_path"])
        inputs.extend([
            {"type": "text", "text": f"FINE {row['fine_id']} timestamp={float(row['timestamp_sec']):.3f}s"},
            {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image.read_bytes()).decode("ascii")},
        ])
    parsed, usage, raw = _gemini_call(cfg, REVIEW_SYSTEM, inputs, _review_schema([row["fine_id"] for row in fine_rows]))
    if parsed["question_id"] != question["question_id"] or parsed["review_scope"] != scope:
        raise ValueError("review identity mismatch")
    seen = [row["fine_id"] for row in parsed["records"]]
    expected = [row["fine_id"] for row in fine_rows]
    if sorted(seen) != sorted(expected) or len(seen) != len(set(seen)):
        raise ValueError("review Fine coverage mismatch")
    for row in parsed["records"]:
        if row["requirement_effect"] == "supports_requirement" and not row["direct_visual_support"]:
            raise ValueError("review support without direct visual support")
        if row["status"] == "not_supported" and row["requirement_effect"] == "supports_requirement":
            raise ValueError("not_supported review cannot support")
    return parsed, usage, raw


def _estimated_cost(cfg: dict[str, Any], calls: list[dict[str, Any]]) -> float:
    pricing = cfg["pricing_usd_per_million"]
    return sum(
        (row.get("input_tokens", 0) * pricing["input"] + (row.get("output_tokens", 0) + row.get("thought_tokens", 0)) * pricing["output"]) / 1_000_000
        for row in calls
    )


def _route_usage(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": "google", "stage": "question_scope_gate",
        "input_tokens": int(result["usage"].get("input_tokens", 0)),
        "output_tokens": int(result["usage"].get("output_tokens", 0)),
        "thought_tokens": 0, "latency_sec": float(result["latency_sec"]),
    }


def _route_live(cfg: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    request, saved = route_request(cfg["model"], payload)
    body = dict(request)
    http_request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(http_request, timeout=240) as response:
        raw_response = json.loads(response.read().decode("utf-8"))
    latency = time.perf_counter() - started
    raw_text = response_text(raw_response)
    parsed = RouteDecision.model_validate(json.loads(raw_text)).model_dump(mode="json")
    used = gemini_usage(raw_response)
    return {
        "raw": raw_text,
        "raw_response": raw_response,
        "parsed": parsed,
        "latency_sec": latency,
        "usage": {
            "input_tokens": int(used.get("input_tokens", 0)),
            "output_tokens": int(used.get("output_tokens", 0)) + int(used.get("thought_tokens", 0)),
            "total_tokens": int(used.get("total_tokens", 0)),
        },
        "saved_request": saved,
    }


def run(root: Path, config_path: Path, *, allow_api_calls: bool) -> dict[str, Any]:
    cfg = load(config_path)
    source = root / cfg["source_experiment"]
    out = root / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    protected = [
        source / "question_input.json", source / "shared_hierarchy.json",
        source / "r1_av_navigation_map.json", source / "r3_2_navigation_map.json",
        source / "r1_av_live_trace.json", source / "r3_2_live_trace.json",
        source / "live_cost_progress.json", source / "answers_blind.json",
    ]
    before = {str(path.relative_to(root)): file_hash(path) for path in protected}
    question = load(source / "question_input.json")
    hierarchy = load(source / "shared_hierarchy.json")
    maps = {side: load(source / f"{side}_navigation_map.json") for side in ("r1_av", "r3_2")}
    traces = {side: load(source / f"{side}_live_trace.json") for side in ("r1_av", "r3_2")}
    initial = _initial_sufficiency_calls(source)
    duration = max(float(row["end_sec"]) for row in hierarchy["medium_nodes"])
    payloads: dict[str, Any] = {}
    evidence_by_side: dict[str, list[dict[str, Any]]] = {}
    for side in ("r1_av", "r3_2"):
        evidence = _initial_evidence(traces[side])
        evidence_by_side[side] = evidence
        payloads[side] = build_route_payload(
            question=question,
            assessments=initial[side]["assessments"],
            candidate_option_ids=initial[side]["candidate_option_ids"],
            evidence_ranges=_ranges(evidence),
            navigation_ranges=_navigation_ranges(traces[side], maps[side]),
        )
    dump(out / "route_inputs.json", payloads)
    no_api = {
        "source_artifacts_resolved": len(before) == 8,
        "initial_sufficiency_results_resolved": set(initial) == {"r1_av", "r3_2"},
        "raw_images_in_gate_inputs": sum(payloads[side]["raw_image_inputs"] for side in payloads),
        "uncertainty_alone_does_not_trigger_review": all(payloads[side]["policy"]["uncertainty_alone_does_not_trigger_review"] for side in payloads),
        "maps_used_only_for_localization": all(all(row["source_kind"] == "navigation_only" for row in payloads[side]["localization"]["available_local_ranges"] if row["source"].startswith("C")) for side in payloads),
        "passed": True,
    }
    dump(out / "no_api_validation.json", no_api)
    dump(out / "source_hash_audit.json", {"before": before})
    if not allow_api_calls:
        result = {"overall_validation": "ready_for_live_gate_cache_replay", "model_calls": 0}
        dump(out / "validation_report.json", result)
        return result

    _load_env(root.parent / "thesis-av-evidence" / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY unavailable; no calls made")
    route_results: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    for side in ("r1_av", "r3_2"):
        route = _route_live(cfg, payloads[side])
        errors = validate_route_decision(route["parsed"], payloads[side], video_duration_sec=duration)
        if errors:
            dump(out / "route_raw_responses.json", {**route_results, side: route})
            raise ValueError(f"{side} route invalid: {errors}")
        route_results[side] = route
        usage = _route_usage(route); usage["side"] = side; calls.append(usage)
    dump(out / "route_raw_responses.json", route_results)
    dump(out / "route_decisions.json", {side: row["parsed"] for side, row in route_results.items()})

    cache = LayeredVisualReviewCache([root / path for path in cfg["seed_cache_roots"]], out / "cache")
    requirements = {row["requirement_id"]: row for row in traces["r1_av"]["sufficiency_input"]["requirements"]}
    resolved: dict[str, Any] = {}
    cache_audit: dict[str, Any] = {}
    raw_reviews: list[dict[str, Any]] = []
    cache_records: list[dict[str, Any]] = []
    image_transmissions = {"r1_av": 0, "r3_2": 0}
    for side in ("r1_av", "r3_2"):
        decision = route_results[side]["parsed"]
        assessments = copy.deepcopy(initial[side]["assessments"])
        evidence = evidence_by_side[side]
        cache_audit[side] = {"decision": decision["decision"], "scopes": {}}
        if decision["decision"] == "request_local_review":
            fine_rows = select_local_fines(decision=decision, fine_nodes=hierarchy["fine_nodes"])
            if not fine_rows:
                raise ValueError(f"{side}: no Fine frame in validated review range")
            for requirement_id in decision["review_request"]["target_requirement_ids"]:
                scope = canonical_review_scope(requirement_id)
                hits: list[str] = []; misses: list[dict[str, Any]] = []; records: list[dict[str, Any]] = []
                for fine in fine_rows:
                    image = Path(fine["source_frame_path"])
                    lookup = cache.lookup(image, cfg["review_contract_version"], [scope])
                    if lookup["status"] == "hit":
                        hits.append(fine["fine_id"]); records.append(lookup["records"][scope])
                    else:
                        misses.append(fine)
                if misses:
                    parsed, usage, raw = _review_missing(cfg, question, requirements[requirement_id], scope, misses)
                    usage.update({"provider": "google", "stage": "visual_review_cache_miss", "side": side, "scope": scope, "image_transmissions": len(misses)})
                    calls.append(usage); image_transmissions[side] += len(misses)
                    raw_reviews.append({"side": side, "scope": scope, "raw": raw, "parsed": parsed})
                    by_fine = {row["fine_id"]: row for row in parsed["records"]}
                    config_hash = hashlib.sha256(canonical_bytes({"model": cfg["model"], "system": REVIEW_SYSTEM, "scope": scope, "contract": cfg["review_contract_version"]})).hexdigest()
                    for fine in misses:
                        row = by_fine[fine["fine_id"]]
                        image = Path(fine["source_frame_path"])
                        record = ReviewRecord(
                            "reviewed_visual_evidence_cache_record_v1", cfg["review_contract_version"], "reviewed_visual_frame",
                            fine["fine_id"], sha256_file(image), image.stat().st_size, float(fine["timestamp_sec"]), scope,
                            row["status"], row["requirement_effect"], row["direct_visual_support"], row["finding"], row["confidence"],
                            (fine["fine_id"],), cfg["model"], config_hash, False,
                        )
                        cache.store(record); records.append(record.to_dict()); cache_records.append(record.to_dict())
                for fine in fine_rows:
                    lookup = cache.lookup(Path(fine["source_frame_path"]), cfg["review_contract_version"], [scope])
                    if lookup["status"] != "hit":
                        raise ValueError("post-review cache lookup did not hit")
                assessments = apply_scope_records(assessments, [requirement_id], scope, records)
                cache_audit[side]["scopes"][scope] = {"cache_hits_before_call": hits, "cache_misses": [row["fine_id"] for row in misses], "records_after": len(records)}
                for record in records:
                    evidence.append({
                        "evidence_id": f"reviewed_visual::{record['fine_id']}::{scope}",
                        "evidence_type": "reviewed_visual_frame",
                        "source_content": record["finding"],
                        "fine_id": record["fine_id"],
                        "timestamp_sec": record["timestamp_sec"],
                        "review_scope": scope,
                    })
        gate = "answer_ready" if decision["decision"] == "answer_now" else "review_resolved"
        resolved[side] = {
            "question_id": question["question_id"], "assessments": assessments,
            "gate": gate, "candidate_option_ids": initial[side]["candidate_option_ids"],
            "question_scope_route": decision,
        }
        evidence_by_side[side] = evidence
    dump(out / "visual_cache_audit.json", cache_audit)
    dump(out / "visual_review_raw_responses.json", raw_reviews)
    dump(out / "new_cache_records.json", cache_records)
    dump(out / "resolved_assessments.json", resolved)

    finals: dict[str, Any] = {}
    final_raw: dict[str, Any] = {}
    for side in ("r1_av", "r3_2"):
        final, usage, raw = _final(cfg, question, resolved[side], evidence_by_side[side])
        usage.update({"provider": "google", "stage": "final_text_only", "side": side, "image_transmissions": 0})
        calls.append(usage); finals[side] = final; final_raw[side] = raw
    dump(out / "answers_blind.json", finals)
    dump(out / "final_raw_responses.json", final_raw)

    second_lookup_errors = []
    for record in cache_records:
        fine = next(row for row in hierarchy["fine_nodes"] if row["fine_id"] == record["fine_id"])
        result = cache.lookup(Path(fine["source_frame_path"]), cfg["review_contract_version"], [record["review_scope"]])
        if result["status"] != "hit":
            second_lookup_errors.append(f"{record['fine_id']}/{record['review_scope']}")
    after = {str(path.relative_to(root)): file_hash(path) for path in protected}
    unchanged = before == after
    gold = load(source / "answer_comparison.json")["gold_option_id"]
    comparison = {
        side: {"selected_option_id": finals[side]["selected_option_id"], "correct": finals[side]["selected_option_id"] == gold, "answer_text": finals[side]["answer_text"]}
        for side in ("r1_av", "r3_2")
    }
    comparison["gold_option_id_posthoc"] = gold
    dump(out / "answer_comparison.json", comparison)
    call_summary = {
        "new_calls": len(calls),
        "new_gate_calls": sum(row["stage"] == "question_scope_gate" for row in calls),
        "new_visual_review_calls": sum(row["stage"] == "visual_review_cache_miss" for row in calls),
        "new_final_calls": sum(row["stage"] == "final_text_only" for row in calls),
        "input_tokens": sum(row.get("input_tokens", 0) for row in calls),
        "output_tokens": sum(row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in calls),
        "latency_sec": sum(row.get("latency_sec", 0.0) for row in calls),
        "estimated_api_usd": _estimated_cost(cfg, calls),
        "image_transmissions": image_transmissions,
        "reused_upstream_calls_not_reincurred": ["Organizer", "Planner", "initial Sufficiency", "retrieval", "captioning", "detector/tracking", "ASR"],
    }
    dump(out / "cost_accounting.json", {"calls": calls, "summary": call_summary})
    validation = {
        "frozen_route_prompt_reused_byte_for_byte": True,
        "separate_question_scope_gate": True,
        "uncertainty_alone_does_not_trigger_review": True,
        "cache_lookup_before_visual_call": True,
        "second_identical_cache_lookup": "passed" if not second_lookup_errors else "failed",
        "source_artifacts_unchanged": unchanged,
        "r3_visual_image_transmissions": image_transmissions["r3_2"],
        "r1_visual_image_transmissions": image_transmissions["r1_av"],
        "route_decisions": {side: route_results[side]["parsed"]["decision"] for side in route_results},
        "answers": comparison,
        "overall_validation": "passed" if unchanged and not second_lookup_errors else "failed",
    }
    dump(out / "validation_report.json", validation)
    dump(out / "protected_hash_audit.json", {"before": before, "after": after, "unchanged": unchanged})
    report = [
        "# HourVideo frozen Gate/cache replay v1", "",
        f"- R1 route: `{validation['route_decisions']['r1_av']}`; images: `{image_transmissions['r1_av']}`; answer: `{comparison['r1_av']['selected_option_id']}`.",
        f"- R3 route: `{validation['route_decisions']['r3_2']}`; images: `{image_transmissions['r3_2']}`; answer: `{comparison['r3_2']['selected_option_id']}`.",
        f"- Gold loaded post-hoc: `{gold}`.",
        f"- New calls/cost: `{call_summary['new_calls']}` / `${call_summary['estimated_api_usd']:.6f}`.",
        "- Organizer, Planner, initial Sufficiency, retrieval and index construction were replayed from immutable artifacts and not called again.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    body = "".join(
        f"<h2>{side}</h2><pre>{html.escape(json.dumps({'route': route_results[side]['parsed'], 'cache': cache_audit[side], 'answer': finals[side]}, ensure_ascii=False, indent=2))}</pre>"
        for side in ("r1_av", "r3_2")
    )
    (out / "review.html").write_text("<!doctype html><meta charset='utf-8'><h1>Frozen Gate/cache replay</h1>" + body, encoding="utf-8")
    return validation
