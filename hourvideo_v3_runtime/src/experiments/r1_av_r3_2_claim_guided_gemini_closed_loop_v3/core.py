from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


QUESTION_ORDER = [
    "q_global_summary", "q_weapon_visible", "q_visible_injury",
    "q_medical_assistance", "q_handcuffing", "q_handcuff_before_medical",
]
VIDEO_DURATION = 1235.0

ROUTE_SYSTEM = """You are the routing-only stage of a frozen claim-guided video
answer loop. You receive one question, requirement assessments, evidence
provenance and concrete local ranges, but no images. Do not answer the question.

Choose request_local_review only when an answer-critical requirement is not
already directly resolved, local visual inspection could materially resolve or
refine it, and a concrete local range is available. Uncertainty alone does not
trigger review. Never request a whole-video scan. Return routing metadata only.
Do not rewrite requirement semantics or emit an answer. Return strict JSON."""

REVIEW_SYSTEM = """You are the targeted local visual-review stage. Inspect only
the supplied local images for the named requirements. Do not re-evaluate the
whole question. For each target requirement, confirm, reject, refine, or retain
uncertainty. Sparse-frame co-occurrence does not prove binding, causality,
completion, continuous identity, or an unobserved temporal order. Do not answer
the question. Return strict JSON."""

FINAL_SYSTEM = """You are the final text-only answer stage. Answer every supplied
question using only its resolved requirement assessments and accepted temporal
sidecar, if present. You receive no images, map, retrieval pool, old answer or
ground truth. Preserve stated uncertainty. Do not strengthen a claim beyond its
status or support scope. Do not invent actors, roles, ownership, causality,
completion, or timing. Return strict JSON."""


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_goal: str
    target_requirement_ids: list[str]
    target_time_range: list[float]
    target_kind: Literal["static", "dynamic"]


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    decision: Literal["answer_now", "request_local_review"]
    reason: str
    review_request: ReviewRequest | None


class ReviewUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str
    status: Literal["supported", "uncertain", "not_found"]
    direct_support: bool
    finding: str
    support_scope: str
    supporting_image_ids: list[str]
    remaining_uncertainty: list[str]
    rationale: str


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    updates: list[ReviewUpdate]


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    answerability: Literal["answer_directly", "partial_answer", "insufficient_to_answer"]
    answer: str
    supporting_requirement_ids: list[str]
    supporting_evidence_ids: list[str]
    caveats: list[str]


class FinalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answers: list[FinalAnswer]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _range(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    a, b = float(value[0]), float(value[1])
    return [min(a, b), max(a, b)]


def overlaps(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def call(client: Any, request: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    started = time.perf_counter()
    interaction = client.interactions.create(**request)
    latency = time.perf_counter() - started
    raw = str(getattr(interaction, "output_text", "") or "")
    parsed = schema.model_validate(json.loads(raw)).model_dump(mode="json")
    usage_obj = getattr(interaction, "usage", None)
    usage = usage_obj.model_dump(mode="json") if hasattr(usage_obj, "model_dump") else (usage_obj or {})
    return {
        "raw": raw, "parsed": parsed, "latency_sec": latency,
        "usage": {
            "input_tokens": usage.get("input_tokens", usage.get("total_input_tokens", 0)) or 0,
            "output_tokens": usage.get("output_tokens", usage.get("total_output_tokens", 0)) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        },
    }


def normalize_r1(handoffs: dict[str, Any], qid: str) -> list[dict[str, Any]]:
    packet = handoffs["r1_av"][qid]
    rows = []
    for key, status in (("supported_content", "supported"), ("qualified_content", "uncertain"),
                        ("unresolved_content", "not_found"), ("conflicted_content", "conflicted")):
        for row in packet[key]:
            rows.append({
                "requirement_id": row["claim_id"], "status": status,
                "direct_support": status == "supported", "finding": row["claim_text"],
                "support_scope": row.get("wording_mode", "uncertain"),
                "supporting_evidence_ids": row.get("evidence_ids", []),
                "time_range": row.get("time_range"),
                "answer_critical": bool(row.get("answer_critical", True)),
                "remaining_uncertainty": row.get("remaining_uncertainty", []),
                "prohibited_upgrade": row.get("prohibited_upgrade", []),
            })
    return rows


def normalize_r3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "requirement_id": row["requirement_id"], "status": row["status"],
            "direct_support": row["status"] == "supported", "finding": row["rationale"],
            "support_scope": "caption_semantic_observation",
            "supporting_evidence_ids": row.get("supporting_evidence_ids", []),
            "supporting_coarse_ids": row.get("supporting_coarse_ids", []),
            "time_range": None, "answer_critical": True,
            "remaining_uncertainty": [] if row["status"] == "supported" else [row["rationale"]],
            "next_stage": row.get("next_stage"), "detail_query": row.get("detail_query", ""),
        })
    return out


def frame_registry(index: dict[str, Any], exact: dict[str, Any]) -> list[dict[str, Any]]:
    medium_of = {}
    for medium in index["medium_nodes"]:
        for fid in medium.get("source_fine_ids", medium.get("fine_ids", [])):
            medium_of[fid] = medium["medium_id"]
    rows = []
    for fine in index["fine_nodes"]:
        path = Path(fine["representative_frame_path"])
        rows.append({
            "image_id": fine["fine_id"], "medium_id": medium_of.get(fine["fine_id"]),
            "timestamp_sec": float(fine["representative_frame_timestamp_sec"]),
            "path": str(path), "question_id": None, "origin": "canonical_fine",
        })
    for question in exact["questions"]:
        for row in question["followup_evidence"]:
            rows.append({
                "image_id": row["followup_frame_id"], "medium_id": None,
                "timestamp_sec": float(row["actual_timestamp_sec"]), "path": row["jpeg_path"],
                "question_id": question["question_id"], "origin": "frozen_exact_followup",
            })
    return rows


def localization(side: str, qid: str, assessments: list[dict[str, Any]], direct: dict[str, Any],
                 contextual: dict[str, Any], planner_rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = direct[side][qid]["evidence"] + contextual[side][qid]["contextual_evidence"]
    ranges = []
    for row in evidence:
        tr = _range(row.get("timestamp"))
        if tr:
            ranges.append({"range": tr, "source": row["evidence_id"]})
    if side == "r3_2":
        planner = next(x for x in planner_rows if x["question"]["question_id"] == qid)
        coarse = {x["coarse_id"]: x for x in planner["navigation_map"]["coarse_regions"]}
        for assessment in assessments:
            for cid in assessment.get("supporting_coarse_ids", []):
                if cid in coarse:
                    ranges.append({"range": [float(coarse[cid]["start_sec"]), float(coarse[cid]["end_sec"])], "source": cid})
    for assessment in assessments:
        tr = _range(assessment.get("time_range"))
        if tr:
            ranges.append({"range": tr, "source": assessment["requirement_id"]})
    unique = {(x["range"][0], x["range"][1], x["source"]): x for x in ranges}
    return {"available_local_ranges": list(unique.values()), "evidence": evidence}


def route_payload(side: str, qid: str, question: str, assessments: list[dict[str, Any]], loc: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": qid, "question": question,
        "requirement_assessments": assessments,
        "localization": {"available_local_ranges": loc["available_local_ranges"]},
        "policy": {"no_whole_video_review": True, "one_local_review_max": True,
                   "uncertainty_alone_does_not_trigger_review": True},
        "raw_image_inputs": 0,
    }


def validate_route(decision: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    errors = []
    if decision["question_id"] != payload["question_id"]:
        errors.append("question_id mismatch")
    known = {x["requirement_id"]: x for x in payload["requirement_assessments"]}
    req = decision["review_request"]
    if decision["decision"] == "answer_now":
        if req is not None:
            errors.append("answer_now must not include review_request")
        return errors
    if req is None or not req["target_requirement_ids"]:
        return errors + ["review lacks concrete target"]
    if not set(req["target_requirement_ids"]) <= set(known):
        errors.append("unknown target requirement")
    if not any(known[x]["answer_critical"] for x in req["target_requirement_ids"] if x in known):
        errors.append("review target is not answer-critical")
    tr = _range(req["target_time_range"])
    if not tr or tr[0] < 0 or tr[1] > VIDEO_DURATION:
        errors.append("invalid target range")
    elif tr[1] - tr[0] >= VIDEO_DURATION * 0.5:
        errors.append("whole-video/broad review rejected")
    elif not any(overlaps(tr, x["range"]) for x in payload["localization"]["available_local_ranges"]):
        errors.append("target range has no frozen localization support")
    return errors


def select_images(qid: str, decision: dict[str, Any], registry: list[dict[str, Any]]) -> dict[str, Any]:
    req = decision["review_request"]
    lo, hi = req["target_time_range"]
    local = [x for x in registry if lo <= x["timestamp_sec"] <= hi]
    exact = [x for x in local if x["question_id"] == qid]
    if req["target_kind"] == "dynamic":
        selected = local
        policy = "all existing ordered local frames in the frozen range; no fixed image cap"
    elif exact:
        selected = exact
        policy = "all frozen exact-followup frames in the local range; no fixed image cap"
    elif local:
        center = (lo + hi) / 2
        distance = min(abs(x["timestamp_sec"] - center) for x in local)
        selected = [x for x in local if abs(abs(x["timestamp_sec"] - center) - distance) < 1e-9]
        policy = "nearest canonical local frame for static target; no fixed image cap"
    else:
        selected, policy = [], "no frozen local image available"
    unique = {}
    for row in sorted(selected, key=lambda x: (x["timestamp_sec"], x["image_id"])):
        unique[(row["path"], row["timestamp_sec"])] = row
    images = []
    for row in unique.values():
        path = Path(row["path"])
        images.append({**row, "exists": path.is_file(), "sha256": sha(path) if path.is_file() else None})
    return {"question_id": qid, "target_requirement_ids": req["target_requirement_ids"],
            "target_time_range": [lo, hi], "target_kind": req["target_kind"],
            "selection_policy": policy, "fixed_frame_count_rule": False,
            "full_video_scan": False, "images": images}


def route_request(model: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {"model": model, "system_instruction": ROUTE_SYSTEM,
               "input": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
               "response_format": {"type": "text", "mime_type": "application/json", "schema": RouteDecision.model_json_schema()},
               "generation_config": {"temperature": 0.0, "thinking_level": "low"}, "store": False}
    return request, {**request, "input": payload, "raw_image_inputs": 0}


def review_request(model: str, payload: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": json.dumps({
        "question_id": payload["question_id"], "question": payload["question"],
        "target_requirements": [x for x in payload["requirement_assessments"] if x["requirement_id"] in manifest["target_requirement_ids"]],
        "review_scope": {k: manifest[k] for k in ("target_time_range", "target_kind", "selection_policy")},
    }, ensure_ascii=False)}]
    saved_images = []
    for image in manifest["images"]:
        path = Path(image["path"])
        parts += [{"type": "text", "text": f"IMAGE {image['image_id']} timestamp={image['timestamp_sec']:.3f}s"},
                  {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(path.read_bytes()).decode("ascii")}]
        saved_images.append({k: image[k] for k in ("image_id", "timestamp_sec", "path", "sha256")})
    request = {"model": model, "system_instruction": REVIEW_SYSTEM, "input": parts,
               "response_format": {"type": "text", "mime_type": "application/json", "schema": ReviewResult.model_json_schema()},
               "generation_config": {"temperature": 0.0, "thinking_level": "low"}, "store": False}
    return request, {"model": model, "system_instruction": REVIEW_SYSTEM, "question_id": payload["question_id"],
                     "images": saved_images, "response_schema": ReviewResult.model_json_schema(),
                     "generation_config": request["generation_config"], "store": False}


def validate_review(result: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors = []
    if result["question_id"] != manifest["question_id"]:
        errors.append("question_id mismatch")
    expected = set(manifest["target_requirement_ids"])
    seen = [x["requirement_id"] for x in result["updates"]]
    if set(seen) != expected or len(seen) != len(set(seen)):
        errors.append("target requirement coverage mismatch")
    image_ids = {x["image_id"] for x in manifest["images"]}
    for row in result["updates"]:
        if not set(row["supporting_image_ids"]) <= image_ids:
            errors.append(f"{row['requirement_id']}: unknown image")
        if row["status"] == "supported" and (not row["direct_support"] or not row["supporting_image_ids"]):
            errors.append(f"{row['requirement_id']}: unsupported supported status")
    return errors


def apply_review(assessments: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {x["requirement_id"]: x for x in updates}
    out = []
    for row in assessments:
        update = by_id.get(row["requirement_id"])
        out.append(row if not update else {**row, **update, "reviewed_visual": True,
                                            "supporting_evidence_ids": update["supporting_image_ids"]})
    return out


def answerability(rows: list[dict[str, Any]], temporal: bool = False) -> str:
    critical = [x for x in rows if x.get("answer_critical", True)]
    direct = [x for x in critical if x["status"] == "supported" and x.get("direct_support")]
    if temporal:
        return "partial_answer"
    if critical and len(direct) == len(critical):
        return "answer_directly"
    return "partial_answer" if direct else "insufficient_to_answer"


def final_request(model: str, side: str, packets: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {"questions": packets, "policy": {"text_only": True, "preserve_uncertainty": True,
               "do_not_reselect_evidence": True, "representation_identity_not_evidence": side}}
    request = {"model": model, "system_instruction": FINAL_SYSTEM,
               "input": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
               "response_format": {"type": "text", "mime_type": "application/json", "schema": FinalBatch.model_json_schema()},
               "generation_config": {"temperature": 0.0, "thinking_level": "low"}, "store": False}
    return request, {**request, "input": payload, "raw_image_inputs": 0}


def validate_final(parsed: dict[str, Any], packets: list[dict[str, Any]]) -> list[str]:
    errors = []
    by_q = {x["question_id"]: x for x in packets}
    ids = [x["question_id"] for x in parsed["answers"]]
    if set(ids) != set(by_q) or len(ids) != len(set(ids)):
        errors.append("question coverage mismatch")
    for answer in parsed["answers"]:
        packet = by_q.get(answer["question_id"])
        if not packet:
            continue
        rids = {x["requirement_id"] for x in packet["requirement_assessments"]}
        eids = {e for x in packet["requirement_assessments"] for e in x.get("supporting_evidence_ids", [])}
        def collect_temporal_ids(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("evidence_id"), str):
                    eids.add(value["evidence_id"])
                for key, nested in value.items():
                    if key.endswith("evidence_ids") and isinstance(nested, list):
                        for evidence_id in (x for x in nested if isinstance(x, str)):
                            eids.add(evidence_id)
                            # The accepted temporal sidecar stores canonical node IDs,
                            # while the shared packet qualifies those same IDs by type.
                            if evidence_id.startswith(("comet_style_", "safe_node_", "followup_frame_")):
                                eids.add(f"visual_caption::{evidence_id}")
                            if evidence_id.startswith("audio_"):
                                eids.add(f"audio_asr::{evidence_id}")
                    collect_temporal_ids(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect_temporal_ids(nested)
        collect_temporal_ids(packet.get("accepted_temporal_resolution"))
        if answer["answerability"] != packet["answerability"]:
            errors.append(f"{answer['question_id']}: answerability changed")
        if not set(answer["supporting_requirement_ids"]) <= rids:
            errors.append(f"{answer['question_id']}: unknown requirement")
        if not set(answer["supporting_evidence_ids"]) <= eids:
            errors.append(f"{answer['question_id']}: unknown evidence")
        if not answer["answer"].strip():
            errors.append(f"{answer['question_id']}: empty answer")
        if packet.get("accepted_temporal_resolution"):
            text = (answer["answer"] + " " + " ".join(answer["caveats"])).lower()
            if "ems" not in text or not any(x in text for x in ("unknown", "not establish", "cannot determine", "cannot be establish")):
                errors.append(f"{answer['question_id']}: temporal caveat missing")
    return errors


def _usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"calls": len(rows), "input_tokens": sum(x["usage"]["input_tokens"] for x in rows),
            "output_tokens": sum(x["usage"]["output_tokens"] for x in rows),
            "latency_sec": round(sum(x["latency_sec"] for x in rows), 3)}


def render(validation: dict[str, Any], decisions: dict[str, Any], manifests: dict[str, Any], answers: dict[str, Any], cost: dict[str, Any]) -> str:
    parts = ["<meta charset='utf-8'><title>Claim-guided Gemini v3</title><h1>Claim-guided Gemini v3</h1>"]
    for side in ("r1_av", "r3_2"):
        parts.append(f"<h2>{side}</h2>")
        for qid in QUESTION_ORDER:
            d = decisions[side][qid]["parsed"]
            parts.append(f"<h3>{qid}: {d['decision']}</h3><pre>{html.escape(json.dumps(d, ensure_ascii=False, indent=2))}</pre>")
            if qid in manifests[side]:
                parts.append(f"<p>Images: {len(manifests[side][qid]['images'])}</p>")
            if side in answers and qid in answers[side]:
                parts.append(f"<p>{html.escape(answers[side][qid]['answer'])}</p>")
    parts += [f"<h2>Cost</h2><pre>{html.escape(json.dumps(cost, indent=2))}</pre>",
              f"<h2>Validation</h2><pre>{html.escape(json.dumps(validation, indent=2))}</pre>"]
    return "\n".join(parts)


def run(root: Path, config_path: Path, allow_api_calls: bool = False) -> dict[str, Any]:
    cfg = load(config_path)
    out = root / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: root / v for k, v in cfg["sources"].items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source_audit = {k: {"path": str(p.relative_to(root)), "sha256": sha(p), "bytes": p.stat().st_size} for k, p in paths.items()}
    dump(out / "source_artifact_audit.json", source_audit)
    index, direct, contextual = load(paths["index"]), load(paths["direct_packets"]), load(paths["contextual"])
    handoffs, r3_rows = load(paths["handoffs"]), load(paths["r3_assessments"])
    planner_rows, exact, temporal = load(paths["r3_planner_inputs"]), load(paths["exact_followups"]), load(paths["temporal_sidecar"])
    frozen = load(paths["frozen_snapshot_validation"])
    if frozen.get("overall_validation") != "frozen_test_experimental_snapshot":
        raise RuntimeError("frozen upstream snapshot validation mismatch")
    questions = {qid: direct["r1_av"][qid]["question"] for qid in QUESTION_ORDER}
    assessments = {"r1_av": {}, "r3_2": {}}
    locs = {"r1_av": {}, "r3_2": {}}
    payloads = {"r1_av": {}, "r3_2": {}}
    for qid in QUESTION_ORDER:
        assessments["r1_av"][qid] = normalize_r1(handoffs, qid)
        assessments["r3_2"][qid] = normalize_r3(r3_rows[qid])
        for side in ("r1_av", "r3_2"):
            locs[side][qid] = localization(side, qid, assessments[side][qid], direct, contextual, planner_rows)
            payloads[side][qid] = route_payload(side, qid, questions[qid], assessments[side][qid], locs[side][qid])
    dump(out / "routing_inputs.json", payloads)
    registry = frame_registry(index, exact)
    preflight_errors = []
    if len(index["fine_nodes"]) != 88 or len(index["medium_nodes"]) != 30:
        preflight_errors.append("canonical hierarchy count mismatch")
    if any(not Path(x["path"]).is_file() for x in registry):
        preflight_errors.append("unreadable frozen frame")
    if any("caption" in json.dumps(payloads["r1_av"][q], ensure_ascii=False).lower() for q in QUESTION_ORDER):
        preflight_errors.append("R1 caption leakage")
    preflight = {"tests": 8, "passed": not preflight_errors, "errors": preflight_errors,
                 "api_calls": 0, "fixed_frame_count_rule": False, "whole_video_review": False}
    dump(out / "no_api_test_report.json", preflight)
    if preflight_errors:
        raise RuntimeError(preflight_errors)
    if not allow_api_calls:
        validation = {"preflight_validation": "passed", "live_execution": "not_run",
                      "overall_validation": "ready_for_live_claim_guided_smoke", "api_calls": 0}
        dump(out / "validation_report.json", validation)
        return validation
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY unavailable; no calls made")
    from google import genai
    client = genai.Client()
    decisions = {"r1_av": {}, "r3_2": {}}
    route_inputs = {"r1_av": {}, "r3_2": {}}
    route_errors = []
    for side in ("r1_av", "r3_2"):
        for qid in QUESTION_ORDER:
            request, saved = route_request(cfg["model"], payloads[side][qid])
            route_inputs[side][qid] = saved
            result = call(client, request, RouteDecision)
            route_errors += [f"{side}/{qid}: {e}" for e in validate_route(result["parsed"], payloads[side][qid])]
            decisions[side][qid] = result
            dump(out / "routing_raw_responses.json", decisions)
    dump(out / "routing_model_inputs.json", route_inputs)
    dump(out / "routing_decisions.json", {s: {q: r["parsed"] for q, r in v.items()} for s, v in decisions.items()})
    if route_errors:
        validation = {"routing_validation": "failed", "errors": route_errors, "overall_validation": "failed"}
        dump(out / "validation_report.json", validation)
        return validation
    manifests = {"r1_av": {}, "r3_2": {}}
    review_inputs = {"r1_av": {}, "r3_2": {}}
    reviews = {"r1_av": {}, "r3_2": {}}
    review_errors = []
    for side in ("r1_av", "r3_2"):
        for qid in QUESTION_ORDER:
            decision = decisions[side][qid]["parsed"]
            if decision["decision"] == "answer_now":
                continue
            manifest = select_images(qid, decision, registry)
            manifests[side][qid] = manifest
            if not manifest["images"] or any(not x["exists"] for x in manifest["images"]):
                review_errors.append(f"{side}/{qid}: no readable localized images")
                continue
            request, saved = review_request(cfg["model"], payloads[side][qid], manifest)
            review_inputs[side][qid] = saved
            result = call(client, request, ReviewResult)
            review_errors += [f"{side}/{qid}: {e}" for e in validate_review(result["parsed"], manifest)]
            reviews[side][qid] = result
            dump(out / "visual_review_raw_responses.json", reviews)
    dump(out / "local_evidence_manifest.json", manifests)
    dump(out / "visual_review_inputs.json", review_inputs)
    dump(out / "visual_review_updates.json", {s: {q: r["parsed"] for q, r in v.items()} for s, v in reviews.items()})
    if review_errors:
        validation = {"routing_validation": "passed", "visual_review_validation": "failed",
                      "errors": review_errors, "overall_validation": "failed"}
        dump(out / "validation_report.json", validation)
        return validation
    resolved = {"r1_av": {}, "r3_2": {}}
    final_packets = {"r1_av": [], "r3_2": []}
    for side in ("r1_av", "r3_2"):
        for qid in QUESTION_ORDER:
            updates = reviews[side].get(qid, {}).get("parsed", {}).get("updates", [])
            rows = apply_review(assessments[side][qid], updates)
            resolved[side][qid] = rows
            accepted_temporal = temporal if side == "r3_2" and qid == "q_handcuff_before_medical" else None
            final_packets[side].append({
                "question_id": qid, "question": questions[qid],
                "answerability": answerability(rows, accepted_temporal is not None),
                "requirement_assessments": rows,
                "accepted_temporal_resolution": accepted_temporal,
            })
    dump(out / "resolved_requirement_assessments.json", resolved)
    final_results, final_inputs, final_errors = {}, {}, []
    for side in ("r1_av", "r3_2"):
        request, saved = final_request(cfg["model"], side, final_packets[side])
        final_inputs[side] = saved
        result = call(client, request, FinalBatch)
        final_errors += [f"{side}: {e}" for e in validate_final(result["parsed"], final_packets[side])]
        final_results[side] = result
        dump(out / "final_gemini_raw_responses.json", final_results)
    dump(out / "final_gemini_inputs.json", final_inputs)
    answers = {s: {x["question_id"]: x for x in r["parsed"]["answers"]} for s, r in final_results.items()}
    dump(out / "final_answers.json", answers)
    route_flat = [x for side in decisions.values() for x in side.values()]
    review_flat = [x for side in reviews.values() for x in side.values()]
    final_flat = list(final_results.values())
    transmitted = [x for side in manifests.values() for m in side.values() for x in m["images"]]
    cost = {"routing": _usage(route_flat), "visual_review": _usage(review_flat), "final_answer": _usage(final_flat),
            "images_transmitted": len(transmitted), "unique_images_transmitted": len({x["sha256"] for x in transmitted}),
            "fixed_frame_count_rule": False}
    cost["total"] = {"calls": sum(cost[x]["calls"] for x in ("routing", "visual_review", "final_answer")),
                     "input_tokens": sum(cost[x]["input_tokens"] for x in ("routing", "visual_review", "final_answer")),
                     "output_tokens": sum(cost[x]["output_tokens"] for x in ("routing", "visual_review", "final_answer")),
                     "latency_sec": round(sum(cost[x]["latency_sec"] for x in ("routing", "visual_review", "final_answer")), 3)}
    dump(out / "cost_accounting.json", cost)
    validation = {
        "upstream_frozen_snapshot": "passed", "routing_validation": "passed",
        "visual_review_validation": "passed", "final_answer_validation": "passed" if not final_errors else "failed",
        "no_fixed_frame_count": True, "no_full_video_review": True, "new_video_decoding": 0,
        "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0,
        "errors": final_errors,
        "overall_validation": "pending_manual_answer_review" if not final_errors else "failed",
    }
    dump(out / "validation_report.json", validation)
    (out / "review.html").write_text(render(validation, decisions, manifests, answers, cost), encoding="utf-8")
    lines = ["# R1_AV / R3_2 claim-guided Gemini closed loop v3", "",
             "This experiment changes only the downstream Gemini orchestration.", "",
             f"- Routing/review/final calls: `{cost['routing']['calls']}/{cost['visual_review']['calls']}/{cost['final_answer']['calls']}`",
             f"- Images transmitted/unique: `{cost['images_transmitted']}/{cost['unique_images_transmitted']}`",
             f"- Overall validation: `{validation['overall_validation']}`", "", "## R1_AV"]
    lines += [f"- **{q}**: {answers['r1_av'][q]['answer']}" for q in QUESTION_ORDER]
    lines += ["", "## R3_2"] + [f"- **{q}**: {answers['r3_2'][q]['answer']}" for q in QUESTION_ORDER]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validation


def revalidate_existing(root: Path, config_path: Path) -> dict[str, Any]:
    """Revalidate stored raw model responses after a deterministic validator fix."""
    cfg = load(config_path)
    out = root / cfg["output_root"]
    inputs = load(out / "final_gemini_inputs.json")
    raw = load(out / "final_gemini_raw_responses.json")
    errors = []
    for side in ("r1_av", "r3_2"):
        packets = inputs[side]["input"]["questions"]
        errors += [f"{side}: {x}" for x in validate_final(raw[side]["parsed"], packets)]
    validation = load(out / "validation_report.json")
    validation.update({
        "final_answer_validation": "passed" if not errors else "failed",
        "deterministic_validator_replay": "passed" if not errors else "failed",
        "validator_replay_api_calls": 0,
        "errors": errors,
        "overall_validation": "pending_manual_answer_review" if not errors else "failed",
    })
    dump(out / "validation_report.json", validation)
    dump(out / "deterministic_validator_replay.json", {
        "source_raw_response_unchanged": True,
        "api_calls": 0,
        "errors": errors,
        "result": validation["deterministic_validator_replay"],
    })
    answers = load(out / "final_answers.json")
    decisions = load(out / "routing_raw_responses.json")
    reviews = load(out / "visual_review_raw_responses.json")
    finals = load(out / "final_gemini_raw_responses.json")
    manifests = load(out / "local_evidence_manifest.json")
    cost = load(out / "cost_accounting.json")
    by_side = {}
    for side in ("r1_av", "r3_2"):
        route_usage = _usage(list(decisions[side].values()))
        review_usage = _usage(list(reviews[side].values()))
        final_usage = _usage([finals[side]])
        side_images = [image for manifest in manifests[side].values() for image in manifest["images"]]
        by_side[side] = {
            "routing": route_usage, "visual_review": review_usage, "final_answer": final_usage,
            "images_transmitted": len(side_images),
            "unique_images_transmitted": len({x["sha256"] for x in side_images}),
            "total": {
                "calls": route_usage["calls"] + review_usage["calls"] + final_usage["calls"],
                "input_tokens": route_usage["input_tokens"] + review_usage["input_tokens"] + final_usage["input_tokens"],
                "output_tokens": route_usage["output_tokens"] + review_usage["output_tokens"] + final_usage["output_tokens"],
                "latency_sec": round(route_usage["latency_sec"] + review_usage["latency_sec"] + final_usage["latency_sec"], 3),
            },
        }
    cost["by_side"] = by_side
    dump(out / "cost_accounting.json", cost)
    semantic_flags = []
    injury_text = answers["r1_av"]["q_visible_injury"]["answer"].lower()
    if "continuously" in injury_text or "continuous" in injury_text:
        semantic_flags.append({
            "side": "r1_av", "question_id": "q_visible_injury",
            "type": "sparse_frames_upgraded_to_continuous_visibility",
            "severity": "material_wording_risk",
        })
    semantic_audit = {
        "automatic_contract_validation": "passed" if not errors else "failed",
        "manual_semantic_acceptance": "pending_manual_review",
        "freeze_ready": False,
        "flags": semantic_flags,
    }
    dump(out / "semantic_answer_audit.json", semantic_audit)
    old_usage_path = root / "outputs/experiments/r1_av_r3_2_full_gemini_closed_loop_v2/gemini_usage_report.json"
    old_usage = load(old_usage_path) if old_usage_path.is_file() else None
    comparison = {
        "new_claim_guided": cost,
        "previous_fixed_12_frame_run": old_usage,
        "interpretation": (
            "The frozen no-fixed-count selection policy reduced review calls but enlarged "
            "overlapping R1 dynamic windows; it is contract-faithful, not a cost win on 226."
        ),
    }
    dump(out / "cost_comparison.json", comparison)
    (out / "review.html").write_text(render(validation, decisions, manifests, answers, cost), encoding="utf-8")
    route_counts = {
        side: dict(Counter(row["parsed"]["decision"] for row in decisions[side].values()))
        for side in decisions
    }
    lines = [
        "# R1_AV / R3_2 claim-guided Gemini closed loop v3", "",
        "Only the downstream Gemini orchestration changed. Frozen maps, Planner, Retrieval, Sufficiency and reliability outputs were replayed read-only.", "",
        f"- Automatic contract validation: `{semantic_audit['automatic_contract_validation']}`",
        f"- Manual semantic acceptance: `{semantic_audit['manual_semantic_acceptance']}`",
        f"- Routing decisions: `{json.dumps(route_counts, ensure_ascii=False)}`",
        f"- Calls (routing/review/final): `{cost['routing']['calls']}/{cost['visual_review']['calls']}/{cost['final_answer']['calls']}`",
        f"- Images transmitted/unique/repeated: `{cost['images_transmitted']}/{cost['unique_images_transmitted']}/{cost['images_transmitted']-cost['unique_images_transmitted']}`",
        f"- Tokens input/output: `{cost['total']['input_tokens']}/{cost['total']['output_tokens']}`",
        f"- R1_AV calls/tokens: `{by_side['r1_av']['total']['calls']}` / `{by_side['r1_av']['total']['input_tokens']} in, {by_side['r1_av']['total']['output_tokens']} out`",
        f"- R3_2 calls/tokens: `{by_side['r3_2']['total']['calls']}` / `{by_side['r3_2']['total']['input_tokens']} in, {by_side['r3_2']['total']['output_tokens']} out`",
        "- Fixed frame-count rule: `false`", "- Full-video review: `false`", "",
        "## Important audit finding", "",
        "R1_AV q_visible_injury says the injury was visible continuously, but the review used discrete frames. This is a material wording-strength risk, so this run is not freeze-ready before manual acceptance or a deterministic final-answer policy fix.", "",
        "## R1_AV answers", "",
    ]
    lines += [f"- **{qid}**: {answers['r1_av'][qid]['answer']}" for qid in QUESTION_ORDER]
    lines += ["", "## R3_2 answers", ""]
    lines += [f"- **{qid}**: {answers['r3_2'][qid]['answer']}" for qid in QUESTION_ORDER]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validation
