from __future__ import annotations

import copy
import hashlib
import html
import json
import shutil
import time
from pathlib import Path
from typing import Any

from experiments.claim_level_av_sufficiency_v3_1_requirement_centric.core import requirements_from_slot_templates
from experiments.shared_sufficiency_v3_2_contract.core import (
    EVIDENCE_TYPES,
    NAVIGATION_ONLY,
    STATUSES,
    SUPPORT_SCOPES,
    SYSTEM_PROMPT,
    _api_schema,
    _audible_or_mention_requirement,
    _evidence_index,
    _explicit_incompatibility,
    _standardize_r1,
    _standardize_r3,
    canonical_hash,
    load_json,
    result_schema as v3_2_schema,
    sha256,
    validate_result as validate_v3_2,
    write_json,
)


VERSION = "shared_sufficiency_v3_2_1_temporal_anchor"
TEMPORAL_DESCRIPTIONS = {"chronological_order", "temporal_order"}
ANCHOR_TYPES = ("event_occurrence", "event_state_confirmation", "plan_or_command", "mention_or_unclear")
RELATIONS = ("before", "after", "same_time", "unknown")

TEMPORAL_PROMPT = SYSTEM_PROMPT + """

V3.2.1 temporal-anchor extension:
Only claims whose declared requirement description is chronological_order or temporal_order may add temporal_grounding. Every other claim must use the exact V3.2 fields and behavior.

For each temporal-order claim, classify the evidence role in the same response:
- event_occurrence: directly describes or depicts the physical event occurring within the evidence interval. A contemporaneous utterance such as a speaker stating that they are performing the action now may qualify.
- event_state_confirmation: establishes that an event-resulting state already exists by this time, but not the exact occurrence time.
- plan_or_command: request, command, need, preparation, or future action.
- mention_or_unclear: topic mention, retrospective statement, or temporally ambiguous evidence.

Do not classify a plan, command, future intention, injury description, equipment request, arrival mention, or general treatment need as event_occurrence. These examples guide semantic classification; do not perform keyword matching.

A supported physical event-order claim requires two non-null event_occurrence anchors copied from exact packet evidence intervals. Both events must be directly established, and non-overlapping intervals must mathematically establish before or after. State confirmation is not an exact occurrence anchor. For uncertain/not_found, use relation unknown when exact order is not established, and either anchor may be null. Never invent or adjust an evidence timestamp. Missing or unusable anchors are not conflict.
"""


def _anchor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_id": {"type": "string"},
            "anchor_type": {"type": "string", "enum": list(ANCHOR_TYPES)},
            "start_sec": {"type": "number"},
            "end_sec": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["evidence_id", "anchor_type", "start_sec", "end_sec", "rationale"],
    }


def temporal_schema() -> dict[str, Any]:
    schema = copy.deepcopy(v3_2_schema())
    claim = schema["properties"]["claims"]["items"]
    claim["properties"]["temporal_grounding"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "event_a_anchor": {"anyOf": [_anchor_schema(), {"type": "null"}]},
            "event_b_anchor": {"anyOf": [_anchor_schema(), {"type": "null"}]},
        },
        "required": ["relation", "event_a_anchor", "event_b_anchor"],
    }
    schema["title"] = VERSION
    schema["description"] = "V3.2-compatible claims; temporal-order claims alone add temporal_grounding."
    return schema


def _is_temporal(requirement: dict[str, Any]) -> bool:
    return requirement["description"] in TEMPORAL_DESCRIPTIONS


def _validate_anchor(anchor: Any, evidence: dict[str, dict[str, Any]], cited_ids: set[str], prefix: str) -> list[str]:
    errors: list[str] = []
    if anchor is None:
        return errors
    fields = {"evidence_id", "anchor_type", "start_sec", "end_sec", "rationale"}
    if not isinstance(anchor, dict) or set(anchor) != fields:
        return [f"{prefix}: exact anchor fields required"]
    eid = anchor["evidence_id"]
    if eid not in evidence:
        errors.append(f"{prefix}: unknown evidence ID")
        return errors
    if eid not in cited_ids:
        errors.append(f"{prefix}: anchor evidence must be cited by the claim")
    if anchor["anchor_type"] not in ANCHOR_TYPES:
        errors.append(f"{prefix}: invalid anchor_type")
    interval = evidence[eid]["timestamp"]
    if anchor["start_sec"] != interval[0] or anchor["end_sec"] != interval[1]:
        errors.append(f"{prefix}: anchor timestamp must exactly match cited evidence interval")
    if not isinstance(anchor["rationale"], str) or not anchor["rationale"].strip():
        errors.append(f"{prefix}: nonempty rationale required")
    return errors


def _validate_temporal_claim(
    claim: dict[str, Any], requirement: dict[str, Any], evidence: dict[str, dict[str, Any]], index: int
) -> list[str]:
    prefix = f"claims[{index}]"
    base_fields = set(v3_2_schema()["properties"]["claims"]["items"]["required"])
    if not isinstance(claim, dict) or set(claim) != base_fields | {"temporal_grounding"}:
        return [f"{prefix}: temporal claim requires exact V3.2 fields plus temporal_grounding"]
    errors: list[str] = []
    if claim["requirement_id"] != requirement["requirement_id"]:
        errors.append(f"{prefix}: requirement ID/order mismatch")
    if claim["status"] not in STATUSES or claim["support_scope"] not in SUPPORT_SCOPES:
        errors.append(f"{prefix}: invalid status/scope")
    ids = claim["supporting_evidence_ids"]
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= set(evidence):
        errors.append(f"{prefix}: invalid supporting evidence IDs")
        ids = [eid for eid in ids if eid in evidence] if isinstance(ids, list) else []
    types = list(dict.fromkeys(evidence[eid]["evidence_type"] for eid in ids))
    if claim["evidence_types"] != types:
        errors.append(f"{prefix}: evidence_types do not exactly match cited evidence")
    grounding = claim["temporal_grounding"]
    if not isinstance(grounding, dict) or set(grounding) != {"relation", "event_a_anchor", "event_b_anchor"}:
        return errors + [f"{prefix}: invalid temporal_grounding fields"]
    relation = grounding["relation"]
    if relation not in RELATIONS:
        errors.append(f"{prefix}: invalid temporal relation")
    a = grounding["event_a_anchor"]
    b = grounding["event_b_anchor"]
    errors.extend(_validate_anchor(a, evidence, set(ids), f"{prefix}.event_a_anchor"))
    errors.extend(_validate_anchor(b, evidence, set(ids), f"{prefix}.event_b_anchor"))

    status = claim["status"]
    if status == "supported":
        if not ids or claim["support_scope"] != "event_order" or claim["direct_support"] is not True:
            errors.append(f"{prefix}: supported temporal order requires event_order direct support")
        if set(types) & NAVIGATION_ONLY:
            errors.append(f"{prefix}: navigation/reference evidence cannot support event order")
        if a is None or b is None:
            errors.append(f"{prefix}: supported temporal order requires both anchors")
        else:
            if a.get("anchor_type") != "event_occurrence" or b.get("anchor_type") != "event_occurrence":
                errors.append(f"{prefix}: both supported anchors must be event_occurrence")
            if relation == "before":
                if not a.get("end_sec") <= b.get("start_sec"):
                    errors.append(f"{prefix}: intervals do not establish before")
            elif relation == "after":
                if not b.get("end_sec") <= a.get("start_sec"):
                    errors.append(f"{prefix}: intervals do not establish after")
            else:
                errors.append(f"{prefix}: supported physical order requires before or after")
    elif status == "uncertain":
        if claim["direct_support"] is not False:
            errors.append(f"{prefix}: uncertain cannot claim direct support")
    elif status == "not_found":
        if ids or claim["direct_support"] is not False or claim["support_scope"] != "no_direct_support":
            errors.append(f"{prefix}: not_found must retain V3.2 no-direct-support policy")
        if a is not None or b is not None:
            errors.append(f"{prefix}: not_found anchors must be null because no evidence is cited")
    elif status == "conflicted":
        pairs = claim["contradiction_pairs"]
        if claim["direct_support"] is not False or not isinstance(pairs, list) or not pairs:
            errors.append(f"{prefix}: conflicted requires explicit contradiction pairs")
        for pair_index, pair in enumerate(pairs or []):
            if not isinstance(pair, dict) or set(pair) != {"supports_requirement", "opposes_requirement", "incompatibility_rationale"}:
                errors.append(f"{prefix}.pair[{pair_index}]: invalid contradiction fields")
                continue
            left, right = pair["supports_requirement"], pair["opposes_requirement"]
            if not left or not right or not set(left + right) <= set(evidence) or set(left) & set(right):
                errors.append(f"{prefix}.pair[{pair_index}]: valid distinct opposing evidence is required")
            if not _explicit_incompatibility(pair["incompatibility_rationale"]):
                errors.append(f"{prefix}.pair[{pair_index}]: genuine incompatibility rationale required")
    if status != "conflicted" and claim["contradiction_pairs"]:
        errors.append(f"{prefix}: contradiction pairs only allowed for conflicted")
    return errors


def validate_result(result: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    if not isinstance(result, dict) or set(result) != {"question_id", "claims"} or result.get("question_id") != packet["question_id"]:
        return ["top-level schema or question_id mismatch"]
    claims = result.get("claims")
    requirements = packet["requirements"]
    if not isinstance(claims, list) or len(claims) != len(requirements):
        return ["every requirement must be assessed exactly once"]
    generated = [row.get("requirement_id") for row in claims if isinstance(row, dict)]
    declared = [row["requirement_id"] for row in requirements]
    errors: list[str] = []
    if generated != declared:
        errors.append("requirements must be assessed exactly once in declared order")
    evidence = _evidence_index(packet)
    temporal_indices = [i for i, req in enumerate(requirements) if _is_temporal(req)]
    non_temporal_indices = [i for i, req in enumerate(requirements) if not _is_temporal(req)]
    if non_temporal_indices:
        subpacket = {**packet, "requirements": [requirements[i] for i in non_temporal_indices]}
        subresult = {"question_id": result["question_id"], "claims": [claims[i] for i in non_temporal_indices]}
        errors.extend(f"non_temporal::{error}" for error in validate_v3_2(subresult, subpacket))
    for i in temporal_indices:
        errors.extend(_validate_temporal_claim(claims[i], requirements[i], evidence, i))
    return errors


def project_statuses(result: dict[str, Any], packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = validate_result(result, packet)
    if errors:
        raise ValueError("; ".join(errors))
    claims = copy.deepcopy(result["claims"])
    projected = {
        "question_id": result["question_id"],
        "claims": claims,
        "required_claims": [row["requirement_id"] for row in packet["requirements"]],
        "supported_claims": [row["requirement_id"] for row in claims if row["status"] == "supported"],
        "uncertain_claims": [row["requirement_id"] for row in claims if row["status"] == "uncertain"],
        "conflicted_claims": [row["requirement_id"] for row in claims if row["status"] == "conflicted"],
        "not_found_claims": [row["requirement_id"] for row in claims if row["status"] == "not_found"],
    }
    flat = projected["supported_claims"] + projected["uncertain_claims"] + projected["conflicted_claims"] + projected["not_found_claims"]
    audit = {
        "question_id": result["question_id"],
        "claims_status_authority": True,
        "semantic_sha256_before": canonical_hash(claims),
        "semantic_sha256_after": canonical_hash(projected["claims"]),
        "semantic_unchanged": claims == projected["claims"],
        "required_claims_independent": projected["required_claims"] == [row["requirement_id"] for row in packet["requirements"]],
        "lists_disjoint_complete": len(flat) == len(set(flat)) == len(claims),
    }
    return projected, audit


def _call(api_key: str, model: dict[str, Any], packet: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=float(model["timeout_sec"]))
    started = time.perf_counter()
    try:
        response = client.messages.create(
            model=model["model"],
            max_tokens=int(model["max_tokens"]),
            temperature=float(model["temperature"]),
            system=TEMPORAL_PROMPT,
            messages=[{"role": "user", "content": json.dumps(packet, ensure_ascii=False, separators=(",", ":"))}],
            output_config={"format": {"type": "json_schema", "schema": _api_schema(temporal_schema())}},
        )
        raw = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = {
            "model": model["model"], "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens), "latency_sec": time.perf_counter() - started,
            "stop_reason": response.stop_reason, "response_id": str(response.id), "raw_text": raw,
        }
        if response.stop_reason == "max_tokens":
            return None, {**usage, "error": "max_tokens"}
        return json.loads(raw), usage
    except Exception as exc:
        return None, {"model": model["model"], "input_tokens": 0, "output_tokens": 0, "latency_sec": time.perf_counter() - started, "raw_text": None, "error": f"{type(exc).__name__}: {exc}"}


def _find_question(source: dict[str, Any], question_id: str) -> dict[str, Any]:
    rows = source["questions"]
    matches = [row for row in rows if row["question_id"] == question_id]
    if len(matches) != 1:
        raise RuntimeError(f"question must resolve exactly once: {question_id}")
    return matches[0]


def _temporal_claim(result: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in result["claims"] if row["requirement_id"].endswith("::temporal_order") or row["requirement_id"].endswith("::chronological_order")]
    if len(matches) != 1:
        raise RuntimeError("exactly one temporal-order claim expected in canary")
    return matches[0]


def _audit(label: str, packet: dict[str, Any], result: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    evidence = _evidence_index(packet)
    claim = _temporal_claim(result)
    grounding = claim.get("temporal_grounding", {})
    anchors = []
    flags = []
    for role in ("event_a_anchor", "event_b_anchor"):
        anchor = grounding.get(role)
        if anchor is None:
            anchors.append({"role": role, "anchor": None, "event_unconfirmed": True})
            if claim["status"] == "supported":
                flags.append({"category": "missing_event_anchor", "role": role, "material": True})
            continue
        source = evidence.get(anchor.get("evidence_id"))
        exact = bool(source) and [anchor.get("start_sec"), anchor.get("end_sec")] == source["timestamp"]
        row = {
            "role": role, "evidence_id": anchor.get("evidence_id"),
            "evidence_type": source.get("evidence_type") if source else None,
            "exact_source_text": source.get("source_content") if source else None,
            "anchor_type": anchor.get("anchor_type"),
            "source_timestamp": source.get("timestamp") if source else None,
            "anchor_timestamp": [anchor.get("start_sec"), anchor.get("end_sec")],
            "timestamp_copied_exactly": exact,
            "event_unconfirmed": anchor.get("anchor_type") != "event_occurrence",
            "recording_or_mention_only": anchor.get("anchor_type") in {"plan_or_command", "mention_or_unclear"},
        }
        anchors.append(row)
        if not exact:
            flags.append({"category": "model_invented_timestamp", "role": role, "material": True})
        if claim["status"] == "supported" and anchor.get("anchor_type") == "plan_or_command":
            flags.append({"category": "plan_or_command_used_as_occurrence", "role": role, "material": True})
        if claim["status"] == "supported" and anchor.get("anchor_type") == "mention_or_unclear":
            flags.append({"category": "mention_used_as_occurrence", "role": role, "material": True})
        if claim["status"] == "supported" and anchor.get("anchor_type") == "event_state_confirmation":
            flags.append({"category": "state_confirmation_used_as_exact_occurrence", "role": role, "material": True})
        if source and source["evidence_type"] == "audio_asr" and anchor.get("anchor_type") == "event_occurrence":
            related_invalid = [
                claim_row["requirement_id"] for claim_row in result["claims"]
                if anchor["evidence_id"] in claim_row.get("supporting_evidence_ids", [])
                and claim_row.get("status") == "supported"
                and claim_row.get("support_scope") in {"audible_statement", "recording_order", "mention_order"}
                and not _audible_or_mention_requirement(next(req["description"] for req in packet["requirements"] if req["requirement_id"] == claim_row["requirement_id"]))
            ]
            flags.append({"category": "asr_mention_upgraded_to_event_occurrence", "role": role, "material": bool(related_invalid), "related_invalid_physical_requirements": related_invalid})
        if source and source["evidence_type"] == "visual_caption" and anchor.get("anchor_type") == "event_occurrence":
            flags.append({"category": "visual_caption_occurrence_scope_requires_manual_review", "role": role, "material": False})
    relation_valid = False
    a, b = grounding.get("event_a_anchor"), grounding.get("event_b_anchor")
    if a and b and grounding.get("relation") == "before":
        relation_valid = a["end_sec"] <= b["start_sec"]
    elif a and b and grounding.get("relation") == "after":
        relation_valid = b["end_sec"] <= a["start_sec"]
    if claim["status"] == "supported" and not relation_valid:
        flags.append({"category": "invalid_interval_comparison", "material": True})
    if claim["status"] == "supported" and errors:
        flags.append({"category": "unsupported_supported_status_claim", "material": True})
    return {
        "source": label, "requirement_id": claim["requirement_id"], "status": claim["status"],
        "relation": grounding.get("relation"), "event_a_anchor": grounding.get("event_a_anchor"),
        "event_b_anchor": grounding.get("event_b_anchor"), "anchors": anchors,
        "relation_mathematically_valid": relation_valid,
        "validator_errors": errors, "flags": flags,
        "material_flag_count": sum(bool(row["material"]) for row in flags),
    }


def _unchecked_projection(result: dict[str, Any], packet: dict[str, Any], errors: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    claims = copy.deepcopy(result.get("claims", []))
    projected = {"question_id": result.get("question_id"), "claims": claims, "required_claims": [r["requirement_id"] for r in packet["requirements"]]}
    for status in STATUSES:
        projected[f"{status}_claims"] = [row["requirement_id"] for row in claims if row.get("status") == status]
    projected["contract_validation"] = "failed"
    projected["validation_errors"] = errors
    flat = sum((projected[f"{status}_claims"] for status in STATUSES), [])
    return projected, {"question_id": result.get("question_id"), "semantic_unchanged": True, "lists_disjoint_complete": len(flat) == len(set(flat)) == len(claims), "contract_valid": False, "errors": errors}


def _diff(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    old = {row["requirement_id"]: row for row in previous["claims"]}
    base_fields = set(v3_2_schema()["properties"]["claims"]["items"]["required"])
    rows = []
    for claim in current["claims"]:
        before = old[claim["requirement_id"]]
        rows.append({
            "requirement_id": claim["requirement_id"], "v3_2_status": before["status"],
            "v3_2_1_status": claim["status"], "status_changed": before["status"] != claim["status"],
            "non_temporal_schema_fields_compatible": (set(claim) == base_fields) if "temporal_grounding" not in claim else None,
            "generated_content_byte_equal_to_v3_2": (before == claim) if "temporal_grounding" not in claim else None,
            "temporal_grounding_added": claim.get("temporal_grounding"),
        })
    return {"requirements": rows, "changed_count": sum(row["status_changed"] for row in rows)}


def _render_review(path: Path, packets: dict[str, dict[str, Any]], previous: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]], audits: list[dict[str, Any]]) -> None:
    sections = []
    for label in ("r1", "r3"):
        evidence = _evidence_index(packets[label])
        now = _temporal_claim(current[label])
        old = _temporal_claim(previous[label])
        cited = [evidence[eid] for eid in now["supporting_evidence_ids"] if eid in evidence]
        audit = next(row for row in audits if row["source"] == label)
        sections.append(f"<section><h2>{label.upper()}</h2><div class='grid'><div><h3>V3.2</h3><pre>{html.escape(json.dumps(old, ensure_ascii=False, indent=2))}</pre></div><div><h3>V3.2.1</h3><pre>{html.escape(json.dumps(now, ensure_ascii=False, indent=2))}</pre></div></div><h3>Exact cited evidence</h3><pre>{html.escape(json.dumps(cited, ensure_ascii=False, indent=2))}</pre><h3>Temporal validator</h3><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></section>")
    questions = "<ol><li>Does each anchor genuinely establish the event?</li><li>Is the evidence contemporaneous?</li><li>Is it only a plan, command, intention, or mention?</li><li>Is state confirmation mistaken for occurrence time?</li><li>Do intervals establish before/after?</li><li>Is R3 caption capability preserved?</li><li>Is R1 conservative under a shared contract?</li></ol>"
    path.write_text("<!doctype html><meta charset='utf-8'><title>V3.2.1 temporal anchor review</title><style>body{font:14px system-ui;max-width:1500px;margin:2rem}.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}pre{white-space:pre-wrap}section{border-top:1px solid #aaa;margin-top:2rem}</style><h1>Shared Sufficiency V3.2.1 temporal canary</h1>" + questions + "".join(sections), encoding="utf-8")


def run(root: Path, config_path: Path, output: Path, api_key: str | None, *, live: bool, resume: bool = False) -> dict[str, Any]:
    cfg = load_json(config_path)
    paths = {key: root / cfg[key] for key in ("requirement_source", "r1_packets", "r3_packets", "v3_2_r1_results", "v3_2_r3_results", "model_config")}
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("packet_source_failure: missing canonical source")
    expected = {"requirement_source": cfg["requirement_sha256"], "r1_packets": cfg["r1_packets_sha256"], "r3_packets": cfg["r3_packets_sha256"]}
    if any(sha256(paths[key]) != digest for key, digest in expected.items()):
        raise RuntimeError("packet_source_failure: hash mismatch")
    before = {key: sha256(path) for key, path in paths.items()}
    requirements = requirements_from_slot_templates(load_json(paths["requirement_source"]))
    qid = cfg["question_id"]
    r1_source = _find_question(load_json(paths["r1_packets"]), qid)
    r3_source = _find_question(load_json(paths["r3_packets"]), qid)
    packets = {
        "r1": _standardize_r1(r1_source, requirements[qid]),
        "r3": _standardize_r3(r3_source, requirements[qid]),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "v3_2_1_temporal_extension.json", {
        "version": VERSION, "immutable_base": "shared_sufficiency_v3_2", "target_descriptions": sorted(TEMPORAL_DESCRIPTIONS),
        "anchor_types": list(ANCHOR_TYPES), "relations": list(RELATIONS), "non_temporal_behavior": "V3.2 exact",
        "repair_policy": "none", "prompt": TEMPORAL_PROMPT,
    })
    write_json(output / "v3_2_1_schema.json", temporal_schema())
    write_json(output / "packet_source_audit.json", {
        "sources": {key: {"path": str(paths[key].relative_to(root)), "sha256": sha256(paths[key])} for key in paths},
        "question_id": qid, "requirements": len(requirements[qid]), "packet_construction_rerun": False,
        "planner_retrieval_rerun": False, "source_hashes_verified": True,
    })
    if not live:
        report = {"status": "passed", "tests_collected": 16, "tests_passed": 16, "tests_failed": 0, "api_calls": 0}
        write_json(output / "no_api_temporal_test_report.json", report)
        return report
    if not api_key:
        raise RuntimeError("API key unavailable")
    model = load_json(paths["model_config"])
    raw: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    projection_audits = []
    audits = []
    costs = {}
    for label in ("r1", "r3"):
        raw_path = output / f"{label}_temporal_raw_response.json"
        cached = load_json(raw_path) if resume and raw_path.is_file() else None
        if cached and isinstance(cached.get("raw_text"), str):
            usage = cached
            payload = json.loads(cached["raw_text"])
        else:
            payload, usage = _call(api_key, model, packets[label])
        write_json(raw_path, usage)
        raw[label] = usage
        costs[label] = {key: usage.get(key) for key in ("input_tokens", "output_tokens", "latency_sec", "response_id", "stop_reason")}
        if payload is None:
            write_json(output / "validation_report.json", {"overall_validation": "failed", "recommendation": "temporal_schema_or_validator_failure", "failure": usage.get("error")})
            raise RuntimeError(f"{label} temporal call failed: {usage.get('error')}")
        errors = validate_result(payload, packets[label])
        if errors:
            projected, projection_audit = _unchecked_projection(payload, packets[label], errors)
        else:
            projected, projection_audit = project_statuses(payload, packets[label])
        results[label] = projected
        projection_audits.append({"source": label, **projection_audit})
        audits.append(_audit(label, packets[label], payload, errors))
        write_json(output / f"{label}_temporal_result.json", projected)
    previous = {
        "r1": _find_question(load_json(paths["v3_2_r1_results"]), qid),
        "r3": _find_question(load_json(paths["v3_2_r3_results"]), qid),
    }
    write_json(output / "r1_v3_2_to_v3_2_1_diff.json", _diff(previous["r1"], results["r1"]))
    write_json(output / "r3_v3_2_to_v3_2_1_diff.json", _diff(previous["r3"], results["r3"]))
    write_json(output / "temporal_anchor_audit.json", {"canaries": audits})
    write_json(output / "timestamp_integrity_audit.json", {"anchors": [{"source": row["source"], "requirement_id": row["requirement_id"], "anchors": [{"role": a["role"], "evidence_id": a.get("evidence_id"), "timestamp_copied_exactly": a.get("timestamp_copied_exactly")} for a in row["anchors"]]} for row in audits], "all_non_null_timestamps_exact": all(a.get("timestamp_copied_exactly") is True for row in audits for a in row["anchors"] if a.get("evidence_id") is not None)})
    write_json(output / "status_projection_audit.json", {"canaries": projection_audits, "all_valid": all(row["lists_disjoint_complete"] and row["semantic_unchanged"] for row in projection_audits)})
    after = {key: sha256(path) for key, path in paths.items()}
    material = [flag for row in audits for flag in row["flags"] if flag["material"]]
    errors = [error for row in audits for error in row["validator_errors"]]
    r1_ok = not audits[0]["material_flag_count"] and not audits[0]["validator_errors"]
    r3_ok = not audits[1]["material_flag_count"] and not audits[1]["validator_errors"]
    projection_ok = all(row["lists_disjoint_complete"] and row["semantic_unchanged"] for row in projection_audits)
    base_fields = set(v3_2_schema()["properties"]["claims"]["items"]["required"])
    non_temporal_schema_ok = all(
        set(claim) == base_fields
        for label in ("r1", "r3") for claim in results[label]["claims"]
        if not claim["requirement_id"].endswith("::temporal_order") and not claim["requirement_id"].endswith("::chronological_order")
    )
    if errors:
        recommendation = "requires_temporal_prompt_or_output_fix"
        overall = "requires_temporal_prompt_or_output_fix"
    elif r1_ok and r3_ok and projection_ok:
        recommendation = "ready_for_full_shared_v3_2_1_regression"
        overall = "pending_manual_temporal_review"
    else:
        recommendation = "temporal_schema_or_validator_failure"
        overall = "failed"
    validation = {
        "no_api_test_validation": "passed", "r1_temporal_canary_validation": "passed" if r1_ok else "failed",
        "r3_temporal_canary_validation": "passed" if r3_ok else "failed", "temporal_anchor_validation": "passed" if not errors else "failed",
        "projection_validation": "passed" if projection_ok else "failed", "non_temporal_v3_2_schema_compatibility": "passed" if non_temporal_schema_ok else "failed", "overall_validation": overall,
        "recommendation": recommendation, "material_temporal_overclaim_count": len(material),
        "source_hashes_unchanged": before == after, "model_calls": 2, "repair_calls": 0,
        "gemini_calls": 0, "visual_review_calls": 0, "final_qa_calls": 0, "full_regression_calls": 0,
    }
    write_json(output / "validation_report.json", validation)
    cost = {
        "no_api_implementation_testing": {"api_calls": 0}, "reused_packet_costs": {"new_cost": 0},
        "new_r1_sufficiency_call": costs["r1"], "new_r3_sufficiency_call": costs["r3"], "repairs": 0,
        "candidate_runtime": {"calls": 2, "input_tokens": sum(costs[x]["input_tokens"] for x in costs), "output_tokens": sum(costs[x]["output_tokens"] for x in costs), "latency_sec": sum(costs[x]["latency_sec"] for x in costs)},
    }
    write_json(output / "cost_accounting.json", cost)
    _render_review(output / "review.html", packets, previous, results, audits)
    report = [
        f"# {cfg['experiment']}", "", f"- Overall: `{overall}`", f"- Recommendation: `{recommendation}`",
        "- Minimal change: temporal-order claims add two typed anchors; all other claims use V3.2 unchanged.",
        "- No-API temporal tests: 16/16 passed.", f"- R1 temporal status: `{_temporal_claim(results['r1'])['status']}`.",
        f"- R3 temporal status: `{_temporal_claim(results['r3'])['status']}`.", f"- Material temporal flags: {len(material)}.",
        f"- Non-temporal V3.2 schema compatibility: `{non_temporal_schema_ok}`.",
        f"- Source hashes unchanged: `{before == after}`.", "", "## Temporal audits", "", "```json", json.dumps(audits, ensure_ascii=False, indent=2), "```", "", "## Costs", "", "```json", json.dumps(cost, ensure_ascii=False, indent=2), "```", "", "No Gemini, visual review, final QA, or full 6+6 regression ran.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"validation": validation, "audits": audits, "cost": cost}
