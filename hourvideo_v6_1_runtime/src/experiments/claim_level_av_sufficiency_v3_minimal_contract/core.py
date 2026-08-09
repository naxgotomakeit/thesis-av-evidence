from __future__ import annotations

import copy
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT = "egopolice_sufficiency_v3_contract_smoke_v1"
CONTRACT = "claim_level_av_sufficiency_v3_minimal_contract"
STATUS_ENUM = ("supported", "uncertain", "conflicted", "not_found")
STATUS_LISTS = {
    "supported": "supported_claims",
    "uncertain": "uncertain_claims",
    "conflicted": "conflicted_claims",
    "not_found": "missing_claims",
}
CLAIM_TYPES = (
    "object_presence", "actor_action", "actor_object_binding",
    "spoken_command", "reported_event", "physical_event", "state",
    "temporal_order", "causal_relation", "intervention", "outcome",
    "identity", "location", "cross_modal_event",
)
SUPPORT_MODES = (
    "visual_caption", "audio_content", "audio_event", "cross_modal",
    "temporal_inference",
)

V3_SYSTEM_PROMPT = """You are a claim-level audiovisual index sufficiency analyst.
Return only atomic semantic claims supported, left uncertain, conflicted, or not
found in the supplied evidence packet. Associate every claim only with
predeclared requirement IDs. Preserve modality, polarity, uncertainty, and
rejected inferences. Evidence timestamps locate source material; they do not by
themselves establish an event's true start, end, completion, actor binding, or
causal relation. Provide claim_event_time only when the evidence semantically
supports that interval. Do not answer the question and do not emit classification
lists, required-claim lists, criticality, reuse flags, or review-control fields.
Return strict JSON matching the supplied V3 semantic schema."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def semantic_schema() -> dict[str, Any]:
    claim = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim_id": {"type": "string", "minLength": 1},
            "requirement_ids": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "claim_text": {"type": "string", "minLength": 1},
            "claim_type": {"type": "string", "enum": list(CLAIM_TYPES)},
            "status": {"type": "string", "enum": list(STATUS_ENUM)},
            "support_mode": {
                "type": "array", "uniqueItems": True,
                "items": {"type": "string", "enum": list(SUPPORT_MODES)},
            },
            "evidence_ids": {
                "type": "array", "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "confidence": {
                "type": "string", "enum": ["high", "medium", "low"],
            },
            "remaining_uncertainty": {
                "type": "array", "items": {"type": "string"},
            },
            "rejected_inferences": {
                "type": "array", "items": {"type": "string"},
            },
            "claim_event_time": {
                "type": ["array", "null"],
                "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
            },
        },
        "required": [
            "claim_id", "requirement_ids", "claim_text", "claim_type",
            "status", "support_mode", "evidence_ids", "confidence",
            "remaining_uncertainty", "rejected_inferences",
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": CONTRACT,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string", "minLength": 1},
            "claims": {"type": "array", "minItems": 1, "items": claim},
        },
        "required": ["question_id", "claims"],
    }


def requirements_from_slot_templates(
    slot_templates: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for question_id, slots in slot_templates.items():
        rows = []
        for slot in slots:
            slot_id = str(slot["slot_id"])
            rows.append({
                "requirement_id": f"{question_id}::{slot_id}",
                "description": slot_id,
                "answer_critical": bool(
                    slot.get("essential") or slot.get("conditional_essential")
                ),
                "source_slot_id": slot_id,
                "source_essential": bool(slot.get("essential")),
                "source_conditional_essential": bool(
                    slot.get("conditional_essential")
                ),
            })
        result[question_id] = rows
    return result


def build_v3_input(
    packet: dict[str, Any], requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the future LLM input without running inference."""
    return {
        "question_id": packet["question_id"],
        "question": packet["question"],
        "question_type": packet.get("question_type"),
        "requirements": copy.deepcopy(requirements),
        "selected_phase_ids": copy.deepcopy(packet.get("selected_phase_ids") or []),
        "visual_evidence": copy.deepcopy(packet.get("visual_evidence") or []),
        "audio_evidence": copy.deepcopy(packet.get("audio_evidence") or []),
        "typed_av_links": copy.deepcopy(packet.get("typed_av_links") or []),
        "existing_uncertainty": copy.deepcopy(
            packet.get("existing_uncertainty") or []
        ),
        "rejected_bindings": copy.deepcopy(packet.get("rejected_bindings") or []),
    }


def _evidence_index(v3_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for modality, field in (("visual", "visual_evidence"), ("audio", "audio_evidence")):
        for row in v3_input.get(field) or []:
            evidence_id = row.get("evidence_id")
            if evidence_id in result:
                raise ValueError(f"duplicate evidence_id: {evidence_id}")
            timestamp = row.get("timestamp") or []
            if (
                not isinstance(timestamp, list) or len(timestamp) != 2
                or not all(isinstance(x, (int, float)) for x in timestamp)
                or timestamp[0] > timestamp[1]
            ):
                raise ValueError(f"invalid evidence timestamp: {evidence_id}")
            result[evidence_id] = {
                "modality": modality,
                "timestamp": [timestamp[0], timestamp[1]],
            }
    return result


def validate_semantic_payload(
    payload: dict[str, Any], v3_input: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if set(payload) != {"question_id", "claims"}:
        errors.append("semantic payload contains missing or forbidden top-level fields")
    if payload.get("question_id") != v3_input.get("question_id"):
        errors.append("question_id mismatch")
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return errors + ["claims must be a nonempty list"]
    requirement_ids = {
        row["requirement_id"] for row in v3_input.get("requirements") or []
    }
    evidence = _evidence_index(v3_input)
    allowed_fields = set(semantic_schema()["properties"]["claims"]["items"]["properties"])
    required_fields = set(semantic_schema()["properties"]["claims"]["items"]["required"])
    seen = set()
    covered_requirements = set()
    for index, claim in enumerate(claims):
        prefix = str(claim.get("claim_id") or f"claim[{index}]")
        if set(claim) - allowed_fields:
            errors.append(f"{prefix}: forbidden semantic fields")
        if not required_fields <= set(claim):
            errors.append(f"{prefix}: missing required semantic fields")
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip() or claim_id in seen:
            errors.append(f"{prefix}: claim_id must be nonempty and unique")
        seen.add(claim_id)
        linked = claim.get("requirement_ids")
        if (
            not isinstance(linked, list) or not linked
            or len(linked) != len(set(linked))
            or not set(linked) <= requirement_ids
        ):
            errors.append(f"{prefix}: unknown, duplicate, or empty requirement_ids")
        else:
            covered_requirements.update(linked)
        if claim.get("status") not in STATUS_ENUM:
            errors.append(f"{prefix}: invalid status")
        if claim.get("claim_type") not in CLAIM_TYPES:
            errors.append(f"{prefix}: invalid claim_type")
        modes = claim.get("support_mode")
        if not isinstance(modes, list) or not set(modes) <= set(SUPPORT_MODES):
            errors.append(f"{prefix}: invalid support_mode")
        ids = claim.get("evidence_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= set(evidence):
            errors.append(f"{prefix}: invalid evidence_ids")
        event_time = claim.get("claim_event_time")
        if event_time is not None and (
            not isinstance(event_time, list) or len(event_time) != 2
            or not all(isinstance(x, (int, float)) for x in event_time)
            or event_time[0] > event_time[1]
        ):
            errors.append(f"{prefix}: invalid or reversed claim_event_time")
        if claim.get("confidence") not in {"high", "medium", "low"}:
            errors.append(f"{prefix}: invalid confidence")
        for field in ("remaining_uncertainty", "rejected_inferences"):
            if not isinstance(claim.get(field), list):
                errors.append(f"{prefix}: {field} must be a list")
    missing = requirement_ids - covered_requirements
    if missing:
        errors.append("requirements without claims: " + ", ".join(sorted(missing)))
    return errors


def _status_and_answerability(
    claims: list[dict[str, Any]], requirements: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], str, list[str]]:
    lists = {field: [] for field in STATUS_LISTS.values()}
    for claim in claims:
        lists[STATUS_LISTS[claim["status"]]].append(claim["claim_id"])
    critical = {
        row["requirement_id"] for row in requirements if row["answer_critical"]
    }
    per_requirement: dict[str, list[str]] = {rid: [] for rid in critical}
    for claim in claims:
        for rid in set(claim["requirement_ids"]) & critical:
            per_requirement[rid].append(claim["status"])
    if any("conflicted" in states for states in per_requirement.values()):
        answerability = "conflicted"
    else:
        satisfied = sum("supported" in states for states in per_requirement.values())
        answerability = (
            "sufficient" if satisfied == len(per_requirement)
            else "partial" if satisfied
            else "insufficient"
        )
    required_claims = [
        claim["claim_id"] for claim in claims
        if set(claim["requirement_ids"]) & critical
    ]
    return lists, answerability, required_claims


def project_semantic_payload(
    payload: dict[str, Any], v3_input: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = validate_semantic_payload(payload, v3_input)
    if errors:
        raise ValueError("; ".join(errors))
    before = copy.deepcopy(payload["claims"])
    evidence = _evidence_index(v3_input)
    requirements = {
        row["requirement_id"]: row for row in v3_input["requirements"]
    }
    projected_claims = []
    review_candidates = []
    for claim in payload["claims"]:
        ids = claim["evidence_ids"]
        envelope = None
        if ids:
            envelope = [
                min(evidence[eid]["timestamp"][0] for eid in ids),
                max(evidence[eid]["timestamp"][1] for eid in ids),
            ]
        modalities = sorted({evidence[eid]["modality"] for eid in ids})
        critical = any(requirements[rid]["answer_critical"] for rid in claim["requirement_ids"])
        reusable = bool(ids) and claim["status"] != "not_found"
        row = {
            **copy.deepcopy(claim),
            "answer_critical": critical,
            "reusable": reusable,
            "support_modalities": modalities,
            "evidence_time_envelope": envelope,
        }
        projected_claims.append(row)
        visual_ids = [eid for eid in ids if evidence[eid]["modality"] == "visual"]
        if critical and claim["status"] in {"uncertain", "conflicted", "not_found"} and visual_ids:
            visual_range = [
                min(evidence[eid]["timestamp"][0] for eid in visual_ids),
                max(evidence[eid]["timestamp"][1] for eid in visual_ids),
            ]
            review_candidates.append({
                "claim_id": claim["claim_id"],
                "requirement_ids": list(claim["requirement_ids"]),
                "candidate_existing_evidence_ids": visual_ids,
                "target_time_range": visual_range,
                "available_evidence_modality": modalities,
                "candidate_only": True,
            })
    lists, answerability, required_claims = _status_and_answerability(
        payload["claims"], v3_input["requirements"]
    )
    top_uncertainty = []
    for claim in projected_claims:
        if claim["answer_critical"] and claim["status"] != "supported":
            top_uncertainty.extend(claim["remaining_uncertainty"] or [claim["claim_text"]])
    result = {
        "question_id": payload["question_id"],
        "claims": projected_claims,
        **lists,
        "required_claims": required_claims,
        "answerability_from_current_index": answerability,
        "review_candidates": review_candidates,
        "remaining_uncertainty": list(dict.fromkeys(top_uncertainty)),
    }
    after_semantic = [
        {key: claim[key] for key in before[index]}
        for index, claim in enumerate(projected_claims)
    ]
    audit = {
        "semantic_claims_sha256_before": canonical_sha256(before),
        "semantic_claims_sha256_after": canonical_sha256(after_semantic),
        "semantic_claims_deep_equal": before == after_semantic,
        "claim_order_unchanged": [x["claim_id"] for x in before] == [x["claim_id"] for x in after_semantic],
        "per_claim": [
            {
                "claim_id": old["claim_id"],
                "before_sha256": canonical_sha256(old),
                "after_sha256": canonical_sha256(new),
                "unchanged": old == new,
            }
            for old, new in zip(before, after_semantic)
        ],
    }
    if not audit["semantic_claims_deep_equal"]:
        raise AssertionError("projection modified the semantic payload")
    return result, audit


def build_legacy_adapter_view(
    canonical: dict[str, Any],
) -> dict[str, Any]:
    candidates = {row["claim_id"]: row for row in canonical["review_candidates"]}
    claims = []
    requests = []
    for claim in canonical["claims"]:
        candidate = candidates.get(claim["claim_id"])
        # The old time_range is localization-only in this adapter. Prefer the
        # asserted event time; otherwise use the explicitly labelled evidence envelope.
        time_range = claim.get("claim_event_time") or claim.get("evidence_time_envelope") or []
        claims.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "claim_type": claim["claim_type"],
            "status": claim["status"],
            "support_mode": list(claim["support_mode"]),
            "evidence_ids": list(claim["evidence_ids"]),
            "time_range": list(time_range),
            "reasoning_summary": "",
            "confidence": claim["confidence"],
            "answer_critical": claim["answer_critical"],
            "reusable": claim["reusable"],
            "needs_raw_visual_review": candidate is not None,
            "review_reason": (
                "Answer-critical unresolved claim has local visual evidence."
                if candidate else None
            ),
            "review_target": claim["claim_text"] if candidate else None,
            "remaining_uncertainty": list(claim["remaining_uncertainty"]),
            "rejected_inferences": list(claim["rejected_inferences"]),
        })
        if candidate:
            requests.append({
                "claim_id": claim["claim_id"],
                "answer_critical": True,
                "reason": "Answer-critical unresolved claim has local visual evidence.",
                "target_time_range": list(candidate["target_time_range"]),
                "target_visual_question": claim["claim_text"],
                "candidate_existing_evidence_ids": list(
                    candidate["candidate_existing_evidence_ids"]
                ),
                "expected_resolution": ["confirm", "reject", "remain_uncertain"],
            })
    return {
        "question_id": canonical["question_id"],
        "claims": claims,
        "required_claims": list(canonical["required_claims"]),
        "supported_claims": list(canonical["supported_claims"]),
        "uncertain_claims": list(canonical["uncertain_claims"]),
        "conflicted_claims": list(canonical["conflicted_claims"]),
        "missing_claims": list(canonical["missing_claims"]),
        "answerability_from_current_index": canonical["answerability_from_current_index"],
        "visual_review_requests": requests,
        "remaining_uncertainty": list(canonical["remaining_uncertainty"]),
    }


def compatibility_fixture(v3_input: dict[str, Any]) -> dict[str, Any]:
    """Non-semantic no-API fixture: it deliberately asserts no video fact."""
    claims = []
    for index, requirement in enumerate(v3_input["requirements"], 1):
        rid = requirement["requirement_id"]
        claims.append({
            "claim_id": f"contract_fixture_{index:03d}",
            "requirement_ids": [rid],
            "claim_text": f"No inference was run for requirement {rid}.",
            "claim_type": "state",
            "status": "not_found",
            "support_mode": [],
            "evidence_ids": [],
            "confidence": "low",
            "remaining_uncertainty": ["No model inference was run."],
            "rejected_inferences": [],
            "claim_event_time": None,
        })
    return {"question_id": v3_input["question_id"], "claims": claims}


def compatibility_report(
    *, packet_path: Path, requirements_by_question: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    from src.experiments.claim_handoff_routing_v2_2.core import route_claim
    from src.experiments.claim_reliability_gate_v2_1.core import (
        extract_risk_features, gate_claim,
    )
    from src.experiments.claim_level_av_sufficiency_v2.core import validate_result
    from src.experiments.claim_guided_gemini_review_loop_v2.core import (
        build_final_batch_request, build_final_claim_packets,
        build_stage_a_payload,
    )

    packets = load_json(packet_path)["questions"]
    rows = []
    for packet in packets:
        qid = packet["question_id"]
        v3_input = build_v3_input(packet, requirements_by_question[qid])
        fixture = compatibility_fixture(v3_input)
        canonical, preservation = project_semantic_payload(fixture, v3_input)
        legacy = build_legacy_adapter_view(canonical)
        legacy_errors = validate_result(legacy, packet)
        visual = {x["evidence_id"]: x for x in packet["visual_evidence"]}
        audio = {x["evidence_id"]: x for x in packet["audio_evidence"]}
        consumer_errors = []
        routed = []
        for claim in legacy["claims"]:
            try:
                feature = extract_risk_features(
                    claim, question_id=qid, visual=visual, audio=audio,
                    overclaim_ids=set(),
                )
                gated = gate_claim(claim, feature)
                routed.append(route_claim(gated, feature=feature, cache_claims=[]))
            except Exception as exc:  # recorded as a compatibility failure
                consumer_errors.append(f"{claim['claim_id']}: {exc}")
        review_consumer_errors = []
        final_consumer_errors = []
        try:
            handoff_packet = {
                "question_id": qid,
                "question": packet["question"],
                "supported_content": [x for x in routed if x["handoff_status"] == "supported_content"],
                "qualified_content": [x for x in routed if x["handoff_status"] == "qualified_content"],
                "unresolved_content": [x for x in routed if x["handoff_status"] == "unresolved_content"],
                "conflicted_content": [x for x in routed if x["handoff_status"] == "conflicted_content"],
            }
            stage_a = build_stage_a_payload(handoff_packet, {}, {})
        except Exception as exc:
            review_consumer_errors.append(str(exc))
            stage_a = None
        if stage_a is not None:
            try:
                final_packets = build_final_claim_packets({qid: stage_a}, {})
                _, saved_final = build_final_batch_request(final_packets)
                if saved_final.get("raw_image_inputs") != 0:
                    final_consumer_errors.append("final batch unexpectedly contains images")
            except Exception as exc:
                final_consumer_errors.append(str(exc))
        rows.append({
            "question_id": qid,
            "v3_input_valid": bool(v3_input["requirements"]),
            "semantic_fixture_kind": "no-inference not_found contract fixture",
            "semantic_fixture_is_video_result": False,
            "semantic_validation_errors": validate_semantic_payload(fixture, v3_input),
            "legacy_validator_errors": legacy_errors,
            "reliability_gate_load_errors": consumer_errors,
            "handoff_routed_claim_count": len(routed),
            "gemini_review_payload_load_errors": review_consumer_errors,
            "final_answer_payload_load_errors": final_consumer_errors,
            "claim_semantics_unchanged": preservation["semantic_claims_deep_equal"],
        })
    valid = all(
        row["v3_input_valid"]
        and not row["semantic_validation_errors"]
        and not row["legacy_validator_errors"]
        and not row["reliability_gate_load_errors"]
        and not row["gemini_review_payload_load_errors"]
        and not row["final_answer_payload_load_errors"]
        and row["claim_semantics_unchanged"]
        for row in rows
    )
    return {
        "packet_path": str(packet_path),
        "packet_sha256": file_sha256(packet_path),
        "question_count": len(rows),
        "valid": valid,
        "questions": rows,
        "api_calls": 0,
        "upstream_reruns": 0,
    }


def frozen_contract_audit(repo: Path) -> dict[str, Any]:
    from src.experiments.claim_level_av_sufficiency_v2 import core as old

    sources = {
        "v2_sufficiency": Path(old.__file__),
        "reliability_gate": repo / "src/experiments/claim_reliability_gate_v2_1/core.py",
        "handoff": repo / "src/experiments/claim_handoff_routing_v2_2/core.py",
        "gemini_review_loop": repo / "src/experiments/claim_guided_gemini_review_loop_v2/core.py",
    }
    return {
        "contract": CONTRACT,
        "audited_sources": [
            {"role": role, "path": str(path.relative_to(repo)), "sha256": file_sha256(path)}
            for role, path in sources.items()
        ],
        "old_status_enum": sorted(old.ALLOWED_STATUS),
        "old_semantic_fields": [
            "claim_id", "claim_text", "claim_type", "status", "support_mode",
            "evidence_ids", "confidence", "remaining_uncertainty",
            "rejected_inferences", "time_range (mixed event/localization semantics)",
        ],
        "old_redundant_or_derived_fields": [
            "supported_claims", "uncertain_claims", "conflicted_claims",
            "missing_claims", "required_claims", "answer_critical", "reusable",
            "needs_raw_visual_review", "review_reason", "review_target",
            "visual_review_requests", "answerability_from_current_index",
        ],
        "downstream_dependencies": {
            "claim_reliability_gate_v2_1": [
                "claims with claim_type/status/support_mode/evidence_ids/time_range/answer_critical/reusable/review fields/uncertainty",
                "required_claims_by_question.json", "sufficiency_decisions.json",
                "visual_review_requests.json", "request_payloads.json",
            ],
            "claim_handoff_routing_v2_2": [
                "gated reliability claims including answer_critical, time_range, evidence_ids, support modalities and resolution metadata"
            ],
            "claim_guided_gemini_review_loop_v2": [
                "required claims", "sufficiency decisions", "request payloads",
                "routed handoff claims; old review suggestions are comparison-only",
            ],
        },
        "legacy_fields_without_new_semantic_judgment": [
            "four status lists from claim.status",
            "required_claims from requirement_ids and predeclared criticality",
            "answer_critical from the requirement specification",
            "reusable from evidence presence and status",
            "time_range localization view from claim_event_time or evidence_time_envelope",
            "review-control fields from the canonical review-candidate view",
            "answerability from critical-requirement status coverage",
            "top-level remaining_uncertainty from answer-critical non-supported claims",
        ],
        "unresolved_semantic_dependencies": [
            "requirement specifications must be declared before inference",
            "claim_type and support_mode remain LLM semantic judgments",
            "claim_event_time is optional and cannot be derived from the evidence envelope",
            "Gemini Stage A remains the authority for an actual review decision",
        ],
        "source_hashes": {
            "old_schema": hashlib.sha256(
                canonical_json(old.claim_schema()).encode("utf-8")
            ).hexdigest(),
            "old_validator": hashlib.sha256(
                inspect.getsource(old.validate_result).encode("utf-8")
            ).hexdigest(),
            "old_normalizer": hashlib.sha256(
                inspect.getsource(old.normalize_result_schema).encode("utf-8")
            ).hexdigest(),
            "old_prompt": hashlib.sha256(old.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
    }


def run_smoke(repo: Path, config_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    config = load_json(config_path)
    output = repo / config["output_root"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True)
    slot_path = repo / config["requirement_source"]
    requirements = requirements_from_slot_templates(load_json(slot_path))
    input_paths = {name: repo / path for name, path in config["packet_paths"].items()}
    frozen_paths = [slot_path, *input_paths.values()]
    hashes_before = {str(path.relative_to(repo)): file_sha256(path) for path in frozen_paths}

    audit = frozen_contract_audit(repo)
    audit["requirement_source"] = {
        "path": str(slot_path.relative_to(repo)), "sha256": file_sha256(slot_path),
        "stable_id_rule": "question_id + '::' + slot_id",
        "criticality_rule": "essential OR conditional_essential",
    }
    dump_json(output / "frozen_contract_audit.json", audit)
    dump_json(output / "v3_semantic_schema.json", semantic_schema())
    (output / "v3_system_prompt.txt").write_text(
        V3_SYSTEM_PROMPT + "\n", encoding="utf-8"
    )
    projection_spec = {
        "status_lists": STATUS_LISTS,
        "required_claims": "claim IDs linked to predeclared answer-critical requirements, preserving claim order",
        "answer_critical": "any linked predeclared requirement is answer-critical",
        "reusable": "has evidence and status is not not_found",
        "evidence_time_envelope": "minimum evidence start and maximum evidence end across referenced evidence IDs",
        "claim_event_time": "optional LLM assertion; never derived or reordered",
        "answerability": "deterministic coverage of answer-critical requirements",
        "review_candidate": "answer-critical AND non-supported status AND referenced visual evidence with valid timestamps",
        "actual_review_authority": "Gemini Stage A",
        "semantic_mutations_allowed": [],
    }
    dump_json(output / "deterministic_projection_spec.json", projection_spec)
    adapter = {
        "claims.time_range": "claim_event_time if present, else evidence_time_envelope as localization-only fallback",
        "claims.reasoning_summary": "empty compatibility string; no semantic content invented",
        "claims.answer_critical": "projected answer_critical",
        "claims.reusable": "projected reusable",
        "claims.needs_raw_visual_review": "membership in canonical review-candidate view",
        "claims.review_reason": "fixed generic control explanation or null",
        "claims.review_target": "unchanged claim_text or null",
        "status lists": "deterministic status projection",
        "required_claims": "deterministic requirement linkage",
        "answerability_from_current_index": "deterministic critical-requirement coverage",
        "visual_review_requests": "canonical review candidates in legacy shape",
        "remaining_uncertainty": "deduplicated answer-critical non-supported uncertainty",
    }
    dump_json(output / "legacy_adapter_mapping.json", adapter)

    reports = {}
    filenames = {
        "uniform": "uniform_compatibility_report.json",
        "selective": "selective_compatibility_report.json",
        "dense_r3": "dense_r3_compatibility_report.json",
    }
    for name, path in input_paths.items():
        report = compatibility_report(
            packet_path=path, requirements_by_question=requirements,
        )
        reports[name] = report
        dump_json(output / filenames[name], report)

    hashes_after = {str(path.relative_to(repo)): file_sha256(path) for path in frozen_paths}
    validation = {
        "validation_status": (
            "passed_claim_level_av_sufficiency_v3_contract_smoke"
            if all(row["valid"] for row in reports.values()) and hashes_before == hashes_after
            else "failed_validation"
        ),
        "compatibility": {name: row["valid"] for name, row in reports.items()},
        "api_calls": 0,
        "haiku_calls": 0,
        "gemini_calls": 0,
        "organizer_planner_retrieval_audio_selector_reruns": 0,
        "legacy_consumers_loaded": [
            "claim_reliability_gate_v2_1",
            "claim_handoff_routing_v2_2",
            "claim_guided_gemini_review_loop_v2 payload builder",
            "claim_guided_gemini_review_loop_v2 final batch payload builder",
        ],
        "old_files_unchanged": hashes_before == hashes_after,
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "semantic_fixture_is_not_a_video_result": True,
    }
    dump_json(output / "validation_report.json", validation)
    report_lines = [
        "# Claim-level AV Sufficiency V3 minimal-contract smoke",
        "", f"- Status: `{validation['validation_status']}`",
        "- API/model calls: 0", "- Upstream reruns: 0",
        "- Compatibility fixtures deliberately assert no video facts.",
        "- The smoke validates contracts and consumer loading, not Sufficiency quality.",
        "- Legacy loading covered validation, reliability, handoff, Stage A, and final-batch payload construction.",
        "",
    ]
    for name, report in reports.items():
        report_lines.append(
            f"- {name}: valid={report['valid']}, questions={report['question_count']}, packet_sha256={report['packet_sha256']}"
        )
    (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return validation
