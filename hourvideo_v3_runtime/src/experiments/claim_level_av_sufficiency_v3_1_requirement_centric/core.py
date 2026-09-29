from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.claim_level_av_sufficiency_v2.core import validate_result
from src.experiments.claim_level_av_sufficiency_v3_minimal_contract.core import (
    CLAIM_TYPES, STATUS_ENUM, STATUS_LISTS, SUPPORT_MODES, build_v3_input,
    canonical_sha256, dump_json, file_sha256, load_json,
    requirements_from_slot_templates,
)


CONTRACT = "claim_level_av_sufficiency_v3_1_requirement_centric"
EXPERIMENT = "egopolice_uniform_weapon_v3_1_canary_v1"

SYSTEM_PROMPT = """You are a requirement-centric audiovisual index sufficiency analyst.
For every predeclared requirement in the input, return exactly one primary
assessment claim using that exact requirement_id. Return an assessment even
when evidence is insufficient, using uncertain, conflicted, or not_found as
appropriate. Do not omit, rename, infer, or remap requirement IDs. Auxiliary
claims may capture additional atomic observations but do not satisfy requirement
coverage. Preserve modality, evidence provenance, uncertainty, polarity, and
rejected inferences. Evidence timestamps locate evidence and do not by themselves
prove an event's true start, end, completion, actor binding, or causality. Do not
answer the question. Return strict JSON matching the supplied schema."""


def _claim_properties(*, include_requirement: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    if include_requirement:
        properties["requirement_id"] = {"type": "string", "minLength": 1}
    properties.update({
        "claim_id": {"type": "string", "minLength": 1},
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
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "remaining_uncertainty": {"type": "array", "items": {"type": "string"}},
        "rejected_inferences": {"type": "array", "items": {"type": "string"}},
        "claim_event_time": {
            "type": ["array", "null"], "items": {"type": "number"},
            "minItems": 2, "maxItems": 2,
        },
    })
    return properties


def semantic_schema() -> dict[str, Any]:
    assessment_properties = _claim_properties(include_requirement=True)
    auxiliary_properties = _claim_properties(include_requirement=False)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": CONTRACT,
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string", "minLength": 1},
            "requirement_assessments": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": assessment_properties,
                    "required": list(assessment_properties),
                },
            },
            "auxiliary_claims": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": auxiliary_properties,
                    "required": list(auxiliary_properties),
                },
            },
        },
        "required": ["question_id", "requirement_assessments", "auxiliary_claims"],
    }


def _evidence_index(v3_input: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for modality, field in (("visual", "visual_evidence"), ("audio", "audio_evidence")):
        for row in v3_input.get(field) or []:
            evidence_id = row.get("evidence_id")
            timestamp = row.get("timestamp") or []
            if evidence_id in result:
                raise ValueError(f"duplicate evidence_id: {evidence_id}")
            if (
                not isinstance(timestamp, list) or len(timestamp) != 2
                or not all(isinstance(value, (int, float)) for value in timestamp)
                or timestamp[0] > timestamp[1]
            ):
                raise ValueError(f"invalid evidence timestamp: {evidence_id}")
            result[evidence_id] = {
                "modality": modality, "timestamp": list(timestamp),
            }
    return result


def _validate_claim(
    claim: Any, *, allowed_fields: set[str], required_fields: set[str],
    evidence_ids: set[str], prefix: str,
) -> list[str]:
    errors = []
    if not isinstance(claim, dict):
        return [f"{prefix}: must be an object"]
    if set(claim) != required_fields:
        errors.append(f"{prefix}: fields must exactly match the semantic schema")
    if set(claim) - allowed_fields:
        errors.append(f"{prefix}: forbidden fields")
    for field in ("claim_id", "claim_text"):
        if not isinstance(claim.get(field), str) or not claim.get(field):
            errors.append(f"{prefix}.{field}: must be a nonempty string")
    if claim.get("claim_type") not in CLAIM_TYPES:
        errors.append(f"{prefix}: invalid claim_type")
    if claim.get("status") not in STATUS_ENUM:
        errors.append(f"{prefix}: invalid status")
    modes = claim.get("support_mode")
    if (
        not isinstance(modes, list) or len(modes) != len(set(modes))
        or not set(modes) <= set(SUPPORT_MODES)
    ):
        errors.append(f"{prefix}: invalid support_mode array")
    ids = claim.get("evidence_ids")
    if (
        not isinstance(ids, list) or len(ids) != len(set(ids))
        or not set(ids) <= evidence_ids
    ):
        errors.append(f"{prefix}: invalid evidence_ids")
    if claim.get("confidence") not in {"high", "medium", "low"}:
        errors.append(f"{prefix}: invalid confidence")
    for field in ("remaining_uncertainty", "rejected_inferences"):
        value = claim.get(field)
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            errors.append(f"{prefix}: {field} must be an array of strings")
    event_time = claim.get("claim_event_time")
    if event_time is not None and (
        not isinstance(event_time, list) or len(event_time) != 2
        or not all(isinstance(value, (int, float)) for value in event_time)
        or event_time[0] > event_time[1]
    ):
        errors.append(f"{prefix}: invalid or reversed claim_event_time")
    return errors


def validate_payload(
    payload: dict[str, Any], v3_input: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    if not isinstance(payload, dict) or set(payload) != {
        "question_id", "requirement_assessments", "auxiliary_claims"
    }:
        return ["top-level fields must exactly match V3.1 schema"], {}
    if payload.get("question_id") != v3_input.get("question_id"):
        errors.append("question_id mismatch")
    assessments = payload.get("requirement_assessments")
    auxiliary = payload.get("auxiliary_claims")
    if not isinstance(assessments, list) or not assessments:
        errors.append("requirement_assessments must be a nonempty array")
        assessments = []
    if not isinstance(auxiliary, list):
        errors.append("auxiliary_claims must be an array")
        auxiliary = []
    declared = [row["requirement_id"] for row in v3_input.get("requirements") or []]
    generated = [row.get("requirement_id") for row in assessments if isinstance(row, dict)]
    duplicates = sorted({rid for rid in generated if generated.count(rid) > 1})
    missing = sorted(set(declared) - set(generated))
    unknown = sorted(set(generated) - set(declared))
    if duplicates:
        errors.append("duplicate requirement assessments: " + ", ".join(duplicates))
    if missing:
        errors.append("missing requirement assessments: " + ", ".join(missing))
    if unknown:
        errors.append("unknown requirement assessments: " + ", ".join(unknown))
    if len(generated) != len(declared) or set(generated) != set(declared):
        errors.append("generated requirement IDs do not exactly equal declared IDs")
    evidence = _evidence_index(v3_input)
    schema = semantic_schema()
    assessment_schema = schema["properties"]["requirement_assessments"]["items"]
    auxiliary_schema = schema["properties"]["auxiliary_claims"]["items"]
    for index, claim in enumerate(assessments):
        errors.extend(_validate_claim(
            claim, allowed_fields=set(assessment_schema["properties"]),
            required_fields=set(assessment_schema["required"]),
            evidence_ids=set(evidence), prefix=f"requirement_assessments[{index}]",
        ))
    for index, claim in enumerate(auxiliary):
        errors.extend(_validate_claim(
            claim, allowed_fields=set(auxiliary_schema["properties"]),
            required_fields=set(auxiliary_schema["required"]),
            evidence_ids=set(evidence), prefix=f"auxiliary_claims[{index}]",
        ))
        if isinstance(claim, dict) and "requirement_id" in claim:
            errors.append(f"auxiliary_claims[{index}] cannot satisfy a requirement")
    claim_ids = [
        row.get("claim_id") for row in [*assessments, *auxiliary]
        if isinstance(row, dict)
    ]
    if any(not claim_id for claim_id in claim_ids) or len(claim_ids) != len(set(claim_ids)):
        errors.append("claim IDs must be nonempty and unique across all claims")
    coverage = {
        "declared_requirement_ids": declared,
        "generated_requirement_ids": generated,
        "missing_requirement_ids": missing,
        "unknown_requirement_ids": unknown,
        "duplicate_requirement_ids": duplicates,
        "exact_set_equality": not missing and not unknown and not duplicates
        and len(generated) == len(declared),
        "coverage_ratio": (
            len(set(generated) & set(declared)) / len(declared) if declared else 1.0
        ),
        "auxiliary_claim_count": len(auxiliary),
        "auxiliary_claims_count_toward_coverage": False,
    }
    return errors, coverage


def project_payload(
    payload: dict[str, Any], v3_input: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors, coverage = validate_payload(payload, v3_input)
    if errors:
        raise ValueError("; ".join(errors))
    before_assessments = copy.deepcopy(payload["requirement_assessments"])
    before_auxiliary = copy.deepcopy(payload["auxiliary_claims"])
    evidence = _evidence_index(v3_input)
    requirements = {row["requirement_id"]: row for row in v3_input["requirements"]}

    def enrich(claim: dict[str, Any], *, requirement_id: str | None) -> dict[str, Any]:
        ids = claim["evidence_ids"]
        envelope = None if not ids else [
            min(evidence[eid]["timestamp"][0] for eid in ids),
            max(evidence[eid]["timestamp"][1] for eid in ids),
        ]
        critical = bool(
            requirement_id is not None
            and requirements[requirement_id]["answer_critical"]
        )
        return {
            **copy.deepcopy(claim),
            "answer_critical": critical,
            "reusable": bool(ids) and claim["status"] != "not_found",
            "support_modalities": sorted({evidence[eid]["modality"] for eid in ids}),
            "evidence_time_envelope": envelope,
            "assessment_origin": "required" if requirement_id else "auxiliary",
        }

    assessments = [enrich(row, requirement_id=row["requirement_id"]) for row in before_assessments]
    auxiliary = [enrich(row, requirement_id=None) for row in before_auxiliary]
    all_claims = [*assessments, *auxiliary]
    lists = {field: [] for field in STATUS_LISTS.values()}
    for row in all_claims:
        lists[STATUS_LISTS[row["status"]]].append(row["claim_id"])
    review_candidates = []
    for row in assessments:
        visual_ids = [
            eid for eid in row["evidence_ids"] if evidence[eid]["modality"] == "visual"
        ]
        if row["answer_critical"] and row["status"] != "supported" and visual_ids:
            review_candidates.append({
                "claim_id": row["claim_id"],
                "requirement_id": row["requirement_id"],
                "candidate_existing_evidence_ids": visual_ids,
                "target_time_range": [
                    min(evidence[eid]["timestamp"][0] for eid in visual_ids),
                    max(evidence[eid]["timestamp"][1] for eid in visual_ids),
                ],
                "candidate_only": True,
            })
    critical_statuses = [
        row["status"] for row in assessments
        if requirements[row["requirement_id"]]["answer_critical"]
    ]
    if "conflicted" in critical_statuses:
        answerability = "conflicted"
    elif all(status == "supported" for status in critical_statuses):
        answerability = "sufficient"
    elif "supported" in critical_statuses:
        answerability = "partial"
    else:
        answerability = "insufficient"
    uncertainty = []
    for row in assessments:
        if row["answer_critical"] and row["status"] != "supported":
            uncertainty.extend(row["remaining_uncertainty"] or [row["claim_text"]])
    projected = {
        "question_id": payload["question_id"],
        "requirement_assessments": assessments,
        "auxiliary_claims": auxiliary,
        **lists,
        "required_claims": [row["claim_id"] for row in assessments],
        "answerability_from_current_index": answerability,
        "review_candidates": review_candidates,
        "remaining_uncertainty": list(dict.fromkeys(uncertainty)),
    }
    after_assessments = [
        {key: row[key] for key in original}
        for original, row in zip(before_assessments, assessments)
    ]
    after_auxiliary = [
        {key: row[key] for key in original}
        for original, row in zip(before_auxiliary, auxiliary)
    ]
    audit = {
        "coverage": coverage,
        "assessment_semantics_sha256_before": canonical_sha256(before_assessments),
        "assessment_semantics_sha256_after": canonical_sha256(after_assessments),
        "assessment_semantics_unchanged": before_assessments == after_assessments,
        "auxiliary_semantics_sha256_before": canonical_sha256(before_auxiliary),
        "auxiliary_semantics_sha256_after": canonical_sha256(after_auxiliary),
        "auxiliary_semantics_unchanged": before_auxiliary == after_auxiliary,
        "claim_order_unchanged": [x["claim_id"] for x in [*before_assessments, *before_auxiliary]]
        == [x["claim_id"] for x in [*after_assessments, *after_auxiliary]],
        "per_claim": [{
            "claim_id": old["claim_id"],
            "before_sha256": canonical_sha256(old),
            "after_sha256": canonical_sha256(new),
            "unchanged": old == new,
        } for old, new in zip(
            [*before_assessments, *before_auxiliary],
            [*after_assessments, *after_auxiliary],
        )],
    }
    if not audit["assessment_semantics_unchanged"] or not audit["auxiliary_semantics_unchanged"]:
        raise AssertionError("projection changed semantic objects")
    return projected, audit


def build_legacy_view(projected: dict[str, Any]) -> dict[str, Any]:
    candidates = {row["claim_id"]: row for row in projected["review_candidates"]}
    claims = []
    requests = []
    for source in [*projected["requirement_assessments"], *projected["auxiliary_claims"]]:
        candidate = candidates.get(source["claim_id"])
        event_or_localization = source.get("claim_event_time") or source.get("evidence_time_envelope") or []
        claims.append({
            "claim_id": source["claim_id"], "claim_text": source["claim_text"],
            "claim_type": source["claim_type"], "status": source["status"],
            "support_mode": list(source["support_mode"]),
            "evidence_ids": list(source["evidence_ids"]),
            "time_range": list(event_or_localization), "reasoning_summary": "",
            "confidence": source["confidence"],
            "answer_critical": source["answer_critical"],
            "reusable": source["reusable"],
            "needs_raw_visual_review": candidate is not None,
            "review_reason": "Answer-critical unresolved claim has local visual evidence." if candidate else None,
            "review_target": source["claim_text"] if candidate else None,
            "remaining_uncertainty": list(source["remaining_uncertainty"]),
            "rejected_inferences": list(source["rejected_inferences"]),
        })
        if candidate:
            requests.append({
                "claim_id": source["claim_id"], "answer_critical": True,
                "reason": "Answer-critical unresolved claim has local visual evidence.",
                "target_time_range": list(candidate["target_time_range"]),
                "target_visual_question": source["claim_text"],
                "candidate_existing_evidence_ids": list(candidate["candidate_existing_evidence_ids"]),
                "expected_resolution": ["confirm", "reject", "remain_uncertain"],
            })
    return {
        "question_id": projected["question_id"], "claims": claims,
        "required_claims": list(projected["required_claims"]),
        "supported_claims": list(projected["supported_claims"]),
        "uncertain_claims": list(projected["uncertain_claims"]),
        "conflicted_claims": list(projected["conflicted_claims"]),
        "missing_claims": list(projected["missing_claims"]),
        "answerability_from_current_index": projected["answerability_from_current_index"],
        "visual_review_requests": requests,
        "remaining_uncertainty": list(projected["remaining_uncertainty"]),
    }


def _api_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _api_schema(item) for key, item in value.items() if key not in {
            "$schema", "uniqueItems", "minItems", "maxItems", "minLength",
        }}
    if isinstance(value, list):
        return [_api_schema(item) for item in value]
    return value


def _call_once(
    *, api_key: str, model_config: dict[str, Any], model_input: dict[str, Any],
    output: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=float(model_config["timeout_sec"]))
    started = time.perf_counter()
    try:
        response = client.messages.create(
            model=model_config["model"], max_tokens=int(model_config["max_tokens"]),
            temperature=float(model_config["temperature"]), system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [{
                "type": "text", "text": json.dumps(model_input, ensure_ascii=False, allow_nan=False),
            }]}],
            output_config={"format": {"type": "json_schema", "schema": _api_schema(semantic_schema())}},
        )
        latency = time.perf_counter() - started
        raw = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        envelope = {
            "raw_text": raw, "model": model_config["model"],
            "usage": {"input_tokens": int(response.usage.input_tokens), "output_tokens": int(response.usage.output_tokens)},
            "latency_sec": latency, "finish_reason": response.stop_reason,
            "response_id": str(response.id), "retry_count": 0,
        }
        dump_json(output / "raw_model_response.json", envelope)
        if response.stop_reason == "max_tokens":
            return None, {**envelope, "error": "response truncated"}
        try:
            return json.loads(raw), envelope
        except Exception as exc:
            return None, {**envelope, "error": f"JSON parse failed: {exc}"}
    except Exception as exc:
        envelope = {
            "raw_text": None, "model": model_config["model"], "usage": {},
            "latency_sec": time.perf_counter() - started, "finish_reason": None,
            "retry_count": 0, "api_error": f"{type(exc).__name__}: {exc}",
        }
        dump_json(output / "raw_model_response.json", envelope)
        return None, envelope


def run_canary(repo: Path, config_path: Path, *, allow_api_calls: bool) -> dict[str, Any]:
    repo = repo.resolve()
    config = load_json(config_path)
    output = repo / config["output_root"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True)
    paths = {
        "packet": repo / config["pre_sufficiency_packet"],
        "requirements": repo / config["requirement_source"],
        "model_config": repo / config["model_config"],
        "v3_core": repo / config["v3_core"],
        "v3_schema": repo / config["v3_schema"],
        "v3_prompt": repo / config["v3_prompt"],
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("one or more frozen inputs are missing")
    if file_sha256(paths["packet"]) != config["pre_sufficiency_packet_sha256"]:
        raise RuntimeError("packet hash mismatch")
    frozen_before = {role: file_sha256(path) for role, path in paths.items()}
    packets = load_json(paths["packet"])["questions"]
    matches = [row for row in packets if row["question_id"] == config["question_id"]]
    if len(matches) != 1:
        raise RuntimeError("frozen Weapon packet is missing or ambiguous")
    packet = matches[0]
    requirements_by_q = requirements_from_slot_templates(load_json(paths["requirements"]))
    requirements = requirements_by_q[config["question_id"]]
    model_input = build_v3_input(packet, requirements)
    requirement_spec = {
        "question_id": packet["question_id"], "requirements": requirements,
        "source_path": str(paths["requirements"].relative_to(repo)),
        "source_sha256": file_sha256(paths["requirements"]),
        "stable_id_rule": "question_id::slot_id",
        "requirements_changed_for_canary": False,
    }
    dump_json(output / "requirement_specification.json", requirement_spec)
    dump_json(output / "semantic_schema.json", semantic_schema())
    (output / "semantic_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    dump_json(output / "frozen_input_manifest.json", {
        "files": [{
            "role": role, "path": str(path.relative_to(repo)),
            "sha256": frozen_before[role], "size_bytes": path.stat().st_size,
            "modified_time_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        } for role, path in paths.items()]
    })

    # Prove that an actual model-shaped not_found assessment passes the old
    # adapter without creating a new claim.
    not_found_fixture = {
        "question_id": packet["question_id"],
        "requirement_assessments": [{
            "requirement_id": row["requirement_id"],
            "claim_id": f"fixture_{index:02d}",
            "claim_text": "The available evidence does not establish this requirement.",
            "claim_type": "state", "status": "not_found", "support_mode": [],
            "evidence_ids": [], "confidence": "low",
            "remaining_uncertainty": ["The requirement remains unresolved."],
            "rejected_inferences": [], "claim_event_time": None,
        } for index, row in enumerate(requirements, 1)],
        "auxiliary_claims": [],
    }
    fixture_errors, _ = validate_payload(not_found_fixture, model_input)
    fixture_projected, fixture_audit = project_payload(not_found_fixture, model_input)
    fixture_legacy = build_legacy_view(fixture_projected)
    fixture_legacy_errors = validate_result(fixture_legacy, packet)
    not_found_audit = {
        "valid": not fixture_errors and not fixture_legacy_errors,
        "semantic_validation_errors": fixture_errors,
        "legacy_validation_errors": fixture_legacy_errors,
        "input_assessment_claim_count": len(not_found_fixture["requirement_assessments"]),
        "legacy_claim_count": len(fixture_legacy["claims"]),
        "sentinel_or_new_claim_synthesized": len(fixture_legacy["claims"]) != len(not_found_fixture["requirement_assessments"]),
        "semantic_objects_unchanged": fixture_audit["assessment_semantics_unchanged"],
    }
    dump_json(output / "not_found_legacy_preflight.json", not_found_audit)
    if not not_found_audit["valid"] or not_found_audit["sentinel_or_new_claim_synthesized"]:
        result = {"validation_status": "failed_not_found_legacy_preflight", **not_found_audit}
        dump_json(output / "validation_report.json", result)
        return result
    if not allow_api_calls:
        result = {
            "validation_status": "passed_no_api_preflight",
            "planned_haiku_calls": 1, "actual_model_calls": 0,
            "not_found_legacy_compatible": True,
        }
        dump_json(output / "validation_report.json", result)
        return result
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY unavailable")
    model_config = load_json(paths["model_config"])
    payload, envelope = _call_once(
        api_key=os.environ["ANTHROPIC_API_KEY"], model_config=model_config,
        model_input=model_input, output=output,
    )
    cost = {
        "haiku_calls": 1, "input_tokens": envelope.get("usage", {}).get("input_tokens", 0),
        "output_tokens": envelope.get("usage", {}).get("output_tokens", 0),
        "latency_sec": envelope.get("latency_sec", 0), "retries": 0,
        "gemini_calls": 0, "image_inputs": 0,
    }
    dump_json(output / "cost_accounting.json", cost)
    if payload is None:
        dump_json(output / "semantic_payload.json", {"status": "unavailable"})
        result = {"validation_status": "failed_model_or_parse", "error": envelope.get("error") or envelope.get("api_error")}
        dump_json(output / "validation_report.json", result)
        return result
    dump_json(output / "semantic_payload.json", payload)
    errors, coverage = validate_payload(payload, model_input)
    dump_json(output / "requirement_coverage_audit.json", coverage)
    dump_json(output / "semantic_validation_report.json", {
        "valid": not errors, "errors": errors, "semantic_retry": False,
        "semantic_repair": False,
    })
    if errors:
        for name in ("deterministic_projection.json", "projection_audit.json", "legacy_adapter_output.json", "legacy_compatibility_report.json"):
            dump_json(output / name, {"status": "not_run_due_to_semantic_validation"})
        result = {
            "validation_status": "failed_v3_1_semantic_validation",
            "errors": errors, "coverage": coverage,
            "haiku_calls": 1, "retries": 0,
        }
        dump_json(output / "validation_report.json", result)
        (output / "REPORT.md").write_text(f"# {EXPERIMENT}\n\n- Status: failed semantic validation\n- Errors: `{errors}`\n", encoding="utf-8")
        return result
    projected, audit = project_payload(payload, model_input)
    dump_json(output / "deterministic_projection.json", projected)
    dump_json(output / "projection_audit.json", audit)
    legacy = build_legacy_view(projected)
    dump_json(output / "legacy_adapter_output.json", legacy)
    legacy_errors = validate_result(legacy, packet)
    legacy_report = {
        "valid": not legacy_errors, "errors": legacy_errors,
        "not_found_primary_assessments_preserved_as_same_claims": all(
            any(old["claim_id"] == new["claim_id"] and new["status"] == "not_found" for new in legacy["claims"])
            for old in payload["requirement_assessments"] if old["status"] == "not_found"
        ),
        "sentinel_claims_synthesized": False,
    }
    dump_json(output / "legacy_compatibility_report.json", legacy_report)
    frozen_after = {role: file_sha256(path) for role, path in paths.items()}
    result = {
        "validation_status": "passed_uniform_weapon_v3_1_canary" if not legacy_errors and frozen_before == frozen_after else "failed_validation",
        "declared_requirement_ids": coverage["declared_requirement_ids"],
        "generated_requirement_ids": coverage["generated_requirement_ids"],
        "exact_coverage": coverage["exact_set_equality"],
        "coverage_ratio": coverage["coverage_ratio"],
        "semantic_valid": True,
        "projection_semantics_unchanged": audit["assessment_semantics_unchanged"] and audit["auxiliary_semantics_unchanged"],
        "legacy_compatible": not legacy_errors,
        "haiku_calls": 1, "gemini_calls": 0, "retries": 0,
        "other_questions_run": 0, "selective_run": False, "dense_r3_run": False,
        "frozen_inputs_unchanged": frozen_before == frozen_after,
    }
    dump_json(output / "validation_report.json", result)
    role_id = config["required_audit_requirement_id"]
    role = next(row for row in payload["requirement_assessments"] if row["requirement_id"] == role_id)
    report = [
        f"# {EXPERIMENT}", "", f"- Status: `{result['validation_status']}`",
        f"- Exact coverage: `{coverage['exact_set_equality']}` ({coverage['coverage_ratio']:.0%})",
        f"- Declared IDs: `{coverage['declared_requirement_ids']}`",
        f"- Generated IDs: `{coverage['generated_requirement_ids']}`",
        f"- Audited assessment: `{json.dumps(role, ensure_ascii=False)}`",
        f"- Legacy compatible: `{not legacy_errors}`; sentinel synthesis: `False`",
        f"- Haiku calls: 1; Gemini calls: 0; retries: 0",
        "- No other questions, Selective, Dense R3, reliability, handoff, or final-answer stages ran.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result
