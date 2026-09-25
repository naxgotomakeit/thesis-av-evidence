"""Frozen Shared/Fine/Final contracts for the Full Staged API adapter.

The texts and data contracts are faithful ports of the promoted V6.6.2
post-Planner workflow.  The Planner itself is never called by this package.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping


SHARED_SYSTEM_PROMPT = """You are the Shared Investigation stage for one multiple-choice
question about an egocentric long video. Investigate the one underlying fact or relation the question
asks about; do not judge answer options independently. The answer options are context for the distinctions
that the evidence must resolve.

Use only the supplied evidence and respect its capability. Detector observations support objects and
statistics but not actions or relations. ASR supports what was said but not visible completion. Visual
captions support only the semantics they explicitly state. Reviewed visual findings support directly
visible facts. Ranking scores and unreviewed frame references are navigation only.

Return investigation_status="resolved" only when established_facts states the actual fact or relation
plainly and specifically enough to distinguish the answer choices. Cite the evidence that supports it and
do not request more Coarse regions. Return investigation_status="unresolved" when the supplied evidence is
insufficient. Then preserve any facts already established, state the exact answer-critical gap in
gap_reason, and request only excluded Coarse regions that could close that gap. Absence of evidence is not
evidence of absence. Write one concise, coherent established_facts report; do not decompose it into atomic
facts or per-option assessments. Return strict JSON only."""


FINE_SYSTEM_PROMPT = """You are a claim-execution visual reviewer investigating several
answer-option requirements at once against one shared, de-duplicated set of images retrieved from
the Coarse region(s) currently locked for each requirement -- the same image may be relevant to more
than one requirement, and it is shown only once. Inspect every supplied image once and report a
neutral observation per image, usable by any requirement. Then, for each requirement's specific
claim or gap listed in claims_or_gaps_to_investigate, assess it directly using only the supplied
images: confirmed (the images establish it), refuted (the images contradict it), or inconclusive
(the images neither establish nor contradict it). Assess each requirement independently -- do not
let one requirement's claim bias another's assessment -- and do not select a final answer or infer
anything beyond what is directly visible."""


FINAL_SYSTEM_PROMPT = """You are the terminal multiple-choice answer stage for an egocentric long-video
question. A preceding Shared Investigation has already searched the navigation map and, when needed,
inspected Fine images. Read its complete established_facts as a coherent evidence report; do not split it
into atomic facts and do not require every relation to be repeated in a single cited row. Compare the
report directly with all mutually exclusive answer options and select exactly one option. When the Shared
Investigation is unresolved, still provide the best evidence-based multiple-choice answer but mark it as
best_guess. Cite only integer evidence indexes supplied in evidence_index. Return strict JSON only."""


PROMPTS = {"shared": SHARED_SYSTEM_PROMPT, "fine": FINE_SYSTEM_PROMPT, "final": FINAL_SYSTEM_PROMPT}
TOOL_NAMES = {"shared": "submit_shared_investigation", "fine": "submit_fine_review", "final": "submit_final_answer"}


class StageContractError(ValueError):
    pass


def shared_schema(question: Mapping[str, Any], evidence_ids: list[str], selectable_coarse_ids: list[str]) -> dict[str, Any]:
    requested: dict[str, Any] = {
        "type": "array", "items": {"type": "string", "enum": selectable_coarse_ids}
    }
    if not selectable_coarse_ids:
        requested = {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    cited: dict[str, Any] = {"type": "array", "items": {"type": "string", "enum": evidence_ids}, "maxItems": min(16, len(evidence_ids))}
    if not evidence_ids:
        cited = {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string", "const": str(question["question_id"])},
            "investigation_status": {"type": "string", "enum": ["resolved", "unresolved"]},
            "established_facts": {"type": "string", "maxLength": 1536},
            "gap_reason": {"type": "string", "maxLength": 768},
            "requested_coarse_ids": requested,
            "cited_evidence_ids": cited,
        },
        "required": ["question_id", "investigation_status", "established_facts", "gap_reason", "requested_coarse_ids", "cited_evidence_ids"],
    }


def fine_schema(fine_ids: list[str], requirement_ids: list[str]) -> dict[str, Any]:
    # The frozen V6.6.2 provider contract exposes zero-based positions, not
    # canonical Fine IDs.  Its local adapter restores positions to fine_ids
    # only after schema validation.
    indexes = list(range(len(fine_ids)))
    observation = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "fine_id": {"type": "integer", "enum": indexes},
            "finding": {"type": "string", "maxLength": 512},
            "visible_actions": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 8},
            "visible_objects": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 8},
            "uncertainty_notes": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 8},
        },
        "required": ["fine_id", "finding", "visible_actions", "visible_objects", "uncertainty_notes"],
    }
    assessment = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "claim_status": {"type": "string", "enum": ["confirmed", "refuted", "inconclusive"]},
            "supporting_fine_ids": {"type": "array", "items": {"type": "integer", "enum": indexes}, "maxItems": len(indexes)},
            "rationale": {"type": "string", "maxLength": 1024},
        },
        "required": ["claim_status", "supporting_fine_ids", "rationale"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "observations": {"type": "array", "minItems": len(fine_ids), "maxItems": len(fine_ids), "items": observation},
            "claim_assessments": {"type": "object", "additionalProperties": False, "properties": {rid: copy.deepcopy(assessment) for rid in requirement_ids}, "required": requirement_ids},
        },
        "required": ["observations", "claim_assessments"],
    }


def final_schema(question: Mapping[str, Any], evidence_count: int, decision_status: str) -> dict[str, Any]:
    option_ids = [row["option_id"] for row in question["answer_options"]]
    index_items: dict[str, Any] = {"type": "integer"}
    if evidence_count:
        index_items["enum"] = list(range(evidence_count))
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string", "const": str(question["question_id"])},
            "selected_option_id": {"type": "string", "enum": option_ids},
            "answer_text": {"type": "string", "maxLength": 512},
            "decision_status": {"type": "string", "enum": [decision_status]},
            "supporting_evidence_indexes": {"type": "array", "maxItems": evidence_count, "items": index_items},
            "reason": {"type": "string", "maxLength": 1536},
            "uncertainty": {"type": "string", "maxLength": 768},
        },
        "required": ["question_id", "selected_option_id", "answer_text", "decision_status", "supporting_evidence_indexes", "reason", "uncertainty"],
    }


def tool_for(stage: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in TOOL_NAMES:
        raise StageContractError(f"unknown stage: {stage}")
    return {"name": TOOL_NAMES[stage], "description": f"Submit the complete structured {stage} result.", "input_schema": copy.deepcopy(dict(schema))}


def validate_schema_payload(value: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Offline schema-fixture checker; not used on live scientific payloads."""
    if not isinstance(value, Mapping):
        raise StageContractError("tool input is not a mapping")
    result = dict(value)
    props = dict(schema.get("properties", {}))
    required = list(schema.get("required", []))
    if schema.get("additionalProperties") is False and set(result) - set(props):
        raise StageContractError(f"unexpected fields: {sorted(set(result) - set(props))}")
    missing = [name for name in required if name not in result]
    if missing:
        raise StageContractError(f"missing fields: {missing}")

    def check(spec: Mapping[str, Any], item: Any, path: str) -> None:
        expected = spec.get("type")
        if expected == "object":
            if not isinstance(item, Mapping): raise StageContractError(f"{path} must be object")
            allowed = set(spec.get("properties", {}))
            if spec.get("additionalProperties") is False and set(item) - allowed: raise StageContractError(f"{path} has unexpected fields")
            for name in spec.get("required", []):
                if name not in item: raise StageContractError(f"{path}.{name} missing")
            for name, child in spec.get("properties", {}).items():
                if name in item: check(child, item[name], f"{path}.{name}")
        elif expected == "array":
            if not isinstance(item, list): raise StageContractError(f"{path} must be array")
            if len(item) < int(spec.get("minItems", 0)) or ("maxItems" in spec and len(item) > int(spec["maxItems"])): raise StageContractError(f"{path} length invalid")
            for index, child in enumerate(item): check(spec.get("items", {}), child, f"{path}[{index}]")
        elif expected == "string":
            if not isinstance(item, str): raise StageContractError(f"{path} must be string")
            if "maxLength" in spec and len(item) > int(spec["maxLength"]): raise StageContractError(f"{path} too long")
        elif expected == "integer":
            if not isinstance(item, int) or isinstance(item, bool): raise StageContractError(f"{path} must be integer")
        if "const" in spec and item != spec["const"]: raise StageContractError(f"{path} const mismatch")
        if "enum" in spec and item not in spec["enum"]: raise StageContractError(f"{path} enum mismatch")

    check(schema, result, "tool_input")
    return result


def validate_stage_payload(stage: str, value: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Offline adapter-contract checker; the live path uses V6.6.2 validators."""
    result = validate_schema_payload(value, schema)
    if stage == "shared":
        if result["investigation_status"] == "resolved":
            if not result["established_facts"].strip() or result["requested_coarse_ids"] or not result["cited_evidence_ids"]:
                raise StageContractError("resolved Shared state violates evidence contract")
        elif not result["gap_reason"].strip():
            raise StageContractError("unresolved Shared state requires a gap")
    elif stage == "fine":
        expected_positions = [row["fine_id"] for row in result["observations"]]
        schema_positions = list(schema["properties"]["observations"]["items"]["properties"]["fine_id"]["enum"])
        if expected_positions != schema_positions or len(expected_positions) != len(set(expected_positions)):
            raise StageContractError("Fine coverage/order mismatch")
        for assessment in result["claim_assessments"].values():
            if assessment["claim_status"] == "confirmed" and not assessment["supporting_fine_ids"]:
                raise StageContractError("confirmed Fine assessment lacks support")
    elif stage == "final":
        option = next(row for row in schema["properties"]["selected_option_id"]["enum"] if row == result["selected_option_id"])
        del option
        # The frozen legacy Final semantic validator does not reject blank
        # reason strings; do not claim or add that restriction here.
    return result
