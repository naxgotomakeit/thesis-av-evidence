from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable

PROMPT_VERSION = "v2"
OPERATIONS = {"identify_source", "identify_object", "identify_person", "describe_sound", "describe_action", "measure_delay", "count_occurrences", "compare_events", "other"}
ANCHOR_MODALITIES = {"time", "speech", "acoustic", "visual", "unknown"}
PRIMARY_MODALITIES = {"time", "speech", "acoustic", "visual", "none", "uncertain"}
RESOLVER_MODALITIES = {"visual", "speech", "acoustic"}
AUDIO_ROLES = {"direct_answer", "temporal_anchor", "supporting_evidence", "irrelevant", "uncertain"}
TEMPORAL_RELATIONS = {"during", "before", "after", "between", "sequence", "count_within", "none", "uncertain"}
VISUAL_ROUTES = {"anchor_guided_local_refinement", "coarse_event_retrieval", "coarse_then_local_refinement", "global_visual_fallback", "not_required", "uncertain"}

PLAN_SCHEMA: dict[str, Any] = {
    "answer_requirement": {"operation": sorted(OPERATIONS), "description": "short free-text description"},
    "anchor_cues": [{"text": "cue copied or paraphrased from the question", "modality": sorted(ANCHOR_MODALITIES)}],
    "primary_anchor_modality": sorted(PRIMARY_MODALITIES),
    "resolver_modalities": sorted(RESOLVER_MODALITIES),
    "audio_role": sorted(AUDIO_ROLES),
    "temporal_relation": sorted(TEMPORAL_RELATIONS),
    "requires_local_visual_inspection": "boolean",
    "visual_route": sorted(VISUAL_ROUTES),
    "fallback_route": "short description",
    "planner_confidence": "number from 0 to 1",
    "rationale": "maximum two short sentences explaining evidence roles without answering the question",
}

JSON_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_requirement": {"type": "object", "additionalProperties": False, "properties": {"operation": {"type": "string", "enum": sorted(OPERATIONS)}, "description": {"type": "string"}}, "required": ["operation", "description"]},
        "anchor_cues": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"text": {"type": "string"}, "modality": {"type": "string", "enum": sorted(ANCHOR_MODALITIES)}}, "required": ["text", "modality"]}},
        "primary_anchor_modality": {"type": "string", "enum": sorted(PRIMARY_MODALITIES)},
        "resolver_modalities": {"type": "array", "description": "Unique subset; uniqueness is enforced by the independent validator because the API schema grammar does not support uniqueItems.", "items": {"type": "string", "enum": sorted(RESOLVER_MODALITIES)}},
        "audio_role": {"type": "string", "enum": sorted(AUDIO_ROLES)},
        "temporal_relation": {"type": "string", "enum": sorted(TEMPORAL_RELATIONS)},
        "requires_local_visual_inspection": {"type": "boolean"},
        "visual_route": {"type": "string", "enum": sorted(VISUAL_ROUTES)},
        "fallback_route": {"type": "string"},
        "planner_confidence": {"type": "number", "description": "Value from 0 to 1; bounds are enforced by the independent validator because the API schema grammar does not support minimum/maximum."},
        "rationale": {"type": "string"},
    },
    "required": ["answer_requirement", "anchor_cues", "primary_anchor_modality", "resolver_modalities", "audio_role", "temporal_relation", "requires_local_visual_inspection", "visual_route", "fallback_route", "planner_confidence", "rationale"],
}

SYSTEM_PROMPT = """You are the Task 5A question-understanding planner for an audio-visual evidence system.
Produce only a retrieval plan; never answer the original question. Distinguish anchor modalities, which locate an event or time, from resolver modalities, which supply the information required for an answer. Audio can be a direct answer, temporal anchor, supporting evidence, irrelevant, or uncertain. Do not force visual inspection when audio can directly resolve the question, and do not assume every modality is required.
Use only the raw question and deterministic cues supplied in the user message. Never infer or use ground-truth answers, answer options, annotation context, reference timestamps, media, prior retrieval results, or human notes. Do not turn vague time expressions into numeric intervals.
Return exactly one JSON object conforming to the supplied schema, with no Markdown or commentary."""

USER_PROMPT_TEMPLATE = """Plan evidence retrieval for this raw question without answering it.

RAW QUESTION:
{question}

DETERMINISTIC CUES FROM QUESTION TEXT:
{cues_json}

FIXED OUTPUT SCHEMA AND FIELD DEFINITIONS:
{schema_json}

Return strict JSON only. Every field is required. resolver_modalities must be a unique subset of visual, speech, acoustic. rationale must contain at most two short sentences."""

CORRECTION_TEMPLATE = """Your previous response failed strict JSON/schema validation.
Validation errors:
{errors_json}

Return a corrected JSON object only. Preserve the planning task and obey the same schema. Do not answer the original question and do not introduce information not present in the raw question or deterministic cues."""

CODEX_ANALYSIS_INSTRUCTIONS = """After Claude planning is complete, semantically assess what the plan believes the question asks, whether anchor and resolver roles fit the raw question and deterministic cues, and whether anchor modality, resolvers, audio role, visual-inspection flag, and visual route are mutually consistent. Distinguish schema validity, semantic labeling quality, and likely functional impact. A debatable anchor/audio label is normally labeling-only when the required resolver and executable visual inspection route remain intact. Do not answer the question, retrieve evidence, use ground-truth answers, reference timestamps, media, or prior retrieval results, and do not alter Claude's plan."""


def _seconds(mm: str, ss: str) -> float:
    return float(int(mm) * 60 + int(ss))


def extract_cues(question: str) -> dict[str, list[dict[str, Any]]]:
    time_cues: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    clock_range = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})(?!\d)")
    for match in clock_range.finditer(question):
        time_cues.append({"raw_text": match.group(0), "cue_type": "explicit_time_range", "start_sec": _seconds(match.group(1), match.group(2)), "end_sec": _seconds(match.group(3), match.group(4)), "source": "question_text"})
        occupied.append(match.span())
    between_clock = re.compile(r"\bbetween\s+(\d{1,2}):(\d{2})\s+and\s+(\d{1,2}):(\d{2})\b", re.I)
    for match in between_clock.finditer(question):
        # Prefer the semantically complete "between ... and ..." span over
        # independent timestamp cues.
        time_cues.append({"raw_text": match.group(0), "cue_type": "explicit_time_range", "start_sec": _seconds(match.group(1), match.group(2)), "end_sec": _seconds(match.group(3), match.group(4)), "source": "question_text"})
        occupied.append(match.span())
    between_seconds = re.compile(r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s+seconds?\b", re.I)
    for match in between_seconds.finditer(question):
        time_cues.append({"raw_text": match.group(0), "cue_type": "explicit_time_range", "start_sec": float(match.group(1)), "end_sec": float(match.group(2)), "source": "question_text"})
        occupied.append(match.span())
    clock = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
    for match in clock.finditer(question):
        if any(a <= match.start() and match.end() <= b for a, b in occupied):
            continue
        value = _seconds(match.group(1), match.group(2))
        time_cues.append({"raw_text": match.group(0), "cue_type": "explicit_timestamp", "start_sec": value, "end_sec": value, "source": "question_text"})
    relative: list[dict[str, Any]] = []
    relative_pattern = re.compile(r"\b(?:at the (?:very )?start|at the beginning|after|before|during|between)\b", re.I)
    for match in relative_pattern.finditer(question):
        relative.append({"raw_text": match.group(0), "cue_type": "relative_temporal_expression", "source": "question_text"})
    quoted: list[dict[str, Any]] = []
    quote_pattern = re.compile(r'"([^"\r\n]+)"|\'([^\'\r\n]+)\'|“([^”\r\n]+)”|‘([^’\r\n]+)’')
    for match in quote_pattern.finditer(question):
        text = next(group for group in match.groups() if group is not None)
        quoted.append({"raw_text": match.group(0), "text": text, "cue_type": "quoted_phrase", "source": "question_text"})
    return {"time_cues": time_cues, "quoted_phrases": quoted, "relative_temporal_expressions": relative}


def render_user_prompt(question: str, cues: dict[str, Any]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        cues_json=json.dumps(cues, ensure_ascii=False, indent=2),
        schema_json=json.dumps(PLAN_SCHEMA, ensure_ascii=False, indent=2),
    )


def shared_prompt_text() -> str:
    return "\n\n".join((SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, json.dumps(PLAN_SCHEMA, ensure_ascii=False, indent=2), json.dumps(JSON_OUTPUT_SCHEMA, ensure_ascii=False, indent=2), CORRECTION_TEMPLATE))


def prompt_hash() -> str:
    return sha256(shared_prompt_text().encode("utf-8")).hexdigest()


def parse_model_json(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"answer_requirement", "anchor_cues", "primary_anchor_modality", "resolver_modalities", "audio_role", "temporal_relation", "requires_local_visual_inspection", "visual_route", "fallback_route", "planner_confidence", "rationale"}
    missing, extra = required - plan.keys(), plan.keys() - required
    errors += [f"missing field: {x}" for x in sorted(missing)] + [f"unexpected field: {x}" for x in sorted(extra)]
    ar = plan.get("answer_requirement")
    if not isinstance(ar, dict) or set(ar) != {"operation", "description"}:
        errors.append("answer_requirement must contain exactly operation and description")
    else:
        if ar["operation"] not in OPERATIONS: errors.append("invalid answer_requirement.operation")
        if not isinstance(ar["description"], str) or not ar["description"].strip(): errors.append("answer_requirement.description must be non-empty text")
    anchors = plan.get("anchor_cues")
    if not isinstance(anchors, list): errors.append("anchor_cues must be a list")
    else:
        for i, item in enumerate(anchors):
            if not isinstance(item, dict) or set(item) != {"text", "modality"}: errors.append(f"anchor_cues[{i}] has invalid structure")
            elif not isinstance(item["text"], str) or item["modality"] not in ANCHOR_MODALITIES: errors.append(f"anchor_cues[{i}] has invalid values")
    if plan.get("primary_anchor_modality") not in PRIMARY_MODALITIES: errors.append("invalid primary_anchor_modality")
    resolvers = plan.get("resolver_modalities")
    if not isinstance(resolvers, list) or any(x not in RESOLVER_MODALITIES for x in resolvers) or len(resolvers) != len(set(resolvers)): errors.append("resolver_modalities must be a unique valid list")
    if plan.get("audio_role") not in AUDIO_ROLES: errors.append("invalid audio_role")
    if plan.get("temporal_relation") not in TEMPORAL_RELATIONS: errors.append("invalid temporal_relation")
    if type(plan.get("requires_local_visual_inspection")) is not bool: errors.append("requires_local_visual_inspection must be boolean")
    if plan.get("visual_route") not in VISUAL_ROUTES: errors.append("invalid visual_route")
    if not isinstance(plan.get("fallback_route"), str) or not plan.get("fallback_route", "").strip(): errors.append("fallback_route must be non-empty text")
    confidence = plan.get("planner_confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1: errors.append("planner_confidence must be between 0 and 1")
    rationale = plan.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip(): errors.append("rationale must be non-empty text")
    elif len(re.findall(r"[.!?](?:\s|$)", rationale.strip())) > 2: errors.append("rationale must contain at most two sentences")
    return errors


@dataclass
class ModelReply:
    text: str
    latency_sec: float
    input_tokens: int
    output_tokens: int


def plan_with_retry(
    question: str,
    cues: dict[str, Any],
    request: Callable[[str, str, int], ModelReply],
) -> dict[str, Any]:
    user_prompt = render_user_prompt(question, cues)
    attempts: list[dict[str, Any]] = []
    messages = user_prompt
    for number in (1, 2):
        reply = request(SYSTEM_PROMPT, messages, number)
        errors: list[str] = []
        parsed = None
        try: parsed = parse_model_json(reply.text)
        except (json.JSONDecodeError, ValueError) as exc: errors = [f"JSON parse error: {exc}"]
        if parsed is not None: errors = validate_plan(parsed)
        attempts.append({"request_number": number, "rendered_prompt": messages, "raw_response": reply.text, "parsed_plan": parsed, "validation_errors": errors, "latency_sec": reply.latency_sec, "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens})
        if not errors:
            return {"plan": parsed, "validation_status": "valid", "retry_required": number == 2, "attempts": attempts}
        if number == 1:
            messages = user_prompt + "\n\n" + CORRECTION_TEMPLATE.format(errors_json=json.dumps(errors, ensure_ascii=False, indent=2))
    return {"plan": None, "validation_status": "failed_after_retry", "retry_required": True, "attempts": attempts}


def safe_question_cases(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """An allowlist boundary prevents answers, contexts, and reference fields entering Task 5A."""
    return [{"case_id": str(row["case_id"]), "question": str(row["question"])} for row in rows]


def semantic_diagnostic(question: str, cues: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    """Assess plan semantics from question text/cues only, without changing the plan."""
    operation = plan.get("answer_requirement", {}).get("operation", "other")
    description = plan.get("answer_requirement", {}).get("description", "unspecified requirement")
    primary = plan.get("primary_anchor_modality")
    resolvers = set(plan.get("resolver_modalities", []))
    audio_role = plan.get("audio_role")
    local_visual = plan.get("requires_local_visual_inspection") is True
    visual_route = plan.get("visual_route")
    has_explicit_time = bool(cues.get("time_cues"))
    has_quoted_speech = bool(cues.get("quoted_phrases"))
    local_routes = {"anchor_guided_local_refinement", "coarse_then_local_refinement"}
    issues: list[str] = []
    execution_issues: list[str] = []

    expected: set[str] = set()
    if operation == "describe_sound": expected.add("acoustic")
    elif operation == "identify_object": expected.add("visual")
    elif operation == "identify_person": expected.update(("speech", "visual"))
    elif operation == "describe_action": expected.add("visual")
    elif operation == "measure_delay": expected.add("speech" if has_quoted_speech else "acoustic")
    elif operation == "count_occurrences" and has_quoted_speech: expected.add("speech")

    missing = expected - resolvers
    if missing:
        execution_issues.append("Resolver plan may omit question-indicated modality/modalities: " + ", ".join(sorted(missing)) + ".")
    if "visual" in resolvers and (not local_visual or visual_route == "not_required"):
        execution_issues.append("Visual resolution is requested but visual inspection/route would not execute it consistently.")
    if local_visual and visual_route not in local_routes and visual_route not in {"coarse_event_retrieval", "global_visual_fallback"}:
        execution_issues.append("Local visual inspection is enabled without an executable visual route.")
    if audio_role == "irrelevant" and resolvers.intersection({"speech", "acoustic"}):
        execution_issues.append("Audio is labeled irrelevant while an audio resolver is required.")

    if has_explicit_time and primary not in {"time", "speech", "acoustic"}:
        issues.append("An explicit textual time cue exists but the primary anchor label does not reflect a time-capable anchor.")
    if has_quoted_speech and primary == "acoustic":
        issues.append("Quoted speech is labeled with a generic acoustic primary anchor rather than speech.")
    if audio_role == "direct_answer" and resolvers == {"visual"}:
        issues.append("Audio is labeled direct-answer although only visual evidence is designated to resolve the answer.")

    visual_execution_intact = "visual" in resolvers and local_visual and visual_route in local_routes.union({"coarse_event_retrieval", "global_visual_fallback"})
    if execution_issues:
        status, impact = "inconsistent", "may_omit_required_evidence"
        problem = " ".join(execution_issues + issues)
    elif issues:
        status = "questionable"
        impact = "labeling_quality_only" if visual_execution_intact or not expected.intersection({"visual"}) else "may_affect_ranking_or_budget"
        problem = " ".join(issues)
    else:
        status, impact, problem = "plausible", "none", "none"

    anchor_analysis = f"Primary anchor is {primary}; explicit-time cue present={has_explicit_time}, quoted-speech cue present={has_quoted_speech}. Resolver modalities are {sorted(resolvers)} for operation {operation}."
    routing = f"audio_role={audio_role}, requires_local_visual_inspection={local_visual}, visual_route={visual_route}. " + ("The declared routes and resolver roles are mutually executable." if not execution_issues else "Execution consistency concern: " + " ".join(execution_issues))
    human_check = "Confirm the anchor label and evidence-role labels against the raw question; then verify that every required resolver has an executable route." if problem != "none" else "Confirm that the planned anchor and resolver roles match the raw question before routing execution."
    return {
        "analysis_status": status,
        "question_understanding": f"The planner interprets the task as {operation}: {description}",
        "anchor_resolver_analysis": anchor_analysis,
        "routing_consistency": routing,
        "functional_impact": impact,
        "possible_problem": problem,
        "recommended_human_check": human_check,
    }


def aggregate_semantic_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {key: 0 for key in ("plausible", "questionable", "inconsistent")}
    execution_cases, labeling_cases = [], []
    for record in records:
        diagnostic = record["codex_semantic_diagnostic"]
        statuses[diagnostic["analysis_status"]] += 1
        if diagnostic["functional_impact"] in {"may_affect_ranking_or_budget", "may_omit_required_evidence"}: execution_cases.append(record["case_id"])
        if diagnostic["functional_impact"] == "labeling_quality_only": labeling_cases.append(record["case_id"])
    ready = statuses["inconsistent"] == 0
    return {
        "overall_assessment": "Plans are structurally valid; semantic labels require targeted human review before execution." if statuses["questionable"] else "Plans are structurally valid and semantically plausible, pending human review.",
        "status_counts": statuses,
        "recurring_strengths": ["Plans distinguish anchor cues from resolver modalities.", "Visual inspection is not forced for audio-resolvable questions."],
        "recurring_semantic_label_issues": ["Primary-anchor and audio-role labels can be semantically debatable without changing the executable resolver route."] if labeling_cases else [],
        "cases_whose_routing_may_affect_actual_execution": execution_cases,
        "cases_with_labeling_only_issues": labeling_cases,
        "ready_for_limited_task5b_prototype": ready,
        "readiness_caveat": "Limited prototype readiness means schema-valid routing plans with no detected evidence-omission inconsistency; it does not establish semantic correctness and still requires human review.",
    }
