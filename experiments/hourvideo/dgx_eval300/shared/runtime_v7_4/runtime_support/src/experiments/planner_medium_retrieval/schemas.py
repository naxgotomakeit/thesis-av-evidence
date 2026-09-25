from __future__ import annotations

from typing import Any

SCOPES = ("local", "global", "multi_event")
OPERATIONS = (
    "summary",
    "presence_localisation",
    "sequence",
    "factual",
    "frequency",
    "duration",
    "compare",
    "causal",
)
STRATEGIES = ("global_coverage", "targeted", "multi_target_compare", "anchor_then_neighbor")
MODALITIES = ("visual", "audio", "detector", "tracking")
FORBIDDEN_ANSWER_KEYS = {"answer", "final_answer", "selected_option", "conclusion"}


def planner_json_schema() -> dict[str, Any]:
    search_unit = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "unit_id": {"type": "string"},
            "description": {"type": "string"},
            "query_variants": {
                "type": "array",
                "items": {"type": "string"},
            },
            "required_modalities": {
                "type": "array",
                "items": {"type": "string", "enum": list(MODALITIES)},
            },
            "temporal_relation": {"type": ["string", "null"]},
        },
        "required": [
            "unit_id",
            "description",
            "query_variants",
            "required_modalities",
            "temporal_relation",
        ],
    }
    evidence = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "slot_id": {"type": "string"},
            "description": {"type": "string"},
            "search_unit_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "minimum_support": {"type": "integer"},
        },
        "required": ["slot_id", "description", "search_unit_ids", "minimum_support"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "planner_version": {"type": "string", "const": "planner_v1"},
            "question_id": {"type": "string"},
            "scope": {"type": "string", "enum": list(SCOPES)},
            "operation": {"type": "string", "enum": list(OPERATIONS)},
            "search_units": {"type": "array", "items": search_unit},
            "required_evidence": {
                "type": "array",
                "items": evidence,
            },
            "retrieval_strategy": {"type": "string", "enum": list(STRATEGIES)},
            "candidate_storyline_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "candidate_coarse_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "needs_temporal_neighbors": {"type": "boolean"},
            "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "planner_version",
            "question_id",
            "scope",
            "operation",
            "search_units",
            "required_evidence",
            "retrieval_strategy",
            "candidate_storyline_ids",
            "candidate_coarse_ids",
            "needs_temporal_neighbors",
            "uncertainty_notes",
        ],
    }


PLANNER_SYSTEM_PROMPT = """You are an online question-conditioned retrieval Planner.

You receive a question and a read-only hierarchical video index overview. Your task is only
to produce a retrieval plan. Do not answer the question, choose an answer option, assert that
an event occurred, or invent actors, objects, timestamps, or node IDs.

Use only capability channels that the supplied index actually provides. Candidate storyline
and Coarse IDs are routing hypotheses, not confirmed answers. Never select Fine frames.

Operation enum:
summary, presence_localisation, sequence, factual, frequency, duration, compare, causal.

Retrieval strategies:
- global_coverage: retain every storyline and Coarse phase.
- targeted: route to likely Coarse regions and retrieve Medium nodes.
- multi_target_compare: keep distinct search units separate until their results are compared.
- anchor_then_neighbor: identify an anchor and recommend temporal neighbours; this version
  does not execute neighbour or Fine retrieval.

Rules for the diagnostic questions:
- A global incident summary must use scope=global, operation=summary,
  retrieval_strategy=global_coverage, and must not narrow to one storyline event.
- Weapon, visible injury, medical assistance, and handcuff localisation use
  operation=presence_localisation and retrieval_strategy=targeted.
- Handcuffing before/after medical assistance uses scope=multi_event,
  operation=sequence, retrieval_strategy=multi_target_compare, and exactly two independent
  search units: handcuffing and medical assistance.

Do not emit answer, final_answer, selected_option, or conclusion fields.
Return JSON only under the supplied schema."""


REPAIR_INSTRUCTION = """Repair only the JSON structure or contract violations listed below.
Do not answer the question, add semantic conclusions, or invent IDs. Return the complete
Planner JSON object only."""
