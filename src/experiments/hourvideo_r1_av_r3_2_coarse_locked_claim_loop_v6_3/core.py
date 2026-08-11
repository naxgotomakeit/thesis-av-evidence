from __future__ import annotations

import base64
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.fine_reranking.core import rerank_fines_for_medium, select_diverse_fines, temporally_diverse
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import (
    FINAL_SYSTEM,
    SUFFICIENCY_SYSTEM,
    _anthropic_call,
    _estimated_cost,
    _gemini_call,
    _load_env,
)
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import (
    _final,
    _final_schema,
    _fine_registry,
    _lexical_rows,
    _normalize,
    _parent_map,
    _planner_map,
    _query_text,
    option_requirements,
)
from experiments.planner_medium_retrieval.core import SiglipTextEncoder, lexical_similarity


def _coarse_locked_planner_schema(requirements: list[dict[str, Any]], coarse_ids: list[str]) -> dict[str, Any]:
    # V6.3 (2026-08-11): replaces V6.1's per-Coarse-per-option judgment array (one {coarse_id, selected,
    # reason} entry for EVERY Coarse region, for EVERY option -- 5 options x 36 Coarse = 180 judgments on
    # a real case, the confirmed root cause of a planner_max_tokens truncation) with a sparse
    # selected_coarse_ids list: the model states only what it wants, not a "selected=false, here is why
    # not" essay for every region it doesn't want. Verified against real V6.1 code+data before this
    # change: (1) core.py's _locked_coarse_ids only ever reads the selected=True subset -- the excluded
    # judgments' reasons were never used to build retrieval scope; (2) _excluded_judgments_union already
    # collapsed all per-option exclusion reasons for a region down to one representative reason, so the
    # other N-1 were always discarded; (3) that one surviving reason feeds ONLY the shared investigation's
    # requested_coarse_ids escalation path, and a real audit of all 20 shared_investigation.json files
    # from V6.1's actual 10-video run found requested_coarse_ids non-empty in 0/20 -- the escalation path
    # this reason text existed to support was never exercised in practice. The escalation path itself is
    # kept (not removed): see _excluded_judgments_union below, which now sources per-region context from
    # the Coarse map's own navigation_summary instead of a planner-authored reason.
    unit = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "search_description": {"type": "string"},
            "query_variants": {"type": "array", "items": {"type": "string"}},
            "modality_strategy": {"type": "string"},
            "selected_coarse_ids": {"type": "array", "items": {"type": "string", "enum": coarse_ids}},
            "selection_reason": {"type": "string"},
        },
        "required": ["search_description", "query_variants", "modality_strategy", "selected_coarse_ids", "selection_reason"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "requirement_plans": {
                "type": "object", "additionalProperties": False,
                "properties": {row["requirement_id"]: unit for row in requirements},
                "required": [row["requirement_id"] for row in requirements],
            },
            # Named for what V6 actually does, unlike v1/V3's "hard_filtering_allowed: false" (which
            # asserted the opposite -- that Coarse hints were only a soft sort priority, never a real
            # filter). V6's retrieval genuinely restricts candidates to the locked Coarse set, so the
            # artifact should say so rather than carry over a field whose meaning no longer applies.
            "coarse_lock_is_hard_scope": {"type": "boolean", "const": True},
        },
        "required": ["question_id", "requirement_plans", "coarse_lock_is_hard_scope"],
    }


def _project_coarse_locked_plan(value: dict[str, Any], requirements: list[dict[str, Any]]) -> dict[str, Any]:
    keyed = value["requirement_plans"]
    return {
        "question_id": value["question_id"],
        "requirement_plans": [
            {"requirement_id": row["requirement_id"], **keyed[row["requirement_id"]]}
            for row in requirements
        ],
        "coarse_lock_is_hard_scope": value["coarse_lock_is_hard_scope"],
    }


def _validate_coarse_locked_plan(
    plan: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], coarse_ids: set[str],
) -> None:
    if plan.get("question_id") != question["question_id"] or plan.get("coarse_lock_is_hard_scope") is not True:
        raise ValueError("Planner identity or hard-filtering contract invalid")
    if [row.get("requirement_id") for row in plan.get("requirement_plans", [])] != [row["requirement_id"] for row in requirements]:
        raise ValueError("Planner requirement coverage/order invalid")
    for row in plan["requirement_plans"]:
        if not row["query_variants"]:
            raise ValueError(f"empty Planner query: {row['requirement_id']}")
        selected = row["selected_coarse_ids"]
        if len(selected) != len(set(selected)):
            raise ValueError(f"Planner selected_coarse_ids has duplicates: {row['requirement_id']}")
        if not set(selected) <= coarse_ids:
            raise ValueError(f"Planner selected an unknown coarse_id: {row['requirement_id']}")
        # Zero-selected is still legitimate (unchanged from V6.1's rationale, docstring below kept for
        # context): on a "which happened first, A or B" question the wrong-category distractor options
        # (neither A nor B) can be correctly ruled out from the option text alone, with no Coarse region
        # genuinely supporting them -- forcing at least one selection assumed every option restates the
        # same underlying question (true for a duration comparison, false here). selection_reason is
        # still required either way, to say so plainly rather than silently return an empty list.
        if not row["selection_reason"].strip():
            raise ValueError(f"empty Planner selection_reason: {row['requirement_id']}")


_EMPTY_SELECTION_REASON_FALLBACK = (
    "Planner did not return a compliant selection_reason for this requirement; defaulted to a "
    "placeholder. The selected_coarse_ids for this requirement (if any) are still honored as-is."
)


def _repair_planner_coverage(plan: dict[str, Any], coarse_ids: list[str]) -> dict[str, Any]:
    # V6.3: repair no longer needs to force full per-Coarse coverage (there is no "judge every region"
    # requirement left to satisfy). What the sparse schema's enum constraint still cannot guarantee on
    # its own is (a) no duplicate ids within one requirement's selection and (b) a non-empty
    # selection_reason -- same class of gap _downgrade_noncompliant_investigation exists for elsewhere
    # in this module (schema enums bound values but not uniqueness/non-emptiness).
    coarse_id_set = set(coarse_ids)
    repaired = copy.deepcopy(plan)
    for row in repaired["requirement_plans"]:
        seen = list(dict.fromkeys(row["selected_coarse_ids"]))
        row["selected_coarse_ids"] = [cid for cid in seen if cid in coarse_id_set]
        if not row["selection_reason"].strip():
            row["selection_reason"] = _EMPTY_SELECTION_REASON_FALLBACK
    return repaired


def _call_coarse_locked_planner(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    coarse_ids: list[str], map_doc: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = {"question": question, "requirements": requirements, "navigation_map": _planner_map(map_doc)}
    schema = _coarse_locked_planner_schema(requirements, coarse_ids)
    coarse_id_set = set(coarse_ids)
    attempts = int(cfg.get("max_validation_retries", 2))
    last_error: ValueError | RuntimeError | None = None
    plan: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for _ in range(attempts):
        # V6.3's sparse selected_coarse_ids output is far smaller than V6.1's per-region judgment array
        # (see _coarse_locked_planner_schema), but a wide semantic navigation map (r3_2 sides have run up
        # to 47 Coarse regions vs. r1_av's <=17) could still in principle push output past
        # planner_max_tokens on a case with many genuinely relevant regions across many options.
        # _anthropic_call raises RuntimeError (not ValueError) in that case; treat it the same as a
        # validation failure -- retry, and if every attempt is exhausted, surface it rather than crash
        # with no chance to try again -- since there is no plan payload to fall back to, repair cannot
        # help here.
        try:
            provider, usage = _anthropic_call(cfg, COARSE_LOCKED_PLANNER_PROMPT, payload, schema, int(cfg["anthropic"]["planner_max_tokens"]))
        except RuntimeError as error:
            last_error = error
            continue
        plan = _project_coarse_locked_plan(provider, requirements)
        try:
            _validate_coarse_locked_plan(plan, question, requirements, coarse_id_set)
            return plan, usage, payload
        except ValueError as error:
            last_error = error
    if plan is None:
        raise last_error
    repaired = _repair_planner_coverage(plan, coarse_ids)
    try:
        _validate_coarse_locked_plan(repaired, question, requirements, coarse_id_set)
        return repaired, usage, payload
    except ValueError:
        raise last_error


def _locked_coarse_ids(requirement_plan: dict[str, Any]) -> set[str]:
    return set(requirement_plan["selected_coarse_ids"])


def _rank_within_locked_coarse(
    cfg: dict[str, Any], question: dict[str, Any], requirement_plan: dict[str, Any], hierarchy: dict[str, Any],
    lexical: list[dict[str, str]], medium_embeddings: np.ndarray, parent: dict[str, str], encoder: Any,
    locked_coarse_ids: set[str],
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    text = _query_text(question, requirement_plan)
    started = time.perf_counter(); query = np.asarray(encoder.encode([text])[0]); encode_sec = time.perf_counter() - started
    candidates = [
        (index, medium, lex) for index, (medium, lex) in enumerate(zip(hierarchy["medium_nodes"], lexical))
        if parent[medium["medium_id"]] in locked_coarse_ids
    ]
    if not candidates:
        raise ValueError("no Medium nodes inside the locked Coarse set")
    raw = np.array([float(medium_embeddings[index] @ query) for index, _, _ in candidates])
    visual = _normalize(raw)
    rows = []
    for position, (index, medium, lex) in enumerate(candidates):
        lexical_score, terms = lexical_similarity(text, lex["lexical_text"])
        combined = float(cfg["ranking"]["visual_weight"]) * float(visual[position]) + float(cfg["ranking"]["lexical_weight"]) * lexical_score
        rows.append({
            "medium_id": medium["medium_id"], "start_sec": medium["start_sec"], "end_sec": medium["end_sec"],
            "parent_coarse_id": parent[medium["medium_id"]], "lexical_source": lex["lexical_source"],
            "visual_score_raw": float(raw[position]), "visual_score_normalized": float(visual[position]),
            "lexical_score": lexical_score, "matched_terms": terms,
            "combined_score": combined, "source_fine_ids": medium["source_fine_ids"],
        })
    rows.sort(key=lambda row: (-row["combined_score"], row["start_sec"], row["medium_id"]))
    for rank, row in enumerate(rows, 1): row["rank"] = rank
    return rows, query, encode_sec


def _allocate_fine_quota_by_coarse(
    fine_rows: list[dict[str, Any]], parent: dict[str, str], cap: int,
) -> list[dict[str, Any]]:
    # A flat, score-ranked truncation (the original version of this cap) re-introduces at the Fine
    # level exactly the coverage loss that Coarse-locking was built to fix at the Medium level: a
    # low-scoring-but-locked Coarse region could get zero images while a high-scoring one takes all
    # `cap` slots. Instead, give every locked Coarse region a guaranteed base share first, and only let
    # score compete for whatever slots are left over.
    if len(fine_rows) <= cap:
        return fine_rows
    by_coarse: dict[str, list[dict[str, Any]]] = {}
    for row in fine_rows:
        by_coarse.setdefault(parent[row["medium_id"]], []).append(row)
    base_quota = max(1, cap // len(by_coarse))
    selected: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for rows in by_coarse.values():
        selected.extend(rows[:base_quota])
        leftover.extend(rows[base_quota:])
    leftover.sort(key=lambda row: row["medium_rank"])
    slots_left = max(0, cap - len(selected))
    selected.extend(leftover[:slots_left])
    return selected


def _coarse_scoped_retrieval(
    cfg: dict[str, Any], side: str, question: dict[str, Any], plan: dict[str, Any], map_doc: dict[str, Any],
    hierarchy: dict[str, Any], projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: np.ndarray, fine_by_id: dict[str, dict[str, Any]], fine_embeddings: np.ndarray,
    row_by_id: dict[str, int], encoder: Any, locked_coarse_ids_by_requirement: dict[str, set[str]],
) -> dict[str, Any]:
    parent = _parent_map(map_doc); medium_by_id = {row["medium_id"]: row for row in hierarchy["medium_nodes"]}
    lexical = _lexical_rows(side, projections, captions)
    output = {
        "question_id": question["question_id"], "candidate_universe_count": len(hierarchy["medium_nodes"]),
        "requirements": {}, "ranking_formula": "0.6*normalized SigLIP + 0.3*lexical, Coarse-scoped, no Top-K truncation",
    }
    for requirement_plan in plan["requirement_plans"]:
        rid = requirement_plan["requirement_id"]
        locked = locked_coarse_ids_by_requirement[rid]
        rows, query, encode_sec = _rank_within_locked_coarse(cfg, question, requirement_plan, hierarchy, lexical, medium_embeddings, parent, encoder, locked)
        fine_rows = []
        for medium_row in rows:
            medium = dict(medium_by_id[medium_row["medium_id"]])
            medium["child_fine_ids"] = medium["source_fine_ids"]
            ranking = rerank_fines_for_medium(
                question_id=question["question_id"], search_unit_id=rid, medium=medium,
                fine_by_id=fine_by_id, fine_embeddings=fine_embeddings, row_by_id=row_by_id, query_embedding=query,
            )
            chosen, reason = select_diverse_fines(
                {rid: ranking}, strategy="top_relevance_then_temporal_diversity",
                max_fines=int(cfg["ranking"]["max_fine_per_medium"]), minimum_gap=float(cfg["ranking"]["minimum_fine_gap_sec"]),
            )
            for item in chosen:
                fine_rows.append({
                    "fine_id": item["fine_id"], "medium_id": medium["medium_id"], "timestamp_sec": item["representative_frame_timestamp_sec"],
                    "source_frame_path": item["representative_frame_path"], "frame_path": item["representative_frame_path"],
                    "siglip_score_raw": item["siglip_score_raw"], "selection_reason": item["selection_reason"],
                    "one_fine_reason": reason, "medium_rank": medium_row["rank"],
                })
        # Real Gemini calls against this video's 12-Coarse map (~46 Fine images once a broad Coarse lock
        # is legitimately needed) returned HTTP 400 "invalid argument" above ~30 images/request, hence a
        # hard cap. Below the cap is per-Coarse quota allocation (_allocate_fine_quota_by_coarse), not a
        # flat score-ranked truncation -- see that function for why. This is still not fully claim-aware
        # (dense-near-a-boundary vs spread-across-a-region sampling remains deferred).
        cap = int(cfg["ranking"].get("max_total_fine_evidence_per_claim", 20))
        capped_rows = _allocate_fine_quota_by_coarse(fine_rows, parent, cap)
        output["requirements"][rid] = {
            "locked_coarse_ids": sorted(locked), "all_medium_rankings": rows,
            "selected_medium_ids": [row["medium_id"] for row in rows], "selected_fine_evidence": capped_rows,
            "allowed_image_ids": [row["fine_id"] for row in capped_rows], "query_encode_sec": encode_sec,
        }
    return output


OPTION_STATUS_VALUES = ("supported", "refuted", "unresolved")
INVESTIGATION_STATUS_VALUES = ("resolved", "unresolved")

# Confirmed on real traffic (819c8af7 option D, pre-V6.1): free text argued "Option D is supported,
# not refuted" while the model's own status field said "refuted" -- a direct self-contradiction that
# nothing caught. Generic on purpose (status vocabulary differs between the shared investigation and
# the option-mapping step that consume it) -- a heuristic phrase check, not a proof of consistency.
_STATUS_REVERSAL_PHRASES = {
    "refuted": ("is supported", "is the correct answer", "not refuted"),
    "supported": ("is refuted", "is incorrect", "is not the correct answer", "not supported"),
}


def _claim_text_contradicts_status(status: str, claim_text: str) -> bool:
    lowered = claim_text.lower()
    return any(phrase in lowered for phrase in _STATUS_REVERSAL_PHRASES.get(status, ()))


# V6.1: a single shared investigation establishes ONE fact about the video for the whole question,
# instead of each answer option separately investigating "is my own text true?". This directly targets
# two real diagnosed problems from the 2026-08-06/07 pilot: (1) independent per-option investigation
# could honestly support two mutually exclusive options at once (7 of 13 confirmed finals) because each
# option's investigation never learns about the other's evidence; (2) when every option's planner
# locked zero Coarse regions, the old per-option SFC had nothing to fall back on except the planner's
# own exclusion reasoning, which reads "not evidenced at Coarse level" as "did not happen" -- confirmed
# as a real false-refutation on ab93e55b's option C (0/41 Coarse regions selected, then refuted from
# text inference alone, never a single image viewed). Still free-text output, not a structured relation
# graph -- this is not a return to V5.3's failed contract-compiler approach.
SHARED_INVESTIGATION_SYSTEM = SUFFICIENCY_SYSTEM + """
You are investigating ONE underlying question about a video -- not judging any single answer option.
The answer options are supplied only as context for what distinction ultimately matters; your job is to
establish the actual fact(s) the question is asking about (e.g. which of two events lasted longer, what
specific action two people performed together, what the exact sequence of steps was before a named
event), using only the supplied evidence. Set investigation_status to "resolved" once you can state that
fact plainly and specifically enough to distinguish between the answer options, or "unresolved" if the
evidence so far does not let you state it with confidence. When resolved, state established_facts as
plain prose describing what you can now assert actually happened, phrased so it is directly checkable
against any answer option's wording -- include any comparative reasoning (durations, sequence, who did
what) inline as ordinary sentences, not a separate structured relation. Whenever the supplied evidence
list is non-empty, cite at least one cited_evidence_ids entry that actually supports established_facts;
never mark resolved based on a guess with nothing cited, and never mark resolved when the evidence list
is empty -- an empty evidence list (Coarse map AND the Medium-level lexical fallback both turned up
nothing) means the described event is simply not evidenced in this video's available text, which is an
UNRESOLVED conclusion, never a "resolved" claim that it definitely did not happen. Absence of evidence is
not evidence of absence. For "unresolved", state gap_reason plainly and never leave established_facts as
a substitute guess. The payload's excluded_coarse_regions field lists every Coarse region no answer
option's search currently locks, each with its own navigation_summary from the video's semantic map --
read those summaries. Only if a specific one now looks like it might actually bear on the fact you're
establishing should you name that exact region in requested_coarse_ids. Never leave gap_reason empty
while unresolved, or established_facts empty while resolved. If nothing was excluded, requested_coarse_ids
must be an empty array. Before you commit to established_facts, decide investigation_status first and
make sure established_facts is consistent with it, not its opposite.

When resolved, also break established_facts down into atomic_facts: a list of individual, checkable
statements, each paired with exactly the evidence_ids that specific statement rests on (not the whole
evidence list -- the specific citation(s) for that one statement). Be precise and conservative in each
statement's own wording: state only what the cited evidence actually says, not a more specific label or
category you are inferring or guessing at (e.g. if the evidence says "a messy room", the atomic fact must
say "a messy room", not upgrade it to "a bedroom" -- a guess belongs in your reasoning, not asserted as a
fact with nothing behind it but a hunch). established_facts may still read as a natural narrative; the
atomic_facts breakdown is what downstream logic will check claim-by-claim against its own citations, so
each one must be honest about exactly how specific the cited evidence actually is.
"""


def _shared_investigation_schema(
    question: dict[str, Any], evidence: list[dict[str, Any]], excluded_coarse_ids: list[str],
) -> dict[str, Any]:
    evidence_ids = [row["evidence_id"] for row in evidence]
    # Same empty-enum/const limitations as the old per-option claim schema (confirmed against the real
    # Anthropic API): fall back to an unconstrained array and clip hallucinated ids after the call.
    requested = (
        {"type": "array", "items": {"type": "string", "enum": excluded_coarse_ids}}
        if excluded_coarse_ids else {"type": "array", "items": {"type": "string"}}
    )
    cited = (
        {"type": "array", "items": {"type": "string", "enum": evidence_ids}}
        if evidence_ids else {"type": "array", "items": {"type": "string"}}
    )
    fact_item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "evidence_ids": cited,
        },
        "required": ["text", "evidence_ids"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "investigation_status": {"type": "string", "enum": list(INVESTIGATION_STATUS_VALUES)},
            "established_facts": {"type": "string"},
            "atomic_facts": {"type": "array", "items": fact_item},
            "gap_reason": {"type": "string"},
            "requested_coarse_ids": requested,
            "cited_evidence_ids": cited,
        },
        "required": [
            "question_id", "investigation_status", "established_facts", "atomic_facts", "gap_reason",
            "requested_coarse_ids", "cited_evidence_ids",
        ],
    }


_MENTIONED_SHORT_ID_PATTERN = re.compile(r"\b[MC]\d{2,4}\b")


def _evidence_ids_mentioned_in_text(text: str, evidence: list[dict[str, Any]]) -> set[str]:
    # established_facts prose informally cites Medium/Coarse IDs inline (e.g. "chopping red pepper and
    # mushrooms (M009, M010)") separately from the structured cited_evidence_ids array. Real traffic
    # (70f2a750, 2026-08-07) showed this drifts badly: established_facts mentioned 23 distinct Medium
    # IDs but only 11 made it into cited_evidence_ids -- 12 were referenced in prose and then simply
    # never cited, M009 (the one detail that actually distinguished two answer options) among them.
    # Only a mentioned short id that matches a REAL evidence_id's trailing component counts as a
    # citation -- an incidental "M12"-looking substring with no matching evidence is not one.
    mentioned_short_ids = set(_MENTIONED_SHORT_ID_PATTERN.findall(text))
    return {
        row["evidence_id"] for row in evidence
        if row["evidence_id"].rsplit("::", 1)[-1] in mentioned_short_ids
    }


def _validate_shared_investigation(
    value: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]], excluded_coarse_ids: set[str],
) -> None:
    if value.get("question_id") != question["question_id"]:
        raise ValueError("Investigation question mismatch")
    evidence_ids = {row["evidence_id"] for row in evidence}
    if not set(value["requested_coarse_ids"]) <= excluded_coarse_ids:
        raise ValueError("Investigation requested an unknown/non-excluded Coarse region")
    if not set(value["cited_evidence_ids"]) <= evidence_ids:
        raise ValueError("Investigation cited unknown evidence")
    if value["investigation_status"] == "resolved":
        if not value["established_facts"].strip() or value["requested_coarse_ids"]:
            raise ValueError("resolved investigation must state established_facts and request nothing further")
        # Unlike the old per-option claim validator, there is no "this option's text is categorically
        # off-topic" escape hatch here: the shared investigation is about the question's own core fact,
        # not any one option's wording, so zero evidence can never legitimately support "resolved" --
        # this is exactly the false-refutation-from-silence bug confirmed on ab93e55b's option C.
        if not evidence:
            raise ValueError("resolved investigation requires at least the Medium-fallback evidence to exist")
        if not value["cited_evidence_ids"]:
            raise ValueError("resolved investigation must cite evidence")
        # Every evidence ID established_facts actually references inline must also be a formal
        # citation -- cited_evidence_ids must not be a silently-incomplete sample of what the prose
        # actually drew on (see _evidence_ids_mentioned_in_text).
        mentioned_evidence_ids = _evidence_ids_mentioned_in_text(value["established_facts"], evidence)
        if not mentioned_evidence_ids <= set(value["cited_evidence_ids"]):
            raise ValueError("established_facts references evidence not present in cited_evidence_ids")
        if _claim_text_contradicts_status("supported", value["established_facts"]):
            raise ValueError("established_facts argues the opposite of investigation_status=resolved")
        # atomic_facts is what the fact-verification gate (_verify_decisive_facts) checks claim-by-claim
        # against its own citations -- a resolved investigation with no breakdown gives that gate nothing
        # to check, silently defeating it.
        if not value["atomic_facts"]:
            raise ValueError("resolved investigation must break established_facts into atomic_facts")
        for fact in value["atomic_facts"]:
            if not fact["text"].strip():
                raise ValueError("atomic_facts entry must not have empty text")
            if not set(fact["evidence_ids"]) <= evidence_ids:
                raise ValueError("atomic_facts entry cited unknown evidence")
    elif not value["gap_reason"].strip():
        raise ValueError("unresolved investigation must state gap_reason")


def _downgrade_noncompliant_investigation(investigation: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    downgraded = copy.deepcopy(investigation)
    if downgraded["investigation_status"] != "resolved":
        return downgraded
    mentioned_evidence_ids = _evidence_ids_mentioned_in_text(downgraded["established_facts"], evidence)
    resolved_without_grounds = (
        not evidence or not downgraded["cited_evidence_ids"]
        or not mentioned_evidence_ids <= set(downgraded["cited_evidence_ids"])
        or not downgraded.get("atomic_facts")
    )
    # Real crash (9a7a189e r1_av, 2026-08-08, all-4-fixes 10-video run): this function only ever handled
    # the missing-grounds failures above -- when _validate_shared_investigation instead rejected a
    # resolved investigation for self-contradiction (established_facts arguing the opposite of its own
    # status), the downgrade left it untouched, _validate_shared_investigation raised the identical error
    # a second time, and it propagated uncaught and crashed run_live. Latent since V6.1's original
    # implementation, never triggered until this run. Same treatment as the other ungrounded cases.
    self_contradicting = _claim_text_contradicts_status("supported", downgraded["established_facts"])
    if resolved_without_grounds or self_contradicting:
        downgraded["investigation_status"] = "unresolved"
        downgraded["gap_reason"] = downgraded["gap_reason"].strip() or (
            "Model returned investigation_status=resolved without sufficient grounding evidence "
            "(no citations, established_facts referenced evidence it never formally cited, no "
            "atomic_facts breakdown, or established_facts argued the opposite of its own resolved "
            "status); downgraded to unresolved rather than accepting an ungrounded, under-cited, or "
            "self-contradictory conclusion."
        )
        downgraded["established_facts"] = ""
        downgraded["atomic_facts"] = []
    return downgraded


def _call_shared_investigation(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    excluded_judgments: list[dict[str, Any]], round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    excluded_coarse_ids = [j["coarse_id"] for j in excluded_judgments]
    payload = {
        "question": question, "evidence": evidence, "round": round_number,
        "excluded_coarse_regions": excluded_judgments,
    }
    schema = _shared_investigation_schema(question, evidence, excluded_coarse_ids)
    known_evidence_ids = {row["evidence_id"] for row in evidence}
    allowed_coarse = set(excluded_coarse_ids)
    attempts = int(cfg.get("max_validation_retries", 2))
    last_error: ValueError | RuntimeError | None = None
    investigation: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for _ in range(attempts):
        try:
            provider, usage = _anthropic_call(cfg, SHARED_INVESTIGATION_SYSTEM, payload, schema, int(cfg["anthropic"]["claim_max_tokens"]))
        except RuntimeError as error:
            last_error = error
            continue
        investigation = dict(provider)
        investigation["requested_coarse_ids"] = [cid for cid in investigation["requested_coarse_ids"] if cid in allowed_coarse]
        investigation["cited_evidence_ids"] = [eid for eid in investigation["cited_evidence_ids"] if eid in known_evidence_ids]
        investigation["atomic_facts"] = [
            {**fact, "evidence_ids": [eid for eid in fact["evidence_ids"] if eid in known_evidence_ids]}
            for fact in investigation["atomic_facts"]
        ]
        try:
            _validate_shared_investigation(investigation, question, evidence, allowed_coarse)
            return investigation, usage
        except ValueError as error:
            last_error = error
    if investigation is None:
        raise last_error
    downgraded = _downgrade_noncompliant_investigation(investigation, evidence)
    try:
        _validate_shared_investigation(downgraded, question, evidence, allowed_coarse)
        return downgraded, usage
    except ValueError:
        raise last_error


_STOPWORDS = {
    "this", "that", "with", "from", "have", "does", "option", "text", "assess", "whether", "correctly",
    "answers", "exact", "question", "camera", "wearer", "video", "there", "which", "about", "into", "than",
}


def _significant_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


def _is_decisive_fact(fact_text: str, requirements: list[dict[str, Any]]) -> bool:
    # Cheap, local, no-API: only worth verifying a fact if its wording actually touches vocabulary that
    # distinguishes between the answer options -- most established_facts content (scene-setting detail no
    # option's wording turns on) doesn't affect which option wins, and checking every fact against raw
    # evidence text would cost far more than the final decision needs.
    #
    # Real bug caught by a zero-API replay against the actual historical 824e7896 artifact (2026-08-08):
    # checking whether the fact's WHOLE word set intersects each option's word set is wrong -- a shared
    # word present in every option (here "room", from "Living Room" appearing in all 5 options) makes
    # every option's overlap non-empty, masking the one genuinely decisive word ("bedroom", present only
    # in option C) that was actually in the same fact. Must check per-word, not per-fact-as-a-whole: a
    # fact is decisive if ANY of its words individually appears in some but not all options.
    fact_words = _significant_words(fact_text)
    option_word_sets = [_significant_words(rp["description"]) for rp in requirements]
    for word in fact_words:
        touches = [word in words for words in option_word_sets]
        if any(touches) and not all(touches):
            return True
    return False


FACT_VERIFICATION_SYSTEM = """You are checking whether a single claimed fact about a video is directly
supported by the raw evidence text cited for it, or whether it asserts something more specific than that
text actually states. A paraphrase, synonym, or a more precise description of the exact same real-world
detail counts as supported (e.g. "takes off gloves" supports a claimed fact of "removes gloves"). A
specific label or category the cited text only vaguely permits but never actually states does NOT count
as supported (e.g. a caption saying only "a messy room" does not support a claimed fact of "Bedroom" --
that is a guess dressed up as a fact, not something the text actually says). Set entailed to true only if
the cited text plausibly describes the same real-world detail the claimed fact asserts; otherwise false.
State a short reason either way."""


def _fact_verification_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "entailed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["entailed", "reason"],
    }


def _call_fact_verification(cfg: dict[str, Any], fact_text: str, cited_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {"claimed_fact": fact_text, "cited_evidence_text": cited_text}
    schema = _fact_verification_schema()
    provider, usage = _anthropic_call(cfg, FACT_VERIFICATION_SYSTEM, payload, schema, int(cfg["anthropic"]["claim_max_tokens"]))
    return provider, usage


def _verify_decisive_facts(
    cfg: dict[str, Any], atomic_facts: list[dict[str, Any]], requirements: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Fix per user+codex design (2026-08-08): a shared-investigation fact can overclaim beyond what its
    # own cited evidence actually says (real case: "a messy room" cited as support for the specific claim
    # "Bedroom", which then wrongly refuted an answer option). This gate checks only DECISIVE facts (ones
    # that could actually change which option wins -- see _is_decisive_fact) against their own cited
    # evidence text: zero lexical overlap is treated as unsupported for free, no API call (safe direction
    # -- worst case an unusually-worded true paraphrase gets flagged uncertain, never a false claim waved
    # through); anything else gets one small, cheap, text-only verification call. established_facts
    # itself is never rewritten -- this produces a separate, auditable verdict per fact.
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    results: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for fact in atomic_facts:
        text = fact["text"]
        fact_evidence_ids = [eid for eid in fact["evidence_ids"] if eid in evidence_by_id]
        if not fact_evidence_ids or not _is_decisive_fact(text, requirements):
            results.append({
                "text": text, "evidence_ids": fact["evidence_ids"], "checked": False,
                "verified": True, "reason": "",
            })
            continue
        cited_text = " ".join(evidence_by_id[eid]["source_content"] for eid in fact_evidence_ids)
        score, _terms = lexical_similarity(text, cited_text)
        if score <= 0:
            results.append({
                "text": text, "evidence_ids": fact["evidence_ids"], "checked": True, "verified": False,
                "reason": "No lexical overlap between this claim and its cited evidence text (checked locally, no API call).",
            })
            continue
        verdict, usage = _call_fact_verification(cfg, text, cited_text)
        calls.append({"stage": "fact_verification", **usage})
        results.append({
            "text": text, "evidence_ids": fact["evidence_ids"], "checked": True,
            "verified": bool(verdict["entailed"]), "reason": verdict["reason"],
        })
    return results, calls


def _medium_lexical_fallback(
    side: str, question: dict[str, Any], query_variants: list[str], hierarchy: dict[str, Any],
    projections: list[dict[str, Any]], captions: list[dict[str, Any]], top_k: int = 8,
) -> list[dict[str, Any]]:
    # Fix I (real forensic finding on ab93e55b, 2026-08-07): a fine-grained action can be genuinely
    # absent from every Coarse region's compressed summary. That must be CHECKED, not assumed, before
    # concluding "not found" from the planner's own Coarse-level exclusion reasoning alone -- so when no
    # option locked any Coarse region at all, fall back to a full-video lexical scan over the same
    # Medium-level captions/ASR channel _lexical_rows already exposes for retrieval ranking, bypassing
    # Coarse boundaries entirely. No API cost -- pure lexical scoring, reused from existing retrieval code.
    lexical = _lexical_rows(side, projections, captions)
    query_text = " ".join(query_variants) if query_variants else question["question_text"]
    scored = []
    for medium, lex in zip(hierarchy["medium_nodes"], lexical):
        score, terms = lexical_similarity(query_text, lex["lexical_text"])
        if score > 0:
            scored.append((score, terms, medium, lex))
    scored.sort(key=lambda row: -row[0])
    shared_id = f"{question['question_id']}::shared"
    evidence = []
    for score, terms, medium, lex in scored[:top_k]:
        evidence.append({
            "evidence_id": f"medium_lexical_fallback::{question['question_id']}::{medium['medium_id']}",
            "evidence_type": "medium_lexical_fallback", "source_content": lex["lexical_text"],
            "interval": [medium["start_sec"], medium["end_sec"]], "lexical_score": score, "matched_terms": terms,
            "retrieved_for_requirement_id": shared_id,
        })
    return evidence


_EXCLUDED_REGION_SUMMARY_MAX_CHARS = 240


def _excluded_judgments_union(map_doc: dict[str, Any], locked: set[str]) -> list[dict[str, Any]]:
    # V6.3: excluded regions are now a pure set difference (every Coarse region in the map that no
    # option's sparse selection currently locks), and the context handed to the shared investigator for
    # each one is the Coarse map's OWN navigation_summary -- not a planner-authored "why I excluded this"
    # reason (the sparse planner no longer produces one; see _coarse_locked_planner_schema for why this
    # is safe to drop). navigation_summary is clipped defensively: the map's organizer prompt already
    # asks for a concise per-region summary, but nothing enforces a hard length ceiling, and this text
    # gets repeated once per excluded region in every shared_investigation call.
    excluded = []
    for coarse in map_doc["coarse_regions"]:
        if coarse["coarse_id"] in locked:
            continue
        excluded.append({
            "coarse_id": coarse["coarse_id"], "interval": [coarse["start_sec"], coarse["end_sec"]],
            "navigation_summary": coarse["navigation_summary"][:_EXCLUDED_REGION_SUMMARY_MAX_CHARS],
        })
    return sorted(excluded, key=lambda row: row["coarse_id"])


def _shared_search_plan(plan: dict[str, Any], shared_id: str) -> dict[str, Any]:
    descriptions = [rp["search_description"] for rp in plan["requirement_plans"]]
    variants = [v for rp in plan["requirement_plans"] for v in rp["query_variants"]]
    return {
        "requirement_id": shared_id,
        "search_description": " ".join(dict.fromkeys(descriptions)),
        "query_variants": list(dict.fromkeys(variants)),
        "modality_strategy": plan["requirement_plans"][0]["modality_strategy"],
        "selected_coarse_ids": [],
    }


OPTION_MAPPING_SYSTEM = SUFFICIENCY_SYSTEM + """
You are mapping a set of already-established, individually-verified facts about a video onto a fixed set
of mutually exclusive answer options. verified_facts is the ONLY evidence you have access to -- you do not
see the full video evidence pool and must not guess at or invent details beyond what verified_facts states;
if verified_facts does not address a detail some option's wording depends on, that is simply not something
you can determine, not something you may infer from general plausibility. For each option, set status to
"supported" (verified_facts establishes this option as the correct answer), "refuted" (verified_facts
directly and positively contradicts this option), or "unresolved" (neither is established from
verified_facts). Silence is not contradiction: if verified_facts simply never addresses a specific detail
an option's wording depends on, that is "unresolved", never "refuted" -- refuted requires verified_facts to
affirmatively show something incompatible with the option, not merely fail to mention it. In the ordinary
case exactly one option should end up "supported", since these are mutually exclusive answers to the same
question -- but if verified_facts genuinely does not distinguish between two or more options, it is correct
and expected to mark more than one "unresolved" or even "supported" rather than force an arbitrary pick;
do not fabricate a distinguishing detail verified_facts does not actually contain. For every option you
mark "supported" or "refuted", list in used_facts the exact text of every verified_facts entry your
conclusion actually rests on (copy the text verbatim) -- this becomes the option's own citation trail, so
it must be complete and must not include a fact irrelevant to that option's conclusion. State a short
reason for every option, including ones you mark unresolved, and make sure the reason argues for that
exact status, not its opposite.
"""

# Real bug (70f2a750, 2026-08-07): mapping refuted option E with the reason "...these are not explicitly
# characterized as 'waste disposal' in the established facts, making this addition unsupported by the
# evidence" -- treating "my summary didn't mention it" as "the evidence contradicts it". Same
# absence-is-not-evidence-of-absence principle already enforced for the shared investigation itself
# (_validate_shared_investigation's zero-evidence guard), extended here as a phrase-level heuristic guard
# since "refuted needs positive contradiction" isn't otherwise mechanically checkable.
_CLOSED_WORLD_REFUTATION_PHRASES = (
    "not explicitly", "not mentioned", "no evidence of", "not documented", "not described",
    "not stated", "no mention of", "does not mention", "not addressed",
)


def _reason_is_closed_world_refutation(status: str, reason: str) -> bool:
    if status != "refuted":
        return False
    lowered = reason.lower()
    return any(phrase in lowered for phrase in _CLOSED_WORLD_REFUTATION_PHRASES)


def _option_mapping_schema(requirements: list[dict[str, Any]], fact_texts: list[str]) -> dict[str, Any]:
    req_ids = [row["requirement_id"] for row in requirements]
    # Same empty-enum limitation as elsewhere in this module: falls back to an unconstrained array (only
    # possible if verified_facts is somehow empty, which _resolve_requirements already short-circuits
    # around before ever calling this) and clips any hallucinated fact text after the call.
    used_facts_field = (
        {"type": "array", "items": {"type": "string", "enum": fact_texts}}
        if fact_texts else {"type": "array", "items": {"type": "string"}}
    )
    item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "requirement_id": {"type": "string", "enum": req_ids},
            "status": {"type": "string", "enum": list(OPTION_STATUS_VALUES)},
            "used_facts": used_facts_field,
            "reason": {"type": "string"},
        },
        "required": ["requirement_id", "status", "used_facts", "reason"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "option_assessments": {"type": "array", "minItems": len(req_ids), "maxItems": len(req_ids), "items": item},
        },
        "required": ["question_id", "option_assessments"],
    }


def _validate_option_mapping(
    value: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], fact_texts: list[str],
) -> None:
    req_ids = [row["requirement_id"] for row in requirements]
    known_facts = set(fact_texts)
    if value.get("question_id") != question["question_id"]:
        raise ValueError("Option mapping question mismatch")
    seen = [row.get("requirement_id") for row in value.get("option_assessments", [])]
    if sorted(seen) != sorted(req_ids):
        raise ValueError("Option mapping coverage/uniqueness invalid")
    for row in value["option_assessments"]:
        if not row["reason"].strip():
            raise ValueError(f"Option mapping reason must not be empty: {row['requirement_id']}")
        if not set(row["used_facts"]) <= known_facts:
            raise ValueError(f"Option mapping used_facts references an unknown fact: {row['requirement_id']}")
        if _claim_text_contradicts_status(row["status"], row["reason"]):
            raise ValueError(f"Option mapping reason argues the opposite of its own status: {row['requirement_id']}")
        if _reason_is_closed_world_refutation(row["status"], row["reason"]):
            raise ValueError(f"Option mapping refuted from silence, not positive contradiction: {row['requirement_id']}")
        # A citation trail is exactly the point of used_facts (see codex's correction, 2026-08-08): a
        # supported/refuted verdict with nothing in used_facts would let a conclusion stand with no
        # traceable basis, silently reopening the original ungrounded-claim failure this whole citation
        # discipline exists to close.
        if row["status"] in ("supported", "refuted") and not row["used_facts"]:
            raise ValueError(f"supported/refuted mapping must cite used_facts: {row['requirement_id']}")


def _repair_option_mapping_coverage(
    value: dict[str, Any], requirements: list[dict[str, Any]], fact_texts: list[str],
) -> dict[str, Any]:
    known_facts = set(fact_texts)
    repaired = copy.deepcopy(value)
    by_rid: dict[str, dict[str, Any]] = {}
    for row in repaired.get("option_assessments", []):
        by_rid.setdefault(row.get("requirement_id"), row)
    fixed = []
    for rp in requirements:
        rid = rp["requirement_id"]
        row = by_rid.get(rid) or {"requirement_id": rid, "status": "unresolved", "used_facts": [], "reason": ""}
        row.setdefault("used_facts", [])
        row["used_facts"] = [f for f in row["used_facts"] if f in known_facts]
        noncompliant = (
            not row.get("reason", "").strip()
            or _claim_text_contradicts_status(row["status"], row.get("reason", ""))
            or _reason_is_closed_world_refutation(row["status"], row.get("reason", ""))
            or (row["status"] in ("supported", "refuted") and not row["used_facts"])
        )
        if noncompliant:
            row = {
                **row, "status": "unresolved", "used_facts": [],
                "reason": "Model did not return a compliant mapping for this option; treated as unresolved.",
            }
        fixed.append(row)
    repaired["option_assessments"] = fixed
    return repaired


def _call_option_mapping(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    verified_facts: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Deliberately ONLY verified_facts -- no established_facts, no full evidence pool (codex's correction,
    # 2026-08-08): giving mapping raw evidence access let it independently re-derive an unverified reading
    # on its own (confirmed real case: mapping read the same "messy room" caption directly and could, in
    # principle, re-invent "Bedroom" itself, bypassing the whole fact-verification gate). Constraining
    # mapping's entire knowledge to the already-verified fact set closes that side door; if verified_facts
    # can't decide something, that is unresolved, not something to go looking for in raw evidence.
    fact_texts = [f["text"] for f in verified_facts]
    payload = {"question": question, "requirements": requirements, "verified_facts": verified_facts}
    schema = _option_mapping_schema(requirements, fact_texts)
    attempts = int(cfg.get("max_validation_retries", 2))
    last_error: ValueError | RuntimeError | None = None
    mapping: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for _ in range(attempts):
        try:
            provider, usage = _anthropic_call(cfg, OPTION_MAPPING_SYSTEM, payload, schema, int(cfg["anthropic"]["claim_max_tokens"]))
        except RuntimeError as error:
            last_error = error
            continue
        mapping = provider
        try:
            _validate_option_mapping(mapping, question, requirements, fact_texts)
            return mapping, usage
        except ValueError as error:
            last_error = error
    if mapping is None:
        raise last_error
    repaired = _repair_option_mapping_coverage(mapping, requirements, fact_texts)
    try:
        _validate_option_mapping(repaired, question, requirements, fact_texts)
        return repaired, usage
    except ValueError:
        raise last_error


BATCH_CLAIM_EXECUTION_SYSTEM = """You are a claim-execution visual reviewer investigating several
answer-option requirements at once against one shared, de-duplicated set of images retrieved from
the Coarse region(s) currently locked for each requirement -- the same image may be relevant to more
than one requirement, and it is shown only once. Inspect every supplied image once and report a
neutral observation per image, usable by any requirement. Then, for each requirement's specific
claim or gap listed in claims_or_gaps_to_investigate, assess it directly using only the supplied
images: confirmed (the images establish it), refuted (the images contradict it), or inconclusive
(the images neither establish nor contradict it). Assess each requirement independently -- do not
let one requirement's claim bias another's assessment -- and do not select a final answer or infer
anything beyond what is directly visible."""


def _batch_claim_execution_schema(fine_ids: list[str], requirement_ids: list[str]) -> dict[str, Any]:
    observation_item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "fine_id": {"type": "string", "enum": fine_ids},
            "finding": {"type": "string"},
            "visible_actions": {"type": "array", "items": {"type": "string"}},
            "visible_objects": {"type": "array", "items": {"type": "string"}},
            "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["fine_id", "finding", "visible_actions", "visible_objects", "uncertainty_notes"],
    }
    assessment_unit = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "claim_status": {"type": "string", "enum": ["confirmed", "refuted", "inconclusive"]},
            "supporting_fine_ids": {"type": "array", "items": {"type": "string", "enum": fine_ids}},
            "rationale": {"type": "string"},
        },
        "required": ["claim_status", "supporting_fine_ids", "rationale"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "observations": {
                "type": "array", "minItems": len(fine_ids), "maxItems": len(fine_ids), "items": observation_item,
            },
            "claim_assessments": {
                "type": "object", "additionalProperties": False,
                "properties": {rid: assessment_unit for rid in requirement_ids}, "required": requirement_ids,
            },
        },
        "required": ["observations", "claim_assessments"],
    }


def _validate_batch_claim_execution(value: dict[str, Any], fine_ids: list[str], requirement_ids: list[str]) -> None:
    observed = [row.get("fine_id") for row in value.get("observations", [])]
    if observed != fine_ids or len(observed) != len(set(observed)):
        raise ValueError("Batch claim execution Fine coverage/order mismatch")
    if set(value.get("claim_assessments", {})) != set(requirement_ids):
        raise ValueError("Batch claim execution requirement coverage invalid")
    known = set(fine_ids)
    for rid, assessment in value["claim_assessments"].items():
        if set(assessment["supporting_fine_ids"]) - known:
            raise ValueError(f"Batch claim execution cited unknown Fine: {rid}")
        if assessment["claim_status"] == "confirmed" and not assessment["supporting_fine_ids"]:
            raise ValueError(f"confirmed batch claim execution lacks supporting Fine evidence: {rid}")


def _dedupe_fine_rows(fine_rows_by_requirement: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    # Mirrors V3's question_batched_visual_review dedup (_dedupe_fines): the same Fine frame can be a
    # retrieval candidate for more than one still-open requirement in a round, and should only be
    # transmitted to Gemini once. V6's original per-requirement loop paid for it every time.
    by_hash: dict[str, dict[str, Any]] = {}
    targets: dict[str, list[str]] = {}
    for rid, rows in fine_rows_by_requirement.items():
        for row in rows:
            digest = sha256_file(Path(row["source_frame_path"]))
            if digest not in by_hash:
                by_hash[digest] = {**row, "image_sha256": digest}
            targets.setdefault(digest, []).append(rid)
    ordered = sorted(by_hash.values(), key=lambda row: (float(row["timestamp_sec"]), row["fine_id"]))
    return ordered, {digest: list(dict.fromkeys(ids)) for digest, ids in targets.items()}


def _allocate_fine_quota_by_requirement(
    unique_fines: list[dict[str, Any]], targets: dict[str, list[str]], cap: int,
) -> list[dict[str, Any]]:
    # The per-requirement cap (_allocate_fine_quota_by_coarse) bounds each requirement's OWN candidate
    # pool to `max_total_fine_evidence_per_claim`, but batching several requirements together for
    # cross-option dedup means the batch's total unique Fine count can still exceed a single request's
    # safe size even when every individual requirement stayed under its own cap (confirmed on real
    # traffic: 4 requirements each within their own limit still deduped to 29 unique images in one
    # batch, and that batch got HTTP 400 "invalid argument" -- the failure threshold depends on total
    # payload size, not image count alone, so even a count below an earlier-observed working count can
    # still fail). Same per-group-quota-then-leftover pattern as the per-Coarse cap, grouped by whichever
    # requirement each unique image primarily serves.
    if len(unique_fines) <= cap:
        return unique_fines
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for row in unique_fines:
        by_rid.setdefault(targets[row["image_sha256"]][0], []).append(row)
    base_quota = max(1, cap // len(by_rid))
    selected: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for rows in by_rid.values():
        selected.extend(rows[:base_quota])
        leftover.extend(rows[base_quota:])
    leftover.sort(key=lambda row: row["medium_rank"])
    slots_left = max(0, cap - len(selected))
    selected.extend(leftover[:slots_left])
    keep = {row["image_sha256"] for row in selected}
    return [row for row in unique_fines if row["image_sha256"] in keep]


def _execute_claims_batch(
    cfg: dict[str, Any], question: dict[str, Any], claims_or_gaps_by_requirement: dict[str, str],
    unique_fines: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fine_ids = [row["fine_id"] for row in unique_fines]
    requirement_ids = list(claims_or_gaps_by_requirement)
    payload = {
        "question_id": question["question_id"],
        "claims_or_gaps_to_investigate": claims_or_gaps_by_requirement,
        "ordered_images": [{"fine_id": row["fine_id"], "timestamp_sec": row["timestamp_sec"]} for row in unique_fines],
    }
    inputs: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    for row in unique_fines:
        image = Path(row["source_frame_path"])
        inputs.extend([
            {"type": "text", "text": f"FINE {row['fine_id']} timestamp={float(row['timestamp_sec']):.3f}s"},
            {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image.read_bytes()).decode("ascii")},
        ])
    schema = _batch_claim_execution_schema(fine_ids, requirement_ids)
    attempts = int(cfg.get("max_validation_retries", 2))
    last_error: ValueError | None = None
    for _ in range(attempts):
        result, usage, raw = _gemini_call({"gemini": cfg["gemini"]}, BATCH_CLAIM_EXECUTION_SYSTEM, inputs, schema)
        try:
            _validate_batch_claim_execution(result, fine_ids, requirement_ids)
            return result, usage, raw
        except ValueError as error:
            last_error = error
    raise last_error


def _map_level_evidence(
    side: str, requirement_id: str, locked_coarse_ids: set[str], map_doc: dict[str, Any],
    hierarchy: dict[str, Any], projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    parent: dict[str, str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    coarse_by_id = {row["coarse_id"]: row for row in map_doc["coarse_regions"]}
    for coarse_id in sorted(locked_coarse_ids):
        coarse = coarse_by_id[coarse_id]
        evidence.append({
            "evidence_id": f"semantic_coarse::{requirement_id}::{coarse_id}", "evidence_type": "semantic_coarse_summary",
            "source_content": coarse["navigation_summary"], "interval": [coarse["start_sec"], coarse["end_sec"]],
            "retrieved_for_requirement_id": requirement_id,
        })
    lexical = _lexical_rows(side, projections, captions)
    for medium, lex in zip(hierarchy["medium_nodes"], lexical):
        if parent[medium["medium_id"]] not in locked_coarse_ids:
            continue
        evidence.append({
            "evidence_id": f"medium_detail::{requirement_id}::{medium['medium_id']}", "evidence_type": lex["lexical_source"],
            "source_content": lex["lexical_text"], "interval": [medium["start_sec"], medium["end_sec"]],
            "retrieved_for_requirement_id": requirement_id,
        })
    ids = [row["evidence_id"] for row in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence ID")
    return evidence


def _batch_assessment_as_evidence(requirement_id: str, round_number: int, assessment: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": f"claim_execution::{requirement_id}::round{round_number}",
        "evidence_type": "claim_execution_result",
        "source_content": assessment["rationale"],
        "claim_status": assessment["claim_status"],
        "supporting_fine_ids": assessment["supporting_fine_ids"],
        "retrieved_for_requirement_id": requirement_id,
    }


def _resolve_requirements(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], plan: dict[str, Any],
    side: str, map_doc: dict[str, Any], hierarchy: dict[str, Any], projections: list[dict[str, Any]],
    captions: list[dict[str, Any]], medium_embeddings: np.ndarray, fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: np.ndarray, row_by_id: dict[str, int], encoder: Any, parent: dict[str, str], max_rounds: int,
) -> tuple[
    dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]],
]:
    # V6.1: one shared investigation establishes the question's own underlying fact once, using the
    # UNION of every option's own Coarse locks, instead of five independent per-option investigations
    # that could each honestly ground a different, mutually exclusive option. A single option-mapping
    # call at the end projects the one established fact onto every option's status. See the module-level
    # comment above SHARED_INVESTIGATION_SYSTEM for the two real bugs this design targets.
    calls: list[dict[str, Any]] = []
    shared_id = f"{question['question_id']}::shared"

    locked_by_rid = {rp["requirement_id"]: _locked_coarse_ids(rp) for rp in plan["requirement_plans"]}
    locked: set[str] = set().union(*locked_by_rid.values()) if locked_by_rid else set()

    fallback_evidence: list[dict[str, Any]] = []
    if not locked:
        query_variants = [v for rp in plan["requirement_plans"] for v in rp["query_variants"]]
        fallback_evidence = _medium_lexical_fallback(side, question, query_variants, hierarchy, projections, captions)

    evidence: list[dict[str, Any]] = _map_level_evidence(
        side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
    ) + fallback_evidence
    image_findings: dict[str, dict[str, Any]] = {}

    def investigation_call(round_number: int, stage: str = "shared_investigation") -> dict[str, Any]:
        excluded = _excluded_judgments_union(map_doc, locked)
        doc, usage = _call_shared_investigation(cfg, question, evidence, excluded, round_number=round_number)
        calls.append({"stage": stage, "round": round_number, "requirement_id": shared_id, **usage})
        return doc

    investigation = investigation_call(round_number=0)

    for round_number in range(1, max_rounds + 1):
        if investigation["investigation_status"] != "unresolved":
            break
        if round_number > 1 and not investigation["requested_coarse_ids"]:
            # Stalled: no new Coarse requested, so retrieval would see the exact same locked set and
            # the exact same Fine images as last round -- re-executing burns budget for no new
            # information (same rationale as the pre-V6.1 per-option stall detection).
            break

        if investigation["requested_coarse_ids"]:
            # Fix C2: give the newly-unlocked region's map text a free look before paying for images.
            locked = locked | set(investigation["requested_coarse_ids"])
            evidence = _map_level_evidence(
                side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
            ) + fallback_evidence
            investigation = investigation_call(round_number, stage="shared_investigation_recheck")
            if investigation["investigation_status"] != "unresolved":
                break

        if not locked:
            # Nothing to retrieve images for -- the Medium fallback already ran once above and found
            # nothing worth escalating from; further rounds cannot add information.
            break

        retrieval = _coarse_scoped_retrieval(
            cfg, side, question, {"requirement_plans": [_shared_search_plan(plan, shared_id)]},
            map_doc, hierarchy, projections, captions, medium_embeddings, fine_by_id, fine_embeddings, row_by_id,
            encoder, {shared_id: locked},
        )
        fine_rows = retrieval["requirements"][shared_id]["selected_fine_evidence"]

        # Fix C3: reuse an already-seen image's finding as text instead of paying to re-transmit it.
        for row in fine_rows:
            if row["fine_id"] in image_findings:
                evidence = evidence + [{
                    "evidence_id": f"cached_observation::{shared_id}::{row['fine_id']}::round{round_number}",
                    "evidence_type": "cached_visual_observation",
                    "source_content": image_findings[row["fine_id"]]["finding"],
                    "retrieved_for_requirement_id": shared_id,
                }]
        fresh_rows = [row for row in fine_rows if row["fine_id"] not in image_findings]

        if fresh_rows:
            unique_fines, targets = _dedupe_fine_rows({shared_id: fresh_rows})
            batch_cap = int(cfg["ranking"].get("max_total_fine_evidence_per_batch", 16))
            unique_fines = _allocate_fine_quota_by_requirement(unique_fines, targets, batch_cap)
            claim_or_gap = investigation["established_facts"] or investigation["gap_reason"]
            execution, exec_usage, exec_raw = _execute_claims_batch(cfg, question, {shared_id: claim_or_gap}, unique_fines)
            calls.append({
                "stage": "claim_execution_batch", "round": round_number, "requirement_ids": [shared_id],
                "image_transmissions": len(unique_fines), **exec_usage,
            })
            for observation in execution["observations"]:
                image_findings[observation["fine_id"]] = observation
            assessment = execution["claim_assessments"][shared_id]
            evidence = evidence + [_batch_assessment_as_evidence(shared_id, round_number, assessment)]
        elif not fine_rows:
            evidence = evidence + [{
                "evidence_id": f"claim_execution::{shared_id}::round{round_number}::empty",
                "evidence_type": "claim_execution_result",
                "source_content": "No Fine evidence available in the currently locked Coarse set.",
                "claim_status": "inconclusive", "supporting_fine_ids": [], "retrieved_for_requirement_id": shared_id,
            }]
        # else: every candidate Fine was already cached above -- no fresh transmission or placeholder needed.

        investigation = investigation_call(round_number)

    fact_verification: list[dict[str, Any]] = []
    if investigation["investigation_status"] == "resolved":
        fact_verification, verification_calls = _verify_decisive_facts(
            cfg, investigation.get("atomic_facts", []), requirements, evidence,
        )
        calls.extend(verification_calls)
        unverified_texts = {row["text"] for row in fact_verification if row["checked"] and not row["verified"]}
        verified_facts = [f for f in investigation.get("atomic_facts", []) if f["text"] not in unverified_texts]

        if not verified_facts:
            # Every atomic fact that could have decided anything failed verification -- nothing
            # trustworthy left to hand to mapping, so don't spend a call asking it to guess from an
            # empty basis. Same deterministic treatment as an unresolved investigation.
            claims = {
                rp["requirement_id"]: {
                    "requirement_id": rp["requirement_id"], "option_status": "unresolved", "claim_text": "",
                    "gap_reason": (
                        "Every established fact that could have decided an option failed independent "
                        "verification against its own cited evidence."
                    ),
                    "cited_evidence_ids": [],
                }
                for rp in requirements
            }
        else:
            mapping, mapping_usage = _call_option_mapping(cfg, question, requirements, verified_facts)
            calls.append({"stage": "option_mapping", "requirement_id": shared_id, **mapping_usage})
            mapping_by_rid = {row["requirement_id"]: row for row in mapping["option_assessments"]}
            # Final citations are built FROM the facts a claim actually used, not by taking the
            # investigation's whole citation list and subtracting unverified ones (codex's correction,
            # 2026-08-08): the same evidence_id can legitimately back both a verified and an unverified
            # claim (M017 doesn't prove "Bedroom" but still proves "organizing a desk in a messy room"),
            # so subtractive filtering would have discarded a still-valid citation.
            fact_evidence_by_text = {f["text"]: f["evidence_ids"] for f in verified_facts}
            claims = {}
            for rp in requirements:
                rid = rp["requirement_id"]
                row = mapping_by_rid[rid]
                status = row["status"]
                used_texts = row.get("used_facts", [])
                used_evidence_ids = sorted({eid for text in used_texts for eid in fact_evidence_by_text.get(text, [])})
                claims[rid] = {
                    "requirement_id": rid, "option_status": status,
                    "claim_text": (
                        f"{' '.join(used_texts)} Applied to this option: {row['reason']}"
                        if status in ("supported", "refuted") else ""
                    ),
                    "gap_reason": row["reason"] if status == "unresolved" else "",
                    "cited_evidence_ids": used_evidence_ids if status in ("supported", "refuted") else [],
                }
    else:
        # Deterministic, no LLM call: an unresolved shared investigation gives no grounds to assert any
        # option's status beyond "we could not determine this" -- asking a mapping call to guess from
        # admittedly-insufficient facts would just reintroduce an ungrounded per-option guess.
        claims = {
            rp["requirement_id"]: {
                "requirement_id": rp["requirement_id"], "option_status": "unresolved", "claim_text": "",
                "gap_reason": investigation["gap_reason"], "cited_evidence_ids": [],
            }
            for rp in requirements
        }

    evidence_by_rid = {rp["requirement_id"]: evidence for rp in requirements}
    return claims, evidence_by_rid, calls, investigation, fact_verification


COARSE_LOCKED_PLANNER_PROMPT = """You are a Coarse-region locking planner for a video question. For each
answer-option requirement, give a search_description, query_variants, and modality_strategy exactly as a
retrieval planner would. From the supplied navigation map, choose selected_coarse_ids: only the Coarse
regions that are actually plausibly relevant to checking this requirement -- do not list a region you are
excluding, and do not explain why every other region was left out; you are naming what to search, not
auditing the whole map. A recurring or interruptible activity may appear in more than one Coarse region
across the timeline; if a requirement concerns such an activity, select every Coarse region where it
plausibly recurs, not only the most recent or most salient one -- missing a real occurrence is a more
costly mistake than including one extra region. It is correct to select zero Coarse regions for a
requirement when the option's own text does not address what the question actually asks (for example, a
"which happened first, A or B" question where this option names neither A nor B) -- say so plainly in
selection_reason; do not force a selection just to have one. A region's summary describing an action in
different words than the option's text is not, by itself, grounds for leaving it out (for example "running
on the spot" / "jogging in place" / "high knees" could all be describing the same real-world action as an
option worded "Sprinting Drills"). Judge whether it is plausibly the same action, not whether the exact
phrase matches. selection_reason is one short statement covering this requirement's selection as a whole
(what you searched for and, in brief, why these regions and not others) -- never leave it empty."""


BUDGET_EXHAUSTED_CAVEAT_MARKER = "map_based_guess_budget_exhausted"
# Real traffic (10-video pilot, 2026-08-07) showed 7 of 13 "confirmed" finals had MORE THAN ONE option
# simultaneously marked option_status=="supported" in the same side's resolutions -- independent
# per-option investigation can honestly ground two mutually exclusive options at once (each sees its own
# evidence pool and never learns the other is also grounded), and nothing previously stopped _final from
# picking one of them and calling it "confirmed". A confirmed answer must be the UNIQUE supported option,
# not just "a" supported option -- ambiguity between multiple grounded options is a genuinely different,
# more honest failure mode than "nothing was grounded," so it gets its own marker string.
AMBIGUOUS_SUPPORT_CAVEAT_MARKER = "multiple_options_simultaneously_supported"

# Maps SFC's own three-way option_status onto the status vocabulary _final/_final_schema (reused
# unchanged from v1) already expects, so the free-text rationale and the structured label agree:
# "refuted" (a confirmed rejection) must never collapse into "supported" the way a bare resolved:bool
# did, which fed _final a status/rationale pair that flatly contradicted each other on real traffic.
_OPTION_STATUS_TO_ASSESSMENT_STATUS = {"supported": "supported", "refuted": "not_found", "unresolved": "uncertain"}


def _claim_to_assessment(claim: dict[str, Any]) -> dict[str, Any]:
    status = _OPTION_STATUS_TO_ASSESSMENT_STATUS[claim["option_status"]]
    if claim["option_status"] == "unresolved":
        rationale = "Unresolved after the claim/audit loop budget: " + claim["gap_reason"]
    else:
        rationale = claim["claim_text"]
    return {
        "requirement_id": claim["requirement_id"], "status": status,
        "direct_support": claim["option_status"] == "supported",
        "supporting_evidence_ids": claim["cited_evidence_ids"], "rationale": rationale,
    }


def _finalize_answer(
    answer: dict[str, Any], question: dict[str, Any], per_requirement_claim: dict[str, Any],
) -> dict[str, Any]:
    selected_rid = f"{question['question_id']}::option_{answer['selected_option_id'].lower()}"
    selected_claim = per_requirement_claim.get(selected_rid)
    if selected_claim is None:
        return {**answer, "final_status": "unknown"}

    # Ported from V3's original guard, which V6 never carried over: _final is an LLM call and can
    # still pick an option its own claim just refuted if a genuinely supported alternative exists.
    if selected_claim["option_status"] == "refuted":
        better_exists = any(
            c["option_status"] == "supported" for rid, c in per_requirement_claim.items() if rid != selected_rid
        )
        if better_exists:
            raise ValueError("Final selected a refuted option while a supported option exists")

    # _final is reused unchanged from v1 and has no citation requirement of its own (unlike
    # _validate_claim's rule for individual options, added after real traffic hit a bare, uncited
    # "supported" claim). Confirmed on 819c8af7: all 5 options ended up refuted, and _final still had
    # to pick one -- if it had labeled that pick "confirmed" without citing anything, the same
    # ungrounded-claim failure mode this whole design exists to close would resurface one layer up.
    other_supported_rids = [
        rid for rid, claim in per_requirement_claim.items()
        if rid != selected_rid and claim["option_status"] == "supported"
    ]
    if selected_claim["option_status"] == "supported" and answer["supporting_evidence_ids"] and not other_supported_rids:
        return {**answer, "final_status": "confirmed"}

    if selected_claim["option_status"] == "supported" and answer["supporting_evidence_ids"] and other_supported_rids:
        # The selected option is itself properly grounded, but it is not the ONLY grounded option --
        # per-option investigation independently supported multiple mutually exclusive options (real
        # traffic: 7 of 13 confirmed finals this way, e.g. a workout question where both "Squats" and
        # "Sprinting Drills" ended up supported). Picking one and calling it "confirmed" would assert an
        # exclusivity the investigation never actually established -- this is honestly a tie, not a win.
        if not any(AMBIGUOUS_SUPPORT_CAVEAT_MARKER in row for row in answer["caveats"]):
            other_labels = ", ".join(rid.split("::option_")[-1].upper() for rid in sorted(other_supported_rids))
            forced = (
                f"{AMBIGUOUS_SUPPORT_CAVEAT_MARKER}: option(s) {other_labels} were also independently "
                "marked option_status=supported; the investigation never established that the selected "
                "option is uniquely correct, so this is an unconfirmed best guess among multiple "
                "grounded candidates, not a confidently verified answer."
            )
            answer = {**answer, "caveats": [*answer["caveats"], forced]}
        return {**answer, "final_status": "budget_exhausted_guess"}

    # Either option_status isn't "supported" (refuted with no better alternative, or unresolved), or
    # it is "supported" but _final cited nothing: either way this is a forced best guess, not a
    # confirmed answer. HourVideo is scored multiple-choice, so the answer is still submitted -- only
    # the honesty of the caveat is enforced here, not the choice itself.
    if not any(BUDGET_EXHAUSTED_CAVEAT_MARKER in row for row in answer["caveats"]):
        forced = (
            f"{BUDGET_EXHAUSTED_CAVEAT_MARKER}: the selected option's underlying claim did not reach "
            "option_status=supported within the claim/audit loop budget; this is an unconfirmed "
            "map/evidence-based best guess, not a confidently verified answer."
        )
        answer = {**answer, "caveats": [*answer["caveats"], forced]}
    return {**answer, "final_status": "budget_exhausted_guess"}


def _case_paths(cfg: dict[str, Any], root: Path, video_uid: str | None) -> list[tuple[dict[str, Any], Path]]:
    source = root / cfg["source_experiment"]
    cases_root = source / "cases"
    rows = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        uid = case_dir.name
        if video_uid is not None and uid != video_uid:
            continue
        if not (case_dir / "question_input.json").is_file():
            continue
        case_cfg = load_json(source / "case_configs" / f"{uid}.json")
        rows.append((case_cfg, case_dir))
    return rows


def _offline_audit(case_cfg: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    required = [
        "question_input.json", "shared_hierarchy.json", "r1_medium_projection.json", "r3_medium_captions.json",
        "medium_siglip.float32.npy", "audio_asr.json", "r1_av_navigation_map.json", "r3_2_navigation_map.json",
    ]
    missing = [name for name in required if not (case_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{case_cfg['video_uid']}: missing upstream artifacts {missing}")
    question = load_json(case_dir / "question_input.json")
    return {"video_uid": case_cfg["video_uid"], "question_id": question["question_id"], "option_requirement_count": len(option_requirements(question))}


def preflight(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path); output = root / cfg["output_root"]; output.mkdir(parents=True, exist_ok=True)
    cases = _case_paths(cfg, root, video_uid)
    if not cases:
        raise RuntimeError("no matching cases found under source_experiment")
    audits = [_offline_audit(case_cfg, case_dir) for case_cfg, case_dir in cases]
    option_count = sum(row["option_requirement_count"] for row in audits)
    result = {
        "overall_validation": "ready_for_live_pilot", "video_count": len(audits), "question_count": len(audits),
        "total_option_requirements": option_count,
        "budget_formula": "coarse_scoped_upper_bound_not_yet_tight",
        "fixed_calls": {"planner": len(audits) * 2, "sfc_claim_round0": option_count * 2},
        "worst_case_sfc_claim_calls": option_count * 2 * (int(cfg["max_claim_rounds"]) + 1),
        "worst_case_claim_execution_calls": option_count * 2 * int(cfg["max_claim_rounds"]),
        "max_claim_rounds": int(cfg["max_claim_rounds"]),
        "audits": audits,
    }
    write_json(output / "live_preflight.json", result)
    return result


def _write_run_manifest(root: Path, cfg: dict[str, Any], config_path: Path, video_uid: str | None) -> dict[str, Any]:
    # We iterated on core.py multiple times today while re-running against the same video with no
    # record of which code version produced which output -- this is the fix: a hash of the config and
    # of this module itself, written before every live run, so a result can be traced back to the
    # exact code/config that produced it.
    import datetime

    output = root / cfg["output_root"]; output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "video_uid_filter": video_uid,
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "core_module_path": str(Path(__file__)), "core_module_sha256": sha256_file(Path(__file__)),
        "max_claim_rounds": cfg.get("max_claim_rounds"),
        "review_contract_version": cfg.get("review_contract_version"),
    }
    stamp = manifest["written_at_utc"].replace(":", "-")
    write_json(output / f"run_manifest_{stamp}.json", manifest)
    write_json(output / "run_manifest_latest.json", manifest)
    return manifest


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path); output = root / cfg["output_root"]
    pre = preflight(root, config_path, video_uid)
    if pre["overall_validation"] != "ready_for_live_pilot":
        raise RuntimeError("preflight failed")
    _load_env(root.parent / "thesis-av-evidence" / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("API keys unavailable before live calls")
    _write_run_manifest(root, cfg, config_path, video_uid)
    encoder = SiglipTextEncoder(cfg["siglip_text"])
    summaries: dict[str, Any] = {}
    for case_cfg, case_dir in _case_paths(cfg, root, video_uid):
        uid = case_cfg["video_uid"]
        question = load_json(case_dir / "question_input.json")
        hierarchy = load_json(case_dir / "shared_hierarchy.json")
        projections = load_json(case_dir / "r1_medium_projection.json")
        captions = load_json(case_dir / "r3_medium_captions.json")
        medium_embeddings = np.load(case_dir / "medium_siglip.float32.npy", allow_pickle=False).astype(np.float32)
        medium_embeddings /= np.maximum(np.linalg.norm(medium_embeddings, axis=1, keepdims=True), 1e-12)
        requirements = option_requirements(question)
        fine_by_id, fine_embeddings, row_by_id = _fine_registry(case_cfg, hierarchy)
        maps = {
            "r1_av": load_json(case_dir / "r1_av_navigation_map.json"),
            "r3_2": load_json(case_dir / "r3_2_navigation_map.json"),
        }
        case_out = output / "cases" / uid
        calls_path = case_out / "live_api_calls.json"
        calls = load_json(calls_path) if calls_path.is_file() else []
        side_results: dict[str, Any] = {}

        for side in ("r1_av", "r3_2"):
            side_out = case_out / side; side_out.mkdir(parents=True, exist_ok=True)
            map_doc = maps[side]
            parent = _parent_map(map_doc)
            coarse_ids = [row["coarse_id"] for row in map_doc["coarse_regions"]]

            planner_path = side_out / "planner.json"
            if planner_path.is_file():
                planner_doc = load_json(planner_path)
            else:
                plan, usage, payload = _call_coarse_locked_planner(cfg, question, requirements, coarse_ids, map_doc)
                planner_doc = {"input": payload, "output": plan, "usage": usage}
                write_json(planner_path, planner_doc)
                calls.append({"stage": "planner", "side": side, **usage})
                write_json(calls_path, calls)
            plan = planner_doc["output"]

            resolutions_path = side_out / "resolutions.json"
            if resolutions_path.is_file():
                resolutions_doc = load_json(resolutions_path)
            else:
                claims, evidence_by_rid, round_calls, investigation, fact_verification = _resolve_requirements(
                    cfg, question, requirements, plan, side, map_doc, hierarchy, projections,
                    captions, medium_embeddings, fine_by_id, fine_embeddings, row_by_id, encoder, parent,
                    int(cfg["max_claim_rounds"]),
                )
                calls.extend(round_calls)
                resolutions_doc = {"claims": claims, "evidence": evidence_by_rid}
                write_json(resolutions_path, resolutions_doc)
                write_json(calls_path, calls)
                # Persisted separately per the user's explicit request: every option's status is derived
                # from this ONE shared result, so the result itself -- what was established, whether it
                # was ever resolved, and exactly which evidence/images backed it -- must be inspectable
                # on its own, not only implicitly inside five duplicated per-option claim_text strings.
                # established_facts and atomic_facts are kept exactly as the model produced them --
                # fact_verification is a distinct, separately-stored verdict per fact (2026-08-08 fix),
                # never merged back into or overwriting the original, so the record stays auditable: what
                # was originally claimed, which specific statement got flagged, and why.
                write_json(side_out / "shared_investigation.json", {
                    "investigation_status": investigation["investigation_status"],
                    "established_facts": investigation["established_facts"],
                    "atomic_facts": investigation.get("atomic_facts", []),
                    "fact_verification": fact_verification,
                    "gap_reason": investigation["gap_reason"],
                    "cited_evidence_ids": investigation["cited_evidence_ids"],
                    "evidence": evidence_by_rid[requirements[0]["requirement_id"]],
                })
            per_requirement_claim = resolutions_doc["claims"]
            per_requirement_evidence = resolutions_doc["evidence"]

            resolved = {
                "question_id": question["question_id"],
                "assessments": [_claim_to_assessment(per_requirement_claim[row["requirement_id"]]) for row in requirements],
            }
            # V6.1: every requirement's evidence list is the SAME shared investigation's evidence (one
            # investigation feeds all options), so flattening per-rid would duplicate every row once per
            # option -- dedupe by evidence_id when building _final's payload.
            seen_evidence_ids: set[str] = set()
            all_evidence: list[dict[str, Any]] = []
            for rid in per_requirement_evidence:
                for row in per_requirement_evidence[rid]:
                    if row["evidence_id"] not in seen_evidence_ids:
                        seen_evidence_ids.add(row["evidence_id"])
                        all_evidence.append(row)

            final_path = side_out / "final_answer.json"
            if final_path.is_file():
                final_doc = load_json(final_path)
            else:
                answer, usage, raw = _final(cfg, question, requirements, resolved, all_evidence)
                answer = _finalize_answer(answer, question, per_requirement_claim)
                final_doc = {"answer": answer, "usage": usage}
                write_json(side_out / "final_raw_response.json", raw)
                write_json(final_path, final_doc)
                calls.append({"stage": "final", "side": side, "image_transmissions": 0, **usage})
                write_json(calls_path, calls)
            side_results[side] = final_doc["answer"]

        write_json(case_out / "answers_blind.json", side_results)
        summaries[uid] = {"question_id": question["question_id"], "answers": side_results, "calls": len(calls)}
        write_json(output / "live_progress.json", summaries)
    return _summarize_live(root, cfg, video_uid)


def _summarize_live(root: Path, cfg: dict[str, Any], video_uid: str | None) -> dict[str, Any]:
    # Always aggregates across every case that has completed so far, regardless of the video_uid
    # filter used for *this* run_live invocation -- summarizing only the current filter's scope
    # silently overwrote answers_blind.json/cost_accounting.json down to just the latest single-video
    # run every time, discarding every previously-completed video's recorded answer.
    output = root / cfg["output_root"]
    all_cases = _case_paths(cfg, root, None)
    all_calls, cases = [], []
    for case_cfg, case_dir in all_cases:
        uid = case_cfg["video_uid"]
        calls_path = output / "cases" / uid / "live_api_calls.json"
        calls = load_json(calls_path) if calls_path.is_file() else []
        all_calls.extend({"video_uid": uid, **row} for row in calls)
        answers_path = output / "cases" / uid / "answers_blind.json"
        if answers_path.is_file():
            cases.append({"video_uid": uid, "question_id": case_cfg["question_id"], "answers": load_json(answers_path)})
    cost = {"calls": all_calls, "estimated_api_usd": _estimated_cost(cfg, all_calls)}
    write_json(output / "cost_accounting.json", cost)
    write_json(output / "answers_blind.json", cases)
    validation = {
        "video_count": len(cases), "blind_answers_complete": len(cases) == len(all_cases),
        "gold_loaded_before_predictions": False, "overall_validation": "passed_live_pending_posthoc_evaluation",
    }
    write_json(output / "validation_report.json", validation)
    return {"validation": validation, "cost": cost}


def _find_gold(doc: Any, qid: str) -> str:
    stack = [doc]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            identity = value.get("question_id", value.get("qid", value.get("id")))
            if identity == qid:
                for key in ("correct_option", "correct_answer", "correct_answer_label", "answer", "label", "ground_truth"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                raise ValueError("matched question lacks gold label")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    raise ValueError(f"gold question not found: {qid}")


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); output = root / cfg["output_root"]
    blind_path = output / "answers_blind.json"
    if not blind_path.is_file():
        raise RuntimeError("blind answers absent")
    blind_hash = sha256_file(blind_path)
    annotation = load_json(Path(cfg["annotation_path"])) if "annotation_path" in cfg else None
    cases = load_json(blind_path)
    rows = []
    for case in cases:
        gold = _find_gold(annotation, case["question_id"]) if annotation is not None else None
        row = {"video_uid": case["video_uid"], "question_id": case["question_id"], "gold_option_id": gold}
        for side in ("r1_av", "r3_2"):
            selected = case["answers"][side]["selected_option_id"]
            row[side] = {"selected_option_id": selected, "correct": (selected == gold) if gold is not None else None}
        rows.append(row)
    counts = {
        "r1_correct": sum(bool(row["r1_av"]["correct"]) for row in rows),
        "r3_correct": sum(bool(row["r3_2"]["correct"]) for row in rows),
    }
    result = {"answers_blind_sha256_before_gold_load": blind_hash, "cases": rows, "summary": counts}
    write_json(output / "posthoc_evaluation.json", result)
    return result
