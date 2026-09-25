from __future__ import annotations

import json
from typing import Any

from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_3_local import core as _base
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import FINAL_SYSTEM, _gemini_call
from experiments.hourvideo_v6_3_local_final_evidence_enum_v1 import core as _enum


preflight = _base.preflight
evaluate = _base.evaluate


FACT_VERIFICATION_SYSTEM = _base.FACT_VERIFICATION_SYSTEM + """

Apply semantic entailment rather than exact-word matching. In egocentric-video evidence, "Ego", "the
camera wearer", and "the wearer" denote the same viewpoint person unless the evidence explicitly says
otherwise. A more specific observation entails its ordinary broader description: for example, peeling
and chopping onions entails preparing ingredients. Do not reject a claim merely because the evidence is
more specific, uses a synonymous participant label, or distributes support across several cited
snippets. Directly observing the camera wearer acting in a named room establishes that the wearer was in
and interacted in that room when the question asks which locations were visited or used; it does not by
itself establish a particular traversal order. Still reject genuinely added objects, actions, locations,
or temporal relations that the cited evidence does not support.
"""


OPTION_MAPPING_SYSTEM = _base.OPTION_MAPPING_SYSTEM + """

First identify the question's answer operator, then judge each option as a candidate answer to that
operator, not merely as a statement that its named event occurred somewhere in the video.

- For "which event happened first: X or Y?", if verified facts establish X before Y, the option naming X
  is supported and the option naming Y is refuted. Never reverse this by arguing that Y occurred after X.
- For a question asking for the complete list or sequence of locations/events, an option that omits an
  established requested item is not an exact supported answer merely because every item it does list is
  true. Compare the complete candidate with the complete established set or sequence.
- For sequence, before/after, and fill-in-the-blank questions, map the fact at the requested temporal slot;
  do not support a different fact merely because it also occurred during the video.
- If verified facts establish an answer outside all supplied mutually exclusive options, do not invent a
  match. Apply the same consequence consistently to every incompatible option.
"""


def _verify_decisive_facts_semantically(
    cfg: dict[str, Any], atomic_facts: list[dict[str, Any]], requirements: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify every decisive cited fact semantically, without the contradictory lexical-zero shortcut."""
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    results: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for fact in atomic_facts:
        text = fact["text"]
        fact_evidence_ids = [eid for eid in fact["evidence_ids"] if eid in evidence_by_id]
        if not fact_evidence_ids or not _base._is_decisive_fact(text, requirements):
            results.append({
                "text": text, "evidence_ids": fact["evidence_ids"], "checked": False,
                "verified": True, "reason": "",
            })
            continue
        cited_text = " ".join(evidence_by_id[eid]["source_content"] for eid in fact_evidence_ids)
        verdict, usage = _base._call_fact_verification(cfg, text, cited_text)
        calls.append({"stage": "fact_verification", "semantic_zero_overlap_enabled": True, **usage})
        results.append({
            "text": text, "evidence_ids": fact["evidence_ids"], "checked": True,
            "verified": bool(verdict["entailed"]), "reason": verdict["reason"],
        })
    return results, calls


def _refuted_option_ids(question: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    status_by_requirement = {
        row["requirement_id"]: row["status"] for row in resolved.get("assessments", [])
    }
    return [
        option["option_id"] for option in question["answer_options"]
        if status_by_requirement.get(
            f"{question['question_id']}::option_{option['option_id'].lower()}"
        ) == "not_found"
    ]


def _validate_final_provenance(
    result: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> None:
    known_evidence = {row["evidence_id"] for row in evidence}
    known_requirements = {row["requirement_id"] for row in requirements}
    if (
        result["question_id"] != question["question_id"]
        or set(result["supporting_evidence_ids"]) - known_evidence
        or set(result["supporting_requirement_ids"]) - known_requirements
    ):
        raise ValueError("Final answer provenance invalid")


def _final_with_conflict_feedback(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    resolved: dict[str, Any], evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = {
        "question": question, "requirements": requirements,
        "resolved_assessments": resolved, "evidence": evidence,
    }
    schema = _enum._final_schema_with_evidence_enum(question, requirements, evidence)
    base_input = {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
    inputs = [base_input]
    attempts = int(cfg.get("max_validation_retries", 2))
    refuted = _refuted_option_ids(question, resolved)
    conflicted_selections: list[str] = []
    last_error: ValueError | None = None
    for attempt in range(attempts):
        result, usage, raw = _gemini_call({"gemini": cfg["gemini"]}, FINAL_SYSTEM, inputs, schema)
        try:
            _validate_final_provenance(result, question, requirements, evidence)
            selected = result["selected_option_id"]
            if selected in refuted:
                conflicted_selections.append(selected)
                raise ValueError(
                    f"Final selected option {selected}, but option mapping marked it refuted/not_found"
                )
            usage["final_evidence_id_adapter"] = "exact_payload_evidence_enum_v1"
            usage["allowed_evidence_id_count"] = len(evidence)
            usage["final_option_consistency_adapter"] = "refuted_conflict_feedback_retry_v1"
            usage["refuted_final_option_ids"] = refuted
            usage["final_validator_feedback_retry"] = attempt > 0
            usage["previous_conflicted_selections"] = conflicted_selections
            return result, usage, raw
        except ValueError as error:
            last_error = error
            status_rows = [
                {"requirement_id": row["requirement_id"], "status": row["status"]}
                for row in resolved.get("assessments", [])
            ]
            feedback = (
                "VALIDATOR ERROR FROM THE PREVIOUS RESPONSE: " + str(error) + "\n"
                f"Option-mapping statuses are: {json.dumps(status_rows, ensure_ascii=False)}.\n"
                "Re-read the question operator, resolved assessments, and evidence. Regenerate the entire "
                "JSON answer. Do not silently repeat a refuted option. If the assessments do not support a "
                "confident answer, choose only a non-refuted best candidate and state the uncertainty in caveats."
            )
            inputs = [base_input, {"type": "text", "text": feedback}]
    raise ValueError(
        "Final consistency unresolved after validator-feedback retry; "
        f"refuted_options={refuted}, selections={conflicted_selections}, last_error={last_error}"
    )


def run_live(root, config_path, video_uid=None):
    original_final = _base._final
    original_downgrade = _base._downgrade_noncompliant_investigation
    original_execution = _base._execute_claims_batch
    original_verify = _base._verify_decisive_facts
    original_fact_prompt = _base.FACT_VERIFICATION_SYSTEM
    original_mapping_prompt = _base.OPTION_MAPPING_SYSTEM
    _enum._base_downgrade = original_downgrade
    _base._final = _final_with_conflict_feedback
    _base._downgrade_noncompliant_investigation = _enum._downgrade_inconsistent_resolved_investigation
    _base._execute_claims_batch = _enum._execute_claims_batch_with_indexed_fines
    _base._verify_decisive_facts = _verify_decisive_facts_semantically
    _base.FACT_VERIFICATION_SYSTEM = FACT_VERIFICATION_SYSTEM
    _base.OPTION_MAPPING_SYSTEM = OPTION_MAPPING_SYSTEM
    try:
        return _base.run_live(root, config_path, video_uid=video_uid)
    finally:
        _base._final = original_final
        _base._downgrade_noncompliant_investigation = original_downgrade
        _base._execute_claims_batch = original_execution
        _base._verify_decisive_facts = original_verify
        _base.FACT_VERIFICATION_SYSTEM = original_fact_prompt
        _base.OPTION_MAPPING_SYSTEM = original_mapping_prompt
