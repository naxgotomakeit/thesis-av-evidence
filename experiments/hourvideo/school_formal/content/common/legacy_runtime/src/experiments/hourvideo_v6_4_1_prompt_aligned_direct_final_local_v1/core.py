from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import sha256_file, write_json
from experiments.hourvideo_v6_4_pruned_direct_final_local_v1 import core as _v64


# Standalone on purpose: V6.4 prepended the legacy requirement-centric/answer_ready
# prompt even though its schema and execution unit are now one shared question.
SHARED_INVESTIGATION_SYSTEM = """You are the Shared Investigation stage for one multiple-choice
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


def _fine_gap_target(question: dict[str, Any], investigation: dict[str, Any]) -> str:
    options = "\n".join(
        f"{row['option_id']}: {row['text']}" for row in question["answer_options"]
    )
    established = investigation["established_facts"].strip() or "None yet."
    gap = investigation["gap_reason"].strip()
    return (
        f"Exact question: {question['question_text']}\n"
        f"Answer options:\n{options}\n"
        f"Facts already established (context only; do not merely verify or repeat them): {established}\n"
        f"Answer-critical unresolved gap to investigate: {gap}\n"
        "Inspect the supplied images for directly visible facts that close this gap. Preserve who did "
        "what, temporal order, duration, and location when they matter to the question. Do not choose "
        "an answer option."
    )


def _call_shared(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    excluded: list[dict[str, Any]], round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = _v64.PRUNED_SHARED_INVESTIGATION_SYSTEM
    _v64.PRUNED_SHARED_INVESTIGATION_SYSTEM = SHARED_INVESTIGATION_SYSTEM
    try:
        return _v64._call_pruned_shared_investigation(
            cfg, question, evidence, excluded, round_number,
        )
    finally:
        _v64.PRUNED_SHARED_INVESTIGATION_SYSTEM = original


def _resolve_to_shared(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    plan: dict[str, Any], side: str, map_doc: dict[str, Any], hierarchy: dict[str, Any],
    projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: Any, fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: Any, row_by_id: dict[str, int], encoder: Any,
    parent: dict[str, str], max_rounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    shared_id = f"{question['question_id']}::shared"
    locked_by_rid = {
        row["requirement_id"]: _v64._base._locked_coarse_ids(row)
        for row in plan["requirement_plans"]
    }
    locked: set[str] = set().union(*locked_by_rid.values()) if locked_by_rid else set()
    query_variants = [
        variant for row in plan["requirement_plans"] for variant in row["query_variants"]
    ]
    fallback = (
        _v64._base._medium_lexical_fallback(
            side, question, query_variants, hierarchy, projections, captions,
        )
        if not locked else []
    )
    evidence = _v64._base._map_level_evidence(
        side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
    ) + fallback
    image_findings: dict[str, dict[str, Any]] = {}

    def call_shared(round_number: int, stage: str) -> dict[str, Any]:
        excluded = _v64._base._excluded_judgments_union(map_doc, locked)
        doc, usage = _call_shared(cfg, question, evidence, excluded, round_number)
        calls.append({
            "stage": stage, "round": round_number,
            "requirement_id": shared_id, **usage,
        })
        return doc

    investigation = call_shared(0, "shared_investigation")
    for round_number in range(1, max_rounds + 1):
        if investigation["investigation_status"] != "unresolved":
            break
        if round_number > 1 and not investigation["requested_coarse_ids"]:
            break
        if investigation["requested_coarse_ids"]:
            locked |= set(investigation["requested_coarse_ids"])
            evidence = _v64._base._map_level_evidence(
                side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
            ) + fallback
            investigation = call_shared(round_number, "shared_investigation_recheck")
            if investigation["investigation_status"] != "unresolved":
                break
        if not locked:
            break
        retrieval = _v64._base._coarse_scoped_retrieval(
            cfg, side, question,
            {"requirement_plans": [_v64._base._shared_search_plan(plan, shared_id)]},
            map_doc, hierarchy, projections, captions, medium_embeddings,
            fine_by_id, fine_embeddings, row_by_id, encoder, {shared_id: locked},
        )
        fine_rows = retrieval["requirements"][shared_id]["selected_fine_evidence"]
        for row in fine_rows:
            if row["fine_id"] in image_findings:
                evidence.append({
                    "evidence_id": (
                        f"cached_observation::{shared_id}::{row['fine_id']}::round{round_number}"
                    ),
                    "evidence_type": "cached_visual_observation",
                    "source_content": image_findings[row["fine_id"]]["finding"],
                    "retrieved_for_requirement_id": shared_id,
                })
        fresh = [row for row in fine_rows if row["fine_id"] not in image_findings]
        if fresh:
            unique, targets = _v64._base._dedupe_fine_rows({shared_id: fresh})
            cap = int(cfg["ranking"].get("max_total_fine_evidence_per_batch", 16))
            unique = _v64._base._allocate_fine_quota_by_requirement(unique, targets, cap)
            target = _fine_gap_target(question, investigation)
            execution, usage, _ = (
                _v64._latest_guard._execute_claims_batch_with_bounded_text_retry(
                    cfg, question, {shared_id: target}, unique,
                )
            )
            calls.append({
                "stage": "claim_execution_batch", "round": round_number,
                "requirement_ids": [shared_id], "image_transmissions": len(unique), **usage,
            })
            for observation in execution["observations"]:
                image_findings[observation["fine_id"]] = observation
            evidence.append(_v64._base._batch_assessment_as_evidence(
                shared_id, round_number, execution["claim_assessments"][shared_id],
            ))
        elif not fine_rows:
            evidence.append({
                "evidence_id": f"claim_execution::{shared_id}::round{round_number}::empty",
                "evidence_type": "claim_execution_result",
                "source_content": "No Fine evidence available in the currently locked Coarse set.",
                "claim_status": "inconclusive",
                "supporting_fine_ids": [],
                "retrieved_for_requirement_id": shared_id,
            })
        investigation = call_shared(round_number, "shared_investigation")

    diagnostics = {
        "pruned_stages": [
            "atomic_coverage_reviewer", "fact_verification", "option_mapping",
            "mapping_consistency_guard", "mapping_based_final_consistency",
        ],
        "fact_verification_calls": 0,
        "option_mapping_calls": 0,
        "shared_prompt": "standalone_single_question_v1",
        "fine_target": "question_options_established_context_and_gap_v1",
    }
    return investigation, evidence, calls, diagnostics


def _manifest(root: Path, cfg: dict[str, Any], config_path: Path, video_uid: str | None) -> None:
    write_json(root / cfg["output_root"] / "run_manifest_latest.json", {
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "video_uid_filter": video_uid,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "module_sha256": sha256_file(Path(__file__)),
        "base_v6_4_sha256": sha256_file(Path(_v64.__file__)),
        "gold_loaded": False,
    })


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    original_resolver = _v64._resolve_to_shared
    original_manifest = _v64._manifest
    _v64._resolve_to_shared = _resolve_to_shared
    _v64._manifest = _manifest
    try:
        return _v64.run_live(root, config_path, video_uid=video_uid)
    finally:
        _v64._resolve_to_shared = original_resolver
        _v64._manifest = original_manifest


preflight = _v64.preflight
summarize = _v64.summarize
evaluate = _v64.evaluate
