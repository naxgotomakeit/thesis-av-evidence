from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_v6_5_prompt_aligned_direct_final_local_v1 import core as _v65


FORMAL_VERSION = "v6.5.2"
PROMOTION_SOURCE = "hourvideo_v6_5_prompt_aligned_direct_final_local_v1"
FINE_FEEDBACK_CONTRACT = "complete_reviewed_visual_observations_v1"
_BASE_DIRECT_FINAL = _v65._v641._v64._direct_final

SHARED_INVESTIGATION_SYSTEM = _v65.SHARED_INVESTIGATION_SYSTEM


def _reviewed_observation_as_evidence(
    shared_id: str,
    round_number: int,
    observation: dict[str, Any],
    fine_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fine_id = observation["fine_id"]
    fine = fine_by_id[fine_id]
    content = {
        "fine_id": fine_id,
        "finding": observation["finding"],
        "visible_actions": observation.get("visible_actions", []),
        "visible_objects": observation.get("visible_objects", []),
        "uncertainty_notes": observation.get("uncertainty_notes", []),
    }
    return {
        "evidence_id": f"reviewed_visual_observation::{shared_id}::{fine_id}",
        "evidence_type": "reviewed_visual_observation",
        "source_content": json.dumps(content, ensure_ascii=False, separators=(",", ":")),
        "fine_id": fine_id,
        "timestamp_sec": float(fine["timestamp_sec"]),
        "observation_round": round_number,
        "retrieved_for_requirement_id": shared_id,
    }


def _call_shared(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    excluded: list[dict[str, Any]], round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    v641 = _v65._v641
    original = v641.SHARED_INVESTIGATION_SYSTEM
    v641.SHARED_INVESTIGATION_SYSTEM = SHARED_INVESTIGATION_SYSTEM
    try:
        return v641._call_shared(cfg, question, evidence, excluded, round_number)
    finally:
        v641.SHARED_INVESTIGATION_SYSTEM = original


def _resolve_to_shared(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    plan: dict[str, Any], side: str, map_doc: dict[str, Any], hierarchy: dict[str, Any],
    projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: Any, fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: Any, row_by_id: dict[str, int], encoder: Any,
    parent: dict[str, str], max_rounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    v641 = _v65._v641
    v64 = v641._v64
    calls: list[dict[str, Any]] = []
    shared_id = f"{question['question_id']}::shared"
    locked_by_rid = {
        row["requirement_id"]: v64._base._locked_coarse_ids(row)
        for row in plan["requirement_plans"]
    }
    locked: set[str] = set().union(*locked_by_rid.values()) if locked_by_rid else set()
    query_variants = [
        variant for row in plan["requirement_plans"] for variant in row["query_variants"]
    ]
    fallback = (
        v64._base._medium_lexical_fallback(
            side, question, query_variants, hierarchy, projections, captions,
        )
        if not locked else []
    )
    evidence = v64._base._map_level_evidence(
        side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
    ) + fallback
    image_findings: dict[str, dict[str, Any]] = {}

    def call_shared(round_number: int, stage: str) -> dict[str, Any]:
        excluded = v64._base._excluded_judgments_union(map_doc, locked)
        doc, usage = _call_shared(cfg, question, evidence, excluded, round_number)
        calls.append({
            "stage": stage,
            "round": round_number,
            "requirement_id": shared_id,
            "reviewed_observations_available": sum(
                row.get("evidence_type") == "reviewed_visual_observation" for row in evidence
            ),
            **usage,
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
            evidence = v64._base._map_level_evidence(
                side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
            ) + fallback + [
                _reviewed_observation_as_evidence(
                    shared_id, 0, observation, fine_by_id,
                )
                for observation in image_findings.values()
            ]
            investigation = call_shared(round_number, "shared_investigation_recheck")
            if investigation["investigation_status"] != "unresolved":
                break
        if not locked:
            break
        retrieval = v64._base._coarse_scoped_retrieval(
            cfg, side, question,
            {"requirement_plans": [v64._base._shared_search_plan(plan, shared_id)]},
            map_doc, hierarchy, projections, captions, medium_embeddings,
            fine_by_id, fine_embeddings, row_by_id, encoder, {shared_id: locked},
        )
        fine_rows = retrieval["requirements"][shared_id]["selected_fine_evidence"]
        fresh = [row for row in fine_rows if row["fine_id"] not in image_findings]
        if fresh:
            unique, targets = v64._base._dedupe_fine_rows({shared_id: fresh})
            cap = int(cfg["ranking"].get("max_total_fine_evidence_per_batch", 16))
            unique = v64._base._allocate_fine_quota_by_requirement(unique, targets, cap)
            target = v641._fine_gap_target(question, investigation)
            execution, usage, _ = (
                v64._latest_guard._execute_claims_batch_with_bounded_text_retry(
                    cfg, question, {shared_id: target}, unique,
                )
            )
            calls.append({
                "stage": "claim_execution_batch",
                "round": round_number,
                "requirement_ids": [shared_id],
                "image_transmissions": len(unique),
                "reviewed_observations_returned": len(execution["observations"]),
                **usage,
            })
            for observation in execution["observations"]:
                image_findings[observation["fine_id"]] = observation
                evidence.append(_reviewed_observation_as_evidence(
                    shared_id, round_number, observation, fine_by_id,
                ))
            evidence.append(v64._base._batch_assessment_as_evidence(
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
        "shared_prompt": "standalone_single_question_v1_unchanged",
        "fine_target": "question_options_established_context_and_gap_v1",
        "fine_feedback_contract": FINE_FEEDBACK_CONTRACT,
        "reviewed_visual_observation_count": len(image_findings),
    }
    return investigation, evidence, calls, diagnostics


def _direct_final_with_8b(
    cfg: dict[str, Any], question: dict[str, Any], investigation: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    final_cfg = copy.deepcopy(cfg)
    final_cfg["gemini"] = copy.deepcopy(cfg["final"])
    answer, usage, raw, payload = _BASE_DIRECT_FINAL(
        final_cfg, question, investigation, evidence,
    )
    usage.update({
        "provider_role": "semantic_final",
        "visual_model": cfg["gemini"]["model"],
        "final_model": cfg["final"]["model"],
    })
    return answer, usage, raw, payload


def _manifest(
    root: Path, cfg: dict[str, Any], config_path: Path, video_uid: str | None,
) -> None:
    write_json(root / cfg["output_root"] / "run_manifest_latest.json", {
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formal_version": FORMAL_VERSION,
        "promotion_source": PROMOTION_SOURCE,
        "video_uid_filter": video_uid,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "module_sha256": sha256_file(Path(__file__)),
        "fine_feedback_contract": FINE_FEEDBACK_CONTRACT,
        "visual_model": cfg["gemini"]["model"],
        "final_model": cfg["final"]["model"],
        "gold_loaded": False,
    })


def preflight(
    root: Path, config_path: Path, video_uid: str | None = None,
) -> dict[str, Any]:
    result = _v65.preflight(root, config_path, video_uid)
    cfg = load_json(config_path)
    result.update({
        "formal_version": FORMAL_VERSION,
        "promotion_source": PROMOTION_SOURCE,
        "fine_feedback_contract": FINE_FEEDBACK_CONTRACT,
        "visual_model": cfg["gemini"]["model"],
        "final_model": cfg["final"]["model"],
    })
    write_json(root / cfg["output_root"] / "preflight.json", result)
    return result


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    v641 = _v65._v641
    v64 = v641._v64
    original_resolver = v641._resolve_to_shared
    original_direct_final = v64._direct_final
    original_manifest = v641._manifest
    original_base_preflight = v64.preflight
    v641._resolve_to_shared = _resolve_to_shared
    v64._direct_final = _direct_final_with_8b
    v641._manifest = _manifest
    v64.preflight = preflight
    try:
        return v641.run_live(root, config_path, video_uid=video_uid)
    finally:
        v641._resolve_to_shared = original_resolver
        v64._direct_final = original_direct_final
        v641._manifest = original_manifest
        v64.preflight = original_base_preflight


summarize = _v65.summarize
evaluate = _v65.evaluate
