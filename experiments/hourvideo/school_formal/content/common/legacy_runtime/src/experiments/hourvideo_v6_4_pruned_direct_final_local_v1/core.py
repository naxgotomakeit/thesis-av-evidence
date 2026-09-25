from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_3_local import core as _base
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import (
    _fine_registry,
    _lexical_rows,
    _parent_map,
    option_requirements,
)
from experiments.hourvideo_v6_3_local_temporal_operator_guard_v1 import core as _latest_guard
from experiments.planner_medium_retrieval.core import SiglipTextEncoder


SIDES = ("r1_av", "r3_2")

DIRECT_FINAL_SYSTEM = """You are the terminal multiple-choice answer stage for an egocentric long-video
question. A preceding Shared Investigation has already searched the navigation map and, when needed,
inspected Fine images. Read its complete established_facts as a coherent evidence report; do not split it
into atomic facts and do not require every relation to be repeated in a single cited row. Compare the
report directly with all mutually exclusive answer options and select exactly one option. When the Shared
Investigation is unresolved, still provide the best evidence-based multiple-choice answer but mark it as
best_guess. Cite only integer evidence indexes supplied in evidence_index. Return strict JSON only."""

PRUNED_SHARED_INVESTIGATION_SYSTEM = _base.SUFFICIENCY_SYSTEM + """
Investigate ONE underlying question about a video, rather than judging answer options independently.
Use only the supplied evidence. Return resolved only when established_facts states the actual fact(s)
plainly and specifically enough to distinguish the answer choices. A resolved report must cite supporting
evidence and must not request more Coarse regions. Return unresolved when evidence is insufficient; then
state the gap plainly and request only excluded Coarse regions that could close it. Absence of evidence is
not evidence of absence. Write one coherent established_facts report. Do not decompose it into atomic
facts or repeat it in per-option assessments. Return strict JSON only.
"""


def _pruned_shared_schema(
    question: dict[str, Any], evidence: list[dict[str, Any]], excluded_coarse_ids: list[str],
) -> dict[str, Any]:
    evidence_ids = [row["evidence_id"] for row in evidence]
    requested = (
        {"type": "array", "items": {"type": "string", "enum": excluded_coarse_ids}}
        if excluded_coarse_ids else {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    )
    cited = (
        {"type": "array", "items": {"type": "string", "enum": evidence_ids}}
        if evidence_ids else {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    )
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string", "const": question["question_id"]},
            "investigation_status": {"type": "string", "enum": ["resolved", "unresolved"]},
            "established_facts": {"type": "string"},
            "gap_reason": {"type": "string"},
            "requested_coarse_ids": requested,
            "cited_evidence_ids": cited,
        },
        "required": [
            "question_id", "investigation_status", "established_facts", "gap_reason",
            "requested_coarse_ids", "cited_evidence_ids",
        ],
    }


def _validate_pruned_shared(
    value: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    allowed_coarse: set[str],
) -> None:
    if value.get("question_id") != question["question_id"]:
        raise ValueError("Shared question identity mismatch")
    evidence_ids = {row["evidence_id"] for row in evidence}
    if not set(value["requested_coarse_ids"]) <= allowed_coarse:
        raise ValueError("Shared requested an unknown/non-excluded Coarse region")
    if not set(value["cited_evidence_ids"]) <= evidence_ids:
        raise ValueError("Shared cited unknown evidence")
    if value["investigation_status"] == "resolved":
        if not value["established_facts"].strip():
            raise ValueError("resolved Shared report must state established_facts")
        if value["requested_coarse_ids"]:
            raise ValueError("resolved Shared report must not request more Coarse regions")
        if not evidence or not value["cited_evidence_ids"]:
            raise ValueError("resolved Shared report requires cited evidence")
        mentioned = _base._evidence_ids_mentioned_in_text(value["established_facts"], evidence)
        if not mentioned <= set(value["cited_evidence_ids"]):
            raise ValueError("established_facts references evidence absent from cited_evidence_ids")
        if _base._claim_text_contradicts_status("supported", value["established_facts"]):
            raise ValueError("established_facts contradicts investigation_status=resolved")
    elif not value["gap_reason"].strip():
        raise ValueError("unresolved Shared report must state gap_reason")


def _combine_shared_usages(
    usages: list[dict[str, Any]], errors: list[str],
    attempt_telemetry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    combined = dict(usages[-1])
    accounting_rows = attempt_telemetry or usages
    for key in ("input_tokens", "output_tokens"):
        if all(row.get(key) is not None for row in accounting_rows):
            combined[key] = sum(int(row[key]) for row in accounting_rows)
    combined["latency_sec"] = sum(
        float(row.get("latency_sec") or 0.0) for row in accounting_rows
    )
    combined["shared_contract"] = "coherent_report_without_atomic_facts_v1"
    combined["shared_attempt_count"] = len(accounting_rows)
    combined["shared_validator_feedback_retry"] = len(accounting_rows) > 1
    combined["previous_validation_errors"] = errors
    combined["attempt_telemetry"] = copy.deepcopy(accounting_rows)
    combined["attempt_telemetry_complete"] = all(
        row.get("input_tokens") is not None and row.get("output_tokens") is not None
        for row in accounting_rows
    )
    return combined


def _call_pruned_shared_investigation(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    excluded_judgments: list[dict[str, Any]], round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    excluded_ids = [row["coarse_id"] for row in excluded_judgments]
    allowed_coarse = set(excluded_ids)
    known_evidence = {row["evidence_id"] for row in evidence}
    base_payload = {
        "question": question, "evidence": evidence, "round": round_number,
        "excluded_coarse_regions": excluded_judgments,
    }
    schema = _pruned_shared_schema(question, evidence, excluded_ids)
    attempts = max(1, int(cfg.get("max_validation_retries", 2)))
    errors: list[str] = []
    usages: list[dict[str, Any]] = []
    attempt_telemetry: list[dict[str, Any]] = []
    last_value: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(attempts):
        payload = copy.deepcopy(base_payload)
        if errors:
            retry_instruction = "Regenerate the full JSON object and correct this contract error."
            if "reached max_tokens" in errors[-1]:
                retry_instruction = (
                    "The previous response was truncated at max_tokens. Return only one compact JSON "
                    "object now: no analysis, preamble, repetition, or exhaustive narration. State only "
                    "the decisive established facts or gap needed for this question."
                )
            payload["validator_feedback"] = {
                "previous_error": errors[-1],
                "instruction": retry_instruction,
            }
        try:
            provider, usage = _base._anthropic_call(
                cfg, PRUNED_SHARED_INVESTIGATION_SYSTEM, payload, schema,
                int(cfg["anthropic"]["claim_max_tokens"]),
            )
        except RuntimeError as error:
            last_error = error
            errors.append(str(error))
            failed = copy.deepcopy(getattr(error, "attempt_telemetry", {}))
            failed.update({
                "attempt": attempt + 1,
                "status": "max_tokens" if "reached max_tokens" in str(error) else "provider_error",
                "failure_reason": str(error),
                "retry_triggered": attempt + 1 < attempts,
            })
            attempt_telemetry.append(failed)
            continue
        usages.append(usage)
        value = dict(provider)
        value["requested_coarse_ids"] = list(dict.fromkeys(
            cid for cid in value["requested_coarse_ids"] if cid in allowed_coarse
        ))
        value["cited_evidence_ids"] = list(dict.fromkeys(
            eid for eid in value["cited_evidence_ids"] if eid in known_evidence
        ))
        last_value = value
        try:
            _validate_pruned_shared(value, question, evidence, allowed_coarse)
            attempt_telemetry.append({
                **copy.deepcopy(usage),
                "attempt": attempt + 1,
                "status": "success",
                "failure_reason": None,
                "retry_triggered": False,
            })
            return value, _combine_shared_usages(usages, errors, attempt_telemetry)
        except ValueError as error:
            last_error = error
            errors.append(str(error))
            attempt_telemetry.append({
                **copy.deepcopy(usage),
                "attempt": attempt + 1,
                "status": "validation_failed",
                "failure_reason": str(error),
                "retry_triggered": attempt + 1 < attempts,
            })

    if last_value is None:
        raise last_error or RuntimeError("Shared report generation failed")
    # A contradictory resolved/request-more combination is safely represented as unresolved. This
    # does not choose an answer; it preserves requested regions and lets the existing retrieval loop
    # gather the missing evidence instead of accepting a false terminal state.
    if last_value["investigation_status"] == "resolved":
        last_value["investigation_status"] = "unresolved"
        last_value["gap_reason"] = last_value["gap_reason"].strip() or (
            "The generated Shared report did not satisfy the resolved-state evidence contract; "
            "additional evidence is required."
        )
    _validate_pruned_shared(last_value, question, evidence, allowed_coarse)
    usage = _combine_shared_usages(usages, errors, attempt_telemetry)
    usage["shared_terminal_state_downgraded"] = True
    return last_value, usage


def _direct_final_schema(
    question: dict[str, Any], evidence_count: int, decision_status: str,
) -> dict[str, Any]:
    option_ids = [row["option_id"] for row in question["answer_options"]]
    evidence_items: dict[str, Any] = {"type": "integer"}
    if evidence_count:
        evidence_items["enum"] = list(range(evidence_count))
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string", "const": question["question_id"]},
            "selected_option_id": {"type": "string", "enum": option_ids},
            "answer_text": {"type": "string", "maxLength": 512},
            "decision_status": {"type": "string", "enum": [decision_status]},
            "supporting_evidence_indexes": {
                "type": "array",
                "maxItems": evidence_count,
                "items": evidence_items,
            },
            "reason": {"type": "string", "maxLength": 1536},
            "uncertainty": {"type": "string", "maxLength": 768},
        },
        "required": [
            "question_id", "selected_option_id", "answer_text", "decision_status",
            "supporting_evidence_indexes", "reason", "uncertainty",
        ],
    }


def _direct_final(
    cfg: dict[str, Any], question: dict[str, Any], investigation: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_id = {row["evidence_id"]: row for row in evidence}
    cited = []
    for evidence_id in investigation.get("cited_evidence_ids", []):
        if evidence_id in by_id and evidence_id not in {row["evidence_id"] for row in cited}:
            cited.append(by_id[evidence_id])
    payload = {
        "question": question,
        "shared_investigation": {
            "investigation_status": investigation["investigation_status"],
            "established_facts": investigation["established_facts"],
            "gap_reason": investigation["gap_reason"],
        },
        "evidence_index": [
            {
                "index": index,
                "evidence_id": row["evidence_id"],
                "evidence_type": row["evidence_type"],
                "source_content": row["source_content"],
                **({"interval": row["interval"]} if "interval" in row else {}),
                **({"timestamp_sec": row["timestamp_sec"]} if "timestamp_sec" in row else {}),
            }
            for index, row in enumerate(cited)
        ],
    }
    expected_status = "grounded" if investigation["investigation_status"] == "resolved" else "best_guess"
    schema = _direct_final_schema(question, len(cited), expected_status)
    result, usage, raw = _latest_guard._gemini_call_with_schema_preflight(
        {"gemini": cfg["gemini"]}, DIRECT_FINAL_SYSTEM,
        [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
        schema,
    )
    if result["question_id"] != question["question_id"]:
        raise ValueError("Direct Final question identity mismatch")
    selected = result["selected_option_id"]
    option_by_id = {row["option_id"]: row["text"] for row in question["answer_options"]}
    if result["answer_text"].strip().casefold() != option_by_id[selected].strip().casefold():
        raise ValueError("Direct Final answer_text does not match selected option")
    raw_indexes = result["supporting_evidence_indexes"]
    if any(not isinstance(i, int) or i < 0 or i >= len(cited) for i in raw_indexes):
        raise ValueError("Direct Final returned invalid evidence index")
    indexes = list(dict.fromkeys(raw_indexes))
    if result["decision_status"] != expected_status:
        raise ValueError(
            f"Direct Final status mismatch: expected {expected_status} from Shared "
            f"{investigation['investigation_status']}"
        )
    answer = {
        "question_id": result["question_id"],
        "selected_option_id": selected,
        "answer_text": result["answer_text"],
        "decision_status": result["decision_status"],
        "supporting_evidence_ids": [cited[index]["evidence_id"] for index in indexes],
        "reason": result["reason"],
        "uncertainty": result["uncertainty"],
        "final_status": "confirmed" if result["decision_status"] == "grounded" else "budget_exhausted_guess",
    }
    usage = {
        **usage,
        "direct_final_adapter": "shared_established_facts_to_local_final_v1",
        "input_evidence_count": len(cited),
        "duplicate_evidence_indexes_removed": len(raw_indexes) - len(indexes),
    }
    return answer, usage, raw, payload


def _resolve_to_shared(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]],
    plan: dict[str, Any], side: str, map_doc: dict[str, Any], hierarchy: dict[str, Any],
    projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: np.ndarray, fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: np.ndarray, row_by_id: dict[str, int], encoder: Any,
    parent: dict[str, str], max_rounds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    shared_id = f"{question['question_id']}::shared"
    locked_by_rid = {
        row["requirement_id"]: _base._locked_coarse_ids(row)
        for row in plan["requirement_plans"]
    }
    locked: set[str] = set().union(*locked_by_rid.values()) if locked_by_rid else set()
    query_variants = [
        variant for row in plan["requirement_plans"] for variant in row["query_variants"]
    ]
    fallback = (
        _base._medium_lexical_fallback(side, question, query_variants, hierarchy, projections, captions)
        if not locked else []
    )
    evidence = _base._map_level_evidence(
        side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
    ) + fallback
    image_findings: dict[str, dict[str, Any]] = {}

    def call_shared(round_number: int, stage: str) -> dict[str, Any]:
        excluded = _base._excluded_judgments_union(map_doc, locked)
        doc, usage = _call_pruned_shared_investigation(
            cfg, question, evidence, excluded, round_number=round_number,
        )
        calls.append({
            "stage": stage, "round": round_number, "requirement_id": shared_id, **usage,
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
            evidence = _base._map_level_evidence(
                side, shared_id, locked, map_doc, hierarchy, projections, captions, parent,
            ) + fallback
            investigation = call_shared(round_number, "shared_investigation_recheck")
            if investigation["investigation_status"] != "unresolved":
                break
        if not locked:
            break
        retrieval = _base._coarse_scoped_retrieval(
            cfg, side, question,
            {"requirement_plans": [_base._shared_search_plan(plan, shared_id)]},
            map_doc, hierarchy, projections, captions, medium_embeddings,
            fine_by_id, fine_embeddings, row_by_id, encoder, {shared_id: locked},
        )
        fine_rows = retrieval["requirements"][shared_id]["selected_fine_evidence"]
        for row in fine_rows:
            if row["fine_id"] in image_findings:
                evidence.append({
                    "evidence_id": f"cached_observation::{shared_id}::{row['fine_id']}::round{round_number}",
                    "evidence_type": "cached_visual_observation",
                    "source_content": image_findings[row["fine_id"]]["finding"],
                    "retrieved_for_requirement_id": shared_id,
                })
        fresh = [row for row in fine_rows if row["fine_id"] not in image_findings]
        if fresh:
            unique, targets = _base._dedupe_fine_rows({shared_id: fresh})
            cap = int(cfg["ranking"].get("max_total_fine_evidence_per_batch", 16))
            unique = _base._allocate_fine_quota_by_requirement(unique, targets, cap)
            target = investigation["established_facts"] or investigation["gap_reason"]
            execution, usage, raw = _latest_guard._execute_claims_batch_with_bounded_text_retry(
                cfg, question, {shared_id: target}, unique,
            )
            calls.append({
                "stage": "claim_execution_batch", "round": round_number,
                "requirement_ids": [shared_id], "image_transmissions": len(unique), **usage,
            })
            for observation in execution["observations"]:
                image_findings[observation["fine_id"]] = observation
            evidence.append(_base._batch_assessment_as_evidence(
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
    }
    return investigation, evidence, calls, diagnostics


def _planner_source_path(root: Path, cfg: dict[str, Any], uid: str, side: str) -> Path:
    return root / cfg["planner_source_experiment"] / "live" / "cases" / uid / side / "planner.json"


def preflight(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    output = root / cfg["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    cases = _base._case_paths(cfg, root, video_uid)
    if not cases:
        raise RuntimeError("no matching Pilot10 cases")
    audits = []
    for case_cfg, case_dir in cases:
        offline = _base._offline_audit(case_cfg, case_dir)
        planners = {
            side: {
                "path": str(_planner_source_path(root, cfg, case_cfg["video_uid"], side)),
                "sha256": sha256_file(_planner_source_path(root, cfg, case_cfg["video_uid"], side)),
            }
            for side in SIDES
        }
        audits.append({**offline, "planner_sources": planners})
    result = {
        "status": "ready",
        "experiment": cfg["experiment"],
        "case_count": len(audits),
        "run_count": len(audits) * len(SIDES),
        "sides": list(SIDES),
        "source_experiment": cfg["source_experiment"],
        "planner_reused": True,
        "semantic_model": cfg["anthropic"]["model"],
        "visual_and_final_model": cfg["gemini"]["model"],
        "decision_path": ["planner", "shared", "optional_fine", "shared", "direct_final"],
        "pruned_stages": ["fact_verification", "option_mapping", "mapping_based_final"],
        "gold_loaded": False,
        "audits": audits,
    }
    write_json(output / "preflight.json", result)
    return result


def _manifest(root: Path, cfg: dict[str, Any], config_path: Path, video_uid: str | None) -> None:
    write_json(root / cfg["output_root"] / "run_manifest_latest.json", {
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "video_uid_filter": video_uid,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "module_sha256": sha256_file(Path(__file__)),
        "base_v6_3_sha256": sha256_file(Path(_base.__file__)),
        "gold_loaded": False,
    })


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    output = root / cfg["output_root"]
    preflight(root, config_path, video_uid)
    _manifest(root, cfg, config_path, video_uid)
    encoder = SiglipTextEncoder(cfg["siglip_text"])
    for case_cfg, case_dir in _base._case_paths(cfg, root, video_uid):
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
        calls_path = case_out / "live_model_calls.json"
        calls = load_json(calls_path) if calls_path.is_file() else []
        answers = load_json(case_out / "answers_blind.json") if (case_out / "answers_blind.json").is_file() else {}
        for side in SIDES:
            side_out = case_out / side
            side_out.mkdir(parents=True, exist_ok=True)
            final_path = side_out / "final_answer.json"
            if final_path.is_file():
                answers[side] = load_json(final_path)["answer"]
                continue
            planner_source = _planner_source_path(root, cfg, uid, side)
            planner_doc = load_json(planner_source)
            write_json(side_out / "planner.json", {
                **planner_doc,
                "reused_from": str(planner_source),
                "reused_sha256": sha256_file(planner_source),
            })
            investigation, evidence, round_calls, diagnostics = _resolve_to_shared(
                cfg, question, requirements, planner_doc["output"], side, maps[side], hierarchy,
                projections, captions, medium_embeddings, fine_by_id, fine_embeddings, row_by_id,
                encoder, _parent_map(maps[side]), int(cfg["max_claim_rounds"]),
            )
            calls.extend({"side": side, **row} for row in round_calls)
            write_json(side_out / "shared_investigation.json", {
                **investigation,
                "evidence": evidence,
                "v6_4_diagnostics": diagnostics,
            })
            answer, usage, raw, payload = _direct_final(cfg, question, investigation, evidence)
            write_json(side_out / "direct_final_input.json", payload)
            write_json(side_out / "direct_final_raw.json", raw)
            write_json(final_path, {"answer": answer, "usage": usage})
            calls.append({"stage": "direct_final", "side": side, "image_transmissions": 0, **usage})
            write_json(calls_path, calls)
            answers[side] = answer
            write_json(case_out / "answers_blind.json", answers)
        write_json(case_out / "answers_blind.json", answers)
    return summarize(root, config_path)


def summarize(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    output = root / cfg["output_root"]
    rows = []
    calls = []
    for case_cfg, _ in _base._case_paths(cfg, root, None):
        uid = case_cfg["video_uid"]
        answer_path = output / "cases" / uid / "answers_blind.json"
        if answer_path.is_file():
            answers = load_json(answer_path)
            if all(side in answers for side in SIDES):
                rows.append({"video_uid": uid, "question_id": case_cfg["question_id"], "answers": answers})
        calls_path = output / "cases" / uid / "live_model_calls.json"
        if calls_path.is_file():
            calls.extend({"video_uid": uid, **row} for row in load_json(calls_path))
    write_json(output / "answers_blind.json", rows)
    stage_counts = {}
    for call in calls:
        stage_counts[call["stage"]] = stage_counts.get(call["stage"], 0) + 1
    result = {
        "completed_cases": len(rows),
        "completed_runs": sum(side in row["answers"] for row in rows for side in SIDES),
        "target_cases": len(_base._case_paths(cfg, root, None)),
        "gold_loaded_before_predictions": False,
        "stage_counts": stage_counts,
        "fact_verification_calls": stage_counts.get("fact_verification", 0),
        "option_mapping_calls": stage_counts.get("option_mapping", 0),
    }
    write_json(output / "validation_report.json", result)
    return result


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    output = root / cfg["output_root"]
    blind_path = output / "answers_blind.json"
    blind_hash = sha256_file(blind_path)
    annotations = load_json(Path(cfg["annotation_path"]))
    rows = []
    for case in load_json(blind_path):
        gold = _base._find_gold(annotations, case["question_id"])
        row = {"video_uid": case["video_uid"], "question_id": case["question_id"], "gold_option_id": gold}
        for side in SIDES:
            selected = case["answers"][side]["selected_option_id"]
            row[side] = {
                "selected_option_id": selected,
                "correct": selected == gold,
                "final_status": case["answers"][side]["final_status"],
            }
        rows.append(row)
    result = {
        "answers_blind_sha256_before_gold_load": blind_hash,
        "cases": rows,
        "summary": {
            side: {
                "correct": sum(row[side]["correct"] for row in rows),
                "total": len(rows),
            }
            for side in SIDES
        },
    }
    write_json(output / "posthoc_evaluation.json", result)
    return result
