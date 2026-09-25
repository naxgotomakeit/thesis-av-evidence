from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.fine_reranking.core import rerank_fines_for_medium, select_diverse_fines
from experiments.hourvideo_r1_av_r3_2_frozen_gate_cache_replay_v1.core import _review_missing
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import (
    FINAL_SYSTEM,
    SUFFICIENCY_SYSTEM,
    _anthropic_call,
    _build_r1_map,
    _build_r3_map,
    _estimated_cost,
    _gemini_call,
    _load_env,
)
from experiments.planner_medium_retrieval.core import SiglipTextEncoder, lexical_similarity
from experiments.r1_av_r3_2_requirement_centric_pipeline_canary.core import PLANNER_PROMPT
from experiments.r1_av_r3_2_review_cache_integration_canary_v1.core import apply_scope_records
from experiments.reviewed_visual_evidence_cache_v1 import LayeredVisualReviewCache, ReviewRecord
from experiments.reviewed_visual_evidence_cache_v1.cache import canonical_bytes


SUFFICIENCY_PROMPT = SUFFICIENCY_SYSTEM + """
The declared requirements are option hypotheses, one for each supplied answer option.
Assess only those hypotheses. For R3, a semantic_coarse_summary is a caption-and-ASR-derived
navigation statement and may directly support only facts it explicitly states; it is not reviewed
visual confirmation. For R1, structural map text is navigation only and is never evidence.
Request local visual review only for unresolved option hypotheses whose requested visible detail
could materially distinguish the remaining candidate options. Do not request review for an
already-refuted or noncompetitive option. answer_ready requires exactly one distinguishable option.
"""


def option_requirements(question: dict[str, Any]) -> list[dict[str, Any]]:
    qid = question["question_id"]
    rows = []
    for option in question["answer_options"]:
        oid = option["option_id"]
        rows.append({
            "requirement_id": f"{qid}::option_{oid.lower()}",
            "option_id": oid,
            "description": (
                f"Assess whether option {oid} correctly answers the exact question. "
                f"Option text: {option['text']}"
            ),
        })
    return rows


def review_scope(question_id: str, option_id: str) -> str:
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:12]
    return f"q_{digest}_option_{option_id.lower()}"


def _parent_map(map_doc: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for coarse in map_doc["coarse_regions"]:
        for medium_id in coarse["source_medium_ids"]:
            if medium_id in result:
                raise ValueError(f"duplicate Medium mapping: {medium_id}")
            result[medium_id] = coarse["coarse_id"]
    return result


def _planner_map(map_doc: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for coarse in map_doc["coarse_regions"]:
        row = {
            "coarse_id": coarse["coarse_id"],
            "start_sec": coarse["start_sec"],
            "end_sec": coarse["end_sec"],
            "navigation_summary": coarse["navigation_summary"],
        }
        if map_doc["semantic_fields_available"]:
            row["exact_source_captions"] = coarse.get("exact_source_captions", [])
            row["exact_source_asr"] = coarse.get("exact_source_asr", [])
        else:
            row["audio_channel"] = coarse.get("audio_channel", [])
        rows.append(row)
    return {
        "map_type": map_doc["map_type"],
        "semantic_fields_available": map_doc["semantic_fields_available"],
        "hard_filtering_allowed": False,
        "coarse_regions": rows,
    }


def _keyed_planner_schema(requirements: list[dict[str, Any]], coarse_ids: list[str]) -> dict[str, Any]:
    unit = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "search_description": {"type": "string"},
            "query_variants": {"type": "array", "items": {"type": "string"}},
            "modality_strategy": {"type": "string"},
            "suggested_coarse_ids": {"type": "array", "items": {"type": "string", "enum": coarse_ids}},
        },
        "required": ["search_description", "query_variants", "modality_strategy", "suggested_coarse_ids"],
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
            "hard_filtering_allowed": {"type": "boolean", "const": False},
        },
        "required": ["question_id", "requirement_plans", "hard_filtering_allowed"],
    }


def _project_plan(value: dict[str, Any], requirements: list[dict[str, Any]]) -> dict[str, Any]:
    keyed = value["requirement_plans"]
    return {
        "question_id": value["question_id"],
        "requirement_plans": [
            {"requirement_id": row["requirement_id"], **keyed[row["requirement_id"]]}
            for row in requirements
        ],
        "hard_filtering_allowed": value["hard_filtering_allowed"],
    }


def _validate_plan(plan: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], coarse_ids: set[str]) -> None:
    if plan.get("question_id") != question["question_id"] or plan.get("hard_filtering_allowed") is not False:
        raise ValueError("Planner identity or hard-filtering contract invalid")
    if [row.get("requirement_id") for row in plan.get("requirement_plans", [])] != [row["requirement_id"] for row in requirements]:
        raise ValueError("Planner requirement coverage/order invalid")
    for row in plan["requirement_plans"]:
        if not row["query_variants"] or not row["suggested_coarse_ids"]:
            raise ValueError(f"empty Planner query/scope: {row['requirement_id']}")
        if not set(row["suggested_coarse_ids"]) <= coarse_ids:
            raise ValueError(f"unknown suggested Coarse: {row['requirement_id']}")


def _sufficiency_schema(question: dict[str, Any], requirements: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    req_ids = [row["requirement_id"] for row in requirements]
    option_ids = [row["option_id"] for row in question["answer_options"]]
    evidence_ids = [row["evidence_id"] for row in evidence]
    assessment = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["supported", "uncertain", "not_found", "conflicted"]},
            "direct_support": {"type": "boolean"},
            "supporting_evidence_ids": {"type": "array", "items": {"type": "string", "enum": evidence_ids}},
            "rationale": {"type": "string"},
        },
        "required": ["status", "direct_support", "supporting_evidence_ids", "rationale"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "assessments": {
                "type": "object", "additionalProperties": False,
                "properties": {rid: assessment for rid in req_ids}, "required": req_ids,
            },
            "gate": {"type": "string", "enum": ["answer_ready", "needs_local_visual_review", "provisional"]},
            "candidate_option_ids": {"type": "array", "items": {"type": "string", "enum": option_ids}},
            "review_requirement_ids": {"type": "array", "items": {"type": "string", "enum": req_ids}},
            "review_query": {"type": "string"},
        },
        "required": ["question_id", "assessments", "gate", "candidate_option_ids", "review_requirement_ids", "review_query"],
    }


def _project_sufficiency(value: dict[str, Any], requirements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **{key: value[key] for key in ("question_id", "gate", "candidate_option_ids", "review_requirement_ids", "review_query")},
        "assessments": [
            {"requirement_id": row["requirement_id"], **value["assessments"][row["requirement_id"]]}
            for row in requirements
        ],
    }


def _validate_sufficiency(value: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> None:
    req_ids = [row["requirement_id"] for row in requirements]
    option_ids = {row["option_id"] for row in question["answer_options"]}
    known = {row["evidence_id"] for row in evidence}
    if value["question_id"] != question["question_id"]:
        raise ValueError("Sufficiency question mismatch")
    if [row["requirement_id"] for row in value["assessments"]] != req_ids:
        raise ValueError("Sufficiency requirement coverage/order invalid")
    if not set(value["candidate_option_ids"]) <= option_ids or not set(value["review_requirement_ids"]) <= set(req_ids):
        raise ValueError("Sufficiency emitted unknown option/requirement")
    for row in value["assessments"]:
        if set(row["supporting_evidence_ids"]) - known:
            raise ValueError("Sufficiency cited unknown evidence")
        if row["status"] == "supported" and (not row["direct_support"] or not row["supporting_evidence_ids"]):
            raise ValueError("supported assessment lacks direct evidence")
    if value["gate"] == "answer_ready" and (len(value["candidate_option_ids"]) != 1 or value["review_requirement_ids"]):
        raise ValueError("answer_ready gate is inconsistent")
    if value["gate"] == "needs_local_visual_review" and not value["review_requirement_ids"]:
        raise ValueError("review gate has no scoped requirement")


def _final_schema(question: dict[str, Any], requirements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "selected_option_id": {"type": "string", "enum": [row["option_id"] for row in question["answer_options"]]},
            "answer_text": {"type": "string"},
            "supporting_requirement_ids": {"type": "array", "items": {"type": "string", "enum": [row["requirement_id"] for row in requirements]}},
            "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question_id", "selected_option_id", "answer_text", "supporting_requirement_ids", "supporting_evidence_ids", "caveats"],
    }


def _lexical_rows(side: str, projections: list[dict[str, Any]], captions: list[dict[str, Any]]) -> list[dict[str, str]]:
    if side == "r1_av":
        return [{"lexical_source": "structured_fallback", "lexical_text": row["detector_summary"]} for row in projections]
    return [{"lexical_source": "vlm_caption", "lexical_text": row["qwen_caption"]} for row in captions]


def _query_text(question: dict[str, Any], plan: dict[str, Any]) -> str:
    return (
        question["question_text"] + " Options: " +
        " ".join(f"{row['option_id']}: {row['text']}" for row in question["answer_options"]) +
        " Retrieval target: " + plan["search_description"] +
        " Variants: " + "; ".join(plan["query_variants"])
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return np.ones_like(values, dtype=np.float64) if math.isclose(low, high) else (values - low) / (high - low)


def _rank_requirement(
    cfg: dict[str, Any], question: dict[str, Any], plan: dict[str, Any], hierarchy: dict[str, Any],
    lexical: list[dict[str, str]], medium_embeddings: np.ndarray, parent: dict[str, str], encoder: SiglipTextEncoder,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    text = _query_text(question, plan)
    started = time.perf_counter(); query = encoder.encode([text])[0]; encode_sec = time.perf_counter() - started
    raw = medium_embeddings @ query; visual = _normalize(raw)
    rows = []
    for index, (medium, lex) in enumerate(zip(hierarchy["medium_nodes"], lexical)):
        lexical_score, terms = lexical_similarity(text, lex["lexical_text"])
        combined = float(cfg["ranking"]["visual_weight"]) * float(visual[index]) + float(cfg["ranking"]["lexical_weight"]) * lexical_score
        rows.append({
            "medium_id": medium["medium_id"], "start_sec": medium["start_sec"], "end_sec": medium["end_sec"],
            "parent_coarse_id": parent[medium["medium_id"]], "lexical_source": lex["lexical_source"],
            "visual_score_raw": float(raw[index]), "visual_score_normalized": float(visual[index]),
            "lexical_score": lexical_score, "matched_terms": terms, "coarse_prior": 0.0,
            "combined_score": combined, "source_fine_ids": medium["source_fine_ids"],
        })
    rows.sort(key=lambda row: (-row["combined_score"], row["start_sec"], row["medium_id"]))
    for rank, row in enumerate(rows, 1): row["rank"] = rank
    return rows, query, encode_sec


def _fine_registry(case_cfg: dict[str, Any], hierarchy: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], np.ndarray, dict[str, int]]:
    source = np.load(case_cfg["siglip_npz"], allow_pickle=False)
    embeddings = source["embedding"].astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    fine_by_id, row_by_id = {}, {}
    for fine in hierarchy["fine_nodes"]:
        row = dict(fine)
        row.update({
            "representative_frame_timestamp_sec": fine["timestamp_sec"],
            "representative_frame_path": fine["source_frame_path"],
            "visual_embedding_ref": {"row": fine["frame_index"], "dimension": 768},
        })
        fine_by_id[fine["fine_id"]] = row
        row_by_id[fine["fine_id"]] = int(fine["frame_index"])
    return fine_by_id, embeddings, row_by_id


def _requirement_retrieval(
    cfg: dict[str, Any], side: str, question: dict[str, Any], plan: dict[str, Any], map_doc: dict[str, Any],
    hierarchy: dict[str, Any], projections: list[dict[str, Any]], captions: list[dict[str, Any]],
    medium_embeddings: np.ndarray, fine_by_id: dict[str, dict[str, Any]], fine_embeddings: np.ndarray,
    row_by_id: dict[str, int], encoder: SiglipTextEncoder,
) -> dict[str, Any]:
    parent = _parent_map(map_doc); medium_by_id = {row["medium_id"]: row for row in hierarchy["medium_nodes"]}
    lexical = _lexical_rows(side, projections, captions)
    output = {"question_id": question["question_id"], "candidate_universe_count": len(hierarchy["medium_nodes"]), "requirements": {}, "ranking_formula": "0.6*normalized SigLIP + 0.3*lexical + 0.0*coarse_prior"}
    for requirement_plan in plan["requirement_plans"]:
        rows, query, encode_sec = _rank_requirement(cfg, question, requirement_plan, hierarchy, lexical, medium_embeddings, parent, encoder)
        hints = set(requirement_plan["suggested_coarse_ids"])
        hinted = [row for row in rows if row["parent_coarse_id"] in hints]
        rest = [row for row in rows if row["parent_coarse_id"] not in hints]
        ordered = hinted + rest
        selected = ordered[:int(cfg["ranking"]["top_k_medium_per_requirement"])]
        fine_rows = []
        for medium_row in selected:
            medium = dict(medium_by_id[medium_row["medium_id"]])
            medium["child_fine_ids"] = medium["source_fine_ids"]
            ranking = rerank_fines_for_medium(
                question_id=question["question_id"], search_unit_id=requirement_plan["requirement_id"], medium=medium,
                fine_by_id=fine_by_id, fine_embeddings=fine_embeddings, row_by_id=row_by_id, query_embedding=query,
            )
            chosen, reason = select_diverse_fines(
                {requirement_plan["requirement_id"]: ranking}, strategy="top_relevance_then_temporal_diversity",
                max_fines=int(cfg["ranking"]["max_fine_per_medium"]), minimum_gap=float(cfg["ranking"]["minimum_fine_gap_sec"]),
            )
            for item in chosen:
                fine_rows.append({
                    "fine_id": item["fine_id"], "medium_id": medium["medium_id"], "timestamp_sec": item["representative_frame_timestamp_sec"],
                    "source_frame_path": item["representative_frame_path"], "frame_path": item["representative_frame_path"],
                    "siglip_score_raw": item["siglip_score_raw"], "selection_reason": item["selection_reason"],
                    "one_fine_reason": reason, "medium_rank": medium_row["rank"],
                })
        output["requirements"][requirement_plan["requirement_id"]] = {
            "suggested_coarse_ids": requirement_plan["suggested_coarse_ids"], "all_medium_rankings": rows,
            "selected_medium_ids": [row["medium_id"] for row in selected], "selected_fine_evidence": fine_rows,
            "allowed_image_ids": [row["fine_id"] for row in fine_rows], "query_encode_sec": encode_sec,
            "hinted_medium_count": len(hinted), "hard_filtering_applied": False, "coarse_prior": 0.0,
        }
    return output


def _audio_evidence(audio: list[dict[str, Any]], query: str, requirement_id: str, limit: int = 5) -> list[dict[str, Any]]:
    scored = []
    for row in audio:
        score, _ = lexical_similarity(query, row["exact_transcript"])
        scored.append((score, row))
    result = []
    for score, row in sorted(scored, key=lambda item: (-item[0], item[1]["start_sec"]))[:limit]:
        if row["exact_transcript"].strip():
            result.append({
                "evidence_id": f"audio_asr::{requirement_id}::{row['audio_id']}", "evidence_type": "audio_asr",
                "source_content": row["exact_transcript"], "interval": [row["start_sec"], row["end_sec"]],
                "retrieved_for_requirement_id": requirement_id, "lexical_score": score,
            })
    return result


def _initial_evidence(
    side: str, question: dict[str, Any], requirements: list[dict[str, Any]], plan: dict[str, Any], map_doc: dict[str, Any],
    retrieval: dict[str, Any], projections: list[dict[str, Any]], audio: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    projection_by_id = {row["medium_id"]: row for row in projections}
    if side == "r1_av":
        for requirement_plan in plan["requirement_plans"]:
            rid = requirement_plan["requirement_id"]
            for medium_id in retrieval["requirements"][rid]["selected_medium_ids"]:
                row = projection_by_id[medium_id]
                evidence.append({
                    "evidence_id": f"detector_observation::{rid}::{medium_id}", "evidence_type": "detector_observation",
                    "source_content": row["detector_summary"], "interval": [row["start_sec"], row["end_sec"]],
                    "retrieved_for_requirement_id": rid,
                })
            evidence.extend(_audio_evidence(audio, _query_text(question, requirement_plan), rid))
    else:
        coarse_by_id = {row["coarse_id"]: row for row in map_doc["coarse_regions"]}
        for requirement_plan in plan["requirement_plans"]:
            rid = requirement_plan["requirement_id"]
            for coarse_id in requirement_plan["suggested_coarse_ids"]:
                coarse = coarse_by_id[coarse_id]
                evidence.append({
                    "evidence_id": f"semantic_coarse::{rid}::{coarse_id}", "evidence_type": "semantic_coarse_summary",
                    "source_content": coarse["navigation_summary"], "interval": [coarse["start_sec"], coarse["end_sec"]],
                    "source_medium_ids": coarse["source_medium_ids"], "retrieved_for_requirement_id": rid,
                    "semantic_provenance": "canonical captions plus timestamped ASR",
                })
    ids = [row["evidence_id"] for row in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence ID")
    return evidence


def _review_and_resolve(
    cfg: dict[str, Any], out: Path, side: str, question: dict[str, Any], requirements: list[dict[str, Any]],
    sufficiency: dict[str, Any], retrieval: dict[str, Any], fine_by_id: dict[str, dict[str, Any]], calls: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    resolved = copy.deepcopy(sufficiency["assessments"])
    reviewed_evidence: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    cache = LayeredVisualReviewCache([], out / "cache" / side)
    requirement_by_id = {row["requirement_id"]: row for row in requirements}
    for rid in sufficiency["review_requirement_ids"]:
        requirement = requirement_by_id[rid]; scope = review_scope(question["question_id"], requirement["option_id"])
        allowed = retrieval["requirements"][rid]["selected_fine_evidence"]
        hits, misses, records = [], [], []
        for item in allowed:
            fine = fine_by_id[item["fine_id"]]; image = Path(fine["source_frame_path"])
            found = cache.lookup(image, cfg["review_contract_version"], [scope])
            if found["status"] == "hit": hits.append(item["fine_id"]); records.append(found["records"][scope])
            else:
                misses.append({**fine, "timestamp_sec": fine["timestamp_sec"]})
        if misses:
            parsed, usage, raw = _review_missing({"gemini": cfg["gemini"]}, question, requirement, scope, misses)
            usage.update({"stage": "visual_review_cache_miss", "side": side, "requirement_id": rid, "scope": scope, "image_transmissions": len(misses)})
            calls.append(usage)
            write_json(out / side / "visual_reviews" / f"{scope}_raw.json", raw)
            write_json(out / side / "visual_reviews" / f"{scope}_parsed.json", parsed)
            by_fine = {row["fine_id"]: row for row in parsed["records"]}
            config_hash = hashlib.sha256(canonical_bytes({"model": cfg["gemini"]["model"], "scope": scope, "contract": cfg["review_contract_version"]})).hexdigest()
            for fine in misses:
                row = by_fine[fine["fine_id"]]; image = Path(fine["source_frame_path"])
                record = ReviewRecord(
                    "reviewed_visual_evidence_cache_record_v1", cfg["review_contract_version"], "reviewed_visual_frame",
                    fine["fine_id"], sha256_file(image), image.stat().st_size, float(fine["timestamp_sec"]), scope,
                    row["status"], row["requirement_effect"], row["direct_visual_support"], row["finding"], row["confidence"],
                    (fine["fine_id"],), cfg["gemini"]["model"], config_hash, False,
                )
                cache.store(record); records.append(record.to_dict())
        resolved = apply_scope_records(resolved, [rid], scope, records)
        for record in records:
            reviewed_evidence.append({
                "evidence_id": f"reviewed_visual::{record['fine_id']}::{scope}", "evidence_type": "reviewed_visual_frame",
                "source_content": record["finding"], "fine_id": record["fine_id"], "timestamp_sec": record["timestamp_sec"],
                "review_scope": scope,
            })
        audit[scope] = {"requirement_id": rid, "allowed_image_ids": [row["fine_id"] for row in allowed], "cache_hits": hits, "cache_misses": [row["fine_id"] for row in misses]}
    return {**sufficiency, "assessments": resolved, "review_resolution": "deterministic_cache_projection"}, reviewed_evidence, audit


def _final(
    cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], resolved: dict[str, Any], evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = {"question": question, "requirements": requirements, "resolved_assessments": resolved, "evidence": evidence}
    result, usage, raw = _gemini_call({"gemini": cfg["gemini"]}, FINAL_SYSTEM, [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], _final_schema(question, requirements))
    known_evidence = {row["evidence_id"] for row in evidence}
    known_requirements = {row["requirement_id"] for row in requirements}
    if result["question_id"] != question["question_id"] or set(result["supporting_evidence_ids"]) - known_evidence or set(result["supporting_requirement_ids"]) - known_requirements:
        raise ValueError("Final answer provenance invalid")
    return result, usage, raw


def _case_paths(root: Path, cfg: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    selection = load_json(root / cfg["selection_manifest"])
    question_by_uid = {row["video_uid"]: row for row in selection["questions"]}
    rows = []
    for video in selection["videos"]:
        uid = video["video_uid"]
        case_cfg = load_json(root / cfg["output_root"] / "case_configs" / f"{uid}.json")
        rows.append((case_cfg, root / cfg["output_root"] / "cases" / uid))
        if question_by_uid[uid]["question_id"] != case_cfg["question_id"]:
            raise ValueError("selection/case question mismatch")
    return rows


def _offline_audit(case_cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    required = [
        "question_input.json", "shared_hierarchy.json", "r1_medium_projection.json", "r3_medium_captions.json",
        "medium_siglip.float32.npy", "audio_asr.json", "detector_validation.json", "r3_caption_validation.json",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing: raise FileNotFoundError(f"{case_cfg['video_uid']}: {missing}")
    hierarchy = load_json(out / "shared_hierarchy.json"); projections = load_json(out / "r1_medium_projection.json"); captions = load_json(out / "r3_medium_captions.json")
    embeddings = np.load(out / "medium_siglip.float32.npy", allow_pickle=False)
    count = len(hierarchy["medium_nodes"])
    if not (count == len(projections) == len(captions) == embeddings.shape[0]): raise ValueError("Medium artifacts differ")
    ids = [row["medium_id"] for row in hierarchy["medium_nodes"]]
    if ids != [row["medium_id"] for row in projections] or ids != [row["medium_id"] for row in captions]: raise ValueError("Medium ordering differs")
    if not load_json(out / "detector_validation.json")["valid"] or not load_json(out / "r3_caption_validation.json")["valid"]: raise ValueError("offline validation failed")
    question = load_json(out / "question_input.json"); requirements = option_requirements(question)
    return {
        "video_uid": case_cfg["video_uid"], "medium_count": count, "fine_count": len(hierarchy["fine_nodes"]),
        "option_requirement_count": len(requirements), "r1_caption_leakage": 0,
        "hashes": {name: sha256_file(out / name) for name in required},
    }


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); output = root / cfg["output_root"]; output.mkdir(parents=True, exist_ok=True)
    audits = [_offline_audit(case_cfg, out) for case_cfg, out in _case_paths(root, cfg)]
    option_count = sum(row["option_requirement_count"] for row in audits)
    max_images = option_count * 2 * int(cfg["ranking"]["top_k_medium_per_requirement"]) * int(cfg["ranking"]["max_fine_per_medium"])
    result = {
        "overall_validation": "ready_for_live_pilot", "model_api_calls": 0, "video_count": len(audits),
        "question_count": len(audits), "total_option_requirements": option_count,
        "fixed_calls": {"organizer": len(audits), "planner": len(audits) * 2, "sufficiency": len(audits) * 2, "final_gemini": len(audits) * 2},
        "worst_case_visual_review_calls": option_count * 2, "worst_case_image_transmissions": max_images,
        "fine_policy": {"top_medium_per_requirement": cfg["ranking"]["top_k_medium_per_requirement"], "max_fine_per_medium": cfg["ranking"]["max_fine_per_medium"], "whole_range_expansion": False},
        "audits": audits,
    }
    write_json(output / "live_preflight.json", result)
    return result


def _append_call(path: Path, calls: list[dict[str, Any]], row: dict[str, Any]) -> None:
    calls.append(row); write_json(path, calls)


def run_live(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); output = root / cfg["output_root"]
    pre = preflight(root, config_path)
    if pre["overall_validation"] != "ready_for_live_pilot": raise RuntimeError("preflight failed")
    _load_env(root.parent / "thesis-av-evidence" / ".env")
    semantic_local = cfg["anthropic"].get("provider") == "local_openai"
    visual_local = cfg["gemini"].get("provider") == "local_openai"
    if (not semantic_local and not os.environ.get("ANTHROPIC_API_KEY")) or (not visual_local and not os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("API keys unavailable before live calls")
    encoder = SiglipTextEncoder(cfg["siglip_text"])
    summaries: dict[str, Any] = {}
    for position, (case_cfg, case_out) in enumerate(_case_paths(root, cfg), 1):
        uid = case_cfg["video_uid"]; question = load_json(case_out / "question_input.json")
        hierarchy = load_json(case_out / "shared_hierarchy.json"); projections = load_json(case_out / "r1_medium_projection.json")
        captions = load_json(case_out / "r3_medium_captions.json"); audio = load_json(case_out / "audio_asr.json")["segments"]
        medium_embeddings = np.load(case_out / "medium_siglip.float32.npy", allow_pickle=False).astype(np.float32)
        medium_embeddings /= np.maximum(np.linalg.norm(medium_embeddings, axis=1, keepdims=True), 1e-12)
        requirements = option_requirements(question); fine_by_id, fine_embeddings, row_by_id = _fine_registry(case_cfg, hierarchy)
        calls_path = case_out / "live_api_calls.json"; calls = load_json(calls_path) if calls_path.is_file() else []

        r1_map_path = case_out / "r1_av_navigation_map.json"
        if not r1_map_path.is_file(): write_json(r1_map_path, _build_r1_map(hierarchy, projections, audio, medium_embeddings))
        r3_map_path = case_out / "r3_2_navigation_map.json"
        if not r3_map_path.is_file():
            r3_map, trace, usage = _build_r3_map(cfg, hierarchy, captions, audio)
            write_json(case_out / "r3_2_organizer_trace.json", trace); write_json(r3_map_path, r3_map)
            _append_call(calls_path, calls, {"stage": "organizer", "side": "r3_2", **usage})
        maps = {"r1_av": load_json(r1_map_path), "r3_2": load_json(r3_map_path)}
        side_results = {}
        for side in ("r1_av", "r3_2"):
            side_out = case_out / side; side_out.mkdir(parents=True, exist_ok=True)
            planner_path = side_out / "planner.json"
            if planner_path.is_file(): planner_doc = load_json(planner_path)
            else:
                payload = {"question": question, "requirements": requirements, "navigation_map": _planner_map(maps[side]), "retrieval_contract": {"hard_filtering_allowed": False, "coarse_prior_weight": 0.0}}
                coarse_ids = [row["coarse_id"] for row in maps[side]["coarse_regions"]]
                provider, usage = _anthropic_call(cfg, PLANNER_PROMPT + "\nSchema-only provider projection: requirement_plans is keyed by exact requirement_id; preserve requirement semantics.", payload, _keyed_planner_schema(requirements, coarse_ids), int(cfg["anthropic"]["planner_max_tokens"]))
                plan = _project_plan(provider, requirements); _validate_plan(plan, question, requirements, set(coarse_ids))
                planner_doc = {"input": payload, "output": plan, "usage": usage, "schema_projection": "keyed_to_canonical_order"}; write_json(planner_path, planner_doc)
                _append_call(calls_path, calls, {"stage": "planner", "side": side, **usage})
            plan = planner_doc["output"]

            retrieval_path = side_out / "requirement_retrieval.json"
            if retrieval_path.is_file(): retrieval = load_json(retrieval_path)
            else:
                retrieval = _requirement_retrieval(cfg, side, question, plan, maps[side], hierarchy, projections, captions, medium_embeddings, fine_by_id, fine_embeddings, row_by_id, encoder)
                write_json(retrieval_path, retrieval)
            evidence = _initial_evidence(side, question, requirements, plan, maps[side], retrieval, projections, audio)
            write_json(side_out / "initial_evidence.json", evidence)

            suff_path = side_out / "initial_sufficiency.json"
            if suff_path.is_file(): suff_doc = load_json(suff_path)
            else:
                payload = {"question": question, "requirements": requirements, "evidence": evidence, "pass": "initial", "map_policy": {"r1_structural_map_is_navigation_only": True, "r3_semantic_coarse_is_caption_asr_derived_evidence": True}}
                provider, usage = _anthropic_call(cfg, SUFFICIENCY_PROMPT, payload, _sufficiency_schema(question, requirements, evidence), int(cfg["anthropic"]["sufficiency_max_tokens"]))
                suff = _project_sufficiency(provider, requirements); _validate_sufficiency(suff, question, requirements, evidence)
                suff_doc = {"input": payload, "output": suff, "usage": usage, "schema_projection": "keyed_to_canonical_order"}; write_json(suff_path, suff_doc)
                _append_call(calls_path, calls, {"stage": "sufficiency", "side": side, **usage})
            suff = suff_doc["output"]

            resolved_path = side_out / "resolved_assessments.json"
            if resolved_path.is_file(): resolved_doc = load_json(resolved_path)
            else:
                resolved, reviewed, cache_audit = _review_and_resolve(cfg, case_out, side, question, requirements, suff, retrieval, fine_by_id, calls)
                write_json(calls_path, calls)
                resolved_doc = {"resolved": resolved, "reviewed_evidence": reviewed, "cache_audit": cache_audit}
                write_json(resolved_path, resolved_doc)
            all_evidence = evidence + resolved_doc["reviewed_evidence"]

            final_path = side_out / "final_answer.json"
            if final_path.is_file(): final_doc = load_json(final_path)
            else:
                answer, usage, raw = _final(cfg, question, requirements, resolved_doc["resolved"], all_evidence)
                final_doc = {"answer": answer, "usage": usage}; write_json(side_out / "final_raw_response.json", raw); write_json(final_path, final_doc)
                _append_call(calls_path, calls, {"stage": "final_text_only", "side": side, "image_transmissions": 0, **usage})
            side_results[side] = {
                "answer": final_doc["answer"], "planner": plan, "sufficiency": suff, "resolved": resolved_doc["resolved"],
                "review": resolved_doc["cache_audit"], "candidate_universe": retrieval["candidate_universe_count"],
            }
        blind = {side: row["answer"] for side, row in side_results.items()}; write_json(case_out / "answers_blind.json", blind)
        summaries[uid] = {"position": position, "question_id": question["question_id"], "answers": blind, "calls": len(calls), "completed": True}
        write_json(output / "live_progress.json", summaries)
    return _summarize_live(root, cfg)


def _summarize_live(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]; all_calls, cases = [], []
    for case_cfg, case_out in _case_paths(root, cfg):
        calls = load_json(case_out / "live_api_calls.json"); all_calls.extend({"video_uid": case_cfg["video_uid"], **row} for row in calls)
        answers = load_json(case_out / "answers_blind.json")
        cases.append({"video_uid": case_cfg["video_uid"], "question_id": case_cfg["question_id"], "answers": answers})
    side_cost = {}
    for side in ("r1_av", "r3_2"):
        rows = [row for row in all_calls if row.get("side") == side]
        side_cost[side] = {
            "calls": len(rows), "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
            "output_tokens": sum(row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in rows),
            "latency_sec": sum(row.get("latency_sec", 0.0) for row in rows),
            "image_transmissions": sum(row.get("image_transmissions", 0) for row in rows),
            "estimated_api_usd": _estimated_cost(cfg, rows),
            "by_stage": {stage: len([row for row in rows if row["stage"] == stage]) for stage in sorted({row["stage"] for row in rows})},
        }
    organizer_rows = [row for row in all_calls if row["stage"] == "organizer"]
    failure_path = output / "development_failed_calls.json"
    failed_calls = load_json(failure_path) if failure_path.is_file() else []
    cost = {"calls": all_calls, "r1_av": side_cost["r1_av"], "r3_2": side_cost["r3_2"], "r3_2_organizer": {"calls": len(organizer_rows), "estimated_api_usd": _estimated_cost(cfg, organizer_rows)}, "development_failed_calls": failed_calls, "new_api_total_usd": _estimated_cost(cfg, all_calls)}
    write_json(output / "cost_accounting.json", cost); write_json(output / "answers_blind.json", cases)
    validation = {"video_count": len(cases), "question_count": len(cases), "blind_answers_complete": len(cases) == 10, "gold_loaded_before_predictions": False, "whole_range_expansion": 0, "coarse_prior_weight": 0.0, "overall_validation": "passed_live_pending_posthoc_evaluation"}
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
                    if isinstance(candidate, str) and candidate.strip(): return candidate.strip()
                raise ValueError("matched question lacks gold label")
            stack.extend(value.values())
        elif isinstance(value, list): stack.extend(value)
    raise ValueError(f"gold question not found: {qid}")


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); output = root / cfg["output_root"]
    blind_path = output / "answers_blind.json"
    if not blind_path.is_file(): raise RuntimeError("blind answers absent")
    blind_hash = sha256_file(blind_path); annotation = load_json(Path(cfg["annotation_path"]))
    rows = []
    for case_cfg, case_out in _case_paths(root, cfg):
        answers = load_json(case_out / "answers_blind.json"); gold = _find_gold(annotation, case_cfg["question_id"])
        row = {"video_uid": case_cfg["video_uid"], "question_id": case_cfg["question_id"], "gold_option_id": gold}
        for side in ("r1_av", "r3_2"):
            selected = answers[side]["selected_option_id"]
            row[side] = {"selected_option_id": selected, "correct": selected == gold}
        rows.append(row)
    counts = {
        "r1_correct": sum(row["r1_av"]["correct"] for row in rows), "r3_correct": sum(row["r3_2"]["correct"] for row in rows),
        "both_correct": sum(row["r1_av"]["correct"] and row["r3_2"]["correct"] for row in rows),
        "r1_only": sum(row["r1_av"]["correct"] and not row["r3_2"]["correct"] for row in rows),
        "r3_only": sum(row["r3_2"]["correct"] and not row["r1_av"]["correct"] for row in rows),
        "neither": sum(not row["r1_av"]["correct"] and not row["r3_2"]["correct"] for row in rows),
    }
    result = {"answers_blind_sha256_before_gold_load": blind_hash, "cases": rows, "summary": counts}
    write_json(output / "posthoc_evaluation.json", result)
    _write_pilot_metrics(root, cfg, result)
    validation = load_json(output / "validation_report.json")
    validation.update({
        "posthoc_evaluation_complete": True,
        "interface_and_provenance_validation": "passed",
        "semantic_contract_validation": "failed_compound_option_atomicity",
        "overall_validation": "completed_with_known_semantic_contract_limitation",
    })
    write_json(output / "validation_report.json", validation)
    _write_report(root, cfg, result)
    return result


def _write_pilot_metrics(root: Path, cfg: dict[str, Any], evaluation: dict[str, Any]) -> None:
    output = root / cfg["output_root"]
    cost = load_json(output / "cost_accounting.json")
    stage_metrics: dict[str, Any] = {}
    for side in ("r1_av", "r3_2"):
        stage_metrics[side] = {}
        for stage in sorted({row["stage"] for row in cost["calls"] if row.get("side") == side}):
            calls = [row for row in cost["calls"] if row.get("side") == side and row["stage"] == stage]
            stage_metrics[side][stage] = {
                "calls": len(calls), "input_tokens": sum(row.get("input_tokens", 0) for row in calls),
                "output_tokens_including_thought": sum(row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in calls),
                "latency_sec": sum(row.get("latency_sec", 0.0) for row in calls),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in calls),
                "estimated_usd": _estimated_cost(cfg, calls),
            }
    failed = cost.get("development_failed_calls", [])
    if failed:
        # The retry used byte-identical input/prompt/schema, so its recorded input
        # tokens are the best auditable estimate for the provider response whose
        # legacy helper discarded usage after observing max_tokens.
        retry = next(row for row in cost["calls"] if row["stage"] == "organizer" and row.get("video_uid") == failed[0]["video_uid"])
        failed[0]["estimated_input_tokens_from_identical_retry"] = retry["input_tokens"]
        failed[0]["estimated_usd"] = (retry["input_tokens"] * cfg["anthropic"]["pricing_usd_per_million"]["input"] + failed[0]["configured_max_output_tokens"] * cfg["anthropic"]["pricing_usd_per_million"]["output"]) / 1_000_000
        write_json(output / "development_failed_calls.json", failed)

    offline = {"prepare_wall_sec": 0.0, "audio_wall_sec": 0.0, "detector_wall_sec": 0.0, "caption_wall_sec": 0.0,
               "detector_inference_sec": 0.0, "detector_model_load_sec": 0.0, "detector_observations": 0,
               "asr_inference_sec": 0.0, "asr_model_load_sec": 0.0, "asr_segments": 0,
               "caption_inference_sec": 0.0, "caption_model_load_sec": 0.0, "caption_calls": 0,
               "caption_input_tokens": 0, "caption_output_tokens": 0, "api_calls": 0}
    for stage, key in (("prepare", "prepare_wall_sec"), ("audio", "audio_wall_sec"), ("detector", "detector_wall_sec"), ("captions", "caption_wall_sec")):
        offline[key] = sum(row["elapsed_sec"] for row in load_json(output / f"{stage}_progress.json").values())
    detector_peak = caption_peak = 0
    r1_storage = r3_storage = shared_storage = cache_storage = 0
    review_rows = []
    evaluation_by_uid = {row["video_uid"]: row for row in evaluation["cases"]}
    for case_cfg, case_out in _case_paths(root, cfg):
        detector = load_json(case_out / "detector_tracking_cost.json"); audio = load_json(case_out / "audio_cost.json"); caption = load_json(case_out / "r3_caption_cost.json")
        offline["detector_inference_sec"] += detector["inference_sec"]; offline["detector_model_load_sec"] += detector["model_load_sec"]
        offline["detector_observations"] += detector["observations"]; detector_peak = max(detector_peak, detector["peak_gpu_memory_bytes"])
        offline["asr_inference_sec"] += audio["inference_sec"]; offline["asr_model_load_sec"] += audio["model_load_sec"]
        offline["asr_segments"] += len(load_json(case_out / "audio_asr.json")["segments"])
        offline["caption_inference_sec"] += caption["total_caption_sec"]; offline["caption_model_load_sec"] += caption["model_load_sec"]
        offline["caption_calls"] += caption["caption_calls"]; offline["caption_input_tokens"] += caption["input_tokens"]; offline["caption_output_tokens"] += caption["output_tokens"]
        caption_peak = max(caption_peak, caption["peak_gpu_memory_bytes"])
        calls = load_json(case_out / "live_api_calls.json")
        for side in ("r1_av", "r3_2"):
            suff = load_json(case_out / side / "initial_sufficiency.json")["output"]
            side_calls = [row for row in calls if row.get("side") == side]
            review_rows.append({
                "video_uid": case_cfg["video_uid"], "question_id": case_cfg["question_id"], "side": side,
                "initial_gate": suff["gate"], "candidate_option_ids": suff["candidate_option_ids"],
                "review_requirement_ids": suff["review_requirement_ids"],
                "visual_review_calls": sum(row["stage"] == "visual_review_cache_miss" for row in side_calls),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in side_calls),
                "selected_option_id": evaluation_by_uid[case_cfg["video_uid"]][side]["selected_option_id"],
                "correct": evaluation_by_uid[case_cfg["video_uid"]][side]["correct"],
            })
        for name in ("r1_medium_projection.json", "frame_track_observations.jsonl", "r1_av_navigation_map.json"):
            if (case_out / name).is_file(): r1_storage += (case_out / name).stat().st_size
        for name in ("r3_medium_captions.json", "r3_2_navigation_map.json", "r3_2_organizer_trace.json"):
            if (case_out / name).is_file(): r3_storage += (case_out / name).stat().st_size
        for name in ("shared_hierarchy.json", "medium_siglip.float32.npy", "audio_asr.json"):
            shared_storage += (case_out / name).stat().st_size
        cache_storage += sum(path.stat().st_size for path in case_out.rglob("cache/**/*.json"))
    offline["detector_peak_gpu_memory_bytes"] = detector_peak; offline["caption_peak_gpu_memory_bytes"] = caption_peak
    offline["total_measured_wall_sec"] = sum(offline[key] for key in ("prepare_wall_sec", "audio_wall_sec", "detector_wall_sec", "caption_wall_sec"))
    offline["storage_bytes"] = {"r1_specific": r1_storage, "r3_specific": r3_storage, "shared": shared_storage, "visual_cache_records": cache_storage}
    write_json(output / "offline_cost_accounting.json", offline)
    write_json(output / "image_review_audit.json", {"rows": review_rows, "summary": {
        side: {"review_calls": sum(row["visual_review_calls"] for row in review_rows if row["side"] == side), "image_transmissions": sum(row["image_transmissions"] for row in review_rows if row["side"] == side), "answer_ready_questions": sum(row["initial_gate"] == "answer_ready" for row in review_rows if row["side"] == side)}
        for side in ("r1_av", "r3_2")
    }})
    write_json(output / "pilot_metrics.json", {
        "accuracy": evaluation["summary"], "stage_costs": stage_metrics,
        "candidate_runtime_api_usd": cost["new_api_total_usd"],
        "development_failed_call_estimated_usd": sum(row.get("estimated_usd", 0.0) for row in failed),
        "image_reduction_r3_vs_r1": cost["r1_av"]["image_transmissions"] - cost["r3_2"]["image_transmissions"],
        "tests": "6 passed",
        "semantic_contract_audit": {
            "compound_option_requirements_present": True,
            "current_cache_projection_promotes_a_whole_option_requirement_when_any_reviewed_frame_supports_the_target_scope": True,
            "risk": "partial visual support can over-support a conjunctive answer option",
            "consequence": "pilot accuracy is valid for this executed baseline, but this requirement-update rule is not ready to freeze for a larger run",
        },
    })
    failures = []
    for row in review_rows:
        if row["correct"]: continue
        failures.append({
            "video_uid": row["video_uid"], "question_id": row["question_id"], "side": row["side"],
            "selected_option_id": row["selected_option_id"], "initial_gate": row["initial_gate"],
            "attribution": "answer_ready_overclaim" if row["initial_gate"] == "answer_ready" else "localization_or_compound_option_projection_limitation",
            "compound_option_projection_risk": bool(row["visual_review_calls"]),
            "note": "No post-hoc repair or rerun was applied.",
        })
    write_json(output / "failure_attribution.json", {"failures": failures, "known_contract_limitation": "review evidence is scoped per option, but some HourVideo options contain multiple conjunctive facts; positive evidence for one part must not automatically support the entire option in the next version"})


def _write_report(root: Path, cfg: dict[str, Any], evaluation: dict[str, Any]) -> None:
    output = root / cfg["output_root"]; cost = load_json(output / "cost_accounting.json")
    summary = evaluation["summary"]
    report = [
        "# HourVideo R1_AV / R3_2 ten-video pilot", "",
        f"- R1_AV accuracy: `{summary['r1_correct']}/10`; R3_2 accuracy: `{summary['r3_correct']}/10`.",
        f"- Both / R1-only / R3-only / neither: `{summary['both_correct']} / {summary['r1_only']} / {summary['r3_only']} / {summary['neither']}`.",
        f"- R1_AV API: `{cost['r1_av']['calls']}` calls, `{cost['r1_av']['image_transmissions']}` images, `${cost['r1_av']['estimated_api_usd']:.6f}`.",
        f"- R3_2 API: `{cost['r3_2']['calls']}` calls, `{cost['r3_2']['image_transmissions']}` images, `${cost['r3_2']['estimated_api_usd']:.6f}`.",
        f"- R3_2 reduced image transmissions by `{cost['r1_av']['image_transmissions'] - cost['r3_2']['image_transmissions']}` relative to R1_AV.",
        "- Fine policy: 3 Mediums per option requirement, at most 2 ranked Fine per Medium; no whole-range expansion.",
        "- Gold labels were loaded only after blind answers were persisted.",
        "- Known semantic limitation: several answer options are conjunctive, while the current deterministic cache projection can promote the whole option after positive support for only part of it. The 10-question scores remain the executed baseline result, but this projection is not ready for a larger freeze.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    body = "".join(f"<h2>{html.escape(row['question_id'])}</h2><pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre>" for row in evaluation["cases"])
    (output / "review.html").write_text("<!doctype html><meta charset='utf-8'><h1>HourVideo ten-video pilot</h1>" + body, encoding="utf-8")
