from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.r1_r3_v2_map_aware_all_medium_retrieval_pair import core as shared


EXPERIMENT = "egopolice_r1_av_dual_channel_retrieval_smoke_v1"


def source_paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / f"configs/experiments/{EXPERIMENT}.json",
        "questions": root / "configs/experiments/planner_medium_retrieval_v1/questions_226.json",
        "r1_av_map": root / "outputs/experiments/r1_av_structural_audio_timeline_v1/r1_av_navigation_map.json",
        "r1_av_validation": root / "outputs/experiments/r1_av_structural_audio_timeline_v1/validation_report.json",
        "canonical_index": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json",
        "r1_index": root / "outputs/experiments/egopolice_r1_structured_index_organizer_canary_v1/hierarchical_index_r1_caption_free.json",
        "projection": root / "outputs/experiments/egopolice_two_stage_organizer_r1_canary_v1/unified_medium_envelopes.json",
        "medium_embeddings": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/embeddings/medium_pooled_siglip.float32.npy",
        "diagnostic_reference": root / "configs/experiments/egopolice_r1_structural_map_navigation_smoke_v1.json",
    }


def load_sources(root: Path) -> dict[str, Any]:
    paths = source_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing_sources:{missing}")
    return {
        "paths": paths,
        "config": shared.load_json(paths["config"]),
        "questions": shared.load_json(paths["questions"])["questions"],
        "r1_av_map": shared.load_json(paths["r1_av_map"]),
        "r1_av_validation": shared.load_json(paths["r1_av_validation"]),
        "base": shared.load_json(paths["canonical_index"]),
        "r1_index": shared.load_json(paths["r1_index"]),
        "projection": shared.load_json(paths["projection"]),
        "references": shared.load_json(paths["diagnostic_reference"])["posthoc_diagnostic_reference"]["queries"],
    }


def planner_map_view(r1_av_map: dict[str, Any]) -> dict[str, Any]:
    visual_rows = [row for row in r1_av_map["timeline_nodes"] if row["node_type"] == "visual_structural"]
    return {
        "map_type": r1_av_map["map_type"],
        "semantic_fields_available": False,
        "av_semantic_fusion_available": False,
        "timeline_nodes": [dict(row) for row in r1_av_map["timeline_nodes"]],
        "coarse_regions": [{
            "coarse_id": row["node_id"],
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "summary": row["navigation_text"],
            "source_medium_ids": row["source_medium_ids"],
            "adapter_derived": True,
            "source_field": "timeline_nodes.visual_structural.navigation_text",
        } for row in visual_rows],
        "has_storyline": False,
        "storyline_events": [],
        "hard_filtering_allowed": False,
        "map_policy": {
            "visual_and_audio_are_parallel_navigation_nodes": True,
            "audio_transcript_means_audible_statement_or_mention_only": True,
            "audio_is_not_visual_confirmation": True,
            "audio_does_not_prove_physical_event_occurrence": True,
            "all_mediums_remain_eligible": True,
            "all_audio_nodes_remain_eligible": True,
            "coarse_prior_weight": 0.0,
            "map_is_not_sufficiency_evidence": True,
        },
    }


def planner_input(question: dict[str, Any], map_view: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": {
            "question_id": question["question_id"],
            "question_text": question["question"],
            "answer_options": list(question.get("answer_options", [])),
        },
        "navigation_map": map_view,
        "available_retrieval_channels": {
            "medium_structured_fallback_lexical": True,
            "medium_siglip": True,
            "audio_exact_transcript_lexical": True,
            "fine_references": True,
        },
        "retrieval_contract": {
            "medium_candidate_universe": "30/30",
            "audio_candidate_universe": "107/107",
            "hard_filtering_allowed": False,
            "coarse_prior_weight": 0.0,
            "fine_reranking": False,
        },
    }


def audio_rank(plan: dict[str, Any], question: dict[str, Any], audio_nodes: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    started = time.perf_counter()
    ranked = []
    for audio in audio_nodes:
        best_score, best_unit, best_terms = -1.0, None, []
        for unit in plan["search_units"]:
            query = shared.unit_query_text(question, unit)
            score, terms = shared.lexical_similarity(query, audio["transcript"])
            if score > best_score:
                best_score, best_unit, best_terms = score, unit["unit_id"], terms
        ranked.append({
            "audio_id": audio["audio_id"],
            "start_sec": float(audio["start_sec"]),
            "end_sec": float(audio["end_sec"]),
            "exact_transcript": audio["transcript"],
            "source_type": audio.get("source_type", "unclear"),
            "asr_status": audio.get("asr_status"),
            "lexical_score": best_score,
            "matched_terms": best_terms,
            "best_search_unit_id": best_unit,
            "evidence_type": "audio_asr",
            "support_scope": "audible_statement_or_mention_only",
            "visual_confirmation": False,
            "physical_event_occurrence_confirmed": False,
        })
    ranked.sort(key=lambda row: (-row["lexical_score"], row["start_sec"], row["audio_id"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked, ranked[:top_k], time.perf_counter() - started


def audio_reference_audit(question_id: str, ranking: list[dict[str, Any]], reference: dict[str, Any], top_k: int) -> dict[str, Any]:
    timestamps = [float(value) for value in reference.get("timestamps_sec", [])]
    if question_id == "q_global_summary":
        return {"top_k_contains_audio": bool(ranking[:top_k]), "reference_overlapping_audio_ids": [], "reference_audio_ranks": {}}
    overlapping = [row for row in ranking if any(row["start_sec"] <= timestamp < row["end_sec"] for timestamp in timestamps)]
    return {
        "top_k_contains_reference_overlapping_audio": any(row["rank"] <= top_k for row in overlapping),
        "reference_overlapping_audio_ids": [row["audio_id"] for row in overlapping],
        "reference_audio_ranks": {row["audio_id"]: row["rank"] for row in overlapping},
    }


def preflight(root: Path) -> dict[str, Any]:
    data = load_sources(root)
    base, r1_av, config = data["base"], data["r1_av_map"], data["config"]
    timeline = r1_av["timeline_nodes"]
    visual = [row for row in timeline if row["node_type"] == "visual_structural"]
    audio = [row for row in timeline if row["node_type"] == "audio_asr"]
    lexical = shared.build_lexical_records(data["projection"], base, "r1")
    r1_integrity, _ = shared.validate_lexical_integrity(lexical, shared.build_lexical_records(data["projection"], base, "r3"), data["projection"], base)
    checks = {
        "r1_av_source_passed": data["r1_av_validation"].get("overall_validation") == "passed",
        "canonical_index_hash": shared.sha256_file(data["paths"]["canonical_index"]) == "8b88e4a75c55d241b18e254b471da2bc38387fc6a9e6ccdf7b66337a1e75381f",
        "five_visual_nodes": len(visual) == 5,
        "107_audio_nodes": len(audio) == 107,
        "audio_ids_unique": len({row["node_id"] for row in audio}) == 107,
        "audio_exact_against_canonical": all(row["node_id"] == source["audio_id"] and row["transcript"] == source["transcript"] and row["start_sec"] == source["start_sec"] and row["end_sec"] == source["end_sec"] for row, source in zip(audio, base["audio_nodes"])),
        "six_questions": [row["question_id"] for row in data["questions"]] == shared.QUESTION_IDS,
        "r1_lexical_integrity": r1_integrity["valid"],
        "30_medium_88_fine": (len(base["medium_nodes"]), len(base["fine_nodes"])) == (30, 88),
        "coarse_prior_zero": config["ranking"]["coarse_prior_weight"] == 0.0,
        "fine_reranking_false": config["ranking"]["fine_reranking"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"preflight_failed:{[key for key, value in checks.items() if not value]}")
    return {"valid": True, "checks": checks, "passed": sum(checks.values()), "required": len(checks)}


def _review(path: Path, rows: list[dict[str, Any]]) -> None:
    sections = []
    for row in rows:
        medium = "".join(f"<li>#{x['rank']} {x['medium_id']} [{x['start_sec']:.1f},{x['end_sec']:.1f}) score={x['combined_score']:.3f}</li>" for x in row["medium_top_k"])
        audio = "".join(f"<li>#{x['rank']} {x['audio_id']} [{x['start_sec']:.1f},{x['end_sec']:.1f}) score={x['lexical_score']:.3f}: {html.escape(x['exact_transcript'])}</li>" for x in row["audio_top_k"])
        plan = html.escape(json.dumps(row["planner_output"], ensure_ascii=False, indent=2))
        sections.append(f"<section><h2>{row['question_id']}</h2><pre>{plan}</pre><div class='grid'><div><h3>Medium</h3><ol>{medium}</ol></div><div><h3>Audio</h3><ol>{audio}</ol></div></div></section>")
    path.write_text("<!doctype html><meta charset='utf-8'><style>body{font:14px system-ui;margin:24px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap}section{border-top:1px solid #aaa}</style><h1>R1_AV dual-channel retrieval</h1>" + "".join(sections), encoding="utf-8", newline="\n")


def run(root: Path, output: Path, api_key: str) -> dict[str, Any]:
    started = time.perf_counter()
    preflight_report = preflight(root)
    data = load_sources(root)
    paths, config, questions, base = data["paths"], data["config"], data["questions"], data["base"]
    before = {name: shared.sha256_file(path) for name, path in paths.items()}
    map_view = planner_map_view(data["r1_av_map"])
    inputs = [planner_input(question, map_view) for question in questions]
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "planner_call_checkpoints"
    plans = [shared.plan_one(api_key, config, payload, checkpoints / f"{payload['question']['question_id']}.json") for payload in inputs]

    lexical = shared.build_lexical_records(data["projection"], base, "r1")
    medium_embeddings = np.asarray(np.load(paths["medium_embeddings"], mmap_mode="r"), dtype=np.float32)
    medium_embeddings = medium_embeddings / np.maximum(np.linalg.norm(medium_embeddings, axis=1, keepdims=True), 1e-12)
    from experiments.planner_medium_retrieval.core import SiglipTextEncoder
    encoder = SiglipTextEncoder(config["siglip"])
    parent = shared._parent_map({"coarse_regions": data["r1_av_map"]["visual_structural_regions"]})
    audio_nodes = base["audio_nodes"]
    rows, medium_rankings, audio_rankings, packets, retentions = [], [], [], [], []
    siglip_latency = 0.0
    fine_ids = {row["fine_id"] for row in base["fine_nodes"]}
    for question, plan_doc in zip(questions, plans):
        query_texts = [shared.unit_query_text(question, unit) for unit in plan_doc["plan"]["search_units"]]
        encode_started = time.perf_counter(); query_embeddings = encoder.encode(query_texts); encode_sec = time.perf_counter() - encode_started; siglip_latency += encode_sec
        medium_full, medium_top, medium_sec = shared.rank_all_mediums(question, plan_doc["plan"], lexical, base["medium_nodes"], medium_embeddings, query_embeddings, {**config, "ranking": {**config["ranking"], "top_k": config["ranking"]["top_k_medium"]}}, parent)
        audio_full, audio_top, audio_sec = audio_rank(plan_doc["plan"], question, audio_nodes, int(config["ranking"]["top_k_audio"]))
        fine = shared._fine_refs(medium_top, base)
        if len(medium_full) != 30 or len(audio_full) != 107 or any(item["coarse_prior"] != 0.0 for item in medium_full):
            raise RuntimeError(f"candidate_contract_failed:{question['question_id']}")
        medium_retention = shared._reference_retention(question["question_id"], medium_full, data["references"], int(config["ranking"]["top_k_medium"]))
        audio_retention = audio_reference_audit(question["question_id"], audio_full, data["references"][question["question_id"]], int(config["ranking"]["top_k_audio"]))
        medium_rankings.append({"question_id": question["question_id"], "candidate_count": 30, "full_ranking": medium_full, "top_k": medium_top, "latency_sec": medium_sec})
        audio_rankings.append({"question_id": question["question_id"], "candidate_count": 107, "full_ranking": audio_full, "top_k": audio_top, "latency_sec": audio_sec})
        packet = {
            "question_id": question["question_id"],
            "question_text": question["question"],
            "planner_search_units": plan_doc["plan"]["search_units"],
            "selected_medium_evidence": [{**item, "evidence_type": "detector_observation", "map_text_used_as_evidence": False} for item in medium_top],
            "selected_fine_references": fine,
            "selected_audio_evidence": audio_top,
            "candidate_universe": {"medium": "30/30", "audio": "107/107"},
            "map_used_for_hard_pruning": False,
            "map_used_as_evidence": False,
            "embedding_scores_are_navigation_only": True,
            "fine_frames_reviewed": False,
        }
        packets.append(packet)
        retentions.append({"question_id": question["question_id"], "medium": medium_retention, "audio": audio_retention})
        rows.append({"question_id": question["question_id"], "planner_output": plan_doc["plan"], "medium_top_k": medium_top, "audio_top_k": audio_top, "medium_retention": medium_retention, "audio_retention": audio_retention})

    cost = shared._call_records(plans)
    cost.update({"local_siglip_text_encoding_latency_sec": siglip_latency, "other_model_api_calls": 0})
    after = {name: shared.sha256_file(path) for name, path in paths.items()}
    unchanged = before == after
    all_fine_valid = all(item["fine_id"] in fine_ids for packet in packets for item in packet["selected_fine_references"])
    validation = {
        "source_validation": "passed" if unchanged else "failed",
        "planner_validation": "passed" if len(plans) == 6 else "failed",
        "medium_retrieval_validation": "passed" if all(row["candidate_count"] == 30 for row in medium_rankings) else "failed",
        "audio_retrieval_validation": "passed" if all(row["candidate_count"] == 107 for row in audio_rankings) else "failed",
        "evidence_packet_validation": "passed" if len(packets) == 6 and all_fine_valid else "failed",
        "overall_validation": "passed" if unchanged and len(plans) == 6 and all_fine_valid else "failed",
        "planner_calls": cost["calls"],
        "schema_repair_calls": cost["schema_repair_calls"],
        "candidate_universe": {"medium": "30/30", "audio": "107/107"},
        "hard_pruning": False,
        "coarse_prior_weight": 0.0,
        "fine_reranking": False,
        "sufficiency_calls": 0,
        "final_answer_calls": 0,
        "qa_calls": 0,
    }
    outputs = {
        "input_manifest.json": {"experiment": EXPERIMENT, "source_hashes": before, "question_count": 6},
        "preflight_report.json": preflight_report,
        "planner_inputs.json": inputs,
        "planner_outputs.json": plans,
        "medium_retrieval_rankings.json": medium_rankings,
        "audio_retrieval_rankings.json": audio_rankings,
        "pre_sufficiency_evidence_packets.json": packets,
        "diagnostic_reference_retention_audit.json": {"used_posthoc_only": True, "rows": retentions},
        "provenance_audit.json": {"map_is_evidence": False, "audio_ranked_from_canonical_exact_transcript": True, "medium_ranked_from_exact_fallback_and_frozen_siglip": True, "fine_references_reviewed": False},
        "cost_accounting.json": cost,
        "protected_hash_audit.json": {"status": "passed" if unchanged else "failed", "before": before, "after": after, "unchanged": unchanged},
        "validation_report.json": validation,
    }
    for filename, value in outputs.items():
        shared.write_json(output / filename, value)
    _review(output / "review.html", rows)
    (output / "REPORT.md").write_text("\n".join([
        "# EgoPolice R1_AV dual-channel retrieval smoke v1",
        "",
        f"Overall validation: **{validation['overall_validation']}**.",
        "",
        "- Planner: R1_AV visual structural and independent exact-ASR timeline nodes",
        "- Medium retrieval: `30/30`, structured fallback + frozen SigLIP",
        "- Audio retrieval: `107/107`, exact-transcript lexical ranking",
        "- Fine reranking: disabled; canonical child Fine references returned",
        "- Map hard pruning and Coarse prior: disabled",
        f"- Planner calls/repairs: `{cost['calls']}` / `{cost['schema_repair_calls']}`",
        f"- Planner input/output tokens: `{cost['input_tokens']}` / `{cost['output_tokens']}`",
        f"- Planner latency: `{cost['latency_sec']:.3f}s`",
        "- Sufficiency, Final Gemini and QA calls: `0`",
        f"- Total local run latency: `{time.perf_counter() - started:.3f}s`",
    ]) + "\n", encoding="utf-8", newline="\n")
    return validation
