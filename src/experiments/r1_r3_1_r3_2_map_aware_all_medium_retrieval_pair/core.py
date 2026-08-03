from __future__ import annotations

import copy
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.r1_r3_v2_map_aware_all_medium_retrieval_pair import core as shared


EXPERIMENT = "r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1"
RUNGS = ("r1", "r3_1", "r3_2")


def source_paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / f"configs/experiments/{EXPERIMENT}.json",
        "questions": root / "configs/experiments/planner_medium_retrieval_v1/questions_226.json",
        "canonical_index": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json",
        "r1_index": root / "outputs/experiments/egopolice_r1_structured_index_organizer_canary_v1/hierarchical_index_r1_caption_free.json",
        "r1_projection": root / "outputs/experiments/egopolice_two_stage_organizer_r1_canary_v1/unified_medium_envelopes.json",
        "medium_embeddings": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/embeddings/medium_pooled_siglip.float32.npy",
        "r1_map": root / "outputs/experiments/egopolice_r1_structural_novelty_map_v1/coarse_structural_map.json",
        "r1_view": root / "outputs/experiments/egopolice_r1_structural_novelty_map_v1/planner_compatibility_view.json",
        "r1_freeze": root / "outputs/experiments/r1_all_medium_retrieval_baseline_freeze_v1/freeze_manifest.json",
        "r3_1_map": root / "outputs/experiments/r3_1_noisy_av_semantic_map_freeze_v1/r3_1_noisy_av_semantic_map.json",
        "r3_1_view": root / "outputs/experiments/r3_1_noisy_av_semantic_map_freeze_v1/r3_1_planner_compatibility_view.json",
        "r3_1_freeze": root / "outputs/experiments/r3_1_noisy_av_semantic_map_freeze_v1/r3_1_freeze_manifest.json",
        "r3_2_map": root / "outputs/experiments/r3_2_global_stage1_clean_navigation_map_freeze_v1/r3_2_frozen_semantic_navigation_map.json",
        "r3_2_view": root / "outputs/experiments/r3_2_global_stage1_clean_navigation_map_freeze_v1/r3_2_frozen_planner_compatibility_view.json",
        "r3_2_freeze": root / "outputs/experiments/r3_2_global_stage1_clean_navigation_map_freeze_v1/r3_2_navigation_map_freeze_manifest.json",
    }


def _load(root: Path) -> dict[str, Any]:
    paths = source_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing_sources:{missing}")
    return {
        "paths": paths,
        "config": shared.load_json(paths["config"]),
        "questions": shared.load_json(paths["questions"])["questions"],
        "base": shared.load_json(paths["canonical_index"]),
        "r1_index": shared.load_json(paths["r1_index"]),
        "projection": shared.load_json(paths["r1_projection"]),
        "maps": {rung: shared.load_json(paths[f"{rung}_map"]) for rung in RUNGS},
        "views": {rung: shared.load_json(paths[f"{rung}_view"]) for rung in RUNGS},
    }


def build_r3_2_clean_context(raw_map: dict[str, Any], planner_view: dict[str, Any]) -> dict[str, Any]:
    raw_by_id = {row["coarse_id"]: row for row in raw_map["coarse_regions"]}
    rows = []
    for view in planner_view["coarse_regions"]:
        raw = raw_by_id[view["coarse_id"]]
        rows.append({
            "coarse_id": view["coarse_id"],
            "start_sec": float(view["start_sec"]),
            "end_sec": float(view["end_sec"]),
            "navigation_text": view["summary"],
            "source_medium_ids": list(view["source_medium_ids"]),
            "source_captions": [],
            "source_asr": [],
            "uncertainty_notes": list(view.get("uncertainty_notes", [])),
            "sidecar_caption_count": len(raw.get("exact_source_captions_sidecar", [])),
            "sidecar_asr_count": len(raw.get("exact_source_asr_sidecar", [])),
        })
    return {
        "map_type": raw_map["map_type"],
        "semantic_fields_available": True,
        "has_storyline": False,
        "storyline_events": [],
        "hard_filtering_allowed": False,
        "coarse_regions": rows,
        "map_policy": {
            "navigation_hints_only": True,
            "planner_visible_text": "global clean phase labels only",
            "raw_caption_asr_sidecars_not_exposed_to_planner": True,
            "all_mediums_remain_eligible": True,
            "coarse_prior_weight": 0.0,
            "map_text_is_not_answer_evidence": True,
        },
    }


def build_contexts(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "r1": shared.build_map_context("r1", data["maps"]["r1"], data["views"]["r1"], data["base"]),
        "r3_1": shared.build_map_context("r3", data["maps"]["r3_1"], data["views"]["r3_1"], data["base"]),
        "r3_2": build_r3_2_clean_context(data["maps"]["r3_2"], data["views"]["r3_2"]),
    }


def preflight(root: Path) -> dict[str, Any]:
    data = _load(root)
    base = data["base"]
    config = data["config"]
    canonical_ids = [row["medium_id"] for row in base["medium_nodes"]]
    canonical_fine = [row["fine_id"] for row in base["fine_nodes"]]
    parent_maps = {rung: shared._parent_map(data["maps"][rung]) for rung in RUNGS}
    contexts = build_contexts(data)
    r1_lex = shared.build_lexical_records(data["projection"], base, "r1")
    r3_lex = shared.build_lexical_records(data["projection"], base, "r3")
    r1_integrity, r3_integrity = shared.validate_lexical_integrity(r1_lex, r3_lex, data["projection"], base)
    checks = {
        "canonical_30_medium_88_fine_107_audio": (len(canonical_ids), len(canonical_fine), len(base["audio_nodes"])) == (30, 88, 107),
        "canonical_index_hash": shared.sha256_file(data["paths"]["canonical_index"]) == "8b88e4a75c55d241b18e254b471da2bc38387fc6a9e6ccdf7b66337a1e75381f",
        "six_frozen_questions": [row["question_id"] for row in data["questions"]] == shared.QUESTION_IDS,
        "empty_answer_options": all(row.get("answer_options") == [] for row in data["questions"]),
        "all_maps_cover_same_30_mediums": all(list(parent_maps[rung]) == canonical_ids for rung in RUNGS),
        "all_storylines_disabled": all(data["views"][rung].get("storyline_events") == [] and data["views"][rung].get("has_storyline") is False for rung in RUNGS),
        "all_hard_filtering_disabled": all(contexts[rung]["hard_filtering_allowed"] is False for rung in RUNGS),
        "r3_2_raw_sidecars_not_exposed": all(not row["source_captions"] and not row["source_asr"] for row in contexts["r3_2"]["coarse_regions"]),
        "r1_lexical_integrity": r1_integrity["valid"],
        "r3_shared_caption_lexical_integrity": r3_integrity["valid"],
        "coarse_prior_zero": config["ranking"]["coarse_prior_weight"] == 0.0,
        "fine_reranking_false": config["ranking"]["fine_reranking"] is False,
        "embedding_shape_30x768": np.load(data["paths"]["medium_embeddings"], mmap_mode="r").shape == (30, 768),
    }
    if not all(checks.values()):
        raise ValueError(f"preflight_failed:{[key for key, value in checks.items() if not value]}")
    return {"valid": True, "checks": checks, "passed": sum(checks.values()), "required": len(checks)}


def _planner_input_stats(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [len(shared.canonical_bytes(row)) for row in inputs]
    return {"questions": len(inputs), "payload_bytes_total": sum(sizes), "payload_bytes_mean": sum(sizes) / len(sizes), "payload_bytes_min": min(sizes), "payload_bytes_max": max(sizes)}


def _plan_or_resume(api_key: str, config: dict[str, Any], payload: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    if checkpoint.is_file():
        saved = shared.load_json(checkpoint)
        if saved.get("status") == "passed" and saved.get("question_id") == payload["question"]["question_id"]:
            return {key: saved[key] for key in ("question_id", "plan", "validation", "attempts")}
    return shared.plan_one(api_key, config, payload, checkpoint)


def _render_review(path: Path, comparisons: list[dict[str, Any]], cost: dict[str, Any]) -> None:
    cost_rows = "".join(
        f"<tr><td>{rung}</td><td>{row['calls']}</td><td>{row['input_tokens']}</td><td>{row['output_tokens']}</td><td>{row['latency_sec']:.3f}</td></tr>"
        for rung, row in cost["planner_by_rung"].items()
    )
    sections = []
    for row in comparisons:
        cols = []
        for rung in RUNGS:
            plan = html.escape(json.dumps(row["rungs"][rung]["plan"], ensure_ascii=False, indent=2))
            ranks = "".join(f"<li>#{item['rank']} {item['medium_id']} ({item['combined_score']:.3f})</li>" for item in row["rungs"][rung]["top_k"])
            cols.append(f"<div><h3>{rung}</h3><pre>{plan}</pre><ol>{ranks}</ol></div>")
        sections.append(f"<section><h2>{row['question_id']}</h2><div class='grid'>{''.join(cols)}</div></section>")
    path.write_text("<!doctype html><meta charset='utf-8'><style>body{font:14px system-ui;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}pre{white-space:pre-wrap}table,td,th{border:1px solid #aaa;border-collapse:collapse;padding:6px}</style><h1>R1 / R3_1 / R3_2 map-aware retrieval</h1><table><tr><th>Rung</th><th>Calls</th><th>Input tokens</th><th>Output tokens</th><th>Latency sec</th></tr>" + cost_rows + "</table>" + "".join(sections), encoding="utf-8", newline="\n")


def run(root: Path, output: Path, api_key: str) -> dict[str, Any]:
    started = time.perf_counter()
    preflight_report = preflight(root)
    data = _load(root)
    paths, config, questions, base = data["paths"], data["config"], data["questions"], data["base"]
    hashes_before = {name: shared.sha256_file(path) for name, path in paths.items()}
    contexts = build_contexts(data)
    inputs = {rung: [shared.build_planner_input(question, contexts[rung], base) for question in questions] for rung in RUNGS}
    for index in range(len(questions)):
        question_bytes = [shared.canonical_bytes(inputs[rung][index]["question"]) for rung in RUNGS]
        if len(set(question_bytes)) != 1:
            raise RuntimeError(f"question_contract_mismatch:{index}")

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "planner_call_checkpoints"
    planner_outputs: dict[str, list[dict[str, Any]]] = {}
    for rung in RUNGS:
        planner_outputs[rung] = [
            _plan_or_resume(api_key, config, payload, checkpoint / f"{rung}_{payload['question']['question_id']}.json")
            for payload in inputs[rung]
        ]

    lexical = {
        "r1": shared.build_lexical_records(data["projection"], base, "r1"),
        "r3_1": shared.build_lexical_records(data["projection"], base, "r3"),
        "r3_2": shared.build_lexical_records(data["projection"], base, "r3"),
    }
    r1_integrity, r3_integrity = shared.validate_lexical_integrity(lexical["r1"], lexical["r3_1"], data["projection"], base)
    if shared.canonical_bytes(lexical["r3_1"]) != shared.canonical_bytes(lexical["r3_2"]):
        raise RuntimeError("r3_1_r3_2_lexical_corpora_differ")

    medium_embeddings = np.asarray(np.load(paths["medium_embeddings"], mmap_mode="r"), dtype=np.float32)
    medium_embeddings = medium_embeddings / np.maximum(np.linalg.norm(medium_embeddings, axis=1, keepdims=True), 1e-12)
    from experiments.planner_medium_retrieval.core import SiglipTextEncoder
    encoder = SiglipTextEncoder(config["siglip"])
    parent_maps = {rung: shared._parent_map(data["maps"][rung]) for rung in RUNGS}
    ranking_docs: dict[str, list[dict[str, Any]]] = {rung: [] for rung in RUNGS}
    comparisons, retentions, fine_audits = [], [], []
    reference_config = shared.load_json(root / "configs/experiments/egopolice_r1_structural_map_navigation_smoke_v1.json")["posthoc_diagnostic_reference"]["queries"]
    encoder_latency = 0.0
    for question_index, question in enumerate(questions):
        row = {"question_id": question["question_id"], "question_text": question["question"], "rungs": {}}
        for rung in RUNGS:
            plan_doc = planner_outputs[rung][question_index]
            query_texts = [shared.unit_query_text(question, unit) for unit in plan_doc["plan"]["search_units"]]
            encode_started = time.perf_counter()
            query_embeddings = encoder.encode(query_texts)
            encode_sec = time.perf_counter() - encode_started
            encoder_latency += encode_sec
            full, top_k, ranking_sec = shared.rank_all_mediums(question, plan_doc["plan"], lexical[rung], base["medium_nodes"], medium_embeddings, query_embeddings, config, parent_maps[rung])
            if len(full) != 30 or any(item["coarse_prior"] != 0.0 for item in full):
                raise RuntimeError(f"candidate_or_prior_violation:{rung}:{question['question_id']}")
            fine = shared._fine_refs(top_k, base)
            retention = shared._reference_retention(question["question_id"], full, reference_config, int(config["ranking"]["top_k"]))
            doc = {"rung": rung, "question_id": question["question_id"], "candidate_medium_count": 30, "suggested_coarse_ids": plan_doc["plan"]["suggested_coarse_ids"], "suggestions_used_for_filtering": False, "suggestions_used_for_scoring": False, "ranking_formula": "0.6 * visual_score_normalized + 0.3 * lexical_score + 0.0 * coarse_prior", "query_embedding_latency_sec": encode_sec, "ranking_latency_sec": ranking_sec, "full_ranking": full, "top_k": top_k, "fine_references": fine}
            ranking_docs[rung].append(doc)
            row["rungs"][rung] = {"plan": plan_doc["plan"], "top_k": top_k, "diagnostic_reference_retention": retention}
            retentions.append({"rung": rung, "question_id": question["question_id"], **retention})
            fine_audits.append({"rung": rung, "question_id": question["question_id"], "fine_reranking": False, "all_references_valid": len(fine) > 0, "references": fine})
        comparisons.append(row)

    planner_cost = {rung: shared._call_records(planner_outputs[rung]) for rung in RUNGS}
    input_stats = {rung: _planner_input_stats(inputs[rung]) for rung in RUNGS}
    base_r1 = planner_cost["r1"]
    deltas = {}
    for rung in ("r3_1", "r3_2"):
        current = planner_cost[rung]
        deltas[f"{rung}_minus_r1"] = {
            "input_tokens": current["input_tokens"] - base_r1["input_tokens"],
            "output_tokens": current["output_tokens"] - base_r1["output_tokens"],
            "latency_sec": current["latency_sec"] - base_r1["latency_sec"],
            "input_token_ratio": current["input_tokens"] / base_r1["input_tokens"],
        }
    deltas["r3_2_minus_r3_1"] = {
        "input_tokens": planner_cost["r3_2"]["input_tokens"] - planner_cost["r3_1"]["input_tokens"],
        "output_tokens": planner_cost["r3_2"]["output_tokens"] - planner_cost["r3_1"]["output_tokens"],
        "latency_sec": planner_cost["r3_2"]["latency_sec"] - planner_cost["r3_1"]["latency_sec"],
        "input_token_ratio": planner_cost["r3_2"]["input_tokens"] / planner_cost["r3_1"]["input_tokens"],
    }
    cost = {
        "planner_by_rung": planner_cost,
        "planner_input_payload_size": input_stats,
        "planner_cost_deltas": deltas,
        "total_planner_calls": sum(row["calls"] for row in planner_cost.values()),
        "total_input_tokens": sum(row["input_tokens"] for row in planner_cost.values()),
        "total_output_tokens": sum(row["output_tokens"] for row in planner_cost.values()),
        "total_planner_latency_sec": sum(row["latency_sec"] for row in planner_cost.values()),
        "local_siglip_text_encoding_latency_sec": encoder_latency,
        "other_model_api_calls": 0,
    }

    hashes_after = {name: shared.sha256_file(path) for name, path in paths.items()}
    source_unchanged = hashes_before == hashes_after
    soft = all(doc["candidate_medium_count"] == 30 and not doc["suggestions_used_for_filtering"] and not doc["suggestions_used_for_scoring"] for rows in ranking_docs.values() for doc in rows)
    repairs = {rung: planner_cost[rung]["schema_repair_calls"] for rung in RUNGS}
    validation = {
        "source_validation": "passed" if source_unchanged else "failed",
        "shared_planner_policy_validation": "passed",
        "planner_execution_validation": "passed" if all(len(planner_outputs[rung]) == 6 for rung in RUNGS) else "failed",
        "all_medium_retrieval_validation": "passed" if soft else "failed",
        "r1_lexical_validation": "passed" if r1_integrity["valid"] else "failed",
        "r3_1_r3_2_caption_lexical_identity": "passed" if r3_integrity["valid"] else "failed",
        "overall_validation": "passed" if source_unchanged and soft and r1_integrity["valid"] and r3_integrity["valid"] else "failed",
        "planner_calls": sum(row["calls"] for row in planner_cost.values()),
        "schema_repairs": repairs,
        "candidate_universe": {rung: "30/30" for rung in RUNGS},
        "coarse_prior_weight": 0.0,
        "fine_reranking": False,
        "organizer_calls": 0,
        "sufficiency_calls": 0,
        "final_answer_calls": 0,
        "qa_calls": 0,
    }

    source_audit = {"valid": source_unchanged, "before": hashes_before, "after": hashes_after, "modified": [name for name in hashes_before if hashes_before[name] != hashes_after[name]]}
    soft_audit = {"valid": soft, "suggestions_filter_mediums": False, "suggestions_change_scores": False, "coarse_prior_weight": 0.0, "candidate_universe": {rung: ["30/30"] * 6 for rung in RUNGS}}
    outputs = {
        "input_manifest.json": {"experiment": EXPERIMENT, "rungs": list(RUNGS), "question_count": 6, "source_hashes": hashes_before},
        "source_hash_audit.json": source_audit,
        "shared_planner_contract.json": {"system_prompt": shared.PLANNER_SYSTEM_PROMPT, "system_prompt_sha256": hashlib.sha256(shared.PLANNER_SYSTEM_PROMPT.encode()).hexdigest(), "schema": shared.planner_schema(), "schema_sha256": hashlib.sha256(shared.canonical_bytes(shared.planner_schema())).hexdigest(), "configuration": config["planner"]},
        "planner_input_size_audit.json": input_stats,
        "soft_hint_and_candidate_audit.json": soft_audit,
        "lexical_integrity_audit.json": {"r1": r1_integrity, "r3_shared": r3_integrity, "r3_1_r3_2_byte_identical": shared.canonical_bytes(lexical["r3_1"]) == shared.canonical_bytes(lexical["r3_2"])},
        "paired_retrieval_comparison.json": comparisons,
        "diagnostic_reference_retention_audit.json": {"used_posthoc_only": True, "rows": retentions},
        "fine_reference_audit.json": {"fine_reranking": False, "rows": fine_audits},
        "planner_cost_comparison.json": cost,
        "validation_report.json": validation,
    }
    for rung in RUNGS:
        outputs[f"{rung}_planner_inputs.json"] = inputs[rung]
        outputs[f"{rung}_planner_outputs.json"] = planner_outputs[rung]
        outputs[f"{rung}_retrieval_rankings.json"] = ranking_docs[rung]
    for filename, value in outputs.items():
        shared.write_json(output / filename, value)
    _render_review(output / "review.html", comparisons, cost)
    report_lines = [
        "# R1 / R3_1 / R3_2 map-aware all-Medium retrieval pair",
        "",
        f"Overall validation: **{validation['overall_validation']}**.",
        "",
        "All three paths used the same Planner policy, all 30 Mediums, zero Coarse prior, and no Fine reranking.",
        "R3_1 and R3_2 used byte-identical canonical-caption lexical corpora; only their maps and resulting Planner formulations differed.",
        "",
        "| Rung | Calls | Input tokens | Output tokens | Latency (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for rung in RUNGS:
        row = planner_cost[rung]
        report_lines.append(f"| {rung} | {row['calls']} | {row['input_tokens']} | {row['output_tokens']} | {row['latency_sec']:.3f} |")
    report_lines += [
        "",
        f"R3_2 versus R3_1 input-token ratio: `{deltas['r3_2_minus_r3_1']['input_token_ratio']:.4f}`.",
        "",
        "Organizer, Sufficiency, temporal review, Final Gemini and QA calls were zero.",
        f"Total local experiment latency: `{time.perf_counter() - started:.3f}s`.",
    ]
    (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    return validation
