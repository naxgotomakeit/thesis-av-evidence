from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.fine_reranking.core import (
    build_fine_query_text,
    load_fine_embeddings,
    rerank_fines_for_medium,
    select_diverse_fines,
)
from experiments.planner_medium_retrieval.core import SiglipTextEncoder


RUNGS = ("r1_av", "r3_2")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_question(
    *, question: str, question_id: str, plan: dict[str, Any], medium_doc: dict[str, Any],
    medium_by_id: dict[str, dict[str, Any]], fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: np.ndarray, row_by_id: dict[str, int], encoder: SiglipTextEncoder,
    max_fines: int, minimum_gap: float,
) -> tuple[dict[str, Any], float]:
    units = {x["unit_id"]: x for x in plan["search_units"]}
    query_rows = [{
        "search_unit_id": unit["unit_id"],
        "query_text": build_fine_query_text(question, unit),
    } for unit in plan["search_units"]]
    started = time.perf_counter()
    encoded = encoder.encode([x["query_text"] for x in query_rows])
    encoding_sec = time.perf_counter() - started
    query_by_unit = {row["search_unit_id"]: encoded[i] for i, row in enumerate(query_rows)}
    selected, full_ranking, per_medium = [], [], []
    for medium_row in medium_doc["top_k"]:
        medium = medium_by_id[medium_row["medium_id"]]
        unit_ids = [x for x in medium_row.get("matched_search_unit_ids", []) if x in units]
        if not unit_ids:
            unit_ids = list(units)
        rankings = {}
        for unit_id in unit_ids:
            rows = rerank_fines_for_medium(
                question_id=question_id, search_unit_id=unit_id, medium=medium,
                fine_by_id=fine_by_id, fine_embeddings=fine_embeddings,
                row_by_id=row_by_id, query_embedding=query_by_unit[unit_id],
            )
            rankings[unit_id] = rows
            full_ranking.extend(rows)
        chosen, single_reason = select_diverse_fines(
            rankings, strategy=plan["temporal_strategy"], max_fines=max_fines,
            minimum_gap=minimum_gap,
        )
        chosen_rows = []
        for row in chosen:
            fine = fine_by_id[row["fine_id"]]
            saved = {
                "question_id": question_id,
                "medium_rank": medium_row["rank"], "medium_id": medium["medium_id"],
                "parent_coarse_id": medium_row["parent_coarse_id"],
                "fine_id": row["fine_id"],
                "timestamp_sec": float(fine["representative_frame_timestamp_sec"]),
                "frame_path": fine["representative_frame_path"],
                "siglip_score_raw": float(row["siglip_score_raw"]),
                "selection_reason": row["selection_reason"],
                "search_unit_ids": row["search_unit_ids"],
                "selection_policy": "frozen Fine SigLIP reranking; at most 2 relevant/time-diverse Fine per retrieved Medium",
            }
            selected.append(saved); chosen_rows.append(saved)
        per_medium.append({
            "medium_id": medium["medium_id"], "medium_rank": medium_row["rank"],
            "child_fine_count": len(medium["child_fine_ids"]),
            "selected_fine_ids": [x["fine_id"] for x in chosen_rows],
            "single_selection_reason": single_reason,
        })
    return {
        "question_id": question_id,
        "medium_top_k_count": len(medium_doc["top_k"]),
        "candidate_child_fine_reference_count": len(medium_doc["fine_references"]),
        "fine_query_records": query_rows,
        "fine_ranking": full_ranking,
        "selected_fine_evidence": selected,
        "per_medium_selection": per_medium,
        "gemini_contract": {
            "allowed_image_ids": [x["fine_id"] for x in selected],
            "whole_medium_expansion_allowed": False,
            "claim_range_expansion_allowed": False,
            "unranked_child_fines_allowed": False,
            "memory_cache_enabled": False,
        },
    }, encoding_sec


def validate(results: dict[str, list[dict[str, Any]]], index: dict[str, Any], max_fines: int) -> list[str]:
    errors = []
    fine_by_id = {x["fine_id"]: x for x in index["fine_nodes"]}
    medium_by_id = {x["medium_id"]: x for x in index["medium_nodes"]}
    for rung, questions in results.items():
        if len(questions) != 6:
            errors.append(f"{rung}: expected six questions")
        for question in questions:
            counts = {}
            allowed = set(question["gemini_contract"]["allowed_image_ids"])
            selected_ids = [x["fine_id"] for x in question["selected_fine_evidence"]]
            if not allowed <= set(selected_ids) or len(selected_ids) != len(set(selected_ids)):
                errors.append(f"{rung}/{question['question_id']}: ID contract mismatch")
            per_requirement = question["gemini_contract"].get("allowed_image_ids_by_requirement", {})
            if any(not set(ids) <= allowed for ids in per_requirement.values()):
                errors.append(f"{rung}/{question['question_id']}: requirement allowlist mismatch")
            for row in question["selected_fine_evidence"]:
                counts[row["medium_id"]] = counts.get(row["medium_id"], 0) + 1
                if row["fine_id"] not in fine_by_id:
                    errors.append(f"{rung}/{question['question_id']}: unknown Fine")
                elif row["fine_id"] not in medium_by_id[row["medium_id"]]["child_fine_ids"]:
                    errors.append(f"{rung}/{question['question_id']}: parent mismatch")
                if not Path(row["frame_path"]).is_file():
                    errors.append(f"{rung}/{question['question_id']}: unreadable frame")
                if not np.isfinite(row["siglip_score_raw"]):
                    errors.append(f"{rung}/{question['question_id']}: nonfinite score")
            if any(x > max_fines for x in counts.values()):
                errors.append(f"{rung}/{question['question_id']}: per-Medium budget exceeded")
            contract = question["gemini_contract"]
            if contract["whole_medium_expansion_allowed"] or contract["claim_range_expansion_allowed"] or contract["unranked_child_fines_allowed"]:
                errors.append(f"{rung}/{question['question_id']}: expansion enabled")
    return errors


def run(root: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load(config_path)
    out = root / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: root / v for k, v in cfg["sources"].items()}
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError([str(p) for p in paths.values() if not p.is_file()])
    hashes_before = {k: sha(p) for k, p in paths.items()}
    index = load(paths["index"])
    if len(index["fine_nodes"]) != 88 or len(index["medium_nodes"]) != 30:
        raise RuntimeError("canonical hierarchy count mismatch")
    fine_config = load(paths["fine_config"])
    fine_embeddings, row_by_id, embedding_path = load_fine_embeddings(index, paths["index"])
    fine_by_id = {x["fine_id"]: x for x in index["fine_nodes"]}
    medium_by_id = {x["medium_id"]: x for x in index["medium_nodes"]}
    rankings = {
        "r1_av": load(paths["r1_medium_rankings"]),
        "r3_2": load(paths["r3_2_medium_rankings"]),
    }
    planners = {
        "r1_av": load(paths["r1_planner_outputs"]),
        "r3_2": load(paths["r3_2_planner_outputs"]),
    }
    direct_packets = load(paths["direct_packets"])
    question_text = {}
    planner_inputs = load(root / "outputs/experiments/r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1/r1_planner_inputs.json")
    for row in planner_inputs:
        question_text[row["question"]["question_id"]] = row["question"]["question_text"]
    encoder = SiglipTextEncoder(fine_config["siglip"])
    results, encoding_latency = {"r1_av": [], "r3_2": []}, {"r1_av": 0.0, "r3_2": 0.0}
    for rung in RUNGS:
        planner_by_q = {x["question_id"]: x["plan"] for x in planners[rung]}
        for medium_doc in rankings[rung]:
            qid = medium_doc["question_id"]
            result, latency = select_question(
                question=question_text[qid], question_id=qid, plan=planner_by_q[qid],
                medium_doc=medium_doc, medium_by_id=medium_by_id, fine_by_id=fine_by_id,
                fine_embeddings=fine_embeddings, row_by_id=row_by_id, encoder=encoder,
                max_fines=int(fine_config["max_fine_per_medium"]),
                minimum_gap=float(fine_config["minimum_timestamp_gap_sec"]),
            )
            results[rung].append(result); encoding_latency[rung] += latency
    for rung in RUNGS:
        for result in results[rung]:
            qid = result["question_id"]
            packet = direct_packets[rung][qid]
            medium_ids_by_requirement = {x["requirement_id"]: [] for x in packet["requirements"]}
            for evidence in packet["evidence"]:
                provenance = evidence.get("source_provenance") or {}
                medium_id = provenance.get("medium_id")
                for requirement_id in provenance.get("retrieved_for_requirement_ids", []):
                    if requirement_id in medium_ids_by_requirement and medium_id:
                        medium_ids_by_requirement[requirement_id].append(medium_id)
            selected_by_medium = {}
            for row in result["selected_fine_evidence"]:
                selected_by_medium.setdefault(row["medium_id"], []).append(row["fine_id"])
            allowed_by_requirement = {}
            for requirement_id, medium_ids in medium_ids_by_requirement.items():
                allowed_by_requirement[requirement_id] = list(dict.fromkeys(
                    fine_id for medium_id in medium_ids for fine_id in selected_by_medium.get(medium_id, [])
                ))
            allowed_union = list(dict.fromkeys(
                fine_id for ids in allowed_by_requirement.values() for fine_id in ids
            ))
            result["requirement_bound_fine_evidence"] = {
                "retrieved_medium_ids_by_requirement": {
                    key: list(dict.fromkeys(value)) for key, value in medium_ids_by_requirement.items()
                },
                "allowed_image_ids_by_requirement": allowed_by_requirement,
                "question_union_allowed_image_ids": allowed_union,
                "requirements_without_retrieved_fine_evidence": [key for key, value in allowed_by_requirement.items() if not value],
            }
            result["gemini_contract"]["allowed_image_ids"] = allowed_union
            result["gemini_contract"]["allowed_image_ids_by_requirement"] = allowed_by_requirement
    errors = validate(results, index, int(fine_config["max_fine_per_medium"]))
    hashes_after = {k: sha(p) for k, p in paths.items()}
    if hashes_before != hashes_after:
        errors.append("protected source hash changed")
    for rung in RUNGS:
        dump(out / f"{rung}_fine_retrieval.json", results[rung])
        dump(out / f"{rung}_gemini_image_allowlist.json", {
            "rung": rung, "memory_cache_enabled": False,
            "questions": [{"question_id": x["question_id"], **x["gemini_contract"]} for x in results[rung]],
        })
    diagnostic = load(paths["v3_diagnostic_manifest"])
    comparison = {"previous_v3": {}, "retrieval_bound": {}}
    for rung in RUNGS:
        comparison["previous_v3"][rung] = {
            q: len(x["images"]) for q, x in diagnostic[rung].items()
        }
        comparison["retrieval_bound"][rung] = {
            x["question_id"]: len(x["gemini_contract"]["allowed_image_ids"]) for x in results[rung]
        }
    dump(out / "image_candidate_count_comparison.json", comparison)
    source_audit = {"before": hashes_before, "after": hashes_after,
                    "unchanged": hashes_before == hashes_after,
                    "fine_embedding_path": str(embedding_path), "fine_embedding_shape": list(fine_embeddings.shape)}
    dump(out / "source_hash_audit.json", source_audit)
    validation = {
        "source_validation": "passed" if hashes_before == hashes_after else "failed",
        "fine_reranking_validation": "passed" if not errors else "failed",
        "gemini_image_contract_validation": "passed" if not errors else "failed",
        "errors": errors, "overall_validation": "passed" if not errors else "failed",
        "api_calls": 0, "gemini_calls": 0, "video_decoding": 0,
        "memory_cache_enabled": False,
    }
    dump(out / "validation_report.json", validation)
    cost = {"api_calls": 0, "local_siglip_text_encoding_latency_sec": encoding_latency,
            "total_latency_sec": time.perf_counter() - started}
    dump(out / "cost_accounting.json", cost)
    rows = []
    for qid in question_text:
        cells = []
        for rung in RUNGS:
            q = next(x for x in results[rung] if x["question_id"] == qid)
            cells.append(f"<td>{html.escape(', '.join(x['fine_id'] for x in q['selected_fine_evidence']))}</td>")
        rows.append(f"<tr><td>{qid}</td>{''.join(cells)}</tr>")
    (out / "review.html").write_text("<meta charset='utf-8'><h1>Retrieval-bound Fine evidence</h1><table border=1><tr><th>Question</th><th>R1_AV</th><th>R3_2</th></tr>" + "".join(rows) + "</table>", encoding="utf-8")
    report = ["# R1_AV / R3_2 retrieval-bound Fine evidence v1", "",
              f"- Overall: `{validation['overall_validation']}`",
              "- Gemini receives only Fine IDs selected by the frozen Fine SigLIP reranker.",
              "- Whole-Medium and claim-range image expansion: disabled.",
              "- Gemini memory/cache: intentionally not implemented in this experiment.",
              "- API/Gemini/video decoding calls: `0/0/0`.", "", "## Counts"]
    for rung in RUNGS:
        report.append(f"- {rung}: " + ", ".join(f"{x['question_id']}={len(x['gemini_contract']['allowed_image_ids'])}" for x in results[rung]))
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return validation
