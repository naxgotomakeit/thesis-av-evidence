from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_question_batched_visual_review_v2.core import build_batch_plan
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import _estimated_cost, _gemini_call, _load_env
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import (
    _final,
    _validate_sufficiency,
    option_requirements,
)
from experiments.reviewed_visual_evidence_cache_v1.cache import canonical_bytes
from experiments.reviewed_visual_evidence_cache_v1_gemini_canary.core import usage as gemini_usage


REVIEW_SYSTEM = """You are a question-level visual evidence comparator for a multiple-choice video question. Inspect every supplied image once. Treat all answer options symmetrically and never assume that any option is correct. First report neutral observations for every image. Then assess every option against the same reviewed images. An option is visually supported only when every answer-critical clause is directly established. Report missing, unverified, and contradicted clauses explicitly. Do not select the final answer, do not use retrieval rank as evidence, and do not infer unshown continuity. Return strict JSON."""


def comparison_schema(question: dict[str, Any], fine_ids: list[str]) -> dict[str, Any]:
    option_ids = [row["option_id"] for row in question["answer_options"]]
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "observations": {
                "type": "array", "minItems": len(fine_ids), "maxItems": len(fine_ids),
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "fine_id": {"type": "string", "enum": fine_ids},
                        "finding": {"type": "string"},
                        "visible_actions": {"type": "array", "items": {"type": "string"}},
                        "visible_objects": {"type": "array", "items": {"type": "string"}},
                        "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["fine_id", "finding", "visible_actions", "visible_objects", "uncertainty_notes"],
                },
            },
            "option_assessments": {
                "type": "array", "minItems": len(option_ids), "maxItems": len(option_ids),
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "option_id": {"type": "string", "enum": option_ids},
                        "status": {"type": "string", "enum": ["supported", "partially_supported", "contradicted", "not_observed"]},
                        "direct_visual_support": {"type": "boolean"},
                        "supporting_fine_ids": {"type": "array", "items": {"type": "string", "enum": fine_ids}},
                        "supported_clauses": {"type": "array", "items": {"type": "string"}},
                        "unverified_clauses": {"type": "array", "items": {"type": "string"}},
                        "contradicted_clauses": {"type": "array", "items": {"type": "string"}},
                        "rationale": {"type": "string"},
                    },
                    "required": ["option_id", "status", "direct_visual_support", "supporting_fine_ids", "supported_clauses", "unverified_clauses", "contradicted_clauses", "rationale"],
                },
            },
        },
        "required": ["question_id", "observations", "option_assessments"],
    }


def validate_comparison(value: dict[str, Any], question: dict[str, Any], fine_ids: list[str]) -> None:
    if value.get("question_id") != question["question_id"]:
        raise ValueError("comparison question mismatch")
    observed = [row.get("fine_id") for row in value.get("observations", [])]
    if observed != fine_ids or len(observed) != len(set(observed)):
        raise ValueError("comparison Fine coverage/order mismatch")
    expected_options = [row["option_id"] for row in question["answer_options"]]
    actual_options = [row.get("option_id") for row in value.get("option_assessments", [])]
    if actual_options != expected_options or len(actual_options) != len(set(actual_options)):
        raise ValueError("comparison option coverage/order mismatch")
    known_fines = set(fine_ids)
    for row in value["option_assessments"]:
        if set(row["supporting_fine_ids"]) - known_fines:
            raise ValueError("comparison cited unknown Fine")
        if row["status"] == "supported":
            if not row["direct_visual_support"] or not row["supporting_fine_ids"] or row["unverified_clauses"] or row["contradicted_clauses"]:
                raise ValueError("supported option violates whole-option visual contract")
        elif row["direct_visual_support"]:
            raise ValueError("non-supported option cannot claim direct whole-option support")
    if "selected_option_id" in value or "answer" in value:
        raise ValueError("visual comparator selected a final answer")


def batch_key(side: str, question: dict[str, Any], unique_fines: list[dict[str, Any]], contract: str) -> str:
    payload = {
        "side": side,
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "answer_options": question["answer_options"],
        "image_sha256": [row["image_sha256"] for row in unique_fines],
        "contract": contract,
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _cache_path(out: Path, side: str, key: str) -> Path:
    return out / "cache" / side / f"{key}.json"


def _store_immutable(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError("immutable symmetric-review cache collision")


def _call_comparator(cfg: dict[str, Any], question: dict[str, Any], unique_fines: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = {
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "answer_options": question["answer_options"],
        "ordered_images": [{"fine_id": row["fine_id"], "timestamp_sec": row["timestamp_sec"]} for row in unique_fines],
        "comparison_policy": {
            "all_options_equal_status": True,
            "single_option_confirmation_prompts": False,
            "whole_option_support_requires_all_critical_clauses": True,
        },
    }
    inputs: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
    for row in unique_fines:
        image = Path(row["source_frame_path"])
        inputs.extend([
            {"type": "text", "text": f"FINE {row['fine_id']} timestamp={float(row['timestamp_sec']):.3f}s"},
            {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image.read_bytes()).decode("ascii")},
        ])
    result, usage, raw = _gemini_call({"gemini": cfg["gemini"]}, REVIEW_SYSTEM, inputs, comparison_schema(question, [row["fine_id"] for row in unique_fines]))
    validate_comparison(result, question, [row["fine_id"] for row in unique_fines])
    return result, usage, raw


def project_resolved_sufficiency(initial: dict[str, Any], comparison: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    by_option = {row["option_id"]: row for row in comparison["option_assessments"]}
    assessments = []
    for initial_row, option in zip(initial["assessments"], question["answer_options"]):
        visual = by_option[option["option_id"]]
        status = initial_row["status"]
        direct = initial_row["direct_support"]
        cited = list(initial_row["supporting_evidence_ids"])
        visual_ids = [f"reviewed_visual::{fine_id}::{scope}" for fine_id in visual["supporting_fine_ids"]]
        if visual["status"] == "supported":
            status, direct = "supported", True
            cited = list(dict.fromkeys(cited + visual_ids))
        elif visual["status"] == "partially_supported":
            if status != "supported": status, direct = "uncertain", False
            cited = list(dict.fromkeys(cited + visual_ids))
        elif visual["status"] == "contradicted":
            status, direct = ("conflicted", False) if status == "supported" else ("not_found", False)
            cited = list(dict.fromkeys(cited + visual_ids))
        # not_observed preserves stronger caption/ASR support but cannot create it.
        assessments.append({
            "requirement_id": initial_row["requirement_id"], "status": status,
            "direct_support": direct, "supporting_evidence_ids": cited,
            "rationale": initial_row["rationale"] + " Symmetric visual comparison: " + visual["rationale"],
        })
    supported = [row for row in assessments if row["status"] == "supported"]
    if len(supported) == 1:
        candidate = [next(option["option_id"] for option, row in zip(question["answer_options"], assessments) if row is supported[0])]
        gate = "answer_ready"
    else:
        candidate = [option["option_id"] for option, row in zip(question["answer_options"], assessments) if row["status"] in {"supported", "uncertain"}]
        gate = "provisional"
    resolved = {
        "question_id": question["question_id"], "assessments": assessments,
        "gate": gate, "candidate_option_ids": candidate,
        "review_requirement_ids": [], "review_query": "",
    }
    requirements = option_requirements(question)
    _validate_sufficiency(resolved, question, requirements, evidence)
    return resolved


def _protected_files(source: Path) -> list[Path]:
    files = [source / "answers_blind.json", source / "posthoc_evaluation.json", source / "cost_accounting.json"]
    for case in sorted((source / "cases").iterdir()):
        if not case.is_dir() or not (case / "answers_blind.json").is_file(): continue
        files.extend([case / "question_input.json", case / "answers_blind.json"])
        for side in ("r1_av", "r3_2"):
            files.extend([case / side / "initial_sufficiency.json", case / side / "initial_evidence.json", case / side / "requirement_retrieval.json"])
    return files


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); source = root / cfg["source_experiment"]; out = root / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    protected = _protected_files(source); before = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    plan = build_batch_plan(source)
    errors = []
    for case in plan["cases"]:
        question = load_json(source / "cases" / case["video_uid"] / "question_input.json")
        for side in ("r1_av", "r3_2"):
            row = case["sides"][side]
            hashes = [fine["image_sha256"] for fine in row["unique_fines"]]
            if len(hashes) != len(set(hashes)): errors.append(f"{case['video_uid']}/{side}: duplicate Fine hash")
            row["batch_cache_key"] = batch_key(side, question, row["unique_fines"], cfg["review_contract_version"])
    after = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    if before != after: errors.append("protected source changed")
    validation = {
        "question_symmetric_options": "passed" if not errors else "failed",
        "single_option_confirmation_prompts": 0,
        "old_visual_calls": sum(case["sides"][side]["old_option_scoped_calls"] for case in plan["cases"] for side in ("r1_av", "r3_2")),
        "planned_visual_calls": sum(case["sides"][side]["new_question_scoped_calls"] for case in plan["cases"] for side in ("r1_av", "r3_2")),
        "old_image_transmissions": sum(case["sides"][side]["old_image_transmissions"] for case in plan["cases"] for side in ("r1_av", "r3_2")),
        "planned_image_transmissions": sum(case["sides"][side]["new_image_transmissions"] for case in plan["cases"] for side in ("r1_av", "r3_2")),
        "haiku_post_review_calls": 0,
        "protected_sources_unchanged": before == after,
        "model_api_calls": 0, "errors": errors,
        "overall_validation": "passed_no_api_ready_for_live_v3" if not errors else "failed",
    }
    write_json(out / "input_manifest.json", {"source_experiment": cfg["source_experiment"], "protected_hashes": before})
    write_json(out / "question_symmetric_review_plan.json", plan)
    write_json(out / "visual_comparison_schema.json", comparison_schema({"question_id": "fixture", "answer_options": [{"option_id": "A"}, {"option_id": "B"}]}, ["F001"]))
    write_json(out / "validation_report.json", validation)
    (out / "REPORT.md").write_text("# Question-symmetric visual review v3\n\nNo-API preflight only.\n", encoding="utf-8")
    return validation


def _scope(question_id: str) -> str:
    return "q_" + hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16] + "_symmetric_visual_review_v3"


def _recover_calls(out: Path, plan: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for case in plan["cases"]:
        uid = case["video_uid"]
        for side in ("r1_av", "r3_2"):
            side_out = out / "cases" / uid / side
            if not (side_out / "result.json").is_file(): continue
            result = load_json(side_out / "result.json")
            if result.get("review_skipped_answer_ready"): continue
            cache = load_json(_cache_path(out, side, case["sides"][side]["batch_cache_key"]))
            review_usage = dict(cache["usage"]); review_usage.update({"stage": "question_symmetric_visual_comparison", "side": side, "video_uid": uid, "image_transmissions": len(case["sides"][side]["unique_fines"])})
            final_raw = load_json(side_out / "final_raw_response.json")
            final_usage = {"provider": "google", "model": cfg["gemini"]["model"], **gemini_usage(final_raw), "latency_sec": None, "latency_recovered": False, "stage": "final_after_symmetric_review", "side": side, "video_uid": uid, "image_transmissions": 0}
            calls.extend([review_usage, final_usage])
    return calls


def run_live(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); source = root / cfg["source_experiment"]; out = root / cfg["output_root"]
    validation = preflight(root, config_path)
    if validation["overall_validation"] != "passed_no_api_ready_for_live_v3": raise RuntimeError("preflight failed")
    _load_env(root.parent / "thesis-av-evidence" / ".env")
    if not os.environ.get("GEMINI_API_KEY"): raise RuntimeError("Gemini API key unavailable")
    plan = load_json(out / "question_symmetric_review_plan.json"); outputs: dict[str, Any] = {}
    for case_plan in plan["cases"]:
        uid = case_plan["video_uid"]; case = source / "cases" / uid; question = load_json(case / "question_input.json"); requirements = option_requirements(question); outputs[uid] = {}
        for side in ("r1_av", "r3_2"):
            side_plan = case_plan["sides"][side]; side_out = out / "cases" / uid / side; side_out.mkdir(parents=True, exist_ok=True); result_path = side_out / "result.json"
            if result_path.is_file(): outputs[uid][side] = load_json(result_path); continue
            if not side_plan["batch_required"]:
                reused = load_json(case / side / "final_answer.json"); result = {"review_skipped_answer_ready": True, "answer": reused["answer"], "new_model_calls": 0}; write_json(result_path, result); outputs[uid][side] = result; continue
            cache_path = _cache_path(out, side, side_plan["batch_cache_key"])
            if cache_path.is_file():
                cache = load_json(cache_path); comparison = cache["comparison"]
            else:
                comparison, usage, raw = _call_comparator(cfg, question, side_plan["unique_fines"])
                usage.update({"stage": "question_symmetric_visual_comparison", "side": side, "video_uid": uid, "image_transmissions": len(side_plan["unique_fines"])})
                cache = {"contract_version": cfg["review_contract_version"], "side": side, "batch_cache_key": side_plan["batch_cache_key"], "comparison": comparison, "usage": usage, "raw_response": raw}
                _store_immutable(cache_path, cache)
            scope = _scope(question["question_id"])
            observation_by_id = {row["fine_id"]: row for row in comparison["observations"]}
            fine_by_id = {row["fine_id"]: row for row in side_plan["unique_fines"]}
            visual_evidence = [{"evidence_id": f"reviewed_visual::{fine_id}::{scope}", "evidence_type": "reviewed_visual_frame", "source_content": observation["finding"], "fine_id": fine_id, "timestamp_sec": fine_by_id[fine_id]["timestamp_sec"], "visible_actions": observation["visible_actions"], "visible_objects": observation["visible_objects"], "uncertainty_notes": observation["uncertainty_notes"]} for fine_id, observation in observation_by_id.items()]
            evidence = load_json(case / side / "initial_evidence.json") + visual_evidence
            initial = load_json(case / side / "initial_sufficiency.json")["output"]
            resolved = project_resolved_sufficiency(initial, comparison, question, evidence, scope)
            answer, final_usage, final_raw = _final({**cfg, "ranking": {}, "siglip_text": {}}, question, requirements, resolved, evidence)
            final_usage.update({"stage": "final_after_symmetric_review", "side": side, "video_uid": uid, "image_transmissions": 0})
            selected_req = f"{question['question_id']}::option_{answer['selected_option_id'].lower()}"
            selected = next(row for row in resolved["assessments"] if row["requirement_id"] == selected_req)
            if selected["status"] in {"not_found", "conflicted"} and any(row["status"] in {"supported", "uncertain"} for row in resolved["assessments"]):
                write_json(side_out / "invalid_final_raw_response.json", final_raw)
                raise ValueError("Final selected a visually rejected option")
            result = {"review_skipped_answer_ready": False, "comparison": comparison, "resolved_sufficiency": resolved, "answer": answer, "new_model_calls": 2}
            write_json(side_out / "symmetric_visual_comparison.json", comparison); write_json(side_out / "resolved_sufficiency.json", resolved); write_json(side_out / "final_raw_response.json", final_raw); write_json(result_path, result); outputs[uid][side] = result
            write_json(out / "live_progress.json", outputs)
    calls = _recover_calls(out, plan, cfg)
    write_json(out / "live_calls.json", calls)
    write_json(out / "answers_blind.json", [{"video_uid": uid, "answers": {side: row[side]["answer"] for side in ("r1_av", "r3_2")}} for uid, row in outputs.items()])
    write_json(out / "cost_accounting.json", {"calls": calls, "new_api_usd": _estimated_cost(cfg, calls), "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in calls)})
    return {"completed_cases": len(outputs), "model_calls": len(calls), "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in calls)}


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); source = root / cfg["source_experiment"]; out = root / cfg["output_root"]
    if not (out / "answers_blind.json").is_file(): raise RuntimeError("blind answers missing")
    blind_hash = sha256_file(out / "answers_blind.json"); old_eval = load_json(source / "posthoc_evaluation.json"); gold = {row["video_uid"]: row["gold_option_id"] for row in old_eval["cases"]}; old_rows = {row["video_uid"]: row for row in old_eval["cases"]}
    rows = []
    for item in load_json(out / "answers_blind.json"):
        uid = item["video_uid"]; row = {"video_uid": uid, "question_id": old_rows[uid]["question_id"], "gold_option_id": gold[uid]}
        for side in ("r1_av", "r3_2"):
            selected = item["answers"][side]["selected_option_id"]
            row[side] = {"selected_option_id": selected, "correct": selected == gold[uid], "old_selected_option_id": old_rows[uid][side]["selected_option_id"]}
        rows.append(row)
    summary = {"r1_correct": sum(row["r1_av"]["correct"] for row in rows), "r3_correct": sum(row["r3_2"]["correct"] for row in rows)}
    write_json(out / "posthoc_evaluation.json", {"answers_blind_sha256_before_gold_load": blind_hash, "cases": rows, "summary": summary})
    old_cost = load_json(source / "cost_accounting.json"); new_cost = load_json(out / "cost_accounting.json"); plan = load_json(out / "question_symmetric_review_plan.json")
    ready = {(case["video_uid"], side) for case in plan["cases"] for side in ("r1_av", "r3_2") if not case["sides"][side]["batch_required"]}
    reused = [row for row in old_cost["calls"] if row["stage"] in {"organizer", "planner", "sufficiency"} or (row["stage"] == "final_text_only" and (row["video_uid"], row["side"]) in ready)]
    official = reused + new_cost["calls"]
    comparison = {
        "old": {"api_calls": len(old_cost["calls"]), "image_transmissions": old_cost["r1_av"]["image_transmissions"] + old_cost["r3_2"]["image_transmissions"], "total_tokens": sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in old_cost["calls"]), "estimated_usd": old_cost["new_api_total_usd"], "accuracy": old_eval["summary"]},
        "v3_comparable_runtime": {"api_calls": len(official), "image_transmissions": new_cost["image_transmissions"], "total_tokens": sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in official), "estimated_usd": _estimated_cost(cfg, official), "accuracy": summary},
        "successful_incremental_replay": {"api_calls": len(new_cost["calls"]), "image_transmissions": new_cost["image_transmissions"], "estimated_usd": new_cost["new_api_usd"]},
        "haiku_post_review_calls": 0,
    }
    comparison["per_side"] = {}
    for side in ("r1_av", "r3_2"):
        old_side = [row for row in old_cost["calls"] if row.get("side") == side]
        reused_side = [row for row in reused if row.get("side") == side]
        new_side = [row for row in new_cost["calls"] if row.get("side") == side]
        full_side = reused_side + new_side
        comparison["per_side"][side] = {
            "old": {
                "api_calls": len(old_side),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in old_side),
                "total_tokens": sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in old_side),
                "estimated_usd": _estimated_cost(cfg, old_side),
            },
            "v3_comparable_runtime": {
                "api_calls": len(full_side),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in full_side),
                "total_tokens": sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in full_side),
                "estimated_usd": _estimated_cost(cfg, full_side),
            },
            "successful_incremental_replay": {
                "api_calls": len(new_side),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in new_side),
                "estimated_usd": _estimated_cost(cfg, new_side),
            },
        }
    write_json(out / "old_vs_v3_cost_comparison.json", comparison)
    validation = load_json(out / "validation_report.json"); validation.update({"live_replay": "passed", "posthoc_evaluation": "passed", "model_api_calls": len(new_cost["calls"]), "image_transmissions": new_cost["image_transmissions"], "overall_validation": "passed_live_v3"}); write_json(out / "validation_report.json", validation)
    (out / "REPORT.md").write_text(f"# Question-symmetric visual review v3 - live replay\n\n- Accuracy: R1 `{summary['r1_correct']}/10`; R3 `{summary['r3_correct']}/10`.\n- Image transmissions: `{comparison['old']['image_transmissions']} -> {comparison['v3_comparable_runtime']['image_transmissions']}`.\n- Comparable cost: `${comparison['old']['estimated_usd']:.6f} -> ${comparison['v3_comparable_runtime']['estimated_usd']:.6f}`.\n- Gemini saw every option symmetrically in one question-level visual call. Haiku post-review calls: `0`.\n", encoding="utf-8")
    return {"summary": summary, "cost_comparison": comparison}
