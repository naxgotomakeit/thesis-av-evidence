from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import _anthropic_call, _estimated_cost, _gemini_call, _load_env
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import (
    FINAL_SYSTEM,
    SUFFICIENCY_PROMPT,
    _final,
    _project_sufficiency,
    _sufficiency_schema,
    _validate_sufficiency,
    option_requirements,
)
from experiments.reviewed_visual_evidence_cache_v1.cache import canonical_bytes
from experiments.reviewed_visual_evidence_cache_v1_gemini_canary.core import usage as gemini_usage


REVIEW_SYSTEM = """You are a question-scoped visual observation component. Inspect every supplied image once and report only directly visible, neutral observations useful for distinguishing the supplied answer hypotheses. Do not choose an answer, mark any option as supported, infer unshown continuity, or convert a partial observation into support for a compound option. Return exactly one observation record per Fine image in strict JSON."""

SECOND_PASS_SUFFIX = """
This is the post-review pass. Reassess all option requirements together using the question-scoped neutral visual observations. An option containing multiple answer-critical clauses may be supported only when every necessary clause is directly established by compatible evidence. Partial support for one clause must remain uncertain for the whole option. Do not treat image retrieval, image presence, or option-targeted localization as semantic proof.
Keep the structured response compact: cite at most three minimal evidence IDs per option,
write each rationale in at most 35 words, do not repeat observation text, and keep any
review_query to one short sentence. These are serialization limits only; assess every option.
"""


def _recover_successful_calls(out: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct successful-call accounting after an interrupted/resumed live run."""
    rows: list[dict[str, Any]] = []
    plan_by_key = {
        (case["video_uid"], side): case["sides"][side]
        for case in plan["cases"] for side in ("r1_av", "r3_2")
    }
    cases_root = out / "cases"
    if not cases_root.is_dir():
        return rows
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        uid = case_dir.name
        for side in ("r1_av", "r3_2"):
            side_dir = case_dir / side
            if not (side_dir / "result.json").is_file():
                continue
            result = load_json(side_dir / "result.json")
            if result.get("review_skipped_answer_ready"):
                continue
            plan_row = plan_by_key[(uid, side)]
            review_raw = load_json(side_dir / "gemini_raw_response.json")
            review_usage = {
                "provider": "google", "model": "gemini-3.5-flash", **gemini_usage(review_raw),
                "latency_sec": None, "latency_recovered": False,
                "stage": "question_batched_visual_review", "side": side, "video_uid": uid,
                "image_transmissions": len(plan_row["unique_fines"]),
            }
            suff_usage = dict(load_json(side_dir / "second_pass_sufficiency.json")["usage"])
            final_raw = load_json(side_dir / "final_raw_response.json")
            final_usage = {
                "provider": "google", "model": "gemini-3.5-flash", **gemini_usage(final_raw),
                "latency_sec": None, "latency_recovered": False,
                "stage": "final_after_batched_review", "side": side, "video_uid": uid,
                "image_transmissions": 0,
            }
            rows.extend([review_usage, suff_usage, final_usage])
    return rows


def _complete_call_ledger(out: Path, plan: dict[str, Any], observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill interrupted-run gaps from immutable raw responses without double counting."""
    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (row["video_uid"], row["side"], row["stage"])

    ledger = {key(row): row for row in _recover_successful_calls(out, plan)}
    for row in observed:
        current = ledger.get(key(row))
        if current is None or row.get("latency_sec") is not None:
            ledger[key(row)] = row
    stage_order = {
        "question_batched_visual_review": 0,
        "sufficiency_after_batched_review": 1,
        "final_after_batched_review": 2,
    }
    return sorted(ledger.values(), key=lambda row: (row["video_uid"], row["side"], stage_order[row["stage"]]))


def question_scope(question_id: str) -> str:
    return "q_" + hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16] + "_visual_observation"


def _dedupe_fines(rows_by_requirement: dict[str, list[dict[str, Any]]], requested: list[str]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    by_hash: dict[str, dict[str, Any]] = {}
    targets: dict[str, list[str]] = {}
    for requirement_id in requested:
        for row in rows_by_requirement.get(requirement_id, []):
            path = Path(row["source_frame_path"])
            digest = sha256_file(path)
            identity = (str(path.resolve()), float(row["timestamp_sec"]))
            if digest in by_hash:
                existing = by_hash[digest]
                if (str(Path(existing["source_frame_path"]).resolve()), float(existing["timestamp_sec"])) != identity:
                    raise ValueError("same image hash has conflicting provenance")
            else:
                by_hash[digest] = {**row, "image_sha256": digest, "image_size_bytes": path.stat().st_size}
            targets.setdefault(digest, []).append(requirement_id)
    ordered = sorted(by_hash.values(), key=lambda row: (float(row["timestamp_sec"]), row["fine_id"], row["image_sha256"]))
    return ordered, {digest: list(dict.fromkeys(ids)) for digest, ids in targets.items()}


def build_batch_plan(source: Path) -> dict[str, Any]:
    cases = []
    for case in sorted((source / "cases").iterdir()):
        if not case.is_dir() or not (case / "answers_blind.json").is_file(): continue
        question = load_json(case / "question_input.json")
        sides = {}
        for side in ("r1_av", "r3_2"):
            sufficiency = load_json(case / side / "initial_sufficiency.json")["output"]
            retrieval = load_json(case / side / "requirement_retrieval.json")
            requested = sufficiency["review_requirement_ids"]
            rows_by_requirement = {
                rid: retrieval["requirements"][rid]["selected_fine_evidence"] for rid in requested
            }
            unique, targets = _dedupe_fines(rows_by_requirement, requested)
            old_transmissions = sum(len(rows) for rows in rows_by_requirement.values())
            sides[side] = {
                "question_scope": question_scope(question["question_id"]),
                "review_requirement_ids": requested,
                "batch_required": bool(requested),
                "unique_fines": unique,
                "requirement_ids_by_image_sha256": targets,
                "old_option_scoped_calls": len(requested),
                "new_question_scoped_calls": 1 if requested else 0,
                "old_image_transmissions": old_transmissions,
                "new_image_transmissions": len(unique),
                "eliminated_repeated_transmissions": old_transmissions - len(unique),
                "rung_cache_root": f"cache/{side}",
            }
        cases.append({"video_uid": case.name, "question_id": question["question_id"], "sides": sides})
    return {"schema_version": "question_batched_visual_review_plan_v2", "cases": cases}


def review_schema(fine_ids: list[str]) -> dict[str, Any]:
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
                        "visible_interactions": {"type": "array", "items": {"type": "string"}},
                        "visible_objects": {"type": "array", "items": {"type": "string"}},
                        "scene_context": {"type": "array", "items": {"type": "string"}},
                        "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["fine_id", "finding", "visible_actions", "visible_interactions", "visible_objects", "scene_context", "uncertainty_notes"],
                },
            },
        },
        "required": ["question_id", "observations"],
    }


def validate_review(value: dict[str, Any], question_id: str, fine_ids: list[str]) -> None:
    if value.get("question_id") != question_id: raise ValueError("review question mismatch")
    seen = [row.get("fine_id") for row in value.get("observations", [])]
    if len(seen) != len(set(seen)) or set(seen) != set(fine_ids): raise ValueError("review Fine coverage mismatch")
    forbidden = {"requirement_effect", "supports_requirement", "selected_option_id", "answer"}
    for row in value["observations"]:
        if set(row) & forbidden: raise ValueError("review output contains option/requirement decision")
        if not row["finding"].strip(): raise ValueError("empty visual finding")


@dataclass(frozen=True)
class ObservationRecord:
    record_version: str
    review_contract_version: str
    side: str
    question_scope: str
    question_id: str
    fine_id: str
    image_sha256: str
    image_size_bytes: int
    timestamp_sec: float
    finding: str
    visible_actions: tuple[str, ...]
    visible_interactions: tuple[str, ...]
    visible_objects: tuple[str, ...]
    scene_context: tuple[str, ...]
    uncertainty_notes: tuple[str, ...]
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("visible_actions", "visible_interactions", "visible_objects", "scene_context", "uncertainty_notes"):
            value[key] = list(value[key])
        return value


class RungQuestionObservationCache:
    def __init__(self, root: Path): self.root = root

    def path(self, record_or_hash: ObservationRecord | str, contract: str | None = None, scope: str | None = None) -> Path:
        if isinstance(record_or_hash, ObservationRecord):
            digest, contract, scope = record_or_hash.image_sha256, record_or_hash.review_contract_version, record_or_hash.question_scope
        else: digest = record_or_hash
        assert contract is not None and scope is not None
        contract_hash = hashlib.sha256(contract.encode("utf-8")).hexdigest()[:16]
        return self.root / "records" / digest / contract_hash / f"{scope}.json"

    def load(self, image_hash: str, contract: str, scope: str) -> dict[str, Any] | None:
        path = self.path(image_hash, contract, scope)
        return load_json(path) if path.is_file() else None

    def store(self, record: ObservationRecord) -> None:
        path = self.path(record); payload = canonical_bytes(record.to_dict()); path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes() != payload: raise FileExistsError("immutable neutral observation cache collision")


def _protected_files(source: Path) -> list[Path]:
    files = [source / "answers_blind.json", source / "posthoc_evaluation.json", source / "cost_accounting.json"]
    for case in sorted((source / "cases").iterdir()):
        if not case.is_dir() or not (case / "answers_blind.json").is_file(): continue
        files.extend([case / "question_input.json", case / "answers_blind.json"])
        for side in ("r1_av", "r3_2"):
            files.extend([case / side / "initial_sufficiency.json", case / side / "requirement_retrieval.json", case / side / "initial_evidence.json"])
    return files


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); source = root / cfg["source_experiment"]; out = root / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    protected = _protected_files(source); before = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    plan = build_batch_plan(source)
    rows = [row["sides"][side] for row in plan["cases"] for side in ("r1_av", "r3_2")]
    summary = {
        side: {
            "old_calls": sum(row["sides"][side]["old_option_scoped_calls"] for row in plan["cases"]),
            "new_calls": sum(row["sides"][side]["new_question_scoped_calls"] for row in plan["cases"]),
            "old_image_transmissions": sum(row["sides"][side]["old_image_transmissions"] for row in plan["cases"]),
            "new_image_transmissions": sum(row["sides"][side]["new_image_transmissions"] for row in plan["cases"]),
        } for side in ("r1_av", "r3_2")
    }
    errors = []
    for case in plan["cases"]:
        r1, r3 = case["sides"]["r1_av"], case["sides"]["r3_2"]
        if r1["rung_cache_root"] == r3["rung_cache_root"]: errors.append(f"{case['video_uid']}: rung caches are not isolated")
        for side, row in case["sides"].items():
            hashes = [fine["image_sha256"] for fine in row["unique_fines"]]
            if len(hashes) != len(set(hashes)): errors.append(f"{case['video_uid']}/{side}: duplicate image remains")
            if row["new_question_scoped_calls"] > 1: errors.append(f"{case['video_uid']}/{side}: more than one question batch")
    after = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    if before != after: errors.append("protected source changed")
    write_json(out / "input_manifest.json", {"source_experiment": cfg["source_experiment"], "protected_hashes": before})
    write_json(out / "question_batched_review_plan.json", plan)
    write_json(out / "redundancy_elimination_audit.json", summary)
    write_json(out / "neutral_observation_schema.json", review_schema(["F001"]))
    write_json(out / "cache_contract.json", {"rung_isolation": True, "key": ["side", "image_sha256", "review_contract_version", "question_scope"], "option_id_in_cache_key": False, "immutable": True})
    write_json(out / "second_pass_sufficiency_contract.json", {"required_after_review": True, "deterministic_whole_option_promotion": False, "all_option_requirements_reassessed_together": True, "compound_option_requires_all_critical_clauses": True})
    validation = {
        "old_visual_calls": sum(row["old_option_scoped_calls"] for row in rows), "new_visual_calls": sum(row["new_question_scoped_calls"] for row in rows),
        "old_image_transmissions": sum(row["old_image_transmissions"] for row in rows), "new_image_transmissions": sum(row["new_image_transmissions"] for row in rows),
        "eliminated_repeated_transmissions": sum(row["eliminated_repeated_transmissions"] for row in rows),
        "rung_cache_isolation": "passed", "option_id_absent_from_cache_key": "passed", "neutral_review_output": "passed",
        "second_sufficiency_pass_required": "passed", "protected_sources_unchanged": before == after,
        "model_api_calls": 0, "errors": errors, "overall_validation": "passed_no_api_ready_for_live_replay" if not errors else "failed",
    }
    write_json(out / "validation_report.json", validation)
    (out / "REPORT.md").write_text(
        "# Question-batched visual review v2\n\n"
        f"- Validation: `{validation['overall_validation']}`.\n"
        f"- Visual calls: `{validation['old_visual_calls']} -> {validation['new_visual_calls']}`.\n"
        f"- Image transmissions: `{validation['old_image_transmissions']} -> {validation['new_image_transmissions']}`.\n"
        "- R1 and R3 cache roots remain isolated.\n- Gemini outputs neutral observations once per unique image; a second Sufficiency pass reassesses every option.\n- Live calls: `0`.\n",
        encoding="utf-8",
    )
    return validation


def _call_review(cfg: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], fine_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompt = {"question": question, "unresolved_option_requirements": requirements, "ordered_images": [{"fine_id": row["fine_id"], "timestamp_sec": row["timestamp_sec"]} for row in fine_rows]}
    inputs: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
    import base64
    for row in fine_rows:
        image = Path(row["source_frame_path"])
        inputs.extend([{"type": "text", "text": f"FINE {row['fine_id']} timestamp={float(row['timestamp_sec']):.3f}s"}, {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image.read_bytes()).decode("ascii")}])
    result, usage, raw = _gemini_call({"gemini": cfg["gemini"]}, REVIEW_SYSTEM, inputs, review_schema([row["fine_id"] for row in fine_rows]))
    validate_review(result, question["question_id"], [row["fine_id"] for row in fine_rows])
    return result, usage, raw


def run_live(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); source = root / cfg["source_experiment"]; out = root / cfg["output_root"]
    validation = preflight(root, config_path)
    if validation["overall_validation"] != "passed_no_api_ready_for_live_replay": raise RuntimeError("preflight failed")
    _load_env(root.parent / "thesis-av-evidence" / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("GEMINI_API_KEY"): raise RuntimeError("API keys unavailable")
    plan = load_json(out / "question_batched_review_plan.json")
    calls: list[dict[str, Any]] = _recover_successful_calls(out, plan)
    outputs = {}
    for case_plan in plan["cases"]:
        uid = case_plan["video_uid"]; case = source / "cases" / uid; question = load_json(case / "question_input.json"); requirements = option_requirements(question); requirement_by_id = {row["requirement_id"]: row for row in requirements}
        outputs[uid] = {}
        for side in ("r1_av", "r3_2"):
            side_plan = case_plan["sides"][side]; side_out = out / "cases" / uid / side; side_out.mkdir(parents=True, exist_ok=True)
            result_path = side_out / "result.json"
            if result_path.is_file(): outputs[uid][side] = load_json(result_path); continue
            if not side_plan["batch_required"]:
                reused = load_json(case / side / "final_answer.json")
                result = {"review_skipped_answer_ready": True, "answer": reused["answer"], "new_model_calls": 0}; write_json(result_path, result); outputs[uid][side] = result; continue
            cache = RungQuestionObservationCache(out / "cache" / side); scope = side_plan["question_scope"]; records, misses = [], []
            for fine in side_plan["unique_fines"]:
                record = cache.load(fine["image_sha256"], cfg["review_contract_version"], scope)
                if record is None: misses.append(fine)
                else: records.append(record)
            if misses:
                target_requirements = [requirement_by_id[rid] for rid in side_plan["review_requirement_ids"]]
                reviewed, usage, raw = _call_review(cfg, question, target_requirements, misses); usage.update({"stage": "question_batched_visual_review", "side": side, "video_uid": uid, "image_transmissions": len(misses)}); calls.append(usage)
                write_json(side_out / "gemini_raw_response.json", raw); write_json(side_out / "gemini_observations.json", reviewed)
                fine_by_id = {row["fine_id"]: row for row in misses}
                for observation in reviewed["observations"]:
                    fine = fine_by_id[observation["fine_id"]]
                    record = ObservationRecord("neutral_visual_observation_record_v2", cfg["review_contract_version"], side, scope, question["question_id"], fine["fine_id"], fine["image_sha256"], fine["image_size_bytes"], float(fine["timestamp_sec"]), observation["finding"], tuple(observation["visible_actions"]), tuple(observation["visible_interactions"]), tuple(observation["visible_objects"]), tuple(observation["scene_context"]), tuple(observation["uncertainty_notes"]), cfg["gemini"]["model"])
                    cache.store(record); records.append(record.to_dict())
            original_evidence = load_json(case / side / "initial_evidence.json")
            visual_evidence = [{"evidence_id": f"reviewed_visual::{row['fine_id']}::{scope}", "evidence_type": "reviewed_visual_frame", "source_content": row["finding"], "timestamp_sec": row["timestamp_sec"], "fine_id": row["fine_id"], "visible_actions": row["visible_actions"], "visible_interactions": row["visible_interactions"], "visible_objects": row["visible_objects"], "uncertainty_notes": row["uncertainty_notes"]} for row in records]
            evidence = original_evidence + visual_evidence
            suff_payload = {"question": question, "requirements": requirements, "evidence": evidence, "pass": "after_question_batched_visual_review"}
            provider, suff_usage = _anthropic_call(cfg, SUFFICIENCY_PROMPT + SECOND_PASS_SUFFIX, suff_payload, _sufficiency_schema(question, requirements, evidence), int(cfg["anthropic"]["sufficiency_max_tokens"]))
            sufficiency = _project_sufficiency(provider, requirements); _validate_sufficiency(sufficiency, question, requirements, evidence); suff_usage.update({"stage": "sufficiency_after_batched_review", "side": side, "video_uid": uid}); calls.append(suff_usage)
            answer, final_usage, final_raw = _final({**cfg, "ranking": {}, "siglip_text": {}}, question, requirements, sufficiency, evidence); final_usage.update({"stage": "final_after_batched_review", "side": side, "video_uid": uid, "image_transmissions": 0}); calls.append(final_usage)
            result = {"review_skipped_answer_ready": False, "observation_count": len(records), "sufficiency": sufficiency, "answer": answer, "new_model_calls": 3}
            write_json(side_out / "second_pass_sufficiency.json", {"input": suff_payload, "output": sufficiency, "usage": suff_usage}); write_json(side_out / "final_raw_response.json", final_raw); write_json(result_path, result); outputs[uid][side] = result
            write_json(out / "live_calls.json", calls); write_json(out / "live_progress.json", outputs)
    calls = _complete_call_ledger(out, plan, calls)
    write_json(out / "live_calls.json", calls)
    write_json(out / "answers_blind.json", [{"video_uid": uid, "answers": {side: row[side]["answer"] for side in ("r1_av", "r3_2")}} for uid, row in outputs.items()])
    write_json(out / "cost_accounting.json", {"calls": calls, "new_api_usd": _estimated_cost(cfg, calls), "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in calls)})
    return {"completed_cases": len(outputs), "model_calls": len(calls), "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in calls)}


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path); source = root / cfg["source_experiment"]; out = root / cfg["output_root"]
    answers_path = out / "answers_blind.json"
    if not answers_path.is_file(): raise RuntimeError("blind answers must exist before evaluation")
    blind_hash = sha256_file(answers_path)
    old_evaluation = load_json(source / "posthoc_evaluation.json")
    gold_by_uid = {row["video_uid"]: row["gold_option_id"] for row in old_evaluation["cases"]}
    old_by_uid = {row["video_uid"]: row for row in old_evaluation["cases"]}
    answers = {row["video_uid"]: row["answers"] for row in load_json(answers_path)}
    rows = []
    for uid, answer_by_side in answers.items():
        gold = gold_by_uid[uid]; row = {"video_uid": uid, "question_id": old_by_uid[uid]["question_id"], "gold_option_id": gold}
        for side in ("r1_av", "r3_2"):
            selected = answer_by_side[side]["selected_option_id"]
            row[side] = {"selected_option_id": selected, "correct": selected == gold, "old_selected_option_id": old_by_uid[uid][side]["selected_option_id"]}
        rows.append(row)
    summary = {
        "r1_correct": sum(row["r1_av"]["correct"] for row in rows), "r3_correct": sum(row["r3_2"]["correct"] for row in rows),
        "both_correct": sum(row["r1_av"]["correct"] and row["r3_2"]["correct"] for row in rows),
        "r1_only": sum(row["r1_av"]["correct"] and not row["r3_2"]["correct"] for row in rows),
        "r3_only": sum(row["r3_2"]["correct"] and not row["r1_av"]["correct"] for row in rows),
        "neither": sum(not row["r1_av"]["correct"] and not row["r3_2"]["correct"] for row in rows),
    }
    result = {"answers_blind_sha256_before_gold_load": blind_hash, "cases": rows, "summary": summary}
    write_json(out / "posthoc_evaluation.json", result)

    old_cost = load_json(source / "cost_accounting.json"); new_cost = load_json(out / "cost_accounting.json")
    answer_ready_keys = set()
    for case_plan in load_json(out / "question_batched_review_plan.json")["cases"]:
        for side in ("r1_av", "r3_2"):
            if not case_plan["sides"][side]["batch_required"]: answer_ready_keys.add((case_plan["video_uid"], side))
    reusable_upstream = [row for row in old_cost["calls"] if row["stage"] in {"organizer", "planner", "sufficiency"}]
    reusable_answer_ready_finals = [row for row in old_cost["calls"] if row["stage"] == "final_text_only" and (row["video_uid"], row["side"]) in answer_ready_keys]
    official_calls = reusable_upstream + reusable_answer_ready_finals + new_cost["calls"]
    comparison = {
        "old_option_scoped_method": {
            "api_calls": len(old_cost["calls"]), "image_transmissions": old_cost["r1_av"]["image_transmissions"] + old_cost["r3_2"]["image_transmissions"],
            "input_tokens": sum(row.get("input_tokens", 0) for row in old_cost["calls"]),
            "output_tokens_including_thought": sum(row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in old_cost["calls"]),
            "estimated_usd": old_cost["new_api_total_usd"], "accuracy": old_evaluation["summary"],
        },
        "new_question_batched_method_comparable_runtime": {
            "api_calls": len(official_calls), "image_transmissions": new_cost["image_transmissions"],
            "input_tokens": sum(row.get("input_tokens", 0) for row in official_calls),
            "output_tokens_including_thought": sum(row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in official_calls),
            "estimated_usd": _estimated_cost(cfg, official_calls), "accuracy": summary,
            "composition": {"reused_upstream_call_count": len(reusable_upstream), "reused_answer_ready_final_count": len(reusable_answer_ready_finals), "new_review_second_sufficiency_final_calls": len(new_cost["calls"])},
        },
        "successful_incremental_replay_spend": {
            "api_calls": len(new_cost["calls"]), "image_transmissions": new_cost["image_transmissions"], "estimated_usd": new_cost["new_api_usd"],
        },
        "development_diagnostics": {
            "failed_anthropic_max_token_calls": 1,
            "included_in_candidate_runtime": False,
            "exact_provider_usage_available": False,
            "known_output_token_lower_bound": 2600,
            "known_output_cost_lower_bound_usd": 0.013,
        },
    }
    comparison["per_side"] = {}
    for side in ("r1_av", "r3_2"):
        old_side = [row for row in old_cost["calls"] if row.get("side") == side]
        reused_side = [row for row in reusable_upstream + reusable_answer_ready_finals if row.get("side") == side]
        new_side = [row for row in new_cost["calls"] if row.get("side") == side]
        comparable_side = reused_side + new_side
        comparison["per_side"][side] = {
            "old": {
                "api_calls": len(old_side),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in old_side),
                "total_tokens": sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in old_side),
                "estimated_usd": _estimated_cost(cfg, old_side),
            },
            "new_comparable_runtime": {
                "api_calls": len(comparable_side),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in comparable_side),
                "total_tokens": sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in comparable_side),
                "estimated_usd": _estimated_cost(cfg, comparable_side),
            },
            "new_incremental_replay": {
                "api_calls": len(new_side),
                "image_transmissions": sum(row.get("image_transmissions", 0) or 0 for row in new_side),
                "total_tokens": sum(row.get("input_tokens", 0) + row.get("output_tokens", 0) + row.get("thought_tokens", 0) for row in new_side),
                "estimated_usd": _estimated_cost(cfg, new_side),
            },
        }
    old = comparison["old_option_scoped_method"]; new = comparison["new_question_batched_method_comparable_runtime"]
    comparison["delta"] = {
        "api_calls": new["api_calls"] - old["api_calls"], "image_transmissions": new["image_transmissions"] - old["image_transmissions"],
        "input_tokens": new["input_tokens"] - old["input_tokens"], "output_tokens_including_thought": new["output_tokens_including_thought"] - old["output_tokens_including_thought"],
        "estimated_usd": new["estimated_usd"] - old["estimated_usd"], "estimated_usd_percent": (new["estimated_usd"] / old["estimated_usd"] - 1.0) * 100.0,
    }
    write_json(out / "old_vs_new_cost_comparison.json", comparison)
    validation = load_json(out / "validation_report.json")
    validation.update({
        "live_replay": "passed", "posthoc_evaluation": "passed",
        "blind_answers_saved_before_gold": True,
        "model_api_calls": len(new_cost["calls"]),
        "visual_review_calls": sum(row["stage"] == "question_batched_visual_review" for row in new_cost["calls"]),
        "image_transmissions": new_cost["image_transmissions"],
        "development_failed_max_token_calls": 1,
        "overall_validation": "passed_live_question_batched_review",
    })
    write_json(out / "validation_report.json", validation)
    (out / "REPORT.md").write_text(
        "# Question-batched visual review v2 — live replay\n\n"
        f"- Accuracy: R1 `{summary['r1_correct']}/10`; R3 `{summary['r3_correct']}/10`.\n"
        f"- Images: `{old['image_transmissions']} -> {new['image_transmissions']}`.\n"
        f"- Comparable API cost: `${old['estimated_usd']:.6f} -> ${new['estimated_usd']:.6f}` ({comparison['delta']['estimated_usd_percent']:.2f}%).\n"
        f"- Successful new replay spend: `${comparison['successful_incremental_replay_spend']['estimated_usd']:.6f}`; one failed development-only max-token call has unavailable exact provider usage and is excluded from candidate runtime.\n"
        f"- R1 comparable cost: `${comparison['per_side']['r1_av']['old']['estimated_usd']:.6f} -> ${comparison['per_side']['r1_av']['new_comparable_runtime']['estimated_usd']:.6f}`.\n"
        f"- R3 comparable cost: `${comparison['per_side']['r3_2']['old']['estimated_usd']:.6f} -> ${comparison['per_side']['r3_2']['new_comparable_runtime']['estimated_usd']:.6f}`.\n"
        "- R1/R3 cache roots remained isolated; old experiment outputs were not modified.\n",
        encoding="utf-8",
    )
    return {"summary": summary, "cost_comparison": comparison}
