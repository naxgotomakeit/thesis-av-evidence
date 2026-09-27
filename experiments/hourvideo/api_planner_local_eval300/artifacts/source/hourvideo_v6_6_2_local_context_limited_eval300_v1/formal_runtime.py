from __future__ import annotations

import datetime
import ast
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_v6_6_1_contract_telemetry_v1 import core as v661
from experiments.hourvideo_v6_6_2_shared_coarse_contract_v1 import core as v662
from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1.prompt_contract import (
    PLANNER_R1_SYSTEM,
    PLANNER_R3_SYSTEM,
)


SIDES = ("r1_av", "r3_2")
PROMPT_SHA = {
    "planner_r1": "c3e27709f94f823f5470e13e21082abdde5dd35b3429913c0d89077387afd799",
    "planner_r3": "d9b9ccadad1d713620a027118e1ab4919f9bb119b070e340a6f3f9944f908100",
    "shared": "a1017b5a3542fde6b907e35e8f050623826902c0e15a0696c3c43cb74967a19b",
    "fine": "f36170283fae6c6e61f05539a88387e9afc10f3694635e75d2d10001e99fcf42",
    "final": "c7145de296ed0ff79f5b05e1f0bd99e7eddb3d76c0253fde235913ad214223b1",
}


def _deferred_versioned_canonical_summary(root: Path, config_path: Path) -> dict[str, Any]:
    """Never write a shared terminal report from a per-side execution pass.

    The old summarizer scanned the whole output root after each side and wrote
    the same validation_report.json, which allowed the final side to overwrite
    the first and to absorb live-gate/cross-side telemetry.  Formal and API
    runners must instead invoke a versioned fixed-population canonical
    aggregator after all routes have durable terminal states.
    """
    return {
        "status": "DEFERRED_TO_VERSIONED_CANONICAL_SUMMARY",
        "gold_loaded": False,
        "shared_top_level_validation_report_written": False,
        "identity_contract": "question_id_plus_route",
    }


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _paths(root: Path, cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    return root / cfg["eligibility_root"], root / cfg["planner_source_experiment"], root / cfg["output_root"]


def verify_frozen_inputs(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    eligibility, _, output = _paths(root, cfg)
    prompt_manifest_path = root / cfg["prompt_freeze_manifest"]
    capacity_path = root / cfg["capacity_summary"]
    prompt = load_json(prompt_manifest_path)
    capacity = load_json(capacity_path)
    if prompt["status"] != "FROZEN_CONFIRMED" or prompt["prompt_content_sha256"] != PROMPT_SHA:
        raise RuntimeError("Prompt freeze manifest does not match confirmed Prompt SHA")
    if capacity["counts"] != {
        "questions": 300, "routes": 600, "r1_eligible": 300, "r1_context_overflow": 0,
        "r3_eligible": 150, "r3_context_overflow": 150, "total_eligible": 450,
        "total_context_overflow": 150, "common_eligible_questions": 150,
    }:
        raise RuntimeError("capacity population differs from confirmed 300/150/450 freeze")
    if cfg["anthropic"]["provider"] != "local_openai" or cfg["gemini"]["provider"] != "local_openai" or cfg["final"]["provider"] != "local_openai":
        raise RuntimeError("all providers must be local_openai")
    if any(cfg[name]["api_key"] != "EMPTY" for name in ("anthropic", "gemini", "final")):
        raise RuntimeError("all local API keys must be EMPTY")
    if cfg["concurrency"] != 1 or cfg["planner_max_output_tokens"] != 8192 or cfg["qwen3_effective_max_model_len"] != 36544:
        raise RuntimeError("local capacity/deployment freeze mismatch")
    for name, artifact in capacity["artifacts"].items():
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError(f"eligibility artifact SHA mismatch: {name}")
    route_rows = [json.loads(line) for line in (eligibility / "route_capacity.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(route_rows) != 600 or len({row["route_key"] for row in route_rows}) != 600:
        raise RuntimeError("route capacity population is not 600 unique routes")
    common = (eligibility / "common_eligible_question_ids.txt").read_text(encoding="utf-8").splitlines()
    r3 = (eligibility / "r3_eligible_question_ids.txt").read_text(encoding="utf-8").splitlines()
    if common != r3 or len(common) != 150:
        raise RuntimeError("paired context-feasible subset is not the frozen R3 eligible set")
    report = {
        "status": "FROZEN_INPUTS_VERIFIED", "checked_at_utc": _now(), "model_called": False,
        "prompt_freeze_manifest_sha256": sha256_file(prompt_manifest_path),
        "capacity_summary_sha256": sha256_file(capacity_path),
        "route_capacity_sha256": capacity["artifacts"]["route_capacity.jsonl"]["sha256"],
        "counts": capacity["counts"], "prompt_sha256": PROMPT_SHA,
    }
    write_json(output.parent / "preflight/frozen_input_verification.json", report)
    return report


def initialize_status_population(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    eligibility, _, output = _paths(root, cfg)
    rows = [json.loads(line) for line in (eligibility / "route_capacity.jsonl").read_text(encoding="utf-8").splitlines()]
    table = output.parent / "preflight/formal_status_table_initial.jsonl"
    if table.exists():
        existing = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines()]
        if existing != rows:
            raise RuntimeError("existing initial formal status table differs from frozen eligibility")
    else:
        for row in rows:
            _append(table, row)
    overflow = 0
    for row in rows:
        if row["eligible"]:
            continue
        overflow += 1
        side_out = output / "cases" / row["question_id"] / row["route"]
        status_path = side_out / "route_status.json"
        status = {
            "video_uid": row["video_uid"], "question_id": row["question_id"], "side": row["route"],
            "formal_status": "context_overflow_pre_model", "execution_status": "failed",
            "reasoning_termination": None, "failure_kind": "context_overflow_pre_model",
            "failure_reason": row["eligibility_reason"], "model_request_count": 0,
            "prediction_present": False, "counts_in_fixed_denominator": True,
            "scored_correct": False, "planner_input_tokens": row["planner_input_tokens"],
            "required_total_tokens": row["required_total_tokens"], "max_model_len": row["max_model_len"],
        }
        if status_path.exists() and load_json(status_path) != status:
            raise RuntimeError(f"refusing to overwrite different overflow status: {row['route_key']}")
        write_json(status_path, status)
    result = {"routes": 600, "prewritten_context_overflow": overflow, "model_requests_for_overflow": 0}
    write_json(output.parent / "preflight/status_initialization.json", result)
    return result


def write_metric_parity_matrix(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    _, _, output = _paths(root, cfg)
    rows = [
        ("correct", "gold and non-empty prediction and exact option-label equality", "same", "post-hoc only"),
        ("fixed_denominator_accuracy", "correct / frozen population denominator", "same", "R1=300; R3=300; paired=150"),
        ("strict_completion", "non-empty prediction with terminal valid output; fallback/downgrade labelled separately", "aligned", "no prediction is incomplete"),
        ("timeout", "terminal timeout remains in denominator with no prediction unless one was durably written", "same", "never dropped"),
        ("failed", "terminal parser/contract/runtime failure remains in denominator", "same", "V6.6.2 retains stage/failure_kind"),
        ("missing_prediction", "frozen population row without a valid prediction", "same", "counts as incorrect"),
        ("e2e_boundary", "route/task start immediately before online reasoning through terminal prediction/failure including retrieval and retries", "aligned", "V6.6.2 post-Planner E2E excludes Frozen Planner generation"),
        ("latency_distribution", "completed terminal rows; mean, median and linear-interpolated P90/P95 over the same declared population", "common analysis contract", "missing E2E reported, never imputed"),
        ("selected_final_route_cost", "attempts on the route that produced the final prediction", "aligned concept", "reported separately from actual total"),
        ("actual_total_cost", "all actual attempts including retries and failed routes", "same", "authoritative resource total"),
        ("stage_names", "method-native stages retained", "not renamed", "V6.6.2 Planner/Shared/Fine/Final are not VideoSEAL Planner/Retriever/Inspector"),
        ("images", "actual physical images sent to the vision model", "same unit", "V6.6.2 Fine images and VideoSEAL Inspector frames remain separately named"),
        ("planner_cost", "separate from online E2E", "method-specific", "V6.6.2 Frozen Planner cost reported separately"),
    ]
    path = output.parent / "preflight/METRIC_PARITY_MATRIX.tsv"
    text = "metric\tdefinition\tparity\tnote\n" + "".join("\t".join(row) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError("existing metric parity matrix differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    limitation = {
        "cross_machine_comparable": ["accuracy", "completion", "failure_count", "model_call_count", "fixed_denominator"],
        "descriptive_only_across_machines": ["absolute_route_e2e", "model_latency", "GPU_time"],
        "prohibited_claim": "No hardware-independent V6.6.2 speedup claim against VideoSEAL from absolute latency.",
        "matrix_sha256": sha256_file(path),
    }
    write_json(output.parent / "preflight/machine_difference_limitation.json", limitation)
    return limitation


def build_static_closure(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    _, _, output = _paths(root, cfg)
    seeds = [
        Path(__file__),
        root / "scripts/experiments/run_hourvideo_v6_6_2_local_context_limited_eval300_v1.py",
        root / "scripts/experiments/serve_hourvideo_v6_6_2_local_context_limited_v1.sh",
    ]
    queue = [path for path in seeds if path.suffix == ".py"]
    code: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in code or not path.is_file():
            continue
        code.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        try:
            module_parts = list(path.relative_to(root / "src").with_suffix("").parts)
        except ValueError:
            module_parts = []
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and module_parts:
                    base_parts = module_parts[:-node.level]
                    if node.module:
                        base_parts.extend(node.module.split("."))
                    base = ".".join(base_parts)
                else:
                    base = node.module or ""
                names = [base]
                names.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
            for name in names:
                if not name.startswith("experiments"):
                    continue
                candidate = root / "src" / Path(*name.split("."))
                resolved = candidate.with_suffix(".py") if candidate.with_suffix(".py").is_file() else candidate / "__init__.py"
                if resolved.is_file() and resolved not in code:
                    queue.append(resolved)
    prompt_dir = (root / cfg["prompt_freeze_manifest"]).parent
    eligibility = root / cfg["eligibility_root"]
    source = root / cfg["source_experiment"]
    files: set[Path] = set(code) | {config_path, *seeds}
    files.update(path for path in prompt_dir.iterdir() if path.is_file())
    files.update(path for path in eligibility.iterdir() if path.is_file())
    files.add(output.parent / "services/service_manifest.json")
    files.update(path for path in (root / "tests/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1").glob("*.py"))
    files.update((source / "case_configs").glob("*.json"))
    files.update((source / "cases").glob("*/question_input.json"))
    files.update({source / "source_manifest.json", source / "ordered_routes.jsonl"})
    source_manifest = load_json(source / "source_manifest.json")
    for assets in source_manifest["video_assets"].values():
        for name, value in assets["asset_paths"].items():
            path = Path(value)
            if sha256_file(path) != assets["asset_sha256"][name]:
                raise RuntimeError(f"source asset closure SHA mismatch: {path}")
            files.add(path)
        siglip = Path(assets["siglip_npz"])
        if sha256_file(siglip) != assets["siglip_npz_sha256"]:
            raise RuntimeError(f"Fine embedding closure SHA mismatch: {siglip}")
        files.add(siglip)
    rows = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(files, key=str)]
    tree_sha = hashlib.sha256("".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode()).hexdigest()
    result = {"schema_version": "v6_6_2_local_context_limited_static_closure_v1", "created_at_utc": _now(), "file_count": len(rows), "tree_sha256": tree_sha, "all_source_asset_sha_verified": True, "files": rows}
    write_json(output.parent / "preflight/static_closure_manifest.json", result)
    return result


def _selected_schema(question: dict[str, Any], requirements: list[dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    unit = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "search_description": {"type": "string"},
            "query_variants": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
            "modality_strategy": {"type": "string"},
            "selected_coarse_ids": {"type": "array", "items": {"type": "integer", "enum": ids}, "uniqueItems": True},
            "selection_reason": {"type": "string"},
        },
        "required": ["search_description", "query_variants", "modality_strategy", "selected_coarse_ids", "selection_reason"],
    }
    rids = [row["requirement_id"] for row in requirements]
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string", "const": question["question_id"]},
            "requirement_plans": {"type": "object", "additionalProperties": False, "properties": {rid: unit for rid in rids}, "required": rids},
            "coarse_lock_is_hard_scope": {"type": "boolean", "const": True},
        },
        "required": ["question_id", "requirement_plans", "coarse_lock_is_hard_scope"],
    }


def _validate_and_project_selected(raw: dict[str, Any], question: dict[str, Any], requirements: list[dict[str, Any]], coarse_ids: list[str]) -> dict[str, Any]:
    if raw.get("question_id") != question["question_id"] or raw.get("coarse_lock_is_hard_scope") is not True:
        raise ValueError("Planner identity or hard-scope field invalid")
    expected = [row["requirement_id"] for row in requirements]
    plans = raw.get("requirement_plans")
    if not isinstance(plans, dict) or list(plans) != expected:
        raise ValueError("Planner requirement coverage/order invalid")
    canonical = []
    for rid in expected:
        row = plans[rid]
        selected = row.get("selected_coarse_ids")
        if not isinstance(selected, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
            raise ValueError(f"selected_coarse_ids must be integer IDs: {rid}")
        if len(selected) != len(set(selected)) or any(value < 0 or value >= len(coarse_ids) for value in selected):
            raise ValueError(f"selected_coarse_ids contains duplicate/invalid ID: {rid}")
        if not str(row.get("selection_reason") or "").strip() or not row.get("query_variants"):
            raise ValueError(f"Planner concise fields invalid: {rid}")
        selected_set = set(selected)
        canonical.append({
            "requirement_id": rid, "search_description": row["search_description"],
            "query_variants": row["query_variants"], "modality_strategy": row["modality_strategy"],
            "selection_reason": row["selection_reason"],
            "selected_coarse_ids": [coarse_ids[index] for index in selected],
            "coarse_judgments": [
                {"coarse_id": coarse_id, "selected": index in selected_set, "reason": row["selection_reason"]}
                for index, coarse_id in enumerate(coarse_ids)
            ],
        })
    plan = {"question_id": question["question_id"], "requirement_plans": canonical, "coarse_lock_is_hard_scope": True}
    v661._base._validate_coarse_locked_plan(plan, question, requirements, set(coarse_ids))
    return plan


def eligible_rows(root: Path, cfg: dict[str, Any], question_id: str | None = None) -> list[dict[str, Any]]:
    eligibility, _, _ = _paths(root, cfg)
    rows = [json.loads(line) for line in (eligibility / "route_capacity.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row["eligible"] and (question_id is None or row["question_id"] == question_id)]


def generate_planners(root: Path, config_path: Path, question_id: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    _, planner_root, output = _paths(root, cfg)
    source = root / cfg["source_experiment"]
    telemetry = planner_root / "planner_attempts.jsonl"
    generated = failed = reused = 0
    for route in eligible_rows(root, cfg, question_id):
        qid, side = route["question_id"], route["route"]
        destination = planner_root / "cases" / qid / side / "planner.json"
        if destination.is_file():
            reused += 1
            continue
        question = load_json(source / "cases" / qid / "question_input.json")
        case_cfg = load_json(source / "case_configs" / f"{qid}.json")
        map_name = "r1_av_navigation_map.json" if side == "r1_av" else "r3_2_navigation_map.json"
        map_path = Path(case_cfg["asset_paths"][map_name])
        map_doc = load_json(map_path)
        coarse_ids = [row["coarse_id"] for row in map_doc["coarse_regions"]]
        requirements = v661.option_requirements(question)
        canonical_payload = {"question": question, "requirements": requirements, "navigation_map": v661._base._planner_map(map_doc)}
        base_payload, provider_ids = v661._base._local_indexed_planner_contract(canonical_payload, coarse_ids)
        schema = _selected_schema(question, requirements, provider_ids)
        system = PLANNER_R1_SYSTEM if side == "r1_av" else PLANNER_R3_SYSTEM
        errors: list[str] = []
        accepted = None
        accepted_usage = None
        for attempt in range(1, int(cfg["max_validation_retries"]) + 2):
            payload = json.loads(json.dumps(base_payload))
            if errors:
                payload["validator_feedback"] = {"previous_error": errors[-1], "instruction": "Regenerate the complete JSON for the same question and unchanged navigation map; fix only this contract error."}
            request_uuid = str(uuid.uuid4())
            start = time.monotonic()
            common = {"request_uuid": request_uuid, "question_id": qid, "video_uid": route["video_uid"], "route": side, "stage": "planner", "attempt": attempt, "is_retry": attempt > 1, "payload_sha256": _digest(payload), "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(), "map_sha256": route["map_sha256"]}
            _append(telemetry, {"event": "request_start", "written_at_utc": _now(), **common})
            try:
                raw, usage = v661._base._anthropic_call(cfg, system, payload, schema, int(cfg["planner_max_output_tokens"]))
                plan = _validate_and_project_selected(raw, question, requirements, coarse_ids)
            except Exception as error:
                record = getattr(error, "attempt_telemetry", {}) or {}
                errors.append(f"{type(error).__name__}: {error}")
                _append(telemetry, {"event": "attempt_end", "written_at_utc": _now(), **common, "status": "failed", "failure_reason": errors[-1], "retry_triggered": attempt <= int(cfg["max_validation_retries"]), "e2e_sec": time.monotonic() - start, **{key: record.get(key) for key in ("model", "input_tokens", "output_tokens", "latency_sec", "response_id", "stop_reason", "raw_text")}})
                continue
            _append(telemetry, {"event": "attempt_end", "written_at_utc": _now(), **common, "status": "accepted", "failure_reason": None, "retry_triggered": False, "e2e_sec": time.monotonic() - start, **{key: usage.get(key) for key in ("model", "input_tokens", "output_tokens", "latency_sec", "response_id", "stop_reason", "raw_text")}})
            accepted, accepted_usage = plan, usage
            break
        if accepted is None:
            failed += 1
            write_json(output / "cases" / qid / side / "route_status.json", {"video_uid": route["video_uid"], "question_id": qid, "side": side, "formal_status": "planner_contract_exhausted", "execution_status": "failed", "reasoning_termination": None, "failure_kind": "planner_contract_exhausted", "failure_reason": errors[-1] if errors else "unknown", "prediction_present": False, "counts_in_fixed_denominator": True})
            continue
        write_json(destination, {"schema_version": "v6_6_2_local_selected_coarse_frozen_planner_v1", "created_at_utc": _now(), "question_id": qid, "video_uid": route["video_uid"], "route": side, "input_asset_path": str(map_path), "input_asset_sha256": route["map_sha256"], "input_payload_sha256": _digest(base_payload), "planner_model": cfg["anthropic"]["model"], "provider": "local_openai", "temperature": 0.0, "max_tokens": 8192, "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(), "output": accepted, "usage": accepted_usage, "gold_loaded": False})
        generated += 1
    result = {"generated": generated, "failed": failed, "reused": reused, "eligible_considered": generated + failed + reused}
    write_json(planner_root / "planner_generation_progress.json", result)
    return result


def run_pipeline_for_available_planners(root: Path, config_path: Path, question_id: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    _, planner_root, _ = _paths(root, cfg)
    original_case_paths = v661._case_paths
    original_sides = v661._configured_sides
    original_terminal_summarize = v662.summarize
    results = {}
    v662.summarize = _deferred_versioned_canonical_summary
    try:
        for side in SIDES:
            allowed = {
                row["question_id"] for row in eligible_rows(root, cfg, question_id)
                if row["route"] == side and (planner_root / "cases" / row["question_id"] / side / "planner.json").is_file()
            }
            if not allowed:
                results[side] = {"eligible_with_planner": 0}
                continue
            def filtered_case_paths(cfg_: dict[str, Any], root_: Path, identity_filter: str | None):
                rows = original_case_paths(cfg_, root_, identity_filter)
                return [row for row in rows if row[0]["question_id"] in allowed]
            v661._case_paths = filtered_case_paths
            v661._configured_sides = lambda cfg_, selected=side: (selected,)
            try:
                results[side] = v662.run_live(root, config_path, video_uid=None)
            finally:
                v661._case_paths = original_case_paths
                v661._configured_sides = original_sides
    finally:
        v662.summarize = original_terminal_summarize
    return results


def run_gate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    eligibility, _, output = _paths(root, cfg)
    gate_qid = (eligibility / "common_eligible_question_ids.txt").read_text(encoding="utf-8").splitlines()[0]
    planner = generate_planners(root, config_path, gate_qid)
    pipeline = run_pipeline_for_available_planners(root, config_path, gate_qid)
    route_results = []
    for side in SIDES:
        status_path = output / "cases" / gate_qid / side / "route_status.json"
        if not status_path.is_file():
            raise RuntimeError(f"gate route produced no durable status: {gate_qid}/{side}")
        status = load_json(status_path)
        attempts_path = output / "cases" / gate_qid / "model_attempts.jsonl"
        route_results.append({"question_id": gate_qid, "route": side, "status": status, "attempt_telemetry_present": attempts_path.is_file(), "prediction_present": (output / "cases" / gate_qid / side / "final_answer.json").is_file()})
    result = {"status": "PASS", "gate_question_id": gate_qid, "correctness_checked": False, "planner": planner, "pipeline": pipeline, "routes": route_results}
    write_json(output.parent / "preflight/live_gate_report.json", result)
    return result


def run_formal(root: Path, config_path: Path) -> dict[str, Any]:
    generate_planners(root, config_path, None)
    return run_pipeline_for_available_planners(root, config_path, None)
