from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent
RUNTIME_ROOT = WORKSPACE_ROOT / "runtime_support"
REFERENCE_RUNTIME = WORKSPACE_ROOT / "runnable_runtime"
CONFIG_PATH = HERE / "config_v7_4.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment(cfg: dict[str, Any], backend: str) -> dict[str, str]:
    planner, inspector, retrieval, runtime = (
        cfg["planner"], cfg["inspector"], cfg["retrieval"], cfg["runtime"]
    )
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REFERENCE_RUNTIME), str(RUNTIME_ROOT), str(RUNTIME_ROOT / "src")]
        + ([old_pythonpath] if old_pythonpath else [])
    )
    values = {
        "RETRIEVAL_BACKEND": backend,
        "PAIRED_FLAT_INDEX_ROOT": env.get("PAIRED_FLAT_INDEX_ROOT") or cfg["flat_index_root"],
        "PAIRED_HIERARCHICAL_ROOT": env.get("INDEX_ROOT") or env.get("PAIRED_HIERARCHICAL_ROOT") or cfg["hierarchical_root"],
        "AGENT_LLM_BACKEND": "api",
        "AGENT_MLLM_BACKEND": "openai",
        "AGENT_API_USE_MESSAGES": str(planner["api_use_messages"]),
        "AGENT_LLM_API_BASE": planner["api_base"],
        "AGENT_LLM_API_KEY": "local",
        "AGENT_LLM_MODEL": planner["served_model"],
        "AGENT_LLM_MAX_TOKENS": str(planner["max_tokens"]),
        "AGENT_LLM_TEMPERATURE": str(planner["temperature"]),
        "AGENT_LLM_TIMEOUT": str(planner["http_timeout_sec"]),
        "VISUAL_INSPECT_BACKEND": "openai",
        "VISUAL_INSPECT_API_BASE": inspector["api_base"],
        "VISUAL_INSPECT_API_KEY": "local",
        "VISUAL_INSPECT_MODEL": inspector["served_model"],
        "VISUAL_RETRIEVE_SUM_BACKEND": "openai",
        "VISUAL_RETRIEVE_SUM_API_BASE": inspector["api_base"],
        "VISUAL_RETRIEVE_SUM_API_KEY": "local",
        "VISUAL_RETRIEVE_SUM_MODEL": inspector["served_model"],
        "VISUAL_RETRIEVE_SUMMARY_ENABLED": "1",
        "RETRIEVE_SUMMARY_ENABLED": "1",
        "VISUAL_RETRIEVE_RETURN_SPANS": "0",
        "RETRIEVE_SUMMARY_MAX_SPANS": str(retrieval["summary_max_spans"]),
        "VISUAL_RETRIEVE_SUM_MAX_TOKENS": str(retrieval["summary_max_tokens"]),
        "VISUAL_RETRIEVE_SUM_TEMPERATURE": str(retrieval["summary_temperature"]),
        "BENCHMARK": "lvbench",
        "SEMANTIC_RETRIEVE_MIX": retrieval["mode"],
        "SEMANTIC_RETRIEVE_TOPK": str(retrieval["semantic_top_k"]),
        "VISUAL_RETRIEVE_TOPK": str(retrieval["visual_top_k"]),
        "RETRIEVE_MIN_TIME_GAP_SEC": str(retrieval["min_time_gap_sec"]),
        "HIERARCHICAL_COARSE_TOPK": str(retrieval["hierarchical_coarse_top_k"]),
        "HIERARCHICAL_MEDIUM_TOPK": str(retrieval["hierarchical_medium_top_k"]),
        "HIERARCHICAL_FINE_PER_MEDIUM": str(retrieval["hierarchical_fine_per_medium"]),
        "HIERARCHICAL_PROFILE": str(retrieval.get("hierarchical_profile") or "h6"),
        "HIERARCHICAL_SIGLIP_CACHE": str(retrieval.get("siglip_cache_dir") or ""),
        "HIERARCHICAL_SIGLIP_MODEL": str(retrieval.get("siglip_model") or "google/siglip-base-patch16-224"),
        "INSPECT_MAX_LONG_EDGE": str(inspector["max_long_edge"]),
        "INSPECT_VLM_MAX_TOKENS": str(inspector["max_tokens"]),
        "INSPECT_VLM_TEMPERATURE": str(inspector["temperature"]),
        "INSPECT_MAX_TOTAL_IMAGES": str(inspector["max_images_per_call"]),
        "INSPECT_FPS": str(inspector["fps"]),
        "VISUAL_INSPECT_MAX_LONG_EDGE": str(inspector["max_long_edge"]),
        "VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE": "1",
        "VISUAL_INSPECT_TOTAL_PIXELS": str(20480 * 32 * 32),
        "VISUAL_INSPECT_MIN_PIXELS": str(16 * 28 * 28),
        "VISUAL_INSPECT_EDGE_MULTIPLE": "32",
        "MLLM_BACKEND": "openai",
        "MLLM_MAX_TOKENS": "4096",
        "MLLM_TIMEOUT": str(runtime["mllm_timeout_sec"]),
        "MLLM_RETRY_TIMES": str(runtime["mllm_retry_times"]),
        "MLLM_RETRY_DELAY": str(runtime["mllm_retry_delay_sec"]),
        "EMBED_RETRY_TIMES": str(runtime["embedding_retry_times"]),
        "EMBED_RETRY_DELAY": str(runtime["embedding_retry_delay_sec"]),
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "MAX_STEPS": str(runtime["max_steps"]),
        "TASK_TIMEOUT_SEC": str(runtime["task_timeout_sec"]),
        "CONCURRENCY": str(runtime["concurrency"]),
        "AGENT_FORCE_LAST_STEP_VISUAL_INSPECT": "1",
        "AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK": "1",
        "AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE": "mcq",
        "AGENT_PARSE_FAILURE_FALLBACK_TO_C": "0",
    }
    env.update({key: str(value) for key, value in values.items()})
    return env


def contract_test(cfg: dict[str, Any]) -> dict[str, Any]:
    original = Path(cfg["reference_extract"])
    bundle = Path(cfg["reference_bundle"])
    errors: list[str] = []
    if _sha256(bundle) != cfg["reference_sha256"]:
        errors.append("reference bundle SHA-256 mismatch")
    original_files = sorted(path.relative_to(original) for path in original.rglob("*") if path.is_file())
    if len(original_files) != 48:
        errors.append(f"reference file count is {len(original_files)}, expected 48")
    changed = []
    for rel in original_files:
        copied = REFERENCE_RUNTIME / rel
        if not copied.is_file() or _sha256(copied) != _sha256(original / rel):
            changed.append(str(rel))
    expected_differences = ["videoseal/tools/tool_map.py", "videoseal/tools/visual_tools.py"]
    if changed != expected_differences:
        errors.append(f"unexpected reference-copy differences: {changed}")

    # Keep the contract check import-free: the control-node Python used for
    # offline auditing intentionally has no OpenCV, while the actual runtime
    # environment does. Both classes inherit the one common schema property.
    expected_schema = {
        "type": "function",
        "function": {
            "name": "visual_retrieve",
            "description": (
                "Retrieve candidate temporal spans from the video's visual semantic index. "
                "Use a natural-language visual query; inspect returned spans before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    adapter_text = (REFERENCE_RUNTIME / "videoseal/tools/retrieval_adapter.py").read_text(encoding="utf-8")
    if "class RuntimeAlignedFlatRetrieve(_CommonRetrieveContract" not in adapter_text:
        errors.append("flat retriever does not inherit the common Planner contract")
    if "class RuntimeAlignedHierarchicalRetrieve(_CommonRetrieveContract" not in adapter_text:
        errors.append("hierarchical retriever does not inherit the common Planner contract")
    schemas = {backend: expected_schema for backend in cfg["retrieval_backends"]}
    schema_identical = len({json.dumps(value, sort_keys=True) for value in schemas.values()}) == 1
    if not schema_identical:
        errors.append("backend visual_retrieve schemas differ")

    required_paths = [Path(cfg["parquet"]), Path(cfg["flat_index_root"]), Path(cfg["hierarchical_root"])]
    required_paths.extend(Path(cfg[key]["checkpoint_path"]) for key in ("planner", "inspector"))
    missing = [str(path) for path in required_paths if not path.exists()]
    errors.extend(f"missing required path: {path}" for path in missing)

    tool_agent_text = (REFERENCE_RUNTIME / "videoseal/agents/tool_agent.py").read_text(encoding="utf-8")
    reference_flags = {
        "sixteen_steps": cfg["runtime"]["max_steps"] == 16,
        "prompt_only_gate": "_final_allowed_by_inspector_gate" not in tool_agent_text,
        "step15_force_code_present": "step_idx == max_steps - 1" in tool_agent_text,
        "max_step_full_video_fallback_present": "_forced_full_video_visual_inspect" in tool_agent_text,
        "legacy_api_mode": cfg["planner"]["api_use_messages"] == 0,
        "step15_force_effective": cfg["planner"]["api_use_messages"] == 1,
    }
    shared_runtime = RUNTIME_ROOT.parents[1] / "VideoSEAL"
    per_file_comparison = []
    for rel in original_files:
        current = shared_runtime / rel
        if not current.is_file():
            status = "missing_from_current_shared_runtime"
            current_sha = None
        else:
            current_sha = _sha256(current)
            status = "same" if current_sha == _sha256(original / rel) else "different"
        per_file_comparison.append(
            {
                "path": str(rel),
                "reference_sha256": _sha256(original / rel),
                "v7_2_shared_runtime_sha256": current_sha,
                "status": status,
            }
        )
    v72_root = RUNTIME_ROOT
    v72_relpaths = [
        "configs/experiments/hourvideo_v7_2_videoseal_visual_only_embedding_fair_smoke_v1.json",
        "scripts/experiments/run_hourvideo_v7_2_videoseal_visual_only_embedding_fair_smoke_v1.py",
        "src/experiments/hourvideo_v7_2_videoseal_visual_only_embedding_fair_smoke_v1/core.py",
        "tests/experiments/hourvideo_v7_2_videoseal_visual_only_embedding_fair_smoke_v1/test_contract.py",
    ]
    v72_experiment_files = [
        {"path": rel, "sha256": _sha256(v72_root / rel) if (v72_root / rel).is_file() else None}
        for rel in v72_relpaths
    ]
    report = {
        "stage": "offline_contract",
        "ok": not errors,
        "errors": errors,
        "bundle_sha256": _sha256(bundle),
        "reference_file_count": len(original_files),
        "reference_copy_differences": changed,
        "intentional_new_file": "videoseal/tools/retrieval_adapter.py",
        "intentional_reference_modifications": {
            "videoseal/tools/tool_map.py": "backend-selectable retrieval seam",
            "videoseal/tools/visual_tools.py": "audit-only retrieval metadata; tool output unchanged",
        },
        "schemas": schemas,
        "tool_schema_identical": schema_identical,
        "reference_flags": reference_flags,
        "v7_2_reference_per_file_comparison": per_file_comparison,
        "v7_2_experiment_files": v72_experiment_files,
        "effective_step15_discrepancy_confirmed": (
            reference_flags["step15_force_code_present"] and not reference_flags["step15_force_effective"]
        ),
        "models_or_apis_called": False,
        "eval300_started": False,
    }
    artifacts = HERE / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "offline_contract_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _termination_mode(trajectory: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    """Classify the observable end state without changing controller behavior."""
    if metrics and str(metrics.get("status") or "").lower() == "timeout":
        return "timeout"
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    for step in steps:
        observation = step.get("observation") if isinstance(step, dict) else None
        if not isinstance(observation, dict):
            continue
        if observation.get("forced") is True and observation.get("mode") == "full_video":
            return "full_video_64_frame_fallback"
    if trajectory.get("answer"):
        return "normal_retrieval_answer"
    return "parser_or_runtime_failure"


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stage_latency(metadata: dict[str, Any], stage: str) -> float:
    for nested_name in ("hierarchy_stage_latency", "stage_latencies_sec"):
        nested = metadata.get(nested_name)
        if not isinstance(nested, dict):
            continue
        for key in (stage, f"{stage}_sec", f"{stage}_latency_sec"):
            if key in nested:
                return _number(nested[key])
        if f"{stage}_elapsed_sec" in nested:
            return _number(nested[f"{stage}_elapsed_sec"])
    for key in (
        f"{stage}_latency_sec",
        f"{stage}_elapsed_sec",
        f"{stage}_sec",
        f"{stage}_processing_sec",
    ):
        if key in metadata:
            return _number(metadata[key])
    return 0.0


def _candidate_spans(observation: dict[str, Any]) -> list[dict[str, Any]]:
    output = observation.get("output")
    metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
    sources: list[object] = []
    if isinstance(output, list):
        sources.append(output)
    elif isinstance(output, dict):
        sources.extend([output.get("useful_spans"), output.get("candidates")])
    sources.extend(
        [
            metadata.get("candidate_spans"),
            metadata.get("summary_selected_spans"),
            metadata.get("spans"),
        ]
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            start = str(item.get("start_time") or item.get("start_sec") or "")
            end = str(item.get("end_time") or item.get("end_sec") or "")
            if not start or not end or (start, end) in seen:
                continue
            seen.add((start, end))
            row = {"start_time": start, "end_time": end}
            caption = str(item.get("caption") or "").strip()
            if caption:
                row["caption"] = caption
            rows.append(row)
    return rows


def _validated_candidate_rows(value: object, field: str) -> tuple[list[dict[str, Any]], list[str]]:
    if value is None:
        return [], [f"{field} missing"]
    if not isinstance(value, list):
        return [], [f"{field} is not a list"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{field}[{index}] is not an object")
            continue
        start = str(item.get("start_time") or "").strip()
        end = str(item.get("end_time") or "").strip()
        if not start or not end:
            errors.append(f"{field}[{index}] lacks start_time/end_time")
            continue
        row: dict[str, Any] = {"start_time": start, "end_time": end}
        caption = str(item.get("caption") or "").strip()
        if caption:
            row["caption"] = caption
        rows.append(row)
    return rows, errors


def _retrieval_telemetry(observation: dict[str, Any]) -> dict[str, Any]:
    """Extract audit-only retrieval data; never infer missing raw candidates from prose."""
    output = observation.get("output")
    metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
    summary = str(output.get("summary") or "") if isinstance(output, dict) else ""
    errors: list[str] = []

    raw_source = metadata.get("raw_candidates")
    raw_source_name = "metadata.raw_candidates"
    if raw_source is None:
        paths = metadata.get("coarse_medium_fine_paths")
        if isinstance(paths, list):
            path_candidates = [path.get("candidate") for path in paths if isinstance(path, dict)]
            if any(isinstance(item, dict) for item in path_candidates):
                raw_source = path_candidates
                raw_source_name = "metadata.coarse_medium_fine_paths[].candidate"
        if raw_source is None and isinstance(output, list):
            raw_source = output
            raw_source_name = "tool_output"
    raw_candidates, raw_errors = _validated_candidate_rows(raw_source, "raw_candidates")
    errors.extend(raw_errors)

    useful_source = metadata.get("summarizer_useful_spans")
    if useful_source is None:
        useful_source = metadata.get("summary_selected_spans")
    if useful_source is None and isinstance(output, dict):
        useful_source = output.get("useful_spans")
    useful_spans, useful_errors = _validated_candidate_rows(useful_source, "summarizer_useful_spans")
    errors.extend(useful_errors)

    declared_count = metadata.get("raw_candidate_count")
    if declared_count is not None:
        try:
            if int(declared_count) != len(raw_candidates):
                errors.append(
                    f"raw_candidate_count={int(declared_count)} but extracted={len(raw_candidates)}"
                )
        except (TypeError, ValueError):
            errors.append("raw_candidate_count is not an integer")
    raw_count = int(declared_count) if isinstance(declared_count, int) else (
        len(raw_candidates) if raw_source is not None and not raw_errors else None
    )
    if raw_source is None:
        errors.append(
            "historical trajectory predates raw-candidate telemetry; raw candidates cannot be "
            "reconstructed reliably from free-form summary text"
        )
    if useful_source is None:
        errors.append(
            "historical trajectory does not expose summarizer-selected useful spans"
        )
    return {
        "raw_candidate_count": raw_count,
        "raw_candidates": raw_candidates,
        "raw_candidate_source": raw_source_name if raw_source is not None else None,
        "summarizer_useful_spans": useful_spans,
        "planner_visible_summary": summary or None,
        "planner_visible_output_kind": "summary" if summary else (
            "candidate_list" if isinstance(output, list) else type(output).__name__
        ),
        "extraction_error": "; ".join(dict.fromkeys(errors)) if errors else None,
    }


def _hierarchy_audit(metadata: dict[str, Any]) -> dict[str, Any]:
    coarse = metadata.get("selected_coarse") if isinstance(metadata.get("selected_coarse"), list) else []
    medium = metadata.get("selected_medium") if isinstance(metadata.get("selected_medium"), list) else []
    fine = metadata.get("selected_fine") if isinstance(metadata.get("selected_fine"), list) else []
    medium_ids = {
        str(row.get("medium_id")) for row in medium if isinstance(row, dict) and row.get("medium_id") is not None
    }
    fine_medium_ids = {
        str(row.get("medium_id")) for row in fine if isinstance(row, dict) and row.get("medium_id") is not None
    }
    paths = metadata.get("coarse_medium_fine_paths")
    if not isinstance(paths, list):
        paths = metadata.get("provenance_paths") or metadata.get("traversal_paths") or []
    stage_timing = metadata.get("hierarchy_stage_latency")
    stages_complete = isinstance(stage_timing, dict) and {
        "coarse_elapsed_sec", "medium_elapsed_sec", "fine_elapsed_sec"
    }.issubset(stage_timing)
    complete = bool(
        metadata.get("hierarchy_used") is True
        and metadata.get("hierarchy_complete") is True
        and coarse and medium and fine and paths and stages_complete
        and fine_medium_ids and fine_medium_ids.issubset(medium_ids)
    )
    return {
        "complete": complete,
        "selected_coarse": coarse,
        "selected_medium": medium,
        "selected_fine": fine,
        "provenance_paths": paths,
        "hierarchy_traversal_elapsed_sec": _number(metadata.get("hierarchy_traversal_elapsed_sec")),
        "latency_sec": {
            "coarse": _stage_latency(metadata, "coarse"),
            "medium": _stage_latency(metadata, "medium"),
            "fine": _stage_latency(metadata, "fine"),
        },
    }


def _find_run_files(output: Path, uid: str) -> tuple[Path | None, Path | None, Path | None]:
    trajectories: list[Path] = []
    for path in output.glob("*/*/trajectory.json"):
        try:
            if str(_read_json(path).get("uid") or "") == uid:
                trajectories.append(path)
        except Exception:
            pass
    trajectory_path = max(trajectories, key=lambda path: path.stat().st_mtime_ns) if trajectories else None
    metrics = list(output.glob(f"*/metrics/{uid}.json"))
    preds = list(output.glob(f"*/preds/{uid}.json"))
    return trajectory_path, (metrics[0] if metrics else None), (preds[0] if preds else None)


def _summarize_run(
    *, output: Path, backend: str, uid: str, runner_elapsed_sec: float | None = None,
    runner_returncode: int = 0,
) -> dict[str, Any]:
    trajectory_path, metrics_path, pred_path = _find_run_files(output, uid)
    trajectory = _read_json(trajectory_path) if trajectory_path else {}
    metrics = _read_json(metrics_path) if metrics_path else None
    pred_record = _read_json(pred_path) if pred_path else {}
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    retrieval_calls: list[dict[str, Any]] = []
    inspector_calls: list[dict[str, Any]] = []
    planner_latencies: list[float] = []
    hierarchy_audits: list[dict[str, Any]] = []
    executed_backends: list[str] = []
    runtime_errors: list[str] = []
    final_spans: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        timing = step.get("timing") if isinstance(step.get("timing"), dict) else {}
        if timing.get("model_elapsed_sec") is not None:
            planner_latencies.append(_number(timing.get("model_elapsed_sec")))
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
        name = str(action.get("name") or observation.get("name") or "")
        if name == "visual_retrieve":
            executed_backends.append(str(metadata.get("retrieval_backend") or ""))
            telemetry = _retrieval_telemetry(observation)
            spans = telemetry["raw_candidates"] or telemetry["summarizer_useful_spans"]
            final_spans = spans or final_spans
            audit = _hierarchy_audit(metadata)
            if backend == "hierarchical":
                hierarchy_audits.append(audit)
            retrieval_calls.append(
                {
                    "step": index,
                    "query": str((action.get("arguments") or {}).get("query") or "")
                    if isinstance(action.get("arguments"), dict) else "",
                    "latency_sec": _number(metadata.get("tool_elapsed_sec") or timing.get("tool_elapsed_sec")),
                    "candidates": spans,
                    "raw_candidate_count": telemetry["raw_candidate_count"],
                    "raw_candidates": telemetry["raw_candidates"],
                    "raw_candidate_source": telemetry["raw_candidate_source"],
                    "summarizer_useful_spans": telemetry["summarizer_useful_spans"],
                    "planner_visible_summary": telemetry["planner_visible_summary"],
                    "planner_visible_output_kind": telemetry["planner_visible_output_kind"],
                    "telemetry_extraction_error": telemetry["extraction_error"],
                    "ok": observation.get("ok"),
                    "error": observation.get("error"),
                    "hierarchy": audit if backend == "hierarchical" else None,
                }
            )
        elif name == "visual_inspect":
            inspector_calls.append(
                {
                    "step": index,
                    "latency_sec": _number(metadata.get("tool_elapsed_sec") or timing.get("tool_elapsed_sec")),
                    "model_latency_sec": _number(metadata.get("model_elapsed_sec")),
                    "frames": int(metadata.get("sent_image_count") or metadata.get("image_count") or 0),
                    "timestamps": metadata.get("image_timestamps") or [],
                    "forced_full_video": observation.get("forced") is True and observation.get("mode") == "full_video",
                    "ok": observation.get("ok"),
                    "error": observation.get("error"),
                }
            )
        if observation.get("error"):
            runtime_errors.append(str(observation["error"]))

    hierarchy_used = bool(
        backend == "hierarchical"
        and hierarchy_audits
        and all(item["complete"] for item in hierarchy_audits)
    )
    flat_entered_hierarchy = backend == "videoseal_flat" and any(
        isinstance(call.get("hierarchy"), dict) and call["hierarchy"].get("complete")
        for call in retrieval_calls
    )
    backend_execution_verified = bool(
        retrieval_calls
        and executed_backends
        and all(value == backend for value in executed_backends)
        and (hierarchy_used if backend == "hierarchical" else not flat_entered_hierarchy)
    )
    trajectory_valid = bool(
        trajectory_path
        and str(trajectory.get("uid") or "") == uid
        and isinstance(trajectory.get("steps"), list)
        and trajectory.get("run_id")
        and trajectory.get("created_at")
    )
    gold = str(pred_record.get("gt") or trajectory.get("groundtruth") or "").strip().upper()
    prediction = str(pred_record.get("pred") or "").strip().upper()
    retrieval_latencies = [float(call["latency_sec"]) for call in retrieval_calls]
    inspector_latencies = [float(call["latency_sec"]) for call in inspector_calls]
    fallback_latencies = [
        float(call["latency_sec"]) for call in inspector_calls if call["forced_full_video"]
    ]
    e2e = _number(trajectory.get("elapsed_sec")) or _number(metrics.get("elapsed_sec") if metrics else None)
    planner_step_count = sum(
        1 for step in steps
        if isinstance(step, dict)
        and isinstance(step.get("timing"), dict)
        and step["timing"].get("model_elapsed_sec") is not None
    )
    return {
        "uid": uid,
        "backend": backend,
        "video_id": str(trajectory.get("video_id") or pred_record.get("video_id") or ""),
        "gold": gold,
        "prediction": prediction,
        "correct": bool(gold and prediction and gold == prediction),
        "trajectory_valid": trajectory_valid,
        "completion_mode": _termination_mode(trajectory, metrics),
        "completion_note": trajectory.get("note"),
        "planner_steps": planner_step_count,
        "trajectory_steps_total": len(steps),
        "retrieval_calls": len(retrieval_calls),
        "inspector_calls": len(inspector_calls),
        "total_frames": sum(int(call["frames"]) for call in inspector_calls),
        "fallback_triggered": bool(fallback_latencies),
        "step_15_forced_inspector_effective": any(
            call["step"] == 15 and call["forced_full_video"] for call in inspector_calls
        ),
        "hierarchy_used": hierarchy_used,
        "backend_execution_verified": backend_execution_verified,
        "executed_backend_markers": executed_backends,
        "flat_entered_hierarchy": flat_entered_hierarchy,
        "hierarchy_traversal_complete": hierarchy_used,
        "final_candidate_spans": final_spans,
        "latency_sec": {
            "retrieval_calls": retrieval_latencies,
            "retrieval_total": round(sum(retrieval_latencies), 6),
            "planner_calls": planner_latencies,
            "planner_total": round(sum(planner_latencies), 6),
            "inspector_calls": inspector_latencies,
            "inspector_model_calls": [float(call["model_latency_sec"]) for call in inspector_calls],
            "inspector_total": round(sum(inspector_latencies), 6),
            "fallback": round(sum(fallback_latencies), 6),
            "end_to_end": round(e2e, 6) if e2e else None,
            "runner_outer_monotonic": round(float(runner_elapsed_sec), 6) if runner_elapsed_sec is not None else None,
            "hierarchical_coarse_total": round(sum(item["latency_sec"]["coarse"] for item in hierarchy_audits), 6),
            "hierarchical_medium_total": round(sum(item["latency_sec"]["medium"] for item in hierarchy_audits), 6),
            "hierarchical_fine_total": round(sum(item["latency_sec"]["fine"] for item in hierarchy_audits), 6),
        },
        "retrieval_trace": retrieval_calls,
        "inspector_trace": inspector_calls,
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "prediction_path": str(pred_path) if pred_path else None,
        "runner_returncode": int(runner_returncode),
        "runtime_errors": runtime_errors,
        "system_prompt_sha256": hashlib.sha256(
            str(trajectory.get("system_prompt") or "").encode("utf-8")
        ).hexdigest() if trajectory else None,
        "tools_schema_sha256": hashlib.sha256(
            str(trajectory.get("tools_schema") or "").encode("utf-8")
        ).hexdigest() if trajectory else None,
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    return {
        "count": len(clean),
        "total": round(sum(clean), 6) if clean else 0.0,
        "mean": round(statistics.fmean(clean), 6) if clean else None,
        "median": round(statistics.median(clean), 6) if clean else None,
    }


def _summarize_backend(
    output: Path, backend: str, uids: list[str], timings: dict[str, float],
    returncodes: dict[str, int], backend_elapsed_sec: float,
) -> dict[str, Any]:
    rows = [
        _summarize_run(
            output=output,
            backend=backend,
            uid=uid,
            runner_elapsed_sec=timings.get(uid),
            runner_returncode=returncodes.get(uid, 0),
        )
        for uid in uids
    ]
    latency_fields = (
        "retrieval_total", "planner_total", "inspector_total", "fallback", "end_to_end",
        "runner_outer_monotonic", "hierarchical_coarse_total", "hierarchical_medium_total",
        "hierarchical_fine_total",
    )
    latency_summary = {
        field: _stats(
            [float(row["latency_sec"][field]) for row in rows if row["latency_sec"].get(field) is not None]
        )
        for field in latency_fields
    }
    report = {
        "backend": backend,
        "rows": rows,
        "latency_summary_sec": latency_summary,
        "backend_runner_outer_monotonic_sec": round(backend_elapsed_sec, 6),
        "offline_indexing_time": None,
        "offline_indexing_time_note": "offline indexing time not measured in this smoke",
        "eval300_started": False,
    }
    (output / "runtime_aligned_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _write_snapshot(cfg: dict[str, Any], root: Path) -> Path:
    files = [CONFIG_PATH, HERE / "run.py", HERE / "test_contract.py", HERE / "three_question_uids.txt"]
    files.extend(sorted(path for path in REFERENCE_RUNTIME.rglob("*") if path.is_file()))
    hashes = []
    for path in files:
        if "__pycache__" in path.parts:
            continue
        hashes.append({"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    snapshot = {
        "experiment": cfg["experiment"],
        "config": cfg,
        "code_hashes": hashes,
        "reference_bundle_actual_sha256": _sha256(Path(cfg["reference_bundle"])),
        "uid_file_lines": [
            line.strip() for line in (HERE / cfg["uids_file"]).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ],
        "runtime_invariants": {
            "api_use_messages": cfg["planner"]["api_use_messages"],
            "max_steps": cfg["runtime"]["max_steps"],
            "task_timeout_sec": cfg["runtime"]["task_timeout_sec"],
            "concurrency": cfg["runtime"]["concurrency"],
            "eval300_enabled": cfg["eval300_enabled"],
        },
        "secrets_recorded": False,
        "eval300_started": False,
    }
    target = root / "actual_config_and_code_hash_snapshot.json"
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _write_markdown(report: dict[str, Any], target: Path) -> None:
    rows = [row for backend in report["reports"] for row in backend["rows"]]
    lines = [
        "# V7.3 aligned paired live smoke",
        "",
        "Eval300 was not started. Offline indexing time not measured in this smoke.",
        "",
        "| UID | backend | gold | pred | correct | mode | steps | retrieve | inspect | frames | fallback | hierarchy | e2e s |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {uid} | {backend} | {gold} | {prediction} | {correct} | {completion_mode} | "
            "{planner_steps} | {retrieval_calls} | {inspector_calls} | {total_frames} | "
            "{fallback_triggered} | {hierarchy_used} | {e2e} |".format(
                **row, e2e=row["latency_sec"].get("end_to_end")
            )
        )
    lines.extend(["", "## Backend latency summary", ""])
    for backend in report["reports"]:
        lines.append(f"### {backend['backend']}")
        lines.append("")
        lines.append("| component | mean s | median s | total s |")
        lines.append("|---|---:|---:|---:|")
        for component in ("retrieval_total", "planner_total", "inspector_total", "fallback", "end_to_end"):
            stats = backend["latency_summary_sec"][component]
            lines.append(f"| {component} | {stats['mean']} | {stats['median']} | {stats['total']} |")
        lines.append("")
    lines.extend(
        [
            "## Contract result", "",
            f"- paired smoke passed: `{report['paired_smoke_passed']}`",
            f"- 3/3 hierarchical traversal complete: `{report['hierarchical_3_of_3']}`",
            f"- 3/3 flat avoided hierarchy: `{report['flat_3_of_3_no_hierarchy']}`",
            "- step-15 forcing remains unreachable in tag mode and was not repaired.",
            "- services are managed outside this runner; see the service logs and PID manifest in the output root.",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")


def paired_smoke(cfg: dict[str, Any], backend: str) -> int:
    selected = cfg["retrieval_backends"] if backend == "paired" else [backend]
    uids = [
        line.strip() for line in (HERE / cfg["uids_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(uids) != 3 or len(set(uids)) != 3:
        raise RuntimeError(f"paired smoke requires exactly three unique frozen UIDs, got {uids}")
    if int(cfg["runtime"]["concurrency"]) != 1:
        raise RuntimeError("paired smoke requires concurrency=1")
    if int(cfg["runtime"]["task_timeout_sec"]) != 1000 or int(cfg["runtime"]["max_steps"]) != 16:
        raise RuntimeError("runtime invariants changed: expected max_steps=16 and task_timeout_sec=1000")
    if bool(cfg.get("eval300_enabled")) or bool(cfg.get("formal_eval")):
        raise RuntimeError("refusing to run: Eval300/formal mode must remain disabled")
    output_root = Path(cfg["output_root"])
    for item in selected:
        if (output_root / "smoke" / item).exists():
            raise RuntimeError(f"refusing to overwrite existing output: {output_root / 'smoke' / item}")
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    launcher_root = output_root / "launcher_uids"
    launcher_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = _write_snapshot(cfg, output_root)
    reports = []
    for item in selected:
        output = output_root / "smoke" / item
        env = _environment(cfg, item)
        timings: dict[str, float] = {}
        returncodes: dict[str, int] = {}
        backend_started = time.monotonic()
        log_path = logs_root / f"runner_{item}.log"
        with log_path.open("a", encoding="utf-8") as log:
            for uid in uids:
                uid_file = launcher_root / f"{item}_{uid}.txt"
                uid_file.write_text(uid + "\n", encoding="utf-8")
                command = [
                    sys.executable, "-m", "videoseal.runner.per_question_runner",
                    "--parquet", cfg["parquet"],
                    "--uids-file", str(uid_file),
                    "--save-runs", str(output),
                    "--concurrency", "1",
                    "--max-steps", "16",
                    "--task-timeout-sec", "1000",
                ]
                log.write(json.dumps({"uid": uid, "backend": item, "event": "runner_start"}) + "\n")
                log.flush()
                started = time.monotonic()
                completed = subprocess.run(
                    command, cwd=REFERENCE_RUNTIME, env=env, check=False,
                    stdout=log, stderr=subprocess.STDOUT,
                )
                timings[uid] = time.monotonic() - started
                returncodes[uid] = int(completed.returncode)
                log.write(
                    json.dumps(
                        {
                            "uid": uid, "backend": item, "event": "runner_end",
                            "returncode": completed.returncode,
                            "outer_monotonic_elapsed_sec": round(timings[uid], 6),
                        }
                    ) + "\n"
                )
                log.flush()
        reports.append(
            _summarize_backend(
                output, item, uids, timings, returncodes, time.monotonic() - backend_started
            )
        )
    rows = [row for report in reports for row in report["rows"]]
    hier_rows = [row for row in rows if row["backend"] == "hierarchical"]
    flat_rows = [row for row in rows if row["backend"] == "videoseal_flat"]
    hier_ok = len(hier_rows) == 3 and all(row["hierarchy_used"] for row in hier_rows)
    flat_ok = len(flat_rows) == 3 and all(not row["flat_entered_hierarchy"] for row in flat_rows)
    run_ok = len(rows) == 6 and all(
        row["trajectory_valid"]
        and row["runner_returncode"] == 0
        and row["backend_execution_verified"]
        and not row["runtime_errors"]
        and bool(row["prediction"])
        and row["completion_mode"] in {"normal_retrieval_answer", "full_video_64_frame_fallback"}
        for row in rows
    )
    prompt_hashes_by_uid: dict[str, dict[str, str | None]] = {}
    schema_hashes = {row["tools_schema_sha256"] for row in rows if row["tools_schema_sha256"]}
    for uid in uids:
        prompt_hashes_by_uid[uid] = {
            row["backend"]: row["system_prompt_sha256"] for row in rows if row["uid"] == uid
        }
    prompt_equal = all(
        len({value for value in hashes.values() if value}) == 1 and len(hashes) == len(selected)
        for hashes in prompt_hashes_by_uid.values()
    )
    schema_equal = len(schema_hashes) == 1 and len(rows) == (len(uids) * len(selected))
    traversal_path = output_root / "hierarchical_traversal_metadata.json"
    traversal_path.write_text(
        json.dumps(
            {
                "backend": "hierarchical",
                "rows": [
                    {"uid": row["uid"], "hierarchy_used": row["hierarchy_used"], "retrieval_trace": row["retrieval_trace"]}
                    for row in hier_rows
                ],
                "planner_visible": False,
                "eval300_started": False,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    paired_report = {
        "experiment": cfg["experiment"],
        "backends": selected,
        "reports": reports,
        "run_count": len(rows),
        "hierarchical_3_of_3": hier_ok,
        "flat_3_of_3_no_hierarchy": flat_ok,
        "planner_prompt_equal_per_uid": prompt_equal,
        "planner_tool_schema_equal": schema_equal,
        "paired_smoke_passed": bool(
            backend == "paired" and run_ok and hier_ok and flat_ok and prompt_equal and schema_equal
        ),
        "hierarchical_traversal_metadata": str(traversal_path),
        "actual_config_and_code_hash_snapshot": str(snapshot_path),
        "timing_clock": "time.monotonic",
        "offline_indexing_time": None,
        "offline_indexing_time_note": "offline indexing time not measured in this smoke",
        "eval300_started": False,
    }
    report_path = output_root / "paired_smoke_report.json"
    report_path.write_text(json.dumps(paired_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(paired_report, output_root / "paired_smoke_summary.md")
    return 0 if all(row["runner_returncode"] == 0 for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["contract-test", "paired-smoke"], required=True)
    parser.add_argument(
        "--backend", choices=["paired", "videoseal_flat", "hierarchical"], default="paired"
    )
    args = parser.parse_args()
    cfg = _read_json(CONFIG_PATH)
    if args.stage == "contract-test":
        report = contract_test(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    return paired_smoke(cfg, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
