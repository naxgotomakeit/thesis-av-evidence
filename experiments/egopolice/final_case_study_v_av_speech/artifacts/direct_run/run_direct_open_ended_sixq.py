#!/usr/bin/env python3
"""Isolated 226 Direct-v1.2 open-ended V/AV-Speech runner.

Default mode is a no-API preflight. ``--execute`` is required for provider use.
The shared repositories and frozen bundle are read-only inputs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request


ROOT = Path("/cs/student/project_msc/2025/rai/xinanx01/codex_226_case_study_v1")
OUT = ROOT / "direct_open_ended_v_av_sixq_v1"
CONTRACT = ROOT / "direct_open_ended_contract_v1"
PAIR = ROOT / "v_av_speech_v2_2"
CANONICAL = Path("/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/main_system/isolated_workspaces/hourvideo_direct_api_eval300_v1")
QUESTIONS = Path("/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/main_system/thesis-av-evidence-hourvideo/configs/experiments/planner_medium_retrieval_v1/questions_226.json")
FRAMES = Path("/cs/student/project_msc/2025/rai/xinanx01/226_case_study_bundle_v1/media/frames_1fps")
HIERARCHY = ROOT / "r3_visual_226_v1/shared/source_cases/540772226/shared_hierarchy.json"
V_MAP = PAIR / "V/attempt_01/cases/540772226/parsed_map.json"
AV_MAP = PAIR / "AV_SPEECH/attempt_01/cases/540772226/parsed_map.json"
PAIR_MANIFEST = PAIR / "V_AV_SPEECH_MATCHED_PAIR_MANIFEST.json"
PAIR_SUMMARY = PAIR / "V_AV_SPEECH_MATCHED_PAIR_SUMMARY.json"
CONFIG = CANONICAL / "config/direct_v1_anthropic_smoke.json"

QUESTION_SHA = "ce1e1de3a35fc5afd54662230db7e3ec1198249527bf95ee1374b5c95584791d"
V_MAP_SHA = "fc480efee797c379afbb31c5d1bdad0d14f44fe777fbce15b562ab55a22d9d65"
AV_MAP_SHA = "c8af9b585d661448c496c44128db05452c0334b412e6031b8d1022f2d23110f1"
EXPECTED_IDS = [
    "q_global_summary", "q_weapon_visible", "q_visible_injury",
    "q_medical_assistance", "q_handcuffing", "q_handcuff_before_medical",
]
VISIBLE_IDS = {"q_weapon_visible", "q_visible_injury"}
CONDITIONS = [("V", V_MAP), ("AV_SPEECH", AV_MAP)]

MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0.0
MAX_TOKENS = 512
TIMEOUT_SEC = 120
MAX_TRANSPORT_RETRIES = 1
MAX_STRUCTURAL_CORRECTIONS = 1
MAX_TURNS = 32
MAX_NEW_PER_TURN = 3
MAX_UNIQUE = 16
MAX_TOTAL_USD = 50.0
PRICING = {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.1}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(value)
    os.replace(temp, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def verify_sha_list(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        target = path.parent / rel
        actual = sha(target) if target.is_file() else None
        rows.append({"path": str(target), "expected": expected, "actual": actual, "pass": actual == expected})
    if not rows or not all(row["pass"] for row in rows):
        raise RuntimeError(f"SHA list failed: {path}")
    return rows


def no_forbidden_question_fields(value: object) -> bool:
    forbidden = {"correct_answer", "gold_answer", "answer_key", "winning_option", "correctness"}
    if isinstance(value, dict):
        return not (set(value) & forbidden) and all(no_forbidden_question_fields(v) for v in value.values())
    if isinstance(value, list):
        return all(no_forbidden_question_fields(v) for v in value)
    return True


def load_questions() -> list[dict[str, str]]:
    if sha(QUESTIONS) != QUESTION_SHA:
        raise RuntimeError("question SHA mismatch")
    raw = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if not no_forbidden_question_fields(raw):
        raise RuntimeError("forbidden gold/correctness field in questions")
    rows = raw.get("questions")
    if not isinstance(rows, list) or [row.get("question_id") for row in rows] != EXPECTED_IDS:
        raise RuntimeError("six-question identity/order mismatch")
    result = []
    for row in rows:
        text = row.get("question")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("empty question text")
        result.append({"question_id": row["question_id"], "question_text": text})
    return result


def verify_pair() -> dict[str, object]:
    if sha(V_MAP) != V_MAP_SHA or sha(AV_MAP) != AV_MAP_SHA:
        raise RuntimeError("formal v2.2 map SHA mismatch")
    manifest = json.loads(PAIR_MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for path, meta in manifest["files"].items():
        target = Path(path)
        actual = sha(target) if target.is_file() else None
        if actual != meta["sha256"] or not target.is_file() or target.stat().st_size != meta["size_bytes"]:
            failures.append(path)
    if failures:
        raise RuntimeError(f"matched-pair manifest failures: {len(failures)}")
    summary = json.loads(PAIR_SUMMARY.read_text(encoding="utf-8"))
    gates = {
        "state_complete": summary.get("state") == "V_AV_SPEECH_MAP_PAIR_COMPLETE",
        "shared_visual_identity": summary.get("shared_visual_identity_exact_match") is True,
        "caption_identity": summary.get("caption_count_exact_match") is True,
        "organizer_prompt_identity": summary.get("organizer_prompt_exact_match") is True,
        "provider_schema_identity": summary.get("provider_schema_exact_match") is True,
        "validator_identity": summary.get("local_strict_validator_exact_match") is True,
        "only_organizer_input_difference": summary.get("only_input_difference") == "timeline[*].overlapping_asr content",
        "non_speech_zero": summary.get("non_speech_usage") == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"matched-pair provenance gate failed: {gates}")
    return {"manifest_file_count": len(manifest["files"]), "gates": gates, "shared_visual_hashes": summary["shared_visual_hashes"]}


def project_semantic_map(source: Path, condition: str) -> tuple[dict[str, object], dict[str, object]]:
    """Remove raw ASR lists symmetrically; preserve Organizer semantics and all visual fields."""
    value = json.loads(source.read_text(encoding="utf-8"))
    groups = value.get("coarse_regions")
    if not isinstance(groups, list):
        raise RuntimeError("map has no coarse_regions")
    removed_records = 0
    for group in groups:
        if not isinstance(group, dict) or "exact_source_asr" not in group or not isinstance(group["exact_source_asr"], list):
            raise RuntimeError("map group exact_source_asr contract mismatch")
        removed_records += len(group["exact_source_asr"])
        group["exact_source_asr"] = []
    def raw_audio_present(node: object) -> bool:
        if isinstance(node, dict):
            for key, child in node.items():
                low = key.lower()
                if low in {"transcript", "overlapping_asr"}:
                    return True
                if low == "exact_source_asr" and child != []:
                    return True
                if raw_audio_present(child):
                    return True
        elif isinstance(node, list):
            return any(raw_audio_present(child) for child in node)
        return False
    if raw_audio_present(value):
        raise RuntimeError("raw speech content survived Direct semantic-map projection")
    report = {
        "condition": condition,
        "source_path": str(source),
        "source_sha256": sha(source),
        "rule": "set every coarse_regions[*].exact_source_asr to []; no other field/value changed",
        "raw_asr_records_removed": removed_records,
        "group_count": len(groups),
        "raw_audio_present_after_projection": False,
    }
    return value, report


def credential_available(config: dict[str, object]) -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    path = Path(str(config["credential_env_path"]))
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY=") and line.split("=", 1)[1].strip().strip('"').strip("'"):
            return True
    return False


def load_key(config: dict[str, object]) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        path = Path(str(config["credential_env_path"]))
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError("credential unavailable")
    return key


def preflight() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    contract_checks = verify_sha_list(CONTRACT / "sha256.txt")
    questions = load_questions()
    pair = verify_pair()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["model"] != MODEL or config["max_output_tokens"] != MAX_TOKENS or config["timeout_sec"] != TIMEOUT_SEC or config["max_retries"] != MAX_TRANSPORT_RETRIES:
        raise RuntimeError("canonical provider config mismatch")
    frames = sorted(FRAMES.glob("frame_*.jpg"))
    names = [path.name for path in frames]
    expected_names = [f"frame_{i:05d}.jpg" for i in range(1235)]
    if names != expected_names or not all(path.is_file() and path.stat().st_size > 0 for path in frames):
        raise RuntimeError("formal 1-FPS frame pool mismatch")
    hierarchy = json.loads(HIERARCHY.read_text(encoding="utf-8"))
    if hierarchy.get("duration_sec") != 1235.0 or len(hierarchy.get("fine_nodes", [])) != 83:
        raise RuntimeError("frozen hierarchy mismatch")

    projections = OUT / "map_projections"
    projections.mkdir(exist_ok=True)
    projection_reports = []
    projected_paths = {}
    for condition, source in CONDITIONS:
        projected, report = project_semantic_map(source, condition)
        target = projections / f"{condition}.semantic_only_map.json"
        atomic_json(target, projected)
        report["projected_path"] = str(target)
        report["projected_sha256"] = sha(target)
        projection_reports.append(report)
        projected_paths[condition] = str(target)
    atomic_json(OUT / "map_projection_validation.json", {
        "status": "PASS",
        "purpose": "Direct answerer receives Organizer semantic map but never raw ASR records",
        "same_projection_rule_for_both_conditions": True,
        "reports": projection_reports,
    })
    atomic_json(OUT / "questions_freeze.json", {
        "source": str(QUESTIONS), "source_sha256": QUESTION_SHA,
        "gold_or_correctness_read": False, "questions": questions,
    })
    route_order = [
        {"ordinal": i * 2 + j + 1, "route_id": f"{q['question_id']}__{condition}",
         "question_id": q["question_id"], "condition": condition,
         "question_text": q["question_text"], "map_path": projected_paths[condition],
         "source_map_path": str(dict(CONDITIONS)[condition]),
         "source_map_sha256": V_MAP_SHA if condition == "V" else AV_MAP_SHA}
        for i, q in enumerate(questions) for j, (condition, _) in enumerate(CONDITIONS)
    ]
    atomic_json(OUT / "route_manifest.json", {
        "schema_version": "direct_open_ended_226_sixq_routes_v1",
        "status": "FROZEN_NOT_STARTED",
        "order": "for each question in questions_226.json order: V then AV_SPEECH",
        "routes": route_order,
        "route_count": 12,
    })
    report = {
        "status": "PRE_EXECUTION_READY",
        "created_at_utc": now(),
        "hostname": os.uname().nodename,
        "output_namespace": str(OUT),
        "contract_sha_checks": contract_checks,
        "questions_sha256": QUESTION_SHA,
        "pair_validation": pair,
        "map_projection_validation": projection_reports,
        "frame_pool": {"path": str(FRAMES), "count": len(frames), "continuous_00000_01234": True},
        "model_config": {"model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "timeout_sec": TIMEOUT_SEC,
                         "transport_retries": MAX_TRANSPORT_RETRIES, "structural_corrections": MAX_STRUCTURAL_CORRECTIONS,
                         "max_turns": MAX_TURNS, "max_new_per_turn": MAX_NEW_PER_TURN, "max_unique": MAX_UNIQUE},
        "credential_available": credential_available(config),
        "raw_asr_direct_answerer_access": False,
        "gold_or_correctness_read": False,
        "api_calls": 0,
    }
    if not report["credential_available"]:
        raise RuntimeError("credential unavailable")
    atomic_json(OUT / "launch_preflight.json", report)
    freeze_files = [
        OUT / "run_direct_open_ended_sixq.py", OUT / "route_manifest.json", OUT / "questions_freeze.json",
        OUT / "map_projection_validation.json", OUT / "launch_preflight.json",
        projections / "V.semantic_only_map.json", projections / "AV_SPEECH.semantic_only_map.json",
    ]
    lines = [f"{sha(path)}  {path.relative_to(OUT)}" for path in freeze_files]
    atomic_bytes(OUT / "pre_execution_sha256.txt", ("\n".join(lines) + "\n").encode())
    verify_sha_list(OUT / "pre_execution_sha256.txt")
    return report


def short_reason_schema() -> dict[str, object]:
    return {"type": "string", "minLength": 1, "maxLength": 240}


def direct_tools(remaining: int) -> list[dict[str, object]]:
    tools = []
    allowed = min(MAX_NEW_PER_TURN, remaining)
    if allowed > 0:
        tools.append({
            "name": "inspect_frames",
            "description": "Request only the allowed number of new, valid video timestamps for the most informative next visual check.",
            "input_schema": {"type": "object", "additionalProperties": False,
                "properties": {"timestamps_sec": {"type": "array", "minItems": 1, "maxItems": allowed, "items": {"type": "number"}},
                               "reason": short_reason_schema()},
                "required": ["timestamps_sec", "reason"]},
        })
    tools.append({
        "name": "final_answer",
        "description": "Finish with a concise open-ended answer supported by the available evidence.",
        "input_schema": {"type": "object", "additionalProperties": False,
            "properties": {"answer": {"type": "string", "minLength": 1}, "reason": short_reason_schema()},
            "required": ["answer", "reason"]},
    })
    return tools


def budget_state(unique_count: int) -> str:
    remaining = max(0, MAX_UNIQUE - unique_count)
    if remaining == 0:
        return f"Visual inspection state: {MAX_UNIQUE} inspected; 0 remaining. No inspection is allowed; return final_answer."
    return f"Visual inspection state: {unique_count} inspected; {remaining} remaining; maximum new images this turn: {min(MAX_NEW_PER_TURN, remaining)}."


def resolve_frame(requested: float, duration_sec: float = 1235.0) -> dict[str, object]:
    if not math.isfinite(requested) or requested < 0 or requested > duration_sec:
        raise ValueError(f"requested timestamp outside video duration: {requested}")
    floor, ceil = int(math.floor(requested)), int(math.ceil(requested))
    candidates = sorted({floor, ceil}, key=lambda index: (abs(index - requested), index))
    limit = int(math.floor(duration_sec))
    ordered = list(candidates)
    for distance in range(1, limit + 1):
        for index in (candidates[0] - distance, candidates[-1] + distance):
            if 0 <= index <= limit and index not in ordered:
                ordered.append(index)
    for index in ordered:
        path = FRAMES / f"frame_{index:05d}.jpg"
        if path.is_file():
            digest = sha(path)
            return {"requested_timestamp_sec": requested, "resolved_timestamp_sec": float(index), "frame_index": index,
                    "frame_path": str(path), "expected_sha256": digest, "observed_sha256": digest}
    raise ValueError(f"no valid original 1-FPS frame near {requested}")


def call_provider(payload: dict[str, object], key: str, route_dir: Path, call_index: int) -> tuple[dict[str, object], dict[str, object]]:
    last_error = None
    records = []
    for attempt in range(MAX_TRANSPORT_RETRIES + 1):
        attempt_no = attempt + 1
        request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_path = route_dir / "requests" / f"call_{call_index:02d}_attempt_{attempt_no:02d}.json"
        atomic_bytes(request_path, request_bytes)
        started = time.perf_counter()
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=request_bytes, method="POST",
            headers={"content-type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01", "anthropic-beta": "prompt-caching-2024-07-31"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SEC) as response:
                raw = response.read()
            response_path = route_dir / "responses" / f"call_{call_index:02d}_attempt_{attempt_no:02d}.json"
            atomic_bytes(response_path, raw)
            value = json.loads(raw.decode("utf-8"))
            record = {"call_index": call_index, "attempt_index": attempt_no, "status": "response_received",
                      "request_path": str(request_path), "request_sha256": sha_bytes(request_bytes),
                      "response_path": str(response_path), "response_sha256": sha_bytes(raw),
                      "response_id": value.get("id"), "latency_sec": time.perf_counter() - started,
                      "usage": value.get("usage", {})}
            records.append(record)
            return value, {"attempts": records, "accepted": record}
        except urllib.error.HTTPError as error:
            body = error.read()
            error_path = route_dir / "responses" / f"call_{call_index:02d}_attempt_{attempt_no:02d}_http_{error.code}.json"
            atomic_bytes(error_path, body)
            last_error = error
            records.append({"call_index": call_index, "attempt_index": attempt_no, "status": f"http_{error.code}",
                            "request_path": str(request_path), "request_sha256": sha_bytes(request_bytes),
                            "response_path": str(error_path), "response_sha256": sha_bytes(body),
                            "latency_sec": time.perf_counter() - started})
        except Exception as error:
            last_error = error
            records.append({"call_index": call_index, "attempt_index": attempt_no,
                            "status": f"provider_error:{type(error).__name__}",
                            "request_path": str(request_path), "request_sha256": sha_bytes(request_bytes),
                            "latency_sec": time.perf_counter() - started})
    raise RuntimeError(json.dumps({"error": f"provider_failure_after_{MAX_TRANSPORT_RETRIES + 1}_attempts:{type(last_error).__name__}", "attempts": records}))


def parse_response(response: dict[str, object]) -> tuple[dict[str, object], dict[str, object] | None, str]:
    blocks = response.get("content")
    if not isinstance(blocks, list):
        return {"invalid": "content_not_array"}, None, "user_message"
    tools = [block for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use"]
    non_tools = [block for block in blocks if not (isinstance(block, dict) and block.get("type") == "tool_use")]
    if not tools:
        return {"invalid": "no_tool_use"}, None, "user_message"
    if len(tools) > 1:
        return {"invalid": "multiple_tool_use"}, None, "user_message"
    if non_tools:
        return {"invalid": "tool_plus_text"}, None, "user_message"
    tool = tools[0]
    identity = {"id": tool.get("id"), "name": tool.get("name"), "input": tool.get("input")}
    if not isinstance(identity["id"], str) or not identity["id"] or not isinstance(identity["input"], dict):
        return {"invalid": "invalid_tool_identity_or_input"}, None, "user_message"
    if identity["name"] == "inspect_frames":
        inp = identity["input"]
        return {"action": "inspect_frames", "timestamps_sec": inp.get("timestamps_sec"), "reason": inp.get("reason")}, identity, "tool_result"
    if identity["name"] == "final_answer":
        inp = identity["input"]
        return {"action": "final_answer", "answer": inp.get("answer"), "reason": inp.get("reason")}, identity, "tool_result"
    return {"invalid": f"unknown_tool:{identity['name']}"}, identity, "tool_result"


def action_error(action: dict[str, object], remaining: int) -> str | None:
    if "invalid" in action:
        return str(action["invalid"])
    reason = action.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        return "invalid_reason"
    if action.get("action") == "final_answer":
        answer = action.get("answer")
        return None if isinstance(answer, str) and answer.strip() else "invalid_answer"
    if action.get("action") != "inspect_frames":
        return "unsupported_action"
    if remaining <= 0:
        return "answer_only_budget_exhausted"
    values = action.get("timestamps_sec")
    # Canonical controller validates the fixed 3-per-turn ceiling, then budgets
    # resolved *new physical paths*. The provider schema is tighter when fewer
    # than three unique slots remain, but duplicate resolutions can still make
    # an otherwise overlong request fit the unique-frame budget.
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_NEW_PER_TURN:
        return "invalid_timestamp_count"
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        return "invalid_timestamps"
    return None


def cost_from_usage(usage: dict[str, object]) -> dict[str, float]:
    ordinary = int(usage.get("input_tokens") or 0) * PRICING["input"] / 1_000_000
    write = int(usage.get("cache_creation_input_tokens") or 0) * PRICING["cache_write"] / 1_000_000
    read = int(usage.get("cache_read_input_tokens") or 0) * PRICING["cache_read"] / 1_000_000
    output = int(usage.get("output_tokens") or 0) * PRICING["output"] / 1_000_000
    return {"ordinary_input_usd": ordinary, "cache_creation_usd": write, "cache_read_usd": read,
            "output_usd": output, "total_cache_aware_usd": ordinary + write + read + output}


def route_file_manifest(route_dir: Path) -> None:
    files = []
    for path in sorted(route_dir.rglob("*")):
        if path.is_file() and path.name != "route_sha256.txt":
            files.append({"path": str(path.relative_to(route_dir)), "sha256": sha(path), "size_bytes": path.stat().st_size})
    atomic_bytes(route_dir / "route_sha256.txt", ("\n".join(f"{row['sha256']}  {row['path']}" for row in files) + "\n").encode())


def run_route(route: dict[str, object], key: str, prompt: str, spend_before: float) -> tuple[dict[str, object], float]:
    route_dir = OUT / str(route["condition"]) / str(route["question_id"])
    if route_dir.exists():
        raise RuntimeError(f"route directory already exists; refusing overwrite/replay: {route_dir}")
    route_dir.mkdir(parents=True)
    map_path = Path(str(route["map_path"]))
    map_raw = map_path.read_text(encoding="utf-8")
    input_record = {**route, "projected_map_sha256": sha(map_path), "direct_contract_sha256_file": sha(CONTRACT / "sha256.txt"),
                    "model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "timeout_sec": TIMEOUT_SEC,
                    "started_at_utc": now(), "raw_asr_direct_answerer_access": False, "gold_or_correctness_read": False}
    atomic_json(route_dir / "route_input.json", input_record)

    system = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "VIDEO MAP (frozen native representation):\n" + map_raw, "cache_control": {"type": "ephemeral"}},
    ]
    seen: set[str] = set()
    inspected: list[dict[str, object]] = []
    history: list[dict[str, object]] = [{"role": "user", "content": [{"type": "text", "text":
        f"Question: {route['question_text']}\n{budget_state(0)}\nReturn the next Direct action."}]}]
    turns = []
    provider_attempts = []
    rounds = 0
    corrections = 0
    call_index = 0
    terminal_status = None
    final_answer = None
    final_reason = None
    total_cost = 0.0
    pending_correction: tuple[str, str | None] | None = None
    route_started = time.perf_counter()

    while terminal_status is None:
        if rounds >= MAX_TURNS:
            terminal_status = "runtime_failure:turn_limit_exhausted"
            break
        remaining = MAX_UNIQUE - len(seen)
        if spend_before + total_cost >= MAX_TOTAL_USD:
            terminal_status = "runtime_failure:hard_api_budget_exhausted"
            break
        if pending_correction is not None:
            message, tool_id = pending_correction
            if tool_id:
                history.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id,
                    "is_error": True, "content": [{"type": "text", "text": message}]}]})
            else:
                history.append({"role": "user", "content": [{"type": "text", "text": message}]})
            pending_correction = None

        payload = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                   "system": system, "messages": history, "tools": direct_tools(remaining), "tool_choice": {"type": "any"}}
        call_index += 1
        try:
            response, call_record = call_provider(payload, key, route_dir, call_index)
        except Exception as error:
            try:
                detail = json.loads(str(error))
                provider_attempts.extend(detail.get("attempts", []))
                terminal_status = "runtime_failure:provider"
                atomic_json(route_dir / "provider_failure.json", detail)
            except Exception:
                terminal_status = f"runtime_failure:provider:{type(error).__name__}"
            break
        provider_attempts.extend(call_record["attempts"])
        usage = call_record["accepted"].get("usage", {})
        cost = cost_from_usage(usage)
        total_cost += cost["total_cache_aware_usd"]
        action, tool_identity, correction_mode = parse_response(response)
        if tool_identity is not None:
            history.append({"role": "assistant", "content": [{"type": "tool_use", **tool_identity}]})
        else:
            history.append({"role": "assistant", "content": [{"type": "text", "text":
                f"[Structurally invalid Direct response: {action.get('invalid')}.]"}]})
        error = action_error(action, remaining)
        round_record = {"turn_index": rounds + 1, "provider_call_index": call_index,
                        "provider_response_id": response.get("id"), "action": action,
                        "usage": usage, "cost": cost, "controller_validation": None,
                        "unique_frame_count_before": len(seen)}
        if error is None and action["action"] == "final_answer":
            rounds += 1
            final_answer = str(action["answer"]).strip()
            final_reason = str(action["reason"]).strip()
            round_record.update({"controller_validation": "accepted", "unique_frame_count_after": len(seen)})
            turns.append(round_record)
            terminal_status = "final_answer"
            break

        annotated = []
        new_frames = []
        duplicate_count = 0
        if error is None:
            requested = [float(value) for value in action["timestamps_sec"]]
            try:
                resolved = [resolve_frame(value) for value in requested]
            except Exception as resolver_error:
                error = f"frame_resolution:{resolver_error}"
                resolved = []
            turn_paths: set[str] = set()
            for frame in resolved:
                path = str(frame["frame_path"])
                duplicate_turn = path in turn_paths
                duplicate_seen = path in seen
                row = {**frame, "duplicate_of_seen_frame": duplicate_seen, "duplicate_of_turn_frame": duplicate_turn}
                annotated.append(row)
                turn_paths.add(path)
                if duplicate_turn or duplicate_seen:
                    duplicate_count += 1
                else:
                    new_frames.append(row)
            if error is None and len(seen) + len(new_frames) > MAX_UNIQUE:
                error = "global_unique_image_budget_exceeded"

        rounds += 1
        if error is not None:
            round_record.update({"controller_validation": f"rejected:{error}", "resolved_frames": annotated,
                                 "images_transmitted": 0, "duplicate_requests": duplicate_count,
                                 "unique_frame_count_after": len(seen)})
            turns.append(round_record)
            if corrections < MAX_STRUCTURAL_CORRECTIONS:
                corrections += 1
                remaining = MAX_UNIQUE - len(seen)
                message = ("Visual inspection budget is exhausted. Return final_answer with a non-empty answer and reason."
                           if remaining == 0 else
                           f"Invalid action. Remaining visual budget: {remaining} unique images. Return exactly one valid action: inspect_frames with at most {min(MAX_NEW_PER_TURN, remaining)} timestamps, or final_answer with a non-empty answer and reason.")
                pending_correction = (message, str(tool_identity["id"]) if correction_mode == "tool_result" and tool_identity else None)
                continue
            terminal_status = f"invalid_action:{error}"
            break

        annotated.sort(key=lambda row: (row["resolved_timestamp_sec"], row["requested_timestamp_sec"], row["frame_path"]))
        new_frames.sort(key=lambda row: (row["resolved_timestamp_sec"], row["requested_timestamp_sec"], row["frame_path"]))
        for frame in new_frames:
            seen.add(str(frame["frame_path"]))
            inspected.append(frame)
        round_record.update({"controller_validation": "accepted", "resolved_frames": annotated,
                             "images_transmitted": len(new_frames), "duplicate_requests": duplicate_count,
                             "unique_frame_count_after": len(seen)})
        turns.append(round_record)
        lead = ("New chronologically ordered original video frames follow. " if new_frames else
                "The requested timestamps resolved only to already inspected physical frames, so no new image was transmitted. ")
        content: list[dict[str, object]] = [{"type": "text", "text": lead + budget_state(len(seen)) + " Reassess and return the next Direct action."}]
        for frame in new_frames:
            content.append({"type": "text", "text": f"Frame timestamp={frame['resolved_timestamp_sec']:.3f}s (requested {frame['requested_timestamp_sec']:.3f}s)."})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(Path(str(frame["frame_path"])).read_bytes()).decode("ascii")}})
        history.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": str(tool_identity["id"]), "content": content}]})

    visible_grounding = route["question_id"] not in VISIBLE_IDS or len(seen) > 0
    protocol_valid = terminal_status == "final_answer" and bool(final_answer) and bool(final_reason) and len(seen) <= MAX_UNIQUE
    completed = protocol_valid and visible_grounding
    if terminal_status == "final_answer" and not visible_grounding:
        terminal_status = "visible_grounding_failure:no_inspected_original_frames"
        completed = False
    result = {
        "route_id": route["route_id"], "question_id": route["question_id"], "condition": route["condition"],
        "question_text": route["question_text"], "terminal_status": terminal_status, "completed": completed,
        "final_answer": final_answer, "final_reason": final_reason, "rounds": rounds,
        "inspection_turn_count": sum(turn["action"].get("action") == "inspect_frames" and turn["controller_validation"] == "accepted" for turn in turns),
        "unique_frame_count": len(seen), "inspected_frames": inspected,
        "requested_timestamps_sec": [t for turn in turns if turn["action"].get("action") == "inspect_frames" for t in (turn["action"].get("timestamps_sec") or [])],
        "resolved_timestamps_sec": [frame["resolved_timestamp_sec"] for frame in inspected],
        "duplicate_requests": sum(int(turn.get("duplicate_requests", 0)) for turn in turns),
        "structural_corrections": corrections, "provider_attempt_count": len(provider_attempts),
        "provider_response_ids": [turn.get("provider_response_id") for turn in turns],
        "usage": {"calls": call_index, "cache_aware_usd": total_cost,
                  "input_tokens": sum(int(a.get("usage", {}).get("input_tokens") or 0) for a in provider_attempts),
                  "cache_creation_input_tokens": sum(int(a.get("usage", {}).get("cache_creation_input_tokens") or 0) for a in provider_attempts),
                  "cache_read_input_tokens": sum(int(a.get("usage", {}).get("cache_read_input_tokens") or 0) for a in provider_attempts),
                  "output_tokens": sum(int(a.get("usage", {}).get("output_tokens") or 0) for a in provider_attempts)},
        "route_wall_time_sec": time.perf_counter() - route_started,
        "visible_question": route["question_id"] in VISIBLE_IDS, "visible_grounding_pass": visible_grounding,
        "protocol_valid": protocol_valid, "max_unique_frames_pass": len(seen) <= MAX_UNIQUE,
        "gold_or_correctness_read": False, "raw_asr_direct_answerer_access": False,
        "finished_at_utc": now(),
    }
    atomic_json(route_dir / "turn_history.json", turns)
    atomic_json(route_dir / "provider_attempts.json", provider_attempts)
    atomic_json(route_dir / "conversation_history_final.json", history)
    atomic_json(route_dir / "route_validation.json", {
        "status": "PASS" if completed else "FAIL", "protocol_valid": protocol_valid,
        "visible_grounding_pass": visible_grounding, "unique_frame_count": len(seen),
        "unique_frame_ceiling": MAX_UNIQUE, "physical_paths_unique": len(seen) == len({f["frame_path"] for f in inspected}),
        "frame_hashes_pass": all(f["expected_sha256"] == f["observed_sha256"] for f in inspected),
    })
    atomic_json(route_dir / "route_result.json", result)
    route_file_manifest(route_dir)
    return result, total_cost


def execute() -> None:
    preflight()
    verify_sha_list(OUT / "pre_execution_sha256.txt")
    if any((OUT / condition).exists() for condition, _ in CONDITIONS):
        raise RuntimeError("condition output already exists; refusing overwrite")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    key = load_key(config)
    prompt = (CONTRACT / "open_ended_direct_prompt.txt").read_text(encoding="utf-8")
    manifest = json.loads((OUT / "route_manifest.json").read_text(encoding="utf-8"))
    atomic_json(OUT / "execution_started.json", {"status": "RUNNING", "started_at_utc": now(), "route_count": 12,
        "pre_execution_sha256": sha(OUT / "pre_execution_sha256.txt"), "api_authorized": True})
    results = []
    total_spend = 0.0
    for route in manifest["routes"]:
        print(f"START {route['ordinal']:02d}/12 {route['route_id']}", flush=True)
        result, spent = run_route(route, key, prompt, total_spend)
        total_spend += spent
        results.append(result)
        append_jsonl(OUT / "execution_journal.jsonl", {"timestamp": now(), "route_id": route["route_id"],
            "completed": result["completed"], "terminal_status": result["terminal_status"],
            "unique_frame_count": result["unique_frame_count"], "spend_usd": spent})
        print(f"DONE  {route['ordinal']:02d}/12 {route['route_id']} status={result['terminal_status']} frames={result['unique_frame_count']}", flush=True)
    atomic_json(OUT / "route_results.json", results)
    atomic_json(OUT / "execution_complete.json", {"status": "ROUTES_FROZEN", "finished_at_utc": now(),
        "route_count": len(results), "completed_count": sum(bool(r["completed"]) for r in results),
        "total_cache_aware_usd": total_spend, "gold_or_correctness_read": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        report = preflight()
        print(json.dumps({"status": report["status"], "credential_available": report["credential_available"],
                          "api_calls": 0, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
