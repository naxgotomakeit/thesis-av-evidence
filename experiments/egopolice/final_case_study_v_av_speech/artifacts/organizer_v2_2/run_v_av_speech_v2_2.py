from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
SESSION = OUT.parent
RUN = SESSION / "r3_visual_226_v1"
CONTRACT = SESSION / "organizer_v2_2_speech_contract"
BUNDLE = Path("/cs/student/project_msc/2025/rai/xinanx01/226_case_study_bundle_v1")
SPEECH = BUNDLE / "asr/enhanced_asr_nodes.json"
UID = "540772226"
RUNTIME = Path(
    "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/main_system/"
    "thesis-av-evidence-hourvideo/hourvideo_v6_1_runtime"
)
CANONICAL_RUNNER = RUNTIME / (
    "src/experiments/hourvideo_r3_keyframe_caption_3frame_action_preserving_v1/"
    "run_organizer_variant_c_controlled_diagnostic_v2.py"
)
SOURCE_CASE = RUN / "caption_runtime/index/cases" / UID

PROMPT_SHA = "532c58e7df9c2e86e12669f5c9dffc1fc4c607cfb516ce12668f9369be271a1d"
SCHEMA_SHA = "20b247005d3714d1807a806f1d4632585b3d9d3bd6f0f7e54399bc33abd82723"
SPEECH_SHA = "a88ee06b9ef5c9b855e01a05540e286c43bc5c9474b28ed313a0eb3bc67ccf2e"
RUNNER_SHA = "dd2da49836430efb67a3572cb31cef506e66ad3559579ee5d6b55178807584f0"
STRICT_VALIDATOR_SHA = "688f26b6d5050626e7f626120466541a520d40d5da1e06706e279ab8a34ee566"
MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0.0
MAX_TOKENS = 64_000
TIMEOUT_SEC = 900.0
PRICING = {"input": 1.0, "output": 5.0}
CONDITIONS = ("V", "AV_SPEECH")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text_new(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def load_runner():
    if sha_file(CANONICAL_RUNNER) != RUNNER_SHA:
        raise RuntimeError("canonical runner SHA changed")
    for path in (RUNTIME, RUNTIME / "src", CANONICAL_RUNNER.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("organizer_v2_2_symmetric_226", CANONICAL_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical Organizer runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    actual = sha_bytes(inspect.getsource(module.validate_raw_schema_strict).encode("utf-8"))
    if actual != STRICT_VALIDATOR_SHA:
        raise RuntimeError(f"canonical strict validator identity changed: {actual}")
    return module


def verify_sha_manifest(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        target = path.parent / name
        if not target.is_file() or sha_file(target) != expected:
            raise RuntimeError(f"SHA manifest mismatch: {target}")


def verify_inputs() -> tuple[dict, str, dict, dict, list[dict], list[dict]]:
    verify_sha_manifest(CONTRACT / "organizer_v2_2_sha256.txt")
    config = load_json(CONTRACT / "organizer_v2_2_config.json")
    prompt = (CONTRACT / "system_prompt_v2_2.txt").read_text(encoding="utf-8")
    schema = load_json(CONTRACT / "provider_schema_v2_2.json")
    if sha_bytes(prompt.encode("utf-8")) != PROMPT_SHA:
        raise RuntimeError("v2.2 prompt SHA mismatch")
    if sha_bytes(canonical_bytes(schema)) != SCHEMA_SHA:
        raise RuntimeError("v2.2 schema SHA mismatch")
    if sha_file(SPEECH) != SPEECH_SHA:
        raise RuntimeError("speech artifact SHA mismatch")
    bad_visual = []
    for relative, identity in config["frozen_visual_identity"].items():
        path = SESSION / relative
        if not path.is_file() or sha_file(path) != identity["sha256"]:
            bad_visual.append(relative)
    if bad_visual:
        raise RuntimeError(f"frozen visual SHA mismatch: {bad_visual}")
    hierarchy = load_json(SOURCE_CASE / "shared_hierarchy.json")
    captions = load_json(SOURCE_CASE / "r3_medium_captions.json")
    caption_validation = load_json(SOURCE_CASE / "r3_caption_validation.json")
    speech_document = load_json(SPEECH)
    speech_nodes = speech_document.get("nodes")
    if len(hierarchy.get("medium_nodes", [])) != 28 or len(captions) != 28 or not caption_validation.get("valid"):
        raise RuntimeError("frozen visual hierarchy/caption contract failed")
    if not isinstance(speech_nodes, list) or len(speech_nodes) != 107:
        raise RuntimeError("frozen speech contract failed")
    return config, prompt, schema, hierarchy, captions, speech_nodes


def align_speech(mediums: list[dict], speech_nodes: list[dict]) -> list[list[dict]]:
    aligned = []
    for medium in mediums:
        selected = [
            copy.deepcopy(node) for node in speech_nodes
            if float(node["start_sec"]) < float(medium["end_sec"])
            and float(node["end_sec"]) > float(medium["start_sec"])
        ]
        aligned.append(selected)
    return aligned


def blank_audio(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    for row in result["timeline"]:
        row["overlapping_asr"] = []
    return result


def build_payloads(module, hierarchy: dict, captions: list[dict], speech_nodes: list[dict]) -> tuple[dict, dict, dict]:
    v_payload = module.payload_for(hierarchy, captions)
    if len(v_payload.get("timeline", [])) != 28 or not all(row["overlapping_asr"] == [] for row in v_payload["timeline"]):
        raise RuntimeError("canonical V payload construction failed")
    av_payload = copy.deepcopy(v_payload)
    aligned = align_speech(hierarchy["medium_nodes"], speech_nodes)
    for row, nodes in zip(av_payload["timeline"], aligned):
        row["overlapping_asr"] = nodes
    if blank_audio(v_payload) != blank_audio(av_payload):
        raise RuntimeError("V/AV payloads differ outside overlapping_asr")
    for medium, actual, expected in zip(hierarchy["medium_nodes"], av_payload["timeline"], aligned):
        if canonical_bytes(actual["overlapping_asr"]) != canonical_bytes(expected):
            raise RuntimeError(f"ASR alignment copy mismatch: {medium['medium_id']}")
    differing_paths = [
        f"$.timeline[{index}].overlapping_asr"
        for index, (v_row, av_row) in enumerate(zip(v_payload["timeline"], av_payload["timeline"]))
        if v_row["overlapping_asr"] != av_row["overlapping_asr"]
    ]
    validation = {
        "valid": True,
        "medium_count": 28,
        "caption_count": 28,
        "speech_source_node_count": 107,
        "speech_counts_by_medium_index": [len(nodes) for nodes in aligned],
        "speech_presentations": sum(map(len, aligned)),
        "v_all_overlapping_asr_empty": True,
        "av_alignment_rule": "asr.start_sec < medium.end_sec AND asr.end_sec > medium.start_sec",
        "av_node_objects_preserved_without_field_changes": True,
        "visual_and_contract_fields_exact_match": True,
        "differing_json_paths": differing_paths,
        "all_differences_are_overlapping_asr": all(path.endswith(".overlapping_asr") for path in differing_paths),
        "clean_visual_payload_sha256_v": sha_bytes(canonical_bytes(blank_audio(v_payload))),
        "clean_visual_payload_sha256_av": sha_bytes(canonical_bytes(blank_audio(av_payload))),
        "question_filtering": False,
        "transcript_rewrite": False,
        "confidence_filtering": False,
        "empty_transcript_filtering": False,
        "non_speech_usage": 0,
    }
    return v_payload, av_payload, validation


def request_body(prompt: str, schema: dict, payload: dict) -> dict:
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": prompt,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }


def build_preflight() -> dict:
    forbidden = [
        OUT / "PRE_API_FREEZE.json", OUT / "pre_api_freeze_sha256.txt", OUT / "payloads",
        OUT / "V", OUT / "AV_SPEECH", OUT / "V_AV_SPEECH_MATCHED_PAIR_SUMMARY.json",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"refusing to overwrite preflight/live artifacts: {present}")
    config, prompt, schema, hierarchy, captions, speech_nodes = verify_inputs()
    module = load_runner()
    module.SYSTEM_PROMPT = prompt
    v_payload, av_payload, symmetry = build_payloads(module, hierarchy, captions, speech_nodes)
    generated_schema = module.organizer_core._api_schema(module.organizer_core._organizer_schema(28))
    if generated_schema != schema:
        raise RuntimeError("runtime provider schema differs from frozen v2.2 schema")
    v_body, av_body = request_body(prompt, schema, v_payload), request_body(prompt, schema, av_payload)
    if {key: value for key, value in v_body.items() if key != "messages"} != {key: value for key, value in av_body.items() if key != "messages"}:
        raise RuntimeError("outgoing request methods differ outside user content")
    paths = {
        "v_payload": OUT / "payloads/v_input_payload.json",
        "av_payload": OUT / "payloads/av_speech_input_payload.json",
        "symmetry": OUT / "payloads/payload_symmetry_validation.json",
        "v_request": OUT / "payloads/v_request_body_no_key.json",
        "av_request": OUT / "payloads/av_speech_request_body_no_key.json",
    }
    write_json_new(paths["v_payload"], v_payload)
    write_json_new(paths["av_payload"], av_payload)
    write_json_new(paths["symmetry"], symmetry)
    write_json_new(paths["v_request"], v_body)
    write_json_new(paths["av_request"], av_body)
    preflight = {
        "state": "V_AV_SPEECH_V2_2_PRE_API_FROZEN",
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "api_calls": 0,
        "frozen_visual_sha_verification": "PASS",
        "frozen_caption_count": 28,
        "frozen_speech_sha256": SPEECH_SHA,
        "frozen_speech_node_count": 107,
        "prompt_sha256": PROMPT_SHA,
        "schema_sha256": SCHEMA_SHA,
        "validator_source_sha256": STRICT_VALIDATOR_SHA,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout_sec": TIMEOUT_SEC,
        "payload_symmetry": symmetry,
        "request_hashes": {
            "V": sha_bytes(canonical_bytes(v_body)),
            "AV_SPEECH": sha_bytes(canonical_bytes(av_body)),
        },
        "non_speech_usage": 0,
    }
    write_json_new(OUT / "PRE_API_FREEZE.json", preflight)
    included = [
        OUT / "run_v_av_speech_v2_2.py",
        *paths.values(),
        OUT / "PRE_API_FREEZE.json",
        CONTRACT / "organizer_v2_2_sha256.txt",
        CONTRACT / "system_prompt_v2_2.txt",
        CONTRACT / "provider_schema_v2_2.json",
        CONTRACT / "organizer_v2_2_config.json",
        CONTRACT / "speech_alignment_contract.json",
        SPEECH,
    ]
    included.extend(SESSION / relative for relative in config["frozen_visual_identity"])
    write_text_new(
        OUT / "pre_api_freeze_sha256.txt",
        "\n".join(f"{sha_file(path)}  {path}" for path in included) + "\n",
    )
    return preflight


def verify_pre_api_freeze() -> dict:
    manifest = OUT / "pre_api_freeze_sha256.txt"
    if not manifest.is_file():
        raise RuntimeError("pre-API freeze is absent")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"pre-API SHA mismatch: {path}")
    preflight = load_json(OUT / "PRE_API_FREEZE.json")
    if preflight.get("state") != "V_AV_SPEECH_V2_2_PRE_API_FROZEN" or preflight.get("api_calls") != 0:
        raise RuntimeError("pre-API freeze state invalid")
    return preflight


def persist_provider_response(module, case_out: Path, record: dict) -> None:
    module.write_json_atomic(case_out / "provider_record.json", record)
    module.write_json_atomic(case_out / "raw_response.json", {
        "response_id": record["response_id"],
        "stop_reason": record["stop_reason"],
        "raw_text": record["raw_text"],
        "provider_response": record["provider_response"],
    })
    module.write_json_atomic(case_out / "usage.json", {
        **record["usage"],
        "provider": "anthropic",
        "model": record["model_returned"],
        "response_id": record["response_id"],
        "stop_reason": record["stop_reason"],
        "latency_sec": record["latency_sec"],
        "max_tokens_requested": MAX_TOKENS,
        "pricing_usd_per_million": PRICING,
        "estimated_cost": module.organizer_core._cost_record(record["usage"], PRICING),
    })


def validate_map_asr(map_doc: dict, speech_nodes: list[dict]) -> None:
    for region in map_doc["coarse_regions"]:
        expected = [
            node for node in speech_nodes
            if float(node["start_sec"]) < float(region["end_sec"])
            and float(node["end_sec"]) > float(region["start_sec"])
        ]
        if canonical_bytes(region["exact_source_asr"]) != canonical_bytes(expected):
            raise RuntimeError(f"map exact_source_asr mismatch: {region['coarse_id']}")


def attempt_manifest(condition: str, attempt: int, result: dict, response_id: str | None) -> dict:
    attempt_root = OUT / condition / f"attempt_{attempt:02d}"
    files = {}
    for path in sorted(attempt_root.rglob("*")):
        if path.is_file() and path.name not in {"attempt_summary.json", "attempt_sha256.json"}:
            files[str(path.relative_to(OUT))] = {"sha256": sha_file(path), "size_bytes": path.stat().st_size}
    summary = {
        "condition": condition,
        "attempt": attempt,
        "maximum_attempts": 2,
        "created_at_utc": utc_now(),
        "organizer_calls_this_attempt": 1,
        "response_id": response_id,
        "status": result.get("status"),
        "strict_validation_pass": result.get("status") == "valid",
        "repair_applied": False,
        "manual_modification": False,
        "non_speech_usage": 0,
        "files": files,
    }
    write_json_new(attempt_root / "attempt_summary.json", summary)
    all_files = {}
    for path in sorted(attempt_root.rglob("*")):
        if path.is_file() and path.name != "attempt_sha256.json":
            all_files[str(path.relative_to(OUT))] = {"sha256": sha_file(path), "size_bytes": path.stat().st_size}
    write_json_new(attempt_root / "attempt_sha256.json", {"files": all_files})
    return summary


def run_one_attempt(module, client, condition: str, attempt: int, payload: dict, hierarchy: dict, captions: list[dict], speech_nodes: list[dict], prompt: str, schema: dict, expected_request_sha: str) -> tuple[dict, str | None]:
    case_out = OUT / condition / f"attempt_{attempt:02d}/cases" / UID
    if case_out.exists():
        raise RuntimeError(f"refusing to overwrite existing attempt: {case_out}")
    case_out.mkdir(parents=True, exist_ok=False)
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    request_hashes = module.persist_request(case_out, UID, payload, content, schema)
    stored_body = load_json(case_out / "request_body_no_key.json")
    if sha_bytes(canonical_bytes(stored_body)) != expected_request_sha:
        raise RuntimeError(f"{condition} attempt request differs from pre-API freeze")
    print(f"{condition}_V2_2_ATTEMPT_START {attempt}/2", flush=True)
    started = time.monotonic()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=prompt,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
    except Exception as exc:
        failure = {
            "status": "api_call_failure",
            "condition": condition,
            "attempt": attempt,
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "request_hashes": request_hashes,
            "repair_applied": False,
            "non_speech_usage": 0,
        }
        module.write_json_atomic(case_out / "failure.json", failure)
        return failure, None
    latency = time.monotonic() - started
    record = module.provider_record(response, latency, request_hashes)
    persist_provider_response(module, case_out, record)
    try:
        if record["stop_reason"] == "max_tokens":
            raise RuntimeError("provider response reached max_tokens")
        parsed = json.loads(record["raw_text"])
        ends = module.validate_raw_schema_strict(parsed, len(captions))
        active_speech = speech_nodes if condition == "AV_SPEECH" else []
        map_doc = module.organizer_core._map_from_output(hierarchy, captions, active_speech, parsed)
        coverage = module.validate_map(hierarchy, parsed, map_doc)
        validate_map_asr(map_doc, active_speech)
        expected_counts = [len(row["overlapping_asr"]) for row in payload["timeline"]]
        validation = {
            **coverage,
            "schema_valid": True,
            "navigation_summary_non_empty": True,
            "boundaries_strictly_increasing": True,
            "duplicate_boundaries": False,
            "last_boundary_matches": True,
            "zero_coverage_groups": 0,
            "uncertainty_only_terminal_groups": 0,
            "prompt_sha256_matches": record["prompt_sha256"] == PROMPT_SHA,
            "provider_schema_sha256_matches": request_hashes["schema_sha256"] == SCHEMA_SHA,
            "condition": condition,
            "speech_source_sha256": SPEECH_SHA,
            "speech_counts_by_medium_index": expected_counts,
            "speech_presentations": sum(expected_counts),
            "overlapping_asr_all_empty": all(count == 0 for count in expected_counts),
            "aligned_speech_verified": condition == "AV_SPEECH",
            "exact_source_asr_verified": True,
            "question_content_read": False,
            "fine_caption_generated": False,
            "repair_applied": False,
            "manual_modification": False,
            "non_speech_usage": 0,
        }
        module.write_json_atomic(case_out / "parsed_output.json", parsed)
        module.write_json_atomic(case_out / "parsed_map.json", map_doc)
        module.write_json_atomic(case_out / "validation.json", validation)
        usage = load_json(case_out / "usage.json")
        statistics = module.group_statistics(ends, len(captions))
        result = {
            "status": "valid",
            "condition": condition,
            "video_uid": UID,
            "completed_at_utc": utc_now(),
            "hostname": socket.gethostname(),
            "prompt_name": "organizer_v2_2_symmetric_speech_aware_v1",
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "statistics": statistics,
            "validation": validation,
            "usage": usage,
            "request_hashes": request_hashes,
            "output_hashes": {
                "raw_text_sha256": sha_bytes(record["raw_text"].encode("utf-8")),
                "parsed_output_sha256": sha_bytes(canonical_bytes(parsed)),
                "parsed_map_sha256": sha_bytes(canonical_bytes(map_doc)),
            },
        }
        module.write_json_atomic(case_out / "result.json", result)
        module.write_json_atomic(case_out / "file_hashes.json", module.artifact_hashes(case_out))
        return result, record["response_id"]
    except Exception as exc:
        failure = {
            "status": "parser_schema_or_map_validation_failure",
            "condition": condition,
            "attempt": attempt,
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "provider_record_persisted_before_failure": True,
            "raw_response_persisted_before_failure": True,
            "usage_persisted_before_failure": True,
            "request_hashes": request_hashes,
            "repair_applied": False,
            "manual_modification": False,
            "non_speech_usage": 0,
        }
        module.write_json_atomic(case_out / "failure.json", failure)
        module.write_json_atomic(case_out / "file_hashes.json", module.artifact_hashes(case_out))
        return failure, record["response_id"]


def freeze_condition(condition: str, accepted_attempt: int, attempts: list[dict], config: dict) -> dict:
    case = OUT / condition / f"attempt_{accepted_attempt:02d}/cases" / UID
    result = load_json(case / "result.json")
    validation = load_json(case / "validation.json")
    map_path = case / "parsed_map.json"
    if result.get("status") != "valid" or not validation.get("valid") or not map_path.is_file():
        raise RuntimeError(f"cannot freeze invalid condition: {condition}")
    condition_root = OUT / condition
    manifest_path = condition_root / f"{condition}_MAP_V2_2_MANIFEST.json"
    summary_path = condition_root / f"{condition}_MAP_V2_2_SUMMARY.json"
    sha_path = condition_root / f"{condition.lower()}_map_v2_2_sha256.txt"
    included = [
        OUT / "run_v_av_speech_v2_2.py",
        OUT / "PRE_API_FREEZE.json",
        OUT / "pre_api_freeze_sha256.txt",
        OUT / "payloads/payload_symmetry_validation.json",
        OUT / ("payloads/v_input_payload.json" if condition == "V" else "payloads/av_speech_input_payload.json"),
        CONTRACT / "organizer_v2_2_sha256.txt",
        CONTRACT / "system_prompt_v2_2.txt",
        CONTRACT / "provider_schema_v2_2.json",
        CONTRACT / "organizer_v2_2_config.json",
        SPEECH,
    ]
    included.extend(SESSION / relative for relative in config["frozen_visual_identity"])
    for attempt in range(1, accepted_attempt + 1):
        included.extend(path for path in sorted((condition_root / f"attempt_{attempt:02d}").rglob("*")) if path.is_file())
    included = list(dict.fromkeys(included))
    manifest = {
        "state": f"{condition}_MAP_V2_2_COMPLETE",
        "condition": condition,
        "accepted_attempt": accepted_attempt,
        "files": {str(path): {"sha256": sha_file(path), "size_bytes": path.stat().st_size} for path in included},
    }
    summary = {
        "state": f"{condition}_MAP_V2_2_COMPLETE",
        "condition": condition,
        "created_at_utc": utc_now(),
        "video_uid": UID,
        "attempt_count": accepted_attempt,
        "maximum_attempts": 2,
        "provider_response_ids": [row["response_id"] for row in attempts],
        "accepted_attempt": accepted_attempt,
        "strict_validation": "PASS",
        "group_count": result["statistics"]["coarse_count"],
        "boundaries": result["statistics"]["boundaries"],
        "final_map": str(map_path),
        "final_map_sha256": sha_file(map_path),
        "prompt_sha256": PROMPT_SHA,
        "schema_sha256": SCHEMA_SHA,
        "validator_sha256": STRICT_VALIDATOR_SHA,
        "repair_applied": False,
        "manual_modification": False,
        "non_speech_usage": 0,
    }
    write_json_new(manifest_path, manifest)
    write_json_new(summary_path, summary)
    checks = [*included, manifest_path, summary_path]
    write_text_new(sha_path, "\n".join(f"{sha_file(path)}  {path}" for path in checks) + "\n")
    return summary


def run_condition(module, client, condition: str, payload: dict, hierarchy: dict, captions: list[dict], speech_nodes: list[dict], prompt: str, schema: dict, preflight: dict, config: dict) -> dict | None:
    condition_root = OUT / condition
    if condition_root.exists():
        raise RuntimeError(f"refusing to overwrite condition output: {condition_root}")
    attempts = []
    for attempt in (1, 2):
        result, response_id = run_one_attempt(
            module, client, condition, attempt, payload, hierarchy, captions, speech_nodes,
            prompt, schema, preflight["request_hashes"][condition],
        )
        summary = attempt_manifest(condition, attempt, result, response_id)
        attempts.append(summary)
        if result.get("status") == "valid":
            return freeze_condition(condition, attempt, attempts, config)
        print(f"{condition}_V2_2_ATTEMPT_INVALID {attempt}/2 status={result.get('status')}", flush=True)
    write_json_new(condition_root / f"{condition}_MAP_V2_2_FAILED.json", {
        "state": f"{condition}_MAP_V2_2_FAILED",
        "attempt_count": 2,
        "maximum_attempts": 2,
        "provider_response_ids": [row["response_id"] for row in attempts],
        "repair_applied": False,
        "manual_modification": False,
        "non_speech_usage": 0,
    })
    return None


def verify_condition_sha(summary: dict) -> None:
    condition = summary["condition"]
    path = OUT / condition / f"{condition.lower()}_map_v2_2_sha256.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        target = Path(name)
        if not target.is_file() or sha_file(target) != expected:
            raise RuntimeError(f"condition SHA mismatch: {target}")


def freeze_pair(v_summary: dict, av_summary: dict, config: dict) -> dict:
    v_payload = load_json(OUT / "payloads/v_input_payload.json")
    av_payload = load_json(OUT / "payloads/av_speech_input_payload.json")
    symmetry = load_json(OUT / "payloads/payload_symmetry_validation.json")
    if blank_audio(v_payload) != blank_audio(av_payload) or not symmetry.get("all_differences_are_overlapping_asr"):
        raise RuntimeError("final pair payload symmetry failed")
    verify_condition_sha(v_summary)
    verify_condition_sha(av_summary)
    visual_hashes = {relative: identity["sha256"] for relative, identity in config["frozen_visual_identity"].items()}
    pair = {
        "state": "V_AV_SPEECH_MAP_PAIR_COMPLETE",
        "created_at_utc": utc_now(),
        "video_uid": UID,
        "conditions": {"V": v_summary, "AV_SPEECH": av_summary},
        "speech_artifact_sha256": SPEECH_SHA,
        "speech_counts_by_medium_index": symmetry["speech_counts_by_medium_index"],
        "alignment_rule": symmetry["av_alignment_rule"],
        "shared_visual_identity_exact_match": True,
        "shared_visual_hashes": visual_hashes,
        "caption_count_exact_match": True,
        "caption_sha256": visual_hashes["r3_visual_226_v1/caption_runtime/index/cases/540772226/r3_medium_captions.json"],
        "organizer_prompt_exact_match": True,
        "organizer_prompt_sha256": PROMPT_SHA,
        "provider_schema_exact_match": True,
        "provider_schema_sha256": SCHEMA_SHA,
        "local_strict_validator_exact_match": True,
        "local_strict_validator_sha256": STRICT_VALIDATOR_SHA,
        "model_config_attempt_policy_exact_match": True,
        "only_input_difference": "timeline[*].overlapping_asr content",
        "payload_visual_and_contract_fields_exact_match": True,
        "repair_applied": False,
        "manual_modification": False,
        "non_speech_usage": 0,
        "retrieval_qa_or_visual_selection_executed": False,
    }
    summary_path = OUT / "V_AV_SPEECH_MATCHED_PAIR_SUMMARY.json"
    manifest_path = OUT / "V_AV_SPEECH_MATCHED_PAIR_MANIFEST.json"
    sha_path = OUT / "v_av_speech_matched_pair_sha256.txt"
    write_json_new(summary_path, pair)
    included = [
        OUT / "run_v_av_speech_v2_2.py",
        OUT / "PRE_API_FREEZE.json",
        OUT / "pre_api_freeze_sha256.txt",
        *(path for path in sorted((OUT / "payloads").rglob("*")) if path.is_file()),
        *(path for condition in CONDITIONS for path in sorted((OUT / condition).rglob("*")) if path.is_file()),
        CONTRACT / "organizer_v2_2_sha256.txt",
        CONTRACT / "ORGANIZER_V2_2_SPEECH_CONTRACT.md",
        CONTRACT / "system_prompt_v2_2.txt",
        CONTRACT / "provider_schema_v2_2.json",
        CONTRACT / "v2_1_to_v2_2_diff.txt",
        CONTRACT / "speech_alignment_contract.json",
        CONTRACT / "organizer_v2_2_config.json",
        SPEECH,
    ]
    included.extend(SESSION / relative for relative in config["frozen_visual_identity"])
    included = list(dict.fromkeys(included))
    write_json_new(manifest_path, {
        "state": "V_AV_SPEECH_MAP_PAIR_COMPLETE",
        "files": {str(path): {"sha256": sha_file(path), "size_bytes": path.stat().st_size} for path in included},
    })
    checks = [*included, summary_path, manifest_path]
    write_text_new(sha_path, "\n".join(f"{sha_file(path)}  {path}" for path in checks) + "\n")
    return pair


def run_live() -> int:
    forbidden = [OUT / "V", OUT / "AV_SPEECH", OUT / "V_AV_SPEECH_MATCHED_PAIR_SUMMARY.json"]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"refusing to overwrite live outputs: {present}")
    preflight = verify_pre_api_freeze()
    config, prompt, schema, hierarchy, captions, speech_nodes = verify_inputs()
    module = load_runner()
    module.SYSTEM_PROMPT = prompt
    module.PROMPT_NAME = "organizer_v2_2_symmetric_speech_aware_v1"
    v_payload = load_json(OUT / "payloads/v_input_payload.json")
    av_payload = load_json(OUT / "payloads/av_speech_input_payload.json")
    fresh_v, fresh_av, fresh_symmetry = build_payloads(module, hierarchy, captions, speech_nodes)
    if canonical_bytes(v_payload) != canonical_bytes(fresh_v) or canonical_bytes(av_payload) != canonical_bytes(fresh_av):
        raise RuntimeError("frozen payloads differ from deterministic reconstruction")
    if fresh_symmetry != load_json(OUT / "payloads/payload_symmetry_validation.json"):
        raise RuntimeError("payload symmetry validation changed")
    module.organizer_core._load_dedicated_key(CANONICAL_RUNNER.parent / ".env.haiku45_test")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("credential unavailable; no API call attempted")
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=TIMEOUT_SEC)
    v_summary = run_condition(module, client, "V", v_payload, hierarchy, captions, speech_nodes, prompt, schema, preflight, config)
    if v_summary is None:
        print(json.dumps({"state": "V_MAP_V2_2_FAILED_STOP", "non_speech_usage": 0}))
        return 2
    av_summary = run_condition(module, client, "AV_SPEECH", av_payload, hierarchy, captions, speech_nodes, prompt, schema, preflight, config)
    if av_summary is None:
        print(json.dumps({"state": "AV_SPEECH_MAP_V2_2_FAILED_STOP", "non_speech_usage": 0}))
        return 2
    pair = freeze_pair(v_summary, av_summary, config)
    print(json.dumps({
        "state": pair["state"],
        "V": {"attempts": v_summary["attempt_count"], "response_ids": v_summary["provider_response_ids"], "groups": v_summary["group_count"], "boundaries": v_summary["boundaries"]},
        "AV_SPEECH": {"attempts": av_summary["attempt_count"], "response_ids": av_summary["provider_response_ids"], "groups": av_summary["group_count"], "boundaries": av_summary["boundaries"]},
        "shared_visual_identity_exact_match": True,
        "only_input_difference": pair["only_input_difference"],
        "non_speech_usage": 0,
    }, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.live:
        raise RuntimeError("select exactly one of --preflight-only or --live")
    if args.preflight_only:
        print(json.dumps(build_preflight(), ensure_ascii=False, indent=2))
        return 0
    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())
