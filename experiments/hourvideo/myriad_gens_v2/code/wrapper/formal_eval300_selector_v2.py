from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import formal_eval300_selector as v1


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/myriadfs/home/ucemxna/Scratch/workspace/envs/videoseal/bin/python3.12")
PROFILE_PATH = ROOT / "config/gens_hybrid_symmetric_mcq_cap16_v2_long_output.json"
V1_PROFILE_PATH = ROOT / "config/gens_hybrid_symmetric_mcq_cap16_v1.json"
V1_ROOT = ROOT / "outputs/formal_eval300_gens_hybrid_symmetric_mcq_cap16_v1_20260822T165403Z"
EXPECTED_PROFILE_SHA = "3479770e71ff5f68f0669dc6b51ce3db1922fe8ae39b2f7a0ff493634aa8d0f6"
EXPECTED_V1_RAW_MANIFEST_SHA = "25d32a8e95d0d9d0c56e78cbe619263efd1d368bc6a0824009cc411f1a224834"
METHOD = "gens_hybrid_symmetric_mcq_cap16_v2_long_output"
DISPLAY_NAME = "GenS-Hybrid-SymmetricMCQ-cap16-v2-long-output"
PRIVATE_PREFIX = v1.PRIVATE_PREFIX
STAGE_A_FILES = (
    "question_metadata.json",
    "clip_per_option_queries.json",
    "clip_per_option_query_telemetry.json",
    "clip_top256.json",
    "stage_a_complete.json",
    "stage_a_timings.json",
)
SMOKE_UIDS = (
    "6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31",
    "4572b198-2c1c-4920-bcf0-95fcebe12261_17_39",
    "6fd90f8d-7a4d-425d-a812-3268db0b0342_8_9",
    "6fd90f8d-7a4d-425d-a812-3268db0b0342_17_25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "controller", "status", "report"):
        item = sub.add_parser(name)
        item.add_argument("--formal-root", required=True, type=Path)
    worker = sub.add_parser("worker")
    worker.add_argument("--formal-root", required=True, type=Path)
    worker.add_argument("--worker", required=True, type=int, choices=(0, 1))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return v1.sha256_file(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def profile_diff(old: Any, new: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(old) | set(new)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old:
                changes.append({"path": path, "kind": "added", "new": new[key]})
            elif key not in new:
                changes.append({"path": path, "kind": "removed", "old": old[key]})
            else:
                changes.extend(profile_diff(old[key], new[key], path))
        return changes
    if old != new:
        return [{"path": prefix, "kind": "changed", "old": old, "new": new}]
    return []


def validate_profile_diff() -> list[dict[str, Any]]:
    old = json.loads(V1_PROFILE_PATH.read_text(encoding="utf-8"))
    new = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    changes = profile_diff(old, new)
    paths = {item["path"] for item in changes}
    if "gens.generation.max_new_tokens" not in paths:
        raise RuntimeError("V2 profile does not change the output budget")
    if any(
        path != "gens.generation.max_new_tokens"
        and not path.startswith("amendment")
        and not path.startswith("observability")
        for path in paths
    ):
        raise RuntimeError(f"unauthorized V1-to-V2 profile change: {changes}")
    if old["gens"]["generation"]["max_new_tokens"] != 512:
        raise RuntimeError("V1 output budget drift")
    if new["gens"]["generation"]["max_new_tokens"] != 4096:
        raise RuntimeError("V2 output budget must be 4096")
    return changes


def v2_profile_and_records(full_manifests: bool = False):
    from gens_baseline.config import load_profile, resolve_profile_path, verify_frozen_artifacts
    from gens_baseline.pipeline import build_access_policy
    from gens_baseline.query import load_selector
    from gens_baseline.symmetric_mcq import load_symmetric_template

    if sha256_file(PROFILE_PATH) != EXPECTED_PROFILE_SHA:
        raise RuntimeError("V2 profile SHA drift")
    validate_profile_diff()
    profile = load_profile(PROFILE_PATH)
    frozen = verify_frozen_artifacts(profile, full_manifests=full_manifests)
    policy, allowlist_path, output_root = build_access_policy(profile)
    records = load_selector(profile, policy)
    load_symmetric_template(profile)
    full_query_path = resolve_profile_path(profile, profile["gens_full_query"]["template_path"])
    hashes = {
        "profile_sha256": profile["_profile_sha256"],
        "v1_profile_sha256": sha256_file(V1_PROFILE_PATH),
        "clip_symmetric_query_template_sha256": profile["symmetric_mcq"]["clip_query_template_sha256"],
        "gens_full_mcq_query_template_sha256": sha256_file(full_query_path),
        "clip_image_embedding_cache_tree_sha256": profile["cached_image_embeddings"]["global_tree_sha256"],
        "canonical_frame_cache_tree_sha256": profile["candidate_cache"]["total_tree_sha256"],
    }
    expected = {
        "profile_sha256": EXPECTED_PROFILE_SHA,
        "v1_profile_sha256": v1.EXPECTED_PROFILE_SHA,
        "clip_symmetric_query_template_sha256": v1.EXPECTED_CLIP_QUERY_SHA,
        "gens_full_mcq_query_template_sha256": v1.EXPECTED_GENS_QUERY_SHA,
        "clip_image_embedding_cache_tree_sha256": v1.EXPECTED_EMBEDDING_TREE_SHA,
        "canonical_frame_cache_tree_sha256": v1.EXPECTED_FRAME_TREE_SHA,
    }
    if hashes != expected:
        raise RuntimeError(f"frozen V2 identity mismatch: {hashes}")
    return profile, policy, allowlist_path, output_root, records, frozen, hashes


def raw_manifest_sha(root: Path) -> tuple[int, str]:
    paths = sorted((root / "questions").glob("*/gens_raw_response*.txt"))
    digest = hashlib.sha256()
    for path in paths:
        file_sha = sha256_file(path)
        digest.update(str(path.relative_to(root)).encode() + b"\0" + file_sha.encode() + b"\n")
    return len(paths), digest.hexdigest()


def input_identity(
    query: str, chronological_candidates: list[dict[str, Any]], profile: dict[str, Any]
) -> str:
    from gens_baseline.gens_stage import load_instruction_template

    value = {
        "query": query,
        "chronological_candidates_without_clip_signals": chronological_candidates,
        "instruction_template_sha256": profile["gens"]["instruction_template_sha256"],
        "instruction": load_instruction_template(profile),
        "resolution": profile["gens"]["resolution"],
        "frame_label_template": profile["gens"]["frame_label_template"],
        "processor": profile["gens"]["processor"],
        "model_revision": profile["gens"]["revision"],
    }
    return sha256_bytes(canonical_bytes(value))


def build_context_audit(profile: dict[str, Any], records: list[dict[str, Any]], formal_root: Path) -> dict[str, Any]:
    from transformers import AutoProcessor
    from gens_baseline.config import resolve_profile_path
    from gens_baseline.gens_stage import build_gens_messages, load_instruction_template
    from gens_baseline.query import build_retrieval_query

    processor = AutoProcessor.from_pretrained(
        resolve_profile_path(profile, profile["gens"]["local_path"]),
        local_files_only=True, trust_remote_code=False,
    )
    instruction = load_instruction_template(profile)
    template = resolve_profile_path(profile, profile["gens_full_query"]["template_path"]).read_text(encoding="utf-8")
    rows = []
    for record in records:
        question_dir = formal_root / "questions" / record["qa_uid"]
        top = json.loads((question_dir / "clip_top256.json").read_text(encoding="utf-8"))
        v1.validate_top256(top, record)
        candidates = [
            {key: value for key, value in item.items() if not str(key).startswith("clip_")}
            for item in top
        ]
        query = build_retrieval_query(record, template)
        messages = build_gens_messages(candidates, query, instruction, 112)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(processor.tokenizer.encode(prompt, add_special_tokens=False))
        image_pad_count = prompt.count("<|image_pad|>")
        if image_pad_count != 256:
            raise RuntimeError("context audit expected exactly 256 image placeholders")
        context_tokens = prompt_tokens + 15 * image_pad_count
        max_new = int(profile["gens"]["generation"]["max_new_tokens"])
        limit = int(profile["gens"]["context_limit_tokens"])
        margin = limit - context_tokens - max_new
        chronological = sorted(candidates, key=lambda item: (item["timestamp_sec"], item["frame_index"]))
        row = {
            "qa_uid": record["qa_uid"],
            "context_tokens": context_tokens,
            "prompt_template_tokens_before_image_expansion": prompt_tokens,
            "max_new_tokens": max_new,
            "context_limit_tokens": limit,
            "safe_margin_tokens": margin,
            "fits": margin >= 0,
            "clip_top256_sha256": sha256_file(question_dir / "clip_top256.json"),
            "gens_full_query_sha256": sha256_bytes(query.encode("utf-8")),
            "gens_input_identity_sha256": input_identity(query, chronological, profile),
        }
        rows.append(row)
    path = formal_root / "preflight/context_window_audit_300.jsonl"
    v1.atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    margins = [row["safe_margin_tokens"] for row in rows]
    contexts = [row["context_tokens"] for row in rows]
    summary = {
        "status": "PASS" if all(row["fits"] for row in rows) else "FAIL",
        "questions": len(rows),
        "context_tokens_min": min(contexts),
        "context_tokens_mean": statistics.fmean(contexts),
        "context_tokens_median": statistics.median(contexts),
        "context_tokens_max": max(contexts),
        "max_new_tokens": 4096,
        "context_limit_tokens": 11264,
        "safe_margin_tokens_min": min(margins),
        "audit_jsonl_sha256": sha256_file(path),
    }
    v1.atomic_json(formal_root / "preflight/context_window_audit_summary.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("one or more Eval300 inputs do not fit input+4096 context budget")
    return summary


def prepare(formal_root: Path) -> int:
    v1.assert_start_environment()
    formal_root = formal_root.resolve()
    if formal_root == V1_ROOT.resolve() or V1_ROOT.resolve() in formal_root.parents:
        raise RuntimeError("V2 output must not be inside the immutable V1 formal root")
    if formal_root.exists() and any(formal_root.iterdir()):
        marker = formal_root / "FORMAL_RUN.json"
        if not marker.exists():
            raise RuntimeError("refusing a pre-existing non-formal V2 output directory")
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("profile_sha256") != EXPECTED_PROFILE_SHA:
            raise RuntimeError("V2 resume marker profile drift")

    profile, _, _, _, records, frozen, hashes = v2_profile_and_records(True)
    raw_count, raw_sha = raw_manifest_sha(V1_ROOT)
    if (raw_count, raw_sha) != (600, EXPECTED_V1_RAW_MANIFEST_SHA):
        raise RuntimeError("immutable V1 raw-response manifest drift")
    old_marker = json.loads((V1_ROOT / "FORMAL_RUN.json").read_text(encoding="utf-8"))
    if old_marker["frozen_hashes"]["profile_sha256"] != v1.EXPECTED_PROFILE_SHA:
        raise RuntimeError("V1 formal marker profile drift")

    for name in ("preflight", "questions", "shards", "smoke", "logs", "status", "telemetry"):
        (formal_root / name).mkdir(parents=True, exist_ok=True)
    for worker in (0, 1):
        for suffix in ("jsonl", "txt"):
            source_name = f"gpu{worker}_150.jsonl" if suffix == "jsonl" else f"gpu{worker}_uids.txt"
            shutil.copyfile(V1_ROOT / "shards" / source_name, formal_root / "shards" / source_name)

    record_index = {row["qa_uid"]: row for row in records}
    top_hashes = {}
    for uid, record in record_index.items():
        source_dir = V1_ROOT / "questions" / uid
        target_dir = formal_root / "questions" / uid
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in STAGE_A_FILES:
            source = source_dir / name
            target = target_dir / name
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256_file(source) != sha256_file(target):
                raise RuntimeError(f"reused Stage A artifact changed during copy: {uid}/{name}")
        top = json.loads((target_dir / "clip_top256.json").read_text(encoding="utf-8"))
        v1.validate_top256(top, record)
        top_hashes[uid] = sha256_file(target_dir / "clip_top256.json")
        v1.atomic_json(target_dir / "stage_a_reuse.json", {
            "qa_uid": uid,
            "source_formal_root": str(V1_ROOT),
            "source_top256_sha256": top_hashes[uid],
            "stage_a_rerun": False,
            "clip_image_reencoding": False,
            "new_frame_extraction": False,
        })
    if len(top_hashes) != 300:
        raise RuntimeError("Stage A reuse does not cover 300 questions")

    shard_report = {}
    combined = []
    for worker in (0, 1):
        path = formal_root / "shards" / f"gpu{worker}_150.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 150 or len({row["qa_uid"] for row in rows}) != 150:
            raise RuntimeError("V2 shard cardinality error")
        combined.extend(row["qa_uid"] for row in rows)
        shard_report[f"gpu{worker}"] = {
            "qa_count": 150,
            "jsonl_path": str(path),
            "jsonl_sha256": sha256_file(path),
            "uid_list_path": str(formal_root / "shards" / f"gpu{worker}_uids.txt"),
            "uid_list_sha256": sha256_file(formal_root / "shards" / f"gpu{worker}_uids.txt"),
            "first_qa_uid": rows[0]["qa_uid"],
            "last_qa_uid": rows[-1]["qa_uid"],
        }
    if len(combined) != 300 or len(set(combined)) != 300 or set(combined) != set(record_index):
        raise RuntimeError("V2 shards overlap or omit UIDs")

    context_summary = build_context_audit(profile, records, formal_root)
    test_command = [
        str(PYTHON), "-m", "unittest", "-v",
        "tests.test_offline_contract", "tests.test_symmetric_mcq_contract",
        "tests.test_v2_long_output_contract",
    ]
    started = time.perf_counter()
    completed = subprocess.run(test_command, cwd=ROOT / "wrapper", env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    v1.atomic_text(formal_root / "preflight/offline_contract.log", completed.stdout)
    count_match = re.search(r"Ran (\d+) tests?", completed.stdout)
    passed = int(count_match.group(1)) if count_match and completed.returncode == 0 else 0
    contract = {
        "command": test_command,
        "returncode": completed.returncode,
        "tests_passed": passed,
        "tests_expected": 42,
        "status": "42/42_PASS" if completed.returncode == 0 and passed == 42 else "FAIL",
        "wall_clock_sec": time.perf_counter() - started,
        "finished_at": v1.utc_now(),
    }
    v1.atomic_json(formal_root / "preflight/offline_contract_result.json", contract)
    if contract["status"] != "42/42_PASS":
        raise RuntimeError("V2 offline contract failed")

    diffs = validate_profile_diff()
    report = {
        "schema_version": 2,
        "formal_root": str(formal_root),
        "method": DISPLAY_NAME,
        "method_key": METHOD,
        "profile_path": str(PROFILE_PATH),
        "profile_sha256": EXPECTED_PROFILE_SHA,
        "v1_formal_root": str(V1_ROOT),
        "v1_status": "superseded_before_downstream_due_to_output_truncation",
        "v1_raw_response_files": raw_count,
        "v1_raw_response_manifest_sha256": raw_sha,
        "profile_diff": diffs,
        "frozen_hashes": hashes,
        "verified_manifests": frozen,
        "stage_a": {
            "source": str(V1_ROOT), "reruns": 0, "questions": 300,
            "top256_each": True, "uid_to_top256_sha256": top_hashes,
            "new_frame_extractions": 0, "clip_image_reencodings": 0,
        },
        "context_window": context_summary,
        "offline_contract": contract,
        "shards": shard_report,
        "smoke_uids": list(SMOKE_UIDS),
        "api_calls": 0,
        "private_gold_reads": 0,
        "A_to_E_predictions": 0,
        "created_at": v1.utc_now(),
    }
    marker = formal_root / "FORMAL_RUN.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["profile_sha256"] != EXPECTED_PROFILE_SHA or previous["shards"] != shard_report:
            raise RuntimeError("V2 formal resume marker drift")
    v1.atomic_json(marker, report)
    v1.atomic_json(formal_root / "preflight/preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def load_records_and_template():
    from gens_baseline.config import resolve_profile_path

    profile, policy, _, _, records, _, _ = v2_profile_and_records(False)
    template = resolve_profile_path(profile, profile["gens_full_query"]["template_path"]).read_text(encoding="utf-8")
    return profile, policy, records, template


def load_shard(formal_root: Path, worker: int, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marker = json.loads((formal_root / "FORMAL_RUN.json").read_text(encoding="utf-8"))
    info = marker["shards"][f"gpu{worker}"]
    path = Path(info["jsonl_path"])
    if sha256_file(path) != info["jsonl_sha256"]:
        raise RuntimeError("V2 shard SHA drift")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = {row["qa_uid"]: row for row in records}
    if len(rows) != 150 or any(index[row["qa_uid"]] != row for row in rows):
        raise RuntimeError("V2 shard differs from frozen selector")
    return rows


def terminal_complete(question_dir: Path, record: dict[str, Any]) -> bool:
    try:
        finished = json.loads((question_dir / "finished_at.json").read_text(encoding="utf-8"))
        if finished["qa_uid"] != record["qa_uid"]:
            return False
        if finished["status"] == "ok":
            required = (
                "gens_raw_response.txt", "generated_token_ids.json", "generation_telemetry.json",
                "gens_parser_result.json", "gens_parsed_output.json", "final_selected_frames.json",
                "selector_provenance.json", "finished_at.json",
            )
            if not all((question_dir / name).is_file() for name in required):
                return False
            telemetry = json.loads((question_dir / "generation_telemetry.json").read_text(encoding="utf-8"))
            selected = json.loads((question_dir / "final_selected_frames.json").read_text(encoding="utf-8"))
            return telemetry["finish_reason"] in ("eos", "length") and 1 <= len(selected) <= 16
        if finished["status"] == "final_selector_failure":
            return (question_dir / "gens_failure.json").is_file()
        return False
    except Exception:
        return False


def process_question(model, profile, record: dict[str, Any], template: str, stage_a_dir: Path, output_dir: Path, attempt: int = 1) -> dict[str, Any]:
    from gens_baseline.contract import cap_gens_frames
    from gens_baseline.gens_stage import chronological_gens_candidates
    from gens_baseline.parser import ParserFailure, map_parsed_frames, parse_gens_response
    from gens_baseline.query import build_retrieval_query

    top_path = stage_a_dir / "clip_top256.json"
    top = json.loads(top_path.read_text(encoding="utf-8"))
    v1.validate_top256(top, record)
    source_complete = json.loads((stage_a_dir / "stage_a_complete.json").read_text(encoding="utf-8"))
    if source_complete["top256_sha256"] != sha256_file(top_path):
        raise RuntimeError("Stage A SHA drift")
    query = build_retrieval_query(record, template)
    full_chronological = chronological_gens_candidates(top)
    prompt_candidates = [
        {key: value for key, value in item.items() if not str(key).startswith("clip_")}
        for item in top
    ]
    if any(any(str(key).startswith("clip_") for key in item) for item in prompt_candidates):
        raise RuntimeError("CLIP signal leaked into GenS input")
    prompt_chronological = chronological_gens_candidates(prompt_candidates)
    identity = input_identity(query, prompt_chronological, profile)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, returned_candidates, gens_timing, context_tokens, token_ids, finish_reason, eos_observed = model.select_observed(query, prompt_candidates)
    if returned_candidates != prompt_chronological:
        raise RuntimeError("GenS chronological input drift")
    max_new = int(profile["gens"]["generation"]["max_new_tokens"])
    raw_attempt = output_dir / f"gens_raw_response.attempt{attempt}.txt"
    token_attempt = output_dir / f"generated_token_ids.attempt{attempt}.json"
    telemetry_attempt = output_dir / f"generation_telemetry.attempt{attempt}.json"
    v1.atomic_text(raw_attempt, raw)
    v1.atomic_json(token_attempt, token_ids)
    generation = {
        "attempt": attempt,
        "generated_token_count": len(token_ids),
        "eos_observed": eos_observed,
        "finish_reason": finish_reason,
        "context_tokens": context_tokens,
        "max_new_tokens": max_new,
        "context_limit_tokens": int(profile["gens"]["context_limit_tokens"]),
        "safe_margin_tokens": int(profile["gens"]["context_limit_tokens"]) - context_tokens - max_new,
        "generation_latency_ms": gens_timing["gens_generate"],
        "image_read_ms": gens_timing["gens_image_read"],
        "preprocessing_ms": gens_timing["gens_processor_preprocess"],
        "preprocess_total_ms": gens_timing["gens_preprocess"],
        "prompt_text_sha256": gens_timing["prompt_text_sha256"],
        "gens_input_identity_sha256": identity,
        "clip_top256_sha256": sha256_file(top_path),
        "gens_full_query_sha256": sha256_bytes(query.encode("utf-8")),
    }
    v1.atomic_json(telemetry_attempt, generation)

    parse_started = time.perf_counter()
    try:
        parsed = parse_gens_response(raw, len(returned_candidates))
        if not parsed:
            raise ParserFailure("empty_selection", "GenS returned zero legal frames")
        mapped = map_parsed_frames(parsed, full_chronological)
        selected, relevance_order = cap_gens_frames(mapped, 16)
        if not 1 <= len(selected) <= 16:
            raise ParserFailure("empty_selection", "GenS cap produced zero frames")
        parser_ms = (time.perf_counter() - parse_started) * 1000.0
        parser_result = {"status": "ok", "parser_ms": parser_ms, "parsed_frame_count": len(parsed)}
    except ParserFailure as exc:
        parser_ms = (time.perf_counter() - parse_started) * 1000.0
        parser_result = {"status": "failure", "code": exc.code, "message": str(exc), "parser_ms": parser_ms}
        v1.atomic_json(output_dir / f"gens_parser_result.attempt{attempt}.json", parser_result)
        return {"status": "parser_failure", "finish_reason": finish_reason, "generation": generation, "parser": parser_result}

    v1.atomic_text(output_dir / "gens_full_mcq_query.txt", query)
    v1.atomic_json(output_dir / "gens_input_candidates_no_clip_signals.json", returned_candidates)
    shutil.copyfile(raw_attempt, output_dir / "gens_raw_response.txt")
    shutil.copyfile(token_attempt, output_dir / "generated_token_ids.json")
    shutil.copyfile(telemetry_attempt, output_dir / "generation_telemetry.json")
    v1.atomic_json(output_dir / "gens_parser_result.json", parser_result)
    v1.atomic_json(output_dir / "gens_parsed_output.json", [item.to_dict() for item in parsed])
    v1.atomic_json(output_dir / "final_selected_frames.json", selected)
    provenance = {
        "qa_uid": record["qa_uid"], "video_id": record["video_id"],
        "method": METHOD, "profile_sha256": EXPECTED_PROFILE_SHA,
        "v1_formal_root": str(V1_ROOT), "v1_status": "superseded_before_downstream_due_to_output_truncation",
        "clip_top256_sha256": generation["clip_top256_sha256"],
        "gens_full_query_sha256": generation["gens_full_query_sha256"],
        "gens_input_identity_sha256": identity,
        "selected_frames": selected, "gens_relevance_order": relevance_order,
        "clip_scores_or_winning_option_sent_to_gens": False,
        "stage_a_rerun": False, "new_frame_extraction": False, "clip_image_reencoding": False,
        "gold_read": False, "api_calls": 0, "A_to_E_prediction_generated": False,
    }
    v1.atomic_json(output_dir / "selector_provenance.json", provenance)
    v1.atomic_json(output_dir / "stage_b_timings.json", {
        "image_read_ms": generation["image_read_ms"],
        "gens_preprocessing_ms": generation["preprocessing_ms"],
        "gens_preprocess_total_ms": generation["preprocess_total_ms"],
        "gens_generate_ms": generation["generation_latency_ms"],
        "parser_ms": parser_ms, "context_tokens": context_tokens,
        "generated_token_count": len(token_ids), "finish_reason": finish_reason,
        "selected_frame_count": len(selected), "attempt": attempt,
    })
    v1.atomic_json(output_dir / "selector_output.json", {
        "schema_version": 2, "qa_uid": record["qa_uid"], "video_id": record["video_id"],
        "method": METHOD, "status": "ok", "selected_frames": selected,
        "actual_frame_count": len(selected), "contains_gold": False,
        "api_downstream_run": False, "A_to_E_prediction": None,
    })
    v1.atomic_json(output_dir / "finished_at.json", {
        "qa_uid": record["qa_uid"], "video_id": record["video_id"],
        "status": "ok", "finished_at": v1.utc_now(),
    })
    return {"status": "ok", "finish_reason": finish_reason, "generation": generation, "parser": parser_result, "selected": selected}


def fatal_exception(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    terms = (
        "outofmemory", "out of memory", "cuda", "cublas", "cudnn", "nccl", "illegal memory",
        "hash drift", "sha drift", "profile drift", "model", "accessviolation", "private", "network",
        "socket", "cross-video", "path", "no such file", "contextoverflow", "context limit",
    )
    return any(term in name or term in message for term in terms)


def smoke(formal_root: Path) -> int:
    v1.assert_start_environment()
    from smoke_telemetry import ResourceMonitor, SafetyAudit
    import torch
    from gens_baseline.gens_stage import GenSSelector

    marker = json.loads((formal_root / "FORMAL_RUN.json").read_text(encoding="utf-8"))
    if marker["offline_contract"]["status"] != "42/42_PASS" or marker["context_window"]["status"] != "PASS":
        raise RuntimeError("V2 preflight is not complete")
    audit = SafetyAudit(PRIVATE_PREFIX)
    audit.install()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V2 smoke must see exactly one CUDA GPU")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    physical = int(os.environ.get("GENS_PHYSICAL_GPU_INDEX", "0"))
    monitor = ResourceMonitor(formal_root / "smoke/gpu.csv", 1.0, physical)
    monitor.start()
    started = time.perf_counter()
    model = None
    try:
        profile, policy, records, template = load_records_and_template()
        index = {row["qa_uid"]: row for row in records}
        model = GenSSelector(profile, policy, "cuda:0")
        outcomes = []
        for uid in SMOKE_UIDS:
            result = process_question(
                model, profile, index[uid], template,
                formal_root / "questions" / uid,
                formal_root / "smoke/questions" / uid,
            )
            outcomes.append({"qa_uid": uid, **result})
        validations = []
        for position, outcome in enumerate(outcomes):
            uid = outcome["qa_uid"]
            new_dir = formal_root / "smoke/questions" / uid
            old_dir = V1_ROOT / "questions" / uid
            checks: dict[str, Any] = {
                "qa_uid": uid,
                "new_status": outcome["status"],
                "finish_reason": outcome.get("finish_reason"),
                "not_length": outcome.get("finish_reason") != "length",
            }
            if position < 2:
                checks.update({
                    "kind": "v1_success",
                    "raw_byte_identical": (new_dir / "gens_raw_response.txt").read_bytes() == (old_dir / "gens_raw_response.txt").read_bytes(),
                    "parsed_item_identical": json.loads((new_dir / "gens_parsed_output.json").read_text()) == json.loads((old_dir / "gens_parsed_output.json").read_text()),
                    "selected_item_identical": json.loads((new_dir / "final_selected_frames.json").read_text()) == json.loads((old_dir / "final_selected_frames.json").read_text()),
                })
            else:
                selected = json.loads((new_dir / "final_selected_frames.json").read_text()) if outcome["status"] == "ok" else []
                parsed = json.loads((new_dir / "gens_parsed_output.json").read_text()) if outcome["status"] == "ok" else []
                checks.update({
                    "kind": "v1_truncated_failure",
                    "parser_success": outcome["status"] == "ok",
                    "selected_count": len(selected),
                    "selected_count_1_to_16": 1 <= len(selected) <= 16,
                    "gens_input_indices_1_to_256": all(1 <= int(item["gens_input_index"]) <= 256 for item in parsed),
                    "zero_based_candidate_indices_0_to_255": all(0 <= int(item["gens_input_index"]) - 1 <= 255 for item in parsed),
                })
            generated = json.loads((new_dir / "generation_telemetry.json").read_text()) if outcome["status"] == "ok" else outcome["generation"]
            context_row = next(json.loads(line) for line in (formal_root / "preflight/context_window_audit_300.jsonl").read_text().splitlines() if json.loads(line)["qa_uid"] == uid)
            checks.update({
                "top256_hash_matches_preflight": generated["clip_top256_sha256"] == context_row["clip_top256_sha256"],
                "query_hash_matches_preflight": generated["gens_full_query_sha256"] == context_row["gens_full_query_sha256"],
                "input_identity_matches_preflight": generated["gens_input_identity_sha256"] == context_row["gens_input_identity_sha256"],
                "context_matches_preflight": generated["context_tokens"] == context_row["context_tokens"],
            })
            validations.append(checks)
        passed = all(
            check["new_status"] == "ok"
            and check["not_length"]
            and check["top256_hash_matches_preflight"]
            and check["query_hash_matches_preflight"]
            and check["input_identity_matches_preflight"]
            and check["context_matches_preflight"]
            and (
                check.get("kind") != "v1_success"
                or (check["raw_byte_identical"] and check["parsed_item_identical"] and check["selected_item_identical"])
            )
            and (
                check.get("kind") != "v1_truncated_failure"
                or (check["parser_success"] and check["selected_count_1_to_16"] and check["gens_input_indices_1_to_256"])
            )
            for check in validations
        )
        report = {
            "status": "PASS" if passed else "FAIL", "profile_sha256": EXPECTED_PROFILE_SHA,
            "uids": list(SMOKE_UIDS), "validations": validations,
            "wall_clock_sec": time.perf_counter() - started,
            "api_calls": audit.network_attempts, "private_gold_reads": audit.private_access_attempts,
            "finished_at": v1.utc_now(),
        }
        v1.atomic_json(formal_root / "smoke/SMOKE_REPORT.json", report)
        if not passed:
            v1.atomic_json(formal_root / "STOP_FATAL.json", {"type": "SmokeValidationFailure", "report": report, "time": v1.utc_now()})
            return 2
        return 0
    finally:
        if model is not None:
            model.close()
        monitor.stop()
        resource = {
            "wall_clock_sec": time.perf_counter() - started,
            "cpu_rss_peak_bytes": monitor.cpu_rss_peak_bytes,
            "torch_cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "torch_cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "nvidia_smi_device_memory_peak_mib": monitor.device_memory_peak_mib,
            "nvidia_smi_samples": monitor.samples, "telemetry_errors": monitor.errors,
            "api_calls": audit.network_attempts, "private_gold_reads": audit.private_access_attempts,
            "finished_at": v1.utc_now(),
        }
        v1.atomic_json(formal_root / "smoke/resource_summary.json", resource)


def worker_main(formal_root: Path, worker: int) -> int:
    v1.assert_start_environment()
    from smoke_telemetry import ResourceMonitor, SafetyAudit
    import torch
    from gens_baseline.gens_stage import GenSSelector

    audit = SafetyAudit(PRIVATE_PREFIX)
    audit.install()
    physical = int(os.environ.get("GENS_PHYSICAL_GPU_INDEX", str(worker)))
    monitor = ResourceMonitor(formal_root / "telemetry" / f"worker{worker}_gens_gpu.csv", 1.0, physical)
    monitor.start()
    started = time.perf_counter()
    model = None
    processed = legal = resumed = task_failures = runtime_retries = parser_failures = length_finishes = 0
    fatal = None
    status_path = formal_root / "status" / f"worker{worker}_gens.json"
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("each V2 worker must see exactly one CUDA GPU")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        profile, policy, records, template = load_records_and_template()
        shard = load_shard(formal_root, worker, records)
        stop_path = formal_root / "STOP_FATAL.json"
        v1.atomic_json(status_path, {"status": "loading_model", "worker": worker, "pid": os.getpid(), "pgid": os.getpgrp(), "started_at": v1.utc_now()})
        load_started = time.perf_counter()
        model = GenSSelector(profile, policy, "cuda:0")
        torch.cuda.synchronize(0)
        model_load_ms = (time.perf_counter() - load_started) * 1000.0
        for position, record in enumerate(shard, 1):
            if stop_path.exists():
                raise RuntimeError("peer requested stop after a fatal error")
            qdir = formal_root / "questions" / record["qa_uid"]
            if terminal_complete(qdir, record):
                resumed += 1
                processed += 1
                if json.loads((qdir / "finished_at.json").read_text())["status"] == "ok":
                    legal += 1
                continue
            result = None
            last_exc = None
            for attempt in (1, 2):
                try:
                    result = process_question(model, profile, record, template, qdir, qdir, attempt)
                    break
                except Exception as exc:
                    last_exc = exc
                    v1.atomic_json(qdir / f"gens_attempt{attempt}_failure.json", {
                        "qa_uid": record["qa_uid"], "attempt": attempt,
                        "type": type(exc).__name__, "message": str(exc), "time": v1.utc_now(),
                    })
                    if fatal_exception(exc):
                        raise
                    if attempt == 1:
                        runtime_retries += 1
                        continue
            if result is None:
                task_failures += 1
                result = {"status": "runtime_failure", "finish_reason": "error"}
                v1.atomic_json(qdir / "gens_failure.json", {
                    "qa_uid": record["qa_uid"], "type": type(last_exc).__name__,
                    "message": str(last_exc), "retries": 1, "time": v1.utc_now(),
                })
                v1.atomic_json(qdir / "finished_at.json", {"qa_uid": record["qa_uid"], "video_id": record["video_id"], "status": "final_selector_failure", "finished_at": v1.utc_now()})
            elif result["status"] == "parser_failure":
                parser_failures += 1
                task_failures += 1
                length_finishes += int(result["finish_reason"] == "length")
                v1.atomic_json(qdir / "gens_failure.json", {
                    "qa_uid": record["qa_uid"], "type": "ParserFailure",
                    "message": result["parser"]["message"], "finish_reason": result["finish_reason"],
                    "retries": 0, "time": v1.utc_now(),
                })
                v1.atomic_json(qdir / "finished_at.json", {"qa_uid": record["qa_uid"], "video_id": record["video_id"], "status": "final_selector_failure", "finished_at": v1.utc_now()})
            else:
                legal += 1
                length_finishes += int(result["finish_reason"] == "length")
            processed += 1
            v1.atomic_json(status_path, {
                "status": "running", "worker": worker, "pid": os.getpid(), "pgid": os.getpgrp(),
                "model_load_ms": model_load_ms, "processed": processed, "legal": legal,
                "resumed": resumed, "task_failures": task_failures, "parser_failures": parser_failures,
                "length_finishes": length_finishes, "runtime_retries": runtime_retries,
                "total": 150, "last_qa_uid": record["qa_uid"], "last_position": position, "updated_at": v1.utc_now(),
            })
            print(json.dumps({"worker": worker, "position": position, "qa_uid": record["qa_uid"], "processed": processed, "legal": legal, "status": result["status"], "finish_reason": result.get("finish_reason")}), flush=True)
        v1.atomic_json(status_path, {
            "status": "complete", "worker": worker, "pid": os.getpid(), "pgid": os.getpgrp(),
            "model_load_ms": model_load_ms, "processed": processed, "legal": legal,
            "resumed": resumed, "task_failures": task_failures, "parser_failures": parser_failures,
            "length_finishes": length_finishes, "runtime_retries": runtime_retries,
            "total": 150, "finished_at": v1.utc_now(),
        })
        return 0
    except Exception as exc:
        fatal = {"worker": worker, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "time": v1.utc_now()}
        v1.atomic_json(formal_root / "STOP_FATAL.json", fatal)
        v1.atomic_json(status_path, {"status": "fatal", **fatal, "pid": os.getpid(), "pgid": os.getpgrp(), "processed": processed, "total": 150})
        return 2
    finally:
        if model is not None:
            model.close()
        monitor.stop()
        resource = {
            "worker": worker, "stage": "gens_v2", "wall_clock_sec": time.perf_counter() - started,
            "cpu_rss_peak_bytes": monitor.cpu_rss_peak_bytes,
            "torch_cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else None,
            "torch_cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)) if torch.cuda.is_available() else None,
            "nvidia_smi_device_memory_peak_mib": monitor.device_memory_peak_mib,
            "nvidia_smi_samples": monitor.samples, "telemetry_errors": monitor.errors,
            "api_calls": audit.network_attempts, "private_gold_reads": audit.private_access_attempts,
            "processed": processed, "legal": legal, "task_failures": task_failures,
            "parser_failures": parser_failures, "length_finishes": length_finishes,
            "runtime_retries": runtime_retries, "fatal": fatal, "finished_at": v1.utc_now(),
        }
        v1.atomic_json(formal_root / "telemetry" / f"worker{worker}_gens_resource.json", resource)


def percentile95(values: list[int]) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))])


def build_final_report(formal_root: Path) -> dict[str, Any]:
    _, _, records, _ = load_records_and_template()
    legal = failures = eos = length = 0
    tokens: list[int] = []
    selected_distribution: Counter[int] = Counter()
    v1_raw_same = v1_parsed_same = v1_selected_same = 0
    v1_success_total = v1_recovered = 0
    old_failures = 0
    failure_rows = []
    per_question = []
    for record in records:
        uid = record["qa_uid"]
        qdir = formal_root / "questions" / uid
        old = V1_ROOT / "questions" / uid
        was_old_success = (old / "gens_raw_response.txt").exists()
        if was_old_success:
            v1_success_total += 1
        else:
            old_failures += 1
        finished = json.loads((qdir / "finished_at.json").read_text()) if (qdir / "finished_at.json").exists() else {"status": "missing"}
        if finished["status"] == "ok":
            legal += 1
            generation = json.loads((qdir / "generation_telemetry.json").read_text())
            timing = json.loads((qdir / "stage_b_timings.json").read_text())
            tokens.append(int(generation["generated_token_count"]))
            eos += int(generation["finish_reason"] == "eos")
            length += int(generation["finish_reason"] == "length")
            selected = json.loads((qdir / "final_selected_frames.json").read_text())
            selected_distribution[len(selected)] += 1
            if was_old_success:
                v1_raw_same += int((qdir / "gens_raw_response.txt").read_bytes() == (old / "gens_raw_response.txt").read_bytes())
                v1_parsed_same += int(json.loads((qdir / "gens_parsed_output.json").read_text()) == json.loads((old / "gens_parsed_output.json").read_text()))
                v1_selected_same += int(selected == json.loads((old / "final_selected_frames.json").read_text()))
            else:
                v1_recovered += 1
            per_question.append({"qa_uid": uid, "status": "ok", "selected_frame_count": len(selected), "generated_token_count": generation["generated_token_count"], "finish_reason": generation["finish_reason"], **timing})
        else:
            failures += 1
            failure = json.loads((qdir / "gens_failure.json").read_text()) if (qdir / "gens_failure.json").exists() else {"type": "MissingOutput", "message": "no terminal result"}
            failure_rows.append({"qa_uid": uid, **failure})
            per_question.append({"qa_uid": uid, "status": "failure", "failure": failure})
    resources = [json.loads((formal_root / "telemetry" / f"worker{worker}_gens_resource.json").read_text()) for worker in (0, 1)]
    started_times = [json.loads((formal_root / "status" / f"worker{worker}_gens.json").read_text()).get("started_at") for worker in (0, 1)]
    report = {
        "status": "PASS" if legal + failures == 300 else "FAIL",
        "profile_sha256": EXPECTED_PROFILE_SHA, "method": METHOD,
        "legal_selected_frame_questions": legal, "final_failures": failures,
        "parser_failures": sum(item.get("parser_failures", 0) for item in resources),
        "eos_finishes": eos, "length_finishes": length,
        "output_tokens": {
            "min": min(tokens) if tokens else None, "mean": statistics.fmean(tokens) if tokens else None,
            "median": statistics.median(tokens) if tokens else None, "p95": percentile95(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
        "selected_frame_count_distribution": dict(sorted(selected_distribution.items())),
        "all_legal_selected_counts_1_to_16": sum(selected_distribution.values()) == legal and all(1 <= key <= 16 for key in selected_distribution),
        "v1_229_success_consistency": {
            "total": v1_success_total, "raw_byte_identical": v1_raw_same,
            "parsed_selection_identical": v1_parsed_same, "selected_frames_identical": v1_selected_same,
        },
        "v1_71_failures": {"total": old_failures, "recovered": v1_recovered, "still_failed": old_failures - v1_recovered},
        "failures": failure_rows,
        "worker_wall_clock_sec": {str(item["worker"]): item["wall_clock_sec"] for item in resources},
        "gpu_hours": sum(item["wall_clock_sec"] for item in resources) / 3600.0,
        "resource_peaks": resources,
        "per_question": per_question,
        "api_calls": sum(item["api_calls"] for item in resources),
        "private_gold_reads": sum(item["private_gold_reads"] for item in resources),
        "stage_a_reruns": 0, "new_frame_extractions": 0, "clip_image_reencodings": 0,
        "A_to_E_predictions": 0, "finished_at": v1.utc_now(),
    }
    v1.atomic_json(formal_root / "FINAL_SELECTOR_REPORT.json", report)
    return report


def controller(formal_root: Path) -> int:
    v1.assert_start_environment()
    marker = json.loads((formal_root / "FORMAL_RUN.json").read_text(encoding="utf-8"))
    smoke_report = json.loads((formal_root / "smoke/SMOKE_REPORT.json").read_text(encoding="utf-8"))
    if marker["profile_sha256"] != EXPECTED_PROFILE_SHA or smoke_report["status"] != "PASS":
        raise RuntimeError("V2 controller requires matching preflight and PASS smoke")
    started = time.perf_counter()
    processes = []
    launch_records = []
    for worker in (0, 1):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker)
        env["GENS_PHYSICAL_GPU_INDEX"] = str(worker)
        command = [str(PYTHON), str(Path(__file__)), "worker", "--formal-root", str(formal_root), "--worker", str(worker)]
        log_path = formal_root / "logs" / f"worker{worker}_gens.log"
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((process, handle))
        launch_records.append({"worker": worker, "pid": process.pid, "pgid": os.getpgid(process.pid), "cuda_visible_devices": str(worker), "command": command, "log": str(log_path)})
    v1.atomic_json(formal_root / "status/gens_worker_processes.json", {"workers": launch_records, "launched_at": v1.utc_now()})
    v1.atomic_json(formal_root / "status/controller.json", {"status": "running", "current_stage": "B_GENS_V2", "pid": os.getpid(), "pgid": os.getpgrp(), "workers": launch_records, "started_at": v1.utc_now()})
    while any(process.poll() is None for process, _ in processes):
        if (formal_root / "STOP_FATAL.json").exists():
            for process, _ in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            break
        time.sleep(2)
    returncodes = []
    for process, handle in processes:
        try:
            returncodes.append(process.wait(timeout=30))
        finally:
            handle.close()
    if returncodes != [0, 0]:
        v1.atomic_json(formal_root / "status/controller.json", {"status": "stopped", "current_stage": "B_GENS_V2", "returncodes": returncodes, "finished_at": v1.utc_now()})
        return 2
    report = build_final_report(formal_root)
    report["overall_wall_clock_sec"] = time.perf_counter() - started
    v1.atomic_json(formal_root / "FINAL_SELECTOR_REPORT.json", report)
    v1.atomic_json(formal_root / "status/controller.json", {"status": "complete", "current_stage": "COMPLETE", "legal": report["legal_selected_frame_questions"], "failures": report["final_failures"], "finished_at": v1.utc_now()})
    return 0


def show_status(formal_root: Path) -> int:
    result = {}
    for path in sorted((formal_root / "status").glob("*.json")):
        result[path.name] = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    root = args.formal_root.resolve()
    if args.command == "prepare":
        return prepare(root)
    if args.command == "smoke":
        return smoke(root)
    if args.command == "worker":
        return worker_main(root, args.worker)
    if args.command == "controller":
        return controller(root)
    if args.command == "status":
        return show_status(root)
    if args.command == "report":
        print(json.dumps(build_final_report(root), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
