from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_PROFILE_SHA = "e817014952c76d5cf770062a6ca5d2324f5595b560c7baec1f2db40a0ce45b31"
EXPECTED_CLIP_QUERY_SHA = "8391f7c5dbd15fb122f270fdbb8d652820cd2d6d6507a6606d3ff0430adb716f"
EXPECTED_GENS_QUERY_SHA = "f32b2f1b38241811e461c1c5e6c47f0b5e54abdb9710867e87ec557bfa788985"
EXPECTED_EMBEDDING_TREE_SHA = "0a2d41faca48e52f558860220aad2c14c6452e52e0125a8f7daa8502693910c7"
EXPECTED_FRAME_TREE_SHA = "831695bc7c594f62b6b000b7534fe896c34cb061d40944b1ed31a8ed754bb632"
METHOD = "gens_hybrid_symmetric_mcq_cap16_v1"
DISPLAY_NAME = "GenS-Hybrid-SymmetricMCQ-cap16-v1"
PRIVATE_PREFIX = "/myriadfs/home/ucemxna/Scratch/evaluation_private"
ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config/gens_hybrid_symmetric_mcq_cap16_v1.json"
PYTHON = Path("/myriadfs/home/ucemxna/Scratch/workspace/envs/videoseal/bin/python3.12")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--formal-root", required=True, type=Path)
    worker = sub.add_parser("worker")
    worker.add_argument("--formal-root", required=True, type=Path)
    worker.add_argument("--worker", required=True, type=int, choices=(0, 1))
    worker.add_argument("--stage", required=True, choices=("clip", "gens"))
    validate = sub.add_parser("validate-stage-a")
    validate.add_argument("--formal-root", required=True, type=Path)
    controller = sub.add_parser("controller")
    controller.add_argument("--formal-root", required=True, type=Path)
    status = sub.add_parser("status")
    status.add_argument("--formal-root", required=True, type=Path)
    return parser.parse_args()


def assert_start_environment() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before Python/torch starts")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("formal selector must be fully offline")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(key):
            raise RuntimeError(f"secret-bearing environment variable must be unset: {key}")


def profile_and_records(full_manifests: bool = False):
    from gens_baseline.config import load_profile, resolve_profile_path, verify_frozen_artifacts
    from gens_baseline.pipeline import build_access_policy
    from gens_baseline.query import load_selector
    from gens_baseline.symmetric_mcq import load_symmetric_template

    if sha256_file(PROFILE_PATH) != EXPECTED_PROFILE_SHA:
        raise RuntimeError("profile SHA drift")
    profile = load_profile(PROFILE_PATH)
    frozen = verify_frozen_artifacts(profile, full_manifests=full_manifests)
    policy, allowlist_path, output_root = build_access_policy(profile)
    records = load_selector(profile, policy)
    load_symmetric_template(profile)
    full_query_path = resolve_profile_path(profile, profile["gens_full_query"]["template_path"])
    hashes = {
        "profile_sha256": profile["_profile_sha256"],
        "clip_symmetric_query_template_sha256": profile["symmetric_mcq"]["clip_query_template_sha256"],
        "gens_full_mcq_query_template_sha256": sha256_file(full_query_path),
        "clip_image_embedding_cache_tree_sha256": profile["cached_image_embeddings"]["global_tree_sha256"],
        "canonical_frame_cache_tree_sha256": profile["candidate_cache"]["total_tree_sha256"],
    }
    expected = {
        "profile_sha256": EXPECTED_PROFILE_SHA,
        "clip_symmetric_query_template_sha256": EXPECTED_CLIP_QUERY_SHA,
        "gens_full_mcq_query_template_sha256": EXPECTED_GENS_QUERY_SHA,
        "clip_image_embedding_cache_tree_sha256": EXPECTED_EMBEDDING_TREE_SHA,
        "canonical_frame_cache_tree_sha256": EXPECTED_FRAME_TREE_SHA,
    }
    if hashes != expected:
        raise RuntimeError(f"frozen SHA mismatch: {hashes}")
    embedding_sums = Path(profile["_config_dir"]) / profile["cached_image_embeddings"]["root"] / "CACHE_SHA256SUMS.txt"
    if sha256_file(embedding_sums.resolve()) != EXPECTED_EMBEDDING_TREE_SHA:
        raise RuntimeError("embedding CACHE_SHA256SUMS tree identity drift")
    return profile, policy, allowlist_path, output_root, records, frozen, hashes


def prepare(formal_root: Path) -> int:
    assert_start_environment()
    if formal_root.exists() and not (formal_root / "FORMAL_RUN.json").exists():
        existing = {path.name for path in formal_root.iterdir()}
        if not existing.issubset({"preflight", "shards"}):
            raise RuntimeError("refusing a pre-existing non-formal output directory")
    formal_root.mkdir(parents=True, exist_ok=True)
    profile, policy, allowlist_path, output_root, records, frozen, hashes = profile_and_records(True)
    formal_root = formal_root.resolve()
    if formal_root.parent != output_root.resolve():
        raise RuntimeError("formal root must be a new direct child of the frozen output root")

    forbidden_fields = {str(x).lower() for x in profile["inputs"]["forbidden_fields"]}
    input_dir = Path(profile["inputs"]["selector"]).parent.resolve()
    input_files = sorted(path.name for path in input_dir.iterdir() if path.is_file())
    if Path(PRIVATE_PREFIX).resolve() == input_dir or Path(PRIVATE_PREFIX).resolve() in input_dir.parents:
        raise RuntimeError("public selector input directory is under private prefix")
    for record in records:
        if forbidden_fields.intersection(str(key).lower() for key in record):
            raise RuntimeError("selector leaked a forbidden field")

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[str(record["video_id"])].append(record)
    videos = sorted(by_video)
    counts = {video: len(by_video[video]) for video in videos}
    if len(records) != 300 or len({x["qa_uid"] for x in records}) != 300 or len(videos) != 12:
        raise RuntimeError("Eval300 cardinality drift")
    if set(counts.values()) != {25}:
        raise RuntimeError(f"Eval300 must contain exactly 25 questions per video: {counts}")

    shard_report: dict[str, Any] = {}
    all_uids: list[str] = []
    for worker, selected_videos in enumerate((videos[:6], videos[6:])):
        shard_records = [record for video in selected_videos for record in by_video[video]]
        lines = b"".join(canonical_bytes(record) + b"\n" for record in shard_records)
        shard_path = formal_root / "shards" / f"gpu{worker}_150.jsonl"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        if shard_path.exists() and shard_path.read_bytes() != lines:
            raise RuntimeError("resume shard bytes differ from deterministic shard")
        if not shard_path.exists():
            atomic_text(shard_path, lines.decode("utf-8"))
        uid_lines = "".join(f"{record['qa_uid']}\n" for record in shard_records)
        uid_path = formal_root / "shards" / f"gpu{worker}_uids.txt"
        if uid_path.exists() and uid_path.read_text(encoding="utf-8") != uid_lines:
            raise RuntimeError("resume UID list differs from deterministic shard")
        if not uid_path.exists():
            atomic_text(uid_path, uid_lines)
        all_uids.extend(record["qa_uid"] for record in shard_records)
        shard_report[f"gpu{worker}"] = {
            "gpu": worker,
            "video_ids": selected_videos,
            "video_count": len(selected_videos),
            "qa_count": len(shard_records),
            "jsonl_path": str(shard_path),
            "jsonl_sha256": sha256_file(shard_path),
            "uid_list_path": str(uid_path),
            "uid_list_sha256": sha256_file(uid_path),
            "first_qa_uid": shard_records[0]["qa_uid"],
            "last_qa_uid": shard_records[-1]["qa_uid"],
        }
    if len(all_uids) != 300 or len(set(all_uids)) != 300 or set(all_uids) != {x["qa_uid"] for x in records}:
        raise RuntimeError("shards overlap or omit Eval300 UIDs")

    test_command = [
        str(PYTHON), "-m", "unittest", "-v",
        "tests.test_offline_contract", "tests.test_symmetric_mcq_contract",
    ]
    contract_started = time.perf_counter()
    completed = subprocess.run(
        test_command, cwd=ROOT / "wrapper", env=os.environ.copy(),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    atomic_text(formal_root / "preflight" / "offline_contract_35.log", completed.stdout)
    contract_count = 35 if "Ran 35 tests" in completed.stdout and completed.stdout.rstrip().endswith("OK") else completed.stdout.count(" ... ok")
    contract = {
        "command": test_command,
        "returncode": completed.returncode,
        "tests_passed": contract_count,
        "tests_expected": 35,
        "status": "35/35_PASS" if completed.returncode == 0 and contract_count == 35 else "FAIL",
        "wall_clock_sec": time.perf_counter() - contract_started,
        "finished_at": utc_now(),
    }
    atomic_json(formal_root / "preflight" / "offline_contract_result.json", contract)
    if contract["status"] != "35/35_PASS":
        raise RuntimeError("offline contract did not pass 35/35")

    report = {
        "schema_version": 1,
        "formal_root": str(formal_root),
        "method": DISPLAY_NAME,
        "method_key": METHOD,
        "created_at": utc_now(),
        "frozen_hashes": hashes,
        "verified_manifests": frozen,
        "full_artifact_manifest_verification": "PASS",
        "offline_contract": contract,
        "eval300": {
            "records": 300,
            "unique_qa_uids": 300,
            "unique_videos": 12,
            "questions_per_video": counts,
            "selector_sha256": profile["inputs"]["selector_sha256"],
            "uid_order_sha256": profile["inputs"]["uid_order_sha256"],
        },
        "shards": shard_report,
        "input_boundary": {
            "public_input_directory": str(input_dir),
            "files": input_files,
            "contains_private_gold": False,
            "private_prefix": PRIVATE_PREFIX,
            "private_prefix_is_outside_input_directory": True,
            "runtime_positive_allowlist_and_python_audit_block_private_prefix": True,
            "network_socket_connect_blocked": True,
        },
        "frozen_execution": {
            "candidate_fps": 1,
            "clip_queries_per_question": 5,
            "clip_aggregation": "max_over_options",
            "clip_top_k": 256,
            "gens_resolution": 112,
            "final_cap": 16,
            "api_downstream": False,
            "produces_A_to_E_prediction": False,
        },
        "disclosure": (
            "GenS-Hybrid-cap16 with symmetric MCQ query adaptation; symmetric per-option "
            "multi-query/max aggregation is our fairness adaptation, not a public GenS author implementation."
        ),
    }
    marker = formal_root / "FORMAL_RUN.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["frozen_hashes"] != hashes or previous["shards"] != shard_report:
            raise RuntimeError("formal resume marker drift")
    else:
        atomic_json(marker, report)
    atomic_json(formal_root / "preflight" / "preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def load_shard(formal_root: Path, worker: int) -> list[dict[str, Any]]:
    marker = json.loads((formal_root / "FORMAL_RUN.json").read_text(encoding="utf-8"))
    info = marker["shards"][f"gpu{worker}"]
    path = Path(info["jsonl_path"])
    if sha256_file(path) != info["jsonl_sha256"]:
        raise RuntimeError("formal shard SHA drift")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 150 or len({x["qa_uid"] for x in rows}) != 150:
        raise RuntimeError("worker shard cardinality drift")
    if set(x["video_id"] for x in rows) != set(info["video_ids"]):
        raise RuntimeError("worker shard video set drift")
    return rows


def validate_top256(value: Any, record: dict[str, Any]) -> None:
    if not isinstance(value, list) or len(value) != 256:
        raise RuntimeError("top-256 must contain exactly 256 rows")
    if len({(x["video_id"], int(x["frame_index"])) for x in value}) != 256:
        raise RuntimeError("top-256 contains duplicate frames")
    if any(x["video_id"] != record["video_id"] for x in value):
        raise RuntimeError("top-256 contains cross-video frame")
    if [int(x["clip_rank"]) for x in value] != list(range(1, 257)):
        raise RuntimeError("top-256 ranks are not 1..256")
    keys = [(-float(x["clip_score"]), float(x["timestamp_sec"]), int(x["frame_index"])) for x in value]
    if keys != sorted(keys):
        raise RuntimeError("top-256 ordering drift")
    for row in value:
        if tuple(row["clip_option_scores"].keys()) != ("A", "B", "C", "D", "E"):
            raise RuntimeError("per-option score telemetry drift")
        winner = max((float(score), -"ABCDE".index(label), label) for label, score in row["clip_option_scores"].items())[2]
        if row["clip_winning_option_label"] != winner:
            raise RuntimeError("winning-option telemetry drift")


def stage_a_complete(question_dir: Path, record: dict[str, Any]) -> bool:
    try:
        top_path = question_dir / "clip_top256.json"
        complete = json.loads((question_dir / "stage_a_complete.json").read_text(encoding="utf-8"))
        value = json.loads(top_path.read_text(encoding="utf-8"))
        validate_top256(value, record)
        return complete["qa_uid"] == record["qa_uid"] and complete["top256_sha256"] == sha256_file(top_path)
    except Exception:
        return False


def final_complete(question_dir: Path, record: dict[str, Any]) -> bool:
    try:
        if not stage_a_complete(question_dir, record):
            return False
        raw = (question_dir / "gens_raw_response.txt").read_text(encoding="utf-8")
        parsed = json.loads((question_dir / "gens_parsed_output.json").read_text(encoding="utf-8"))
        selected = json.loads((question_dir / "final_selected_frames.json").read_text(encoding="utf-8"))
        finished = json.loads((question_dir / "finished_at.json").read_text(encoding="utf-8"))
        if not raw.strip() or not isinstance(parsed, list) or not parsed or not 1 <= len(selected) <= 16:
            return False
        if finished["qa_uid"] != record["qa_uid"] or finished["status"] != "ok":
            return False
        order = [(x["timestamp_sec"], x["frame_index"]) for x in selected]
        return order == sorted(order) and len(set(order)) == len(order)
    except Exception:
        return False


def fatal_exception(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    terms = (
        "outofmemory", "out of memory", "cuda", "cublas", "cudnn", "nccl",
        "illegal memory", "embedding", "hash drift", "sha drift", "accessviolation",
        "private", "network", "socket", "service", "cross-video",
    )
    return any(term in name or term in message for term in terms)


def resource_summary(monitor, torch, started: float, audit, stage: str, worker: int) -> dict[str, Any]:
    monitor.stop()
    return {
        "worker": worker,
        "stage": stage,
        "started_elapsed_wall_clock_sec": time.perf_counter() - started,
        "cpu_rss_peak_bytes": monitor.cpu_rss_peak_bytes,
        "torch_cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "torch_cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "nvidia_smi_device_memory_peak_mib": monitor.device_memory_peak_mib,
        "nvidia_smi_samples": monitor.samples,
        "telemetry_errors": monitor.errors,
        "private_read_attempts": audit.private_access_attempts,
        "network_attempts": audit.network_attempts,
        "finished_at": utc_now(),
    }


def worker_main(formal_root: Path, worker: int, stage: str) -> int:
    assert_start_environment()
    from smoke_telemetry import ResourceMonitor, SafetyAudit

    audit = SafetyAudit(PRIVATE_PREFIX)
    audit.install()
    started = time.perf_counter()
    physical_gpu = int(os.environ.get("GENS_PHYSICAL_GPU_INDEX", str(worker)))
    monitor = ResourceMonitor(formal_root / "telemetry" / f"worker{worker}_{stage}_gpu.csv", 1.0, physical_gpu)
    monitor.start()
    model = None
    fatal = None
    task_failures = 0
    retries = 0
    parser_failures = 0
    completed_count = 0
    resumed_count = 0
    try:
        import torch
        from gens_baseline.cache import load_frozen_cache_candidates
        from gens_baseline.config import resolve_profile_path
        from gens_baseline.contract import cap_gens_frames
        from gens_baseline.gens_stage import GenSSelector, chronological_gens_candidates
        from gens_baseline.parser import ParserFailure, map_parsed_frames, parse_gens_response
        from gens_baseline.query import build_retrieval_query
        from gens_baseline.symmetric_mcq import CachedSymmetricClipRetriever, build_option_queries, load_symmetric_template

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("each formal worker must see exactly one CUDA GPU")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        profile, policy, _, _, records, _, hashes = profile_and_records(False)
        record_index = {x["qa_uid"]: x for x in records}
        shard = load_shard(formal_root, worker)
        if any(record_index[x["qa_uid"]] != x for x in shard):
            raise RuntimeError("shard differs from frozen selector")
        stop_path = formal_root / "STOP_FATAL.json"
        status_path = formal_root / "status" / f"worker{worker}_{stage}.json"
        atomic_json(status_path, {"status": "loading_model", "stage": stage, "worker": worker, "pid": os.getpid(), "pgid": os.getpgrp(), "started_at": utc_now()})
        load_started = time.perf_counter()
        if stage == "clip":
            model = CachedSymmetricClipRetriever(profile, policy, "cuda:0")
        else:
            model = GenSSelector(profile, policy, "cuda:0")
        torch.cuda.synchronize(0)
        model_load_ms = (time.perf_counter() - load_started) * 1000.0
        atomic_json(status_path, {"status": "running", "stage": stage, "worker": worker, "pid": os.getpid(), "pgid": os.getpgrp(), "model_load_ms": model_load_ms, "completed": 0, "total": 150, "updated_at": utc_now()})

        current_video = None
        candidates = None
        if stage == "clip":
            clip_template = load_symmetric_template(profile)
        else:
            full_template_path = resolve_profile_path(profile, profile["gens_full_query"]["template_path"])
            full_template = full_template_path.read_text(encoding="utf-8")

        for position, record in enumerate(shard, 1):
            if stop_path.exists():
                raise RuntimeError("peer requested stop after a fatal error")
            question_dir = formal_root / "questions" / record["qa_uid"]
            question_dir.mkdir(parents=True, exist_ok=True)
            if stage == "clip" and stage_a_complete(question_dir, record):
                resumed_count += 1
                completed_count += 1
                continue
            if stage == "gens" and final_complete(question_dir, record):
                resumed_count += 1
                completed_count += 1
                continue
            per_question_started = time.perf_counter()
            succeeded = False
            last_error: BaseException | None = None
            for attempt in (1, 2):
                if attempt == 2:
                    retries += 1
                try:
                    if stage == "clip":
                        load_candidates_started = time.perf_counter()
                        if current_video != record["video_id"]:
                            candidates = load_frozen_cache_candidates(
                                profile=profile, video_id=record["video_id"],
                                duration_sec=float(record["video_duration_sec"]), policy=policy,
                            )
                            current_video = record["video_id"]
                        candidate_load_ms = (time.perf_counter() - load_candidates_started) * 1000.0
                        queries = build_option_queries(record, clip_template)
                        top256, query_telemetry, timing = model.retrieve(queries, candidates)
                        validate_top256(top256, record)
                        atomic_json(question_dir / "question_metadata.json", {"qa_uid": record["qa_uid"], "video_id": record["video_id"], "worker": worker, "shard_position": position})
                        atomic_json(question_dir / "clip_per_option_queries.json", queries)
                        atomic_json(question_dir / "clip_per_option_query_telemetry.json", query_telemetry)
                        atomic_json(question_dir / "clip_top256.json", top256)
                        top_sha = sha256_file(question_dir / "clip_top256.json")
                        timing.update({"candidate_manifest_load_ms": candidate_load_ms, "question_wall_clock_ms": (time.perf_counter() - per_question_started) * 1000.0})
                        atomic_json(question_dir / "stage_a_timings.json", timing)
                        atomic_json(question_dir / "stage_a_complete.json", {"qa_uid": record["qa_uid"], "video_id": record["video_id"], "top256_count": 256, "top256_sha256": top_sha, "finished_at": utc_now()})
                    else:
                        if not stage_a_complete(question_dir, record):
                            raise RuntimeError("stage A prerequisite is incomplete or invalid")
                        top256 = json.loads((question_dir / "clip_top256.json").read_text(encoding="utf-8"))
                        validate_top256(top256, record)
                        full_query = build_retrieval_query(record, full_template)
                        full_chronological = chronological_gens_candidates(top256)
                        prompt_candidates = [
                            {key: value for key, value in item.items() if not str(key).startswith("clip_")}
                            for item in top256
                        ]
                        if any(any(str(key).startswith("clip_") for key in item) for item in prompt_candidates):
                            raise RuntimeError("CLIP signal leaked into GenS input")
                        raw, prompt_chronological, gens_timing, context_tokens = model.select(full_query, prompt_candidates)
                        raw_attempt = question_dir / f"gens_raw_response.attempt{attempt}.txt"
                        atomic_text(raw_attempt, raw)
                        parse_started = time.perf_counter()
                        parsed = parse_gens_response(raw, len(prompt_chronological))
                        parser_ms = (time.perf_counter() - parse_started) * 1000.0
                        if not parsed:
                            raise ParserFailure("empty_selection", "GenS returned zero legal frames")
                        mapped = map_parsed_frames(parsed, full_chronological)
                        selected, relevance_order = cap_gens_frames(mapped, 16)
                        if not 1 <= len(selected) <= 16:
                            raise ParserFailure("empty_selection", "GenS cap produced zero frames")
                        atomic_text(question_dir / "gens_full_mcq_query.txt", full_query)
                        atomic_json(question_dir / "gens_input_candidates_no_clip_signals.json", prompt_chronological)
                        atomic_text(question_dir / "gens_raw_response.txt", raw)
                        atomic_json(question_dir / "gens_parsed_output.json", [item.to_dict() for item in parsed])
                        atomic_json(question_dir / "final_selected_frames.json", selected)
                        timing = {
                            "image_read_ms": gens_timing.get("gens_image_read"),
                            "gens_preprocessing_ms": gens_timing.get("gens_processor_preprocess"),
                            "gens_preprocess_total_ms": gens_timing["gens_preprocess"],
                            "gens_generate_ms": gens_timing["gens_generate"],
                            "parser_ms": parser_ms,
                            "context_tokens": context_tokens,
                            "selected_frame_count": len(selected),
                            "attempt": attempt,
                            "question_wall_clock_ms": (time.perf_counter() - per_question_started) * 1000.0,
                        }
                        provenance = {
                            "qa_uid": record["qa_uid"], "video_id": record["video_id"],
                            "profile_sha256": EXPECTED_PROFILE_SHA,
                            "clip_top256_sha256": sha256_file(question_dir / "clip_top256.json"),
                            "gens_full_query_sha256": sha256_bytes(full_query.encode("utf-8")),
                            "selected_frames": selected,
                            "gens_relevance_order": relevance_order,
                            "clip_scores_or_winning_option_sent_to_gens": False,
                            "gold_read": False, "api_calls": 0, "A_to_E_prediction_generated": False,
                        }
                        atomic_json(question_dir / "selector_provenance.json", provenance)
                        atomic_json(question_dir / "stage_b_timings.json", timing)
                        atomic_json(question_dir / "selector_output.json", {
                            "schema_version": 1, "qa_uid": record["qa_uid"], "video_id": record["video_id"],
                            "method": METHOD, "status": "ok", "selected_frames": selected,
                            "actual_frame_count": len(selected), "contains_gold": False,
                            "api_downstream_run": False, "A_to_E_prediction": None,
                        })
                        atomic_json(question_dir / "finished_at.json", {"qa_uid": record["qa_uid"], "video_id": record["video_id"], "status": "ok", "finished_at": utc_now()})
                    succeeded = True
                    break
                except Exception as exc:
                    last_error = exc
                    if fatal_exception(exc):
                        raise
                    if isinstance(exc, ParserFailure):
                        parser_failures += 1
                    atomic_json(question_dir / f"{stage}_attempt{attempt}_failure.json", {"qa_uid": record["qa_uid"], "attempt": attempt, "type": type(exc).__name__, "message": str(exc), "time": utc_now()})
                    if attempt == 1:
                        continue
            if not succeeded:
                task_failures += 1
                atomic_json(question_dir / f"{stage}_failure.json", {"qa_uid": record["qa_uid"], "type": type(last_error).__name__, "message": str(last_error), "retries": 1, "time": utc_now()})
            else:
                completed_count += 1
            atomic_json(status_path, {
                "status": "running", "stage": stage, "worker": worker, "pid": os.getpid(), "pgid": os.getpgrp(),
                "model_load_ms": model_load_ms, "completed": completed_count, "resumed": resumed_count,
                "task_failures": task_failures, "parser_failures": parser_failures, "retries": retries,
                "total": 150, "last_qa_uid": record["qa_uid"], "last_position": position, "updated_at": utc_now(),
            })
            print(json.dumps({"worker": worker, "stage": stage, "position": position, "qa_uid": record["qa_uid"], "completed": completed_count, "success": succeeded}), flush=True)

        model.close()
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        result_status = "complete" if task_failures == 0 and completed_count == 150 else "complete_with_task_failures"
        atomic_json(status_path, {"status": result_status, "stage": stage, "worker": worker, "pid": os.getpid(), "pgid": os.getpgrp(), "model_load_ms": model_load_ms, "completed": completed_count, "resumed": resumed_count, "task_failures": task_failures, "parser_failures": parser_failures, "retries": retries, "total": 150, "finished_at": utc_now()})
        return_code = 0 if task_failures == 0 and completed_count == 150 else 3
    except Exception as exc:
        fatal = {"worker": worker, "stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "time": utc_now()}
        atomic_json(formal_root / "STOP_FATAL.json", fatal)
        atomic_json(formal_root / "status" / f"worker{worker}_{stage}.json", {"status": "fatal", **fatal, "pid": os.getpid(), "pgid": os.getpgrp(), "completed": completed_count, "total": 150})
        return_code = 2
    finally:
        try:
            if model is not None:
                model.close()
            import torch
            summary = resource_summary(monitor, torch, started, audit, stage, worker)
            summary.update({"completed": completed_count, "resumed": resumed_count, "task_failures": task_failures, "parser_failures": parser_failures, "retries": retries, "fatal": fatal})
            atomic_json(formal_root / "telemetry" / f"worker{worker}_{stage}_resource.json", summary)
        except Exception:
            traceback.print_exc()
    return return_code


def validate_stage_a(formal_root: Path) -> int:
    assert_start_environment()
    _, _, _, _, records, _, _ = profile_and_records(False)
    seen = set()
    hashes = {}
    for record in records:
        question_dir = formal_root / "questions" / record["qa_uid"]
        if not stage_a_complete(question_dir, record):
            raise RuntimeError(f"stage A incomplete or invalid: {record['qa_uid']}")
        if record["qa_uid"] in seen:
            raise RuntimeError("duplicate UID during stage A validation")
        seen.add(record["qa_uid"])
        hashes[record["qa_uid"]] = sha256_file(question_dir / "clip_top256.json")
    if len(seen) != 300:
        raise RuntimeError("stage A does not cover exactly 300 UIDs")
    result = {"status": "PASS", "questions": 300, "top256_each": True, "unique_and_sorted_each": True, "uid_to_top256_sha256": hashes, "validated_at": utc_now()}
    atomic_json(formal_root / "stage_a_validation.json", result)
    print(json.dumps({"status": "PASS", "stage": "A", "questions": 300}))
    return 0


def controller(formal_root: Path) -> int:
    assert_start_environment()
    marker = json.loads((formal_root / "FORMAL_RUN.json").read_text(encoding="utf-8"))
    if marker["offline_contract"]["status"] != "35/35_PASS":
        raise RuntimeError("formal preflight contract is not 35/35 PASS")
    started = time.perf_counter()
    atomic_json(formal_root / "status" / "controller.json", {"status": "running", "current_stage": "A_CLIP", "pid": os.getpid(), "pgid": os.getpgrp(), "started_at": utc_now()})
    stage_summaries = []
    for stage in ("clip", "gens"):
        if stage == "gens":
            subprocess.run([str(PYTHON), str(Path(__file__)), "validate-stage-a", "--formal-root", str(formal_root)], check=True, env=os.environ.copy())
        current_stage = "A_CLIP" if stage == "clip" else "B_GENS"
        processes = []
        launch_records = []
        for worker in (0, 1):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(worker)
            env["GENS_PHYSICAL_GPU_INDEX"] = str(worker)
            command = [str(PYTHON), str(Path(__file__)), "worker", "--formal-root", str(formal_root), "--worker", str(worker), "--stage", stage]
            log = (formal_root / "logs" / f"worker{worker}_{stage}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((process, handle))
            launch_records.append({"worker": worker, "pid": process.pid, "pgid": os.getpgid(process.pid), "cuda_visible_devices": str(worker), "command": command, "log": str(log)})
        atomic_json(formal_root / "status" / f"{stage}_worker_processes.json", {"stage": current_stage, "workers": launch_records, "launched_at": utc_now()})
        atomic_json(formal_root / "status" / "controller.json", {"status": "running", "current_stage": current_stage, "pid": os.getpid(), "pgid": os.getpgrp(), "workers": launch_records, "updated_at": utc_now()})
        stage_started = time.perf_counter()
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
        stage_summaries.append({"stage": current_stage, "wall_clock_sec": time.perf_counter() - stage_started, "returncodes": returncodes})
        if returncodes != [0, 0]:
            atomic_json(formal_root / "status" / "controller.json", {"status": "stopped", "current_stage": current_stage, "returncodes": returncodes, "finished_at": utc_now()})
            return 2

    _, _, _, _, records, _, _ = profile_and_records(False)
    finished = sum(final_complete(formal_root / "questions" / record["qa_uid"], record) for record in records)
    resources = []
    for worker in (0, 1):
        for stage in ("clip", "gens"):
            resources.append(json.loads((formal_root / "telemetry" / f"worker{worker}_{stage}_resource.json").read_text(encoding="utf-8")))
    worker_seconds = {str(worker): sum(x["started_elapsed_wall_clock_sec"] for x in resources if x["worker"] == worker) for worker in (0, 1)}
    report = {
        "status": "PASS" if finished == 300 else "FAIL", "finished_questions": finished,
        "overall_wall_clock_sec": time.perf_counter() - started,
        "worker_wall_clock_sec": worker_seconds,
        "gpu_hours": sum(worker_seconds.values()) / 3600.0,
        "stage_summaries": stage_summaries,
        "clip_cold_load_ms": {str(w): json.loads((formal_root / "status" / f"worker{w}_clip.json").read_text(encoding="utf-8"))["model_load_ms"] for w in (0, 1)},
        "gens_cold_load_ms": {str(w): json.loads((formal_root / "status" / f"worker{w}_gens.json").read_text(encoding="utf-8"))["model_load_ms"] for w in (0, 1)},
        "api_calls": sum(x["network_attempts"] for x in resources),
        "private_reads": sum(x["private_read_attempts"] for x in resources),
        "produced_A_to_E_predictions": False,
        "method_disclosure": "GenS-Hybrid-cap16 with symmetric MCQ query adaptation; symmetric multi-query/max aggregation is our fairness adaptation, not the GenS authors' public original implementation.",
        "finished_at": utc_now(),
    }
    atomic_json(formal_root / "FINAL_SELECTOR_REPORT.json", report)
    atomic_json(formal_root / "status" / "controller.json", {"status": "complete", "current_stage": "COMPLETE", "finished_questions": finished, "finished_at": utc_now()})
    return 0 if finished == 300 else 2


def show_status(formal_root: Path) -> int:
    values = {}
    for path in sorted((formal_root / "status").glob("*.json")):
        values[path.name] = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(values, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args.formal_root)
    if args.command == "worker":
        return worker_main(args.formal_root.resolve(), args.worker, args.stage)
    if args.command == "validate-stage-a":
        return validate_stage_a(args.formal_root.resolve())
    if args.command == "controller":
        return controller(args.formal_root.resolve())
    if args.command == "status":
        return show_status(args.formal_root.resolve())
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
