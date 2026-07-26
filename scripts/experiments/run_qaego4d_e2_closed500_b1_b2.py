#!/usr/bin/env python3
"""Formal E2 Closed-500 B1/B2 under the frozen Amendment #6 indexes.

B0 is deliberately *not* inferred here: its Formal E1 Closed Uniform-8
records are an immutable reused baseline.  Retrieval is cached before Qwen is
loaded, then B1 and B2 run in one FP16 HuggingFace Qwen session.  A CUDA OOM or
external-contention failure is checkpointed and terminates the invocation so
that no setting is silently changed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/diagnostic_caches/qaego4d_e2_huggingface")

from src.experiments.qaego4d_e1.core import (  # noqa: E402
    CONTEXT_SCOPE, CanonicalCase, atomic_write_json, build_prompt, load_json,
    sha256_file, stable_hash,
)
from src.experiments.qaego4d_e1.runner import (  # noqa: E402
    DEFAULT_ANNOTATION_ROOT, DEFAULT_CONFIG as E1_CONFIG, _load_all_cases,
    _parse_closed_prediction,
)
from src.experiments.qaego4d_e1_fp16_validation.runner import FP16ValidationModel  # noqa: E402
from src.experiments.qaego4d_e2.core import retrieve_amendment6_b1, retrieve_amendment6_b2  # noqa: E402
from src.experiments.qaego4d_e2.retrieval import SharedCRadioAlignedEncoder  # noqa: E402


FORMAL_CONFIG = ROOT / "configs/experiments/qaego4d_e2_closed500_b1_b2_formal_v1.json"
INDEX_CONFIG = ROOT / "configs/experiments/qaego4d_e2_b1_b2_v2_amendment4.json"
MANIFEST = ROOT / "configs/eval_manifests/qaego4d_closed_eval_ids.json"
INDEX_ROOT = ROOT / "outputs/experiments/qaego4d_e2_closed500_amendment6_indexes_v1"
E1_CLOSED_ROOT = ROOT / "outputs/experiments/qaego4d_e1_closed_fp16_v1"
OUTPUT = ROOT / "outputs/experiments/qaego4d_e2_closed500_b1_b2_amendment6_v1"
MODEL = Path("/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/Qwen2.5-VL-7B-Instruct")
CRADIO_CACHE = Path("/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/diagnostic_caches/cradio_v4")
EXPECTED_INDEX_CONFIG_HASH = "e2818d8797163eb773869c39f5b6a5e0b22dcd9eeef17ca9b6420b6ea5a8aa5b"
EXPECTED_MANIFEST_HASH = "44c461e674edd4645d57eccd673601f869a75f2c52e274fd38d666b4e696579d"
PROTOCOL_FILES = tuple(ROOT / f"THESIS_EXPERIMENT_FREEZE_V2.2{suffix}" for suffix in (
    "_FINAL.md", "_AMENDMENT_1.md", "_AMENDMENT_2.md", "_AMENDMENT_3.md",
    "_AMENDMENT_4.md", "_AMENDMENT_5.md", "_AMENDMENT_6.md",
))
RUN_ID = "qaego4d-e2-closed500-b1-b2-fp16-amendment6-v1"


class ContractError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def scope_key(case: CanonicalCase) -> str:
    return stable_hash([case.clip_uid, float(case.clip_start_sec), float(case.clip_end_sec), case.video_uid])[:20]


def index_path(case: CanonicalCase, root: Path) -> Path:
    return root / "cases" / case.clip_uid / scope_key(case) / "amendment4_event_index.json"


def vectors_path(case: CanonicalCase, root: Path) -> Path:
    return index_path(case, root).with_name("fine_cradio_vectors.npz")


def video_frame_path(case: CanonicalCase, root: Path, timeline_index: int) -> Path:
    return index_path(case, root).parent / "frames" / f"frame_{timeline_index:05d}.jpg"


def interval_distance(timestamp: float, start: float, end: float) -> float:
    return 0.0 if start <= timestamp < end else min(abs(timestamp - start), abs(timestamp - end))


def event_overlaps(event: dict[str, Any], case: CanonicalCase) -> bool:
    if math.isclose(case.evidence_start_sec, case.evidence_end_sec, abs_tol=1e-9):
        return float(event["start_s"]) <= case.evidence_start_sec < float(event["end_s"])
    return float(event["start_s"]) < case.evidence_end_sec and float(event["end_s"]) > case.evidence_start_sec


def frame_hit(case: CanonicalCase, frames: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [float(frame["timestamp_sec"]) for frame in frames]
    if math.isclose(case.evidence_start_sec, case.evidence_end_sec, abs_tol=1e-9):
        distances = [abs(value - case.evidence_start_sec) for value in timestamps]
        nearest = min(distances) if distances else None
        # E2 has no point fallback selection; report proximity, not a fabricated hit.
        return {"definition": "point_gt_proximity_only", "any_hit": False, "count_hit": 0,
                "fraction_hit": 0.0, "nearest_selected_frame_distance_s": nearest}
    inside = [value for value in timestamps if case.evidence_start_sec <= value < case.evidence_end_sec]
    return {"definition": "interval_frame_timestamp_inside_gt", "any_hit": bool(inside),
            "count_hit": len(inside), "fraction_hit": len(inside) / len(frames) if frames else 0.0,
            "nearest_selected_frame_distance_s": min((interval_distance(value, case.evidence_start_sec, case.evidence_end_sec) for value in timestamps), default=None)}


def load_cases_and_contract() -> tuple[list[CanonicalCase], dict[str, Any]]:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_HASH:
        raise ContractError("Closed manifest SHA256 mismatch")
    if sha256_file(INDEX_CONFIG) != EXPECTED_INDEX_CONFIG_HASH:
        raise ContractError("Amendment #6 index config SHA256 mismatch")
    if not MODEL.is_dir():
        raise ContractError(f"Formal Qwen model missing: {MODEL}")
    e1 = read(E1_CONFIG)
    all_cases, manifest_hashes = _load_all_cases(e1, DEFAULT_ANNOTATION_ROOT)
    cases = all_cases["closed"]
    if len(cases) != 500 or len({case.question_id for case in cases}) != 500:
        raise ContractError(f"Expected 500 unique Closed cases, got {len(cases)}")
    if manifest_hashes["closed"] != EXPECTED_MANIFEST_HASH:
        raise ContractError("E1 Closed loader manifest hash does not match frozen manifest")
    formal = read(FORMAL_CONFIG)
    if formal["index_source"]["frozen_config_sha256"] != EXPECTED_INDEX_CONFIG_HASH:
        raise ContractError("Formal E2 config does not reference the frozen Amendment #6 index config")
    protocol_hashes = {path.name: sha256_file(path) for path in PROTOCOL_FILES}
    if len(protocol_hashes) != len(PROTOCOL_FILES):
        raise ContractError("Protocol hash collection failed")
    return cases, {"formal": formal, "e1": e1, "manifest_hash": manifest_hashes["closed"], "protocol_hashes": protocol_hashes}


def audit_index_contract(cases: list[CanonicalCase]) -> dict[str, Any]:
    build = read(INDEX_ROOT / "BUILD_SUMMARY.json")
    if build.get("completed_valid_clips") != 148 or build.get("completed_closed_questions_covered") != 500:
        raise ContractError("Closed Amendment #6 index build is incomplete")
    if build.get("config_hash") != EXPECTED_INDEX_CONFIG_HASH:
        raise ContractError("Index build config hash mismatch")
    missing: list[str] = []
    malformed: list[str] = []
    seen_clip: set[str] = set()
    metadata: dict[str, Any] = {}
    for case in cases:
        if case.clip_uid in seen_clip:
            continue
        seen_clip.add(case.clip_uid)
        event_file, vector_file = index_path(case, INDEX_ROOT), vectors_path(case, INDEX_ROOT)
        if not event_file.is_file() or not vector_file.is_file():
            missing.append(case.clip_uid)
            continue
        try:
            event = read(event_file)
            fine, medium = event["fine_events"], event["medium_events"]
            with np.load(vector_file, allow_pickle=False) as arrays:
                shape = tuple(arrays["embeddings"].shape)
            if not fine or not medium or shape[0] != len(fine) or shape[1] != 1536:
                malformed.append(case.clip_uid)
                continue
            for item in fine:
                frame = item.get("representative_frame", {})
                if not video_frame_path(case, INDEX_ROOT, int(frame["timeline_index"])).is_file():
                    malformed.append(case.clip_uid)
                    break
            metadata[case.clip_uid] = {"fine_count": len(fine), "medium_count": len(medium), "event_index_path": str(event_file)}
        except Exception as error:
            malformed.append(f"{case.clip_uid}: {error!r}")
    if missing or malformed or len(seen_clip) != 148:
        raise ContractError(f"Index integrity failure: unique={len(seen_clip)} missing={missing[:3]} malformed={malformed[:3]}")
    return {"unique_clips": len(seen_clip), "index_metadata": metadata,
            "index_build_summary_sha256": sha256_file(INDEX_ROOT / "BUILD_SUMMARY.json"),
            "index_root": str(INDEX_ROOT), "index_config_hash": EXPECTED_INDEX_CONFIG_HASH}


def audit_b0_reuse(cases: list[CanonicalCase], manifest_hash: str, e1_config: dict[str, Any]) -> dict[str, Any]:
    records = read(E1_CLOSED_ROOT / "records.json")
    rows = [row for row in records if row.get("condition") == "uniform_8" and row.get("task") == "closed" and row.get("success") is True]
    keys = [str(row.get("question_id")) for row in rows]
    expected = [case.question_id for case in cases]
    if len(rows) != 500 or len(set(keys)) != 500 or set(keys) != set(expected):
        raise ContractError(f"E1 B0 reuse mismatch: successful_uniform8={len(rows)}, unique={len(set(keys))}")
    fields = {str(row.get("config_hash")) for row in rows}
    prompts = {str(row.get("prompt_hash")) for row in rows}
    if len(fields) != 1 or any(row.get("manifest_hash") != manifest_hash for row in rows):
        raise ContractError("E1 B0 records have inconsistent config or manifest provenance")
    by_question = {str(row["question_id"]): row for row in rows}
    prompt_mismatch: list[str] = []
    for case in cases:
        prompt, options, _ = build_prompt(
            case, open_template=e1_config["prompts"]["open"], closed_template=e1_config["prompts"]["closed"],
        )
        recorded = by_question[case.question_id]
        if recorded.get("prompt") != prompt or recorded.get("prompt_hash") != stable_hash(prompt) or recorded.get("options") != options:
            prompt_mismatch.append(case.question_id)
    if prompt_mismatch:
        raise ContractError(f"E1 B0 prompt/options drift for {len(prompt_mismatch)} records, e.g. {prompt_mismatch[:3]}")
    return {"path": str((E1_CLOSED_ROOT / "records.json").resolve()), "sha256": sha256_file(E1_CLOSED_ROOT / "records.json"),
            "record_count": len(rows), "config_hashes": sorted(fields), "n_prompt_hashes": len(prompts)}


def _valid_retrieval(record: Any, *, run_hash: str, method: str, case: CanonicalCase) -> bool:
    return bool(isinstance(record, dict) and record.get("success") is True and record.get("run_contract_hash") == run_hash
                and record.get("method") == method and record.get("question_id") == case.question_id
                and record.get("clip_uid") == case.clip_uid and isinstance(record.get("selected_frames"), list)
                and 1 <= len(record["selected_frames"]) <= 8)


def _valid_answer(record: Any, *, run_hash: str, method: str, case: CanonicalCase) -> bool:
    required = {"success", "run_contract_hash", "method", "question_id", "clip_uid", "prediction", "correct", "raw_output", "event_hit", "frame_hit", "answer_model_time_s", "total_latency_s"}
    return bool(isinstance(record, dict) and required <= set(record) and record.get("success") is True
                and record.get("run_contract_hash") == run_hash and record.get("method") == method
                and record.get("question_id") == case.question_id and record.get("clip_uid") == case.clip_uid
                and record.get("answer_parse_valid") is True)


def _path(root: Path, kind: str, method: str, case: CanonicalCase) -> Path:
    return root / kind / method / f"{case.question_id}.json"


def _load_event_index(case: CanonicalCase) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = read(index_path(case, INDEX_ROOT))
    return payload["fine_events"], payload["medium_events"]


def run_retrieval(cases: list[CanonicalCase], *, output: Path, run_hash: str, device: str) -> dict[str, int]:
    existing = 0
    pending = [(case, method) for case in cases for method in ("b1", "b2")
               if not (_path(output, "retrieval_checkpoints", method, case).is_file() and _valid_retrieval(read(_path(output, "retrieval_checkpoints", method, case)), run_hash=run_hash, method=method, case=case))]
    if not pending:
        return {"hits": 1000, "misses": 0}
    encoder = SharedCRadioAlignedEncoder(cache_root=CRADIO_CACHE, device_name=device)
    query_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    try:
        for ordinal, case in enumerate(cases, start=1):
            todo = [method for method in ("b1", "b2") if (case, method) in pending]
            if not todo:
                existing += 2
                continue
            query, qmeta = encoder.encode_question(case.question)
            query_cache[case.question_id] = (query, qmeta)
            fine, medium = _load_event_index(case)
            for method in todo:
                started = time.perf_counter()
                out = retrieve_amendment6_b1(query_embedding=query, fine_events=fine) if method == "b1" else retrieve_amendment6_b2(query_embedding=query, medium_events=medium, fine_events=fine)
                frames = out["final_frames"]
                indices = [int(item["source_frame_index"]) for item in frames]
                if not 1 <= len(frames) <= 8 or len(indices) != len(set(indices)):
                    raise ContractError(f"{method}/{case.question_id}: invalid final <=8 unique frames")
                final_ids = {str(item["event_id"]) for item in frames}
                fine_map = {str(item["event_id"]): item for item in fine}
                relevant = [item for item in fine if event_overlaps(item, case)]
                event_hit_value = any(str(item["event_id"]) in final_ids for item in relevant)
                record = {
                    "schema_version": "qaego4d-e2-closed500-retrieval-v1", "created_at": now(), "success": True,
                    "run_id": RUN_ID, "run_contract_hash": run_hash, "method": method, "question_id": case.question_id,
                    "clip_uid": case.clip_uid, "video_uid": case.video_uid, "context_scope": CONTEXT_SCOPE,
                    "query_definition": "raw_question_only", "question": case.question,
                    "answer_fields_used_for_retrieval": False, "gt_used_for_selection": False,
                    "query_encode_time_s": qmeta["retrieval_time_s"], "retrieval_time_s": time.perf_counter() - started,
                    "selected_frames": frames, "final_selected_fine_event_ids": [str(item["event_id"]) for item in frames],
                    "all_fine_count": len(fine), "event_hit": {"any_hit": event_hit_value, "gt_used_posthoc_only": True},
                    "frame_hit": frame_hit(case, frames),
                    "all_candidate_event_scores": out["ranked_fine_events"] if method == "b1" else None,
                    "stage1_candidate_event_scores": out.get("ranked_medium_events"),
                    "stage2_candidate_event_scores": out.get("ranked_fine_events") if method == "b2" else None,
                    "stage1_total_medium_count": out.get("stage1_total_medium_count"),
                    "stage1_selected_medium_count": out.get("stage1_selected_medium_count"),
                    "stage1_selected_medium_ids_in_rank_order": out.get("stage1_selected_medium_ids_in_rank_order"),
                    "stage1_cumulative_fine_count_after_each_medium": out.get("stage1_cumulative_fine_count_after_each_medium"),
                    "stage2_fine_candidate_count_before_topk": out.get("stage2_fine_candidate_count_before_topk"),
                    "stage1_medium_score_time_s": out.get("stage1_medium_score_time_s"),
                    "stage1_ranking_time_s": out.get("stage1_ranking_time_s"),
                    "stage1_expansion_time_s": out.get("stage1_expansion_time_s"),
                    "stage2_fine_score_time_s": out.get("stage2_fine_score_time_s"),
                    "stage2_ranking_time_s": out.get("stage2_ranking_time_s"),
                    "final_frame_event_metadata": [fine_map[str(item["event_id"])] for item in frames],
                }
                atomic_write_json(_path(output, "retrieval_checkpoints", method, case), record)
    finally:
        encoder.close()
        gc.collect()
    return {"hits": existing, "misses": len(pending)}


def load_images(case: CanonicalCase, frames: list[dict[str, Any]]) -> tuple[list[Image.Image], float]:
    started = time.perf_counter()
    images: list[Image.Image] = []
    try:
        for frame in frames:
            path = video_frame_path(case, INDEX_ROOT, int(frame["timeline_index"]))
            if not path.is_file():
                raise ContractError(f"Missing indexed representative JPEG: {path}")
            images.append(Image.open(path).convert("RGB"))
        return images, time.perf_counter() - started
    except Exception:
        for image in images:
            image.close()
        raise


def run_answers(cases: list[CanonicalCase], *, output: Path, run_hash: str, e1_config: dict[str, Any], device: str) -> dict[str, Any]:
    pending = [(case, method) for method in ("b1", "b2") for case in cases
               if not (_path(output, "answer_checkpoints", method, case).is_file() and _valid_answer(read(_path(output, "answer_checkpoints", method, case)), run_hash=run_hash, method=method, case=case))]
    if not pending:
        return {"cache_hits": 1000, "cache_misses": 0, "model_loaded": False}
    model = FP16ValidationModel(model_path=MODEL, max_pixels=262144, seed=int(e1_config["decoding"]["seed"]), minimum_free_vram_gib=20.0)
    model_meta = {"model_load_time_s": model.model_load_time_s, "pre_model_load_memory": model.pre_model_load_memory, "post_model_load_memory": model.post_model_load_memory}
    try:
        for method in ("b1", "b2"):
            for case in cases:
                checkpoint = _path(output, "answer_checkpoints", method, case)
                if checkpoint.is_file() and _valid_answer(read(checkpoint), run_hash=run_hash, method=method, case=case):
                    continue
                retrieval = read(_path(output, "retrieval_checkpoints", method, case))
                started = time.perf_counter()
                images: list[Image.Image] = []
                try:
                    images, cache_read = load_images(case, retrieval["selected_frames"])
                    prompt, options, gt_index = build_prompt(case, open_template=e1_config["prompts"]["open"], closed_template=e1_config["prompts"]["closed"])
                    answer = model.infer(prompt=prompt, images=images, max_new_tokens=int(e1_config["decoding"]["closed_max_new_tokens"]))
                    prediction = _parse_closed_prediction(answer["raw_output"])
                    if prediction is None:
                        raise ContractError(f"Malformed closed model output {case.question_id}/{method}: {answer['raw_output']!r}")
                    gt_letter = chr(65 + int(gt_index))
                    record = {
                        "schema_version": "qaego4d-e2-closed500-answer-v1", "created_at": now(), "success": True,
                        "run_id": RUN_ID, "run_contract_hash": run_hash, "formal_result": True, "stage": "E2",
                        "method": method, "task": "closed", "question_id": case.question_id, "sample_id": case.question_id,
                        "clip_uid": case.clip_uid, "video_uid": case.video_uid, "context_scope": CONTEXT_SCOPE,
                        "config_hash": stable_hash(e1_config), "index_config_hash": EXPECTED_INDEX_CONFIG_HASH,
                        "manifest_hash": EXPECTED_MANIFEST_HASH, "backend": "huggingface", "dtype": "float16",
                        "model": "Qwen2.5-VL-7B-Instruct", "max_pixels": 262144, "prompt": prompt,
                        "prompt_hash": stable_hash(prompt), "options": options,
                        "selected_frame_timestamps_sec": [float(frame["timestamp_sec"]) for frame in retrieval["selected_frames"]],
                        "selected_frame_indices": [int(frame["source_frame_index"]) for frame in retrieval["selected_frames"]],
                        "selected_frames": retrieval["selected_frames"], "frames_shown": len(images), "frames_decoded_online": 0,
                        "video_decode_time_s": 0.0, "online_cached_frame_read_time_s": cache_read,
                        "query_encode_time_s": retrieval["query_encode_time_s"], "retrieval_time_s": retrieval["retrieval_time_s"],
                        "retrieval_stage1_time_s": sum(float(retrieval.get(key) or 0.0) for key in ("stage1_medium_score_time_s", "stage1_ranking_time_s", "stage1_expansion_time_s")) if method == "b2" else None,
                        "retrieval_stage2_time_s": sum(float(retrieval.get(key) or 0.0) for key in ("stage2_fine_score_time_s", "stage2_ranking_time_s")) if method == "b2" else None,
                        "all_fine_count": retrieval["all_fine_count"], "candidate_reduction_ratio": (1.0 - float(retrieval.get("stage2_fine_candidate_count_before_topk") or retrieval["all_fine_count"]) / float(retrieval["all_fine_count"])) if method == "b2" else 0.0,
                        "selected_medium_count": retrieval.get("stage1_selected_medium_count"), "event_hit": retrieval["event_hit"], "frame_hit": retrieval["frame_hit"],
                        "raw_output": answer["raw_output"], "prediction": prediction, "answer_parse_valid": True,
                        "ground_truth_after_prediction": case.answer, "ground_truth_option_letter_after_prediction": gt_letter,
                        "correct": prediction == gt_letter, "answer_preprocess_time_s": answer["answer_preprocess_time_s"],
                        "answer_model_time_s": answer["answer_model_time_s"], "total_latency_s": time.perf_counter() - started,
                        "text_tokens": answer["text_tokens"], "visual_tokens": answer["visual_tokens"], "total_input_tokens": answer["total_input_tokens"],
                        "output_tokens": answer["output_tokens"], "model_calls": 1, "api_calls": 0,
                        "peak_vram_gib": answer["peak_vram_gib"], "cuda_memory": model.last_inference_telemetry,
                    }
                    atomic_write_json(checkpoint, record)
                except BaseException as error:
                    telemetry = model.last_inference_telemetry or {}
                    failure = {"created_at": now(), "success": False, "run_id": RUN_ID, "run_contract_hash": run_hash,
                               "method": method, "question_id": case.question_id, "clip_uid": case.clip_uid,
                               "error": repr(error), "is_cuda_oom": model.is_cuda_oom(error), "cuda_memory": telemetry}
                    atomic_write_json(_path(output, "failures", method, case), failure)
                    if model.is_cuda_oom(error):
                        model.recover_after_oom()
                    raise RuntimeError(f"E2 safely paused on {method}/{case.question_id}: {error!r}") from error
                finally:
                    for image in images:
                        image.close()
    finally:
        model.close()
        gc.collect()
    return {"cache_hits": 1000 - len(pending), "cache_misses": len(pending), "model_loaded": True, **model_meta}


def stat(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "mean": float(statistics.mean(values)), "median": float(statistics.median(values)), "min": float(min(values)), "max": float(max(values))}


def wilson(success: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0: return [None, None]
    p = success / total; denom = 1 + z*z/total; centre = (p + z*z/(2*total))/denom
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return [centre-margin, centre+margin]


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0: return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5).pvalue)
    except Exception:
        return 1.0


def paired_bootstrap(left: list[bool], right: list[bool], *, seed: int = 20260726, reps: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed); diff = np.asarray(right, dtype=float) - np.asarray(left, dtype=float); n = len(diff)
    draws = np.mean(diff[rng.integers(0, n, size=(reps, n))], axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def paired_comparison(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    amap, bmap = {row["question_id"]: row for row in a}, {row["question_id"]: row for row in b}
    if set(amap) != set(bmap): raise ContractError("Pairwise record keys mismatch")
    aa, bb = [bool(amap[key]["correct"]) for key in amap], [bool(bmap[key]["correct"]) for key in amap]
    b01 = sum(x and not y for x, y in zip(aa, bb)); c10 = sum(not x and y for x, y in zip(aa, bb))
    delta = statistics.mean(np.asarray(bb, float) - np.asarray(aa, float))
    discordance = (b01+c10)/len(aa)
    # Approximate paired McNemar MDE at alpha=.05 / power=.80, reported explicitly.
    mde = (1.959963984540054 + 0.8416212335729143) * math.sqrt(discordance / len(aa))
    return {"n": len(aa), "left_correct_right_wrong": b01, "left_wrong_right_correct": c10,
            "both_correct": sum(x and y for x,y in zip(aa,bb)), "both_wrong": sum(not x and not y for x,y in zip(aa,bb)),
            "accuracy_difference_right_minus_left": float(delta), "mcnemar_exact_p": exact_mcnemar(b01,c10),
            "paired_bootstrap_95ci": paired_bootstrap(aa,bb), "mde_absolute_accuracy_approx": float(mde),
            "mde_definition": "Approximate two-sided alpha=.05, power=.80 paired-McNemar MDE using observed discordance; interpret cautiously."}


def cluster_bootstrap(rows_by_method: dict[str, list[dict[str, Any]]], *, reps: int = 5000) -> dict[str, Any]:
    methods = list(rows_by_method); keys = sorted({row["clip_uid"] for row in rows_by_method[methods[0]]})
    per = {method: defaultdict(list) for method in methods}
    for method in methods:
        for row in rows_by_method[method]: per[method][row["clip_uid"]].append(float(row["correct"]))
    rng=np.random.default_rng(20260726); out={}
    for method in methods:
        vals=np.array([statistics.mean(per[method][key]) for key in keys]); draws=np.mean(vals[rng.integers(0,len(vals),size=(reps,len(vals)))],axis=1)
        out[method]={"n_clusters":len(keys),"mean_clip_accuracy":float(vals.mean()),"cluster_bootstrap_95ci":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]}
    return out


def build_analysis(cases: list[CanonicalCase], *, output: Path, provenance: dict[str, Any]) -> None:
    answer_rows={method: [] for method in ("b1","b2")}
    for method in answer_rows:
        for case in cases:
            path=_path(output,"answer_checkpoints",method,case)
            if not path.is_file() or not _valid_answer(read(path),run_hash=provenance["run_contract_hash"],method=method,case=case):
                raise ContractError(f"Cannot analyze incomplete or invalid {method} checkpoint {case.question_id}")
            answer_rows[method].append(read(path))
    e1=[row for row in read(E1_CLOSED_ROOT/"records.json") if row.get("task")=="closed" and row.get("condition")=="uniform_8" and row.get("success")]
    b0={row["question_id"]: {**row,"method":"b0","prediction":row["parsed_closed_option"],"correct":row["parsed_closed_option"]==row["ground_truth_option_letter_after_prediction"]} for row in e1}
    if set(b0)!={case.question_id for case in cases}: raise ContractError("B0 reuse record mismatch at analysis")
    rows_by_method={"b0":[b0[case.question_id] for case in cases], **answer_rows}
    allrows=[]
    for method, rows in rows_by_method.items():
        allrows.extend(rows)
    write_jsonl_atomic(output/"per_question_results.jsonl", allrows)
    accuracy={}
    for method, rows in rows_by_method.items():
        good=sum(bool(row["correct"]) for row in rows); accuracy[method]={"correct":good,"total":len(rows),"accuracy":good/len(rows),"wilson_95ci":wilson(good,len(rows))}
    pairs={"b0_vs_b1":paired_comparison(rows_by_method["b0"],rows_by_method["b1"]),"b0_vs_b2":paired_comparison(rows_by_method["b0"],rows_by_method["b2"]),"b1_vs_b2":paired_comparison(rows_by_method["b1"],rows_by_method["b2"])}
    build_records=read(INDEX_ROOT/"per_clip_build_records.json")["records"]
    offline={row["clip_uid"]:row for row in build_records}
    efficiency={}
    breakeven=[]
    for method, rows in rows_by_method.items():
        fields=["video_decode_time_s","answer_preprocess_time_s","answer_model_time_s","total_latency_s","retrieval_time_s","online_cached_frame_read_time_s"]
        efficiency[method]={"n":len(rows),"latency":{field:stat([float(row[field]) for row in rows if row.get(field) is not None]) for field in fields},"mean_frames_shown":statistics.mean(float(row["frames_shown"]) for row in rows),"mean_visual_tokens":statistics.mean(float(row["visual_tokens"]) for row in rows),"mean_output_tokens":statistics.mean(float(row["output_tokens"]) for row in rows),"peak_vram_gib":stat([float(row["peak_vram_gib"]) for row in rows if row.get("peak_vram_gib") is not None])}
    by_clip={method:defaultdict(list) for method in rows_by_method}
    for method, rows in rows_by_method.items():
        for row in rows: by_clip[method][row["clip_uid"]].append(row)
    for method in ("b1","b2"):
        for clip, rs in by_clip[method].items():
            base=statistics.mean(float(x["total_latency_s"]) for x in by_clip["b0"][clip]); online=statistics.mean(float(x["total_latency_s"]) for x in rs)
            # B2 index cost includes B1 shared work plus hierarchy; B1 omits hierarchy component.
            timing=offline[clip]["offline_timing"]
            b1cost=sum(float(timing.get(key) or 0) for key in ("offline_parent_open_time_s","offline_video_seek_time_s","offline_cpu_decode_s","offline_image_encode_s","offline_storage_write_s","offline_storage_metadata_s","offline_dinov2_feature_time_s","offline_fine_segmentation_time_s","offline_fine_representative_selection_time_s","offline_cradio_fine_only_feature_time_s","offline_index_cache_time_s"))
            b2cost=b1cost+sum(float(timing.get(key) or 0) for key in ("offline_safe_merge_time_s","offline_fluid_loose_time_s","offline_medium_finalize_time_s","offline_medium_representation_time_s"))
            cost=b1cost if method=="b1" else b2cost; saving=base-online
            breakeven.append({"method":method,"clip_uid":clip,"offline_cost_s":cost,"n_questions":len(rs),"b0_online_s":base,"method_online_s":online,"online_saving_s":saving,"N_star":cost/saving if saving>0 else None})
    candidate=[float(row["candidate_reduction_ratio"]) for row in answer_rows["b2"]]
    summary={"schema_version":"qaego4d-e2-closed500-summary-v1","run_id":RUN_ID,"provenance":provenance,"record_integrity":{"b0_reused":500,"b1":len(answer_rows["b1"]),"b2":len(answer_rows["b2"]),"complete":True},"accuracy":accuracy,"paired_comparisons":pairs,"clip_cluster_bootstrap":cluster_bootstrap(rows_by_method),"efficiency":efficiency,"b2_candidate_reduction_ratio":stat(candidate),"offline_index_build":{"source":str(INDEX_ROOT/"BUILD_REPORT.md"),"per_clip_count":len(build_records),"per_clip_records_sha256":sha256_file(INDEX_ROOT/"per_clip_build_records.json")},"break_even":{"definition":"N*=offline per-clip cost/(B0 online latency-method online latency); finite only if observed online saving is positive.","per_clip":breakeven,"summary":{method:stat([float(x["N_star"]) for x in breakeven if x["method"]==method and x["N_star"] is not None]) for method in ("b1","b2")}},"efficiency40_appendix":{"path":str(ROOT/"outputs/experiments/qaego4d_e2_efficiency40_v1"),"status":"referenced only; inspect separately for exact frozen-contract consistency"}}
    atomic_write_json(output/"summary.json",summary)
    csv="method,correct,total,accuracy,wilson_low,wilson_high\n"+"".join(f"{m},{v['correct']},{v['total']},{v['accuracy']},{v['wilson_95ci'][0]},{v['wilson_95ci'][1]}\n" for m,v in accuracy.items())
    (output/"accuracy_summary.csv").write_text(csv,encoding="utf-8")
    report=["# Formal E2 Closed-500 — B0/B1/B2", "", "B0 is the immutable Formal E1 Closed Uniform-8 baseline; this run generated only B1 and B2 under Amendment #6.","","## Accuracy", "", "| Method | Correct / 500 | Accuracy | Wilson 95% CI |","|---|---:|---:|---:|"]
    report += [f"| {m.upper()} | {v['correct']}/500 | {v['accuracy']:.4%} | [{v['wilson_95ci'][0]:.4%}, {v['wilson_95ci'][1]:.4%}] |" for m,v in accuracy.items()]
    report += ["","## Pairwise tests",""] + [f"- **{name}**: Δ(right−left)={value['accuracy_difference_right_minus_left']:.4%}; McNemar exact p={value['mcnemar_exact_p']:.6g}; bootstrap 95% CI={value['paired_bootstrap_95ci']}; approximate MDE={value['mde_absolute_accuracy_approx']:.4%}." for name,value in pairs.items()]
    report += ["","## Efficiency", "", "Per-query online latency is logged separately from offline index creation. Break-even N* is reported only where B1/B2 were empirically faster than B0.", ""]
    (output/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4)); names=list(accuracy); vals=[accuracy[n]["accuracy"] for n in names]; errs=[[vals[i]-accuracy[n]["wilson_95ci"][0] for i,n in enumerate(names)],[accuracy[n]["wilson_95ci"][1]-vals[i] for i,n in enumerate(names)]]; ax.bar(names,vals,yerr=errs,capsize=4); ax.set_ylim(0,1); ax.set_ylabel("Closed accuracy"); fig.tight_layout(); fig.savefig(output/"figures"/"accuracy_wilson.png",dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4)); comps=["retrieval_time_s","online_cached_frame_read_time_s","answer_preprocess_time_s","answer_model_time_s"]; bottom=np.zeros(3); labels=["B0","B1","B2"]; colors=["#4c78a8","#f58518","#54a24b","#e45756"]
        for colour,field in zip(colors,comps):
            vals=np.array([efficiency[m]["latency"][field]["mean"] or 0 for m in ("b0","b1","b2")]); ax.bar(labels,vals,bottom=bottom,label=field,color=colour); bottom+=vals
        ax.legend(fontsize=7); ax.set_ylabel("mean seconds/query"); fig.tight_layout(); fig.savefig(output/"figures"/"latency_decomposition.png",dpi=160); plt.close(fig)
        finite=[x for x in breakeven if x["N_star"] is not None]; fig,ax=plt.subplots(figsize=(7,4));
        for method in ("b1","b2"):
            values=[x["N_star"] for x in finite if x["method"]==method];
            if values: ax.hist(values,bins=min(25,len(values)),alpha=.6,label=method.upper())
        ax.set_xlabel("break-even queries per clip (N*)"); ax.set_ylabel("clip count"); ax.legend(); fig.tight_layout(); fig.savefig(output/"figures"/"break_even.png",dpi=160); plt.close(fig)
    except Exception as error:
        atomic_write_json(output/"figures"/"FIGURE_ERROR.json",{"error":repr(error)})


def preflight(cases: list[CanonicalCase], *, output: Path, contract: dict[str, Any]) -> dict[str, Any]:
    index=audit_index_contract(cases); b0=audit_b0_reuse(cases,contract["manifest_hash"],contract["e1"])
    try:
        import torch
        gpu={"cuda_available":torch.cuda.is_available()}
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info(); gpu.update({"name":torch.cuda.get_device_name(0),"free_gib":free/1024**3,"total_gib":total/1024**3,"sufficient_for_qwen":free>=20*1024**3})
    except Exception as error: gpu={"error":repr(error)}
    result={"created_at":now(),"run_id":RUN_ID,"question_count":len(cases),"methods_to_infer":["b1","b2"],"expected_new_answer_records":1000,"b0_reuse":b0,"index":index,"gpu":gpu,"formal_config_hash":sha256_file(FORMAL_CONFIG),"e1_answer_config_hash":stable_hash(contract["e1"]),"manifest_hash":contract["manifest_hash"],"protocol_hashes":contract["protocol_hashes"],"host":socket.gethostname()}
    atomic_write_json(output/"PREFLIGHT.json",result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",type=Path,default=OUTPUT); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--preflight-only",action="store_true"); parser.add_argument("--analyze-only",action="store_true"); args=parser.parse_args()
    cases, contract=load_cases_and_contract(); output=args.output_root; output.mkdir(parents=True,exist_ok=True); (output/"figures").mkdir(exist_ok=True)
    frozen_contract={"run_id":RUN_ID,"formal_config_sha256":sha256_file(FORMAL_CONFIG),"e1_answer_config_hash":stable_hash(contract["e1"]),"index_config_sha256":sha256_file(INDEX_CONFIG),"manifest_sha256":contract["manifest_hash"],"protocol_hashes":contract["protocol_hashes"],"index_root":str(INDEX_ROOT.resolve()),"b0_source":str((E1_CLOSED_ROOT/"records.json").resolve()),"model":str(MODEL),"backend":"huggingface","dtype":"float16","max_pixels":262144}
    # The contract intentionally excludes run timestamp: resume must compare
    # only frozen experiment inputs, never a fresh invocation's wall-clock.
    run_hash=stable_hash(frozen_contract); run_contract={**frozen_contract,"run_contract_hash":run_hash,"created_at":now()}
    if (output/"RUN_PROVENANCE.json").exists():
        old=read(output/"RUN_PROVENANCE.json")
        if old.get("run_contract_hash")!=run_hash:
            has_any_checkpoint = any((output / kind).exists() and any((output / kind).rglob("*.json")) for kind in ("retrieval_checkpoints", "answer_checkpoints", "failures"))
            old_frozen = {key: value for key, value in old.items() if key not in {"run_contract_hash", "created_at", "host", "resume_hosts"}}
            if has_any_checkpoint and old_frozen != frozen_contract:
                raise ContractError("Existing formal output has incompatible frozen run contract")
            if has_any_checkpoint and old_frozen == frozen_contract:
                atomic_write_json(output/"RUN_PROVENANCE_PREVIOUS_HOST.json", old)
            # The only possible legacy here is the initial no-inference
            # preflight written before deterministic contract hashing was fixed.
            if not has_any_checkpoint:
                atomic_write_json(output/"RUN_PROVENANCE_INITIAL_PREFLIGHT.json", old)
            atomic_write_json(output/"RUN_PROVENANCE.json",run_contract)
    else:
        atomic_write_json(output/"RUN_PROVENANCE.json",run_contract)
    current = read(output/"RUN_PROVENANCE.json")
    hosts = sorted(set(current.get("resume_hosts", [])) | {current.get("host", ""), socket.gethostname()})
    current["resume_hosts"] = [host for host in hosts if host]
    current["host"] = socket.gethostname()
    atomic_write_json(output/"RUN_PROVENANCE.json", current)
    state=preflight(cases,output=output,contract=contract)
    print(json.dumps({"questions":500,"b0_reused":500,"b1_to_run":500,"b2_to_run":500,"expected_total_formal_rows":1500,"gpu":state["gpu"],"run_id":RUN_ID},sort_keys=True))
    if args.preflight_only: return
    if args.analyze_only: build_analysis(cases,output=output,provenance=run_contract); return
    if not state.get("gpu",{}).get("sufficient_for_qwen"):
        raise ContractError("GPU preflight failed: require >=20 GiB free before Qwen load")
    retrieval=run_retrieval(cases,output=output,run_hash=run_hash,device=args.device); atomic_write_json(output/"RETRIEVAL_PROGRESS.json",retrieval)
    answers=run_answers(cases,output=output,run_hash=run_hash,e1_config=contract["e1"],device=args.device); atomic_write_json(output/"ANSWER_PROGRESS.json",answers)
    build_analysis(cases,output=output,provenance=run_contract)


if __name__ == "__main__":
    main()
