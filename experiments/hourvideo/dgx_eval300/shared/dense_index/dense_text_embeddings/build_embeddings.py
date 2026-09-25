#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai import OpenAI

MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
PROTOCOL_REL = "outputs/dense_semantic_hierarchical_beam_b_frozen_20260827T194716Z"
INDEX_REL = "index/work_index/cases"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_nodes(index_root: Path):
    nodes = []
    sources = []
    case_dirs = sorted(p for p in index_root.iterdir() if p.is_dir())
    if len(case_dirs) != 12:
        raise RuntimeError(f"Expected exactly 12 case directories, found {len(case_dirs)}")
    for case in case_dirs:
        video = case.name
        nav_path = case / "r3_2_navigation_map.json"
        cap_path = case / "r3_medium_captions.json"
        hierarchy_path = case / "shared_hierarchy.json"
        nav = load_json(nav_path)
        captions = load_json(cap_path)
        hierarchy = load_json(hierarchy_path)
        medium_by_id = {m["medium_id"]: m for m in hierarchy["medium_nodes"]}
        parent = {}
        for coarse in nav["coarse_regions"]:
            cid = coarse["coarse_id"]
            text = coarse["navigation_summary"].strip()
            if not text:
                raise RuntimeError(f"Empty Coarse summary: {video}/{cid}")
            nodes.append({
                "video_uid": video,
                "level": "coarse",
                "node_id": cid,
                "stable_node_id": f"{video}:coarse:{cid}",
                "parent_node_id": None,
                "start_sec": float(coarse["start_sec"]),
                "end_sec": float(coarse["end_sec"]),
                "text": text,
                "text_sha256": sha256_text(text),
                "source_field": "navigation_summary",
            })
            for mid in coarse["source_medium_ids"]:
                if mid in parent:
                    raise RuntimeError(f"Duplicate Medium parent: {video}/{mid}")
                parent[mid] = cid
        if set(parent) != set(medium_by_id):
            raise RuntimeError(f"Parent mapping mismatch for {video}")
        caption_by_id = {x["medium_id"]: x for x in captions}
        if set(caption_by_id) != set(medium_by_id):
            raise RuntimeError(f"Caption/hierarchy Medium mismatch for {video}")
        for mid in sorted(medium_by_id):
            item = caption_by_id[mid]
            text = item["qwen_caption"].strip()
            if not text:
                raise RuntimeError(f"Empty Medium caption: {video}/{mid}")
            cid = parent[mid]
            nodes.append({
                "video_uid": video,
                "level": "medium",
                "node_id": mid,
                "stable_node_id": f"{video}:medium:{mid}",
                "parent_node_id": cid,
                "parent_stable_node_id": f"{video}:coarse:{cid}",
                "start_sec": float(item["start_sec"]),
                "end_sec": float(item["end_sec"]),
                "text": text,
                "text_sha256": sha256_text(text),
                "source_field": "qwen_caption",
            })
        sources.extend([
            {"path": str(nav_path), "sha256": file_sha(nav_path)},
            {"path": str(cap_path), "sha256": file_sha(cap_path)},
            {"path": str(hierarchy_path), "sha256": file_sha(hierarchy_path)},
        ])
    stable_ids = [n["stable_node_id"] for n in nodes]
    if len(stable_ids) != len(set(stable_ids)):
        raise RuntimeError("Duplicate stable node IDs")
    return nodes, sources


def batch_is_valid(path: Path, expected_nodes) -> bool:
    if not path.exists():
        return False
    try:
        obj = load_json(path)
        if obj["model"] != MODEL or obj["dimensions"] != DIMENSIONS:
            return False
        if obj["stable_node_ids"] != [n["stable_node_id"] for n in expected_nodes]:
            return False
        if obj["text_sha256"] != [n["text_sha256"] for n in expected_nodes]:
            return False
        arr = np.asarray(obj["embeddings"], dtype=np.float32)
        return arr.shape == (len(expected_nodes), DIMENSIONS) and bool(np.isfinite(arr).all())
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-attempts", type=int, default=4)
    args = ap.parse_args()

    exp = args.experiment_root.resolve()
    out = args.output_root.resolve()
    protocol = exp / PROTOCOL_REL
    index_root = exp / INDEX_REL
    if not (protocol / "MANIFEST.sha256").is_file():
        raise RuntimeError("Frozen protocol manifest missing")
    nodes, sources = build_nodes(index_root)
    if sum(n["level"] == "coarse" for n in nodes) != 186 or sum(n["level"] == "medium" for n in nodes) != 838:
        raise RuntimeError("Frozen node counts do not match 186 Coarse + 838 Medium")

    base = os.environ.get("EMBEDDING_API_BASE", "").strip()
    key = os.environ.get("EMBEDDING_API_KEY", "").strip()
    if not base or not key:
        raise RuntimeError("EMBEDDING_API_BASE and EMBEDDING_API_KEY are required")
    client = OpenAI(api_key=key, base_url=base)
    batches_dir = out / "completed_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "api_call_log.jsonl"

    build_identity = {
        "status": "running",
        "started_at_utc": now(),
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "batch_size": args.batch_size,
        "node_count": len(nodes),
        "coarse_count": 186,
        "medium_count": 838,
        "frozen_protocol_manifest_sha256": file_sha(protocol / "MANIFEST.sha256"),
        "source_files": sources,
        "credentials_recorded": False,
    }
    atomic_json(out / "build_state.json", build_identity)

    total_batches = math.ceil(len(nodes) / args.batch_size)
    for batch_no in range(total_batches):
        chunk = nodes[batch_no * args.batch_size : (batch_no + 1) * args.batch_size]
        batch_path = batches_dir / f"batch_{batch_no:04d}.json"
        if batch_is_valid(batch_path, chunk):
            continue
        if batch_path.exists():
            raise RuntimeError(f"Existing batch is invalid; refusing silent overwrite: {batch_path}")
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            call_started = now()
            t0 = time.monotonic()
            try:
                response = client.embeddings.create(
                    model=MODEL,
                    dimensions=DIMENSIONS,
                    input=[n["text"] for n in chunk],
                    encoding_format="float",
                    timeout=120.0,
                )
                vectors = [x.embedding for x in sorted(response.data, key=lambda x: x.index)]
                arr = np.asarray(vectors, dtype=np.float32)
                if arr.shape != (len(chunk), DIMENSIONS) or not np.isfinite(arr).all():
                    raise RuntimeError(f"Invalid response array shape/values: {arr.shape}")
                usage = getattr(response, "usage", None)
                usage_obj = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}
                obj = {
                    "batch_no": batch_no,
                    "model": MODEL,
                    "dimensions": DIMENSIONS,
                    "stable_node_ids": [n["stable_node_id"] for n in chunk],
                    "text_sha256": [n["text_sha256"] for n in chunk],
                    "embeddings": arr.tolist(),
                    "usage": usage_obj,
                    "response_model": getattr(response, "model", None),
                    "created_at_utc": now(),
                }
                atomic_json(batch_path, obj)
                append_jsonl(log_path, {
                    "batch_no": batch_no,
                    "attempt": attempt,
                    "status": "success",
                    "started_at_utc": call_started,
                    "ended_at_utc": now(),
                    "elapsed_sec": time.monotonic() - t0,
                    "item_count": len(chunk),
                    "usage": usage_obj,
                    "response_id": getattr(response, "id", None),
                })
                last_error = None
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                append_jsonl(log_path, {
                    "batch_no": batch_no,
                    "attempt": attempt,
                    "status": "failure",
                    "started_at_utc": call_started,
                    "ended_at_utc": now(),
                    "elapsed_sec": time.monotonic() - t0,
                    "item_count": len(chunk),
                    "error": last_error,
                })
                if attempt < args.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
        if last_error is not None:
            build_identity.update({"status": "failed", "failed_batch": batch_no, "error": last_error, "ended_at_utc": now()})
            atomic_json(out / "build_state.json", build_identity)
            raise RuntimeError(f"Batch {batch_no} failed after {args.max_attempts} attempts: {last_error}")

    embeddings = {}
    usage_input = usage_total = 0
    for batch_no in range(total_batches):
        chunk = nodes[batch_no * args.batch_size : (batch_no + 1) * args.batch_size]
        obj = load_json(batches_dir / f"batch_{batch_no:04d}.json")
        if not batch_is_valid(batches_dir / f"batch_{batch_no:04d}.json", chunk):
            raise RuntimeError(f"Batch validation failed during materialization: {batch_no}")
        usage_input += int(obj.get("usage", {}).get("prompt_tokens", 0) or 0)
        usage_total += int(obj.get("usage", {}).get("total_tokens", 0) or 0)
        for node, vec in zip(chunk, obj["embeddings"]):
            embeddings[node["stable_node_id"]] = vec

    videos = sorted({n["video_uid"] for n in nodes})
    validation_cases = []
    for video in videos:
        case_out = out / "cases" / video
        case_out.mkdir(parents=True, exist_ok=True)
        for level in ("coarse", "medium"):
            selected = [n for n in nodes if n["video_uid"] == video and n["level"] == level]
            arr = np.asarray([embeddings[n["stable_node_id"]] for n in selected], dtype=np.float32)
            npy_path = case_out / f"{level}_text_embedding_3_large.float32.npy"
            np.save(npy_path, arr, allow_pickle=False)
            records = []
            for row, n in enumerate(selected):
                rec = {k: v for k, v in n.items() if k != "text"}
                rec.update({
                    "row_index": row,
                    "model": MODEL,
                    "dimensions": DIMENSIONS,
                    "embedding_artifact": npy_path.name,
                })
                records.append(rec)
            atomic_json(case_out / f"{level}_nodes.json", records)
            validation_cases.append({
                "video_uid": video,
                "level": level,
                "node_count": len(selected),
                "shape": list(arr.shape),
                "finite": bool(np.isfinite(arr).all()),
                "unique_stable_node_ids": len({n["stable_node_id"] for n in selected}) == len(selected),
                "npy_sha256": file_sha(npy_path),
                "metadata_sha256": file_sha(case_out / f"{level}_nodes.json"),
            })

    calls = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            calls = [json.loads(line) for line in f if line.strip()]
    report = {
        "status": "valid",
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "video_count": len(videos),
        "coarse_count": 186,
        "medium_count": 838,
        "total_node_count": len(nodes),
        "success_node_count": len(embeddings),
        "failed_node_count": 0,
        "successful_api_calls": sum(x["status"] == "success" for x in calls),
        "failed_api_attempts": sum(x["status"] == "failure" for x in calls),
        "input_tokens": usage_input,
        "total_tokens": usage_total,
        "all_expected_nodes_present": len(embeddings) == len(nodes),
        "all_dimensions_3072": all(x["shape"][1] == DIMENSIONS for x in validation_cases),
        "all_finite": all(x["finite"] for x in validation_cases),
        "all_ids_unique": all(x["unique_stable_node_ids"] for x in validation_cases),
        "text_sha_match": True,
        "cases": validation_cases,
        "completed_at_utc": now(),
    }
    atomic_json(out / "VALIDATION_REPORT.json", report)
    atomic_json(out / "SOURCE_LINEAGE.json", {
        "frozen_protocol_directory": str(protocol),
        "frozen_protocol_manifest_sha256": file_sha(protocol / "MANIFEST.sha256"),
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "text_fields": {"coarse": "navigation_summary", "medium": "qwen_caption"},
        "source_files": sources,
        "credentials_recorded": False,
    })
    build_identity.update({"status": "completed", "ended_at_utc": now(), "validation_report": "VALIDATION_REPORT.json"})
    atomic_json(out / "build_state.json", build_identity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
