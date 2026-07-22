from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.baselines.egopolice_formal.runner import (
    ROOT,
    atomic_write_json,
    load_formal_contract,
    sha256_file,
    stable_sha256,
    utc_now,
)
from src.diagnostics.cradio_v4.runner import (
    RADIO_ADAPTOR,
    RADIO_CHECKPOINT_FILENAME,
    RADIO_HF_REPO,
    RADIO_HF_REVISION,
    _load_model,
    _to_device,
)

from .constants import (
    CRADIO_RETRIEVAL_SCHEMA_VERSION,
    INDEX_MINIMUM_FREE_VRAM_BYTES,
    cradio_frozen_configuration,
)
from .indexer import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_INDEX_ROOT,
    index_paths,
    validate_index,
)
from .retrieval import build_raw_question_queries, cosine_ranking, select_top8_unique


DEFAULT_OUTPUT_DIR = ROOT / "outputs/experiments/B1/formal_ablation98_v1"
TextEncoderFactory = Callable[..., Any]


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class OfficialSiglip2GTextEncoder:
    def __init__(self, *, cache_root: Path, device_name: str) -> None:
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root / "huggingface")
        os.environ["TORCH_HOME"] = str(cache_root / "torch")
        import torch
        from huggingface_hub import hf_hub_download

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for official siglip2-g text encoding")
        self.torch = torch
        self.device = torch.device(device_name)
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        if free_bytes < INDEX_MINIMUM_FREE_VRAM_BYTES:
            raise RuntimeError(
                f"C-RADIO text encoding requires 12 GiB free VRAM; observed {free_bytes}"
            )
        checkpoint_path = Path(hf_hub_download(
            repo_id=RADIO_HF_REPO, filename=RADIO_CHECKPOINT_FILENAME,
            revision=RADIO_HF_REVISION,
        ))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model, _, self.warnings, self.model_load_latency_sec = _load_model(
            self.device, checkpoint_path
        )
        self.actual_model_loading_mode = "official_cradio_v4_so400m_siglip2g_bfloat16_autocast"

    def encode(self, raw_question: str) -> dict[str, Any]:
        torch = self.torch
        started = time.perf_counter()
        tokenized = _to_device(
            self.model.adaptors[RADIO_ADAPTOR].tokenizer([raw_question]), self.device
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            embedding = self.model.adaptors[RADIO_ADAPTOR].encode_text(
                tokenized, normalize=True
            )
        torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started
        embedding = embedding.float()
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return {
            "embedding": embedding.cpu().numpy()[0],
            "latency_sec": latency,
            "peak_gpu_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)),
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def retrieval_paths(output_dir: Path, question_id: str) -> tuple[Path, Path]:
    if not question_id or any(char in question_id for char in "/\\"):
        raise ValueError(f"Unsafe question ID: {question_id!r}")
    root = output_dir / "retrieval_checkpoints"
    return root / f"{question_id}.json", root / f"{question_id}.ranking.npz"


def retrieval_fingerprint(
    *, query: Any, index_metadata: dict[str, Any], question_manifest_sha256: str,
) -> str:
    return stable_sha256({
        "schema_version": CRADIO_RETRIEVAL_SCHEMA_VERSION,
        "question_manifest_sha256": question_manifest_sha256,
        "question_id": query.question_id,
        "video_id": query.video_id,
        "raw_question": query.raw_question,
        "index_fingerprint": index_metadata["index_fingerprint"],
        "query_definition": "raw_question_exactly",
        "top_k": 8,
        "allocation": "eight_highest_ranked_unique_1fps_frames_no_other_heuristic",
        "configuration": cradio_frozen_configuration(),
    })


def load_index_arrays(index_root: Path, video_id: str) -> dict[str, np.ndarray]:
    artifact_path, _ = index_paths(index_root, video_id)
    with np.load(artifact_path, allow_pickle=False) as arrays:
        embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
        timestamps = np.asarray(arrays["timestamps_sec"], dtype=np.float64)
        frame_indices = np.asarray(arrays["frame_indices"], dtype=np.int64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return {
        "embeddings": embeddings,
        "timestamps_sec": timestamps,
        "frame_indices": frame_indices,
    }


def validate_retrieval_checkpoint(
    *, output_dir: Path, query: Any, index_metadata: dict[str, Any],
    question_manifest_sha256: str,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    json_path, ranking_path = retrieval_paths(output_dir, query.question_id)
    if not json_path.is_file() or not ranking_path.is_file():
        return False, None, "retrieval checkpoint or ranking artifact missing"
    try:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        expected_fingerprint = retrieval_fingerprint(
            query=query, index_metadata=index_metadata,
            question_manifest_sha256=question_manifest_sha256,
        )
        if record.get("schema_version") != CRADIO_RETRIEVAL_SCHEMA_VERSION:
            raise ValueError("schema mismatch")
        if record.get("retrieval_fingerprint") != expected_fingerprint:
            raise ValueError("retrieval fingerprint mismatch")
        if record.get("raw_question_text") != query.raw_question:
            raise ValueError("raw question mismatch")
        if record.get("retrieval_query_text") != query.raw_question:
            raise ValueError("retrieval query is not the exact raw question")
        if record.get("retrieval_query_equals_raw_question") is not True:
            raise ValueError("raw-query equality flag missing")
        if record.get("answer_fields_used_for_retrieval") is not False:
            raise ValueError("answer-field exclusion flag missing")
        if record.get("gt_used_for_ranking") is not False:
            raise ValueError("GT exclusion flag missing")
        if record.get("ranking_artifact_sha256") != sha256_file(ranking_path):
            raise ValueError("ranking artifact hash mismatch")
        with np.load(ranking_path, allow_pickle=False) as arrays:
            similarities = np.asarray(arrays["similarities"], dtype=np.float32)
            ranking = np.asarray(arrays["ranking_indices"], dtype=np.int64)
            timestamps = np.asarray(arrays["timestamps_sec"], dtype=np.float64)
        if (
            similarities.ndim != 1 or ranking.shape != similarities.shape
            or timestamps.shape != similarities.shape
            or set(ranking.tolist()) != set(range(ranking.size))
            or not np.isfinite(similarities).all()
        ):
            raise ValueError("invalid full ranking arrays")
        selected = select_top8_unique(ranking=ranking, timestamps_sec=timestamps)
        expected_top8 = [float(timestamps[index]) for index in selected]
        observed = [float(row["timestamp_sec"]) for row in record.get("top_8_ranked", [])]
        if len(observed) != 8 or observed != expected_top8 or len(set(observed)) != 8:
            raise ValueError("Top-8 selection mismatch")
    except Exception as exc:
        return False, None, str(exc)
    return True, record, None


def _quarantine_retrieval(output_dir: Path, question_id: str, reason: str) -> None:
    json_path, ranking_path = retrieval_paths(output_dir, question_id)
    suffix = f".invalid.{int(time.time())}.{uuid.uuid4().hex[:8]}"
    moved = []
    for path in (json_path, ranking_path):
        if path.exists():
            destination = path.with_name(path.name + suffix)
            os.replace(path, destination)
            moved.append(str(destination))
    atomic_write_json(
        output_dir / "retrieval_quarantine" / f"{question_id}{suffix}.json",
        {"quarantined_at": utc_now(), "reason": reason, "moved": moved},
    )


def write_top1_concentration(
    *, output_dir: Path, records: Sequence[dict[str, Any]], expected_question_count: int,
) -> dict[str, Any]:
    if len(records) != expected_question_count:
        raise ValueError("Top-1 concentration requires all retrieval records")
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)
    rows = []
    sharing = 0
    for video_id, video_rows in sorted(by_video.items()):
        counts = Counter(float(row["top_8_ranked"][0]["timestamp_sec"]) for row in video_rows)
        most_timestamp, most_count = counts.most_common(1)[0]
        sharing += sum(count for count in counts.values() if count > 1)
        rows.append({
            "video_id": video_id,
            "question_count": len(video_rows),
            "unique_top1_timestamp_count": len(counts),
            "most_common_top1_timestamp_sec": most_timestamp,
            "most_common_top1_count": most_count,
            "most_common_top1_share": most_count / len(video_rows),
            "top1_timestamp_counts": {str(key): value for key, value in sorted(counts.items())},
        })
    payload = {
        "schema_version": "egopolice-b1-top1-concentration-v1",
        "diagnostic_only": True,
        "correction_applied": False,
        "question_count": len(records),
        "video_count": len(by_video),
        "questions_whose_top1_is_shared_with_another_question_in_same_video": sharing,
        "rate": sharing / len(records),
        "per_video": rows,
    }
    atomic_write_json(output_dir / "top1_concentration_diagnostic.json", payload)
    return payload


def prepare_retrieval(
    *, data_root: Path, index_root: Path, output_dir: Path, cache_root: Path,
    video_manifest_path: Path, question_manifest_path: Path,
    readiness_path: Path, config_path: Path, device_name: str,
    dry_run: bool, text_encoder_factory: TextEncoderFactory = OfficialSiglip2GTextEncoder,
) -> dict[str, Any]:
    contract = load_formal_contract(
        data_root=data_root, video_manifest_path=video_manifest_path,
        question_manifest_path=question_manifest_path,
        readiness_path=readiness_path, config_path=config_path,
    )
    queries = build_raw_question_queries(contract["questions"])
    if len(queries) != 98 or len({row.question_id for row in queries}) != 98:
        raise ValueError("B1 retrieval must enumerate exactly 98 unique questions")
    if any(query.raw_question != question["question"] for query, question in zip(queries, contract["questions"])):
        raise ValueError("Retrieval query differs from raw question")

    index_metadata: dict[str, dict[str, Any]] = {}
    missing_indexes = []
    for video in contract["videos"].values():
        passed, metadata, reason = validate_index(
            index_root=index_root, video=video,
            video_manifest_sha256=contract["video_manifest_sha256"],
        )
        if passed:
            index_metadata[video["video_id"]] = metadata or {}
        else:
            missing_indexes.append({"video_id": video["video_id"], "reason": reason})
    structural_top8 = all(
        int(np.ceil(contract["videos"][query.video_id]["duration_sec"])) >= 8
        for query in queries
    )
    result = {
        "question_count": len(queries),
        "video_count": len(contract["videos"]),
        "question_count_by_duration_class": dict(Counter(
            row["duration_class"] for row in contract["questions"]
        )),
        "retrieval_query_equals_raw_question_count": sum(
            query.raw_question == question["question"]
            for query, question in zip(queries, contract["questions"])
        ),
        "retrieval_visible_fields": ["question_id", "video_id", "raw_question"],
        "answer_fields_used_for_retrieval": False,
        "gt_used_for_ranking": False,
        "structurally_can_return_8_unique_1fps_frames_for_every_question": structural_top8,
        "valid_index_count": len(index_metadata),
        "missing_index_count": len(missing_indexes),
        "missing_indexes": missing_indexes,
        "dry_run": dry_run,
        "text_encoder_loaded": False,
        "retrievals_completed": 0,
    }
    if dry_run:
        return result
    if missing_indexes:
        raise ValueError(f"Formal retrieval blocked by missing indexes: {missing_indexes}")

    valid_records = []
    pending = []
    invalid = []
    for query in queries:
        passed, record, reason = validate_retrieval_checkpoint(
            output_dir=output_dir, query=query,
            index_metadata=index_metadata[query.video_id],
            question_manifest_sha256=contract["question_manifest_sha256"],
        )
        if passed:
            valid_records.append(record or {})
        else:
            if reason != "retrieval checkpoint or ranking artifact missing":
                invalid.append({"question_id": query.question_id, "reason": str(reason)})
                _quarantine_retrieval(output_dir, query.question_id, str(reason))
            pending.append(query)
    if not pending:
        concentration = write_top1_concentration(
            output_dir=output_dir, records=valid_records, expected_question_count=98
        )
        return {
            **result, "valid_retrieval_count": 98, "pending_retrieval_count": 0,
            "already_complete": True, "top1_concentration": concentration,
        }

    visual_cache = {
        video_id: load_index_arrays(index_root, video_id) for video_id in index_metadata
    }
    encoder = text_encoder_factory(cache_root=cache_root, device_name=device_name)
    session = {
        "schema_version": "egopolice-b1-retrieval-session-v1",
        "session_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        "started_at": utc_now(),
        "question_manifest_sha256": contract["question_manifest_sha256"],
        "model_load_count": 1,
        "model_load_latency_sec": float(encoder.model_load_latency_sec),
        "actual_model_loading_mode": encoder.actual_model_loading_mode,
        "completed_question_ids": [],
        "status": "running",
    }
    session_path = output_dir / "retrieval_sessions" / f"{session['session_id']}.json"
    atomic_write_json(session_path, session)
    try:
        for query in pending:
            arrays = visual_cache[query.video_id]
            encoded = encoder.encode(query.raw_question)
            search_started = time.perf_counter()
            similarities, ranking = cosine_ranking(
                raw_question_embedding=encoded["embedding"],
                normalized_visual_embeddings=arrays["embeddings"],
            )
            similarity_sec = time.perf_counter() - search_started
            selected = select_top8_unique(
                ranking=ranking, timestamps_sec=arrays["timestamps_sec"]
            )
            json_path, ranking_path = retrieval_paths(output_dir, query.question_id)
            _atomic_npz(
                ranking_path, similarities=similarities,
                ranking_indices=ranking.astype(np.int32),
                timestamps_sec=arrays["timestamps_sec"],
                frame_indices=arrays["frame_indices"],
            )
            record = {
                "schema_version": CRADIO_RETRIEVAL_SCHEMA_VERSION,
                "created_at": utc_now(),
                "question_id": query.question_id,
                "video_id": query.video_id,
                "raw_question_text": query.raw_question,
                "retrieval_query_text": query.raw_question,
                "retrieval_query_equals_raw_question": True,
                "retrieval_visible_fields": ["question_id", "video_id", "raw_question"],
                "answer_fields_used_for_retrieval": False,
                "gt_used_for_ranking": False,
                "query_rewriting_used": False,
                "index_fingerprint": index_metadata[query.video_id]["index_fingerprint"],
                "retrieval_fingerprint": retrieval_fingerprint(
                    query=query, index_metadata=index_metadata[query.video_id],
                    question_manifest_sha256=contract["question_manifest_sha256"],
                ),
                "allocation_rule": "top_8_unique_by_stable_descending_cosine_rank",
                "temporal_nms": False,
                "spacing_or_diversity_heuristic": False,
                "clustering": False,
                "reranking": False,
                "selected_frame_count": 8,
                "top_8_ranked": [
                    {
                        "rank": rank + 1,
                        "index_row": int(index),
                        "timestamp_sec": float(arrays["timestamps_sec"][index]),
                        "source_frame_index": int(arrays["frame_indices"][index]),
                        "cosine_similarity": float(similarities[index]),
                    }
                    for rank, index in enumerate(selected)
                ],
                "model_input_chronological": sorted(
                    [float(arrays["timestamps_sec"][index]) for index in selected]
                ),
                "query_text_embedding_time_sec": float(encoded["latency_sec"]),
                "similarity_search_time_sec": similarity_sec,
                "peak_text_encoder_vram_bytes": int(encoded["peak_gpu_allocated_memory_bytes"]),
                "full_ranking_count": int(ranking.size),
                "ranking_artifact_path": str(ranking_path),
                "ranking_artifact_sha256": sha256_file(ranking_path),
            }
            atomic_write_json(json_path, record)
            passed, _, reason = validate_retrieval_checkpoint(
                output_dir=output_dir, query=query,
                index_metadata=index_metadata[query.video_id],
                question_manifest_sha256=contract["question_manifest_sha256"],
            )
            if not passed:
                raise ValueError(f"Refusing invalid retrieval checkpoint: {reason}")
            valid_records.append(record)
            session["completed_question_ids"].append(query.question_id)
            atomic_write_json(session_path, session)
        session["status"] = "completed"
        session["ended_at"] = utc_now()
        atomic_write_json(session_path, session)
    except Exception as exc:
        session["status"] = "failed"
        session["ended_at"] = utc_now()
        session["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(session_path, session)
        raise
    finally:
        encoder.close()

    all_records = []
    for query in queries:
        passed, record, reason = validate_retrieval_checkpoint(
            output_dir=output_dir, query=query,
            index_metadata=index_metadata[query.video_id],
            question_manifest_sha256=contract["question_manifest_sha256"],
        )
        if not passed:
            raise ValueError(f"Post-retrieval validation failed: {query.question_id}: {reason}")
        all_records.append(record or {})
    concentration = write_top1_concentration(
        output_dir=output_dir, records=all_records, expected_question_count=98
    )
    return {
        **result, "valid_retrieval_count": 98, "pending_retrieval_count": 0,
        "text_encoder_loaded": True, "text_encoder_load_count": 1,
        "retrievals_completed": len(pending), "session_path": str(session_path),
        "top1_concentration": concentration,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw-question-only formal B1 retrieval checkpoints"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation20_v1.json")
    parser.add_argument("--question-manifest", type=Path, default=ROOT / "config/data/egopolice_ablation_questions_v1.json")
    parser.add_argument("--readiness", type=Path, default=ROOT / "outputs/data_audit/egopolice_ablation20_readiness.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines/egopolice_b0.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_retrieval(
        data_root=args.data_root, index_root=args.index_root,
        output_dir=args.output_dir, cache_root=args.cache_root,
        video_manifest_path=args.video_manifest,
        question_manifest_path=args.question_manifest,
        readiness_path=args.readiness, config_path=args.config,
        device_name=args.device, dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
