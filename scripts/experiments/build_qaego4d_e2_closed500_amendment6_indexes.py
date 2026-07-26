#!/usr/bin/env python3
"""Build only the frozen Amendment #6 offline E2 indexes for Closed-500.

This runner deliberately has no question encoder, no retrieval call, and no
answer-model import.  It constructs one reusable canonical-clip index per
unique Closed-500 ``clip_uid`` and resumes only from atomically written,
validated clip checkpoints.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "HF_HOME",
    "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/diagnostic_caches/qaego4d_e2_huggingface",
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.experiments.qaego4d_e1.core import atomic_write_json, sha256_file
from src.experiments.qaego4d_e2.core import (
    build_amendment4_fine_events,
    build_amendment4_medium_events,
    select_amendment4_fine_representatives,
)
from src.experiments.qaego4d_e2.pipeline import (
    build_shared_fine_artifact,
    decode_canonical_one_fps_to_cache,
    derive_b1_fine_units,
    derive_b2_medium_units,
)
from src.experiments.qaego4d_e2.retrieval import SharedCRadioAlignedEncoder
from src.thesis_av.visual.dinov2_features import DINOv2FeatureExtractor


MAPPING_DEFAULT = ROOT / "configs/qaego4d/qaego4d_closed_canonical_mapping.json"
MANIFEST_DEFAULT = ROOT / "configs/eval_manifests/qaego4d_closed_eval_ids.json"
CONFIG_DEFAULT = ROOT / "configs/experiments/qaego4d_e2_b1_b2_v2_amendment4.json"
OUTPUT_DEFAULT = ROOT / "outputs/experiments/qaego4d_e2_closed500_amendment6_indexes_v1"
CRADIO_CACHE = Path("/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/diagnostic_caches/cradio_v4")

PROTOCOL_FILES = (
    "THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md",
    "THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_1.md",
    "THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_2.md",
    "THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_3.md",
    "THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_4.md",
    "THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_5.md",
    "THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_6.md",
)
TRUSTED_A6_REUSE_ROOTS = (
    ROOT / "outputs/experiments/qaego4d_e2_amendment6_retrieval_dryrun20_v1",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scope_key(entry: dict[str, Any]) -> str:
    return stable_hash([
        entry["clip_uid"], float(entry["canonical_clip_start_sec"]),
        float(entry["canonical_clip_end_sec"]), entry["video_uid"],
    ])[:20]


def index_storage_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def canonical_entries(mapping_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    mapping = read_json(mapping_path)
    manifest = read_json(manifest_path)
    ids = manifest.get("question_ids") if isinstance(manifest, dict) else None
    if not isinstance(mapping, list) or not isinstance(ids, list) or len(ids) != 500:
        raise RuntimeError("Expected frozen Closed-500 mapping and 500 question IDs")
    by_sample = {str(row.get("sample_id")): row for row in mapping if isinstance(row, dict)}
    if len(by_sample) != len(mapping) or set(ids) != set(by_sample):
        raise RuntimeError("Closed manifest/mapping sample_id mismatch")
    grouped: dict[str, dict[str, Any]] = {}
    for sample_id in ids:  # preserve frozen Closed-500 ordering for representative source IDs.
        row = by_sample[str(sample_id)]
        required = {"clip_uid", "video_uid", "clip_start_sec", "clip_end_sec", "parent_video_path", "context_scope"}
        if not required.issubset(row) or row["context_scope"] != "canonical_clip":
            raise RuntimeError(f"Invalid canonical Closed mapping row: {sample_id}")
        entry = {
            "sample_id": str(sample_id),
            "clip_uid": str(row["clip_uid"]),
            "video_uid": str(row["video_uid"]),
            "canonical_clip_start_sec": float(row["clip_start_sec"]),
            "canonical_clip_end_sec": float(row["clip_end_sec"]),
            "parent_video_path": str(row["parent_video_path"]),
            "context_scope": "canonical_clip",
            "closed_sample_ids": [],
        }
        current = grouped.get(entry["clip_uid"])
        if current is None:
            grouped[entry["clip_uid"]] = entry
            current = entry
        comparable = ("video_uid", "canonical_clip_start_sec", "canonical_clip_end_sec", "parent_video_path", "context_scope")
        if any(current[field] != entry[field] for field in comparable):
            raise RuntimeError(f"clip_uid maps to conflicting canonical scopes: {entry['clip_uid']}")
        current["closed_sample_ids"].append(str(sample_id))
    entries = list(grouped.values())
    if len(entries) != 148:
        raise RuntimeError(f"Expected 148 unique Closed clip_uids, got {len(entries)}")
    if any(not Path(entry["parent_video_path"]).is_file() for entry in entries):
        missing = [entry["parent_video_path"] for entry in entries if not Path(entry["parent_video_path"]).is_file()]
        raise RuntimeError(f"Missing Closed parent videos: {missing[:3]}")
    return entries


def valid_index_root(root: Path) -> bool:
    index_path = root / "amendment4_event_index.json"
    vectors_path = root / "fine_cradio_vectors.npz"
    if not index_path.is_file() or not vectors_path.is_file() or not (root / "frames").is_dir():
        return False
    try:
        payload = read_json(index_path)
        fine, medium = payload.get("fine_events"), payload.get("medium_events")
        vectors = np.load(vectors_path, allow_pickle=False)
        return bool(fine) and bool(medium) and payload.get("caption_active_retrieval") is False and (
            vectors["embeddings"].shape[0] == len(fine)
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def trusted_reuse_index(entries: list[dict[str, Any]]) -> dict[str, Path]:
    wanted = {scope_key(entry): entry for entry in entries}
    found: dict[str, Path] = {}
    for trusted_root in TRUSTED_A6_REUSE_ROOTS:
        provenance = trusted_root / "RUN_PROVENANCE.json"
        if not provenance.is_file():
            continue
        values = read_json(provenance)
        if values.get("protocol_hashes", {}).get("THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_6.md") != sha256_file(ROOT / "THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_6.md"):
            continue
        for index_path in trusted_root.glob("cases/*/*/amendment4_event_index.json"):
            key = index_path.parent.name
            if key in wanted and valid_index_root(index_path.parent):
                found[key] = index_path.parent
    return found


def estimate_storage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    reference = ROOT / "outputs/experiments/qaego4d_e2_amendment6_retrieval_dryrun20_v1/offline_index_summary.json"
    if not reference.is_file():
        return {"method": "unavailable", "estimated_new_bytes": None}
    rows = read_json(reference).get("rows", [])
    frames = sum(int(row["counts"]["one_fps_frames"]) for row in rows)
    bytes_total = sum(int(row["offline_timing"]["offline_storage_bytes"]) for row in rows)
    if frames <= 0 or bytes_total <= 0:
        return {"method": "unavailable", "estimated_new_bytes": None}
    requested_frames = sum(round(float(row["canonical_clip_end_sec"]) - float(row["canonical_clip_start_sec"])) for row in entries)
    return {
        "method": "actual_amendment6_dryrun20_storage_bytes_per_frozen_1fps_frame",
        "reference_clips": len(rows),
        "reference_frames": frames,
        "reference_storage_bytes": bytes_total,
        "bytes_per_frame": bytes_total / frames,
        "estimated_new_1fps_frames": requested_frames,
        "estimated_new_bytes": requested_frames * bytes_total / frames,
    }


def build_one(
    *, entry: dict[str, Any], config: dict[str, Any], output_root: Path,
    config_hash: str, dino: DINOv2FeatureExtractor, radio: SharedCRadioAlignedEncoder,
) -> dict[str, Any]:
    """Run precisely the Amendment #6 shared offline pipeline for one clip."""
    clip_start = float(entry["canonical_clip_start_sec"])
    clip_end = float(entry["canonical_clip_end_sec"])
    key = scope_key(entry)
    root = output_root / "cases" / str(entry["clip_uid"]) / key
    root.mkdir(parents=True, exist_ok=True)

    frame_paths, timestamps, source_indices, decode = decode_canonical_one_fps_to_cache(
        parent_video_path=Path(entry["parent_video_path"]), clip_start_s=clip_start,
        clip_end_s=clip_end, output_dir=root / "frames",
    )
    dino_features, dino_meta = dino.extract_or_load(
        video_id=f"amendment6_{entry['clip_uid']}_{key}", frame_paths=frame_paths,
        timestamps=timestamps, cache_dir=root / "dino_features",
    )
    fine_started = time.perf_counter()
    hierarchy_config = {
        **config["b2"]["safe_merge"],
        "representative_fractions": [0.25, 0.5, 0.75],
        "include_dinov2_medoid": True,
    }
    shared = build_shared_fine_artifact(
        clip_uid=str(entry["clip_uid"]), clip_start_s=clip_start, clip_end_s=clip_end,
        timestamps_parent_s=timestamps, frame_paths=frame_paths, dino_features=dino_features,
        comet_config=config["dinov2_temporal_only"]["comet_style"], hierarchy_config=hierarchy_config,
    )
    fine_segmentation_s = time.perf_counter() - fine_started
    fine_units = derive_b1_fine_units(shared)
    medium_units, hierarchy_trace = derive_b2_medium_units(
        shared_fine=shared, hierarchy_config=hierarchy_config,
        medium_config=config["b2"]["fluid_loose"],
    )

    representative_started = time.perf_counter()
    fine_rep_indices, fine_rep_records = select_amendment4_fine_representatives(
        units=fine_units, timestamps=timestamps, dino_embeddings=dino_features,
    )
    fine_representative_s = time.perf_counter() - representative_started
    rep_images = [Image.open(frame_paths[index]).convert("RGB") for index in fine_rep_indices]
    try:
        fine_cradio, cradio_meta = radio.encode_images(rep_images, batch_size=4)
    finally:
        for image in rep_images:
            image.close()
    fine_events = build_amendment4_fine_events(
        units=fine_units, timestamps=timestamps, source_frame_indices=source_indices,
        dino_embeddings=dino_features, fine_cradio_embeddings=fine_cradio,
    )
    medium_started = time.perf_counter()
    medium_events, parent_by_fine = build_amendment4_medium_events(
        medium_units=medium_units, fine_events=fine_events,
    )
    medium_representation_s = time.perf_counter() - medium_started
    if len(parent_by_fine) != len(fine_events):
        raise RuntimeError("Not every Fine event has exactly one Medium parent")
    if len(fine_cradio) != len(fine_events):
        raise RuntimeError("C-RADIO encoded a timeline other than Fine representatives")
    fine_by_id = {str(event["event_id"]): event for event in fine_events}
    if any(
        list(event["retrieval_representation"]["embedding"])
        != list(fine_by_id[str(event["medium_representative_fine_event_id"])]["retrieval_representation"]["embedding"])
        for event in medium_events
    ):
        raise RuntimeError("Medium retrieval vector was not inherited from a medoid child Fine")
    if not np.all((timestamps >= clip_start) & (timestamps < clip_end)):
        raise RuntimeError("Offline timeline escaped canonical clip")

    index_started = time.perf_counter()
    atomic_write_json(root / "amendment4_event_index.json", {
        "fine_events": fine_events,
        "medium_events": medium_events,
        "parent_medium_by_fine_event": parent_by_fine,
        "legacy_b1_prime": "legacy_retired",
        "caption_active_retrieval": False,
    })
    np.savez_compressed(
        root / "fine_cradio_vectors.npz", embeddings=fine_cradio.astype(np.float16),
        fine_event_ids=np.asarray([row["event_id"] for row in fine_events]),
    )
    index_s = time.perf_counter() - index_started
    timing = {
        "offline_parent_open_time_s": decode["offline_parent_open_time_s"],
        "offline_video_seek_time_s": decode["offline_video_seek_time_s"],
        "offline_video_decode_time_s": decode["offline_video_decode_time_s"],
        "offline_cpu_decode_s": decode["offline_cpu_decode_s"],
        "offline_image_encode_s": decode["offline_image_encode_s"],
        "offline_storage_write_s": decode["offline_storage_write_s"],
        "offline_storage_metadata_s": decode["offline_storage_metadata_s"],
        "offline_dinov2_feature_time_s": dino_meta["extraction_sec"],
        "offline_fine_segmentation_time_s": fine_segmentation_s,
        "offline_safe_merge_time_s": hierarchy_trace["timing"]["offline_safe_merge_time_s"],
        "offline_fluid_loose_time_s": hierarchy_trace["timing"]["offline_fluid_loose_time_s"],
        "offline_medium_finalize_time_s": hierarchy_trace["timing"]["offline_medium_finalize_time_s"],
        "offline_fine_representative_selection_time_s": fine_representative_s,
        "offline_cradio_fine_only_feature_time_s": cradio_meta["visual_inference_time_s"],
        "offline_medium_representation_time_s": medium_representation_s,
        "offline_index_cache_time_s": index_s,
        "offline_gpu_compute_s": dino_meta["extraction_sec"] + cradio_meta["visual_inference_time_s"],
        "offline_wall_latency_s": (
            decode["offline_wall_latency_s"] + dino_meta["extraction_sec"] + fine_segmentation_s
            + hierarchy_trace["timing"]["offline_safe_merge_time_s"]
            + hierarchy_trace["timing"]["offline_fluid_loose_time_s"]
            + hierarchy_trace["timing"]["offline_medium_finalize_time_s"]
            + fine_representative_s + cradio_meta["visual_inference_time_s"]
            + medium_representation_s + index_s
        ),
    }
    return {
        "success": True, "reused": False, "clip_uid": entry["clip_uid"], "video_uid": entry["video_uid"],
        "key": key, "canonical_interval": [clip_start, clip_end], "context_scope": "canonical_clip",
        "representative_closed_sample_id": entry["sample_id"], "closed_question_count": len(entry["closed_sample_ids"]),
        "event_index_path": str(root / "amendment4_event_index.json"),
        "counts": {
            "one_fps_frames": len(frame_paths), "fine_events": len(fine_events),
            "medium_events": len(medium_events), "cradio_visual_images_encoded": len(fine_cradio),
            "cradio_medium_visual_images_encoded": 0,
        },
        "offline_timing": timing, "index_size_bytes": index_storage_bytes(root),
        "config_hash": config_hash,
        "offline_contract": {
            "dino_temporal_event_construction": True,
            "fine_to_safe_merge_to_fluid_loose_to_medium": True,
            "dino_medoid_fine_keyframes": True,
            "cradio_visual_encoded_only_for_fine_keyframes": True,
            "medium_vector_inherits_medoid_child_fine": True,
            "question_encoding_retrieval_answer_model_qa": False,
        },
    }


def checkpoint_valid(path: Path, *, config_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = read_json(path)
        return bool(row.get("success")) and row.get("config_hash") == config_hash and valid_index_root(
            Path(row["event_index_path"]).parent
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def write_summary(output_root: Path, entries: list[dict[str, Any]], *, preflight: dict[str, Any]) -> None:
    rows = []
    for entry in entries:
        cp = output_root / "checkpoints" / "offline_clips" / f"{scope_key(entry)}.json"
        if checkpoint_valid(cp, config_hash=preflight["config_sha256"]):
            rows.append(read_json(cp))
    def total(field: str) -> float:
        return float(sum(float(row["offline_timing"].get(field, 0.0)) for row in rows))
    summary = {
        "purpose": "formal_e2_closed500_amendment6_offline_index_build_only_no_qa",
        "expected_unique_clips": len(entries), "completed_valid_clips": len(rows),
        "remaining_clips": len(entries) - len(rows),
        "completed_closed_questions_covered": sum(int(row["closed_question_count"]) for row in rows),
        "config_hash": preflight["config_sha256"], "protocol_hashes": preflight["protocol_hashes"],
        "counts": {
            "one_fps_frames": sum(int(row["counts"]["one_fps_frames"] or 0) for row in rows),
            "fine_events": sum(int(row["counts"]["fine_events"] or 0) for row in rows),
            "medium_events": sum(int(row["counts"]["medium_events"] or 0) for row in rows),
            "cradio_fine_keyframes": sum(int(row["counts"]["cradio_visual_images_encoded"] or 0) for row in rows),
        },
        "timing_totals_s": {
            "decode": total("offline_video_decode_time_s"), "dino": total("offline_dinov2_feature_time_s"),
            "fine_segmentation": total("offline_fine_segmentation_time_s"),
            "safe_merge": total("offline_safe_merge_time_s"), "fluid_loose": total("offline_fluid_loose_time_s"),
            "medium": total("offline_medium_finalize_time_s"),
            "cradio_fine_only": total("offline_cradio_fine_only_feature_time_s"),
            "index": total("offline_index_cache_time_s"), "wall_latency_sum": total("offline_wall_latency_s"),
        },
        "index_size_bytes_total": sum(int(row["index_size_bytes"]) for row in rows),
        "reused_existing_amendment6_indexes": sum(bool(row.get("reused")) for row in rows),
        "built_new_indexes": sum(not bool(row.get("reused")) for row in rows),
        "preflight": preflight,
    }
    atomic_write_json(output_root / "BUILD_SUMMARY.json", summary)
    atomic_write_json(output_root / "per_clip_build_records.json", {"records": rows})
    report = [
        "# Formal E2 Closed-500 Amendment #6 Offline Index Build",
        "",
        f"- Unique canonical clips: {summary['expected_unique_clips']}",
        f"- Valid completed clips: {summary['completed_valid_clips']}",
        f"- Closed questions covered: {summary['completed_closed_questions_covered']}/500",
        f"- New indexes: {summary['built_new_indexes']}; trusted Amendment #6 reuses: {summary['reused_existing_amendment6_indexes']}",
        f"- Total current index size: {summary['index_size_bytes_total']} bytes",
        "- Scope: offline indexing only. No QA, retrieval, question encoding, or answer model was run.",
    ]
    (output_root / "BUILD_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=MAPPING_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    entries = canonical_entries(args.mapping, args.manifest)
    config = read_json(args.config)
    config_hash = sha256_file(args.config)
    protocol_hashes = {name: sha256_file(ROOT / name) for name in PROTOCOL_FILES}
    if config.get("active_protocol", {}).get("amendment_4_sha256") != protocol_hashes["THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_4.md"]:
        raise RuntimeError("Frozen E2 offline config does not match Amendment #4")
    reuse = trusted_reuse_index(entries)
    missing = [entry for entry in entries if scope_key(entry) not in reuse]
    disk = shutil.disk_usage(args.output_root.parent)
    estimate = estimate_storage(missing)
    preflight = {
        "manifest_path": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "mapping_path": str(args.mapping), "mapping_sha256": sha256_file(args.mapping),
        "config_path": str(args.config), "config_sha256": config_hash,
        "protocol_hashes": protocol_hashes, "unique_clips": len(entries),
        "unique_parent_videos": len({entry["video_uid"] for entry in entries}),
        "trusted_amendment6_reuse_clips": len(reuse), "missing_index_clips": len(missing),
        "disk_available_bytes_before": disk.free, "disk_total_bytes": disk.total,
        "storage_estimate": estimate,
        "no_qa_no_retrieval_no_question_encoder": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "PREFLIGHT.json", preflight)
    atomic_write_json(args.output_root / "FROZEN_CONFIG_SNAPSHOT.json", {
        "source_path": str(args.config), "source_sha256": config_hash, "config": config,
    })
    atomic_write_json(args.output_root / "CLOSED500_UNIQUE_CLIP_MANIFEST.json", {
        "source_manifest_sha256": preflight["manifest_sha256"], "source_mapping_sha256": preflight["mapping_sha256"],
        "unique_clip_count": len(entries), "entries": entries,
    })
    atomic_write_json(args.output_root / "RUN_PROVENANCE.json", {
        "purpose": "formal_e2_closed500_amendment6_offline_index_build_only_no_qa",
        "config_hash": config_hash, "protocol_hashes": protocol_hashes,
        "manifest_hash": preflight["manifest_sha256"], "mapping_hash": preflight["mapping_sha256"],
        "cache_namespace": "amendment6_fine_dino_medoid_cradio_fine_only",
        "no_question_encoder": True, "no_retrieval": True, "no_answer_model": True,
    })
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return 0

    # A prior valid checkpoint always wins. Trusted external A6 indexes are
    # recorded by reference; this preserves their identity without copying.
    for entry in entries:
        key = scope_key(entry)
        cp = args.output_root / "checkpoints" / "offline_clips" / f"{key}.json"
        if checkpoint_valid(cp, config_hash=config_hash):
            continue
        if key in reuse:
            source = reuse[key]
            atomic_write_json(cp, {
                "success": True, "reused": True, "clip_uid": entry["clip_uid"], "video_uid": entry["video_uid"],
                "key": key, "canonical_interval": [entry["canonical_clip_start_sec"], entry["canonical_clip_end_sec"]],
                "context_scope": "canonical_clip", "representative_closed_sample_id": entry["sample_id"],
                "closed_question_count": len(entry["closed_sample_ids"]), "event_index_path": str(source / "amendment4_event_index.json"),
                "counts": {"one_fps_frames": None, "fine_events": len(read_json(source / "amendment4_event_index.json")["fine_events"]),
                           "medium_events": len(read_json(source / "amendment4_event_index.json")["medium_events"]),
                           "cradio_visual_images_encoded": None, "cradio_medium_visual_images_encoded": 0},
                "offline_timing": {}, "index_size_bytes": index_storage_bytes(source), "config_hash": config_hash,
                "reused_from": str(source),
            })

    dino = DINOv2FeatureExtractor(device=args.device, batch_size=32)
    radio = SharedCRadioAlignedEncoder(cache_root=CRADIO_CACHE, device_name=args.device)
    try:
        for position, entry in enumerate(entries, start=1):
            cp = args.output_root / "checkpoints" / "offline_clips" / f"{scope_key(entry)}.json"
            if checkpoint_valid(cp, config_hash=config_hash):
                continue
            started = time.perf_counter()
            row = build_one(
                entry=entry, config=config, output_root=args.output_root, config_hash=config_hash,
                dino=dino, radio=radio,
            )
            row["build_position"] = position
            row["checkpoint_written_after_build_s"] = time.perf_counter() - started
            atomic_write_json(cp, row)
            write_summary(args.output_root, entries, preflight=preflight)
            print(json.dumps({"completed": position, "clip_uid": entry["clip_uid"], "fine": row["counts"]["fine_events"], "medium": row["counts"]["medium_events"]}, sort_keys=True), flush=True)
    finally:
        radio.close()
        del dino
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_summary(args.output_root, entries, preflight=preflight)
    print(json.dumps({"success": True, "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
