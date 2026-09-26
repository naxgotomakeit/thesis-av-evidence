from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.local_prepare import run_audio


def case_configs(repo: Path, cfg: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    selection = load_json(repo / cfg["selection_manifest"])
    question_by_video = {row["video_uid"]: row for row in selection["questions"]}
    configs: list[tuple[dict[str, Any], Path]] = []
    for video in selection["videos"]:
        uid = video["video_uid"]
        question = question_by_video[uid]
        case = copy.deepcopy(cfg)
        case.update({
            "experiment": cfg["experiment"],
            "video_uid": uid,
            "question_id": question["question_id"],
            "video_path": video["video_path"],
            "frame_dir": video["frame_dir"],
            "has_audio": video["has_audio"],
            "output_root": f"{cfg['output_root']}/cases/{uid}",
            "siglip_npz": f"{cfg['embedding_root']}/siglip/shards/{uid}.npz",
            "siglip_metadata": f"{cfg['embedding_root']}/siglip/shards/{uid}.json",
            "dino_npz": f"{cfg['embedding_root']}/dino_v2/shards/{uid}.npz",
            "dino_metadata": f"{cfg['embedding_root']}/dino_v2/shards/{uid}.json",
            "planned_live_calls": {"organizer": 1, "planner": 2, "sufficiency": 2, "visual_review_max": 10, "final": 2},
        })
        path = repo / cfg["output_root"] / "case_configs" / f"{uid}.json"
        write_json(path, case)
        configs.append((case, path))
    return configs


def prepare_case(repo: Path, case: dict[str, Any]) -> dict[str, Any]:
    out = repo / case["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(case["frame_dir"]).glob("frame_*.jpg"))
    if not frames or [path.name for path in frames] != [f"frame_{i:05d}.jpg" for i in range(len(frames))]:
        raise RuntimeError(f"noncanonical frame cache: {case['video_uid']}")
    sources = [Path(case[key]) for key in ("video_path", "siglip_npz", "siglip_metadata", "dino_npz", "dino_metadata")]
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError(f"missing source for {case['video_uid']}")
    sig = np.load(case["siglip_npz"], allow_pickle=False)
    dino = np.load(case["dino_npz"], allow_pickle=False)
    count = len(frames)
    if sig["embedding"].shape != (count, 768) or dino["embedding"].shape != (count, 384):
        raise RuntimeError(f"embedding/frame shape mismatch: {case['video_uid']}")
    if sig["frame_ids"].tolist() != dino["frame_ids"].tolist() or sig["timestamps_sec"].tolist() != dino["timestamps_sec"].tolist():
        raise RuntimeError(f"embedding identity mismatch: {case['video_uid']}")
    fine_width, medium_width = int(case["fine_duration_sec"]), int(case["medium_duration_sec"])
    fine_nodes: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, count, fine_width), 1):
        end = min(count, start + fine_width)
        center = min(count - 1, (start + end - 1) // 2)
        fine_nodes.append({"fine_id": f"F{index:03d}", "start_sec": float(start), "end_sec": float(end), "timestamp_sec": float(center), "source_frame_path": str(frames[center]), "frame_index": center})
    medium_nodes, pooled = [], []
    for index, start in enumerate(range(0, count, medium_width), 1):
        end = min(count, start + medium_width)
        children = [row for row in fine_nodes if start <= row["start_sec"] < end]
        mid = f"M{index:03d}"
        vector = sig["embedding"][start:end].astype(np.float32).mean(axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        pooled.append(vector)
        medium_nodes.append({"medium_id": mid, "start_sec": float(start), "end_sec": float(end), "duration_sec": float(end-start), "source_fine_ids": [row["fine_id"] for row in children], "representative_frame_paths": [row["source_frame_path"] for row in children], "embedding_ref": {"artifact": "medium_siglip.float32.npy", "row_index": index-1, "dimension": 768}})
        for row in children:
            row["parent_medium_id"] = mid
    np.save(out / "medium_siglip.float32.npy", np.stack(pooled).astype(np.float32), allow_pickle=False)
    safe = load_json(repo / case["safe_question_manifest"])
    questions = [row for row in safe["questions"] if row["question_id"] == case["question_id"]]
    if len(questions) != 1:
        raise RuntimeError(f"question does not resolve: {case['question_id']}")
    hierarchy = {"schema_version": "hourvideo-pilot-hierarchy-v1", "video_uid": case["video_uid"], "duration_sec": float(count), "segmentation_policy": {"fine_fixed_sec": fine_width, "medium_fixed_sec": medium_width, "question_independent": True}, "fine_nodes": fine_nodes, "medium_nodes": medium_nodes}
    write_json(out / "question_input.json", questions[0])
    write_json(out / "shared_hierarchy.json", hierarchy)
    write_json(out / "detector_job.json", {"video_uid": case["video_uid"], "frames": [str(path) for path in frames], "medium_intervals": [{key: row[key] for key in ("medium_id", "start_sec", "end_sec")} for row in medium_nodes], "detector": case["detector"]})
    audit = {"video_uid": case["video_uid"], "frame_count": count, "fine_count": len(fine_nodes), "medium_count": len(medium_nodes), "question_id": case["question_id"], "gold_or_reference_loaded": False, "sources": {path.name: {"path": str(path), "sha256": sha256_file(path)} for path in sources}}
    write_json(out / "source_artifact_audit.json", audit)
    return audit


def audio_case(case: dict[str, Any], out: Path) -> dict[str, Any]:
    if case["has_audio"]:
        return run_audio(case, out)
    result = {"model": None, "device": None, "segment_count": 0, "ffmpeg_extraction_sec": 0.0, "model_load_sec": 0.0, "inference_sec": 0.0, "api_calls": 0, "no_audio_source": True}
    write_json(out / "audio_asr.json", {"video_uid": case["video_uid"], "segments": []})
    write_json(out / "audio_cost.json", result)
    return result


def run_stage(repo: Path, config_path: Path, stage: str, resume: bool = True) -> dict[str, Any]:
    cfg = load_json(config_path)
    configs = case_configs(repo, cfg)
    status: dict[str, Any] = {}
    for position, (case, case_path) in enumerate(configs, 1):
        uid, out = case["video_uid"], repo / case["output_root"]
        started = time.perf_counter()
        if stage == "prepare":
            if not (resume and (out / "shared_hierarchy.json").is_file()): prepare_case(repo, case)
        elif stage == "audio":
            if not (resume and (out / "audio_asr.json").is_file()): audio_case(case, out)
        elif stage == "detector":
            if not (resume and (out / "detector_validation.json").is_file() and load_json(out / "detector_validation.json").get("valid")):
                env = {**os.environ, "PYTHONPATH": str(repo / "src"), "YOLO_CONFIG_DIR": str((out / "ultralytics_runtime").resolve())}
                command = [case["detector"]["python"], "-m", "experiments.hourvideo_r1_av_r3_2_single_video_smoke.detector_runner", "--job", str(out / "detector_job.json"), "--output", str(out)]
                subprocess.run(command, cwd=repo, env=env, check=True)
        elif stage == "captions":
            if not (resume and (out / "r3_caption_validation.json").is_file() and load_json(out / "r3_caption_validation.json").get("valid")):
                rel = case_path.relative_to(repo)
                subprocess.run([sys.executable, "-m", "experiments.hourvideo_r1_av_r3_2_single_video_smoke.caption_runner", "--config", str(rel)], cwd=repo, env={**os.environ, "PYTHONPATH": str(repo / "src")}, check=True)
        else:
            raise ValueError(stage)
        status[uid] = {"position": position, "stage": stage, "completed": True, "elapsed_sec": time.perf_counter()-started}
        write_json(repo / cfg["output_root"] / f"{stage}_progress.json", status)
    return status
