from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def medium_for(index: int, intervals: list[dict[str, Any]]) -> str:
    for row in intervals:
        if float(row["start_sec"]) <= index < float(row["end_sec"]):
            return row["medium_id"]
    raise RuntimeError(f"No Medium interval for frame {index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    job = load(Path(args.job))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = job["detector"]
    checkpoint = Path(cfg["checkpoint"])
    if sha(checkpoint).lower() != cfg["checkpoint_sha256"].lower():
        raise RuntimeError("Detector checkpoint hash mismatch")
    os.environ.setdefault("YOLO_CONFIG_DIR", str((output / "ultralytics_config").resolve()))
    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch

    model_started = time.perf_counter()
    model = YOLO(str(checkpoint))
    model_load_sec = time.perf_counter() - model_started
    frames = [Path(p) for p in job["frames"]]
    kwargs = {
        "device": 0, "imgsz": int(cfg["imgsz"]), "conf": float(cfg["conf"]),
        "iou": float(cfg["iou"]), "max_det": int(cfg["max_det"]), "half": False,
        "augment": False, "agnostic_nms": False, "verbose": False,
        "tracker": cfg["tracker"], "persist": True,
    }
    _ = model.track(source=str(frames[0]), **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    observations = []
    per_frame_timing = []
    names = model.names
    for frame_index, frame in enumerate(frames):
        t0 = time.perf_counter()
        results = model.track(source=str(frame), **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        result = results[0]
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        per_frame_timing.append({"frame_index": frame_index, "timestamp_sec": float(frame_index), "elapsed_sec": elapsed, "detection_count": count})
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            tids = boxes.id.detach().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
            height, width = result.orig_shape
            for local, (bbox, confidence, class_id, track_id) in enumerate(zip(xyxy, confs, classes, tids), 1):
                cid = int(class_id)
                label = str(names[cid] if isinstance(names, dict) else names[cid])
                observations.append({
                    "observation_id": f"D{frame_index:05d}_{local:03d}", "frame_index": frame_index,
                    "timestamp_sec": float(frame_index), "source_frame_path": str(frame),
                    "medium_id": medium_for(frame_index, job["medium_intervals"]),
                    "raw_class_id": cid, "raw_class_label": label, "confidence": float(confidence),
                    "bbox_xyxy_pixels": [float(v) for v in bbox], "image_width": int(width), "image_height": int(height),
                    "local_track_id": None if track_id is None else int(track_id),
                    "association_status": "tracked" if track_id is not None else "untracked",
                })
        if (frame_index + 1) % 100 == 0:
            write(output / "detector_progress.json", {"processed_frames": frame_index + 1, "total_frames": len(frames), "observations": len(observations)})
    total_sec = time.perf_counter() - started
    with (output / "frame_track_observations.jsonl").open("w", encoding="utf-8") as f:
        for row in observations:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    by_medium: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_medium[row["medium_id"]].append(row)
    projections = []
    for interval in job["medium_intervals"]:
        mid = interval["medium_id"]
        rows = by_medium.get(mid, [])
        sampled = int(float(interval["end_sec"]) - float(interval["start_sec"]))
        by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_class[row["raw_class_id"]].append(row)
        stats = []
        for cid, items in by_class.items():
            frames_seen = len({row["frame_index"] for row in items})
            tracks = {row["local_track_id"] for row in items if row["local_track_id"] is not None}
            stats.append({
                "raw_class_id": cid, "raw_class_label": items[0]["raw_class_label"],
                "observed_frame_count": frames_seen, "observed_frame_ratio": frames_seen / max(1, sampled),
                "detection_count": len(items), "local_tracklet_count": len(tracks),
                "mean_confidence": statistics.fmean(row["confidence"] for row in items),
                "max_confidence": max(row["confidence"] for row in items),
            })
        stats.sort(key=lambda x: (-x["observed_frame_ratio"], -x["detection_count"], x["raw_class_id"]))
        selected = stats[:6]
        parts = [f"{sampled} one-second observations."]
        if selected:
            parts.append("Detected: " + "; ".join(f"{s['raw_class_label']} {s['observed_frame_count']}/{sampled} frames, {s['local_tracklet_count']} local tracklets" for s in selected) + ".")
        else:
            parts.append("No checkpoint classes detected above the frozen tracking floor.")
        parts.append("No action, role, identity, ownership, relation, intent, or continuous visibility is inferred.")
        summary = " ".join(parts)
        while len(summary) > 450 and len(selected) > 1:
            selected.pop(); parts[1] = "Detected: " + "; ".join(f"{s['raw_class_label']} {s['observed_frame_count']}/{sampled} frames, {s['local_tracklet_count']} local tracklets" for s in selected) + "."; summary = " ".join(parts)
        if len(summary) > 450:
            raise RuntimeError(f"Projection cannot satisfy 450-character boundary: {mid}")
        projections.append({
            "medium_id": mid, "start_sec": float(interval["start_sec"]), "end_sec": float(interval["end_sec"]),
            "sampled_frame_count": sampled, "detector_summary": summary, "class_statistics": stats,
            "vlm_caption_available": False, "semantic_fields_available": False,
        })
    cost = {
        "implementation": "official Ultralytics YOLOv8x-OIV7 plus official BoT-SORT configuration",
        "ultralytics_version": ultralytics_version, "torch_version": torch.__version__,
        "checkpoint_sha256": sha(checkpoint), "frames": len(frames), "observations": len(observations),
        "model_load_sec": model_load_sec, "inference_sec": total_sec,
        "throughput_fps": len(frames) / total_sec, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "api_calls": 0, "inference_arguments": kwargs,
    }
    write(output / "r1_medium_projection.json", projections)
    write(output / "detector_timing.json", per_frame_timing)
    write(output / "detector_tracking_cost.json", cost)
    write(output / "detector_validation.json", {"processed_frames": len(frames), "expected_frames": len(frames), "mediums": len(projections), "valid": len(projections) == len(job["medium_intervals"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
