#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visual.state_regions import (build_regions, extract_frames, frame_timestamps, save_contact_sheet,
    save_timeline, validate_region_schema, write_regions_json)


def arguments():
    parser = argparse.ArgumentParser(description="Build CLIP visual_state_regions for MVP dev cases.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "visual_mvp.yaml")
    parser.add_argument("--dev-cases", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_inspection" / "dev_cases.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "egosound_manifest.jsonl")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "visual_index")
    return parser.parse_args()


def load_cases(dev_path: Path, manifest_path: Path, priority_ids: list[str], max_cases: int):
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    dev_by_id = {row["case_id"]: row for row in dev}
    manifest_by_id = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["case_id"] in priority_ids:
                manifest_by_id[row["case_id"]] = row
    selected, warnings = [], []
    for case_id in priority_ids:
        if len(selected) >= max_cases:
            break
        if case_id in dev_by_id:
            row, source = dict(dev_by_id[case_id]), "dev_cases"
        elif case_id in manifest_by_id:
            row, source = dict(manifest_by_id[case_id]), "manifest_priority_fallback"
            warnings.append(f"{case_id}:not_in_current_dev_cases_used_manifest_fallback")
        else:
            warnings.append(f"{case_id}:case_not_found")
            continue
        row["selection_source"] = source
        selected.append(row)
    if len(selected) < max_cases:
        for row in dev:
            if len(selected) >= max_cases:
                break
            if row["case_id"] not in {x["case_id"] for x in selected}:
                item = dict(row)
                item["selection_source"] = "dev_cases_fill"
                selected.append(item)
    return selected, warnings


def encode_frames(frame_paths, model, preprocess, device, batch_size):
    batches = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            tensors = [preprocess(Image.open(path).convert("RGB")) for path in frame_paths[start:start + batch_size]]
            batch = torch.stack(tensors).to(device)
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            batches.append(features.float().cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def main():
    args = arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not all(isinstance(case_id, str) for case_id in config.get("priority_case_ids", [])):
        raise SystemExit("All priority_case_ids must be quoted YAML strings")
    import clip
    device_requested = str(config.get("device", "cuda"))
    device = "cuda" if device_requested.startswith("cuda") and torch.cuda.is_available() else "cpu"
    warnings = []
    if device != device_requested:
        warnings.append(f"requested_device_{device_requested}_unavailable_using_{device}")
    cases, case_warnings = load_cases(args.dev_cases, args.manifest, config["priority_case_ids"], int(config["max_cases"]))
    warnings.extend(case_warnings)
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(config["clip_download_root"]))
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(config["dataset_root"])
    summaries = []
    for case in cases:
        video_id = str(case["video_id"])
        output_dir = args.output_root / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = dataset_root / Path(case["video_path"])
        if not video_path.is_file():
            warnings.append(f"{case['case_id']}:missing_video:{video_path}")
            continue
        frame_paths = extract_frames(video_path, output_dir / "frames_1fps", float(config["fps"]), str(config["ffmpeg_path"]))
        if not frame_paths:
            warnings.append(f"{case['case_id']}:no_frames_extracted")
            continue
        duration = float(case["video_duration"])
        timestamps = frame_timestamps(len(frame_paths), float(config["fps"]), duration)
        embeddings = encode_frames(frame_paths, model, preprocess, device, int(config.get("batch_size", 32)))
        np.save(output_dir / "frame_embeddings.npy", embeddings)
        regions, region_embeddings, distances = build_regions(
            timestamps, embeddings, frame_paths, duration, float(config["cosine_distance_threshold"]),
            float(config["min_region_duration_sec"]), output_dir,
        )
        np.save(output_dir / "region_embeddings.npy", region_embeddings)
        schema_errors = [f"{region['region_id']}:{error}" for region in regions for error in validate_region_schema(region)]
        warnings.extend(f"{case['case_id']}:{error}" for error in schema_errors)
        payload = {
            "video_id": video_id, "case_ids": [case["case_id"]], "video_path": case["video_path"],
            "encoder": config["encoder"], "sampling_fps": float(config["fps"]),
            "cosine_distance_threshold": float(config["cosine_distance_threshold"]),
            "min_region_duration_sec": float(config["min_region_duration_sec"]),
            "task_2_question_independent": True,
            "visual_index_inputs": ["video_frames"],
            "visual_index_pipeline": ["video_frames", "clip_image_embeddings", "adjacent_frame_cosine_distances", "visual_state_regions"],
            "excluded_from_visual_index_computation": ["question", "answer", "provided_context", "provided_timestamp"],
            "dataset_provided_qa_reference_interval": {
                "start": case.get("provided_timestamp_start"),
                "end": case.get("provided_timestamp_end"),
                "role": "visualization_overlay_only",
                "used_by_visual_index": False,
                "label": "Dataset-provided QA reference interval (not model prediction; not retrieval result; not gold boundary)"
            },
            "visual_state_region_definition": "A temporally continuous visually stable segment; not a semantic event.",
            "frame_embeddings_path": "frame_embeddings.npy", "region_embeddings_path": "region_embeddings.npy",
            "visual_state_regions": regions,
        }
        write_regions_json(output_dir / "visual_state_regions.json", payload)
        save_timeline(output_dir / "timeline.png", distances, regions, duration,
                      float(config["cosine_distance_threshold"]), case.get("provided_timestamp_start"), case.get("provided_timestamp_end"))
        save_contact_sheet(output_dir / "contact_sheet.jpg", output_dir, regions)
        summaries.append({
            "case_id": case["case_id"], "video_id": video_id, "selection_source": case["selection_source"],
            "video_path": case["video_path"], "extracted_frames": len(frame_paths),
            "visual_state_regions": len(regions), "video_duration": duration,
            "dataset_provided_qa_reference_interval": {
                "start": case.get("provided_timestamp_start"), "end": case.get("provided_timestamp_end"),
                "role": "visualization_overlay_only", "used_by_visual_index": False,
            },
            "output_path": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        })
    summary = {
        "config": config,
        "task_2_question_independent": True,
        "visual_index_pipeline": ["video_frames", "clip_image_embeddings", "adjacent_frame_cosine_distances", "visual_state_regions"],
        "excluded_from_visual_index_computation": ["question", "answer", "provided_context", "provided_timestamp"],
        "qa_reference_interval_policy": "Dataset-provided QA reference interval (not model prediction; not retrieval result; not gold boundary); visualization overlay only.",
        "question_conditioned_retrieval_starts_in": "Task 4",
        "processed_videos": len(summaries), "videos": summaries, "warnings": warnings,
    }
    (args.output_root / "visual_index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Visual MVP index summary", "", "## What Task 2 did", "",
             "Task 2 built a question-independent visual structure index: `video frames → CLIP image embeddings → adjacent-frame cosine distances → visual_state_regions`.",
             "A `visual_state_region` is a temporally continuous visually stable segment, not a semantic event.", "",
             "## What Task 2 did not do", "",
             "The question, answer, provided context, and provided timestamp did not influence frame extraction, embeddings, segmentation boundaries, region pooling, or representative keyframe selection.",
             "Task 2 did not perform audio processing, question-conditioned retrieval, reranking, final QA, or agent reasoning. Question-conditioned retrieval begins only in Task 4.", "",
             "## Why a timestamp appears in the timeline", "",
             "**Dataset-provided QA reference interval (not model prediction; not retrieval result; not gold boundary).**",
             "This interval is retained as separate QA reference metadata and drawn only as a pink visualization overlay for later evaluation. Its overlap with a `visual_state_region` is not a retrieval hit.", "",
             f"Processed videos: {len(summaries)}", "", "| case_id | video_id | frames | visual_state_regions | source | output |",
             "|---|---:|---:|---:|---|---|"]
    for row in summaries:
        lines.append(f"| {row['case_id']} | {row['video_id']} | {row['extracted_frames']} | {row['visual_state_regions']} | {row['selection_source']} | `{row['output_path']}` |")
    lines += ["", "## Warnings", ""] + ([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    (args.output_root / "visual_index_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"processed_videos": len(summaries), "videos": summaries,
                      "summary_json": str(args.output_root / "visual_index_summary.json"),
                      "summary_md": str(args.output_root / "visual_index_summary.md"), "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
