from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.medium_semantic_abstraction.qwen_local import LocalQwen2VL, create_read_only_model_view

from .common import load_json, write_json


PROMPT = """The images are ordered samples from one short first-person video interval.
Write one concise factual English caption that preserves the principal visible activity,
human interaction, scene, and event transition. Use exactly these labeled sections in
one line: EVENT: ... | EGO: ... | CONTEXT: ... | UNCERTAIN: ... . Do not infer intent,
identity, ownership, causality, or an action not supported by the images."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    cfg = load_json(root / args.config)
    out = root / cfg["output_root"]
    hierarchy = load_json(out / "shared_hierarchy.json")
    view, files = create_read_only_model_view(
        Path(cfg["qwen"]["snapshot"]), out / cfg["qwen"].get("model_view_dir", "qwen_model_view")
    )
    model = LocalQwen2VL(model_view=view, seed=int(cfg["qwen"]["seed"]))
    rows, timings = [], []
    started = time.perf_counter()
    for position, medium in enumerate(hierarchy["medium_nodes"], 1):
        images = [Path(p) for p in medium["representative_frame_paths"]]
        if len(images) != int(cfg["qwen"]["images_per_medium"]) and position != len(hierarchy["medium_nodes"]):
            raise RuntimeError(f"Unexpected image count for {medium['medium_id']}: {len(images)}")
        raw, timing = model.describe_images(image_paths=images, prompt=PROMPT, max_new_tokens=int(cfg["qwen"]["max_new_tokens"]))
        if not raw:
            raise RuntimeError(f"Invalid Qwen caption for {medium['medium_id']}: {raw!r}")
        rows.append({
            "medium_id": medium["medium_id"], "start_sec": medium["start_sec"], "end_sec": medium["end_sec"],
            "qwen_caption": raw, "caption_source": "direct_local_qwen2_5_vl_7b_three_ordered_frames",
            "source_frame_paths": [str(p) for p in images],
        })
        timings.append({"medium_id": medium["medium_id"], **timing})
        write_json(out / "r3_caption_progress.json", {"completed": position, "total": len(hierarchy["medium_nodes"])})
    total = time.perf_counter() - started
    cost = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_type": model.model_type,
        "model_load_sec": model.model_load_sec,
        "caption_calls": model.inference_count, "medium_count": len(rows), "total_caption_sec": total,
        "input_tokens": sum(row["input_tokens"] for row in timings), "output_tokens": sum(row["output_tokens"] for row in timings),
        "peak_gpu_memory_bytes": max((row["peak_vram_bytes"] for row in timings), default=0),
        "external_api_calls": 0, "discarded_local_caption_calls_before_contract_fix": 1,
        "model_files": files,
    }
    write_json(out / "r3_medium_captions.json", rows)
    write_json(out / "r3_caption_timing.json", timings)
    write_json(out / "r3_caption_cost.json", cost)
    write_json(out / "r3_caption_validation.json", {"mediums": len(rows), "expected": len(hierarchy["medium_nodes"]), "valid": len(rows) == len(hierarchy["medium_nodes"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
