"""Complete only missing frozen Fluid Loose Medium captions; no Coarse work."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.medium_semantic_abstraction.qwen_local import LocalQwen2VL, create_read_only_model_view
from src.experiments.semantic_coarse_v0_1.experiment import frozen_fluid_loose_frontiers, write_json


def main() -> int:
    config = json.loads((ROOT / "config/experiments/semantic_coarse_v0_1.json").read_text(encoding="utf-8"))
    output = ROOT / "outputs/experiments/semantic_coarse_v0_1"
    checkpoint_path = output / "semantic_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    source = config["source_policy"]
    frontiers, _ = frozen_fluid_loose_frontiers(
        metrics_path=ROOT / source["adaptive_metrics"],
        hierarchy_path=ROOT / source["safe_merge_hierarchies"], expected_video_count=4,
    )
    expected = [(frontier["video_id"], medium_id) for frontier in frontiers for medium_id in frontier["medium_ids"]]
    keyframe_rows = json.loads((output / "keyframe_selection.json").read_text(encoding="utf-8"))
    keyframes = {(row["video_id"], row["medium_id"]): row for row in keyframe_rows}
    existing = {(row["video_id"], row["medium_id"]): row for row in checkpoint["medium"].values()}
    missing = [key for key in expected if key not in existing]
    if len(expected) != 104 or len(existing) != 96 or len(missing) != 8:
        raise RuntimeError(f"Expected frozen 104 / saved 96 / missing 8, got {len(expected)} / {len(existing)} / {len(missing)}")
    if any(key not in keyframes for key in missing):
        raise RuntimeError("A missing Medium caption has no frozen keyframe trace")

    model_view, _ = create_read_only_model_view(
        Path(config["local_model"]["snapshot_path"]), output / "local_model_view"
    )
    model = LocalQwen2VL(model_view=model_view, seed=int(config["seed"]))
    created = []
    for video_id, medium_id in missing:
        trace = keyframes[(video_id, medium_id)]
        raw, timing = model.describe_images(
            image_paths=[Path(trace["selected_keyframe"]["image_path"])],
            prompt=config["prompts"]["medium"],
            max_new_tokens=int(config["local_model"]["max_new_tokens_medium"]),
        )
        record = {
            "video_id": video_id, "medium_id": medium_id,
            "start": trace["start"], "end": trace["end"], "duration": trace["duration"],
            "selected_keyframe": trace["selected_keyframe"],
            "caption": " ".join(raw.strip().split()), "raw_output": raw,
            "caption_source": "direct_local_qwen_image", "timing": timing, "keyframe_trace": trace,
        }
        checkpoint["medium"][f"{video_id}|{medium_id}"] = record
        write_json(checkpoint_path, checkpoint)
        existing[(video_id, medium_id)] = record
        created.append(record)

    ordered = [existing[key] for key in expected]
    if len(ordered) != 104 or len({(row["video_id"], row["medium_id"]) for row in ordered}) != 104:
        raise RuntimeError("Completed Medium captions are not unique and complete")
    for frontier in frontiers:
        rows = [row for row in ordered if row["video_id"] == frontier["video_id"]]
        if [row["medium_id"] for row in rows] != frontier["medium_ids"]:
            raise RuntimeError(f"Caption order differs from frozen Fluid Loose frontier: {frontier['video_id']}")
    write_json(output / "medium_captions.json", ordered)
    result = {
        "status": "pass", "expected_medium_count": 104, "saved_medium_caption_count": len(ordered),
        "new_caption_count": len(created), "preserved_existing_caption_count": 96,
        "new_video_ids": sorted({row["video_id"] for row in created}),
        "local_qwen_image_calls_this_completion": model.inference_count,
        "model_load_sec": model.model_load_sec, "external_api_calls": 0,
    }
    write_json(output / "medium_caption_completion.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
