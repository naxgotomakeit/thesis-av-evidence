"""Small, explicitly bounded Medium-to-Coarse boundary contract smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer
from src.experiments.semantic_coarse_v0_1.experiment import (
    frozen_fluid_loose_frontiers, photometric_transition_diagnostic, posture_state_tokens, write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    config = json.loads((ROOT / "config/experiments/semantic_coarse_v0_1.json").read_text(encoding="utf-8"))
    output = ROOT / "outputs/experiments/semantic_coarse_v0_1"
    checkpoint = json.loads((output / "semantic_checkpoint.json").read_text(encoding="utf-8"))
    source = config["source_policy"]
    frontiers, _ = frozen_fluid_loose_frontiers(
        metrics_path=ROOT / source["adaptive_metrics"],
        hierarchy_path=ROOT / source["safe_merge_hierarchies"], expected_video_count=4,
    )
    captions = list(checkpoint["medium"].values())
    available = {str(row["medium_id"]): row for row in captions}
    candidates = []
    for frontier in frontiers:
        nodes = frontier["medium_nodes"]
        for left_node, right_node in zip(nodes, nodes[1:]):
            left_id, right_id = str(left_node["node_id"]), str(right_node["node_id"])
            if left_id not in available or right_id not in available:
                continue
            left_vector = np.asarray(left_node["pooled_dinov2"], dtype=np.float32)
            right_vector = np.asarray(right_node["pooled_dinov2"], dtype=np.float32)
            left_vector /= max(float(np.linalg.norm(left_vector)), 1e-12)
            right_vector /= max(float(np.linalg.norm(right_vector)), 1e-12)
            candidates.append((frontier, left_node, right_node, available[left_id], available[right_id], float(left_vector @ right_vector)))
    candidates.sort(key=lambda row: (row[0]["video_id"], float(row[1]["start"])))
    threshold = float(config["semantic_coarse"]["strong_visual_transition_veto_cosine"])
    high = [row for row in candidates if row[-1] >= threshold]
    low = [row for row in candidates if row[-1] < threshold]
    high_count = min(len(high), max(1, int(args.count) // 2))
    low_count = int(args.count) - high_count
    candidates = sorted(high[:high_count] + low[:low_count], key=lambda row: (row[0]["video_id"], float(row[1]["start"])))
    if len(candidates) != int(args.count):
        raise RuntimeError(f"Only {len(candidates)} captioned adjacent boundaries available")

    load_started = time.perf_counter()
    semantic_model = SentenceTransformer(
        config["semantic_coarse"]["caption_semantic_encoder"],
        local_files_only=True,
        device="cuda",
    )
    semantic_model_load_sec = time.perf_counter() - load_started
    records = []
    veto_threshold = threshold
    photo = config["semantic_coarse"]["photometric_veto_bypass"]
    for frontier, left_node, right_node, left, right, cosine in candidates:
        diagnostic = photometric_transition_diagnostic(
            frame_root=ROOT / source["sampled_frame_root"], video_id=frontier["video_id"],
            boundary_time=float(left_node["end"]),
            grayscale_correlation_min=float(photo["standardized_grayscale_correlation_min"]),
            mean_luminance_delta_min=float(photo["mean_luminance_delta_min"]),
            mean_rgb_delta_min=float(photo["mean_rgb_delta_min"]),
        )
        encode_started = time.perf_counter()
        caption_embeddings = semantic_model.encode(
            [left["caption"], right["caption"]], normalize_embeddings=True, show_progress_bar=False
        )
        caption_cosine = float(caption_embeddings[0] @ caption_embeddings[1])
        encode_sec = time.perf_counter() - encode_started
        dino_veto = cosine < veto_threshold and not diagnostic["suspected_photometric_only"]
        caption_continuous = caption_cosine >= float(config["semantic_coarse"]["caption_continuity_cosine_min"])
        posture_vocabulary = list(config["semantic_coarse"]["posture_state_conflict_veto"]["tokens"])
        left_postures = posture_state_tokens(left["caption"], posture_vocabulary)
        right_postures = posture_state_tokens(right["caption"], posture_vocabulary)
        posture_conflict = bool(left_postures and right_postures and set(left_postures).isdisjoint(right_postures))
        decision = "MERGE" if caption_continuous and not dino_veto and not posture_conflict else "STOP"
        failed = []
        if not caption_continuous:
            failed.append("caption immediate-state continuity")
        if dino_veto:
            failed.append("strong non-photometric DINO transition")
        if posture_conflict:
            failed.append("explicit immediate posture/state conflict")
        raw = decision
        timing = {"caption_encode_sec": encode_sec, "model_inference_sec": 0.0, "total_semantic_generation_sec": encode_sec}
        parsed = {
            "decision": decision,
            "reason": "all conservative continuity gates passed" if decision == "MERGE" else "; ".join(failed),
            "parse_status": "deterministic_conservative_continuity_rule",
        }
        source_name = "caption_sentence_t5_plus_dino_veto"
        records.append({
            "video_id": frontier["video_id"], "left_medium_id": left["medium_id"], "right_medium_id": right["medium_id"],
            "boundary_time": float(left_node["end"]), "left_caption": left["caption"], "right_caption": right["caption"],
            "adjacent_medium_dino_cosine": cosine, "photometric_diagnostic": diagnostic,
            "caption_sentence_t5_cosine": caption_cosine,
            "caption_continuity_threshold": float(config["semantic_coarse"]["caption_continuity_cosine_min"]),
            "left_posture_state_tokens": left_postures,
            "right_posture_state_tokens": right_postures,
            "posture_state_conflict_veto": posture_conflict,
            "decision_source": source_name, "raw_output": raw, "timing": timing, **parsed,
        })
    result = {
        "status": "pass" if len({row["decision"] for row in records}) > 1 else "review_required_single_decision_class",
        "bounded_boundary_count": len(records), "sentence_t5_load_sec": semantic_model_load_sec,
        "local_qwen_boundary_calls": 0, "sentence_t5_pair_encodes": len(records), "external_api_calls": 0,
        "merge_count": sum(row["decision"] == "MERGE" for row in records),
        "stop_count": sum(row["decision"] == "STOP" for row in records),
        "all_merge": all(row["decision"] == "MERGE" for row in records),
        "records": records,
    }
    write_json(output / "boundary_smoke_v0_1.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
