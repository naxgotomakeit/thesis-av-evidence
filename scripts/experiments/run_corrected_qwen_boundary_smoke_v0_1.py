"""Compare deterministic and corrected-Qwen decisions on ten frozen adjacent boundaries."""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.medium_semantic_abstraction.qwen_local import LocalQwen2VL, create_read_only_model_view
from src.experiments.semantic_coarse_v0_1.experiment import (
    frozen_fluid_loose_frontiers, photometric_transition_diagnostic, posture_state_tokens, write_json,
)


def parse_qwen(raw: str) -> tuple[str, str, str]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return "INVALID", "missing JSON object", "invalid_missing_json"
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "INVALID", "malformed JSON object", "invalid_json"
    decision = str(value.get("decision", "")).strip().upper()
    if decision not in {"MERGE", "STOP"}:
        return "INVALID", "decision is not MERGE or STOP", "invalid_decision"
    reason = " ".join(str(value.get("reason") or "no reason supplied").split())[:240]
    return decision, reason, "parsed"


def main() -> int:
    config = json.loads((ROOT / "config/experiments/semantic_coarse_v0_1.json").read_text(encoding="utf-8"))
    smoke_config = json.loads((ROOT / "config/experiments/semantic_coarse_boundary_smoke_v0_1.json").read_text(encoding="utf-8"))
    output = ROOT / "outputs/experiments/semantic_coarse_v0_1"
    captions = json.loads((output / "medium_captions.json").read_text(encoding="utf-8"))
    caption_map = {(row["video_id"], row["medium_id"]): row for row in captions}
    if len(captions) != 104 or len(caption_map) != 104:
        raise RuntimeError("Corrected boundary smoke requires complete unique 104/104 Medium captions")

    source = config["source_policy"]
    frontiers, _ = frozen_fluid_loose_frontiers(
        metrics_path=ROOT / source["adaptive_metrics"],
        hierarchy_path=ROOT / source["safe_merge_hierarchies"], expected_video_count=4,
    )
    frontier_map = {row["video_id"]: row for row in frontiers}
    selected = smoke_config["boundaries"]
    if len(selected) != int(smoke_config["boundary_count"]) or not 8 <= len(selected) <= 12:
        raise RuntimeError("Smoke boundary count must be frozen within 8-12")
    for item in selected:
        ids = frontier_map[item["video_id"]]["medium_ids"]
        left_index = ids.index(item["left_medium_id"])
        if left_index + 1 >= len(ids) or ids[left_index + 1] != item["right_medium_id"]:
            raise RuntimeError(f"Smoke pair is not adjacent in frozen Fluid Loose frontier: {item}")

    model_view, _ = create_read_only_model_view(
        Path(config["local_model"]["snapshot_path"]), output / "local_model_view"
    )
    qwen = LocalQwen2VL(model_view=model_view, seed=int(config["seed"]))
    sentence_started = time.perf_counter()
    sentence_model = SentenceTransformer(
        config["semantic_coarse"]["caption_semantic_encoder"], local_files_only=True,
        device="cuda" if str(qwen.device) == "cuda" else "cpu",
    )
    sentence_load_sec = time.perf_counter() - sentence_started
    records = []
    photo_config = config["semantic_coarse"]["photometric_veto_bypass"]
    posture_vocabulary = list(config["semantic_coarse"]["posture_state_conflict_veto"]["tokens"])
    for index, item in enumerate(selected, start=1):
        frontier = frontier_map[item["video_id"]]
        nodes = {str(node["node_id"]): node for node in frontier["medium_nodes"]}
        left_node, right_node = nodes[item["left_medium_id"]], nodes[item["right_medium_id"]]
        left = caption_map[(item["video_id"], item["left_medium_id"])]
        right = caption_map[(item["video_id"], item["right_medium_id"])]
        left_vector = np.asarray(left_node["pooled_dinov2"], dtype=np.float32)
        right_vector = np.asarray(right_node["pooled_dinov2"], dtype=np.float32)
        left_vector /= max(float(np.linalg.norm(left_vector)), 1e-12)
        right_vector /= max(float(np.linalg.norm(right_vector)), 1e-12)
        dino_cosine = float(left_vector @ right_vector)
        photometric = photometric_transition_diagnostic(
            frame_root=ROOT / source["sampled_frame_root"], video_id=item["video_id"],
            boundary_time=float(left_node["end"]),
            grayscale_correlation_min=float(photo_config["standardized_grayscale_correlation_min"]),
            mean_luminance_delta_min=float(photo_config["mean_luminance_delta_min"]),
            mean_rgb_delta_min=float(photo_config["mean_rgb_delta_min"]),
        )
        encode_started = time.perf_counter()
        embeddings = sentence_model.encode(
            [left["caption"], right["caption"]], normalize_embeddings=True, show_progress_bar=False
        )
        caption_cosine = float(embeddings[0] @ embeddings[1])
        sentence_encode_sec = time.perf_counter() - encode_started
        dino_veto = (
            dino_cosine < float(config["semantic_coarse"]["strong_visual_transition_veto_cosine"])
            and not photometric["suspected_photometric_only"]
        )
        caption_continuous = caption_cosine >= float(config["semantic_coarse"]["caption_continuity_cosine_min"])
        left_postures = posture_state_tokens(left["caption"], posture_vocabulary)
        right_postures = posture_state_tokens(right["caption"], posture_vocabulary)
        posture_conflict = bool(left_postures and right_postures and set(left_postures).isdisjoint(right_postures))
        deterministic = "MERGE" if caption_continuous and not dino_veto and not posture_conflict else "STOP"
        deterministic_reasons = []
        if not caption_continuous:
            deterministic_reasons.append("caption continuity below threshold")
        if dino_veto:
            deterministic_reasons.append("strong non-photometric DINO transition")
        if posture_conflict:
            deterministic_reasons.append("explicit posture/state conflict")
        if deterministic == "MERGE":
            deterministic_reasons.append("all continuity gates passed")

        ordered = [
            f"FIRST [{left['start']:.1f}-{left['end']:.1f}s]: {left['caption']}",
            f"SECOND [{right['start']:.1f}-{right['end']:.1f}s]: {right['caption']}",
        ]
        raw, qwen_timing = qwen.summarize_text(
            ordered_descriptions=ordered, prompt=smoke_config["qwen_contract_prompt"],
            max_new_tokens=int(smoke_config["qwen_max_new_tokens"]),
        )
        qwen_decision, qwen_reason, parse_status = parse_qwen(raw)
        records.append({
            "smoke_index": index, **item,
            "boundary_time": float(left_node["end"]),
            "left_caption": left["caption"], "right_caption": right["caption"],
            "dino_similarity": dino_cosine,
            "caption_sentence_t5_similarity": caption_cosine,
            "photometric_veto_bypass": bool(photometric["suspected_photometric_only"]),
            "photometric_diagnostic": photometric,
            "left_posture_state_tokens": left_postures,
            "right_posture_state_tokens": right_postures,
            "state_conflict_veto": posture_conflict,
            "deterministic_decision": deterministic,
            "deterministic_reason": "; ".join(deterministic_reasons),
            "qwen_decision": qwen_decision, "qwen_short_reason": qwen_reason,
            "qwen_parse_status": parse_status, "qwen_raw_output": raw,
            "timing": {"sentence_t5_encode_sec": sentence_encode_sec, "qwen": qwen_timing},
        })

    result = {
        "status": "pass" if all(row["qwen_decision"] in {"MERGE", "STOP"} for row in records) else "qwen_output_validation_failure",
        "boundary_count": len(records),
        "coverage_intents": dict(Counter(row["coverage_intent"] for row in records)),
        "deterministic_distribution": dict(Counter(row["deterministic_decision"] for row in records)),
        "qwen_distribution": dict(Counter(row["qwen_decision"] for row in records)),
        "agreement_count": sum(row["deterministic_decision"] == row["qwen_decision"] for row in records),
        "disagreement_count": sum(row["deterministic_decision"] != row["qwen_decision"] for row in records),
        "qwen_boundary_calls": qwen.inference_count,
        "sentence_t5_pair_encodes": len(records),
        "qwen_model_load_sec": qwen.model_load_sec,
        "sentence_t5_load_sec": sentence_load_sec,
        "external_api_calls": 0,
        "generated_coarse": False, "generated_storyline": False, "generated_html": False,
        "discarded_local_qwen_calls_from_reporting_failure": 10,
        "records": records,
    }
    write_json(output / "corrected_qwen_boundary_smoke_v0_1.json", result)
    lines = [
        "# Corrected Qwen boundary contract smoke v0.1", "",
        f"- Boundaries: {len(records)}", f"- Qwen boundary calls: {qwen.inference_count}",
        f"- Deterministic: {result['deterministic_distribution']}", f"- Qwen: {result['qwen_distribution']}",
        f"- Agreement: {result['agreement_count']}/{len(records)}", "",
        "| # | Intent | Time | DINO | Deterministic | Qwen | Qwen reason |", "|---:|---|---:|---:|---|---|---|",
    ]
    for row in records:
        reason = row["qwen_short_reason"].replace("|", "/")
        lines.append(
            f"| {row['smoke_index']} | {row['coverage_intent']} | {row['boundary_time']:.1f}s | "
            f"{row['dino_similarity']:.4f} | {row['deterministic_decision']} | {row['qwen_decision']} | {reason} |"
        )
    (output / "corrected_qwen_boundary_smoke_v0_1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
