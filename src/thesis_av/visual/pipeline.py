"""Executed selected visual pipeline, ported from frozen runners at 7c5c739."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .comet_style import CometStyleConfig, segment as comet_segment
from .deterministic_coarse import build_groups_with_anti_chaining, normalized, pair_decision
from .dinov2_features import DINOv2FeatureExtractor
from .hierarchy import build_boundary_records, build_fine_nodes, build_safe_hierarchy
from .keyframes import select_video_keyframes
from .medium import build_tree_context, fluid_frontier
from .qwen_local import LocalQwen2VL, create_read_only_model_view
from .sampling import extract_frames_1fps
from .segmentation_schema import make_segmentation
from .semantic_helpers import posture_state_tokens, validate_semantic_coarse


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_local_qwen_exact(
    *, output_dir: Path, config: dict[str, Any]
) -> tuple[LocalQwen2VL, list[dict[str, Any]]]:
    """Use the frozen read-only model-view and one persistent local Qwen instance."""
    model_view, model_files = create_read_only_model_view(
        Path(config["local_model"]["snapshot_path"]), output_dir / "local_model_view"
    )
    return LocalQwen2VL(model_view=model_view, seed=int(config["seed"])), model_files


def load_sentence_t5_exact(*, config: dict[str, Any]) -> Any:
    """Mirror the final deterministic runner's local CUDA Sentence-T5 load."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        config["semantic_coarse"]["caption_semantic_encoder"],
        local_files_only=True,
        device="cuda",
    )


def build_structure_exact(
    *, video_id: str, video_path: Path, video_duration: float, root: Path,
    output_dir: Path, ffmpeg_path: Path, config: dict[str, Any], device: str,
) -> dict[str, Any]:
    """Exact runner sequence: ffmpeg -> DINO -> CoMET -> Safe-Merge -> Fluid Loose."""
    frames, timestamps, frame_meta = extract_frames_1fps(
        video_path, output_dir / "frames" / video_id, duration_sec=video_duration,
        jpeg_quality=int(config["sampling"]["jpeg_quality"]), ffmpeg_path=ffmpeg_path,
    )
    extractor = DINOv2FeatureExtractor(device=device, batch_size=int(config["dinov2"]["batch_size"]))
    dino, dino_meta = extractor.extract_or_load(
        video_id=video_id, frame_paths=frames, timestamps=timestamps,
        cache_dir=output_dir / "feature_cache" / video_id,
    )
    internal = comet_segment(dino, timestamps, CometStyleConfig(**config["comet_style"]))
    fine = make_segmentation(
        video_id=video_id, method="comet_style_dinov2", video_duration=video_duration,
        boundaries=internal["boundaries"], frame_timestamps=timestamps,
    )
    boundary_records = build_boundary_records(fine["segments"], internal)
    relative_frames = [path.relative_to(root).as_posix() for path in frames]
    hierarchy = config["hierarchy"]
    fine_nodes = build_fine_nodes(
        segments=fine["segments"], features=dino, timestamps=timestamps,
        frame_paths=relative_frames, boundary_records=boundary_records,
        representative_fractions=tuple(hierarchy["representative_fractions"]),
        include_medoid=bool(hierarchy["include_dinov2_medoid"]),
    )
    tree, merge_trace = build_safe_hierarchy(
        video_id=video_id, video_duration=video_duration, fine_nodes=fine_nodes,
        boundary_records=boundary_records, timestamps=timestamps, frame_paths=relative_frames,
        features=dino, medium_fraction=float(hierarchy["medium_reference_fraction"]),
        coarse_fraction=float(hierarchy["coarse_reference_fraction"]),
        representative_fractions=tuple(hierarchy["representative_fractions"]),
        include_medoid=bool(hierarchy["include_dinov2_medoid"]),
    )
    context = build_tree_context(tree)
    medium_ids, medium_trace = fluid_frontier(
        context, minimum_q_rank=float(config["medium"]["minimum_q_rank"]),
        maximum_local_drop=float(config["medium"]["maximum_local_drop"]), level="medium",
    )
    nodes = context["nodes"]
    return {
        "video_id": video_id, "video_duration": video_duration, "frames": relative_frames,
        "timestamps": timestamps, "dino": dino, "frame_meta": frame_meta, "dino_meta": dino_meta,
        "fine_segmentation": fine, "comet_internal": internal, "boundary_records": boundary_records,
        "safe_merge_hierarchy": tree, "merge_trace": merge_trace,
        "medium_ids": medium_ids, "medium_nodes": [nodes[item] for item in medium_ids],
        "medium_trace": medium_trace,
    }


def select_keyframes_exact(
    *, structure: dict[str, Any], root: Path, output_dir: Path, config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    tree = copy.deepcopy(structure["safe_merge_hierarchy"])
    tree["cuts"]["medium"]["node_ids"] = list(structure["medium_ids"])
    tree["cuts"]["coarse"]["node_ids"] = [tree["root_id"]]
    records, timing = select_video_keyframes(
        hierarchy=tree, features=structure["dino"], timestamps=structure["timestamps"],
        root=root, output_dir=output_dir, weights=config["keyframe_selection"]["quality_weights"],
    )
    for record in records:
        record.pop("coarse_id", None)
    return records, timing


def caption_mediums_exact(
    *, keyframes: list[dict[str, Any]], model: LocalQwen2VL, config: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for trace in keyframes:
        raw, timing = model.describe_images(
            image_paths=[Path(trace["selected_keyframe"]["image_path"])],
            prompt=config["prompts"]["medium"],
            max_new_tokens=int(config["local_model"]["max_new_tokens_medium"]),
        )
        output.append({
            "video_id": trace["video_id"], "medium_id": trace["medium_id"],
            "start": trace["start"], "end": trace["end"], "duration": trace["duration"],
            "selected_keyframe": trace["selected_keyframe"], "caption": " ".join(raw.strip().split()),
            "raw_output": raw, "caption_source": "direct_local_qwen_image",
            "timing": timing, "keyframe_trace": trace,
        })
    return output


def deterministic_semantic_coarse_exact(
    *, structure: dict[str, Any], medium: list[dict[str, Any]],
    sentence_model: Any, config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact executed deterministic runner. Photometric bypass is intentionally not called."""
    rule = config["semantic_coarse"]
    vectors = sentence_model.encode(
        [row["caption"] for row in medium], normalize_embeddings=True, show_progress_bar=False,
    )
    node_map = {str(node["node_id"]): node for node in structure["medium_nodes"]}
    vocabulary = list(rule["posture_tokens"])
    records = []
    for row, caption_embedding in zip(medium, vectors):
        records.append({
            **row, "caption_embedding": np.asarray(caption_embedding, dtype=np.float32),
            "visual_embedding": normalized(np.asarray(node_map[row["medium_id"]]["pooled_dinov2"], dtype=np.float32)),
            "posture_tokens": posture_state_tokens(row["caption"], vocabulary),
        })
    pairs = []
    for left, right in zip(records, records[1:]):
        caption_similarity = float(left["caption_embedding"] @ right["caption_embedding"])
        visual_similarity = float(left["visual_embedding"] @ right["visual_embedding"])
        decision, failures = pair_decision(
            caption_similarity=caption_similarity, visual_similarity=visual_similarity,
            left_postures=left["posture_tokens"], right_postures=right["posture_tokens"],
            caption_min=float(rule["caption_continuity_cosine_min"]),
            visual_min=float(rule["strong_visual_transition_veto_cosine"]),
        )
        pairs.append({
            "video_id": structure["video_id"], "boundary_time": float(left["end"]),
            "left_medium_id": left["medium_id"], "right_medium_id": right["medium_id"],
            "left_caption": left["caption"], "right_caption": right["caption"],
            "caption_sentence_t5_similarity": caption_similarity, "dino_visual_similarity": visual_similarity,
            "left_posture_state_tokens": left["posture_tokens"], "right_posture_state_tokens": right["posture_tokens"],
            "pair_decision": decision, "failed_gates": failures, "qwen_boundary_calls": 0,
        })
    groups, expansions = build_groups_with_anti_chaining(
        records=records, pair_rows=pairs,
        caption_centroid_min=float(rule["caption_centroid_cosine_min"]),
        visual_centroid_min=float(rule["visual_centroid_cosine_min"]),
    )
    coarse = []
    for index, group in enumerate(groups, 1):
        children = [records[position] for position in group]
        coarse.append({
            "video_id": structure["video_id"], "coarse_id": f"semantic_coarse_{index:04d}",
            "start": float(children[0]["start"]), "end": float(children[-1]["end"]),
            "duration": float(children[-1]["end"] - children[0]["start"]),
            "child_medium_ids": [row["medium_id"] for row in children], "child_count": len(children),
            "child_captions": [row["caption"] for row in children],
        })
    return coarse, pairs, expansions


def summarize_coarse_and_story_exact(
    *, video_id: str, medium: list[dict[str, Any]], coarse: list[dict[str, Any]],
    model: LocalQwen2VL, config: dict[str, Any], frozen_coarse: dict[tuple[str, ...], dict[str, Any]] | None = None,
    frozen_story: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact singleton/input orchestration; frozen Qwen outputs may be reused by identity."""
    summarized = []
    frozen_coarse = frozen_coarse or {}
    for group in coarse:
        key = tuple(group["child_medium_ids"])
        if key in frozen_coarse:
            frozen = frozen_coarse[key]
            identity_fields = (
                "video_id",
                "coarse_id",
                "start",
                "end",
                "duration",
                "child_medium_ids",
                "child_count",
                "child_captions",
            )
            mismatches = [
                field for field in identity_fields if frozen.get(field) != group.get(field)
            ]
            if mismatches:
                raise ValueError(
                    "Frozen Coarse summary input identity mismatch: "
                    + ", ".join(mismatches)
                )
            summarized.append(copy.deepcopy(frozen))
            continue
        if group["child_count"] == 1:
            summarized.append({**group, "summary": group["child_captions"][0], "raw_output": group["child_captions"][0], "summary_source": "singleton_copy_through", "timing": {"image_count": 0, "image_decode_sec": 0.0, "image_preprocess_sec": 0.0, "model_inference_sec": 0.0, "total_semantic_generation_sec": 0.0, "input_tokens": 0, "output_tokens": 0, "peak_vram_bytes": 0}})
        else:
            descriptions = [f"[{row['start']:.1f}-{row['end']:.1f}s] {row['caption']}" for row in medium if row["medium_id"] in group["child_medium_ids"]]
            raw, timing = model.summarize_text(ordered_descriptions=descriptions, prompt=config["prompts"]["coarse"], max_new_tokens=int(config["local_model"]["max_new_tokens_coarse"]))
            summarized.append({**group, "summary": " ".join(raw.strip().split()), "raw_output": raw, "summary_source": "local_qwen_text_only", "timing": timing})
    story_inputs = [f"[{row['start']:.1f}-{row['end']:.1f}s] {row['summary']}" for row in summarized]
    if frozen_story is not None:
        if frozen_story["coarse_inputs"] != story_inputs: raise ValueError("Frozen storyline inputs do not match reconstructed Semantic Coarse summaries")
        story = copy.deepcopy(frozen_story)
    else:
        raw, timing = model.summarize_text(ordered_descriptions=story_inputs, prompt=config["prompts"]["story"], max_new_tokens=int(config["local_model"]["max_new_tokens_story"]))
        story = {"video_id": video_id, "coarse_inputs": story_inputs, "story": raw.strip(), "raw_output": raw, "timing": timing}
    return summarized, story
