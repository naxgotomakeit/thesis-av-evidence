from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .core import posthoc_gt_interval_metrics, rank_timestamps
from .runner import (
    RADIO_ADAPTOR,
    RADIO_CHECKPOINT_FILENAME,
    RADIO_GITHUB_REVISION,
    RADIO_HF_REPO,
    RADIO_HF_REVISION,
    SIGLIP2_TEXT_MODEL,
    _load_model,
    _sha256,
    _to_device,
)


VIDEO_ID = "pasadena/YKI08"
QUERY_TEMPLATE = "Question: {question}\nCandidate answer: {option}"
K_VALUES = (1, 5, 8, 10, 20)


def is_concrete_option(option: str) -> bool:
    return option.strip().casefold().rstrip(".") != "none of the above"


def build_option_query(question: str, option: str) -> str:
    return QUERY_TEMPLATE.format(question=question.strip(), option=option.strip())


def load_query_inputs(manifest_path: Path) -> list[dict[str, Any]]:
    """Load retrieval-visible fields only; deliberately excludes GT fields."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in payload["questions"]:
        if item.get("video_id") != VIDEO_ID:
            continue
        options = [str(value) for value in item["options"]]
        if len(options) != 5:
            raise ValueError(f"Expected five options for {item['question_id']}")
        rows.append(
            {
                "question_id": str(item["question_id"]),
                "duration_class": str(item["duration_class"]),
                "question": str(item["question"]),
                "options": options,
            }
        )
    if not rows:
        raise ValueError(f"No frozen questions found for {VIDEO_ID}")
    return rows


def load_posthoc_ground_truth(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load GT only after the pre-GT ranking artifact has been persisted."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(item["question_id"]): {
            "gt_interval_sec": [float(value) for value in item["gt_interval_sec"]],
            "ground_truth_index": int(item["ground_truth_index"]),
            "ground_truth_text": str(item["ground_truth_text"]),
        }
        for item in payload["questions"]
        if item.get("video_id") == VIDEO_ID
    }


def balanced_unique_candidates(
    option_rankings: Sequence[Sequence[int]], concrete_option_indices: Sequence[int], budget: int,
) -> list[int]:
    """Deterministic round-robin allocation, independent of GT and scores."""
    if budget <= 0:
        raise ValueError("Candidate budget must be positive")
    concrete = [int(index) for index in concrete_option_indices]
    if not concrete:
        raise ValueError("At least one concrete option is required")
    arrays = [np.asarray(option_rankings[index], dtype=np.int64) for index in concrete]
    selected: list[int] = []
    seen: set[int] = set()
    depth = 0
    while len(selected) < budget and any(depth < len(array) for array in arrays):
        for array in arrays:
            if depth >= len(array):
                continue
            candidate = int(array[depth])
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
                if len(selected) == budget:
                    break
        depth += 1
    return selected


def union_at_per_option_depth(
    option_rankings: Sequence[Sequence[int]], concrete_option_indices: Sequence[int], depth: int,
) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for rank in range(depth):
        for option_index in concrete_option_indices:
            ranking = option_rankings[int(option_index)]
            if rank >= len(ranking):
                continue
            candidate = int(ranking[rank])
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
    return selected


def candidate_interval_hit(
    candidates: Sequence[int], timestamps: np.ndarray, interval: Sequence[float],
) -> bool:
    start, end = map(float, interval)
    return any(start <= float(timestamps[index]) < end for index in candidates)


def _median_first_rank(rows: Sequence[dict[str, Any]]) -> float | None:
    values = [row["rank_of_first_timestamp_inside_gt_interval"] for row in rows]
    finite = [int(value) for value in values if value is not None]
    return float(statistics.median(finite)) if finite else None


def _aggregate_single_ranking(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "hit_at_k": {
            str(k): {
                "hits": sum(bool(row["gt_interval_hit_at_k"][str(k)]) for row in rows),
                "total": len(rows),
                "rate": (
                    sum(bool(row["gt_interval_hit_at_k"][str(k)]) for row in rows) / len(rows)
                    if rows else None
                ),
            }
            for k in K_VALUES
        },
        "median_first_inside_rank": _median_first_rank(rows),
    }


def _aggregate_balanced(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "hit_at_k": {
            str(k): {
                "hits": sum(bool(row["balanced_fixed_budget"][str(k)]["gt_interval_hit"]) for row in rows),
                "total": len(rows),
                "rate": (
                    sum(bool(row["balanced_fixed_budget"][str(k)]["gt_interval_hit"]) for row in rows) / len(rows)
                    if rows else None
                ),
            }
            for k in K_VALUES
        },
        "median_first_inside_rank": float(
            statistics.median(row["balanced_full_ranking_first_inside_rank"] for row in rows)
        ) if rows else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    *, manifest_path: Path, embedding_path: Path, prior_generic_result_path: Path,
    output_dir: Path, cache_root: Path, device_name: str = "cuda:0",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    import torch
    from huggingface_hub import hf_hub_download
    from torch.nn import functional as F

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official siglip2-g text encoding")

    with np.load(embedding_path) as artifact:
        embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
        timestamps = np.asarray(artifact["timestamps_sec"], dtype=np.float64)
        frame_indices = np.asarray(artifact["frame_indices"], dtype=np.int64)
    visual = torch.from_numpy(embeddings).to(device)
    visual = F.normalize(visual, dim=-1)

    query_rows = load_query_inputs(manifest_path)
    flat_queries = [
        build_option_query(row["question"], option)
        for row in query_rows for option in row["options"]
    ]
    checkpoint_path = Path(hf_hub_download(
        repo_id=RADIO_HF_REPO,
        filename=RADIO_CHECKPOINT_FILENAME,
        revision=RADIO_HF_REVISION,
    ))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model, _, model_warnings, model_load_sec = _load_model(device, checkpoint_path)

    text_started = time.perf_counter()
    tokenized = _to_device(model.adaptors[RADIO_ADAPTOR].tokenizer(flat_queries), device)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        option_text = model.adaptors[RADIO_ADAPTOR].encode_text(tokenized, normalize=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    text_encoding_sec = time.perf_counter() - text_started
    similarities = (F.normalize(option_text.float(), dim=-1) @ visual.T).cpu().numpy()
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    ranking_matrix = np.stack([rank_timestamps(row) for row in similarities]).reshape(len(query_rows), 5, -1)
    similarities = similarities.reshape(len(query_rows), 5, -1)

    ranking_npz_path = output_dir / "option_rankings_pre_gt.npz"
    np.savez_compressed(
        ranking_npz_path,
        similarities=similarities.astype(np.float32),
        ranking_indices=ranking_matrix.astype(np.int32),
        timestamps_sec=timestamps,
        frame_indices=frame_indices,
    )
    pre_gt_json_path = output_dir / "option_rankings_pre_gt.json"
    pre_gt_payload = {
        "schema_version": "cradio-v4-option-rankings-pre-gt-v1",
        "diagnostic_label": "ANSWER-CONDITIONED / OPTION-SEMANTIC DIAGNOSTIC — not formal B1",
        "video_id": VIDEO_ID,
        "gt_fields_loaded": False,
        "query_template": QUERY_TEMPLATE,
        "options_used_independently": True,
        "rankings_completed_before_gt_load": True,
        "model": {
            "model_id": RADIO_HF_REPO,
            "checkpoint_revision": RADIO_HF_REVISION,
            "implementation_revision": RADIO_GITHUB_REVISION,
            "adaptor": RADIO_ADAPTOR,
            "text_model": SIGLIP2_TEXT_MODEL,
            "compute_dtype": "bfloat16 CUDA autocast" if device.type == "cuda" else "float32 CPU",
        },
        "ranking_npz": str(ranking_npz_path),
        "ranking_npz_sha256": _sha256(ranking_npz_path),
        "questions": [
            {
                **row,
                "options": [
                    {
                        "option_index": option_index,
                        "option_text": option,
                        "query": build_option_query(row["question"], option),
                        "concrete_visual_semantics": is_concrete_option(option),
                        "none_of_above_warning": (
                            None if is_concrete_option(option)
                            else "Weak/no positive visual semantics; do not interpret like a concrete action query."
                        ),
                        "top_20": [
                            {
                                "rank": rank + 1,
                                "timestamp_sec": float(timestamps[index]),
                                "source_frame_index": int(frame_indices[index]),
                                "cosine_similarity": float(similarities[q_index, option_index, index]),
                            }
                            for rank, index in enumerate(ranking_matrix[q_index, option_index, :20])
                        ],
                    }
                    for option_index, option in enumerate(row["options"])
                ],
            }
            for q_index, row in enumerate(query_rows)
        ],
    }
    pre_gt_json_path.write_text(json.dumps(pre_gt_payload, indent=2) + "\n", encoding="utf-8")

    # This is the explicit leakage boundary: GT is loaded only after every option
    # and generic ranking has been persisted in immutable pre-GT artifacts.
    ground_truth = load_posthoc_ground_truth(manifest_path)
    previous_generic = json.loads(prior_generic_result_path.read_text(encoding="utf-8"))
    previous_generic_by_id = {
        row["question_id"]: row for row in previous_generic["retrieval"]["questions"]
    }

    question_results: list[dict[str, Any]] = []
    option_csv_rows: list[dict[str, Any]] = []
    generic_rows: list[dict[str, Any]] = []
    correct_rows: list[dict[str, Any]] = []
    for q_index, row in enumerate(query_rows):
        gt = ground_truth[row["question_id"]]
        start, end = gt["gt_interval_sec"]
        option_results = []
        for option_index, option in enumerate(row["options"]):
            metrics = posthoc_gt_interval_metrics(
                ranking_matrix[q_index, option_index], timestamps,
                gt_start_sec=start, gt_end_sec=end, ks=K_VALUES,
            )
            result = {
                "option_index": option_index,
                "option_text": option,
                "is_ground_truth_option": option_index == gt["ground_truth_index"],
                "concrete_visual_semantics": is_concrete_option(option),
                "none_of_above_warning": (
                    None if is_concrete_option(option)
                    else "Weak/no positive visual semantics; do not interpret like a concrete action query."
                ),
                **metrics,
            }
            option_results.append(result)
            option_csv_rows.append({
                "question_id": row["question_id"],
                "duration_class": row["duration_class"],
                "option_index": option_index,
                "option_text": option,
                "is_ground_truth_option": result["is_ground_truth_option"],
                "concrete_visual_semantics": result["concrete_visual_semantics"],
                **{f"hit_at_{k}": int(metrics["gt_interval_hit_at_k"][str(k)]) for k in K_VALUES},
                "first_inside_rank": metrics["rank_of_first_timestamp_inside_gt_interval"],
                "top1_distance_sec": metrics["best_ranked_timestamp_distance_to_gt_interval_sec"],
            })

        previous_row = previous_generic_by_id[row["question_id"]]
        first_inside = previous_row["rank_of_first_timestamp_inside_gt_interval"]
        generic_metrics = {
            "gt_interval_sec": [start, end],
            "gt_interval_hit_at_k": {
                **previous_row["gt_interval_hit_at_k"],
                "20": bool(first_inside is not None and int(first_inside) <= 20),
            },
            "best_ranked_timestamp_distance_to_gt_interval_sec": previous_row[
                "best_ranked_timestamp_distance_to_gt_interval_sec"
            ],
            "rank_of_first_timestamp_inside_gt_interval": first_inside,
            "source": "previous saved generic-question diagnostic; not recomputed",
        }
        generic_metrics["question_id"] = row["question_id"]
        generic_rows.append(generic_metrics)

        concrete_indices = [
            index for index, option in enumerate(row["options"]) if is_concrete_option(option)
        ]
        balanced = {}
        for budget in K_VALUES:
            candidates = balanced_unique_candidates(ranking_matrix[q_index], concrete_indices, budget)
            balanced[str(budget)] = {
                "candidate_count": len(candidates),
                "timestamps_sec": [float(timestamps[index]) for index in candidates],
                "gt_interval_hit": candidate_interval_hit(candidates, timestamps, (start, end)),
            }
        full_balanced = balanced_unique_candidates(
            ranking_matrix[q_index], concrete_indices, len(timestamps)
        )
        first_balanced = next(
            rank for rank, index in enumerate(full_balanced, 1) if start <= timestamps[index] < end
        )
        raw_unions = {}
        for depth in (5, 8, 10, 20):
            candidates = union_at_per_option_depth(ranking_matrix[q_index], concrete_indices, depth)
            raw_unions[str(depth)] = {
                "per_option_depth": depth,
                "unique_candidate_count": len(candidates),
                "timestamps_sec": [float(timestamps[index]) for index in candidates],
                "gt_interval_hit": candidate_interval_hit(candidates, timestamps, (start, end)),
            }
        correct_option = option_results[gt["ground_truth_index"]]
        correct_is_concrete = bool(correct_option["concrete_visual_semantics"])
        if correct_is_concrete:
            correct_rows.append({"question_id": row["question_id"], **correct_option})
        question_results.append({
            "question_id": row["question_id"],
            "duration_class": row["duration_class"],
            "question": row["question"],
            "gt_interval_sec": [start, end],
            "ground_truth_index": gt["ground_truth_index"],
            "ground_truth_text": gt["ground_truth_text"],
            "correct_option_has_concrete_visual_semantics": correct_is_concrete,
            "generic_question_metrics": generic_metrics,
            "option_metrics": option_results,
            "balanced_fixed_budget": balanced,
            "balanced_full_ranking_first_inside_rank": first_balanced,
            "raw_union_by_per_option_depth": raw_unions,
        })

    eligible_ids = {row["question_id"] for row in correct_rows}
    generic_eligible = [row for row in generic_rows if row["question_id"] in eligible_ids]
    balanced_eligible = [row for row in question_results if row["question_id"] in eligible_ids]
    aggregate = {
        "generic_question_all_five": _aggregate_single_ranking(generic_rows),
        "generic_question_concrete_gt_cohort": _aggregate_single_ranking(generic_eligible),
        "correct_option_semantic_concrete_gt_only": _aggregate_single_ranking(correct_rows),
        "all_options_balanced_all_five": _aggregate_balanced(question_results),
        "all_options_balanced_concrete_gt_cohort": _aggregate_balanced(balanced_eligible),
    }

    result = {
        "schema_version": "cradio-v4-option-semantic-diagnostic-v1",
        "diagnostic_label": "ANSWER-CONDITIONED / OPTION-SEMANTIC DIAGNOSTIC — exploratory, not formal B0-B4",
        "video_id": VIDEO_ID,
        "qwen_calls": 0,
        "existing_visual_embeddings_reused": True,
        "visual_embeddings_recomputed": False,
        "rankings_completed_and_saved_before_gt_load": True,
        "pre_gt_ranking_artifact": str(pre_gt_json_path),
        "pre_gt_ranking_artifact_sha256": _sha256(pre_gt_json_path),
        "model": pre_gt_payload["model"],
        "runtime": {
            "model_load_sec": model_load_sec,
            "text_encoding_sec": text_encoding_sec,
            "text_encoding_device": str(device),
            "peak_gpu_allocated_bytes": peak_vram,
            "model_warnings": model_warnings,
        },
        "query_policy": {
            "template": QUERY_TEMPLATE,
            "correct_option_not_used_until_posthoc_evaluation": True,
            "all_five_options_ranked_independently": True,
            "none_of_above_policy": "Retained in raw rankings but excluded from concrete-option candidate merging.",
            "balanced_merge": "Round-robin by ascending option index and within-option rank; duplicates skipped; fixed total budget.",
        },
        "aggregate": aggregate,
        "questions": question_results,
        "interpretation_limit": (
            "Correct-option retrieval is an answer-conditioned post-hoc upper bound, not deployable retrieval. "
            "Balanced all-options retrieval is result-independent but remains exploratory and does not define B1."
        ),
    }
    result_path = output_dir / "option_semantic_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "option_rank_statistics.csv", option_csv_rows)
    summary_rows = []
    for name, values in aggregate.items():
        summary_rows.append({
            "query_formulation": name,
            "n": values["n"],
            **{f"hit_at_{k}": values["hit_at_k"][str(k)]["hits"] for k in K_VALUES},
            "median_first_inside_rank": values["median_first_inside_rank"],
        })
    _write_csv(output_dir / "query_formulation_comparison.csv", summary_rows)

    del model, visual, option_text
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
