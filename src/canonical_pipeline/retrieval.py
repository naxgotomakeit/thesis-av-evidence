"""Canonical Task 5B v1.1 per-case retrieval adapter.

Source lineage: the already reusable ``build_case`` function and the v1.1
visual packaging correction. Offline indexes are reused; none are rebuilt.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Protocol

from scripts.run_task5b_retrieval import build_case
from src.retrieval.task5b import canonical_visual_frames, package_micro_frames

from .state import CaseState, ExecutionMode
from .query_scoring import QueryScoreResult


class QueryScorer(Protocol):
    """Boundary for fresh local question-conditioned modality scoring."""

    def score_case(self, case: dict[str, Any], modalities: list[str]) -> dict[str, QueryScoreResult]: ...


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _declared_fps(project_root: Path, index: dict[str, Any]) -> float | None:
    for value in (index.get("fps"), index.get("sampling_fps"), index.get("frame_sampling_fps")):
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    coarse = project_root / "outputs/visual_index" / str(index.get("video_id")) / "visual_state_regions.json"
    if coarse.is_file():
        metadata = _load(coarse)
        for value in (metadata.get("fps"), metadata.get("sampling_fps"), (metadata.get("config") or {}).get("fps")):
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None


def _interval_distance(timestamp: float, candidate: dict[str, Any]) -> float:
    start, end = float(candidate["start_time"]), float(candidate["end_time"])
    if start <= timestamp < end:
        return 0.0
    return start - timestamp if timestamp < start else timestamp - end


def _attach_dense_frame_lineage(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Add deterministic source identity to dense assets without selecting frames."""
    dense = copy.deepcopy(result["local_visual_refinement"].get("dense_frames", []))
    micro = list(result["local_visual_refinement"].get("micro_windows", []))
    coarse = list(result.get("coarse_visual_candidates", []))
    if dense:
        bounds = {
            "start_sec": min(float(item["timestamp"]) for item in dense),
            "end_sec": max(float(item["timestamp"]) for item in dense),
        }
    else:
        bounds = None
    for frame in dense:
        timestamp = float(frame["timestamp"])
        overlapping_micro = [item for item in micro if _interval_distance(timestamp, item) == 0.0]
        overlapping_coarse = [item for item in coarse if _interval_distance(timestamp, item) == 0.0]
        nearest_micro = (
            [min(micro, key=lambda item: (_interval_distance(timestamp, item), item["candidate_id"]))]
            if not overlapping_micro and micro
            else []
        )
        nearest_coarse = (
            [min(coarse, key=lambda item: (_interval_distance(timestamp, item), item["candidate_id"]))]
            if not overlapping_coarse and coarse
            else []
        )
        micro_sources = overlapping_micro or nearest_micro
        coarse_sources = overlapping_coarse or nearest_coarse
        source_ids = [item["candidate_id"] for item in micro_sources]
        if not source_ids:
            source_ids = [item["candidate_id"] for item in coarse_sources]
        if not source_ids:
            raise RuntimeError("Dense visual frame has no resolvable refinement source candidate")
        frame["source_candidate_ids"] = source_ids
        frame["source_coarse_candidate_ids"] = [item["candidate_id"] for item in coarse_sources]
        frame["source_refinement_interval"] = bounds
        frame["lineage_basis"] = (
            "temporal_overlap_with_refinement_candidate"
            if overlapping_micro or overlapping_coarse
            else "nearest_refinement_candidate"
        )
    return dense


def _apply_v1_1_visual_packaging(
    base: dict[str, Any],
    project_root: Path,
    *,
    include_dense_lineage: bool = True,
) -> dict[str, Any]:
    """Apply the final v1.1 half-open frame packaging without a patch stage."""
    result = copy.deepcopy(base)
    corrected: dict[str, dict[str, Any]] = {}
    for candidate in base["all_candidates"]:
        if candidate.get("candidate_type") != "micro_window":
            continue
        path = project_root / candidate["source_index"]
        index = _load(path)
        source = next((item for item in index["microclips"] if item["microclip_id"] == candidate["microclip_id"]), None)
        if source is None:
            item = copy.deepcopy(candidate)
            item.update({"source_frame_paths": list(candidate.get("frame_paths", [])), "source_frame_timestamps": [], "selected_frame_paths": [], "selected_frame_timestamps": [], "frame_paths": []})
            item.setdefault("warnings", []).append("unable_to_resolve_micro_frame_timestamps")
        else:
            item = package_micro_frames(candidate, source, _declared_fps(project_root, index))
            if index.get("video_id") is not None:
                item["video_id"] = str(index["video_id"])
        corrected[candidate["candidate_id"]] = item
    rewrite = lambda items: [copy.deepcopy(corrected.get(item["candidate_id"], item)) for item in items]
    result["all_candidates"] = rewrite(base["all_candidates"])
    result["selected_candidates"] = rewrite(base["selected_candidates"])
    result["local_visual_refinement"]["micro_windows"] = rewrite(base["local_visual_refinement"]["micro_windows"])
    selected_micro = [item for item in result["selected_candidates"] if item.get("candidate_type") == "micro_window"]
    if include_dense_lineage:
        dense = _attach_dense_frame_lineage(result)
        result["local_visual_refinement"]["dense_frames"] = copy.deepcopy(dense)
    else:
        # Frozen regression references predate provenance-only lineage metadata.
        dense = list(result["local_visual_refinement"].get("dense_frames", []))
    video_id = next((item.get("video_id") for item in selected_micro if item.get("video_id")), None)
    canonical = canonical_visual_frames(selected_micro, dense, video_id=video_id)
    result["visual_evidence_accounting"] = {"source_micro_frame_count": sum(len(item.get("source_frame_paths", [])) for item in selected_micro), "selected_micro_frame_count": sum(len(item.get("selected_frame_paths", [])) for item in selected_micro), "dense_frame_count": len(dense), "unique_selected_visual_frame_count": len(canonical), "old_task5b_selected_visual_frame_count": base["selected_visual_frame_count"], "timestamp_normalization": "video_id + timestamp rounded to nearest millisecond", "deduplication_policy": "dense frame preferred as canonical representation; all provenance retained"}
    result["selected_visual_evidence_frames"] = canonical
    result["local_visual_refinement"]["selected_visual_evidence_frames"] = copy.deepcopy(canonical)
    result["selected_visual_frame_count"] = len(canonical)
    result["task5b_correction_version"] = "v1.1"
    result["correction_scope"] = "visual frame packaging and deduplicated efficiency accounting only"
    return result


def run_planner_guided_retrieval(state: CaseState, project_root: Path, query_scorer: QueryScorer | None = None, runtime_output_root: Path | None = None) -> CaseState:
    """Run Task 5B v1.1, optionally using fresh Task 4-lineage scores.

    Regression replay deliberately retains the historical reference path.
    Generalized live execution supplies ``query_scorer`` and never reads an
    old per-question Task 4 score artifact.
    """
    if state.planner_output is None:
        raise ValueError("Planner output is required before retrieval")
    planner_record = {
        "case_id": state.case_id,
        "raw_question": state.question,
        "plan": state.planner_output,
        "deterministic_cues": state.deterministic_cues,
    }
    safe_manifest = {"video_id": state.video_id, "video_duration": state.video_duration_sec}
    score_results: dict[str, QueryScoreResult] = {}
    if query_scorer is not None:
        requested = list(dict.fromkeys(
            [item for item in state.planner_output.get("resolver_modalities", []) if item in {"visual", "speech", "acoustic"}]
            + ([state.planner_output.get("primary_anchor_modality")] if state.planner_output.get("primary_anchor_modality") in {"visual", "speech", "acoustic"} else [])
        ))
        requested = [item for item in requested if item in state.available_modalities]
        score_results = query_scorer.score_case(
            {"case_id": state.case_id, "video_id": state.video_id, "question": state.question}, requested,
        )
    score_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for modality, result in score_results.items():
        # Historical Task 5B read only Task 4's Top-3 speech score file, while
        # its acoustic online path scored every acoustic row. Preserve that
        # exact distinction. Visual scores annotate coarse retrieval/anchors.
        rows = result.top_k_results if modality == "speech" else result.ranked_results
        id_key = {"visual": "region_id", "speech": "transcript_segment_id", "acoustic": "acoustic_region_id"}[modality]
        score_maps[modality] = {str(item[id_key]): copy.deepcopy(item) for item in rows}
    acoustic_result = score_results.get("acoustic")
    base = build_case(
        planner_record, safe_manifest,
        acoustic_query=acoustic_result.query_vector if acoustic_result else None,
        acoustic_query_time=acoustic_result.query_encode_sec if acoustic_result else 0.0,
        fresh_score_maps=score_maps, allow_historical_score_files=query_scorer is None,
        source_video_path=Path(state.source_video_path) if state.source_video_path else None,
        local_visual_output_root=runtime_output_root / "local_visual" if runtime_output_root else None,
        available_modalities=set(state.available_modalities),
    )
    if score_results:
        audits = {modality: result.audit_dict() for modality, result in score_results.items()}
        base["question_conditioned_scoring"] = {
            "execution_mode": "fresh_online",
            "historical_task4_score_files_used": False,
            "modalities": audits,
        }
        base["runtime"]["query_scoring"] = {
            modality: {
                "encoder_model_load_sec": result.model_load_sec,
                "query_encode_sec": result.query_encode_sec,
                "similarity_search_sec": result.similarity_search_sec,
            }
            for modality, result in score_results.items()
        }
        base["runtime"]["new_embedding_computations"] += len(score_results)
        base["runtime"]["new_embedding_computation_time_sec"] += sum(result.query_encode_sec for result in score_results.values())
        base["runtime"]["reused_retrieval_score_files"] = []
        base["warnings"] = [warning for warning in base["warnings"] if not warning.startswith("missing_input: outputs/retrieval/")]
    else:
        base["question_conditioned_scoring"] = {
            "execution_mode": "historical_regression_reference",
            "historical_task4_score_files_used": True,
            "modalities": {},
        }
    state.retrieval_result = _apply_v1_1_visual_packaging(
        base,
        project_root,
        include_dense_lineage=state.mode is ExecutionMode.EXECUTE_LIVE,
    )
    state.record("planner_guided_retrieval", "task5b_v1_1_completed", offline_indexes_reused=True, fresh_question_scores=bool(score_results), historical_task4_score_dependency=not bool(score_results))
    return state
