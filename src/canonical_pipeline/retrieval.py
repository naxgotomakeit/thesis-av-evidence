"""Canonical Task 5B v1.1 per-case retrieval adapter.

Source lineage: the already reusable ``build_case`` function and the v1.1
visual packaging correction. Offline indexes are reused; none are rebuilt.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from scripts.run_task5b_retrieval import build_case
from src.retrieval.task5b import canonical_visual_frames, package_micro_frames

from .state import CaseState


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


def _apply_v1_1_visual_packaging(base: dict[str, Any], project_root: Path) -> dict[str, Any]:
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


def run_planner_guided_retrieval(state: CaseState, project_root: Path) -> CaseState:
    """Run final Task 5B v1.1 behavior from reusable offline indexes."""
    if state.planner_output is None:
        raise ValueError("Planner output is required before retrieval")
    planner_record = {
        "case_id": state.case_id,
        "raw_question": state.question,
        "plan": state.planner_output,
        "deterministic_cues": state.deterministic_cues,
    }
    safe_manifest = {"video_id": state.video_id, "video_duration": state.video_duration_sec}
    base = build_case(planner_record, safe_manifest, acoustic_query=None, acoustic_query_time=0.0)
    state.retrieval_result = _apply_v1_1_visual_packaging(base, project_root)
    state.record("planner_guided_retrieval", "task5b_v1_1_completed", offline_indexes_reused=True)
    return state
