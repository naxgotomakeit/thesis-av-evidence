from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

from videoseal.tools.base import Tool, ToolOutput

from experiments.planner_medium_retrieval.core import SiglipTextEncoder, lexical_similarity

def select_time_diverse_windows(windows, k, min_gap_sec):
    chosen = []
    for i, (start, end) in enumerate(windows):
        if len(chosen) >= k:
            break
        if all(end <= windows[j][0] - min_gap_sec or start >= windows[j][1] + min_gap_sec for j in chosen):
            chosen.append(i)
    return chosen


@dataclass(frozen=True)
class HierarchicalProfile:
    name: str
    mode: str
    coarse_top_k: Optional[int]
    medium_top_k: Optional[int]
    fine_per_medium: Optional[int]
    output_top_k: int


PROFILES = {
    "h6": HierarchicalProfile("H-6", "hierarchical_gated", 3, 3, 2, 6),
    "h15": HierarchicalProfile("H-15", "hierarchical_gated", 5, 5, 3, 15),
    "h30": HierarchicalProfile("H-30-diagnostic", "all_fine_no_gate", None, None, None, 30),
}

VISUAL_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.3
MEDIUM_TIE_BREAK = ["combined_score desc", "start_sec asc", "medium_id asc"]
FINE_TIE_BREAK = ["siglip_score_raw desc", "timestamp_sec asc", "fine_id asc"]
COARSE_TIE_BREAK = ["lexical_score desc", "start_sec asc", "coarse_id desc"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        return np.ones_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def _coarse_text(node: dict[str, Any]) -> str:
    return " ".join(
        [str(node.get("navigation_summary") or "")]
        + [str(item) for item in node.get("uncertainty_notes") or []]
    ).strip()


def _resolve_case_dir(index_path: str | Path, video_id: Optional[str]) -> Path:
    base = Path(index_path).expanduser().resolve()
    candidates = [base]
    if video_id:
        candidates.extend((base / str(video_id), base / "cases" / str(video_id)))
    for candidate in candidates:
        if (candidate / "r3_2_navigation_map.json").is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve hierarchical case under {base}: {video_id!r}")


def load_case(case_dir: Path) -> dict[str, Any]:
    navigation = _read_json(case_dir / "r3_2_navigation_map.json")
    hierarchy = _read_json(case_dir / "shared_hierarchy.json")
    caption_rows = _read_json(case_dir / "r3_medium_captions.json")
    captions = {str(row["medium_id"]): row for row in caption_rows}
    medium_embeddings = np.load(
        case_dir / "medium_siglip.float32.npy", allow_pickle=False
    ).astype(np.float32)
    fine_source = np.load(case_dir / "fine_siglip.npz", allow_pickle=False)
    fine_embeddings = fine_source["embedding"].astype(np.float32)
    medium_embeddings /= np.maximum(
        np.linalg.norm(medium_embeddings, axis=1, keepdims=True), 1e-12
    )
    fine_embeddings /= np.maximum(
        np.linalg.norm(fine_embeddings, axis=1, keepdims=True), 1e-12
    )
    return {
        "case_dir": case_dir,
        "navigation": navigation,
        "hierarchy": hierarchy,
        "captions": captions,
        "medium_embeddings": medium_embeddings,
        "fine_embeddings": fine_embeddings,
    }


def _canonical_span(start: float, end: float, duration: float, caption: str) -> dict[str, str]:
    start = max(0.0, min(float(start), duration))
    end = max(0.0, min(float(end), duration))
    if not start < end:
        raise ValueError(f"invalid temporal span: {start} >= {end}")
    start_i, end_i = int(math.floor(start)), int(math.ceil(end))
    if end_i <= start_i:
        end_i = min(int(math.ceil(duration)), start_i + 1)
    if end_i <= start_i:
        raise ValueError(f"invalid serialized span: {start_i} >= {end_i}")
    def stamp(value: int) -> str:
        return f"{value // 3600:02d}:{(value % 3600) // 60:02d}:{value % 60:02d}"
    return {
        "start_time": stamp(start_i),
        "end_time": stamp(end_i),
        "caption": str(caption or "").strip(),
    }


def _diverse_per_medium(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not rows or limit <= 0:
        return []
    chosen = [rows[0]]
    for candidate in rows[1:]:
        if len(chosen) >= limit:
            break
        first = chosen[0]
        timestamp_gap = abs(float(first["timestamp_sec"]) - float(candidate["timestamp_sec"]))
        different_interval = (
            first["fine_id"] != candidate["fine_id"]
            and (first["start_sec"], first["end_sec"])
            != (candidate["start_sec"], candidate["end_sec"])
        )
        if timestamp_gap >= 10.0 or different_interval:
            chosen.append(candidate)
    if len(chosen) < limit:
        used = {row["fine_id"] for row in chosen}
        chosen.extend(row for row in rows if row["fine_id"] not in used)
    return chosen[:limit]


def rank_case(
    case: dict[str, Any],
    *,
    query: str,
    query_embedding: np.ndarray,
    profile: HierarchicalProfile,
    min_time_gap_sec: float = 15.0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    total_started = time.monotonic()
    hierarchy = case["hierarchy"]
    navigation = case["navigation"]
    captions = case["captions"]
    medium_nodes = list(hierarchy["medium_nodes"])
    fine_nodes = list(hierarchy["fine_nodes"])
    medium_by_id = {str(row["medium_id"]): row for row in medium_nodes}
    fine_by_id = {str(row["fine_id"]): row for row in fine_nodes}
    duration = float(hierarchy["duration_sec"])
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    query_embedding /= max(float(np.linalg.norm(query_embedding)), 1e-12)

    coarse_started = time.monotonic()
    coarse_ranking: list[dict[str, Any]] = []
    for node in navigation["coarse_regions"]:
        score, terms = lexical_similarity(query, _coarse_text(node))
        coarse_ranking.append(
            {
                "coarse_id": str(node["coarse_id"]),
                "start_sec": float(node["start_sec"]),
                "end_sec": float(node["end_sec"]),
                "lexical_score": float(score),
                "matched_terms": terms,
                "source_medium_ids": [str(value) for value in node.get("source_medium_ids") or []],
                "navigation_summary": str(node.get("navigation_summary") or ""),
                "uncertainty_notes": list(node.get("uncertainty_notes") or []),
            }
        )
    coarse_ranking.sort(
        key=lambda row: (-row["lexical_score"], row["start_sec"], _descending_id(row["coarse_id"]))
    )
    if profile.mode == "hierarchical_gated":
        selected_coarse = coarse_ranking[: int(profile.coarse_top_k or 0)]
    else:
        selected_coarse = list(coarse_ranking)
    selected_coarse_ids = {row["coarse_id"] for row in selected_coarse}
    allowed_medium_ids = {
        medium_id for row in selected_coarse for medium_id in row["source_medium_ids"]
    }
    for rank, row in enumerate(coarse_ranking, 1):
        row["rank"] = rank
        row["selected"] = row["coarse_id"] in selected_coarse_ids
    coarse_sec = time.monotonic() - coarse_started

    medium_started = time.monotonic()
    eligible_mediums = [
        row for row in medium_nodes if str(row["medium_id"]) in allowed_medium_ids
    ]
    if profile.mode == "all_fine_no_gate":
        eligible_mediums = list(medium_nodes)
    if not eligible_mediums:
        raise ValueError("no eligible Medium nodes")
    medium_embeddings = case["medium_embeddings"]
    medium_positions = {str(row["medium_id"]): i for i, row in enumerate(medium_nodes)}
    raw_medium = np.asarray(
        [float(medium_embeddings[medium_positions[str(row["medium_id"])]] @ query_embedding) for row in eligible_mediums],
        dtype=np.float64,
    )
    norm_medium = _normalize(raw_medium)
    medium_ranking: list[dict[str, Any]] = []
    for offset, node in enumerate(eligible_mediums):
        medium_id = str(node["medium_id"])
        caption = str(captions.get(medium_id, {}).get("qwen_caption") or "")
        lexical_score, terms = lexical_similarity(query, caption)
        combined = VISUAL_WEIGHT * float(norm_medium[offset]) + LEXICAL_WEIGHT * float(lexical_score)
        medium_ranking.append(
            {
                "medium_id": medium_id,
                "start_sec": float(node["start_sec"]),
                "end_sec": float(node["end_sec"]),
                "visual_score_raw": float(raw_medium[offset]),
                "visual_score_normalized": float(norm_medium[offset]),
                "lexical_score": float(lexical_score),
                "matched_terms": terms,
                "combined_score": float(combined),
                "caption": caption,
            }
        )
    medium_ranking.sort(key=lambda row: (-row["combined_score"], row["start_sec"], row["medium_id"]))
    if profile.mode == "hierarchical_gated":
        selected_medium = medium_ranking[: int(profile.medium_top_k or 0)]
    else:
        selected_medium = list(medium_ranking)
    selected_medium_ids = {row["medium_id"] for row in selected_medium}
    for rank, row in enumerate(medium_ranking, 1):
        row["rank"] = rank
        row["selected"] = row["medium_id"] in selected_medium_ids
    medium_sec = time.monotonic() - medium_started

    fine_started = time.monotonic()
    fine_embeddings = case["fine_embeddings"]
    fine_ranking: list[dict[str, Any]] = []
    if profile.mode == "hierarchical_gated":
        for medium_rank, medium_row in enumerate(selected_medium, 1):
            medium_id = medium_row["medium_id"]
            child_ids = [str(value) for value in medium_by_id[medium_id]["source_fine_ids"]]
            local: list[dict[str, Any]] = []
            local_raw = np.asarray(
                [float(fine_embeddings[int(fine_by_id[fine_id]["frame_index"])] @ query_embedding) for fine_id in child_ids],
                dtype=np.float64,
            )
            local_norm = _normalize(local_raw)
            for offset, fine_id in enumerate(child_ids):
                node = fine_by_id[fine_id]
                local.append(_fine_row(node, medium_id, medium_rank, medium_row, local_raw[offset], local_norm[offset]))
            local.sort(key=lambda row: (-row["siglip_score_raw"], row["timestamp_sec"], row["fine_id"]))
            for rank, row in enumerate(local, 1):
                row["rank_within_medium"] = rank
            chosen = _diverse_per_medium(local, int(profile.fine_per_medium or 0))
            chosen_ids = {row["fine_id"] for row in chosen}
            for row in local:
                row["selected_before_global_diversity"] = row["fine_id"] in chosen_ids
            fine_ranking.extend(local)
        pre_dedupe = [row for row in fine_ranking if row["selected_before_global_diversity"]]
        pre_dedupe.sort(key=lambda row: (row["medium_rank"], -row["siglip_score_raw"], row["start_sec"], row["fine_id"]))
    else:
        for node in fine_nodes:
            medium_id = str(node["parent_medium_id"])
            medium_row = next(row for row in medium_ranking if row["medium_id"] == medium_id)
            raw = float(fine_embeddings[int(node["frame_index"])] @ query_embedding)
            fine_ranking.append(_fine_row(node, medium_id, medium_row["rank"], medium_row, raw, 0.0))
        fine_ranking.sort(key=lambda row: (-row["siglip_score_raw"], row["timestamp_sec"], row["fine_id"]))
        all_norm = _normalize(np.asarray([row["siglip_score_raw"] for row in fine_ranking], dtype=np.float64))
        for rank, row in enumerate(fine_ranking, 1):
            row["rank_global"] = rank
            row["siglip_score_normalized"] = float(all_norm[rank - 1])
            row["selected_before_global_diversity"] = True
        pre_dedupe = list(fine_ranking)

    deduped: list[dict[str, Any]] = []
    seen_windows: set[tuple[float, float]] = set()
    for row in pre_dedupe:
        key = (float(row["start_sec"]), float(row["end_sec"]))
        if key not in seen_windows:
            seen_windows.add(key)
            deduped.append(row)
    keep = select_time_diverse_windows(
        [(float(row["start_sec"]), float(row["end_sec"])) for row in deduped],
        k=profile.output_top_k,
        min_gap_sec=float(min_time_gap_sec),
    )
    selected_fine = [deduped[index] for index in keep[: profile.output_top_k]]
    selected_fine_ids = {row["fine_id"] for row in selected_fine}
    for row in fine_ranking:
        row["returned"] = row["fine_id"] in selected_fine_ids
    fine_sec = time.monotonic() - fine_started

    spans = [
        _canonical_span(row["start_sec"], row["end_sec"], duration, row["caption"])
        for row in selected_fine
    ]
    coarse_by_medium = {
        medium_id: row["coarse_id"]
        for row in navigation["coarse_regions"]
        for medium_id in row.get("source_medium_ids") or []
    }
    coarse_by_id = {row["coarse_id"]: row for row in coarse_ranking}
    medium_rank_by_id = {row["medium_id"]: row for row in medium_ranking}
    provenance = []
    for candidate_index, fine in enumerate(selected_fine):
        medium_id = fine["medium_id"]
        coarse_id = str(coarse_by_medium[medium_id])
        provenance.append(
            {
                "candidate_index": candidate_index,
                "candidate": spans[candidate_index],
                "coarse": coarse_by_id[coarse_id],
                "medium": medium_rank_by_id[medium_id],
                "fine": fine,
            }
        )
    total_sec = time.monotonic() - total_started
    metadata = {
        "retrieval_backend": "hierarchical",
        "config_name": profile.name,
        "mode": profile.mode,
        "hierarchy_used": True,
        "hierarchy_complete": bool(spans and provenance),
        "all_coarse_expanded": profile.mode == "all_fine_no_gate",
        "all_medium_expanded": profile.mode == "all_fine_no_gate",
        "coarse_gate_applied": profile.mode == "hierarchical_gated",
        "medium_gate_applied": profile.mode == "hierarchical_gated",
        "counts": {
            "coarse_total": len(coarse_ranking),
            "coarse_scored": len(coarse_ranking),
            "coarse_eligible": len(coarse_ranking),
            "coarse_selected": len(selected_coarse),
            "medium_total": len(medium_nodes),
            "medium_scored": len(medium_ranking),
            "medium_eligible": len(eligible_mediums),
            "medium_selected": len(selected_medium),
            "fine_total": len(fine_nodes),
            "fine_scored": len(fine_ranking),
            "fine_eligible": len(fine_ranking),
            "fine_before_dedupe": len(pre_dedupe),
            "fine_after_dedupe": len(deduped),
            "fine_returned": len(selected_fine),
            "raw_candidate_count": len(spans),
        },
        "score_formula": "0.6 * minmax(SigLIP) + 0.3 * caption_lexical",
        "coarse_text_fields": ["navigation_summary", "uncertainty_notes"],
        "coarse_ranking": coarse_ranking,
        "medium_ranking": medium_ranking,
        "fine_ranking": fine_ranking,
        "selected_coarse": selected_coarse,
        "selected_medium": selected_medium,
        "selected_fine": selected_fine,
        "coarse_medium_fine_paths": provenance,
        "tie_break": {"coarse": COARSE_TIE_BREAK, "medium": MEDIUM_TIE_BREAK, "fine": FINE_TIE_BREAK},
        "temporal_diversity": {"min_gap_sec": float(min_time_gap_sec), "selected_indices": keep},
        "planner_visible_contract": "list[start_time,end_time,caption]",
        "planner_visible_summary": None,
        "audio_used": False,
        "latency_sec": {
            "coarse": round(coarse_sec, 6),
            "medium": round(medium_sec, 6),
            "fine": round(fine_sec, 6),
            "total_ranking": round(total_sec, 6),
        },
    }
    return spans, metadata


def _descending_id(value: str) -> tuple[int, ...]:
    # Python keys are ascending; negative code points retain the historical reverse-string tie.
    return tuple(-ord(char) for char in value)


def _fine_row(
    node: dict[str, Any],
    medium_id: str,
    medium_rank: int,
    medium_row: dict[str, Any],
    raw: float,
    normalized: float,
) -> dict[str, Any]:
    return {
        "fine_id": str(node["fine_id"]),
        "medium_id": medium_id,
        "medium_rank": int(medium_rank),
        "start_sec": float(node["start_sec"]),
        "end_sec": float(node["end_sec"]),
        "timestamp_sec": float(node["timestamp_sec"]),
        "frame_index": int(node["frame_index"]),
        "source_frame_path": str(node.get("source_frame_path") or ""),
        "siglip_score_raw": float(raw),
        "siglip_score_normalized": float(normalized),
        "medium_combined_score": float(medium_row["combined_score"]),
        "caption": str(medium_row["caption"]),
    }


_ENCODER_STATE: dict[str, Any] = {"initialized": False, "initialization_sec": 0.0}


@lru_cache(maxsize=1)
def _encoder(model: str, cache_dir: str, max_text_tokens: int) -> SiglipTextEncoder:
    started = time.monotonic()
    encoder = SiglipTextEncoder(
        {
            "model": model,
            "cache_dir": cache_dir,
            "local_files_only": True,
            "embedding_dimension": 768,
            "max_text_tokens": max_text_tokens,
        },
        device="cpu",
    )
    _ENCODER_STATE.update(initialized=True, initialization_sec=time.monotonic() - started)
    return encoder


class HierarchicalVisualRetrieveTool(Tool):
    """Runtime tool; profile changes only the isolated hierarchical search width/gates."""

    def forward(
        self,
        *,
        query: str,
        video_id: Optional[str] = None,
        index_path: Optional[str] = None,
        top_k: Optional[int] = None,
        original_question: Optional[str] = None,
    ) -> ToolOutput:
        del original_question, top_k
        try:
            query = str(query or "").strip()
            if not query:
                raise ValueError("query must be non-empty")
            if not index_path:
                raise ValueError("hierarchical backend requires index_path")
            profile_key = str(os.getenv("HIERARCHICAL_PROFILE") or "h6").strip().lower()
            if profile_key not in PROFILES:
                raise ValueError(f"unknown HIERARCHICAL_PROFILE: {profile_key!r}")
            case_dir = _resolve_case_dir(index_path, video_id)
            case = load_case(case_dir)
            model = str(os.getenv("HIERARCHICAL_SIGLIP_MODEL") or "google/siglip-base-patch16-224")
            cache_dir = str(os.getenv("HIERARCHICAL_SIGLIP_CACHE") or "")
            max_tokens = int(os.getenv("HIERARCHICAL_SIGLIP_MAX_TEXT_TOKENS") or "64")
            was_initialized = bool(_ENCODER_STATE["initialized"])
            encoder = _encoder(model, cache_dir, max_tokens)
            encode_started = time.monotonic()
            query_embedding = np.asarray(encoder.encode([query])[0], dtype=np.float32)
            query_encode_sec = time.monotonic() - encode_started
            spans, metadata = rank_case(
                case,
                query=query,
                query_embedding=query_embedding,
                profile=PROFILES[profile_key],
                min_time_gap_sec=float(os.getenv("RETRIEVE_MIN_TIME_GAP_SEC") or "15"),
            )
            metadata["video_id"] = str(video_id or case_dir.name)
            metadata["video_duration_sec"] = float(case["hierarchy"]["duration_sec"])
            metadata["siglip"] = {
                "model": model,
                "initialization_sec": round(float(_ENCODER_STATE["initialization_sec"]), 6),
                "cold_retrieval": not was_initialized,
                "warm_retrieval": was_initialized,
                "query_encode_sec": round(query_encode_sec, 6),
            }
            metadata["planner_visible_summary"] = None
            metadata["retrieval_latency_sec"] = round(
                float(metadata["latency_sec"]["total_ranking"])
                + query_encode_sec,
                6,
            )
            return ToolOutput(self.name, output=spans, metadata=metadata)
        except Exception as exc:
            return ToolOutput(self.name, error=f"{type(exc).__name__}: {exc}")
