from __future__ import annotations

import contextvars
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional, Type

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
VIDEOSEAL_ROOT = RUNTIME_ROOT.parents[1] / "VideoSEAL"
if str(VIDEOSEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEOSEAL_ROOT))

from videoseal.tools.base import Tool, ToolOutput  # noqa: E402
from videoseal.tools.visual_tools import (  # noqa: E402
    VisualInspectAliasTool,
    VisualRetrieveAliasTool,
)
from videoseal.utils.lvbench_io import parse_choice_letter, parse_choice_letter_smart  # noqa: E402
from videoseal.utils.video.time import sec_to_hhmmss, srt_timestamp_to_seconds  # noqa: E402
from videoseal.utils.video.tooling import select_time_diverse_windows  # noqa: E402

from experiments.fine_reranking.core import (  # noqa: E402
    rerank_fines_for_medium,
    select_diverse_fines,
)
from experiments.planner_medium_retrieval.core import (  # noqa: E402
    SiglipTextEncoder,
    lexical_similarity,
)


RETRIEVE_DESCRIPTION = (
    "Retrieve candidate temporal spans from the video's visual semantic index. "
    "Use a natural-language visual query; inspect returned spans before answering."
)
RETRIEVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "visual_retrieve",
        "description": RETRIEVE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _resolve(path_value: str, *, root: Path = RUNTIME_ROOT) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_case_dir(index_path: str | Path, video_id: Optional[str]) -> Path:
    base = Path(index_path).expanduser().resolve()
    candidates = [base]
    if video_id:
        candidates.extend([base / str(video_id), base / "cases" / str(video_id)])
    for candidate in candidates:
        if (candidate / "r3_2_navigation_map.json").is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve hierarchical case: index_path={base}, video_id={video_id!r}")


def _case_config(case_dir: Path) -> dict[str, Any]:
    config = case_dir.parents[1] / "case_configs" / f"{case_dir.name}.json"
    if not config.is_file():
        raise FileNotFoundError(f"case config not found: {config}")
    return _read_json(config)


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        return np.ones_like(values, dtype=np.float64)
    return (values - low) / (high - low)


@lru_cache(maxsize=1)
def _encoder(model: str, cache_dir: str, max_text_tokens: int) -> SiglipTextEncoder:
    # Retrieval runs on CPU so it cannot disturb an existing GPU experiment.
    return SiglipTextEncoder(
        {
            "model": model,
            "cache_dir": cache_dir,
            "local_files_only": True,
            "embedding_dimension": 768,
            "max_text_tokens": int(max_text_tokens),
        },
        device="cpu",
    )


def _coarse_text(node: dict[str, Any]) -> str:
    # HourVideo smoke policy: visual semantic fields only; ASR is deliberately excluded.
    pieces = [str(node.get("navigation_summary") or "")]
    pieces.extend(str(item) for item in node.get("uncertainty_notes") or [])
    return " ".join(pieces)


def _load_fine_registry(
    hierarchy: dict[str, Any], case_cfg: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], np.ndarray, dict[str, int]]:
    source = np.load(str(case_cfg["siglip_npz"]), allow_pickle=False)
    embeddings = source["embedding"].astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    fine_by_id: dict[str, dict[str, Any]] = {}
    row_by_id: dict[str, int] = {}
    for fine in hierarchy["fine_nodes"]:
        row = dict(fine)
        row["representative_frame_timestamp_sec"] = fine["timestamp_sec"]
        row["representative_frame_path"] = fine["source_frame_path"]
        fine_by_id[str(fine["fine_id"])] = row
        row_by_id[str(fine["fine_id"])] = int(fine["frame_index"])
    return fine_by_id, embeddings, row_by_id


def _canonical_span(start: float, end: float, duration: float, caption: str) -> dict[str, str]:
    start = max(0.0, min(float(start), float(duration)))
    end = max(0.0, min(float(end), float(duration)))
    if not start < end:
        raise ValueError(f"invalid temporal span after clamp: {start} >= {end}, duration={duration}")
    # Source intervals are integer-second boundaries. ceil prevents a fractional end from collapsing.
    start_i = int(math.floor(start))
    end_i = int(math.ceil(end))
    if end_i <= start_i:
        end_i = start_i + 1
    end_i = min(end_i, int(math.ceil(duration)))
    if end_i <= start_i:
        raise ValueError(f"invalid serialized temporal span: {start_i} >= {end_i}")
    return {
        "start_time": sec_to_hhmmss(start_i),
        "end_time": sec_to_hhmmss(end_i),
        "caption": str(caption or "").strip(),
    }


class _CommonRetrieveContract:
    name = "visual_retrieve"

    @property
    def json(self) -> dict[str, Any]:
        return json.loads(json.dumps(RETRIEVE_SCHEMA))


class UnifiedFlatVisualRetrieveTool(_CommonRetrieveContract, VisualRetrieveAliasTool):
    """Original VideoSEAL flat retrieval with a frozen Planner-visible contract."""

    top_k = 6

    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        super().__init__(name=name or "visual_retrieve", description=description or RETRIEVE_DESCRIPTION)

    def forward(self, **kwargs: Any) -> ToolOutput:
        kwargs["top_k"] = int(kwargs.get("top_k") or self.top_k)
        result = super().forward(**kwargs)
        if result.error is not None:
            return result
        if not isinstance(result.output, list):
            return ToolOutput(self.name, error=f"flat backend expected span list, got {type(result.output).__name__}")
        spans = []
        for row in result.output:
            if not isinstance(row, dict):
                continue
            spans.append(
                {
                    "start_time": str(row.get("start_time") or ""),
                    "end_time": str(row.get("end_time") or ""),
                    "caption": str(row.get("caption") or ""),
                }
            )
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "retrieval_backend": "videoseal_flat",
                "candidate_count": len(spans),
                "planner_visible_contract": "list[start_time,end_time,caption]",
            }
        )
        return ToolOutput(self.name, output=spans, metadata=metadata)


class HierarchicalVisualRetrieveTool(_CommonRetrieveContract, Tool):
    """Natural-language Coarse -> Medium -> Fine adapter for HourVideo maps."""

    coarse_top_k = 3
    medium_top_k = 3
    fine_per_medium = 2
    output_top_k = 6
    min_time_gap_sec = 0.0
    visual_weight = 0.6
    lexical_weight = 0.3
    siglip_model = "google/siglip-base-patch16-224"
    siglip_cache_dir = "/cs/student/project_msc/2025/rai/xinanx01/HourVideo/model_cache"
    siglip_max_text_tokens = 64

    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        super().__init__(name=name or "visual_retrieve", description=description or RETRIEVE_DESCRIPTION)

    def forward(
        self,
        *,
        query: str,
        top_k: Optional[int] = None,
        video_id: Optional[str] = None,
        index_path: Optional[str] = None,
        original_question: Optional[str] = None,
    ) -> ToolOutput:
        del original_question
        try:
            query = str(query or "").strip()
            if not query:
                return ToolOutput(self.name, error="query must be non-empty")
            if not index_path:
                return ToolOutput(self.name, error="hierarchical backend requires index_path")
            case_dir = _resolve_case_dir(index_path, video_id)
            navigation = _read_json(case_dir / "r3_2_navigation_map.json")
            hierarchy = _read_json(case_dir / "shared_hierarchy.json")
            captions_list = _read_json(case_dir / "r3_medium_captions.json")
            captions = {str(row["medium_id"]): row for row in captions_list}
            medium_by_id = {str(row["medium_id"]): row for row in hierarchy["medium_nodes"]}
            duration = float(hierarchy["duration_sec"])

            coarse_ranked = []
            for node in navigation["coarse_regions"]:
                score, terms = lexical_similarity(query, _coarse_text(node))
                coarse_ranked.append((float(score), -float(node["start_sec"]), str(node["coarse_id"]), terms, node))
            coarse_ranked.sort(reverse=True)
            selected_coarse = coarse_ranked[: max(1, int(self.coarse_top_k))]
            allowed_medium_ids = {
                str(medium_id)
                for _score, _start, _cid, _terms, node in selected_coarse
                for medium_id in node.get("source_medium_ids") or []
            }
            if not allowed_medium_ids:
                raise ValueError("selected Coarse regions contain no Medium children")

            case_cfg = _case_config(case_dir)
            encoder = _encoder(self.siglip_model, self.siglip_cache_dir, self.siglip_max_text_tokens)
            encode_started = time.monotonic()
            query_embedding = np.asarray(encoder.encode([query])[0], dtype=np.float32)
            query_encode_sec = time.monotonic() - encode_started
            medium_embeddings = np.load(case_dir / "medium_siglip.float32.npy", allow_pickle=False).astype(np.float32)
            medium_embeddings /= np.maximum(np.linalg.norm(medium_embeddings, axis=1, keepdims=True), 1e-12)

            medium_candidates = []
            indices = []
            for index, medium in enumerate(hierarchy["medium_nodes"]):
                if str(medium["medium_id"]) in allowed_medium_ids:
                    indices.append(index)
                    medium_candidates.append(medium)
            raw_scores = np.asarray([float(medium_embeddings[index] @ query_embedding) for index in indices])
            visual_scores = _normalize(raw_scores)
            medium_ranked = []
            for position, medium in enumerate(medium_candidates):
                medium_id = str(medium["medium_id"])
                caption = str(captions.get(medium_id, {}).get("qwen_caption") or "")
                lexical_score, terms = lexical_similarity(query, caption)
                combined = self.visual_weight * float(visual_scores[position]) + self.lexical_weight * lexical_score
                medium_ranked.append(
                    {
                        "medium_id": medium_id,
                        "start_sec": float(medium["start_sec"]),
                        "end_sec": float(medium["end_sec"]),
                        "visual_score_raw": float(raw_scores[position]),
                        "visual_score_normalized": float(visual_scores[position]),
                        "lexical_score": float(lexical_score),
                        "matched_terms": terms,
                        "combined_score": float(combined),
                    }
                )
            medium_ranked.sort(key=lambda row: (-row["combined_score"], row["start_sec"], row["medium_id"]))
            selected_medium = medium_ranked[: max(1, int(self.medium_top_k))]

            fine_by_id, fine_embeddings, row_by_id = _load_fine_registry(hierarchy, case_cfg)
            fine_candidates = []
            for medium_rank, medium_row in enumerate(selected_medium, start=1):
                medium_id = str(medium_row["medium_id"])
                medium = dict(medium_by_id[medium_id])
                medium["child_fine_ids"] = list(medium["source_fine_ids"])
                ranking = rerank_fines_for_medium(
                    question_id=str(video_id or case_dir.name),
                    search_unit_id=query,
                    medium=medium,
                    fine_by_id=fine_by_id,
                    fine_embeddings=fine_embeddings,
                    row_by_id=row_by_id,
                    query_embedding=query_embedding,
                )
                chosen, reason = select_diverse_fines(
                    {query: ranking},
                    strategy="top_relevance_then_temporal_diversity",
                    max_fines=max(1, int(self.fine_per_medium)),
                    minimum_gap=10.0,
                )
                for fine in chosen:
                    fine_candidates.append(
                        {
                            **fine,
                            "medium_rank": medium_rank,
                            "medium_combined_score": medium_row["combined_score"],
                            "one_fine_reason": reason,
                            "caption": str(captions.get(medium_id, {}).get("qwen_caption") or ""),
                        }
                    )
            fine_candidates.sort(
                key=lambda row: (
                    row["medium_rank"],
                    -float(row["siglip_score_raw"]),
                    float(row["start_sec"]),
                    str(row["fine_id"]),
                )
            )

            limit = max(1, int(top_k or self.output_top_k))
            deduped = []
            seen_windows = set()
            for row in fine_candidates:
                key = (float(row["start_sec"]), float(row["end_sec"]))
                if key in seen_windows:
                    continue
                seen_windows.add(key)
                deduped.append(row)
            keep = select_time_diverse_windows(
                [(float(row["start_sec"]), float(row["end_sec"])) for row in deduped],
                k=limit,
                min_gap_sec=max(0.0, float(self.min_time_gap_sec)),
            )
            selected_fine = [deduped[index] for index in keep[:limit]]
            spans = [
                _canonical_span(row["start_sec"], row["end_sec"], duration, row["caption"])
                for row in selected_fine
            ]
            if not spans:
                raise ValueError("hierarchical retrieval produced no inspectable spans")
            metadata = {
                "retrieval_backend": "hierarchical",
                "adapter": "hourvideo_coarse_medium_fine_siglip_v1",
                "query": query,
                "video_id": str(video_id or case_dir.name),
                "video_duration_sec": duration,
                "candidate_count": len(spans),
                "planner_visible_contract": "list[start_time,end_time,caption]",
                "audio_used": False,
                "query_encode_sec": round(query_encode_sec, 6),
                "selected_coarse": [
                    {
                        "coarse_id": coarse_id,
                        "score": score,
                        "matched_terms": terms,
                        "start_sec": float(node["start_sec"]),
                        "end_sec": float(node["end_sec"]),
                    }
                    for score, _start, coarse_id, terms, node in selected_coarse
                ],
                "selected_medium": selected_medium,
                "selected_fine": [
                    {
                        "fine_id": row["fine_id"],
                        "medium_id": row["medium_id"],
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "timestamp_sec": float(row["representative_frame_timestamp_sec"]),
                        "siglip_score_raw": float(row["siglip_score_raw"]),
                        "selection_reason": row.get("selection_reason"),
                    }
                    for row in selected_fine
                ],
            }
            return ToolOutput(self.name, output=spans, metadata=metadata)
        except Exception as exc:
            return ToolOutput(self.name, error=f"{type(exc).__name__}: {exc}")


_INSPECT_BUDGET: contextvars.ContextVar[Optional[dict[str, int]]] = contextvars.ContextVar(
    "videoseal_inspect_budget", default=None
)


@contextmanager
def inspect_budget(total_frames: int, per_call_frames: int) -> Iterator[dict[str, int]]:
    state = {"total": int(total_frames), "per_call": int(per_call_frames), "used": 0, "calls": 0}
    token = _INSPECT_BUDGET.set(state)
    try:
        yield state
    finally:
        _INSPECT_BUDGET.reset(token)


class BudgetedVisualInspectTool(VisualInspectAliasTool):
    """Original Inspector with a shared per-question cumulative frame cap."""

    def forward(self, **kwargs: Any) -> ToolOutput:
        state = _INSPECT_BUDGET.get()
        if state is None:
            return super().forward(**kwargs)
        remaining = int(state["total"] - state["used"])
        if remaining <= 0:
            return ToolOutput(self.name, error="cumulative Inspector frame budget exhausted")
        call_cap = min(int(state["per_call"]), remaining)
        previous = os.environ.get("INSPECT_MAX_TOTAL_IMAGES")
        os.environ["INSPECT_MAX_TOTAL_IMAGES"] = str(call_cap)
        try:
            result = super().forward(**kwargs)
        finally:
            if previous is None:
                os.environ.pop("INSPECT_MAX_TOTAL_IMAGES", None)
            else:
                os.environ["INSPECT_MAX_TOTAL_IMAGES"] = previous
        metadata = dict(result.metadata or {})
        used = int(metadata.get("image_count") or 0)
        state["used"] += used
        state["calls"] += 1
        metadata["cumulative_frame_budget"] = {
            "used": state["used"],
            "total": state["total"],
            "remaining": max(0, state["total"] - state["used"]),
            "per_call_cap": state["per_call"],
        }
        return ToolOutput(result.name, output=result.output, error=result.error, metadata=metadata)


def _configured_tool(base: Type[Tool], name: str, values: dict[str, Any]) -> Type[Tool]:
    return type(name, (base,), dict(values))


def build_tool_map(backend: str, cfg: dict[str, Any]) -> dict[str, Type[Tool]]:
    retrieval = cfg["retrieval"]
    if backend == "videoseal_flat":
        retrieve = _configured_tool(
            UnifiedFlatVisualRetrieveTool,
            "ConfiguredUnifiedFlatVisualRetrieveTool",
            {"top_k": int(retrieval["top_k"])},
        )
    elif backend == "hierarchical":
        siglip = cfg["siglip_text"]
        retrieve = _configured_tool(
            HierarchicalVisualRetrieveTool,
            "ConfiguredHierarchicalVisualRetrieveTool",
            {
                "coarse_top_k": int(retrieval["coarse_top_k"]),
                "medium_top_k": int(retrieval["medium_top_k"]),
                "fine_per_medium": int(retrieval["fine_per_medium"]),
                "output_top_k": int(retrieval["top_k"]),
                "min_time_gap_sec": float(retrieval["min_time_gap_sec"]),
                "visual_weight": float(retrieval["visual_weight"]),
                "lexical_weight": float(retrieval["lexical_weight"]),
                "siglip_model": str(siglip["model"]),
                "siglip_cache_dir": str(siglip["cache_dir"]),
                "siglip_max_text_tokens": int(siglip["max_text_tokens"]),
            },
        )
    else:
        raise ValueError(f"unknown retrieval_backend: {backend}")
    return {"visual_retrieve": retrieve, "visual_inspect": BudgetedVisualInspectTool}


def prepare_flat_smoke_indexes(cfg: dict[str, Any], source_cases: Path, flat_root: Path) -> dict[str, Any]:
    rows = []
    for uid in cfg["smoke_video_uids"]:
        case_dir = source_cases / uid
        hierarchy = _read_json(case_dir / "shared_hierarchy.json")
        captions = _read_json(case_dir / "r3_medium_captions.json")
        docs = {}
        for row in captions:
            start = int(math.floor(float(row["start_sec"])))
            end = int(math.ceil(float(row["end_sec"])))
            docs[f"{start}_{end}"] = {
                "caption": str(row.get("qwen_caption") or ""),
                "source": "hourvideo_medium_caption_flat_smoke_fixture",
                "video_id": uid,
            }
        out_dir = flat_root / uid
        _write_json(out_dir / "semantic_captions.json", docs)
        rows.append(
            {
                "video_uid": uid,
                "documents": len(docs),
                "duration_sec": float(hierarchy["duration_sec"]),
                "index_dir": str(out_dir),
                "formal_videoseal_offline_index": False,
            }
        )
    report = {
        "stage": "prepare_flat_smoke_indexes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_policy": "existing HourVideo Medium captions projected into VideoSEAL semantic_captions schema",
        "rows": rows,
        "ok": len(rows) == len(cfg["smoke_video_uids"]),
    }
    _write_json(flat_root.parent / "prepare_flat_smoke_indexes.json", report)
    return report


def _validate_spans(spans: object, duration: float, top_k: int) -> list[str]:
    errors = []
    if not isinstance(spans, list):
        return [f"output is not a list: {type(spans).__name__}"]
    if len(spans) > top_k:
        errors.append(f"candidate count {len(spans)} exceeds top_k {top_k}")
    seen = set()
    for index, row in enumerate(spans):
        if not isinstance(row, dict):
            errors.append(f"candidate {index} is not an object")
            continue
        if set(row) != {"start_time", "end_time", "caption"}:
            errors.append(f"candidate {index} fields differ: {sorted(row)}")
        try:
            start = float(srt_timestamp_to_seconds(str(row.get("start_time") or "")))
            end = float(srt_timestamp_to_seconds(str(row.get("end_time") or "")))
        except Exception as exc:
            errors.append(f"candidate {index} time parse failed: {exc}")
            continue
        if not 0 <= start < end <= duration:
            errors.append(f"candidate {index} invalid interval: {start}, {end}, duration={duration}")
        key = (start, end)
        if key in seen:
            errors.append(f"candidate {index} duplicates interval {key}")
        seen.add(key)
    return errors


def contract_test(cfg: dict[str, Any], source_cases: Path, flat_root: Path, output_root: Path) -> dict[str, Any]:
    old = {
        name: os.environ.get(name)
        for name in ("SEMANTIC_RETRIEVE_MIX", "VISUAL_RETRIEVE_SUMMARY_ENABLED", "VISUAL_RETRIEVE_RETURN_SPANS")
    }
    os.environ.update(
        {
            "SEMANTIC_RETRIEVE_MIX": "bm25",
            "VISUAL_RETRIEVE_SUMMARY_ENABLED": "0",
            "VISUAL_RETRIEVE_RETURN_SPANS": "1",
        }
    )
    rows = []
    try:
        schemas = {backend: build_tool_map(backend, cfg)["visual_retrieve"]().json for backend in cfg["retrieval_backends"]}
        for uid in cfg["smoke_video_uids"]:
            case = source_cases / uid
            question = _read_json(case / "question_input.json")
            duration = float(_read_json(case / "shared_hierarchy.json")["duration_sec"])
            for backend in cfg["retrieval_backends"]:
                index_path = (flat_root / uid) if backend == "videoseal_flat" else source_cases
                tool = build_tool_map(backend, cfg)["visual_retrieve"]()
                result = tool(
                    query=str(question["question_text"]),
                    video_id=uid,
                    index_path=str(index_path),
                    original_question=str(question["question_text"]),
                )
                errors = [result.error] if result.error else []
                errors.extend(_validate_spans(result.output, duration, int(cfg["retrieval"]["top_k"])))
                if isinstance(result.output, list) and not result.output:
                    errors.append("retrieval returned no candidate spans")
                rows.append(
                    {
                        "video_uid": uid,
                        "backend": backend,
                        "error": result.error,
                        "output": result.output,
                        "metadata": result.metadata,
                        "validation_errors": errors,
                        "ok": not errors,
                    }
                )
        report = {
            "stage": "contract_test",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schemas": schemas,
            "tool_schema_identical": len({json.dumps(value, sort_keys=True) for value in schemas.values()}) == 1,
            "rows": rows,
        }
        report["ok"] = report["tool_schema_identical"] and all(row["ok"] for row in rows)
        _write_json(output_root / "contract_test.json", report)
        return report
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _mcq_prompt(question: dict[str, Any]) -> str:
    lines = [str(question["question_text"]).strip()]
    lines.extend(f"{row['option_id']}. {row['text']}" for row in question.get("answer_options") or [])
    return "\n".join(lines)


@contextmanager
def _frozen_environment(cfg: dict[str, Any]) -> Iterator[None]:
    planner = cfg["planner"]
    inspector = cfg["inspector"]
    values = {
        "AGENT_LLM_API_BASE": str(planner["api_base"]),
        "AGENT_LLM_API_KEY": str(planner["api_key"]),
        "AGENT_LLM_MODEL": str(planner["served_model"]),
        "AGENT_MLLM_BACKEND": "openai",
        "AGENT_API_USE_MESSAGES": str(int(planner["api_use_messages"])),
        "AGENT_LLM_MAX_TOKENS": str(planner["max_tokens"]),
        "AGENT_LLM_TEMPERATURE": str(planner["temperature"]),
        "AGENT_LLM_TIMEOUT": str(planner["timeout_sec"]),
        "AGENT_ENFORCE_INSPECTOR_ANSWER_GATE": "1",
        "AGENT_FORCE_LAST_STEP_VISUAL_INSPECT": "0",
        "AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK": "0",
        "AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK": "0",
        "VISUAL_INSPECT_API_BASE": str(inspector["api_base"]),
        "VISUAL_INSPECT_API_KEY": str(inspector["api_key"]),
        "VISUAL_INSPECT_MODEL": str(inspector["served_model"]),
        "VISUAL_INSPECT_BACKEND": "openai",
        "VISUAL_INSPECT_PROMPT_MODE": "evidence_gate",
        "INSPECT_FPS": str(inspector["fps"]),
        "INSPECT_MAX_TOTAL_IMAGES": str(inspector["per_call_frame_budget"]),
        "INSPECT_MAX_LONG_EDGE": str(inspector["max_long_edge"]),
        "INSPECT_VLM_MAX_TOKENS": str(inspector["max_tokens"]),
        "INSPECT_VLM_TEMPERATURE": str(inspector["temperature"]),
        "INSPECT_GLOBAL_ORDER": "1",
        "VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE": "0",
        "INSPECT_DYNAMIC_MAX_LONG_EDGE": "0",
        "MLLM_RETRY_TIMES": str(cfg["retry_times"]),
        "MLLM_TIMEOUT": str(max(float(planner["timeout_sec"]), float(inspector["timeout_sec"]))),
        "SEMANTIC_RETRIEVE_MIX": "bm25",
        "VISUAL_RETRIEVE_SUMMARY_ENABLED": "0",
        "RETRIEVE_SUMMARY_ENABLED": "0",
        "VISUAL_RETRIEVE_RETURN_SPANS": "1",
        "VISUAL_RETRIEVE_TOPK": str(cfg["retrieval"]["top_k"]),
        "RETRIEVE_MIN_TIME_GAP_SEC": str(cfg["retrieval"]["min_time_gap_sec"]),
        "SUBTITLE_PATH": "",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _service_model(api_base: str) -> dict[str, Any]:
    import requests

    response = requests.get(str(api_base).rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    data = response.json().get("data") or []
    if len(data) != 1:
        raise RuntimeError(f"expected exactly one served model at {api_base}, got {len(data)}")
    return dict(data[0])


def _trajectory_summary(trajectory: dict[str, Any], backend: str) -> dict[str, Any]:
    actions = []
    retrieval_count = 0
    inspector_count = 0
    total_frames = 0
    saw_search_more = False
    parser_errors = []
    for index, step in enumerate(trajectory.get("steps") or [], start=1):
        action = step.get("action") if isinstance(step, dict) else None
        observation = step.get("observation") if isinstance(step, dict) else None
        model_response = str(step.get("model_response") or "")
        row = {"step": index, "action": action, "model_response": model_response, "timing": step.get("timing")}
        if not isinstance(action, dict) and "<final>" not in model_response:
            parser_errors.append(f"step {index}: no parsed tool call or final answer")
        if isinstance(observation, dict):
            name = str(observation.get("name") or "")
            row["tool"] = name
            row["ok"] = observation.get("ok")
            row["output"] = observation.get("output")
            row["metadata"] = observation.get("metadata")
            row["error"] = observation.get("error")
            if observation.get("error"):
                parser_errors.append(str(observation["error"]))
            if name == "visual_retrieve":
                retrieval_count += 1
            elif name == "visual_inspect":
                inspector_count += 1
                metadata = observation.get("metadata") or {}
                total_frames += int(metadata.get("image_count") or 0)
                answer = str((observation.get("output") or {}).get("answer") or "") if isinstance(observation.get("output"), dict) else str(observation.get("output") or "")
                if "SEARCH_MORE" in answer.upper():
                    saw_search_more = True
        actions.append(row)
    answer = str(trajectory.get("answer") or "")
    question = str(trajectory.get("question") or "")
    pred = parse_choice_letter_smart(answer, question) or parse_choice_letter(answer) or ""
    return {
        "backend": backend,
        "actions": actions,
        "retrieval_count": retrieval_count,
        "inspector_count": inspector_count,
        "total_frames": total_frames,
        "saw_search_more": saw_search_more,
        "continued_after_search_more": saw_search_more and retrieval_count > 1,
        "final_answer": answer,
        "parsed_answer": pred,
        "latency_sec": trajectory.get("elapsed_sec"),
        "timeout_or_parser_errors": parser_errors,
    }


def run_paired_smoke(cfg: dict[str, Any], source_cases: Path, flat_root: Path, output_root: Path) -> dict[str, Any]:
    from videoseal.agents.tool_agent import SimpleToolAgent

    planner_model = _service_model(str(cfg["planner"]["api_base"]))
    inspector_model = _service_model(str(cfg["inspector"]["api_base"]))
    expected_planner_path = str(_resolve(cfg["planner"]["checkpoint_path"], root=Path("/")))
    if Path(str(planner_model.get("root") or "")).resolve() != Path(expected_planner_path).resolve():
        raise RuntimeError(
            f"refusing smoke: Planner service root={planner_model.get('root')!r}, expected={expected_planner_path!r}"
        )
    rows = []
    with _frozen_environment(cfg):
        for uid in cfg["smoke_video_uids"]:
            case_dir = source_cases / uid
            question = _read_json(case_dir / "question_input.json")
            case_cfg = _case_config(case_dir)
            video_path = Path(str(case_cfg["video_path"])).resolve()
            for backend in cfg["retrieval_backends"]:
                tools = build_tool_map(backend, cfg)
                index_path = (flat_root / uid) if backend == "videoseal_flat" else source_cases
                backend_root = output_root / "smoke" / backend / uid
                with inspect_budget(
                    int(cfg["inspector"]["per_question_frame_budget"]),
                    int(cfg["inspector"]["per_call_frame_budget"]),
                ) as budget:
                    agent = SimpleToolAgent(tools=tools, llm_backend="api")
                    result = agent.run(
                        question=_mcq_prompt(question),
                        uid=str(question["question_id"]),
                        video_id=uid,
                        video_path=str(video_path),
                        visual_index=str(index_path),
                        groundtruth=None,
                        max_steps=int(cfg["max_steps"]),
                        save_dir=str(backend_root),
                        prompt_type=int(cfg["inspector"]["prompt_type"]),
                    )
                trajectory_path = backend_root / str(result["run_id"]) / "trajectory.json"
                trajectory = _read_json(trajectory_path)
                summary = _trajectory_summary(trajectory, backend)
                summary.update(
                    {
                        "video_uid": uid,
                        "question_id": question["question_id"],
                        "trajectory_path": str(trajectory_path),
                        "agent_result": result,
                        "frame_budget_state": budget,
                    }
                )
                _write_json(backend_root / "smoke_summary.json", summary)
                rows.append(summary)
    report = {
        "stage": "paired_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "planner_checkpoint_id": "CewEhao/VideoSEAL_8B",
        "planner_checkpoint_path": expected_planner_path,
        "planner_service": planner_model,
        "inspector_service": inspector_model,
        "rows": rows,
        "paired_cases": len(cfg["smoke_video_uids"]),
        "ok": all(row["retrieval_count"] >= 1 and row["inspector_count"] >= 1 for row in rows),
    }
    report_name = (
        f"smoke_report_{cfg['retrieval_backends'][0]}.json"
        if len(cfg["retrieval_backends"]) == 1
        else "paired_smoke_report.json"
    )
    _write_json(output_root / report_name, report)
    return report


def load_config(path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    cfg = _read_json(path)
    source_cases = _resolve(cfg["source_cases_root"])
    output_root = _resolve(cfg["output_root"])
    flat_root = output_root / "indexes" / "videoseal_flat_smoke"
    return cfg, source_cases, flat_root, output_root


def preflight(config_path: Path) -> dict[str, Any]:
    cfg, source_cases, flat_root, output_root = load_config(config_path)
    errors = []
    planner_path = _resolve(cfg["planner"]["checkpoint_path"], root=Path("/"))
    if not (planner_path / "model.safetensors.index.json").is_file():
        errors.append(f"trained Planner checkpoint missing: {planner_path}")
    if cfg["planner"]["checkpoint_id"] != "CewEhao/VideoSEAL_8B":
        errors.append("Planner checkpoint_id must be CewEhao/VideoSEAL_8B")
    if cfg["retrieval_backends"] != ["videoseal_flat", "hierarchical"]:
        errors.append("retrieval_backends must explicitly contain flat then hierarchical")
    case_rows = []
    for uid in cfg["smoke_video_uids"]:
        case = source_cases / uid
        required = [
            case / "question_input.json",
            case / "shared_hierarchy.json",
            case / "r3_2_navigation_map.json",
            case / "r3_medium_captions.json",
            case / "medium_siglip.float32.npy",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        case_cfg_path = case.parents[1] / "case_configs" / f"{uid}.json"
        if not case_cfg_path.is_file():
            missing.append(str(case_cfg_path))
        else:
            case_cfg = _read_json(case_cfg_path)
            for key in ("video_path", "siglip_npz"):
                if not Path(str(case_cfg[key])).is_file():
                    missing.append(str(case_cfg[key]))
        if missing:
            errors.extend(missing)
        case_rows.append({"video_uid": uid, "missing": missing, "ok": not missing})
    report = {
        "stage": "preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "videoseal_root": str(VIDEOSEAL_ROOT),
        "planner_checkpoint_path": str(planner_path),
        "source_cases_root": str(source_cases),
        "flat_smoke_index_root": str(flat_root),
        "output_root": str(output_root),
        "cases": case_rows,
        "errors": errors,
        "ok": not errors,
    }
    _write_json(output_root / "preflight.json", report)
    return report
