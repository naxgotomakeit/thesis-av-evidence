from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Type

from videoseal.prompts.visual_tool_prompts import build_visual_retrieve_summary_prompt
from videoseal.tools.base import Tool, ToolOutput
from videoseal.tools import visual_tools as reference_visual_tools
from videoseal.tools.visual_tools import VisualRetrieveAliasTool
from videoseal.utils.agent.env import build_mllm_client_from_env_prefix, env_flag, env_int_first
from videoseal.utils.video.tooling import parse_hhmmss_spans_from_text


RETRIEVE_DESCRIPTION = (
    "Fine-grained retrieval over the visual (LVBench) semantic index; optionally summarizes top hits and returns useful spans."
)


class _CommonRetrieveContract:
    name = "visual_retrieve"

    @property
    def json(self) -> dict[str, Any]:
        return {
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


def _required_root(name: str) -> Path:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        raise RuntimeError(f"{name} is required")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {root}")
    return root


class RuntimeAlignedFlatRetrieve(_CommonRetrieveContract, VisualRetrieveAliasTool):
    """Reference embedding retriever with only its index root made backend-selectable."""

    def __init__(self, name: str | None = None, description: str | None = None, function=None):
        super().__init__(name=name or "visual_retrieve", description=description or RETRIEVE_DESCRIPTION)

    def forward(self, **kwargs: Any) -> ToolOutput:
        started = time.monotonic()
        video_id = str(kwargs.get("video_id") or "").strip()
        if not video_id:
            return ToolOutput(self.name, error="flat retrieval requires video_id")
        kwargs["index_path"] = str(_required_root("PAIRED_FLAT_INDEX_ROOT") / video_id)
        kwargs["top_k"] = int(kwargs.get("top_k") or os.getenv("VISUAL_RETRIEVE_TOPK") or 30)
        result = super().forward(**kwargs)
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "retrieval_backend": "videoseal_flat",
                "config_name": "videoseal_flat_top30",
                "mode": "flat_embedding",
                "hierarchy_used": False,
                "hierarchy_complete": False,
                "retrieval_latency_sec": round(time.monotonic() - started, 6),
            }
        )
        return ToolOutput(self.name, output=result.output, error=result.error, metadata=metadata)


def _hierarchical_base() -> type[Tool]:
    # The experiment source tree is put on PYTHONPATH by the V7.3 launcher.
    # Importing here keeps the 48-file reference runtime unchanged except for
    # this explicit retrieval seam.
    from experiments.hourvideo_v7_1_videoseal_frozen_downstream_smoke_v1.core import (
        HierarchicalVisualRetrieveTool,
    )

    return HierarchicalVisualRetrieveTool


def _line_number(function: Callable[..., Any], marker: str) -> int:
    """Resolve an inherited implementation boundary without copying its logic."""
    source, first_line = inspect.getsourcelines(function)
    matches = [first_line + offset for offset, line in enumerate(source) if marker in line]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {marker!r} boundary in {function.__qualname__}, got {matches}")
    return matches[0]


def _run_with_hierarchy_stage_timing(
    function: Callable[..., ToolOutput], **kwargs: Any
) -> tuple[ToolOutput, dict[str, float]]:
    """Time existing Coarse/Medium/Fine blocks while leaving their behavior intact."""
    code = function.__code__
    boundaries = {
        "coarse": _line_number(function, "coarse_ranked = []"),
        "medium": _line_number(function, "case_cfg = _case_config(case_dir)"),
        "fine": _line_number(function, "fine_by_id, fine_embeddings, row_by_id = _load_fine_registry"),
        "done": _line_number(function, 'metadata = {'),
    }
    observed: dict[str, float] = {}
    previous_trace = sys.gettrace()

    def trace(frame, event, arg):  # type: ignore[no-untyped-def]
        if frame.f_code is code and event == "line":
            for name, line_number in boundaries.items():
                if frame.f_lineno == line_number and name not in observed:
                    observed[name] = time.monotonic()
        return trace

    sys.settrace(trace)
    try:
        result = function(**kwargs)
    finally:
        sys.settrace(previous_trace)

    timing: dict[str, float] = {}
    for stage, following in (("coarse", "medium"), ("medium", "fine"), ("fine", "done")):
        if stage in observed and following in observed:
            timing[f"{stage}_elapsed_sec"] = round(observed[following] - observed[stage], 6)
    return result, timing


def _hierarchy_provenance(
    *, result: ToolOutput, case_root: Path, video_id: str
) -> dict[str, Any]:
    """Join the existing selections into auditable Coarse->Medium->Fine paths."""
    metadata = dict(result.metadata or {})
    selected_coarse = [dict(row) for row in metadata.get("selected_coarse") or [] if isinstance(row, dict)]
    selected_medium = [dict(row) for row in metadata.get("selected_medium") or [] if isinstance(row, dict)]
    selected_fine = [dict(row) for row in metadata.get("selected_fine") or [] if isinstance(row, dict)]

    case_dir = case_root / video_id
    if not (case_dir / "r3_2_navigation_map.json").is_file():
        case_dir = case_root / "cases" / video_id
    navigation = json.loads((case_dir / "r3_2_navigation_map.json").read_text(encoding="utf-8"))
    selected_coarse_ids = {str(row.get("coarse_id") or "") for row in selected_coarse}
    medium_parents: dict[str, list[str]] = {}
    for coarse in navigation.get("coarse_regions") or []:
        coarse_id = str(coarse.get("coarse_id") or "")
        if coarse_id not in selected_coarse_ids:
            continue
        for medium_id in coarse.get("source_medium_ids") or []:
            medium_parents.setdefault(str(medium_id), []).append(coarse_id)

    coarse_by_id = {str(row.get("coarse_id") or ""): row for row in selected_coarse}
    medium_by_id = {str(row.get("medium_id") or ""): row for row in selected_medium}
    candidates = [dict(row) for row in result.output or [] if isinstance(row, dict)] if isinstance(result.output, list) else []
    paths: list[dict[str, Any]] = []
    for candidate_index, fine in enumerate(selected_fine):
        medium_id = str(fine.get("medium_id") or "")
        for coarse_id in medium_parents.get(medium_id, []):
            paths.append(
                {
                    "candidate_index": candidate_index,
                    "candidate": candidates[candidate_index] if candidate_index < len(candidates) else None,
                    "coarse": coarse_by_id.get(coarse_id),
                    "medium": medium_by_id.get(medium_id),
                    "fine": fine,
                }
            )
    return {
        "selected_coarse": selected_coarse,
        "selected_medium": selected_medium,
        "selected_fine": selected_fine,
        "coarse_medium_fine_paths": paths,
        "hierarchy_complete": bool(selected_coarse and selected_medium and selected_fine and paths),
    }


def _legacy_summarize_like_reference_unused(
    *, query: str, original_question: str, candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Historical V7.3 duplicate; retained only for audit and never called."""
    max_useful = env_int_first(("VISUAL_RETRIEVE_SUMMARY_MAX_SPANS", "RETRIEVE_SUMMARY_MAX_SPANS"), 5)
    if max_useful <= 0:
        max_useful = 5
    client = build_mllm_client_from_env_prefix("VISUAL_RETRIEVE_SUM")
    prompt = build_visual_retrieve_summary_prompt(
        query_text=query,
        user_question=original_question or query,
        candidates=candidates,
        max_useful=max_useful,
    )
    started = time.monotonic()
    text = client.generate_text(
        prompt,
        response_json=False,
        max_tokens=int(os.getenv("VISUAL_RETRIEVE_SUM_MAX_TOKENS") or "800"),
        temperature=float(os.getenv("VISUAL_RETRIEVE_SUM_TEMPERATURE") or "0.0"),
    ) or ""
    usage = client.get_last_usage() or {}
    allow = {
        (str(item.get("start_time") or ""), str(item.get("end_time") or ""))
        for item in candidates
    }
    parsed = [item for item in parse_hhmmss_spans_from_text(str(text)) if item in allow]
    selected = []
    wanted = set(parsed)
    for item in candidates:
        pair = (str(item.get("start_time") or ""), str(item.get("end_time") or ""))
        if pair in wanted:
            selected.append({"start_time": pair[0], "end_time": pair[1]})
            if len(selected) >= max_useful:
                break
    if not selected:
        selected = [
            {"start_time": str(item.get("start_time") or ""), "end_time": str(item.get("end_time") or "")}
            for item in candidates[:max_useful]
        ]
    output: dict[str, Any] = {
        "summary": str(text).strip()
        or f"Condensed visual retrieval: selected {len(selected)} spans; further verification may be needed."
    }
    if env_flag("VISUAL_RETRIEVE_RETURN_SPANS", default=False):
        output["useful_spans"] = selected
    metadata = {
        "summary_selected_spans": selected,
        "retrieval_telemetry_version": "v7.3-audit-v1",
        "raw_candidate_count": len(candidates),
        "raw_candidates": [dict(item) for item in candidates],
        "summarizer_useful_spans": [dict(item) for item in selected],
        "model_elapsed_sec": round(time.monotonic() - started, 6),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "visual_request_count": 1,
    }
    return output, metadata


def _make_hierarchical_type() -> Type[Tool]:
    base = _hierarchical_base()

    class RuntimeAlignedHierarchicalRetrieve(_CommonRetrieveContract, base):  # type: ignore[misc,valid-type]
        """Existing Coarse->Medium->Fine retrieval projected to reference output behavior."""

        def __init__(self, name: str | None = None, description: str | None = None, function=None):
            super().__init__(name=name or "visual_retrieve", description=description or RETRIEVE_DESCRIPTION)
            self.coarse_top_k = int(os.getenv("HIERARCHICAL_COARSE_TOPK") or "3")
            self.medium_top_k = int(os.getenv("HIERARCHICAL_MEDIUM_TOPK") or "3")
            self.fine_per_medium = int(os.getenv("HIERARCHICAL_FINE_PER_MEDIUM") or "2")
            self.output_top_k = int(os.getenv("VISUAL_RETRIEVE_TOPK") or "30")
            self.min_time_gap_sec = float(os.getenv("RETRIEVE_MIN_TIME_GAP_SEC") or "15")

        def forward(self, **kwargs: Any) -> ToolOutput:
            query = str(kwargs.get("query") or "").strip()
            original_question = str(kwargs.get("original_question") or "").strip()
            hierarchy_root = _required_root("PAIRED_HIERARCHICAL_ROOT")
            kwargs["index_path"] = str(hierarchy_root)
            kwargs["top_k"] = int(kwargs.get("top_k") or os.getenv("VISUAL_RETRIEVE_TOPK") or 30)
            video_id = str(kwargs.get("video_id") or "").strip()
            traversal_started = time.monotonic()
            result, stage_timing = _run_with_hierarchy_stage_timing(super().forward, **kwargs)
            traversal_elapsed_sec = round(time.monotonic() - traversal_started, 6)
            metadata = dict(result.metadata or {})
            metadata.update(
                {
                    "retrieval_backend": "hierarchical",
                    "hierarchy_used": False,
                    "hierarchy_stage_latency": stage_timing,
                    "hierarchy_traversal_elapsed_sec": traversal_elapsed_sec,
                }
            )
            if not result.error and video_id:
                try:
                    metadata.update(
                        _hierarchy_provenance(
                            result=result,
                            case_root=hierarchy_root,
                            video_id=video_id,
                        )
                    )
                except Exception as exc:
                    metadata["hierarchy_audit_error"] = f"{type(exc).__name__}: {exc}"
                expected_stages = {"coarse_elapsed_sec", "medium_elapsed_sec", "fine_elapsed_sec"}
                metadata["hierarchy_used"] = bool(
                    metadata.get("hierarchy_complete")
                    and expected_stages.issubset(stage_timing)
                )
            result = ToolOutput(self.name, output=result.output, error=result.error, metadata=metadata)
            if result.error or not isinstance(result.output, list):
                return result
            candidates = [dict(item) for item in result.output if isinstance(item, dict)]
            enabled = str(
                os.getenv("VISUAL_RETRIEVE_SUMMARY_ENABLED")
                or os.getenv("RETRIEVE_SUMMARY_ENABLED")
                or "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if not enabled:
                return ToolOutput(self.name, output=candidates, metadata=result.metadata)
            try:
                output, summary_meta = _legacy_summarize_like_reference_unused(
                    query=query, original_question=original_question, candidates=candidates
                )
            except Exception as exc:
                return ToolOutput(self.name, error=f"{type(exc).__name__}: {exc}", metadata=result.metadata)
            metadata = dict(result.metadata or {})
            metadata.update(summary_meta)
            metadata["planner_visible_summary"] = output.get("summary") if isinstance(output, dict) else output
            return ToolOutput(self.name, output=output, metadata=metadata)

    RuntimeAlignedHierarchicalRetrieve.__name__ = "RuntimeAlignedHierarchicalRetrieve"
    return RuntimeAlignedHierarchicalRetrieve


def build_retrieve_tool() -> Type[Tool]:
    backend = str(os.getenv("RETRIEVAL_BACKEND") or "").strip()
    if backend == "videoseal_flat":
        return RuntimeAlignedFlatRetrieve
    if backend == "hierarchical":
        from experiments.hourvideo_v7_4_variant_c_budgets_v1.retriever import (
            HierarchicalVisualRetrieveTool,
        )

        class RuntimeAlignedV74HierarchicalRetrieve(
            _CommonRetrieveContract, HierarchicalVisualRetrieveTool
        ):
            """Internal hierarchy candidates followed by the exact reference summarizer."""

            def forward(self, **kwargs: Any) -> ToolOutput:
                kwargs["index_path"] = str(_required_root("PAIRED_HIERARCHICAL_ROOT"))
                raw_result = super().forward(**kwargs)
                if raw_result.error or not isinstance(raw_result.output, list):
                    return raw_result
                candidates = [dict(item) for item in raw_result.output if isinstance(item, dict)]
                summarized = reference_visual_tools.summarize_visual_retrieval_candidates(
                    tool_name=self.name,
                    query=str(kwargs.get("query") or ""),
                    original_question=str(kwargs.get("original_question") or ""),
                    out_items=candidates,
                    return_spans=env_flag("VISUAL_RETRIEVE_RETURN_SPANS", default=False),
                    log_spans=env_flag("VISUAL_RETRIEVE_LOG_SPANS", default=False),
                )
                metadata = dict(raw_result.metadata or {})
                metadata.update(dict(summarized.metadata or {}))
                return ToolOutput(
                    self.name,
                    output=summarized.output,
                    error=summarized.error,
                    metadata=metadata,
                )

        RuntimeAlignedV74HierarchicalRetrieve.__name__ = "RuntimeAlignedV74HierarchicalRetrieve"
        return RuntimeAlignedV74HierarchicalRetrieve
    if backend == "dense_semantic_beam_b":
        from videoseal.tools.dense_beam_b.providers import get_retriever

        class RuntimeAlignedDenseSemanticBeamBRetrieve(_CommonRetrieveContract, Tool):
            def __init__(self, name: str | None = None, description: str | None = None, function=None):
                super().__init__(name=name or "visual_retrieve", description=description or RETRIEVE_DESCRIPTION)
                self.b = int(os.getenv("DENSE_BEAM_B") or "0")
                if self.b not in (6, 15, 30):
                    raise RuntimeError(f"DENSE_BEAM_B must be 6, 15, or 30; got {self.b}")
                self.retriever = get_retriever()

            def forward(self, **kwargs: Any) -> ToolOutput:
                query = str(kwargs.get("query") or "").strip()
                video_id = str(kwargs.get("video_id") or "").strip()
                original_question = str(kwargs.get("original_question") or "").strip()
                try:
                    raw = self.retriever.retrieve(video_id, query, self.b, original_question)
                except Exception as exc:
                    return ToolOutput(
                        self.name,
                        error=f"{type(exc).__name__}: {exc}",
                        metadata={"retrieval_backend": "dense_semantic_beam_b", "beam_b": self.b, "fallback_used": False},
                    )
                candidates = [dict(item) for item in raw["evidence"]]
                summarized = reference_visual_tools.summarize_visual_retrieval_candidates(
                    tool_name=self.name,
                    query=query,
                    original_question=original_question,
                    out_items=candidates,
                    return_spans=env_flag("VISUAL_RETRIEVE_RETURN_SPANS", default=False),
                    log_spans=env_flag("VISUAL_RETRIEVE_LOG_SPANS", default=False),
                )
                metadata = dict(summarized.metadata or {})
                metadata.update(raw["telemetry"])
                metadata.update({
                    "retrieval_backend": "dense_semantic_beam_b",
                    "config_name": f"dense_semantic_beam_b_h{self.b}",
                    "hierarchy_used": True,
                    "hierarchy_complete": True,
                    "medium_gate_applied": True,
                    "global_fine_fallback": False,
                    "caption_score_fallback": False,
                    "representative_frame_fallback": False,
                    "raw_candidate_count": len(candidates),
                    "raw_candidates": candidates,
                })
                return ToolOutput(self.name, output=summarized.output, error=summarized.error, metadata=metadata)

        RuntimeAlignedDenseSemanticBeamBRetrieve.__name__ = "RuntimeAlignedDenseSemanticBeamBRetrieve"
        return RuntimeAlignedDenseSemanticBeamBRetrieve
    raise RuntimeError(
        "RETRIEVAL_BACKEND must be exactly videoseal_flat, hierarchical, or dense_semantic_beam_b; "
        f"got {backend!r}"
    )
