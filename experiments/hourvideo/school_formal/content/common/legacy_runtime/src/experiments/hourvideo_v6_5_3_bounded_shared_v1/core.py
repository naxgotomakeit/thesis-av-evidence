from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_v6_5_2_complete_fine_feedback_v1 import core as _v652


FORMAL_VERSION = "v6.5.3"
PROMOTION_SOURCE = "hourvideo_v6_5_2_complete_fine_feedback_v1"
SHARED_OUTPUT_CONTRACT = "bounded_coherent_shared_report_v1"
ESTABLISHED_FACTS_MAX_LENGTH = 1536
GAP_REASON_MAX_LENGTH = 768
CITED_EVIDENCE_MAX_ITEMS = 16

_BASE_SHARED_SCHEMA = _v652._v65._v641._v64._pruned_shared_schema
_BASE_PREFLIGHT = _v652.preflight


def _bounded_shared_schema(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    excluded_coarse_ids: list[str],
) -> dict[str, Any]:
    """Keep the V6.5.2 contract while placing hard, task-level output bounds."""
    schema = _BASE_SHARED_SCHEMA(question, evidence, excluded_coarse_ids)
    properties = schema["properties"]
    properties["established_facts"]["maxLength"] = ESTABLISHED_FACTS_MAX_LENGTH
    properties["gap_reason"]["maxLength"] = GAP_REASON_MAX_LENGTH
    properties["cited_evidence_ids"]["maxItems"] = min(
        CITED_EVIDENCE_MAX_ITEMS, len(evidence)
    )
    return schema


def _manifest(
    root: Path, cfg: dict[str, Any], config_path: Path, video_uid: str | None,
) -> None:
    write_json(root / cfg["output_root"] / "run_manifest_latest.json", {
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formal_version": FORMAL_VERSION,
        "promotion_source": PROMOTION_SOURCE,
        "video_uid_filter": video_uid,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "module_sha256": sha256_file(Path(__file__)),
        "fine_feedback_contract": _v652.FINE_FEEDBACK_CONTRACT,
        "shared_output_contract": SHARED_OUTPUT_CONTRACT,
        "shared_output_bounds": {
            "established_facts_max_length": ESTABLISHED_FACTS_MAX_LENGTH,
            "gap_reason_max_length": GAP_REASON_MAX_LENGTH,
            "cited_evidence_max_items": CITED_EVIDENCE_MAX_ITEMS,
            "claim_max_tokens": int(cfg["anthropic"]["claim_max_tokens"]),
        },
        "visual_model": cfg["gemini"]["model"],
        "final_model": cfg["final"]["model"],
        "gold_loaded": False,
    })


def preflight(
    root: Path, config_path: Path, video_uid: str | None = None,
) -> dict[str, Any]:
    result = _BASE_PREFLIGHT(root, config_path, video_uid)
    cfg = load_json(config_path)
    result.update({
        "formal_version": FORMAL_VERSION,
        "promotion_source": PROMOTION_SOURCE,
        "shared_output_contract": SHARED_OUTPUT_CONTRACT,
        "shared_output_bounds": {
            "established_facts_max_length": ESTABLISHED_FACTS_MAX_LENGTH,
            "gap_reason_max_length": GAP_REASON_MAX_LENGTH,
            "cited_evidence_max_items": CITED_EVIDENCE_MAX_ITEMS,
            "claim_max_tokens": int(cfg["anthropic"]["claim_max_tokens"]),
        },
    })
    write_json(root / cfg["output_root"] / "preflight.json", result)
    return result


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    v64 = _v652._v65._v641._v64
    original_schema = v64._pruned_shared_schema
    original_manifest = _v652._manifest
    original_preflight = _v652.preflight
    v64._pruned_shared_schema = _bounded_shared_schema
    _v652._manifest = _manifest
    _v652.preflight = preflight
    try:
        return _v652.run_live(root, config_path, video_uid=video_uid)
    finally:
        v64._pruned_shared_schema = original_schema
        _v652._manifest = original_manifest
        _v652.preflight = original_preflight


summarize = _v652.summarize
evaluate = _v652.evaluate
