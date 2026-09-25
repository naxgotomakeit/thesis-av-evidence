from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_v6_5_2_complete_fine_feedback_v1 import core as _v652
from experiments.hourvideo_v6_5_3_bounded_shared_v1 import core as _v653


FORMAL_VERSION = "v6.5.4"
PROMOTION_SOURCE = "hourvideo_v6_5_3_bounded_shared_v1"
OBSERVATION_PROJECTION = "lossless_semantic_fields_short_id_v1"

_BASE_CALL_SHARED = _v652._call_shared
_BASE_PREFLIGHT = _v653.preflight


def _compact_list(values: list[Any]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _project_shared_evidence(
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Project reviewed observations without model-based summarization or field loss.

    The full evidence rows remain owned by the resolver. Only the payload sent to Shared is
    projected; every short ID has a deterministic one-to-one canonical mapping.
    """
    projected: list[dict[str, Any]] = []
    short_to_canonical: dict[str, str] = {}
    observation_count = 0
    original_chars = 0
    projected_chars = 0
    for row in evidence:
        if row.get("evidence_type") != "reviewed_visual_observation":
            projected.append(copy.deepcopy(row))
            continue
        canonical_id = str(row["evidence_id"])
        short_id = f"V{observation_count:02d}"
        observation_count += 1
        short_to_canonical[short_id] = canonical_id
        content = json.loads(str(row["source_content"]))
        compact_content = " ".join((
            f"finding={content['finding']}",
            f"actions={_compact_list(content.get('visible_actions', []))}",
            f"objects={_compact_list(content.get('visible_objects', []))}",
            f"uncertainty={_compact_list(content.get('uncertainty_notes', []))}",
        ))
        original_chars += len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        compact_row = {
            "evidence_id": short_id,
            "evidence_type": "reviewed_visual_observation",
            "fine_id": str(row["fine_id"]),
            "timestamp_sec": float(row["timestamp_sec"]),
            "source_content": compact_content,
        }
        projected_chars += len(json.dumps(
            compact_row, ensure_ascii=False, separators=(",", ":"),
        ))
        projected.append(compact_row)
    diagnostics = {
        "projection": OBSERVATION_PROJECTION,
        "observation_count": observation_count,
        "original_observation_chars": original_chars,
        "projected_observation_chars": projected_chars,
        "character_reduction": original_chars - projected_chars,
    }
    return projected, short_to_canonical, diagnostics


def _call_shared_compact(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    excluded: list[dict[str, Any]], round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projected, short_to_canonical, projection = _project_shared_evidence(evidence)
    document, usage = _BASE_CALL_SHARED(
        cfg, question, projected, excluded, round_number,
    )
    document["cited_evidence_ids"] = [
        short_to_canonical.get(evidence_id, evidence_id)
        for evidence_id in document["cited_evidence_ids"]
    ]
    usage.update({
        "shared_observation_projection": OBSERVATION_PROJECTION,
        "shared_observation_projection_metrics": projection,
        "shared_observation_short_id_map": short_to_canonical,
    })
    return document, usage


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
        "shared_output_contract": _v653.SHARED_OUTPUT_CONTRACT,
        "observation_projection": OBSERVATION_PROJECTION,
        "shared_output_bounds": {
            "established_facts_max_length": _v653.ESTABLISHED_FACTS_MAX_LENGTH,
            "gap_reason_max_length": _v653.GAP_REASON_MAX_LENGTH,
            "cited_evidence_max_items": _v653.CITED_EVIDENCE_MAX_ITEMS,
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
        "observation_projection": OBSERVATION_PROJECTION,
    })
    write_json(root / cfg["output_root"] / "preflight.json", result)
    return result


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    original_call_shared = _v652._call_shared
    original_manifest = _v653._manifest
    original_preflight = _v653.preflight
    _v652._call_shared = _call_shared_compact
    _v653._manifest = _manifest
    _v653.preflight = preflight
    try:
        return _v653.run_live(root, config_path, video_uid=video_uid)
    finally:
        _v652._call_shared = original_call_shared
        _v653._manifest = original_manifest
        _v653.preflight = original_preflight


summarize = _v653.summarize
evaluate = _v653.evaluate
