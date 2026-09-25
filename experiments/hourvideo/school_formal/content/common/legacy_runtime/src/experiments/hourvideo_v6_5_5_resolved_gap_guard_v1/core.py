from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_v6_5_4_compact_observation_projection_v1 import core as _v654


FORMAL_VERSION = "v6.5.5"
PROMOTION_SOURCE = "hourvideo_v6_5_4_compact_observation_projection_v1"
SHARED_STATE_GUARD = "resolved_requires_empty_gap_v1"

_V64 = _v654._v653._v652._v65._v641._v64
_BASE_VALIDATE_SHARED = _V64._validate_pruned_shared
_BASE_CALL_SHARED_COMPACT = _v654._call_shared_compact
_BASE_PREFLIGHT = _v654.preflight
_BASE_SUMMARIZE = _v654.summarize
_BASE_EVALUATE = _v654.evaluate


def _validate_shared_state(
    value: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    allowed_coarse: set[str],
) -> None:
    """Add one state invariant without changing semantic evidence or answer logic."""
    _BASE_VALIDATE_SHARED(value, question, evidence, allowed_coarse)
    if (
        value["investigation_status"] == "resolved"
        and value["gap_reason"].strip()
    ):
        raise ValueError(
            "resolved Shared report must have empty gap_reason; if an answer-critical gap "
            "remains, return investigation_status=unresolved"
        )


def _call_shared_guarded(
    cfg: dict[str, Any], question: dict[str, Any], evidence: list[dict[str, Any]],
    excluded: list[dict[str, Any]], round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, usage = _BASE_CALL_SHARED_COMPACT(
        cfg, question, evidence, excluded, round_number,
    )
    errors = list(usage.get("previous_validation_errors", []))
    usage.update({
        "shared_state_guard": SHARED_STATE_GUARD,
        "shared_resolved_gap_violation_seen": any(
            "resolved Shared report must have empty gap_reason" in error
            for error in errors
        ),
    })
    return document, usage


def _configured_sides(cfg: dict[str, Any]) -> tuple[str, ...]:
    sides = tuple(cfg.get("execution_sides", _V64.SIDES))
    unknown = set(sides) - set(_V64.SIDES)
    if not sides or unknown:
        raise ValueError(f"invalid execution_sides: {sides}")
    return sides


def _manifest(
    root: Path, cfg: dict[str, Any], config_path: Path, video_uid: str | None,
) -> None:
    write_json(root / cfg["output_root"] / "run_manifest_latest.json", {
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formal_version": FORMAL_VERSION,
        "promotion_source": PROMOTION_SOURCE,
        "video_uid_filter": video_uid,
        "execution_sides": list(_configured_sides(cfg)),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "module_sha256": sha256_file(Path(__file__)),
        "fine_feedback_contract": _v654._v652.FINE_FEEDBACK_CONTRACT,
        "shared_output_contract": _v654._v653.SHARED_OUTPUT_CONTRACT,
        "shared_state_guard": SHARED_STATE_GUARD,
        "observation_projection": _v654.OBSERVATION_PROJECTION,
        "visual_model": cfg["gemini"]["model"],
        "final_model": cfg["final"]["model"],
        "gold_loaded": False,
    })


def preflight(
    root: Path, config_path: Path, video_uid: str | None = None,
) -> dict[str, Any]:
    cfg = load_json(config_path)
    original_sides = _V64.SIDES
    _V64.SIDES = _configured_sides(cfg)
    try:
        result = _BASE_PREFLIGHT(root, config_path, video_uid)
    finally:
        _V64.SIDES = original_sides
    result.update({
        "formal_version": FORMAL_VERSION,
        "promotion_source": PROMOTION_SOURCE,
        "execution_sides": list(_configured_sides(cfg)),
        "shared_state_guard": SHARED_STATE_GUARD,
    })
    write_json(root / cfg["output_root"] / "preflight.json", result)
    return result


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    original_validate = _V64._validate_pruned_shared
    original_sides = _V64.SIDES
    original_call_shared = _v654._call_shared_compact
    original_manifest = _v654._manifest
    original_preflight = _v654.preflight
    _V64._validate_pruned_shared = _validate_shared_state
    _V64.SIDES = _configured_sides(cfg)
    _v654._call_shared_compact = _call_shared_guarded
    _v654._manifest = _manifest
    _v654.preflight = preflight
    try:
        return _v654.run_live(root, config_path, video_uid=video_uid)
    finally:
        _V64._validate_pruned_shared = original_validate
        _V64.SIDES = original_sides
        _v654._call_shared_compact = original_call_shared
        _v654._manifest = original_manifest
        _v654.preflight = original_preflight


def summarize(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    original_sides = _V64.SIDES
    _V64.SIDES = _configured_sides(cfg)
    try:
        return _BASE_SUMMARIZE(root, config_path)
    finally:
        _V64.SIDES = original_sides


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    original_sides = _V64.SIDES
    _V64.SIDES = _configured_sides(cfg)
    try:
        return _BASE_EVALUATE(root, config_path)
    finally:
        _V64.SIDES = original_sides
