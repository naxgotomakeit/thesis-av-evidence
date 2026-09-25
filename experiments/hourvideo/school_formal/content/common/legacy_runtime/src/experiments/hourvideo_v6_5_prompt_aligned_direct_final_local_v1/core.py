from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Any

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_v6_4_1_prompt_aligned_direct_final_local_v1 import core as _v641


FORMAL_VERSION = "v6.5"
PROMOTION_SOURCE = "hourvideo_v6_4_1_prompt_aligned_direct_final_local_v1"
SHARED_INVESTIGATION_SYSTEM = _v641.SHARED_INVESTIGATION_SYSTEM
FINE_TARGET_CONTRACT = "question_options_established_context_and_gap_v1"
_fine_gap_target = _v641._fine_gap_target


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def preflight(
    root: Path, config_path: Path, video_uid: str | None = None,
) -> dict[str, Any]:
    result = _v641.preflight(root, config_path, video_uid)
    result.update({
        "formal_version": FORMAL_VERSION,
        "promotion_source": PROMOTION_SOURCE,
        "shared_prompt_sha256": _text_sha256(SHARED_INVESTIGATION_SYSTEM),
        "fine_target_contract": FINE_TARGET_CONTRACT,
    })
    cfg = load_json(config_path)
    write_json(root / cfg["output_root"] / "preflight.json", result)
    return result


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
        "promoted_v6_4_1_module_sha256": sha256_file(Path(_v641.__file__)),
        "shared_prompt_sha256": _text_sha256(SHARED_INVESTIGATION_SYSTEM),
        "fine_target_contract": FINE_TARGET_CONTRACT,
        "gold_loaded": False,
    })


def run_live(root: Path, config_path: Path, video_uid: str | None = None) -> dict[str, Any]:
    original_manifest = _v641._manifest
    original_base_preflight = _v641._v64.preflight
    _v641._manifest = _manifest
    _v641._v64.preflight = preflight
    try:
        return _v641.run_live(root, config_path, video_uid=video_uid)
    finally:
        _v641._manifest = original_manifest
        _v641._v64.preflight = original_base_preflight


summarize = _v641.summarize
evaluate = _v641.evaluate
