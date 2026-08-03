from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.r3_2_frozen_method_staged_av_organizer.core import (
    build_stage2_payload,
    no_api_tests,
    validate_stage1,
    validate_stage2,
)
from experiments.r3_v2_coarse_semantic_organizer.core import load_json


INDEX = load_json(ROOT / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json")
CONFIG = load_json(ROOT / "configs/experiments/r3_2_frozen_method_staged_av_organizer_canary_v1.json")


def stage1_fixture() -> dict:
    phases = []
    start = 0
    for i, end in enumerate([4, 9, 14, 19, 24, 29]):
        phases.append({"end_medium_index": end, "phase_label": f"phase {i}", "boundary_reason": "meaningful change", "salient_medium_indices": [start], "salient_audio_ids": []})
        start = end + 1
    return {"phases": phases}


def test_no_api_suite_passes() -> None:
    report = no_api_tests(INDEX, CONFIG)
    assert report["status"] == "passed", report


def test_stage1_has_global_semantics_but_no_mechanical_ids() -> None:
    value = stage1_fixture()
    assert validate_stage1(value, INDEX, CONFIG)["valid"]
    assert "phase_id" not in str(value)
    assert "coarse_id" not in str(value)


def test_stage2_inherits_global_phase_record() -> None:
    phase = stage1_fixture()["phases"][0]
    payload = build_stage2_payload(phase, 0, 0, INDEX, 1)
    assert payload["fixed_phase"]["phase_label"] == phase["phase_label"]
    assert payload["fixed_phase"]["boundary_reason"] == phase["boundary_reason"]
    assert payload["neighbouring_context_visual_captions"]


def test_stage2_reference_validation() -> None:
    phase = stage1_fixture()["phases"][0]
    payload = build_stage2_payload(phase, 0, 0, INDEX, 1)
    value = {"direct_visual": [{"claim": "visible", "medium_indices": [0]}], "transcript_evidence": [], "av_interpretation": [], "phase_summary": "summary", "uncertainty": []}
    assert validate_stage2(value, payload)["valid"]
    value["direct_visual"][0]["medium_indices"] = [29]
    assert not validate_stage2(value, payload)["valid"]
