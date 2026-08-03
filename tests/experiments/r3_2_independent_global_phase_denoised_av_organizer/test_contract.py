from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.r3_2_independent_global_phase_denoised_av_organizer.core import (
    build_selector_payload,
    build_summary_payload,
    infer_ranges,
    no_api_tests,
    validate_discovery,
    validate_selection,
)
from experiments.r3_v2_coarse_semantic_organizer.core import load_json


INDEX = load_json(
    ROOT
    / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json"
)


def test_no_api_contract_suite_passes() -> None:
    report = no_api_tests(INDEX, 3, 6)
    assert report["status"] == "passed", report


def test_discovery_is_end_indices_only_and_complete() -> None:
    good = {"groups": [{"end_medium_index": 9}, {"end_medium_index": 19}, {"end_medium_index": 29}]}
    assert validate_discovery(good)["valid"]
    assert infer_ranges(good) == [
        {"group_index": 0, "start_medium_index": 0, "end_medium_index": 9},
        {"group_index": 1, "start_medium_index": 10, "end_medium_index": 19},
        {"group_index": 2, "start_medium_index": 20, "end_medium_index": 29},
    ]
    bad = copy.deepcopy(good)
    bad["groups"][0]["label"] = "not allowed"
    assert not validate_discovery(bad)["valid"]


def test_selector_budget_and_phase_scope_are_enforced() -> None:
    discovery = {"groups": [{"end_medium_index": 4}, {"end_medium_index": 29}]}
    phase = infer_ranges(discovery)[0]
    payload = build_selector_payload(phase, INDEX, 3, 6)
    value = {
        "representative_medium_indices": [0, 2],
        "representative_audio_ids": [],
        "rejected_navigation_noise_medium_indices": [1],
        "rejected_navigation_noise_audio_ids": [],
        "selection_uncertainty": [],
    }
    assert validate_selection(value, phase, payload, 3, 6)["valid"]
    value["representative_medium_indices"] = [0, 1, 2, 3]
    assert not validate_selection(value, phase, payload, 3, 6)["valid"]


def test_rejected_caption_never_reaches_summary_payload() -> None:
    discovery = {"groups": [{"end_medium_index": 4}, {"end_medium_index": 29}]}
    ranges = infer_ranges(discovery)
    selections = [
        {
            "representative_medium_indices": [0],
            "representative_audio_ids": [],
            "rejected_navigation_noise_medium_indices": [1],
            "rejected_navigation_noise_audio_ids": [],
            "selection_uncertainty": [],
        },
        {
            "representative_medium_indices": [5],
            "representative_audio_ids": [],
            "rejected_navigation_noise_medium_indices": [],
            "rejected_navigation_noise_audio_ids": [],
            "selection_uncertainty": [],
        },
    ]
    payload = build_summary_payload(ranges, selections, INDEX)
    assert INDEX["medium_nodes"][1]["qwen_caption"] not in str(payload)
    assert "historical" not in str(payload).lower()
