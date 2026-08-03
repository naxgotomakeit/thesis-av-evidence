from __future__ import annotations

import copy

import pytest

from experiments.r3_v2_coarse_semantic_organizer.core import (
    MODEL_OUTPUT_FIELDS,
    payload_leakage,
    planner_view,
    reconstruct,
    validate_grouping,
    validate_planner_view,
    validate_reconstruction,
)


def group(index: int, start: int, end: int) -> dict:
    return {
        "group_index": index,
        "start_medium_index": start,
        "end_medium_index": end,
        "navigation_summary": f"Phase {index}",
        "uncertainty_notes": [],
    }


def fake_index() -> dict:
    mediums = []
    fine = []
    for index in range(30):
        fine_id = f"F{index:02d}"
        mediums.append({
            "medium_id": f"M{index:02d}", "start_sec": float(index * 10), "end_sec": float((index + 1) * 10),
            "child_fine_ids": [fine_id], "qwen_caption": f"Caption {index}",
        })
        fine.append({"fine_id": fine_id})
    return {
        "medium_nodes": mediums, "fine_nodes": fine,
        "audio_nodes": [{"audio_id": "A01", "start_sec": 5.0, "end_sec": 15.0, "transcript": "x"}],
        "coarse_nodes": [], "storyline_events": [],
        "capabilities": {"has_storyline": False, "has_hierarchy": False},
    }


def test_valid_global_grouping_and_deterministic_reconstruction() -> None:
    value = {"groups": [group(0, 0, 9), group(1, 10, 29)]}
    assert validate_grouping(value)["valid"]
    index = fake_index()
    first = reconstruct(index, value)
    second = reconstruct(index, value)
    assert first == second
    assert validate_reconstruction(index, first)["valid"]
    assert first["storyline_events"] == []
    assert first["hard_filtering_allowed"] is False


@pytest.mark.parametrize(
    "value",
    [
        {"groups": []},
        {"groups": [group(0, 1, 29)]},
        {"groups": [group(0, 0, 10), group(1, 10, 29)]},
        {"groups": [group(0, 0, 9), group(2, 10, 29)]},
        {"groups": [group(0, 0, 28)]},
    ],
)
def test_invalid_coverage_order_overlap_and_indices_fail(value: dict) -> None:
    assert not validate_grouping(value)["valid"]


def test_extra_model_field_fails() -> None:
    row = group(0, 0, 29)
    row["coarse_id"] = "C01"
    assert list(row) != MODEL_OUTPUT_FIELDS
    assert not validate_grouping({"groups": [row]})["valid"]


def test_payload_leakage_detects_forbidden_channels() -> None:
    assert payload_leakage({"ordered_medium_caption_timeline": [{"medium_index": 0, "caption": "safe"}]}) == []
    assert payload_leakage({"question": "where?"}) == ["question"]
    assert payload_leakage({"audio": []}) == ["audio"]


def test_planner_view_is_non_filtering_and_complete() -> None:
    index = fake_index()
    semantic_map = reconstruct(index, {"groups": [group(0, 0, 14), group(1, 15, 29)]})
    view = planner_view(semantic_map)
    assert validate_planner_view(view, index)["valid"]
    assert validate_planner_view(view, index)["frozen_coarse_map_validator_valid"]
    assert validate_planner_view(view, index)["frozen_compatibility_adapter_loadable"]
    assert view["planner_policy"]["all_mediums_remain_eligible"] is True
    assert view["coarse_regions"][0]["coarse_summary"] == view["coarse_regions"][0]["summary"]


def test_planner_view_hard_filtering_or_missing_medium_fails() -> None:
    index = fake_index()
    view = planner_view(reconstruct(index, {"groups": [group(0, 0, 29)]}))
    bad = copy.deepcopy(view)
    bad["hard_filtering_allowed"] = True
    assert not validate_planner_view(bad, index)["valid"]
    bad = copy.deepcopy(view)
    bad["coarse_regions"][0]["source_medium_ids"].pop()
    assert not validate_planner_view(bad, index)["valid"]


def test_audio_half_open_overlap() -> None:
    index = fake_index()
    semantic_map = reconstruct(index, {"groups": [group(0, 0, 0), group(1, 1, 29)]})
    assert semantic_map["coarse_regions"][0]["audio_ids"] == ["A01"]
    assert semantic_map["coarse_regions"][1]["audio_ids"] == ["A01"]


def test_caption_tampering_is_detected() -> None:
    index = fake_index()
    semantic_map = reconstruct(index, {"groups": [group(0, 0, 29)]})
    semantic_map["coarse_regions"][0]["exact_source_captions"][0]["caption"] = "changed"
    assert not validate_reconstruction(index, semantic_map)["valid"]
