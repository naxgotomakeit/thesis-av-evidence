from __future__ import annotations

import copy

import pytest

from experiments.r3_v2_av_coarse_semantic_organizer_v1_1.core import infer_ranges, reconstruct, validate_first_pass, validate_reconstruction


def group(end: int) -> dict:
    return {"end_medium_index": end, "navigation_summary": "phase", "uncertainty_notes": []}


def fake_index() -> dict:
    mediums, fine = [], []
    for index in range(30):
        fine_id = f"F{index:02d}"
        mediums.append({"medium_id": f"M{index:02d}", "start_sec": float(index * 10), "end_sec": float((index + 1) * 10), "child_fine_ids": [fine_id], "qwen_caption": f"Caption {index}"})
        fine.append({"fine_id": fine_id})
    audio = [{"audio_id": f"A{index:03d}", "start_sec": float(index * 2), "end_sec": float(index * 2 + 1), "transcript": f"speech {index}", "source_type": "unclear", "fallback_used": False, "asr_status": "ok"} for index in range(107)]
    return {"medium_nodes": mediums, "fine_nodes": fine, "audio_nodes": audio}


def test_first_start_and_later_starts_are_deterministic() -> None:
    ranges = infer_ranges({"groups": [group(1), group(8), group(29)]})
    assert ranges[0]["start_medium_index"] == 0
    assert ranges[1]["start_medium_index"] == 2
    assert ranges[2]["start_medium_index"] == 9


def test_strictly_increasing_ends_pass() -> None:
    assert validate_first_pass({"groups": [group(1), group(8), group(29)]})["valid"]


@pytest.mark.parametrize("ends", [[1, 1, 29], [8, 7, 29], [1, 28], [1, 30], [-1, 29]])
def test_invalid_end_sequences_fail(ends: list[int]) -> None:
    assert not validate_first_pass({"groups": [group(end) for end in ends]})["valid"]


def test_empty_groups_fail() -> None:
    assert not validate_first_pass({"groups": []})["valid"]


def test_unknown_fields_fail() -> None:
    row = group(29); row["group_index"] = 0
    assert not validate_first_pass({"groups": [row]})["valid"]


def test_no_repair_contract_has_no_repair_fields() -> None:
    result = validate_first_pass({"groups": [group(29)]})
    assert result["valid"]
    assert "repair" not in result


def test_reconstruction_is_complete_and_deterministic() -> None:
    index = fake_index(); grouping = {"groups": [group(4), group(14), group(29)]}
    first, second = reconstruct(index, grouping), reconstruct(index, grouping)
    assert first == second
    validation = validate_reconstruction(index, first)
    assert validation["valid"] and validation["medium_count"] == 30 and validation["fine_count"] == 30
    assert first["storyline_events"] == [] and first["hard_filtering_allowed"] is False


def test_tampered_fine_or_audio_mapping_fails() -> None:
    index = fake_index(); value = reconstruct(index, {"groups": [group(29)]})
    bad = copy.deepcopy(value); bad["coarse_regions"][0]["source_fine_ids"].pop()
    assert not validate_reconstruction(index, bad)["valid"]
    bad = copy.deepcopy(value); bad["coarse_regions"][0]["exact_source_asr"][0]["transcript"] = "changed"
    assert not validate_reconstruction(index, bad)["valid"]
