from __future__ import annotations

from experiments.r3_2_global_stage1_clean_navigation_map_freeze.core import EXPECTED_PHASES, validate_freeze_source


def _fixtures():
    medium_ids = [f"M{i:02d}" for i in range(1, 31)]
    fine_ids = [f"F{i:03d}" for i in range(1, 89)]
    coarse = []
    for i, label in enumerate(EXPECTED_PHASES):
        coarse.append({
            "navigation_summary": label,
            "source_medium_ids": medium_ids[i * 3 : (i + 1) * 3],
            "source_fine_ids": fine_ids[i * 9 : (i + 1) * 9] if i < 8 else (fine_ids[72:80] if i == 8 else fine_ids[80:88]),
        })
    semantic_map = {
        "coarse_regions": coarse,
        "storyline_events": [],
        "has_storyline": False,
        "hard_filtering_allowed": False,
        "provenance": {"historical_phase_output_used": False, "failed_stage2_text_used": False},
    }
    planner = {
        "coarse_regions": [{"summary": label} for label in EXPECTED_PHASES],
        "hard_filtering_allowed": False,
        "planner_policy": {"coarse_prior_affects_ranking": False, "all_mediums_remain_eligible": True},
    }
    validation = {"clean_projection_validation": "passed", "planner_compatibility_validation": "passed", "navigation_noise_audit": "passed"}
    return semantic_map, planner, validation


def test_valid_source_passes() -> None:
    semantic_map, planner, validation = _fixtures()
    assert validate_freeze_source(semantic_map, planner, validation)["status"] == "passed"


def test_historical_phase_dependency_fails() -> None:
    semantic_map, planner, validation = _fixtures()
    semantic_map["provenance"]["historical_phase_output_used"] = True
    assert validate_freeze_source(semantic_map, planner, validation)["status"] == "failed"


def test_hard_filtering_fails() -> None:
    semantic_map, planner, validation = _fixtures()
    planner["hard_filtering_allowed"] = True
    assert validate_freeze_source(semantic_map, planner, validation)["status"] == "failed"

