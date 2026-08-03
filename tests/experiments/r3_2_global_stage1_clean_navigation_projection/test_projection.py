from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.r3_2_global_stage1_clean_navigation_projection.core import planner_view, reconstruct, validate
from experiments.r3_v2_coarse_semantic_organizer.core import load_json


def test_clean_projection() -> None:
    index = load_json(ROOT / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json")
    stage1 = load_json(ROOT / "outputs/experiments/r3_2_frozen_exact_interleaved_av_organizer_canary_v1/stage1_global_phases.json")
    semantic_map = reconstruct(index, stage1)
    view = planner_view(semantic_map)
    report = validate(index, semantic_map, view)
    assert report["status"] == "passed", report
    assert all(row["summary"] == row["event_label"] for row in view["coarse_regions"])
