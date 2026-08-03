from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.r3_2_frozen_exact_interleaved_av_organizer.core import build_interleaved_nodes, no_api_tests
from experiments.r3_v2_coarse_semantic_organizer.core import load_json


INDEX = load_json(ROOT / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json")
CONFIG = load_json(ROOT / "configs/experiments/r3_2_frozen_exact_interleaved_av_organizer_canary_v1.json")


def test_no_api_suite_passes() -> None:
    report = no_api_tests(INDEX, CONFIG)
    assert report["status"] == "passed", report


def test_canonical_nodes_are_interleaved_once() -> None:
    nodes = build_interleaved_nodes(INDEX)
    assert len(nodes) == 137
    assert len({row["id"] for row in nodes}) == 137
    assert [row["start_sec"] for row in nodes] == sorted(row["start_sec"] for row in nodes)


def test_current_repaired_captions_not_historical_caption_text() -> None:
    nodes = build_interleaved_nodes(INDEX)
    visual = [row for row in nodes if row["modality"] == "visual"]
    assert [row["caption"] for row in visual] == [row["qwen_caption"] for row in INDEX["medium_nodes"]]
