from __future__ import annotations

from experiments.r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair.core import _planner_input_stats


def test_planner_input_stats() -> None:
    stats = _planner_input_stats([{"x": 1}, {"x": 22}])
    assert stats["questions"] == 2
    assert stats["payload_bytes_total"] > 0
    assert stats["payload_bytes_max"] >= stats["payload_bytes_min"]

