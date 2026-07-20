from __future__ import annotations

import numpy as np

from src.experiments.semantic_coarse_v0_1.deterministic_coarse import (
    build_groups_with_anti_chaining, pair_decision,
)


def test_pair_rule_requires_caption_visual_and_state_continuity():
    assert pair_decision(caption_similarity=.95, visual_similarity=.94, left_postures=[], right_postures=[], caption_min=.9, visual_min=.85)[0] == "MERGE"
    assert pair_decision(caption_similarity=.89, visual_similarity=.94, left_postures=[], right_postures=[], caption_min=.9, visual_min=.85)[0] == "STOP"
    assert pair_decision(caption_similarity=.95, visual_similarity=.84, left_postures=[], right_postures=[], caption_min=.9, visual_min=.85)[0] == "STOP"
    assert pair_decision(caption_similarity=.95, visual_similarity=.94, left_postures=["standing"], right_postures=["kneeling"], caption_min=.9, visual_min=.85)[0] == "STOP"


def test_anti_chaining_can_stop_pairwise_chain():
    records = [
        {"medium_id": "m1", "caption_embedding": np.array([1.0, 0.0]), "visual_embedding": np.array([1.0, 0.0])},
        {"medium_id": "m2", "caption_embedding": np.array([.95, .31]), "visual_embedding": np.array([.95, .31])},
        {"medium_id": "m3", "caption_embedding": np.array([.80, .60]), "visual_embedding": np.array([.80, .60])},
    ]
    pair_rows = [{"pair_decision": "MERGE"}, {"pair_decision": "MERGE"}]
    groups, trace = build_groups_with_anti_chaining(
        records=records, pair_rows=pair_rows, caption_centroid_min=.94, visual_centroid_min=.94
    )
    assert groups == [[0, 1], [2]]
    assert trace[-1]["anti_chaining_prevented_merge"] is True


def test_grouping_preserves_order_and_lineage():
    records = [
        {"medium_id": f"m{i}", "caption_embedding": np.array([1.0, 0.0]), "visual_embedding": np.array([1.0, 0.0])}
        for i in range(4)
    ]
    groups, _ = build_groups_with_anti_chaining(
        records=records, pair_rows=[{"pair_decision": "MERGE"}] * 3,
        caption_centroid_min=.9, visual_centroid_min=.85,
    )
    assert groups == [[0, 1, 2, 3]]
    assert [i for group in groups for i in group] == list(range(4))
