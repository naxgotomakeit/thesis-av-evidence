from __future__ import annotations

import copy

import numpy as np

from experiments.r1_r3_v2_map_aware_all_medium_retrieval_pair.core import (
    build_lexical_records,
    planner_schema,
    rank_all_mediums,
    validate_lexical_integrity,
    validate_plan,
)


def plan() -> dict:
    return {
        "question_id": "q", "search_units": [{"unit_id": "u1", "description": "target", "query_variants": ["target object"]}],
        "query_variants": ["target object"], "modality_strategy": "visual and audio",
        "temporal_strategy": "localize independently", "suggested_coarse_ids": ["C01"], "hard_filtering_allowed": False,
    }


def test_schema_and_plan_contract() -> None:
    assert planner_schema()["additionalProperties"] is False
    assert validate_plan(plan(), "q", {"C01"})["valid"]


def test_unknown_coarse_or_hard_filtering_fails() -> None:
    bad = plan(); bad["suggested_coarse_ids"] = ["C99"]
    assert not validate_plan(bad, "q", {"C01"})["valid"]
    bad = plan(); bad["hard_filtering_allowed"] = True
    assert not validate_plan(bad, "q", {"C01"})["valid"]


def test_query_variant_union_is_exact() -> None:
    bad = plan(); bad["query_variants"] = []
    assert not validate_plan(bad, "q", {"C01"})["valid"]


def test_lexical_routes_are_separate_and_exact() -> None:
    base = {"medium_nodes": [{"medium_id": "M1", "qwen_caption": "caption", "start_sec": 0, "end_sec": 1}]}
    projection = [{"medium_id": "M1", "detector_summary": "fallback", "embedding_ref": {}, "source_fine_ids": ["F1"]}]
    r1, r3 = build_lexical_records(projection, base, "r1"), build_lexical_records(projection, base, "r3")
    a, b = validate_lexical_integrity(r1, r3, projection, base)
    assert a["valid"] and b["valid"]
    assert r1[0]["lexical_text"] == "fallback" and r3[0]["lexical_text"] == "caption"


def test_all_medium_ranking_has_zero_prior_and_no_fine_reranking() -> None:
    lexical = [{"medium_id": f"M{i}", "lexical_text": f"target {i}", "lexical_source": "structured_fallback", "source_fine_ids": [f"F{i}"]} for i in range(3)]
    media = [{"medium_id": f"M{i}", "start_sec": i, "end_sec": i + 1} for i in range(3)]
    config = {"ranking": {"visual_score_weight": .6, "lexical_score_weight": .3, "coarse_prior_weight": 0.0, "top_k": 2}}
    ranking, selected, _ = rank_all_mediums({"question": "target"}, plan(), lexical, media, np.eye(3, dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32), config, {f"M{i}": "C01" for i in range(3)})
    assert len(ranking) == 3 and len(selected) == 2
    assert all(row["coarse_prior"] == 0.0 and row["coarse_prior_weight"] == 0.0 for row in ranking)
    bad = copy.deepcopy(ranking); bad[0]["coarse_prior"] = 1.0
    assert bad != ranking
