from __future__ import annotations

import numpy as np

from src.diagnostics.cradio_v4.option_semantic import (
    balanced_unique_candidates,
    build_option_query,
    candidate_interval_hit,
    is_concrete_option,
    union_at_per_option_depth,
)


def test_option_queries_are_action_specific_and_none_is_flagged() -> None:
    assert build_option_query("What happens?", "An officer runs.") == (
        "Question: What happens?\nCandidate answer: An officer runs."
    )
    assert is_concrete_option("An officer runs.")
    assert not is_concrete_option("None of the above")


def test_balanced_merge_is_deterministic_result_independent_and_deduplicated() -> None:
    rankings = [
        np.asarray([1, 2, 3, 4]),
        np.asarray([1, 5, 6, 7]),
        np.asarray([8, 9, 10, 11]),
        np.asarray([12, 13, 14, 15]),
        np.asarray([99, 98, 97, 96]),
    ]
    assert balanced_unique_candidates(rankings, [0, 1, 2, 3], 8) == [1, 8, 12, 2, 5, 9, 13, 3]
    assert union_at_per_option_depth(rankings, [0, 1, 2, 3], 2) == [1, 8, 12, 2, 5, 9, 13]


def test_candidate_hit_is_posthoc() -> None:
    timestamps = np.arange(6, dtype=np.float64)
    assert candidate_interval_hit([0, 3], timestamps, [3.0, 4.0])
    assert not candidate_interval_hit([0, 2], timestamps, [3.0, 4.0])

