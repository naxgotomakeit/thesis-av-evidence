import numpy as np
import pytest

from src.retrieval.three_channel import (
    assert_no_leakage_payload,
    cosine_similarities,
    evaluate_ranked_results,
    rank_index,
    safe_retrieval_case,
    temporal_distance,
    temporal_overlap,
    validate_query_dimension,
)


def test_query_embedding_dimensions():
    assert validate_query_dimension(np.zeros(512), np.zeros((2, 512)))
    assert validate_query_dimension(np.zeros(768), np.zeros((1, 768)))
    assert not validate_query_dimension(np.zeros(512), np.zeros((1, 768)))


def test_cosine_similarity_and_descending_order():
    index = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    scores = cosine_similarities(np.array([1, 0]), index)
    assert np.allclose(scores, [1, 0, -1])
    ranked = rank_index(np.array([1, 0]), index, [{"id": i} for i in range(3)], 3)
    assert [x["metadata"]["id"] for x in ranked] == [0, 1, 2]


def test_embedding_metadata_alignment_and_fewer_than_topk():
    with pytest.raises(ValueError):
        rank_index(np.ones(2), np.ones((2, 2)), [{}], 3)
    assert len(rank_index(np.ones(2), np.ones((1, 2)), [{"text": ""}], 3)) == 1


def test_empty_transcript_index():
    result = rank_index(np.ones(3), np.empty((0, 3), dtype=np.float32), [], 3)
    assert result == []


def test_temporal_overlap_and_distance():
    assert temporal_overlap(2, 4, 3, 5)
    assert temporal_distance(2, 4, 3, 5) == 0
    assert temporal_distance(0, 1, 3, 5) == 2
    assert temporal_distance(7, 8, 3, 5) == 2


def test_deterministic_tie_ranking():
    index = np.ones((3, 2), dtype=np.float32)
    rows = [{"id": i} for i in range(3)]
    first = rank_index(np.ones(2), index, rows, 3)
    second = rank_index(np.ones(2), index, rows, 3)
    assert first == second and [x["embedding_row"] for x in first] == [0, 1, 2]


@pytest.mark.parametrize("key,value", [
    ("answer", "secret"), ("answer_options", ["a", "b"]),
    ("provided_timestamp_start", 3.0), ("provided_context", "leak"),
])
def test_no_leakage_keys(key, value):
    payload = {"case_id": "x", "video_id": "v", "question": "raw question", key: value}
    with pytest.raises(ValueError):
        assert_no_leakage_payload(payload)


def test_safe_case_strips_answer_options_and_reference_before_retrieval():
    raw = {"case_id": "c", "video_id": "v", "question": "q?", "answer": "a", "answer_options": ["a"], "provided_timestamp_start": 1.0}
    assert safe_retrieval_case(raw) == {"case_id": "c", "video_id": "v", "question": "q?"}


def test_reference_cannot_change_ranking():
    query = np.array([1.0, 0.0]); index = np.array([[1.0, 0.0], [0.0, 1.0]])
    rows = [{"start_time": 0, "end_time": 1}, {"start_time": 10, "end_time": 11}]
    ranked = rank_index(query, index, rows, 2)
    before = [x["embedding_row"] for x in ranked]
    diagnostic = [{**x, **x["metadata"]} for x in ranked]
    evaluate_ranked_results(diagnostic, 10, 11)
    assert [x["embedding_row"] for x in ranked] == before
