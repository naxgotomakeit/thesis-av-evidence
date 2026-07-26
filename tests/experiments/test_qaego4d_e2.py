from __future__ import annotations

import numpy as np
from pathlib import Path

from src.experiments.qaego4d_e2.core import (
    build_b1_prime_groups,
    build_amendment4_fine_events,
    build_amendment4_medium_events,
    build_event_records,
    compute_retrieval_diagnostics,
    finalize_ranked_frames,
    rank_events,
    retrieve_amendment4_b1,
    retrieve_amendment4_b2,
    retrieve_amendment5_b1,
    retrieve_amendment5_b2,
    retrieve_amendment6_b1,
    retrieve_amendment6_b2,
    select_cradio_medoid,
    select_dino_medoid,
)
from src.experiments.qaego4d_e2.pipeline import (
    build_shared_fine_artifact,
    derive_b1_fine_units,
    derive_b2_medium_units,
)


def _events() -> list[dict]:
    timestamps = np.arange(6, dtype=np.float64)
    source = np.arange(100, 106, dtype=np.int64)
    embeddings = np.eye(6, dtype=np.float32)
    units = [
        {"event_id": "fine_0", "start_s": 0.0, "end_s": 2.0, "leaf_ids": ["fine_0"]},
        {"event_id": "fine_1", "start_s": 2.0, "end_s": 4.0, "leaf_ids": ["fine_1"]},
        {"event_id": "fine_2", "start_s": 4.0, "end_s": 6.0, "leaf_ids": ["fine_2"]},
    ]
    return build_event_records(
        units=units, timestamps=timestamps, source_frame_indices=source,
        cradio_embeddings=embeddings, method="b1",
    )


def test_medoid_tie_is_earlier_timestamp() -> None:
    index, record = select_cradio_medoid(
        timestamps=np.asarray([2.0, 3.0]), visual_embeddings=np.eye(2, dtype=np.float32), frame_indices=[0, 1]
    )
    assert index == 0
    assert record["selected_timestamp_sec"] == 2.0


def test_amendment4_dino_medoid_tie_is_earlier_timestamp() -> None:
    index, record = select_dino_medoid(
        timestamps=np.asarray([2.0, 3.0]), dino_embeddings=np.eye(2, dtype=np.float32), frame_indices=[0, 1]
    )
    assert index == 0
    assert record["selected_timestamp_sec"] == 2.0


def test_shared_event_representation_is_one_normalized_cradio_medoid() -> None:
    events = _events()
    assert all(event["retrieval_representation"]["rule"] == "normalized_cradio_embedding_of_one_cradio_medoid_frame" for event in events)
    assert all(event["retrieval_representation"]["dimension"] == 6 for event in events)
    assert all("pooled" not in event["retrieval_representation"] for event in events)


def test_cosine_ranking_and_final_frames_are_deterministic_and_leq8() -> None:
    events = _events()
    query = np.asarray(events[1]["retrieval_representation"]["embedding"], dtype=np.float32)
    ranked_once = rank_events(query_embedding=query, events=events)
    ranked_twice = rank_events(query_embedding=query, events=events)
    assert [row["event_id"] for row in ranked_once] == [row["event_id"] for row in ranked_twice]
    frames = finalize_ranked_frames(ranked_once, final_event_top_k=8)
    assert len(frames) <= 8
    assert len({row["source_frame_index"] for row in frames}) == len(frames)
    assert [row["timestamp_sec"] for row in frames] == sorted(row["timestamp_sec"] for row in frames)


def test_b1_prime_exact_count_contiguity_and_earlier_tie() -> None:
    fine = [
        {"event_id": f"fine_{index}", "start_s": float(index), "end_s": float(index + 1)}
        for index in range(5)
    ]
    groups = build_b1_prime_groups(fine_events=fine, medium_count=3, clip_start_s=0.0, clip_end_s=5.0)
    assert len(groups) == 3
    assert all(group["leaf_ids"] for group in groups)
    assert groups[0]["end_s"] == groups[1]["start_s"]
    assert groups[1]["end_s"] == groups[2]["start_s"]
    # Target 5/3 is closer to boundary 2; target 10/3 is closer to boundary 3.
    assert [group["end_s"] for group in groups[:-1]] == [2.0, 3.0]


def test_diagnostics_are_posthoc_and_separate_event_from_frame_hit() -> None:
    events = _events()
    ranked = rank_events(
        query_embedding=np.asarray(events[0]["retrieval_representation"]["embedding"]), events=events
    )
    final = finalize_ranked_frames(ranked)
    metrics = compute_retrieval_diagnostics(
        events=events, ranked_events=ranked, final_frames=final, gt_start_s=2.2, gt_end_s=2.8
    )
    assert metrics["gt_used_for_retrieval"] is False
    assert metrics["event_representability"]["gt_overlap_event_exists"] is True
    assert "recall_at_20_event" in metrics and "frame_hit_at_8" in metrics


def test_e2_retrieval_path_has_no_semantic_or_planner_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/experiments/qaego4d_e2/pipeline.py").read_text(encoding="utf-8")
    active_imports = [line.strip() for line in source.splitlines() if line.startswith("from ") or line.startswith("import ")]
    forbidden = ("semantic_helpers", "sentence_transformers", "qwen_local", "planner", "rerank", "audio")
    assert not [line for line in active_imports if any(token in line.lower() for token in forbidden)]


def test_b1_and_b2_share_one_fine_artifact_but_b2_derives_medium() -> None:
    rng = np.random.default_rng(0)
    timestamps = np.arange(10, dtype=np.float64)
    shared = build_shared_fine_artifact(
        clip_uid="toy", clip_start_s=0.0, clip_end_s=10.0, timestamps_parent_s=timestamps,
        frame_paths=[Path(f"/tmp/frame_{i}.jpg") for i in range(10)],
        dino_features=rng.normal(size=(10, 384)).astype(np.float32),
        comet_config={"smoothing_sigma_frames": 1.0, "adaptive_mad_multiplier": 0.5, "minimum_prominence": 0.005, "minimum_segment_duration_sec": 4.0},
        hierarchy_config={"representative_fractions": [0.25, 0.5, 0.75], "include_dinov2_medoid": True},
    )
    b1 = derive_b1_fine_units(shared)
    b2, trace = derive_b2_medium_units(
        shared_fine=shared,
        hierarchy_config={"medium_reference_fraction": 0.5, "coarse_reference_fraction": 0.25, "representative_fractions": [0.25, 0.5, 0.75], "include_dinov2_medoid": True},
        medium_config={"minimum_q_rank": 0.4, "maximum_local_drop": 0.3},
    )
    assert [node["node_id"] for node in shared["fine_nodes"]] == [node["source_fine_segment_id"] for node in b1]
    assert trace["safe_merge_hierarchy"]["cuts"]["fine"]["node_ids"] == [node["node_id"] for node in shared["fine_nodes"]]
    assert b2 and trace["medium_node_ids"]


def test_amendment4_fine_only_cradio_and_medium_reuses_child_vector() -> None:
    timestamps = np.arange(6, dtype=np.float64)
    source = np.arange(100, 106, dtype=np.int64)
    dino = np.asarray([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, 0.1],
    ], dtype=np.float32)
    units = [
        {"event_id": "fine_a", "node_id": "node_a", "source_fine_segment_id": "node_a", "start_s": 0.0, "end_s": 2.0},
        {"event_id": "fine_b", "node_id": "node_b", "source_fine_segment_id": "node_b", "start_s": 2.0, "end_s": 4.0},
        {"event_id": "fine_c", "node_id": "node_c", "source_fine_segment_id": "node_c", "start_s": 4.0, "end_s": 6.0},
    ]
    fine = build_amendment4_fine_events(
        units=units, timestamps=timestamps, source_frame_indices=source, dino_embeddings=dino,
        fine_cradio_embeddings=np.eye(3, dtype=np.float32),
    )
    assert len(fine) == 3  # exactly one C-RADIO vector per Fine, not per timeline frame
    medium, parents = build_amendment4_medium_events(
        medium_units=[{"event_id": "medium_0", "start_s": 0.0, "end_s": 6.0, "leaf_ids": ["node_a", "node_b", "node_c"]}],
        fine_events=fine,
    )
    assert len(medium) == 1 and set(parents) == {"fine_a", "fine_b", "fine_c"}
    selected = medium[0]["medium_representative_fine_event_id"]
    selected_fine = next(item for item in fine if item["event_id"] == selected)
    assert medium[0]["retrieval_representation"]["embedding"] == selected_fine["retrieval_representation"]["embedding"]


def test_amendment4_b1_global_and_b2_top3_descendant_only() -> None:
    fine = []
    for index in range(6):
        fine.append({
            "event_id": f"fine_{index}", "start_s": float(index), "end_s": float(index + 1),
            "representative_frame": {"selected_timestamp_sec": float(index), "source_frame_index": index, "timeline_index": index},
            "retrieval_representation": {"embedding": np.eye(6, dtype=np.float32)[index].tolist()},
        })
    medium = [
        {"event_id": "m0", "start_s": 0.0, "end_s": 2.0, "descendant_fine_event_ids": ["fine_0", "fine_1"], "retrieval_representation": {"embedding": np.eye(6, dtype=np.float32)[0].tolist()}},
        {"event_id": "m1", "start_s": 2.0, "end_s": 4.0, "descendant_fine_event_ids": ["fine_2", "fine_3"], "retrieval_representation": {"embedding": np.eye(6, dtype=np.float32)[1].tolist()}},
        {"event_id": "m2", "start_s": 4.0, "end_s": 5.0, "descendant_fine_event_ids": ["fine_4"], "retrieval_representation": {"embedding": np.eye(6, dtype=np.float32)[2].tolist()}},
        {"event_id": "m3", "start_s": 5.0, "end_s": 6.0, "descendant_fine_event_ids": ["fine_5"], "retrieval_representation": {"embedding": np.eye(6, dtype=np.float32)[3].tolist()}},
    ]
    query = np.asarray([1, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    b1 = retrieve_amendment4_b1(query_embedding=query, fine_events=fine)
    b2 = retrieve_amendment4_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert b1["fine_candidate_count"] == 6 and len(b1["final_frames"]) <= 8
    assert b2["stage1_medium_candidate_count"] == 4
    assert b2["stage1_selected_medium_count"] == 3
    allowed = {fine_id for row in b2["selected_medium_events"] for fine_id in row["descendant_fine_event_ids"]}
    assert {row["event_id"] for row in b2["stage2_fine_candidates"]} == allowed
    assert {row["event_id"] for row in b2["ranked_fine_events"]}.issubset(allowed)
    assert len(b2["final_frames"]) <= 8


def _amendment5_events(*, fine_count: int, medium_descendants: list[list[str]]) -> tuple[list[dict], list[dict], np.ndarray]:
    fine = [{
        "event_id": f"fine_{index}", "start_s": float(index), "end_s": float(index + 1),
        "representative_frame": {"selected_timestamp_sec": float(index), "source_frame_index": index, "timeline_index": index},
        "retrieval_representation": {"embedding": np.eye(fine_count, dtype=np.float32)[index].tolist()},
    } for index in range(fine_count)]
    medium = [{
        "event_id": f"medium_{index}", "start_s": float(index), "end_s": float(index + 1),
        "descendant_fine_event_ids": descendants,
        "retrieval_representation": {"embedding": np.eye(fine_count, dtype=np.float32)[index].tolist()},
    } for index, descendants in enumerate(medium_descendants)]
    query = np.arange(fine_count, 0, -1, dtype=np.float32)
    return fine, medium, query


def test_amendment5_top3_sufficient_requires_no_expansion() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=8, medium_descendants=[["fine_0", "fine_1", "fine_2"], ["fine_3", "fine_4", "fine_5"], ["fine_6", "fine_7"]],
    )
    result = retrieve_amendment5_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert result["stage1_initial_medium_count"] == 3
    assert result["stage1_selected_medium_count"] == 3
    assert result["stage1_expansion_steps"] == 0
    assert result["stage2_fine_candidate_count_before_topk"] == 8
    assert result["final_selected_fine_count"] == 8


def test_amendment5_expands_in_rank_order_then_selects_k() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=10,
        medium_descendants=[["fine_0", "fine_1"], ["fine_2", "fine_3"], ["fine_4"], ["fine_5", "fine_6", "fine_7"], ["fine_8", "fine_9"]],
    )
    result = retrieve_amendment5_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert result["stage1_initial_medium_count"] == 3
    assert result["stage1_expansion_required"] is True
    assert [row["added_medium_rank"] for row in result["stage1_expansion_trace"]] == [4]
    assert result["stage2_fine_candidate_count_before_topk"] == 8
    assert result["final_selected_fine_count"] == 8
    allowed = {fine_id for row in result["selected_medium_events"] for fine_id in row["descendant_fine_event_ids"]}
    assert {row["event_id"] for row in result["ranked_fine_events"]} == allowed


def test_amendment5_overshoot_and_duplicate_descendants_are_controlled() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=9,
        medium_descendants=[["fine_0", "fine_1"], ["fine_1", "fine_2"], ["fine_3"], ["fine_4", "fine_5", "fine_6", "fine_7", "fine_8"]],
    )
    result = retrieve_amendment5_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert result["stage2_fine_candidate_count_before_topk"] == 9
    assert result["target_final_fine_count_K"] == 8
    assert result["final_selected_fine_count"] == 8
    assert len({row["source_frame_index"] for row in result["final_frames"]}) == 8


def test_amendment5_handles_fewer_than_eight_fine_and_medium_events() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=2, medium_descendants=[["fine_0"], ["fine_1"]],
    )
    b1 = retrieve_amendment5_b1(query_embedding=query, fine_events=fine)
    b2 = retrieve_amendment5_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert b1["target_final_fine_count_K"] == b2["target_final_fine_count_K"] == 2
    assert len(b1["final_frames"]) == len(b2["final_frames"]) == 2
    assert b2["stage1_initial_medium_count"] == b2["stage1_selected_medium_count"] == 2
    assert b2["stage1_expansion_steps"] == 0


def test_amendment5_records_all_medium_exhaustion_integrity_failure() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=8, medium_descendants=[["fine_0"], ["fine_1"], ["fine_2"], ["fine_3"]],
    )
    result = retrieve_amendment5_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert result["all_mediums_exhausted"] is True
    assert result["structural_integrity_failure"] is True
    assert result["final_selected_fine_count"] == 4


def test_amendment6_top1_sufficient_selects_one_medium() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=8,
        medium_descendants=[[f"fine_{index}" for index in range(8)], ["fine_0"], ["fine_1"]],
    )
    result = retrieve_amendment6_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert result["stage1_selected_medium_count"] == 1
    assert result["stage1_expansion_steps"] == 0
    assert result["stage1_cumulative_fine_count_after_each_medium"][0]["cumulative_fine_count"] == 8
    assert result["final_selected_fine_count"] == 8


def test_amendment6_top2_sufficient_and_rank_ordered() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=8,
        medium_descendants=[["fine_0", "fine_1", "fine_2"], ["fine_3", "fine_4", "fine_5", "fine_6", "fine_7"], ["fine_0"]],
    )
    result = retrieve_amendment6_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert result["stage1_selected_medium_count"] == 2
    assert result["stage1_expansion_steps"] == 1
    assert [row["medium_rank"] for row in result["stage1_cumulative_fine_count_after_each_medium"]] == [1, 2]
    assert [row["cumulative_fine_count"] for row in result["stage1_cumulative_fine_count_after_each_medium"]] == [3, 8]


def test_amendment6_accumulates_multiple_then_topk_after_overshoot() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=10,
        medium_descendants=[["fine_0", "fine_1", "fine_2"], ["fine_3", "fine_4"], ["fine_5", "fine_6"], ["fine_7", "fine_8", "fine_9"]],
    )
    result = retrieve_amendment6_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert [row["cumulative_fine_count"] for row in result["stage1_cumulative_fine_count_after_each_medium"]] == [3, 5, 7, 10]
    assert result["stage2_fine_candidate_count_before_topk"] == 10
    assert result["final_selected_fine_count"] == 8


def test_amendment6_low_fine_medium_and_duplicate_edge_cases() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=2, medium_descendants=[["fine_0", "fine_0"], ["fine_1"]],
    )
    b1 = retrieve_amendment6_b1(query_embedding=query, fine_events=fine)
    b2 = retrieve_amendment6_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert b1["final_selected_fine_count"] == b2["final_selected_fine_count"] == 2
    assert [row["cumulative_fine_count"] for row in b2["stage1_cumulative_fine_count_after_each_medium"]] == [1, 2]


def test_amendment6_all_medium_exhaustion_is_explicit_without_padding() -> None:
    fine, medium, query = _amendment5_events(
        fine_count=8, medium_descendants=[["fine_0"], ["fine_1"]],
    )
    result = retrieve_amendment6_b2(query_embedding=query, medium_events=medium, fine_events=fine)
    assert result["all_mediums_exhausted"] is True
    assert result["structural_integrity_failure"] is True
    assert result["final_selected_fine_count"] == 2
