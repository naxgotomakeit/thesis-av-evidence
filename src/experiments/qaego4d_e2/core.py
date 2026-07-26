"""Pure shared E2 event/retrieval primitives.

This module intentionally has no answer model, caption, semantic, reranking,
Planner, or audio dependency.  DINO-derived nodes are only temporal units;
C-RADIO is the sole text-to-event representation space.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Iterable

import numpy as np


EPSILON = 1e-12


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array / max(float(np.linalg.norm(array)), EPSILON)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), EPSILON)


def _event_frame_indices(event: dict[str, Any], timestamps: np.ndarray) -> np.ndarray:
    start, end = float(event["start_s"]), float(event["end_s"])
    indices = np.flatnonzero((timestamps >= start) & (timestamps < end))
    if not len(indices):
        raise ValueError(f"Event has no valid 1FPS frame: {event['event_id']}")
    return indices.astype(np.int64)


def select_cradio_medoid(
    *, timestamps: np.ndarray, visual_embeddings: np.ndarray, frame_indices: Iterable[int]
) -> tuple[int, dict[str, Any]]:
    """Select the deterministic, question-independent C-RADIO cosine medoid."""
    indices = np.asarray(list(frame_indices), dtype=np.int64)
    if not len(indices):
        raise ValueError("Cannot choose a medoid from no frames")
    normalized = normalize_rows(np.asarray(visual_embeddings, dtype=np.float32)[indices])
    if len(indices) == 1:
        winner = 0
        mean_similarity = np.asarray([1.0], dtype=np.float64)
    else:
        similarity = normalized @ normalized.T
        mean_similarity = (similarity.sum(axis=1) - 1.0) / (len(indices) - 1)
        # np.argmax resolves an exact tie by local/chronological order because
        # event frame indices are chronological.
        winner = int(np.argmax(mean_similarity))
    selected = int(indices[winner])
    return selected, {
        "selection_rule": "cradio_cosine_medoid_highest_mean_similarity_tie_earlier_timestamp",
        "candidate_frame_count": int(len(indices)),
        "candidate_frame_indices": indices.tolist(),
        "candidate_timestamps_sec": [float(timestamps[index]) for index in indices],
        "mean_cosine_similarity": [float(value) for value in mean_similarity],
        "selected_frame_timeline_index": selected,
        "selected_timestamp_sec": float(timestamps[selected]),
    }


def select_dino_medoid(
    *, timestamps: np.ndarray, dino_embeddings: np.ndarray, frame_indices: Iterable[int]
) -> tuple[int, dict[str, Any]]:
    """Amendment #4 deterministic DINO cosine medoid, with earlier tie break."""
    indices = np.asarray(list(frame_indices), dtype=np.int64)
    if not len(indices):
        raise ValueError("Cannot choose a DINO medoid from no frames")
    normalized = normalize_rows(np.asarray(dino_embeddings, dtype=np.float32)[indices])
    if len(indices) == 1:
        winner = 0
        mean_similarity = np.asarray([1.0], dtype=np.float64)
    else:
        similarity = normalized @ normalized.T
        mean_similarity = (similarity.sum(axis=1) - 1.0) / (len(indices) - 1)
        winner = int(np.argmax(mean_similarity))  # chronological indices make ties earlier.
    selected = int(indices[winner])
    return selected, {
        "selection_rule": "dinov2_cosine_medoid_highest_mean_similarity_tie_earlier_timestamp",
        "candidate_frame_count": int(len(indices)),
        "candidate_frame_indices": indices.tolist(),
        "candidate_timestamps_sec": [float(timestamps[index]) for index in indices],
        "mean_cosine_similarity": [float(value) for value in mean_similarity],
        "selected_frame_timeline_index": selected,
        "selected_timestamp_sec": float(timestamps[selected]),
    }


def build_amendment4_fine_events(
    *, units: Iterable[dict[str, Any]], timestamps: np.ndarray, source_frame_indices: np.ndarray,
    dino_embeddings: np.ndarray, fine_cradio_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    """Build one DINO-medoid / one C-RADIO-vector record per Fine event."""
    items = list(units)
    times = np.asarray(timestamps, dtype=np.float64)
    source = np.asarray(source_frame_indices, dtype=np.int64)
    dino = normalize_rows(np.asarray(dino_embeddings, dtype=np.float32))
    cradio = normalize_rows(np.asarray(fine_cradio_embeddings, dtype=np.float32))
    if not (len(items) == len(cradio)):
        raise ValueError("Fine C-RADIO vectors must be one-for-one with Fine events")
    if not (len(times) == len(source) == len(dino)):
        raise ValueError("Fine timeline identity mismatch")
    selected_indices, selected_records = select_amendment4_fine_representatives(
        units=items, timestamps=times, dino_embeddings=dino
    )
    records: list[dict[str, Any]] = []
    for position, unit in enumerate(items):
        event_id = str(unit["event_id"])
        start, end = float(unit["start_s"]), float(unit["end_s"])
        indices = _event_frame_indices({"event_id": event_id, "start_s": start, "end_s": end}, times)
        representative_index = selected_indices[position]
        representative = selected_records[position]
        fine_node_id = str(unit.get("source_fine_segment_id") or unit.get("node_id") or event_id)
        records.append({
            "event_id": event_id,
            "method": "b1_fine",
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "fine_node_id": fine_node_id,
            "source_fine_ids": [fine_node_id],
            "source_fine_indices": [int(value) for value in indices],
            "representative_frame": {
                **representative,
                "source_frame_index": int(source[representative_index]),
                "timeline_index": representative_index,
            },
            "dino_representative": {
                "timeline_index": representative_index,
                "embedding": dino[representative_index].astype(np.float32).tolist(),
            },
            "retrieval_representation": {
                "rule": "normalized_cradio_embedding_of_dino_medoid_fine_representative",
                "model_space": "C-RADIOv4-SO400M/siglip2-g aligned space",
                "dimension": int(cradio.shape[1]),
                "normalization": "l2",
                "embedding": cradio[position].astype(np.float32).tolist(),
            },
        })
    return records


def select_amendment4_fine_representatives(
    *, units: Iterable[dict[str, Any]], timestamps: np.ndarray, dino_embeddings: np.ndarray,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select one deterministic DINO medoid per Fine, before C-RADIO encoding."""
    times = np.asarray(timestamps, dtype=np.float64)
    dino = normalize_rows(np.asarray(dino_embeddings, dtype=np.float32))
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    for unit in units:
        event_id = str(unit["event_id"])
        indices = _event_frame_indices({
            "event_id": event_id, "start_s": float(unit["start_s"]), "end_s": float(unit["end_s"]),
        }, times)
        index, record = select_dino_medoid(timestamps=times, dino_embeddings=dino, frame_indices=indices)
        selected.append(index)
        records.append(record)
    return selected, records


def build_amendment4_medium_events(
    *, medium_units: Iterable[dict[str, Any]], fine_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build Medium routing vectors solely by reusing child-Fine C-RADIO vectors."""
    by_fine_node = {str(item["fine_node_id"]): item for item in fine_events}
    medium_records: list[dict[str, Any]] = []
    parent_by_fine: dict[str, str] = {}
    for position, unit in enumerate(medium_units):
        event_id = str(unit["event_id"])
        child_nodes = [str(value) for value in unit.get("leaf_ids", [])]
        children = [by_fine_node[value] for value in child_nodes if value in by_fine_node]
        if len(children) != len(child_nodes) or not children:
            raise ValueError(f"Medium {event_id} has invalid child Fine IDs")
        ordered = sorted(children, key=lambda item: (float(item["start_s"]), str(item["event_id"])))
        child_times = np.asarray([float(item["representative_frame"]["selected_timestamp_sec"]) for item in ordered])
        child_dino = np.asarray([item["dino_representative"]["embedding"] for item in ordered], dtype=np.float32)
        representative_local, medoid = select_dino_medoid(
            timestamps=child_times, dino_embeddings=child_dino, frame_indices=range(len(ordered))
        )
        representative_fine = ordered[representative_local]
        for child in ordered:
            fine_event_id = str(child["event_id"])
            if fine_event_id in parent_by_fine:
                raise ValueError(f"Fine {fine_event_id} belongs to multiple Medium events")
            parent_by_fine[fine_event_id] = event_id
        medium_records.append({
            "event_id": event_id,
            "method": "b2_medium_routing",
            "start_s": float(unit["start_s"]),
            "end_s": float(unit["end_s"]),
            "duration_s": float(unit["end_s"]) - float(unit["start_s"]),
            "descendant_fine_event_ids": [str(item["event_id"]) for item in ordered],
            "descendant_fine_node_ids": child_nodes,
            "medium_representative_fine_event_id": str(representative_fine["event_id"]),
            "medium_representative_selection": {
                **medoid,
                "selection_rule": "dinov2_child_fine_representative_cosine_medoid_tie_earlier",
            },
            "retrieval_representation": {
                "rule": "reuse_cradio_vector_of_dino_medoid_child_fine_representative",
                "model_space": "C-RADIOv4-SO400M/siglip2-g aligned space",
                "dimension": int(representative_fine["retrieval_representation"]["dimension"]),
                "normalization": "l2",
                "embedding": list(representative_fine["retrieval_representation"]["embedding"]),
                "reused_from_fine_event_id": str(representative_fine["event_id"]),
            },
        })
    return medium_records, parent_by_fine


def build_event_records(
    *, units: Iterable[dict[str, Any]], timestamps: np.ndarray,
    source_frame_indices: np.ndarray, cradio_embeddings: np.ndarray, method: str,
) -> list[dict[str, Any]]:
    """Attach the one shared C-RADIO medoid retrieval representation to units."""
    times = np.asarray(timestamps, dtype=np.float64)
    source = np.asarray(source_frame_indices, dtype=np.int64)
    embeddings = normalize_rows(np.asarray(cradio_embeddings, dtype=np.float32))
    if not (len(times) == len(source) == len(embeddings)):
        raise ValueError("C-RADIO timeline arrays differ in length")
    records: list[dict[str, Any]] = []
    for position, unit in enumerate(units):
        event_id = str(unit.get("event_id", unit.get("node_id", f"{method}_{position:04d}")))
        start = float(unit.get("start_s", unit.get("start")))
        end = float(unit.get("end_s", unit.get("end")))
        if end <= start:
            raise ValueError(f"Non-positive event duration: {event_id}")
        event = {"event_id": event_id, "start_s": start, "end_s": end}
        timeline_indices = _event_frame_indices(event, times)
        medoid_index, medoid = select_cradio_medoid(
            timestamps=times, visual_embeddings=embeddings, frame_indices=timeline_indices
        )
        provenance = {
            "source_fine_ids": list(unit.get("leaf_ids", [unit.get("source_fine_segment_id", event_id)])),
            "safe_merge_node_id": unit.get("safe_merge_node_id"),
            "fluid_loose_node_id": unit.get("fluid_loose_node_id", unit.get("node_id")),
        }
        records.append({
            "event_id": event_id,
            "method": method,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "source_fine_indices": [int(index) for index in timeline_indices],
            "source_fine_ids": provenance["source_fine_ids"],
            "hierarchy_provenance": provenance,
            "representative_frame": {
                **medoid,
                "source_frame_index": int(source[medoid_index]),
                "timeline_index": medoid_index,
            },
            "retrieval_representation": {
                "rule": "normalized_cradio_embedding_of_one_cradio_medoid_frame",
                "model_space": "C-RADIOv4-SO400M/siglip2-g aligned space",
                "dimension": int(embeddings.shape[1]),
                "normalization": "l2",
                "timeline_index": medoid_index,
                "embedding": embeddings[medoid_index].astype(np.float32).tolist(),
            },
        })
    return records


def _valid_boundaries(fine_events: list[dict[str, Any]]) -> np.ndarray:
    if len(fine_events) < 1:
        raise ValueError("B1-prime requires at least one Fine event")
    return np.asarray([float(item["end_s"]) for item in fine_events[:-1]], dtype=np.float64)


def build_b1_prime_groups(
    *, fine_events: list[dict[str, Any]], medium_count: int,
    clip_start_s: float, clip_end_s: float,
) -> list[dict[str, Any]]:
    """Count-matched, equal-duration contiguous Fine grouping from Amendment #3."""
    if medium_count <= 0 or medium_count > len(fine_events):
        raise ValueError("B1-prime medium_count must be in [1, Fine event count]")
    if not math.isclose(float(fine_events[0]["start_s"]), clip_start_s, abs_tol=1e-6):
        raise ValueError("Fine events must start at canonical clip start")
    if not math.isclose(float(fine_events[-1]["end_s"]), clip_end_s, abs_tol=1e-6):
        raise ValueError("Fine events must end at canonical clip end")
    if medium_count == 1:
        return [{
            "event_id": "b1_prime_0000", "start_s": clip_start_s, "end_s": clip_end_s,
            "leaf_ids": [str(item["event_id"]) for item in fine_events],
            "b1_prime_rule": "equal_duration_fine_boundary_snap",
        }]
    targets = [clip_start_s + j * (clip_end_s - clip_start_s) / medium_count for j in range(1, medium_count)]
    # A boundary is represented by its *left group* Fine-event index.  At step j
    # at least j Fine events must remain left, and at least K-j to the right.
    selected_positions: list[int] = []
    for j, target in enumerate(targets, start=1):
        lower = selected_positions[-1] + 1 if selected_positions else 1
        upper = len(fine_events) - (medium_count - j) - 1
        candidates = list(range(lower, upper + 1))
        if not candidates:
            raise ValueError("No non-empty B1-prime boundary can satisfy ordering")
        winner = min(
            candidates,
            key=lambda pos: (abs(float(fine_events[pos]["start_s"]) - target), float(fine_events[pos]["start_s"]), pos),
        )
        selected_positions.append(winner)
    edges = [0, *selected_positions, len(fine_events)]
    groups: list[dict[str, Any]] = []
    for group_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        members = fine_events[left:right]
        if not members:
            raise AssertionError("B1-prime generated an empty group")
        groups.append({
            "event_id": f"b1_prime_{group_index:04d}",
            "start_s": float(members[0]["start_s"]),
            "end_s": float(members[-1]["end_s"]),
            "leaf_ids": [str(item["event_id"]) for item in members],
            "b1_prime_rule": "equal_duration_fine_boundary_snap",
            "b1_prime_medium_count_only": int(medium_count),
        })
    return groups


def rank_events(*, query_embedding: np.ndarray, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shared normalized cosine ranking with deterministic event-ID tie breaking."""
    query = normalize_rows(np.asarray(query_embedding, dtype=np.float32))
    if query.ndim != 1:
        raise ValueError("Question retrieval vector must be one-dimensional")
    return deterministic_rank_scored(score_events(query_embedding=query, events=events))


def score_events(*, query_embedding: np.ndarray, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score events only; ranking is separate for Amendment #4 latency logs."""
    query = normalize_rows(np.asarray(query_embedding, dtype=np.float32))
    if query.ndim != 1:
        raise ValueError("Question retrieval vector must be one-dimensional")
    ranked: list[dict[str, Any]] = []
    for event in events:
        vector = normalize_rows(np.asarray(event["retrieval_representation"]["embedding"], dtype=np.float32))
        if vector.shape != query.shape:
            raise ValueError("Question/event C-RADIO embedding shape mismatch")
        ranked.append({**event, "cosine_score": float(query @ vector)})
    return ranked


def deterministic_rank_scored(scored_events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frozen score/start/event-ID ordering, separated from cosine computation."""
    ranked = list(scored_events)
    ranked.sort(key=lambda item: (-float(item["cosine_score"]), float(item["start_s"]), str(item["event_id"])))
    for index, event in enumerate(ranked, start=1):
        event["rank"] = index
    return ranked


def retrieve_amendment4_b1(*, query_embedding: np.ndarray, fine_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fine-global B1 retrieval; the query is supplied once by the caller."""
    ranked = rank_events(query_embedding=query_embedding, events=fine_events)
    return {
        "ranked_fine_events": ranked,
        "final_frames": finalize_ranked_frames(ranked, final_event_top_k=8),
        "fine_candidate_count": len(fine_events),
    }


def retrieve_amendment4_b2(
    *, query_embedding: np.ndarray, medium_events: list[dict[str, Any]], fine_events: list[dict[str, Any]],
    medium_top_m: int = 3,
) -> dict[str, Any]:
    """Frozen B2 Medium Top-3 routing followed by global Fine Top-8 retrieval."""
    if medium_top_m != 3:
        raise ValueError("Amendment #4 freezes B2 Stage-1 Medium Top-M to exactly 3")
    ranked_medium = rank_events(query_embedding=query_embedding, events=medium_events)
    selected_medium = ranked_medium[:medium_top_m]
    fine_by_id = {str(item["event_id"]): item for item in fine_events}
    descendant_ids: list[str] = []
    seen: set[str] = set()
    for medium in selected_medium:
        for fine_id in medium["descendant_fine_event_ids"]:
            if fine_id not in fine_by_id:
                raise ValueError(f"Medium {medium['event_id']} references unknown Fine {fine_id}")
            if fine_id not in seen:
                seen.add(fine_id)
                descendant_ids.append(fine_id)
    candidate_fine = [fine_by_id[fine_id] for fine_id in descendant_ids]
    ranked_fine = rank_events(query_embedding=query_embedding, events=candidate_fine)
    return {
        "ranked_medium_events": ranked_medium,
        "selected_medium_events": selected_medium,
        "stage2_fine_candidates": candidate_fine,
        "ranked_fine_events": ranked_fine,
        "final_frames": finalize_ranked_frames(ranked_fine, final_event_top_k=8),
        "stage1_medium_candidate_count": len(medium_events),
        "stage1_selected_medium_count": len(selected_medium),
        "stage2_fine_candidate_count": len(candidate_fine),
    }


def _descendant_fine_union(
    *, selected_medium_events: Iterable[dict[str, Any]], fine_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return the deterministic, ID-deduplicated Fine-descendant union."""
    descendant_ids: list[str] = []
    seen: set[str] = set()
    for medium in selected_medium_events:
        for fine_id in medium["descendant_fine_event_ids"]:
            fine_id = str(fine_id)
            if fine_id not in fine_by_id:
                raise ValueError(f"Medium {medium['event_id']} references unknown Fine {fine_id}")
            if fine_id not in seen:
                seen.add(fine_id)
                descendant_ids.append(fine_id)
    return descendant_ids, [fine_by_id[fine_id] for fine_id in descendant_ids]


def retrieve_amendment5_b1(*, query_embedding: np.ndarray, fine_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Amendment #5 B1: global Fine ranking with K=min(8, N_Fine)."""
    target_k = min(8, len(fine_events))
    ranked = rank_events(query_embedding=query_embedding, events=fine_events)
    final_frames = finalize_ranked_frames(ranked, final_event_top_k=target_k) if target_k else []
    return {
        "ranked_fine_events": ranked,
        "final_frames": final_frames,
        "fine_candidate_count": len(fine_events),
        "target_final_fine_count_K": target_k,
        "final_selected_fine_count": len(final_frames),
    }


def retrieve_amendment5_b2(
    *, query_embedding: np.ndarray, medium_events: list[dict[str, Any]], fine_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Amendment #5 B2: Top-3 Medium prior plus count-only ranked expansion.

    The expansion condition is deliberately only the descendant Fine candidate
    count relative to K.  Ground truth, captions, answer data, and quality
    signals are absent from this API and cannot affect routing.
    """
    target_k = min(8, len(fine_events))
    ranked_medium = rank_events(query_embedding=query_embedding, events=medium_events)
    initial_count = min(3, len(ranked_medium))
    selected_medium = list(ranked_medium[:initial_count])
    fine_by_id = {str(item["event_id"]): item for item in fine_events}
    descendant_ids, candidate_fine = _descendant_fine_union(
        selected_medium_events=selected_medium, fine_by_id=fine_by_id
    )
    expansion_trace: list[dict[str, Any]] = []
    next_rank_index = initial_count
    while len(candidate_fine) < target_k and next_rank_index < len(ranked_medium):
        added_medium = ranked_medium[next_rank_index]
        selected_medium.append(added_medium)
        next_rank_index += 1
        descendant_ids, candidate_fine = _descendant_fine_union(
            selected_medium_events=selected_medium, fine_by_id=fine_by_id
        )
        expansion_trace.append({
            "added_medium_rank": int(added_medium["rank"]),
            "added_medium_event_id": str(added_medium["event_id"]),
            "fine_union_count": len(candidate_fine),
        })
    ranked_fine = rank_events(query_embedding=query_embedding, events=candidate_fine)
    final_frames = finalize_ranked_frames(ranked_fine, final_event_top_k=target_k) if target_k else []
    exhausted = len(selected_medium) == len(ranked_medium)
    structural_integrity_failure = len(candidate_fine) < target_k
    return {
        "ranked_medium_events": ranked_medium,
        "selected_medium_events": selected_medium,
        "stage2_fine_candidates": candidate_fine,
        "stage2_descendant_fine_event_ids": descendant_ids,
        "ranked_fine_events": ranked_fine,
        "final_frames": final_frames,
        "stage1_initial_medium_count": initial_count,
        "stage1_selected_medium_count": len(selected_medium),
        "stage1_expansion_steps": len(expansion_trace),
        "stage1_expansion_required": bool(expansion_trace),
        "stage1_expansion_trace": expansion_trace,
        "stage1_total_medium_count": len(medium_events),
        "stage2_fine_candidate_count_before_topk": len(candidate_fine),
        "target_final_fine_count_K": target_k,
        "final_selected_fine_count": len(final_frames),
        "stage1_candidate_sufficiency_ratio": (len(candidate_fine) / target_k) if target_k else 1.0,
        "all_mediums_exhausted": exhausted,
        "structural_integrity_failure": structural_integrity_failure,
    }


def retrieve_amendment6_b1(*, query_embedding: np.ndarray, fine_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Amendment #6 preserves Amendment #5's shared-K B1 behavior."""
    return retrieve_amendment5_b1(query_embedding=query_embedding, fine_events=fine_events)


def retrieve_amendment6_b2(
    *, query_embedding: np.ndarray, medium_events: list[dict[str, Any]], fine_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Amendment #6 B2: the smallest ranked Medium prefix exposing K Fine.

    This function intentionally does not call the superseded Amendment #5
    Top-3-first path. Expansion is determined only by rank order and the
    deduplicated descendant Fine count, without GT/caption/answer inputs.
    """
    retrieval_started = time.perf_counter()
    target_k = min(8, len(fine_events))
    started = time.perf_counter()
    scored_medium = score_events(query_embedding=query_embedding, events=medium_events)
    stage1_medium_score_time_s = time.perf_counter() - started
    started = time.perf_counter()
    ranked_medium = deterministic_rank_scored(scored_medium)
    stage1_ranking_time_s = time.perf_counter() - started
    fine_by_id = {str(item["event_id"]): item for item in fine_events}
    selected_medium: list[dict[str, Any]] = []
    cumulative_trace: list[dict[str, Any]] = []
    descendant_ids: list[str] = []
    candidate_fine: list[dict[str, Any]] = []
    started = time.perf_counter()
    for medium in ranked_medium:
        selected_medium.append(medium)
        descendant_ids, candidate_fine = _descendant_fine_union(
            selected_medium_events=selected_medium, fine_by_id=fine_by_id
        )
        cumulative_trace.append({
            "medium_rank": int(medium["rank"]),
            "medium_id": str(medium["event_id"]),
            "cumulative_fine_count": len(candidate_fine),
        })
        if len(candidate_fine) >= target_k:
            break
    stage1_expansion_time_s = time.perf_counter() - started
    # Naming preserves the shared schema: this is materializing the exact
    # Stage-1 descendant union, not a new retrieval decision.
    stage2_candidate_expand_time_s = 0.0
    started = time.perf_counter()
    scored_fine = score_events(query_embedding=query_embedding, events=candidate_fine)
    stage2_fine_score_time_s = time.perf_counter() - started
    started = time.perf_counter()
    ranked_fine = deterministic_rank_scored(scored_fine)
    stage2_ranking_time_s = time.perf_counter() - started
    final_frames = finalize_ranked_frames(ranked_fine, final_event_top_k=target_k) if target_k else []
    exhausted = len(selected_medium) == len(ranked_medium)
    structural_integrity_failure = len(candidate_fine) < target_k
    return {
        "ranked_medium_events": ranked_medium,
        "selected_medium_events": selected_medium,
        "stage2_fine_candidates": candidate_fine,
        "stage2_descendant_fine_event_ids": descendant_ids,
        "ranked_fine_events": ranked_fine,
        "final_frames": final_frames,
        "target_final_fine_count_K": target_k,
        "stage1_total_medium_count": len(medium_events),
        "stage1_selected_medium_count": len(selected_medium),
        "stage1_selected_medium_ids_in_rank_order": [str(item["event_id"]) for item in selected_medium],
        "stage1_expansion_steps": max(0, len(selected_medium) - 1),
        "stage1_cumulative_fine_count_after_each_medium": cumulative_trace,
        "stage1_medium_score_time_s": stage1_medium_score_time_s,
        "stage1_ranking_time_s": stage1_ranking_time_s,
        "stage1_expansion_time_s": stage1_expansion_time_s,
        "stage2_candidate_expand_time_s": stage2_candidate_expand_time_s,
        "stage2_fine_score_time_s": stage2_fine_score_time_s,
        "stage2_ranking_time_s": stage2_ranking_time_s,
        "stage2_fine_candidate_count_before_topk": len(candidate_fine),
        "final_selected_fine_count": len(final_frames),
        "all_mediums_exhausted": exhausted,
        "structural_integrity_failure": structural_integrity_failure,
        "total_retrieval_time_s": time.perf_counter() - retrieval_started,
        "total_online_latency_s": time.perf_counter() - retrieval_started,
    }


def finalize_ranked_frames(ranked_events: Iterable[dict[str, Any]], *, final_event_top_k: int = 8) -> list[dict[str, Any]]:
    """One representative per top event, unique/no padding, then chronological input order."""
    if final_event_top_k <= 0 or final_event_top_k > 8:
        raise ValueError("E2 final event top-k must be within the frozen <=8 frame budget")
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for event in list(ranked_events)[:final_event_top_k]:
        frame = event["representative_frame"]
        source_index = int(frame["source_frame_index"])
        if source_index in seen:
            continue
        seen.add(source_index)
        selected.append({
            "event_id": str(event["event_id"]), "rank": int(event["rank"]),
            "timestamp_sec": float(frame["selected_timestamp_sec"]),
            "source_frame_index": source_index, "cosine_score": float(event["cosine_score"]),
            # This is cache-addressing provenance only.  It is not consulted by
            # ranking or selection; E2's cached JPEG grid is numbered by this
            # local 1FPS timeline index, whereas source_frame_index belongs to
            # the parent MP4.
            "timeline_index": int(frame["timeline_index"]),
        })
    return sorted(selected, key=lambda row: (row["timestamp_sec"], row["source_frame_index"], row["event_id"]))


def _distance_to_interval(timestamp: float, start: float, end: float) -> float:
    if start <= timestamp < end:
        return 0.0
    return min(abs(timestamp - start), abs(timestamp - end))


def frame_hit_diagnostics(*, frames: Iterable[dict[str, Any]], gt_start_s: float, gt_end_s: float) -> dict[str, Any]:
    """Shared post-hoc frame-hit calculation for B0 and retrieved methods."""
    values = [float(frame["timestamp_sec"]) for frame in frames]
    inside = [value for value in values if gt_start_s <= value < gt_end_s]
    return {
        "any_inside_gt": bool(inside), "count_inside_gt": len(inside),
        "fraction_inside_gt": len(inside) / len(values) if values else 0.0,
        "nearest_selected_frame_distance_s": min(
            (_distance_to_interval(value, gt_start_s, gt_end_s) for value in values), default=None
        ),
    }


def compute_retrieval_diagnostics(
    *, events: list[dict[str, Any]], ranked_events: list[dict[str, Any]], final_frames: list[dict[str, Any]],
    gt_start_s: float, gt_end_s: float, top_n: int = 20,
) -> dict[str, Any]:
    """Post-hoc-only E2 representability/retrieval/frame diagnostics."""
    if gt_end_s < gt_start_s:
        raise ValueError("Negative GT interval")
    overlaps = [event for event in events if float(event["start_s"]) < gt_end_s and float(event["end_s"]) > gt_start_s]
    relevant_ids = {str(event["event_id"]) for event in overlaps}
    representatives = [float(event["representative_frame"]["selected_timestamp_sec"]) for event in overlaps]
    return {
        "gt_used_for_retrieval": False,
        "event_representability": {
            "gt_overlap_event_exists": bool(overlaps),
            "overlap_event_ids": sorted(relevant_ids),
            "overlap_events": [
                {"event_id": item["event_id"], "start_s": item["start_s"], "end_s": item["end_s"],
                 "overlap_s": max(0.0, min(float(item["end_s"]), gt_end_s) - max(float(item["start_s"]), gt_start_s)),
                 "representative_frame_inside_gt": gt_start_s <= float(item["representative_frame"]["selected_timestamp_sec"]) < gt_end_s,
                }
                for item in overlaps
            ],
            "nearest_representative_frame_distance_s": min((_distance_to_interval(value, gt_start_s, gt_end_s) for value in representatives), default=None),
        },
        "recall_at_20_event": any(str(item["event_id"]) in relevant_ids for item in ranked_events[:top_n]),
        "recall_at_k_event": any(str(item["event_id"]) in relevant_ids for item in ranked_events[:8]),
        "frame_hit_at_8": frame_hit_diagnostics(frames=final_frames, gt_start_s=gt_start_s, gt_end_s=gt_end_s),
    }


def hubness_summary(selections: Iterable[Iterable[dict[str, Any]]]) -> dict[str, Any]:
    """Descriptive selection concentration only; no hard threshold is implied."""
    flattened = [frame for rows in selections for frame in rows]
    counts = Counter((str(item["event_id"]), float(item["timestamp_sec"])) for item in flattened)
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "selected_frame_count": total,
        "unique_event_timestamp_count": len(counts),
        "top_selected_regions": [
            {"event_id": key[0], "timestamp_sec": key[1], "count": count,
             "fraction_of_selected_frames": count / total if total else 0.0}
            for key, count in ordered[:20]
        ],
        "top_region_fraction": ordered[0][1] / total if ordered and total else 0.0,
    }
