"""Formal QaEgo4D E2 event-index preparation (no answer-model inference)."""

from .core import (
    build_b1_prime_groups,
    build_amendment4_fine_events,
    build_amendment4_medium_events,
    build_event_records,
    compute_retrieval_diagnostics,
    frame_hit_diagnostics,
    finalize_ranked_frames,
    retrieve_amendment4_b1,
    retrieve_amendment4_b2,
    retrieve_amendment5_b1,
    retrieve_amendment5_b2,
    retrieve_amendment6_b1,
    retrieve_amendment6_b2,
    select_cradio_medoid,
    select_dino_medoid,
)

__all__ = [
    "build_b1_prime_groups",
    "build_amendment4_fine_events",
    "build_amendment4_medium_events",
    "build_event_records",
    "compute_retrieval_diagnostics",
    "frame_hit_diagnostics",
    "finalize_ranked_frames",
    "retrieve_amendment4_b1",
    "retrieve_amendment4_b2",
    "retrieve_amendment5_b1",
    "retrieve_amendment5_b2",
    "retrieve_amendment6_b1",
    "retrieve_amendment6_b2",
    "select_cradio_medoid",
    "select_dino_medoid",
]
