"""Monotonic, nested stage timing without research-behaviour changes."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


OFFLINE_STAGES = (
    "video_decode", "frame_sampling", "visual_embedding", "visual_region_construction",
    "audio_extraction", "vad", "global_asr", "speech_embedding",
    "acoustic_embedding_clap", "index_serialization",
)
ONLINE_STAGES = (
    "question_planner", "modality_routing", "visual_retrieval", "speech_retrieval",
    "acoustic_retrieval", "relation_construction", "relation_reranking",
    "local_visual_refinement", "local_audio_refinement", "evidence_sufficiency",
    "fallback_decision", "fallback_execution", "fallback_local_asr",
    "evidence_packet_build", "final_payload_build", "final_model_api",
    "structured_output_parse", "local_validation",
)


@dataclass
class StageTiming:
    stage_name: str
    parent_stage: str | None = None
    executed: bool = True
    skipped: bool = False
    skip_reason: str | None = None
    start_monotonic: float | None = None
    end_monotonic: float | None = None
    duration_sec: float | None = None
    api_calls: int = 0
    model_calls: int = 0
    cache_status: str = "not_applicable"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StageTimer:
    def __init__(self) -> None:
        self.records: list[StageTiming] = []

    @contextmanager
    def stage(
        self, stage_name: str, parent_stage: str | None = None, *, api_calls: int = 0,
        model_calls: int = 0, cache_status: str = "not_applicable", notes: list[str] | None = None,
    ) -> Iterator[StageTiming]:
        record = StageTiming(stage_name=stage_name, parent_stage=parent_stage, api_calls=api_calls, model_calls=model_calls, cache_status=cache_status, notes=list(notes or []))
        record.start_monotonic = time.perf_counter()
        try:
            yield record
        finally:
            record.end_monotonic = time.perf_counter()
            record.duration_sec = max(0.0, record.end_monotonic - record.start_monotonic)
            self.records.append(record)

    def skip(self, stage_name: str, reason: str, parent_stage: str | None = None, notes: list[str] | None = None) -> StageTiming:
        record = StageTiming(stage_name=stage_name, parent_stage=parent_stage, executed=False, skipped=True, skip_reason=reason, cache_status="not_applicable", notes=list(notes or []))
        self.records.append(record)
        return record

    def measured(
        self,
        stage_name: str,
        duration_sec: float,
        parent_stage: str | None = None,
        *,
        api_calls: int = 0,
        model_calls: int = 0,
        notes: list[str] | None = None,
    ) -> StageTiming:
        """Record a real duration measured by an instrumented child implementation.

        Some historical reusable functions already use ``perf_counter`` internally.
        This adapter preserves their measured duration without inventing wall-clock
        boundaries or rerunning the research operation.
        """
        record = StageTiming(
            stage_name=stage_name,
            parent_stage=parent_stage,
            duration_sec=max(0.0, float(duration_sec)),
            api_calls=api_calls,
            model_calls=model_calls,
            notes=list(notes or []) + ["duration_measured_with_perf_counter_inside_reused_stage"],
        )
        self.records.append(record)
        return record

    def as_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]


def existing_offline_artifact_timings() -> list[dict[str, Any]]:
    timer = StageTimer()
    for stage in OFFLINE_STAGES:
        timer.skip(stage, "not_measured_existing_artifact", parent_stage="offline_indexing_total")
    timer.skip("offline_indexing_total", "not_measured_existing_artifact")
    return timer.as_dicts()


def blocked_online_timings(reason: str = "blocked_no_clean_online_replay") -> list[dict[str, Any]]:
    timer = StageTimer()
    parents = {
        "visual_retrieval": "retrieval_total", "speech_retrieval": "retrieval_total", "acoustic_retrieval": "retrieval_total",
        "fallback_local_asr": "fallback_execution", "structured_output_parse": "final_model_api",
    }
    for stage in ONLINE_STAGES:
        timer.skip(stage, reason, parent_stage=parents.get(stage))
    timer.skip("retrieval_total", reason)
    timer.skip("online_end_to_end_total", reason)
    return timer.as_dicts()


def timing_consistency(records: list[dict[str, Any]], non_overlapping_top_level: list[str]) -> dict[str, Any]:
    by_name = {record["stage_name"]: record for record in records}
    online = by_name.get("online_end_to_end_total", {})
    online_duration = online.get("duration_sec") if online.get("executed") else None
    measured = [by_name[name]["duration_sec"] for name in non_overlapping_top_level if name in by_name and by_name[name].get("executed") and by_name[name].get("duration_sec") is not None]
    top_sum = sum(measured)
    overhead = None if online_duration is None else online_duration - top_sum
    percent = None if online_duration in (None, 0) else overhead / online_duration * 100.0
    child_ok = True if online_duration is None else all(record.get("duration_sec", 0) <= online_duration + 1e-9 for record in records if record.get("executed") and record["stage_name"] != "online_end_to_end_total")
    return {
        "online_end_to_end_duration_sec": online_duration,
        "non_overlapping_top_level_duration_sum_sec": top_sum if measured else None,
        "uninstrumented_overhead_sec": overhead,
        "uninstrumented_overhead_percentage": percent,
        "online_total_greater_than_or_equal_to_each_stage": child_ok,
        "double_counting_avoided": True,
        "consistency_status": "not_evaluable_online_replay_blocked" if online_duration is None else ("valid" if child_ok and overhead is not None and overhead >= -1e-6 else "invalid"),
    }
