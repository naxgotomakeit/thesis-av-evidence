"""Canonical Task 5C v1.2 local acoustic evidence materialization.

The interval, filename, PCM writing, and sample-rate behavior come directly
from ``scripts/run_task5c_v1_2.py::local_clip_for`` and
``src.retrieval.task5c_v1_2.export_local_clip``.  This module selects no new
evidence; it only materializes already-selected acoustic intervals.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import soundfile as sf

from src.retrieval.task5c_v1_2 import export_local_clip


MATERIALIZED_ROLES = {"direct_evidence", "temporal_anchor", "resolver"}


def materialize_acoustic_candidate(
    candidate: dict[str, Any],
    *,
    case_id: str,
    source_audio: Path,
    output_root: Path,
    role: str,
    project_root: Path,
) -> tuple[str | None, dict[str, Any] | None, str | None, float]:
    """Create/reuse the exact selected-interval WAV and return provenance."""
    if role not in MATERIALIZED_ROLES:
        return None, None, None, 0.0
    start, end = float(candidate["start_time"]), float(candidate["end_time"])
    name = f"{case_id}_{start:.3f}_{end:.3f}_{candidate['candidate_id']}.wav"
    path = output_root / "local_audio_clips" / case_id / name
    started = time.perf_counter()
    reused = False
    try:
        if path.is_file():
            info = sf.info(path)
            duration = float(info.frames) / float(info.samplerate)
            metadata = {
                "requested_selected_interval": {"start_sec": start, "end_sec": end},
                "actual_extracted_interval": {"start_sec": start, "end_sec": start + duration},
                "source_audio_path": str(source_audio), "clip_path": str(path),
                "duration_sec": duration, "sample_rate": int(info.samplerate),
                "source_audio_unchanged": True,
            }
            reused = True
        else:
            metadata = export_local_clip(source_audio, path, start, end)
        try:
            clip_reference = path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            clip_reference = path.as_posix()
        metadata["clip_path"] = clip_reference
        metadata.update({
            "candidate_id": candidate["candidate_id"], "acoustic_evidence_role": role,
            "materialized_now": not reused, "reused_existing": reused,
            "materialization_latency_sec": time.perf_counter() - started,
        })
        return clip_reference, metadata, None, metadata["materialization_latency_sec"]
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, None, f"local_acoustic_clip_export_error:{type(exc).__name__}:{str(exc)[:240]}", elapsed


def materialize_required_acoustic_clips(
    candidates: list[dict[str, Any]],
    *,
    case_id: str,
    source_audio: Path,
    output_root: Path,
    role: str,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[str], float]:
    """Decorate selected acoustic candidates and return unique clip records."""
    clips: list[dict[str, Any]] = []
    warnings: list[str] = []
    total = 0.0
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.get("modality") != "acoustic":
            continue
        reference, metadata, warning, elapsed = materialize_acoustic_candidate(
            candidate, case_id=case_id, source_audio=source_audio,
            output_root=output_root, role=role, project_root=project_root,
        )
        total += elapsed
        candidate["local_audio_clip_reference"] = reference
        if metadata and metadata["clip_path"] not in seen:
            clips.append(metadata)
            seen.add(metadata["clip_path"])
        if warning:
            warnings.append(warning)
    return clips, warnings, total

