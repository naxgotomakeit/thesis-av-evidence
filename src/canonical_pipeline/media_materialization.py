"""Canonical Task 5C v1.2 local acoustic evidence materialization.

The selected interval and PCM export behavior come from the frozen Task 5C
v1.2 implementation.  This module never selects evidence.  It adds a
deterministic provenance contract so a readable but stale WAV cannot be
silently reused, and it can materialize the final Task6 model-facing acoustic
entities without changing Task6 membership.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from src.retrieval.task5c_v1_2 import export_local_clip


AUDIO_MATERIALIZATION_SCHEMA_VERSION = "task5c-v1.2-audio-materialization-v2"
MATERIALIZED_ROLES = {"direct_evidence", "temporal_anchor", "resolver"}
EXPORT_POLICY = {
    "container": "WAV",
    "reader": "src.retrieval.task5c.read_local_audio",
    "writer": "soundfile.write",
    "target_sample_rate_hz": 16000,
    "target_channels": 1,
    "subtype_policy": "soundfile_default_for_wav",
    "interval_policy": "exact_selected_interval_clipped_to_source_bounds",
}


class AudioMaterializationError(RuntimeError):
    """Raised when selected model-facing acoustic evidence has no valid WAV."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_audio_identity(source_audio: Path) -> dict[str, Any]:
    """Return deterministic non-secret identity for one source audio file."""
    resolved = source_audio.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source audio is missing: {resolved}")
    return {
        "resolved_path": resolved.as_posix(),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _expected_provenance(
    source_identity: dict[str, Any], start: float, end: float
) -> dict[str, Any]:
    return {
        "schema_version": AUDIO_MATERIALIZATION_SCHEMA_VERSION,
        "source_audio": source_identity,
        "requested_interval": {"start_sec": start, "end_sec": end},
        "export_policy": EXPORT_POLICY,
    }


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".provenance.json")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _reuse_matches(
    path: Path,
    sidecar: Path,
    expected: dict[str, Any],
) -> tuple[bool, str | None, dict[str, Any] | None]:
    if not path.is_file():
        return False, "wav_missing", None
    saved = _read_json(sidecar)
    if saved is None:
        return False, "provenance_sidecar_missing_or_invalid", None
    if saved.get("contract") != expected:
        return False, "provenance_contract_mismatch", saved
    try:
        info = sf.info(path)
    except Exception:
        return False, "wav_unreadable", saved
    media = saved.get("materialized_media") or {}
    if media.get("sha256") != _sha256(path):
        return False, "wav_content_hash_mismatch", saved
    actual = {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }
    if any(media.get(key) != value for key, value in actual.items()):
        return False, "wav_media_metadata_mismatch", saved
    return True, None, saved


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_acoustic_candidate(
    candidate: dict[str, Any],
    *,
    case_id: str,
    source_audio: Path,
    output_root: Path,
    role: str,
    project_root: Path,
    model_facing: bool = False,
    source_identity: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None, str | None, float]:
    """Create/reuse one exact selected-interval WAV with verified provenance."""
    if role not in MATERIALIZED_ROLES and not model_facing:
        return None, None, None, 0.0
    start, end = float(candidate["start_time"]), float(candidate["end_time"])
    name = f"{case_id}_{start:.3f}_{end:.3f}_{candidate['candidate_id']}.wav"
    path = output_root / "local_audio_clips" / case_id / name
    sidecar = _sidecar_path(path)
    started = time.perf_counter()
    reuse_rejected_reason: str | None = None
    try:
        identity = source_identity or source_audio_identity(source_audio)
        expected = _expected_provenance(identity, start, end)
        reusable, reuse_rejected_reason, saved = _reuse_matches(path, sidecar, expected)
        if reusable:
            info = sf.info(path)
            duration = float(info.frames) / float(info.samplerate)
            export_metadata = {
                "requested_selected_interval": {"start_sec": start, "end_sec": end},
                "actual_extracted_interval": {"start_sec": start, "end_sec": start + duration},
                "source_audio_path": str(source_audio),
                "clip_path": str(path),
                "duration_sec": duration,
                "sample_rate": int(info.samplerate),
                "source_audio_unchanged": True,
            }
            provenance = saved or {}
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_wav = path.with_name(path.stem + ".tmp" + path.suffix)
            export_metadata = export_local_clip(source_audio, temporary_wav, start, end)
            temporary_wav.replace(path)
            info = sf.info(path)
            duration = float(info.frames) / float(info.samplerate)
            provenance = {
                "contract": expected,
                "materialized_media": {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                    "format": str(info.format),
                    "subtype": str(info.subtype),
                    "duration_sec": duration,
                    "actual_interval": {"start_sec": start, "end_sec": start + duration},
                },
            }
            _atomic_json(sidecar, provenance)
        try:
            clip_reference = path.resolve().relative_to(project_root.resolve()).as_posix()
            sidecar_reference = sidecar.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            clip_reference = path.as_posix()
            sidecar_reference = sidecar.as_posix()
        export_metadata.update(
            {
                "clip_path": clip_reference,
                "provenance_path": sidecar_reference,
                "provenance_schema_version": AUDIO_MATERIALIZATION_SCHEMA_VERSION,
                "source_audio_identity": identity,
                "export_policy": EXPORT_POLICY,
                "candidate_id": candidate["candidate_id"],
                "acoustic_evidence_role": role,
                "model_facing_materialization": model_facing,
                "materialized_now": not reusable,
                "reused_existing": reusable,
                "reuse_rejected_reason": reuse_rejected_reason if path.is_file() and not reusable else None,
                "materialization_latency_sec": time.perf_counter() - started,
            }
        )
        return clip_reference, export_metadata, None, export_metadata["materialization_latency_sec"]
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return (
            None,
            None,
            f"local_acoustic_clip_export_error:{type(exc).__name__}:{str(exc)[:240]}",
            elapsed,
        )


def materialize_required_acoustic_clips(
    candidates: list[dict[str, Any]],
    *,
    case_id: str,
    source_audio: Path,
    output_root: Path,
    role: str,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[str], float]:
    """Decorate Task5C role-required candidates; this selects no evidence."""
    acoustic = [item for item in candidates if item.get("modality") == "acoustic"]
    if not acoustic or role not in MATERIALIZED_ROLES:
        return [], [], 0.0
    identity = source_audio_identity(source_audio)
    clips: list[dict[str, Any]] = []
    warnings: list[str] = []
    total = 0.0
    seen: set[str] = set()
    for candidate in acoustic:
        reference, metadata, warning, elapsed = materialize_acoustic_candidate(
            candidate,
            case_id=case_id,
            source_audio=source_audio,
            output_root=output_root,
            role=role,
            project_root=project_root,
            source_identity=identity,
        )
        total += elapsed
        candidate["local_audio_clip_reference"] = reference
        if metadata and metadata["clip_path"] not in seen:
            clips.append(metadata)
            seen.add(metadata["clip_path"])
        if warning:
            warnings.append(warning)
    return clips, warnings, total


def materialize_model_facing_acoustic_entities(
    packet: dict[str, Any],
    *,
    case_id: str,
    source_audio: Path,
    output_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Ensure exactly the already model-facing Task6 acoustic entities have WAVs."""
    model_facing_ids = {
        member["candidate_id"]
        for group in packet.get("retained_evidence_groups", [])
        for member in group.get("retained_candidates", [])
        if member.get("modality") == "acoustic"
    }
    retained = {
        item["candidate_id"]: item
        for item in packet.get("retained_candidates", [])
        if item.get("modality") == "acoustic"
    }
    unknown = model_facing_ids - set(retained)
    if unknown:
        raise AudioMaterializationError(
            f"Task6 groups reference unknown acoustic evidence: {sorted(unknown)}"
        )
    if not model_facing_ids:
        return {
            "model_facing_acoustic_ids": [],
            "materialized_clip_count": 0,
            "materialization_latency_sec": 0.0,
            "warnings": [],
        }
    identity = source_audio_identity(source_audio)
    clips_by_id: dict[str, dict[str, Any]] = {
        item.get("candidate_id"): item
        for item in packet.get("local_audio_clips", [])
        if item.get("candidate_id") in model_facing_ids
    }
    warnings: list[str] = []
    total = 0.0
    for candidate_id in sorted(model_facing_ids):
        candidate = retained[candidate_id]
        role = str(candidate.get("acoustic_evidence_role") or "model_facing")
        reference, metadata, warning, elapsed = materialize_acoustic_candidate(
            candidate,
            case_id=case_id,
            source_audio=source_audio,
            output_root=output_root,
            role=role,
            project_root=project_root,
            model_facing=True,
            source_identity=identity,
        )
        total += elapsed
        if warning or not reference or metadata is None:
            raise AudioMaterializationError(
                f"Model-facing acoustic evidence lacks a valid WAV: {candidate_id}; {warning}"
            )
        candidate["local_audio_clip_reference"] = reference
        clips_by_id[candidate_id] = metadata
    for group in packet.get("retained_evidence_groups", []):
        for member in group.get("retained_candidates", []):
            if member.get("candidate_id") in model_facing_ids:
                member["local_audio_clip_reference"] = retained[member["candidate_id"]][
                    "local_audio_clip_reference"
                ]
    packet["local_audio_clips"] = [clips_by_id[key] for key in sorted(clips_by_id)]
    audit = {
        "contract": "all_and_only_task6_model_facing_acoustic_entities_have_verified_wav",
        "model_facing_acoustic_ids": sorted(model_facing_ids),
        "materialized_clip_count": len(model_facing_ids),
        "materialization_latency_sec": total,
        "warnings": warnings,
        "selection_changed": False,
    }
    packet["model_facing_acoustic_materialization"] = audit
    return audit
