from __future__ import annotations

import copy
import hashlib
import io
import re
import shutil
import subprocess
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


FALLBACK_MODEL = "small"
FALLBACK_SAMPLE_RATE = 16000
# Conservative, configurable fallback threshold: exact normalized matching is preferred.
FUZZY_MATCH_THRESHOLD = 0.85


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_text(text: str) -> str:
    """Unicode/case/punctuation/whitespace normalisation for deterministic matching."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in normalized).split())


def quoted_phrases(cues: dict[str, Any]) -> list[str]:
    return [str(item["text"]) for item in cues.get("quoted_phrases", []) if item.get("text")]


def _phrase_match(text: str, phrase: str, fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD) -> tuple[str | None, float, str]:
    target, observed = normalize_text(phrase), normalize_text(text)
    if not target or not observed:
        return None, 0.0, observed
    if target in observed:
        return "normalized_exact", 1.0, observed
    target_tokens, tokens = target.split(), observed.split()
    width = max(1, len(target_tokens))
    alternatives = [" ".join(tokens[index:index + width]) for index in range(max(1, len(tokens) - width + 1))]
    score = max((SequenceMatcher(None, target, item).ratio() for item in alternatives), default=0.0)
    return ("token_character_fuzzy" if score >= fuzzy_threshold else None), round(score, 6), observed


def _speech_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in candidates if item.get("modality") == "speech" and str(item.get("transcript_text") or item.get("text") or "").strip()]


def _phrase_evidence(candidates: list[dict[str, Any]], phrases: list[str], interval: tuple[float, float] | None = None) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for candidate in _speech_candidates(candidates):
        start, end = float(candidate["start_time"]), float(candidate["end_time"])
        if interval and not (start < interval[1] and interval[0] < end):
            continue
        text = str(candidate.get("transcript_text") or candidate.get("text") or "")
        for phrase in phrases:
            method, score, normalized = _phrase_match(text, phrase)
            if method:
                matches.append({"candidate_id": candidate["candidate_id"], "phrase": phrase, "start_time": start, "end_time": end, "method": method, "score": score, "normalized_text": normalized})
    return matches


def explicit_interval(cues: dict[str, Any], anchor_resolution: dict[str, Any]) -> tuple[float, float] | None:
    for cue in cues.get("time_cues", []):
        if cue.get("start_sec") is not None:
            return float(cue["start_sec"]), float(cue.get("end_sec", cue["start_sec"]))
    for item in anchor_resolution.get("raw_question_intervals", []):
        if item.get("start_sec") is not None:
            return float(item["start_sec"]), float(item.get("end_sec", item["start_sec"]))
    return None


def classify_evidence(record: dict[str, Any], plan: dict[str, Any], cues: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify structural evidence sufficiency without inspecting references or answers."""
    required = set(plan.get("resolver_modalities", []))
    operation = str(plan.get("answer_requirement", {}).get("operation", "other"))
    phrase_list = quoted_phrases(cues)
    hard_interval = explicit_interval(cues, record.get("anchor_resolution", {}))
    reason_codes: list[str] = []
    critical: list[dict[str, str]] = []
    ambiguity: list[str] = []

    if not candidates:
        reason_codes.append("no_selected_candidates")
        critical.append({"type": "no_selected_candidates", "detail": "No selected evidence candidate is available."})

    modality_candidates = {modality: [item for item in candidates if item.get("modality") == modality] for modality in ("speech", "acoustic", "visual")}
    for modality in sorted(required):
        usable = bool(modality_candidates.get(modality))
        if modality == "speech":
            usable = bool(_speech_candidates(candidates))
        if modality == "visual":
            usable = bool(record.get("selected_visual_evidence_frames"))
        if not usable:
            code = f"missing_required_{modality}_evidence"
            reason_codes.append(code)
            critical.append({"type": code, "detail": f"Required {modality} resolver has no usable selected evidence."})

    phrase_matches = _phrase_evidence(candidates, phrase_list, hard_interval)
    if phrase_list and "speech" in required and not phrase_matches:
        reason_codes.append("quoted_phrase_not_found")
        critical.append({"type": "quoted_phrase_not_found", "detail": "No plausible quoted-speech phrase occurrence is present in the candidate evidence."})

    if operation == "count_occurrences" and not phrase_matches:
        reason_codes.append("count_target_phrase_not_found")
        critical.append({"type": "count_target_phrase_not_found", "detail": "No plausible target phrase occurrence is available inside the requested interval."})

    if operation == "measure_delay":
        trigger_matches = phrase_matches
        if not trigger_matches:
            reason_codes.append("delay_trigger_missing")
            critical.append({"type": "delay_trigger_missing", "detail": "The quoted speech trigger is missing."})
        else:
            trigger_end = min(match["end_time"] for match in trigger_matches)
            responses = [item for item in _speech_candidates(candidates) if float(item["start_time"]) >= trigger_end and item["candidate_id"] not in {match["candidate_id"] for match in trigger_matches}]
            if not responses:
                reason_codes.append("delay_response_missing")
                critical.append({"type": "delay_response_missing", "detail": "No plausible speech response follows the trigger."})
            elif len(responses) > 1:
                ambiguity.append("multiple_plausible_responses")
    question = str(record.get("question", "")).casefold()
    if any(token in question for token in ("passenger", "driver", "male", "female", "speaker", "user")):
        ambiguity.append("unresolved_speaker_attribution")

    local = record.get("local_visual_refinement", {})
    if plan.get("requires_local_visual_inspection") and not record.get("selected_visual_evidence_frames"):
        reason_codes.append("required_local_visual_refinement_has_no_canonical_frames")
        critical.append({"type": "required_local_visual_refinement_has_no_canonical_frames", "detail": "Required local visual inspection produced no canonical selected frames."})

    candidate_warnings = [warning for item in candidates for warning in item.get("warnings", [])]
    if any("broad_source" in warning for warning in candidate_warnings):
        ambiguity.append("broad_source_region")
    if any("ambigu" in warning for warning in candidate_warnings):
        ambiguity.append("candidate_explicit_ambiguity_warning")
    if plan.get("requires_local_visual_inspection") and local.get("executed") and any("micro_window_is_detailed_candidate_not_event_label" in warning for warning in candidate_warnings):
        # This is provenance metadata, not a failure; semantic inspection remains a human/future-model task.
        pass

    if critical:
        status = "insufficient"
    elif ambiguity:
        status = "questionable"
    else:
        status = "sufficient"
    fallback_required = status == "insufficient" and "speech" in required and bool(phrase_list)
    return {
        "evidence_status": status,
        "sufficiency_reason_codes": sorted(set(reason_codes + ambiguity)),
        "critical_missing_evidence": critical,
        "ambiguity_flags": sorted(set(ambiguity)),
        "fallback_required": fallback_required,
        "phrase_evidence": phrase_matches,
        "structural_sufficiency_only": True,
        "interpretation": "Structural evidence status only; it does not establish semantic correctness or a final answer.",
    }


def select_fallback_interval(record: dict[str, Any], cues: dict[str, Any]) -> tuple[float, float] | None:
    """Use explicit question time first; never expand it during Task 5C v1 fallback."""
    hard = explicit_interval(cues, record.get("anchor_resolution", {}))
    if hard:
        return hard
    intervals = record.get("anchor_resolution", {}).get("final_search_intervals", [])
    if intervals:
        item = intervals[0]
        return float(item["start_sec"]), float(item["end_sec"])
    return None


def local_audio_path(video_id: str, source_mp4_path: str) -> Path:
    wav = Path("D:/ThesisData/EgoSound/data/Ego4d/audios") / f"{video_id}.wav"
    return wav if wav.is_file() else Path(source_mp4_path)


def read_local_audio(path: Path, start_sec: float, end_sec: float, target_sample_rate: int = FALLBACK_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Read only a local WAV interval; MP4 support uses ffmpeg only if WAV is unavailable."""
    if path.suffix.casefold() != ".wav":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("local_mp4_audio_fallback_unavailable_without_ffmpeg")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{start_sec:.6f}", "-t", f"{max(0.0, end_sec - start_sec):.6f}", "-i", str(path), "-vn", "-ac", "1", "-ar", str(target_sample_rate), "-f", "wav", "-"]
        result = subprocess.run(command, capture_output=True, check=False, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"local_mp4_audio_extraction_failed:{result.stderr.decode(errors='replace')[:200]}")
        samples, sample_rate = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
        return np.asarray(samples.mean(axis=1), dtype=np.float32), int(sample_rate)
    with sf.SoundFile(path) as handle:
        source_rate = int(handle.samplerate)
        start_frame = max(0, int(round(start_sec * source_rate)))
        end_frame = max(start_frame, int(round(end_sec * source_rate)))
        handle.seek(start_frame)
        samples = handle.read(end_frame - start_frame, dtype="float32", always_2d=True)
    mono = np.asarray(samples.mean(axis=1), dtype=np.float32)
    if source_rate != target_sample_rate and mono.size:
        divisor = np.gcd(source_rate, target_sample_rate)
        mono = resample_poly(mono, target_sample_rate // divisor, source_rate // divisor).astype(np.float32)
    return mono, target_sample_rate


def _real_transcriber(model: Any, device: str) -> Callable[[np.ndarray, int], dict[str, Any]]:
    def transcribe(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        if sample_rate != FALLBACK_SAMPLE_RATE:
            raise ValueError("Whisper fallback expects 16 kHz audio")
        return model.transcribe(audio, fp16=device == "cuda", word_timestamps=False, condition_on_previous_text=False, verbose=None)
    return transcribe


def deduplicate_phrase_matches(matches: list[dict[str, Any]], tolerance_sec: float = 0.5) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: (item["start_time"], item["end_time"], item["normalized_text"], item["decoding_pass"])):
        duplicate = next((prior for prior in unique if prior["normalized_text"] == match["normalized_text"] and abs(prior["start_time"] - match["start_time"]) <= tolerance_sec), None)
        if duplicate is None:
            unique.append(match)
        else:
            duplicate.setdefault("duplicate_decoding_passes", []).append(match["decoding_pass"])
    return unique


def run_local_asr_fallback(
    *,
    case_id: str,
    video_id: str,
    record: dict[str, Any],
    cues: dict[str, Any],
    source_audio: Path,
    transcriber: Callable[[np.ndarray, int], dict[str, Any]],
    model_name: str = FALLBACK_MODEL,
) -> dict[str, Any]:
    """Perform deterministic complete-interval plus overlapping-chunk local ASR."""
    started = time.perf_counter()
    interval = select_fallback_interval(record, cues)
    base = {
        "triggered": True, "trigger_reasons": ["critical_speech_evidence_missing"], "modality": "speech",
        "strategy": "local_interval_whisper_complete_pass_plus_4s_windows_2s_stride", "search_interval": None,
        "source_audio_path": str(source_audio), "local_audio_duration_sec": 0.0, "total_decoding_audio_duration_sec": 0.0, "model": f"whisper-{model_name}",
        "model_calls": 0, "decoding_passes": [], "local_transcript_segments": [], "phrase_matches": [], "added_candidates": [],
        "latency_sec": 0.0, "outcome": "still_insufficient", "warnings": [],
    }
    if interval is None:
        base.update({"outcome": "failed_with_error", "warnings": ["no_local_interval_available_for_speech_fallback"], "latency_sec": time.perf_counter() - started})
        return base
    start, end = interval
    base["search_interval"] = {"start_sec": start, "end_sec": end, "hard_question_constraint_preserved": explicit_interval(cues, record.get("anchor_resolution", {})) is not None}
    try:
        audio, sample_rate = read_local_audio(source_audio, start, end)
        duration = len(audio) / sample_rate
        base["local_audio_duration_sec"] = duration
        passes = [("complete_interval", 0.0, duration)]
        chunk_start = 0.0
        chunk_index = 0
        while chunk_start < duration:
            passes.append((f"chunk_{chunk_index:03d}", chunk_start, min(duration, chunk_start + 4.0)))
            chunk_start += 2.0
            chunk_index += 1
        phrase_list = quoted_phrases(cues)
        detections: list[dict[str, Any]] = []
        for pass_name, local_start, local_end in passes:
            if local_end <= local_start:
                continue
            result = transcriber(audio[int(round(local_start * sample_rate)):int(round(local_end * sample_rate))], sample_rate)
            base["model_calls"] += 1
            base["decoding_passes"].append({"name": pass_name, "local_start_sec": local_start, "local_end_sec": local_end})
            base["total_decoding_audio_duration_sec"] += local_end - local_start
            for index, segment in enumerate(result.get("segments", [])):
                absolute_start = start + local_start + float(segment.get("start", 0.0))
                absolute_end = start + local_start + float(segment.get("end", 0.0))
                text = str(segment.get("text", "")).strip()
                transcript = {"segment_id": f"{pass_name}_{index:03d}", "decoding_pass": pass_name, "start_time": round(absolute_start, 6), "end_time": round(absolute_end, 6), "text": text, "normalized_text": normalize_text(text)}
                base["local_transcript_segments"].append(transcript)
                for phrase in phrase_list:
                    method, score, normalized = _phrase_match(text, phrase)
                    if method:
                        detections.append({"phrase": phrase, "start_time": transcript["start_time"], "end_time": transcript["end_time"], "transcript_text": text, "normalized_text": normalized, "phrase_match_method": method, "phrase_match_score": score, "decoding_pass": pass_name})
        base["phrase_matches"] = deduplicate_phrase_matches(detections)
        for match in base["phrase_matches"]:
            candidate = {
                "candidate_id": stable_id("task5c_local_asr", case_id, match["normalized_text"], f'{match["start_time"]:.3f}', f'{match["end_time"]:.3f}'),
                "candidate_type": "speech_segment", "modality": "speech", "transcript_text": match["transcript_text"],
                "start_time": match["start_time"], "end_time": match["end_time"], "normalized_text": match["normalized_text"],
                "phrase_match_method": match["phrase_match_method"], "phrase_match_score": match["phrase_match_score"],
                "source_audio_path": str(source_audio), "source": "local_asr_fallback", "decoding_pass": match["decoding_pass"], "warnings": [],
            }
            base["added_candidates"].append(candidate)
        base["outcome"] = "recovered_evidence" if base["added_candidates"] else "still_insufficient"
    except Exception as exc:  # Recorded, never converted into an invented phrase match.
        base["outcome"] = "failed_with_error"
        base["warnings"].append(f"local_asr_fallback_error:{type(exc).__name__}:{str(exc)[:240]}")
    base["latency_sec"] = time.perf_counter() - started
    return base


def no_fallback() -> dict[str, Any]:
    return {"triggered": False, "trigger_reasons": [], "modality": "speech", "strategy": "not_triggered", "search_interval": None, "source_audio_path": None, "local_audio_duration_sec": 0.0, "total_decoding_audio_duration_sec": 0.0, "model": f"whisper-{FALLBACK_MODEL}", "model_calls": 0, "decoding_passes": [], "local_transcript_segments": [], "phrase_matches": [], "added_candidates": [], "latency_sec": 0.0, "outcome": "not_triggered", "warnings": []}
