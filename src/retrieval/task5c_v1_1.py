from __future__ import annotations

import copy
import hashlib
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.retrieval.task5c import (
    FALLBACK_MODEL,
    FALLBACK_SAMPLE_RATE,
    FUZZY_MATCH_THRESHOLD,
    classify_evidence,
    local_audio_path,
    normalize_text,
    read_local_audio,
)


EPSILON = 1e-6


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, b_start - a_end, a_start - b_end)


def validate_asr_segment(
    *, raw_start_time: float, raw_end_time: float, pass_decode_start_sec: float, pass_decode_end_sec: float,
    hard_question_start_sec: float, hard_question_end_sec: float, text: str, decoding_pass: str, segment_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (valid/effective segment, excluded segment) using overlap with the actual decode pass."""
    raw_start, raw_end = float(raw_start_time), float(raw_end_time)
    base = {
        "segment_id": segment_id, "decoding_pass": decoding_pass,
        "pass_decode_start_sec": pass_decode_start_sec, "pass_decode_end_sec": pass_decode_end_sec,
        "hard_question_start_sec": hard_question_start_sec, "hard_question_end_sec": hard_question_end_sec,
        "raw_start_time": raw_start, "raw_end_time": raw_end, "text": str(text).strip(), "normalized_text": normalize_text(str(text)),
    }
    if raw_end <= pass_decode_start_sec + EPSILON or raw_start >= pass_decode_end_sec - EPSILON or raw_end <= raw_start:
        base.update({"effective_start_time": None, "effective_end_time": None, "timestamp_validity": "excluded_outside_decode_interval", "warnings": ["asr_segment_outside_decode_interval"]})
        return None, base
    effective_start = max(raw_start, pass_decode_start_sec)
    effective_end = min(raw_end, pass_decode_end_sec)
    clipped = effective_start > raw_start + EPSILON or effective_end < raw_end - EPSILON
    base.update({"effective_start_time": effective_start, "effective_end_time": effective_end, "timestamp_validity": "clipped_to_decode_interval" if clipped else "valid", "warnings": ["asr_segment_clipped_to_decode_interval"] if clipped else []})
    return base, None


def phrase_match(text: str, phrase: str) -> tuple[str | None, float]:
    target, observed = normalize_text(phrase), normalize_text(text)
    if not target or not observed:
        return None, 0.0
    if target in observed:
        return "normalized_exact", 1.0
    tokens, target_tokens = observed.split(), target.split()
    width = max(1, len(target_tokens))
    windows = [" ".join(tokens[index:index + width]) for index in range(max(1, len(tokens) - width + 1))]
    score = max((SequenceMatcher(None, target, window).ratio() for window in windows), default=0.0)
    return ("token_character_fuzzy" if score >= FUZZY_MATCH_THRESHOLD else None), round(score, 6)


def deduplicate_matches(matches: list[dict[str, Any]], tolerance_sec: float = 0.5) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: (item["effective_start_time"], item["effective_end_time"], item["normalized_text"], item["decoding_pass"])):
        existing = next((item for item in unique if item["normalized_text"] == match["normalized_text"] and abs(item["effective_start_time"] - match["effective_start_time"]) <= tolerance_sec), None)
        if existing is None:
            unique.append(match)
        else:
            existing.setdefault("duplicate_decoding_passes", []).append(match["decoding_pass"])
    return unique


def _pass_segments(
    *, transcriber: Callable[[np.ndarray, int], dict[str, Any]], audio: np.ndarray, sample_rate: int,
    local_start: float, local_end: float, decode_start: float, decode_end: float,
    hard_start: float, hard_end: float, pass_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clip = audio[int(round(local_start * sample_rate)):int(round(local_end * sample_rate))]
    decoded = transcriber(clip, sample_rate)
    valid, excluded = [], []
    for index, item in enumerate(decoded.get("segments", [])):
        segment, invalid = validate_asr_segment(
            raw_start_time=decode_start + local_start + float(item.get("start", 0.0)), raw_end_time=decode_start + local_start + float(item.get("end", 0.0)),
            pass_decode_start_sec=decode_start + local_start, pass_decode_end_sec=decode_start + local_end,
            hard_question_start_sec=hard_start, hard_question_end_sec=hard_end,
            text=str(item.get("text", "")), decoding_pass=pass_name, segment_id=f"{pass_name}_{index:03d}",
        )
        if segment is not None:
            valid.append(segment)
        if invalid is not None:
            excluded.append(invalid)
    return valid, excluded


def _matches_for_segments(segments: list[dict[str, Any]], phrases: list[str], hard_start: float, hard_end: float) -> list[dict[str, Any]]:
    matches = []
    for segment in segments:
        for phrase in phrases:
            method, score = phrase_match(segment["text"], phrase)
            if method:
                match = copy.deepcopy(segment)
                match.update({"phrase": phrase, "phrase_match_method": method, "phrase_match_score": score, "within_hard_question_interval": hard_start <= segment["effective_start_time"] < hard_end})
                matches.append(match)
    return deduplicate_matches(matches)


def staged_local_asr_fallback(
    *, case_id: str, video_id: str, search_interval: tuple[float, float], hard_question_interval: tuple[float, float],
    phrases: list[str], source_audio: Path, transcriber: Callable[[np.ndarray, int], dict[str, Any]], model_name: str = FALLBACK_MODEL,
) -> dict[str, Any]:
    """Complete local pass first; run overlapping chunks only if it finds no countable phrase evidence."""
    started = time.perf_counter()
    decode_start, decode_end = search_interval
    hard_start, hard_end = hard_question_interval
    fallback = {
        "triggered": True, "modality": "speech", "strategy": "staged_complete_local_pass_then_optional_4s_windows_2s_stride",
        "decode_interval": {"start_sec": decode_start, "end_sec": decode_end}, "hard_question_interval": {"start_sec": hard_start, "end_sec": hard_end},
        "context_padding_sec": max(0.0, decode_end - hard_end), "source_audio_path": str(source_audio), "model": f"whisper-{model_name}",
        "model_calls": 0, "local_audio_duration_sec": 0.0, "total_decoding_audio_duration_sec": 0.0,
        "decoding_passes": [], "valid_transcript_segments": [], "excluded_transcript_segments": [], "phrase_matches": [], "added_candidates": [],
        "early_stop_after_complete_pass": False, "chunk_fallback_triggered": False, "chunk_fallback_reason": None,
        "warnings": [], "outcome": "still_insufficient", "latency_sec": 0.0,
    }
    try:
        audio, sample_rate = read_local_audio(source_audio, decode_start, decode_end)
        duration = len(audio) / sample_rate
        fallback["local_audio_duration_sec"] = duration
        complete_valid, complete_excluded = _pass_segments(
            transcriber=transcriber, audio=audio, sample_rate=sample_rate, local_start=0.0, local_end=duration,
            decode_start=decode_start, decode_end=decode_end, hard_start=hard_start, hard_end=hard_end, pass_name="complete_interval",
        )
        fallback["model_calls"] = 1
        fallback["total_decoding_audio_duration_sec"] = duration
        fallback["decoding_passes"].append({"name": "complete_interval", "pass_decode_start_sec": decode_start, "pass_decode_end_sec": decode_end})
        fallback["valid_transcript_segments"].extend(complete_valid)
        fallback["excluded_transcript_segments"].extend(complete_excluded)
        complete_matches = _matches_for_segments(complete_valid, phrases, hard_start, hard_end)
        usable = [item for item in complete_matches if item["within_hard_question_interval"]]
        if usable:
            fallback["early_stop_after_complete_pass"] = True
            fallback["chunk_fallback_reason"] = "complete_pass_recovered_usable_phrase_evidence"
            fallback["phrase_matches"] = usable
        else:
            fallback["chunk_fallback_triggered"] = True
            fallback["chunk_fallback_reason"] = "complete_pass_found_no_usable_phrase_evidence"
            chunk_start, chunk_index = 0.0, 0
            all_matches = list(complete_matches)
            while chunk_start < duration:
                chunk_end = min(duration, chunk_start + 4.0)
                valid, excluded = _pass_segments(
                    transcriber=transcriber, audio=audio, sample_rate=sample_rate, local_start=chunk_start, local_end=chunk_end,
                    decode_start=decode_start, decode_end=decode_end, hard_start=hard_start, hard_end=hard_end, pass_name=f"chunk_{chunk_index:03d}",
                )
                fallback["model_calls"] += 1
                fallback["total_decoding_audio_duration_sec"] += chunk_end - chunk_start
                fallback["decoding_passes"].append({"name": f"chunk_{chunk_index:03d}", "pass_decode_start_sec": decode_start + chunk_start, "pass_decode_end_sec": decode_start + chunk_end})
                fallback["valid_transcript_segments"].extend(valid)
                fallback["excluded_transcript_segments"].extend(excluded)
                all_matches.extend(_matches_for_segments(valid, phrases, hard_start, hard_end))
                chunk_start += 2.0
                chunk_index += 1
            fallback["phrase_matches"] = [item for item in deduplicate_matches(all_matches) if item["within_hard_question_interval"]]
        for match in fallback["phrase_matches"]:
            fallback["added_candidates"].append({
                "candidate_id": stable_id("task5c_local_asr", case_id, match["normalized_text"], f'{match["effective_start_time"]:.3f}', f'{match["effective_end_time"]:.3f}'),
                "candidate_type": "speech_segment", "modality": "speech", "transcript_text": match["text"],
                "start_time": match["effective_start_time"], "end_time": match["effective_end_time"],
                "raw_start_time": match["raw_start_time"], "raw_end_time": match["raw_end_time"],
                "effective_start_time": match["effective_start_time"], "effective_end_time": match["effective_end_time"],
                "timestamp_validity": match["timestamp_validity"], "normalized_text": match["normalized_text"],
                "phrase_match_method": match["phrase_match_method"], "phrase_match_score": match["phrase_match_score"],
                "source_audio_path": str(source_audio), "source": "local_asr_fallback", "decoding_pass": match["decoding_pass"], "warnings": list(match["warnings"]),
            })
        fallback["outcome"] = "recovered_evidence" if fallback["added_candidates"] else "still_insufficient"
    except Exception as exc:
        fallback["outcome"] = "failed_with_error"
        fallback["warnings"].append(f"local_asr_fallback_error:{type(exc).__name__}:{str(exc)[:240]}")
    fallback["latency_sec"] = time.perf_counter() - started
    return fallback


def broad_acoustic_diagnostics(candidates: list[dict[str, Any]], plan: dict[str, Any], tolerance: float = 1e-4) -> list[dict[str, Any]]:
    required = "acoustic" in set(plan.get("resolver_modalities", [])) or plan.get("audio_role") == "direct_answer"
    diagnostics = []
    for candidate in candidates:
        if candidate.get("modality") != "acoustic":
            continue
        source_start, source_end = candidate.get("source_start_time"), candidate.get("source_end_time")
        clipped = source_start is not None and source_end is not None and (abs(float(source_start) - float(candidate["start_time"])) > tolerance or abs(float(source_end) - float(candidate["end_time"])) > tolerance)
        locally_scored = any(candidate.get(key) is not None for key in ("local_acoustic_score", "local_acoustic_similarity_score", "locally_recomputed_acoustic_score"))
        ambiguous = clipped and candidate.get("local_audio_clip_reference") is None and not locally_scored
        diagnostics.append({"candidate_id": candidate["candidate_id"], "source_start_time": source_start, "source_end_time": source_end, "selected_start_time": candidate["start_time"], "selected_end_time": candidate["end_time"], "clipped_from_broader_source": clipped, "local_audio_clip_reference": candidate.get("local_audio_clip_reference"), "locally_recomputed_acoustic_score": locally_scored, "required_as_direct_or_resolver_evidence": required, "ambiguity_codes": ["broad_source_acoustic_region", "local_acoustic_semantics_not_verified"] if ambiguous else []})
    return diagnostics


def classify_v1_1(record_context: dict[str, Any], plan: dict[str, Any], cues: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assessment = classify_evidence(record_context, plan, cues, candidates)
    acoustic = broad_acoustic_diagnostics(candidates, plan)
    broad_required = any(item["ambiguity_codes"] and item["required_as_direct_or_resolver_evidence"] for item in acoustic)
    if broad_required and assessment["evidence_status"] == "sufficient":
        assessment["evidence_status"] = "questionable"
    codes = set(assessment["sufficiency_reason_codes"])
    flags = set(assessment["ambiguity_flags"])
    for item in acoustic:
        if item["ambiguity_codes"] and item["required_as_direct_or_resolver_evidence"]:
            codes.update(item["ambiguity_codes"])
            flags.update(item["ambiguity_codes"])
    assessment["sufficiency_reason_codes"] = sorted(codes)
    assessment["ambiguity_flags"] = sorted(flags)
    assessment["fallback_required"] = assessment["evidence_status"] == "insufficient" and "speech" in set(plan.get("resolver_modalities", [])) and bool(cues.get("quoted_phrases"))
    return assessment, acoustic


def visual_accounting(task5b_input: dict[str, Any], legacy_count: int | None = None) -> dict[str, Any]:
    local = task5b_input.get("local_visual_refinement", {})
    micro = local.get("micro_windows", [])
    dense = local.get("dense_frames", [])
    canonical = task5b_input.get("selected_visual_evidence_frames", [])
    return {
        "source_micro_frame_count": sum(len(item.get("source_frame_paths", [])) for item in micro),
        "selected_micro_frame_count": sum(len(item.get("selected_frame_paths", item.get("frame_paths", []))) for item in micro),
        "dense_frame_count": len(dense),
        "unique_selected_visual_frame_count": len(canonical),
        "selected_visual_frames": len(canonical),
        "legacy_task5c_visual_frame_count": legacy_count,
        "deduplication_policy": "canonical Task 5B v1.1 selected_visual_evidence_frames; dense provenance is preferred when timestamps duplicate",
    }


def evaluate_candidates(candidates: list[dict[str, Any]], reference: tuple[float, float]) -> dict[str, Any]:
    overlaps = [interval_overlap(float(item["start_time"]), float(item["end_time"]), *reference) for item in candidates]
    distances = [interval_distance(float(item["start_time"]), float(item["end_time"]), *reference) for item in candidates]
    hit = any(value > 0 for value in overlaps)
    return {"reference_role": "post-hoc weak reference only; not a precise gold boundary and not used for evidence generation", "reference_interval_hit": hit, "maximum_temporal_overlap_sec": max(overlaps, default=0.0), "minimum_distance_to_reference_sec": min(distances, default=None), "selected_evidence_recall": 1.0 if hit else 0.0, "selected_candidate_count": len(candidates)}
