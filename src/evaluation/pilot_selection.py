"""Gold-blind, deterministic stratification metadata for the EgoSound pilot."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


TARGET_QUESTION_TYPES = {
    "Sound Source Identification": 4,
    "Sound Characteristics": 3,
    "Cross-Modal Reasoning": 4,
    "Counting": 3,
    "Temporal Information": 2,
    "Spatial Location (Direction & Distance)": 2,
    "Inferential & Contextual Causality": 2,
}

CLOCK = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)")
MODERATE_TIME = re.compile(r"\b(?:at the (?:very )?start|at the beginning|at the end|before|after|between|during)\b", re.I)
WEAK_TIME = re.compile(r"\b(?:when|while|then|following|followed|throughout|span|clip|earlier|later)\b", re.I)
SPEECH = re.compile(r"\b(?:say|said|says|spoken|speech|voice|phrase|conversation|respond|reply|exclaim|remark|counting|mention|mentioned|tell|told|utter|audio line)\b", re.I)
VISUAL = re.compile(r"\b(?:visual|visibly|camera|shown|displayed|focus|object|card|action|where|direction)\b", re.I)


def temporal_metadata(case: dict[str, Any]) -> dict[str, Any]:
    """Describe dataset and raw-question localization without using gold answers."""
    question = str(case.get("question") or "")
    clocks = CLOCK.findall(question)
    if clocks:
        strength = "strong"
    elif MODERATE_TIME.search(question):
        strength = "moderate"
    elif WEAK_TIME.search(question):
        strength = "weak"
    else:
        strength = "none"
    start, end = case.get("provided_timestamp_start"), case.get("provided_timestamp_end")
    available = isinstance(start, (int, float)) and isinstance(end, (int, float))
    return {
        "temporal_hint_strength": strength,
        "provided_timestamp_available": available,
        "provided_timestamp_start_sec": float(start) if available else None,
        "provided_timestamp_end_sec": float(end) if available else None,
        "provided_timestamp_width_sec": max(0.0, float(end) - float(start)) if available else None,
        "question_derived_timestamp_executable": bool(clocks),
        "provided_timestamp_directly_used_by_canonical_runtime": False,
        "timestamp_usage_note": "Canonical retrieval may execute a timestamp parsed from raw question text; the dataset provided_timestamp field remains unavailable to online execution.",
    }


def expected_modalities(case: dict[str, Any]) -> list[str]:
    """Assign selection-only modality strata from question metadata/text."""
    question = str(case.get("question") or "")
    kind = str(case.get("question_type") or "")
    if kind == "Sound Characteristics":
        return ["acoustic"]
    speech_required = bool(SPEECH.search(question)) and not bool(re.search(r"\b(?:excluding|exclude|without)\s+speech\b", question, re.I))
    if kind == "Sound Source Identification":
        return ["speech", "visual"] if speech_required else ["acoustic", "visual"]
    if kind == "Spatial Location (Direction & Distance)":
        return ["acoustic", "visual"]
    if kind == "Cross-Modal Reasoning":
        return ["speech" if speech_required else "acoustic", "visual"]
    if kind == "Counting":
        if speech_required:
            return ["speech"]
        return ["visual"] if re.search(r"\b(?:transfer|place|unwrap|action)\b", question, re.I) else ["acoustic"]
    if kind == "Temporal Information":
        return ["speech"] if speech_required else ["acoustic"]
    if kind == "Inferential & Contextual Causality":
        return ["speech" if speech_required else "acoustic", "visual"]
    return ["visual"] if VISUAL.search(question) else ["acoustic"]


def selection_tags(case: dict[str, Any]) -> list[str]:
    question = str(case.get("question") or "")
    kind = str(case.get("question_type") or "")
    tags: set[str] = set()
    if kind == "Counting":
        tags.add("count_or_aggregation")
    if kind == "Cross-Modal Reasoning" or len(expected_modalities(case)) > 1:
        tags.add("cross_modal")
    if kind == "Sound Characteristics":
        tags.add("direct_acoustic")
    if "speech" in expected_modalities(case):
        tags.add("speech_focused")
    if re.search(r"\b(?:before|after|followed|following|sequence|quickly|time|duration)\b", question, re.I):
        tags.add("temporal_relation")
    if re.search(r"\b(?:source|speaker|male|female|passenger|who|where|direction)\b", question, re.I):
        tags.add("source_or_speaker_attribution")
    width = temporal_metadata(case)["provided_timestamp_width_sec"]
    if kind == "Counting" or (isinstance(width, float) and width >= 10) or re.search(r"\b(?:background|distant|multiple|continuous|throughout|most of the clip)\b", question, re.I):
        tags.add("challenging_or_fallback_likely")
    return sorted(tags)


def _selection_view(row: dict[str, Any]) -> dict[str, Any]:
    """Strict allowlist ensures answers cannot influence stratification."""
    return {key: row.get(key) for key in ("case_id", "video_id", "question", "question_type", "provided_timestamp", "provided_timestamp_start", "provided_timestamp_end", "media_valid", "video_duration", "audio_duration")}


def select_case_ids(rows: list[dict[str, Any]], existing_rows: list[dict[str, Any]], indexed_video_ids: set[str], target_count: int = 20) -> tuple[list[str], dict[str, Any]]:
    """Greedily satisfy type quotas while balancing hints, tags and videos."""
    existing = [_selection_view(row) for row in existing_rows]
    selected = list(existing)
    selected_ids = {str(row["case_id"]) for row in selected}
    eligible = [_selection_view(row) for row in rows if str(row.get("video_id")) in indexed_video_ids and row.get("media_valid") is True and str(row.get("case_id")) not in selected_ids]
    type_counts = Counter(str(row.get("question_type")) for row in selected)
    hint_counts = Counter(temporal_metadata(row)["temporal_hint_strength"] for row in selected)
    video_counts = Counter(str(row.get("video_id")) for row in selected)
    tag_counts = Counter(tag for row in selected for tag in selection_tags(row))
    for question_type, target in TARGET_QUESTION_TYPES.items():
        while type_counts[question_type] < target and len(selected) < target_count:
            candidates = [row for row in eligible if row["question_type"] == question_type and row["case_id"] not in selected_ids]
            if not candidates:
                raise ValueError(f"Insufficient eligible cases for stratum: {question_type}")
            candidates.sort(key=lambda row: (
                hint_counts[temporal_metadata(row)["temporal_hint_strength"]],
                -sum(tag_counts[tag] == 0 for tag in selection_tags(row)),
                video_counts[str(row["video_id"])],
                -float(temporal_metadata(row)["provided_timestamp_width_sec"] or 0.0) if question_type in {"Counting", "Temporal Information"} else 0.0,
                str(row["case_id"]),
            ))
            chosen = candidates[0]
            selected.append(chosen)
            selected_ids.add(str(chosen["case_id"]))
            type_counts[question_type] += 1
            hint_counts[temporal_metadata(chosen)["temporal_hint_strength"]] += 1
            video_counts[str(chosen["video_id"])] += 1
            tag_counts.update(selection_tags(chosen))
    if len(selected) != target_count:
        raise ValueError(f"Stratified selection produced {len(selected)} rather than {target_count} cases")
    audit = {
        "selection_method": "deterministic question-metadata stratification; gold answers and prior predictions excluded",
        "target_question_type_counts": TARGET_QUESTION_TYPES,
        "selected_question_type_counts": dict(sorted(type_counts.items())),
        "selected_hint_counts": dict(sorted(hint_counts.items())),
        "selected_video_counts": dict(sorted(video_counts.items())),
        "selected_tag_counts": dict(sorted(tag_counts.items())),
    }
    return [str(row["case_id"]) for row in selected], audit
