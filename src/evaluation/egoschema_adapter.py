"""Leakage-safe EgoSchema I/O adapter for the frozen canonical baseline.

This module changes no planner, retrieval, sufficiency, reranking, or final-QA
policy. It only separates retrieval-visible fields from final multiple-choice
fields and post-hoc labels.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "egoschema-comparison-pilot-v1"


class EgoSchemaAdapterError(ValueError):
    """Raised when the comparison manifest violates the adapter contract."""


class RetrievalProtocol(str, Enum):
    """Fields visible while selecting evidence."""

    STANDARD_MULTIPLE_CHOICE = "protocol_a_standard_multiple_choice"
    QUESTION_ONLY = "protocol_b_question_only_retrieval"


@dataclass(frozen=True)
class EgoSchemaRuntimeCase:
    """Gold-free case data shared by retrieval and final selection adapters."""

    case_id: str
    video_id: str
    question: str
    options: tuple[str, str, str, str, str]
    video_path: str
    duration_sec: float
    audio_available: bool


def _validated_source(path: Path) -> dict[str, Any]:
    """Read and validate the complete on-disk manifest, including post-hoc data."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_version") != MANIFEST_VERSION:
        raise EgoSchemaAdapterError(
            f"Unsupported EgoSchema manifest version: {value.get('manifest_version')!r}"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or not 20 <= len(cases) <= 30:
        raise EgoSchemaAdapterError("EgoSchema pilot must contain 20–30 cases")
    case_ids = [row.get("case_id") for row in cases]
    video_ids = [row.get("video_id") for row in cases]
    if len(case_ids) != len(set(case_ids)) or len(video_ids) != len(set(video_ids)):
        raise EgoSchemaAdapterError("Pilot cases and videos must be unique")
    for row in cases:
        options = row.get("options")
        if not isinstance(options, list) or len(options) != 5 or not all(
            isinstance(option, str) and option.strip() for option in options
        ):
            raise EgoSchemaAdapterError(f"Case {row.get('case_id')} needs five options")
        forbidden = {"answer", "gold", "gold_label", "correct_option"}.intersection(row)
        if forbidden:
            raise EgoSchemaAdapterError(
                f"Runtime case contains post-hoc fields: {row.get('case_id')} {sorted(forbidden)}"
            )
    labels = value.get("posthoc_evaluation", {}).get("labels_by_case_id", {})
    if set(labels) != set(case_ids):
        raise EgoSchemaAdapterError("Post-hoc labels must cover exactly the runtime cases")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    """Load only runtime-safe fields; gold is not returned to online code."""
    value = _validated_source(path)
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "posthoc_evaluation"
    }


def runtime_case(manifest: dict[str, Any], case_id: str) -> EgoSchemaRuntimeCase:
    """Return one runtime-safe case; post-hoc labels are never copied."""
    source = next((row for row in manifest["cases"] if row["case_id"] == case_id), None)
    if source is None:
        raise KeyError(f"Unknown EgoSchema comparison case: {case_id}")
    options = tuple(source["options"])
    return EgoSchemaRuntimeCase(
        case_id=str(source["case_id"]),
        video_id=str(source["video_id"]),
        question=str(source["question"]),
        options=(options[0], options[1], options[2], options[3], options[4]),
        video_path=str(source["video_path"]),
        duration_sec=float(source["duration_sec"]),
        audio_available=bool(source["audio_available"]),
    )


def retrieval_question(case: EgoSchemaRuntimeCase, protocol: RetrievalProtocol) -> str:
    """Construct only the protocol-defined retrieval-visible text."""
    if protocol is RetrievalProtocol.QUESTION_ONLY:
        return case.question
    if protocol is RetrievalProtocol.STANDARD_MULTIPLE_CHOICE:
        options = "\n".join(f"Option {index}: {text}" for index, text in enumerate(case.options))
        return f"{case.question}\n\nAnswer options available during retrieval:\n{options}"
    raise EgoSchemaAdapterError(f"Unknown retrieval protocol: {protocol!r}")


def canonical_manifest_row(
    case: EgoSchemaRuntimeCase,
    protocol: RetrievalProtocol,
) -> dict[str, Any]:
    """Create the gold-free row consumed by the unchanged canonical runner."""
    return {
        "case_id": case.case_id,
        "video_id": case.video_id,
        "question": retrieval_question(case, protocol),
        "video_duration": case.duration_sec,
        "video_path": case.video_path,
        "available_modalities": ["visual"] if not case.audio_available else ["visual", "audio"],
        "dataset_adapter": MANIFEST_VERSION,
        "retrieval_protocol": protocol.value,
    }


def final_selection_payload(
    case: EgoSchemaRuntimeCase,
    protocol: RetrievalProtocol,
    final_evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    """Reveal options at final selection while preserving evidence unchanged."""
    return {
        "case_id": case.case_id,
        "question": case.question,
        "options": list(case.options),
        "retrieval_protocol": protocol.value,
        "final_evidence_payload": copy.deepcopy(final_evidence_payload),
        "required_output": {
            "selected_option_index": "integer 0–4 or null",
            "answer_status": "answered | answered_with_uncertainty | insufficient_evidence | query_or_premise_inconsistent",
        },
    }


def load_posthoc_label(
    manifest_path: Path,
    case_id: str,
    *,
    raw_prediction_saved: bool,
    validated_prediction_saved: bool,
) -> dict[str, Any]:
    """Expose gold only after raw and validated predictions are durable."""
    if not raw_prediction_saved or not validated_prediction_saved:
        raise EgoSchemaAdapterError("Gold access is forbidden before prediction and validation are saved")
    source = _validated_source(manifest_path)
    label = source["posthoc_evaluation"]["labels_by_case_id"].get(case_id)
    if label is None:
        raise KeyError(f"No post-hoc label for EgoSchema case: {case_id}")
    return copy.deepcopy(label)
