"""Gemini Interactions request construction for the Task 7B pilot."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from src.final_qa.task7b_validation import FinalQAResult, with_uncertainty_ids


SYSTEM_INSTRUCTION = """You are the final grounded multimodal QA component.
Answer only from the supplied evidence. Never force an answer and never use unseen evidence.
Cite only supplied evidence IDs. Distinguish direct observation from inference.
Preserve every inherited pipeline uncertainty and assess each one exactly once.
Do not infer speaker identity without verified speaker evidence.
CLAP scores are retrieval provenance, never semantic sound verification.
Do not invent objects, sounds, people, or events.
Use query_or_premise_inconsistent if supplied evidence contradicts a required premise.
Use insufficient_evidence if answer-required evidence is absent or unusable.
Do not provide hidden chain-of-thought. reasoning_summary must be brief and evidence-grounded.
"""


def gemini_key_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _safe_media_bytes(path: Path, loader: Callable[[Path], bytes] | None) -> bytes:
    return loader(path) if loader else path.read_bytes()


def _text_metadata(payload: dict[str, Any], inherited: list[dict[str, Any]]) -> str:
    groups = []
    for group in payload["evidence_groups"]:
        groups.append({
            "group_id": group["group_id"],
            "group_type": group["group_type"],
            "visual_evidence": [{"evidence_id": item["evidence_id"], "start_sec": item["start_sec"], "end_sec": item["end_sec"], "roles": item["roles"], "frames": [{"timestamp_sec": frame["timestamp_sec"], "presentation_order": frame["presentation_order"], "selection_rank": frame["selection_rank"]} for frame in item["frames"]]} for item in group["visual_evidence"]],
            "speech_evidence": group["speech_evidence"],
            "acoustic_evidence": [{"evidence_id": item["evidence_id"], "start_sec": item["start_sec"], "end_sec": item["end_sec"], "roles": item["roles"], "acoustic_evidence_role": item["acoustic_evidence_role"], "semantic_interpretation_pending": item["semantic_interpretation_pending"]} for item in group["acoustic_evidence"]],
            "relations": group["relations"],
            "unresolved_ambiguities": group["unresolved_ambiguities"],
            "missing_information": group["missing_information"],
        })
    metadata = {
        "case_id": payload["case_id"],
        "question": payload["question"],
        "operation": payload["operation"],
        "answer_required_modalities": payload["answer_required_modalities"],
        "supporting_modalities": payload["supporting_modalities"],
        "evidence_groups": groups,
        "inherited_pipeline_uncertainties": inherited,
        "final_answer_policy": payload["final_answer_policy"],
    }
    return "Original question and evidence metadata:\n" + json.dumps(metadata, ensure_ascii=False, indent=2)


def build_request(
    payload: dict[str, Any],
    root: Path,
    model: str,
    thinking_level: str,
    include_media_data: bool,
    media_loader: Callable[[Path], bytes] | None = None,
    response_model: type[Any] = FinalQAResult,
    uncertainties_override: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    inherited = (
        [dict(item) for item in uncertainties_override]
        if uncertainties_override is not None
        else with_uncertainty_ids(payload["pipeline_uncertainties"])
    )
    inputs: list[dict[str, Any]] = [{"type": "text", "text": _text_metadata(payload, inherited)}]
    media: list[dict[str, Any]] = []
    delivered: set[str] = set()
    speech_count = sum(len(group["speech_evidence"]) for group in payload["evidence_groups"])
    if speech_count:
        delivered.add("speech")

    for group in payload["evidence_groups"]:
        for visual in group["visual_evidence"]:
            for frame in visual["frames"]:
                path = _resolve(root, frame["frame_path"])
                if not path.is_file():
                    raise FileNotFoundError(f"Required retained image is missing: {frame['frame_path']}")
                media.append({"evidence_id": visual["evidence_id"], "path": frame["frame_path"], "timestamp_sec": frame["timestamp_sec"], "mime_type": "image/jpeg", "byte_count": path.stat().st_size, "kind": "image"})
                inputs.append({"type": "text", "text": f"Visual evidence {visual['evidence_id']} frame at {frame['timestamp_sec']:.3f}s (presentation_order={frame['presentation_order']}, selection_rank={frame['selection_rank']})."})
                part: dict[str, Any] = {"type": "image", "mime_type": "image/jpeg"}
                if include_media_data:
                    part["data"] = base64.b64encode(_safe_media_bytes(path, media_loader)).decode("ascii")
                inputs.append(part)
                delivered.add("visual")
        for acoustic in group["acoustic_evidence"]:
            value = acoustic.get("audio_clip_path")
            path = _resolve(root, value) if value else None
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Required retained audio is missing: {value}")
            media.append({"evidence_id": acoustic["evidence_id"], "path": value, "start_sec": acoustic["start_sec"], "end_sec": acoustic["end_sec"], "mime_type": "audio/wav", "byte_count": path.stat().st_size, "kind": "audio"})
            inputs.append({"type": "text", "text": f"Local raw audio evidence {acoustic['evidence_id']} covers {acoustic['start_sec']:.3f}-{acoustic['end_sec']:.3f}s. Interpret the supplied audio itself; do not use CLAP as semantic evidence."})
            part = {"type": "audio", "mime_type": "audio/wav"}
            if include_media_data:
                part["data"] = base64.b64encode(_safe_media_bytes(path, media_loader)).decode("ascii")
            inputs.append(part)
            delivered.add("acoustic")
    if any(item["kind"] == "audio" for item in media):
        inputs.append({"type": "text", "text": "Assess the supplied raw audio for overlapping speech, background noise, reverberation, low target-sound volume, simultaneous sound sources, clipping, or poor signal quality. Add signal_interference_uncertainty only when observed interference materially affects interpretation; do not assume it exists."})
    inputs.append({"type": "text", "text": "Return only the structured JSON required by the attached response_format schema. Include a brief reasoning_summary, not hidden reasoning."})

    request = {
        "model": model,
        "system_instruction": SYSTEM_INSTRUCTION,
        "input": inputs,
        "response_format": {"type": "text", "mime_type": "application/json", "schema": response_model.model_json_schema()},
        "generation_config": {"thinking_level": thinking_level},
        "store": False,
    }
    manifest = {
        "case_id": payload["case_id"], "model": model, "media": media,
        "evidence_ids": sorted({item["evidence_id"] for item in media} | {item["evidence_id"] for group in payload["evidence_groups"] for item in group["speech_evidence"]}),
        "uncertainty_ids": [item["uncertainty_id"] for item in inherited],
        "store": False, "thinking_level": thinking_level, "previous_interaction_id_used": False,
        "response_schema": response_model.__name__, "inline_media_data_saved": False,
    }
    return request, manifest, delivered


def retryable_exception(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    name = type(exc).__name__.casefold()
    return isinstance(exc, (ConnectionError, TimeoutError)) or any(token in name for token in ("servererror", "serviceunavailable", "timeout", "connectionerror"))


def safe_error(exc: Exception) -> dict[str, Any]:
    return {"error_class": type(exc).__name__, "status_code": getattr(exc, "status_code", None), "message": "Provider or parsing error; secret-bearing exception text was not persisted."}


def usage_metadata(interaction: Any) -> tuple[dict[str, Any], list[str]]:
    usage = getattr(interaction, "usage", None)
    warnings: list[str] = []
    data = usage.model_dump(mode="json") if usage is not None and hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else {})
    fields = {
        "provider_input_token_count": data.get("total_input_tokens"),
        "provider_output_token_count": data.get("total_output_tokens"),
        "provider_total_token_count": data.get("total_tokens"),
        "input_tokens_by_modality": data.get("input_tokens_by_modality"),
        "output_tokens_by_modality": data.get("output_tokens_by_modality"),
    }
    for key, value in fields.items():
        if value is None:
            warnings.append(f"usage_field_unavailable:{key}")
    return fields, warnings


@dataclass
class CallOutcome:
    parsed: Any | None
    raw_text: str | None
    interaction: Any | None
    attempts: list[dict[str, Any]]
    raw_attempt_outputs: list[dict[str, Any]]
    infrastructure_succeeded: bool
    error: dict[str, Any] | None


def call_with_one_technical_retry(client: Any, request: dict[str, Any], response_model: type[Any] = FinalQAResult) -> CallOutcome:
    attempts: list[dict[str, Any]] = []
    raw_attempt_outputs: list[dict[str, Any]] = []
    for attempt_number in (1, 2):
        started = time.perf_counter()
        try:
            interaction = client.interactions.create(**request)
            raw_text = getattr(interaction, "output_text", None)
            raw_attempt_outputs.append({"attempt_number": attempt_number, "raw_model_text": None if raw_text is None else str(raw_text)})
            if not raw_text or not str(raw_text).strip():
                raise ValueError("empty_structured_output")
            parsed = response_model.model_validate_json(raw_text)
            attempts.append({"attempt_number": attempt_number, "latency_sec": time.perf_counter() - started, "success": True, "retry_reason": None, "first_call_billable": None})
            return CallOutcome(parsed, str(raw_text), interaction, attempts, raw_attempt_outputs, True, None)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            attempts.append({"attempt_number": attempt_number, "latency_sec": time.perf_counter() - started, "success": False, "retry_reason": "invalid_or_empty_structured_output", "error": safe_error(exc), "first_call_billable": None})
            if attempt_number == 2:
                return CallOutcome(None, None, None, attempts, raw_attempt_outputs, False, safe_error(exc))
        except Exception as exc:
            attempts.append({"attempt_number": attempt_number, "latency_sec": time.perf_counter() - started, "success": False, "retry_reason": "technical_provider_error", "error": safe_error(exc), "first_call_billable": None})
            if attempt_number == 2 or not retryable_exception(exc):
                return CallOutcome(None, None, None, attempts, raw_attempt_outputs, False, safe_error(exc))
    raise AssertionError("unreachable")
