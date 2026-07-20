"""Existing-provider boundary and crash-safe batch journaling for the experiment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import MODALITIES, NATURES, SCOPES, sha256_text, validate_annotation


SYSTEM_RUBRIC_A = """You are annotating evidence requirements from QUESTION TEXT ONLY.
Do not answer the question. Do not assume dataset identity, answer options, gold labels, timestamps, video, audio, or retrieved evidence.

Assign three experimental axes by the minimum evidence likely needed:

TEMPORAL SCOPE
- local: one contiguous temporal neighborhood likely suffices; a short local transition is allowed.
- multi_event: two or more distinguishable events/states and their temporal relation are likely needed.
- global: broad coverage, distributed stages, overall activity, or video-level progression is likely needed.
- unclear: question text alone cannot support a reliable scope decision.

EVIDENCE NATURE
- static: primarily state/object/attribute/scene/configuration/identity evidence.
- dynamic: primarily motion/action/change/order/interaction/process evidence.
- uncertain: text alone is insufficient, or static and dynamic are both genuinely necessary with no clear primary.
If both are relevant but one is primary, set the other as nature_secondary and nature_ambiguity=true. Otherwise secondary is null. Never force a clean label when genuinely mixed.

MODALITY
- visual: wording indicates visual evidence is sufficient or primary.
- audio: wording explicitly requires speech/sound/noise/auditory evidence.
- audio_visual: answering likely requires linking auditory evidence to a visual actor/event/state/context.
- indeterminate_from_question: wording alone does not reveal a safe modality choice.

Set taxonomy_failure=true only when one or more axes cannot represent the requirement even with unclear/uncertain/indeterminate. State the missing distinction briefly.
Return one independent strict record per item. Reasons must be descriptive, not chain-of-thought, and at most 20 words per axis."""


SYSTEM_RUBRIC_B = """Classify only the evidence demand expressed by each QUESTION. Do not answer it and do not use or infer dataset identity, options, labels, timestamps, media, or retrieved evidence.

Use the least evidence sufficient in principle:
- scope local = one continuous neighborhood; multi_event = distinct events/states must be related; global = distributed/video-wide coverage; unclear = text cannot decide.
- nature static = state-like recognition; dynamic = motion/change/order/process; uncertain = neither primary is defensible. If both matter and one leads, record the other as secondary and flag ambiguity.
- modality visual = vision is indicated; audio = speech/sound is explicitly needed; audio_visual = auditory evidence must be linked to visible context; indeterminate_from_question = wording cannot safely select modality.

Use taxonomy_failure only if these axes still omit a necessary evidence distinction. Produce strict independent JSON records. Each axis reason is descriptive and no more than 20 words; provide no hidden reasoning or answer."""


def annotation_json_schema(max_items: int) -> dict[str, Any]:
    annotation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "item_id": {"type": "string"},
            "scope": {"type": "string", "enum": list(SCOPES)},
            "nature_primary": {"type": "string", "enum": list(NATURES)},
            "nature_secondary": {
                "anyOf": [
                    {"type": "string", "enum": ["static", "dynamic"]},
                    {"type": "null"},
                ]
            },
            "nature_ambiguity": {"type": "boolean"},
            "modality": {"type": "string", "enum": list(MODALITIES)},
            "confidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "scope": {"type": "number"},
                    "nature": {"type": "number"},
                    "modality": {"type": "number"},
                },
                "required": ["scope", "nature", "modality"],
            },
            "reason_short": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "scope": {"type": "string"},
                    "nature": {"type": "string"},
                    "modality": {"type": "string"},
                },
                "required": ["scope", "nature", "modality"],
            },
            "taxonomy_failure": {"type": "boolean"},
            "taxonomy_failure_reason": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            },
        },
        "required": [
            "item_id", "scope", "nature_primary", "nature_secondary", "nature_ambiguity",
            "modality", "confidence", "reason_short", "taxonomy_failure",
            "taxonomy_failure_reason",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "annotations": {
                "type": "array",
                "items": annotation,
            }
        },
        "required": ["annotations"],
    }


def question_only_payload(rows: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """Build an opaque-ID prompt containing no record metadata beyond question text."""
    mapping: dict[str, str] = {}
    items = []
    for index, row in enumerate(rows):
        item_id = f"item_{index:03d}"
        mapping[item_id] = str(row["record_id"])
        items.append({"item_id": item_id, "question": str(row["question"])})
    return json.dumps({"items": items}, ensure_ascii=False), mapping


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class BatchResult:
    annotations: list[dict[str, Any]]
    usage: dict[str, Any]
    api_call_made: bool


class AnthropicTaxonomyAnnotator:
    """A compact structured-output annotator using the repo's configured provider."""

    def __init__(self, *, model: str, api_key: str, max_tokens: int, temperature: float = 0):
        import anthropic

        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.client = anthropic.Anthropic(api_key=api_key)

    def annotate_batch(
        self,
        *,
        rows: list[dict[str, Any]],
        system_rubric: str,
        pass_name: str,
        batch_index: int,
        journal_dir: Path,
    ) -> BatchResult:
        user_payload, item_mapping = question_only_payload(rows)
        request_hash = sha256_text(
            json.dumps(
                {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system_sha256": sha256_text(system_rubric),
                    "user_payload": json.loads(user_payload),
                    "output_schema": annotation_json_schema(len(rows)),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        journal_path = journal_dir / f"{pass_name}_batch_{batch_index:03d}.json"
        if journal_path.exists():
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            if (
                journal.get("state") == "definitive_rejection"
                and journal.get("request_sha256") != request_hash
            ):
                preserved = journal_path.with_name(
                    f"{journal_path.stem}_definitive_rejection_{str(journal['request_sha256'])[:10]}.json"
                )
                if not preserved.exists():
                    journal_path.replace(preserved)
                journal = {}
            if journal and journal.get("request_sha256") != request_hash:
                raise RuntimeError(f"Existing journal fingerprint mismatch: {journal_path}")
            if journal:
                state = journal.get("state")
                if state == "validated":
                    return BatchResult(journal["annotations"], journal["usage"], False)
                if state == "response_saved":
                    return self._validate_saved(journal_path, journal, rows, item_mapping)
                if state in {"pending", "sent"}:
                    raise RuntimeError(
                        f"Ambiguous provider attempt state '{state}' at {journal_path}; refusing duplicate call"
                    )
                if state == "definitive_rejection":
                    raise RuntimeError(f"Provider definitively rejected unchanged request: {journal_path}")
                raise RuntimeError(f"Unsupported journal state '{state}' at {journal_path}")

        base = {
            "schema_version": "planner-taxonomy-provider-attempt-v1",
            "pass": pass_name,
            "batch_index": batch_index,
            "model": self.model,
            "request_sha256": request_hash,
            "record_ids": [row["record_id"] for row in rows],
            "state": "pending",
        }
        _atomic_json(journal_path, base)
        sent = {**base, "state": "sent", "sent_at_unix": time.time()}
        _atomic_json(journal_path, sent)
        started = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_rubric,
                messages=[{"role": "user", "content": user_payload}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": annotation_json_schema(len(rows)),
                    }
                },
            )
        except Exception as error:
            if getattr(error, "status_code", None) == 400:
                rejected = {
                    **sent,
                    "state": "definitive_rejection",
                    "rejection_type": type(error).__name__,
                    "rejection_note": "HTTP 400 request validation rejection; no model response was produced.",
                    "rejected_at_unix": time.time(),
                }
                _atomic_json(journal_path, rejected)
            raise
        latency = time.perf_counter() - started
        response_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = {
            "provider": "anthropic",
            "model": self.model,
            "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens),
            "latency_sec": latency,
            "estimated_cost_usd": None,
            "cost_note": "No reliable provider pricing table is configured in this repository.",
        }
        saved = {
            **sent,
            "state": "response_saved",
            "response_text": response_text,
            "provider_response_id": str(getattr(response, "id", "")),
            "usage": usage,
            "response_saved_at_unix": time.time(),
        }
        _atomic_json(journal_path, saved)
        return self._validate_saved(journal_path, saved, rows, item_mapping, api_call_made=True)

    @staticmethod
    def _validate_saved(
        journal_path: Path,
        journal: dict[str, Any],
        rows: list[dict[str, Any]],
        item_mapping: dict[str, str],
        api_call_made: bool = False,
    ) -> BatchResult:
        parsed = json.loads(journal["response_text"])
        records = parsed.get("annotations")
        if not isinstance(records, list) or len(records) != len(rows):
            raise ValueError("Provider annotation count differs from requested batch")
        by_item: dict[str, dict[str, Any]] = {}
        for record in records:
            item_id = str(record.get("item_id"))
            if item_id in by_item:
                raise ValueError(f"Duplicate annotation item_id: {item_id}")
            validate_annotation(record, item_id)
            by_item[item_id] = record
        if set(by_item) != set(item_mapping):
            raise ValueError("Provider item IDs differ from requested opaque IDs")
        source_by_id = {row["record_id"]: row for row in rows}
        annotations = []
        for item_id, record_id in item_mapping.items():
            source = source_by_id[record_id]
            annotation = {key: value for key, value in by_item[item_id].items() if key != "item_id"}
            annotations.append({**source, **annotation})
        validated = {
            **journal,
            "state": "validated",
            "annotations": annotations,
            "validated_at_unix": time.time(),
        }
        _atomic_json(journal_path, validated)
        return BatchResult(annotations, journal["usage"], api_call_made)


def configured_anthropic() -> tuple[str, str]:
    model = os.environ.get("ANTHROPIC_MODEL")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not model or not key:
        raise RuntimeError("ANTHROPIC_MODEL and ANTHROPIC_API_KEY must be configured")
    return model, key
