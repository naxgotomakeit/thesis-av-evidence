"""Shared offline-only runner, prompt builder, adapter, and parser for A/B/D.

All variants use one approved draft prompt and one payload builder.  This module
has no provider transport: rendering requests offline is possible, calling an
API is not.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gens_haiku_eval300.structured_v3 import (
    StructuredV3Config,
    parse_structured_response,
    tool_schema,
)


VARIANTS = frozenset({"A", "B", "D"})


@dataclass(frozen=True)
class AbdDraftConfig:
    """ABD-only wrapper around the frozen GenS-v3 model/output contract."""

    schema_version: str
    experiment_id: str
    base_gens_config_path: str
    system_prompt_path: str
    map_prefix: str
    frame_timestamp_template: str
    message_layout: str
    cache_policy: str
    single_turn: bool
    structural_correction_retries: int
    api_enabled_by_default: bool
    provider_execution_available: bool
    question_text_source: str
    source_record: str
    base_config: StructuredV3Config

    @classmethod
    def load(cls, path: Path) -> "AbdDraftConfig":
        path = path.resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent.parent
        base_path = Path(raw["base_gens_config_path"])
        prompt_path = Path(raw["system_prompt_path"])
        if not base_path.is_absolute():
            base_path = root / base_path
        if not prompt_path.is_absolute():
            prompt_path = root / prompt_path
        raw["base_gens_config_path"] = str(base_path)
        raw["system_prompt_path"] = str(prompt_path)
        config = cls(base_config=StructuredV3Config.load(base_path), **raw)
        if config.api_enabled_by_default or config.provider_execution_available:
            raise ValueError("ABD draft must remain offline-only")
        if not config.single_turn or config.structural_correction_retries != 0:
            raise ValueError("ABD must remain single-turn with no correction")
        if config.cache_policy != "disabled_no_cache_control":
            raise ValueError("ABD cache controls must remain disabled")
        if config.map_prefix != "VIDEO MAP (frozen native representation):\n":
            raise ValueError("unexpected map prefix")
        if config.frame_timestamp_template != (
            "Frame timestamp={resolved_timestamp_sec:.3f}s from video start."
        ):
            raise ValueError("unexpected frame timestamp template")
        if not Path(config.system_prompt_path).is_file():
            raise ValueError("missing ABD common system prompt")
        return config

    @property
    def system_prompt(self) -> str:
        return Path(self.system_prompt_path).read_text(encoding="utf-8").rstrip("\n")

    @property
    def provider(self) -> str:
        return self.base_config.provider

    @property
    def model(self) -> str:
        return self.base_config.model

    @property
    def max_output_tokens(self) -> int:
        return self.base_config.max_output_tokens

    @property
    def temperature(self) -> float:
        return self.base_config.temperature

    @property
    def timeout_sec(self) -> float:
        return self.base_config.timeout_sec

    @property
    def max_retries(self) -> int:
        return self.base_config.max_retries

    @property
    def tool_choice(self) -> dict[str, Any]:
        return self.base_config.tool_choice

    @property
    def pricing(self) -> dict[str, Any]:
        return self.base_config.pricing


def parse_response(content: Any) -> dict[str, Any]:
    """Use the exact GenS-v3 Direct-aligned parser, without extra validation."""
    return parse_structured_response(content)


def _question_text(row: dict[str, Any]) -> str:
    options = row["options"]
    return (
        f"Question: {row['question']}\n"
        "Options:\n"
        + "\n".join(f"{choice}. {options[choice]}" for choice in "ABCDE")
        + "\nReturn the final_answer action."
    )


@dataclass(frozen=True)
class GensPromptBuilder:
    """Render the shared ABD payload with variant differences limited to evidence."""

    config: AbdDraftConfig

    @property
    def system_prompt(self) -> str:
        return self.config.system_prompt

    def _content(self, row: dict[str, Any], *, encode_images: bool) -> list[dict[str, Any]]:
        variant = str(row["variant"])
        if variant not in VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        content: list[dict[str, Any]] = []
        map_item = row.get("map")
        if map_item is not None:
            path = Path(map_item["path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != map_item["sha256"]:
                raise ValueError(f"map SHA mismatch while rendering: {path}")
            content.append({"type": "text", "text": self.config.map_prefix + raw.decode("utf-8")})
        for image in row["images"]:
            path = Path(image["path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != image["sha256"]:
                raise ValueError(f"image SHA mismatch while rendering: {path}")
            content.append({
                "type": "text",
                "text": self.config.frame_timestamp_template.format(
                    resolved_timestamp_sec=float(image["resolved_timestamp_sec"])
                ),
            })
            data = base64.b64encode(raw).decode("ascii") if encode_images else "<validated-original-jpeg-bytes>"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
            })
        content.append({"type": "text", "text": _question_text(row)})
        return content

    def build_provider_payload(
        self, row: dict[str, Any], *, encode_images: bool = True
    ) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": self._content(row, encode_images=encode_images)}],
            "tools": [tool_schema()],
            "tool_choice": dict(self.config.tool_choice),
        }

    def request_preview(self, row: dict[str, Any]) -> dict[str, Any]:
        """Separate the real model-visible shape from audit-only provenance."""
        payload = self.build_provider_payload(row, encode_images=False)
        backend = {
            "question_id": row["question_id"],
            "variant": row["variant"],
            "map": row.get("map"),
            "images": [
                {
                    "path": image["path"],
                    "sha256": image["sha256"],
                    "requested_timestamp_sec": image["requested_timestamp_sec"],
                    "resolved_timestamp_sec": image["resolved_timestamp_sec"],
                    "direct_transmission_index": image["direct_transmission_index"],
                    "presentation_index": image["presentation_index"],
                }
                for image in row["images"]
            ],
            "source_provenance": row["source_provenance"],
        }
        return {
            "draft_only": True,
            "provider_payload_renderable_offline": True,
            "provider_execution_available": False,
            "prompt_status": "common_abd_prompt_applied_to_draft_only",
            "model_visible_request_preview": payload,
            "backend_metadata_not_sent_to_model": backend,
        }


class DraftOnlyProviderAdapter:
    """Non-network adapter: preview is allowed; execution is impossible."""

    def preview(self, builder: GensPromptBuilder, row: dict[str, Any]) -> dict[str, Any]:
        return builder.request_preview(row)

    def call(self, _payload: dict[str, Any]) -> None:
        raise RuntimeError(
            "real API transport is absent from abd_draft_v1; API calls are hard-disabled"
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_manifest(rows: list[dict[str, Any]], variant: str) -> None:
    if len(rows) != 300:
        raise ValueError(f"expected 300 rows, found {len(rows)}")
    qids = [str(row["question_id"]) for row in rows]
    if len(set(qids)) != 300:
        raise ValueError("duplicate question identity")
    for row in rows:
        if row.get("variant") != variant:
            raise ValueError("variant mismatch")
        if set(row["options"]) != set("ABCDE"):
            raise ValueError(f"invalid option labels for {row['question_id']}")
        if variant == "A" and (row.get("map") is None or row["images"]):
            raise ValueError("A must contain one map and zero images")
        if variant == "B" and row.get("map") is not None:
            raise ValueError("B must not contain a map")
        if variant == "D" and row.get("map") is None:
            raise ValueError("D must contain a map")
        indices = [image["presentation_index"] for image in row["images"]]
        if indices != list(range(len(indices))):
            raise ValueError(f"non-contiguous presentation order for {row['question_id']}")
        stamps = [image["resolved_timestamp_sec"] for image in row["images"]]
        if stamps != sorted(stamps):
            raise ValueError(f"non-chronological GenS presentation for {row['question_id']}")


@dataclass
class DraftRunner:
    variant: str
    config_path: Path
    manifest_path: Path

    def run(self, *, preview_ids: Iterable[str] = (), preview_count: int = 1) -> dict[str, Any]:
        config = AbdDraftConfig.load(self.config_path)
        rows = load_jsonl(self.manifest_path)
        validate_manifest(rows, self.variant)
        by_id = {row["question_id"]: row for row in rows}
        selected = list(preview_ids)
        if not selected:
            selected = [row["question_id"] for row in rows[:preview_count]]
        unknown = [qid for qid in selected if qid not in by_id]
        if unknown:
            raise ValueError(f"unknown preview question ids: {unknown}")
        builder = GensPromptBuilder(config)
        adapter = DraftOnlyProviderAdapter()
        previews = [adapter.preview(builder, by_id[qid]) for qid in selected]
        return {
            "status": "DRY_RUN_ONLY",
            "variant": self.variant,
            "questions_validated": len(rows),
            "preview_question_ids": selected,
            "request_previews": previews,
            "api_calls": 0,
            "smoke_launches": 0,
            "formal_preflight_launches": 0,
            "formal_experiment_launches": 0,
            "provider_execution_available": False,
        }


def run_cli(variant: str, root: Path) -> None:
    if variant not in VARIANTS:
        raise ValueError(variant)
    parser = argparse.ArgumentParser(description=f"A/B/D {variant} offline draft runner")
    parser.add_argument("--mode", choices=["dry-run"], default="dry-run")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config/abd_draft_v1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / f"drafts/abd_direct_eval300_v1/inputs/{variant}.jsonl",
    )
    parser.add_argument("--preview-question-id", action="append", default=[])
    parser.add_argument("--preview-count", type=int, default=1)
    args = parser.parse_args()
    if args.preview_count < 0:
        raise SystemExit("--preview-count must be non-negative")
    result = DraftRunner(variant, args.config, args.manifest).run(
        preview_ids=args.preview_question_id,
        preview_count=args.preview_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
