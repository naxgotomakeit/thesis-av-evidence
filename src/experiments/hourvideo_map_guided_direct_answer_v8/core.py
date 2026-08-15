from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from experiments.reviewed_visual_evidence_cache_v1_gemini_canary.core import (
    response_text,
    usage as gemini_usage,
)


MAP_SYSTEM = """You answer one multiple-choice video question using the supplied
semantic navigation map as a guide. The map is the only video-derived evidence
available in this stage. Select an answer directly when the map clearly
distinguishes one option.

Request visual review only when an answer-critical detail is unclear, evidence is
insufficient, or map statements are semantically conflicting. General caution or
the mere availability of images is not a reason to request review. If review is
needed, autonomously choose map-existing Coarse or Medium IDs and exact map image
IDs, explain the exact detail to inspect, and request the smallest useful image
set. Never request more than 25 images; fewer is preferred whenever sufficient.
Do not request a whole-video scan. Return strict JSON."""


REVIEW_SYSTEM = """You answer one multiple-choice video question using the
supplied semantic navigation map plus a map-routed set of chronologically labeled
images. Inspect the images only to resolve the explicit uncertainty or semantic
conflict reported by the map-only stage. Do not treat sparse images as proof of
unseen continuity, duration, causality, or identity. Select exactly one option and
return strict JSON."""


CACHED_COMMON_SYSTEM = """You answer multiple-choice video questions using a
supplied semantic navigation map as the only video-derived evidence. Follow the
STAGE instruction provided after the cached map. The first stage may answer or
request a small targeted visual review; the review stage must select exactly one
option. Never request a whole-video scan or more than 25 images. Return strict
JSON matching the supplied response schema."""


QUESTION_FIRST_ADDENDUM = """\n\nQUESTION-FIRST DECISION POLICY:
Before selecting an answer or deciding whether to inspect images, first
understand the question as an information task. Identify the question type,
target entities, exact requested relation, all scope constraints, literal
differences between the options, and the information needed to distinguish
them. Do not choose an answer during this question-understanding step.

Then inspect the navigation map for the required information. Do not replace
the requested relation with a broad whole-video theme, and do not treat keyword
overlap as sufficient when the question asks about a more specific relation.
After understanding the question, either answer directly when the map
distinguishes an option or request the smallest useful map-routed image set to
resolve the missing distinction. During visual review, preserve and apply the
same question understanding rather than reinterpreting the task."""


STRICT_EVIDENCE_ONLY_ADDENDUM = """\n\nSTRICT EVIDENCE-ONLY CONTRACT:
- Never use commonsense, real-world plausibility, stereotypes, or prior
  expectations to select or eliminate an option. An option that seems unlikely
  remains possible unless supplied evidence contradicts it.
- Every answer-critical predicate and relation must be explicitly supported by
  the supplied map or by an actually reviewed image. This includes actor,
  action, object, location, time, ordering, shared participation, identity, and
  causality.
- A word or action appearing somewhere in the map does not prove that every
  named actor performed it, that it occurred at the questioned location/time,
  or that the complete relation in an option is true.
- Separate direct evidence from inference. Do not fill a missing relation with
  assumptions. Sparse captions or images cannot establish non-occurrence.
- If the map supports only parts of the required relation, mark the case
  unclear/insufficient/semantic_conflict and request the smallest targeted set
  of map images capable of checking the missing predicates.
- Cite the exact Coarse/Medium/image IDs supporting the decision. After visual
  review, select the best-supported required option, but explicitly preserve
  any unresolved predicates instead of resolving them with commonsense."""


NO_COMMONSENSE_ONLY_ADDENDUM = """\n\nNO-COMMONSENSE CONTRACT:
- Do not use commonsense, real-world plausibility, stereotypes, or prior
  expectations to select or eliminate an option.
- In particular, never reject an option merely because the described event
  seems unusual, unlikely, or inappropriate for the apparent setting.
- Base the answer on the supplied navigation map and, when requested under the
  normal review policy above, the actually reviewed images."""


def stage_system(cfg: dict[str, Any], base: str) -> str:
    additions = ""
    if cfg.get("question_first_decision"):
        additions += QUESTION_FIRST_ADDENDUM
    if cfg.get("strict_evidence_only"):
        additions += STRICT_EVIDENCE_ONLY_ADDENDUM
    elif cfg.get("no_commonsense_only"):
        additions += NO_COMMONSENSE_ONLY_ADDENDUM
    return base + additions


QUESTION_UNDERSTANDING_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": [
                "temporal", "comparison", "shared_action", "location", "identity",
                "count", "causal", "object_action", "summary", "other",
            ],
        },
        "target_entities": {"type": "array", "items": {"type": "string"}},
        "requested_relation": {"type": "string"},
        "temporal_constraints": {"type": "array", "items": {"type": "string"}},
        "spatial_constraints": {"type": "array", "items": {"type": "string"}},
        "participant_constraints": {"type": "array", "items": {"type": "string"}},
        "other_constraints": {"type": "array", "items": {"type": "string"}},
        "critical_option_differences": {"type": "array", "items": {"type": "string"}},
        "information_needed": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "task_type", "target_entities", "requested_relation", "temporal_constraints",
        "spatial_constraints", "participant_constraints", "other_constraints",
        "critical_option_differences", "information_needed",
    ],
    "additionalProperties": False,
}


MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "question_id": {"type": "string"},
        "map_status": {
            "type": "string",
            "enum": ["answerable", "unclear", "insufficient", "semantic_conflict"],
        },
        "provisional_option_id": {"type": "string"},
        "reasoning": {"type": "string"},
        "supporting_coarse_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_medium_ids": {"type": "array", "items": {"type": "string"}},
        "requested_coarse_ids": {"type": "array", "items": {"type": "string"}},
        "requested_medium_ids": {"type": "array", "items": {"type": "string"}},
        "requested_map_image_ids": {"type": "array", "items": {"type": "string"}},
        "review_goal": {"type": "string"},
    },
    "required": [
        "question_id", "map_status", "provisional_option_id", "reasoning",
        "supporting_coarse_ids", "supporting_medium_ids", "requested_coarse_ids",
        "requested_medium_ids", "requested_map_image_ids", "review_goal",
    ],
    "additionalProperties": False,
}


MAP_SCHEMA_QUESTION_FIRST = {
    **MAP_SCHEMA,
    "properties": {
        "question_understanding": QUESTION_UNDERSTANDING_SCHEMA,
        **MAP_SCHEMA["properties"],
    },
    "required": ["question_understanding", *MAP_SCHEMA["required"]],
}


CACHED_MAP_STAGE_INSTRUCTION = MAP_SYSTEM + """
Set response_stage to map_only. Complete the map-decision fields normally.
For the visual-review-only fields, set selected_option_id equal to the
provisional option (or NONE), answer_text to an empty string,
visual_review_changed_answer to false, supporting_image_ids to an empty list,
and remaining_uncertainty to a concise list of unresolved details or an empty
list when answerable."""


CACHED_REVIEW_STAGE_INSTRUCTION = REVIEW_SYSTEM + """
Set response_stage to visual_review. Preserve question_understanding from the
map-only decision. Complete the final-answer fields from the supplied images.
Copy map_status, provisional_option_id, requested IDs, and review_goal from the
map-only decision so the unified response contract remains stable."""


LOCAL_REVIEW_IMAGE_ID_ADDENDUM = """
Image evidence uses two explicitly paired identifiers. IMGxx is the only identifier
namespace allowed in supporting_image_ids. MAPIMGxxxx is shown only
as a stable map-catalog cross-reference; never copy a MAPIMGxxxx value into
supporting_image_ids. Cite only IMGxx labels from allowed_supporting_image_ids.
Do not infer, interpolate, or invent an image identifier that was not sent."""


def review_stage_instruction(cfg: dict[str, Any], base: str) -> str:
    if cfg.get("local_review_image_ids_only", False):
        return base + "\n" + LOCAL_REVIEW_IMAGE_ID_ADDENDUM
    return base


CACHED_UNIFIED_SCHEMA = {
    "type": "object",
    "properties": {
        "response_stage": {"type": "string", "enum": ["map_only", "visual_review"]},
        "question_understanding": QUESTION_UNDERSTANDING_SCHEMA,
        **MAP_SCHEMA["properties"],
        "selected_option_id": {"type": "string"},
        "answer_text": {"type": "string"},
        "visual_review_changed_answer": {"type": "boolean"},
        "supporting_image_ids": {"type": "array", "items": {"type": "string"}},
        "remaining_uncertainty": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "response_stage", "question_understanding", *MAP_SCHEMA["required"],
        "selected_option_id", "answer_text", "visual_review_changed_answer",
        "supporting_image_ids", "remaining_uncertainty",
    ],
    "additionalProperties": False,
}


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "question_id": {"type": "string"},
        "selected_option_id": {"type": "string"},
        "answer_text": {"type": "string"},
        "reasoning": {"type": "string"},
        "visual_review_changed_answer": {"type": "boolean"},
        "supporting_coarse_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_medium_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_image_ids": {"type": "array", "items": {"type": "string"}},
        "remaining_uncertainty": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "question_id", "selected_option_id", "answer_text", "reasoning",
        "visual_review_changed_answer", "supporting_coarse_ids",
        "supporting_medium_ids", "supporting_image_ids", "remaining_uncertainty",
    ],
    "additionalProperties": False,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_view(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "answer_options": [
            {"option_id": row["option_id"], "text": row["text"]}
            for row in question["answer_options"]
        ],
    }


def map_ids(navigation_map: dict[str, Any]) -> tuple[set[str], set[str]]:
    coarse_ids: set[str] = set()
    medium_ids: set[str] = set()
    for region in navigation_map.get("coarse_regions", []):
        coarse_ids.add(str(region["coarse_id"]))
        medium_ids.update(str(value) for value in region.get("source_medium_ids", []))
    return coarse_ids, medium_ids


def map_image_catalog(navigation_map: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose stable IDs for only the frame references already contained in the map."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for region in navigation_map.get("coarse_regions", []):
        coarse_id = str(region["coarse_id"])
        for caption in region.get("exact_source_captions", []):
            medium_id = str(caption["medium_id"])
            paths = [Path(value) for value in caption.get("source_frame_paths", [])]
            start = float(caption.get("start_sec", region["start_sec"]))
            end = float(caption.get("end_sec", region["end_sec"]))
            for slot, path in enumerate(paths):
                key = (medium_id, str(path))
                if key in seen:
                    continue
                seen.add(key)
                number = _frame_number(path)
                approximate = float(number) if number is not None else start + (slot + 1) * (end - start) / (len(paths) + 1)
                position = "middle" if len(paths) % 2 and slot == len(paths) // 2 else ("early" if slot < len(paths) / 2 else "late")
                rows.append({
                    "map_image_id": f"MAPIMG{len(rows) + 1:04d}",
                    "coarse_id": coarse_id,
                    "medium_id": medium_id,
                    "approx_timestamp_sec": approximate,
                    "within_medium_position": position,
                    "path": str(path),
                })
    return rows


def validate_map_decision(
    decision: dict[str, Any], question: dict[str, Any], navigation_map: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    option_ids = {row["option_id"] for row in question["answer_options"]}
    coarse_ids, medium_ids = map_ids(navigation_map)
    if decision.get("question_id") != question["question_id"]:
        errors.append("question_id mismatch")
    if decision.get("map_status") not in {"answerable", "unclear", "insufficient", "semantic_conflict"}:
        errors.append("invalid map_status")
    if not set(decision.get("supporting_coarse_ids", [])) <= coarse_ids:
        errors.append("unknown supporting coarse ID")
    if not set(decision.get("supporting_medium_ids", [])) <= medium_ids:
        errors.append("unknown supporting medium ID")
    requested_coarse = set(decision.get("requested_coarse_ids", []))
    requested_medium = set(decision.get("requested_medium_ids", []))
    catalog = {row["map_image_id"]: row for row in map_image_catalog(navigation_map)}
    requested_images = list(decision.get("requested_map_image_ids", []))
    if not requested_coarse <= coarse_ids:
        errors.append("unknown requested coarse ID")
    if not requested_medium <= medium_ids:
        errors.append("unknown requested medium ID")
    if decision.get("map_status") == "answerable":
        if decision.get("provisional_option_id") not in option_ids:
            errors.append("answerable decision has invalid option")
        # The schema requires this string, so an answerable response may say
        # "No review needed". Only concrete requested IDs trigger a review.
        if requested_coarse or requested_medium or requested_images:
            errors.append("answerable decision improperly requests review")
    else:
        if not requested_coarse and not requested_medium:
            errors.append("review decision has no localized map IDs")
        if not decision.get("review_goal", "").strip():
            errors.append("review decision has no review goal")
        if not 1 <= len(requested_images) <= 25:
            errors.append("review must request between 1 and 25 exact map image IDs")
        if len(requested_images) != len(set(requested_images)):
            errors.append("duplicate requested map image ID")
        if not set(requested_images) <= set(catalog):
            errors.append("unknown requested map image ID")
        for image_id in requested_images:
            row = catalog.get(image_id)
            if row and row["coarse_id"] not in requested_coarse and row["medium_id"] not in requested_medium:
                errors.append(f"requested image outside requested regions: {image_id}")
        provisional = decision.get("provisional_option_id")
        if provisional not in option_ids | {"NONE"}:
            errors.append("invalid provisional option")
    return errors


def validate_question_understanding(decision: dict[str, Any]) -> list[str]:
    value = decision.get("question_understanding")
    if not isinstance(value, dict):
        return ["missing question understanding"]
    errors: list[str] = []
    if not str(value.get("requested_relation", "")).strip():
        errors.append("question understanding has no requested relation")
    if not value.get("target_entities"):
        errors.append("question understanding has no target entities")
    if not value.get("critical_option_differences"):
        errors.append("question understanding has no option differences")
    if not value.get("information_needed"):
        errors.append("question understanding has no information needs")
    return errors


def _frame_number(path: Path) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    return int(match.group(1)) if match else None


def select_review_images(
    navigation_map: dict[str, Any], decision: dict[str, Any], maximum: int
) -> list[dict[str, Any]]:
    catalog = {row["map_image_id"]: row for row in map_image_catalog(navigation_map)}
    chosen = [
        catalog[image_id]
        for image_id in decision.get("requested_map_image_ids", [])
        if image_id in catalog
    ][:maximum]
    chosen.sort(key=lambda row: (row["approx_timestamp_sec"], row["medium_id"], row["path"]))
    output = []
    for index, row in enumerate(chosen, 1):
        path = Path(row["path"])
        output.append({
            **row,
            "image_id": f"IMG{index:02d}",
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        })
    return output


def validate_review(
    result: dict[str, Any], question: dict[str, Any], navigation_map: dict[str, Any],
    images: list[dict[str, Any]], local_image_ids_only: bool = False,
) -> list[str]:
    errors: list[str] = []
    option_ids = {row["option_id"] for row in question["answer_options"]}
    coarse_ids, medium_ids = map_ids(navigation_map)
    # The review prompt exposes both the transmission label (IMGxx) and the
    # stable map-catalog label (MAPIMGxxxx); either names the same sent image.
    if local_image_ids_only:
        image_ids = {row["image_id"] for row in images if row.get("image_id")}
    else:
        image_ids = {
            value
            for row in images
            for value in (row.get("image_id"), row.get("map_image_id"))
            if value
        }
    if result.get("question_id") != question["question_id"]:
        errors.append("question_id mismatch")
    if result.get("selected_option_id") not in option_ids:
        errors.append("invalid final option")
    if not set(result.get("supporting_coarse_ids", [])) <= coarse_ids:
        errors.append("unknown final coarse ID")
    if not set(result.get("supporting_medium_ids", [])) <= medium_ids:
        errors.append("unknown final medium ID")
    if not set(result.get("supporting_image_ids", [])) <= image_ids:
        errors.append("unknown final image ID")
    return errors


def _gemini_call(
    cfg: dict[str, Any], system: str, inputs: list[dict[str, Any]], schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    body = {
        "model": cfg["gemini"]["model"],
        "system_instruction": system,
        "input": inputs,
        "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
        "generation_config": {
            "temperature": 0.0,
            "thinking_level": cfg["gemini"].get("thinking_level", "low"),
            "max_output_tokens": int(cfg["gemini"]["max_output_tokens"]),
        },
        "store": False,
    }
    request = urllib.request.Request(
        cfg["gemini"]["endpoint"],
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=int(cfg["gemini"]["timeout_sec"])) as response:
        raw = json.loads(response.read().decode("utf-8"))
    parsed = json.loads(response_text(raw))
    return parsed, {
        "provider": "google",
        "model": cfg["gemini"]["model"],
        **gemini_usage(raw),
        "latency_sec": time.perf_counter() - started,
    }, raw


def _api_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _api_schema(item)
            for key, item in value.items()
            if key not in {"$schema", "uniqueItems", "minItems", "maxItems", "minLength", "minimum", "maximum"}
        }
    if isinstance(value, list):
        return [_api_schema(item) for item in value]
    return value


def _anthropic_call(
    cfg: dict[str, Any], system: str, inputs: list[dict[str, Any]], schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import anthropic

    content: list[dict[str, Any]] = []
    for item in inputs:
        if item["type"] == "text":
            block: dict[str, Any] = {"type": "text", "text": item["text"]}
            if item.get("cache_control"):
                block["cache_control"] = item["cache_control"]
            content.append(block)
        elif item["type"] == "image":
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item["mime_type"],
                    "data": item["data"],
                },
            })
        else:
            raise ValueError(f"unsupported Anthropic input type: {item['type']}")
    started = time.perf_counter()
    request_kwargs: dict[str, Any] = {
        "model": cfg["anthropic"]["model"],
        "max_tokens": int(cfg["anthropic"]["max_output_tokens"]),
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": _api_schema(schema)}},
    }
    thinking_budget = int(cfg["anthropic"].get("thinking_budget_tokens", 0))
    if thinking_budget:
        if thinking_budget < 1024 or thinking_budget >= request_kwargs["max_tokens"]:
            raise ValueError("Anthropic thinking budget must be >=1024 and less than max_output_tokens")
        request_kwargs["thinking"] = {
            "type": "enabled", "budget_tokens": thinking_budget, "display": "summarized",
        }
        # Anthropic requires the default temperature whenever thinking is enabled.
    else:
        request_kwargs["temperature"] = 0.0
    response = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=float(cfg["anthropic"]["timeout_sec"]),
    ).messages.create(**request_kwargs)
    raw_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Anthropic structured response reached max_tokens")
    parsed = json.loads(raw_text)
    response_blocks = [
        block.model_dump(mode="json") if hasattr(block, "model_dump") else {"type": getattr(block, "type", "unknown")}
        for block in response.content
    ]
    thinking_blocks = [block for block in response_blocks if block.get("type") in {"thinking", "redacted_thinking"}]
    raw = {
        "id": str(response.id),
        "model": response.model,
        "stop_reason": response.stop_reason,
        "content": response_blocks,
        "thinking": {
            "enabled": bool(thinking_budget),
            "budget_tokens": thinking_budget,
            "block_count": len(thinking_blocks),
            "summaries": [block.get("thinking", "") for block in thinking_blocks],
        },
        "usage": {
            "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens),
            "cache_creation_input_tokens": int(getattr(response.usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(response.usage, "cache_read_input_tokens", 0) or 0),
        },
        "output_text": raw_text,
    }
    used = raw["usage"]
    return parsed, {
        "provider": "anthropic",
        "model": cfg["anthropic"]["model"],
        "input_tokens": used["input_tokens"],
        "output_tokens": used["output_tokens"],
        "thought_tokens": 0,
        "tool_use_tokens": 0,
        "total_tokens": used["input_tokens"] + used["output_tokens"],
        "cache_creation_input_tokens": used["cache_creation_input_tokens"],
        "cache_read_input_tokens": used["cache_read_input_tokens"],
        "thinking_enabled": bool(thinking_budget),
        "thinking_budget_tokens": thinking_budget,
        "thinking_block_count": len(thinking_blocks),
        "thinking_summary_characters": sum(len(str(block.get("thinking", ""))) for block in thinking_blocks),
        "thinking_token_accounting": "included in Anthropic output_tokens; provider usage does not split thinking from final text",
        "latency_sec": time.perf_counter() - started,
    }, raw


def _provider_call(
    cfg: dict[str, Any], system: str, inputs: list[dict[str, Any]], schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provider = cfg.get("provider", "gemini")
    if provider == "gemini":
        return _gemini_call(cfg, system, inputs, schema)
    if provider == "anthropic":
        return _anthropic_call(cfg, system, inputs, schema)
    raise ValueError(f"unsupported V8 provider: {provider}")


def _cost(cfg: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    provider = cfg.get("provider", "gemini")
    pricing = cfg[provider]["pricing_usd_per_million"]
    normalized = []
    for call in calls:
        non_input_from_total = max(0, int(call.get("total_tokens", 0)) - int(call.get("input_tokens", 0)))
        billable_output = max(
            int(call.get("output_tokens", 0)),
            int(call.get("output_tokens", 0)) + int(call.get("thought_tokens", 0)),
            non_input_from_total,
        )
        cache_creation = int(call.get("cache_creation_input_tokens", 0))
        cache_read = int(call.get("cache_read_input_tokens", 0))
        estimated = (
            int(call.get("input_tokens", 0)) * float(pricing["input"])
            + cache_creation * float(pricing.get("cache_write_5m", pricing["input"]))
            + cache_read * float(pricing.get("cache_read", pricing["input"]))
            + billable_output * float(pricing["output"])
        ) / 1_000_000
        normalized.append({
            **call,
            "estimated_billable_output_tokens": billable_output,
            "estimated_usd": estimated,
        })
    return {
        "currency": "USD",
        "pricing_source": "frozen V8 config; estimate, not provider invoice",
        "model_calls": len(normalized),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in normalized),
        "cache_creation_input_tokens": sum(int(row.get("cache_creation_input_tokens", 0)) for row in normalized),
        "cache_read_input_tokens": sum(int(row.get("cache_read_input_tokens", 0)) for row in normalized),
        "total_input_tokens_including_cache": sum(
            int(row.get("input_tokens", 0))
            + int(row.get("cache_creation_input_tokens", 0))
            + int(row.get("cache_read_input_tokens", 0))
            for row in normalized
        ),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in normalized),
        "thought_tokens": sum(int(row.get("thought_tokens", 0)) for row in normalized),
        "estimated_billable_output_tokens": sum(row["estimated_billable_output_tokens"] for row in normalized),
        "estimated_usd": sum(row["estimated_usd"] for row in normalized),
        "calls": normalized,
    }


def _case_paths(root: Path, cfg: dict[str, Any], video_uid: str) -> tuple[Path, Path]:
    case_dir = root / cfg["source_cases"] / video_uid
    return case_dir / "question_input.json", case_dir / "r3_2_navigation_map.json"


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cached_map_text(navigation_map: dict[str, Any]) -> str:
    return json.dumps({
        "navigation_map": navigation_map,
        "map_image_catalog": map_image_catalog(navigation_map),
    }, ensure_ascii=False, separators=(",", ":"))


def run(root: Path, config_path: Path, execute_api: bool = False) -> dict[str, Any]:
    cfg = load_json(config_path)
    output = root / cfg["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    expected_count = int(cfg.get("expected_question_count", len(cfg["video_uids"])))
    if len(cfg["video_uids"]) != expected_count or len(set(cfg["video_uids"])) != expected_count:
        raise RuntimeError(f"V8 requires exactly {expected_count} unique video_uids")
    source_audit: dict[str, Any] = {}
    cases = []
    preflight_errors: list[str] = []
    for video_uid in cfg["video_uids"]:
        question_path, map_path = _case_paths(root, cfg, video_uid)
        for label, path in (("question", question_path), ("map", map_path)):
            if not path.is_file():
                preflight_errors.append(f"missing {label}: {video_uid}")
        if not question_path.is_file() or not map_path.is_file():
            continue
        question = question_view(load_json(question_path))
        navigation_map = load_json(map_path)
        if question["question_id"].split("_", 1)[0] != video_uid:
            preflight_errors.append(f"question/video mismatch: {video_uid}")
        if navigation_map.get("map_type") != "r3_2_global_av_semantic_coarse":
            preflight_errors.append(f"unexpected map type: {video_uid}")
        source_audit[video_uid] = {
            "question": {"path": str(question_path.relative_to(root)), "sha256": sha256_file(question_path), "bytes": question_path.stat().st_size},
            "map": {"path": str(map_path.relative_to(root)), "sha256": sha256_file(map_path), "bytes": map_path.stat().st_size},
        }
        cases.append({"video_uid": video_uid, "question": question, "map": navigation_map})
    write_json(output / "source_artifact_audit.json", source_audit)
    preflight = {
        "experiment_id": cfg["experiment_id"],
        "passed": not preflight_errors and len(cases) == expected_count,
        "errors": preflight_errors,
        "question_count": len(cases),
        "allowed_initial_inputs": ["question_input.json", "r3_2_navigation_map.json"],
        "conditionally_allowed_input": "only map-referenced images after unclear/insufficient/semantic_conflict",
        "maximum_review_images_per_question": int(cfg["maximum_review_images_per_question"]),
        "gold_read": False,
        "old_prediction_read": False,
        "hierarchy_read": False,
        "retrieval_embedding_read": False,
        "full_video_read": False,
        "api_calls": 0,
        "strict_evidence_only": bool(cfg.get("strict_evidence_only", False)),
        "no_commonsense_only": bool(cfg.get("no_commonsense_only", False)),
        "question_first_decision": bool(cfg.get("question_first_decision", False)),
        "prompt_cache_map": bool(cfg.get("prompt_cache_map", False)),
        "local_review_image_ids_only": bool(cfg.get("local_review_image_ids_only", False)),
    }
    write_json(output / "preflight.json", preflight)
    cache_enabled = bool(cfg.get("prompt_cache_map", False))
    write_json(output / "prompt_contract.json", {
        "common_cached_system": stage_system(cfg, CACHED_COMMON_SYSTEM) if cache_enabled else None,
        "map_system_or_stage_instruction": CACHED_MAP_STAGE_INSTRUCTION if cache_enabled else stage_system(cfg, MAP_SYSTEM),
        "review_system_or_stage_instruction": review_stage_instruction(
            cfg,
            CACHED_REVIEW_STAGE_INSTRUCTION if cache_enabled else stage_system(cfg, REVIEW_SYSTEM),
        ),
        "cached_unified_response_schema": CACHED_UNIFIED_SCHEMA if cache_enabled else None,
        "strict_evidence_only": bool(cfg.get("strict_evidence_only", False)),
        "no_commonsense_only": bool(cfg.get("no_commonsense_only", False)),
        "question_first_decision": bool(cfg.get("question_first_decision", False)),
        "prompt_cache_map": cache_enabled,
        "local_review_image_ids_only": bool(cfg.get("local_review_image_ids_only", False)),
        "thinking": cfg.get("anthropic", {}).get("thinking_budget_tokens", 0),
    })
    if not preflight["passed"]:
        raise RuntimeError(preflight_errors)
    if not execute_api:
        result = {**preflight, "ready": True, "live_execution": "not_run"}
        write_json(output / "validation_report.json", result)
        return result

    _load_env(root.parent / "thesis-av-evidence" / ".env")
    provider = cfg.get("provider", "gemini")
    required_key = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
    if not os.environ.get(required_key):
        raise RuntimeError(f"{required_key} unavailable")
    reuse_root_value = cfg.get("reuse_experiment_root")
    reuse_summary: dict[str, Any] = {}
    reuse_by_question: dict[str, dict[str, Any]] = {}
    reused_calls: list[dict[str, Any]] = []
    if reuse_root_value:
        reuse_root = root / reuse_root_value
        reuse_summary_path = reuse_root / "experiment_summary.json"
        if not reuse_summary_path.is_file():
            raise FileNotFoundError(f"reuse summary unavailable: {reuse_summary_path}")
        reuse_summary = load_json(reuse_summary_path)
        if reuse_summary.get("model") != cfg[provider]["model"] or reuse_summary.get("provider") != provider:
            raise RuntimeError("reuse provider/model mismatch")
        reuse_by_question = {
            row["question"]["question_id"]: row
            for row in reuse_summary.get("cases", [])
            if row.get("valid")
        }
        reused_calls = [dict(row) for row in reuse_summary.get("cost", {}).get("calls", [])]
        write_json(output / "reuse_audit.json", {
            "source": str(reuse_summary_path.relative_to(root)),
            "source_sha256": sha256_file(reuse_summary_path),
            "provider": provider,
            "model": cfg[provider]["model"],
            "reusable_question_ids": sorted(reuse_by_question),
            "reused_call_count": len(reused_calls),
            "reused_images": int(reuse_summary.get("total_images_reviewed", 0)),
            "reused_estimated_usd": float(reuse_summary.get("cost", {}).get("estimated_usd", 0.0)),
            "new_api_calls_for_reused_questions": 0,
        })
    case_results = []
    calls: list[dict[str, Any]] = []
    total_images = 0
    for case in cases:
        video_uid, question, navigation_map = case["video_uid"], case["question"], case["map"]
        case_out = output / "cases" / question["question_id"]
        reused = reuse_by_question.get(question["question_id"])
        if reused is not None:
            reused_case = {**reused, "reused": True, "reused_from": reuse_root_value}
            case_results.append(reused_case)
            total_images += int(reused.get("images_reviewed", 0))
            write_json(case_out / "reused_case_result.json", reused_case)
            continue
        initial_payload = {
            "question": question,
            "navigation_map": navigation_map,
            "map_image_catalog": map_image_catalog(navigation_map),
            "policy": {
                "map_is_navigation_and_evidence": True,
                "answer_directly_when_clear": True,
                "visual_review_triggers": ["unclear", "insufficient", "semantic_conflict"],
                "maximum_images_if_triggered": int(cfg["maximum_review_images_per_question"]),
            },
        }
        write_json(case_out / "map_only_input.json", initial_payload)
        map_cache = cached_map_text(navigation_map)
        if cache_enabled:
            map_system = stage_system(cfg, CACHED_COMMON_SYSTEM)
            map_inputs = [
                {
                    "type": "text", "text": map_cache,
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "type": "text",
                    "text": json.dumps({
                        "stage_instruction": CACHED_MAP_STAGE_INSTRUCTION,
                        "question": question,
                        "policy": initial_payload["policy"],
                    }, ensure_ascii=False),
                },
            ]
        else:
            map_system = stage_system(cfg, MAP_SYSTEM)
            map_inputs = [{"type": "text", "text": json.dumps(initial_payload, ensure_ascii=False)}]
        map_schema = (
            CACHED_UNIFIED_SCHEMA if cache_enabled
            else (MAP_SCHEMA_QUESTION_FIRST if cfg.get("question_first_decision") else MAP_SCHEMA)
        )
        decision, map_usage, map_raw = _provider_call(
            cfg, map_system, map_inputs, map_schema,
        )
        map_usage.update({"question_id": question["question_id"], "stage": "map_only"})
        calls.append(map_usage)
        write_json(case_out / "map_only_raw_response.json", map_raw)
        write_json(case_out / "map_only_decision.json", decision)
        errors = validate_map_decision(decision, question, navigation_map)
        if cfg.get("question_first_decision"):
            errors.extend(validate_question_understanding(decision))
        images: list[dict[str, Any]] = []
        review_result: dict[str, Any] | None = None
        if not errors and decision["map_status"] != "answerable":
            images = select_review_images(
                navigation_map, decision, int(cfg["maximum_review_images_per_question"])
            )
            if not images:
                errors.append("triggered review selected no map-routed images")
            if any(not row["exists"] for row in images):
                errors.append("triggered review contains unreadable image")
            write_json(case_out / "visual_review_manifest.json", images)
            if not errors:
                review_payload = {
                    "question": question,
                    "map_only_decision": decision,
                    "review_scope": decision["review_goal"],
                }
                local_image_ids_only = bool(cfg.get("local_review_image_ids_only", False))
                if local_image_ids_only:
                    review_payload["image_id_contract"] = {
                        "supporting_image_ids_namespace": "local_IMGxx_only",
                        "allowed_supporting_image_ids": [row["image_id"] for row in images],
                        "stable_map_ids_are_cross_references_only": True,
                    }
                if cache_enabled:
                    review_system = stage_system(cfg, CACHED_COMMON_SYSTEM)
                    inputs: list[dict[str, Any]] = [
                        {
                            "type": "text", "text": map_cache,
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        },
                        {
                            "type": "text",
                            "text": json.dumps({
                                "stage_instruction": review_stage_instruction(cfg, CACHED_REVIEW_STAGE_INSTRUCTION),
                                **review_payload,
                            }, ensure_ascii=False),
                        },
                    ]
                else:
                    review_system = review_stage_instruction(cfg, stage_system(cfg, REVIEW_SYSTEM))
                    review_payload["navigation_map"] = navigation_map
                    inputs = [{"type": "text", "text": json.dumps(review_payload, ensure_ascii=False)}]
                for image in images:
                    inputs.append({
                        "type": "text",
                        "text": (
                            f"IMAGE {image['image_id']} coarse={image['coarse_id']} "
                            f"stable_map_id={image['map_image_id']} medium={image['medium_id']} "
                            f"approximately={image['approx_timestamp_sec']:.3f}s"
                        ),
                    })
                    inputs.append({
                        "type": "image", "mime_type": "image/jpeg",
                        "data": base64.b64encode(Path(image["path"]).read_bytes()).decode("ascii"),
                    })
                write_json(case_out / "visual_review_input_sanitized.json", {
                    **review_payload,
                    "images": [{key: row[key] for key in ("image_id", "map_image_id", "coarse_id", "medium_id", "approx_timestamp_sec", "path", "sha256", "bytes")} for row in images],
                })
                review_result, review_usage, review_raw = _provider_call(
                    cfg, review_system, inputs,
                    CACHED_UNIFIED_SCHEMA if cache_enabled else REVIEW_SCHEMA,
                )
                review_usage.update({"question_id": question["question_id"], "stage": "conditional_visual_review", "image_count": len(images)})
                calls.append(review_usage)
                write_json(case_out / "visual_review_raw_response.json", review_raw)
                write_json(case_out / "visual_review_result.json", review_result)
                errors.extend(validate_review(
                    review_result, question, navigation_map, images,
                    local_image_ids_only=bool(cfg.get("local_review_image_ids_only", False)),
                ))
        total_images += len(images)
        if decision.get("map_status") == "answerable":
            option_id = decision.get("provisional_option_id")
            final_answer = {
                "question_id": question["question_id"],
                "selected_option_id": option_id,
                "answer_text": next((row["text"] for row in question["answer_options"] if row["option_id"] == option_id), ""),
                "answer_source": "map_only",
                "map_status": decision.get("map_status"),
                "images_reviewed": 0,
            }
        elif review_result is not None:
            final_answer = {
                "question_id": question["question_id"],
                "selected_option_id": review_result["selected_option_id"],
                "answer_text": review_result["answer_text"],
                "answer_source": "map_plus_conditional_visual_review",
                "map_status": decision.get("map_status"),
                "images_reviewed": len(images),
            }
        else:
            final_answer = {
                "question_id": question["question_id"], "selected_option_id": "NONE",
                "answer_text": "", "answer_source": "validation_failed",
                "map_status": decision.get("map_status"), "images_reviewed": len(images),
            }
        write_json(case_out / "final_answer.json", final_answer)
        case_results.append({
            "video_uid": video_uid,
            "question": question,
            "map_only_decision": decision,
            "visual_review_triggered": decision.get("map_status") != "answerable",
            "images_reviewed": len(images),
            "final_answer": final_answer,
            "validation_errors": errors,
            "valid": not errors,
            "reused": False,
        })

    new_cost = _cost(cfg, calls)
    reused_cost = _cost(cfg, reused_calls)
    cost = _cost(cfg, [*reused_calls, *calls])
    summary = {
        "experiment_id": cfg["experiment_id"],
        "provider": provider,
        "model": cfg[provider]["model"],
        "question_count": len(case_results),
        "map_only_calls": sum(row.get("stage") == "map_only" for row in calls),
        "conditional_visual_review_calls": sum(row.get("stage") == "conditional_visual_review" for row in calls),
        "reused_question_count": sum(bool(row.get("reused")) for row in case_results),
        "new_question_count": sum(not bool(row.get("reused")) for row in case_results),
        "reused_api_calls": len(reused_calls),
        "new_api_calls": len(calls),
        "total_images_reviewed": total_images,
        "maximum_review_images_per_question": int(cfg["maximum_review_images_per_question"]),
        "cost": cost,
        "reused_cost": reused_cost,
        "new_cost": new_cost,
        "cases": case_results,
        "valid": all(row["valid"] for row in case_results),
    }
    write_json(output / "experiment_summary.json", summary)
    source_unchanged = all(
        source_audit[uid][label]["sha256"] == sha256_file(root / source_audit[uid][label]["path"])
        for uid in source_audit for label in ("question", "map")
    )
    validation = {
        "overall_validation": "passed" if summary["valid"] and source_unchanged else "failed",
        "source_artifacts_unchanged": source_unchanged,
        "question_count": len(case_results),
        "new_api_calls": len(calls),
        "reused_api_calls": len(reused_calls),
        "cumulative_api_calls": len(calls) + len(reused_calls),
        "images_reviewed": total_images,
        "errors": [error for row in case_results for error in row["validation_errors"]],
    }
    write_json(output / "validation_report.json", validation)
    return summary
