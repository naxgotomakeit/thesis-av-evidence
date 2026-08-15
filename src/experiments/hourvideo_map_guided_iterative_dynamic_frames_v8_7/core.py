from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from experiments.hourvideo_map_guided_direct_answer_v8.core import (
    QUESTION_UNDERSTANDING_SCHEMA,
    _case_paths,
    _cost,
    _load_env,
    _provider_call,
    load_json,
    map_ids,
    question_view,
    sha256_file,
    validate_question_understanding,
    write_json,
)


SYSTEM = """You answer multiple-choice video questions with a semantic navigation
map and an iterative dynamic-frame tool. The map is navigation and preliminary
video evidence. First understand the question as an information task: identify
the question type, target entities, exact requested relation, constraints,
literal option differences, and information needed. Do not choose an answer
during that understanding step.

If the map clearly distinguishes an option, answer directly. Otherwise request
exactly two precise timestamps guided by the question and map. After each pair
of frames, answer as soon as the evidence distinguishes an option; if it does
not, request exactly two new timestamps using the map and the accumulated
evidence ledger. Do not request evenly spaced coverage or scan the whole video.
Each request must target an unresolved answer-critical distinction.

Do not use commonsense, real-world plausibility, stereotypes, frequency, or
prior expectations to select or eliminate an option. Do not treat sparse frames
as proof of unseen continuity, duration, causality, or identity. DYNIMGxxx is
the only allowed namespace in supporting_image_ids. Never invent an image ID.
Return strict JSON matching the supplied schema."""


RELATION_BINDING_ADDENDUM = """

RELATION-BINDING PASS:
Before answering or requesting frames, convert the question into explicit
evidence slots: participants, action/state, object, location, temporal relation,
and other qualifiers. Resolve modifier-scope ambiguity explicitly; do not assume
physical co-location unless the wording or evidence requires it.

Normalize every answer option into the same relation representation. Semantic
paraphrase matching between map wording and option wording is allowed and is not
commonsense inference. Inferring an unobserved event because it is typical,
likely, or narratively natural remains forbidden.

For every option, build an option_binding. Combine exact captions, coarse
summaries, ASR, participants, and scene descriptions only when their time
windows overlap or the map explicitly links them. A binding must cite its time
window and source Coarse/Medium IDs, plus any reviewed DYNIMG IDs used. Mark each
option supported, contradicted, or unknown and state the missing evidence. Answer only when exactly one option
has a complete supported binding. Otherwise request two frames specifically for
the missing slot that distinguishes the remaining options. Keep bindings concise."""


MAP_STAGE = """STAGE: map_only. Use the question and cached navigation map.
Set response_stage to map_only. Either set action=answer and answer from the map,
or set action=request_more and request exactly two dynamic timestamps. When
action=request_more, selected_option_id must be NONE and answer_text must be an
empty string; do not put a provisional option in selected_option_id. No images
have been reviewed yet, so observations and supporting_image_ids must be empty."""


REVIEW_STAGE = """STAGE: dynamic_visual_review. Inspect only the two newly
supplied frames, while using the cached map and prior evidence ledger. Record a
literal observation for each new frame. Either answer now or request exactly two
new, previously unseen timestamps that target the remaining distinction. When
action=request_more, selected_option_id must be NONE and answer_text must be an
empty string. Never request a timestamp listed in already_reviewed_timestamps.
Cite only DYNIMGxxx IDs listed in allowed_supporting_image_ids."""


REPLAN_STAGE = """STAGE: action_replan. No new images are supplied in this
call. The previous action was rejected for the stated controller reason. Do not
answer and do not repeat the rejected action. Set response_stage to
dynamic_visual_review, action to request_more, selected_option_id to NONE,
answer_text to an empty string, observations to an empty list, and request
exactly two timestamps not listed in already_reviewed_timestamps. Each new
timestamp must target the unresolved answer-critical distinction."""


FRAME_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp_sec": {"type": "number"},
        "inspection_goal": {"type": "string"},
    },
    "required": ["timestamp_sec", "inspection_goal"],
    "additionalProperties": False,
}


OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string"},
        "observation": {"type": "string"},
    },
    "required": ["image_id", "observation"],
    "additionalProperties": False,
}


NORMALIZED_RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "participants": {"type": "array", "items": {"type": "string"}},
        "action_or_state": {"type": "string"},
        "object": {"type": "string"},
        "location": {"type": "string"},
        "temporal_relation": {"type": "string"},
        "qualifiers": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "participants", "action_or_state", "object", "location",
        "temporal_relation", "qualifiers",
    ],
    "additionalProperties": False,
}


EVIDENCE_WINDOW_SCHEMA = {
    "type": "object",
    "properties": {
        "start_sec": {"type": "number"},
        "end_sec": {"type": "number"},
        "participant_evidence": {"type": "array", "items": {"type": "string"}},
        "action_evidence": {"type": "array", "items": {"type": "string"}},
        "qualifier_evidence": {"type": "array", "items": {"type": "string"}},
        "source_coarse_ids": {"type": "array", "items": {"type": "string"}},
        "source_medium_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "start_sec", "end_sec", "participant_evidence", "action_evidence",
        "qualifier_evidence", "source_coarse_ids", "source_medium_ids",
    ],
    "additionalProperties": False,
}


OPTION_BINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "option_id": {"type": "string"},
        "normalized_relation": {"type": "string"},
        "aligned_time_windows": {"type": "array", "items": {"type": "string"}},
        "evidence_summary": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["supported", "contradicted", "unknown"]},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "option_id", "normalized_relation", "aligned_time_windows",
        "evidence_summary", "source_ids", "verdict", "missing_evidence",
    ],
    "additionalProperties": False,
}


RELATION_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "participants": {"type": "array", "items": {"type": "string"}},
        "action_or_state": {"type": "string"},
        "object": {"type": "string"},
        "location": {"type": "string"},
        "temporal_relation": {"type": "string"},
        "other_qualifiers": {"type": "array", "items": {"type": "string"}},
        "scope_ambiguities": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "participants", "action_or_state", "object", "location",
        "temporal_relation", "other_qualifiers", "scope_ambiguities",
    ],
    "additionalProperties": False,
}


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_stage": {"type": "string", "enum": ["map_only", "dynamic_visual_review"]},
        "action": {"type": "string", "enum": ["answer", "request_more"]},
        "question_understanding": QUESTION_UNDERSTANDING_SCHEMA,
        "question_id": {"type": "string"},
        "selected_option_id": {"type": "string"},
        "answer_text": {"type": "string"},
        "reasoning": {"type": "string"},
        "supporting_coarse_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_medium_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_image_ids": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": OBSERVATION_SCHEMA},
        "requested_frames": {"type": "array", "items": FRAME_REQUEST_SCHEMA},
        "remaining_uncertainty": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "response_stage", "action", "question_understanding", "question_id",
        "selected_option_id", "answer_text", "reasoning", "supporting_coarse_ids",
        "supporting_medium_ids", "supporting_image_ids", "observations",
        "requested_frames", "remaining_uncertainty",
    ],
    "additionalProperties": False,
}


RESPONSE_SCHEMA_RELATION_BINDING = {
    **RESPONSE_SCHEMA,
    "properties": {
        **RESPONSE_SCHEMA["properties"],
        "relation_query": RELATION_QUERY_SCHEMA,
        "option_bindings": {"type": "array", "items": OPTION_BINDING_SCHEMA},
    },
    "required": [*RESPONSE_SCHEMA["required"], "relation_query", "option_bindings"],
}


def system_prompt(cfg: dict[str, Any]) -> str:
    if cfg.get("relation_binding_pass", False):
        return SYSTEM + RELATION_BINDING_ADDENDUM
    return SYSTEM


def response_schema(cfg: dict[str, Any]) -> dict[str, Any]:
    return RESPONSE_SCHEMA_RELATION_BINDING if cfg.get("relation_binding_pass", False) else RESPONSE_SCHEMA


def cached_map_text(navigation_map: dict[str, Any]) -> str:
    return json.dumps(
        {"navigation_map": navigation_map}, ensure_ascii=False, separators=(",", ":")
    )


def observable_commonsense_violations(result: dict[str, Any]) -> list[str]:
    if result.get("action") != "answer":
        return []
    text = " ".join([
        str(result.get("reasoning", "")), str(result.get("answer_text", "")),
    ]).lower()
    violations: list[str] = []
    explicit_prior_markers = {
        "logical sequence": "logical_sequence_prior",
        "would have been": "unobserved_would_have_prior",
        "most commonly": "frequency_prior",
        "most likely": "likelihood_prior",
        "typically": "typicality_prior",
        "usually": "usuality_prior",
        "plausib": "plausibility_prior",
    }
    for marker, label in explicit_prior_markers.items():
        if marker in text:
            violations.append(label)
    if any(marker in text for marker in ("not shown", "not observed", "not visible")) and any(
        marker in text for marker in ("would have", "likely", "logical", "common", "typically", "usually", "probably")
    ):
        violations.append("unobserved_claim_resolved_by_prior")
    return sorted(set(violations))


def medium_intervals(navigation_map: dict[str, Any]) -> dict[str, tuple[str, float, float]]:
    result: dict[str, tuple[str, float, float]] = {}
    for coarse in navigation_map.get("coarse_regions", []):
        for medium in coarse.get("exact_source_captions", []):
            result[medium["medium_id"]] = (
                coarse["coarse_id"], float(medium["start_sec"]), float(medium["end_sec"])
            )
    return result


def frame_inventory(navigation_map: dict[str, Any]) -> list[tuple[int, Path]]:
    example: Path | None = None
    for coarse in navigation_map.get("coarse_regions", []):
        for medium in coarse.get("exact_source_captions", []):
            paths = medium.get("source_frame_paths", [])
            if paths:
                example = Path(paths[0])
                break
        if example is not None:
            break
    if example is None:
        return []
    rows: list[tuple[int, Path]] = []
    for path in example.parent.glob("frame_*.jpg"):
        match = re.search(r"frame_(\d+)\.jpg$", path.name)
        if match:
            rows.append((int(match.group(1)), path))
    return sorted(rows)


def resolve_requested_frames(
    requests: list[dict[str, Any]], inventory: list[tuple[int, Path]],
    already_seen_seconds: set[int], start_index: int,
    navigation_map: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    output: list[dict[str, Any]] = []
    if not inventory:
        return [], ["dynamic frame inventory unavailable"]
    for offset, request in enumerate(requests):
        target = float(request["timestamp_sec"])
        second, path = min(inventory, key=lambda row: (abs(row[0] - target), row[0]))
        if second in already_seen_seconds or any(row["resolved_timestamp_sec"] == second for row in output):
            errors.append(f"duplicate dynamic frame second: {second}")
            continue
        coarse_id = ""
        medium_id = ""
        if navigation_map is not None:
            matches = [
                (medium, coarse_id_value)
                for medium, (coarse_id_value, start, end) in medium_intervals(navigation_map).items()
                if start <= second <= end
            ]
            if matches:
                medium_id, coarse_id = matches[0]
        output.append({
            "image_id": f"DYNIMG{start_index + offset:03d}",
            "requested_timestamp_sec": target,
            "resolved_timestamp_sec": second,
            "coarse_id": coarse_id or request.get("coarse_id", ""),
            "medium_id": medium_id or request.get("medium_id", ""),
            "inspection_goal": request["inspection_goal"],
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return output, errors


def validate_response(
    result: dict[str, Any], question: dict[str, Any], navigation_map: dict[str, Any],
    batch_size: int, current_images: list[dict[str, Any]], all_image_ids: set[str],
    relation_binding_required: bool = False,
) -> list[str]:
    errors: list[str] = []
    option_ids = {row["option_id"] for row in question["answer_options"]}
    coarse_ids, medium_ids = map_ids(navigation_map)
    intervals = medium_intervals(navigation_map)
    video_start = min((start for _, start, _ in intervals.values()), default=0.0)
    video_end = max((end for _, _, end in intervals.values()), default=0.0)
    if result.get("question_id") != question["question_id"]:
        errors.append("question_id mismatch")
    errors.extend(validate_question_understanding(result))
    if not set(result.get("supporting_coarse_ids", [])) <= coarse_ids:
        errors.append("unknown supporting coarse ID")
    if not set(result.get("supporting_medium_ids", [])) <= medium_ids:
        errors.append("unknown supporting medium ID")
    if not set(result.get("supporting_image_ids", [])) <= all_image_ids:
        errors.append("unknown supporting dynamic image ID")
    current_ids = {row["image_id"] for row in current_images}
    observation_ids = {row.get("image_id") for row in result.get("observations", [])}
    if current_images and observation_ids != current_ids:
        errors.append("observations must cover exactly the current image pair")
    if not current_images and result.get("observations"):
        errors.append("map-only response contains image observations")
    action = result.get("action")
    requests = result.get("requested_frames", [])
    if action == "answer":
        if result.get("selected_option_id") not in option_ids:
            errors.append("answer action has invalid option")
        if requests:
            errors.append("answer action also requests frames")
    elif action == "request_more":
        if result.get("selected_option_id") != "NONE":
            errors.append("request_more must select NONE")
        if len(requests) != batch_size:
            errors.append(f"request_more must request exactly {batch_size} frames")
        requested_seconds: list[float] = []
        for request in requests:
            timestamp = float(request.get("timestamp_sec", -1))
            requested_seconds.append(timestamp)
            if not video_start <= timestamp <= video_end:
                errors.append("requested timestamp outside video map range")
            if not str(request.get("inspection_goal", "")).strip():
                errors.append("dynamic frame request has no inspection goal")
        if len(requested_seconds) != len(set(requested_seconds)):
            errors.append("duplicate requested timestamp")
    else:
        errors.append("invalid action")
    if relation_binding_required:
        relation_query = result.get("relation_query")
        if not isinstance(relation_query, dict):
            errors.append("missing relation query")
        else:
            if not relation_query.get("participants"):
                errors.append("relation query has no participants")
            if not str(relation_query.get("action_or_state", "")).strip():
                errors.append("relation query has no action or state")
        bindings = result.get("option_bindings")
        if not isinstance(bindings, list):
            errors.append("missing option bindings")
        else:
            binding_ids = [row.get("option_id") for row in bindings]
            if len(binding_ids) != len(option_ids) or set(binding_ids) != option_ids:
                errors.append("option bindings must cover every option exactly once")
            for binding in bindings:
                source_ids = set(binding.get("source_ids", []))
                if not source_ids <= coarse_ids | medium_ids | all_image_ids:
                    errors.append("option binding has unknown source ID")
            if action == "answer":
                supported = [row.get("option_id") for row in bindings if row.get("verdict") == "supported"]
                if supported != [result.get("selected_option_id")]:
                    errors.append("answer must match exactly one supported option binding")
    return errors


def _call(
    cfg: dict[str, Any], map_cache: str, payload: dict[str, Any],
    images: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs: list[dict[str, Any]] = [
        {"type": "text", "text": map_cache, "cache_control": {"type": "ephemeral", "ttl": "5m"}},
        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
    ]
    for image in images:
        inputs.append({
            "type": "text",
            "text": (
                f"IMAGE {image['image_id']} requested={image['requested_timestamp_sec']:.3f}s "
                f"actual={image['resolved_timestamp_sec']}s coarse={image['coarse_id']} "
                f"medium={image['medium_id']} goal={image['inspection_goal']}"
            ),
        })
        inputs.append({
            "type": "image", "mime_type": "image/jpeg",
            "data": base64.b64encode(Path(image["path"]).read_bytes()).decode("ascii"),
        })
    return _provider_call(cfg, system_prompt(cfg), inputs, response_schema(cfg))


def _replan_call(
    cfg: dict[str, Any], map_cache: str, question: dict[str, Any],
    previous_result: dict[str, Any], controller_reasons: list[str],
    ledger: list[dict[str, Any]], seen_seconds: set[int], remaining: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = {
        "stage_instruction": REPLAN_STAGE,
        "question": question,
        "question_understanding": previous_result["question_understanding"],
        "controller_rejection_reasons": controller_reasons,
        "rejected_response": previous_result,
        "prior_evidence_ledger": ledger,
        "already_reviewed_timestamps": sorted(seen_seconds),
        "dynamic_frame_policy": {
            "batch_size": 2, "remaining_frame_budget": remaining,
        },
    }
    return _call(cfg, map_cache, payload, [])


def run(root: Path, config_path: Path, execute_api: bool = False) -> dict[str, Any]:
    cfg = load_json(config_path)
    output = root / cfg["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    batch_size = int(cfg.get("dynamic_frame_batch_size", 2))
    maximum = int(cfg.get("maximum_dynamic_frames_per_question", 24))
    errors: list[str] = []
    if batch_size != 2:
        errors.append("V8.7 requires dynamic_frame_batch_size=2")
    if maximum < 2 or maximum % 2:
        errors.append("maximum_dynamic_frames_per_question must be a positive multiple of two")
    video_uids = cfg["video_uids"]
    if len(video_uids) != int(cfg.get("expected_question_count", len(video_uids))):
        errors.append("question count mismatch")
    cases: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for video_uid in video_uids:
        question_path, map_path = _case_paths(root, cfg, video_uid)
        if not question_path.is_file() or not map_path.is_file():
            errors.append(f"missing source case: {video_uid}")
            continue
        question = question_view(load_json(question_path))
        navigation_map = load_json(map_path)
        inventory = frame_inventory(navigation_map)
        if not inventory:
            errors.append(f"missing 1fps frame inventory: {video_uid}")
        cases.append({
            "video_uid": video_uid, "question": question,
            "navigation_map": navigation_map, "inventory": inventory,
        })
        source_audit[video_uid] = {
            "question": {"path": str(question_path.relative_to(root)), "sha256": sha256_file(question_path)},
            "map": {"path": str(map_path.relative_to(root)), "sha256": sha256_file(map_path)},
            "dynamic_frame_directory": str(inventory[0][1].parent) if inventory else None,
            "dynamic_frame_count": len(inventory),
        }
    write_json(output / "source_artifact_audit.json", source_audit)
    preflight = {
        "experiment_id": cfg["experiment_id"], "passed": not errors,
        "errors": errors, "question_count": len(cases), "api_calls": 0,
        "gold_read": False, "old_prediction_read": False,
        "dynamic_frame_batch_size": batch_size,
        "maximum_dynamic_frames_per_question": maximum,
        "frame_source": "nearest frame from existing 1fps inventory",
        "prompt_cache_map": True,
    }
    write_json(output / "preflight.json", preflight)
    write_json(output / "prompt_contract.json", {
        "system": system_prompt(cfg), "map_stage": MAP_STAGE, "review_stage": REVIEW_STAGE,
        "replan_stage": REPLAN_STAGE,
        "response_schema": response_schema(cfg), "thinking": cfg.get("anthropic", {}).get("thinking_budget_tokens", 0),
        "relation_binding_pass": bool(cfg.get("relation_binding_pass", False)),
        "observable_commonsense_gate": bool(cfg.get("observable_commonsense_gate", False)),
        "action_replan_on_duplicate": bool(cfg.get("action_replan_on_duplicate", False)),
    })
    if errors:
        raise RuntimeError(errors)
    if not execute_api:
        result = {**preflight, "ready": True, "live_execution": "not_run"}
        write_json(output / "validation_report.json", result)
        return result

    _load_env(root.parent / "thesis-av-evidence" / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY unavailable")
    calls: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    total_images = 0
    for case in cases:
        question = case["question"]
        navigation_map = case["navigation_map"]
        case_out = output / "cases" / question["question_id"]
        map_cache = cached_map_text(navigation_map)
        initial_payload = {
            "stage_instruction": MAP_STAGE,
            "question": question,
            "dynamic_frame_policy": {
                "batch_size": batch_size, "maximum_total_frames": maximum,
                "frames_reviewed_so_far": 0,
            },
        }
        write_json(case_out / "map_only_input.json", initial_payload)
        result, usage, raw = _call(cfg, map_cache, initial_payload, [])
        usage.update({"question_id": question["question_id"], "stage": "map_only", "image_count": 0})
        calls.append(usage)
        write_json(case_out / "map_only_raw_response.json", raw)
        write_json(case_out / "map_only_result.json", result)
        relation_binding_required = bool(cfg.get("relation_binding_pass", False))
        case_errors = validate_response(
            result, question, navigation_map, batch_size, [], set(),
            relation_binding_required=relation_binding_required,
        )
        seen_seconds: set[int] = set()
        all_images: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        round_index = 0
        replan_index = 0
        replans: list[dict[str, Any]] = []
        maximum_replans = int(cfg.get("maximum_action_replans_per_question", 0))
        commonsense_gate = bool(cfg.get("observable_commonsense_gate", False))
        replan_duplicates = bool(cfg.get("action_replan_on_duplicate", False))
        initial_violations = observable_commonsense_violations(result) if commonsense_gate else []
        if not case_errors and initial_violations and replan_index < maximum_replans:
            replan_index += 1
            rejected = result
            result, usage, raw = _replan_call(
                cfg, map_cache, question, rejected,
                [f"observable_no_commonsense_violation:{x}" for x in initial_violations],
                ledger, seen_seconds, maximum - len(all_images),
            )
            usage.update({
                "question_id": question["question_id"],
                "stage": f"action_replan_{replan_index}", "image_count": 0,
            })
            calls.append(usage)
            write_json(case_out / f"replan_{replan_index:02d}_raw_response.json", raw)
            write_json(case_out / f"replan_{replan_index:02d}_result.json", result)
            replan_errors = validate_response(
                result, question, navigation_map, batch_size, [], set(),
                relation_binding_required=relation_binding_required,
            )
            if result.get("action") != "request_more":
                replan_errors.append("action replan did not request more frames")
            replans.append({"reason": initial_violations, "rejected": rejected, "result": result, "errors": replan_errors})
            case_errors.extend(replan_errors)
        while not case_errors and result["action"] == "request_more" and len(all_images) < maximum:
            images, resolution_errors = resolve_requested_frames(
                result["requested_frames"], case["inventory"], seen_seconds,
                len(all_images) + 1, navigation_map,
            )
            if resolution_errors and replan_duplicates and replan_index < maximum_replans:
                replan_index += 1
                rejected = result
                result, usage, raw = _replan_call(
                    cfg, map_cache, question, rejected, resolution_errors,
                    ledger, seen_seconds, maximum - len(all_images),
                )
                usage.update({
                    "question_id": question["question_id"],
                    "stage": f"action_replan_{replan_index}", "image_count": 0,
                })
                calls.append(usage)
                write_json(case_out / f"replan_{replan_index:02d}_raw_response.json", raw)
                write_json(case_out / f"replan_{replan_index:02d}_result.json", result)
                replan_errors = validate_response(
                    result, question, navigation_map, batch_size, [],
                    {row["image_id"] for row in all_images},
                    relation_binding_required=relation_binding_required,
                )
                if result.get("action") != "request_more":
                    replan_errors.append("action replan did not request more frames")
                replans.append({"reason": resolution_errors, "rejected": rejected, "result": result, "errors": replan_errors})
                case_errors.extend(replan_errors)
                continue
            case_errors.extend(resolution_errors)
            if case_errors:
                break
            round_index += 1
            for image in images:
                seen_seconds.add(image["resolved_timestamp_sec"])
            all_images.extend(images)
            remaining = maximum - len(all_images)
            review_payload = {
                "stage_instruction": REVIEW_STAGE,
                "question": question,
                "question_understanding": result["question_understanding"],
                "prior_evidence_ledger": ledger,
                "already_reviewed_timestamps": sorted(seen_seconds),
                "current_frame_manifest": [
                    {key: row[key] for key in (
                        "image_id", "requested_timestamp_sec", "resolved_timestamp_sec",
                        "coarse_id", "medium_id", "inspection_goal",
                    )} for row in images
                ],
                "allowed_supporting_image_ids": [row["image_id"] for row in all_images],
                "dynamic_frame_policy": {
                    "batch_size": batch_size, "frames_reviewed_so_far": len(all_images),
                    "remaining_frame_budget": remaining,
                    "must_answer_now": remaining < batch_size,
                },
            }
            write_json(case_out / f"round_{round_index:02d}_input_sanitized.json", {
                **review_payload,
                "images": [{**row, "path": row["path"]} for row in images],
            })
            next_result, usage, raw = _call(cfg, map_cache, review_payload, images)
            usage.update({
                "question_id": question["question_id"],
                "stage": f"dynamic_visual_review_round_{round_index}",
                "image_count": len(images),
            })
            calls.append(usage)
            write_json(case_out / f"round_{round_index:02d}_raw_response.json", raw)
            write_json(case_out / f"round_{round_index:02d}_result.json", next_result)
            all_ids = {row["image_id"] for row in all_images}
            round_errors = validate_response(
                next_result, question, navigation_map, batch_size, images, all_ids,
                relation_binding_required=relation_binding_required,
            )
            if remaining < batch_size and next_result.get("action") != "answer":
                round_errors.append("dynamic frame budget exhausted before answer")
            observations_by_id = {
                row["image_id"]: row["observation"]
                for row in next_result.get("observations", [])
            }
            ledger.extend([
                {
                    "image_id": image["image_id"],
                    "timestamp_sec": image["resolved_timestamp_sec"],
                    "coarse_id": image["coarse_id"],
                    "medium_id": image["medium_id"],
                    "observation": observations_by_id.get(image["image_id"], ""),
                }
                for image in images
            ])
            rounds.append({
                "round": round_index, "images": images, "result": next_result,
                "validation_errors": round_errors,
            })
            case_errors.extend(round_errors)
            result = next_result
            compliance_violations = observable_commonsense_violations(result) if commonsense_gate else []
            if not case_errors and compliance_violations:
                if replan_index < maximum_replans and remaining >= batch_size:
                    replan_index += 1
                    rejected = result
                    result, usage, raw = _replan_call(
                        cfg, map_cache, question, rejected,
                        [f"observable_no_commonsense_violation:{x}" for x in compliance_violations],
                        ledger, seen_seconds, remaining,
                    )
                    usage.update({
                        "question_id": question["question_id"],
                        "stage": f"action_replan_{replan_index}", "image_count": 0,
                    })
                    calls.append(usage)
                    write_json(case_out / f"replan_{replan_index:02d}_raw_response.json", raw)
                    write_json(case_out / f"replan_{replan_index:02d}_result.json", result)
                    replan_errors = validate_response(
                        result, question, navigation_map, batch_size, [],
                        {row["image_id"] for row in all_images},
                        relation_binding_required=relation_binding_required,
                    )
                    if result.get("action") != "request_more":
                        replan_errors.append("action replan did not request more frames")
                    replans.append({"reason": compliance_violations, "rejected": rejected, "result": result, "errors": replan_errors})
                    case_errors.extend(replan_errors)
                else:
                    case_errors.extend([
                        f"observable no-commonsense violation: {value}"
                        for value in compliance_violations
                    ])
        total_images += len(all_images)
        if not case_errors and result.get("action") == "answer":
            final_answer = {
                "question_id": question["question_id"],
                "selected_option_id": result["selected_option_id"],
                "answer_text": result["answer_text"],
                "answer_source": "map_only" if not all_images else "map_plus_iterative_dynamic_frames",
                "images_reviewed": len(all_images), "review_rounds": round_index,
            }
        else:
            final_answer = {
                "question_id": question["question_id"], "selected_option_id": "NONE",
                "answer_text": "", "answer_source": "validation_failed",
                "images_reviewed": len(all_images), "review_rounds": round_index,
            }
        write_json(case_out / "dynamic_frame_manifest.json", all_images)
        write_json(case_out / "evidence_ledger.json", ledger)
        write_json(case_out / "final_answer.json", final_answer)
        case_results.append({
            "video_uid": case["video_uid"], "question": question,
            "rounds": rounds, "action_replans": replans, "final_answer": final_answer,
            "validation_errors": case_errors, "valid": not case_errors,
        })
    cost = _cost(cfg, calls)
    summary = {
        "experiment_id": cfg["experiment_id"], "provider": cfg["provider"],
        "model": cfg[cfg["provider"]]["model"], "question_count": len(case_results),
        "api_calls": len(calls), "total_images_reviewed": total_images,
        "dynamic_frame_batch_size": batch_size,
        "maximum_dynamic_frames_per_question": maximum,
        "cost": cost, "cases": case_results,
        "valid": all(row["valid"] for row in case_results),
    }
    write_json(output / "experiment_summary.json", summary)
    write_json(output / "validation_report.json", {
        "overall_validation": "passed" if summary["valid"] else "failed",
        "question_count": len(case_results), "api_calls": len(calls),
        "images_reviewed": total_images,
        "errors": [error for row in case_results for error in row["validation_errors"]],
    })
    return summary
