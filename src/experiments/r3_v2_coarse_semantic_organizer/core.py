from __future__ import annotations

import copy
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are an offline, question-agnostic semantic Organizer for a video navigation map. You receive the complete ordered timeline of Medium-level visual captions. Globally organize the whole timeline into contiguous, coherent semantic event phases. Group consecutive records when they form one event phase even if wording, viewpoint, or visible objects change. Split for meaningful changes in scene, principal activity, human interaction, event state, incident phase, or response phase. Do not split merely for wording, camera-angle, object, or short noisy-caption changes, and do not merge clearly different phases merely because people or objects persist. Preserve caption-supported actions, interactions, visible relations, scene transitions, incident developments, response phases, and explicit uncertainty. Navigation summaries must be concise, faithful navigation aids rather than final answers or reviewed visual confirmation. Return only the required groups JSON. Generate no IDs, timestamps, source IDs, audio IDs, Storyline, retrieval choices, scores, entities/actions/locations side fields, or answers."""

MODEL_OUTPUT_FIELDS = [
    "group_index", "start_medium_index", "end_medium_index",
    "navigation_summary", "uncertainty_notes",
]
FORBIDDEN_PAYLOAD_TERMS = {
    "detector", "tracking", "structured_fallback", "storyline", "coarse_summary",
    "question", "answer_options", "experiment_name", "rung", "audio", "asr",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["groups"],
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": MODEL_OUTPUT_FIELDS,
                    "properties": {
                        "group_index": {"type": "integer"},
                        "start_medium_index": {"type": "integer"},
                        "end_medium_index": {"type": "integer"},
                        "navigation_summary": {"type": "string"},
                        "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }


def build_payload(index: dict[str, Any]) -> dict[str, Any]:
    rows = index["medium_nodes"]
    if len(rows) != 30:
        raise ValueError(f"expected_30_mediums:{len(rows)}")
    timeline = []
    for position, row in enumerate(rows):
        caption = row.get("qwen_caption")
        if not isinstance(caption, str) or not caption:
            raise ValueError(f"missing_caption:{position}")
        timeline.append({"medium_index": position, "caption": caption})
    return {"ordered_medium_caption_timeline": timeline}


def payload_leakage(payload: dict[str, Any]) -> list[str]:
    text = canonical_bytes(payload).decode("utf-8").lower()
    return sorted(term for term in FORBIDDEN_PAYLOAD_TERMS if f'"{term}"' in text)


def validate_grouping(value: Any, medium_count: int = 30) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != {"groups"} or not isinstance(value.get("groups"), list):
        return {"valid": False, "errors": ["invalid_wrapper"], "group_count": 0}
    groups = value["groups"]
    if not groups:
        errors.append("groups_empty")
    expected_start = 0
    seen_indices: list[int] = []
    covered: list[int] = []
    for position, group in enumerate(groups):
        if not isinstance(group, dict) or list(group) != MODEL_OUTPUT_FIELDS:
            errors.append(f"group_{position}:fields_or_order")
            continue
        if type(group["group_index"]) is not int:
            errors.append(f"group_{position}:group_index_not_integer")
            continue
        seen_indices.append(group["group_index"])
        start, end = group["start_medium_index"], group["end_medium_index"]
        if type(start) is not int or type(end) is not int:
            errors.append(f"group_{position}:range_not_integer")
            continue
        if start != expected_start:
            errors.append(f"group_{position}:expected_start_{expected_start}_got_{start}")
        if start < 0 or end >= medium_count or start > end:
            errors.append(f"group_{position}:invalid_range")
        else:
            covered.extend(range(start, end + 1))
        expected_start = end + 1
        if not isinstance(group["navigation_summary"], str) or not group["navigation_summary"].strip():
            errors.append(f"group_{position}:empty_navigation_summary")
        notes = group["uncertainty_notes"]
        if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
            errors.append(f"group_{position}:invalid_uncertainty_notes")
    if seen_indices != list(range(len(groups))):
        errors.append("group_indices_not_canonical")
    if covered != list(range(medium_count)):
        errors.append("medium_coverage_not_exact")
    if groups and expected_start != medium_count:
        errors.append("last_group_does_not_end_at_final_medium")
    return {
        "valid": not errors,
        "errors": errors,
        "group_count": len(groups),
        "covered_medium_indices": covered,
    }


def _overlapping_audio(audio: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [copy.deepcopy(row) for row in audio if float(row["start_sec"]) < end and float(row["end_sec"]) > start]


def reconstruct(index: dict[str, Any], grouping: dict[str, Any]) -> dict[str, Any]:
    validation = validate_grouping(grouping, len(index["medium_nodes"]))
    if not validation["valid"]:
        raise ValueError(f"invalid_grouping:{validation['errors']}")
    mediums = index["medium_nodes"]
    audio = index["audio_nodes"]
    regions = []
    for group in grouping["groups"]:
        selected = mediums[group["start_medium_index"] : group["end_medium_index"] + 1]
        start, end = float(selected[0]["start_sec"]), float(selected[-1]["end_sec"])
        attached = _overlapping_audio(audio, start, end)
        regions.append({
            "coarse_id": f"C{group['group_index'] + 1:02d}",
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "source_medium_ids": [row["medium_id"] for row in selected],
            "source_fine_ids": [fine_id for row in selected for fine_id in row["child_fine_ids"]],
            "navigation_summary": group["navigation_summary"],
            "uncertainty_notes": list(group["uncertainty_notes"]),
            "exact_source_captions": [
                {"medium_id": row["medium_id"], "caption": row["qwen_caption"]} for row in selected
            ],
            "audio_ids": [row["audio_id"] for row in attached],
            "map_type": "semantic_coarse",
            "semantic_fields_available": True,
            "semantic_source": "canonical_qwen_caption",
        })
    return {
        "map_type": "semantic_coarse",
        "semantic_fields_available": True,
        "semantic_source": "canonical_qwen_caption",
        "coarse_regions": regions,
        "storyline_events": [],
        "has_storyline": False,
        "hard_filtering_allowed": False,
        "provenance": {
            "semantic_grouping_and_navigation_summary": "one global Organizer call",
            "coarse_ids_boundaries_coverage_fine_and_audio": "deterministic code",
            "audio_overlap_rule": "half-open [start_sec,end_sec)",
            "captions": "byte-identical canonical repaired Qwen captions",
            "asr_used_for_grouping": False,
            "question_conditioned": False,
        },
    }


def planner_view(semantic_map: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_type": "semantic_coarse",
        "coarse_regions": [
            {
                "coarse_id": row["coarse_id"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "source_medium_ids": list(row["source_medium_ids"]),
                "source_audio_ids": list(row["audio_ids"]),
                "event_label": "",
                "summary": row["navigation_summary"],
                "uncertainty_notes": list(row["uncertainty_notes"]),
                "coarse_summary": row["navigation_summary"],
                "exact_source_captions": copy.deepcopy(row["exact_source_captions"]),
                "adapter_metadata": {
                    "adapter_derived": True,
                    "source_field": "navigation_summary",
                    "semantic_summary": True,
                    "map_type": "semantic_coarse",
                    "stored_index_modified": False,
                    "hard_filtering_allowed": False,
                },
            }
            for row in semantic_map["coarse_regions"]
        ],
        "storyline_events": [],
        "has_storyline": False,
        "hard_filtering_allowed": False,
        "planner_policy": {
            "map_may_guide_query_formulation": True,
            "suggested_coarse_are_explanatory_hints": True,
            "all_mediums_remain_eligible": True,
            "map_text_is_not_sufficiency_evidence": True,
        },
    }


def validate_reconstruction(index: dict[str, Any], semantic_map: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    mediums = index["medium_nodes"]
    fine_ids = {row["fine_id"] for row in index["fine_nodes"]}
    audio_ids = {row["audio_id"] for row in index["audio_nodes"]}
    flattened_mediums = [item for row in semantic_map["coarse_regions"] for item in row["source_medium_ids"]]
    if flattened_mediums != [row["medium_id"] for row in mediums]:
        errors.append("medium_coverage_or_order_mismatch")
    flattened_fine = [item for row in semantic_map["coarse_regions"] for item in row["source_fine_ids"]]
    canonical_fine = [item for row in mediums for item in row["child_fine_ids"]]
    if flattened_fine != canonical_fine or set(flattened_fine) != fine_ids:
        errors.append("fine_mapping_mismatch")
    if any(item not in audio_ids for row in semantic_map["coarse_regions"] for item in row["audio_ids"]):
        errors.append("unknown_audio_id")
    for position, row in enumerate(semantic_map["coarse_regions"]):
        source = [item for item in mediums if item["medium_id"] in row["source_medium_ids"]]
        if row["start_sec"] != float(source[0]["start_sec"]) or row["end_sec"] != float(source[-1]["end_sec"]):
            errors.append(f"coarse_{position}:boundary_mismatch")
        expected_audio = [item["audio_id"] for item in _overlapping_audio(index["audio_nodes"], row["start_sec"], row["end_sec"])]
        if row["audio_ids"] != expected_audio:
            errors.append(f"coarse_{position}:audio_overlap_mismatch")
        expected_captions = [{"medium_id": item["medium_id"], "caption": item["qwen_caption"]} for item in source]
        if canonical_bytes(row["exact_source_captions"]) != canonical_bytes(expected_captions):
            errors.append(f"coarse_{position}:caption_integrity_mismatch")
    if semantic_map["storyline_events"] or semantic_map["has_storyline"]:
        errors.append("storyline_not_disabled")
    if semantic_map["hard_filtering_allowed"]:
        errors.append("hard_filtering_enabled")
    return {"valid": not errors, "errors": errors, "medium_count": len(flattened_mediums), "fine_count": len(flattened_fine)}


def validate_planner_view(view: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if view.get("storyline_events") != [] or view.get("has_storyline") is not False:
        errors.append("storyline_present")
    if view.get("hard_filtering_allowed") is not False:
        errors.append("hard_filtering_not_false")
    flattened = [item for row in view.get("coarse_regions", []) for item in row.get("source_medium_ids", [])]
    if flattened != [row["medium_id"] for row in index["medium_nodes"]]:
        errors.append("medium_coverage_not_exact")
    for position, row in enumerate(view.get("coarse_regions", [])):
        if row.get("coarse_summary") != row.get("summary"):
            errors.append(f"coarse_{position}:adapter_summary_mismatch")
        metadata = row.get("adapter_metadata", {})
        if metadata.get("source_field") != "navigation_summary" or metadata.get("hard_filtering_allowed") is not False:
            errors.append(f"coarse_{position}:adapter_metadata_invalid")
    policy = view.get("planner_policy", {})
    if policy.get("all_mediums_remain_eligible") is not True or policy.get("map_text_is_not_sufficiency_evidence") is not True:
        errors.append("planner_policy_invalid")
    # Exercise the existing isolated Coarse-only compatibility consumer using
    # its exact narrow field contract. The richer view remains available to a
    # future Planner prompt, while this proves that no frozen consumer change
    # is needed to load the mechanical map.
    from experiments.egopolice_coarse_map_planner_compatibility.core import (
        adapt_coarse_map,
        validate_coarse_map,
    )

    strict_contract = {
        "coarse_regions": [
            {
                "coarse_id": row["coarse_id"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "source_medium_ids": list(row["source_medium_ids"]),
                "source_audio_ids": list(row["source_audio_ids"]),
                "event_label": row["event_label"],
                "summary": row["summary"],
                "uncertainty_notes": list(row["uncertainty_notes"]),
            }
            for row in view.get("coarse_regions", [])
        ],
        "storyline_events": [],
        "has_storyline": False,
    }
    frozen_validation = validate_coarse_map(strict_contract, index["medium_nodes"])
    if not frozen_validation["valid"]:
        errors.extend(f"frozen_consumer:{item}" for item in frozen_validation["errors"])
    try:
        adapted = adapt_coarse_map(strict_contract, index)
        frozen_adapter_loadable = (
            len(adapted["coarse_nodes"]) == len(strict_contract["coarse_regions"])
            and [row["medium_id"] for row in adapted["medium_nodes"]]
            == [row["medium_id"] for row in index["medium_nodes"]]
            and adapted["storyline_events"] == []
        )
    except Exception as error:
        frozen_adapter_loadable = False
        errors.append(f"frozen_adapter_exception:{type(error).__name__}:{error}")
    if not frozen_adapter_loadable:
        errors.append("frozen_adapter_not_loadable")
    return {
        "valid": not errors,
        "errors": errors,
        "coarse_count": len(view.get("coarse_regions", [])),
        "medium_coverage_count": len(flattened),
        "frozen_coarse_map_validator_valid": frozen_validation["valid"],
        "frozen_compatibility_adapter_loadable": frozen_adapter_loadable,
    }


def _call(api_key: str, config: dict[str, Any], payload: dict[str, Any], system: str = SYSTEM_PROMPT) -> tuple[str, dict[str, Any]]:
    import anthropic

    started = time.perf_counter()
    response = anthropic.Anthropic(api_key=api_key).messages.create(
        model=config["model"],
        max_tokens=int(config["max_tokens"]),
        temperature=float(config["temperature"]),
        system=system,
        messages=[{"role": "user", "content": canonical_bytes(payload).decode("utf-8")}],
        output_config={"format": {"type": "json_schema", "schema": output_schema()}},
    )
    raw = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return raw, {
        "provider": "anthropic",
        "model": config["model"],
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "latency_sec": time.perf_counter() - started,
        "stop_reason": response.stop_reason,
        "response_id": str(response.id),
        "request_id": str(getattr(response, "_request_id", "") or ""),
    }


def _historical_audit(root: Path, config: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    provenance = load_json(root / config["historical_provenance"])
    organizer = provenance["organizer"]
    return {
        "reference_only": True,
        "historical_model": organizer["model"],
        "historical_prompt_source": organizer["prompt_source"],
        "historical_attempts": organizer["attempts"],
        "historical_coarse_count": len(historical["coarse_nodes"]),
        "historical_storyline_count": len(historical["storyline_events"]),
        "semantic_behavior_preserved": [
            "global timeline understanding", "semantic phase grouping", "action and interaction awareness",
            "event-transition awareness", "concise navigation summaries", "uncertainty preservation",
        ],
        "responsibilities_removed": [
            "Storyline generation", "model-generated Coarse IDs", "model-generated timestamps",
            "model-generated Fine/audio IDs", "model-generated coverage bookkeeping",
            "entities/actions/locations duplicate fields", "branch selection and hard-pruning instructions",
        ],
        "historical_map_is_not_an_accuracy_target": True,
    }


def _review_html(path: Path, index: dict[str, Any], semantic_map: dict[str, Any], historical: dict[str, Any]) -> None:
    timeline = "".join(
        f"<tr><td>{i}</td><td><code>{html.escape(row['medium_id'])}</code></td><td>{row['start_sec']:.3f}–{row['end_sec']:.3f}</td><td>{html.escape(row['qwen_caption'])}</td></tr>"
        for i, row in enumerate(index["medium_nodes"])
    )
    sections = []
    for row in semantic_map["coarse_regions"]:
        captions = "".join(f"<li><code>{html.escape(item['medium_id'])}</code>: {html.escape(item['caption'])}</li>" for item in row["exact_source_captions"])
        sections.append(
            f"<section><h2>{row['coarse_id']} · {row['start_sec']:.3f}–{row['end_sec']:.3f}s</h2>"
            f"<p><b>Mediums:</b> {html.escape(', '.join(row['source_medium_ids']))}</p>"
            f"<p><b>Navigation summary:</b> {html.escape(row['navigation_summary'])}</p>"
            f"<p><b>Uncertainty:</b> {html.escape(str(row['uncertainty_notes']))}</p>"
            f"<p><b>Audio IDs (deterministic overlap):</b> {html.escape(str(row['audio_ids']))}</p>"
            f"<details open><summary>Exact canonical captions</summary><ol>{captions}</ol></details>"
            "<p>Coherent phase? ____ Important transitions preserved? ____ Unsupported invention? ____ Useful navigation? ____ Notes: ____</p></section>"
        )
    historical_rows = "".join(
        f"<tr><td>{html.escape(row['coarse_id'])}</td><td>{row['start_sec']:.3f}–{row['end_sec']:.3f}</td><td>{html.escape(row.get('coarse_summary',''))}</td></tr>"
        for row in historical["coarse_nodes"]
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>R3-v2 semantic map review</title>
<style>body{{font:14px system-ui;margin:24px;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:7px;vertical-align:top}}th{{background:#eef}}section{{border-top:3px solid #345;padding:12px 0}}code{{font-weight:bold}}</style>
<h1>R3-v2 global caption-semantic map manual review</h1>
<p>This is a Planner navigation map, not direct evidence, reviewed visual confirmation, or a hard candidate gate.</p>
<h2>Full ordered 30-caption input</h2><table><tr><th>Index</th><th>Medium</th><th>Interval</th><th>Exact caption</th></tr>{timeline}</table>
<h1>New reconstructed Coarse phases</h1>{''.join(sections)}
<h1>Historical Organizer reference only</h1><p>Different grouping is not a failure.</p><table><tr><th>Coarse</th><th>Interval</th><th>Historical summary</th></tr>{historical_rows}</table>
"""
    path.write_text(document, encoding="utf-8", newline="\n")


def prepare(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = load_json(config_path)
    canonical_path = root / config["canonical_index"]
    historical_path = root / config["historical_organized_index"]
    protected_rel = [
        config["canonical_index"], config["historical_organized_index"], config["historical_prompt"],
        config["historical_schema"], config["historical_provenance"], config["planner_consumer"],
        "outputs/experiments/egopolice_r1_structural_novelty_map_v1/coarse_structural_map.json",
    ]
    protected = {item: sha256_file(root / item) for item in protected_rel}
    if protected[config["canonical_index"]] != config["canonical_index_sha256"]:
        raise RuntimeError("canonical_index_hash_mismatch")
    index = load_json(canonical_path)
    historical = load_json(historical_path)
    payload = build_payload(index)
    captions = [row["qwen_caption"] for row in index["medium_nodes"]]
    payload_captions = [row["caption"] for row in payload["ordered_medium_caption_timeline"]]
    caption_hash = hashlib.sha256(canonical_bytes(captions)).hexdigest()
    payload_hash = hashlib.sha256(canonical_bytes(payload_captions)).hexdigest()
    checks = {
        "canonical_hash_matches": protected[config["canonical_index"]] == config["canonical_index_sha256"],
        "medium_count_30": len(index["medium_nodes"]) == 30,
        "fine_count_88": len(index["fine_nodes"]) == 88,
        "audio_count_107": len(index["audio_nodes"]) == 107,
        "caption_count_30": len(captions) == 30,
        "captions_byte_identical_in_payload": caption_hash == payload_hash,
        "model_family_matches_historical": config["model"] == load_json(root / config["historical_provenance"])["organizer"]["model"],
        "temperature_matches_historical": float(config["temperature"]) == 0.0,
        "payload_leakage_zero": not payload_leakage(payload),
        "storyline_disabled": config["storyline_enabled"] is False,
        "hard_filtering_disabled": config["hard_filtering_allowed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"preflight_failed:{[key for key, value in checks.items() if not value]}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input_manifest.json", {
        "canonical_index": config["canonical_index"],
        "canonical_index_sha256": protected[config["canonical_index"]],
        "historical_organized_index": config["historical_organized_index"],
        "historical_organized_index_sha256": protected[config["historical_organized_index"]],
        "video": index["video"], "medium_count": 30, "fine_count": 88, "audio_count": 107,
        "caption_array_sha256": caption_hash, "model": config["model"],
        "temperature": config["temperature"], "max_tokens": config["max_tokens"],
        "max_schema_repairs": config["max_schema_repairs"],
    })
    write_json(output / "historical_organizer_semantic_audit.json", _historical_audit(root, config, historical))
    write_json(output / "caption_integrity_report.json", {
        "status": "passed", "caption_count": 30, "canonical_caption_array_sha256": caption_hash,
        "payload_caption_array_sha256": payload_hash, "byte_identical": True,
        "detector_tracking_payload_leakage": [], "historical_summary_payload_leakage": [],
        "question_or_answer_payload_leakage": [], "asr_grouping_payload_count": 0,
    })
    write_json(output / "organizer_input_payload.json", payload)
    return {"config": config, "index": index, "historical": historical, "payload": payload, "protected": protected, "checks": checks}


def run(root: Path, config_path: Path, output: Path, api_key: str | None) -> dict[str, Any]:
    prepared = prepare(root, config_path, output)
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY_missing_before_model_call")
    config, index, historical, payload = prepared["config"], prepared["index"], prepared["historical"], prepared["payload"]
    attempts: list[dict[str, Any]] = []
    raw, usage = _call(api_key, config, payload)
    try:
        parsed, parse_error = json.loads(raw), None
    except Exception as error:
        parsed, parse_error = None, f"{type(error).__name__}:{error}"
    validation = validate_grouping(parsed) if parsed is not None else {"valid": False, "errors": ["json_parse_failed"], "group_count": 0}
    attempts.append({"attempt": 1, "raw_response": raw, "usage": usage, "parse_error": parse_error, "validation": validation, "schema_only_repair": False})
    if not validation["valid"] and int(config["max_schema_repairs"]) == 1:
        repair_payload = {
            "original_input": payload,
            "invalid_output": raw,
            "schema_errors": validation["errors"],
            "instruction": "Return the same semantic grouping content corrected only to satisfy the exact output schema and contiguous 0..29 coverage. Do not reconsider or change semantic grouping decisions except where required to express the original decision validly.",
        }
        repair_system = SYSTEM_PROMPT + " This is the one allowed schema-only repair. Preserve the original semantic decisions; repair only syntax, schema, indices, ordering, and coverage."
        raw, repair_usage = _call(api_key, config, repair_payload, repair_system)
        try:
            parsed, parse_error = json.loads(raw), None
        except Exception as error:
            parsed, parse_error = None, f"{type(error).__name__}:{error}"
        validation = validate_grouping(parsed) if parsed is not None else {"valid": False, "errors": ["json_parse_failed"], "group_count": 0}
        attempts.append({"attempt": 2, "raw_response": raw, "usage": repair_usage, "parse_error": parse_error, "validation": validation, "schema_only_repair": True})
    write_json(output / "organizer_raw_response.json", {"attempts": attempts, "selected_attempt": len(attempts)})
    write_json(output / "semantic_grouping.json", parsed if parsed is not None else {"groups": []})
    write_json(output / "grouping_validation_report.json", validation)
    if not validation["valid"]:
        raise RuntimeError(f"organizer_output_invalid:{validation['errors']}")
    reconstruction_started = time.perf_counter()
    semantic_map = reconstruct(index, parsed)
    reconstruction_latency = time.perf_counter() - reconstruction_started
    reconstruction_validation = validate_reconstruction(index, semantic_map)
    write_json(output / "deterministic_coarse_reconstruction.json", semantic_map)
    write_json(output / "audio_overlap_audit.json", {
        "status": "passed" if reconstruction_validation["valid"] else "failed",
        "policy": "half-open [start_sec,end_sec)", "model_selected_audio_ids": 0,
        "coarse_audio": [{"coarse_id": row["coarse_id"], "interval": [row["start_sec"], row["end_sec"]], "audio_ids": row["audio_ids"]} for row in semantic_map["coarse_regions"]],
    })
    write_json(output / "r3_v2_semantic_map.json", semantic_map)
    compatibility_started = time.perf_counter()
    compatibility = planner_view(semantic_map)
    planner_validation = validate_planner_view(compatibility, index)
    compatibility_latency = time.perf_counter() - compatibility_started
    write_json(output / "planner_compatibility_view.json", compatibility)
    write_json(output / "planner_compatibility_report.json", {
        **planner_validation,
        "frozen_planner_modified": False, "planner_model_calls": 0,
        "coarse_summary_adapter_derived": True, "all_mediums_later_eligible": True,
        "suggested_coarse_are_non_filtering_hints": True, "map_text_not_sufficiency_evidence": True,
    })
    write_json(output / "semantic_review_checklist.json", {
        "status": "pending_manual_review", "review_path": str(output / "review.html"),
        "questions": [
            "Are Coarse regions coherent semantic event phases?", "Are action and interaction changes preserved?",
            "Are meaningful incident phases distinguishable?", "Are unrelated phases merged?",
            "Is the map excessively fragmented?", "Are summaries grounded in exact source captions?",
            "Does any summary invent unsupported semantics?", "Can a Planner understand the timeline?",
            "Is the map useful without Storyline?", "Is it clearly navigation rather than final evidence?",
        ],
    })
    _review_html(output / "review.html", index, semantic_map, historical)
    protected_after = {item: sha256_file(root / item) for item in prepared["protected"]}
    protected_unchanged = prepared["protected"] == protected_after
    write_json(output / "protected_artifact_hash_audit.json", {
        "status": "passed" if protected_unchanged else "failed", "before": prepared["protected"],
        "after": protected_after, "unchanged": protected_unchanged,
    })
    total_usage = {
        "calls": len(attempts),
        "input_tokens": sum(item["usage"]["input_tokens"] for item in attempts),
        "output_tokens": sum(item["usage"]["output_tokens"] for item in attempts),
        "latency_sec": sum(item["usage"]["latency_sec"] for item in attempts),
        "schema_repair_calls": len(attempts) - 1,
    }
    historical_cost = _historical_audit(root, config, historical)["historical_attempts"]
    write_json(output / "cost_accounting.json", {
        "historical_organizer_reference_only": {
            "calls": len(historical_cost), "input_tokens": sum(item["input_tokens"] for item in historical_cost),
            "output_tokens": sum(item["output_tokens"] for item in historical_cost), "latency_sec": sum(item["latency_sec"] for item in historical_cost),
        },
        "new_organizer": total_usage,
        "deterministic_reconstruction_latency_sec": reconstruction_latency,
        "planner_compatibility_validation_latency_sec": compatibility_latency,
        "candidate_runtime": {**total_usage, "deterministic_processing_latency_sec": reconstruction_latency + compatibility_latency},
        "other_model_api_calls": 0, "estimated_cost_usd": None, "cost_note": "No pricing table configured; cost not fabricated.",
    })
    statuses = {
        "source_validation": "passed" if all(prepared["checks"].values()) and protected_unchanged else "failed",
        "caption_integrity_validation": "passed",
        "organizer_schema_validation": "passed" if validation["valid"] else "failed",
        "deterministic_reconstruction_validation": "passed" if reconstruction_validation["valid"] else "failed",
        "downstream_compatibility_validation": "passed" if planner_validation["valid"] else "failed",
        "semantic_acceptance": "pending_manual_review",
    }
    structural_pass = all(value == "passed" for key, value in statuses.items() if key != "semantic_acceptance")
    statuses["overall_validation"] = "pending_manual_semantic_review" if structural_pass else "failed_structural_validation"
    recommendation = "ready_for_r3_v2_semantic_map_freeze_review" if structural_pass else "organizer_output_invalid"
    durations = [row["duration_sec"] for row in semantic_map["coarse_regions"]]
    validation_report = {
        **statuses, "recommendation": recommendation,
        "coarse_count": len(semantic_map["coarse_regions"]),
        "medium_count_per_coarse": [len(row["source_medium_ids"]) for row in semantic_map["coarse_regions"]],
        "duration_sec_per_coarse": durations, "longest_coarse_duration_sec": max(durations),
        "storyline_count": 0, "hard_filtering_allowed": False,
        "model_calls": len(attempts), "schema_repair_calls": len(attempts) - 1,
        "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0,
        "temporal_review_calls": 0, "final_gemini_calls": 0, "qa_calls": 0,
        "protected_sources_unchanged": protected_unchanged,
        "semantic_review_path": str(output / "review.html"),
    }
    write_json(output / "validation_report.json", validation_report)
    report_lines = [
        "# R3-v2 Coarse semantic Organizer canary v1", "",
        f"- Structural validation: `{'passed' if structural_pass else 'failed'}`",
        "- Semantic acceptance: `pending_manual_review`",
        f"- Overall validation: `{statuses['overall_validation']}`",
        f"- Recommendation: `{recommendation}`", "",
        "## Contract", "",
        "One global, question-agnostic Organizer call used all 30 byte-identical canonical repaired Qwen captions. ASR, detector/tracking summaries, historical summaries, questions, and answer options were absent from grouping input.", "",
        f"- Model: `{config['model']}`; temperature: `{config['temperature']}`; max tokens: `{config['max_tokens']}`",
        f"- Calls: `{len(attempts)}`; schema repairs: `{len(attempts)-1}`",
        f"- Tokens: `{total_usage['input_tokens']}` input / `{total_usage['output_tokens']}` output",
        f"- Model latency: `{total_usage['latency_sec']:.3f}s`",
        f"- Coarse phases: `{len(semantic_map['coarse_regions'])}`; Medium membership: `{[len(row['source_medium_ids']) for row in semantic_map['coarse_regions']]}`",
        "- Storyline: `0`; hard filtering allowed: `false`", "",
        "## Safety and scope", "",
        "Planner compatibility was validated without a Planner call. Planner, Retrieval, Sufficiency, temporal review, Final Gemini, QA, HourVideo, and R1 replay calls were all zero. Historical and canonical artifacts were hash-verified unchanged.", "",
        "Manual semantic review is required in `review.html` before any formal semantic-map freeze.",
    ]
    (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    return validation_report
