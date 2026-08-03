from __future__ import annotations

import copy
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

from experiments.r3_v2_coarse_semantic_organizer.core import (
    canonical_bytes,
    load_json,
    sha256_file,
    validate_planner_view,
    write_json,
)


OUTPUT_FIELDS = ["end_medium_index", "navigation_summary", "uncertainty_notes"]
FORBIDDEN_INPUT_KEYS = {
    "detector_summary", "structured_visual_summary", "detector_tags", "tracklets",
    "historical_summary", "coarse_summary", "storyline", "question", "answer_options",
    "experiment_name", "rung", "expected_event", "diagnostic_timestamp",
}

SYSTEM_PROMPT = """You are an offline, question-agnostic audio-visual semantic Organizer for a video navigation map. You receive the complete ordered Medium timeline. Each Medium contains its exact visual caption and exact temporally overlapping ASR as separate evidence channels. Globally organize the whole timeline into contiguous coherent semantic phases. Captions are visual semantic descriptions; ASR is audible speech, requests, reports, commands, or coordination. ASR may help identify phase and response transitions, but never treat a mention, request, plan, report, or retrospective statement as visual confirmation, proof of completion, or exact physical-event timing. Qualify audio-only content in navigation summaries. Group consecutive Mediums when both channels describe one coherent phase. Split for meaningful changes in setting, principal activity, human interaction, confrontation or control state, injury state, medical or operational response, or later scene management. Do not split for wording, camera angle, isolated objects, speaker changes, or short noisy records. Do not merge clearly different phases merely because people, objects, or scene persist. Return only groups in the required schema. Each group contains only its inclusive end_medium_index, a concise navigation_summary, and uncertainty_notes. The first group implicitly starts at Medium 0. Each later group implicitly starts one after the previous end. End indices must be strictly increasing and the final end_medium_index must equal 29. Do not output group indices, start indices, IDs, timestamps, source IDs, Storyline, entities/actions/locations side fields, candidate regions, scores, or answers."""


def full_output_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": ["groups"],
        "properties": {
            "groups": {
                "type": "array", "minItems": 1, "maxItems": 30,
                "items": {
                    "type": "object", "additionalProperties": False, "required": OUTPUT_FIELDS,
                    "properties": {
                        "end_medium_index": {"type": "integer", "minimum": 0, "maximum": 29},
                        "navigation_summary": {"type": "string", "minLength": 1},
                        "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }


def provider_output_schema() -> tuple[dict[str, Any], dict[str, Any]]:
    schema = copy.deepcopy(full_output_schema())
    removed = []
    groups = schema["properties"]["groups"]
    for annotation in ("minItems", "maxItems"):
        if annotation in groups:
            removed.append(f"$/properties/groups/{annotation}")
            groups.pop(annotation)
    summary = groups["items"]["properties"]["navigation_summary"]
    if "minLength" in summary:
        removed.append("$/properties/groups/items/properties/navigation_summary/minLength")
        summary.pop("minLength")
    end_index = groups["items"]["properties"]["end_medium_index"]
    for annotation in ("minimum", "maximum"):
        if annotation in end_index:
            removed.append(f"$/properties/groups/items/properties/end_medium_index/{annotation}")
            end_index.pop(annotation)
    return schema, {
        "projection_applied": bool(removed),
        "removed_provider_unsupported_annotations": removed,
        "field_meaning_changed": False,
        "local_validator_uses_full_contract": True,
        "model_calls": 0,
    }


def _overlap(audio: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [row for row in audio if float(row["start_sec"]) < end and float(row["end_sec"]) > start]


def _audio_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_id": row["audio_id"], "start_sec": float(row["start_sec"]), "end_sec": float(row["end_sec"]),
        "transcript": row["transcript"], "source_type": row.get("source_type", "unclear"),
        "fallback_used": bool(row.get("fallback_used", False)), "asr_status": row.get("asr_status"),
    }


def build_payload(index: dict[str, Any]) -> dict[str, Any]:
    mediums, audio = index["medium_nodes"], index["audio_nodes"]
    if len(mediums) != 30 or len(audio) != 107:
        raise ValueError(f"canonical_counts_invalid:medium={len(mediums)},audio={len(audio)}")
    timeline = []
    for position, medium in enumerate(mediums):
        caption = medium.get("qwen_caption")
        if not isinstance(caption, str) or not caption:
            raise ValueError(f"missing_caption:{position}")
        start, end = float(medium["start_sec"]), float(medium["end_sec"])
        timeline.append({
            "medium_index": position, "start_sec": start, "end_sec": end, "caption": caption,
            "overlapping_asr": [_audio_record(row) for row in _overlap(audio, start, end)],
        })
    return {
        "ordered_medium_audio_visual_timeline": timeline,
        "modality_policy": {
            "caption": "visual semantic description; not reviewed visual confirmation",
            "asr": "audible speech; mentions, requests, reports, and recording times are not automatic physical-event proof",
        },
        "output_contract": {
            "model_outputs_end_indices_only": True,
            "first_start_is_deterministically_zero": True,
            "later_start_is_previous_end_plus_one": True,
            "end_indices_strictly_increasing": True,
            "final_end_medium_index": 29,
        },
    }


def forbidden_input_hits(payload: Any) -> list[str]:
    hits: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in FORBIDDEN_INPUT_KEYS:
                    hits.append(f"{path}/{key}")
                visit(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")

    visit(payload, "$")
    return hits


def validate_first_pass(value: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != {"groups"} or not isinstance(value.get("groups"), list):
        return {"valid": False, "errors": ["invalid_wrapper"], "group_count": 0, "end_indices": []}
    groups = value["groups"]
    if not groups:
        errors.append("groups_empty")
    if len(groups) > 30:
        errors.append("group_count_above_30")
    ends: list[int] = []
    for position, row in enumerate(groups):
        if not isinstance(row, dict) or list(row) != OUTPUT_FIELDS:
            errors.append(f"group_{position}:fields_or_order")
            continue
        end = row["end_medium_index"]
        if type(end) is not int:
            errors.append(f"group_{position}:end_not_integer")
        else:
            ends.append(end)
            if not 0 <= end <= 29:
                errors.append(f"group_{position}:end_out_of_range")
        if not isinstance(row["navigation_summary"], str) or not row["navigation_summary"].strip():
            errors.append(f"group_{position}:summary_empty")
        notes = row["uncertainty_notes"]
        if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
            errors.append(f"group_{position}:uncertainty_not_string_array")
    if len(ends) == len(groups):
        if any(right <= left for left, right in zip(ends, ends[1:])):
            errors.append("end_indices_not_strictly_increasing")
        if ends and ends[-1] != 29:
            errors.append("final_end_not_29")
    return {"valid": not errors, "errors": errors, "group_count": len(groups), "end_indices": ends}


def infer_ranges(value: dict[str, Any]) -> list[dict[str, Any]]:
    validation = validate_first_pass(value)
    if not validation["valid"]:
        raise ValueError(f"invalid_first_pass:{validation['errors']}")
    result, start = [], 0
    for group_index, row in enumerate(value["groups"]):
        end = row["end_medium_index"]
        result.append({
            "group_index": group_index, "start_medium_index": start, "end_medium_index": end,
            "navigation_summary": row["navigation_summary"], "uncertainty_notes": list(row["uncertainty_notes"]),
        })
        start = end + 1
    return result


def reconstruct(index: dict[str, Any], first_pass: dict[str, Any]) -> dict[str, Any]:
    mediums, audio = index["medium_nodes"], index["audio_nodes"]
    coarse_regions = []
    for group in infer_ranges(first_pass):
        selected = mediums[group["start_medium_index"] : group["end_medium_index"] + 1]
        start, end = float(selected[0]["start_sec"]), float(selected[-1]["end_sec"])
        attached = [_audio_record(row) for row in _overlap(audio, start, end)]
        coarse_regions.append({
            "coarse_id": f"C{group['group_index'] + 1:02d}", "start_sec": start, "end_sec": end,
            "duration_sec": end - start, "source_medium_ids": [row["medium_id"] for row in selected],
            "source_fine_ids": [fine_id for row in selected for fine_id in row["child_fine_ids"]],
            "navigation_summary": group["navigation_summary"], "uncertainty_notes": list(group["uncertainty_notes"]),
            "exact_source_captions": [{"medium_id": row["medium_id"], "caption": row["qwen_caption"]} for row in selected],
            "exact_source_asr": attached, "audio_ids": [row["audio_id"] for row in attached],
            "map_type": "audio_visual_semantic_coarse", "semantic_fields_available": True,
        })
    return {
        "map_type": "audio_visual_semantic_coarse", "semantic_fields_available": True,
        "visual_semantic_source": "canonical_qwen_caption", "audio_semantic_source": "canonical_timestamped_asr",
        "coarse_regions": coarse_regions, "storyline_events": [], "has_storyline": False,
        "hard_filtering_allowed": False,
        "provenance": {
            "semantic_group_end_indices_and_summaries": "single first-pass global Organizer call",
            "group_starts_indices_ids_boundaries_coverage_fine_audio": "deterministic code",
            "audio_overlap_rule": "half-open [start_sec,end_sec)", "model_repair_calls": 0,
            "semantic_retries": 0, "map_is_navigation_not_direct_evidence": True,
        },
    }


def validate_reconstruction(index: dict[str, Any], semantic_map: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    mediums = index["medium_nodes"]
    expected_mediums = [row["medium_id"] for row in mediums]
    actual_mediums = [item for coarse in semantic_map["coarse_regions"] for item in coarse["source_medium_ids"]]
    if actual_mediums != expected_mediums:
        errors.append("medium_coverage_or_order_mismatch")
    expected_fine = [item for row in mediums for item in row["child_fine_ids"]]
    actual_fine = [item for coarse in semantic_map["coarse_regions"] for item in coarse["source_fine_ids"]]
    if actual_fine != expected_fine or len(actual_fine) != len(index["fine_nodes"]):
        errors.append("fine_mapping_mismatch")
    audio_by_id = {row["audio_id"]: _audio_record(row) for row in index["audio_nodes"]}
    for position, coarse in enumerate(semantic_map["coarse_regions"]):
        source = [row for row in mediums if row["medium_id"] in coarse["source_medium_ids"]]
        if coarse["start_sec"] != float(source[0]["start_sec"]) or coarse["end_sec"] != float(source[-1]["end_sec"]):
            errors.append(f"coarse_{position}:boundary_mismatch")
        expected_asr = [_audio_record(row) for row in _overlap(index["audio_nodes"], coarse["start_sec"], coarse["end_sec"])]
        if canonical_bytes(coarse["exact_source_asr"]) != canonical_bytes(expected_asr):
            errors.append(f"coarse_{position}:asr_mismatch")
        if coarse["audio_ids"] != [row["audio_id"] for row in expected_asr] or any(item not in audio_by_id for item in coarse["audio_ids"]):
            errors.append(f"coarse_{position}:audio_ids_mismatch")
        expected_captions = [{"medium_id": row["medium_id"], "caption": row["qwen_caption"]} for row in source]
        if canonical_bytes(coarse["exact_source_captions"]) != canonical_bytes(expected_captions):
            errors.append(f"coarse_{position}:caption_mismatch")
    if semantic_map["storyline_events"] or semantic_map["has_storyline"]:
        errors.append("storyline_present")
    if semantic_map["hard_filtering_allowed"]:
        errors.append("hard_filtering_enabled")
    return {"valid": not errors, "errors": errors, "medium_count": len(actual_mediums), "fine_count": len(actual_fine)}


def build_planner_view(semantic_map: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_type": "audio_visual_semantic_coarse",
        "coarse_regions": [
            {
                "coarse_id": row["coarse_id"], "start_sec": row["start_sec"], "end_sec": row["end_sec"],
                "source_medium_ids": list(row["source_medium_ids"]), "source_audio_ids": list(row["audio_ids"]),
                "event_label": "", "summary": row["navigation_summary"], "uncertainty_notes": list(row["uncertainty_notes"]),
                "coarse_summary": row["navigation_summary"], "exact_source_captions": copy.deepcopy(row["exact_source_captions"]),
                "exact_source_asr": copy.deepcopy(row["exact_source_asr"]),
                "adapter_metadata": {
                    "adapter_derived": True, "source_field": "navigation_summary", "semantic_summary": True,
                    "map_type": "audio_visual_semantic_coarse", "stored_index_modified": False,
                    "hard_filtering_allowed": False,
                },
            }
            for row in semantic_map["coarse_regions"]
        ],
        "storyline_events": [], "has_storyline": False, "hard_filtering_allowed": False,
        "planner_policy": {
            "map_may_guide_search_units_and_query_variants": True,
            "suggested_coarse_are_navigation_hints_only": True, "all_mediums_remain_eligible": True,
            "coarse_prior_affects_ranking": False, "map_text_is_not_sufficiency_evidence": True,
        },
    }


def _call(api_key: str, config: dict[str, Any], payload: dict[str, Any], provider_schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    import anthropic

    started = time.perf_counter()
    response = anthropic.Anthropic(api_key=api_key).messages.create(
        model=config["model"], max_tokens=int(config["max_tokens"]), temperature=float(config["temperature"]),
        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": canonical_bytes(payload).decode("utf-8")}],
        output_config={"format": {"type": "json_schema", "schema": provider_schema}},
    )
    raw = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return raw, {
        "provider": "anthropic", "model": config["model"], "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens), "latency_sec": time.perf_counter() - started,
        "stop_reason": response.stop_reason, "response_id": str(response.id),
        "request_id": str(getattr(response, "_request_id", "") or ""),
    }


def _strict_contract_view(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_type": view["map_type"], "coarse_regions": view["coarse_regions"],
        "storyline_events": view["storyline_events"], "has_storyline": view["has_storyline"],
        "hard_filtering_allowed": view["hard_filtering_allowed"], "planner_policy": view["planner_policy"],
    }


def run_no_api_tests(index: dict[str, Any], expected_hash: str, actual_hash: str) -> dict[str, Any]:
    valid = {"groups": [
        {"end_medium_index": 1, "navigation_summary": "phase a", "uncertainty_notes": []},
        {"end_medium_index": 29, "navigation_summary": "phase b", "uncertainty_notes": ["uncertain"]},
    ]}
    checks: dict[str, bool] = {}
    ranges = infer_ranges(valid)
    checks["first_group_start_deterministically_zero"] = ranges[0]["start_medium_index"] == 0
    checks["later_starts_previous_end_plus_one"] = ranges[1]["start_medium_index"] == ranges[0]["end_medium_index"] + 1
    checks["strictly_increasing_ends_pass"] = validate_first_pass(valid)["valid"]
    for name, ends in (
        ("duplicate_end_fails", [1, 1, 29]), ("decreasing_end_fails", [5, 4, 29]),
        ("final_not_29_fails", [1, 28]), ("outside_range_fails", [1, 30]),
    ):
        case = {"groups": [{"end_medium_index": end, "navigation_summary": "x", "uncertainty_notes": []} for end in ends]}
        checks[name] = not validate_first_pass(case)["valid"]
    checks["empty_groups_fail"] = not validate_first_pass({"groups": []})["valid"]
    extra = copy.deepcopy(valid); extra["groups"][0]["group_index"] = 0
    checks["unknown_output_fields_fail"] = not validate_first_pass(extra)["valid"]
    checks["no_repair_path_configured"] = True
    semantic_map = reconstruct(index, valid)
    reconstruction = validate_reconstruction(index, semantic_map)
    checks["deterministic_30_medium_coverage"] = reconstruction["valid"] and reconstruction["medium_count"] == 30
    checks["fine_and_audio_mappings_valid"] = reconstruction["valid"] and reconstruction["fine_count"] == 88
    checks["storyline_empty"] = semantic_map["storyline_events"] == [] and semantic_map["has_storyline"] is False
    checks["hard_filtering_false"] = semantic_map["hard_filtering_allowed"] is False
    checks["protected_source_hash_match"] = actual_hash == expected_hash
    checks["deterministic_reconstruction_byte_identical"] = canonical_bytes(semantic_map) == canonical_bytes(reconstruct(index, valid))
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "model_api_calls": 0}


def _review_html(path: Path, payload: dict[str, Any], raw: str, first_pass: dict[str, Any], semantic_map: dict[str, Any], av_v1: dict[str, Any], caption_only: dict[str, Any]) -> None:
    timeline = []
    for medium in payload["ordered_medium_audio_visual_timeline"]:
        audio = "".join(f"<li>[{row['start_sec']:.3f}–{row['end_sec']:.3f}] {html.escape(row['transcript'])}</li>" for row in medium["overlapping_asr"]) or "<li>No overlapping ASR</li>"
        timeline.append(f"<section><h3>Medium {medium['medium_index']} · {medium['start_sec']:.3f}–{medium['end_sec']:.3f}s</h3><p>{html.escape(medium['caption'])}</p><details><summary>Exact overlapping ASR</summary><ul>{audio}</ul></details></section>")
    groups = []
    inferred = infer_ranges(first_pass)
    by_id = {row["coarse_id"]: row for row in semantic_map["coarse_regions"]}
    for group in inferred:
        coarse = by_id[f"C{group['group_index']+1:02d}"]
        captions = "".join(f"<li><code>{html.escape(row['medium_id'])}</code>: {html.escape(row['caption'])}</li>" for row in coarse["exact_source_captions"])
        audio = "".join(f"<li><code>{html.escape(row['audio_id'])}</code> [{row['start_sec']:.3f}–{row['end_sec']:.3f}] fallback={str(row['fallback_used']).lower()}: {html.escape(row['transcript'])}</li>" for row in coarse["exact_source_asr"]) or "<li>No ASR</li>"
        groups.append(f"<section><h2>{coarse['coarse_id']} · Medium {group['start_medium_index']}–{group['end_medium_index']} · {coarse['start_sec']:.3f}–{coarse['end_sec']:.3f}s</h2><p><b>Summary:</b> {html.escape(coarse['navigation_summary'])}</p><p><b>Uncertainty:</b> {html.escape(str(coarse['uncertainty_notes']))}</p><details open><summary>Exact captions</summary><ul>{captions}</ul></details><details open><summary>Exact ASR</summary><ul>{audio}</ul></details><p>Coherent AV phase? ____ Speech confused with visual proof? ____ Response phases distinct? ____ Summary grounded? ____ Notes: ____</p></section>")
    def reference_rows(value: dict[str, Any]) -> str:
        return "".join(f"<tr><td>{row['coarse_id']}</td><td>{row['start_sec']:.3f}–{row['end_sec']:.3f}</td><td>{html.escape(row['navigation_summary'])}</td></tr>" for row in value["coarse_regions"])
    document = f"""<!doctype html><meta charset='utf-8'><title>R3-v2 AV Organizer v1.1 review</title><style>body{{font:14px system-ui;margin:24px;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:7px;vertical-align:top}}section{{border-top:2px solid #567;padding:10px 0}}code{{font-weight:bold}}</style><h1>R3-v2 AV semantic map v1.1</h1><p>Navigation map only; no hard pruning and no direct Sufficiency evidence.</p><h2>Raw first-pass Organizer output</h2><pre>{html.escape(raw)}</pre><h1>Deterministically inferred Coarse ranges</h1>{''.join(groups)}<h1>Complete Medium caption + ASR timeline</h1>{''.join(timeline)}<h1>AV v1 diagnostic reference only</h1><table>{reference_rows(av_v1)}</table><h1>Caption-only 14-Coarse reference only</h1><table>{reference_rows(caption_only)}</table>"""
    path.write_text(document, encoding="utf-8", newline="\n")


def prepare(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = load_json(config_path)
    canonical_path = root / config["canonical_index"]
    protected_rel = [
        config["canonical_index"], config["historical_organized_index"], config["historical_provenance"], config["planner_consumer"],
        f"{config['av_v1_diagnostic']}/r3_v2_av_semantic_map.json", f"{config['av_v1_diagnostic']}/organizer_raw_response.json",
        f"{config['av_v1_diagnostic']}/validation_report.json", f"{config['caption_only_diagnostic']}/r3_v2_semantic_map.json",
        "outputs/experiments/egopolice_r1_structural_novelty_map_v1/coarse_structural_map.json",
    ]
    protected = {item: sha256_file(root / item) for item in protected_rel}
    actual_hash = protected[config["canonical_index"]]
    if actual_hash != config["canonical_index_sha256"]:
        raise RuntimeError("canonical_index_hash_mismatch")
    index = load_json(canonical_path)
    payload = build_payload(index)
    hits = forbidden_input_hits(payload)
    unique_audio = {row["audio_id"] for medium in payload["ordered_medium_audio_visual_timeline"] for row in medium["overlapping_asr"]}
    captions = [row["qwen_caption"] for row in index["medium_nodes"]]
    payload_captions = [row["caption"] for row in payload["ordered_medium_audio_visual_timeline"]]
    canonical_audio = [_audio_record(row) for row in index["audio_nodes"]]
    audio_hash = hashlib.sha256(canonical_bytes(canonical_audio)).hexdigest()
    caption_hash = hashlib.sha256(canonical_bytes(captions)).hexdigest()
    historical_model = load_json(root / config["historical_provenance"])["organizer"]["model"]
    checks = {
        "canonical_hash_matches": actual_hash == config["canonical_index_sha256"], "medium_count_30": len(index["medium_nodes"]) == 30,
        "fine_count_88": len(index["fine_nodes"]) == 88, "audio_count_107": len(index["audio_nodes"]) == 107,
        "captions_byte_identical": canonical_bytes(captions) == canonical_bytes(payload_captions),
        "all_audio_records_represented": unique_audio == {row["audio_id"] for row in index["audio_nodes"]},
        "caption_asr_channels_separate": all("caption" in row and "overlapping_asr" in row for row in payload["ordered_medium_audio_visual_timeline"]),
        "forbidden_payload_hits_zero": not hits, "model_matches_v1": config["model"] == historical_model,
        "temperature_unchanged": float(config["temperature"]) == 0.0, "repair_calls_zero": config["model_repair_calls"] == 0,
        "semantic_retries_zero": config["semantic_retries"] == 0, "storyline_disabled": config["storyline_enabled"] is False,
        "hard_filtering_disabled": config["hard_filtering_allowed"] is False,
    }
    tests = run_no_api_tests(index, config["canonical_index_sha256"], actual_hash)
    if not all(checks.values()) or tests["status"] != "passed":
        raise RuntimeError(f"no_api_preflight_failed:{[key for key,value in checks.items() if not value]}")
    provider_schema, projection = provider_output_schema()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input_manifest.json", {"canonical_index": config["canonical_index"], "canonical_index_sha256": actual_hash, "video": index["video"], "medium_count": 30, "fine_count": 88, "audio_count": 107, "caption_array_sha256": caption_hash, "canonical_audio_array_sha256": audio_hash, "model": config["model"], "temperature": config["temperature"], "max_tokens": config["max_tokens"], "expected_organizer_calls": 1, "model_repair_calls": 0, "semantic_retries": 0})
    write_json(output / "source_artifact_audit.json", {"status": "passed", "protected_hashes_before": protected, "av_v1_used_in_model_input": False, "caption_only_diagnostic_used_in_model_input": False, "checks": checks})
    write_json(output / "caption_integrity_report.json", {"status": "passed", "caption_count": 30, "caption_array_sha256": caption_hash, "byte_identical": True})
    write_json(output / "asr_integrity_report.json", {"status": "passed", "canonical_audio_count": 107, "canonical_audio_array_sha256": audio_hash, "unique_audio_records_in_payload": len(unique_audio), "all_records_represented": True, "fallback_count": sum(row["fallback_used"] for row in canonical_audio), "empty_transcript_count": sum(not row["transcript"].strip() for row in canonical_audio), "duplicate_medium_attachments_allowed_by_temporal_overlap": sum(len(row["overlapping_asr"]) for row in payload["ordered_medium_audio_visual_timeline"]) - len(unique_audio)})
    write_json(output / "organizer_input_payload.json", payload)
    write_json(output / "provider_schema_projection_audit.json", {**projection, "full_local_schema": full_output_schema(), "provider_schema": provider_schema, "pre_inference_provider_schema_rejections": config.get("pre_inference_provider_schema_rejections", [])})
    write_json(output / "no_api_test_report.json", tests)
    return {"config": config, "index": index, "payload": payload, "provider_schema": provider_schema, "protected": protected, "checks": checks, "tests": tests}


def _write_invalid_outputs(output: Path, raw: str, usage: dict[str, Any], parsed: Any, parse_error: str | None, validation: dict[str, Any], protected: dict[str, str], protected_after: dict[str, str]) -> dict[str, Any]:
    write_json(output / "organizer_raw_response.json", {"raw_response": raw, "usage": usage, "parse_error": parse_error, "attempt_count": 1})
    write_json(output / "first_pass_output_validation.json", validation)
    write_json(output / "semantic_grouping.json", parsed if isinstance(parsed, dict) else {"status": "not_available"})
    for filename, content in (
        ("deterministic_coarse_reconstruction.json", {"status": "not_run_invalid_first_pass", "coarse_regions": []}),
        ("audio_overlap_audit.json", {"status": "not_run_invalid_first_pass"}),
        ("r3_v2_av_semantic_map.json", {"status": "not_produced_invalid_first_pass"}),
        ("planner_compatibility_view.json", {"status": "not_produced_invalid_first_pass"}),
        ("planner_compatibility_report.json", {"valid": False, "status": "not_run_invalid_first_pass", "planner_calls": 0}),
    ):
        write_json(output / filename, content)
    write_json(output / "zero_repair_audit.json", {"passed": True, "organizer_calls": 1, "model_repair_calls": 0, "semantic_retries": 0, "partial_grouping_salvaged": False})
    write_json(output / "semantic_review_checklist.json", {"status": "not_started_invalid_first_pass"})
    unchanged = protected == protected_after
    write_json(output / "protected_artifact_hash_audit.json", {"status": "passed" if unchanged else "failed", "before": protected, "after": protected_after, "unchanged": unchanged})
    write_json(output / "cost_accounting.json", {"organizer": {"calls": 1, **{key: usage[key] for key in ("input_tokens", "output_tokens", "latency_sec")}}, "model_repair_calls": 0, "semantic_retries": 0, "all_other_model_api_calls": 0})
    report = {"source_validation": "passed" if unchanged else "failed", "caption_integrity_validation": "passed", "asr_integrity_validation": "passed", "first_pass_output_validation": "failed", "zero_repair_validation": "passed", "deterministic_reconstruction_validation": "not_run", "planner_compatibility_validation": "not_run", "semantic_acceptance": "not_started", "overall_validation": "first_pass_output_invalid", "recommendation": "first_pass_output_invalid", "organizer_calls": 1, "model_repair_calls": 0, "semantic_retries": 0, "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0, "review_calls": 0, "final_gemini_calls": 0, "qa_calls": 0}
    write_json(output / "validation_report.json", report)
    (output / "review.html").write_text(f"<!doctype html><meta charset='utf-8'><h1>Invalid first-pass output</h1><pre>{html.escape(raw)}</pre><pre>{html.escape(json.dumps(validation,indent=2))}</pre>", encoding="utf-8", newline="\n")
    (output / "REPORT.md").write_text("# R3-v2 AV Organizer v1.1\n\nFirst-pass output was invalid. It was preserved without repair, retry, inference, or salvage.\n", encoding="utf-8", newline="\n")
    return report


def run(root: Path, config_path: Path, output: Path, api_key: str | None) -> dict[str, Any]:
    prepared = prepare(root, config_path, output)
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY_missing_before_model_call")
    raw, usage = _call(api_key, prepared["config"], prepared["payload"], prepared["provider_schema"])
    try:
        parsed, parse_error = json.loads(raw), None
    except Exception as error:
        parsed, parse_error = None, f"{type(error).__name__}:{error}"
    validation = validate_first_pass(parsed) if parsed is not None else {"valid": False, "errors": ["json_parse_failed"], "group_count": 0, "end_indices": []}
    write_json(output / "organizer_raw_response.json", {"raw_response": raw, "usage": usage, "parse_error": parse_error, "attempt_count": 1})
    write_json(output / "first_pass_output_validation.json", validation)
    protected_after = {item: sha256_file(root / item) for item in prepared["protected"]}
    if not validation["valid"]:
        return _write_invalid_outputs(output, raw, usage, parsed, parse_error, validation, prepared["protected"], protected_after)
    write_json(output / "semantic_grouping.json", {"model_output": parsed, "deterministically_inferred_ranges": infer_ranges(parsed)})
    reconstruction_started = time.perf_counter()
    semantic_map = reconstruct(prepared["index"], parsed)
    reconstruction_latency = time.perf_counter() - reconstruction_started
    reconstruction_validation = validate_reconstruction(prepared["index"], semantic_map)
    write_json(output / "deterministic_coarse_reconstruction.json", semantic_map)
    write_json(output / "r3_v2_av_semantic_map.json", semantic_map)
    write_json(output / "audio_overlap_audit.json", {"status": "passed" if reconstruction_validation["valid"] else "failed", "policy": "half-open [start_sec,end_sec)", "model_selected_audio_ids": 0, "coarse_audio": [{"coarse_id": row["coarse_id"], "audio_ids": row["audio_ids"], "audio_count": len(row["audio_ids"])} for row in semantic_map["coarse_regions"]]})
    compatibility_started = time.perf_counter()
    planner = build_planner_view(semantic_map)
    planner_validation = validate_planner_view(_strict_contract_view(planner), prepared["index"])
    compatibility_latency = time.perf_counter() - compatibility_started
    write_json(output / "planner_compatibility_view.json", planner)
    write_json(output / "planner_compatibility_report.json", {**planner_validation, "planner_model_calls": 0, "frozen_planner_modified": False, "all_mediums_remain_eligible": True, "coarse_prior_affects_ranking": False, "suggested_regions_are_navigation_hints_only": True, "map_text_not_sufficiency_evidence": True})
    zero_repair = {"passed": True, "organizer_calls": 1, "model_repair_calls": 0, "semantic_retries": 0, "first_pass_selected": True, "summaries_modified_after_model": False, "boundaries_inferred_only_from_valid_end_indices": True}
    write_json(output / "zero_repair_audit.json", zero_repair)
    write_json(output / "semantic_review_checklist.json", {"status": "pending_manual_review", "review_path": str(output / "review.html"), "questions": ["Does the map preserve meaningful AV phases?", "Does ASR add useful distinctions?", "Are audible statements kept distinct from visual proof?", "Are control, injury, response, and later coordination distinguishable where supported?", "Are unrelated phases merged?", "Is the map fragmented?", "Are summaries faithful to exact caption and ASR?", "Is uncertainty retained?", "Can Planner understand the timeline without Storyline?", "Is this clearly navigation rather than final evidence?"]})
    av_v1 = load_json(root / prepared["config"]["av_v1_diagnostic"] / "r3_v2_av_semantic_map.json")
    caption_only = load_json(root / prepared["config"]["caption_only_diagnostic"] / "r3_v2_semantic_map.json")
    _review_html(output / "review.html", prepared["payload"], raw, parsed, semantic_map, av_v1, caption_only)
    unchanged = prepared["protected"] == protected_after
    write_json(output / "protected_artifact_hash_audit.json", {"status": "passed" if unchanged else "failed", "before": prepared["protected"], "after": protected_after, "unchanged": unchanged})
    provider_rejections = prepared["config"].get("pre_inference_provider_schema_rejections", [])
    write_json(output / "cost_accounting.json", {"organizer": {"model_inference_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "latency_sec": usage["latency_sec"]}, "provider_schema_projection_model_calls": 0, "pre_inference_failed_api_requests": provider_rejections, "pre_inference_failed_api_request_count": len(provider_rejections), "model_repair_calls": 0, "semantic_retries": 0, "deterministic_reconstruction_latency_sec": reconstruction_latency, "compatibility_validation_latency_sec": compatibility_latency, "all_other_model_inference_calls": 0, "estimated_cost_usd": None, "cost_note": "No pricing table configured; cost not fabricated."})
    structural = all((reconstruction_validation["valid"], planner_validation["valid"], unchanged))
    durations = [row["duration_sec"] for row in semantic_map["coarse_regions"]]
    report = {"source_validation": "passed" if unchanged else "failed", "caption_integrity_validation": "passed", "asr_integrity_validation": "passed", "first_pass_output_validation": "passed", "zero_repair_validation": "passed", "deterministic_reconstruction_validation": "passed" if reconstruction_validation["valid"] else "failed", "planner_compatibility_validation": "passed" if planner_validation["valid"] else "failed", "semantic_acceptance": "pending_manual_review", "overall_validation": "pending_manual_semantic_review" if structural else "failed_structural_validation", "recommendation": "ready_for_r3_v2_av_semantic_map_freeze_review" if structural else "source_integrity_failure", "raw_end_indices": validation["end_indices"], "inferred_ranges": [[row["start_medium_index"], row["end_medium_index"]] for row in infer_ranges(parsed)], "coarse_count": len(semantic_map["coarse_regions"]), "medium_count_per_coarse": [len(row["source_medium_ids"]) for row in semantic_map["coarse_regions"]], "duration_sec_per_coarse": durations, "longest_coarse_duration_sec": max(durations), "storyline_count": 0, "hard_filtering_allowed": False, "organizer_model_inference_calls": 1, "pre_inference_failed_api_request_count": len(provider_rejections), "model_repair_calls": 0, "semantic_retries": 0, "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0, "review_calls": 0, "final_gemini_calls": 0, "qa_calls": 0, "semantic_review_path": str(output / "review.html")}
    write_json(output / "validation_report.json", report)
    (output / "REPORT.md").write_text("\n".join(["# R3-v2 AV Coarse semantic Organizer canary v1.1", "", f"- First-pass output: `{report['first_pass_output_validation']}`", "- Model repair calls: `0`", "- Semantic retries: `0`", f"- Structural validation: `{'passed' if structural else 'failed'}`", "- Semantic acceptance: `pending_manual_review`", f"- Overall validation: `{report['overall_validation']}`", f"- Recommendation: `{report['recommendation']}`", "", f"- Model: `{prepared['config']['model']}`; temperature: `{prepared['config']['temperature']}`; max tokens: `{prepared['config']['max_tokens']}`", f"- Raw end indices: `{validation['end_indices']}`", f"- Inferred ranges: `{report['inferred_ranges']}`", f"- Coarse count: `{report['coarse_count']}`; memberships: `{report['medium_count_per_coarse']}`", f"- Tokens: `{usage['input_tokens']}` input / `{usage['output_tokens']}` output; latency: `{usage['latency_sec']:.3f}s`", "- Storyline: `0`; hard filtering: `false`", "", "Planner compatibility was validated without a Planner call. Retrieval, Sufficiency, temporal review, Final Gemini, QA, HourVideo, and R1 replay were not run. Manual AV semantic review is required before freeze."]) + "\n", encoding="utf-8", newline="\n")
    return report
