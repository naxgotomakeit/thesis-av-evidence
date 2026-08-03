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
from experiments.r3_v2_av_coarse_semantic_organizer_v1_1.core import (
    _audio_record,
    _overlap,
    build_payload,
    forbidden_input_hits,
)


DISCOVERY_PROMPT = """You are the global phase-discovery stage of an offline, question-agnostic audio-visual navigation Organizer. You receive the complete ordered timeline of exact visual captions and aligned ASR as separate channels. Discover contiguous semantic phases from the supplied timeline itself.

Create a boundary only for a meaningful change in setting, principal activity, human interaction, control state, incident state, response state, or scene-management phase. Keep consecutive records together when they describe one coherent phase. Do not create boundaries for caption wording changes, camera angle, isolated objects, repeated background details, speaker changes, or short anomalous records. Do not let an isolated cross-context caption create or name a phase. ASR can establish audible requests, reports, commands, coordination, and phase transitions, but a mention or request is not automatic visual confirmation or proof of completion.

Return only the inclusive end_medium_index of every discovered phase. The first phase starts implicitly at 0; each later phase begins after the prior end; indices must strictly increase; the final end must be 29. Do not output phase count targets, labels, summaries, reasons, start indices, IDs, timestamps, evidence selections, Storyline, retrieval decisions, or answers."""

SELECTOR_PROMPT = """You are the local evidence-selection stage of an offline audio-visual navigation Organizer. You receive one fixed contiguous phase containing exact visual captions and aligned ASR. Select a small evidence subset that reliably characterizes the phase and its meaningful internal transition.

Prefer evidence about phase-defining setting, activity, interaction, state, response, or coordination. Prefer repeated, temporally consistent, cross-modally compatible, or directly human-centered evidence. Treat isolated scenic descriptions, background-object inventories, cross-context anomalies, and records unrelated to the phase's principal development as navigation noise. Ordinary low-salience records need not be selected or labelled noise. Keep caption and ASR provenance separate. An audible request, report, plan, or mention is not automatic visual proof or proof that an action completed.

Select at least one and at most the supplied visual budget of Medium indices, and at most the supplied audio budget of audio IDs. A selected or rejected item must exist inside this phase. Representative and rejected-noise sets must be disjoint. Output only the required arrays and selection_uncertainty. Do not write a phase label or summary and do not change the fixed phase."""

SUMMARY_PROMPT = """You are the navigation-summary stage of an offline audio-visual Organizer. Fixed phase boundaries and bounded representative evidence were already selected. For each phase you receive only its selected visual captions and selected ASR; exhaustive records and rejected navigation noise are absent.

Write one concise semantic navigation summary that captures the phase-defining setting, activity, interaction, state, response, or transition supported by the supplied evidence. Preserve modality: visible caption content may be described as visual semantics; audible reports, requests, commands, plans, or coordination must remain qualified as audible and must not become visual proof, proof of completion, or exact physical-event timing. Include uncertainty when the selected evidence is incomplete or cross-modal interpretation is not directly bound.

Do not change grouping, introduce evidence not supplied, enumerate incidental objects, generate Storyline, retrieval decisions, or answers. Return exactly one summary for every group_index in order."""

DISCOVERY_FIELDS = ["end_medium_index"]
SELECTION_FIELDS = [
    "representative_medium_indices",
    "representative_audio_ids",
    "rejected_navigation_noise_medium_indices",
    "rejected_navigation_noise_audio_ids",
    "selection_uncertainty",
]
SUMMARY_FIELDS = ["group_index", "navigation_summary", "uncertainty_notes"]


def discovery_schema() -> dict[str, Any]:
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
                    "required": DISCOVERY_FIELDS,
                    "properties": {"end_medium_index": {"type": "integer"}},
                },
            }
        },
    }


def selector_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": SELECTION_FIELDS,
        "properties": {
            "representative_medium_indices": {"type": "array", "items": {"type": "integer"}},
            "representative_audio_ids": {"type": "array", "items": {"type": "string"}},
            "rejected_navigation_noise_medium_indices": {"type": "array", "items": {"type": "integer"}},
            "rejected_navigation_noise_audio_ids": {"type": "array", "items": {"type": "string"}},
            "selection_uncertainty": {"type": "array", "items": {"type": "string"}},
        },
    }


def summary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summaries"],
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": SUMMARY_FIELDS,
                    "properties": {
                        "group_index": {"type": "integer"},
                        "navigation_summary": {"type": "string"},
                        "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }


def validate_discovery(value: Any, medium_count: int = 30) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or list(value) != ["groups"] or not isinstance(value.get("groups"), list):
        return {"valid": False, "errors": ["invalid_wrapper"], "group_count": 0, "end_indices": []}
    groups = value["groups"]
    if not groups or len(groups) > medium_count:
        errors.append("invalid_group_count")
    ends: list[int] = []
    for position, row in enumerate(groups):
        if not isinstance(row, dict) or list(row) != DISCOVERY_FIELDS:
            errors.append(f"group_{position}:fields_or_order")
            continue
        end = row["end_medium_index"]
        if type(end) is not int or not 0 <= end < medium_count:
            errors.append(f"group_{position}:invalid_end")
        else:
            ends.append(end)
    if len(ends) == len(groups):
        if any(right <= left for left, right in zip(ends, ends[1:])):
            errors.append("ends_not_strictly_increasing")
        if ends and ends[-1] != medium_count - 1:
            errors.append("final_end_invalid")
    return {"valid": not errors, "errors": errors, "group_count": len(groups), "end_indices": ends}


def infer_ranges(discovery: dict[str, Any], medium_count: int = 30) -> list[dict[str, int]]:
    audit = validate_discovery(discovery, medium_count)
    if not audit["valid"]:
        raise ValueError(f"invalid_discovery:{audit['errors']}")
    result: list[dict[str, int]] = []
    start = 0
    for group_index, row in enumerate(discovery["groups"]):
        result.append({"group_index": group_index, "start_medium_index": start, "end_medium_index": row["end_medium_index"]})
        start = row["end_medium_index"] + 1
    return result


def build_selector_payload(
    phase: dict[str, int], index: dict[str, Any], visual_budget: int, audio_budget: int
) -> dict[str, Any]:
    mediums = index["medium_nodes"]
    selected = mediums[phase["start_medium_index"] : phase["end_medium_index"] + 1]
    start_sec, end_sec = float(selected[0]["start_sec"]), float(selected[-1]["end_sec"])
    return {
        "fixed_phase": copy.deepcopy(phase),
        "visual_budget": visual_budget,
        "audio_budget": audio_budget,
        "visual_captions": [
            {
                "medium_index": i,
                "medium_id": mediums[i]["medium_id"],
                "start_sec": float(mediums[i]["start_sec"]),
                "end_sec": float(mediums[i]["end_sec"]),
                "caption": mediums[i]["qwen_caption"],
            }
            for i in range(phase["start_medium_index"], phase["end_medium_index"] + 1)
        ],
        "asr": [_audio_record(row) for row in _overlap(index["audio_nodes"], start_sec, end_sec)],
    }


def validate_selection(
    value: Any,
    phase: dict[str, int],
    selector_payload: dict[str, Any],
    visual_budget: int,
    audio_budget: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or list(value) != SELECTION_FIELDS:
        return {"valid": False, "errors": ["fields_or_order"]}
    for field in SELECTION_FIELDS:
        if not isinstance(value.get(field), list):
            errors.append(f"{field}:not_array")
    if errors:
        return {"valid": False, "errors": errors}
    reps = value["representative_medium_indices"]
    rejected = value["rejected_navigation_noise_medium_indices"]
    rep_audio = value["representative_audio_ids"]
    rejected_audio = value["rejected_navigation_noise_audio_ids"]
    if not reps or len(reps) > visual_budget or not all(type(item) is int for item in reps + rejected):
        errors.append("invalid_visual_selection")
    if len(rep_audio) > audio_budget or not all(isinstance(item, str) for item in rep_audio + rejected_audio):
        errors.append("invalid_audio_selection")
    if any(len(items) != len(set(items)) for items in (reps, rejected, rep_audio, rejected_audio)):
        errors.append("duplicate_selection")
    if set(reps) & set(rejected) or set(rep_audio) & set(rejected_audio):
        errors.append("representative_rejected_overlap")
    lo, hi = phase["start_medium_index"], phase["end_medium_index"]
    if any(item < lo or item > hi for item in reps + rejected):
        errors.append("medium_outside_phase")
    known_audio = {row["audio_id"] for row in selector_payload["asr"]}
    if any(item not in known_audio for item in rep_audio + rejected_audio):
        errors.append("audio_outside_phase_or_unknown")
    if not all(isinstance(item, str) for item in value["selection_uncertainty"]):
        errors.append("invalid_selection_uncertainty")
    return {"valid": not errors, "errors": errors}


def build_summary_payload(
    ranges: list[dict[str, int]], selections: list[dict[str, Any]], index: dict[str, Any]
) -> dict[str, Any]:
    mediums = index["medium_nodes"]
    audio_by_id = {row["audio_id"]: _audio_record(row) for row in index["audio_nodes"]}
    groups = []
    for phase, selection in zip(ranges, selections):
        groups.append({
            "group_index": phase["group_index"],
            "selected_visual_captions": [
                {
                    "medium_index": i,
                    "medium_id": mediums[i]["medium_id"],
                    "start_sec": float(mediums[i]["start_sec"]),
                    "end_sec": float(mediums[i]["end_sec"]),
                    "caption": mediums[i]["qwen_caption"],
                }
                for i in selection["representative_medium_indices"]
            ],
            "selected_asr": [audio_by_id[item] for item in selection["representative_audio_ids"]],
            "selection_uncertainty": list(selection["selection_uncertainty"]),
        })
    return {"fixed_groups_with_selected_evidence_only": groups}


def validate_summaries(value: Any, group_count: int) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or list(value) != ["summaries"] or not isinstance(value.get("summaries"), list):
        return {"valid": False, "errors": ["invalid_wrapper"], "summary_count": 0}
    rows = value["summaries"]
    if len(rows) != group_count:
        errors.append("summary_count_mismatch")
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or list(row) != SUMMARY_FIELDS:
            errors.append(f"summary_{position}:fields_or_order")
            continue
        if type(row["group_index"]) is not int or row["group_index"] != position:
            errors.append(f"summary_{position}:group_index_mismatch")
        if not isinstance(row["navigation_summary"], str) or not row["navigation_summary"].strip():
            errors.append(f"summary_{position}:empty")
        if not isinstance(row["uncertainty_notes"], list) or not all(isinstance(item, str) for item in row["uncertainty_notes"]):
            errors.append(f"summary_{position}:invalid_uncertainty")
    return {"valid": not errors, "errors": errors, "summary_count": len(rows)}


def reconstruct(
    index: dict[str, Any], discovery: dict[str, Any], selections: list[dict[str, Any]], summaries: dict[str, Any]
) -> dict[str, Any]:
    ranges = infer_ranges(discovery, len(index["medium_nodes"]))
    if len(selections) != len(ranges) or not validate_summaries(summaries, len(ranges))["valid"]:
        raise ValueError("selection_or_summary_count_mismatch")
    mediums = index["medium_nodes"]
    audio_by_id = {row["audio_id"]: _audio_record(row) for row in index["audio_nodes"]}
    coarse_regions = []
    for phase, selection, summary in zip(ranges, selections, summaries["summaries"]):
        nodes = mediums[phase["start_medium_index"] : phase["end_medium_index"] + 1]
        start_sec, end_sec = float(nodes[0]["start_sec"]), float(nodes[-1]["end_sec"])
        attached = [_audio_record(row) for row in _overlap(index["audio_nodes"], start_sec, end_sec)]
        reps = selection["representative_medium_indices"]
        rejected = selection["rejected_navigation_noise_medium_indices"]
        rep_audio = selection["representative_audio_ids"]
        rejected_audio = selection["rejected_navigation_noise_audio_ids"]
        coarse_regions.append({
            "coarse_id": f"C{phase['group_index'] + 1:02d}",
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "source_medium_ids": [row["medium_id"] for row in nodes],
            "source_fine_ids": [fine_id for row in nodes for fine_id in row["child_fine_ids"]],
            "navigation_summary": summary["navigation_summary"],
            "uncertainty_notes": summary["uncertainty_notes"],
            "representative_medium_ids": [mediums[i]["medium_id"] for i in reps],
            "representative_audio_ids": rep_audio,
            "rejected_navigation_noise_medium_ids": [mediums[i]["medium_id"] for i in rejected],
            "rejected_navigation_noise_audio_ids": rejected_audio,
            "representative_source_captions": [
                {"medium_id": mediums[i]["medium_id"], "caption": mediums[i]["qwen_caption"]} for i in reps
            ],
            "representative_source_asr": [audio_by_id[item] for item in rep_audio],
            "exact_source_captions": [
                {"medium_id": row["medium_id"], "caption": row["qwen_caption"]} for row in nodes
            ],
            "exact_source_asr": attached,
            "audio_ids": [row["audio_id"] for row in attached],
            "map_type": "r3_2_independent_denoised_av_semantic_coarse",
            "semantic_fields_available": True,
        })
    return {
        "map_type": "r3_2_independent_denoised_av_semantic_coarse",
        "semantic_fields_available": True,
        "visual_semantic_source": "canonical_qwen_caption_selected_per_independently_discovered_phase",
        "audio_semantic_source": "canonical_timestamped_asr_selected_per_independently_discovered_phase",
        "coarse_regions": coarse_regions,
        "storyline_events": [],
        "has_storyline": False,
        "hard_filtering_allowed": False,
        "provenance": {
            "phase_discovery": "global AV inference without historical phase scaffold",
            "phase_evidence_selection": "bounded local selection",
            "navigation_summary": "selected evidence only",
            "mechanical_reconstruction": "deterministic",
            "historical_phase_content_used_in_generation": False,
            "all_source_evidence_retained_in_sidecar": True,
            "all_mediums_retrieval_eligible": True,
            "coarse_prior_affects_ranking": False,
            "map_text_is_not_sufficiency_evidence": True,
        },
    }


def validate_map(index: dict[str, Any], semantic_map: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    actual = [item for row in semantic_map["coarse_regions"] for item in row["source_medium_ids"]]
    expected = [row["medium_id"] for row in index["medium_nodes"]]
    if actual != expected:
        errors.append("medium_coverage_mismatch")
    actual_fine = [item for row in semantic_map["coarse_regions"] for item in row["source_fine_ids"]]
    expected_fine = [item for row in index["medium_nodes"] for item in row["child_fine_ids"]]
    if actual_fine != expected_fine or len(actual_fine) != len(index["fine_nodes"]):
        errors.append("fine_coverage_mismatch")
    for position, row in enumerate(semantic_map["coarse_regions"]):
        if position and semantic_map["coarse_regions"][position - 1]["end_sec"] != row["start_sec"]:
            errors.append(f"coarse_{position}:nonadjacent")
        expected_asr = [_audio_record(item) for item in _overlap(index["audio_nodes"], row["start_sec"], row["end_sec"])]
        if canonical_bytes(expected_asr) != canonical_bytes(row["exact_source_asr"]):
            errors.append(f"coarse_{position}:audio_mismatch")
    if semantic_map["storyline_events"] or semantic_map["has_storyline"]:
        errors.append("storyline_present")
    if semantic_map["hard_filtering_allowed"]:
        errors.append("hard_filtering_enabled")
    return {"valid": not errors, "errors": errors, "medium_count": len(actual), "fine_count": len(actual_fine)}


def planner_view(semantic_map: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_type": semantic_map["map_type"],
        "coarse_regions": [
            {
                "coarse_id": row["coarse_id"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "source_medium_ids": row["source_medium_ids"],
                "source_audio_ids": row["audio_ids"],
                "event_label": "",
                "summary": row["navigation_summary"],
                "coarse_summary": row["navigation_summary"],
                "uncertainty_notes": row["uncertainty_notes"],
                "representative_source_captions": row["representative_source_captions"],
                "representative_source_asr": row["representative_source_asr"],
                "adapter_metadata": {
                    "adapter_derived": True,
                    "source_field": "navigation_summary",
                    "semantic_summary": True,
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
            "map_may_guide_search_units_and_query_variants": True,
            "suggested_coarse_are_navigation_hints_only": True,
            "all_mediums_remain_eligible": True,
            "coarse_prior_affects_ranking": False,
            "map_text_is_not_sufficiency_evidence": True,
        },
    }


def _strict_view(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("map_type", "coarse_regions", "storyline_events", "has_storyline", "hard_filtering_allowed", "planner_policy")}


def _call(api_key: str, config: dict[str, Any], prompt: str, payload: dict[str, Any], schema: dict[str, Any], max_tokens_key: str) -> tuple[str, dict[str, Any]]:
    import anthropic

    started = time.perf_counter()
    response = anthropic.Anthropic(api_key=api_key).messages.create(
        model=config["model"],
        max_tokens=int(config[max_tokens_key]),
        temperature=float(config["temperature"]),
        system=prompt,
        messages=[{"role": "user", "content": canonical_bytes(payload).decode("utf-8")}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
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


def no_api_tests(index: dict[str, Any], visual_budget: int, audio_budget: int) -> dict[str, Any]:
    good_discovery = {"groups": [{"end_medium_index": 4}, {"end_medium_index": 29}]}
    ranges = infer_ranges(good_discovery)
    payload = build_selector_payload(ranges[0], index, visual_budget, audio_budget)
    audio_ids = [row["audio_id"] for row in payload["asr"]]
    selection = {
        "representative_medium_indices": [0],
        "representative_audio_ids": audio_ids[:1],
        "rejected_navigation_noise_medium_indices": [1],
        "rejected_navigation_noise_audio_ids": [],
        "selection_uncertainty": [],
    }
    checks: dict[str, bool] = {}
    checks["valid_discovery"] = validate_discovery(good_discovery)["valid"]
    bad = {"groups": [{"end_medium_index": 4}, {"end_medium_index": 4}, {"end_medium_index": 29}]}
    checks["duplicate_end_fails"] = not validate_discovery(bad)["valid"]
    bad = {"groups": [{"end_medium_index": 4}, {"end_medium_index": 28}]}
    checks["incomplete_end_fails"] = not validate_discovery(bad)["valid"]
    checks["valid_selection"] = validate_selection(selection, ranges[0], payload, visual_budget, audio_budget)["valid"]
    bad_selection = copy.deepcopy(selection)
    bad_selection["representative_medium_indices"] = []
    checks["empty_visual_selection_fails"] = not validate_selection(bad_selection, ranges[0], payload, visual_budget, audio_budget)["valid"]
    bad_selection = copy.deepcopy(selection)
    bad_selection["representative_medium_indices"] = [0]
    bad_selection["rejected_navigation_noise_medium_indices"] = [0]
    checks["selection_overlap_fails"] = not validate_selection(bad_selection, ranges[0], payload, visual_budget, audio_budget)["valid"]
    second_payload = build_selector_payload(ranges[1], index, visual_budget, audio_budget)
    second_selection = {
        "representative_medium_indices": [5],
        "representative_audio_ids": [],
        "rejected_navigation_noise_medium_indices": [],
        "rejected_navigation_noise_audio_ids": [],
        "selection_uncertainty": [],
    }
    selections = [selection, second_selection]
    summary_payload = build_summary_payload(ranges, selections, index)
    serialized = canonical_bytes(summary_payload).decode("utf-8")
    checks["rejected_caption_isolated_from_summary"] = index["medium_nodes"][1]["qwen_caption"] not in serialized
    summaries = {"summaries": [
        {"group_index": 0, "navigation_summary": "phase one", "uncertainty_notes": []},
        {"group_index": 1, "navigation_summary": "phase two", "uncertainty_notes": []},
    ]}
    checks["valid_summaries"] = validate_summaries(summaries, 2)["valid"]
    semantic_map = reconstruct(index, good_discovery, selections, summaries)
    checks["deterministic_complete_map"] = validate_map(index, semantic_map)["valid"] and canonical_bytes(semantic_map) == canonical_bytes(reconstruct(index, good_discovery, selections, summaries))
    view = planner_view(semantic_map)
    checks["planner_no_hard_filter"] = not view["hard_filtering_allowed"] and view["planner_policy"]["all_mediums_remain_eligible"] and not view["planner_policy"]["coarse_prior_affects_ranking"]
    checks["storyline_absent"] = not semantic_map["storyline_events"] and not semantic_map["has_storyline"]
    checks["historical_content_not_in_generation_payloads"] = "historical" not in canonical_bytes(build_payload(index)).decode("utf-8").lower() and "historical" not in serialized.lower()
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "model_api_calls": 0}


def prepare(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = load_json(config_path)
    protected_rel = [config["canonical_index"], config["r3_1_control"], config["historical_reference"], config["planner_consumer"]]
    protected = {item: sha256_file(root / item) for item in protected_rel}
    if protected[config["canonical_index"]] != config["canonical_index_sha256"]:
        raise RuntimeError("canonical_hash_mismatch")
    index = load_json(root / config["canonical_index"])
    payload = build_payload(index)
    tests = no_api_tests(index, int(config["max_visual_evidence_per_phase"]), int(config["max_audio_evidence_per_phase"]))
    checks = {
        "canonical_counts": (len(index["medium_nodes"]), len(index["fine_nodes"]), len(index["audio_nodes"])) == (30, 88, 107),
        "forbidden_input_hits_zero": not forbidden_input_hits(payload),
        "historical_phase_fields_absent": not any(key in canonical_bytes(payload).decode("utf-8").lower() for key in ("phase_id", "historical", "nine_phase")),
        "zero_repair": config["model_repair_calls"] == 0,
        "zero_retry": config["semantic_retries"] == 0,
        "storyline_disabled": config["storyline_enabled"] is False,
        "hard_filtering_disabled": config["hard_filtering_allowed"] is False,
    }
    if tests["status"] != "passed" or not all(checks.values()):
        raise RuntimeError(f"no_api_preflight_failed:{tests}:{checks}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input_manifest.json", {
        "canonical_index": config["canonical_index"],
        "canonical_index_sha256": protected[config["canonical_index"]],
        "caption_hash": hashlib.sha256(canonical_bytes([row["qwen_caption"] for row in index["medium_nodes"]])).hexdigest(),
        "audio_hash": hashlib.sha256(canonical_bytes([_audio_record(row) for row in index["audio_nodes"]])).hexdigest(),
        "medium_count": 30,
        "fine_count": 88,
        "audio_count": 107,
        "model": config["model"],
        "temperature": config["temperature"],
        "historical_phase_scaffold_used": False,
    })
    write_json(output / "source_artifact_audit.json", {"status": "passed", "protected_hashes_before": protected, "checks": checks})
    write_json(output / "global_discovery_input_payload.json", payload)
    write_json(output / "global_discovery_schema.json", discovery_schema())
    write_json(output / "selector_schema.json", selector_schema())
    write_json(output / "summary_schema.json", summary_schema())
    write_json(output / "no_api_test_report.json", tests)
    return {"config": config, "index": index, "payload": payload, "protected": protected, "tests": tests, "checks": checks}


def _parse(raw: str) -> tuple[Any, str | None]:
    try:
        return json.loads(raw), None
    except Exception as error:
        return None, f"{type(error).__name__}:{error}"


def _fail(output: Path, stage: str, usages: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    report = {
        "source_validation": "passed",
        "failed_stage": stage,
        "overall_validation": f"{stage}_first_pass_invalid",
        "errors": errors,
        "model_calls": len(usages),
        "repair_calls": 0,
        "semantic_retries": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "sufficiency_calls": 0,
        "final_answer_calls": 0,
    }
    write_json(output / "validation_report.json", report)
    write_json(output / "cost_accounting.json", {"calls": usages, "model_calls": len(usages), "repair_calls": 0, "semantic_retries": 0})
    (output / "REPORT.md").write_text(f"# Independent R3_2 canary\n\n{stage} first-pass output invalid; no repair or retry was performed.\n", encoding="utf-8", newline="\n")
    return report


def _noise_hits(view: dict[str, Any]) -> list[str]:
    terms = ("snowy mountain", "snow-covered ground", "clouds moving", "bird feeder", "furry object", "white bucket", "black cloth")
    text = canonical_bytes(view).decode("utf-8").lower()
    return [term for term in terms if term in text]


def run(root: Path, config_path: Path, output: Path, api_key: str | None) -> dict[str, Any]:
    prepared = prepare(root, config_path, output)
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY_missing_before_model_call")
    config, index = prepared["config"], prepared["index"]
    usages: list[dict[str, Any]] = []

    raw_discovery, discovery_usage = _call(api_key, config, DISCOVERY_PROMPT, prepared["payload"], discovery_schema(), "discovery_max_tokens")
    usages.append({"stage": "global_phase_discovery", **discovery_usage})
    discovery, discovery_parse_error = _parse(raw_discovery)
    discovery_audit = validate_discovery(discovery) if discovery is not None else {"valid": False, "errors": ["json_parse_failed"], "group_count": 0, "end_indices": []}
    write_json(output / "global_discovery_raw_response.json", {"raw_response": raw_discovery, "usage": discovery_usage, "parse_error": discovery_parse_error, "attempt_count": 1})
    write_json(output / "independent_phase_boundaries.json", discovery if isinstance(discovery, dict) else {"status": "not_available"})
    write_json(output / "global_discovery_validation.json", discovery_audit)
    if not discovery_audit["valid"]:
        return _fail(output, "global_discovery", usages, discovery_audit["errors"])

    ranges = infer_ranges(discovery)
    selector_payloads: list[dict[str, Any]] = []
    selector_raw: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    selector_audits: list[dict[str, Any]] = []
    visual_budget = int(config["max_visual_evidence_per_phase"])
    audio_budget = int(config["max_audio_evidence_per_phase"])
    for phase in ranges:
        payload = build_selector_payload(phase, index, visual_budget, audio_budget)
        selector_payloads.append(payload)
        raw, usage = _call(api_key, config, SELECTOR_PROMPT, payload, selector_schema(), "selector_max_tokens")
        usages.append({"stage": f"phase_selector_{phase['group_index']}", **usage})
        value, parse_error = _parse(raw)
        audit = validate_selection(value, phase, payload, visual_budget, audio_budget) if value is not None else {"valid": False, "errors": ["json_parse_failed"]}
        selector_raw.append({"group_index": phase["group_index"], "raw_response": raw, "usage": usage, "parse_error": parse_error, "attempt_count": 1})
        selector_audits.append({"group_index": phase["group_index"], **audit})
        if not audit["valid"]:
            write_json(output / "phase_selector_input_payloads.json", selector_payloads)
            write_json(output / "phase_selector_raw_responses.json", selector_raw)
            write_json(output / "phase_selector_validation.json", selector_audits)
            return _fail(output, f"phase_selector_{phase['group_index']}", usages, audit["errors"])
        selections.append(value)
    write_json(output / "phase_selector_input_payloads.json", selector_payloads)
    write_json(output / "phase_selector_raw_responses.json", selector_raw)
    write_json(output / "phase_evidence_selections.json", selections)
    write_json(output / "phase_selector_validation.json", {"status": "passed", "audits": selector_audits})

    summary_payload = build_summary_payload(ranges, selections, index)
    selected_serialized = canonical_bytes(summary_payload).decode("utf-8")
    rejected_medium_ids = [index["medium_nodes"][i]["medium_id"] for selection in selections for i in selection["rejected_navigation_noise_medium_indices"]]
    rejected_caption_leaks = [medium_id for medium_id in rejected_medium_ids if medium_id in selected_serialized]
    isolation_audit = {
        "status": "passed" if not rejected_caption_leaks else "failed",
        "rejected_medium_ids": rejected_medium_ids,
        "rejected_medium_ids_in_summary_payload": rejected_caption_leaks,
        "exhaustive_source_fields_present": "exact_source" in selected_serialized,
        "historical_phase_content_present": "historical" in selected_serialized.lower(),
    }
    write_json(output / "summary_evidence_isolation_audit.json", isolation_audit)
    if isolation_audit["status"] != "passed" or isolation_audit["exhaustive_source_fields_present"] or isolation_audit["historical_phase_content_present"]:
        return _fail(output, "summary_evidence_isolation", usages, ["summary_payload_not_isolated"])
    write_json(output / "summary_input_payload.json", summary_payload)
    raw_summary, summary_usage = _call(api_key, config, SUMMARY_PROMPT, summary_payload, summary_schema(), "summary_max_tokens")
    usages.append({"stage": "batched_navigation_summary", **summary_usage})
    summaries, summary_parse_error = _parse(raw_summary)
    summary_audit = validate_summaries(summaries, len(ranges)) if summaries is not None else {"valid": False, "errors": ["json_parse_failed"], "summary_count": 0}
    write_json(output / "summary_raw_response.json", {"raw_response": raw_summary, "usage": summary_usage, "parse_error": summary_parse_error, "attempt_count": 1})
    write_json(output / "phase_navigation_summaries.json", summaries if isinstance(summaries, dict) else {"status": "not_available"})
    write_json(output / "summary_validation.json", summary_audit)
    if not summary_audit["valid"]:
        return _fail(output, "summary", usages, summary_audit["errors"])

    semantic_map = reconstruct(index, discovery, selections, summaries)
    map_audit = validate_map(index, semantic_map)
    view = planner_view(semantic_map)
    planner_audit = validate_planner_view(_strict_view(view), index)
    write_json(output / "r3_2_independent_denoised_av_semantic_map.json", semantic_map)
    write_json(output / "deterministic_reconstruction_audit.json", map_audit)
    write_json(output / "planner_compatibility_view.json", view)
    write_json(output / "planner_compatibility_report.json", {**planner_audit, "planner_calls": 0, "all_mediums_remain_eligible": True, "coarse_prior_affects_ranking": False})

    # Historical semantics are first loaded here, after the candidate map is complete.
    historical = load_json(root / config["historical_reference"])
    historical_rows = historical.get("phases", [])
    posthoc = {
        "candidate_generated_without_historical_phase_content": True,
        "comparison_loaded_only_after_candidate_reconstruction": True,
        "candidate_group_count": len(semantic_map["coarse_regions"]),
        "historical_group_count": len(historical_rows),
        "candidate_ranges_sec": [[row["start_sec"], row["end_sec"]] for row in semantic_map["coarse_regions"]],
        "historical_ranges_sec": [[row["start_sec"], row["end_sec"]] for row in historical_rows],
        "historical_group_count_is_not_acceptance_target": True,
    }
    write_json(output / "historical_posthoc_comparison.json", posthoc)
    noise_hits = _noise_hits(view)
    write_json(output / "navigation_noise_audit.json", {"status": "passed" if not noise_hits else "flagged_for_manual_review", "posthoc_terms_found": noise_hits, "terms_not_used_in_prompt_or_selection": True})

    after = {item: sha256_file(root / item) for item in prepared["protected"]}
    unchanged = after == prepared["protected"]
    write_json(output / "protected_hash_audit.json", {"status": "passed" if unchanged else "failed", "before": prepared["protected"], "after": after, "unchanged": unchanged})
    structural = map_audit["valid"] and planner_audit["valid"] and unchanged
    report = {
        "source_validation": "passed" if unchanged else "failed",
        "historical_phase_independence_validation": "passed",
        "global_discovery_validation": "passed",
        "phase_selector_validation": "passed",
        "summary_evidence_isolation_validation": "passed",
        "summary_validation": "passed",
        "deterministic_reconstruction_validation": "passed" if map_audit["valid"] else "failed",
        "planner_compatibility_validation": "passed" if planner_audit["valid"] else "failed",
        "navigation_noise_audit": "passed" if not noise_hits else "flagged_for_manual_review",
        "semantic_acceptance": "pending_manual_review",
        "overall_validation": "pending_manual_semantic_review" if structural else "failed_structural_validation",
        "recommendation": "ready_for_independent_r3_2_manual_review" if structural else "independent_r3_2_contract_failure",
        "coarse_count": len(semantic_map["coarse_regions"]),
        "end_indices": discovery_audit["end_indices"],
        "medium_count": map_audit["medium_count"],
        "fine_count": map_audit["fine_count"],
        "selected_visual_count": sum(len(row["representative_medium_indices"]) for row in selections),
        "selected_audio_count": sum(len(row["representative_audio_ids"]) for row in selections),
        "rejected_navigation_noise_medium_count": sum(len(row["rejected_navigation_noise_medium_indices"]) for row in selections),
        "storyline_count": 0,
        "hard_filtering_allowed": False,
        "all_mediums_retrieval_eligible": True,
        "model_calls": len(usages),
        "repair_calls": 0,
        "semantic_retries": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "sufficiency_calls": 0,
        "final_answer_calls": 0,
    }
    write_json(output / "validation_report.json", report)
    write_json(output / "cost_accounting.json", {
        "calls": usages,
        "model_calls": len(usages),
        "input_tokens": sum(row["input_tokens"] for row in usages),
        "output_tokens": sum(row["output_tokens"] for row in usages),
        "summed_latency_sec": sum(row["latency_sec"] for row in usages),
        "repair_calls": 0,
        "semantic_retries": 0,
        "all_other_model_calls": 0,
    })
    sections = "".join(
        f"<section><h2>{html.escape(row['coarse_id'])}: {row['start_sec']:.3f}-{row['end_sec']:.3f}s</h2>"
        f"<p><b>Summary:</b> {html.escape(row['navigation_summary'])}</p>"
        f"<p><b>Mediums:</b> {html.escape(', '.join(row['source_medium_ids']))}</p>"
        f"<p><b>Selected visual:</b> {html.escape(', '.join(row['representative_medium_ids']))}</p>"
        f"<p><b>Selected audio:</b> {html.escape(', '.join(row['representative_audio_ids']) or 'none')}</p>"
        f"<p><b>Rejected navigation noise:</b> {html.escape(', '.join(row['rejected_navigation_noise_medium_ids']) or 'none')}</p>"
        f"<details><summary>Exact source captions and ASR</summary><pre>{html.escape(json.dumps({'captions': row['exact_source_captions'], 'asr': row['exact_source_asr']}, ensure_ascii=False, indent=2))}</pre></details></section>"
        for row in semantic_map["coarse_regions"]
    )
    (output / "review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Independent R3_2 review</title>"
        "<style>body{font:14px system-ui;margin:24px;line-height:1.5}section{border-top:2px solid #567;padding:12px}pre{white-space:pre-wrap}</style>"
        "<h1>Independent R3_2 denoised AV map</h1><p>No historical phase count, boundary, label, or summary entered generation.</p>"
        + sections
        + f"<h2>Post-hoc historical comparison</h2><pre>{html.escape(json.dumps(posthoc, indent=2))}</pre>",
        encoding="utf-8",
        newline="\n",
    )
    (output / "REPORT.md").write_text("\n".join([
        "# Independent R3_2 global-phase denoised AV Organizer canary",
        "",
        "- Historical nine-phase count/boundaries/labels/summaries used in generation: `false`",
        f"- Independently discovered Coarse groups: `{report['coarse_count']}`",
        f"- End indices: `{report['end_indices']}`",
        f"- Medium/Fine coverage: `{report['medium_count']}/30`, `{report['fine_count']}/88`",
        f"- Selected visual/audio evidence: `{report['selected_visual_count']}/{report['selected_audio_count']}`",
        f"- Post-hoc navigation-noise terms: `{noise_hits}`",
        "- Storyline: `0`; hard filtering: `false`; all Mediums eligible: `true`",
        f"- Model calls: `{report['model_calls']}`; repairs/retries: `0/0`",
        "- Planner compatibility: `passed` (no Planner call)",
        "- Semantic acceptance: `pending_manual_review`",
        "",
        "Historical nine-phase output was read only after the candidate map was complete and was used only for descriptive comparison. Planner, Retrieval, Sufficiency, temporal review, Final Gemini, and QA were not run.",
    ]) + "\n", encoding="utf-8", newline="\n")
    return report
