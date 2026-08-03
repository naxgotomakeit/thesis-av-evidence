from __future__ import annotations

import copy
import hashlib
import html
import json
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
from experiments.r3_2_independent_global_phase_denoised_av_organizer.core import (
    _call,
    _strict_view,
    planner_view,
    reconstruct,
    validate_discovery,
    validate_map,
)


STAGE1_FIELDS = [
    "end_medium_index",
    "phase_label",
    "boundary_reason",
    "salient_medium_indices",
    "salient_audio_ids",
]
STAGE2_FIELDS = [
    "direct_visual",
    "transcript_evidence",
    "av_interpretation",
    "phase_summary",
    "uncertainty",
]

STAGE1_PROMPT = """You are segmenting a chronological audio-visual evidence stream from first-person body-camera footage into a small number of meaningful incident phases.

The input contains exact timestamped visual captions and timestamped speech transcripts. Use both modalities and the complete global timeline to identify major changes in activity, interaction, spoken commands, apparent scene purpose, restraint, injury response, or other meaningful operational phases. Audio may reveal a phase transition that is visually subtle. Visual evidence may provide context for nearby speech.

Rules:
- Produce between the supplied minimum and maximum number of chronological phases.
- Phase boundaries must reflect meaningful activity changes, not every minor visual or speech change.
- Do not create or characterize a phase from incidental objects, repeated scenery, isolated cross-context captions, or short noisy records.
- Judge salient evidence by whether it explains the global incident progression or a meaningful phase transition, not merely whether it describes a concrete local action.
- Do not infer legal status, guilt, motive, or unsupported identity.
- Temporal proximity alone does not prove causality.
- Treat transcripts as heard text with uncertain speaker/source.
- Keep unsupported interpretations uncertain.
- Cite only a few salient Medium indices and audio IDs that characterize the phase and motivated its boundary.

Output only inclusive end_medium_index, phase_label, boundary_reason, salient_medium_indices, and salient_audio_ids for each phase. The first phase starts implicitly at Medium 0; later phases start after the prior end; end indices strictly increase; the final end is 29. Do not output start indices, phase IDs, timestamps, Coarse IDs, Fine IDs, Storyline, retrieval decisions, scores, or answers."""

STAGE2_PROMPT = """You are analysing one validated chronological phase from first-person body-camera footage. The fixed phase label, boundary reason, and globally selected salient evidence come from a preceding Organizer that saw the complete video timeline. Use that global phase meaning when deciding what matters locally.

You receive the phase's exact visual captions, timestamped speech transcripts, and limited neighbouring context. Combine the modalities into a concise factual phase account while ignoring incidental scenery and object interactions that do not contribute to the supplied global phase meaning.

Distinguish:
1. DIRECT_VISUAL: semantics explicitly stated by supplied visual captions;
2. TRANSCRIPT_EVIDENCE: what is audibly stated;
3. AV_INTERPRETATION: optional interpretation supported jointly by cited visual and audio evidence;
4. UNCERTAINTY: what remains unclear.

Rules:
- Do not change the fixed phase or its global meaning.
- AV_INTERPRETATION is optional and requires at least one visual and one audio citation.
- Temporal proximity alone does not establish causality.
- Do not infer guilt, motive, unsupported identity, ownership, or intent.
- Do not classify speech as dispatch, radio, direct command, or report unless supported.
- Preserve modality conflicts and ambiguity.
- Ignore incidental scenery unless necessary to understand the event.
- Retain important actions, spoken commands, weapons, restraint, injury, medical response, and changes in interaction when supported.
- Evidence IDs must come from the supplied phase or neighbouring context.

Return only direct_visual, transcript_evidence, av_interpretation, phase_summary, and uncertainty. Do not output phase IDs, boundaries, Storyline, retrieval decisions, or answers."""


def stage1_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": ["phases"],
        "properties": {"phases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": STAGE1_FIELDS,
            "properties": {
                "end_medium_index": {"type": "integer"},
                "phase_label": {"type": "string"},
                "boundary_reason": {"type": "string"},
                "salient_medium_indices": {"type": "array", "items": {"type": "integer"}},
                "salient_audio_ids": {"type": "array", "items": {"type": "string"}},
            },
        }}},
    }


def stage2_schema() -> dict[str, Any]:
    visual_claim = {
        "type": "object", "additionalProperties": False, "required": ["claim", "medium_indices"],
        "properties": {"claim": {"type": "string"}, "medium_indices": {"type": "array", "items": {"type": "integer"}}},
    }
    audio_claim = {
        "type": "object", "additionalProperties": False, "required": ["claim", "audio_ids"],
        "properties": {"claim": {"type": "string"}, "audio_ids": {"type": "array", "items": {"type": "string"}}},
    }
    av_claim = {
        "type": "object", "additionalProperties": False, "required": ["claim", "medium_indices", "audio_ids"],
        "properties": {
            "claim": {"type": "string"},
            "medium_indices": {"type": "array", "items": {"type": "integer"}},
            "audio_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    return {
        "type": "object", "additionalProperties": False, "required": STAGE2_FIELDS,
        "properties": {
            "direct_visual": {"type": "array", "items": visual_claim},
            "transcript_evidence": {"type": "array", "items": audio_claim},
            "av_interpretation": {"type": "array", "items": av_claim},
            "phase_summary": {"type": "string"},
            "uncertainty": {"type": "array", "items": {"type": "string"}},
        },
    }


def validate_stage1(value: Any, index: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or list(value) != ["phases"] or not isinstance(value.get("phases"), list):
        return {"valid": False, "errors": ["invalid_wrapper"], "phase_count": 0, "end_indices": []}
    phases = value["phases"]
    if not int(config["minimum_phase_count"]) <= len(phases) <= int(config["maximum_phase_count"]):
        errors.append("phase_count_out_of_frozen_range")
    discovery = {"groups": []}
    audio_by_id = {row["audio_id"]: row for row in index["audio_nodes"]}
    start = 0
    for position, row in enumerate(phases):
        if not isinstance(row, dict) or list(row) != STAGE1_FIELDS:
            errors.append(f"phase_{position}:fields_or_order")
            continue
        discovery["groups"].append({"end_medium_index": row["end_medium_index"]})
        if not isinstance(row["phase_label"], str) or not row["phase_label"].strip():
            errors.append(f"phase_{position}:empty_label")
        if not isinstance(row["boundary_reason"], str) or not row["boundary_reason"].strip():
            errors.append(f"phase_{position}:empty_boundary_reason")
        salient_m, salient_a = row["salient_medium_indices"], row["salient_audio_ids"]
        if not isinstance(salient_m, list) or not salient_m or not all(type(item) is int for item in salient_m):
            errors.append(f"phase_{position}:invalid_salient_mediums")
            salient_m = []
        if len(salient_m) > int(config["max_salient_visual_per_phase"]):
            errors.append(f"phase_{position}:visual_budget_exceeded")
        if not isinstance(salient_a, list) or not all(isinstance(item, str) for item in salient_a):
            errors.append(f"phase_{position}:invalid_salient_audio")
            salient_a = []
        if len(salient_a) > int(config["max_salient_audio_per_phase"]):
            errors.append(f"phase_{position}:audio_budget_exceeded")
        if len(salient_m) != len(set(salient_m)) or len(salient_a) != len(set(salient_a)):
            errors.append(f"phase_{position}:duplicate_salient_evidence")
        end = row["end_medium_index"] if type(row["end_medium_index"]) is int else -1
        salient_scope = config.get("stage1_salient_scope", "phase")
        if salient_scope == "global":
            if any(item < 0 or item >= len(index["medium_nodes"]) for item in salient_m):
                errors.append(f"phase_{position}:unknown_salient_medium")
        elif any(item < start or item > end for item in salient_m):
            errors.append(f"phase_{position}:salient_medium_outside_phase")
        if 0 <= start < len(index["medium_nodes"]) and 0 <= end < len(index["medium_nodes"]):
            phase_start = float(index["medium_nodes"][start]["start_sec"])
            phase_end = float(index["medium_nodes"][end]["end_sec"])
            for audio_id in salient_a:
                audio = audio_by_id.get(audio_id)
                if audio is None:
                    errors.append(f"phase_{position}:unknown_audio:{audio_id}")
                elif salient_scope != "global" and not (float(audio["start_sec"]) < phase_end and float(audio["end_sec"]) > phase_start):
                    errors.append(f"phase_{position}:audio_outside_phase:{audio_id}")
        start = end + 1
    discovery_audit = validate_discovery(discovery, len(index["medium_nodes"])) if len(discovery["groups"]) == len(phases) else {"valid": False, "errors": ["invalid_phase_rows"], "end_indices": []}
    errors.extend(discovery_audit["errors"])
    return {"valid": not errors, "errors": errors, "phase_count": len(phases), "end_indices": discovery_audit["end_indices"]}


def stage1_to_discovery(value: dict[str, Any]) -> dict[str, Any]:
    return {"groups": [{"end_medium_index": row["end_medium_index"]} for row in value["phases"]]}


def build_stage2_payload(
    stage1_phase: dict[str, Any], phase_index: int, start: int, index: dict[str, Any], context_count: int
) -> dict[str, Any]:
    end = stage1_phase["end_medium_index"]
    mediums = index["medium_nodes"]
    phase_nodes = []
    for medium_index in range(start, end + 1):
        medium = mediums[medium_index]
        phase_nodes.append({
            "medium_index": medium_index, "medium_id": medium["medium_id"],
            "start_sec": float(medium["start_sec"]), "end_sec": float(medium["end_sec"]),
            "caption": medium["qwen_caption"], "role": "PHASE",
        })
    context_indices = list(range(max(0, start - context_count), start)) + list(range(end + 1, min(len(mediums), end + 1 + context_count)))
    context_nodes = [{
        "medium_index": i, "medium_id": mediums[i]["medium_id"],
        "start_sec": float(mediums[i]["start_sec"]), "end_sec": float(mediums[i]["end_sec"]),
        "caption": mediums[i]["qwen_caption"], "role": "CONTEXT",
    } for i in context_indices]
    phase_start, phase_end = float(mediums[start]["start_sec"]), float(mediums[end]["end_sec"])
    context_start = float(mediums[context_indices[0]]["start_sec"]) if context_indices else phase_start
    context_end = float(mediums[context_indices[-1]]["end_sec"]) if context_indices else phase_end
    phase_audio = [{**_audio_record(row), "role": "PHASE"} for row in _overlap(index["audio_nodes"], phase_start, phase_end)]
    context_audio = [{**_audio_record(row), "role": "CONTEXT"} for row in _overlap(index["audio_nodes"], context_start, context_end) if not (float(row["start_sec"]) < phase_end and float(row["end_sec"]) > phase_start)]
    return {
        "fixed_phase": {
            "phase_index": phase_index,
            "start_medium_index": start,
            "end_medium_index": end,
            "phase_label": stage1_phase["phase_label"],
            "boundary_reason": stage1_phase["boundary_reason"],
        },
        "phase_visual_captions": phase_nodes,
        "phase_asr": phase_audio,
        "neighbouring_context_visual_captions": context_nodes,
        "neighbouring_context_asr": context_audio,
    }


def validate_stage2(value: Any, payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, dict) or list(value) != STAGE2_FIELDS:
        return {"valid": False, "errors": ["fields_or_order"]}
    if not isinstance(value["phase_summary"], str) or not value["phase_summary"].strip():
        errors.append("empty_phase_summary")
    if not isinstance(value["uncertainty"], list) or not all(isinstance(item, str) for item in value["uncertainty"]):
        errors.append("invalid_uncertainty")
    allowed_mediums = {row["medium_index"] for field in ("phase_visual_captions", "neighbouring_context_visual_captions") for row in payload[field]}
    allowed_audio = {row["audio_id"] for field in ("phase_asr", "neighbouring_context_asr") for row in payload[field]}
    for field in ("direct_visual", "transcript_evidence", "av_interpretation"):
        if not isinstance(value[field], list):
            errors.append(f"{field}:not_array")
    if errors and any(error.endswith("not_array") for error in errors):
        return {"valid": False, "errors": errors}
    for i, claim in enumerate(value["direct_visual"]):
        if not isinstance(claim, dict) or list(claim) != ["claim", "medium_indices"] or not isinstance(claim["claim"], str) or not isinstance(claim["medium_indices"], list) or not claim["medium_indices"]:
            errors.append(f"direct_visual_{i}:invalid")
        elif any(type(item) is not int or item not in allowed_mediums for item in claim["medium_indices"]):
            errors.append(f"direct_visual_{i}:invalid_reference")
    for i, claim in enumerate(value["transcript_evidence"]):
        if not isinstance(claim, dict) or list(claim) != ["claim", "audio_ids"] or not isinstance(claim["claim"], str) or not isinstance(claim["audio_ids"], list) or not claim["audio_ids"]:
            errors.append(f"transcript_{i}:invalid")
        elif any(item not in allowed_audio for item in claim["audio_ids"]):
            errors.append(f"transcript_{i}:invalid_reference")
    for i, claim in enumerate(value["av_interpretation"]):
        if not isinstance(claim, dict) or list(claim) != ["claim", "medium_indices", "audio_ids"] or not claim.get("medium_indices") or not claim.get("audio_ids"):
            errors.append(f"av_{i}:invalid")
        elif any(type(item) is not int or item not in allowed_mediums for item in claim["medium_indices"]) or any(item not in allowed_audio for item in claim["audio_ids"]):
            errors.append(f"av_{i}:invalid_reference")
    return {"valid": not errors, "errors": errors}


def reconstruct_from_stages(index: dict[str, Any], stage1: dict[str, Any], stage2: list[dict[str, Any]]) -> dict[str, Any]:
    discovery = stage1_to_discovery(stage1)
    selections = [{
        "representative_medium_indices": list(row["salient_medium_indices"]),
        "representative_audio_ids": list(row["salient_audio_ids"]),
        "rejected_navigation_noise_medium_indices": [],
        "rejected_navigation_noise_audio_ids": [],
        "selection_uncertainty": [],
    } for row in stage1["phases"]]
    summaries = {"summaries": [{
        "group_index": i,
        "navigation_summary": row["phase_summary"],
        "uncertainty_notes": list(row["uncertainty"]),
    } for i, row in enumerate(stage2)]}
    semantic_map = reconstruct(index, discovery, selections, summaries)
    semantic_map["map_type"] = "r3_2_frozen_method_staged_av_semantic_coarse"
    semantic_map["visual_semantic_source"] = "canonical_qwen_caption_global_phase_then_local_fusion"
    semantic_map["audio_semantic_source"] = "canonical_timestamped_asr_global_phase_then_local_fusion"
    semantic_map["provenance"].update({
        "phase_discovery": "global frozen-method-style AV phase grouping without historical result",
        "phase_evidence_selection": "global Stage 1 salient evidence",
        "navigation_summary": "phase-local Stage 2 conditioned on global phase meaning",
        "stage3_storyline_run": False,
    })
    for i, (coarse, phase, local) in enumerate(zip(semantic_map["coarse_regions"], stage1["phases"], stage2)):
        coarse["map_type"] = semantic_map["map_type"]
        coarse["phase_label"] = phase["phase_label"]
        coarse["boundary_reason"] = phase["boundary_reason"]
        coarse["direct_visual"] = copy.deepcopy(local["direct_visual"])
        coarse["transcript_evidence"] = copy.deepcopy(local["transcript_evidence"])
        coarse["av_interpretation"] = copy.deepcopy(local["av_interpretation"])
    return semantic_map


def no_api_tests(index: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    phases = []
    ends = [4, 9, 14, 19, 24, 29]
    start = 0
    for i, end in enumerate(ends):
        phase_start, phase_end = float(index["medium_nodes"][start]["start_sec"]), float(index["medium_nodes"][end]["end_sec"])
        audio = _overlap(index["audio_nodes"], phase_start, phase_end)
        phases.append({
            "end_medium_index": end, "phase_label": f"phase {i}", "boundary_reason": "meaningful transition",
            "salient_medium_indices": [start], "salient_audio_ids": [audio[0]["audio_id"]] if audio else [],
        })
        start = end + 1
    stage1 = {"phases": phases}
    checks: dict[str, bool] = {"valid_stage1": validate_stage1(stage1, index, config)["valid"]}
    payloads, stage2 = [], []
    start = 0
    for i, phase in enumerate(phases):
        payload = build_stage2_payload(phase, i, start, index, int(config["context_medium_count_each_side"]))
        payloads.append(payload)
        local = {"direct_visual": [{"claim": "visible", "medium_indices": [start]}], "transcript_evidence": [], "av_interpretation": [], "phase_summary": "summary", "uncertainty": []}
        checks[f"valid_stage2_{i}"] = validate_stage2(local, payload)["valid"]
        stage2.append(local)
        start = phase["end_medium_index"] + 1
    semantic_map = reconstruct_from_stages(index, stage1, stage2)
    checks["complete_deterministic_map"] = validate_map(index, semantic_map)["valid"] and canonical_bytes(semantic_map) == canonical_bytes(reconstruct_from_stages(index, stage1, stage2))
    bad = copy.deepcopy(stage1); bad["phases"][0]["phase_label"] = ""
    checks["empty_label_fails"] = not validate_stage1(bad, index, config)["valid"]
    bad_local = copy.deepcopy(stage2[0]); bad_local["direct_visual"][0]["medium_indices"] = [29]
    checks["unknown_local_reference_fails"] = not validate_stage2(bad_local, payloads[0])["valid"]
    generation_contract = STAGE1_PROMPT + STAGE2_PROMPT + canonical_bytes(stage1_schema()).decode("utf-8") + canonical_bytes(stage2_schema()).decode("utf-8")
    checks["historical_result_absent"] = not any(term in generation_contract.lower() for term in ("nine phase", "p01", "initial positioning", "armed confrontation"))
    checks["storyline_absent"] = not semantic_map["storyline_events"] and not semantic_map["has_storyline"]
    checks["hard_filtering_disabled"] = semantic_map["hard_filtering_allowed"] is False
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "model_api_calls": 0}


def prepare(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = load_json(config_path)
    protected_rel = [config["canonical_index"], config["r3_1_control"], config["historical_reference"], config["historical_stage_source"], config["planner_consumer"]]
    protected = {item: sha256_file(root / item) for item in protected_rel}
    if protected[config["canonical_index"]] != config["canonical_index_sha256"]:
        raise RuntimeError("canonical_hash_mismatch")
    index = load_json(root / config["canonical_index"])
    payload = build_payload(index)
    payload["phase_policy"] = {
        "minimum_phase_count": int(config["minimum_phase_count"]),
        "maximum_phase_count": int(config["maximum_phase_count"]),
        "max_salient_visual_per_phase": int(config["max_salient_visual_per_phase"]),
        "max_salient_audio_per_phase": int(config["max_salient_audio_per_phase"]),
    }
    tests = no_api_tests(index, config)
    serialized = canonical_bytes(payload).decode("utf-8").lower()
    checks = {
        "canonical_counts": (len(index["medium_nodes"]), len(index["fine_nodes"]), len(index["audio_nodes"])) == (30, 88, 107),
        "forbidden_input_hits_zero": not forbidden_input_hits(payload),
        "historical_result_absent": not any(term in serialized for term in ("historical", "nine_phase", "phase_id", "p01")),
        "zero_repair": config["model_repair_calls"] == 0,
        "zero_retry": config["semantic_retries"] == 0,
        "storyline_disabled": config["storyline_enabled"] is False,
        "hard_filtering_disabled": config["hard_filtering_allowed"] is False,
    }
    if tests["status"] != "passed" or not all(checks.values()):
        raise RuntimeError(f"no_api_preflight_failed:{tests}:{checks}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input_manifest.json", {
        "canonical_index": config["canonical_index"], "canonical_index_sha256": protected[config["canonical_index"]],
        "caption_hash": hashlib.sha256(canonical_bytes([row["qwen_caption"] for row in index["medium_nodes"]])).hexdigest(),
        "audio_hash": hashlib.sha256(canonical_bytes([_audio_record(row) for row in index["audio_nodes"]])).hexdigest(),
        "medium_count": 30, "fine_count": 88, "audio_count": 107,
        "model": config["model"], "temperature": config["temperature"],
        "historical_phase_scaffold_used": False,
        "expected_calls": "1 global Stage 1 + one Stage 2 call per discovered phase",
    })
    write_json(output / "source_artifact_audit.json", {"status": "passed", "protected_hashes_before": protected, "checks": checks})
    write_json(output / "stage1_input_payload.json", payload)
    write_json(output / "stage1_schema.json", stage1_schema())
    write_json(output / "stage2_schema.json", stage2_schema())
    write_json(output / "no_api_test_report.json", tests)
    return {"config": config, "index": index, "payload": payload, "protected": protected, "tests": tests}


def _parse(raw: str) -> tuple[Any, str | None]:
    try:
        return json.loads(raw), None
    except Exception as error:
        return None, f"{type(error).__name__}:{error}"


def _fail(output: Path, stage: str, usages: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    report = {"failed_stage": stage, "overall_validation": f"{stage}_first_pass_invalid", "errors": errors, "model_calls": len(usages), "repair_calls": 0, "semantic_retries": 0, "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0, "final_answer_calls": 0}
    write_json(output / "validation_report.json", report)
    write_json(output / "cost_accounting.json", {"calls": usages, "model_calls": len(usages), "repair_calls": 0, "semantic_retries": 0})
    (output / "REPORT.md").write_text(f"# R3_2 frozen-method staged AV Organizer\n\n{stage} first-pass output invalid; no repair or retry.\n", encoding="utf-8", newline="\n")
    return report


def _noise_hits(value: dict[str, Any]) -> list[str]:
    terms = ("snowy mountain", "snow-covered ground", "clouds moving", "bird feeder", "furry object", "white bucket", "black cloth")
    text = canonical_bytes(value).decode("utf-8").lower()
    return [term for term in terms if term in text]


def run(root: Path, config_path: Path, output: Path, api_key: str | None) -> dict[str, Any]:
    prepared = prepare(root, config_path, output)
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY_missing_before_model_call")
    config, index = prepared["config"], prepared["index"]
    usages: list[dict[str, Any]] = []
    raw1, usage1 = _call(api_key, config, STAGE1_PROMPT, prepared["payload"], stage1_schema(), "stage1_max_tokens")
    usages.append({"stage": "global_phase_grouping", **usage1})
    value1, parse1 = _parse(raw1)
    audit1 = validate_stage1(value1, index, config) if value1 is not None else {"valid": False, "errors": ["json_parse_failed"], "phase_count": 0, "end_indices": []}
    write_json(output / "stage1_raw_response.json", {"raw_response": raw1, "usage": usage1, "parse_error": parse1, "attempt_count": 1})
    write_json(output / "stage1_global_phases.json", value1 if isinstance(value1, dict) else {"status": "not_available"})
    write_json(output / "stage1_validation.json", audit1)
    if not audit1["valid"]:
        return _fail(output, "stage1", usages, audit1["errors"])

    payloads, raw_records, results, audits = [], [], [], []
    start = 0
    for phase_index, phase in enumerate(value1["phases"]):
        payload = build_stage2_payload(phase, phase_index, start, index, int(config["context_medium_count_each_side"]))
        payloads.append(payload)
        raw, usage = _call(api_key, config, STAGE2_PROMPT, payload, stage2_schema(), "stage2_max_tokens")
        usages.append({"stage": f"local_phase_fusion_{phase_index}", **usage})
        value, parse_error = _parse(raw)
        audit = validate_stage2(value, payload) if value is not None else {"valid": False, "errors": ["json_parse_failed"]}
        raw_records.append({"phase_index": phase_index, "raw_response": raw, "usage": usage, "parse_error": parse_error, "attempt_count": 1})
        audits.append({"phase_index": phase_index, **audit})
        if not audit["valid"]:
            write_json(output / "stage2_input_payloads.json", payloads)
            write_json(output / "stage2_raw_responses.json", raw_records)
            write_json(output / "stage2_validation.json", audits)
            return _fail(output, f"stage2_{phase_index}", usages, audit["errors"])
        results.append(value)
        start = phase["end_medium_index"] + 1
    write_json(output / "stage2_input_payloads.json", payloads)
    write_json(output / "stage2_raw_responses.json", raw_records)
    write_json(output / "stage2_phase_accounts.json", results)
    write_json(output / "stage2_validation.json", {"status": "passed", "audits": audits})

    semantic_map = reconstruct_from_stages(index, value1, results)
    map_audit = validate_map(index, semantic_map)
    view = planner_view(semantic_map)
    for planner_row, map_row in zip(view["coarse_regions"], semantic_map["coarse_regions"]):
        planner_row["event_label"] = map_row["phase_label"]
    planner_audit = validate_planner_view(_strict_view(view), index)
    write_json(output / "r3_2_frozen_method_denoised_av_semantic_map.json", semantic_map)
    write_json(output / "deterministic_reconstruction_audit.json", map_audit)
    write_json(output / "planner_compatibility_view.json", view)
    write_json(output / "planner_compatibility_report.json", {**planner_audit, "planner_calls": 0, "all_mediums_remain_eligible": True, "coarse_prior_affects_ranking": False})

    r31 = load_json(root / config["r3_1_control"])
    r31_hits, r32_hits = _noise_hits(r31), _noise_hits(view)
    comparison = {"same_canonical_caption_asr_input": True, "r3_1": {"coarse_count": len(r31["coarse_regions"]), "posthoc_navigation_noise_terms": r31_hits}, "r3_2": {"coarse_count": len(semantic_map["coarse_regions"]), "posthoc_navigation_noise_terms": r32_hits}, "planned_difference": "frozen-method global phase meaning is propagated into local fusion", "posthoc_terms_not_used_in_generation": True}
    write_json(output / "r3_1_vs_r3_2_comparison.json", comparison)
    write_json(output / "navigation_noise_audit.json", {"status": "passed" if not r32_hits else "flagged_for_manual_review", "posthoc_terms_found": r32_hits, "terms_not_used_in_prompt_or_generation": True})

    historical = load_json(root / config["historical_reference"])
    write_json(output / "historical_posthoc_comparison.json", {"candidate_generated_without_historical_phase_content": True, "comparison_loaded_only_after_candidate_reconstruction": True, "candidate_group_count": len(semantic_map["coarse_regions"]), "historical_group_count": len(historical.get("phases", [])), "historical_group_count_is_not_acceptance_target": True})
    after = {item: sha256_file(root / item) for item in prepared["protected"]}
    unchanged = after == prepared["protected"]
    write_json(output / "protected_hash_audit.json", {"status": "passed" if unchanged else "failed", "before": prepared["protected"], "after": after, "unchanged": unchanged})
    structural = map_audit["valid"] and planner_audit["valid"] and unchanged
    report = {
        "source_validation": "passed" if unchanged else "failed", "historical_phase_independence_validation": "passed",
        "stage1_global_validation": "passed", "stage2_local_fusion_validation": "passed",
        "deterministic_reconstruction_validation": "passed" if map_audit["valid"] else "failed",
        "planner_compatibility_validation": "passed" if planner_audit["valid"] else "failed",
        "navigation_noise_audit": "passed" if not r32_hits else "flagged_for_manual_review",
        "semantic_acceptance": "pending_manual_review", "overall_validation": "pending_manual_semantic_review" if structural else "failed_structural_validation",
        "recommendation": "ready_for_r3_2_frozen_method_manual_review" if structural else "r3_2_frozen_method_contract_failure",
        "coarse_count": len(semantic_map["coarse_regions"]), "end_indices": audit1["end_indices"], "phase_labels": [row["phase_label"] for row in value1["phases"]],
        "medium_count": map_audit["medium_count"], "fine_count": map_audit["fine_count"], "storyline_count": 0, "hard_filtering_allowed": False,
        "all_mediums_retrieval_eligible": True, "model_calls": len(usages), "repair_calls": 0, "semantic_retries": 0,
        "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0, "final_answer_calls": 0,
    }
    write_json(output / "validation_report.json", report)
    write_json(output / "cost_accounting.json", {"calls": usages, "model_calls": len(usages), "input_tokens": sum(row["input_tokens"] for row in usages), "output_tokens": sum(row["output_tokens"] for row in usages), "summed_latency_sec": sum(row["latency_sec"] for row in usages), "repair_calls": 0, "semantic_retries": 0, "all_other_model_calls": 0})
    sections = "".join(f"<section><h2>{html.escape(row['coarse_id'])} — {html.escape(row['phase_label'])} ({row['start_sec']:.3f}-{row['end_sec']:.3f}s)</h2><p>{html.escape(row['navigation_summary'])}</p><p><b>Boundary:</b> {html.escape(row['boundary_reason'])}</p><p><b>Mediums:</b> {html.escape(', '.join(row['source_medium_ids']))}</p><p><b>Salient visual/audio:</b> {html.escape(', '.join(row['representative_medium_ids']))} / {html.escape(', '.join(row['representative_audio_ids']) or 'none')}</p><details><summary>Grounded phase account</summary><pre>{html.escape(json.dumps({'direct_visual': row['direct_visual'], 'transcript_evidence': row['transcript_evidence'], 'av_interpretation': row['av_interpretation'], 'uncertainty': row['uncertainty_notes']}, ensure_ascii=False, indent=2))}</pre></details></section>" for row in semantic_map["coarse_regions"])
    (output / "review.html").write_text("<!doctype html><meta charset='utf-8'><title>R3_2 frozen-method review</title><style>body{font:14px system-ui;margin:24px;line-height:1.5}section{border-top:2px solid #567;padding:12px}pre{white-space:pre-wrap}</style><h1>R3_2 frozen-method staged AV map</h1><p>Global phase meaning was propagated into local fusion. Historical phase content was absent.</p>" + sections + f"<h2>R3_1 comparison</h2><pre>{html.escape(json.dumps(comparison, indent=2))}</pre>", encoding="utf-8", newline="\n")
    (output / "REPORT.md").write_text("\n".join(["# R3_2 frozen-method staged AV Organizer canary", "", "- Historical phase output used in generation: `false`", f"- Independently generated phases: `{report['coarse_count']}`", f"- End indices: `{report['end_indices']}`", f"- Labels: `{report['phase_labels']}`", f"- Medium/Fine coverage: `{report['medium_count']}/30`, `{report['fine_count']}/88`", f"- R3_1 post-hoc noise terms: `{r31_hits}`", f"- R3_2 post-hoc noise terms: `{r32_hits}`", "- Storyline: `0`; hard filtering: `false`; all Mediums eligible: `true`", f"- Calls: `{report['model_calls']}`; repairs/retries: `0/0`", "- Planner compatibility: `passed` without a Planner call", "- Semantic acceptance: `pending_manual_review`", "", "Planner, Retrieval, Sufficiency, temporal review, Final Gemini, and QA were not run."]) + "\n", encoding="utf-8", newline="\n")
    return report
