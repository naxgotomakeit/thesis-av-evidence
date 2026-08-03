from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

from experiments.r3_v2_coarse_semantic_organizer.core import canonical_bytes, load_json, sha256_file, validate_planner_view, write_json
from experiments.r3_v2_av_coarse_semantic_organizer_v1_1.core import _audio_record
from experiments.r3_2_independent_global_phase_denoised_av_organizer.core import _strict_view, planner_view, validate_map
from experiments.r3_2_frozen_method_staged_av_organizer.core import (
    build_stage2_payload,
    no_api_tests as staged_no_api_tests,
    reconstruct_from_stages,
    stage1_schema,
    stage2_schema,
    validate_stage1,
    validate_stage2,
)


SYSTEM_PROMPT = (
    "Use only the supplied canonical repaired visual captions, enhanced-ASR TRANSCRIPT nodes, "
    "timestamps, and validated phase records. Every transcript has uncertain speaker/source. "
    "Never use ground truth, manual annotations, questions, answers, historical phase output, or external knowledge."
)

STAGE1_PROMPT = """You are segmenting a chronological audio-visual evidence stream from first-person body-camera footage into a small number of meaningful incident phases.

The input contains timestamped visual observations and timestamped speech transcripts. Use both modalities to identify major changes in activity, interaction, spoken commands, apparent scene purpose, restraint, injury response, or other meaningful operational phases. Audio may reveal a phase transition that is visually subtle. Visual evidence may provide context for nearby speech.

Rules:
- Produce between 5 and 12 chronological phases.
- Phase boundaries must reflect meaningful activity changes, not every minor visual or speech change.
- Do not create a separate phase for incidental objects or repeated scenery.
- Do not infer legal status, guilt, motive, or identities not supported by the evidence.
- Temporal proximity alone does not prove causality.
- Treat transcript content as heard text with uncertain speaker/source.
- Keep unsupported interpretations uncertain.
- Do not list every node ID.
- For each phase, cite only a few salient visual and transcript node IDs that motivated the boundary.

For this Medium-indexed deterministic adapter, return only:
{
  "phases": [{
    "end_medium_index": 0,
    "phase_label": "...",
    "boundary_reason": "...",
    "salient_medium_indices": [],
    "salient_audio_ids": []
  }]
}
The first phase starts implicitly at Medium 0, later phases start after the prior end, end indices strictly increase, and the final end is 29. Do not output start indices, phase IDs, timestamps, Coarse IDs, Storyline, retrieval decisions, or answers."""

STAGE2_PROMPT = """You are analysing one chronological phase from first-person body-camera footage.

You are given a validated global phase record, visual observations, timestamped speech transcripts, and limited neighbouring context. Combine the modalities into a concise factual phase account. The global phase label and boundary reason are the semantic context for deciding what matters locally.

Distinguish between:
1. DIRECT_VISUAL: what is explicitly stated by visual captions,
2. TRANSCRIPT_EVIDENCE: what is heard in the transcript,
3. AV_INTERPRETATION: an optional interpretation supported jointly by nearby visual and transcript evidence,
4. UNCERTAINTY: what remains unclear.

Rules:
- AV_INTERPRETATION is optional and must cite visual and audio evidence.
- Do not force a cross-modal interpretation.
- Temporal proximity alone does not establish causality.
- Do not infer guilt, motive, legal status, danger level, or unsupported identities.
- Preserve conflicts between modalities.
- Ignore incidental scenery unless needed to understand the event.
- Retain important actions, spoken commands, weapons, restraint, injury, medical response and changes in interaction.
- Evidence references must exist in the supplied phase or neighbouring context.

Return only direct_visual, transcript_evidence, av_interpretation, phase_summary, and uncertainty. Do not output phase IDs, boundaries, Storyline, retrieval decisions, or answers."""


def build_interleaved_nodes(index: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for medium_index, medium in enumerate(index["medium_nodes"]):
        nodes.append({
            "id": medium["medium_id"], "medium_index": medium_index,
            "start_sec": float(medium["start_sec"]), "end_sec": float(medium["end_sec"]),
            "modality": "visual", "caption": medium["qwen_caption"],
        })
    for audio in index["audio_nodes"]:
        nodes.append({
            "id": audio["audio_id"], "start_sec": float(audio["start_sec"]), "end_sec": float(audio["end_sec"]),
            "modality": "audio", "transcript": audio["transcript"], "source_type": "unclear",
        })
    return sorted(nodes, key=lambda row: (row["start_sec"], 0 if row["modality"] == "visual" else 1, row["id"]))


def build_stage1_user_text(index: dict[str, Any]) -> str:
    nodes = build_interleaved_nodes(index)
    coverage = {"start_sec": min(row["start_sec"] for row in nodes), "end_sec": max(row["end_sec"] for row in nodes)}
    return STAGE1_PROMPT + "\n\nVIDEO COVERAGE:\n" + json.dumps(coverage, separators=(",", ":")) + "\n\nINTERLEAVED NODES:\n" + json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))


def _call_text(api_key: str, config: dict[str, Any], user_text: str, schema: dict[str, Any], max_tokens_key: str) -> tuple[str, dict[str, Any]]:
    import anthropic

    started = time.perf_counter()
    response = anthropic.Anthropic(api_key=api_key).messages.create(
        model=config["model"], max_tokens=int(config[max_tokens_key]), temperature=float(config["temperature"]),
        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_text}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    raw = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return raw, {"provider": "anthropic", "model": config["model"], "input_tokens": int(response.usage.input_tokens), "output_tokens": int(response.usage.output_tokens), "latency_sec": time.perf_counter() - started, "stop_reason": response.stop_reason, "response_id": str(response.id), "request_id": str(getattr(response, "_request_id", "") or "")}


def no_api_tests(index: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    base = staged_no_api_tests(index, config)
    nodes = build_interleaved_nodes(index)
    visual = [row for row in nodes if row["modality"] == "visual"]
    audio = [row for row in nodes if row["modality"] == "audio"]
    checks = dict(base["checks"])
    checks["interleaved_counts"] = (len(visual), len(audio), len(nodes)) == (30, 107, 137)
    checks["canonical_caption_byte_identity"] = [row["caption"] for row in visual] == [row["qwen_caption"] for row in index["medium_nodes"]]
    checks["canonical_asr_byte_identity"] = [row["transcript"] for row in audio] == [row["transcript"] for row in index["audio_nodes"]]
    checks["each_node_occurs_once"] = len({row["id"] for row in nodes}) == 137
    checks["time_ordered"] = [row["start_sec"] for row in nodes] == sorted(row["start_sec"] for row in nodes)
    text = build_stage1_user_text(index).lower()
    checks["historical_result_absent_from_prompt"] = not any(term in text for term in ("initial positioning and dispatch", "p01", "nine phase", "historical phase"))
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "model_api_calls": 0}


def prepare(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = load_json(config_path)
    protected_rel = [config["canonical_index"], config["r3_1_control"], config["historical_reference"], config["historical_interleaved_timeline"], config["historical_stage_source"], config["planner_consumer"]]
    protected = {item: sha256_file(root / item) for item in protected_rel}
    if protected[config["canonical_index"]] != config["canonical_index_sha256"]:
        raise RuntimeError("canonical_hash_mismatch")
    index = load_json(root / config["canonical_index"])
    tests = no_api_tests(index, config)
    if tests["status"] != "passed":
        raise RuntimeError(f"no_api_preflight_failed:{tests}")
    output.mkdir(parents=True, exist_ok=True)
    user_text = build_stage1_user_text(index)
    write_json(output / "input_manifest.json", {
        "canonical_index": config["canonical_index"], "canonical_index_sha256": protected[config["canonical_index"]],
        "caption_hash": hashlib.sha256(canonical_bytes([row["qwen_caption"] for row in index["medium_nodes"]])).hexdigest(),
        "audio_hash": hashlib.sha256(canonical_bytes([_audio_record(row) for row in index["audio_nodes"]])).hexdigest(),
        "visual_node_count": 30, "audio_node_count": 107, "interleaved_node_count": 137,
        "historical_caption_text_reused": False, "historical_phase_output_used": False,
        "canonical_repaired_caption_text_used": True, "model": config["model"], "temperature": config["temperature"],
    })
    write_json(output / "source_artifact_audit.json", {"status": "passed", "protected_hashes_before": protected})
    write_json(output / "interleaved_input_audit.json", {"status": "passed", "node_count": 137, "visual_count": 30, "audio_count": 107, "each_node_once": True, "time_ordered": True})
    (output / "stage1_input.txt").write_text(user_text, encoding="utf-8", newline="\n")
    write_json(output / "stage1_schema.json", stage1_schema())
    write_json(output / "stage2_schema.json", stage2_schema())
    write_json(output / "no_api_test_report.json", tests)
    return {"config": config, "index": index, "user_text": user_text, "protected": protected, "tests": tests}


def _parse(raw: str) -> tuple[Any, str | None]:
    try:
        return json.loads(raw), None
    except Exception as error:
        return None, f"{type(error).__name__}:{error}"


def _fail(output: Path, stage: str, usages: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    report = {"failed_stage": stage, "overall_validation": f"{stage}_first_pass_invalid", "errors": errors, "model_calls": len(usages), "repair_calls": 0, "semantic_retries": 0, "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0, "final_answer_calls": 0}
    write_json(output / "validation_report.json", report)
    write_json(output / "cost_accounting.json", {"calls": usages, "model_calls": len(usages), "repair_calls": 0, "semantic_retries": 0})
    (output / "REPORT.md").write_text(f"# R3_2 frozen exact-interleaved Organizer\n\n{stage} first-pass output invalid; no repair or retry.\n", encoding="utf-8", newline="\n")
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
    raw1, usage1 = _call_text(api_key, config, prepared["user_text"], stage1_schema(), "stage1_max_tokens")
    usages.append({"stage": "global_interleaved_phase_grouping", **usage1})
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
        user_text = STAGE2_PROMPT + "\n\nPHASE INPUT:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        raw, usage = _call_text(api_key, config, user_text, stage2_schema(), "stage2_max_tokens")
        usages.append({"stage": f"local_phase_fusion_{phase_index}", **usage})
        value, parse_error = _parse(raw)
        audit = validate_stage2(value, payload) if value is not None else {"valid": False, "errors": ["json_parse_failed"]}
        raw_records.append({"phase_index": phase_index, "raw_response": raw, "usage": usage, "parse_error": parse_error, "attempt_count": 1})
        audits.append({"phase_index": phase_index, **audit})
        if not audit["valid"]:
            write_json(output / "stage2_input_payloads.json", payloads); write_json(output / "stage2_raw_responses.json", raw_records); write_json(output / "stage2_validation.json", audits)
            return _fail(output, f"stage2_{phase_index}", usages, audit["errors"])
        results.append(value)
        start = phase["end_medium_index"] + 1
    write_json(output / "stage2_input_payloads.json", payloads); write_json(output / "stage2_raw_responses.json", raw_records); write_json(output / "stage2_phase_accounts.json", results); write_json(output / "stage2_validation.json", {"status": "passed", "audits": audits})

    semantic_map = reconstruct_from_stages(index, value1, results)
    semantic_map["map_type"] = "r3_2_frozen_exact_interleaved_av_semantic_coarse"
    semantic_map["visual_semantic_source"] = "canonical_repaired_qwen_caption_interleaved_once"
    semantic_map["audio_semantic_source"] = "canonical_timestamped_asr_interleaved_once"
    semantic_map["provenance"].update({"phase_discovery": "historical-method global interleaved AV grouping without historical output", "navigation_summary": "historical-method local fusion conditioned on global phase meaning", "stage3_storyline_run": False})
    for coarse in semantic_map["coarse_regions"]:
        coarse["map_type"] = semantic_map["map_type"]
    map_audit = validate_map(index, semantic_map)
    view = planner_view(semantic_map)
    for planner_row, map_row, phase in zip(view["coarse_regions"], semantic_map["coarse_regions"], value1["phases"]):
        planner_row["event_label"] = phase["phase_label"]
        map_row["phase_label"] = phase["phase_label"]
        map_row["boundary_reason"] = phase["boundary_reason"]
    planner_audit = validate_planner_view(_strict_view(view), index)
    write_json(output / "r3_2_frozen_exact_interleaved_av_semantic_map.json", semantic_map); write_json(output / "deterministic_reconstruction_audit.json", map_audit); write_json(output / "planner_compatibility_view.json", view); write_json(output / "planner_compatibility_report.json", {**planner_audit, "planner_calls": 0, "all_mediums_remain_eligible": True, "coarse_prior_affects_ranking": False})

    r31 = load_json(root / config["r3_1_control"]); r31_hits, r32_hits = _noise_hits(r31), _noise_hits(view)
    comparison = {"same_canonical_caption_asr_content": True, "r3_1": {"coarse_count": len(r31["coarse_regions"]), "posthoc_navigation_noise_terms": r31_hits}, "r3_2": {"coarse_count": len(semantic_map["coarse_regions"]), "posthoc_navigation_noise_terms": r32_hits}, "planned_difference": "historical global interleaved phase method and local semantic handoff", "posthoc_terms_not_used_in_generation": True}
    write_json(output / "r3_1_vs_r3_2_comparison.json", comparison); write_json(output / "navigation_noise_audit.json", {"status": "passed" if not r32_hits else "flagged_for_manual_review", "posthoc_terms_found": r32_hits, "terms_not_used_in_prompt_or_generation": True})
    historical = load_json(root / config["historical_reference"]); write_json(output / "historical_posthoc_comparison.json", {"candidate_generated_without_historical_phase_content": True, "comparison_loaded_only_after_candidate_reconstruction": True, "candidate_group_count": len(semantic_map["coarse_regions"]), "historical_group_count": len(historical.get("phases", [])), "historical_group_count_is_not_acceptance_target": True})
    after = {item: sha256_file(root / item) for item in prepared["protected"]}; unchanged = after == prepared["protected"]
    write_json(output / "protected_hash_audit.json", {"status": "passed" if unchanged else "failed", "before": prepared["protected"], "after": after, "unchanged": unchanged})
    structural = map_audit["valid"] and planner_audit["valid"] and unchanged
    report = {"source_validation": "passed" if unchanged else "failed", "canonical_interleaved_input_validation": "passed", "historical_phase_independence_validation": "passed", "stage1_global_validation": "passed", "stage2_local_fusion_validation": "passed", "deterministic_reconstruction_validation": "passed" if map_audit["valid"] else "failed", "planner_compatibility_validation": "passed" if planner_audit["valid"] else "failed", "navigation_noise_audit": "passed" if not r32_hits else "flagged_for_manual_review", "semantic_acceptance": "pending_manual_review", "overall_validation": "pending_manual_semantic_review" if structural else "failed_structural_validation", "recommendation": "ready_for_r3_2_exact_method_manual_review" if structural else "r3_2_exact_method_contract_failure", "coarse_count": len(semantic_map["coarse_regions"]), "end_indices": audit1["end_indices"], "phase_labels": [row["phase_label"] for row in value1["phases"]], "medium_count": map_audit["medium_count"], "fine_count": map_audit["fine_count"], "storyline_count": 0, "hard_filtering_allowed": False, "all_mediums_retrieval_eligible": True, "model_calls": len(usages), "repair_calls": 0, "semantic_retries": 0, "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0, "final_answer_calls": 0}
    write_json(output / "validation_report.json", report); write_json(output / "cost_accounting.json", {"calls": usages, "model_calls": len(usages), "input_tokens": sum(row["input_tokens"] for row in usages), "output_tokens": sum(row["output_tokens"] for row in usages), "summed_latency_sec": sum(row["latency_sec"] for row in usages), "repair_calls": 0, "semantic_retries": 0, "all_other_model_calls": 0})
    sections = "".join(f"<section><h2>{html.escape(row['coarse_id'])} — {html.escape(phase['phase_label'])} ({row['start_sec']:.3f}-{row['end_sec']:.3f}s)</h2><p>{html.escape(row['navigation_summary'])}</p><p><b>Boundary:</b> {html.escape(phase['boundary_reason'])}</p><p><b>Mediums:</b> {html.escape(', '.join(row['source_medium_ids']))}</p></section>" for row, phase in zip(semantic_map["coarse_regions"], value1["phases"]))
    (output / "review.html").write_text("<!doctype html><meta charset='utf-8'><title>R3_2 exact interleaved review</title><style>body{font:14px system-ui;margin:24px;line-height:1.5}section{border-top:2px solid #567;padding:12px}</style><h1>R3_2 frozen exact-interleaved map</h1>" + sections + f"<pre>{html.escape(json.dumps(comparison, indent=2))}</pre>", encoding="utf-8", newline="\n")
    (output / "REPORT.md").write_text("\n".join(["# R3_2 frozen exact-interleaved AV Organizer canary", "", "- Current canonical captions + ASR, each interleaved exactly once.", "- Historical phase output used in generation: `false`", f"- Phases/end indices: `{report['coarse_count']}/{report['end_indices']}`", f"- Labels: `{report['phase_labels']}`", f"- Coverage: `{report['medium_count']}/30 Medium`, `{report['fine_count']}/88 Fine`", f"- R3_1 noise terms: `{r31_hits}`", f"- R3_2 noise terms: `{r32_hits}`", "- Storyline: `0`; hard filtering: `false`; all Mediums eligible: `true`", f"- Calls: `{report['model_calls']}`; repairs/retries: `0/0`", "- Semantic acceptance: `pending_manual_review`", "", "Planner, Retrieval, Sufficiency, temporal review, Final Gemini, and QA were not run."]) + "\n", encoding="utf-8", newline="\n")
    return report
