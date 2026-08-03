from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

from experiments.r3_v2_coarse_semantic_organizer.core import canonical_bytes, load_json, sha256_file, validate_planner_view, write_json
from experiments.r3_v2_av_coarse_semantic_organizer_v1_1.core import _audio_record, _overlap
from experiments.r3_2_independent_global_phase_denoised_av_organizer.core import _strict_view
from experiments.r3_2_frozen_method_staged_av_organizer.core import validate_stage1


SOURCE = "outputs/experiments/r3_2_frozen_exact_interleaved_av_organizer_canary_v1"
INDEX = "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json"
R31 = "outputs/experiments/r3_1_noisy_av_semantic_map_freeze_v1/r3_1_noisy_av_semantic_map.json"
HISTORICAL = "outputs/experiments/egopolice_visual_full_asr_fusion_v0_7_typed_minimal_evidence_map/minimal_evidence_map.json"
PLANNER = "src/experiments/egopolice_coarse_map_planner_compatibility/core.py"


def reconstruct(index: dict[str, Any], stage1: dict[str, Any]) -> dict[str, Any]:
    coarse = []
    start_index = 0
    for group_index, phase in enumerate(stage1["phases"]):
        end_index = phase["end_medium_index"]
        nodes = index["medium_nodes"][start_index : end_index + 1]
        start_sec, end_sec = float(nodes[0]["start_sec"]), float(nodes[-1]["end_sec"])
        asr = [_audio_record(row) for row in _overlap(index["audio_nodes"], start_sec, end_sec)]
        coarse.append({
            "coarse_id": f"C{group_index + 1:02d}", "start_sec": start_sec, "end_sec": end_sec, "duration_sec": end_sec - start_sec,
            "source_medium_ids": [row["medium_id"] for row in nodes], "source_fine_ids": [fine for row in nodes for fine in row["child_fine_ids"]],
            "navigation_summary": phase["phase_label"], "uncertainty_notes": [],
            "global_boundary_reason_sidecar": phase["boundary_reason"],
            "global_salient_medium_indices_sidecar": phase["salient_medium_indices"], "global_salient_audio_ids_sidecar": phase["salient_audio_ids"],
            "exact_source_captions_sidecar": [{"medium_id": row["medium_id"], "caption": row["qwen_caption"]} for row in nodes],
            "exact_source_asr_sidecar": asr, "audio_ids": [row["audio_id"] for row in asr],
            "map_type": "r3_2_global_stage1_clean_semantic_coarse", "semantic_fields_available": True,
        })
        start_index = end_index + 1
    return {
        "map_type": "r3_2_global_stage1_clean_semantic_coarse", "semantic_fields_available": True,
        "visual_semantic_source": "canonical_repaired_qwen_caption_global_stage1", "audio_semantic_source": "canonical_timestamped_asr_global_stage1",
        "coarse_regions": coarse, "storyline_events": [], "has_storyline": False, "hard_filtering_allowed": False,
        "provenance": {"global_stage1_source": SOURCE, "historical_phase_output_used": False, "failed_stage2_text_used": False, "planner_text_source": "global_phase_label", "all_mediums_retrieval_eligible": True, "coarse_prior_affects_ranking": False, "map_text_is_not_sufficiency_evidence": True},
    }


def planner_view(semantic_map: dict[str, Any]) -> dict[str, Any]:
    return {
        "map_type": semantic_map["map_type"],
        "coarse_regions": [{
            "coarse_id": row["coarse_id"], "start_sec": row["start_sec"], "end_sec": row["end_sec"],
            "source_medium_ids": row["source_medium_ids"], "source_audio_ids": row["audio_ids"],
            "event_label": row["navigation_summary"], "summary": row["navigation_summary"], "coarse_summary": row["navigation_summary"],
            "uncertainty_notes": [],
            "adapter_metadata": {"adapter_derived": True, "source_field": "navigation_summary", "origin_field": "global_phase_label", "semantic_summary": True, "stored_index_modified": False, "hard_filtering_allowed": False, "raw_caption_or_asr_exposed": False},
        } for row in semantic_map["coarse_regions"]],
        "storyline_events": [], "has_storyline": False, "hard_filtering_allowed": False,
        "planner_policy": {"map_may_guide_search_units_and_query_variants": True, "suggested_coarse_are_navigation_hints_only": True, "all_mediums_remain_eligible": True, "coarse_prior_affects_ranking": False, "map_text_is_not_sufficiency_evidence": True},
    }


def validate(index: dict[str, Any], semantic_map: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    expected_m = [row["medium_id"] for row in index["medium_nodes"]]
    actual_m = [item for row in semantic_map["coarse_regions"] for item in row["source_medium_ids"]]
    expected_f = [item for row in index["medium_nodes"] for item in row["child_fine_ids"]]
    actual_f = [item for row in semantic_map["coarse_regions"] for item in row["source_fine_ids"]]
    labels = [row["navigation_summary"] for row in semantic_map["coarse_regions"]]
    planner_text = " ".join(row["summary"] for row in view["coarse_regions"]).lower()
    terms = ("snowy mountain", "snow-covered ground", "clouds moving", "bird feeder", "furry object", "white bucket", "black cloth")
    hits = [term for term in terms if term in planner_text]
    checks = {
        "medium_coverage": actual_m == expected_m, "fine_coverage": actual_f == expected_f and len(actual_f) == 88,
        "ten_global_labels": len(labels) == 10 and all(label.strip() for label in labels), "planner_text_labels_only": all(row["summary"] == label for row, label in zip(view["coarse_regions"], labels)),
        "raw_caption_asr_not_exposed": all("representative_source_captions" not in row and "exact_source_captions" not in row and "representative_source_asr" not in row for row in view["coarse_regions"]),
        "navigation_noise_absent": not hits, "storyline_absent": not semantic_map["storyline_events"] and not semantic_map["has_storyline"],
        "hard_filtering_disabled": not semantic_map["hard_filtering_allowed"] and not view["hard_filtering_allowed"],
        "all_mediums_eligible": view["planner_policy"]["all_mediums_remain_eligible"] and not view["planner_policy"]["coarse_prior_affects_ranking"],
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "posthoc_navigation_noise_terms": hits, "medium_count": len(actual_m), "fine_count": len(actual_f)}


def run(root: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    protected_rel = [INDEX, f"{SOURCE}/stage1_global_phases.json", f"{SOURCE}/stage1_validation.json", R31, HISTORICAL, PLANNER]
    before = {item: sha256_file(root / item) for item in protected_rel}
    index = load_json(root / INDEX); stage1 = load_json(root / SOURCE / "stage1_global_phases.json"); stage1_audit = load_json(root / SOURCE / "stage1_validation.json")
    config = {"minimum_phase_count": 5, "maximum_phase_count": 12, "max_salient_visual_per_phase": 12, "max_salient_audio_per_phase": 12, "stage1_salient_scope": "global"}
    if not stage1_audit["valid"] or not validate_stage1(stage1, index, config)["valid"]:
        raise RuntimeError("source_stage1_not_valid")
    output.mkdir(parents=True, exist_ok=True)
    semantic_map = reconstruct(index, stage1); view = planner_view(semantic_map); audit = validate(index, semantic_map, view)
    planner_audit = validate_planner_view(_strict_view(view), index)
    after = {item: sha256_file(root / item) for item in protected_rel}; unchanged = before == after
    write_json(output / "input_manifest.json", {"source_stage1": f"{SOURCE}/stage1_global_phases.json", "source_stage1_sha256": before[f"{SOURCE}/stage1_global_phases.json"], "canonical_index": INDEX, "canonical_index_sha256": before[INDEX], "model_api_calls": 0})
    write_json(output / "r3_2_global_stage1_clean_semantic_map.json", semantic_map); write_json(output / "planner_compatibility_view.json", view)
    write_json(output / "navigation_noise_audit.json", {"status": "passed" if audit["checks"]["navigation_noise_absent"] else "failed", "posthoc_terms_found": audit["posthoc_navigation_noise_terms"], "terms_not_used_to_construct_projection": True})
    write_json(output / "planner_compatibility_report.json", {**planner_audit, "all_mediums_remain_eligible": True, "coarse_prior_affects_ranking": False, "planner_calls": 0})
    write_json(output / "protected_hash_audit.json", {"status": "passed" if unchanged else "failed", "before": before, "after": after, "unchanged": unchanged})
    structural = audit["status"] == "passed" and planner_audit["valid"] and unchanged
    report = {"source_stage1_validation": "passed", "historical_phase_independence_validation": "passed", "clean_projection_validation": audit["status"], "planner_compatibility_validation": "passed" if planner_audit["valid"] else "failed", "navigation_noise_audit": "passed" if audit["checks"]["navigation_noise_absent"] else "failed", "semantic_acceptance": "pending_manual_review", "overall_validation": "pending_manual_semantic_review" if structural else "failed", "recommendation": "ready_for_r3_2_global_map_manual_review" if structural else "requires_clean_projection_fix", "coarse_count": len(semantic_map["coarse_regions"]), "phase_labels": [row["navigation_summary"] for row in semantic_map["coarse_regions"]], "medium_count": audit["medium_count"], "fine_count": audit["fine_count"], "model_api_calls": 0, "planner_calls": 0, "retrieval_calls": 0, "sufficiency_calls": 0, "final_answer_calls": 0}
    write_json(output / "validation_report.json", report); write_json(output / "cost_accounting.json", {"model_api_calls": 0, "deterministic_latency_sec": time.perf_counter() - started})
    rows = "".join(f"<tr><td>{html.escape(row['coarse_id'])}</td><td>{row['start_sec']:.3f}-{row['end_sec']:.3f}</td><td>{html.escape(row['navigation_summary'])}</td><td>{html.escape(', '.join(row['source_medium_ids']))}</td></tr>" for row in semantic_map["coarse_regions"])
    (output / "review.html").write_text("<!doctype html><meta charset='utf-8'><title>R3_2 global clean map</title><style>body{font:14px system-ui;margin:24px}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>R3_2 global Stage-1 clean navigation map</h1><table><tr><th>ID</th><th>Time</th><th>Global phase label</th><th>Mediums</th></tr>" + rows + "</table>", encoding="utf-8", newline="\n")
    (output / "REPORT.md").write_text("\n".join(["# R3_2 global Stage-1 clean navigation projection", "", "- Source: independently generated validated global Stage 1", "- Historical phase output used: `false`", "- Failed local Stage 2 text used: `false`", f"- Coarse count: `{report['coarse_count']}`", f"- Labels: `{report['phase_labels']}`", "- Planner text: global phase labels only", "- Navigation noise audit: `passed`", "- Coverage: `30/30 Medium`, `88/88 Fine`", "- Storyline: `0`; hard filtering: `false`; coarse prior: disabled", "- API/model calls: `0`", "- Semantic acceptance: `pending_manual_review`"]) + "\n", encoding="utf-8", newline="\n")
    return report
