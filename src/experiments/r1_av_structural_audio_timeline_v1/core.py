from __future__ import annotations

import copy
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

from experiments.r3_v2_coarse_semantic_organizer.core import canonical_bytes, load_json, sha256_file, write_json


SOURCE_R1_MAP = "outputs/experiments/egopolice_r1_structural_novelty_map_v1/coarse_structural_map.json"
SOURCE_R1_VALIDATION = "outputs/experiments/egopolice_r1_structural_novelty_map_v1/validation_report.json"
CANONICAL_INDEX = "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json"
EXPECTED_INDEX_SHA256 = "8b88e4a75c55d241b18e254b471da2bc38387fc6a9e6ccdf7b66337a1e75381f"


def visual_timeline_node(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_type": "visual_structural",
        "node_id": region["coarse_id"],
        "start_sec": float(region["start_sec"]),
        "end_sec": float(region["end_sec"]),
        "navigation_text": region["retrieval_text"],
        "source_medium_ids": list(region["source_medium_ids"]),
        "source_fine_ids": list(region["source_fine_ids"]),
        "temporally_overlapping_audio_ids": list(region["source_audio_ids"]),
        "dominant_terms": list(region["dominant_terms"]),
        "distinctive_terms": list(region["distinctive_terms"]),
        "semantic_summary": False,
        "audio_relationship": "temporal_overlap_reference_only",
    }


def audio_timeline_node(audio: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_type": "audio_asr",
        "node_id": audio["audio_id"],
        "start_sec": float(audio["start_sec"]),
        "end_sec": float(audio["end_sec"]),
        "transcript": audio["transcript"],
        "source_type": audio.get("source_type", "unclear"),
        "asr_status": audio.get("asr_status"),
        "fallback_used": bool(audio.get("fallback_used", False)),
        "source_audio_ids": [audio["audio_id"]],
        "evidence_scope": "audible_statement_or_mention_only",
        "visual_confirmation": False,
        "physical_event_claim": False,
    }


def timeline_sort_key(node: dict[str, Any]) -> tuple[float, int, float, str]:
    type_order = 0 if node["node_type"] == "visual_structural" else 1
    return (float(node["start_sec"]), type_order, float(node["end_sec"]), node["node_id"])


def build_map(r1_map: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    visual_nodes = [visual_timeline_node(row) for row in r1_map["coarse_regions"]]
    audio_nodes = [audio_timeline_node(row) for row in index["audio_nodes"]]
    timeline = sorted(visual_nodes + audio_nodes, key=timeline_sort_key)
    return {
        "schema_version": "r1_av_structural_audio_timeline_v1",
        "rung_name": "R1_AV",
        "map_type": "structural_visual_plus_audio_timeline",
        "semantic_fields_available": False,
        "av_semantic_fusion_available": False,
        "visual_semantic_source": "detector_tracking_structural_novelty",
        "audio_semantic_source": "canonical_timestamped_asr",
        "timeline_ordering": "start_sec, visual_before_audio_on_exact_tie, end_sec, node_id",
        "timeline_nodes": timeline,
        "visual_structural_regions": copy.deepcopy(r1_map["coarse_regions"]),
        "canonical_audio_nodes": copy.deepcopy(index["audio_nodes"]),
        "storyline_events": [],
        "has_storyline": False,
        "hard_filtering_allowed": False,
        "provenance": {
            "source_r1_map": SOURCE_R1_MAP,
            "canonical_index": CANONICAL_INDEX,
            "visual_boundaries_modified": False,
            "visual_navigation_text_modified": False,
            "audio_nodes_nested_under_visual_nodes": False,
            "audio_nodes_duplicated": False,
            "asr_used_to_change_visual_boundaries": False,
            "semantic_event_phases_inferred": False,
            "question_conditioned": False,
            "model_api_calls": 0,
        },
    }


def validate(r1_map: dict[str, Any], index: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    timeline = result["timeline_nodes"]
    visual = [row for row in timeline if row["node_type"] == "visual_structural"]
    audio = [row for row in timeline if row["node_type"] == "audio_asr"]
    source_audio = index["audio_nodes"]
    source_medium_ids = [item for row in r1_map["coarse_regions"] for item in row["source_medium_ids"]]
    source_fine_ids = [item for row in r1_map["coarse_regions"] for item in row["source_fine_ids"]]
    expected_audio_projection = [audio_timeline_node(row) for row in source_audio]
    actual_audio_by_id = {row["node_id"]: row for row in audio}
    caption_keys = {"qwen_caption", "caption", "vlm_summary", "semantic_view_text"}
    serialized = canonical_bytes(result).decode("utf-8").lower()
    checks = {
        "five_visual_nodes": len(visual) == len(r1_map["coarse_regions"]) == 5,
        "107_audio_nodes": len(audio) == len(source_audio) == 107,
        "112_total_timeline_nodes": len(timeline) == 112,
        "timeline_sorted_deterministically": timeline == sorted(timeline, key=timeline_sort_key),
        "visual_ids_unique": len({row["node_id"] for row in visual}) == len(visual),
        "audio_ids_unique": len({row["node_id"] for row in audio}) == 107,
        "audio_projection_exact": all(actual_audio_by_id.get(row["node_id"]) == row for row in expected_audio_projection),
        "visual_regions_byte_semantic_unchanged": canonical_bytes(result["visual_structural_regions"]) == canonical_bytes(r1_map["coarse_regions"]),
        "visual_boundaries_unchanged": [(row["start_sec"], row["end_sec"]) for row in visual] == [(float(row["start_sec"]), float(row["end_sec"])) for row in r1_map["coarse_regions"]],
        "visual_navigation_text_unchanged": [row["navigation_text"] for row in visual] == [row["retrieval_text"] for row in r1_map["coarse_regions"]],
        "medium_coverage_30_unique": len(source_medium_ids) == 30 and len(set(source_medium_ids)) == 30,
        "fine_coverage_88_unique": len(source_fine_ids) == 88 and len(set(source_fine_ids)) == 88,
        "audio_not_nested": all("transcript" not in row and "exact_source_asr" not in row for row in visual),
        "audio_scope_conservative": all(row["evidence_scope"] == "audible_statement_or_mention_only" and row["visual_confirmation"] is False and row["physical_event_claim"] is False for row in audio),
        "caption_fields_absent_from_timeline": all(not (caption_keys & set(row)) for row in timeline),
        "storyline_absent": result["storyline_events"] == [] and result["has_storyline"] is False,
        "hard_filtering_disabled": result["hard_filtering_allowed"] is False,
        "no_question_or_rung_leakage_in_navigation_text": "question_text" not in serialized and "expected_answer" not in serialized,
        "no_semantic_fusion_claim": result["av_semantic_fusion_available"] is False and result["semantic_fields_available"] is False,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "visual_node_count": len(visual), "audio_node_count": len(audio), "timeline_node_count": len(timeline), "medium_count": len(source_medium_ids), "fine_count": len(source_fine_ids)}


def run(root: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    protected = [SOURCE_R1_MAP, SOURCE_R1_VALIDATION, CANONICAL_INDEX]
    before = {path: sha256_file(root / path) for path in protected}
    if before[CANONICAL_INDEX] != EXPECTED_INDEX_SHA256:
        raise RuntimeError("canonical_index_hash_mismatch")
    r1_map = load_json(root / SOURCE_R1_MAP)
    r1_validation = load_json(root / SOURCE_R1_VALIDATION)
    index = load_json(root / CANONICAL_INDEX)
    if r1_validation.get("status") != "passed":
        raise RuntimeError("source_r1_map_not_valid")

    result = build_map(r1_map, index)
    audit = validate(r1_map, index, result)
    determinism = canonical_bytes(result) == canonical_bytes(build_map(r1_map, index))
    after = {path: sha256_file(root / path) for path in protected}
    unchanged = before == after
    if audit["status"] != "passed" or not determinism or not unchanged:
        raise RuntimeError("r1_av_validation_failed")

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input_manifest.json", {
        "experiment": "r1_av_structural_audio_timeline_v1",
        "rung_name": "R1_AV",
        "source_files": {path: {"sha256": before[path], "size_bytes": (root / path).stat().st_size} for path in protected},
        "model_api_calls": 0,
    })
    write_json(output / "r1_av_navigation_map.json", result)
    write_json(output / "timeline_validation_report.json", audit)
    write_json(output / "audio_node_integrity_audit.json", {
        "status": "passed",
        "canonical_audio_count": 107,
        "timeline_audio_count": 107,
        "each_audio_id_once": True,
        "exact_transcript_and_interval_preserved": True,
        "nested_under_visual_node": False,
        "physical_event_claims_generated": 0,
    })
    write_json(output / "visual_map_preservation_audit.json", {
        "status": "passed",
        "source_map_sha256": before[SOURCE_R1_MAP],
        "visual_region_count": 5,
        "visual_boundaries_modified": False,
        "retrieval_text_modified": False,
        "medium_coverage": "30/30",
        "fine_coverage": "88/88",
    })
    write_json(output / "determinism_report.json", {
        "status": "passed" if determinism else "failed",
        "byte_identical_second_construction": determinism,
        "map_sha256": hashlib.sha256(canonical_bytes(result)).hexdigest(),
    })
    write_json(output / "protected_hash_audit.json", {"status": "passed" if unchanged else "failed", "before": before, "after": after, "unchanged": unchanged})
    validation = {
        "source_validation": "passed",
        "visual_preservation_validation": "passed",
        "audio_node_validation": "passed",
        "timeline_validation": audit["status"],
        "determinism_validation": "passed" if determinism else "failed",
        "overall_validation": "passed",
        "map_type": result["map_type"],
        "visual_nodes": 5,
        "audio_nodes": 107,
        "timeline_nodes": 112,
        "medium_coverage": "30/30",
        "fine_coverage": "88/88",
        "model_api_calls": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "sufficiency_calls": 0,
        "final_answer_calls": 0,
        "protected_sources_unchanged": unchanged,
    }
    write_json(output / "validation_report.json", validation)
    write_json(output / "cost_accounting.json", {"model_api_calls": 0, "deterministic_map_construction_latency_sec": time.perf_counter() - started})

    rows = []
    for node in result["timeline_nodes"]:
        if node["node_type"] == "visual_structural":
            content = node["navigation_text"]
        else:
            content = node["transcript"]
        rows.append(f"<tr><td>{html.escape(node['node_type'])}</td><td>{html.escape(node['node_id'])}</td><td>{node['start_sec']:.3f}–{node['end_sec']:.3f}</td><td>{html.escape(content)}</td></tr>")
    (output / "review.html").write_text("<!doctype html><meta charset='utf-8'><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px;vertical-align:top}</style><h1>R1_AV structural + audio timeline</h1><p>Five visual structural nodes and 107 independent ASR nodes. No semantic AV fusion is claimed.</p><table><tr><th>Type</th><th>ID</th><th>Interval</th><th>Navigation content</th></tr>" + "".join(rows) + "</table>", encoding="utf-8", newline="\n")
    (output / "REPORT.md").write_text("\n".join([
        "# R1_AV structural plus audio timeline v1",
        "",
        "R1_AV preserves the frozen R1 structural map and packages its five visual regions with all 107 canonical ASR records as independent chronological nodes.",
        "",
        "- Visual nodes: `5`; ASR nodes: `107`; total timeline nodes: `112`",
        "- Medium/Fine coverage: `30/30`, `88/88`",
        "- Visual boundaries and retrieval text changed: `false`",
        "- ASR nested or duplicated: `false`",
        "- AV semantic fusion claimed: `false`",
        "- Storyline and hard filtering: disabled",
        "- Model/API, Planner, Retrieval and downstream calls: `0`",
    ]) + "\n", encoding="utf-8", newline="\n")
    return validation
