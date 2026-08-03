from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

from experiments.r3_v2_coarse_semantic_organizer.core import load_json, sha256_file, write_json


SOURCE_ROOT = "outputs/experiments/r3_2_global_stage1_clean_navigation_projection_v1"
SOURCE_FILES = (
    "r3_2_global_stage1_clean_semantic_map.json",
    "planner_compatibility_view.json",
    "validation_report.json",
    "navigation_noise_audit.json",
    "planner_compatibility_report.json",
    "protected_hash_audit.json",
)

EXPECTED_PHASES = (
    "Initial positioning and arrival at scene",
    "Perimeter reconnaissance and property approach",
    "Building exterior observation and fence interaction",
    "Continued perimeter assessment and fence examination",
    "Tactical positioning with firearm deployment",
    "Suspect apprehension and restraint",
    "Suspect secured and evidence documentation",
    "Injury assessment and medical response initiation",
    "Extended medical intervention and scene management",
    "Casualty care and emergency services coordination",
)


def validate_freeze_source(semantic_map: dict[str, Any], planner_view: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    coarse = semantic_map.get("coarse_regions", [])
    medium_ids = [medium_id for row in coarse for medium_id in row.get("source_medium_ids", [])]
    fine_ids = [fine_id for row in coarse for fine_id in row.get("source_fine_ids", [])]
    labels = tuple(row.get("navigation_summary") for row in coarse)
    planner_labels = tuple(row.get("summary") for row in planner_view.get("coarse_regions", []))
    checks = {
        "source_structural_validation_passed": validation.get("clean_projection_validation") == "passed",
        "source_planner_validation_passed": validation.get("planner_compatibility_validation") == "passed",
        "source_noise_audit_passed": validation.get("navigation_noise_audit") == "passed",
        "expected_ten_phase_labels": labels == EXPECTED_PHASES,
        "medium_coverage_30_unique": len(medium_ids) == 30 and len(set(medium_ids)) == 30,
        "fine_coverage_88_unique": len(fine_ids) == 88 and len(set(fine_ids)) == 88,
        "storyline_disabled": semantic_map.get("storyline_events") == [] and semantic_map.get("has_storyline") is False,
        "hard_filtering_disabled": semantic_map.get("hard_filtering_allowed") is False and planner_view.get("hard_filtering_allowed") is False,
        "coarse_prior_disabled": planner_view.get("planner_policy", {}).get("coarse_prior_affects_ranking") is False,
        "all_mediums_retrieval_eligible": planner_view.get("planner_policy", {}).get("all_mediums_remain_eligible") is True,
        "planner_uses_phase_labels_only": planner_labels == labels,
        "historical_phase_output_not_used": semantic_map.get("provenance", {}).get("historical_phase_output_used") is False,
        "failed_stage2_text_not_used": semantic_map.get("provenance", {}).get("failed_stage2_text_used") is False,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def run(root: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    source_hashes_before = {
        f"{SOURCE_ROOT}/{name}": sha256_file(root / SOURCE_ROOT / name)
        for name in SOURCE_FILES
    }
    semantic_map = load_json(root / SOURCE_ROOT / "r3_2_global_stage1_clean_semantic_map.json")
    planner_view = load_json(root / SOURCE_ROOT / "planner_compatibility_view.json")
    source_validation = load_json(root / SOURCE_ROOT / "validation_report.json")
    source_noise = load_json(root / SOURCE_ROOT / "navigation_noise_audit.json")
    source_planner = load_json(root / SOURCE_ROOT / "planner_compatibility_report.json")

    audit = validate_freeze_source(semantic_map, planner_view, source_validation)
    if source_noise.get("status") != "passed":
        audit["checks"]["source_noise_artifact_passed"] = False
    else:
        audit["checks"]["source_noise_artifact_passed"] = True
    audit["checks"]["source_planner_artifact_valid"] = source_planner.get("valid") is True
    audit["status"] = "passed" if all(audit["checks"].values()) else "failed"
    if audit["status"] != "passed":
        raise RuntimeError(f"r3_2_freeze_source_invalid: {audit['checks']}")

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input_manifest.json", {
        "rung": "R3_2",
        "definition": "independently generated global AV Stage-1 clean semantic navigation map",
        "source_root": SOURCE_ROOT,
        "source_hashes": source_hashes_before,
        "model_api_calls": 0,
    })
    write_json(output / "manual_semantic_acceptance.json", {
        "semantic_acceptance": "passed_by_manual_review",
        "review_decision": "approved_for_freeze_by_user",
        "accepted_scope": "ten-phase R3_2 Planner navigation map for video 226",
        "accepted_properties": [
            "global AV phase labels provide useful semantic navigation",
            "known caption-noise terms are absent from the Planner-visible map",
            "all 30 Medium and 88 Fine nodes remain represented",
            "Storyline and hard pruning remain disabled",
        ],
        "scope_limits": [
            "not a Sufficiency evidence source",
            "not reviewed visual confirmation",
            "not a freeze of Retrieval, Sufficiency, final answer, or full R3 QA",
            "not a dataset-level generalization result",
        ],
    })
    write_json(output / "source_validation_audit.json", audit)
    write_json(output / "r3_2_frozen_semantic_navigation_map.json", copy.deepcopy(semantic_map))
    write_json(output / "r3_2_frozen_planner_compatibility_view.json", copy.deepcopy(planner_view))

    copied_map_hash = sha256_file(output / "r3_2_frozen_semantic_navigation_map.json")
    copied_view_hash = sha256_file(output / "r3_2_frozen_planner_compatibility_view.json")
    source_hashes_after = {
        path: sha256_file(root / path)
        for path in source_hashes_before
    }
    protected_unchanged = source_hashes_before == source_hashes_after
    write_json(output / "protected_source_hash_audit.json", {
        "status": "passed" if protected_unchanged else "failed",
        "before": source_hashes_before,
        "after": source_hashes_after,
        "unchanged": protected_unchanged,
    })

    freeze_manifest = {
        "rung": "R3_2",
        "freeze_status": "freeze_candidate_accepted_for_226_navigation_map",
        "manual_semantic_acceptance": "passed_by_manual_review",
        "source_root": SOURCE_ROOT,
        "source_hashes": source_hashes_before,
        "frozen_map_sha256": copied_map_hash,
        "frozen_planner_view_sha256": copied_view_hash,
        "coarse_count": 10,
        "medium_count": 30,
        "fine_count": 88,
        "phase_labels": list(EXPECTED_PHASES),
        "map_policy": {
            "map_type": "r3_2_global_stage1_clean_semantic_coarse",
            "semantic_input": "canonical repaired Qwen captions plus canonical timestamped ASR in one global interleaved Stage-1 view",
            "planner_text": "global phase labels only",
            "raw_caption_asr": "retained in map sidecar, not exposed as Planner summary text",
            "storyline": False,
            "hard_filtering_allowed": False,
            "coarse_prior_affects_ranking": False,
            "all_mediums_remain_retrieval_eligible": True,
            "map_text_is_sufficiency_evidence": False,
        },
        "historical_nine_phase_output_used": False,
        "failed_stage2_text_used": False,
        "freeze_scope": "R3_2 semantic navigation map only",
    }
    write_json(output / "r3_2_navigation_map_freeze_manifest.json", freeze_manifest)

    overall = "passed" if protected_unchanged else "failed"
    report = {
        "source_validation": audit["status"],
        "manual_semantic_acceptance": "passed_by_manual_review",
        "freeze_candidate_validation": overall,
        "overall_validation": "passed_r3_2_navigation_map_frozen" if overall == "passed" else "failed",
        "coarse_count": 10,
        "medium_count": 30,
        "fine_count": 88,
        "storyline_count": 0,
        "hard_filtering_allowed": False,
        "coarse_prior_affects_ranking": False,
        "model_api_calls": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "sufficiency_calls": 0,
        "final_answer_calls": 0,
        "protected_sources_unchanged": protected_unchanged,
        "next_step": "paired R1/R3_1/R3_2 map-aware Planner plus all-Medium retrieval validation",
    }
    write_json(output / "validation_report.json", report)
    write_json(output / "cost_accounting.json", {
        "model_api_calls": 0,
        "freeze_projection_latency_sec": time.perf_counter() - started,
        "source_stage1_model_call_reused_not_recounted": True,
    })
    (output / "REPORT.md").write_text(
        "\n".join([
            "# R3_2 global Stage-1 clean navigation map freeze",
            "",
            "The user-approved manual semantic review is recorded as passed for the isolated video-226 navigation-map scope.",
            "",
            "- Frozen map: independently generated ten-phase global AV Stage-1 map",
            "- Historical nine-phase result used: `false`",
            "- Failed Stage-2 summaries used: `false`",
            "- Coverage: `30/30 Medium`, `88/88 Fine`",
            "- Storyline: `0`",
            "- Hard filtering: `false`; coarse prior: disabled",
            "- Planner-visible text: global clean phase labels only",
            "- Manual semantic acceptance: `passed_by_manual_review`",
            "- API/model calls for this freeze: `0`",
            "- Scope: navigation map only; Retrieval, Sufficiency and full R3 QA are not frozen here.",
        ]) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report

