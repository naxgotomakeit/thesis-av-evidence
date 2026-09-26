#!/usr/bin/env python3
"""Materialise Direct-owned, no-gold evidence audit manifests.

R1/R3 records intentionally remain pending until their approved frozen
question-time navigation path is executed.  GenS records are finalized from
the native frozen selector and SHA-verified against the local 1-fps cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direct_api_prep.adapters import gens_package, map_identity
from direct_api_prep.evidence import pending_navigation_package

RUNTIME = Path("<PROJECT_MSC_USER_ROOT>/msc_thesis/main_system/thesis-av-evidence-hourvideo/hourvideo_v6_1_runtime")
SOURCE = RUNTIME / "outputs/experiments/hourvideo_v6_6_2_formal_eval300_v2/source"
R3_ROOT = Path("<PROJECT_MSC_USER_ROOT>/msc_thesis/main_system/isolated_workspaces/hourvideo_v7_4_variant_c_budgets_v1/work_index")
GENS = Path("<PROJECT_MSC_USER_ROOT>/msc_thesis/main_system/transfer_inbox/gens_eval300_selector_v2/unpacked/gens_eval300_selector_v2_manifest_only/selected_frames_downstream.jsonl")
FRAME_RUNTIME = Path("<PROJECT_MSC_USER_ROOT>/HourVideo/runtime")
OUT = ROOT / "evidence_packages"


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frame_resolver(relative: str) -> Path:
    # GenS's portable <video_uid>/frame_NNNNN.jpg maps to this verified local
    # HourVideo 1-fps cache layout; no frame is copied or regenerated.
    video_uid, filename = Path(relative).parts
    return FRAME_RUNTIME / video_uid / "frames_1fps" / filename


def main() -> None:
    routes = [json.loads(line) for line in (SOURCE / "ordered_routes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    question_rows = {row["question_id"]: row for row in routes if row["route"] == "r1_av"}
    gens_rows = {row["qa_uid"]: row for row in (json.loads(line) for line in GENS.read_text(encoding="utf-8").splitlines() if line)}
    if set(question_rows) != set(gens_rows) or len(question_rows) != 300:
        raise RuntimeError("frozen Eval300 IDs do not match between source closure and GenS")
    for qid, route in question_rows.items():
        video = route["video_uid"]
        r1_map = SOURCE / "video_assets" / video / "r1_av_navigation_map.json"
        r3_map = R3_ROOT / "cases" / video / "r3_2_navigation_map.json"
        r1 = pending_navigation_package(qid, "R1", map_text_evidence=map_identity(r1_map, representation="frozen_r1_structural_asr_navigation_map"), cost_boundaries={"offline_construction_cost": "frozen source asset", "question_time_navigation_retrieval": "pending approved reuse of V6.6.2 frozen navigation", "direct_answering_api_cost": "not-run", "total_operational_cost": "not-run"})
        r3 = pending_navigation_package(qid, "R3", map_text_evidence=map_identity(r3_map, representation="frozen_r3_hierarchical_semantic_navigation_map"), cost_boundaries={"offline_construction_cost": "frozen source asset", "question_time_navigation_retrieval": "pending approved reuse of V6.6.2 frozen navigation", "direct_answering_api_cost": "not-run", "total_operational_cost": "not-run"})
        gens = gens_package(gens_rows[qid], resolve_frame_path=frame_resolver, verify=True)
        write(OUT / "R1" / f"{qid}.json", r1.as_dict())
        write(OUT / "R3" / f"{qid}.json", r3.as_dict())
        write(OUT / "GenS" / f"{qid}.json", gens.as_dict())
    write(OUT / "manifest.json", {"schema_version": "hourvideo_direct_evidence_package_manifest_v1", "question_count_per_method": 300, "methods": ["R1", "R3", "GenS"], "gold_loaded": False, "api_calls_made": False, "r1_r3_state": "pending_question_time_navigation", "gens_state": "finalized_native_frozen_selection", "physical_image_hard_limit": 16})


if __name__ == "__main__":
    main()
