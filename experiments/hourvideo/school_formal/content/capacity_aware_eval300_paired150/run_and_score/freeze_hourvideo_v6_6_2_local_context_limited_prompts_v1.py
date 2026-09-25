from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json  # noqa: E402
from experiments.hourvideo_v6_6_1_contract_telemetry_v1 import core as v661  # noqa: E402
from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1.prompt_contract import (  # noqa: E402
    LOCAL_STRUCTURED_OUTPUT_CONTRACT,
    PLANNER_COMMON_CORE,
    PLANNER_R1_SYSTEM,
    PLANNER_R3_SYSTEM,
    R1_INPUT_REPRESENTATION,
    R3_INPUT_REPRESENTATION,
)


EXPERIMENT = "hourvideo_v6_6_2_local_context_limited_eval300_v1"
OUTPUT = ROOT / "outputs/experiments" / EXPERIMENT
PROMPT_DIR = OUTPUT / "prompts/candidate_v1"
VALIDATION_DIR = OUTPUT / "validation"
SAMPLE_QID = "6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31"
SAMPLE_VIDEO = "6fd90f8d-7a4d-425d-a812-3268db0b0342"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_non_gold_sample() -> dict:
    source = ROOT / "outputs/experiments/hourvideo_v6_6_2_formal_eval300_v2/source"
    question = load_json(source / "cases" / SAMPLE_QID / "question_input.json")
    case_cfg = load_json(source / "case_configs" / f"{SAMPLE_QID}.json")
    if set(question) != {"question_id", "video_uid", "question_text", "answer_options"}:
        raise ValueError("sample question contains non-input fields")
    hierarchy = load_json(Path(case_cfg["asset_paths"]["shared_hierarchy.json"]))
    fine_ids = [row["fine_id"] for row in hierarchy["fine_nodes"]]
    if len(fine_ids) != len(set(fine_ids)):
        raise ValueError("sample Fine IDs are not unique")
    medium_ids = {row["medium_id"] for row in hierarchy["medium_nodes"]}
    if any(row["parent_medium_id"] not in medium_ids for row in hierarchy["fine_nodes"]):
        raise ValueError("sample Fine parent identity is invalid")

    requirements = v661.option_requirements(question)
    routes = []
    for side, name in (("r1_av", "r1_av_navigation_map.json"), ("r3_2", "r3_2_navigation_map.json")):
        map_path = Path(case_cfg["asset_paths"][name])
        map_doc = load_json(map_path)
        coarse_ids = [row["coarse_id"] for row in map_doc["coarse_regions"]]
        if len(coarse_ids) != len(set(coarse_ids)):
            raise ValueError(f"duplicate sample Coarse IDs: {side}")
        provider_ids = list(range(len(coarse_ids)))
        schema = v661._base._coarse_locked_planner_schema(requirements, provider_ids)
        Draft202012Validator.check_schema(schema)
        frozen = load_json(
            ROOT / "outputs/experiments/hourvideo_v6_6_2_formal_eval300_v1/frozen_planner/cases"
            / SAMPLE_QID / side / "planner.json"
        )
        v661._base._validate_coarse_locked_plan(
            frozen["output"], question, requirements, set(coarse_ids),
        )
        projected = v661._base._planner_map(map_doc)
        if side == "r1_av":
            fields_match = (
                projected["semantic_fields_available"] is False
                and all("audio_channel" in row and "exact_source_captions" not in row for row in projected["coarse_regions"])
                and all("visual_structured_fallback" in row for row in map_doc["coarse_regions"])
            )
        else:
            fields_match = (
                projected["semantic_fields_available"] is True
                and all("exact_source_captions" in row and "audio_channel" not in row for row in projected["coarse_regions"])
            )
        if not fields_match:
            raise ValueError(f"Planner representation description does not match {side} map")
        routes.append({
            "question_id": SAMPLE_QID, "video_uid": SAMPLE_VIDEO, "route": side,
            "gold_fields_loaded": False, "schema_valid": True,
            "coarse_ids_unique_and_valid": True, "fine_ids_unique_and_parented": True,
            "map_fields_match_prompt_description": True,
            "map_sha256": sha256_file(map_path),
            "retained_non_gold_planner_schema_valid": True,
        })
    return {"sample_question_id": SAMPLE_QID, "accuracy_read": False, "routes": routes}


def main() -> int:
    shared = v661._v652.SHARED_INVESTIGATION_SYSTEM
    fine = v661._base.BATCH_CLAIM_EXECUTION_SYSTEM
    final = v661._V64.DIRECT_FINAL_SYSTEM
    prompt_texts = {
        "planner_common_core.txt": PLANNER_COMMON_CORE,
        "planner_r1_system.txt": PLANNER_R1_SYSTEM,
        "planner_r3_system.txt": PLANNER_R3_SYSTEM,
        "shared_system.txt": shared,
        "fine_system.txt": fine,
        "final_system.txt": final,
    }
    for filename, text in prompt_texts.items():
        write_text(PROMPT_DIR / filename, text + "\n")
    diff = "".join(difflib.unified_diff(
        PLANNER_R1_SYSTEM.splitlines(keepends=True),
        PLANNER_R3_SYSTEM.splitlines(keepends=True),
        fromfile="planner_r1_system.txt", tofile="planner_r3_system.txt",
    ))
    write_text(PROMPT_DIR / "planner_r1_vs_r3.diff", diff)
    validation = validate_non_gold_sample()
    write_json(VALIDATION_DIR / "prompt_candidate_non_gold_validation.json", validation)
    manifest = {
        "schema_version": "v6_6_2_local_context_limited_prompt_candidate_v1",
        "status": "AWAITING_USER_CONFIRMATION",
        "external_api_used": False, "model_called": False, "accuracy_read": False,
        "planner_core_identical_between_routes": True,
        "planner_output_contract_identical_between_routes": True,
        "only_route_specific_text": "input representation description",
        "shared_fine_final_identical_between_routes": True,
        "common_interface_correction": (
            "The local formatting instruction now names selected_coarse_ids, the existing schema field, "
            "instead of the nonexistent coarse_judgment field. Schema and decision rules are unchanged."
        ),
        "components": {
            "planner_common_core": {"sha256": digest(PLANNER_COMMON_CORE), "chars": len(PLANNER_COMMON_CORE)},
            "r1_input_representation": {"sha256": digest(R1_INPUT_REPRESENTATION), "chars": len(R1_INPUT_REPRESENTATION)},
            "r3_input_representation": {"sha256": digest(R3_INPUT_REPRESENTATION), "chars": len(R3_INPUT_REPRESENTATION)},
            "local_structured_output_contract": {"sha256": digest(LOCAL_STRUCTURED_OUTPUT_CONTRACT), "chars": len(LOCAL_STRUCTURED_OUTPUT_CONTRACT)},
        },
        "prompt_sha256": {
            "planner_r1": digest(PLANNER_R1_SYSTEM), "planner_r3": digest(PLANNER_R3_SYSTEM),
            "shared": digest(shared), "fine": digest(fine), "final": digest(final),
        },
        "files": {
            filename: {"path": str(PROMPT_DIR / filename), "sha256": sha256_file(PROMPT_DIR / filename)}
            for filename in [*prompt_texts, "planner_r1_vs_r3.diff"]
        },
        "validation_path": str(VALIDATION_DIR / "prompt_candidate_non_gold_validation.json"),
        "validation_sha256": sha256_file(VALIDATION_DIR / "prompt_candidate_non_gold_validation.json"),
    }
    write_json(PROMPT_DIR / "prompt_candidate_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
