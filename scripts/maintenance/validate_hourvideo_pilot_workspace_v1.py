from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


MODULES = (
    "claim_level_av_sufficiency_v2",
    "claim_level_av_sufficiency_v3_1_requirement_centric",
    "claim_level_av_sufficiency_v3_minimal_contract",
    "egopolice_r1_av_dual_channel_retrieval_smoke_v1",
    "fine_reranking",
    "planner_medium_retrieval",
    "question_scoped_visual_review_gate_v1",
    "r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1",
    "r1_av_r3_2_cached_visual_review_full_closed_loop_v1",
    "r1_av_r3_2_claim_guided_gemini_closed_loop_v3",
    "r1_av_r3_2_requirement_centric_pipeline_canary",
    "r1_av_r3_2_requirement_centric_pipeline_canary_v1_1",
    "r1_av_r3_2_requirement_centric_pipeline_canary_v1_2",
    "r1_av_r3_2_requirement_centric_pipeline_six_question_regression_v1",
    "r1_av_r3_2_retrieval_bound_fine_evidence_v1",
    "r1_av_r3_2_review_cache_integration_canary_v1",
    "r1_av_structural_audio_timeline_v1",
    "r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair",
    "r1_r3_v2_map_aware_all_medium_retrieval_pair",
    "r3_2_frozen_exact_interleaved_av_organizer",
    "r3_2_frozen_method_staged_av_organizer",
    "r3_2_global_stage1_clean_navigation_map_freeze",
    "r3_2_global_stage1_clean_navigation_projection",
    "r3_2_independent_global_phase_denoised_av_organizer",
    "r3_2_semantic_coarse_first_sufficiency_canary_v1_2",
    "r3_v2_av_coarse_semantic_organizer_v1_1",
    "r3_v2_coarse_semantic_organizer",
    "reviewed_visual_evidence_cache_v1",
    "reviewed_visual_evidence_cache_v1_gemini_canary",
    "shared_sufficiency_v3_2_1_temporal_anchor",
    "shared_sufficiency_v3_2_contract",
)

ARTIFACT_DIRS = (
    "r1_av_r3_2_map_planner_upstream_freeze_v1",
    "r3_2_global_stage1_clean_navigation_map_freeze_v1",
    "r1_av_r3_2_requirement_centric_pipeline_six_question_regression_v1",
    "r1_av_r3_2_retrieval_bound_fine_evidence_v1",
    "question_scoped_visual_review_gate_v1",
    "r1_av_r3_2_review_cache_integration_canary_v1",
    "reviewed_visual_evidence_cache_v1_gemini_canary_v1",
    "r1_av_r3_2_cached_visual_review_full_closed_loop_v1",
    "r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1",
    "r1_av_structural_audio_timeline_v1",
    "egopolice_r1_av_dual_channel_retrieval_smoke_v1",
    "r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1",
    "r3_2_semantic_coarse_first_sufficiency_canary_v1_2",
    "shared_temporal_review_handoff_canary_v1_1",
    "caption_fixed_rich_index_v1_1",
)

KEY_PARITY_FILES = (
    "outputs/experiments/r1_av_structural_audio_timeline_v1/r1_av_navigation_map.json",
    "outputs/experiments/r3_2_global_stage1_clean_navigation_map_freeze_v1/r3_2_frozen_semantic_navigation_map.json",
    "outputs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1/planner_outputs.json",
    "outputs/experiments/r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1/r3_2_planner_outputs.json",
    "outputs/experiments/r1_av_r3_2_requirement_centric_pipeline_six_question_regression_v1/sufficiency_results.json",
    "outputs/experiments/r1_av_r3_2_cached_visual_review_full_closed_loop_v1/resolved_final_handoff.json",
    "outputs/experiments/r1_av_r3_2_cached_visual_review_full_closed_loop_v1/final_answers.json",
    "outputs/experiments/r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1/r1_av_engineering_freeze_manifest.json",
    "outputs/experiments/r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1/r3_2_engineering_freeze_manifest.json",
    "outputs/experiments/r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1/shared_pipeline_interface_contract.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    return rows


def selected_auxiliary_files(root: Path) -> list[Path]:
    selected = []
    for prefix in ("configs/experiments", "scripts/experiments", "docs/experiments", "tests/experiments"):
        folder = root / prefix
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            text = path.as_posix()
            if any(module in text for module in MODULES):
                selected.append(path)
    return sorted(selected)


def compare_relative_files(source: Path, target: Path, files: list[Path]) -> list[str]:
    errors = []
    for source_path in files:
        relative = source_path.relative_to(source)
        target_path = target / relative
        if not target_path.is_file():
            errors.append(f"missing target file: {relative.as_posix()}")
        elif sha256(source_path) != sha256(target_path):
            errors.append(f"hash mismatch: {relative.as_posix()}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--target-workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    source, target = args.source_workspace.resolve(), args.target_workspace.resolve()
    if source == target:
        raise SystemExit("source and target workspaces must differ")

    errors: list[str] = []
    module_rows = []
    compiled = 0
    for module in MODULES:
        source_dir = source / "src/experiments" / module
        target_dir = target / "src/experiments" / module
        source_manifest, target_manifest = tree_manifest(source_dir), tree_manifest(target_dir)
        equal = source_manifest == target_manifest
        module_rows.append({"module": module, "file_count": len(target_manifest), "byte_identical": equal})
        if not equal:
            errors.append(f"module tree mismatch: {module}")
        for path in target_dir.rglob("*.py"):
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
                compiled += 1
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"compile failure {path.relative_to(target)}: {exc}")

    auxiliary = selected_auxiliary_files(source)
    errors += compare_relative_files(source, target, auxiliary)

    artifact_rows = []
    for name in ARTIFACT_DIRS:
        source_manifest = tree_manifest(source / "outputs/experiments" / name)
        target_manifest = tree_manifest(target / "outputs/experiments" / name)
        equal = source_manifest == target_manifest
        artifact_rows.append({"artifact_dir": name, "file_count": len(target_manifest), "byte_identical": equal})
        if not equal:
            errors.append(f"artifact tree mismatch: {name}")

    key_rows = []
    for relative_text in KEY_PARITY_FILES:
        relative = Path(relative_text)
        source_path, target_path = source / relative, target / relative
        source_hash = sha256(source_path)
        target_hash = sha256(target_path)
        equal = source_hash == target_hash
        key_rows.append({"path": relative.as_posix(), "source_sha256": source_hash, "target_sha256": target_hash, "byte_identical": equal})
        if not equal:
            errors.append(f"key parity mismatch: {relative.as_posix()}")

    sys.path.insert(0, str(target / "src"))
    from experiments.r1_av_r3_2_cached_visual_review_full_closed_loop_v1.core import run as run_cached_closed_loop
    from experiments.r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1.core import run as run_freeze

    validation_root = target / "outputs/workspace_validation/hourvideo_pilot_v1"
    closed_cfg = load(target / "configs/experiments/r1_av_r3_2_cached_visual_review_full_closed_loop_v1.json")
    closed_cfg["output_root"] = "outputs/workspace_validation/hourvideo_pilot_v1/closed_loop"
    closed_cfg_path = validation_root / "closed_loop_config.json"
    dump(closed_cfg_path, closed_cfg)
    closed_result = run_cached_closed_loop(target, closed_cfg_path, allow_api_calls=False)
    if closed_result.get("overall_validation") != "ready_for_live_final_gemini":
        errors.append("no-API cached closed-loop replay failed")

    freeze_cfg = load(target / "configs/experiments/r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1.json")
    freeze_cfg["output_root"] = "outputs/workspace_validation/hourvideo_pilot_v1/freeze"
    freeze_cfg_path = validation_root / "freeze_config.json"
    dump(freeze_cfg_path, freeze_cfg)
    freeze_result = run_freeze(target, freeze_cfg_path)
    if freeze_result.get("overall_validation") != "frozen_engineering_test_snapshot":
        errors.append("engineering freeze replay failed")

    cache_audit = load(validation_root / "closed_loop/visual_cache_hit_audit.json")
    if cache_audit.get("cache_hits") != 22 or cache_audit.get("cache_misses") != 0:
        errors.append("reviewed-visual cache parity failed")

    manifest = {
        "contract": "hourvideo_pilot_clean_workspace_migration_v1",
        "source_workspace": str(source),
        "target_workspace": str(target),
        "source_head": "87e49d570007391007677cbba137fa5fa4cb0d17",
        "module_count": len(MODULES),
        "compiled_python_files": compiled,
        "auxiliary_file_count": len(auxiliary),
        "artifact_directory_count": len(ARTIFACT_DIRS),
        "module_parity": module_rows,
        "artifact_parity": artifact_rows,
        "key_output_parity": key_rows,
        "no_api_cached_closed_loop": closed_result,
        "engineering_freeze_replay": freeze_result,
        "cache_parity": {
            "lookups": cache_audit.get("requested_cache_lookups"),
            "hits": cache_audit.get("cache_hits"),
            "misses": cache_audit.get("cache_misses"),
            "image_transmissions": cache_audit.get("image_transmissions"),
        },
        "model_api_calls": 0,
        "errors": errors,
        "overall_validation": "passed_clean_and_complete_parity" if not errors else "failed",
    }
    dump(target / "docs/workspace_migration/hourvideo_pilot_workspace_manifest_v1.json", manifest)
    print(json.dumps({
        "overall_validation": manifest["overall_validation"],
        "modules": len(MODULES),
        "compiled_python_files": compiled,
        "auxiliary_files": len(auxiliary),
        "artifact_dirs": len(ARTIFACT_DIRS),
        "cache_hits": cache_audit.get("cache_hits"),
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

