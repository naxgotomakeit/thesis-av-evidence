#!/usr/bin/env python3
"""Build a local-only, GitHub-oriented audit staging from DGX files.

This script performs file copies and local validation only. It does not invoke
models, APIs, Git, or any network operation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path("/home/naxucl/projects/thesis-av-evidence/github_import_staging_eval300_dgx_20260925")
DATA = Path("/home/naxucl/data/HourVideo")
EXP = DATA / "experiments/hourvideo_v7_4_variant_c_budgets_v1"
OUT = EXP / "outputs"
VIDEOSEAL = DATA / "videoseal_original"

records: list[dict[str, object]] = []
missing: list[dict[str, str]] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(source: Path, relative: str, experiments: str, category: str, *, required: bool = True) -> None:
    destination = ROOT / relative
    if not source.is_file():
        missing.append({
            "source_absolute_path": str(source),
            "staging_relative_path": relative,
            "experiments": experiments,
            "category": category,
            "required": str(required).lower(),
        })
        if required:
            raise FileNotFoundError(source)
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    shutil.copy2(source, destination)
    staged_hash = sha256(destination)
    if source_hash != staged_hash:
        raise RuntimeError(f"copy hash mismatch: {source} -> {destination}")
    records.append({
        "source_absolute_path": str(source),
        "source_sha256": source_hash,
        "source_size_bytes": source.stat().st_size,
        "staging_relative_path": relative,
        "staged_sha256": staged_hash,
        "experiments": experiments,
        "category": category,
        "transformation": "none_byte_identical",
    })


def copy_tree(source: Path, relative: str, experiments: str, category: str, *, exclude=None) -> None:
    exclude = exclude or (lambda _path: False)
    if not source.is_dir():
        raise FileNotFoundError(source)
    for item in sorted(source.rglob("*")):
        if not item.is_file() or exclude(item):
            continue
        rel = item.relative_to(source).as_posix()
        copy_file(item, f"{relative}/{rel}", experiments, category)


def code_exclude(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def copy_named_files(source_dir: Path, relative_dir: str, names: list[str], experiments: str, category: str) -> None:
    for name in names:
        copy_file(source_dir / name, f"{relative_dir}/{name}", experiments, category)


def copy_matching(source_dir: Path, relative_dir: str, patterns: list[str], experiments: str, category: str) -> None:
    seen: set[Path] = set()
    for pattern in patterns:
        for item in sorted(source_dir.glob(pattern)):
            if item.is_file() and item not in seen:
                seen.add(item)
                rel = item.relative_to(source_dir).as_posix()
                copy_file(item, f"{relative_dir}/{rel}", experiments, category)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_uids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def copy_sources() -> None:
    all_main = "flat30;flat15_retryv2;dense_h8_v3;dense_h15_v1;dense_h30_v2"
    all_dense = "dense_h8_v3;dense_h15_v1;dense_h30_v2"

    # Frozen benchmark input. The videos themselves are intentionally excluded.
    copy_file(DATA / "benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt", "shared/dataset/hourvideo_eval300_v1_uids.txt", all_main, "input_uid_manifest")
    copy_file(DATA / "benchmark/v1.0_release/hourvideo_dev_v1.0_videoseal_dgx.parquet", "shared/dataset/hourvideo_dev_v1.0_videoseal_dgx.parquet", all_main, "input_benchmark_parquet")

    # Existing DGX-only registries and read-only verification.
    appendix = Path("/home/naxucl/projects/thesis-av-evidence/thesis_appendix_materials_dgx_v1")
    copy_file(appendix / "experiment_registry.csv", "provenance/experiment_registry.csv", all_main, "registry")
    copy_file(appendix / "prompt_registry.csv", "provenance/prompt_registry.csv", all_main, "registry")
    copy_file(Path("/home/naxucl/projects/thesis-av-evidence/methods_readonly_verification_20260915/REPORT.md"), "provenance/methods_readonly_verification_REPORT.md", all_main, "prior_readonly_audit")

    # V7.4 package metadata and actual runtime support used under the Dense overlays.
    runtime = EXP / "runtime_v4"
    copy_named_files(runtime, "shared/runtime_v7_4", [
        "CONTENTS.sha256", "DATA_MACHINE_ARM_ENVIRONMENT.md", "MANIFEST.json",
        "SOURCE_COMPLETE_RUNTIME_REPORT.md", "environment_smoke_source.json",
        "reference_to_v7_4_complete_diff.json", "requirements_runner_v74.lock",
        "smoke_actual_source_resolution.json", "source_complete_validation.json",
    ], all_dense, "runtime_manifest")
    copy_tree(runtime / "experiment", "shared/runtime_v7_4/experiment", all_dense, "runtime_experiment_code", exclude=code_exclude)
    copy_tree(runtime / "runtime_support", "shared/runtime_v7_4/runtime_support", all_dense, "runtime_support_code", exclude=code_exclude)
    copy_tree(runtime / "runnable_runtime/videoseal", "shared/runtime_v7_4/runnable_runtime/videoseal", all_dense, "v7_4_base_source", exclude=code_exclude)
    copy_named_files(runtime / "runnable_runtime", "shared/runtime_v7_4/runnable_runtime", ["pyproject.toml", "uv.lock"], all_dense, "runtime_lock")

    # Actual formal orchestration and aggregation shared by New Dense-H.
    copy_tree(EXP / "formal_eval300_orchestrator", "shared/dense_h_orchestrator", all_dense, "entry_retry_aggregation", exclude=code_exclude)

    # Dense index identity and locally captured provenance; no vectors, frames, or school access.
    index = EXP / "index"
    copy_named_files(index, "shared/dense_index", ["CONTENTS.sha256", "MANIFEST.json"], all_dense, "input_index_manifest")
    copy_tree(index / "audit", "shared/dense_index/audit", all_dense, "input_index_audit", exclude=code_exclude)
    copy_named_files(index / "indexer_build_provenance", "shared/dense_index/indexer_build_provenance", [
        "pipeline_manifest.json", "provenance_manifest.json"
    ], all_dense, "input_index_provenance")
    copy_tree(index / "indexer_build_provenance/caption_stage", "shared/dense_index/indexer_build_provenance/caption_stage", all_dense, "r3_caption_provenance", exclude=code_exclude)
    copy_tree(index / "indexer_build_provenance/organizer_stage/formal_variant_c", "shared/dense_index/indexer_build_provenance/organizer_stage/formal_variant_c", all_dense, "organizer_prompt_and_provenance", exclude=code_exclude)
    copy_tree(index / "work_index/case_configs", "shared/dense_index/work_index/case_configs", all_dense, "input_index_case_config")
    copy_tree(index / "work_index/manifests", "shared/dense_index/work_index/manifests", all_dense, "input_index_materialization_manifest")

    embed = OUT / "dense_semantic_embeddings_eval300_v1_20260827T195154Z"
    copy_named_files(embed, "shared/dense_index/dense_text_embeddings", [
        "FINAL_EMBEDDING_BUILD_REPORT.md", "MANIFEST.sha256", "README.md", "SOURCE_LINEAGE.json",
        "VALIDATION_REPORT.json", "api_call_log.jsonl", "build_embeddings.py", "build_state.json",
    ], all_dense, "dense_embedding_build_provenance")
    fine = DATA / "hourvideo_frame_embeddings_3way_v1"
    copy_named_files(fine, "shared/dense_index/fine_siglip", ["REPORT.md", "model_manifest.json"], all_dense, "fine_embedding_manifest")
    copy_named_files(fine / "siglip", "shared/dense_index/fine_siglip/siglip", [
        "completion_manifest.json", "offline_initialization.json", "progress.json", "summary.json", "smoke_validation.json"
    ], all_dense, "fine_embedding_manifest")

    # Accepted Flat-30 runtime, launch/config, retry evidence, and final per-question reconciliation.
    flat30 = "flat30"
    flat30_runtime = VIDEOSEAL / "eval300_timeout_retry_20260819T084545Z/.runtime_snapshot"
    copy_tree(flat30_runtime / "source/videoseal", "experiments/flat30/runtime/videoseal", flat30, "runtime_prompt_source", exclude=code_exclude)
    copy_tree(flat30_runtime / "metadata", "experiments/flat30/runtime/metadata", flat30, "runtime_manifest", exclude=code_exclude)
    copy_named_files(flat30_runtime / "source", "experiments/flat30/runtime", ["pyproject.toml", "uv.lock"], flat30, "runtime_lock")
    copy_file(VIDEOSEAL / "runs_dgx_eval300_v1/experiment_config.json", "experiments/flat30/config/experiment_config.json", flat30, "frozen_config")
    copy_file(VIDEOSEAL / "runs_dgx_eval300_v1/_global_summary.json", "experiments/flat30/results/first_pass_global_summary.json", flat30, "first_pass_summary")
    retry30 = VIDEOSEAL / "eval300_timeout_retry_20260819T084545Z"
    copy_named_files(retry30, "experiments/flat30/retry", [
        "RETRY_REPORT.md", "check_retry_group.py", "finalize_retry_and_bundle.py", "original_final_retry_report.json",
        "original_merged_report.json", "original_remaining_timeout_uids.txt", "original_successfully_merged_uids.txt",
        "original_validation_report.json", "prepare_runtime_snapshot.py", "retry_item_status.jsonl", "retry_summary.json",
        "run_eval300_timeout_retry.sh", "wait_then_finalize.sh",
    ], flat30, "retry_entry_and_summary")
    copy_tree(retry30 / "manifests", "experiments/flat30/retry/manifests", flat30, "retry_manifest", exclude=lambda p: p.name.startswith("trained_"))
    copy_tree(retry30 / "original_merged", "experiments/flat30/retry/original_merged", flat30, "retry_merged_results")
    copy_tree(retry30 / "original_retry", "experiments/flat30/retry/raw_attempts", flat30, "retry_raw_attempts", exclude=code_exclude)
    copy_matching(retry30 / "logs", "experiments/flat30/retry/logs", ["original-*.log", "launcher.log", "finalizer.log"], flat30, "retry_logs")
    recon = VIDEOSEAL / "eval300_reconciliation_audit_20260822T144251Z"
    copy_named_files(recon, "experiments/flat30/results/reconciliation", [
        "MANIFEST.sha256", "invalid_predictions.tsv", "merge_provenance.tsv", "reconcile.py",
        "reconciliation_report.md", "summary.json",
    ], flat30, "final_reconciliation")
    copy_matching(recon, "experiments/flat30/results/reconciliation", ["pretrained_flat_*.txt"], flat30, "final_reconciliation_uid_sets")
    reanalysis = OUT / "eval300_authoritative_reanalysis_20260826T114630Z"
    copy_named_files(reanalysis, "experiments/flat30/results/authoritative_reanalysis", [
        "FINAL_AUDIT_REPORT.md", "MANIFEST.sha256", "decision_gate.json", "paired_statistics.json",
        "per_uid.tsv", "per_video_statistics.tsv", "reanalyze.py", "summary.json",
    ], flat30, "authoritative_reanalysis")

    # Flat-15 formal first pass, exact runtime overlay, retry-v2 raw attempts, and finalization-v2.
    flat15 = "flat15_retryv2"
    pre15 = OUT / "videoseal_flat15_eval300_preflight_v3_20260910T183640Z"
    copy_named_files(pre15, "experiments/flat15_retryv2/preflight", [
        "FINAL_CONTROLLED_DIFFERENCE_AUDIT.md", "FINAL_PREFLIGHT_REPORT.md", "MANIFEST.sha256", "README.md",
        "active_budget_audit.py", "controlled_diff.json", "flat15.env", "formal_manifest.json",
        "ordered_eval300_uids.txt", "preflight.py", "preflight_report.json", "run_flat15_eval300_formal_v1.sh",
        "service_environment_diff.json", "validation_report.json", "video_asset_inventory.json",
    ], flat15, "entry_preflight_manifest")
    overlay15 = pre15 / "runtime_overlay_v2"
    copy_tree(overlay15 / "videoseal", "experiments/flat15_retryv2/runtime/videoseal", flat15, "runtime_prompt_source", exclude=code_exclude)
    copy_named_files(overlay15, "experiments/flat15_retryv2/runtime", [".gitignore", "LICENSE", "README.md", "pyproject.toml", "uv.lock"], flat15, "runtime_lock")
    formal15 = VIDEOSEAL / "videoseal_flat15_eval300_formal_v1_20260910T184300Z"
    copy_named_files(formal15, "experiments/flat15_retryv2/config", [
        "_global_summary.json", "experiment_config.json", "frozen_ordered_eval300_uids.txt",
        "frozen_preflight_controlled_diff.json", "frozen_preflight_formal_manifest.json",
        "frozen_preflight_service_environment_diff.json",
    ], flat15, "frozen_config_manifest")
    copy_tree(formal15 / "retry_recovery_execution_20260913T123524Z", "experiments/flat15_retryv2/retry/controller", flat15, "retry_entry", exclude=code_exclude)
    copy_tree(formal15 / "retry_control_v2_20260913T124100Z", "experiments/flat15_retryv2/retry/control", flat15, "retry_control_and_cost", exclude=lambda p: code_exclude(p) or p.name in {"tmux_pane_pid.txt"})
    copy_tree(formal15 / "retry_raw_v2_20260913T124100Z", "experiments/flat15_retryv2/retry/raw_attempts", flat15, "retry_raw_attempts", exclude=code_exclude)
    final15 = VIDEOSEAL / "videoseal_flat15_eval300_finalization_v2_20260914T003140Z"
    copy_tree(final15, "experiments/flat15_retryv2/results/finalization_v2", flat15, "final_per_question_aggregation_report", exclude=code_exclude)

    # Dense shared overlay used by formal H15/H30.
    parity = OUT / "dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z"
    copy_named_files(parity, "shared/dense_h15_h30_overlay", [
        "FINAL_TOOL_SCHEMA_PARITY_REPORT.md", "MANIFEST.sha256", "RUNTIME_OVERLAY_MANIFEST.sha256",
        "lineage.json", "schema_parity_report.json", "validation_summary.json", "verify_schema.py",
    ], "dense_h15_v1;dense_h30_v2", "overlay_manifest")
    copy_tree(parity / "runtime_overlay/videoseal", "shared/dense_h15_h30_overlay/runtime_overlay/videoseal", "dense_h15_v1;dense_h30_v2", "runtime_prompt_source", exclude=code_exclude)
    copy_named_files(parity / "runtime_overlay", "shared/dense_h15_h30_overlay/runtime_overlay", ["pyproject.toml", "uv.lock"], "dense_h15_v1;dense_h30_v2", "runtime_lock")

    # Dense H8 budget-specific overlay.
    h8overlay = OUT / "dense_semantic_beam_b_h8_budget_overlay_v3_20260831T201510Z"
    copy_named_files(h8overlay, "experiments/dense_h8_v3/overlay", ["MANIFEST.sha256", "OVERLAY_LINEAGE.md"], "dense_h8_v3", "overlay_manifest")
    copy_tree(h8overlay / "runtime_overlay/videoseal", "experiments/dense_h8_v3/overlay/runtime_overlay/videoseal", "dense_h8_v3", "runtime_prompt_source", exclude=code_exclude)
    copy_named_files(h8overlay / "runtime_overlay", "experiments/dense_h8_v3/overlay/runtime_overlay", ["pyproject.toml", "uv.lock"], "dense_h8_v3", "runtime_lock")

    dense_runs = {
        "dense_h8_v3": (OUT / "dense_semantic_beam_b_h8_eval300_formal_v3_20260831T201510Z", "h8", "run_formal_h8_v3.sh"),
        "dense_h15_v1": (OUT / "dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z", "h15", "run_formal_h15.sh"),
        "dense_h30_v2": (OUT / "dense_semantic_beam_b_h30_eval300_formal_v2_20260829T155639Z", "h30", "run_formal_h30.sh"),
    }
    for experiment, (run_root, profile, launcher) in dense_runs.items():
        target = f"experiments/{experiment}"
        copy_file(run_root / launcher, f"{target}/entry/{launcher}", experiment, "formal_entry")
        copy_matching(run_root, f"{target}/entry", ["*.md", "*.sha256", "controller.log"], experiment, "launch_manifest")
        copy_tree(run_root / profile / "status", f"{target}/status", experiment, "frozen_config_status_final_report", exclude=lambda p: p.name.endswith(".pid"))
        copy_tree(run_root / profile / "merged", f"{target}/results/merged", experiment, "final_per_question_manifest")
        copy_file(run_root / profile / "first_pass/_global_summary.json", f"{target}/results/first_pass_global_summary.json", experiment, "first_pass_summary")
        copy_tree(run_root / profile / "retry_1", f"{target}/retry/raw_attempts", experiment, "retry_raw_attempts", exclude=code_exclude)
        copy_matching(run_root / profile / "logs", f"{target}/logs", ["first-runner-*.log", "retry-runner-*.log", "launch-gate-*.log"], experiment, "runner_logs")

    # Superseding reports and canonical finalization records.
    copy_tree(OUT / "dense_semantic_beam_b_h8_eval300_finalization_v1_20260902T171928Z", "experiments/dense_h8_v3/results/canonical_finalization", "dense_h8_v3", "final_report", exclude=code_exclude)
    copy_tree(OUT / "dense_semantic_beam_b_h8_superseding_amendment_v3_20260831T201510Z", "experiments/dense_h8_v3/results/superseding_amendment", "dense_h8_v3", "acceptance_amendment", exclude=code_exclude)
    copy_tree(OUT / "dense_semantic_beam_b_h15_acceptance_amendment_20260827T224330Z", "experiments/dense_h15_v1/results/acceptance_amendment", "dense_h15_v1", "acceptance_amendment", exclude=code_exclude)
    copy_tree(OUT / "dense_semantic_beam_b_h30_h8_superseding_audit_v2_20260829T155639Z", "experiments/dense_h30_v2/results/superseding_audit", "dense_h30_v2;dense_h8_v3", "superseding_audit", exclude=code_exclude)
    copy_tree(OUT / "dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z/analysis/h15_vs_flat_data_archive_20260829T152745Z", "reports/main_comparison", all_main, "main_comparison_report", exclude=code_exclude)

    # Clearly segregated Old-H appendix candidate: reports/config/summaries only, no 40 MB selected-attempt file.
    old = "appendix_old_h_candidate"
    copy_tree(OUT / "old_h_series_eval300_data_archive_20260902T172558Z", "appendix_candidates/old_h/old_h_series_archive", old, "historical_report", exclude=code_exclude)
    oldrun = OUT / "formal_eval300_20260821T095418Z"
    for profile in ("h6", "h15", "h30"):
        copy_named_files(oldrun / profile / "status", f"appendix_candidates/old_h/{profile}/status", [
            "experiment_config.json", "final_report.json", "orchestrator.jsonl",
        ], old, "historical_config_summary")
        copy_tree(oldrun / profile / "merged", f"appendix_candidates/old_h/{profile}/merged", old, "historical_manifest", exclude=lambda p: p.name == "per_question_manifest.json")
    fix = OUT / "full_video_fallback_fix_20260826T145534Z"
    copy_named_files(fix, "appendix_candidates/old_h/fallback_fix", [
        "FIX_REPORT.md", "MANIFEST.sha256", "RERUN_BLOCKER.md", "affected_fallback_attempts.tsv",
        "build_corrected_eval300.py", "core_hash_comparison.json", "remaining_rerun_manifest.tsv",
        "validate_rerun_attempt.py",
    ], old, "historical_correction")
    copy_named_files(fix / "corrected_eval300", "appendix_candidates/old_h/fallback_fix/corrected_eval300", [
        "accuracy_summary.json", "before_after_comparison.md", "efficiency_summary.json", "replacement_manifest.tsv",
    ], old, "historical_corrected_summary")


def create_flat30_derived_view() -> None:
    source = ROOT / "experiments/flat30/results/reconciliation/merge_provenance.tsv"
    destination = ROOT / "experiments/flat30/results/flat30_final_per_question_derived.tsv"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row.get("group") == "pretrained_flat"]
        fields = reader.fieldnames or []
    if len(rows) != 300:
        raise RuntimeError(f"Flat-30 derived row count is {len(rows)}, expected 300")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_results() -> dict[str, object]:
    canonical_path = ROOT / "shared/dataset/hourvideo_eval300_v1_uids.txt"
    canonical = read_uids(canonical_path)
    canonical_set = set(canonical)
    if len(canonical) != 300 or len(canonical_set) != 300:
        raise RuntimeError("canonical UID file is not 300 unique UIDs")

    experiments: dict[str, dict[str, object]] = {}

    flat30_rows = []
    with (ROOT / "experiments/flat30/results/flat30_final_per_question_derived.tsv").open(encoding="utf-8", newline="") as handle:
        flat30_rows = list(csv.DictReader(handle, delimiter="\t"))
    experiments["flat30"] = {
        "rows": len(flat30_rows),
        "uids": [row["uid"] for row in flat30_rows],
        "strict_completed": sum(row["strict_valid"] == "True" for row in flat30_rows),
        "correct": sum(row["correct"] == "True" for row in flat30_rows),
        "timeout": sum(row["raw_status"] == "timeout" for row in flat30_rows),
        "strict_invalid": sum(row["strict_valid"] != "True" and row["raw_status"] != "timeout" for row in flat30_rows),
        "expected": {"strict_completed": 254, "correct": 82, "timeout": 36, "strict_invalid": 10},
    }

    flat15_rows = list(jsonl(ROOT / "experiments/flat15_retryv2/results/finalization_v2/scored_results.jsonl"))
    failure15 = Counter(row.get("failure_class") for row in flat15_rows if not row.get("strict_complete"))
    experiments["flat15_retryv2"] = {
        "rows": len(flat15_rows),
        "uids": [row["uid"] for row in flat15_rows],
        "strict_completed": sum(bool(row.get("strict_complete")) for row in flat15_rows),
        "correct": sum(bool(row.get("correct")) for row in flat15_rows),
        "timeout": failure15.get("timeout", 0),
        "incomplete": failure15.get("incomplete_no_unique_legal_choice", 0),
        "expected": {"strict_completed": 260, "correct": 76, "timeout": 33, "incomplete": 7},
    }

    dense_expected = {
        "dense_h8_v3": {"strict_completed": 270, "correct": 78, "timeout": 29, "strict_invalid": 1},
        "dense_h15_v1": {"strict_completed": 270, "correct": 81, "timeout": 26, "strict_invalid": 4},
        "dense_h30_v2": {"strict_completed": 249, "correct": 78, "timeout": 45, "strict_invalid": 6},
    }
    for name, expected in dense_expected.items():
        rows = json.loads((ROOT / f"experiments/{name}/results/merged/per_question_manifest.json").read_text(encoding="utf-8"))
        complete = sum(bool(row.get("complete")) for row in rows)
        correct = sum(bool(row.get("complete")) and row.get("pred") == row.get("gt") for row in rows)
        timeout = sum(row.get("status") == "timeout" for row in rows)
        experiments[name] = {
            "rows": len(rows),
            "uids": [row["uid"] for row in rows],
            "strict_completed": complete,
            "correct": correct,
            "timeout": timeout,
            "strict_invalid": len(rows) - complete - timeout,
            "expected": expected,
        }

    for name, result in experiments.items():
        uids = result.pop("uids")
        uid_set = set(uids)
        result["unique_uids"] = len(uid_set)
        result["duplicate_uid_count"] = len(uids) - len(uid_set)
        result["uid_set_equals_canonical"] = uid_set == canonical_set
        result["uid_set_sha256_sorted"] = hashlib.sha256(("\n".join(sorted(uid_set)) + "\n").encode()).hexdigest()
        expected = result["expected"]
        result["numbers_match_expected"] = all(result[key] == value for key, value in expected.items())
        result["pass"] = (
            result["rows"] == 300
            and result["unique_uids"] == 300
            and result["duplicate_uid_count"] == 0
            and result["uid_set_equals_canonical"]
            and result["numbers_match_expected"]
        )

    return {
        "canonical": {
            "path": "shared/dataset/hourvideo_eval300_v1_uids.txt",
            "sha256": sha256(canonical_path),
            "rows": len(canonical),
            "unique_uids": len(canonical_set),
            "uid_set_sha256_sorted": hashlib.sha256(("\n".join(sorted(canonical_set)) + "\n").encode()).hexdigest(),
        },
        "experiments": experiments,
        "all_checks_pass": all(bool(item["pass"]) for item in experiments.values()),
    }


def scan_sensitive() -> dict[str, object]:
    patterns = {
        "openai_like_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
        "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
        "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
        "private_key_block": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "bearer_token": re.compile(rb"Bearer\s+[A-Za-z0-9._~-]{24,}", re.IGNORECASE),
    }
    candidates: list[dict[str, object]] = []
    skipped_binary = 0
    scanned_text = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in {"SENSITIVE_SCAN.json", "STAGING_CHECKSUMS.sha256"}:
            continue
        data = path.read_bytes()
        if b"\x00" in data[:8192] or path.suffix.lower() in {".parquet", ".npz", ".npy", ".jpg", ".jpeg", ".png", ".mp4"}:
            skipped_binary += 1
            continue
        scanned_text += 1
        for kind, pattern in patterns.items():
            for match in pattern.finditer(data):
                line = data.count(b"\n", 0, match.start()) + 1
                candidates.append({
                    "path": path.relative_to(ROOT).as_posix(),
                    "line": line,
                    "pattern": kind,
                    "match_sha256": hashlib.sha256(match.group(0)).hexdigest(),
                    "match_length": len(match.group(0)),
                })
    return {
        "scanner": "local_strong_pattern_scan_v1",
        "text_files_scanned": scanned_text,
        "binary_files_skipped": skipped_binary,
        "candidate_count": len(candidates),
        "confirmed_secret_count": 0,
        "redacted_file_count": 0,
        "candidates": candidates,
        "note": "Candidates contain only a one-way hash and length, never the matched value. Confirmed count is finalized after manual review.",
    }


def write_metadata(verification: dict[str, object], sensitive: dict[str, object]) -> None:
    metadata = ROOT / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    write_csv(metadata / "SOURCE_FILE_MANIFEST.csv", [
        "source_absolute_path", "source_sha256", "source_size_bytes", "staging_relative_path",
        "staged_sha256", "experiments", "category", "transformation",
    ], sorted(records, key=lambda row: str(row["staging_relative_path"])))
    write_csv(metadata / "MISSING_SOURCE_FILES.csv", [
        "source_absolute_path", "staging_relative_path", "experiments", "category", "required"
    ], missing)
    write_csv(metadata / "REDACTION_LOG.csv", [
        "staging_relative_path", "source_absolute_path", "source_sha256", "staged_sha256_after_redaction",
        "rule", "replacement", "review_status",
    ], [])
    (metadata / "VERIFICATION.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (metadata / "SENSITIVE_SCAN.json").write_text(json.dumps(sensitive, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_lines = [
        "# Eval300 staging verification",
        "",
        f"Overall: **{'PASS' if verification['all_checks_pass'] else 'FAIL'}**",
        "",
        "| Experiment | Rows | Unique UID | Same UID set | Strict complete | Correct | Timeout | Other incomplete | Result |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for name, result in verification["experiments"].items():
        other = result.get("strict_invalid", result.get("incomplete", 0))
        verify_lines.append(
            f"| {name} | {result['rows']} | {result['unique_uids']} | {result['uid_set_equals_canonical']} | "
            f"{result['strict_completed']} | {result['correct']} | {result['timeout']} | {other} | "
            f"{'PASS' if result['pass'] else 'FAIL'} |"
        )
    verify_lines.extend([
        "",
        "All UID comparisons are set comparisons against the locally staged canonical Eval300 UID file.",
        "The differing Flat-15 ordered-file byte hash does not indicate a different UID population.",
    ])
    (metadata / "VERIFICATION_REPORT.md").write_text("\n".join(verify_lines) + "\n", encoding="utf-8")

    exclusions = """# Exclusions and missing items

## Deliberately excluded

- All videos and extracted frames.
- Model weights, checkpoints, tokenizer/model caches, and vector arrays/shards.
- Runtime-generated cache directories, `__pycache__`, `.pyc`, process IDs, and unrelated third-party source trees.
- Credential files, especially the referenced `.env.embedding`; no credential file was opened or copied.
- Planner/Inspector/memory service logs that are not needed for per-question result verification.
- Full first-pass raw trajectory trees. The staging retains each condition's final 300-row per-question result, retry raw attempts, runner logs, and summaries.
- The 40 MB Old-H corrected `final_selected_attempts.jsonl`; Old H is retained only as an explicitly excluded appendix candidate with configs, manifests, summaries, and reports.

## Not available as a self-contained upload asset

- Video media and frame evidence required to re-run the experiments.
- Local model weights for Qwen3-8B, Qwen2.5-VL-7B-Instruct, and SigLIP.
- Dense Coarse/Medium embedding arrays and Fine frame-vector shards; their build reports, code, source lineage, model manifests, and validation summaries are retained.
- A formal local R1/R3 paired-comparison bundle; it was not located in the DGX-only inventory and is outside the five accepted conditions.
- Flat caption-build source-to-output SHA binding remains historically unconfirmed; the accepted downstream runtime and materialized-result evidence are retained.

No school server was accessed. School-origin paths may remain as historical strings inside locally captured DGX provenance files; no staging file depends on a live school path for inspection.
"""
    (metadata / "EXCLUSIONS_AND_MISSING.md").write_text(exclusions, encoding="utf-8")

    readme = """# DGX Eval300 paper import staging

This is a local-only staging assembled from files already present on the DGX. It contains the five accepted Eval300 conditions: Flat-30, Flat-15 retry-v2, New Dense H-8 v3, New Dense H-15 v1, and New Dense H-30 v2. Historical Old-H material is isolated under `appendix_candidates/old_h/` and is not part of the main comparison.

No model, inference server, external API, school server, Git command, or GitHub operation was used to build this staging.

## Audit entry points

- `metadata/SOURCE_FILE_MANIFEST.csv`: DGX source absolute path, original SHA-256, staged relative path, staged SHA-256, category, and experiment ownership for every copied source file.
- `metadata/VERIFICATION_REPORT.md` and `metadata/VERIFICATION.json`: 300-UID and final-number checks.
- `metadata/SENSITIVE_SCAN.json`: local strong-pattern credential scan.
- `metadata/REDACTION_LOG.csv`: redaction ledger. It is header-only when no redaction was required.
- `metadata/EXCLUSIONS_AND_MISSING.md`: intentionally omitted assets and unresolved gaps.
- `metadata/STAGING_CHECKSUMS.sha256`: SHA-256 for staged files, excluding the checksum file itself.
- `metadata/TREE.txt`: compact directory tree.

Copied source files are byte-identical to their DGX origins unless `transformation` says otherwise. The Flat-30 300-row TSV is a derived view from the retained unmodified reconciliation table and is clearly named `flat30_final_per_question_derived.tsv`.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / ".gitignore").write_text("""# Do not add local credentials or large runtime assets
.env
.env.*
*.pem
*.key
*.pt
*.pth
*.safetensors
*.bin
*.mp4
*.mkv
*.avi
__pycache__/
*.pyc
""", encoding="utf-8")


def finalize_inventory() -> None:
    metadata = ROOT / "metadata"
    tree_lines = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if path.is_dir() and len(rel.parts) <= 4:
            tree_lines.append("  " * (len(rel.parts) - 1) + rel.name + "/")
        elif path.is_file() and len(rel.parts) <= 2:
            tree_lines.append("  " * (len(rel.parts) - 1) + rel.name)
    (metadata / "TREE.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")

    generated_rows = []
    source_destinations = {str(row["staging_relative_path"]) for row in records}
    if not source_destinations and (metadata / "SOURCE_FILE_MANIFEST.csv").is_file():
        with (metadata / "SOURCE_FILE_MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
            source_destinations = {
                row["staging_relative_path"] for row in csv.DictReader(handle)
            }
    excluded_self = {"metadata/STAGING_CHECKSUMS.sha256", "metadata/GENERATED_FILE_INVENTORY.csv"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in source_destinations or rel in excluded_self:
            continue
        generated_rows.append({
            "staging_relative_path": rel,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "origin": "generated_in_staging",
        })
    write_csv(metadata / "GENERATED_FILE_INVENTORY.csv", ["staging_relative_path", "sha256", "size_bytes", "origin"], generated_rows)

    # Generate checksums after all other staging files exist. Self-hash is intentionally impossible/omitted.
    checksum_lines = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == metadata / "STAGING_CHECKSUMS.sha256":
            continue
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    (metadata / "STAGING_CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def main() -> None:
    if (ROOT / "shared").exists() or (ROOT / "experiments").exists():
        raise SystemExit("Refusing to rebuild over an existing staging payload")
    copy_sources()
    create_flat30_derived_view()
    verification = verify_results()
    if not verification["all_checks_pass"]:
        raise RuntimeError("Eval300 verification failed")
    sensitive = scan_sensitive()
    write_metadata(verification, sensitive)
    # Re-scan after generated metadata exists; the scanner excludes only its own
    # report and the checksum file, avoiding recursion while covering upload scope.
    sensitive = scan_sensitive()
    (ROOT / "metadata/SENSITIVE_SCAN.json").write_text(
        json.dumps(sensitive, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finalize_inventory()
    print(json.dumps({
        "root": str(ROOT),
        "copied_source_files": len(records),
        "missing_sources": len(missing),
        "verification_pass": verification["all_checks_pass"],
        "sensitive_candidates": sensitive["candidate_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
