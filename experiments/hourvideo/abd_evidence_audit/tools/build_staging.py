#!/usr/bin/env python3
"""Build a copy-only, allowlisted archival staging tree from an existing workspace.

This script performs no model/API calls and never writes to the source workspace.
It is retained in the archive so that the inclusion policy is inspectable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".sha256", ".txt"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("staging_root", type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    staging = args.staging_root.resolve()

    if not (source / "outputs/abd_eval300_evidence_audit_final_v1/final_manifest.json").is_file():
        raise SystemExit("source_root does not contain the expected final ABD manifest")
    if not (staging / "tools/build_staging.py").is_file():
        raise SystemExit("staging_root must already contain tools/build_staging.py")
    if (staging / "content").exists():
        raise SystemExit("refusing to overwrite an existing staging content tree")

    copies: list[tuple[Path, Path, str]] = []

    def add_file(src_rel: str, dst_rel: str, role: str) -> None:
        copies.append((source / src_rel, staging / dst_rel, role))

    def add_dir(src_rel: str, dst_rel: str, role: str, exclude_names: set[str] | None = None) -> None:
        excluded = exclude_names or set()
        base = source / src_rel
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and path.name not in excluded
                and "__pycache__" not in path.parts
                and path.suffix.lower() not in {".pyc", ".pyo"}
            ):
                rel = path.relative_to(base)
                copies.append((path, staging / dst_rel / rel, role))

    # ABD formal results, exact runtime records, frozen configuration, and implementation.
    add_dir(
        "outputs/abd_eval300_analysis_v1",
        "content/abd_lineage/00_formal/analysis",
        "ABD formal analysis result/report/manifest",
        {"ABD_RESULTS_PACKAGE.tar.gz"},
    )
    add_dir(
        "outputs/abd_eval300_formal_v1",
        "content/abd_lineage/00_formal/raw_run",
        "ABD formal raw per-task/run record",
        {"runner.lock", "launch_authorized_remaining_897.sh"},
    )
    for name in ["abd_common_system_prompt_v1.txt", "abd_draft_v1.json", "abd_formal_candidate_v1.json"]:
        add_file(f"config/{name}", f"content/abd_lineage/00_formal/config/{name}", "ABD frozen configuration/prompt")
    add_dir("src/abd_draft_v1", "content/abd_lineage/00_formal/runtime_source/abd_draft_v1", "ABD runtime source")

    draft_files = [
        "COMMON_ABD_PROMPT_AND_EXACT_DIFF.json",
        "COMMON_ABD_PROMPT_AND_EXACT_DIFF.md",
        "ORIGINAL_GENS_V3_PROMPT_AND_CONTRACT.md",
        "OUTPUT_CONTRACT_COMPARISON.md",
        "README.md",
        "original_gens_v3_contract.json",
        "formal_candidate/FROZEN_FILES_AND_RECOVERY.md",
        "formal_candidate/PREFLIGHT_REPORT.md",
        "formal_candidate/capacity_per_task.csv",
        "formal_candidate/formal_candidate_manifest.json",
        "formal_candidate/freeze_record.json",
        "formal_candidate/preflight_report.json",
        "limited_batch_v1/BATCH_PREFLIGHT_REPORT.json",
        "limited_batch_v1/BATCH_PREFLIGHT_REPORT.md",
        "limited_batch_v1/authorization_APPROVED_FIRST_QUESTION_ABD.json",
        "limited_batch_v1/batch_control_manifest.json",
        "limited_batch_v1/execution_identity_mapping.json",
        "continuation_v1/TARGETED_PREFLIGHT_REPORT.json",
        "continuation_v1/authorization_APPROVED_REMAINING_897.json",
        "continuation_v1/continuation_control_manifest.json",
        "continuation_v1/execution_identity_mapping.json",
    ]
    for rel in draft_files:
        add_file(
            f"drafts/abd_direct_eval300_v1/{rel}",
            f"content/abd_lineage/00_formal/frozen_inputs_and_controls/{rel}",
            "ABD frozen launch/input identity provenance",
        )

    abd_scripts = [
        "analyze_abd_results_v1.py",
        "preflight_abd_formal_candidate_v1.py",
        "run_abd_formal_v1.py",
        "preflight_abd_limited_batch_v1.py",
        "run_abd_limited_batch_v1.py",
        "preflight_abd_continuation_v1.py",
        "run_abd_continuation_v1.py",
        "freeze_abd_raw_closure_v1.py",
        "prepare_abd_evidence_audit_v1.py",
        "verify_abd_evidence_audit_inputs_v1.py",
        "merge_abd_audit_batch_fragments_v1.py",
        "register_abd_valid_second_pass_batches_v1.py",
        "freeze_abd_blinded_audit_v1.py",
        "finalize_abd_evidence_audit_v1.py",
        "build_abd_evidence_audit_correction_blinded_inputs_v1.py",
        "validate_abd_correction_results_v1.py",
        "diagnose_abd_audit_disagreements_v1.py",
        "prepare_abd_disagreement_independent_review_v1.py",
        "freeze_abd_disagreement_independent_review_v1.py",
        "compare_abd_disagreement_independent_review_v1.py",
        "analyze_abd_strict_support_disagreements_v1.py",
        "finalize_abd_evidence_audit_final_v1.py",
        "score_abd_formal_v1.py",
    ]
    for name in abd_scripts:
        add_file(f"scripts/{name}", f"content/abd_lineage/code/{name}", "ABD generation/freeze/validation/comparison script")

    # Initial ABD evidence audit: authoritative root artifacts plus the 192 accepted review batches.
    initial = source / "outputs/abd_eval300_evidence_audit_v1"
    for path in sorted(initial.iterdir()):
        if path.is_file() and path.name != "ABD_EVIDENCE_AUDIT_RESULTS.tar.gz":
            copies.append((path, staging / "content/abd_lineage/01_initial_audit_900" / path.name, "ABD initial 900 freeze/report/reproducibility artifact"))
    add_dir(
        "outputs/abd_eval300_evidence_audit_v1/reviews",
        "content/abd_lineage/01_initial_audit_900/reviews",
        "ABD initial accepted review batch",
    )

    # Correction inputs: retain the minimum complete metadata, not the archive or unpacked images/maps.
    corr_base = "outputs/abd_eval300_evidence_audit_correction_inputs_v1"
    for rel in [
        "BUILD_REPORT.json",
        "ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS.tar.gz.sha256",
        "ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS/README.md",
        "ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS/MANIFEST.json",
        "ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS/MANIFEST.sha256",
    ]:
        add_file(f"{corr_base}/{rel}", f"content/abd_lineage/02_correction_176/input_package_metadata/{rel}", "ABD 176 correction input package manifest/SHA provenance")
    add_dir(
        "outputs/abd_eval300_evidence_audit_correction_validation_v1",
        "content/abd_lineage/02_correction_176/validation",
        "ABD 176 correction returned reviews/validation/freeze",
    )

    # Independent 31: keep frozen outputs and manifests; omit reviewer item images/maps/input payload.
    independent_root = "outputs/abd_eval300_disagreement_independent_review_v1"
    for name in [
        "BLINDED_REVIEW_FREEZE.json",
        "POST_FREEZE_COMPARISON_MANIFEST.json",
        "POST_FREEZE_COMPARISON_REPORT.md",
        "PREPARATION_FREEZE.json",
        "TARGET_116_HANDLING_RECOMMENDATION.md",
        "THREE_WAY_COMPARISON.csv",
        "identity_mapping_private.json",
    ]:
        add_file(f"{independent_root}/{name}", f"content/abd_lineage/03_independent_review_31/{name}", "ABD 31 independent review freeze/comparison/mapping")
    add_dir(
        f"{independent_root}/blinded_reviewer_output",
        "content/abd_lineage/03_independent_review_31/blinded_reviewer_output",
        "ABD 31 frozen independent review output",
    )
    for name in ["AUDIT_CRITERIA_FROZEN.md", "UNIFIED_REVIEW_INSTRUCTIONS.md", "batch_manifest.json", "package_manifest.json"]:
        add_file(f"{independent_root}/reviewer_materials/{name}", f"content/abd_lineage/03_independent_review_31/reviewer_materials/{name}", "ABD 31 reviewer package metadata/instructions")

    add_dir(
        "outputs/abd_eval300_evidence_audit_disagreement_diagnostic_v1",
        "content/abd_lineage/04_derived_diagnostics/disagreement_diagnostic",
        "ABD disagreement diagnostic derived statistic/report",
    )
    add_dir(
        "outputs/abd_eval300_strict_support_disagreement_stats_v1",
        "content/abd_lineage/04_derived_diagnostics/strict_support_statistics",
        "ABD strict-support derived statistic/report",
    )
    add_dir(
        "outputs/abd_eval300_evidence_audit_final_v1",
        "content/abd_lineage/05_final_authoritative_900",
        "ABD final authoritative freeze artifact",
    )

    # Separate historical R1/R3/GenS evidence audit namespace.
    historical = source / "audit/codex_evidence_audit_r1_r3_gens_v3_v1"
    for path in sorted(historical.iterdir()):
        if path.is_file():
            copies.append((path, staging / "content/historical_r1_r3_gens_audit_900/audit" / path.name, "Historical R1/R3/GenS 900 freeze/report/manifest"))
    add_dir(
        "audit/codex_evidence_audit_r1_r3_gens_v3_v1/reviews",
        "content/historical_r1_r3_gens_audit_900/audit/reviews",
        "Historical R1/R3/GenS accepted review batch",
    )
    historical_scripts = [
        "materialize_direct_evidence_audit.py",
        "prepare_codex_evidence_audit_packages.py",
        "render_codex_evidence_audit_contact_sheets.py",
        "update_codex_evidence_audit_progress.py",
        "freeze_codex_evidence_audit.py",
        "summarize_codex_evidence_audit.py",
        "validate_codex_evidence_audit_gold_scores.py",
    ]
    for name in historical_scripts:
        add_file(f"scripts/{name}", f"content/historical_r1_r3_gens_audit_900/code/{name}", "Historical R1/R3/GenS audit preparation/freeze/summary script")

    # Validate the allowlist and destinations before writing.
    missing = [str(src) for src, _, _ in copies if not src.is_file()]
    if missing:
        raise SystemExit("missing allowlisted source files:\n" + "\n".join(missing))
    destinations = [dst for _, dst, _ in copies]
    if len(destinations) != len(set(destinations)):
        raise SystemExit("duplicate staging destination in allowlist")

    project_user_root = source.as_posix().split("/msc_thesis/", 1)[0]
    replacements = [
        (str(source), "<SOURCE_WORKSPACE>"),
        (project_user_root, "<PROJECT_MSC_USER_ROOT>"),
        (str(staging.parent), "<MSC_USER_ROOT>"),
    ]
    manifest_rows: list[dict[str, str | int]] = []
    for src, dst, role in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha256(src)
        shutil.copyfile(src, dst)
        transform = "exact_copy"
        replacements_made = 0
        if dst.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = dst.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            changed = text
            for needle, replacement in replacements:
                n = changed.count(needle)
                if n:
                    replacements_made += n
                    changed = changed.replace(needle, replacement)
            if changed != text:
                dst.write_text(changed, encoding="utf-8")
                transform = "absolute_personal_path_redaction_only"
        manifest_rows.append(
            {
                "source_portable_path": src.relative_to(source).as_posix(),
                "original_sha256": original_hash,
                "original_bytes": src.stat().st_size,
                "staged_path": dst.relative_to(staging).as_posix(),
                "staged_sha256": sha256(dst),
                "staged_bytes": dst.stat().st_size,
                "role": role,
                "transformation": transform,
                "path_replacements": replacements_made,
            }
        )

    with (staging / "SOURCE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    donot = staging / "DO_NOT_COMMIT"
    donot.mkdir(exist_ok=True)
    (donot / "SOURCE_LOCATIONS_LOCAL_ONLY.md").write_text(
        "# Local-only source location\n\n"
        "Do not commit this directory. It intentionally records a machine/user-specific path for local provenance.\n\n"
        f"- `SOURCE_WORKSPACE`: `{source}`\n"
        "- No videos, frames, contact sheets, maps, model weights, caches, virtualenvs, credentials, or omitted archives were copied here.\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": "github_abd_evidence_audit_import_staging_v1",
        "source_files_copied": len(manifest_rows),
        "exact_copies": sum(r["transformation"] == "exact_copy" for r in manifest_rows),
        "path_redacted_copies": sum(r["transformation"] != "exact_copy" for r in manifest_rows),
        "model_or_api_calls": 0,
        "source_workspace_writes": 0,
        "git_operations": 0,
    }
    (staging / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
