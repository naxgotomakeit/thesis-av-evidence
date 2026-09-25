#!/usr/bin/env python3
"""Verify the local DGX staging without invoking models, APIs, Git, or network."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/naxucl/projects/thesis-av-evidence/github_import_staging_eval300_dgx_20260925")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest_path = ROOT / "metadata/SOURCE_FILE_MANIFEST.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    staged_paths = [row["staging_relative_path"] for row in rows]
    errors: list[str] = []
    if len(staged_paths) != len(set(staged_paths)):
        errors.append("duplicate staging_relative_path in source manifest")

    for row in rows:
        source = Path(row["source_absolute_path"])
        staged = ROOT / row["staging_relative_path"]
        if not source.is_file():
            errors.append(f"missing source: {source}")
            continue
        if not staged.is_file():
            errors.append(f"missing staged file: {staged}")
            continue
        source_actual = sha256(source)
        staged_actual = sha256(staged)
        if source_actual != row["source_sha256"]:
            errors.append(f"source changed/hash mismatch: {source}")
        if staged_actual != row["staged_sha256"]:
            errors.append(f"staged hash mismatch: {staged}")
        if row["transformation"] == "none_byte_identical" and source_actual != staged_actual:
            errors.append(f"non-identical untransformed copy: {staged}")

    checksum_errors = 0
    checksum_path = ROOT / "metadata/STAGING_CHECKSUMS.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            checksum_errors += 1
    if checksum_errors:
        errors.append(f"staging checksum errors: {checksum_errors}")

    verification = json.loads((ROOT / "metadata/VERIFICATION.json").read_text(encoding="utf-8"))
    if not verification.get("all_checks_pass"):
        errors.append("Eval300 UID/result verification is not PASS")
    sensitive = json.loads((ROOT / "metadata/SENSITIVE_SCAN.json").read_text(encoding="utf-8"))
    if sensitive.get("candidate_count") or sensitive.get("confirmed_secret_count"):
        errors.append("sensitive scan has unresolved findings")

    forbidden_suffixes = {
        ".mp4", ".mkv", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".webp",
        ".pt", ".pth", ".safetensors", ".onnx", ".ckpt", ".npy", ".npz",
    }
    forbidden_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden_files:
        errors.append(f"forbidden binary assets found: {len(forbidden_files)}")
    symlinks = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_symlink()]
    if symlinks:
        errors.append(f"symlinks found: {len(symlinks)}")
    git_dirs = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob(".git")]
    if git_dirs:
        errors.append(f"Git metadata found: {len(git_dirs)}")

    files = [path for path in ROOT.rglob("*") if path.is_file()]
    result = {
        "status": "PASS" if not errors else "FAIL",
        "source_manifest_rows": len(rows),
        "all_source_copies_byte_verified": not any("source" in item or "staged" in item or "copy" in item for item in errors),
        "checksum_entries_verified": len(checksum_path.read_text(encoding="utf-8").splitlines()),
        "total_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "largest_file_bytes": max(path.stat().st_size for path in files),
        "sensitive_candidates": sensitive.get("candidate_count"),
        "forbidden_binary_assets": len(forbidden_files),
        "symlinks": len(symlinks),
        "git_metadata_dirs": len(git_dirs),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
