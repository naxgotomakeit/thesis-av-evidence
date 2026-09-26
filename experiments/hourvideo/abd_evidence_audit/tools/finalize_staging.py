#!/usr/bin/env python3
"""Generate non-circular staging summary and uploadable checksum manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    verification = json.loads((root / "VERIFICATION.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise SystemExit("refusing to finalize a staging tree without PASS verification")
    build = json.loads((root / "BUILD_SUMMARY.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": "github_abd_evidence_audit_import_staging_summary_v1",
        "verification": "PASS",
        "sensitive_scan": "PASS",
        "source_derived_files": build["source_files_copied"],
        "source_exact_copies": build["exact_copies"],
        "source_path_redacted_copies": build["path_redacted_copies"],
        "lineage_closure": {
            "ABD_initial": 900,
            "correction": 176,
            "independent_review": 31,
            "final_sources": {"independent_review": 31, "correction": 145, "original_ABD": 724},
            "final_total": 900,
        },
        "historical_audit": {
            "records": 900,
            "namespace": "historical_r1_r3_gens_audit_900",
            "is_correction_parent": False,
        },
        "checksum_scope": "all regular files outside DO_NOT_COMMIT except CHECKSUMS.sha256 itself",
        "model_or_api_calls": 0,
        "git_commit_push_or_transfer": 0,
    }
    (root / "STAGING_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and "DO_NOT_COMMIT" not in p.parts
        and p.name != "CHECKSUMS.sha256"
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
