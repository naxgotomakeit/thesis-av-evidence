# DGX Eval300 paper import staging

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
