# ABD Formal and Evidence-Audit Archival Import

This staging tree is a copy-only, audit-oriented import package for `thesis-av-evidence`. It contains the ABD formal run, evidence audit, correction, independent review, derived diagnostics, and final freeze, together with a separate historical R1/R3/GenS evidence audit used for comparison statistics.

No model, external API, or experiment was run while building this package. No source workspace file was modified, moved, or deleted. No Git commit, push, or transfer was performed.

## Two separate 900-record namespaces

The repeated labels `E0001` through `E0900` are local identifiers, not global experiment identities.

1. `content/abd_lineage/` is the ABD lineage: formal A/B/D run → initial ABD audit 900 → correction review 176 (116 targets + 60 controls) → independent review of 31 disagreement controls → final authoritative 900. The final selection is 31 independent-review records + 145 remaining correction records + 724 original ABD records = 900.
2. `content/historical_r1_r3_gens_audit_900/` is the earlier R1 Direct / R3 Direct / GenS historical evidence audit. It has its own identity mapping, freeze, reviews, and reports. It supports historical comparison statistics, but it is **not** the parent of the ABD correction set.

Never join these namespaces on `E####` alone. Within ABD, use `audit_id` with the ABD identity mapping (`variant`, `question_id`, `task_id`). Within the historical audit, use `neutral_id` with its historical identity mapping (`source_group`, `source_identity`, `question_id`).

## Integrity and portability

`SOURCE_MANIFEST.csv` records a portable source-relative path, original SHA-256, staged path, staged SHA-256, role, and any transformation. Exact experimental prompts and result content are retained. Machine/user-specific absolute path prefixes were replaced in uploadable text copies with portable placeholders; those rows are explicitly marked `absolute_personal_path_redaction_only`. Original hashes remain recorded, and closure verification checks the source originals as well as the staged copies.

The final manifest's payload is verified against original source hashes; exact staged copies are checked directly, while path-redacted copies are linked through `SOURCE_MANIFEST.csv`. `CHECKSUMS.sha256` covers the uploadable tree except itself. `DO_NOT_COMMIT/` is local-only and excluded from uploadable counts and checksums.

## Scope

- `00_formal`: formal analysis, raw per-task/run records, frozen configuration and prompt, launch controls, runtime source, and formal scripts.
- `01_initial_audit_900`: accepted audit reviews, freeze, manifests, reports, and reproducibility records.
- `02_correction_176`: minimal correction-package manifest/SHA provenance plus corrected reviews, freeze, validation, and reconstructed identity mapping.
- `03_independent_review_31`: frozen reviewer output, blinded-review freeze, private post-review mapping, and post-freeze comparison.
- `04_derived_diagnostics`: disagreement diagnostic and strict-support statistics.
- `05_final_authoritative_900`: every final authoritative artifact from `abd_eval300_evidence_audit_final_v1`.
- `historical_r1_r3_gens_audit_900`: historical comparison audit freeze, reports, scored records, accepted review batches, manifests, and scripts.

See `INCLUDED_EXCLUDED.md`, `VERIFICATION.md`, and `SENSITIVE_SCAN_REPORT.md` before importing.
