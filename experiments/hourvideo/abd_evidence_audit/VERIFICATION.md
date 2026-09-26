# Verification

Overall result: **PASS**

This verification was read-only with respect to all experimental source files. It made no model/API calls and ran no experiment.

| Check | Result | Detail |
|---|---:|---|
| source files present | PASS | `[]` |
| staged files present | PASS | `[]` |
| original SHA-256 values match | PASS | `[]` |
| staged SHA-256 values match | PASS | `[]` |
| only declared transformations | PASS | `[]` |
| ABD initial identity closure | PASS | `{"mapping": 900, "reviews": 900, "unique_ids": 900}` |
| ABD correction 176 closure | PASS | `{"mapping": 176, "outside_initial": [], "reviews": 176, "roles": {"control": 60, "target": 116}}` |
| ABD independent 31 closure | PASS | `{"mapping": 31, "outside_controls": [], "reviews": 31}` |
| ABD 900 → 176 → 31 → final 900 identity closure | PASS | `{"31_plus_145_plus_724": 900, "final_rows": 900, "source_counts": {"correction": 145, "original_abd": 724, "sol_independent_review": 31}}` |
| final_manifest payload verification | PASS | `{"errors": [], "label_freeze_hash_ok": true, "verification_modes": {"exact_staged_payload": 10, "source_original_plus_declared_path_redaction": 1}}` |
| historical R1/R3/GenS 900 closure | PASS | `{"batch_review_rows": 900, "frozen_reviews": 900, "mapping": 900}` |
| no ABD/historical identity conflation | PASS | `{"ABD_identity_fields": ["audit_id", "variant", "question_id", "task_id"], "historical_identity_fields": ["neutral_id", "source_group", "source_identity", "question_id"], "same_semantic_tuple_at_same_label": 0, "separate_staging_roots": true, "shared_local_E_labels": 900}` |
| uploadable sensitive scan | PASS | `{"credential_hits": [], "env_files": [], "files_scanned": 2399, "personal_absolute_path_files": []}` |

For final-manifest members whose only staged change is declared absolute-path redaction, verification checks the manifest against the source original SHA/byte count and separately checks the staged SHA recorded in `SOURCE_MANIFEST.csv`.
