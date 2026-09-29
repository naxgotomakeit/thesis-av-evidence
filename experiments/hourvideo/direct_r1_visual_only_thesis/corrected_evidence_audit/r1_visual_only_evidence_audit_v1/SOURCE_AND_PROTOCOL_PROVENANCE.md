# Source and protocol provenance

- Preparation package: `/cs/student/project_msc/2025/rai/xinanx01/r1_visual_only_evidence_audit_preparation_v1`
- Preparation checklist SHA-256: `ca374f35505e2c5800ed2a51ce64825de066c5241889155a4c3600e1f24c6494`
- Thesis data package: `/cs/student/project_msc/2025/rai/xinanx01/direct_visual_only_eval300_thesis_data_v1`
- Thesis per-question CSV SHA-256: `352e22f0b445eca99797f351402d1e3ecf59d06d07ad3e0f35e504c12388dc07`
- Old final label artifact: `/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/main_system/isolated_workspaces/hourvideo_direct_api_eval300_v1/audit/codex_evidence_audit_r1_r3_gens_v3_v1/evidence_audit_frozen.jsonl`
- Old final label SHA-256: `120fa48bba3fdba4dc53a71e160c4def946910e9e018764a9df777b32bc8c329`
- Reuse validation: `../r1_visual_only_evidence_audit_preparation_v1/reuse_125_label_validation.json` (125/125 pass; gold_loaded=false)
- New audit execution instruction: `REVIEW_EXECUTION_PROTOCOL.md`
- New labels: 175 routes reviewed, including 174 answered and one no_final_answer.
- Reused labels: 125 old final records, each linked to old neutral ID, old frozen artifact SHA, old package SHA, and unchanged new route artifact.
- Gold linkage occurred only after the 300-record frozen label file existed and passed uniqueness/schema checks.

Protocol differences retained in the report: exact old/current build equivalence unverified; R1-only rather than cross-method mixing; clarified Direct evidence-source enum without changing support categories.
