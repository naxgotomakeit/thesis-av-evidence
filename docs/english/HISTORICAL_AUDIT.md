# Historical R1/R3/GenS evidence audit

English presentation summary, not a replacement experimental record. The linked frozen originals remain authoritative for byte-level provenance. Numerical values and artifact identities reproduced here are copied without alteration; this summary does not introduce new results or revise frozen labels. See the [source register](../ENGLISH_READING_GUIDE.md) for original-file hashes and checksum membership.

[Frozen report](../../experiments/hourvideo/abd_evidence_audit/content/historical_r1_r3_gens_audit_900/audit/EVIDENCE_AUDIT_REPORT.md).

This is the historical R1 Direct / R3 Direct / GenS v3 population, not the ABD population. The Codex-assisted audit covers 900 routes in 192 batches, without duplicate or missing records. It is not human review or an independent reviewer-model audit.

| Method | Historical correct / 300 | Final answers | supported | partially_supported | unsupported | contradicted | unreviewable | no_final_answer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 88 | 300 | 67 | 133 | 69 | 28 | 3 | 0 |
| R3 | 103 | 298 | 78 | 127 | 48 | 42 | 3 | 2 |
| GenS v3 | 93 | 300 | 36 | 90 | 156 | 12 | 6 | 0 |

R1 88 is the earlier formal result, superseded for the thesis by visual-only 85/300. Historical labels are not automatically applicable to the rerun evidence chain.

Correct answers labelled unsupported/contradicted number 15 for R1, 24 for R3 and 48 for GenS v3. Partially supported is excluded from that metric. Accuracy and evidence sufficiency are different measures; an audit label must not replace official scoring or claim access to the model's latent reasoning.

After all 192 batch records were written and 900/900 coverage was checked, a broad file search exposed a small amount of existing gold information before the merged SHA was recorded. The report states that batch reviews were not subsequently changed. Preserve this timing limitation: the workflow was not strictly blind throughout.

Direct evidence-source labels were inconsistent in early batches: 106 R1 and 115 R3 records used `not_applicable`. They cannot form a reliable causal distribution of map versus image dependence. The report calls for separate preregistered review rather than post-score relabelling.

The report's GenS v2→v3 answer-stage comparison records 59→93 correct, with 41 newly correct and seven formerly correct lost. Among newly correct records, support labels are 1 supported, 9 partial, 29 unsupported and 2 unreviewable. Prompt, interface and forced-choice changes prevent treating this as a pure parser effect. This answer-stage v2 label must not be confused with the Myriad GenS V2 frame-selector identity.

The report retains 12 unreviewable records and 190 further-review flags. Case selection used the first two examples per method/support category in canonical Eval300 order, not favourable correctness outcomes.

[Scored rows](../../experiments/hourvideo/abd_evidence_audit/content/historical_r1_r3_gens_audit_900/audit/scored_evidence_audit.jsonl) / [freeze](../../experiments/hourvideo/abd_evidence_audit/content/historical_r1_r3_gens_audit_900/audit/evidence_audit_freeze.json).
