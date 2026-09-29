# Corrected R1 visual-only evidence-support audit

This directory preserves the submitted-thesis corrected-R1 visual-only evidence-support results/provenance package. The [recovered package](r1_visual_only_evidence_audit_v1/) is preserved byte-for-byte; this navigation note does not change thesis result authority or reinterpret its judgments.

The final audit population is **300 questions**: **175 newly reviewed** under the corrected visual-only audit flow (174 answered routes and one without a final answer), plus **125 historical labels reused only after recorded validation**.

| Final label | Count |
|---|---:|
| supported | 58 |
| partially_supported | 118 |
| unsupported | 86 |
| contradicted | 37 |
| unreviewable | 0 |
| no_final_answer | 1 |

The corrected R1 correctness closure is **85/300, with 299 answered**. The frozen labels were finalized before correctness was joined, as recorded by the execution protocol and finalization code. Evidence support and answer correctness are distinct measures.

## Evidence and integrity

- [Final report](r1_visual_only_evidence_audit_v1/R1_VISUAL_ONLY_EVIDENCE_AUDIT_FINAL_REPORT.md), [summary](r1_visual_only_evidence_audit_v1/r1_visual_only_evidence_audit_summary.json), [300 frozen labels](r1_visual_only_evidence_audit_v1/r1_visual_only_evidence_labels_frozen.jsonl), and [post-freeze outcome join](r1_visual_only_evidence_audit_v1/r1_visual_only_evidence_labels_with_outcomes.csv).
- [175 batch judgments](r1_visual_only_evidence_audit_v1/reviews/), [125-label reuse validation](r1_visual_only_evidence_audit_v1/reuse_125_label_validation.json), [execution protocol](r1_visual_only_evidence_audit_v1/REVIEW_EXECUTION_PROTOCOL.md), and [source/protocol provenance](r1_visual_only_evidence_audit_v1/SOURCE_AND_PROTOCOL_PROVENANCE.md).
- The unchanged [SHA256SUMS.txt](r1_visual_only_evidence_audit_v1/SHA256SUMS.txt) covers 65 files. Run `sha256sum -c SHA256SUMS.txt` inside the recovered package. The manifest itself and `sha256_validation.log` are not entries in that manifest.

## Limits and publication boundary

This is a **POST-REVIEW / DEBLINDED RESULTS ARCHIVE**, not the original blinded review input package. Final labels contain identity linkage; the outcome join contains predictions and correctness. Do not provide these as blinded-review inputs.

Historical absolute school paths are preserved as provenance and are not credentials. Referenced preparation inputs, source frames/media, external review and gold/correctness source packages, and `identity_mapping_PRIVATE.json` are not included. The preparation verification log describes a separate historical package, not additional files bundled here. Finalization code depends on external inputs; this is not a self-contained semantic re-review bundle.

The exact historical reviewer software/model build is not preserved. Do not join neutral IDs across audit namespaces without `audit_origin` / source identity / question ID; new-review and reused historical IDs can overlap. The preparation-stage consistency report and inherited review wording remain unchanged, including R3-named evidence references in five reused records; this navigation note does not resolve or revise those references.
