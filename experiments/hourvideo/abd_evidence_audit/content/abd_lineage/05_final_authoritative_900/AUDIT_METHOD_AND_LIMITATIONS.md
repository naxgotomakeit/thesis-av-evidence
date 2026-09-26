# Audit method and limitations

The ABD evidence audit is a Codex-assisted evidence-support judgment over 900 routes (300 each for A, B, and D). Reviewers used only the map, images, and model-visible timestamps actually supplied to the answer model. The five labels remain distinct: `supported`, `partially_supported`, `unsupported`, `contradicted`, and `unreviewable`.

Reviews were completed in batches and included limited blinded re-review. The final per-record selection rule was fixed before correctness linkage: **independent Sol review > correction review > original ABD review**. This produced 31 Sol-selected records, 145 correction-selected records, and 724 original-ABD-selected records. All source records and historical versions remain preserved; the final file records the selected source and SHA for every route.

This must not be described as a fresh Sol review of all 900 records, nor as complete reviewer agreement. The previously reported five-class and strict-supported consistency analyses apply only to the records actually checked. Those selected disagreement/control samples do not estimate the error rate across all 900 routes. The audit is model-assisted interpretation, not objective ground truth and not an observation of the answering model's latent reasoning.

Strict support rate is `supported / 300` per arm. `partially_supported`, `unsupported`, and `contradicted` retain separate meanings; “non-supported” is only a derived binary grouping and must not be renamed `unsupported`. Correctness was opened only after the final label artifact was frozen and verified.
