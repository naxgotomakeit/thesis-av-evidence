# Blinded independent evidence review

Status: complete.

This was a Codex semantic review performed with `gpt-5.6-sol`. It used only the frozen criteria, unified instructions, anonymous manifests, and anonymous item evidence in the permitted reviewer-materials package. No identity mapping, gold/correctness data, method/group labels, prior reviews, diagnostics, or original-answer reasons were used. External API calls: 0.

## Coverage

- Batches completed: R01-R08 (8/8)
- Records reviewed: 31/31
- Complete maps read: 23/23
- Supplied JPEGs visually opened: 233/233
- Review IDs: 31 unique; exact match to package manifest

## Label totals

- `supported`: 5
- `partially_supported`: 6
- `unsupported`: 12
- `contradicted`: 8
- `unreviewable`: 0

Conservative coverage rules were applied to counts, durations, ordering, immediately-next events, non-occurrence, and exhaustive whole-video inventories. Missing modalities were treated as evidence limitations rather than technical failures. One question-relevant evidence limitation was an apparently semantically unrelated supplied map; this was flagged within the affected anonymous record without seeking outside evidence.

## Validation and artifacts

All JSONL records parse successfully, contain the required fields, use only the permitted five labels, and have image-opening attestations matching the manifest. `missing_key_conditions` is now a non-empty string in all 31 records: concrete missing conditions are stated where applicable, while judgments with no missing key condition explicitly begin with `None;`. This schema-only repair did not change labels, rationales, evidence references, flags, map attestations, or image counts.

Per-batch atomic review and viewing files are present. The merged deliverable is `blinded_reviews.jsonl`; it exactly matches the ordered concatenation of R01-R08. SHA-256 verification also matched the anonymous manifest for all 31 inputs, 23 maps, and 233 images. Completion details and the rechecked SHA-256 values for all nine JSONL artifacts are in `progress.json`.
