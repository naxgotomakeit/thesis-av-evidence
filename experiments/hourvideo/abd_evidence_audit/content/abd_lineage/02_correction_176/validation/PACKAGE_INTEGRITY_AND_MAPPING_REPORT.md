# Correction package integrity and mapping validation

## Package integrity

- Incoming absolute path: `<MSC_USER_ROOT>/abd_correction_incoming_v1/correction_results.tar.gz`
- Archive SHA-256: `aef2cd42ebc3f39b4da6407040d210cb0c3120b2a857f96a7ae20d9578a39a96` (expected; PASS).
- Corrected review SHA-256: `3ab37d3b323018ba063688790112639cb0053de4a0d1fb0407da2f012d35c991` (expected; PASS).
- Internal SHA256SUMS: all four declared payload hashes match.
- Archive safety: 5 unique regular files; no absolute paths, traversal, links, devices, or duplicate members.
- Records: 176 rows, 176 unique IDs, exact ID and order match to the original correction input manifest.
- Schema and label values: PASS.

## Identity recovery

- Targets: exactly 116 frozen `unreviewable` records.
- Controls: exactly 60, with 20 from each original workflow; no overlap or duplicates.
- Correction audit ID equals the preserved opaque original audit ID; question/arm association used exact identity-map keys.
- Important provenance limitation: the package builder did not save a standalone target/control mapping file. The mapping in `reconstructed_identity_mapping.json` was deterministically reconstructed from the frozen ID set, batch manifest, preserved builder seed and exact identity-map keys. No question, image, map, or fuzzy matching was used.
- Frozen audit protected-tree SHA before/after validation: `cc8b33d408c76bb02376af8b49121a50fb6b1576b28aed893bfe3e7b4ee1fe8f`. It matches the correction-input build anchor.

File integrity passing does not establish semantic audit quality; scale consistency is reported separately.
