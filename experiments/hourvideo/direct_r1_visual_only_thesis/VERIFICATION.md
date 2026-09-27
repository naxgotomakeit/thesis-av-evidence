# Verification

Status: **PASS**

Verification was read-only against the source experiments. No model, API,
inference, scoring, or experiment execution was performed.

## Source and staged-file integrity

- Source-manifest rows: 1,688.
- Missing source files: 0.
- Missing staged files: 0.
- Original SHA-256 mismatches: 0.
- Staged SHA-256 mismatches: 0.
- Byte-for-byte copies: 79.
- Explicit personal-path-redaction copies: 1,609.

## Thesis result identity

- Corrected R1: 85/300, 299 valid/completed answers — PASS.
- Frozen R3: 103/300, 298 valid/completed answers — PASS.
- Old formal R1: 88/300, 300 predictions — confirmed and segregated as
  `SUPERSEDED_FOR_THESIS`.
- Partition: 175 rerun R1 + 125 reused R1 = 300 — PASS.
- Staged rerun route artifacts / statuses / unique question IDs:
  175 / 175 / 175 — PASS.

## Lineage and hash closure

- `route_reuse_manifest_v8`: 175 rerun, 125 reusable R1, 300 reusable R3;
  both reuse groups marked fully eligible — PASS.
- Reused source artifacts verified by original SHA-256: 425/425 — PASS.
- Final merged R1/R3 source artifacts verified by original SHA-256:
  600/600 — PASS.
- `FINAL_AUDIT_SHA256.txt` entries verified: 19/19 — PASS.
- Final v8 code snapshot hashes verified: 15/15 — PASS.
- Final report, no-gold merge, scored per-question output, accuracy summary,
  v8-v13 manifests, and merge/score script hashes all match the frozen audit
  chain — PASS.

The source chronology is also consistent: the old 88 canonical summary
predates the corrected 85 final report, which predates the thesis data
package. No 85-to-88 successor lineage was found; the evidenced direction is
old 88 to corrected 85.

Five formally unknown provider outcomes remain documented limitations. Their
missing tokens, cost, responses, and latency were not fabricated or filled
with zero.
