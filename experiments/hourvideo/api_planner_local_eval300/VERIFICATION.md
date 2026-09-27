# Verification

Status: **PASS**

Verification was read-only against the source experiment. No model, API,
inference, scoring, or experiment execution was performed.

## Source and staged-file integrity

- Source-manifest rows: 4,820.
- Missing source files: 0.
- Missing staged files: 0.
- Original SHA-256 mismatches: 0.
- Staged SHA-256 mismatches: 0.
- Byte-for-byte copies: 4,215.
- Explicit personal-path-redaction copies: 605.

## Thesis result identity

- R1 correct: 78/300 — PASS.
- R3 correct: 77/300 — PASS.
- Planner API cost: R1 `$4.965456999999997`, R3
  `$10.096272999999993` — PASS.
- Planner-only input tokens: R1 `2,870,342`, R3 `8,510,653` — PASS.
- Full-chain Planner + post-Planner tokens: R1 `42,709,209`, R3
  `36,680,380` — PASS.
- Eval300 UID rows / unique UIDs: 300 / 300 — PASS.

## Per-question evidence

- `answers_blind.json`: 300.
- R1 Planner / final answer / route status: 299 / 293 / 300.
- R3 Planner / final answer / route status: 300 / 292 / 300.
- Missing predictions agree with canonical accuracy: R1 7, R3 8 — PASS.

## Formal raw closure

The 5,966 files enumerated by the original canonical manifest were checked
against the current school source one by one:

- Missing raw files: 0.
- Raw file SHA-256 mismatches: 0.
- Raw byte-size mismatches: 0.
- Recomputed file count: 5,966.
- Recomputed total bytes: 184,533,682.
- Recomputed tree SHA-256:
  `020ad1a5d12805a3bf03c5d719911aa9fc98deed7c3be1d89e5bf0218c37bcd8`.
- Canonical before/after closure identities match — PASS.

The staged upload subset intentionally omits duplicated large logs identified
in `INCLUDED_EXCLUDED.md`; the complete raw inventory and hashes remain in the
staged canonical manifest.
