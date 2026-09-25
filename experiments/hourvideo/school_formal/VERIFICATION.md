# Verification summary

## Integrity

- Source records: 2,524.
- Missing source or staged files during closure verification: 0.
- Original-file SHA-256 mismatches: 0.
- Staged-file SHA-256 mismatches against the manifest: 0.
- Redacted copies are enumerated in `provenance/REDACTIONS.json`; only path
  prefixes changed.

## Critical identities and counts

| Check | Verified value |
|---|---|
| Myriad GenS V2 selector original SHA-256 | `016e078a147215864fd978e30be7820bc741be694948d68e458383473af96587` |
| GenS selector rows / formal route statuses / scored correct | 300 / 300 / 93 |
| Direct route statuses / canonical rows | 600 / 600 |
| Capacity-aware canonical rows | 600 |
| Frozen paired-150 UID count and SHA-256 | 150; `0219ac63d3e577ae2b1aebd36e9ec8a5793d4597c3013d73ff12fd18b8c20057` |
| Full Staged paired-100 route statuses | 100 |

## Independent paired-150 telemetry recomputation

The 150 IDs in `eligibility/common_eligible_question_ids.txt` were applied to
the staged per-question `model_attempts.jsonl` records. Every attempt was
counted, including retries and failed routes.

| Route | Questions | Calls | Input tokens | Output tokens | Image transmissions |
|---|---:|---:|---:|---:|---:|
| R1 (`r1_av`) | 150 | 1,831 | 20,362,068 | 1,493,281 | 7,222 |
| R3 (`r3_2`) | 150 | 1,449 | 12,425,668 | 1,280,158 | 5,172 |

This confirms that paired-150 is a read-only extraction/recomputation from the
capacity-aware Eval300 records, not an independent paired-150 model run. These
are Post-Planner totals.

