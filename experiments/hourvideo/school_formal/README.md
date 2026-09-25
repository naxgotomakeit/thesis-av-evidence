# School-server thesis import staging v1

This is a first-import staging tree for the school-server portion of the
thesis repository. It was assembled by copying and, where necessary, redacting
copies of existing artifacts. No original experiment file was modified,
moved, or deleted. No model or external API was called, and nothing was pushed
to GitHub.

## Result classification

| Classification | Experiment/material | Thesis correspondence |
|---|---|---|
| Formal main result | Direct-v1.2 R1/R3 Eval300 | R1 88/300 (29.33%); R3 103/300 (34.33%) |
| Formal main API baseline | GenS Haiku structured v3 | 93/300 (31.0%); uses the Myriad GenS V2 selector whose original SHA-256 is `016e078a147215864fd978e30be7820bc741be694948d68e458383473af96587` |
| Formal main result | R1/R3 capacity-aware Eval300 | R1 81/300 (27.0%); R3 52/300 (17.33%), with 150 R3 context-overflow routes retained in the denominator |
| Formal paired analysis | Capacity-aware paired-150 | Frozen pre-execution UID set; R1 51/150 and R3 52/150. Post-Planner totals: 1,831 vs 1,449 calls and 20,362,068 vs 12,425,668 input tokens |
| Formal preprocessing/index infrastructure | R1 object/tracking and latest R3 semantic-caption indexes | Supports the main R1/R3 experiments; recorded components are 58.093 min for R1 and 53.437 min for R3 on the exact 12-video subset |
| Formal supplementary result | Full Staged R3 vs Direct R3 paired-100 | 24/100 vs 32/100; exact McNemar `p=0.1686375439` |

The paired-150 figures above are not from a separately rerun paired experiment.
The UID intersection was frozen before execution, then the paired figures were
extracted and recomputed from the capacity-aware Eval300 per-question records.
The 1,831/1,449 calls and 20.36M/12.43M input-token figures are explicitly
Post-Planner; Planner-inclusive totals are reported separately in the retained
thesis report.

## Provenance and upload boundary

- `provenance/SOURCE_MANIFEST.jsonl` records every copied source by a portable
  root alias, original SHA-256, staging path, staged SHA-256, machine,
  experiment, classification, result correspondence, and redaction status.
- `DO_NOT_COMMIT/SOURCE_MANIFEST_PRIVATE.jsonl` records the requested original
  absolute path for every source. It is local-only and ignored by Git.
- `provenance/REDACTIONS.json` enumerates every changed copy. Redaction is
  limited to absolute personal/school/scratch path prefixes; experiment
  prompts, predictions, scores, and numeric telemetry were not semantically
  changed.
- `provenance/CREDENTIAL_SCAN.json` is the uploadable-tree secret/path scan.
- `provenance/CHECKSUMS.sha256` is the final checksum list for uploadable files.

See `INCLUDED_EXCLUDED.md`, `CANDIDATES.md`, `MISSING_FILES.md`, and
`VERIFICATION.md` before importing this tree into the GitHub repository.

