# Direct R1 visual-only thesis correction — GitHub import staging

This package archives the thesis-authoritative Direct result used for Table
5.5. It was assembled from existing artifacts without running a model, API,
scoring job, or experiment. No source artifact was modified, moved, or
deleted.

## Thesis-authoritative result

| Method | Correct / 300 | Completed |
|---|---:|---:|
| R1 visual-only correction | 85 / 300 | 299 / 300 |
| Frozen Direct R3 | 103 / 300 | 298 / 300 |

## Required version relationship

```text
old formal R1: 88/300, 300 predictions
  -> 175 routes contained model-visible non-empty audio_channel ASR
  -> visual-only controlled correction/recovery
  -> 175 routes rerun + 125 R1 routes reused after SHA verification
  -> corrected R1: 85/300, 299 completed
  -> direct_visual_only_eval300_thesis_data_v1
  -> Table 5.5 thesis-authoritative result
```

The old 88/300 result is retained under
`artifacts/SUPERSEDED_FOR_THESIS/old_formal_88/`. It is historical provenance
and is **SUPERSEDED-FOR-THESIS**; it must not be presented as the thesis R1
main result. The 85/300 result and `thesis_data_package/` are
**THESIS-AUTHORITATIVE**.

The observed 88 to 85 change cannot be attributed solely to ASR removal.
The correction was run later and also differs in recovery/transport handling;
cross-time provider-side nondeterminism remains a documented limitation.

## Archive layout

- `artifacts/thesis_data_package/`: Table 5.5 per-question data, statistics,
  report, build logic, and package checksums.
- `artifacts/final_outputs/`, `artifacts/final/`: recovered Eval300 outputs,
  final report, and final audit checksum chain.
- `artifacts/manifests/`: route-reuse, source identity, input/map/frame
  inventories, and code/protocol lineage.
- `artifacts/formal_segments/`: v8-v13 manifests, locks, ledgers, execution
  records, and exactly 175 rerun route artifacts/statuses.
- `artifacts/SUPERSEDED_FOR_THESIS/old_formal_88/`: old 88 canonical summary
  and the 600 old/frozen route artifacts required to audit 125 reused R1
  routes, frozen R3, and old-to-new lineage.
- `artifacts/source/`, `artifacts/config/`, `artifacts/scripts/`,
  `artifacts/validation/`: final prompt/runtime/config, v8-v13 code,
  validation, merge, and scoring logic.

Absolute school-user path prefixes in uploadable text files were replaced by
portable tokens only. Every such transformation is explicit in
`SOURCE_MANIFEST.csv`; exact local paths are isolated in `DO_NOT_COMMIT/`.
No experimental answer or score was fabricated or inferred from the 88 result.

See `VERIFICATION.md`, `SENSITIVE_SCAN.md`, and `INCLUDED_EXCLUDED.md` before
importing this package.
