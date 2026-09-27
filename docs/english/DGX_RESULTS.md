# DGX results and historical comparisons

English presentation summary, not a replacement experimental record. The linked frozen originals remain authoritative for byte-level provenance. Numerical values and artifact identities reproduced here are copied without alteration; this summary does not introduce new results or revise frozen labels. See the [source register](../ENGLISH_READING_GUIDE.md) for original-file hashes and checksum membership.

## New Dense H main comparison

[Frozen comparison report](../../experiments/hourvideo/dgx_eval300/reports/main_comparison/FINAL_H15_VS_FLAT_DATA_ARCHIVE.md). The population is 300 unique HourVideo questions from 12 videos. The native Planner is Qwen3-8B and the Inspector is Qwen2.5-VL-7B-Instruct; formal concurrency is 1.

| Condition | Correct / 300 | Strict completed / 300 |
|---|---:|---:|
| Flat-30 | 82 | 254 |
| Dense H-8 | 78 | 270 |
| Dense H-15 | 81 | 270 |
| Dense H-30 | 78 | 249 |

Strict completion requires a successful metric, one valid A–E prediction, and a completed trajectory with a nonempty answer. Missing/invalid outcomes remain incorrect in the fixed denominator. Completed-only accuracy is a separate conditional metric.

Dense H uses global Top-B Coarse routing, global Top-B Medium routing under selected Coarse nodes, local Fine SigLIP ranking and at most B Flat-compatible caption segments. Inspector image counts are independent of B. Old lexical-gated H and global-Fine variants are excluded.

H-15 and Flat-30 differ by one correct answer, with 38 Flat-only and 37 H-15-only correct answers; exact McNemar p=1.0. On 232 mutually completed questions, mean E2E latency is 317.35 seconds for Flat and 247.71 seconds for H-15, a mean paired difference of −69.63 seconds. This supports an efficiency observation, not superior accuracy.

H-8 has 233 mutually completed questions with Flat and a mean paired difference of −76.36 seconds; the reported bootstrap interval is [−108.61, −44.52] seconds. H-8 versus H-15 has no significant paired latency difference (p=0.2342). H-30 versus Flat has no reliable paired E2E improvement in the reported analysis.

Method-efficiency totals use one final-selected attempt per UID; actual experiment resource consumption includes first passes and retries. Do not combine replaced attempts into final-method latency. Inspector image transmissions exclude locally scored frames and textual Summarizer evidence. Stale `paired_with_flat` fields are not the authoritative final comparisons. Use the report's recomputed UID pairings.

Flat-15 is documented separately in its [English final report](../../experiments/hourvideo/dgx_eval300/experiments/flat15_retryv2/results/finalization_v2/FINAL_REPORT.md): 76/300 correct and 260/300 strict completed. See [condition-specific records](../../THESIS_ARTIFACT_INDEX.md#direct-condition-and-implementation-links).

## Old-H: appendix and historical only

[Frozen Old-H report](../../experiments/hourvideo/dgx_eval300/appendix_candidates/old_h/old_h_series_archive/OLD_H_SERIES_EVAL300_DATA_ARCHIVE.md). Old H-6/H-15 used lexical-gated routing; Old H-30 used global-Fine SigLIP. They are distinct from New Dense H.

| Condition | Original correct / 300 | Corrected correct / 300 | Corrected strict completed |
|---|---:|---:|---:|
| Old H-6 | 64 | 61 | 279 |
| Old H-15 | 64 | 63 | 286 |
| Old H-30 | 66 | 66 | 257 |

Thirteen selected fallback attempts were rerun after an endpoint fix and replaced one-for-one. The corrected view is the historical series' citable version. Do not mix original accuracy with corrected efficiency. Corrected mean E2E latency among completed questions is 273.57 / 268.43 / 288.41 seconds for H-6/H-15/H-30. Correction compute is recorded separately. Pairwise p-values are unadjusted; the report does not establish a family-wise significant ranking.

## Flat reconciliation snapshot

[Frozen reconciliation report](../../experiments/hourvideo/dgx_eval300/experiments/flat30/results/reconciliation/reconciliation_report.md), snapshot 2026-08-22T14:42:51Z. Its Chinese group labels mean “pre-training Planner + VideoSEAL Flat,” “post-training Planner + VideoSEAL Flat,” and “pre-training Planner + H-6 hierarchical indexer.”

The snapshot reports Flat 82/300 correct with 254 valid outcomes, trained-Planner Flat 75/300 with 251 valid outcomes, and H-6 63/300 with 268 valid outcomes while H-6 retry was still in progress. That H-6 snapshot is not the later corrected final Old-H result. No final paired comparison was frozen at this snapshot. Scoring accepts only the explicit prediction field matching A–E; trajectory text cannot substitute for it.
