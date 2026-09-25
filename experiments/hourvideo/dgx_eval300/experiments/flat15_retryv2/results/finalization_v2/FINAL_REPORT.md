# Flat-15 Eval300 canonical final report

## Final status

**COMPLETE / VALID**

This additive finalization did not rerun a model, retry, backfill, index build, or prediction parser, and it made no writes to the historical formal-run roots. The data-bearing metrics, trajectories, and predictions used by the merge still match their pre-retry hashes. Two shared service logs are an explicit exception to the broader pre-retry snapshot: the approved retry-v2 reused the same Planner and Visual services and had already appended `logs/planner.log` and `logs/visual.log` before finalization. `FIRST_PASS_PRE_RETRY_SNAPSHOT_RECHECK.json` records the 924-pass/2-expected-difference result.

An earlier additive finalization attempt (`videoseal_flat15_eval300_finalization_v1_20260914T002232Z`) stopped before gold access when it incorrectly required provider embedding requests to equal persisted retrieval actions. It is diagnostic only and is excluded from this closure.

## 1. Retry-v2 acceptance and freeze

- Frozen retry manifest: exactly 57 unique UIDs; its set equals the 57 strict first-pass failures and has zero overlap with the retained 243 strict first-pass successes.
- Terminal retry-v2 artifacts: 57 metrics and 57 trajectories, covering the manifest exactly; runner exit code 0; source state `RETRY_RAW_COMPLETE_PENDING_FREEZE`.
- Terminal statuses: 17 success, 33 timeout, 7 incomplete.
- All 17 metric successes satisfy the strict rule: complete metric schema, finished trajectory, non-empty answer, and the frozen parser yields exactly one legal A-E prediction. No timeout or incomplete was promoted.
- There are 24 prediction files because `run_one` writes one whenever the agent returns normally: 17 contain legal A-E predictions and are success; 7 contain an empty prediction and are incomplete. The 33 hard-timeout workers terminate before that write.
- Infrastructure-stop sidecar: absent. Populated retry gold fields: 0. Infrastructure-error matches in failed tool observations: 0.
- Retry-v2 raw/control snapshot entries: 148; snapshot manifest SHA-256: `9c00c3392bc53865e45a0034e963c7b67e797ddb65c22a97a94c641ddaae21ba`. Immediate readback and final readback both pass.
- The older 926-entry first-pass pre-retry snapshot now has 924 matches and two expected shared-log differences (`logs/planner.log`, `logs/visual.log`) caused by the later approved retry service activity. No metric, trajectory, prediction, configuration, or frozen UID entry differs.

## 2. Gold-free merge and prediction freeze

- Frozen population: 300 unique UIDs in frozen order.
- Selected records: 243 strict first-pass successes plus all 57 retry-v2 terminal outcomes.
- Retry-v2 recoveries: 17 strict successes; final strict completion: **260/300 (86.67%)**.
- Preserved failures: 33 timeout and 7 incomplete; retry-v1 selected records: 0; backfill: not run; post-processing/backfilled predictions: not used.
- Frozen gold-free prediction SHA-256: `f1c478a6175f56e7b40ba99dd41d21a0ec9e50df9b93e7fb2893fffcfcfd8481`.
- Gold was loaded only in the separate scoring invocation after this hash was independently read back and verified.

## 3. Final score

| Metric | Result |
|---|---:|
| Correct / 300 | **76/300 (25.33%)** |
| Strict completed / 300 | **260/300 (86.67%)** |
| Correct / strict completed | **76/260 (29.23%)** |
| Final timeout | 33 |
| Final incomplete | 7 |

`Correct / 300` is the primary fixed-denominator result; every retained failure is not correct. Completed-only accuracy is conditional on the 260 completed-item subset and is not interchangeable with fixed-300 accuracy.

## 4. Formal calls, tokens, images, and elapsed time

All table values below are sums of per-attempt metric fields. “Actual formal” includes superseded first-pass failures plus their retry attempts; “final-selected” contains one selected attempt per UID. For killed timeouts, metrics are reconstructed from the last persisted trajectory, so an in-flight call and the remainder to the 1000-second wrapper timeout are absent: all-attempt calls/tokens/images/metric-elapsed values are reproducible **recorded lower bounds**, not exact consumed totals.

| Scope | Attempts | Planner calls | Retrieval calls | Inspector calls | Planner tokens | Visual tokens | Images | Elapsed hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| First pass, actual | 300 | 1303 | 579 | 324 | 3636762 | 10507930 | 20224 | 31.31 |
| Formal retry-v2, actual | 57 | 388 | 178 | 105 | 1329048 | 3433698 | 6720 | 11.02 |
| Actual formal total | 357 | 1691 | 757 | 429 | 4965810 | 13941628 | 26944 | 42.33 |
| Final-selected 300 | 300 | 1245 | 540 | 310 | 3369608 | 10003538 | 19328 | 28.89 |

Final-selected strict-completed latency (n=260): mean 271.47s, median 229.50s, P90 475.57s, P95 588.65s; sum 19.61h. Timeout/incomplete attempts are excluded from this distribution but retained in final-selected and actual-formal totals.

Sequential batch wall time, which includes the hard-timeout remainder omitted by terminal trajectory metrics, was 32.05h for first pass and 11.59h for formal retry-v2 (43.64h combined, excluding the calendar gap, smoke, retry-v1 diagnostics, and idle server lifetime).

Embedding cost:

- First pass: **unknown**, because no historical provider-usage ledger exists; it is not reported as zero.
- Formal retry-v2: 184 successful provider requests, 1729 provider-reported input tokens, **USD 0.00022477**. This is six more than the 178 persisted retrieval actions: six timeout UIDs each completed one embedding request before the worker was killed, but never committed the enclosing tool step. `retry_embedding_reconciliation.json` closes this boundary per UID.
- Actual formal total embedding cost: **unknown** because the first-pass component is unknown; the known retry-v2 component is USD 0.00022477.

Separate non-formal-method overhead:

- Recovery smoke: 1 embedding request / 6 input tokens / USD 0.00000078; 1 local Summarizer call / 3440 tokens; 0 Planner and 0 Inspector calls. Excluded from formal totals.
- Invalid retry-v1 diagnostic: 33 Planner calls / 59936 tokens; 2 Inspector calls / 52133 tokens / 128 images; 33 failed retrieval attempts, 0 provider requests, recorded checkpoint elapsed 1574.28s. Preserved as anomalous investment and excluded from merge/formal-method efficiency.

## 5. Same-definition comparison

The result rows use each method's latest archived final-selected 300-UID view: strict completion, fixed-denominator Correct/300, and completed-only accuracy. The efficiency columns use one selected attempt per UID, while completed E2E mean uses only strict-completed items. Historical Flat-30/H values come from `/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z/analysis/h15_vs_flat_data_archive_20260829T152745Z/FINAL_H15_VS_FLAT_DATA_ARCHIVE.md` (result table lines 311-315 and efficiency tables beginning at line 328).

| Method | Strict completed | Correct / 300 | Correct / completed | Timeout | Other strict failure | Completed E2E mean | Planner | Retrieval | Inspector | Images | Selected elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat-15 (this run) | 260 (86.67%) | 76 (25.33%) | 29.23% | 33 | 7 incomplete | 271.47s | 1245 | 540 | 310 | 19328 | 28.89h |
| Flat-30 | 254 (84.67%) | 82 (27.33%) | 32.28% | 36 | 10 strict-invalid | 318.99s | 1300 | 566 | 354 | 21492 | 32.62h |
| H-8 | 270 (90.00%) | 78 (26.00%) | 28.89% | 29 | 1 strict-invalid | 247.03s | 1251 | 442 | 220 | 13348 | 26.19h |
| H-15 | 270 (90.00%) | 81 (27.00%) | 30.00% | 26 | 4 strict-invalid | 250.27s | 1154 | 486 | 302 | 18336 | 26.10h |
| H-30 | 249 (83.00%) | 78 (26.00%) | 31.33% | 45 | 6 strict-invalid | 305.70s | 1307 | 413 | 245 | 14892 | 33.45h |

The “other strict failure” labels differ by runner outcome: this Flat-15 run records 7 explicit `incomplete` outcomes, while the archived comparison reports `status=success` but strict-invalid counts for the other methods. They are all excluded from strict completion and Correct/300, but are not claimed to be the same failure mechanism. Historical token totals and embedding fees are not present in this comparison table and are therefore not imputed.

## 6. Artifacts and integrity

- `retry_validation_per_uid.jsonl`: raw-response hashes, trajectory answers, frozen parser results, statuses, and infrastructure checks for all 57 retry UIDs.
- `final_predictions_gold_free.jsonl`: frozen 300-UID prediction/status closure with failures retained and no gold fields.
- `scored_results.jsonl`: separate post-freeze scored view containing gold.
- `cost_summary.json`: actual-formal, final-selected, smoke, anomalous, and embedding-cost scopes.
- `FIRST_PASS_PRE_RETRY_SNAPSHOT_RECHECK.json`: exact expected/current hashes for the two shared service logs and confirmation that the other 924 pre-retry snapshot entries still match.
- `retry_v2_source_snapshot.sha256`, `final_predictions_gold_free.sha256`, `SOURCE_MANIFEST.sha256`, and `MANIFEST.sha256`: source and finalization integrity manifests. Their corresponding verification files record successful readback.
