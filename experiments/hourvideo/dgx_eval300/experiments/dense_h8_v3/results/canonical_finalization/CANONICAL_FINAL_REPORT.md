# New Dense Semantic Beam-B H-8 Eval300 Canonical Final Report

## Final status

**`COMPLETE / VALID`**

This report is a read-only archival verification of the formal H-8 v3 run. No question was rerun, no prediction or trajectory was changed, and the stale parent `state.txt` was not overwritten.

## 1. Authoritative inputs

- Parent formal run: `data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h8_eval300_formal_v3_20260831T201510Z`
- Frozen UID list: `data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt`
- UID-list SHA-256: `6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1`
- Final merged manifest SHA-256: `c99be195585900531024659fef09b58841205fc05123a104a0c6081463337e65`
- Formal backend: `dense_semantic_beam_b`
- Beam budget: `B=8`
- Formal concurrency: `1`

The invalid H-8 v2 infrastructure run is not an input to this report.

## 2. UID and merge acceptance

| Check | Result | Status |
|---|---:|---|
| Frozen UID count | 300 | PASS |
| Frozen unique UID count | 300 | PASS |
| First-pass unique UID count | 300 | PASS |
| Merged record count | 300 | PASS |
| Merged unique UID count | 300 | PASS |
| Merged UID set equals frozen set | yes | PASS |
| Retry-list count | 46 | PASS |
| Retry-list unique count | 46 | PASS |
| Retry artifacts | 46/46 | PASS |
| Final records selected from retry | 46 | PASS |
| Final records selected from first pass | 254 | PASS |
| Retry-selected UID set equals frozen retry list | yes | PASS |

The merge uses one terminal record per frozen UID. For every UID in the 46-entry retry list, the retry attempt replaces the corresponding first-pass attempt. The old and new attempt are not combined in final method metrics.

The launch-gate UID `6fd90f8d-7a4d-425d-a812-3268db0b0342_14_5` occurs once in first pass, zero times in retry, and once in the merged view. It is not double-counted.

## 3. Final fixed-denominator results

| Metric | Count | Rate |
|---|---:|---:|
| Strict completed / 300 | 270 | 90.00% |
| Correct / 300 | 78 | 26.00% |
| Correct / strict completed | 78/270 | 28.89% |
| Final timeout / 300 | 29 | 9.67% |
| Success but strict invalid / 300 | 1 | 0.33% |
| Final not strict completed / 300 | 30 | 10.00% |

`Correct / 300` is the primary fixed-denominator accuracy: unfinished and invalid cases are treated as not correct. `Correct / strict completed` is conditional on completion and must not replace the fixed-300 result.

## 4. Dense H-8 retrieval integrity

All 346 stored attempt trajectories—300 first pass plus 46 retry—were inspected. They contain 841 `visual_retrieve` events.

| Requirement | Passing events | Status |
|---|---:|---|
| backend=`dense_semantic_beam_b` | 841/841 | PASS |
| `B=8` | 841/841 | PASS |
| `hierarchy_complete=true` | 841/841 | PASS |
| `hierarchy_used=true` | 841/841 | PASS |
| Dense Medium gate applied | 841/841 | PASS |
| One query embedding per retrieval | 841/841 | PASS |
| One SigLIP query encoding per retrieval | 841/841 | PASS |
| No global Fine fallback | 841/841 | PASS |
| No caption-score fallback | 841/841 | PASS |
| No representative-frame fallback | 841/841 | PASS |

No lexical/BM25 event or old global-Fine H path was found. The one recorded full-video forced fallback is a frozen downstream VideoSEAL control-flow branch, not a Retriever fallback and not an old-H path.

## 5. Old H-8 v2 exclusion

- Merged records referring to old H-8 v2: 0.
- Attempt trajectories referring to old H-8 v2: 0.
- Every selected result belongs to the formal H-8 v3 root.
- H-8 v2's 600 infrastructure-failure metrics were not used for resume, retry selection, merging, scoring, latency, or this report.

The 29 final timeout rows correctly have no completed prediction/trajectory artifact. Their terminal metric and elapsed time remain represented in the merged record; absence of a prediction is not treated as missing UID coverage.

## 6. E2E definitions and timeout treatment

Three cumulative quantities must be kept separate.

### 6.1 Reported E2E distribution

The canonical E2E mean, median and percentiles are calculated over the **270 final-selected strict-completed attempts only**:

| Statistic | Value |
|---|---:|
| Count | 270 |
| Mean | 247.03 seconds |
| Median | 205.99 seconds |
| P90 | 432.76 seconds |
| P95 | 580.99 seconds |
| Completed-attempt elapsed sum | 66,699.29 seconds / 18.53 hours |

The 29 final timeouts and one final strict-invalid attempt are excluded from this distribution. Consequently, `mean × 270` equals the strict-completed elapsed sum; it is not the cost of all 300 UIDs.

### 6.2 Final-selected cumulative E2E

Summing exactly one terminal selected attempt for every UID gives:

- 300 selected attempts;
- 94,278.59 seconds = 26.19 hours;
- includes 29 selected timeout attempts totaling 27,428.17 seconds;
- includes the one selected strict-invalid attempt totaling 151.13 seconds;
- excludes the 46 superseded first-pass attempts whose retry became terminal.

This is the appropriate cumulative cost for the final merged method view, but it is not the actual total compute consumed by the experimental run.

### 6.3 Actual first-pass plus retry cost

The actual experimental attempt cost includes every executed attempt:

| Phase | Attempts | Elapsed sum |
|---|---:|---:|
| First pass | 300 | 100,787.25 seconds / 28.00 hours |
| Formal retry | 46 | 34,649.23 seconds / 9.62 hours |
| Total | 346 | 135,436.48 seconds / 37.62 hours |

This total includes first-pass attempts later replaced by retry, all timeout elapsed values, successful recoveries, and unrecovered retry attempts. Timeout cost uses the recorded `metric.elapsed_sec`; it is not replaced mechanically with the nominal 1,000-second timeout ceiling.

Therefore:

- use the 270-completed distribution to describe latency among valid completed outputs;
- use 26.19 hours to describe one-final-attempt-per-UID cumulative E2E;
- use 37.62 hours to describe actual compute consumed by first pass plus formal retry.

These quantities answer different questions and must not be interchanged.

## 7. State-file amendment

The parent contains both:

- `finished_utc.txt=2026-09-02T11:18:44Z`, complete final/merged artifacts and full retry coverage;
- `state.txt=FORMAL_RUNNING`.

The launch script writes `FORMAL_RUNNING` before first pass but does not write a terminal state after finalization. The stale label is therefore a non-data finalization omission. The original file remains unchanged for provenance. This versioned amendment supplies the authoritative archival state `COMPLETE / VALID` without rewriting history.

## 8. Acceptance conclusion

The New Dense Semantic Beam-B H-8 Eval300 formal v3 result passes final archival acceptance:

- exact frozen 300-UID coverage;
- exact 46-attempt retry replacement;
- fixed-300 scoring denominator;
- complete B=8 Dense Coarse → Dense Medium → local Fine execution evidence;
- no old H-8 v2 contamination;
- explicit, non-overlapping E2E definitions;
- original artifacts preserved unchanged.

Canonical status: **`COMPLETE / VALID`**.

