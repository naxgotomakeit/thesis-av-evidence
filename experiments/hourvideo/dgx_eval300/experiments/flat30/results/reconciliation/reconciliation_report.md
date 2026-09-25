# HourVideo Eval300 reconciliation audit

Snapshot UTC: `2026-08-22T14:42:51Z`

## Inputs

Canonical manifest: `/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt`
SHA-256: `6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1`

### 训练前 Planner + VideoSEAL Flat

- First pass: `/home/naxucl/data/HourVideo/videoseal_original/runs_dgx_eval300_v1`
- Retry: `/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/original_retry`
- Retry plan: `/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/manifests/original_timeout_uids.txt`

### 训练后 Planner + VideoSEAL Flat

- First pass: `/home/naxucl/data/HourVideo/videoseal_original/runs_dgx_eval300_videoseal8b_v1`
- Retry: `/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/trained_retry`
- Retry plan: `/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/manifests/trained_timeout_uids.txt`

### 训练前 Planner + H-6 hierarchical indexer

- First pass: `/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/formal_eval300_20260821T095418Z/h6/first_pass`
- Retry: `/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/formal_eval300_20260821T095418Z/h6/retry_1`
- Retry plan: `/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/formal_eval300_20260821T095418Z/h6/status/retry-pending-20260821T095441Z.txt`

## Unified results

| Group | First valid | First correct | First acc. | First timeout | Retry attempted | Retry valid recovered | Retry correct recovered | Retry timeout | Final valid | Final correct | Completed-only acc. | Full-300 acc. | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 训练前 Planner + VideoSEAL Flat | 219 | 73 | 73/219 = 33.3% | 72 | 72 | 35 | 9 | 36 | 254 | 82 | 82/254 = 32.3% | 82/300 = 27.3% | FINAL |
| 训练后 Planner + VideoSEAL Flat | 227 | 70 | 70/227 = 30.8% | 65 | 65 | 24 | 5 | 41 | 251 | 75 | 75/251 = 29.9% | 75/300 = 25.0% | FINAL |
| 训练前 Planner + H-6 hierarchical indexer | 268 | 63 | 63/268 = 23.5% | 28 | 2 | 0 | 0 | 1 | 268 | 63 | 63/268 = 23.5% | 63/300 = 21.0% | RETRY IN PROGRESS |

## Detailed counts

### 训练前 Planner + VideoSEAL Flat

First pass: attempted=300, status-success=228, strict-valid=219, invalid=9, timeout=72, error=0, missing=0, correct=73, incorrect=146; completed-only 73/219 = 33.3%; full-set 73/300 = 24.3%.

Retry: planned=72, attempted=72, raw-success=36, strict-valid=35, invalid=1, timeout=36, error=0, pending=0, correct=9, incorrect=26.

Merged: valid=254, invalid=10, timeout=36, error=0, missing/pending=0, correct=82, incorrect=172; recovered-valid=35, recovered-correct=9; completed-only 82/254 = 32.3%; full-set 82/300 = 27.3%. Status: **FINAL**.

### 训练后 Planner + VideoSEAL Flat

First pass: attempted=300, status-success=235, strict-valid=227, invalid=8, timeout=65, error=0, missing=0, correct=70, incorrect=157; completed-only 70/227 = 30.8%; full-set 70/300 = 23.3%.

Retry: planned=65, attempted=65, raw-success=24, strict-valid=24, invalid=0, timeout=41, error=0, pending=0, correct=5, incorrect=19.

Merged: valid=251, invalid=8, timeout=41, error=0, missing/pending=0, correct=75, incorrect=176; recovered-valid=24, recovered-correct=5; completed-only 75/251 = 29.9%; full-set 75/300 = 25.0%. Status: **FINAL**.

### 训练前 Planner + H-6 hierarchical indexer

First pass: attempted=300, status-success=272, strict-valid=268, invalid=4, timeout=28, error=0, missing=0, correct=63, incorrect=205; completed-only 63/268 = 23.5%; full-set 63/300 = 21.0%.

Retry: planned=32, attempted=2, raw-success=1, strict-valid=0, invalid=1, timeout=1, error=0, pending=30, correct=0, incorrect=0.

Merged: valid=268, invalid=1, timeout=1, error=0, missing/pending=30, correct=63, incorrect=205; recovered-valid=0, recovered-correct=0; completed-only 63/268 = 23.5%; full-set 63/300 = 21.0%. Status: **RETRY IN PROGRESS**.

## Paired comparison

Not frozen: H-6 retry was still running at the snapshot time. No paired UID list or McNemar result was generated. The full-300 snapshot values remain visible above.

## Scoring rule

Only the explicit prediction field, normalized with `str(value).strip().upper()`, is accepted when it fully matches `^[A-E]$`. Trajectory answer text is never used to replace it.
