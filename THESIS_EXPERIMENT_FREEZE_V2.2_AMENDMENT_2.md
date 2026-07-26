# Protocol Amendment #2 — Oracle≤8 zero-decodable-frame clarification

**Date:** 2026-07-24  
**Base protocol:** `THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md`  
**Prior amendment:** `THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_1.md`

## Problem

V2.2 distinguishes:

- non-zero GT interval: select frames inside `[start, end)`;
- point GT `start == end`: corresponding/nearest decodable-frame fallback.

One Formal Open E1 sample has a non-zero GT interval shorter than one source-frame interval and contains no decodable source frame. Therefore the existing protocol does not define a valid Oracle≤8 selection.

## Clarification

For an annotated non-zero GT interval `[start, end)` for which no decodable source-video frame timestamp `t` satisfies `start <= t < end`, select exactly one decodable frame within the canonical clip minimizing temporal distance to the GT interval.

Define distance:

- `d(t, [start,end]) = 0` if `t` lies inside the interval;
- otherwise `d = min(|t-start|, |t-end|)`.

Break exact ties by choosing the earlier timestamp.

Record the distinct frame-selection rule:

`interval_zero_decodable_nearest_frame`

This fallback applies only to physically undecodable non-zero GT intervals containing zero decodable source frames.

It does **not**:

- expand the GT interval by ±T;
- change normal interval sampling;
- change point-GT handling;
- change the ≤8 frame budget;
- change any prompt/model/backend/dtype setting;
- change B0/B1/B2; or
- invalidate prior formal results.

## Audit evidence

- Closed affected: 0/500.
- Open affected: 1/1850.
- Silent prior violations: 0.

## Active protocol

The active protocol is now:

`THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md`

+

`THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_1.md`

+

`THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_2.md`
