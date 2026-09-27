# R1 Direct Visual-Only Correction: Recovered Eval300 Final Report

## 1. Final status

**PASS with disclosed unresolved provider-cost uncertainty.** The corrected R1 population is complete and structurally paired with the frozen R3 Eval300 population. Gold was loaded only after the no-gold structural merge passed.

- Corrected R1: 175 unique routes, comprising 174 final answers and one retained runtime failure.
- Reused R1: 125 frozen routes, all with final answers.
- Combined R1 visual-only Eval300: 300 unique routes, 299 final answers and one retained failure.
- Frozen R3 comparison: 300 unique routes, 298 final answers and two retained failures.
- R1/R3 question pairing: exact.
- Old replaced R1 routes and all smoke routes: excluded.
- Main accuracy denominator: 300 for each method; failures were not silently removed.

## 2. Segmented formal execution and recovery lineage

| Segment | Frozen manifest SHA-256 | Newly attempted scope | Outcome used in corrected 175 |
|---|---|---:|---|
| v8 | `d556d817cc0ba6b49f0dfa7b6e4cf291387f9a5f55aa86ec02d7cb0b80517c61` | 27 reached | 26 direct successes + one offline-recovered QA success with unknown provider cost |
| v9 | `ed182c03541257934ef0db5fdf309fb961f5e77574147f4261bc062763207073` | 52 reached | 51 direct successes + one offline-recovered QA success with unknown provider cost |
| v10 | `082aaa0f2a78e6c3071a82255c230ad570b8e8ad03a5683068b1d64e058ab03a` | 44 reached | 43 direct successes + one offline-recovered QA success with unknown provider cost |
| v11 | `3b1d6118e21060ef4fb6d74c99325668f5eadc203fd4abd865e2d892ef6b4dae` | 22 reached | 21 direct successes + one offline-recovered QA success with unknown provider cost |
| v12 | `d89d2a8dce094a0207e597ce778ce14105df413525b96607e4d4a0ff0ab2f50c` | 18 reached | 17 direct successes + one offline-recovered QA success with unknown provider cost |
| v13 | `629870ab6c17d7648becd868c8468d40f712466f05101d5833f2d1c975375d97` | 12 | 11 final answers + one retained frame-resolution failure; run completed |

Each of the five formal network incidents was recovered under the same frozen rule: the original request outcome remained unknown, the in-process retry produced a complete QA result, offline checks confirmed the full result chain, recovery sent zero requests, the unknown request was not resent, and its `$0.252560` reservation remained charged against the cap. A route without a complete valid final result was not eligible for this recovery rule.

The v11 inherited ledger had a provenance-label defect: some unknown attempts were labelled as settled even though their reservation total remained retained. v12 corrected classification based on the absence of `actual_cost_usd`. The final v13 ledger identifies ten unresolved attempts and preserves exactly `$2.525600` in total reservations.

## 3. Structural merge

The merge was performed without gold by `scripts/merge_v13_recovered_eval300.py`.

| Check | Result |
|---|---|
| Corrected R1 unique routes | 175 |
| Reused R1 unique routes | 125 |
| Combined R1 unique routes | 300 |
| Frozen R3 routes | 300 |
| R1/R3 question sets equal | PASS |
| Corrected/reused R1 overlap | none |
| Old replaced 175 excluded | PASS |
| Smoke excluded | PASS |
| Failures retained | R1: 1; R3: 2 |
| Fixed main denominator | 300 per method |

Primary no-gold evidence:

- `outputs/r1_visual_only_recovered_eval300_v1/merged_results_no_gold.json`
- `outputs/r1_visual_only_recovered_eval300_v1/structural_summary.json`

## 4. Post-hoc gold results

Gold source: `${PROJECT_MSC_ROOT}/HourVideo/benchmark/v1.0_release/json/dev_v1.0_annotations.json`, SHA-256 `e1af087df035d524ee64d92d34d2e81f29461fdeb5fa72cfda679d7c9909daf3`.

| Metric | R1 visual-only | R3 frozen |
|---|---:|---:|
| Population | 300 | 300 |
| Correct | 85 | 103 |
| Main accuracy | **28.33%** | **34.33%** |
| Final/valid answers | 299 | 298 |
| Completion rate | 99.67% | 99.33% |
| Retained failures | 1 | 2 |
| Accuracy among valid answers | 28.43% (85/299) | 34.56% (103/298) |

Paired outcomes:

- Both correct: 51
- R1 only correct: 34
- R3 only correct: 52
- Neither correct: 163
- R1 minus R3: -18 correct answers, or **-6.00 percentage points** on the fixed denominator.

The R1 retained failure was an out-of-duration frame request at 1680 seconds. The R3 retained failures were one turn-limit exhaustion and one exhausted structural-action correction after an out-of-duration request at 1900 seconds.

## 5. Resources and cost

| Quantity | R1 visual-only Eval300 | R3 frozen Eval300 |
|---|---:|---:|
| Known-response provider attempts | 1,956 | 1,444 |
| Unique images, summed by route | 4,217 | 2,838 |
| Input tokens | 10,790,322 | 6,188,889 |
| Cache-creation input tokens | 718,375 | 1,415,926 |
| Cache-read input tokens | 24,229,721 | 43,169,630 |
| Output tokens | 183,730 | 144,978 |
| Known settled route cost | `$15.02991285` | `$13.00064950` |

Correction-execution accounting:

- Corrected 175 known settled route cost: `$8.53059750`.
- Aggregate correction ledger settled amount: `$8.61269850`; this includes `$0.08210100` of settled historical smoke cost.
- Formal-run unknown reservations: `$1.26280000` (five network-result-unknown attempts).
- All unknown reservations carried by the correction ledger: `$2.52560000` (five historical smoke unknowns plus five formal unknowns).
- Remaining cap availability at completion: `$3.86170150` under the frozen reservation accounting.

Unknown reservations are **not claimed as actual spend**. No token, latency, response, or usage values were fabricated for those attempts. Method route cost and correction-project cash accounting are different scopes and must not be substituted for one another.

## 6. Evidence identity

| Artifact | SHA-256 |
|---|---|
| Structural merged results | `57425a522b4bd0bde9de077d96c19149fa1965c1d04913bbb97fc96b82dd1da0` |
| Structural summary | `321cdb731a9516f8c5cc65c221abf71ead00ad68e60bbe0a9204400ccb1c8f34` |
| Gold-scored per-question results | `17550fb4f0bd244fd49c220728a6bf668402b31ead56979eaf23bd5d6fca2cd1` |
| Accuracy summary | `af7304b6d12ec0db0c53c48df7d584a05834c7ccefdf616f253c457c5039082c` |
| Accuracy CSV | `b860e5d90a87ff49ed162b98adc108c4b6ff1c9c68f294cff0b06a3349e1c3cd` |
| Merge script | `f1cff68d1292a062a4cf05bb3337e9ef366464f6f4381aaa5b975d15216ee17e` |
| Scoring script | `497824ab79ae827217b2edd89e1771cb7e7cf63bb411536273e855c00d71ea4f` |

## 7. Interpretation constraints

- This result replaces the prior ASR-exposed R1 routes only for the specified 175-question correction scope; it does not rewrite historical artifacts.
- The 125 reused R1 routes and 300 R3 routes retain their original source identities and costs.
- Provider outcomes and billing for the ten unknown requests cannot be established from local artifacts; their reservations remain disclosed.
- Historical frame hashes prove identity against the frozen local frame inventory, not cross-time identity of the external model service.
- Accuracy scoring is post-hoc and did not influence route inclusion, failure handling, or recovery decisions.

