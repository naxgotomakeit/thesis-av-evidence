# Complete thesis-ready statistics — Planner-only API ablation

Source: final `canonical_summary_v2`; raw formal closure SHA-256 `020ad1a5d12805a3bf03c5d719911aa9fc98deed7c3be1d89e5bf0218c37bcd8`.

## 1. Main outcome table

| Route | Correct / 300 | Accuracy | Wilson 95% CI | Predictions | Prediction completion | normal_success | downgraded_recovery | failed | Missing prediction |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| R1 | 78/300 | 26.00% | [21.36%, 31.24%] | 293/300 | 97.67% | 48 | 245 | 7 | 7 |
| R3 | 77/300 | 25.67% | [21.05%, 30.90%] | 292/300 | 97.33% | 119 | 173 | 8 | 8 |

R3 − R1 accuracy: **-0.33 pp**; prediction completion: **-0.33 pp**.

## 2. Paired R1 vs R3

Correctness: both correct 50; R1-only 28; R3-only 27; both wrong 195. Exact two-sided McNemar p=1; fixed-seed bootstrap 95% CI [-5.00, 4.67] pp.
Completion (prediction present): both 285; R1-only 8; R3-only 7; neither 0. McNemar p=1; bootstrap 95% CI [-3.00, 2.33] pp.

## 3. R3 local-capacity attribution subsets

### context_overflow_pre_model (n=150)

Planner attempted/accepted 150/150; terminal 150; predictions 145; correct 29 (19.33%); completion 145/150 (96.67%); normal/downgraded/failed 48/97/5; failures {'final_contract_exhaustion': 1, 'fine_visual_max_tokens_or_validation_exhaustion': 4}.
### eligible_pre_model (n=150)

Planner attempted/accepted 150/150; terminal 150; predictions 147; correct 48 (32.00%); completion 147/150 (98.00%); normal/downgraded/failed 71/76/3; failures {'final_contract_exhaustion': 1, 'fine_visual_max_tokens_or_validation_exhaustion': 2}.

## 4. Planner API cost

| Route | Logical calls | Attempts | Retries | Failed attempts | Input tokens | Output tokens | Total tokens | USD | USD/question | Latency mean / median / P90 / P95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R1 | 300 | 302 | 2 | 3 | 2,870,342 | 274,447 | 3,144,789 | $4.965457 | $0.016552 | 17.73 / 17.79 / 20.46 / 21.16 |
| R3 | 300 | 301 | 1 | 1 | 8,510,653 | 268,932 | 8,779,585 | $10.096273 | $0.033654 | 17.18 / 17.74 / 20.33 / 21.00 |

R3 − R1: 5,634,796 Planner tokens (179.18%); $5.130816 (103.33%).

## 5. Canonical post-Planner downstream

| Route | Shared | Fine | Final | All local calls | Input tokens | Output tokens | Total tokens | Fine images | Images/question | Post-Planner E2E mean / median / P90 / P95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R1 | 2062 | 687 | 309 | 3058 | 36,990,771 | 2,573,649 | 39,564,420 | 10,676 | 35.59 | 133.40 / 128.44 / 206.03 / 235.31 |
| R3 | 1840 | 628 | 321 | 2789 | 25,426,437 | 2,474,358 | 27,900,795 | 9,855 | 32.85 | 119.13 / 110.61 / 227.76 / 256.19 |

R3 relative to R1: local calls -269 (-8.80%); tokens -11,663,625 (-29.48%); Fine images -821 (-7.69%).

## 6. Full Planner + downstream accounting

| Route | Total requests | Total tokens | Fine images | Planner API USD | Mean total modeled latency |
|---|---:|---:|---:|---:|---:|
| R1 | 3,360 | 42,709,209 | 10,676 | $4.965457 | 151.27s |
| R3 | 3,090 | 36,680,380 | 9,855 | $10.096273 | 136.30s |

No local-GPU USD cost is stated because no authoritative GPU/electricity price was recorded.

## 7. Paired efficiency

| Metric | Paired n | R1 mean / median | R3 mean / median | R3 − R1 mean | Bootstrap 95% CI |
|---|---:|---|---|---:|---|
| fine_images | 300 | 35.59 / 32.00 | 32.85 / 32.00 | -2.74 | [-7.75, 2.34] |
| downstream_calls | 300 | 10.19 / 11.00 | 9.30 / 9.00 | -0.90 | [-1.69, -0.11] |
| post_planner_e2e_sec | 299 | 133.40 / 128.44 | 118.91 / 110.02 | -14.49 | [-25.91, -3.12] |

Canonical v2 stores aggregate downstream-token and total-modeled-latency distributions, not paired per-route vectors; these are therefore not re-derived from raw data or zero-imputed.

## 8. Failure analysis

| Category | Count | R1 | R3 |
|---|---:|---:|---:|
| downstream_local_context_overflow | 3 | 3 | 0 |
| final_contract_exhaustion | 3 | 1 | 2 |
| fine_visual_max_tokens_or_validation_exhaustion | 8 | 2 | 6 |
| planner_provider_exhausted | 1 | 1 | 0 |

The complete route list is in `FAILURE_BREAKDOWN.csv`.

## 9. Descriptive local vs Planner-API comparison

| Condition | Correct / 300 | Accuracy | Predictions / 300 | Completion | R3 execution context |
|---|---:|---:|---:|---:|---|
| Local R1 | 81/300 | 27.00% | 292/300 | 97.33% | — |
| Local R3 | 52/300 | 17.33% | 145/300 | 48.33% | 150 executable; 150 `context_overflow_pre_model` |
| Planner-API R1 | 78/300 | 26.00% | 293/300 | 97.67% | 300 Planner logical calls |
| Planner-API R3 | 77/300 | 25.67% | 292/300 | 97.33% | 300 attempted / 300 accepted; old-overflow subset 150/150 accepted |

This comparison is descriptive: only the Planner backend was intentionally changed; it does not mix populations or establish claims beyond that controlled change.

### Numbers for Table: End-to-End Evaluation of the Proposed Pipeline

| Metric | Local R1 | Local R3 | Planner-API R1 | Planner-API R3 | Staged API | Direct API |
|---|---:|---:|---:|---:|---:|---:|
| Correct / 300 | 81 | 52 | 78 | 77 |  |  |
| Accuracy | 27.00% | 17.33% | 26.00% | 25.67% |  |  |
| Prediction completion | 97.33% | 48.33% | 97.67% | 97.33% |  |  |

### Thesis-ready key findings

- Replacing only the Planner allowed all 150 previously R3 local-context-overflow cases to reach accepted API Planner output and terminal route states.
- Full-population accuracy was 26.00% for R1 and 25.67% for R3; the paired accuracy difference was −0.33 pp (exact McNemar p=1.0).
- R3 retained substantially higher strict normal-success frequency (119 vs 48), while prediction completion was similar (292 vs 293).
- R3 Planner API use was higher by 5,634,796 tokens and $5.130816, reflecting its richer Planner input.
- After the Planner, R3 used 269 fewer local calls and 821 fewer Fine-image transmissions than R1.
- Canonical v2 excludes one post-terminal duplicate Shared call as orchestration overhead; this did not change accuracy, completion, failures, or Planner API cost.
- The 15 durable failures were dominated by downstream Fine visual max-token/validation exhaustion (8), not Planner API context overflow.

Reporting provenance: source = final canonical_summary_v2 (and frozen local canonical v1 only for the descriptive Local-vs-API table); API/model calls during reporting = 0; raw formal artifacts unchanged; raw closure hash unchanged.
