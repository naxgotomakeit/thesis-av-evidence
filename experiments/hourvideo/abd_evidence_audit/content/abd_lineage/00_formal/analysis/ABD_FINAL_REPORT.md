# ABD Eval300 final report

Raw closure SHA-256: `b16175b602bd62e82dd42e44ec5a4249654acf0d75a90a2fda623ad81057649d`.

| Arm | Complete | Correct | Accuracy | Input tokens | Output tokens | Cache create/read | Cost | API mean / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 300/300 | 98 | 32.67% | 9,101,885 | 38,332 | 0/0 | $9.293545 | 2.146s / 2.766s |
| B | 300/300 | 103 | 34.33% | 1,865,812 | 32,206 | 0/0 | $2.026842 | 1.982s / 2.525s |
| D | 300/300 | 103 | 34.33% | 10,658,787 | 35,810 | 0/0 | $10.837837 | 2.416s / 3.245s |

## Paired comparisons

Primary D vs B: both correct 69; D-only 34; B-only 34; neither 163; accuracy difference +0.00%.
Supplementary D vs A: both correct 79; D-only 24; A-only 19; neither 178; accuracy difference +1.67%.

All arm accuracies use 300 as the denominator. Missing and failed tasks would count as incorrect; this run has no missing tasks. No judge calls were made.
