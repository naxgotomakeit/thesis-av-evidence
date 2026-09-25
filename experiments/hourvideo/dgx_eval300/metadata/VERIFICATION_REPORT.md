# Eval300 staging verification

Overall: **PASS**

| Experiment | Rows | Unique UID | Same UID set | Strict complete | Correct | Timeout | Other incomplete | Result |
|---|---:|---:|---|---:|---:|---:|---:|---|
| flat30 | 300 | 300 | True | 254 | 82 | 36 | 10 | PASS |
| flat15_retryv2 | 300 | 300 | True | 260 | 76 | 33 | 7 | PASS |
| dense_h8_v3 | 300 | 300 | True | 270 | 78 | 29 | 1 | PASS |
| dense_h15_v1 | 300 | 300 | True | 270 | 81 | 26 | 4 | PASS |
| dense_h30_v2 | 300 | 300 | True | 249 | 78 | 45 | 6 | PASS |

All UID comparisons are set comparisons against the locally staged canonical Eval300 UID file.
The differing Flat-15 ordered-file byte hash does not indicate a different UID population.
