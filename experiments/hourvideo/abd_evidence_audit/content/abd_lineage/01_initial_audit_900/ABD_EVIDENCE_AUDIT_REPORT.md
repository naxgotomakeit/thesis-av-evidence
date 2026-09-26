# ABD Evidence Audit Report

## Frozen audit completion

- 192/192 batches and 900/900 unique opaque audit IDs passed validation.
- Reviewers actually read all provided maps and viewed all submitted images; per-batch counts are in `review_progress.json`.
- Gold and identity were first loaded only after the blinded audit closure was written and verified, at 2026-09-11T02:40:50+01:00.
- External API calls: 0. Original ABD predictions and historical experiments were not modified.

## Evidence-support distributions

| Arm | supported | partially_supported | unsupported | contradicted | unreviewable |
|---|---:|---:|---:|---:|---:|
| A | 39 | 86 | 84 | 23 | 68 |
| B | 75 | 117 | 46 | 15 | 47 |
| D | 95 | 142 | 33 | 29 | 1 |

## Correct-answer grounding

| Arm | Correct | supported | partial | unsupported | contradicted | unreviewable | unsupported-or-contradicted / correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 98 | 20 | 27 | 20 | 8 | 23 | 28 (28.57%) |
| B | 103 | 31 | 42 | 10 | 3 | 17 | 13 (12.62%) |
| D | 103 | 34 | 52 | 10 | 7 | 0 | 17 (16.50%) |


The labels assess whether the submitted answer is supported by the model-visible evidence. They are not a second attempt to answer the question and are not claims about the model's hidden reasoning.
