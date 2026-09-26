# Control-sample audit consistency

- Overall exact agreement: 29/60 = 48.3%.
- Unweighted Cohen's kappa: 0.299171.
- Workflow 1: 8/20 = 40.0%.
- Workflow 2: 8/20 = 40.0%.
- Workflow 3: 13/20 = 65.0%.

## Confusion matrix

| old \ new | supported | partially_supported | unsupported | contradicted | unreviewable |
|---|---:|---:|---:|---:|---:|
| supported | 3 | 3 | 2 | 3 | 0 |
| partially_supported | 3 | 9 | 9 | 3 | 0 |
| unsupported | 1 | 3 | 10 | 3 | 0 |
| contradicted | 0 | 1 | 0 | 7 | 0 |
| unreviewable | 0 | 0 | 0 | 0 | 0 |

There are 31 changed control labels. Most common directions: partially_supported→unsupported: 9, supported→contradicted: 3, unsupported→contradicted: 3, supported→partially_supported: 3, unsupported→partially_supported: 3, partially_supported→supported: 3, partially_supported→contradicted: 3, supported→unsupported: 2, contradicted→partially_supported: 1, unsupported→supported: 1. The 20-item workflow samples are too small for firm workflow-level conclusions. The disagreements should be read as potential scale/boundary drift, particularly where `supported`, `partially_supported`, and `unsupported` are adjacent interpretations; no post-hoc pass threshold was applied. Full reasons and evidence references are in `control_disagreements.csv`.
