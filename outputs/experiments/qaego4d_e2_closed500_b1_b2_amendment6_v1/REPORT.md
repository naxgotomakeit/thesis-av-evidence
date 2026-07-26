# Formal E2 Closed-500 — B0/B1/B2

B0 is the immutable Formal E1 Closed Uniform-8 baseline; this run generated only B1 and B2 under Amendment #6.

## Accuracy

| Method | Correct / 500 | Accuracy | Wilson 95% CI |
|---|---:|---:|---:|
| B0 | 262/500 | 52.4000% | [48.0208%, 56.7426%] |
| B1 | 288/500 | 57.6000% | [53.2265%, 61.8576%] |
| B2 | 291/500 | 58.2000% | [53.8303%, 62.4447%] |

## Pairwise tests

- **b0_vs_b1**: Δ(right−left)=5.2000%; McNemar exact p=0.02097; bootstrap 95% CI=[0.01, 0.094]; approximate MDE=6.0866%.
- **b0_vs_b2**: Δ(right−left)=5.8000%; McNemar exact p=0.00996632; bootstrap 95% CI=[0.014, 0.102]; approximate MDE=6.1123%.
- **b1_vs_b2**: Δ(right−left)=0.6000%; McNemar exact p=0.812589; bootstrap 95% CI=[-0.028, 0.04]; approximate MDE=4.7213%.

## Efficiency

Per-query online latency is logged separately from offline index creation. Break-even N* is reported only where B1/B2 were empirically faster than B0.

## Figures

Self-contained SVG figures were rendered from the frozen `summary.json` without
rerunning inference: `figures/accuracy_wilson.svg`,
`figures/latency_decomposition.svg`, and `figures/break_even.svg`.
