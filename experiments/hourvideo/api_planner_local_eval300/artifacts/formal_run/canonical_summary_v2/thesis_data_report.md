# Planner-only API ablation — canonical v2

- Raw closure SHA-256: `020ad1a5d12805a3bf03c5d719911aa9fc98deed7c3be1d89e5bf0218c37bcd8` (unchanged after aggregation).
- R1: 78/300 (26.00%), predictions 293/300.
- R3: 77/300 (25.67%), predictions 292/300.
- Paired R3−R1: -0.33 pp; McNemar p=1.
- Planner API cost: R1 $4.965457; R3 $10.096273; total $15.061730.
- Post-terminal overhead excluded: 1 completed local Shared call, 2225 input tokens, 184 output tokens, 2.2346s, 0 images; 1 unmatched Direct Final request-start.

All scientific outcomes use the first durable terminal boundary.
