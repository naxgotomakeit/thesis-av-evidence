# Eval300 timeout recovery report

These are retry-augmented results and must not be described as the original single-pass Eval300 results.

## Original Planner
- First pass: 228 completed + 72 timeout
- Retry recovered: 36
- Retry still timeout/incomplete: 39
- Retry-augmented: 261 completed + 39 timeout/incomplete
- Retry-augmented accuracy (300 denominator): 0.266667

## Trained Planner
- First pass: 235 completed + 65 timeout
- Retry recovered: 23
- Retry still timeout/incomplete: 50
- Retry-augmented: 250 completed + 50 timeout/incomplete
- Retry-augmented accuracy (300 denominator): 0.240000

Only retries with status=success, a finished trajectory, non-empty steps, and a legal A-E prediction were merged.
