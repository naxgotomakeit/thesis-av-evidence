# V1 and smoke history (not formal results)

- V1 (`gens_hybrid_symmetric_mcq_cap16_v1`, profile SHA-256 `e817014952c76d5cf770062a6ca5d2324f5595b560c7baec1f2db40a0ce45b31`) used `max_new_tokens=512`. Seventy-one of 300 deterministic Stage B responses ended at the length limit, so V1 was superseded before downstream evaluation. Its 229 successful selections were byte/parse/selection-identical in V2; V2 recovered all 71 failures.
- V2 changed the generation limit to 4096 and added observability. It reused the frozen V1 Stage A top-256 candidates with zero Stage A reruns, zero new frame extractions, and zero CLIP image re-encodings.
- A four-UID V2 smoke gate preceded the formal run. Smoke outputs and smoke logs are intentionally not staged.
- `code/wrapper/formal_eval300_selector.py` is included only because the formal V2 wrapper directly imports it and reused its Stage A helpers. No V1 launcher or selected-frame result is included.
