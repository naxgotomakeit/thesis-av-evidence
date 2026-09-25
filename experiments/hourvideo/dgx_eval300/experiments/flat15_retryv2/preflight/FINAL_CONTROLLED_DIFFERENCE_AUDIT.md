# Flat-15 v3 Final Controlled-Difference Audit

| Field | Flat-30 | Flat-15 v3 | Classification |
|---|---|---|---|
| Flat `VISUAL_RETRIEVE_TOPK` | 30 caption segments | 15 caption segments | INTENDED SCIENTIFIC DIFFERENCE |
| Pre-service `VLLM_PORT` | unset | unset | ADMINISTRATIVE ENVIRONMENT PARITY RESTORATION |
| Planner API port | 18082 | 18082 | SAME |
| Visual API port | 18083 | 18083 | SAME |
| Planner/Visual CLI and Python environment | frozen reference | byte-identical service scripts / same configured environment | SAME |
| Flat index, Eval300 order, models, prompts, parser, timeout, concurrency, retry, scoring | accepted Flat-30 | same | SAME |
| Full-video endpoint handling | original Flat result unaffected | approved microsecond-safe correction | APPROVED NON-AFFECTING HISTORICAL FIX |

No unexpected scientific difference was found. The later frozen
`smoke_one.sh` assignment of `VLLM_PORT=${PLANNER_PORT}` remains after both
services are health-checked and is not inherited by either service.
