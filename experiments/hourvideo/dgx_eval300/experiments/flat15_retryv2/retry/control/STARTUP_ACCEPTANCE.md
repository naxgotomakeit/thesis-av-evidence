# Flat-15 retry raw v2 startup acceptance

Status: **FORMAL_RETRY_RUNNING**

- tmux: `videoseal_flat15_eval300_retry_v2_20260913T124100Z`
- controller PID: `1099780`
- runner PID: `1099852`
- retry raw directory: `retry_raw_v2_20260913T124100Z`
- retry population: 57 unique UIDs from the frozen retry manifest
- first active UID: `6fd90f8d-7a4d-425d-a812-3268db0b0342_7_7`
- first trajectory: `6fd90f8d-7a4d-425d-a812-3268db0b0342/20260913T124202Z-571n/trajectory.json`
- first real retrieval: PASS (`observation.ok=true`)
- first formal embedding request: PASS, provider-reported 5 input tokens
- Flat Top-K: 15
- concurrency: 1
- per-question hard timeout: 1000 seconds
- Planner: healthy on 127.0.0.1:18082
- Visual/Summarizer/Inspector service: healthy on 127.0.0.1:18083
- gold loaded: false
- backfill/scoring: disabled
- retry-v1 diagnostic records used for skip/merge: false

At acceptance time, the first UID was active and no UID had yet reached a
terminal retry-v2 state. The runner remained alive after the detached launch
command exited.

