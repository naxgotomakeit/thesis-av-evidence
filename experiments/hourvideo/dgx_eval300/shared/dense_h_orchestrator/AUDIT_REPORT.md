# V7.4 formal Eval300 orchestration audit

## Existing runner behavior

- `per_question_runner` returns zero after processing a batch even when individual
  questions time out or fail. Orchestration therefore inspects per-question metrics
  and service health instead of treating process exit zero as experimental success.
- Its built-in `--skip-existing` accepts any non-empty prediction. It is not used by
  this orchestration because timeout post-processing predictions are not trusted.
- A hard task timeout writes a metrics JSON with `status=timeout`,
  `error_type=timeout`, and approximately the configured elapsed time. Its partial
  trajectory lacks a normal `finished_at`; any prediction left or backfilled beside
  it does not make the question complete.
- Strict completion requires metrics `status=success`, a trajectory with non-empty
  `finished_at` and answer, and a legal A-E prediction.

## Isolation

Each profile has separate `first_pass`, `retry_1`, `merged`, `logs`, and `status`
trees. The runner writes its normal per-question metrics separately under each
phase. Finalization reads but never changes first-pass/retry predictions,
trajectories, or metrics. Deployment-smoke and flat-reference outputs are read-only.

## Chain-stopping infrastructure conditions

- Planner or Inspector exits or fails readiness/health checks.
- Port 18082 or 18083 is occupied before a profile starts.
- Non-zero runner process exit.
- OOM, CUDA, EngineCore death, import failure, or model/video/index path failure.
- Three consecutive question-level HTTP failures.
- Frozen UID, runtime, Parquet, index, config, or orchestration hash drift.
- Incomplete attempt coverage before advancing, or a duplicate/residual runner.

Individual question timeout and parser failure are retained and proceed to the one
allowed retry; remaining failures do not prevent finalization or the next profile.

