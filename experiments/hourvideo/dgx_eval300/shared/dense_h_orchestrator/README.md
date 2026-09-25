# V7.4 formal Eval300 serial orchestrator

This directory is orchestration-only. It does not modify the source-complete V7.4
runtime. `orchestrate_eval300.sh --dry-run` performs local path/hash/coverage checks
only. A live run requires an explicit timestamped `FORMAL_ROOT` and `--run`.

Completion means all three are present: metrics status `success`, a trajectory with
`finished_at` and non-empty answer, and a legal A-E prediction. Prediction existence
alone is never sufficient. Each profile has one first-pass attempt and at most one
retry attempt for first-pass incomplete/timeout/error UIDs.

The per-question metrics files are written by the same V7.4
`videoseal.runner.per_question_runner` instrumentation used by the validated
VideoSEAL runs. The finalizer never substitutes for or rewrites them; it adds
read-only aggregate manifests under each profile's `status/` and `merged/` trees.
Three consecutive question-level HTTP failures are treated as persistent service
failure and stop the chain. Individual timeout/parser failures are recorded and
remain eligible for the single retry.
