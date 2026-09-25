# VideoSEAL Flat-15 Eval300 preflight v3

This directory prepares, but does not launch, the controlled Flat-15 condition.

The authoritative runtime is `runtime_overlay_v2`: the clean tree at the Flat-30 recorded Git HEAD, overlaid by every file in the accepted 48-file Flat-30 frozen runtime. Two explicit non-algorithmic lineage changes are present: `common.env` respects the launcher's locked `CODE_ROOT` so imports cannot escape to the dirty checkout, and `tool_agent.py` contains the approved microsecond-safe forced full-video fallback correction. The earlier `runtime_overlay` directory is an incomplete diagnostic copy and MUST NOT be launched. The accepted Flat-30 fallback audit concluded `FLAT_RESULTS_UNAFFECTED` and `FLAT_AFFECTED_UIDS=[]`. Flat retrieval code, caption index, prompts, parser, evaluator, and runner remain the accepted Flat implementation.

The sole experimental-condition change is `VISUAL_RETRIEVE_TOPK=30 -> 15`. This limits retrieved Flat caption segments presented to the retrieval Summarizer; it does not limit Inspector images.

## Service-environment parity restoration

Before Planner and Visual service startup, `VLLM_PORT` is deliberately unset,
matching accepted Flat-30. The prior v2 candidate exported `VLLM_PORT=18082`
too early, which made vLLM allocate Planner's internal NCCL/TCPStore endpoint
on 18083 and collide with the Visual API. The frozen `smoke_one.sh` continues
to set `VLLM_PORT=${PLANNER_PORT}` only after both services pass their health
checks; this historical downstream behavior is unchanged.

## Resume and retry contract

The accepted runner is resumable only with an explicit frozen-order pending UID file. It does not discover strict completion automatically. A continuation must therefore be generated read-only from complete per-question artifacts and must exclude already completed routes. The runner's `--skip-existing` check is prediction-presence based, not the authoritative strict-completion definition, so it must not be used to construct the formal retry population.

Formal retry is a separate output root, exactly as for Flat-30. It contains at most one retry attempt per first-pass strict-incomplete UID. Canonical merging selects one final attempt per UID; old and retry elapsed times are not added to final-selected-attempt efficiency. Actual total compute may be reported separately as first pass plus retry.

This preflight does not silently solve orchestration mistakes: launch tooling must freeze the pending/retry list and its SHA before use.

## Launch guard

`run_flat15_eval300_formal_v1.sh` refuses to run unless `RUN_MODE=first` and an explicit absent `FLAT15_OUTPUT_ROOT` are provided. No service or inference was started while producing this directory.
