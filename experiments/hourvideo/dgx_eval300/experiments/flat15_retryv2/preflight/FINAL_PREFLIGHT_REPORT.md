# Flat-15 v3 Preflight Report

## Verdict

**READY TO LAUNCH.** The only scientific difference from accepted Flat-30 is
`VISUAL_RETRIEVE_TOPK: 30 -> 15`, meaning at most 15 Flat caption-segment
candidates per `visual_retrieve` call.

## Administrative environment parity restoration

`flat15.env` no longer exports `VLLM_PORT` before service startup. This now
matches accepted Flat-30: Planner starts on external port 18082 and Visual
starts on external port 18083 while `VLLM_PORT` is unset. The inherited frozen
`smoke_one.sh` still assigns `VLLM_PORT=${PLANNER_PORT}` only after both
service health checks; that downstream behavior is unchanged.

## Runtime and controlled difference

The runtime service scripts `start_model.sh` and `run_eval300.sh` are bytewise
identical to the accepted Flat-30 frozen reference. The scientific runtime
still differs only in the active Flat retrieval budget in `smoke_one.sh`:
`VISUAL_RETRIEVE_TOPK=15`; `SEMANTIC_RETRIEVE_TOPK=40` is unchanged. The
approved microsecond-safe forced-full-video fallback repair is retained and
the prior audit concluded `FLAT_RESULTS_UNAFFECTED; FLAT_AFFECTED_UIDS=[]`.

## Activity

API calls = 0  
model calls = 0  
questions run = 0  
gold loaded = false
