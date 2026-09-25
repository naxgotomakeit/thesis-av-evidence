# V7.4 data-machine runbook

This deployment preserves the 48-file Eval300 runtime. The only experiment
switch is `RETRIEVAL_BACKEND`; hierarchical width is selected separately with
`HIERARCHICAL_PROFILE=h6|h15|h30`. Do not place QA data inside either bundle.

## Install

Verify the two archive SHA-256 values from their sidecar manifests, extract each
into a new experiment-specific directory, and run the bundled offline contracts
before starting any service. Set `INDEX_ROOT` to the extracted index directory.

```bash
export V74_ROOT=/data/experiments/hourvideo_v7_4_variant_c_budgets_v1
export V74_RUNTIME="$V74_ROOT/runtime"
export INDEX_ROOT="$V74_ROOT/index/work_index"
export PAIRED_HIERARCHICAL_ROOT="$INDEX_ROOT"
export PAIRED_FLAT_INDEX_ROOT=/data/indexes/videoseal_visual_only_flat
export PYTHONPATH="$V74_RUNTIME/runnable_runtime:$V74_RUNTIME/runtime_support:$V74_RUNTIME/runtime_support/src"
python "$V74_RUNTIME/experiment/test_summarizer_contract_v74.py"
python "$V74_RUNTIME/experiment/offline_contract_v74.py"
```

The offline summarizer test uses a deterministic stub and never calls a model
or API. Live execution requires the frozen reference services and environment.
`run.py` gives these deployment path variables precedence over the archived
experiment-machine paths. Keep `VISUAL_RETRIEVE_SUMMARY_ENABLED=1`,
`VISUAL_RETRIEVE_RETURN_SPANS=0`, `VISUAL_RETRIEVE_TOPK=30`, `MAX_STEPS=16`,
`TASK_TIMEOUT_SEC=1000`, and `AGENT_API_USE_MESSAGES=0`.

Profiles:

- `h6`: gated Coarse 3, global Medium 3, two Fine per Medium, at most 6.
- `h15`: gated Coarse 5, global Medium 5, three Fine per Medium, at most 15.
- `h30`: all Coarse and Medium expanded, no front-layer gate, at most 30 Fine.

Run one backend/profile at a time and use distinct absent output directories.
Do not use an Eval300 question as warm-up. This runbook intentionally contains
no UID-list path and does not start smoke or Eval300.
