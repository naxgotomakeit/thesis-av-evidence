# V7.3 VideoSEAL Eval300-runtime-aligned paired experiment

This version is isolated from V7.1, V7.2, the shared VideoSEAL checkout, and the
transferred reference directory. It vendors the 48 transferred runtime files in
`reference_runtime/`. `videoseal/tools/tool_map.py` selects one of two retrieval
implementations, and `videoseal/tools/visual_tools.py` has audit-only retrieval
metadata that does not change its tool response. The new
`videoseal/tools/retrieval_adapter.py` contains the backend seam and hierarchical
telemetry.

No model or API is used by `contract-test`. Eval300 is disabled by policy; this
version's UID file contains exactly the same three informal smoke questions.

## Effective transferred runtime

| Setting | V7.2 smoke | Transferred Eval300 runtime / V7.3 |
|---|---|---|
| Planner | trained `CewEhao/VideoSEAL_8B` | default actually launched by transferred scripts: `Qwen/Qwen3-8B` |
| Planner protocol | tag mode | tag mode (`AGENT_API_USE_MESSAGES=0`) |
| Planner turns | 6 | 16 |
| Final authority | strict code gate requiring structured `SUFFICIENT` | prompt-only rule; controller accepts parsed `<final>` without checking the preceding tool |
| Step 15 | no forced Inspector | flags request forced Inspector + MCQ, but the implementation is only in the messages branch and is not reachable in tag mode |
| After step 16 | fail/no fallback | full-video uniform 64-frame Inspector in MCQ mode |
| Task timeout | no reference-style hard worker timeout; HTTP 600 s | spawn worker hard timeout 1000 s; formal launcher overrides Planner HTTP timeout to 300 s |
| Retrieval | embed, top-6, no summary, minimum gap 0 s | embed, `VISUAL_RETRIEVE_TOPK=30`, semantic top-k 40, summary enabled, summary budget 100, minimum gap 15 s |
| Inspector sampling | 0.2 FPS; 24/call and cumulative 48/question | 2 FPS; 64/call, no cumulative per-question cap |
| Inspector decoding | 1200 tokens, temperature 0 | 4096 tokens, temperature 0.1 |
| Inspector output | strict JSON `SUFFICIENT|SEARCH_MORE` | free-form `Answer: ...`, evidence, confidence and optional `SEARCH_MORE` fields; final MCQ fallback returns a letter |
| Retry | 1 | MLLM 3 attempts, 30 s delay; embedding 5 attempts, 30 s delay |
| Subtitle handling | explicitly disabled | transferred runner forwards any parquet `SUBTITLE_PATH`; this behavior is shared by both V7.3 backends |

The step-15 discrepancy is not repaired here because doing so would no longer be
an exact copy of the runtime that actually ran Eval300. It must be resolved as an
explicit experiment-definition decision before claiming that step 15 was forced.

## Frozen accuracy-efficiency comparison

Both backends use the vendored reference Planner agent, Inspector, prompts,
parsers, frame sampler, 16-turn loop, timeout runner and fallbacks. The only
switch is `RETRIEVAL_BACKEND`:

- `videoseal_flat_top30`: genuine visual-only VideoSEAL embedding index and
  reference embedding scoring/summary path, with at most 30 raw candidates.
- `hierarchical_fixed_width_3_3_6_lexical_coarse`: lexical-guided hierarchical
  visual-semantic retrieval with top-3 Coarse, top-3 eligible Medium and two Fine
  per Medium, for at most six raw candidates.

The raw candidate budgets are intentionally different. Candidate compression is
an efficiency variable of the hierarchical method; this experiment measures an
accuracy-efficiency trade-off and does not claim equal raw candidate budgets.

Both backends share the same Planner, VideoSEAL summarizer, Inspector, frame
sampling rules and budget ceilings, but raw retrieval candidate budgets differ:
flat returns at most 30 and hierarchical at most 6. Actual steps, Inspector calls
and inspected frames are determined by each trajectory and are reported as
accuracy-efficiency outcomes.

Backend identity and hierarchical node paths are metadata only; `ToolOutput` sends
only `output` to the Planner.

## Commands

Offline contract test (safe; no services, models or API):

```bash
cd ${LEGACY_RUNTIME}
python src/experiments/hourvideo_v7_3_videoseal_eval300_runtime_aligned_paired_v1/run.py --stage contract-test
```

For the later three-question smoke, first start the exact transferred model roles
in two terminals. These commands are documentation only and were not run in the
alignment stage:

```bash
VLLM_PYTHON=${SCRATCH_RUNTIME_ROOT}/hourvideo_v72/qaego4d_vllm/bin/python
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
  --model ${SCHOOL_PROJECT_ROOT}/models/Qwen3-8B \
  --served-model-name qwen3-8b-planner-dgx --host 127.0.0.1 --port 18082 \
  --dtype bfloat16 --max-model-len 36864 --gpu-memory-utilization 0.30 \
  --max-num-seqs 1 --enforce-eager --no-enable-log-requests --generation-config vllm
```

```bash
VLLM_PYTHON=${SCRATCH_RUNTIME_ROOT}/hourvideo_v72/qaego4d_vllm/bin/python
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
  --model ${SCHOOL_PROJECT_ROOT}/msc_thesis/models/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen2.5-vl-7b-visual-dgx --host 127.0.0.1 --port 18083 \
  --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.45 \
  --max-num-seqs 1 --limit-mm-per-prompt '{"image":64}' \
  --enforce-eager --no-enable-log-requests --generation-config vllm
```

Load the embedding credentials without printing them, then run only the three
frozen UIDs:

```bash
set -a
source /path/to/private_embedding.env
set +a
cd ${LEGACY_RUNTIME}
python src/experiments/hourvideo_v7_3_videoseal_eval300_runtime_aligned_paired_v1/run.py \
  --stage paired-smoke --backend paired
```

The launcher refuses to overwrite either backend's existing output directory.
It has no Eval300 stage.

The audited source-only data-machine overlay is built by
`build_transfer_package.py`. Deployment and post-run paired statistics are
documented in `DATA_MACHINE_RUNBOOK.md`; building the archive does not transfer
it or run any question.
