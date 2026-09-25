#!/usr/bin/env bash
set -euo pipefail

RUN=/home/naxucl/data/HourVideo/videoseal_original/videoseal_flat15_eval300_formal_v1_20260910T184300Z
EXEC="${RUN}/retry_recovery_execution_20260913T123524Z"
AUDIT="${RUN}/retry_infrastructure_audit_20260913T115931Z"
PRE=/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/videoseal_flat15_eval300_preflight_v3_20260910T183640Z
CODE="${PRE}/runtime_overlay_v2"
: "${SMOKE_ROOT:?Set SMOKE_ROOT to the isolated smoke output directory}"

mkdir -p "${SMOKE_ROOT}"
export CODE_ROOT="${CODE}"
set -a
source "${CODE}/scripts/dgx/common.env"
source "${PRE}/flat15.env"
set +a
source "${AUDIT}/fixed_orchestration/load_embedding_environment.sh"

export PYTHON=/home/naxucl/projects/VideoSEAL/.venv/bin/python
export PYTHONPATH="${EXEC}:${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export AGENT_MLLM_BACKEND=openai MLLM_BACKEND=openai
export VISUAL_RETRIEVE_SUM_BACKEND=openai
export VISUAL_RETRIEVE_SUM_API_BASE=http://127.0.0.1:18083/v1
export VISUAL_RETRIEVE_SUM_API_KEY=local
export VISUAL_RETRIEVE_SUM_MODEL=qwen2.5-vl-7b-visual-dgx
export VISUAL_RETRIEVE_LOG_SPANS=1
# Smoke-only hard cap. The formal retry remains at the frozen value 3.
export MLLM_RETRY_TIMES=1
export EMBEDDING_USAGE_MAX_REQUESTS=1
export EMBEDDING_USAGE_LEDGER="${SMOKE_ROOT}/embedding_usage.jsonl"
export EMBEDDING_INPUT_USD_PER_MILLION=0.13

[[ "${VISUAL_RETRIEVE_TOPK}" == 15 ]] || exit 81
[[ "${TASK_TIMEOUT_SEC}" == 1000 ]] || exit 82
[[ "${CONCURRENCY}" == 1 ]] || exit 83
curl -fsS --max-time 5 http://127.0.0.1:18083/v1/models > "${SMOKE_ROOT}/visual_models.json"

exec "${PYTHON}" "${EXEC}/real_retrieval_smoke.py" \
  --index /home/naxucl/data/HourVideo/videoseal_original/indexes/semantic/6fd90f8d-7a4d-425d-a812-3268db0b0342 \
  --video-id 6fd90f8d-7a4d-425d-a812-3268db0b0342 \
  --query 'kitchen items and their positions' \
  --result "${SMOKE_ROOT}/smoke_result.json"

