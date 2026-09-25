#!/usr/bin/env bash
set -euo pipefail

RUN=/home/naxucl/data/HourVideo/videoseal_original/videoseal_flat15_eval300_formal_v1_20260910T184300Z
EXEC="${RUN}/retry_recovery_execution_20260913T123524Z"
AUDIT="${RUN}/retry_infrastructure_audit_20260913T115931Z"
PRE=/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/videoseal_flat15_eval300_preflight_v3_20260910T183640Z
CODE="${PRE}/runtime_overlay_v2"
PARQUET=/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_dev_v1.0_videoseal_dgx.parquet
RETRY_UIDS="${RUN}/retry_protocol_fix_v1_20260913T041000Z/retry_uids.txt"
: "${RETRY_V2_ROOT:?Set RETRY_V2_ROOT to a new versioned retry raw directory}"
: "${RETRY_V2_CONTROL:?Set RETRY_V2_CONTROL to a new versioned control directory}"

mkdir -p "${RETRY_V2_CONTROL}"
exec 9>"${EXEC}/formal_retry_controller.lock"
flock -n 9 || { echo "another formal retry controller holds the global lock" >&2; exit 66; }

export CODE_ROOT="${CODE}"
set -a
source "${CODE}/scripts/dgx/common.env"
source "${PRE}/flat15.env"
set +a
source "${AUDIT}/fixed_orchestration/load_embedding_environment.sh"

export PYTHON=/home/naxucl/projects/VideoSEAL/.venv/bin/python PARQUET
export PYTHONPATH="${EXEC}:${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export DEFAULT_DATA_ROOT="${DATA_ROOT}" CACHE_DIR="${RUN_ROOT}/cache" INDEXES_ROOT="${RUN_ROOT}/indexes"
export FRAMES_ROOT="${TMPDIR:-/tmp}/videoseal_frames_dgx_smoke" CLEAN_FRAMES=0 CLEAN_FRAMES_ASYNC=1
export AGENT_LLM_BACKEND=api AGENT_MLLM_BACKEND=openai AGENT_API_USE_MESSAGES=0
export AGENT_LLM_API_BASE=http://127.0.0.1:18082/v1 AGENT_LLM_API_KEY=local AGENT_LLM_MODEL=qwen3-8b-planner-dgx AGENT_LLM_MAX_TOKENS=2048 AGENT_LLM_TEMPERATURE=0.0 AGENT_LLM_TIMEOUT=300
export VISUAL_INSPECT_BACKEND=openai VISUAL_INSPECT_API_BASE=http://127.0.0.1:18083/v1 VISUAL_INSPECT_API_KEY=local VISUAL_INSPECT_MODEL=qwen2.5-vl-7b-visual-dgx
export VISUAL_RETRIEVE_SUM_BACKEND=openai VISUAL_RETRIEVE_SUM_API_BASE=http://127.0.0.1:18083/v1 VISUAL_RETRIEVE_SUM_API_KEY=local VISUAL_RETRIEVE_SUM_MODEL=qwen2.5-vl-7b-visual-dgx
export MLLM_BACKEND=openai
export USE_VLLM_META_FILE=0 VLLM_NUM_SERVERS=1 VLLM_PORT="${PLANNER_PORT}"
export VLLM_MODEL="${PLANNER_MODEL}" VLLM_SERVED_MODEL_NAME="${PLANNER_SERVED_NAME}"
export EMBEDDING_USAGE_LEDGER="${RETRY_V2_CONTROL}/embedding_usage.jsonl"
export EMBEDDING_INPUT_USD_PER_MILLION=0.13
unset EMBEDDING_USAGE_MAX_REQUESTS

[[ "${VISUAL_RETRIEVE_TOPK}" == 15 ]] || { echo "Flat-15 Top-K gate failed" >&2; exit 67; }
[[ "${CONCURRENCY}" == 1 ]] || { echo "retry concurrency gate failed" >&2; exit 68; }
[[ "${TASK_TIMEOUT_SEC}" == 1000 ]] || { echo "retry timeout gate failed" >&2; exit 69; }
[[ "$(wc -l < "${RETRY_UIDS}")" == 57 ]] || { echo "retry manifest count gate failed" >&2; exit 71; }
[[ "$(sha256sum "${RETRY_UIDS}" | awk '{print $1}')" == 85899124b010d689fee1dfec084c5edc8ba305a03a6a4478db8f07e2c02dad36 ]] || exit 72

curl -fsS --max-time 5 http://127.0.0.1:18082/v1/models > "${RETRY_V2_CONTROL}/planner_models.json"
curl -fsS --max-time 5 http://127.0.0.1:18083/v1/models > "${RETRY_V2_CONTROL}/visual_models.json"
grep -q 'qwen3-8b-planner-dgx' "${RETRY_V2_CONTROL}/planner_models.json" || exit 73
grep -q 'qwen2.5-vl-7b-visual-dgx' "${RETRY_V2_CONTROL}/visual_models.json" || exit 74

"${PYTHON}" - <<'PY'
from pathlib import Path
import os
import videoseal.agents.tool_agent as ta
import videoseal.tools.visual_tools as vt
root = Path(os.environ["CODE_ROOT"]).resolve()
for module in (ta, vt):
    path = Path(module.__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime isolation gate failed: {path}")
print(f"tool_agent={Path(ta.__file__).resolve()}")
print(f"visual_tools={Path(vt.__file__).resolve()}")
PY

printf 'credential_file_present=true\ncredential_file_readable=true\nembedding_key_set=true\ntop_k=15\nconcurrency=1\ntask_timeout_sec=1000\ngold_loaded=false\n' > "${RETRY_V2_CONTROL}/launch_gate.redacted.txt"

set +e
"${PYTHON}" "${EXEC}/gold_free_retry_runner_v3.py" \
  --parquet "${PARQUET}" --uids "${RETRY_UIDS}" --save-runs "${RETRY_V2_ROOT}" \
  --max-steps 16 --task-timeout-sec 1000 --resume
rc=$?
set -e
printf '%s\n' "${rc}" > "${RETRY_V2_CONTROL}/runner_exit_code.txt"
if [[ ${rc} -eq 0 ]]; then
  printf 'RETRY_RAW_COMPLETE_PENDING_FREEZE\n' > "${RETRY_V2_CONTROL}/state.txt"
else
  printf 'RETRY_STOPPED_EXIT_%s\n' "${rc}" > "${RETRY_V2_CONTROL}/state.txt"
fi
exit "${rc}"

