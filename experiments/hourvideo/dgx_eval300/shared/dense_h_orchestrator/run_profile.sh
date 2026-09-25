#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=formal.env
set -a
# shellcheck disable=SC1091
source "${HERE}/formal.env"
set +a
PROFILE=${1:?profile h6|h15|h30 required}
FORMAL_ROOT=${FORMAL_ROOT:?FORMAL_ROOT must be the explicit resumable run root}
case "${PROFILE}" in h6|h15|h30) ;; *) echo "invalid profile ${PROFILE}" >&2; exit 2;; esac
PROFILE_ROOT=${FORMAL_ROOT}/${PROFILE}
FIRST=${PROFILE_ROOT}/first_pass
RETRY=${PROFILE_ROOT}/retry_1
LOGS=${PROFILE_ROOT}/logs
STATUS=${PROFILE_ROOT}/status
mkdir -p "${FIRST}" "${RETRY}" "${LOGS}" "${STATUS}"

profile_config=${STATUS}/experiment_config.json
"${RUNNER_PYTHON}" - "${profile_config}" "${PROFILE}" "${FORMAL_ROOT}" <<'PY'
import json,os,pathlib,sys
target=pathlib.Path(sys.argv[1])
data={"profile":sys.argv[2],"formal_root":sys.argv[3],"runtime":os.environ.get("RUNTIME"),
      "uids_file":os.environ.get("UIDS_FILE"),"uids_sha256":os.environ.get("UIDS_SHA256"),
      "parquet":os.environ.get("PARQUET"),"hierarchical_index":os.environ.get("HIERARCHICAL_INDEX"),
      "siglip_model":os.environ.get("SIGLIP_MODEL"),"planner_model":os.environ.get("PLANNER_MODEL"),
      "planner_served_name":os.environ.get("PLANNER_SERVED_NAME"),"inspector_model":os.environ.get("INSPECTOR_MODEL"),
      "inspector_served_name":os.environ.get("INSPECTOR_SERVED_NAME"),"concurrency":1,"max_steps":16,
      "http_timeout_sec":300,"task_timeout_sec":1000,"inspector_hard_gate":False,
      "step15_actual_behavior":"reference-unmodified","step16_full_video_fallback_frames":64}
text=json.dumps(data,indent=2,sort_keys=True)+"\n"
if target.exists() and target.read_text()!=text: raise SystemExit(f"profile config drift: {target}")
target.write_text(text)
PY

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${RUNTIME}/runnable_runtime:${RUNTIME}/runtime_support:${RUNTIME}/runtime_support/src"
vllm_bin_dir=$(dirname "${VLLM_PYTHON}")
export PATH="${vllm_bin_dir}:${PATH}"
set -a
# shellcheck disable=SC1090
source "${EMBED_ENV}"
set +a
: "${EMBEDDING_API_KEY:?missing embedding key}" "${EMBEDDING_API_BASE:?missing embedding base}" "${EMBEDDING_MODEL:?missing embedding model}"

export AGENT_LLM_BACKEND=api AGENT_MLLM_BACKEND=openai AGENT_API_USE_MESSAGES=0
export AGENT_LLM_API_BASE=http://${PLANNER_HOST}:${PLANNER_PORT}/v1 AGENT_LLM_API_KEY=local AGENT_LLM_MODEL=${PLANNER_SERVED_NAME}
export AGENT_LLM_MAX_TOKENS=2048 AGENT_LLM_TEMPERATURE=0.0 AGENT_LLM_TIMEOUT=${HTTP_TIMEOUT_SEC}
export VISUAL_INSPECT_BACKEND=openai VISUAL_INSPECT_API_BASE=http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1 VISUAL_INSPECT_API_KEY=local VISUAL_INSPECT_MODEL=${INSPECTOR_SERVED_NAME}
export VISUAL_RETRIEVE_SUM_BACKEND=openai VISUAL_RETRIEVE_SUM_API_BASE=http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1 VISUAL_RETRIEVE_SUM_API_KEY=local VISUAL_RETRIEVE_SUM_MODEL=${INSPECTOR_SERVED_NAME}
export VISUAL_RETRIEVE_SUMMARY_ENABLED=1 RETRIEVE_SUMMARY_ENABLED=1 VISUAL_RETRIEVE_RETURN_SPANS=0 RETRIEVE_SUMMARY_MAX_SPANS=100
export VISUAL_RETRIEVE_SUM_MAX_TOKENS=800 VISUAL_RETRIEVE_SUM_TEMPERATURE=0.0 BENCHMARK=lvbench SEMANTIC_RETRIEVE_MIX=embed
export SEMANTIC_RETRIEVE_TOPK=40 VISUAL_RETRIEVE_TOPK=30 RETRIEVE_MIN_TIME_GAP_SEC=15
export RETRIEVAL_BACKEND=hierarchical HIERARCHICAL_PROFILE=${PROFILE} PAIRED_HIERARCHICAL_ROOT=${HIERARCHICAL_INDEX} INDEX_ROOT=${HIERARCHICAL_INDEX}
siglip_cache=$(dirname "$(dirname "${SIGLIP_MODEL}")")
export HIERARCHICAL_SIGLIP_CACHE=${siglip_cache} HIERARCHICAL_SIGLIP_MODEL=${SIGLIP_MODEL}
export HIERARCHICAL_COARSE_TOPK=3 HIERARCHICAL_MEDIUM_TOPK=3 HIERARCHICAL_FINE_PER_MEDIUM=2
export INSPECT_MAX_LONG_EDGE=1280 INSPECT_VLM_MAX_TOKENS=4096 INSPECT_VLM_TEMPERATURE=0.1 INSPECT_MAX_TOTAL_IMAGES=64 INSPECT_FPS=2
export VISUAL_INSPECT_MAX_LONG_EDGE=1280 VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE=1 VISUAL_INSPECT_TOTAL_PIXELS=20971520 VISUAL_INSPECT_MIN_PIXELS=12544 VISUAL_INSPECT_EDGE_MULTIPLE=32
export MLLM_BACKEND=openai MLLM_MAX_TOKENS=4096 MLLM_TIMEOUT=${HTTP_TIMEOUT_SEC} MLLM_RETRY_TIMES=3 MLLM_RETRY_DELAY=30 EMBED_RETRY_TIMES=5 EMBED_RETRY_DELAY=30
export MAX_STEPS TASK_TIMEOUT_SEC CONCURRENCY AGENT_FORCE_LAST_STEP_VISUAL_INSPECT=1 AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK=1 AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK=1
export AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE=mcq AGENT_PARSE_FAILURE_FALLBACK_TO_C=0

planner_pid=''
inspector_pid=''
memory_pid=''
stop_services() {
  set +e
  [[ -z "${memory_pid}" ]] || kill "${memory_pid}" 2>/dev/null
  [[ -z "${inspector_pid}" ]] || kill -TERM -- "-${inspector_pid}" 2>/dev/null
  [[ -z "${planner_pid}" ]] || kill -TERM -- "-${planner_pid}" 2>/dev/null
  [[ -z "${memory_pid}" ]] || wait "${memory_pid}" 2>/dev/null
  [[ -z "${inspector_pid}" ]] || wait "${inspector_pid}" 2>/dev/null
  [[ -z "${planner_pid}" ]] || wait "${planner_pid}" 2>/dev/null
}
trap stop_services EXIT INT TERM

if ss -ltn | grep -Eq ":(${PLANNER_PORT}|${INSPECTOR_PORT})[[:space:]]"; then echo "profile ports occupied before ${PROFILE}" >&2; exit 10; fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
planner_log=${LOGS}/planner-${stamp}.log; inspector_log=${LOGS}/inspector-${stamp}.log; memory_log=${LOGS}/memory-${stamp}.log
setsid "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server --model "${PLANNER_MODEL}" --served-model-name "${PLANNER_SERVED_NAME}" --host "${PLANNER_HOST}" --port "${PLANNER_PORT}" --dtype bfloat16 --max-model-len "${PLANNER_MAX_MODEL_LEN}" --gpu-memory-utilization "${PLANNER_GPU_MEMORY_UTIL}" --max-num-seqs 1 --enforce-eager --no-enable-log-requests --generation-config vllm >"${planner_log}" 2>&1 & planner_pid=$!
for _ in $(seq 1 450); do curl -fsS "http://${PLANNER_HOST}:${PLANNER_PORT}/v1/models" >/dev/null && break; kill -0 "${planner_pid}" 2>/dev/null || { echo "Planner exited" >&2; exit 11; }; sleep 2; done
curl -fsS "http://${PLANNER_HOST}:${PLANNER_PORT}/v1/models" >/dev/null || { echo "Planner readiness timeout" >&2; exit 12; }
setsid "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server --model "${INSPECTOR_MODEL}" --served-model-name "${INSPECTOR_SERVED_NAME}" --host "${INSPECTOR_HOST}" --port "${INSPECTOR_PORT}" --dtype bfloat16 --max-model-len "${INSPECTOR_MAX_MODEL_LEN}" --gpu-memory-utilization "${INSPECTOR_GPU_MEMORY_UTIL}" --max-num-seqs 1 --limit-mm-per-prompt '{"image":64}' --enforce-eager --no-enable-log-requests --generation-config vllm >"${inspector_log}" 2>&1 & inspector_pid=$!
for _ in $(seq 1 450); do curl -fsS "http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1/models" >/dev/null && break; kill -0 "${inspector_pid}" 2>/dev/null || { echo "Inspector exited" >&2; exit 13; }; sleep 2; done
curl -fsS "http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1/models" >/dev/null || { echo "Inspector readiness timeout" >&2; exit 14; }

(while :; do printf '%s ' "$(date -u +%FT%TZ)"; awk '/MemAvailable|SwapFree/{printf "%s=%s%s ",$1,$2,$3} END{print ""}' /proc/meminfo; ps -o pid=,ppid=,rss=,cmd= --forest -g "${planner_pid},${inspector_pid}" 2>/dev/null || true; sleep 10; done) >"${memory_log}" 2>&1 & memory_pid=$!

run_phase() {
  local kind=$1 phase=$2 source=${3:-}
  local list=${STATUS}/${kind}-pending-${stamp}.txt log=${LOGS}/${kind}-runner-${stamp}.log
  local args=(pending --uids "${UIDS_FILE}" --phase "${phase}" --kind "${kind}" --output "${list}")
  [[ -z "${source}" ]] || args+=(--source "${source}")
  "${RUNNER_PYTHON}" "${HERE}/state_tool.py" "${args[@]}" | tee -a "${STATUS}/orchestrator.jsonl"
  local count; count=$(awk 'NF{n++} END{print n+0}' "${list}")
  [[ "${count}" -gt 0 ]] || return 0
  set +e
  /usr/bin/time -p "${RUNNER_PYTHON}" -m videoseal.runner.per_question_runner --parquet "${PARQUET}" --uids-file "${list}" --save-runs "${phase}" --concurrency 1 --max-steps 16 --task-timeout-sec 1000 >"${log}" 2>&1
  local rc=$?
  set -e
  printf 'runner_exit_code=%s\n' "${rc}" >>"${log}"
  [[ "${rc}" -eq 0 ]] || { echo "runner infrastructure exit ${rc}" >&2; exit 20; }
  curl -fsS "http://${PLANNER_HOST}:${PLANNER_PORT}/v1/models" >/dev/null || { echo "Planner unhealthy" >&2; exit 21; }
  curl -fsS "http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1/models" >/dev/null || { echo "Inspector unhealthy" >&2; exit 22; }
  if grep -Eqi 'CUDA error|CUDA out of memory|out of memory|EngineCore.*died|ModuleNotFoundError|ImportError|No such file.*(index|model|video)' "${log}" "${planner_log}" "${inspector_log}"; then echo "infrastructure signature detected" >&2; exit 23; fi
  "${RUNNER_PYTHON}" "${HERE}/state_tool.py" infra-gate --uids "${list}" --phase "${phase}" --threshold 3 | tee -a "${STATUS}/orchestrator.jsonl" || { echo "persistent HTTP failure gate" >&2; exit 24; }
}

run_phase first "${FIRST}"
"${RUNNER_PYTHON}" "${HERE}/state_tool.py" coverage --uids "${UIDS_FILE}" --phase "${FIRST}" --require-attempted 300 | tee -a "${STATUS}/orchestrator.jsonl"
run_phase retry "${RETRY}" "${FIRST}"
retry_expected=$("${RUNNER_PYTHON}" "${HERE}/state_tool.py" pending --uids "${UIDS_FILE}" --phase /dev/null/not-used --kind retry --source "${FIRST}" --output "${STATUS}/retry-all-${stamp}.txt" | "${RUNNER_PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["pending"])')
"${RUNNER_PYTHON}" "${HERE}/state_tool.py" coverage --uids "${STATUS}/retry-all-${stamp}.txt" --phase "${RETRY}" --require-attempted "${retry_expected}" | tee -a "${STATUS}/orchestrator.jsonl"
"${RUNNER_PYTHON}" "${HERE}/finalize_profile.py" --profile "${PROFILE}" --uids "${UIDS_FILE}" --first "${FIRST}" --retry "${RETRY}" --flat "${FLAT_REFERENCE}" --profile-root "${PROFILE_ROOT}" | tee -a "${STATUS}/orchestrator.jsonl"
