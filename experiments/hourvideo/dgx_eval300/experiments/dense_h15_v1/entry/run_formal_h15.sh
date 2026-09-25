#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1
ORCH=${BASE}/formal_eval300_orchestrator
PARITY=${BASE}/outputs/dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z
OVERLAY=${PARITY}/runtime_overlay
PROFILE_ROOT=${HERE}/h15
FIRST=${PROFILE_ROOT}/first_pass
RETRY=${PROFILE_ROOT}/retry_1
LOGS=${PROFILE_ROOT}/logs
STATUS=${PROFILE_ROOT}/status
mkdir -p "${FIRST}" "${RETRY}" "${LOGS}" "${STATUS}"

set -a
source "${ORCH}/formal.env"
source "${EMBED_ENV}"
set +a

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${OVERLAY}:${RUNTIME}/runtime_support:${RUNTIME}/runtime_support/src"
export PATH="$(dirname "${VLLM_PYTHON}"):${PATH}"
export AGENT_LLM_BACKEND=api AGENT_MLLM_BACKEND=openai AGENT_API_USE_MESSAGES=0
export AGENT_LLM_API_BASE=http://${PLANNER_HOST}:${PLANNER_PORT}/v1 AGENT_LLM_API_KEY=local AGENT_LLM_MODEL=${PLANNER_SERVED_NAME}
export AGENT_LLM_MAX_TOKENS=2048 AGENT_LLM_TEMPERATURE=0.0 AGENT_LLM_TIMEOUT=${HTTP_TIMEOUT_SEC}
export VISUAL_INSPECT_BACKEND=openai VISUAL_INSPECT_API_BASE=http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1 VISUAL_INSPECT_API_KEY=local VISUAL_INSPECT_MODEL=${INSPECTOR_SERVED_NAME}
export VISUAL_RETRIEVE_SUM_BACKEND=openai VISUAL_RETRIEVE_SUM_API_BASE=http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1 VISUAL_RETRIEVE_SUM_API_KEY=local VISUAL_RETRIEVE_SUM_MODEL=${INSPECTOR_SERVED_NAME}
export VISUAL_RETRIEVE_SUMMARY_ENABLED=1 RETRIEVE_SUMMARY_ENABLED=1 VISUAL_RETRIEVE_RETURN_SPANS=0 RETRIEVE_SUMMARY_MAX_SPANS=100
export VISUAL_RETRIEVE_SUM_MAX_TOKENS=800 VISUAL_RETRIEVE_SUM_TEMPERATURE=0.0 BENCHMARK=lvbench SEMANTIC_RETRIEVE_MIX=embed
export SEMANTIC_RETRIEVE_TOPK=40 VISUAL_RETRIEVE_TOPK=30 RETRIEVE_MIN_TIME_GAP_SEC=15
export RETRIEVAL_BACKEND=dense_semantic_beam_b DENSE_BEAM_B=15 DENSE_BEAM_SIGLIP_MODEL=${SIGLIP_MODEL}
export HIERARCHICAL_PROFILE=dense_beam_b_h15 PAIRED_HIERARCHICAL_ROOT=${HIERARCHICAL_INDEX} INDEX_ROOT=${HIERARCHICAL_INDEX}
export INSPECT_MAX_LONG_EDGE=1280 INSPECT_VLM_MAX_TOKENS=4096 INSPECT_VLM_TEMPERATURE=0.1 INSPECT_MAX_TOTAL_IMAGES=64 INSPECT_FPS=2
export VISUAL_INSPECT_MAX_LONG_EDGE=1280 VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE=1 VISUAL_INSPECT_TOTAL_PIXELS=20971520 VISUAL_INSPECT_MIN_PIXELS=12544 VISUAL_INSPECT_EDGE_MULTIPLE=32
export MLLM_BACKEND=openai MLLM_MAX_TOKENS=4096 MLLM_TIMEOUT=${HTTP_TIMEOUT_SEC} MLLM_RETRY_TIMES=3 MLLM_RETRY_DELAY=30 EMBED_RETRY_TIMES=5 EMBED_RETRY_DELAY=30
export MAX_STEPS=16 TASK_TIMEOUT_SEC=1000 CONCURRENCY=1 AGENT_FORCE_LAST_STEP_VISUAL_INSPECT=1 AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK=1 AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK=1
export AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE=mcq AGENT_PARSE_FAILURE_FALLBACK_TO_C=0

started=$(date -u +%FT%TZ)
printf '%s\n' "${started}" >"${STATUS}/started_utc.txt"
printf '%s\n' "dense_beam_b_h15_eval300_formal_v1" >"${STATUS}/requested_tmux_name.txt"

# Immutable prestart gates. No secrets are printed or persisted.
test "$(sha256sum "${OVERLAY}/videoseal/tools/retrieval_adapter.py" | awk '{print $1}')" = 008efa173913759a7713209b649ab3947e99349a2385afd017995778460fc4b0
test "$(sha256sum "${PARITY}/RUNTIME_OVERLAY_MANIFEST.sha256" | awk '{print $1}')" = 17df48f72eef23d422b03dfe4e662e5c7a4b02247b97d8a8d024ff664af84934
test "$(sha256sum "${PARITY}/MANIFEST.sha256" | awk '{print $1}')" = 40f1417e214e21d40188bde45558ca0707adc55586abe5a13842f32ad2cf2340
test "$(sha256sum "${UIDS_FILE}" | awk '{print $1}')" = "${UIDS_SHA256}"
(cd "${PARITY}" && sha256sum -c RUNTIME_OVERLAY_MANIFEST.sha256 >/dev/null && sha256sum -c MANIFEST.sha256 >/dev/null)
"${RUNNER_PYTHON}" "${ORCH}/state_tool.py" preflight --uids "${UIDS_FILE}" --expected-sha "${UIDS_SHA256}" --index "${HIERARCHICAL_INDEX}" --parquet "${PARQUET}" >"${STATUS}/preflight.json"
test "${RETRIEVAL_BACKEND}" = dense_semantic_beam_b
test "${DENSE_BEAM_B}" = 15
test "${CONCURRENCY}" = 1
if ss -ltn | grep -Eq ":(${PLANNER_PORT}|${INSPECTOR_PORT})[[:space:]]"; then echo "formal ports occupied" >&2; exit 10; fi

"${RUNNER_PYTHON}" - "${STATUS}/experiment_config.json" <<'PY'
import json,os,pathlib,sys
keys=['RETRIEVAL_BACKEND','DENSE_BEAM_B','HIERARCHICAL_PROFILE','CONCURRENCY','MAX_STEPS','TASK_TIMEOUT_SEC','AGENT_API_USE_MESSAGES','AGENT_LLM_MODEL','VISUAL_INSPECT_MODEL','VISUAL_RETRIEVE_SUM_MODEL','AGENT_LLM_TEMPERATURE','VISUAL_RETRIEVE_SUM_TEMPERATURE','AGENT_PARSE_FAILURE_FALLBACK_TO_C']
d={k:os.environ.get(k) for k in keys}; d.update({'uid_count':300,'uids_sha256':os.environ['UIDS_SHA256'],'overlay':'/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z/runtime_overlay','tools_schema_sha256':'71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863','planner_model':os.environ['PLANNER_MODEL'],'inspector_model':os.environ['INSPECTOR_MODEL'],'runtime':os.environ['RUNTIME']})
pathlib.Path(sys.argv[1]).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY

planner_pid='' inspector_pid='' memory_pid=''
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

stamp=$(date -u +%Y%m%dT%H%M%SZ)
planner_log=${LOGS}/planner-${stamp}.log
inspector_log=${LOGS}/inspector-${stamp}.log
memory_log=${LOGS}/memory-${stamp}.log
setsid "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server --model "${PLANNER_MODEL}" --served-model-name "${PLANNER_SERVED_NAME}" --host "${PLANNER_HOST}" --port "${PLANNER_PORT}" --dtype bfloat16 --max-model-len "${PLANNER_MAX_MODEL_LEN}" --gpu-memory-utilization "${PLANNER_GPU_MEMORY_UTIL}" --max-num-seqs 1 --enforce-eager --no-enable-log-requests --generation-config vllm >"${planner_log}" 2>&1 & planner_pid=$!
printf '%s\n' "${planner_pid}" >"${STATUS}/planner.pid"
for _ in $(seq 1 450); do curl -fsS "http://${PLANNER_HOST}:${PLANNER_PORT}/v1/models" >/dev/null && break; kill -0 "${planner_pid}" 2>/dev/null || exit 11; sleep 2; done
curl -fsS "http://${PLANNER_HOST}:${PLANNER_PORT}/v1/models" >/dev/null || exit 12
printf '%s\n' "$(date -u +%FT%TZ)" >"${STATUS}/planner_healthy_utc.txt"

setsid "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server --model "${INSPECTOR_MODEL}" --served-model-name "${INSPECTOR_SERVED_NAME}" --host "${INSPECTOR_HOST}" --port "${INSPECTOR_PORT}" --dtype bfloat16 --max-model-len "${INSPECTOR_MAX_MODEL_LEN}" --gpu-memory-utilization "${INSPECTOR_GPU_MEMORY_UTIL}" --max-num-seqs 1 --limit-mm-per-prompt '{"image":64}' --enforce-eager --no-enable-log-requests --generation-config vllm >"${inspector_log}" 2>&1 & inspector_pid=$!
printf '%s\n' "${inspector_pid}" >"${STATUS}/inspector.pid"
for _ in $(seq 1 450); do curl -fsS "http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1/models" >/dev/null && break; kill -0 "${inspector_pid}" 2>/dev/null || exit 13; sleep 2; done
curl -fsS "http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1/models" >/dev/null || exit 14
printf '%s\n' "$(date -u +%FT%TZ)" >"${STATUS}/inspector_healthy_utc.txt"

(while :; do printf '%s ' "$(date -u +%FT%TZ)"; awk '/MemAvailable|SwapFree/{printf "%s=%s%s ",$1,$2,$3} END{print ""}' /proc/meminfo; sleep 10; done) >"${memory_log}" 2>&1 & memory_pid=$!

run_phase() {
  local kind=$1 phase=$2 source=${3:-}
  local list=${STATUS}/${kind}-pending-${stamp}.txt log=${LOGS}/${kind}-runner-${stamp}.log
  local args=(pending --uids "${UIDS_FILE}" --phase "${phase}" --kind "${kind}" --output "${list}")
  [[ -z "${source}" ]] || args+=(--source "${source}")
  "${RUNNER_PYTHON}" "${ORCH}/state_tool.py" "${args[@]}" | tee -a "${STATUS}/orchestrator.jsonl"
  local count; count=$(awk 'NF{n++} END{print n+0}' "${list}")
  [[ "${count}" -gt 0 ]] || return 0
  /usr/bin/time -p "${RUNNER_PYTHON}" -m videoseal.runner.per_question_runner --parquet "${PARQUET}" --uids-file "${list}" --save-runs "${phase}" --concurrency 1 --max-steps 16 --task-timeout-sec 1000 >"${log}" 2>&1
  printf 'runner_exit_code=0\n' >>"${log}"
  curl -fsS "http://${PLANNER_HOST}:${PLANNER_PORT}/v1/models" >/dev/null || exit 21
  curl -fsS "http://${INSPECTOR_HOST}:${INSPECTOR_PORT}/v1/models" >/dev/null || exit 22
  if grep -Eqi 'CUDA error|CUDA out of memory|out of memory|EngineCore.*died|ModuleNotFoundError|ImportError|No such file.*(index|model|video)' "${log}" "${planner_log}" "${inspector_log}"; then exit 23; fi
  "${RUNNER_PYTHON}" "${ORCH}/state_tool.py" infra-gate --uids "${list}" --phase "${phase}" --threshold 3 | tee -a "${STATUS}/orchestrator.jsonl"
}

run_phase first "${FIRST}"
"${RUNNER_PYTHON}" "${ORCH}/state_tool.py" coverage --uids "${UIDS_FILE}" --phase "${FIRST}" --require-attempted 300 | tee -a "${STATUS}/orchestrator.jsonl"
run_phase retry "${RETRY}" "${FIRST}"
retry_expected=$("${RUNNER_PYTHON}" "${ORCH}/state_tool.py" pending --uids "${UIDS_FILE}" --phase /dev/null/not-used --kind retry --source "${FIRST}" --output "${STATUS}/retry-all-${stamp}.txt" | "${RUNNER_PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["pending"])')
"${RUNNER_PYTHON}" "${ORCH}/state_tool.py" coverage --uids "${STATUS}/retry-all-${stamp}.txt" --phase "${RETRY}" --require-attempted "${retry_expected}" | tee -a "${STATUS}/orchestrator.jsonl"
"${RUNNER_PYTHON}" "${ORCH}/finalize_profile.py" --profile h15 --uids "${UIDS_FILE}" --first "${FIRST}" --retry "${RETRY}" --flat "${FLAT_REFERENCE}" --profile-root "${PROFILE_ROOT}" | tee -a "${STATUS}/orchestrator.jsonl"
printf '%s\n' "$(date -u +%FT%TZ)" >"${STATUS}/finished_utc.txt"

stop_services
trap - EXIT INT TERM
if ss -ltn | grep -Eq ":(${PLANNER_PORT}|${INSPECTOR_PORT})[[:space:]]"; then exit 30; fi
