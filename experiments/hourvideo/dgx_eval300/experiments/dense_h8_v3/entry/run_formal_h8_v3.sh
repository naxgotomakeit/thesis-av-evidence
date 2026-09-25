#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1
ORCH=${BASE}/formal_eval300_orchestrator
PARITY=${BASE}/outputs/dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z
H8_OVERLAY_ROOT=${BASE}/outputs/dense_semantic_beam_b_h8_budget_overlay_v3_20260831T201510Z
AMENDMENT=${BASE}/outputs/dense_semantic_beam_b_h8_superseding_amendment_v3_20260831T201510Z
OVERLAY=${H8_OVERLAY_ROOT}/runtime_overlay
PROFILE_ROOT=${HERE}/h8
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
export RETRIEVAL_BACKEND=dense_semantic_beam_b DENSE_BEAM_B=8 DENSE_BEAM_SIGLIP_MODEL=${SIGLIP_MODEL}
export HIERARCHICAL_PROFILE=dense_beam_b_h8 PAIRED_HIERARCHICAL_ROOT=${HIERARCHICAL_INDEX} INDEX_ROOT=${HIERARCHICAL_INDEX}
export INSPECT_MAX_LONG_EDGE=1280 INSPECT_VLM_MAX_TOKENS=4096 INSPECT_VLM_TEMPERATURE=0.1 INSPECT_MAX_TOTAL_IMAGES=64 INSPECT_FPS=2
export VISUAL_INSPECT_MAX_LONG_EDGE=1280 VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE=1 VISUAL_INSPECT_TOTAL_PIXELS=20971520 VISUAL_INSPECT_MIN_PIXELS=12544 VISUAL_INSPECT_EDGE_MULTIPLE=32
export MLLM_BACKEND=openai MLLM_MAX_TOKENS=4096 MLLM_TIMEOUT=${HTTP_TIMEOUT_SEC} MLLM_RETRY_TIMES=3 MLLM_RETRY_DELAY=30 EMBED_RETRY_TIMES=5 EMBED_RETRY_DELAY=30
export MAX_STEPS=16 TASK_TIMEOUT_SEC=1000 CONCURRENCY=1 AGENT_FORCE_LAST_STEP_VISUAL_INSPECT=1 AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK=1 AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK=1
export AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE=mcq AGENT_PARSE_FAILURE_FALLBACK_TO_C=0

started=$(date -u +%FT%TZ)
printf '%s\n' "${started}" >"${STATUS}/started_utc.txt"
printf '%s\n' "dense_beam_b_h8_eval300_formal_v3" >"${STATUS}/requested_tmux_name.txt"

# Immutable prestart gates. No secrets are printed or persisted.
test "$(sha256sum "${OVERLAY}/videoseal/tools/retrieval_adapter.py" | awk '{print $1}')" = 477e331b0636ced674a3db7861181d7d3038eb55027213e66a9b818b05ff8264
test "$(sha256sum "${OVERLAY}/videoseal/tools/dense_beam_b/retriever.py" | awk '{print $1}')" = 47b8dcfea459f9edacf650e9975043714635c8c3b5fd054f78715535f99cd46e
test "$(sha256sum "${PARITY}/RUNTIME_OVERLAY_MANIFEST.sha256" | awk '{print $1}')" = 17df48f72eef23d422b03dfe4e662e5c7a4b02247b97d8a8d024ff664af84934
test "$(sha256sum "${PARITY}/MANIFEST.sha256" | awk '{print $1}')" = 40f1417e214e21d40188bde45558ca0707adc55586abe5a13842f32ad2cf2340
test "$(sha256sum "${H8_OVERLAY_ROOT}/MANIFEST.sha256" | awk '{print $1}')" = 4507fd73e8643c217b26760768a1eab3fd10015d978bbf1d9a29d42c12abae60
test "$(sha256sum "${AMENDMENT}/MANIFEST.sha256" | awk '{print $1}')" = 7f691518f81036e432c23582b680d2133f716c72e0c18803d6470052e6f92e2c
test "$(sha256sum "${BASE}/outputs/dense_semantic_beam_b_h30_h8_superseding_audit_v2_20260829T155639Z/MANIFEST.sha256" | awk '{print $1}')" = ba34712e6b97e5501b18fb21f099e27b3732d1765d810413b56af8c70cd72e7f
test "$(sha256sum "${UIDS_FILE}" | awk '{print $1}')" = "${UIDS_SHA256}"
(cd "${PARITY}" && sha256sum -c MANIFEST.sha256 >/dev/null)
(cd "${H8_OVERLAY_ROOT}" && sha256sum -c MANIFEST.sha256 >/dev/null)
(cd "${AMENDMENT}" && sha256sum -c MANIFEST.sha256 >/dev/null)
"${RUNNER_PYTHON}" "${ORCH}/state_tool.py" preflight --uids "${UIDS_FILE}" --expected-sha "${UIDS_SHA256}" --index "${HIERARCHICAL_INDEX}" --parquet "${PARQUET}" >"${STATUS}/preflight.json"
test "${RETRIEVAL_BACKEND}" = dense_semantic_beam_b
test "${DENSE_BEAM_B}" = 8
test "${CONCURRENCY}" = 1
"${RUNNER_PYTHON}" -c 'from videoseal.tools.dense_beam_b.providers import runtime_gate; runtime_gate()'
"${RUNNER_PYTHON}" - "${STATUS}/actual_import_gate.json" <<'PY'
import hashlib,json,pathlib,sys
import videoseal.tools.retrieval_adapter as adapter
import videoseal.tools.dense_beam_b.providers as providers
import videoseal.tools.dense_beam_b.retriever as retriever
from videoseal.tools.tool_map import build_tool_map
T=build_tool_map(); tool=T['visual_retrieve']()
raw=json.dumps([tool.json,T['visual_inspect']().json])
files={'retrieval_adapter':adapter.__file__,'providers':providers.__file__,'retriever':retriever.__file__}
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out={'files':files,'sha256':{k:sha(v) for k,v in files.items()},'backend':'dense_semantic_beam_b','b':tool.b,'tools_schema_sha256':hashlib.sha256(raw.encode()).hexdigest()}
assert out['sha256']['retrieval_adapter']=='477e331b0636ced674a3db7861181d7d3038eb55027213e66a9b818b05ff8264'
assert out['sha256']['retriever']=='47b8dcfea459f9edacf650e9975043714635c8c3b5fd054f78715535f99cd46e'
assert out['b']==8 and out['tools_schema_sha256']=='71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863'
pathlib.Path(sys.argv[1]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
if ss -ltn | grep -Eq ":(${PLANNER_PORT}|${INSPECTOR_PORT})[[:space:]]"; then echo "formal ports occupied" >&2; exit 10; fi

"${RUNNER_PYTHON}" - "${STATUS}/experiment_config.json" <<'PY'
import json,os,pathlib,sys
keys=['RETRIEVAL_BACKEND','DENSE_BEAM_B','HIERARCHICAL_PROFILE','CONCURRENCY','MAX_STEPS','TASK_TIMEOUT_SEC','AGENT_API_USE_MESSAGES','AGENT_LLM_MODEL','VISUAL_INSPECT_MODEL','VISUAL_RETRIEVE_SUM_MODEL','AGENT_LLM_TEMPERATURE','VISUAL_RETRIEVE_SUM_TEMPERATURE','AGENT_PARSE_FAILURE_FALLBACK_TO_C']
d={k:os.environ.get(k) for k in keys}; d.update({'uid_count':300,'uids_sha256':os.environ['UIDS_SHA256'],'overlay':'/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h8_budget_overlay_v3_20260831T201510Z/runtime_overlay','tools_schema_sha256':'71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863','planner_model':os.environ['PLANNER_MODEL'],'inspector_model':os.environ['INSPECTOR_MODEL'],'runtime':os.environ['RUNTIME']})
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

gate_uid=6fd90f8d-7a4d-425d-a812-3268db0b0342_14_5
gate_list=${STATUS}/launch-gate-first-uid.txt
printf '%s\n' "${gate_uid}" >"${gate_list}"
/usr/bin/time -p "${RUNNER_PYTHON}" -m videoseal.runner.per_question_runner --parquet "${PARQUET}" --uids-file "${gate_list}" --save-runs "${FIRST}" --concurrency 1 --max-steps 16 --task-timeout-sec 1000 >"${LOGS}/launch-gate-${stamp}.log" 2>&1
gate_traj=$(find "${FIRST}/6fd90f8d-7a4d-425d-a812-3268db0b0342" -mindepth 2 -maxdepth 2 -name trajectory.json -type f | head -1)
"${RUNNER_PYTHON}" - "${gate_traj}" "${STATUS}/launch_gate.json" <<'PY'
import json,pathlib,sys
d=json.load(open(sys.argv[1])); retrieval=[]; actions=[]
for step in d.get('steps') or []:
    a=step.get('action') or {}; actions.append(a.get('name'))
    if a.get('name')=='visual_retrieve': retrieval.append(((step.get('observation') or {}).get('metadata') or {}))
assert d.get('uid')=='6fd90f8d-7a4d-425d-a812-3268db0b0342_14_5' and d.get('finished_at')
assert retrieval and 'visual_inspect' in actions
for m in retrieval:
    assert m.get('retrieval_backend')=='dense_semantic_beam_b'
    assert m.get('b')==8 and m.get('hierarchy_complete') is True and m.get('hierarchy_used') is True
    assert m.get('medium_gate_applied') is True and int(m.get('selected_coarse'))<=8 and int(m.get('selected_medium'))<=8 and int(m.get('returned_segments'))<=8
    assert int(m.get('unique_1fps_frames_scored') or 0)>0
    assert m.get('text_embedding_calls')==1 and m.get('siglip_query_encode_calls')==1
    assert m.get('global_fine_fallback') is False and m.get('caption_score_fallback') is False and m.get('representative_frame_fallback') is False
out={'status':'PASS','uid':d['uid'],'trajectory':sys.argv[1],'actions':actions,'retrieval_calls':len(retrieval),'first_retrieval':{k:retrieval[0].get(k) for k in ('retrieval_backend','b','hierarchy_complete','hierarchy_used','total_coarse','selected_coarse','medium_candidates','selected_medium','eligible_flat_segments','unique_1fps_frames_scored','returned_segments','medium_gate_applied','text_embedding_calls','siglip_query_encode_calls')}}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
printf '%s\n' FORMAL_RUNNING >"${STATUS}/state.txt"
run_phase first "${FIRST}"
"${RUNNER_PYTHON}" "${ORCH}/state_tool.py" coverage --uids "${UIDS_FILE}" --phase "${FIRST}" --require-attempted 300 | tee -a "${STATUS}/orchestrator.jsonl"
run_phase retry "${RETRY}" "${FIRST}"
retry_expected=$("${RUNNER_PYTHON}" "${ORCH}/state_tool.py" pending --uids "${UIDS_FILE}" --phase /dev/null/not-used --kind retry --source "${FIRST}" --output "${STATUS}/retry-all-${stamp}.txt" | "${RUNNER_PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["pending"])')
"${RUNNER_PYTHON}" "${ORCH}/state_tool.py" coverage --uids "${STATUS}/retry-all-${stamp}.txt" --phase "${RETRY}" --require-attempted "${retry_expected}" | tee -a "${STATUS}/orchestrator.jsonl"
"${RUNNER_PYTHON}" "${ORCH}/finalize_profile.py" --profile h8 --uids "${UIDS_FILE}" --first "${FIRST}" --retry "${RETRY}" --flat "${FLAT_REFERENCE}" --profile-root "${PROFILE_ROOT}" | tee -a "${STATUS}/orchestrator.jsonl"
printf '%s\n' "$(date -u +%FT%TZ)" >"${STATUS}/finished_utc.txt"

stop_services
trap - EXIT INT TERM
if ss -ltn | grep -Eq ":(${PLANNER_PORT}|${INSPECTOR_PORT})[[:space:]]"; then exit 30; fi
