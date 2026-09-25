#!/usr/bin/env bash
set -Eeuo pipefail

RETRY_ROOT=/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z
CODE_ROOT=/home/naxucl/projects/VideoSEAL
DGX_SCRIPTS=/home/naxucl/projects/VideoSEAL/scripts/dgx
ENV_FILE=/home/naxucl/.env.embedding
LOG_DIR=${RETRY_ROOT}/logs
MANIFEST_DIR=${RETRY_ROOT}/manifests
CHECKER=${RETRY_ROOT}/check_retry_group.py
LAUNCH_LOG=${LOG_DIR}/launcher.log
EXPECTED_UID_SHA=6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1
FROZEN_UIDS=/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt
ORIGINAL_MODEL=/home/naxucl/models/Qwen3-8B
TRAINED_MODEL=/home/naxucl/models/VideoSEAL_8B

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf 'launcher_started_at_utc=%s\nhostname=%s\npid=%s\n' "${started_at}" "$(hostname -f)" "$$"
printf 'retry_root=%s\ncode_root=%s\nenv_file=%s\n' "${RETRY_ROOT}" "${CODE_ROOT}" "${ENV_FILE}"
printf 'semantic_config=max_steps:16,task_timeout_sec:1000,http_timeout_sec:300,concurrency:1,max_num_seqs:1\n'

finish() {
  rc=$?
  printf 'launcher_finished_at_utc=%s\nlauncher_exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rc}"
}
trap finish EXIT

[[ -f "${ENV_FILE}" ]] || { echo "Missing environment file: ${ENV_FILE}" >&2; exit 10; }
[[ "$(stat -c %a "${ENV_FILE}")" == 600 ]] || { echo "Environment file must have mode 600: ${ENV_FILE}" >&2; exit 11; }
[[ "$(sha256sum "${FROZEN_UIDS}" | awk '{print $1}')" == "${EXPECTED_UID_SHA}" ]] || { echo 'Frozen Eval300 UID SHA mismatch.' >&2; exit 12; }
[[ -x /home/naxucl/projects/VideoSEAL/.venv/bin/python ]] || { echo 'Runner Python missing.' >&2; exit 13; }
[[ -x /home/naxucl/envs/vllm-gb10/bin/python ]] || { echo 'vLLM Python missing.' >&2; exit 14; }
if ss -ltn | grep -Eq ':(18082|18083)[[:space:]]'; then
  echo 'Planner or Visual port already in use.' >&2
  exit 15
fi

# Load credentials only inside this launcher. Values are never echoed.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

run_phase() {
  local group=$1
  local phase=$2
  local model=$3
  local served=$4
  local uid_file=$5
  local output=$6
  local mode=$7
  local phase_log=${LOG_DIR}/${group}-${phase}-$(date -u +%Y%m%dT%H%M%SZ).log
  printf 'phase_start group=%s phase=%s uids=%s count=%s output=%s log=%s\n' \
    "${group}" "${phase}" "${uid_file}" "$(wc -l < "${uid_file}")" "${output}" "${phase_log}"
  PLANNER_MODEL="${model}" \
  PLANNER_SERVED_NAME="${served}" \
  RUN_MODE="${mode}" \
  EVAL_NAME="eval300-timeout-retry-${group}-${phase}" \
  EVAL_RUN_ROOT="${output}" \
  EXPERIMENT_NAME="hourvideo_eval300_timeout_retry_${group}" \
  UIDS_FILE="${uid_file}" \
  AGENT_LLM_TIMEOUT=300 \
  CONCURRENCY=1 \
    "${DGX_SCRIPTS}/run_eval300.sh" 2>&1 | tee -a "${phase_log}"
  local rc=${PIPESTATUS[0]}
  printf 'phase_end group=%s phase=%s rc=%s\n' "${group}" "${phase}" "${rc}"
  return "${rc}"
}

run_group() {
  local group=$1
  local model=$2
  local served=$3
  local output=${RETRY_ROOT}/${group}_retry
  local validation=${MANIFEST_DIR}/${group}_validation_uids.txt
  local batch=${MANIFEST_DIR}/${group}_batch_uids.txt

  [[ ! -e "${output}" ]] || { echo "Refusing existing retry output: ${output}" >&2; return 20; }
  run_phase "${group}" validation "${model}" "${served}" "${validation}" "${output}" first
  "${CHECKER}" --uids "${validation}" --run-root "${output}" \
    --report "${RETRY_ROOT}/${group}_validation_report.json" --require-all
  run_phase "${group}" batch "${model}" "${served}" "${batch}" "${output}" resume
  "${CHECKER}" --uids "${MANIFEST_DIR}/${group}_timeout_uids.txt" --run-root "${output}" \
    --report "${RETRY_ROOT}/${group}_final_retry_report.json"
}

cd "${CODE_ROOT}"
run_group original "${ORIGINAL_MODEL}" qwen3-8b-planner-dgx
run_group trained "${TRAINED_MODEL}" videoseal-8b-planner-dgx
printf 'all_retry_groups_finished_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
