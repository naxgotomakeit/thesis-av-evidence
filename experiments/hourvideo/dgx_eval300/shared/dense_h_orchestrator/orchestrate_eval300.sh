#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=formal.env
set -a
# shellcheck disable=SC1091
source "${HERE}/formal.env"
set +a
MODE=${1:---dry-run}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME}/runnable_runtime:${RUNTIME}/runtime_support:${RUNTIME}/runtime_support/src"

preflight() {
  [[ -x "${RUNNER_PYTHON}" && -x "${VLLM_PYTHON}" ]] || { echo "missing interpreter" >&2; exit 30; }
  for path in "${RUNTIME}" "${HIERARCHICAL_INDEX}" "${SIGLIP_MODEL}" "${PLANNER_MODEL}/config.json" "${INSPECTOR_MODEL}/config.json" "${PARQUET}" "${UIDS_FILE}"; do [[ -e "${path}" ]] || { echo "missing path ${path}" >&2; exit 31; }; done
  "${RUNNER_PYTHON}" "${HERE}/state_tool.py" preflight --uids "${UIDS_FILE}" --expected-sha "${UIDS_SHA256}" --index "${HIERARCHICAL_INDEX}" --parquet "${PARQUET}"
}

if [[ "${MODE}" == --dry-run ]]; then
  preflight
  echo "DRY RUN ONLY: no model/API/runner invocation"
  echo "order=h6:first_pass,h6:retry_1,h6:finalize,h15:first_pass,h15:retry_1,h15:finalize,h30:first_pass,h30:retry_1,h30:finalize,combined_summary"
  for p in h6 h15 h30; do
    echo "${p}: FORMAL_ROOT=<absolute-root> ${HERE}/run_profile.sh ${p}"
    echo "${p}: first_pass=300 frozen UIDs; retry_1=only incomplete timeout/error UIDs; concurrency=1"
  done
  exit 0
fi
[[ "${MODE}" == --run ]] || { echo "usage: $0 [--dry-run|--run]" >&2; exit 2; }
: "${FORMAL_ROOT:?set explicit FORMAL_ROOT under ${OUTPUTS_PARENT}}"
case "${FORMAL_ROOT}" in "${OUTPUTS_PARENT}"/formal_eval300_*) ;; *) echo "unsafe FORMAL_ROOT=${FORMAL_ROOT}" >&2; exit 3;; esac
mkdir -p "${FORMAL_ROOT}/status"
exec 9>"${FORMAL_ROOT}/status/orchestrator.lock"
flock -n 9 || { echo "another orchestrator holds ${FORMAL_ROOT}" >&2; exit 4; }
preflight | tee -a "${FORMAL_ROOT}/status/preflight.json"

manifest=${FORMAL_ROOT}/status/frozen_manifest.sha256
tmp=${FORMAL_ROOT}/status/.frozen_manifest.sha256.tmp
sha256sum "${UIDS_FILE}" "${PARQUET}" "${RUNTIME}/MANIFEST.json" "${RUNTIME}/CONTENTS.sha256" "${RUNTIME}/experiment/config_v7_4.json" "${RUNTIME}/runnable_runtime/videoseal/agents/tool_agent.py" "${RUNTIME}/runnable_runtime/videoseal/tools/retrieval_adapter.py" "${RUNTIME}/runnable_runtime/videoseal/tools/visual_tools.py" "${HIERARCHICAL_INDEX}/manifests/materialization_manifest.json" "${HERE}/formal.env" "${HERE}/state_tool.py" "${HERE}/finalize_profile.py" "${HERE}/run_profile.sh" "${HERE}/orchestrate_eval300.sh" >"${tmp}"
if [[ -e "${manifest}" ]]; then cmp -s "${manifest}" "${tmp}" || { echo "configuration/code/index hash drift" >&2; exit 5; }; rm "${tmp}"; else mv "${tmp}" "${manifest}"; fi

for profile in h6 h15 h30; do
  preflight >/dev/null
  if [[ -f "${FORMAL_ROOT}/${profile}/status/final_report.json" ]]; then
    "${RUNNER_PYTHON}" "${HERE}/finalize_profile.py" --profile "${profile}" --uids "${UIDS_FILE}" --first "${FORMAL_ROOT}/${profile}/first_pass" --retry "${FORMAL_ROOT}/${profile}/retry_1" --flat "${FLAT_REFERENCE}" --profile-root "${FORMAL_ROOT}/${profile}" >/dev/null
    echo "resume: verified and skipped finalized profile ${profile}"
    continue
  fi
  FORMAL_ROOT=${FORMAL_ROOT} "${HERE}/run_profile.sh" "${profile}"
  sha256sum -c "${manifest}" >/dev/null || { echo "hash drift after ${profile}" >&2; exit 6; }
  pgrep -f 'videoseal.runner.per_question_runner' >/dev/null && { echo "duplicate/residual runner before next profile" >&2; exit 7; }
done

"${RUNNER_PYTHON}" - "${FORMAL_ROOT}" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]); rows={p:json.loads((root/p/'status/final_report.json').read_text()) for p in ('h6','h15','h30')}
target=root/'status/three_profile_summary.json'; text=json.dumps({'profiles':rows,'flat_reference':'/home/naxucl/data/HourVideo/videoseal_original/runs_dgx_eval300_v1'},indent=2,sort_keys=True)+'\n'
if target.exists() and target.read_text()!=text: raise SystemExit('combined summary drift')
target.write_text(text)
PY
