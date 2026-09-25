#!/usr/bin/env bash
set -euo pipefail

: "${RETRY_V2_ROOT:?}"
: "${RETRY_V2_CONTROL:?}"
: "${RETRY_V2_TMUX_SESSION:?}"
controller=/home/naxucl/data/HourVideo/videoseal_original/videoseal_flat15_eval300_formal_v1_20260910T184300Z/retry_recovery_execution_20260913T123524Z/retry_controller_v3.sh

tmux has-session -t "${RETRY_V2_TMUX_SESSION}" 2>/dev/null && { echo "tmux session already exists" >&2; exit 72; }
mkdir -p "${RETRY_V2_CONTROL}"
tmux new-session -d -s "${RETRY_V2_TMUX_SESSION}" \
  "exec env RETRY_V2_ROOT='${RETRY_V2_ROOT}' RETRY_V2_CONTROL='${RETRY_V2_CONTROL}' bash '${controller}' >>'${RETRY_V2_CONTROL}/controller.log' 2>&1"
tmux has-session -t "${RETRY_V2_TMUX_SESSION}"
tmux list-panes -t "${RETRY_V2_TMUX_SESSION}" -F '#{pane_pid}' > "${RETRY_V2_CONTROL}/tmux_pane_pid.txt"
