#!/usr/bin/env bash
set -Eeuo pipefail

RETRY_SESSION=hourvideo_eval300_timeout_retry
ROOT=/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z
LOG=${ROOT}/logs/finalizer.log

exec > >(tee -a "${LOG}") 2>&1
printf 'finalizer_wait_started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while tmux has-session -t "${RETRY_SESSION}" 2>/dev/null; do
  sleep 60
done
printf 'retry_session_ended_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec /home/naxucl/projects/VideoSEAL/.venv/bin/python "${ROOT}/finalize_retry_and_bundle.py"
