#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${HERE}/runtime_overlay_v2"
ENV_FILE="${HERE}/flat15.env"
UIDS="/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt"
PARQUET="/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_dev_v1.0_videoseal_dgx.parquet"
PYTHON="/home/naxucl/projects/VideoSEAL/.venv/bin/python"

# Launch is intentionally impossible without an explicit, new output root.
: "${FLAT15_OUTPUT_ROOT:?Set FLAT15_OUTPUT_ROOT to a new absent formal output directory}"
[[ ! -e "${FLAT15_OUTPUT_ROOT}" ]] || { echo "Refusing existing output: ${FLAT15_OUTPUT_ROOT}" >&2; exit 20; }
[[ "${RUN_MODE:-}" == "first" ]] || { echo "This entry point is first-pass only; use a separately frozen pending UID list for resume/retry." >&2; exit 21; }

set -a
source "${ENV_FILE}"
set +a
[[ "${VISUAL_RETRIEVE_TOPK}" == 15 ]] || { echo "Flat-15 budget gate failed" >&2; exit 22; }
[[ "${CONCURRENCY}" == 1 ]] || { echo "Concurrency gate failed" >&2; exit 23; }

export CODE_ROOT="${RUNTIME}"
export DATA_ROOT=/home/naxucl/data/HourVideo
export MODEL_ROOT=/home/naxucl/models
export RUN_ROOT=/home/naxucl/data/HourVideo/videoseal_original
export LOG_ROOT="${FLAT15_OUTPUT_ROOT}.logs"
export PARQUET UIDS_FILE="${UIDS}" EVAL_RUN_ROOT="${FLAT15_OUTPUT_ROOT}"
export PYTHONPATH="${RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON

# This invokes the accepted Flat-30 service/runner entry point. It is not run by preflight.
exec "${RUNTIME}/scripts/dgx/run_eval300.sh"
