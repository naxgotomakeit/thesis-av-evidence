#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
unset OPENAI_API_KEY ANTHROPIC_API_KEY AZURE_OPENAI_API_KEY GOOGLE_API_KEY || true

root=/myriadfs/home/ucemxna/Scratch/workspace/external_baselines/gens_qwen25vl3b_top16_v1
formal_root="$root/outputs/formal_eval300_gens_hybrid_symmetric_mcq_cap16_v2_long_output_20260822T214500Z"
python=/myriadfs/home/ucemxna/Scratch/workspace/envs/videoseal/bin/python3.12
export PYTHONPATH="$root/wrapper"

exec "$python" "$root/wrapper/formal_eval300_selector_v2.py" controller \
  --formal-root "$formal_root" >>"$formal_root/logs/controller.log" 2>&1
