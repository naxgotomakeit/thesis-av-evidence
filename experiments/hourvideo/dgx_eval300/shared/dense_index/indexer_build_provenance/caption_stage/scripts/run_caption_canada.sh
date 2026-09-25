#!/usr/bin/env bash
set -euo pipefail

experiment_dir=/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/main_system/thesis-av-evidence-hourvideo/hourvideo_v6_1_runtime/src/experiments/hourvideo_r3_keyframe_caption_3frame_action_preserving_v1
python_bin=/cs/student/project_msc/2025/rai/xinanx01/miniconda3/envs/thesis_av/bin/python

cd "$experiment_dir"
export CUDA_VISIBLE_DEVICES=0
"$python_bin" pipeline.py caption 2>&1 | tee caption_console.log
