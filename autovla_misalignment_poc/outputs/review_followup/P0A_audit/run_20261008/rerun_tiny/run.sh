#!/bin/bash
set -uo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
R=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P0A_audit/run_20261008/rerun_tiny
S=/root/VLA/autovla_misalignment_poc/scripts
nvidia-smi --query-gpu=index,name,memory.used --format=csv > "$R/gpu_before.txt"
date -u > "$R/exp9.start"
timeout 1200 nice -n 19 python "$S/temporal_feedback_window.py" --limit 1 --max-alts 2 --output "$R/exp9" > "$R/exp9.log" 2>&1
echo "exit $?" >> "$R/exp9.log"
date -u > "$R/exp8.start"
timeout 1200 nice -n 19 python "$S/prev_action_state_patching.py" --limit 1 --max-alts 2 --debug --output "$R/exp8" > "$R/exp8.log" 2>&1
echo "exit $?" >> "$R/exp8.log"
date -u > "$R/done"
