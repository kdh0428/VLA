#!/bin/bash
# Stage-0 control-5 pilot: two separate processes (run1, run2), pilot units, all four paths, greedy + T=0.01.
# RTX 5090 only; waits for >= 14 GB free GPU memory before each model load; runs strictly sequentially.
set -euo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup3/p1
O=/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/pilot_c5_20261014
MAN=/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/stage0_manifest.json
[[ "${O:?}" == /root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/* ]] || exit 2
mkdir -p "$O"
wait_gpu() {
  for i in $(seq 1 180); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    if [ "$free" -ge 14000 ]; then echo "$(date -u +%FT%TZ) gpu1 free ${free} MiB" >> "$O/gpu_checks.log"; return 0; fi
    echo "$(date -u +%FT%TZ) waiting: gpu1 free ${free} MiB" >> "$O/gpu_checks.log"; sleep 20
  done
  echo "GPU never had 14 GB free" >> "$O/gpu_checks.log"; exit 3
}
for R in run1 run2; do
  wait_gpu
  nice -n 10 python "$S/p1b_harness.py" --manifest "$MAN" --out "$O/$R" --controls c5 --pilot-only \
     --paths H7,H8,H9,P1A --decodings greedy,t001 > "$O/$R.log" 2>&1
done
python "$S/p1b_compare_runs.py" "$O/run1" "$O/run2" "$O/c5b_cross_process.jsonl" >> "$O/run2.log" 2>&1
date -u > "$O/done"
