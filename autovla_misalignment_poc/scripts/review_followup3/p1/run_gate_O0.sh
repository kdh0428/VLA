#!/bin/bash
# O0 extension (AMENDMENT_P1_03) of the Stage-0 AutoVLA gate per PROTOCOL_P1.md §9.2. RTX 5090 only; >= 14 GB free before each load; sequential.
set -euo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/root/VLA/autovla_misalignment_poc
S=$P/scripts/review_followup3/p1
O=$P/outputs/review_followup3/P1/gate_O0
MAN=$P/outputs/review_followup3/P1/stage0_manifest_O0.json
TOL=$P/outputs/review_followup3/P1/pilot_c5_20261014/tolerances.json
[[ "${O:?}" == /root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/* ]] || exit 2
mkdir -p "$O"
wait_gpu() {
  for i in $(seq 1 180); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    if [ "$free" -ge 14000 ]; then echo "$(date -u +%FT%TZ) gpu1 free ${free} MiB" >> "$O/gpu_checks.log"; return 0; fi
    echo "$(date -u +%FT%TZ) waiting: gpu1 free ${free} MiB" >> "$O/gpu_checks.log"; sleep 20
  done
  exit 3
}
wait_gpu
nice -n 10 python "$S/p1b_harness.py" --manifest "$MAN" --out "$O/run1" --controls c1,c2,c3,c4,c5 --paths H32,P1B --decodings greedy,t001 > "$O/run1.log" 2>&1
wait_gpu
nice -n 10 python "$S/p1b_harness.py" --manifest "$MAN" --out "$O/run2" --controls c5 --paths H32,P1B --decodings greedy,t001 > "$O/run2.log" 2>&1
python "$S/p1b_compare_runs.py" "$O/run1" "$O/run2" "$O/c5b_cross_process.jsonl" > "$O/compare.log" 2>&1
python "$S/p1b_judge.py" "$O/run1" "$O/c5b_cross_process.jsonl" "$TOL" "$O/NOOP_CACHE_AUDIT_O0" > "$O/judge.log" 2>&1
date -u > "$O/done"
