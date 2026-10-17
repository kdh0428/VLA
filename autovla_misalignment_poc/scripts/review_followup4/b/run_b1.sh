#!/bin/bash
# B1 numerical sensitivity (protocol outputs/review_followup4/B/B1/NUMERICAL_SENSITIVITY_PROTOCOL.md)
set -euo pipefail
POC=/root/VLA/autovla_misalignment_poc
S=$POC/scripts/review_followup4/b
OUT=${1:-$POC/outputs/review_followup4/B/B1/run}
[[ "$OUT" == /root/VLA/autovla_misalignment_poc/outputs/review_followup4/B/* || "$OUT" == /tmp/claude-0/* ]] || { echo "bad OUT"; exit 2; }
EXTRA=${2:-}
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
avail=$(df --output=avail -k /root | tail -1 | tr -d ' ')
[ "$avail" -ge 1048576 ] || { echo "disk < 1 GB free"; exit 10; }
for attempt in 1 2 3 4 5 6; do
  bash $S/gpu_wait.sh 14000
  rc=0
  nice -n 10 python $S/b1_runner.py --manifest $POC/outputs/review_followup4/B/B1/b1_manifest_take20.json --out "$OUT" $EXTRA || rc=$?
  [ $rc -eq 0 ] && exit 0
  # retry only on an OOM at load before any record was written
  if [ -s "$OUT/records.jsonl" ]; then echo "failed after records were written (rc $rc)"; exit $rc; fi
  rm -f "$OUT/records.jsonl" "$OUT/logit_cmp.jsonl"
  echo "[run_b1] attempt $attempt failed (rc $rc), retrying after GPU wait"; sleep 60
done
exit 1
