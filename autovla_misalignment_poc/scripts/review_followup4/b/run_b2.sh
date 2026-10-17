#!/bin/bash
# B2 matched-prefix runs (protocol outputs/review_followup4/B/MATCHED_PREFIX_PROTOCOL.md). RTX 5090 only; one model load at a time.
#   bash run_b2.sh gate     # new-arm no-op/cache gate on the Stage-0 manifest       -> B2/gate_stage0
#   bash run_b2.sh pilot    # throughput pilot, B1 dev manifest alt0 t*<=4, no cell outcomes -> B2/pilot_throughput
#   bash run_b2.sh p1c      # main, P1-C cohort (images must be re-fetched first)  -> B2/p1c_main   (after the protocol commit)
#   bash run_b2.sh dev      # main, dev cohort (secondary)                           -> B2/dev_main   (after the protocol commit)
set -euo pipefail
MODE=${1:?usage: run_b2.sh <gate|pilot|p1c|dev>}
POC=/root/VLA/autovla_misalignment_poc
S=$POC/scripts/review_followup4/b
B2=$POC/outputs/review_followup4/B/B2
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case "$MODE" in
  gate)  ARGS=(--cohort stage0 --manifest $POC/outputs/review_followup3/P1/stage0_manifest.json --mode gate --out $B2/gate_stage0) ;;
  pilot) ARGS=(--cohort stage0 --manifest $POC/outputs/review_followup4/B/B1/b1_manifest_take20.json --mode main --timing-only --out $B2/pilot_throughput) ;;
  p1c)   ARGS=(--cohort p1c --manifest $B2/p1c_cohort_t4.json --mode main --out $B2/p1c_main)
         n=$(ls $B2/fetch/shard_*.done 2>/dev/null | wc -l); [ "$n" -eq 26 ] || { echo "P1-C images not re-fetched ($n/26 shards)"; exit 3; } ;;
  dev)   ARGS=(--cohort dev --mode main --out $B2/dev_main) ;;
  *) echo "bad mode"; exit 2 ;;
esac
avail=$(df --output=avail -k /root | tail -1 | tr -d ' ')
[ "$avail" -ge 1048576 ] || { echo "disk < 1 GB free"; exit 10; }
for attempt in 1 2 3 4 5 6; do
  bash $S/gpu_wait.sh 14000
  rc=0
  nice -n 10 python $S/b2_matched_prefix.py "${ARGS[@]}" || rc=$?
  [ $rc -eq 0 ] && exit 0
  echo "[run_b2] attempt $attempt failed (rc $rc); the runner resumes from bank/cells already written"; sleep 60
done
exit 1
