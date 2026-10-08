#!/bin/bash
# Smoke of the P1-A SECONDARY fixed-position arm (4 units), RTX 5090 only; waits for GPU memory and retries an
# OOM at model load (no records written) up to 5 times. The empty attempt dir is moved aside, never deleted.
set -euo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1ab
A=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1A_gt_free_reference/run_20261008/smoke
wait_gpu() { local ok=0; for _ in $(seq 1 720); do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [ "$f" -ge 13000 ]; then ok=$((ok+1)); else ok=0; fi; [ $ok -ge 6 ] && { echo "gpu free $f $(date -u +%T)"; return 0; }; sleep 10; done; echo "gpu busy"; exit 3; }
for att in 1 2 3 4 5; do
  wait_gpu
  if nice -n 19 python "$S/p1a_gt_free_reference.py" --out "$A/raw_fixedpos" --reference "$A/reference" --families G \
      --fixed-positions 1,3 --tokens 013fbdcd9db35b43,12d283af921a5f09 --max-units-per-scene 1 --trace-units 99 > "$A/p1a_fixedpos.log" 2>&1; then break; fi
  n=$( { cat "$A/raw_fixedpos/records.jsonl" 2>/dev/null || true; } | wc -l); [ "$n" -gt 0 ] && { echo "failed after records"; exit 4; }
  mv "$A/p1a_fixedpos.log" "$A/p1a_fixedpos_oom_retry$att.log"; [ -d "$A/raw_fixedpos" ] && mv "$A/raw_fixedpos" "$A/raw_fixedpos_failed_retry$att"
  echo "attempt $att failed at load; retry"; sleep 60
done
[ -f "$A/raw_fixedpos/run_meta.json" ] || { echo "no success"; exit 5; }
nice -n 19 python "$S/p1a_analyze.py" --raw "$A/raw_fixedpos" --out "$A/analysis_fixedpos" > "$A/p1a_fixedpos_analyze.log" 2>&1
