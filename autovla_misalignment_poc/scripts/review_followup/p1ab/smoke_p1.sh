#!/bin/bash
# Smoke test of P1 reference + P1-A + P1-B (<= 5 units each), sequential model loads, RTX 5090 only.
set -euo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1ab
A=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1A_gt_free_reference/run_20261008/smoke
B=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1B_start_position/run_20261008/smoke
mkdir -p "$A" "$B"
TOK=013fbdcd9db35b43,0c931f9db55e5fcc,12d283af921a5f09
{ date -u; free -g; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv; } > "$A/resources_before.txt"
step() { echo "== $1 $(date -u +%T)"; }
# wait until the 5090 has >= NEED MiB free for 3 consecutive checks (other jobs share it)
wait_gpu() { local need=${1:-13000} ok=0; for _ in $(seq 1 360); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
  a=$(free -m | awk '/Mem:/{print $7}')
  if [ "$f" -ge "$need" ] && [ "$a" -ge 6000 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 3 ] && { echo "gpu free ${f} MiB, ram avail ${a} MiB"; return 0; }; sleep 10; done; echo "gpu busy, abort"; exit 3; }
if [ ! -f "$A/reference/manifest.json" ]; then
step reference
wait_gpu 13000
nice -n 19 python "$S/p1_reference.py" --out "$A/reference" --tokens "$TOK" > "$A/reference.log" 2>&1
fi
step p1a
wait_gpu 13000
nice -n 19 python "$S/p1a_gt_free_reference.py" --out "$A/raw" --reference "$A/reference" \
  --tokens 013fbdcd9db35b43,12d283af921a5f09 --max-units-per-scene 1 --trace-units 99 > "$A/p1a.log" 2>&1
step p1b
wait_gpu 13000
nice -n 19 python "$S/p1b_start_position.py" --out "$B/raw" --reference "$A/reference" \
  --tokens 013fbdcd9db35b43,12d283af921a5f09 --max-alts 1 --trace-units 99 > "$B/p1b.log" 2>&1
step done
