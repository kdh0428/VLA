#!/bin/bash
# wait until the RTX 5090 (PCI index 1) has >= ${1:-14000} MiB free for 6 consecutive checks 10 s apart.
NEED=${1:-14000}
ok=0; n=0
while :; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [ "$f" -ge "$NEED" ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 6 ] && { echo "[gpu_wait] free ${f} MiB $(date -u +%FT%TZ)"; exit 0; }
  n=$((n+1)); [ $((n % 30)) -eq 0 ] && echo "[gpu_wait] waiting (free ${f} MiB) $(date -u +%FT%TZ)"
  sleep 10
done
