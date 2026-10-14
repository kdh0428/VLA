#!/bin/bash
# SpatialVLA P1-B/P1-C audit per PROTOCOL_P1.md §8: pilot (C5 -> tolerance) then gate (run1 all arms + C4, run2 separate process).
set -euo pipefail
source /root/VLA/simpler/env.sh
P=/root/VLA/autovla_misalignment_poc
S=$P/scripts/review_followup3/p1
O=$P/outputs/review_followup3/P1/svla_gate
MAN=$P/outputs/review_followup3/P1/svla_manifest.json
[[ "${O:?}" == /root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/* ]] || exit 2
mkdir -p "$O"
wait_gpu() {
  for i in $(seq 1 720); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    avail=$(free -m | awk '/Mem:/{print $7}')
    if [ "$free" -ge 14000 ] && [ "$avail" -ge 10000 ]; then echo "$(date -u +%FT%TZ) gpu1 free ${free} MiB ram ${avail}" >> "$O/gpu_checks.log"; return 0; fi
    echo "$(date -u +%FT%TZ) waiting: gpu1 ${free} MiB ram ${avail}" >> "$O/gpu_checks.log"; sleep 20
  done
  exit 3
}
step() { local name=$1; shift; [ -f "$O/$name.ok" ] && return 0; wait_gpu; nice -n 10 "$@" > "$O/$name.log" 2>&1; touch "$O/$name.ok"; }
step pilot_run1 python "$S/svla_noop_restore.py" --out "$O/pilot_run1" --manifest "$MAN" --units pilot --configs r1e4 --modes natural,natrep --workers 4
step pilot_run2 python "$S/svla_noop_restore.py" --out "$O/pilot_run2" --manifest "$MAN" --units pilot --configs r1e4 --modes natural --tag "#proc2" --workers 4
python "$S/svla_audit.py" pilot "$O/pilot_run1" "$O/pilot_run2" "$O" > "$O/pilot_audit.log" 2>&1
step gate_run1 python "$S/svla_noop_restore.py" --out "$O/run1" --manifest "$MAN" --units all --c4 --workers 4
step gate_run2 python "$S/svla_noop_restore.py" --out "$O/run2" --manifest "$MAN" --units all --modes natural --tag "#proc2" --workers 4
python "$S/svla_audit.py" gate "$O/run1" "$O/run2" "$O/svla_tolerances.json" "$P/outputs/review_followup3/P1/STATE_RESTORE_AUDIT" > "$O/gate_audit.log" 2>&1
date -u > "$O/done"
