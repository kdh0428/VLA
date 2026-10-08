#!/usr/bin/env bash
# P1-D main runs (protocol.md section 7). NOT launched by the preparing agent; run only after approval.
# Requires ~12.5 GB free on the RTX 5090 and ~10 GB free RAM (8 SAPIEN workers + model).
set -euo pipefail
ROOT=/root/VLA/autovla_misalignment_poc
RUN=$ROOT/outputs/review_followup/P1D_dose_control/run_20261008
[ -d "$RUN" ] || { echo "missing $RUN"; exit 1; }
source /root/VLA/simpler/env.sh
cd "$ROOT" || exit 1
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
[ "$free_mb" -ge 13000 ] || { echo "RTX 5090 has only ${free_mb} MiB free; abort"; exit 1; }
mkdir -p "$RUN/rollouts/main"
CFG=r1e1,r2e1,r4e1,r1e4
# Block 1: units 0-39 per task; all arms incl. null (no-op at scale) and gen (re-run noise floor)
nice -n 10 python scripts/review_followup/p1d/p1d_runner.py --out "$RUN/rollouts/main" --units-file "$RUN/units.json" \
  --configs $CFG --modes natural,feedback,corrected,reverse,null \
  --config-modes "r1e4:feedbackD,correctedD;r1e1:gen;r4e1:gen" --seeds 0-39 --workers 8 > "$RUN/rollouts/main/block1.log" 2>&1
# technical gate (protocol 6.1): only verification fields are inspected, never outcomes
python scripts/review_followup/p1d/analyze_p1d.py --run "$RUN/rollouts/main" --episodes "$RUN/rollouts/main/episodes.jsonl" --smoke \
  > "$RUN/rollouts/main/gate_block1.txt"
python - "$RUN/rollouts/main/verification.json" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
bad = []
for k in ("restore_ok", "null_identical_to_N", "prefix_identical_to_N"):
    for c, s in v[k].items():
        a, b = map(int, s.split("/"))
        if (k != "null_identical_to_N" and a != b) or (k == "null_identical_to_N" and a < 0.95 * b):
            bad.append((k, c, s))
if bad:
    print("GATE FAILED", bad); sys.exit(1)
print("gate passed")
EOF
# Block 2: units 40-119 per task; main arms only
nice -n 10 python scripts/review_followup/p1d/p1d_runner.py --out "$RUN/rollouts/main" --units-file "$RUN/units.json" \
  --configs $CFG --modes natural,feedback,corrected,reverse \
  --config-modes "r1e4:feedbackD,correctedD" --seeds 40-119 --workers 8 > "$RUN/rollouts/main/block2.log" 2>&1
nice -n 19 python scripts/review_followup/p1d/analyze_p1d.py --run "$RUN" --episodes "$RUN/rollouts/main/episodes.jsonl" > "$RUN/analysis_stdout.txt"
