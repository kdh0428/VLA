#!/bin/bash
# Phase 2 chain: wait for dev run + all 26 shards, launch run_b2.sh p1c, and enforce the >=95% Normal-token input check
# as soon as the bank is frozen (kills the run before cell outcomes if it fails).
set -uo pipefail
POC=/root/VLA/autovla_misalignment_poc
S=$POC/scripts/review_followup4/b
B2=$POC/outputs/review_followup4/B/B2
until [ "$(ls $B2/fetch/shard_*.done 2>/dev/null | wc -l)" -eq 26 ] && grep -q "\[done\]" $B2/dev_main.log; do sleep 30; done
echo "[chain] start p1c $(date -u +%FT%TZ)"
bash $S/run_b2.sh p1c > $B2/p1c_main.log 2>&1 &
RP=$!
until [ -f $B2/p1c_main/bank_manifest.json ] || ! kill -0 $RP 2>/dev/null; do sleep 10; done
if [ -f $B2/p1c_main/bank_manifest.json ]; then
  if ! /usr/bin/python3 $S/b2_p1c_input_check.py $B2/p1c_main/bank.jsonl $B2/p1c_main/input_check.json; then
    echo "[chain] STOP: input check failed"; pkill -f "run_b2.sh p1c"; pkill -f "b2_matched_prefix.py --cohort p1c"; exit 4
  fi
  echo "[chain] input check PASS"
fi
wait $RP; echo "[chain] p1c finished rc $? $(date -u +%FT%TZ)"
