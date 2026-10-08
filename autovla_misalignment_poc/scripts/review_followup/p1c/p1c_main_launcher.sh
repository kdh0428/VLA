#!/bin/bash
# P1-C MAIN RUN launcher (64 eval logs, shards 6-31). Do not run before: (1) pilot passed (pilot_20261008/PILOT_OK),
# (2) the p1c scripts incl. analyze_p1c.py are committed (stage 2 refuses otherwise; protocol §10), (3) the lead has
# approved the launch. Launch exactly as:
#   nohup bash /root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c/p1c_main_launcher.sh \
#     > /root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008_launcher.out 2>&1 &
# Sequential: per shard (eval_order_key order) fetch CAM_F0/L1/R1 of its eval logs -> stage-1 natural pass -> select ->
# prune; then ED units + H7 + H8 + H9 + H32 (unchanged dev scripts) on all selected scenes; then image cleanup.
# Every model load waits for >= 13 GB free on the RTX 5090, >= 6 GB RAM and no P1-A/B launcher. Resumable: re-run the
# same command after a pause; finished steps are skipped and partial mechanism outputs are never overwritten.
# Pauses (exit codes, protocol §10): 10 free disk < 3 GB, 11 stage-1 error rate > 5% in a shard, 12 GPU wall time
# > 1.5x the pilot estimate, 13 gate not met. Resume only after a dated amendment entry (protocol §10).
set -euo pipefail
RUN=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c
mkdir -p "$RUN/logs"
exec 9> "$RUN/.launcher.lock"
flock -n 9 || { echo "another P1-C launcher is running"; exit 2; }
( while sleep 30; do echo -e "$(date -u +%FT%TZ)\t$(df --output=avail -k / | tail -1 | tr -d ' ')" >> "$RUN/logs/disk_free_kB.tsv"; done ) &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null || true' EXIT
bash "$S/p1c_driver.sh" eval all
echo "main run finished; analysis (frozen script, run once):"
echo "  python $S/analyze_p1c.py --run $RUN --out $RUN/analysis"
