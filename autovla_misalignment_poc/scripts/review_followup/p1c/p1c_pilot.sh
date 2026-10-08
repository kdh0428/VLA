#!/bin/bash
# P1-C pilot (protocol §6) on the 2 pilot_harness_check logs only (never analysed for effects).
#   bash p1c_pilot.sh            (resumable; each finished step is skipped)
# Steps: driver shard 16, 20 (fetch -> stage 1 -> select -> prune) | P1 natural pass on 30 PoC scenes | P3 run with a
# CAM_F0/L1/R1-only symlink tree | ED-wrapper parity on 3 dev scenes | driver stage 2 (units + H7/H8/H9/H32 unchanged)
# | H8 --debug on 2 pilot scenes | masked analysis | checks report | image cleanup.
set -euo pipefail
P=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/pilot_20261008
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c
DEV=/root/VLA/autovla_misalignment_poc/scripts
POC_SCENES=/root/VLA/autovla/dataset/nuplan/navtest_poc
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$P/logs" "$P/checks"
LOG=$P/logs/pilot.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
( while sleep 10; do echo -e "$(date -u +%FT%TZ)\t$(df --output=avail -k / | tail -1 | tr -d ' ')" >> "$P/logs/disk_free_kB.tsv"; done ) &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null || true' EXIT

wait_gpu() { local ok=0 f a busy n=0
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    a=$(free -m | awk '/Mem:/{print $7}')
    busy=0; pgrep -f "p1ab/run_all_main.sh" >/dev/null && busy=1; pgrep -f "p1ab/r8r9_run.sh" >/dev/null && busy=1
    if [ "$f" -ge 13000 ] && [ "$a" -ge 6000 ] && [ $busy -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 6 ] && { say "gpu free ${f} MiB, ram ${a} MiB"; return 0; }
    n=$((n+1)); [ $((n % 60)) -eq 0 ] && say "waiting for GPU (free ${f} MiB, ram ${a} MiB, p1ab running=${busy})"
    sleep 10; done; }
run_gpu() { local name=$1; shift; wait_gpu; local t0=$(date +%s); say "GPU step $name"
  nice -n 19 "$@" > "$P/logs/$name.log" 2>&1 || { say "FAILED $name"; exit 4; }
  local dt=$(( $(date +%s) - t0 ))
  echo "{\"step\": \"$name\", \"s\": $dt, \"rc\": 0, \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$P/gpu_time.jsonl"; say "done $name ${dt}s"; }

say "==== pilot start"
for SH in 16 20; do bash "$S/p1c_driver.sh" pilot shard "$SH"; done

[ -f "$P/checks/p1_stage1/run_meta.jsonl" ] || run_gpu p1_planner_vs_full_extract \
  python "$S/p1c_natural_pass.py" --tokens "$P/checks/p1_tokens.json" --scenes "$POC_SCENES" --output "$P/checks/p1_stage1"
[ -f "$P/checks/p3_stage1_3cam/run_meta.jsonl" ] || run_gpu p3_three_camera_tree \
  python "$S/p1c_natural_pass.py" --tokens "$P/checks/p3_tokens.json" --scenes "$POC_SCENES" --output "$P/checks/p3_stage1_3cam" \
  --sensor-root "$P/checks/p3_sensor_tree"
[ -f "$P/checks/ed_parity/run_meta.jsonl" ] || run_gpu ed_wrapper_parity \
  python "$S/p1c_build_units.py" --stage1 /root/VLA/autovla_misalignment_poc/outputs/full_extract/records.jsonl \
  --check-dev /root/VLA/autovla_misalignment_poc/outputs/equal_distance_perturbation/records.jsonl --limit 3 \
  --scenes "$POC_SCENES" --output "$P/checks/ed_parity"

bash "$S/p1c_driver.sh" pilot stage2

[ -f "$P/checks/h8_debug/run_meta.json" ] || run_gpu h8_debug \
  python "$DEV/prev_action_state_patching.py" --ed-records "$P/units/records.jsonl" --full-records "$P/stage1/records.jsonl" \
  --ah-records "$P/mech/action_history_causal/records.jsonl" --scenes /root/VLA/autovla/dataset/nuplan/navtest_ext \
  --output "$P/checks/h8_debug" --limit 2 --debug

[ -f "$P/analysis_masked/paired_effects.csv" ] || \
  nice -n 19 python "$S/analyze_p1c.py" --run "$P" --out "$P/analysis_masked" --mask-effects > "$P/logs/analysis_masked.log" 2>&1
python "$S/p1c_pilot_checks.py" report --pilot "$P" | tee -a "$LOG"
bash "$S/p1c_driver.sh" pilot cleanup
say "==== pilot finished"
