#!/bin/bash
# P1-C 3-camera streaming driver (decision D7c). One navtest camera shard at a time:
#   fetch (CAM_F0/L1/R1 of the role's logs only) -> stage-1 natural pass -> strata/S sample -> prune images not needed
#   by the selected scenes; after all shards: ED units -> H7 -> H8 -> H9 -> H32 (dev scripts, unchanged) -> remove images.
#
#   bash p1c_driver.sh <pilot|eval> all            whole pipeline (resumable; finished steps are skipped)
#   bash p1c_driver.sh <pilot|eval> fetch <N>      fetch only (CPU/network)
#   bash p1c_driver.sh <pilot|eval> shard <N>      fetch + stage 1 + select + prune for one shard
#   bash p1c_driver.sh <pilot|eval> stage2         units + four mechanism blocks
#   bash p1c_driver.sh <pilot|eval> cleanup        remove this run's image dirs (logged)
#
# Guards (memory: shell-path-guards): every path below is a literal; variables used in paths are checked with ${VAR:?}
# and against an expected prefix; no cd; deletions only through p1c_disk.py (realpath + split role + fetched ledger
# + PoC protection, each logged in DISK_CLEANUP.md first). Free disk is never allowed below 3 GB (pre-check + watchdog).
# GPU: RTX 5090 only (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1); before each model load waits for >= 13 GB free
# GPU memory and >= 6 GB RAM (6 checks, 10 s apart) and for the P1-A/B launcher (run_all_main.sh) to be gone.
# Exit codes: 10 disk pause, 11 stage-1 error rate pause, 12 GPU-time overrun pause, 13 gate not met, 2 usage/guard.
set -euo pipefail
MODE=${1:?usage: p1c_driver.sh <pilot|eval> <all|shard N|stage2|cleanup>}
CMD=${2:?usage}
case "$MODE" in
  pilot) RUN=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/pilot_20261008; ROLE=pilot_harness_check ;;
  eval)  RUN=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008;   ROLE=eval ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac
[[ "${RUN:?}" == /root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/*_20261008 ]] || exit 2
P1C=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication
PILOT=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/pilot_20261008
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c
DEV=/root/VLA/autovla_misalignment_poc/scripts
SENSOR=/root/VLA/autovla/dataset/nuplan/sensor_blobs/test
SCENES=/root/VLA/autovla/dataset/nuplan/navtest_ext
TMP=/root/VLA/autovla/dataset/nuplan/_p1c_tmp
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
MIN_FREE_GB=3
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$RUN/logs" "$RUN/fetch" "$RUN/stage1" "$RUN/selection" "$RUN/mech"
LEDGER=$RUN/fetch/fetched_logs.jsonl; touch "$LEDGER"
LOG=$RUN/logs/driver.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit "${2:-2}"; }
free_kb() { df --output=avail -k / | tail -1 | tr -d ' '; }

wait_gpu() {   # >= 13 GB free on the 5090, >= 6 GB RAM, P1-A/B launcher not running; 6 consecutive checks
  local ok=0 n=0 f a busy
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    a=$(free -m | awk '/Mem:/{print $7}')
    busy=0; pgrep -f "p1ab/run_all_main.sh" >/dev/null && busy=1
    pgrep -f "p1ab/r8r9_run.sh" >/dev/null && busy=1
    if [ "$f" -ge 13000 ] && [ "$a" -ge 6000 ] && [ $busy -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 6 ] && { say "gpu free ${f} MiB, ram avail ${a} MiB"; return 0; }
    n=$((n+1)); [ $((n % 60)) -eq 0 ] && say "waiting for GPU (free ${f} MiB, ram ${a} MiB, p1ab launcher running=${busy})"
    sleep 10
  done
}

gpu_budget_check() {   # eval only: pause when GPU wall time exceeds 1.5x the pilot-updated estimate (protocol §10)
  [ "$MODE" = eval ] || return 0
  python - "$RUN/gpu_time.jsonl" "$PILOT/gpu_estimate.json" <<'PY' || exit 12
import json, os, sys
used = sum(json.loads(l)["s"] for l in open(sys.argv[1])) / 3600 if os.path.exists(sys.argv[1]) else 0.0
lim = 1.5 * json.load(open(sys.argv[2]))["total_gpu_h_estimate"]
print(f"gpu wall used {used:.2f} h, pause limit {lim:.2f} h")
sys.exit(0 if used <= lim else 1)
PY
}

run_gpu() {   # run_gpu <name> cmd... : wait for the GPU, run with nice, record wall time
  local name=$1; shift
  gpu_budget_check
  wait_gpu
  local t0=$(date +%s)
  say "GPU step $name: $*"
  if nice -n 19 "$@" > "$RUN/logs/$name.log" 2>&1; then rc=0; else rc=$?; fi
  local dt=$(( $(date +%s) - t0 ))
  echo "{\"step\": \"$name\", \"s\": $dt, \"rc\": $rc, \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$RUN/gpu_time.jsonl"
  [ $rc -eq 0 ] || die "GPU step $name failed (rc $rc), see $RUN/logs/$name.log" 4
  say "GPU step $name done in ${dt}s"
}

gates_eval() {   # the main run needs a passed pilot; stage 2 additionally needs the analysis script committed
  [ "$MODE" = eval ] || return 0
  [ -f "$PILOT/PILOT_OK" ] || die "pilot checks not passed ($PILOT/PILOT_OK missing)" 13
  [ -f "$PILOT/gpu_estimate.json" ] || die "no GPU estimate from the pilot" 13
  gate_analysis_frozen     # D7: analysis script committed before any evaluation data is produced
}
gate_analysis_frozen() {   # (defined after use in gates_eval; bash resolves at call time)
  [ "$MODE" = eval ] || return 0
  git -C /root/VLA ls-files --error-unmatch autovla_misalignment_poc/scripts/review_followup/p1c/analyze_p1c.py >/dev/null 2>&1 \
    || die "analyze_p1c.py is not committed (protocol §10: freeze before stage-2 output exists)" 13
  git -C /root/VLA diff --quiet HEAD -- autovla_misalignment_poc/scripts/review_followup/p1c/ \
    || die "p1c scripts differ from the committed version" 13
  sha256sum "$S/analyze_p1c.py" > "$RUN/analysis_script_frozen.sha256"
}

fetch_shard() {   # fetch_shard N : CAM_F0/L1/R1 of this role's logs in shard N -> sensor_blobs/test/<log>
  local SH=${1:?}
  [[ "$SH" =~ ^[0-9]+$ ]] && [ "$SH" -ge 6 ] && [ "$SH" -le 31 ] || die "bad shard $SH"
  [ -f "$RUN/fetch/shard_$SH.fetched" ] && { say "shard $SH already fetched"; return 0; }
  local LOGS L need_b need_kb avail PAT t0 dt rc
  LOGS=$(python "$S/p1c_info.py" logs --role "$ROLE" --shard "$SH")
  [ -n "$LOGS" ] || die "no $ROLE logs in shard $SH"
  for L in $LOGS; do
    [[ "$L" =~ ^20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9.]+_veh-[0-9]+_[0-9]+_[0-9]+$ ]] || die "bad log $L"
    if [ -e "$SENSOR/$L" ]; then
      grep -q "\"$L\"" "$LEDGER" || die "$SENSOR/$L exists and was not fetched by this run; refusing to touch it"
    fi
  done
  need_b=$(python "$S/p1c_info.py" budget --logs $LOGS)
  need_kb=$(( need_b * 125 / 100000 + 300000 ))   # pilot: real extraction was ~1.17x the 641 kB/frame budget
  avail=$(free_kb)
  [ "$avail" -ge $(( MIN_FREE_GB * 1000000 + need_kb )) ] || die "free disk ${avail} kB < 3 GB + shard need ${need_kb} kB" 10
  python "$S/p1c_disk.py" clean-tmp --reason "stale extraction scratch before fetching shard $SH"
  mkdir -p -- "$TMP"
  PAT=$RUN/fetch/patterns_$SH.txt; : > "$PAT"
  for L in $LOGS; do for c in CAM_F0 CAM_L1 CAM_R1; do echo "*/$L/$c/*" >> "$PAT"; done; done
  say "shard $SH: streaming $HF/openscene_sensor_test_camera_$SH.tgz, keeping $(echo $LOGS | wc -w) $ROLE logs (~$((need_b/1000000)) MB)"
  # watchdog: abort the stream if free disk drops below 3 GB
  ( while sleep 5; do
      if [ "$(free_kb)" -lt $(( MIN_FREE_GB * 1000000 )) ]; then
        echo "[$(date -u +%FT%TZ)] watchdog: free disk < 3 GB, killing shard $SH stream" >> "$LOG"
        touch "$RUN/fetch/PAUSE_DISK"; pkill -f "openscene_sensor_test_camera_$SH.tgz" || true; exit 0; fi
    done ) & local WD=$!
  t0=$(date +%s)
  set +e
  wget -qO- "$HF/openscene_sensor_test_camera_$SH.tgz" | tar -xz -C "$TMP" --wildcards -T "$PAT"
  rc=("${PIPESTATUS[@]}")
  set -e
  dt=$(( $(date +%s) - t0 ))
  kill "$WD" 2>/dev/null || true; wait "$WD" 2>/dev/null || true
  [ -f "$RUN/fetch/PAUSE_DISK" ] && die "disk watchdog fired during shard $SH" 10
  [ "${rc[0]}" -eq 0 ] || die "wget failed for shard $SH (rc ${rc[*]})" 5
  local peak_used_kb=$(du -sk "$TMP" | cut -f1)
  echo "{\"shard\": $SH, \"download_s\": $dt, \"tar_rc\": ${rc[1]}, \"extracted_kB\": $peak_used_kb, \"free_kB_after\": $(free_kb), \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$RUN/fetch/download_times.jsonl"
  for L in $LOGS; do
    local src=""
    for cand in "$TMP/openscene-v1.1/sensor_blobs/test/$L" "$TMP/openscene-v1.1/sensor_blobs/$L"; do
      [ -d "$cand" ] && src=$cand
    done
    [ -n "$src" ] && [ -d "$src" ] && [[ "$src" == /root/VLA/autovla/dataset/nuplan/_p1c_tmp/openscene-v1.1/sensor_blobs/* ]] \
      || die "log $L not found in shard $SH extraction"
    if [ -e "$SENSOR/$L" ]; then die "$SENSOR/$L appeared during fetch; refusing"; fi
    echo "{\"log\": \"$L\", \"shard\": $SH, \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$LEDGER"
    mv -- "$src" "$SENSOR/$L"
  done
  python "$S/p1c_disk.py" clean-tmp --reason "extraction scratch of shard $SH after moving its $ROLE logs"
  python "$S/p1c_info.py" check-images --logs $LOGS | tee -a "$LOG"
  touch "$RUN/fetch/shard_$SH.fetched"
  say "shard $SH fetched in ${dt}s; free $(( $(free_kb) / 1000 )) MB"
}

shard_pipeline() {
  local SH=${1:?}
  [[ "$SH" =~ ^[0-9]+$ ]] || die "bad shard"
  [ -f "$RUN/selection/shard_$SH.pruned" ] && { say "shard $SH stage 1 already complete"; return 0; }
  local LOGS; LOGS=$(python "$S/p1c_info.py" logs --role "$ROLE" --shard "$SH")
  fetch_shard "$SH"
  python "$S/p1c_info.py" tokens --logs $LOGS --out "$RUN/stage1/tokens_shard_$SH.json"
  run_gpu "stage1_shard$SH" python "$S/p1c_natural_pass.py" --tokens "$RUN/stage1/tokens_shard_$SH.json" --output "$RUN/stage1"
  # technical pause: harness error rate > 5% of the shard's scenes (protocol §10)
  python - "$RUN/stage1" "$RUN/stage1/tokens_shard_$SH.json" <<'PY' || die "stage-1 error rate > 5% in shard $SH" 11
import json, sys, os
toks = set(json.load(open(sys.argv[2])))
done = {json.loads(l)["token"] for l in open(os.path.join(sys.argv[1], "records.jsonl"))} & toks
print(f"stage-1 records {len(done)}/{len(toks)}")
sys.exit(0 if len(toks - done) <= 0.05 * len(toks) else 1)
PY
  if [ ! -f "$RUN/selection/shard_$SH.json" ]; then
    python "$S/p1c_select.py" --stage1 "$RUN/stage1" --logs $LOGS --out "$RUN/selection/shard_$SH.json" | tee -a "$LOG"
  fi
  python "$S/p1c_disk.py" prune --ledger "$LEDGER" --keep "$RUN/selection/shard_$SH.json.keep_files.json" --logs $LOGS \
    --reason "shard $SH: stage 1 done; images not used by any F/S-selected scene" | tee -a "$LOG"
  touch "$RUN/selection/shard_$SH.pruned"
}

mech_step() {   # mech_step <name> <script> [extra args]: unchanged dev script; refuses to overwrite partial output
  local name=$1 script=$2; shift 2
  local out=$RUN/mech/$name
  [ -f "$out/run_meta.json" ] && { say "$name already complete"; return 0; }
  [ -e "$out/records.jsonl" ] && die "$out/records.jsonl exists without run_meta.json (partial); refusing to overwrite"
  run_gpu "$name" python "$DEV/$script" --ed-records "$RUN/units/records.jsonl" --full-records "$RUN/stage1/records.jsonl" \
    --scenes "$SCENES" --output "$out" "$@"
}

stage2() {
  gate_analysis_frozen
  local SHARDS SH sels=()
  SHARDS=$(python "$S/p1c_info.py" shards --role "$ROLE")
  for SH in $SHARDS; do
    [ -f "$RUN/selection/shard_$SH.pruned" ] || die "shard $SH stage 1 not complete; stage 2 needs every shard"
    sels+=("$RUN/selection/shard_$SH.json")
  done
  if [ ! -f "$RUN/units/run_meta.jsonl" ]; then   # resumable; scenes that raised are harness exclusions (not retried)
    run_gpu units python "$S/p1c_build_units.py" --stage1 "$RUN/stage1/records.jsonl" --selection "${sels[@]}" --output "$RUN/units"
  fi
  local AH=$RUN/mech/action_history_causal/records.jsonl
  mech_step action_history_causal action_history_causal.py
  mech_step prev_action_state_patching prev_action_state_patching.py --ah-records "$AH"
  mech_step temporal_feedback_window temporal_feedback_window.py --ah-records "$AH"
  mech_step motion_semantics_ablation motion_semantics_ablation.py --ah-records "$AH"
  touch "$RUN/STAGE2_DONE"
  say "stage 2 complete"
}

cleanup() {
  local L logs
  logs=$(python -c "import json,sys; print(' '.join(sorted({json.loads(l)['log'] for l in open(sys.argv[1]) if l.strip()})))" "$LEDGER")
  [ -n "$logs" ] || { say "nothing to clean"; return 0; }
  python "$S/p1c_disk.py" remove --ledger "$LEDGER" --logs $logs \
    --reason "$MODE run finished (stage 2 complete); retained images of selected scenes no longer needed" | tee -a "$LOG"
}

case "$CMD" in
  fetch)   gates_eval; fetch_shard "${3:?shard number}" ;;
  shard)   gates_eval; shard_pipeline "${3:?shard number}" ;;
  stage2)  gates_eval; stage2 ;;
  cleanup) [ -f "$RUN/STAGE2_DONE" ] || die "stage 2 not complete; refusing cleanup"; cleanup ;;
  all)
    gates_eval
    say "==== $MODE run start; shards: $(python "$S/p1c_info.py" shards --role "$ROLE")"
    for SH in $(python "$S/p1c_info.py" shards --role "$ROLE"); do shard_pipeline "$SH"; done
    stage2
    cleanup
    say "==== $MODE run finished" ;;
  *) echo "bad command $CMD"; exit 2 ;;
esac
