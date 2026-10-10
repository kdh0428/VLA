#!/bin/bash
# P1-C amendment 2 (commit 9b3d4af): re-fetch the CAM_F0/L1/R1 images of the 353 selected scenes, re-run H8 with the
# original command into the original output path, then remove the re-fetched images. New file; the committed driver and
# helpers are reused unchanged (p1c_disk.py does every deletion and logs it in DISK_CLEANUP.md first).
#
#   bash p1c_h8_rerun.sh fetch      stream each navtest camera shard once; keep only the files listed in
#                                   run_20261008/selection/shard_N.json.keep_files.json (resumable per shard)
#   bash p1c_h8_rerun.sh h8         wait for >= 14 GB free on the 5090 for 60 s, run H8 (original command), max 3 attempts;
#                                   an OOM attempt is moved aside to mech/_failed_attemptK_prev_action_state_patching/
#   bash p1c_h8_rerun.sh cleanup    remove the re-fetched log dirs (p1c_disk.py remove; logged)
#
# Guards (memory: shell-path-guards): literal paths, no cd, ${VAR:?} on every path variable, prefix checks before mv.
# Free disk is never allowed below 3 GB (pre-check + watchdog). State/logs: run_20261008/logs/h8_rerun/.
set -euo pipefail
CMD=${1:?usage: p1c_h8_rerun.sh <fetch|h8|cleanup>}
RUN=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c
DEV=/root/VLA/autovla_misalignment_poc/scripts
SENSOR=/root/VLA/autovla/dataset/nuplan/sensor_blobs/test
SCENES=/root/VLA/autovla/dataset/nuplan/navtest_ext
TMP=/root/VLA/autovla/dataset/nuplan/_p1c_tmp
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
ST=$RUN/logs/h8_rerun
MIN_FREE_GB=3
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "${ST:?}"
LEDGER=$ST/fetched_logs.jsonl; touch "$LEDGER"
LOG=$ST/rerun.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
die() { say "STOP: $*"; exit "${2:-2}"; }
free_kb() { df --output=avail -k / | tail -1 | tr -d ' '; }
note_free() { echo -e "$(date -u +%FT%TZ)\t$(free_kb)\t$1" >> "$ST/disk_free_kB.tsv"; }

fetch_shard() {
  local SH=${1:?} KEEP LOGS L PAT need_kb avail t0 dt rc src cand
  [[ "$SH" =~ ^[0-9]+$ ]] && [ "$SH" -ge 6 ] && [ "$SH" -le 31 ] || die "bad shard $SH"
  [ -f "$ST/shard_$SH.fetched" ] && { say "shard $SH already fetched"; return 0; }
  KEEP=$RUN/selection/shard_$SH.json.keep_files.json
  [ -f "$KEEP" ] || die "no keep list for shard $SH"
  # logs of the keep list; every one must be an eval log of this shard in split.csv
  LOGS=$(python - "$KEEP" "$SH" <<'PY'
import json, sys
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c")
import p1c_common as C
keep = json.load(open(sys.argv[1])); split = C.load_split(); prot = C.protected_logs()
logs = sorted({p.split("/")[0] for p in keep})
for l in logs:
    assert C.LOG_RE.match(l) and l not in prot and split[l]["role"] == "eval" and split[l]["shard"] == sys.argv[2], l
print(" ".join(logs))
PY
)
  [ -n "$LOGS" ] || { say "shard $SH: no selected images"; touch "$ST/shard_$SH.fetched"; return 0; }
  for L in $LOGS; do
    [[ "$L" =~ ^20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9.]+_veh-[0-9]+_[0-9]+_[0-9]+$ ]] || die "bad log $L"
    [ -e "$SENSOR/$L" ] && die "$SENSOR/$L already exists; refusing to touch it"
  done
  PAT=$ST/patterns_$SH.txt
  python -c "import json,sys; [print('*/'+p) for p in json.load(open(sys.argv[1]))]" "$KEEP" > "$PAT"
  need_kb=$(( $(wc -l < "$PAT") * 400 + 300000 ))   # <= 400 kB per jpg + margin
  avail=$(free_kb)
  [ "$avail" -ge $(( MIN_FREE_GB * 1000000 + need_kb )) ] || die "free disk ${avail} kB < 3 GB + need ${need_kb} kB" 10
  python "$S/p1c_disk.py" clean-tmp --reason "amendment 2 re-fetch: stale extraction scratch before shard $SH"
  mkdir -p -- "$TMP"
  say "shard $SH: streaming openscene_sensor_test_camera_$SH.tgz, keeping $(wc -l < "$PAT") files of $(echo $LOGS | wc -w) logs"
  ( while sleep 5; do
      if [ "$(free_kb)" -lt $(( MIN_FREE_GB * 1000000 )) ]; then
        echo "[$(date -u +%FT%TZ)] watchdog: free disk < 3 GB, killing shard $SH stream" >> "$LOG"
        touch "$ST/PAUSE_DISK"; pkill -f "openscene_sensor_test_camera_$SH.tgz" || true; exit 0; fi
      echo -e "$(date -u +%FT%TZ)\t$(free_kb)\tshard$SH-stream" >> "$ST/disk_free_kB.tsv"
    done ) & local WD=$!
  t0=$(date +%s)
  set +e
  wget -qO- "$HF/openscene_sensor_test_camera_$SH.tgz" | tar -xz -C "$TMP" --wildcards -T "$PAT"
  rc=("${PIPESTATUS[@]}")
  set -e
  dt=$(( $(date +%s) - t0 ))
  kill "$WD" 2>/dev/null || true; wait "$WD" 2>/dev/null || true
  [ -f "$ST/PAUSE_DISK" ] && die "disk watchdog fired during shard $SH" 10
  [ "${rc[0]}" -eq 0 ] || die "wget failed for shard $SH (rc ${rc[*]})" 5
  local ext_kb; ext_kb=$(du -sk "$TMP" | cut -f1)
  echo "{\"shard\": $SH, \"download_s\": $dt, \"wget_rc\": ${rc[0]}, \"tar_rc\": ${rc[1]}, \"files\": $(wc -l < "$PAT"), \"extracted_kB\": $ext_kb, \"free_kB_after\": $(free_kb), \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$ST/download_times.jsonl"
  for L in $LOGS; do
    src=""
    for cand in "$TMP/openscene-v1.1/sensor_blobs/test/$L" "$TMP/openscene-v1.1/sensor_blobs/$L"; do
      [ -d "$cand" ] && src=$cand
    done
    [ -n "$src" ] && [ -d "$src" ] && [[ "$src" == /root/VLA/autovla/dataset/nuplan/_p1c_tmp/openscene-v1.1/sensor_blobs/* ]] \
      || die "log $L not found in shard $SH extraction"
    [ -e "$SENSOR/$L" ] && die "$SENSOR/$L appeared during fetch; refusing"
    echo "{\"log\": \"$L\", \"shard\": $SH, \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$LEDGER"
    mv -- "$src" "$SENSOR/$L"
  done
  python "$S/p1c_disk.py" clean-tmp --reason "amendment 2 re-fetch: extraction scratch of shard $SH after moving its selected-scene images"
  python - "$KEEP" <<'PY' | tee -a "$LOG"
import json, os, sys
keep = json.load(open(sys.argv[1]))
miss = [p for p in keep if not os.path.isfile(os.path.join("/root/VLA/autovla/dataset/nuplan/sensor_blobs/test", p))]
print(json.dumps({"needed": len(keep), "missing": len(miss), "examples": miss[:3]}))
sys.exit(1 if miss else 0)
PY
  touch "$ST/shard_$SH.fetched"
  note_free "shard$SH-done"
  say "shard $SH fetched in ${dt}s; free $(( $(free_kb) / 1000 )) MB"
}

wait_gpu14() {   # >= 14000 MiB free on the 5090 and >= 6 GB RAM for 60 s (13 checks, 5 s apart)
  local ok=0 n=0 f a
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    a=$(free -m | awk '/Mem:/{print $7}')
    if [ "$f" -ge 14000 ] && [ "$a" -ge 6000 ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 13 ] && { say "gpu free ${f} MiB for 60 s, ram avail ${a} MiB"; return 0; }
    n=$((n+1)); [ $((n % 120)) -eq 0 ] && say "waiting for GPU (free ${f} MiB, ram ${a} MiB)"
    sleep 5
  done
}

h8() {
  local OUT=$RUN/mech/prev_action_state_patching K t0 dt rc fails
  [ -f "$OUT/run_meta.json" ] && { say "H8 already complete"; return 0; }
  [ -e "$OUT" ] && die "$OUT exists without run_meta.json; refusing"
  for K in 2 3 4; do   # attempt 1 was the main run
    wait_gpu14
    t0=$(date +%s)
    say "H8 attempt $K start"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv -i 1 >> "$LOG"
    set +e
    nice -n 19 python "$DEV/prev_action_state_patching.py" --ed-records "$RUN/units/records.jsonl" \
      --full-records "$RUN/stage1/records.jsonl" --scenes "$SCENES" --output "$OUT" \
      --ah-records "$RUN/mech/action_history_causal/records.jsonl" > "$ST/prev_action_state_patching_attempt$K.log" 2>&1
    rc=$?
    set -e
    dt=$(( $(date +%s) - t0 ))
    echo "{\"step\": \"prev_action_state_patching_attempt$K\", \"s\": $dt, \"rc\": $rc, \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$ST/gpu_time.jsonl"
    tail -1 "$ST/prev_action_state_patching_attempt$K.log" | tee -a "$LOG"
    fails=$(python -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['n_unit_fail'], m['n_units'], m['n_scene_fail'])" "$OUT/run_meta.json" 2>/dev/null || echo "NA")
    say "H8 attempt $K: rc $rc, ${dt}s, unit_fail units scene_fail = $fails"
    if [ $rc -eq 0 ] && [ "${fails%% *}" = "0" ]; then return 0; fi
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$ST/prev_action_state_patching_attempt$K.log" && [ $K -lt 4 ]; then
      local AS=$RUN/mech/_failed_attempt${K}_prev_action_state_patching
      [ -e "$AS" ] && die "$AS exists"
      [ -d "$OUT" ] && mv -- "$OUT" "$AS"
      say "OOM in attempt $K; moved output to $AS; waiting 10 min before retry"
      sleep 600
      continue
    fi
    die "H8 attempt $K ended with rc $rc / failures $fails (not retried automatically)" 4
  done
}

case "$CMD" in
  fetch)
    note_free "fetch-start"
    for SH in $(seq 6 31); do fetch_shard "$SH"; done
    say "fetch complete; free $(( $(free_kb) / 1000 )) MB" ;;
  h8) h8 ;;
  cleanup)
    logs=$(python -c "import json,sys; print(' '.join(sorted({json.loads(l)['log'] for l in open(sys.argv[1]) if l.strip()})))" "$LEDGER")
    [ -n "$logs" ] || { say "nothing to clean"; exit 0; }
    python "$S/p1c_disk.py" remove --ledger "$LEDGER" --logs $logs \
      --reason "amendment 2: H8 re-run and analysis finished; re-fetched selected-scene images (p1c_h8_rerun.sh fetch) no longer needed" | tee -a "$LOG"
    note_free "cleanup-done" ;;
  *) echo "bad command $CMD"; exit 2 ;;
esac
