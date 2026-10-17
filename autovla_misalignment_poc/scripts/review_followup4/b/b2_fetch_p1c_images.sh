#!/bin/bash
# Re-fetch ONLY the camera images the 353 P1-C selected scenes need (the P1-C keep lists, 3,831 jpg, ~0.36 GB) from the
# public OpenScene v1.1 navtest camera shards (same URL as p1c_driver.sh). One shard at a time, streamed (wget | tar),
# only the listed files are written. No cd; literal paths; every moved file is appended to a ledger for later removal.
#   bash b2_fetch_p1c_images.sh <shard|all>
set -euo pipefail
WHAT=${1:?usage: b2_fetch_p1c_images.sh <shard number|all>}
RUN=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008
OUT=/root/VLA/autovla_misalignment_poc/outputs/review_followup4/B/B2/fetch
SENSOR=/root/VLA/autovla/dataset/nuplan/sensor_blobs/test
TMP=/root/VLA/autovla/dataset/nuplan/_b2_tmp
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
MIN_FREE_KB=1500000
mkdir -p "$OUT" "$TMP"
LEDGER=$OUT/fetched_files.txt; touch "$LEDGER"
free_kb() { df --output=avail -k /root | tail -1 | tr -d ' '; }

fetch_shard() {
  local SH=$1
  local KEEP=$RUN/selection/shard_$SH.json.keep_files.json
  [ -f "$KEEP" ] || { echo "no keep list for shard $SH"; return 2; }
  [ -f "$OUT/shard_$SH.done" ] && { echo "shard $SH done"; return 0; }
  [ "$(free_kb)" -ge $MIN_FREE_KB ] || { echo "STOP: disk < 1.5 GB free"; exit 10; }
  local PAT=$OUT/patterns_$SH.txt
  python3 -c "import json,sys; [print('*/'+f) for f in json.load(open(sys.argv[1]))]" "$KEEP" > "$PAT"
  local t0=$(date +%s)
  set +e
  wget -qO- "$HF/openscene_sensor_test_camera_$SH.tgz" | tar -xz -C "$TMP" --wildcards -T "$PAT"
  local rc=("${PIPESTATUS[@]}")
  set -e
  [ "${rc[0]}" -eq 0 ] || { echo "wget failed shard $SH (${rc[*]})"; return 5; }
  local n=0 miss=0
  while IFS= read -r rel; do
    local src
    src=$(find "$TMP" -path "*/$rel" -type f | head -1)
    if [ -z "$src" ]; then miss=$((miss+1)); continue; fi
    [[ "$src" == "$TMP"/* ]] || { echo "bad src $src"; exit 2; }
    local dst="$SENSOR/$rel"
    mkdir -p -- "$(dirname -- "$dst")"
    mv -n -- "$src" "$dst"
    echo "$dst" >> "$LEDGER"; n=$((n+1))
  done < <(python3 -c "import json,sys; [print(f) for f in json.load(open(sys.argv[1]))]" "$KEEP")
  echo "{\"shard\": $SH, \"moved\": $n, \"missing\": $miss, \"sec\": $(( $(date +%s) - t0 )), \"free_kB\": $(free_kb)}" >> "$OUT/fetch_log.jsonl"
  [ $miss -eq 0 ] && touch "$OUT/shard_$SH.done"
  # remove empty temp dirs only (no file deletion here)
  find "$TMP" -mindepth 1 -type d -empty -delete
}

if [ "$WHAT" = all ]; then
  for SH in $(seq 6 31); do fetch_shard "$SH"; done
else
  fetch_shard "$WHAT"
fi
