#!/bin/bash
# Fetch the original-scene images navhard stage 1 needs (CAM_F0/L1/R1, 4 history frames) from the
# navtest camera shards, extracting ONLY the listed files. No `cd`; every path is absolute and checked.
#   bash tools/fetch_navhard_originals.sh [shard ...]     (default: every shard in the list file)
set -euo pipefail
LIST=/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset/orig_files_missing_by_shard.json
DEST=/root/VLA/autovla/dataset/nuplan/sensor_blobs/test
TMP=/root/VLA/navhard/_orig_tmp
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
[ -f "$LIST" ] && [ -d "$DEST" ] || { echo "missing list or destination"; exit 1; }
SHARDS=("$@")
if [ ${#SHARDS[@]} -eq 0 ]; then
  mapfile -t SHARDS < <(python3 -c "import json;print('\n'.join(json.load(open('$LIST'))))")
fi
for SH in "${SHARDS[@]}"; do
  [[ "$SH" =~ ^openscene_sensor_test_[0-9]+$ ]] || { echo "bad shard name $SH"; exit 1; }
  PAT=/root/VLA/navhard/_orig_patterns_$SH.txt
  python3 -c "import json;print('\n'.join('*/'+f for f in json.load(open('$LIST'))['$SH']))" > "$PAT"
  rm -rf -- "$TMP"; mkdir -p -- "$TMP"
  echo "######## $(date '+%F %T') $SH: $(wc -l < "$PAT") files"
  wget -qO- "$HF/openscene_sensor_test_camera_${SH##*_}.tgz" | tar -xz -C "$TMP" --wildcards -T "$PAT" || true
  n=0
  while IFS= read -r f; do
    rel=${f#\*/}
    src=$(find "$TMP" -type f -path "*/$rel" | head -1)
    if [ -n "$src" ] && [ -f "$src" ]; then
      mkdir -p -- "$(dirname -- "$DEST/$rel")"
      if [ -e "$DEST/$rel" ]; then n=$((n+1)); else mv -n -- "$src" "$DEST/$rel" && n=$((n+1)); fi
    fi
  done < "$PAT"
  echo "moved $n / $(wc -l < "$PAT")"
  rm -rf -- "$TMP" "$PAT"
done
python3 - <<'PY'
import json, os
d = json.load(open("/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset/orig_files_missing_by_shard.json"))
allf = [f for v in d.values() for f in v]
miss = [f for f in allf if not os.path.exists("/root/VLA/autovla/dataset/nuplan/sensor_blobs/test/" + f)]
print(f"original images present {len(allf) - len(miss)} / {len(allf)}")
PY
