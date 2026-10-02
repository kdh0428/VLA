#!/bin/bash
# Re-fetch the CAM_F0/L1/R1 images of specific logs from a navtest camera shard, decode the listed tokens,
# then delete those images again. Paths are absolute and validated; nothing outside the named log dirs is touched.
#   bash tools/repair_stream_logs.sh <shard_number> <output_run_dir> <tokens.json> <log> [<log> ...]
set -euo pipefail
SHN=$1; RUN=$2; TOK=$3; shift 3
[[ "$SHN" =~ ^[0-9]+$ ]] || { echo "bad shard $SHN"; exit 1; }
B=/root/VLA/autovla/dataset/nuplan/sensor_blobs/test
TMP=/root/VLA/navhard/_repair_tmp
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
POC=/root/VLA/autovla_misalignment_poc
PROTECT=$(python3 -c "import yaml;print(' '.join(yaml.safe_load(open('$POC/configs/scene_filter_navtest_subset.yaml'))['log_names']))")
for L in "$@"; do
  [[ "$L" =~ ^20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9.]+_veh-[0-9]+_[0-9]+_[0-9]+$ ]] || { echo "bad log $L"; exit 1; }
  [[ " $PROTECT " == *" $L "* ]] && { echo "protected log $L"; exit 1; }
  [ -e "$B/$L" ] && { echo "$B/$L already exists - refusing"; exit 1; }
done
PAT=/root/VLA/navhard/_repair_patterns.txt
: > "$PAT"; for L in "$@"; do for c in CAM_F0 CAM_L1 CAM_R1; do echo "*/$L/$c/*" >> "$PAT"; done; done
rm -rf -- "$TMP"; mkdir -p -- "$TMP"
echo "######## $(date '+%F %T') shard $SHN: fetching $# logs"
wget -qO- "$HF/openscene_sensor_test_camera_$SHN.tgz" | tar -xz -C "$TMP" --wildcards -T "$PAT" || true
for L in "$@"; do
  src=$(find "$TMP" -type d -name "$L" | head -1)
  [ -n "$src" ] && [ -d "$src" ] || { echo "log $L not found in shard $SHN"; exit 1; }
  mv -- "$src" "$B/$L"
  echo "$L: $(find "$B/$L" -type f | wc -l) images"
done
rm -rf -- "$TMP" "$PAT"
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
CUDA_VISIBLE_DEVICES=${GPU:-1} python $POC/scripts/expanded_best_of_n.py --scenes /root/VLA/autovla/dataset/nuplan/navtest_ext \
  --tokens "$TOK" --output "$RUN" 2>&1 | grep -E "\[plan\]|\[done\]"
MISSING=$(python3 -c "import json;t=set(json.load(open('$TOK')));d={json.loads(l)['token'] for l in open('$RUN/records.jsonl')};print(len(t-d))")
echo "not decoded: $MISSING"
for L in "$@"; do rm -rf -- "$B/$L"; done
[ "$MISSING" = "0" ] && touch "$RUN/shard_$SHN.done"
echo "######## done shard $SHN"
