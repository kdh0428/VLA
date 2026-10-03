#!/bin/bash
# P4: teacher-force the recorded candidates of one navtest shard. The shard's CAM_F0/L1/R1 images (deleted after the
# original decoding) are re-fetched, the scenes are teacher-forced on the RTX 5090, then ONLY the log dirs this script
# moved in are deleted again. Paths are absolute and validated; PoC logs and pre-existing log dirs are never touched.
#   bash tools/stream_teacher_force.sh <shard_number> <run_dir> <tf_out_dir>
set -euo pipefail
SHN=$1; RUN=$2; OUT=$3
[[ "$SHN" =~ ^[0-9]+$ ]] && [ "$SHN" -ge 6 ] || { echo "bad shard $SHN"; exit 1; }
B=/root/VLA/autovla/dataset/nuplan/sensor_blobs/test
TMP=/root/VLA/navhard/_tf_tmp
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
POC=/root/VLA/autovla_misalignment_poc
TOK=/root/VLA/autovla/dataset/nuplan/_stream_gpu1/tokens_$SHN.json
[ -d "$B" ] && [ -f "$TOK" ] && [ -f "$RUN/records.jsonl" ] || { echo "missing input"; exit 1; }
mkdir -p "$OUT"
LIST=$OUT/_tokens_$SHN.json
python3 - "$TOK" "$RUN" "$LIST" <<'PY'
import json, sys
tok, run, out = sys.argv[1:4]
t = set(json.load(open(tok))); have = {json.loads(l)["token"] for l in open(run + "/records.jsonl")}
json.dump(sorted(t & have), open(out, "w"))
PY
LOGS=$(python3 -c "
import json
ts=json.load(open('$LIST'))
print(' '.join(sorted({json.load(open('/root/VLA/autovla/dataset/nuplan/navtest_ext/'+t+'.json'))['front_camera_paths'][0].split('/')[0] for t in ts})))")
PROTECT=$(python3 -c "import yaml;print(' '.join(yaml.safe_load(open('$POC/configs/scene_filter_navtest_subset.yaml'))['log_names']))")
FETCH=""
for L in $LOGS; do
  [[ "$L" =~ ^20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9.]+_veh-[0-9]+_[0-9]+_[0-9]+$ ]] || { echo "bad log $L"; exit 1; }
  [[ " $PROTECT " == *" $L "* ]] && { echo "protected log $L in shard $SHN"; exit 1; }
  if [ -e "$B/$L" ]; then echo "$L already present - kept as is"; else FETCH="$FETCH $L"; fi
done
echo "######## $(date '+%F %T') shard $SHN: $(python3 -c "import json;print(len(json.load(open('$LIST'))))") scenes, fetch:$FETCH"
MOVED=""
cleanup() {   # always remove the log dirs this run moved in (also when teacher forcing fails or is killed)
  for L in $MOVED; do
    [[ "$L" =~ ^20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9.]+_veh-[0-9]+_[0-9]+_[0-9]+$ ]] && [[ " $PROTECT " != *" $L "* ]] && rm -rf -- "${B:?}/${L:?}"
  done
  rm -rf -- "${TMP:?}"
}
trap cleanup EXIT
if [ -n "$FETCH" ]; then
  PAT=$OUT/_patterns_$SHN.txt; : > "$PAT"
  for L in $FETCH; do for c in CAM_F0 CAM_L1 CAM_R1; do echo "*/$L/$c/*" >> "$PAT"; done; done
  rm -rf -- "${TMP:?}"; mkdir -p -- "$TMP"
  wget -qO- "$HF/openscene_sensor_test_camera_$SHN.tgz" | tar -xz -C "$TMP" --wildcards -T "$PAT" || true
  for L in $FETCH; do
    src=$(find "$TMP" -type d -name "$L" | head -1)
    [ -n "$src" ] && [ -d "$src" ] || { echo "log $L not found in shard $SHN"; continue; }
    mv -- "$src" "$B/$L"; MOVED="$MOVED $L"
  done
  rm -rf -- "${TMP:?}" "$PAT"
fi
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
cd $POC
CUDA_VISIBLE_DEVICES=1 VLA_LOW_MEM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/teacher_force_candidates.py "$RUN" "$OUT" --tokens "$LIST" 2>&1 | grep -E "\[plan\]|\[done\]" || echo "teacher forcing failed"
echo "######## shard $SHN done; errors so far: $(cat "$OUT/errors.jsonl" 2>/dev/null | wc -l)"
