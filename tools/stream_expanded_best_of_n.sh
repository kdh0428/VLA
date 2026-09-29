#!/bin/bash
# 실험 20: navtest 카메라 shard를 하나씩 받아 -> 전처리 -> best-of-N 디코딩 -> 이미지 삭제 (디스크 절약 스트리밍)
#   bash tools/stream_expanded_best_of_n.sh <GPU 0|1> <shard> [<shard> ...]
# 결과: autovla_misalignment_poc/outputs/expanded_best_of_n/gpu<GPU>/records.jsonl (재실행 시 이어서 진행)
# shard 0-5 (PoC 28개 log)는 기존 실험이 쓰므로 지우지 않는다. 이 스크립트에는 6 이상만 넘길 것.
set -uo pipefail
GPU=$1; shift
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
if [ "$GPU" = 0 ]; then export CUDA_VISIBLE_DEVICES=0 VLA_LOW_MEM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else export CUDA_VISIBLE_DEVICES=1; fi
ROOT=/root/VLA
NUP=$ROOT/autovla/dataset/nuplan
POC=$ROOT/autovla_misalignment_poc
OUT=$POC/outputs/${STREAM_OUT:-expanded_best_of_n}/gpu$GPU
WORK=$NUP/_stream_gpu$GPU
OS=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
mkdir -p $OUT $WORK $NUP/navtest_ext
for SH in "$@"; do
  [ "$SH" -lt 6 ] && { echo "refusing shard $SH (< 6)"; continue; }
  [ -f $OUT/shard_$SH.done ] && { echo "shard $SH already done"; continue; }
  echo "######## $(date '+%F %T') shard $SH: download"
  rm -rf $WORK/openscene-v1.1
  wget -qO- $OS/openscene_sensor_test_camera/openscene_sensor_test_camera_$SH.tgz | tar -xz -C $WORK || { echo "download failed $SH"; continue; }
  src=$WORK/openscene-v1.1/sensor_blobs; [ -d $src/test ] && src=$src/test
  LOGS=$(ls $src)
  MOVED=""
  for L in $LOGS; do if [ ! -e $NUP/sensor_blobs/test/$L ]; then mv $src/$L $NUP/sensor_blobs/test/ && MOVED="$MOVED $L"; fi; done
  [ "$SH" -ge 6 ] && MOVED="$LOGS"      # shards >= 6 never belong to the PoC's 28 logs (shards 0-5)
  rm -rf $WORK/openscene-v1.1
  echo "######## $(date '+%F %T') shard $SH: preprocess ($(echo $LOGS | wc -w) logs)"
  python - $WORK/filter_$SH.yaml $LOGS <<'PY'
import sys, yaml
out, logs = sys.argv[1], sys.argv[2:]
nt = yaml.safe_load(open("/root/VLA/autovla/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
nt["log_names"] = [l for l in nt["log_names"] if l in set(logs)]
yaml.safe_dump(nt, open(out, "w"), sort_keys=False)
print("navtest logs in shard:", len(nt["log_names"]))
PY
  python $POC/scripts/preprocess_scenes.py --scene-filter $WORK/filter_$SH.yaml --out $NUP/navtest_ext --anno-out $WORK/anno > $WORK/pre_$SH.log 2>&1 \
    || { echo "preprocess failed for shard $SH (kept images, not marked done)"; continue; }
  tail -1 $WORK/pre_$SH.log
  python - $NUP/navtest_ext $WORK/tokens_$SH.json $LOGS <<'PY'
import sys, os, json
d, out, logs = sys.argv[1], sys.argv[2], set(sys.argv[3:])
toks = sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and json.load(open(os.path.join(d, f)))["front_camera_paths"][0].split("/")[0] in logs)
json.dump(toks, open(out, "w")); print("scenes:", len(toks))
PY
  echo "######## $(date '+%F %T') shard $SH: best-of-N"
  python $POC/scripts/expanded_best_of_n.py --scenes $NUP/navtest_ext --tokens $WORK/tokens_$SH.json --output $OUT > $WORK/bon_$SH.log 2>&1 \
    || { echo "decoding failed for shard $SH (kept images, not marked done)"; continue; }
  tail -1 $WORK/bon_$SH.log
  MISSING=$(python - $WORK/tokens_$SH.json $OUT/records.jsonl <<'PY'
import sys, json
t = set(json.load(open(sys.argv[1]))); d = {json.loads(l)["token"] for l in open(sys.argv[2])}
print(len(t - d))
PY
)
  [ "$MISSING" != "0" ] && { echo "shard $SH: $MISSING scenes not decoded (kept images, not marked done)"; continue; }
  for L in $MOVED; do rm -rf $NUP/sensor_blobs/test/$L; done      # only logs this run added
  touch $OUT/shard_$SH.done
  df -h / | tail -1
done
echo "######## STREAM GPU$GPU DONE"
