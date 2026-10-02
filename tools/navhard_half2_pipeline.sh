#!/bin/bash
# navhard second half (experiment 26, pre-registered): data -> 5090 decoding in log chunks (disk) -> flags -> picks -> scoring.
# Synthetic images are extracted one chunk of logs at a time and deleted (only the listed files) after every scene of
# the chunk is decoded. Original images go to /root/VLA/navhard/original_sensor_blobs (never into sensor_blobs/test).
set -uo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
POC=/root/VLA/autovla_misalignment_poc
V=$POC/outputs/navhard_full_validation/half2
export NAVHARD_SUBSET=$V/subset NAVHARD_JSON=/root/VLA/autovla/dataset/nuplan/navhard_half2_json
export NAVHARD_ORIG_DEST=/root/VLA/navhard/original_sensor_blobs
export NUPLAN_MAPS_ROOT=/root/VLA/autovla/dataset/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/root/VLA/navhard NAVSIM_EXP_ROOT=/root/VLA/navhard/exp
CACHE=/root/VLA/navhard/exp/metric_cache_half2
SYNB=/root/VLA/navhard/navhard_two_stage/sensor_blobs
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2
CH=$V/chunks
step() { echo; echo "######## $(date '+%F %T') $*"; }
die() { echo "STOP: $*"; exit 1; }
cd $POC
mkdir -p $CH $NAVHARD_JSON $NAVHARD_ORIG_DEST

step "chunks (3, by log) + overlap check with the first half"
python - <<'PY' || die "chunking"
import json, os
V = os.environ["NAVHARD_SUBSET"].rsplit("/", 1)[0]
g = json.load(open(f"{V}/subset/groups.json"))
files = [l.strip() for l in open(f"{V}/subset/synthetic_files.txt") if l.strip()]
first = {l.strip() for l in open("/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset/synthetic_files.txt") if l.strip()}
assert not (set(files) & first), "synthetic files shared with the first half"
by_log = {}
for f in files:
    by_log.setdefault(f.split("/")[0], []).append(f)
logs = sorted({x["log"] for x in g})
assert set(by_log) <= set(logs), set(by_log) - set(logs)
bins = [[], [], []]; size = [0, 0, 0]
for lg in sorted(logs, key=lambda l: -len(by_log.get(l, []))):
    k = size.index(min(size)); bins[k].append(lg); size[k] += len(by_log.get(lg, []))
s1 = json.load(open(f"{V}/subset/stage1_tokens.json"))
for k, b in enumerate(bins):
    syn = sorted({t for x in g if x["log"] in b for p in x["pairs"] for t in p})
    toks = (s1 if k == 0 else []) + syn
    json.dump(toks, open(f"{V}/chunks/tokens_{k}.json", "w"))
    fl = sorted(f for lg in b for f in by_log.get(lg, []))
    open(f"{V}/chunks/files_{k}.txt", "w").write("\n".join(fl) + "\n")
    open(f"{V}/chunks/members_{k}.txt", "w").write("\n".join("navhard_two_stage/sensor_blobs/" + f for f in fl) + "\n")
    print(f"chunk {k}: {len(b)} logs, {len(fl)} images (~{len(fl)*0.345/1000:.1f} GB), {len(toks)} tokens")
PY

step "original images (background) + metric caches"
bash /root/VLA/tools/fetch_navhard_originals.sh > $V/fetch_originals.log 2>&1 &
FETCH=$!
PYTHONPATH=/root/VLA/navsim_v2 python scripts/navhard_metric_cache.py --cache $CACHE --workers 3 2>&1 | grep -v -i warn | tail -3
wait $FETCH; tail -1 $V/fetch_originals.log

step "scene JSONs"
python scripts/navhard_scene_json.py | tail -1

for k in 0 1 2; do
  step "chunk $k: extract synthetic images"
  need=$(wc -l < $CH/files_$k.txt)
  free=$(df --output=avail -k / | tail -1)
  [ "$free" -gt $(( need * 345 + 1500000 )) ] || die "disk: ${free} KB free for $need images"
  for part in curr hist; do
    wget -qO- $HF/navsim_v2.2_navhard_two_stage_${part}_sensors.tar.gz | tar -xz -C /root/VLA/navhard -T $CH/members_$k.txt 2>> $CH/tar_$k.err
  done
  have=$(python3 -c "import os;fl=[l.strip() for l in open('$CH/files_$k.txt') if l.strip()];print(sum(os.path.exists('$SYNB/'+f) for f in fl))")
  echo "chunk $k images present $have / $need"; [ "$have" = "$need" ] || die "chunk $k images incomplete"
  step "chunk $k: decode on RTX 5090"
  CUDA_VISIBLE_DEVICES=1 python scripts/expanded_best_of_n.py --scenes $NAVHARD_JSON --tokens $CH/tokens_$k.json \
     --output $V/decode_5090 2>&1 | grep -E "\[plan\]|\[done\]"
  miss=$(python3 -c "import json;t=set(json.load(open('$CH/tokens_$k.json')));d={json.loads(l)['token'] for l in open('$V/decode_5090/records.jsonl')};print(len(t-d))")
  echo "chunk $k not decoded: $miss"; [ "$miss" = "0" ] || die "chunk $k decoding incomplete (images kept)"
  step "chunk $k: delete its synthetic images (listed files only)"
  python3 - "$SYNB" "$CH/files_$k.txt" <<'PY'
import os, sys
base, lst = os.path.realpath(sys.argv[1]), sys.argv[2]
assert base == "/root/VLA/navhard/navhard_two_stage/sensor_blobs"
n = 0; dirs = set()
for rel in (l.strip() for l in open(lst)):
    if not rel or rel.startswith("/") or ".." in rel.split("/"):
        continue
    p = os.path.join(base, rel)
    if os.path.isfile(p):
        os.remove(p); n += 1; dirs.add(os.path.dirname(p))
for d in sorted(dirs, key=len, reverse=True):
    while d.startswith(base + "/") and os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d); d = os.path.dirname(d)
print(f"removed {n} files")
PY
  df -h / | tail -1
done

n=$(wc -l < $V/decode_5090/records.jsonl); [ "$n" -eq 3000 ] || die "decoded $n / 3000"
export PYTHONPATH=/root/VLA/navsim_v2
step "safety flags"; python scripts/navhard_safety_flags.py $V/decode_5090 $V/safety_flags --workers 4 2>&1 | grep -E "\[plan\]|\[done\]"
step "picks"; python scripts/navhard_filter_picks.py $V/safety_flags/flags.jsonl $V/decode_5090 $V/filter | tail -12
step "submissions"; python scripts/navhard_submission.py $V/decode_5090 $V/submissions --picks $V/filter/picks.json 2>&1 | tail -1
while pgrep -f "fv_half1_" >/dev/null; do sleep 60; done
step "scoring"
run() { bash /root/VLA/tools/navhard_score_groups.sh $V/submissions/submission_$1.pkl $V/group_scores/g_$1.json navhard_half2 $CACHE fv_half2_$1 2>&1 | grep "Final extended" | sed "s/^/$1: /"; }
run normal & run ranksum & wait
run max_loglik & run F1 & wait
run F1_maxll
step "HALF2 DONE"
