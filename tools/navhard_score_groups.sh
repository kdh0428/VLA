#!/bin/bash
# Official NAVSIM v2 two-stage scoring of one submission, saving per-group scores (navhard_group_scores.py).
#   bash tools/navhard_score_groups.sh <submission.pkl> <group_scores.json> <split> <metric_cache_dir> <experiment_name>
# e.g. split navhard_half / navhard_half2 (configs in autovla_misalignment_poc/configs/navsim_v2, copy into navsim_v2).
set -euo pipefail
SUB=$1; OUTJ=$2; SPLIT=$3; CACHE=$4; NAME=$5
[ -f "$SUB" ] && [ -d "$CACHE" ] || { echo "missing submission or cache"; exit 1; }
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export PYTHONPATH=/root/VLA/navsim_v2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUPLAN_MAPS_ROOT=/root/VLA/autovla/dataset/nuplan/maps NAVSIM_EXP_ROOT=/root/VLA/navhard/exp OPENSCENE_DATA_ROOT=/root/VLA/navhard
mkdir -p "$(dirname "$OUTJ")"
cd /root/VLA/navsim_v2
GROUP_SCORES_OUT="$OUTJ" python /root/VLA/autovla_misalignment_poc/scripts/navhard_group_scores.py \
  train_test_split="$SPLIT" \
  submission_file_path="$SUB" experiment_name="$NAME" \
  metric_cache_path="$CACHE" \
  navsim_log_path=/root/VLA/autovla/dataset/nuplan/navsim_logs/test \
  synthetic_scenes_path=/root/VLA/navhard/navhard_two_stage/synthetic_scene_pickles \
  worker=${WORKER:-single_machine_thread_pool}
