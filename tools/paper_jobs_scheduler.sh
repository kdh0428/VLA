#!/bin/bash
# Follow-up jobs for P1-P5 (RTX 5090 only), started when the CPU scoring of P2/P3 has finished (host RAM is the limit).
set -uo pipefail
S=/tmp/claude-0/-root-VLA/3d5b895f-ebec-4409-a7c0-03065c2dc74b/scratchpad
POC=/root/VLA/autovla_misalignment_poc; O=$POC/outputs/candidate_oracle
step() { echo; echo "######## $(date '+%F %T') $*"; }
until grep -q "COMPONENTS+CANDIDATES DONE" $S/compcand.log; do sleep 60; done
step "oracle picks + oracle scoring (2 at a time; before the GPU jobs, host RAM)"
(
  source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla; cd $POC; export PYTHONPATH=/root/VLA/navsim_v2
  csv() { ls /root/VLA/navhard/exp/$1/*/*.csv | head -1; }
  for H in half1 half2; do
    if [ $H = half1 ]; then DEC=outputs/navhard_eval/decode_5090; SUBSET=outputs/navhard_eval/subset; SPLIT=navhard_half; CACHE=/root/VLA/navhard/exp/metric_cache_half
    else DEC=outputs/navhard_full_validation/half2/decode_5090; SUBSET=outputs/navhard_full_validation/half2/subset; SPLIT=navhard_half2; CACHE=/root/VLA/navhard/exp/metric_cache_half2; fi
    CS="$(csv fv_${H}_normal)"; for k in $(seq -w 1 16); do CS="$CS $(csv candidate_oracle_${H}_cand$k)"; done
    python scripts/navhard_oracle_picks.py $O/$H/oracle $CS
    NAVHARD_SUBSET=$POC/$SUBSET python scripts/navhard_submission.py $DEC $O/$H/oracle_submissions --picks $O/$H/oracle/oracle_picks.json 2>&1 | tail -1
    for R in oracle17 oracle16; do
      bash /root/VLA/tools/navhard_score_groups.sh $O/$H/oracle_submissions/submission_$R.pkl $O/$H/group_scores/g_$R.json $SPLIT $CACHE candidate_oracle_${H}_$R 2>&1 | grep "Final extended" | sed "s/^/$H $R: /" &
    done; wait
  done
  echo "ORACLE DONE"
) > $S/oracle.log 2>&1
step "P5 motion semantics (RTX 5090)"
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
cd $POC; export CUDA_VISIBLE_DEVICES=1 VLA_LOW_MEM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/motion_semantics_ablation.py 2>&1 | grep -E "\[plan\]|\[tokens\]|\[prog\]|Error" | tail -5
step "P4 teacher forcing (RTX 5090), shard by shard (images re-fetched and deleted again)"
T=$POC/outputs/instability_detection_baselines/teacher_forced
for SH in $(seq 6 31); do
  if   [ $SH -le 13 ]; then R=expanded_best_of_n/gpu1
  elif [ $SH -le 17 ]; then R=expanded_best_of_n_5090/gpu1
  elif [ $SH -le 25 ]; then R=heldout_best_of_n/gpu1
  else R=heldout_best_of_n_5090/gpu1; fi
  bash /root/VLA/tools/stream_teacher_force.sh $SH $POC/outputs/$R $T/${R//\//_} 2>&1 | grep -E "########|\[plan\]|\[done\]|failed|not found"
done
step "SCHEDULER GPU DONE"
until grep -q "OV AB DONE" $S/ov_AB.log; do sleep 60; done
step "P1 phase C: OpenVLA within-step token feedback (RTX 5090)"
source /root/VLA/openvla/env.sh; cd /root/VLA/autovla_misalignment_poc/scripts/cross_vla
python openvla_token_feedback.py $POC/outputs/cross_vla_replication/rollouts/phaseA $POC/outputs/cross_vla_replication/token_feedback 2>&1 | grep -E "^\[plan|^\[done|Error"
step "PHASE C DONE"
