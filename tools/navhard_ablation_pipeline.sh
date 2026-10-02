#!/bin/bash
# Safety-filter ablation (experiment 27): new picks + official group scoring on both navhard halves (CPU only;
# decodes are the RTX 5090 runs of experiments 25-26, read-only).
set -uo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
POC=/root/VLA/autovla_misalignment_poc; A=$POC/outputs/safety_filter_ablation
export PYTHONPATH=/root/VLA/navsim_v2
cd $POC
step() { echo; echo "######## $(date '+%F %T') $*"; }
RULES="filter_only filter_random_s0 filter_random_s1 filter_random_s2 filter_ranksum_natfb filter_maxll_natfb"
for H in half1 half2; do
  if [ $H = half1 ]; then DEC=outputs/navhard_eval/decode_5090; FL=outputs/navhard_eval/safety_flags/flags.jsonl; SUBSET=outputs/navhard_eval/subset
  else DEC=outputs/navhard_full_validation/half2/decode_5090; FL=outputs/navhard_full_validation/half2/safety_flags/flags.jsonl; SUBSET=outputs/navhard_full_validation/half2/subset; fi
  step "$H picks + submissions"
  python scripts/navhard_ablation_picks.py $FL $DEC $A/$H/picks
  NAVHARD_SUBSET=$POC/$SUBSET python scripts/navhard_submission.py $DEC $A/$H/submissions --picks $A/$H/picks/picks.json 2>&1 | tail -1
done
step "scoring (3 at a time)"
run() {   # half rule
  if [ $1 = half1 ]; then SPLIT=navhard_half; CACHE=/root/VLA/navhard/exp/metric_cache_half; else SPLIT=navhard_half2; CACHE=/root/VLA/navhard/exp/metric_cache_half2; fi
  bash /root/VLA/tools/navhard_score_groups.sh $A/$1/submissions/submission_$2.pkl $A/$1/group_scores/g_$2.json $SPLIT $CACHE abl_$1_$2 2>&1 \
    | grep "Final extended" | sed "s/^/$1 $2: /"
}
JOBS=(); for H in half1 half2; do for R in $RULES; do JOBS+=("$H $R"); done; done
for ((i=0; i<${#JOBS[@]}; i+=3)); do
  for j in 0 1 2; do [ $((i+j)) -lt ${#JOBS[@]} ] && run ${JOBS[$((i+j))]} & done; wait
done
step "ABLATION DONE"
