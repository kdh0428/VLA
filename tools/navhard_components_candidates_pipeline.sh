#!/bin/bash
# P2 (safety-filter components) + P3 (every candidate scored officially, for the oracle) on both navhard halves.
# CPU only; decodes and flags of experiments 25-26 (RTX 5090) are reused read-only.
set -uo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
POC=/root/VLA/autovla_misalignment_poc; C=$POC/outputs/safety_filter_components; O=$POC/outputs/candidate_oracle
export PYTHONPATH=/root/VLA/navsim_v2
cd $POC
step() { echo; echo "######## $(date '+%F %T') $*"; }
for H in half1 half2; do
  if [ $H = half1 ]; then DEC=outputs/navhard_eval/decode_5090; FL=outputs/navhard_eval/safety_flags/flags.jsonl; SUBSET=outputs/navhard_eval/subset
  else DEC=outputs/navhard_full_validation/half2/decode_5090; FL=outputs/navhard_full_validation/half2/safety_flags/flags.jsonl; SUBSET=outputs/navhard_full_validation/half2/subset; fi
  step "$H picks + submissions"
  python scripts/navhard_component_oracle_picks.py $FL $DEC $O/$H/picks
  mkdir -p $C/$H/picks; cp $O/$H/picks/component_picks.json $C/$H/picks/
  NAVHARD_SUBSET=$POC/$SUBSET python scripts/navhard_submission.py $DEC $C/$H/submissions --picks $C/$H/picks/component_picks.json 2>&1 | tail -1
  NAVHARD_SUBSET=$POC/$SUBSET python scripts/navhard_submission.py $DEC $O/$H/submissions --picks $O/$H/picks/candidate_picks.json 2>&1 | tail -1
done
run() {   # dir half rule
  if [ $2 = half1 ]; then SPLIT=navhard_half; CACHE=/root/VLA/navhard/exp/metric_cache_half; else SPLIT=navhard_half2; CACHE=/root/VLA/navhard/exp/metric_cache_half2; fi
  bash /root/VLA/tools/navhard_score_groups.sh $1/$2/submissions/submission_$3.pkl $1/$2/group_scores/g_$3.json $SPLIT $CACHE $(basename $1)_$2_$3 2>&1 \
    | grep "Final extended" | sed "s/^/$2 $3: /"
}
JOBS=()
for H in half1 half2; do for R in comp_collision comp_dac; do JOBS+=("$C $H $R"); done; done
for H in half1 half2; do for k in $(seq -w 1 16); do JOBS+=("$O $H cand$k"); done; done
step "scoring ${#JOBS[@]} runs (4 at a time)"
P=${PAR:-4}
for ((i=0; i<${#JOBS[@]}; i+=P)); do
  for ((j=0; j<P; j++)); do [ $((i+j)) -lt ${#JOBS[@]} ] && run ${JOBS[$((i+j))]} & done; wait
done
step "COMPONENTS+CANDIDATES DONE"
