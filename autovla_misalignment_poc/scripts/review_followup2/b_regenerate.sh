#!/usr/bin/env bash
# B (review_followup2, 2026-10-13): regenerate every B output from the raw episode files. POST-HOC, CPU only.
# Outputs: outputs/review_followup2/B_spatialvla_direct/run_20261013/
set -euo pipefail
PY=/root/miniforge3/envs/autovla/bin/python
HERE=/root/VLA/autovla_misalignment_poc/scripts/review_followup2
[ -d "$HERE" ] || exit 1
cd "$HERE" || exit 1
nice -n 19 "$PY" b_extract.py              # episodes.csv, pairs.csv            (~10 s; streams the 315 MB P1-D file)
nice -n 19 "$PY" b_check_conditions.py     # condition_record_check.csv         (~1 min)
nice -n 19 "$PY" b_direct_effects.py       # DIRECT_EFFECTS.csv, B4_TRAJECTORY_VS_SUCCESS.csv, INTERPRETATION.csv (~12 min)
nice -n 19 "$PY" b_figures.py              # fig_direct_effects.png, fig_tcp_vs_success.png
