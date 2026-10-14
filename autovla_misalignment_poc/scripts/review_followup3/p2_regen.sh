#!/usr/bin/env bash
# P2 (review_followup3) full regeneration, CPU only.  Outputs -> outputs/review_followup3/P2/
set -euo pipefail
cd /root/VLA/autovla_misalignment_poc/scripts/review_followup3 || exit 1
PY="nice -n 19 /root/miniforge3/envs/autovla/bin/python"
# P2-A (spec locked: P2/MOTION_BASELINE_SPEC_LOCKED_v1.md + spec_lock_v1.json)
$PY p2_cm_audit_reproduce.py          # CM definition check + exact C1 reproduction
$PY p2_tests.py                       # unit tests (T1-T8)
$PY p2_motion_baselines.py            # CM / CV0 / CTRV-K3 / CTRA-2 / Geo, Delta_model - Delta_b
# P2-B (protocol locked: P2/LEARNED_EXTRAPOLATOR_PROTOCOL.md + extrap_protocol_lock.json)
$PY p2_extrap_data.py stream          # streams OpenScene v1.1 trainval metadata (7.05 GB network, ~56 MB kept)
$PY p2_extrap_data.py samples         # drive split + GT->token projection
for c in "256 1e-3" "512 1e-3" "256 3e-4" "512 3e-4"; do $PY p2_extrap_train.py grid $c; done
$PY p2_extrap_train.py select
for v in prefix prefix_ego; do for s in 0 1 2; do $PY p2_extrap_train.py final $v $s; done; done
$PY p2_extrap_eval.py
$PY p2_figure.py
$PY p2_reports.py
