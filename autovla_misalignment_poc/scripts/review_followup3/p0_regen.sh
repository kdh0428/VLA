#!/usr/bin/env bash
# P0 (review_followup3, 2026-10-14) -- regenerate every P0 deliverable in outputs/review_followup3/P0/.
# POST-HOC re-analysis: CPU only, nice -n 19, no GPU, no model inference. Reads existing results; writes only to P0/.
#   bash /root/VLA/autovla_misalignment_poc/scripts/review_followup3/p0_regen.sh            # offline (uses cached metadata)
#   P0_REFETCH=1 bash .../p0_regen.sh                                                      # re-stream OpenScene metadata
set -euo pipefail
POC=/root/VLA/autovla_misalignment_poc
SCR="$POC/scripts/review_followup3"
OUT="$POC/outputs/review_followup3/P0"
PY=/root/miniforge3/envs/autovla/bin/python
[ -d "$SCR" ] && [ -x "$PY" ] || { echo "missing $SCR or $PY" >&2; exit 1; }
mkdir -p "$OUT/_logs" "$OUT/_external"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
run() { echo "[p0_regen] $*"; nice -n 19 "$PY" "$SCR/$1" "${@:2}" > "$OUT/_logs/${1%.py}${2:+_$2}.log" 2>&1; }

# external public metadata (network; cached copies are used when present)
if [ "${P0_REFETCH:-0}" = "1" ] || [ ! -s "$OUT/_external/openscene_v1.1_trainval_log_metadata.csv" ]; then
  run p0_d_openscene_log_metadata.py test
  run p0_d_openscene_log_metadata.py trainval
fi
if [ ! -s "$OUT/_external/openvla-7b-finetuned-libero-spatial_config.json" ]; then
  curl -sL --fail -o "$OUT/_external/openvla-7b-finetuned-libero-spatial_config.json" \
    https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial/resolve/962318cec55ac10993ff0f5f43eda9a270b4c873/config.json
fi

run p0_a_population.py            # P0-A  (asserts P1-C decisions.json reproduction)
run p0_b_spatial_task_effects.py  # P0-B  (asserts DIRECT_EFFECTS.csv reproduction)
run p0_c_reconcile.py             # P0-C  +4.37 m versions, document number check (p0_c_doc_checks.py)
run p0_c_window_weights.py        # P0-C  R9 / P1-B window, weights, denominators
run p0_d_dependence.py            # P0-D  dependence sensitivity (asserts 15/15 reproduction)
run p0_d_disjoint_inventory.py    # P0-D  P5 preparation: disjoint-log inventory
run p0_e_openvla.py               # P0-E  OpenVLA initial-state re-analysis (asserts analysis.json reproduction)
run p0_manifest_numbers.py        # DATA_MANIFEST.csv, numbers_p0.json
run p0_reports.py                 # markdown reports from the tables
sha256sum "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.md > "$OUT/_logs/SHA256SUMS.txt"
echo "[p0_regen] done -> $OUT"
