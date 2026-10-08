#!/usr/bin/env bash
# Re-runs the ORIGINAL analysis script (scripts/analyze_equal_distance.py, unchanged) on a COPY of
# outputs/equal_distance_perturbation/records.jsonl, so that nothing in the original output dir is touched.
# Purpose: (1) verify the committed summary.json reproduces; (2) obtain summary_strict.json (STRICT=1),
# which is referenced by EQUAL_DISTANCE_strict.md but is not present in the output directory.
# CPU only. Usage: rerun_equal_distance_analysis.sh <scratch_dir>
set -euo pipefail
S="${1:?scratch dir required}"
PY=/root/miniforge3/envs/autovla/bin/python
SCRIPT=/root/VLA/autovla_misalignment_poc/scripts/analyze_equal_distance.py
OUT=/root/VLA/paper_quantitative_package/_parts/recomputed
nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$SCRIPT" --run "$S/ed_full" > "$S/ed_full/console.txt"
nice -n 19 env STRICT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$SCRIPT" --run "$S/ed_strict" > "$S/ed_strict/console.txt"
cp "$S/ed_full/summary.json" "$OUT/equal_distance_summary_rerun.json"
cp "$S/ed_strict/summary_strict.json" "$OUT/equal_distance_summary_strict_rerun.json"
cp "$S/ed_full/EQUAL_DISTANCE.md" "$OUT/EQUAL_DISTANCE_rerun.md"
cp "$S/ed_strict/EQUAL_DISTANCE_strict.md" "$OUT/EQUAL_DISTANCE_strict_rerun.md"
