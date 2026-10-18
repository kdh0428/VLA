#!/usr/bin/env bash
# Regenerate the core statistics tables from the self-contained inputs of this ZIP and compare them with the
# originals shipped in the same ZIP (review_followup4 A5, 2026-10-18). CPU only; no model, no GPU, no network.
#
#   unzip reproducibility_minimal.zip -d <dir>     (or python -m zipfile -e reproducibility_minimal.zip <dir>)
#   bash <dir>/regen.sh                            (env PY=<python 3.9 with numpy/pandas/scipy/pyarrow>; FAST=1 skips
#                                                   the 5-minute A1 coverage simulation)
#
# Step 1 rewrites the hard-coded /root/VLA prefix in the shipped code (files inside <dir> only) so that every input is
# read from <dir>/VLA. Step 2 (regen_compare.py) moves each target output aside, re-runs the script that produced it,
# and compares the new file with the original (exact bytes, then cell by cell with rtol 1e-9).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V="$HERE/VLA"
if [ ! -d "$V/autovla_misalignment_poc/scripts" ] || [ "$V" = "/root/VLA" ]; then
  echo "run regen.sh from the unzipped folder (not from the live repository)" >&2
  exit 1
fi
PY="${PY:-/root/miniforge3/envs/autovla/bin/python}"
grep -rlZ --include='*.py' --include='*.sh' '/root/VLA' "$V" | xargs -0 -r sed -i "s#/root/VLA#$V#g"
if grep -rq --include='*.py' '/root/VLA' "$V"; then
  echo "path rewrite incomplete" >&2
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 19 "$PY" "$HERE/regen_compare.py" "$V" "$PY"
