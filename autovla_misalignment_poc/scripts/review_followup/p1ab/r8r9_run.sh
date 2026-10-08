#!/bin/bash
# R8R9 raw recovery: re-run exp 8 (prev_action_state_patching.py) and exp 9 (temporal_feedback_window.py)
# UNMODIFIED, into new directories, then build units.parquet + cluster statistics.
#   usage: r8r9_run.sh smoke|main
# smoke: exp8 1 scene x 2 units (--debug), exp9 1 scene x 3 units.
# main : exp8 all 1,408 units (B = 93 rows, as the original), exp8 sanity/debug (4 scenes x 2 units, as the original
#        sanity run), exp9 all 1,408 units (B = 17), then the ORIGINAL analysis scripts on the new dirs and the converter.
set -euo pipefail
MODE=${1:?smoke|main}
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
S=/root/VLA/autovla_misalignment_poc/scripts
P=$S/review_followup/p1ab
RUN=/root/VLA/autovla_misalignment_poc/outputs/review_followup/R8R9_raw_recovery/run_20261008
case "$MODE" in smoke) R=$RUN/smoke ;; main) R=$RUN ;; *) echo "bad mode"; exit 2 ;; esac
[[ "$R" == /root/VLA/autovla_misalignment_poc/outputs/review_followup/R8R9_raw_recovery/* ]] || exit 1
mkdir -p "$R/raw" "$R/logs"
for d in "$R/raw/exp8" "$R/raw/exp9"; do [ -e "$d/records.jsonl" ] && { echo "$d exists; refusing"; exit 1; }; done
wait_gpu() { local need=${1:-13000} ok=0; for _ in $(seq 1 360); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
  a=$(free -m | awk '/Mem:/{print $7}')
  if [ "$f" -ge "$need" ] && [ "$a" -ge 6000 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 6 ] && { echo "gpu free ${f} MiB, ram avail ${a} MiB $(date -u +%T)"; return 0; }; sleep 10; done; echo "gpu busy, abort"; exit 3; }
run_gpu() { local out=$1 log=$2; shift 2; local n
  for att in 1 2 3; do
    wait_gpu 13000
    if "$@" > "$log.attempt$att" 2>&1; then cp "$log.attempt$att" "$log"; return 0; fi
    n=$( { cat "$out"/records.jsonl 2>/dev/null || true; } | wc -l)
    if [ "$n" -gt 0 ]; then echo "FAILED after $n records in $out; stop"; exit 4; fi
    if [ -d "$out" ]; then mv "$out" "$out.failed_attempt${att}_$(date -u +%H%M%S)"; fi
    echo "attempt $att failed before any record; retrying"; sleep 60
  done; exit 5; }
{ date -u; free -m; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv; } > "$R/logs/resources_before.txt"
sha256sum "$S/prev_action_state_patching.py" "$S/temporal_feedback_window.py" "$S/analyze_prev_action_state_patching.py" \
  "$S/analyze_temporal_feedback_window.py" "$S/analyze_action_history.py" > "$R/logs/original_script_sha256.txt"
if [ "$MODE" = smoke ]; then
  wait_gpu 13000
  /usr/bin/time -v nice -n 19 python "$S/prev_action_state_patching.py" --limit 1 --max-alts 1 --debug --output "$R/raw/exp8" > "$R/logs/exp8.log" 2>&1
  wait_gpu 13000
  /usr/bin/time -v nice -n 19 python "$S/temporal_feedback_window.py" --limit 1 --max-alts 2 --output "$R/raw/exp9" > "$R/logs/exp9.log" 2>&1
else
  run_gpu "$R/raw/exp8" "$R/logs/exp8.log" nice -n 19 python "$S/prev_action_state_patching.py" --output "$R/raw/exp8"
  run_gpu "$R/raw/exp8/sanity" "$R/logs/exp8_sanity.log" nice -n 19 python "$S/prev_action_state_patching.py" --limit 4 --max-alts 1 --debug --output "$R/raw/exp8/sanity"
  run_gpu "$R/raw/exp9" "$R/logs/exp9.log" nice -n 19 python "$S/temporal_feedback_window.py" --output "$R/raw/exp9"
  # original analysis scripts, pointed at the NEW directories (they write summary/figures only there)
  nice -n 19 python "$S/analyze_prev_action_state_patching.py" --run "$R/raw/exp8" > "$R/logs/analyze_exp8.log" 2>&1 || echo "original exp8 analysis failed (see log)"
  nice -n 19 python "$S/analyze_temporal_feedback_window.py" --run "$R/raw/exp9" > "$R/logs/analyze_exp9.log" 2>&1 || echo "original exp9 analysis failed (see log)"
fi
nice -n 19 python "$P/r8r9_build_units.py" --raw "$R/raw" --out "$R" > "$R/logs/build_units.log" 2>&1
( cd "$R" && find raw -name '*.jsonl' -o -name '*.json' | sort | xargs sha256sum ) > "$R/logs/raw_sha256.txt"
date -u > "$R/logs/done"
