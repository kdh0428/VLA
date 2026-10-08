#!/bin/bash
# SINGLE MAIN-RUN LAUNCHER (lead launches after committing the protocols):
#   reference -> P1-A primary (t_p = t*) -> P1-A secondary fixed positions (t_p = 1, 3) -> P1-B -> R8R9
# Sequential model loads only; before every load waits until the shared RTX 5090 (PCI index 1) has >= 13 GB free
# and RAM >= 6 GB available (6 consecutive checks, 10 s apart, up to 2 h; OOM at model load retried up to 3 times); refuses to overwrite existing raw.
#   usage: bash run_all_main.sh            (all stages; a stage whose final output exists is skipped)
set -euo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
P=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1ab
O=/root/VLA/autovla_misalignment_poc/outputs/review_followup
A=$O/P1A_gt_free_reference/run_20261008
B=$O/P1B_start_position/run_20261008
for d in "$A" "$B"; do [[ "$d" == /root/VLA/autovla_misalignment_poc/outputs/review_followup/*/run_20261008 ]] || exit 1; done
mkdir -p "$A/logs" "$B/logs"
LOG=$O/run_all_main_20261008.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
wait_gpu() { local ok=0; for _ in $(seq 1 720); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
  a=$(free -m | awk '/Mem:/{print $7}')
  if [ "$f" -ge 13000 ] && [ "$a" -ge 6000 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 6 ] && { say "gpu free ${f} MiB, ram avail ${a} MiB"; return 0; }; sleep 10; done; say "gpu busy 2 h, abort"; exit 3; }

# run_gpu OUTDIR LOGFILE cmd...: up to 3 attempts. The shared 5090 can be taken by another process between the
# free-memory check and model load (seen twice in smoke). Retry ONLY if the failed attempt wrote no unit record
# (records.jsonl / references.jsonl absent or empty); the empty attempt dir is moved aside (never deleted).
# A failure after records were written stops the launcher (protocol §7: failed scenes are re-run manually).
run_gpu() { local out=$1 log=$2; shift 2; local n
  for att in 1 2 3; do
    wait_gpu
    if "$@" > "$log.attempt$att" 2>&1; then cp "$log.attempt$att" "$log"; return 0; fi
    n=$( { cat "$out"/records.jsonl "$out"/references.jsonl 2>/dev/null || true; } | wc -l)
    if [ "$n" -gt 0 ]; then say "FAILED after $n records in $out (see $log.attempt$att); stop"; exit 4; fi
    if [ -d "$out" ]; then mv "$out" "$out.failed_attempt${att}_$(date -u +%H%M%S)"; fi
    say "attempt $att failed before any record (likely OOM at load); retrying"; sleep 60
  done; say "3 attempts failed: $out"; exit 5; }

say "stage 1/5 reference"
if [ ! -f "$A/reference/manifest.json" ]; then
  run_gpu "$A/reference" "$A/logs/reference.log" nice -n 19 python "$P/p1_reference.py" --out "$A/reference"
fi

say "stage 2/5 P1-A primary (t_p = t*, families G + E)"
if [ ! -e "$A/raw/records.jsonl" ]; then
  run_gpu "$A/raw" "$A/logs/p1a.log" nice -n 19 python "$P/p1a_gt_free_reference.py" --out "$A/raw" --reference "$A/reference"
fi
nice -n 19 python "$P/p1a_analyze.py" --raw "$A/raw" --out "$A" > "$A/logs/p1a_analyze.log" 2>&1
cp "$A/raw/intervention_trace.jsonl" "$A/intervention_trace.jsonl"

say "stage 3/5 P1-A secondary fixed positions t_p in {1,3} (family G)"
if [ ! -e "$A/raw_fixedpos/records.jsonl" ]; then
  run_gpu "$A/raw_fixedpos" "$A/logs/p1a_fixedpos.log" nice -n 19 python "$P/p1a_gt_free_reference.py" --out "$A/raw_fixedpos" \
    --reference "$A/reference" --families G --fixed-positions 1,3
fi
nice -n 19 python "$P/p1a_analyze.py" --raw "$A/raw_fixedpos" --out "$A/fixedpos" > "$A/logs/p1a_fixedpos_analyze.log" 2>&1
nice -n 19 python "$P/p1_manifest.py" --run-dir "$A" --experiment P1A_gt_free_reference

say "stage 4/5 P1-B"
if [ ! -e "$B/raw/records.jsonl" ]; then
  run_gpu "$B/raw" "$B/logs/p1b.log" nice -n 19 python "$P/p1b_start_position.py" --out "$B/raw" --reference "$A/reference"
fi
nice -n 19 python "$P/p1b_analyze.py" --raw "$B/raw" --out "$B" > "$B/logs/p1b_analyze.log" 2>&1
cp "$B/raw/intervention_trace.jsonl" "$B/intervention_trace.jsonl"
nice -n 19 python "$P/p1_manifest.py" --run-dir "$B" --experiment P1B_start_position

say "stage 5/5 R8R9 (original exp 8 / exp 9 scripts, unmodified)"
bash "$P/r8r9_run.sh" main
nice -n 19 python "$P/p1_manifest.py" --run-dir "$O/R8R9_raw_recovery/run_20261008" --experiment R8R9_raw_recovery
say "all stages done"
