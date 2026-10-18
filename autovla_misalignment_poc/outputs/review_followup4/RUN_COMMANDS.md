# RUN_COMMANDS — review_followup4 (2026-10-18)

All paths are relative to `/root/VLA/autovla_misalignment_poc` unless absolute. Python = conda env `autovla`
(`/root/miniforge3/envs/autovla/bin/python`, Python 3.9.23, numpy 1.23.4, pandas 2.3.3, scipy 1.13.1, pyarrow 21.0.0;
torch 2.8.0+cu128, transformers 4.49 for GPU scripts). GPU scripts pin the RTX 5090 with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` and load the 16 GB checkpoint with `mmap=True` (14.4 GB RAM limit).
Every CPU command was run with `nice -n 19` (or `nice`).

Commits: **5523ce3** A2/A3 · **2a31ad7** A1/A4 · **552b3ac** B1 results + B2 protocol (preregistration, before the
main B2 runs) · **28f1bfd** FEASIBILITY · **c7d1bba** B2 results. The C/A5 finalisation files of 2026-10-18 (listed in
§5) are **not committed**.

## 1. CPU only (re-analysis of stored data; no model)

| item | commit | command (from `scripts/review_followup4/` unless noted) | outputs (`outputs/review_followup4/…`) |
|---|---|---|---|
| A1 verification of the CI methods | 2a31ad7 | `nice -n 19 $PY a1_paired_ci.py` (self-checks: exact trinomial = `stats_core.signflip_ci`, Newcombe vs ratesci reference, Tango MLE) | stdout |
| A1 coverage study + pre-specified method selection | 2a31ad7 | `nice -n 19 $PY a1_coverage_sim.py` (≈ 3 min; seed 20261017; does not read real outcomes) | `A1_A4/A1_METHOD_SELECTION.json`, `A1_COVERAGE_SIM_{iid,cluster}.csv`, `A1_TOST_SIZE_SIM.csv`, `a1_coverage_sim.log` |
| A1 real-data recomputation (old vs new) | 2a31ad7 | `nice -n 19 $PY a1_ci_audit.py` (reads the selection first) | `A1_A4/CI_VALIDITY_AUDIT.csv`, `A1_JUDGMENT_CHANGES.csv`, `A1_OPENVLA_2x2.csv`, `A1_CALLSITES.csv`, `A1_DEGENERACY_DEMO.json`, `a1_audit_meta.json` |
| A2 window semantics (code + traces + stored records) | 5523ce3 | `nice python3 a2_window_semantics.py` | `A2_A3/a2_checks.json`, `a2_step_trace_examples.csv`, `a2_exp9_unit_window.csv` |
| A2 transient vs persistent (P1-B, post hoc) | 5523ce3 | `nice $PY a2_p1b_transient_vs_persistent.py` | `A2_A3/a2_p1b_transient_vs_persistent.json` |
| A3 parity diagnosis of stored exp 33 data | 5523ce3 | `nice $PY a3_parity_diagnosis.py` | `A2_A3/IMPROMPTU_PARITY_DIAGNOSIS.csv`, `a3_parity_summary.json` |
| A4 cohort / estimand map and wording violations | 2a31ad7 | `nice -n 19 $PY a4_cohort_map.py` (asserts every count against its source table and every quoted text on its line) | `A1_A4/COHORT_ESTIMAND_MAP.{md,csv}`, `A4_WORDING_VIOLATIONS.csv` |
| B1 manifest (no outcomes read) | 552b3ac | `nice $PY b/b1_manifest.py` | `B/B1/b1_manifest.json`, `b1_manifest_take20.json` |
| B1 analysis | 552b3ac | `nice $PY b/b1_analyze.py` | `B/B1/NUMERICAL_SENSITIVITY.csv`, `numerical_sensitivity_rows.json` |
| B2 cohort / weights (no outcomes read) | 552b3ac | `nice $PY b/b2_cohort.py` | `B/B2/p1c_cohort_t4.json` |
| B2 P1-C image re-fetch (network, ≈ 90 min, 0.36 GB) | c7d1bba | `bash b/b2_fetch_p1c_images.sh all` | `B/B2/fetch/` (ledger `fetched_files.txt`) |
| B2 input check (stop rule 95%) | c7d1bba | `/usr/bin/python3 b/b2_p1c_input_check.py B/B2/p1c_main/bank.jsonl B/B2/p1c_main/input_check.json` | `B/B2/p1c_main/input_check.json` (2,160/2,160) |
| B2 analysis | c7d1bba | `nice $PY b/b2_analyze.py --run ../../outputs/review_followup4/B/B2/p1c_main --cohort p1c --cohort-manifest ../../outputs/review_followup4/B/B2/p1c_cohort_t4.json --out ../../outputs/review_followup4/B/B2/p1c_analysis`; `nice $PY b/b2_analyze.py --run ../../outputs/review_followup4/B/B2/dev_main --cohort dev --out ../../outputs/review_followup4/B/B2/dev_analysis` | `B/B2/{p1c,dev}_analysis/{MATCHED_PREFIX_RESULTS.csv, matched_prefix_descriptives.csv, unit_contrasts.parquet, analysis_meta.json}` |
| B2 combined table | c7d1bba | `nice $PY b/b2_report_tables.py` | `B/MATCHED_PREFIX_RESULTS.csv` (+ markdown appendix of `MATCHED_PREFIX_RESULTS.md`) |
| §6 feasibility (code / metadata reading only) | 28f1bfd | no script; sources and commits listed in `FEASIBILITY.md` §0 | `FEASIBILITY.md` |

## 2. GPU (RTX 5090) — verification runs

| item | commit | command | cost |
|---|---|---|---|
| A3 same-condition parity runs (Impromptu, 4 scenes) | 5523ce3 | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nice $PY scripts/review_followup4/a3_parity_run.py outputs/review_followup4/A2_A3/a3_parity_run --tokens 00137a5ce92d591c,003a1fdca9505c09,008844070af456e3,00abc794f8ce5195` | ≈ 2 min, peak 17.7 GB |
| A3 unmodified exp_cde rerun (1 scene) | 5523ce3 | `(cd scripts/cross_vla_temporal && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nice $PY exp_cde.py ../../outputs/review_followup4/A2_A3/a3_exp_cde_rerun_limit1 --limit 1)` | ≈ 1 min |
| B1 numerical sensitivity | 552b3ac | `bash scripts/review_followup4/b/run_b1.sh` (waits for ≥ 14 GB free via `b/gpu_wait.sh`; runs `b/b1_runner.py --manifest outputs/review_followup4/B/B1/b1_manifest_take20.json --out outputs/review_followup4/B/B1/run`) | 18.3 min, peak 25.7 GB |
| B2 new-arm gate | 552b3ac | `bash scripts/review_followup4/b/run_b2.sh gate` (→ `B/B2/gate_stage0/`, `gate_judgement.json` PASS) | minutes |
| B2 throughput pilot (no cell outcomes) | 552b3ac | `bash scripts/review_followup4/b/run_b2.sh pilot` (→ `B/B2/pilot_throughput/`) | minutes |

## 3. GPU (RTX 5090) — new effect experiment (B2, after the protocol commit 552b3ac)

| item | commit | command | cost |
|---|---|---|---|
| B2 dev main | c7d1bba | `bash scripts/review_followup4/b/run_b2.sh dev` (→ `B/B2/dev_main/`, log `B/B2/dev_main.log`) | 52.9 min |
| B2 P1-C main + input check | c7d1bba | `bash scripts/review_followup4/b/run_b2_p1c_chain.sh` (runs `run_b2.sh p1c`, then `b2_p1c_input_check.py`; log `B/B2/p1c_main.log`, `p1c_chain.log`) | ≈ 86 min |

The model code is `b/b2_matched_prefix.py` (cells) and `b/b1_runner.py` (B1), both re-using the P1-verified engine
`scripts/review_followup3/p1/p1b_engine.py` unchanged.

## 4. Not run

- **B3** OpenVLA minimal path gate: NOT_RUN (user decision 2026-10-17; checkpoint not on disk) — `OPENVLA_GATE.md`.
- review_followup3 **P3 / P4**: NOT_RUN (historical CP2 decision); **P5**: blocked (no disjoint logs).
- π₀-FAST: no download of weights, no inference.

## 5. Finalisation (C, A5) — CPU only, this round (2026-10-18, uncommitted)

Run from `scripts/review_followup4/c/` in this order:

    nice -n 19 $PY c_numbers.py      # numbers.json + numbers_sources.csv: +1,532 RF4.* rows, rf4_note on changed rows
    nice -n 19 $PY c_tables.py       # tables.tex: rf3 block captions corrected; new rf4 block (Tables 12-14)
    nice -n 19 $PY c_build_zip.py    # outputs/review_followup4/reproducibility_minimal.zip (no staging directory)
    # regeneration check from an unzipped copy (scratchpad), then deleted:
    python3 -m zipfile -e ../../../outputs/review_followup4/reproducibility_minimal.zip /tmp/claude-0/-root-VLA/3d5b895f-ebec-4409-a7c0-03065c2dc74b/scratchpad/repro_check
    bash /tmp/claude-0/-root-VLA/3d5b895f-ebec-4409-a7c0-03065c2dc74b/scratchpad/repro_check/regen.sh
    nice -n 19 $PY c_repro_record.py # copies REGEN_RESULTS into outputs/review_followup4/REPRO_CHECK.{csv,md}
    rm -rf /tmp/claude-0/-root-VLA/3d5b895f-ebec-4409-a7c0-03065c2dc74b/scratchpad/repro_check
    nice -n 19 $PY c_doc_fixes.py    # EXPERIMENT_SUMMARY §6g + in-place A4/A1/A2/A3/B2/B3 corrections; paper package sections
    nice -n 19 $PY c_build_zip.py    # rebuild so the ZIP carries the final documents (data files unchanged; checked by sha)
    nice -n 19 $PY c_source_manifest.py   # outputs/review_followup4/SOURCE_MANIFEST.csv

Files: `c_numbers.py`, `c_tables.py`, `c_doc_fixes.py` (+ `doc_sections/*.md`), `c_build_zip.py`
(+ `zip_text/{README.md, MISSING.md, PINS.csv, regen.sh, regen_compare.py}`), `c_repro_record.py`, `c_source_manifest.py`.

## 6. Regenerating the core tables from the ZIP alone

    python -m zipfile -e reproducibility_minimal.zip <dir>
    PY=/root/miniforge3/envs/autovla/bin/python bash <dir>/regen.sh          # FAST=1 skips the A1 coverage simulation

Targets: A1 coverage selection, A1 corrected CIs, B2 (P1-C, dev), P0-A population effects, P2 motion baselines.
Results of the run on this machine: `REPRO_CHECK.md`.
