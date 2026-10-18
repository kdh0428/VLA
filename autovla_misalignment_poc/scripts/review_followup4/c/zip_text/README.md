# reproducibility_minimal.zip — AutoVLA action-context study (review_followup4, 2026-10-18)

This ZIP lets a third party recompute the core statistics tables of the study from the shipped inputs, and read every
protocol, manifest, judge log and unit-level table the reports cite. It does **not** contain model checkpoints, camera
images, raw logits or raw rollouts (see `MISSING.md`).

Built by `VLA/autovla_misalignment_poc/scripts/review_followup4/c/c_build_zip.py`. `MANIFEST.csv` lists every file with
its byte size, sha256 and original repository path; `BUILD_INFO.json` records the build time and git HEAD;
`PINS.csv` pins the public model / data revisions and checksums.

## 1. Which document wins (precedence)

1. **Final status of this round:** `VLA/autovla_misalignment_poc/outputs/review_followup4/STATUS.md`,
   `CLAIM_CHANGES.md`, `MANUSCRIPT_REVISION.md` (2026-10-18). They supersede everything below where they conflict.
2. **Numbers:** `VLA/paper_quantitative_package/numbers.json` (rows `RF4.*` added 2026-10-18; original rows unchanged,
   rows whose use changed carry `rf4_note`). Where a report and `numbers.json` disagree, `numbers.json` wins, and the
   review_followup4 post-hoc corrections (`RF4.A1.*`) win over the original interval columns for inference.
3. **Final review_followup3 status:** `VLA/review_followup3/STATUS.md`, `CLAIM_EVIDENCE_MATRIX.md`, `REVIEW_COVERAGE.md`
   (2026-10-15), as amended by (1).
4. **Historical:** `VLA/review_followup3/CP2_REPORT.md` (CP2 submission, commit d70d19c) and `CP2_DECISION.md`.
   CP2's decision not to run P3 / P4 is a historical record; the review_followup4 B2 run is a separate new round and does
   not turn P3 into DONE.

**PROVISIONAL / PENDING strings that remain are history.** Several review_followup3 files were written while the
AutoVLA no-op/cache gate or the SpatialVLA A5 decision were still open, and keep the status that was true at run time:
`outputs/review_followup3/P0/{POPULATION_*.csv, DEPENDENCE_SENSITIVITY.csv, WINDOW_WEIGHT_AUDIT.csv, DATA_MANIFEST.csv,
numbers_p0.json, _logs/*}`, `outputs/review_followup3/P2/{MOTION_BASELINE_RESULTS.csv, LEARNED_EXTRAPOLATOR_RESULTS.csv,
*_run_meta.json}`, the protocols `P1/PROTOCOL_P1.md` and `P2/LEARNED_EXTRAPOLATOR_PROTOCOL.md`, and the reports in
`VLA/review_followup3/{P0,P2}/*.md`, `CP2_REPORT.md`. The final statuses were assigned afterwards by
`scripts/review_followup3/cp2/cp2_numbers.py` from the gate judge logs: **0 rows are PROVISIONAL or PENDING** in
`numbers.json` (all six AutoVLA paths PASS; SpatialVLA split-fingerprint C2 PASS 20/20). Do not read the run-time
strings as current status.

## 2. Regenerate the core tables (CPU only)

    python -m zipfile -e reproducibility_minimal.zip <dir>      # or: unzip reproducibility_minimal.zip -d <dir>
    PY=/path/to/python3.9 bash <dir>/regen.sh                  # FAST=1 skips the 3-minute A1 coverage simulation

Requirements: Python 3.9 with numpy 1.23.4, pandas 2.3.3, scipy 1.13.1, pyarrow 21.0.0 (the versions used; conda env
`autovla`). No GPU, model or network is needed.

`regen.sh` rewrites the hard-coded `/root/VLA` prefix of the shipped code to `<dir>/VLA` (files inside `<dir>` only),
moves each target output to `<dir>/_original_outputs/`, re-runs the producing script on the shipped inputs and compares
the new file with the original (exact bytes first, then cell by cell, rtol 1e-9). Results: `<dir>/REGEN_RESULTS.csv`,
logs `<dir>/regen_*.log`.

| target | script | regenerates |
|---|---|---|
| A1_coverage_selection | `scripts/review_followup4/a1_coverage_sim.py` | coverage study and the pre-specified CI method selection |
| A1_corrected_CIs | `scripts/review_followup4/a1_ci_audit.py` | 984 old-vs-corrected paired CI rows, TOST / Holm, 36 judgment changes, OpenVLA paired 2×2 tables |
| B2_p1c, B2_dev | `scripts/review_followup4/b/b2_analyze.py` | matched-prefix contrasts (F / S / P(E4); dev) from the stored cell outcomes |
| P0A_population | `scripts/review_followup3/p0_a_population.py` | selection flow 2,747 → 52 A− / 156 A+, A− frequencies, F and evaluation-frame (P) effects, P = F + S |
| P2_motion_baselines | `scripts/review_followup3/p2_motion_baselines.py` | motion-baseline table (CM / CV0 / CTRV-K3 / CTRA-2 / Geo) from P1-A rollouts |

The result of running exactly this procedure on the build machine is in
`VLA/autovla_misalignment_poc/outputs/review_followup4/REPRO_CHECK.{md,csv}`.

## 3. What is inside (by request item)

| item | where (under `VLA/`) |
|---|---|
| analysis / statistics / intervention code, every round | `autovla_misalignment_poc/scripts/**` (review_followup, review_followup2, review_followup3 incl. `p1/` gate harness and `cp2/`, review_followup4 incl. `b/b2_matched_prefix.py`, `b/b1_runner.py`, `c/`), `autovla_misalignment_poc/src/`, `pra_comparison/pra_labels.py`, `vla_misalignment_poc/analysis/`, `tools/` |
| review_followup4 results | `autovla_misalignment_poc/outputs/review_followup4/**` (A1_A4, A2_A3, B/B1 incl. run rows, B/B2 incl. bank / cells / cohort / gate / pilot / analysis, FEASIBILITY, OPENVLA_GATE, STATUS, CLAIM_CHANGES, MANUSCRIPT_REVISION, RUN_COMMANDS, DOC_EDITS_APPLIED, REPRO_CHECK; `SOURCE_MANIFEST.csv` stays in the repository because it hashes this ZIP) |
| OpenVLA episode records, state mapping, per-task CSV | `autovla_misalignment_poc/outputs/review_followup3/P0/OPENVLA_episode_records.csv` (phase, condition, task, init, δ, success, `initial_state_id`, `task_name`), `OPENVLA_REANALYSIS{,_per_task,_phaseC,_phaseC_dims}.csv`, `OPENVLA_phaseC_feedback_units.csv`, `OPENVLA_physical_units.csv`, `_external/*config.json`; `review_followup4/A1_A4/A1_OPENVLA_2x2.csv` |
| P1 protocol, manifests, tolerance pilot, judge logs | `outputs/review_followup3/P1/{PROTOCOL_P1.md, AMENDMENT_P1_0{1,2,3}.md, stage0_manifest*.json, svla_manifest.json, pilot_c5_20261014/, gate_stage0/, gate_O0/ (judge.log, decodes, comparisons), svla_gate/, NOOP_CACHE_AUDIT.csv, STATE_RESTORE_AUDIT.csv}`; specs `review_followup3/P1/INTERVENTION_SPEC.md` |
| P2 locked protocol + unit summaries | `outputs/review_followup3/P2/{MOTION_BASELINE_SPEC_LOCKED_v1.md, spec_lock_v1.json, LEARNED_EXTRAPOLATOR_PROTOCOL.md, extrap_protocol_lock.json, p2_units.parquet, extrap_units.parquet, *RESULTS.csv, p2_verdict.csv, extrap_coverage.csv}` |
| P3 protocol (historical, not run) | `outputs/review_followup3/P3/MATCHED_REVERSE_PROTOCOL.md` |
| P4 power plan (not run) | `outputs/review_followup3/P4/{POWER_PLAN.md, SPATIAL_STATE_EXPANSION_PROTOCOL.md, planning/, power/}` |
| P5 inventory (blocked) | `outputs/review_followup3/P5/`, `P0/DISJOINT_LOG_INVENTORY*.csv` |
| B2 protocol + unit-level cells / bank | `outputs/review_followup4/B/MATCHED_PREFIX_PROTOCOL.md`; `B/B2/{p1c,dev}_main/{bank.jsonl, bank_manifest.json, cells.jsonl, eligibility.json, run_meta_*.json}`; `B/B2/p1c_cohort_t4.json` (cohort, weights); `B/B2/{p1c,dev}_analysis/unit_contrasts.parquet` |
| paired outcomes, cluster IDs, weights, inclusion | `outputs/review_followup2/B_spatialvla_direct/run_20261013/pairs.csv` (SpatialVLA paired outcomes with state / layout clusters), `review_followup2/A_spatialvla_states/`, `review_followup/P1C_new_log_replication/run_20261008/{analysis/units.parquet, selection/, units/, stage1/records.jsonl}` + `design_20261008/split.csv` (P1-C units, drive / day–vehicle ids, F/S sampling, weights, exclusions), `review_followup/P0B_cluster_stats/run_20261008/units.parquet`, `R8R9_raw_recovery/run_20261008/units.parquet`, `P1A_gt_free_reference/run_20261008/{raw,raw_fixedpos}/records.jsonl`, `full_extract/records.jsonl`, `equal_distance_perturbation/records.jsonl`, `review_followup3/P0/DATA_MANIFEST.csv` |
| manuscript-facing documents | `paper_quantitative_package/` (numbers.json, numbers_sources.csv, tables.tex, PAPER_NUMBERS.md, MANUSCRIPT_REVISION.md, …), `EXPERIMENT_SUMMARY.md` |

No manuscript `.tex` exists in the repository; `MANUSCRIPT_REVISION.md` gives sentence-level replacements instead.
