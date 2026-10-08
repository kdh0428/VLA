# R8R9 protocol: raw recovery re-run of exp 8 (state patching) and exp 9 (temporal window)

Status: **final (lead decisions applied 2026-10-08). Main run not launched.** Smoke (5 units) is in `smoke/`. **Label: RECOVERY re-run of already-reported analyses, not a confirmation.**
Labels: [확인한 사실] / [계산한 결과] / [미확인] / [가설].

## 0. What this is and is not
- P0-A (U1, U2) and P0-B (§1.1) found that exps 8 and 9 have **no per-unit raw on disk**, only summaries. [확인한 사실]
  - The original scripts do write per-unit `records.jsonl`; the files were lost.
  - So P0-B could not compute the AutoVLA primary **Reverse−Full** under log clusters, nor cluster CIs for the window results.
- This run re-executes the **original, unmodified scripts** into new directories:
  - `prev_action_state_patching.py`: all 93 rows, B = 93, as the original;
  - `temporal_feedback_window.py`: 17 rows, B = 17.
- It keeps every per-unit record and then applies the P0-B estimators.
- The script sha256s are recorded in `logs/original_script_sha256.txt`. New code: only `r8r9_run.sh` (launcher) and `r8r9_build_units.py` (converter and statistics).
- **This is not a preregistered confirmation.** The original summary numbers are already known (reverse@emb 33.4% vs GT-history 4.4%, window 47/76/90/96%). It is a recovery and reproduction run. Its results may be reported only as "cluster statistics on recovered raw of an already-reported analysis".

## 1. Population and settings
- The equal-distance set: 208 scenes, 1,408 units (A− 52 scenes / 365 units / 16 logs; A+ 156 / 1,043 / 26).
- Script defaults: seed 0, T = 0.01, RTX 5090 (the original run_meta also says RTX 5090, CUDA_VISIBLE_DEVICES = 1).
- Exp 8 also gets the original's sanity/debug run: `--limit 4 --max-alts 1 --debug`, written to `raw/exp8/sanity/`, which the original analysis expects.

## 2. Contrasts (P0-B estimators: scene-weighted, log clusters, log block bootstrap B = 10,000, log-level sign-flip exact ≤ 20 logs, cluster-t sensitivity, unit/log weights as sensitivities)
- **R8 primary**: A−, `reverse@emb − gt_history`, amplification (Reverse−Full).
  - After the run, its sign-flip p replaces the placeholder (p = 1) in P0-B's AutoVLA primary family (Holm m = 3, with Recent−Normal and Direction−Magnitude), and Holm is recomputed.
  - Interpretation per P0-A F1: reverse@emb copies the position-(k−1) embedding from the **separate Normal row**, not from the reverse branch's own output.
- R8 secondary:
  - recent_gt − normal and gt_history − normal (this harness's own Normal);
  - reverse@L − gt_history for the 10 representative layers;
  - FDE5 versions; A+ and pooled.
- R9:
  - win_w − normal (w = 1..4), gt_history − normal, win_w − gt_history;
  - delay rows; FDE5; A−, A+, pooled.
  - Recovery fractions are computed at the aggregate level only.
- Descriptive critical-window statements need the same threshold rule as the original (f ≥ 0.9 / 0.8), stated together.

## 3. Reproduction check (predeclared)
- `comparison_with_original_summary.csv` lists the unit-weighted amplification rate per group × row, from the recovered raw vs the original `summary.json`.
- **Exact reproduction** = every rate equal within 1e-9.
  - Otherwise report every difference in pp and do not replace the original numbers silently.
  - Possible causes to report: driver or library changes since the original run; current driver 610.43.
- The share of tokens identical to exp-7 raw (`prior_action_history`) is also reported. The originals were 88.6% (exp 8) and 89.2% (exp 9), but those were measured against the **original** exp-7 raw, which is gone (P0-A U5).
- The original analysis scripts are also run on the new directories. Their `summary.json` is written only under `raw/exp8` and `raw/exp9`, for a field-by-field comparison.

## 4. Exclusions, denominators
- Original rules apply: no unit-level exclusion. The OOD rule is condition-level in the original analysis and does not touch these rows.
- Scene/unit runtime failures go to `raw/exp*/errors.jsonl`. They are reported with denominators.
- Paired joins report missing pairs.

## 5. Stopping
- One run, no interim look.
- If the shared GPU OOMs (more than 0 scene failures), re-run only the failed scenes once into `raw/exp*_retry/` with identical settings and report it.
  - The original scripts have no resume option.
  - Merge by unit_id, noting the merge in `analysis.md`. An OOM at model load (no record written) is retried automatically by the launcher, up to 3 times, after the 5090 shows at least 13 GB free for 60 s. The empty attempt dir is moved aside as `*.failed_attempt*`, never deleted. An OOM after records exist stops the launcher, and the protocol rule above applies.

## 6. Estimated cost (from the original run_meta and this smoke)
- Exp 8: 20.5 min. Exp 8 sanity: about 1 min. Exp 9: 6.4 min. Model loads: 3 × about 1 min. Analysis: a few min.
- Total ≈ 32 min of GPU.

## 7. Launch and exact output paths
- Single launcher for all experiments. Order: reference → P1-A → P1-A fixed-position → P1-B → **R8R9 (last)**. It waits for at least 13 GB free on the 5090 before every model load.
```bash
bash /root/VLA/autovla_misalignment_poc/scripts/review_followup/p1ab/run_all_main.sh
```
- Stage 5 of that script runs `bash /root/VLA/autovla_misalignment_poc/scripts/review_followup/p1ab/r8r9_run.sh main`. That script also waits for GPU memory and refuses to overwrite. Then it writes `manifest.json`.
- Run dir R = `/root/VLA/autovla_misalignment_poc/outputs/review_followup/R8R9_raw_recovery/run_20261008/`. Rules are from `/root/VLA/.gitignore`.

| path (relative to R) | content | git |
|---|---|---|
| `protocol.md` | this file | tracked |
| `manifest_prerun.json`, `manifest.json` | sha256 before / after the run | **ignored**; needs `git add -f` |
| `raw/exp8/records.jsonl`, `raw/exp8/errors.jsonl` | exp 8 per-unit raw (about 100 kB per unit, about 140 MB) | ignored |
| `raw/exp8/run_meta.json`, `raw/exp8/summary.json`, `raw/exp8/layer_results.json`, `raw/exp8/narrative.json` | metadata and outputs of the original analysis | tracked by allowlist |
| `raw/exp8/sanity/` (records.jsonl, debug_checks.json, run_meta.json) | original-style debug run | jsonl ignored; run_meta tracked |
| `raw/exp9/records.jsonl`, `raw/exp9/errors.jsonl` | exp 9 per-unit raw | ignored |
| `raw/exp9/run_meta.json`, `raw/exp9/summary.json`, `raw/exp9/early_step_contribution.json` | metadata and original analysis | tracked by allowlist |
| `raw/exp*/units.jsonl`, `raw/exp*/figures/*.png`, `raw/exp*/*.md` | original analysis by-products | jsonl ignored; png and md tracked |
| `units.parquet`, `paired_effects.csv`, `loo_logs.csv`, `condition_summary.csv`, `comparison_with_original_summary.csv` | converter output | ignored |
| `logs/` (exp8.log, exp8_sanity.log, exp9.log, analyze_exp*.log, build_units.log, original_script_sha256.txt, raw_sha256.txt, done) | logs and hashes | ignored |
| `smoke/` | smoke raw, outputs, `smoke_verification.json` | ignored, except `*/run_meta.json` |

- Scripts: `r8r9_build_units.py` is tracked. `r8r9_run.sh` and `run_all_main.sh` are ignored and need `git add -f`.

## 8. Smoke result [확인한 사실 / 계산한 결과]
- Exp 8 `--limit 1 --max-alts 1 --debug`: 2 units, B = 93, 59 s wall clock including model load. No OOM on the shared 5090.
  - **93/93 rows are token-identical to P0-A's earlier tiny re-run** of the same units (deterministic on this GPU).
  - Debug checks: patch@emb ≡ recent_gt and patch@L35 ≡ gt_history (logits L1 = 0); K/V at or before the patch layer unchanged (0.0). The reverse source per step is the Normal row's token (`smoke_verification.json`).
- Exp 9 `--limit 1 --max-alts 2`: 3 units, B = 17, 51 s. GT context offsets are as defined (win_w = [1..w], delay3_len2 = [3, 4]).
- The converter produces units.parquet, paired effects, condition summary, comparison, and LOO.
- The original analysis scripts **fail on 1-scene smoke data** (empty A+ group). They ran on the full data originally, so this is not tested at smoke scale. The launcher continues on their failure.

## 9. Lead decisions (2026-10-08), applied
1. Re-run the **full original 93-row exp 8**, keeping B = 93, the original batch composition, the bf16 numeric path and the identical script. Not a 4-row subset.
2. **Label everything from this run as recovery, not confirmation** (§0). The original numbers were known before this run. Its p-values enter P0-B's primary family only as a recomputation under cluster statistics.
