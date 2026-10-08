# P1-B protocol: same correction budget, different start position (AutoVLA)

Status: **preregistration, final (lead decisions applied 2026-10-08).** No main run has been launched. No main-run outcome has been seen. The smoke test (4 units, `smoke/`) only checked the intervention trace.
Plan reference: `/root/VLA/VLA_experiment_plan_20261008.md` §2, §7, §11.
Labels: [확인한 사실] / [계산한 결과] / [미확인] / [가설].

## 1. Question and hypotheses

Is the initial position special, or does only correction length matter? Exp 9's window sweep varied length with the start fixed at t*+1. Later windows also had fewer free steps afterwards.

- **H_early**: With the same total horizon, the same units and the same number of corrected positions w, a correction that starts right after the perturbation (s = 1) leaves less amplification than the same correction starting two positions later (s = 3).
- **H_null (competing)**: Length matters, but the early–late difference is small. Equivalence is claimed only inside the margins in §5.
- Plan §11 mapping:
  - early consistently stronger → initial-position specificity for this model and horizon;
  - length matters but position does not → the current "short-correction efficiency" conclusion.
  - A length effect alone is **not** evidence of a critical window.

## 2. Design

### 2.1 Units
- The equal-distance set as in exps 7–11 [확인한 사실]: 208 scenes, 1,408 units = scene × forced token f at t*.
- The executed output of every row = `pred[:t*] + [f] + own samples`. Corrections are context-only, as in exps 7–10.

### 2.2 Common subset for the primary analysis (fixed before outcomes)
- Context position j can affect outputs only if j ≤ 8. A window (s, w) covers context positions t*+s … t*+s+w−1.
- Every window with s ∈ {1, 2, 3, 4} and w ∈ {1, 2, 3} is complete iff **t* ≤ 2**.
- **Primary subset = A− units with t* ≤ 2: 46 scenes, 326 units, 15 logs.**
  - Every condition uses the same units. No condition-specific inclusion.
  - Units with t* > 2 are run (windows flagged `truncated`) but not used in total-horizon contrasts.
- A+ with t* ≤ 2: 138 scenes, 920 units, 25 logs (secondary).

### 2.3 Rows (one batched forward per step, B = 53; seed sha256(f"0:{token}:{perturbation}:{k}") shared by all rows of a unit = the exp 9 rule)
- `normal`: no correction.
- `full_gt`, `full_self`: every context position after t*.
- `recent_gt`, `recent_self`: the exp-7 sliding rule, every step.
- **Persistent** `P_{src}_s{s}_w{w}`:
  - offsets o ∈ [s, s+w−1] hold src[j] at every later step (exp 9 definition);
  - P_gt_s1_w = exp 9 win_w; P_gt_s{s}_w1 = exp 9 delay{s}_len1.
- **Transient** `T_{src}_s{s}_w{w}`:
  - src[j] only while j is the most recent context position (j = k−1) and o ∈ [s, s+w−1];
  - w corrected forwards, each with one corrected position (the exp-7 Recent-GT rule restricted to w steps).
- src ∈ {gt, self}. self = the fixed model reference from `p1_reference.py` (P1-A `reference/`, sha256-checked; GT-free, not success-selected).
- s ∈ {1, 2, 3, 4}, w ∈ {1, 2, 3}.
- Budget accounting per row (logged): corrected positions (= w), corrected forwards, corrected (step, position) pairs, and pairs whose token differed from the row's own token (effective).
  - Persistent windows: late windows are **visible in fewer forwards** than early ones (exposure differs).
  - Transient windows: exposure is equal (w forwards). Both designs are reported. The lead/user must approve which one is primary (see §9).

### 2.4 Trace facts from the smoke test [확인한 사실, 4 units]
- The output prefix is locked in all 53 rows. Prefix KV checksum and crop are fine.
- Persistent windows keep the corrected positions at every later step. Transient windows apply only at step k = j+1.
- `P_gt_s4_w3` at t* = 3 is flagged truncated: positions 7, 8, 9, and position 9 can never be context.
- [계산한 결과] Under the B = 53 numeric path, `normal` for the E:original unit diverged from the B = 1 reference at k = 3. Self-reference rows are therefore not no-ops even when f = r[t*]. Every effect uses this run's own `normal`.

## 3. Evaluation unit and weighting (per P0-B)
- Paired per-unit differences, averaged within scene first. **Scene-weighted** pooled estimate.
- Clusters are logs (primary: 15 logs). Unit-weighted and log-uniform weights are sensitivities. Leave-one-log-out is reported.

## 4. Outcomes
- **GT reference family**:
  - `amplification` = A− (existing P/R/A rule) ∧ FDE5 > 3 m;
  - severe thresholds `amp2` / `amp4` (2 / 4 m);
  - continuous FDE5 and ADE5;
  - per-pose GT error curves.
- **Self-reference family**:
  - `dev9_ref` (m, endpoint deviation from the fixed reference);
  - `severe_ref2/3/4`, `ref_amp3`;
  - GT-relative amplification as well.
- GT-reference and self-reference results are reported in separate tables and never pooled.

## 5. Contrasts and tests

**Primary family (Holm, m = 3)**: A−, t* ≤ 2.

| id | contrast | metric |
|---|---|---|
| P1B-w1 | P_gt_s1_w1 − P_gt_s3_w1 | amplification |
| P1B-w2 | P_gt_s1_w2 − P_gt_s3_w2 | amplification |
| P1B-w3 | P_gt_s1_w3 − P_gt_s3_w3 | amplification |

- Two-sided. Negative = early correction leaves less amplification.
- Tests:
  - log block bootstrap B = 10,000 (95% / 90% CI);
  - log-level sign-flip (exact over 2^15);
  - cluster-t sensitivity.
- Equivalence (only if a null is claimed): ±10 pp amplification, ±0.5 m FDE5 / dev9_ref, via the 90% CI (TOST).

**Secondary, total horizon fixed (unadjusted, labelled)**:
- FDE5, amp2, amp4 for the same contrasts.
- early (s = 1) − late (s = 2, 4).
- Transient family.
- Self-reference families (dev9_ref, severe_ref2/3/4, amplification).
- A+ and pooled groups.
- **start × length map**: each window − normal, with levels in `condition_summary.csv`.
- Fraction of the Normal → Full effect, computed at the aggregate level.
- Per-pose deviation curves.

**Secondary, fixed free horizon after release (exp-10 extension; reported separately; never combined with the total-horizon results)**:
- Release pose rel = t* + s + w. Evaluated pose = rel + N.
- **t* ≤ 1 with N = 2** for s ∈ {1, 2, 3} (A− 39 scenes / 277 units / 14 logs). Sensitivity: t* = 0 with N = 3 (17 scenes / 116 units / 9 logs).
- Window effect = metric(window) − metric(normal) **on the same pose**. Contrast = early (s = 1) − late (s = 3) of that effect, w = 1, 2, 3.
- Thresholds 2 / 3 / 4 m on the GT error, or on dev_ref for self.
- Re-divergence: among units with error ≤ 1 m at rel (stab), the share with error > τ at rel+N. Denominators (stab units per condition) are in `free_horizon_rediverge.csv`.
- Only the 10-token in-distribution horizon is used. Exp 10's 20-token extension is **not** repeated: exp 10 found ~0 action mass past token 10, and its extended GT file is not on disk.
- Generation steps are not converted into control latency.

## 6. Exclusions and denominators
- No parsing step exists (action rows only). Runtime errors go to `errors.jsonl` with their stage.
- No outcome-based exclusion.
- Per-condition n_units / n_scenes / n_logs / n_truncated_window are in `condition_summary.csv`. Paired joins report missing pairs and NaN drops.

## 7. Stopping / termination
- One fixed run, no interim outcome look, no extension.
- Abort before analysis if:
  - the reference sha256 check fails;
  - output_prefix_locked < 100%;
  - a cache assert fires;
  - or more than 2% of scenes fail.
- GPU OOM on the shared 5090: failed scenes are re-run once in a separate dir with identical settings and merged by unit_id. Both runs are recorded. An OOM at model load (no record written) is retried automatically by the launcher, up to 3 times, after the 5090 shows at least 13 GB free for 60 s. The empty attempt dir is moved aside as `*.failed_attempt*`, never deleted. An OOM after records exist stops the launcher, and the protocol rule above applies.

## 8. Known limitations
- t* and the A−/A+ strata come from GT, as inherited.
- Primary power is limited: 15 informative clusters at most, and the sign-flip p floor is 2/2^15 ≈ 6e-5.
- Persistent windows confound start with exposure; the transient family addresses this (§2.3).
- bf16 with B = 53 numerics differs from exp 9 (B = 17). P_gt_s1_w is comparable to exp 9 win_w only qualitatively.
- **No 20-token extension (lead decision 8).** The secondary fixed-free-horizon analysis is limited to the 10-token in-distribution plan. That allows at most N = 2 free poses (t\* ≤ 1) or N = 3 (t\* = 0), and only starts s ≤ 3. Longer post-release re-divergence is not tested.

## 9. Lead decisions (2026-10-08), applied
1. **Persistent windows are primary.** The plan adds a start position to the existing length sweep at a fixed total horizon. Transient windows (equal exposure) are the main secondary.
2. Primary contrast: early s = 1 vs late s = 3, w = 1, 2, 3, binary amplification, Holm m = 3. Equivalence margins ±10 pp (amplification) and ±0.5 m (FDE5 / dev9_ref). s = 2 and s = 4 are secondary.
3. No 20-token extension; stated as a limitation (§8).

## 10. Seeds, environment, commands
- Seed 0, T = 0.01, statistics seed 20261008. RTX 5090 only. bf16 weights, float64 logits. HEAD e10374a. AutoVLA ba34eed.
- Seeds use the exp-9 rule: sha256(f"0:{token}:{perturbation}:{k}"), shared by all rows of a unit. This differs from P1-A, which uses one key per scene.
- **Launch** (single script for all experiments; order reference → P1-A → P1-A fixed-position → **P1-B** → R8R9; waits for at least 13 GB free on the 5090 before every model load):
```bash
bash /root/VLA/autovla_misalignment_poc/scripts/review_followup/p1ab/run_all_main.sh
```
- Explicit P1-B commands (as executed by that script):
```bash
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
REF=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1A_gt_free_reference/run_20261008/reference
R=/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1B_start_position/run_20261008
P=/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1ab
nice -n 19 python $P/p1b_start_position.py --out $R/raw --reference $REF
nice -n 19 python $P/p1b_analyze.py --raw $R/raw --out $R
nice -n 19 python $P/p1_manifest.py --run-dir $R --experiment P1B_start_position
```

### 10.1 Exact output paths and git status
- Run dir R = `/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1B_start_position/run_20261008/`. Rules are from `/root/VLA/.gitignore`.
- Outputs:

| path (relative to R) | content | git |
|---|---|---|
| `protocol.md` | this file | tracked |
| `manifest_prerun.json`, `manifest.json` | sha256 before / after the run | **ignored**; needs `git add -f` |
| `raw/records.jsonl`, `raw/intervention_trace.jsonl`, `raw/errors.jsonl` | raw (about 52 kB per unit, about 75 MB in total) | ignored |
| `raw/run_meta.json` | environment | tracked |
| `units.parquet`, `free_horizon_units.parquet`, `condition_summary.csv`, `paired_effects.csv`, `loo_logs.csv`, `free_horizon_rediverge.csv`, `deviation_curves.csv`, `trace_verification.json` | analysis | ignored |
| `intervention_trace.jsonl` | copy of the first 24 traced units | ignored |
| `logs/*.log` | stage logs | ignored |
| `smoke/` | smoke raw and analysis | ignored, except `*/run_meta.json` |

- Input reference (shared, sha256-checked): `/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1A_gt_free_reference/run_20261008/reference/references.jsonl` (ignored).
- Scripts: the `.py` files in `scripts/review_followup/p1ab/` are tracked. The `.sh` files are ignored and need `git add -f`.

## 11. Smoke result and cost [계산한 결과]
- 4 units (2 scenes, t* = 1 and 3; original + alt0 each), B = 53.
- 1.05 s/unit plus about 2.5 s/scene for the prefix; peak GPU 10.9 GB.
- Output prefix locked in all rows (4/4). Window traces logged per (step, position) as specified (`smoke/raw/intervention_trace.jsonl`, `corrected_pairs`).
- Main estimate: 1,408 × 1.05 s + 208 × 2.5 s + model load ≈ **35 min GPU**. It needs the P1-A reference (≈ 7 min) if not built yet.
- Outputs also include `loo_logs.csv` (leave-one-log-out, primary family).
