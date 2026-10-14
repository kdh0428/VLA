# MOTION_BASELINE_AUDIT (P2-A, review_followup3, 2026-10-14)

**Status: PROVISIONAL.** These are AutoVLA numbers from the P1-A rollouts, the C1 input path. They stay provisional until the AutoVLA P1 no-op/cache parity gate is judged (`outputs/review_followup3/P1/NOOP_CACHE_AUDIT.*`; present when this report was generated). Section 6.2 and Figure 1 of the manuscript are held until the gate passes. Tags: driving domain, AutoVLA, open-loop 5 s plan, POST-HOC analysis of existing rollouts.

## 0. Provenance and spec lock
- **Spec adopted:** the MOTION_BASELINE_SPEC from the stopped v1 attempt, unchanged.
  - It was locked at **2026-10-14T10:53:20Z** (sha256 `7330c46d6e33f146…`).
  - Before the lock, v1 had run only the CM-definition check and the C1 reproduction.
  - The v1 directory contains no new-baseline result file (no `p2_units.parquet` and no `MOTION_BASELINE_RESULTS*`). The v1 evaluation script carries a later mtime than the lock but never wrote output.
  - This run first re-ran steps 1–2 and the unit tests, then evaluated. `p2_motion_baselines.py` asserts the sha256 of the verbatim spec copy and of `p2_kinematics.py` before computing anything.
- **Files:**
  - verbatim spec: `MOTION_BASELINE_SPEC_LOCKED_v1.md`;
  - lock record: `spec_lock_v1.json`;
  - the spec text is reproduced in section 4 below.
- **Code:** `scripts/review_followup3/`
  - `p2_kinematics.py`, identical to v1 and sha-locked;
  - `p2_cm_audit_reproduce.py`;
  - `p2_tests.py`;
  - `p2_motion_baselines.py`: v1 plus the lock-file path and a status column;
  - `p2_figure.py`;
  - `p2_reports.py`;
  - `p2_regen.sh`.

## 1. CM definition (code + numeric check, regenerated)
**Code.**
- `scripts/review_followup/p1ab/p1a_gt_free_reference.py` builds `CM = ref[:t_p] + [p] * (10 - t_p)` as **tokens**. These are decoded by the real `ActionTokenizer.decode_token_ids_to_trajectory`.
- C1 (`c_baseline_window.py`) uses `CM_0 = ref[:t] + [r_t] * (10 - t)`.
- The decoder (`ActionTokenizer.rollout`; numpy copy `p1_common.rollout_np`) chains each token as an SE(2) increment in the frame of its own start pose:
  - x_k = x_{k−1} + Rot(h_{k−1}) · disp[a];
  - h_k = h_{k−1} + dyaw[a].

**Numeric check** on all 5553 perturbed P1-A units, using the stored GPU decoder output:

| check | value |
|---|---|
| body-frame increment of every CM suffix segment vs (disp[p], dyaw[p]): max error | 7.5e-06 m, 1.2e-06 rad |
| global-frame displacement repeat deviation (0 for a global-velocity repeat), median over turning units / max | 0.52 m / 7.58 m |
| share of units whose perturbed token turns (abs(dyaw[p]) > 0.5°) | 83.9% |
| abs(dyaw[p]) median / p90 per 0.5 s | 2.2° / 7.1° |
| abs(chord angle − dyaw/2) median / p90 (deviation from a zero-sideslip arc) | 1.7° / 8.1° |
| numpy SE(2) chaining vs stored decoder | 1.9e-05 |

**Finding (rebuttal-relevant).** The existing 'repeat-p constant-motion' baseline is **not** a global constant-velocity repeat.
- It is a **body-frame (local) SE(2) repeat, yaw included**: the perturbed token's translation is applied in the current heading frame, and the heading advances by dyaw[p] every 0.5 s.
- So it is already a discrete constant-turn-rate, constant-speed (constant body twist) extrapolation. It also keeps the token's lateral (chord-angle) component.
- A 'CTRV fitted to the last token' would differ from CM only by imposing zero sideslip. It is therefore **not added**, as a renamed near-duplicate.
- The reviewer's premise that CM ignores curvature is not correct for this implementation.

**Controls that are distinct from CM** (spec section 2.3):
- **CV0:** zero-turn, global constant velocity. It removes the yaw-rate continuation.
- **CTRV-K3:** body twist averaged over the last ≤3 steps. The perturbation enters with weight 1/K_eff.
- **CTRA-2:** constant turn rate plus the acceleration extrapolated from the last two speeds.

## 2. C1 reproduction (exact)
- `c_baseline_window.c1_units` / `c1_tables` were re-run in-process with the original seeds.
- Unit rows equal: True; unit numeric max abs diff 0.0e+00.
- Table rows equal: True; table numeric max abs diff 8.9e-16.
- Reproduced exactly: **True**.

Key cells (family G):

| cell | est [95% CI] | sign-flip p |
|---|---|---|
| G|A-|ex_geo_end | +4.37 [+3.32, +5.19] | 2.4e-04 |
| G|A-|ex_cm_end | +2.35 [+1.27, +3.19] | 7.3e-04 |
| G|A-|ex_cm_sfx | +0.48 [+0.34, +0.59] | 2.4e-04 |
| G|A-|ex_cm_dref | +4.02 [+2.93, +4.78] | 2.4e-04 |
| G|A+|ex_geo_end | +1.64 [+1.29, +1.93] | 3.8e-06 |
| G|A+|ex_cm_end | -0.52 [-0.82, -0.26] | 0.0019 |
| G|A+|ex_cm_sfx | +0.12 [+0.09, +0.16] | 1.9e-05 |
| G|A+|ex_cm_dref | +1.21 [+0.80, +1.52] | 9.5e-05 |
| G|all|ex_geo_end | +2.32 [+1.79, +2.69] | 3.8e-06 |
| G|all|ex_cm_end | +0.20 [-0.32, +0.56] | 0.51 |
| G|all|ex_cm_sfx | +0.21 [+0.15, +0.25] | 3.8e-06 |
| G|all|ex_cm_dref | +1.91 [+1.29, +2.35] | 2.3e-05 |

## 3. Unit tests (`tests_output.txt`)

```
[PASS] T1 compose <-> body_increments round trip (random SE(2) increments): max err 4.441e-15 (tol 1e-09)
[PASS] T2 tokens_to_poses (SE(2) chain) vs p1_common.rollout_np (decoder copy): max err 3.375e-14 (tol 1e-09)
[PASS] T3a bl_cm(prefix) vs stored GPU-decoder CM trajectory (5553 units): max err 1.889e-05 (tol 5e-05)
[PASS] T3b CM endpoint change from bl_cm vs C1 cm_end column: max err 1.714e-05 (tol 5e-05)
[PASS] T4 CTRV with K=1 and no clip == CM (real units): max err 0.000e+00 (tol 1e-09)
[PASS] T6 CTRA with a = 0 (t=0 fallback, or equal last-two speeds) == CM: max err 6.040e-14 (tol 1e-09)
T5 omega->0 (eps, discrete-CTRV vs CV endpoint err, closed-form CTRV vs CV err): 1e-02: 4.43e-01, 4.98e-01; 1e-04: 4.47e-03, 5.03e-03; 1e-06: 4.47e-05, 5.03e-05; 1e-08: 4.47e-07, 4.96e-07
[PASS] T5a discrete CTRV -> CV as omega -> 0 (eps = 1e-8): max err 4.471e-07 (tol 1e-06)
[PASS] T5b closed-form CTRV -> CV as omega -> 0 (eps = 1e-8): max err 4.955e-07 (tol 1e-06)
[PASS] T5c error decreases monotonically with eps: max err 0.000e+00 (tol 0e+00)
[PASS] T5d CV0 == CTRV-K1 when the last yaw increment w_t = 0: max err 1.421e-14 (tol 1e-09)
[PASS] T7 baselines invariant to poses after t (no future / evaluation-suffix leakage): max err 0.000e+00 (tol 0e+00)
[PASS] T8 C1 reproduced exactly (max abs table diff): max err 8.882e-16 (tol 1e-09)

ALL PASS
```

Clip and fallback activation (inputs only; `baseline_fit_diagnostics.csv`):

| family   | group   |   n_units |   share_t0 |   ctrv_keff_mean |   ctrv_keff_lt3 |   ctrv_omega_clip_p |   ctrv_omega_clip_0 |   ctra_acc_clip_p |   ctra_acc_clip_0 |   ctra_omega_clip_p |   ctra_stop_p |   ctra_stop_0 |
|:---------|:--------|----------:|-----------:|-----------------:|----------------:|--------------------:|--------------------:|------------------:|------------------:|--------------------:|--------------:|--------------:|
| E        | A+      |       887 |      0.328 |            1.927 |           0.745 |               0     |                   0 |             0.001 |                 0 |               0     |         0.12  |         0.114 |
| E        | A-      |       313 |      0.316 |            1.923 |           0.76  |               0     |                   0 |             0     |                 0 |               0     |         0.121 |         0.118 |
| G        | A+      |      1089 |      0.328 |            1.904 |           0.768 |               0     |                   0 |             0     |                 0 |               0     |         0.123 |         0.128 |
| G        | A-      |       364 |      0.341 |            1.901 |           0.758 |               0     |                   0 |             0     |                 0 |               0     |         0.124 |         0.121 |
| GF1      | A+      |      1096 |      0     |            2     |           1     |               0     |                   0 |             0     |                 0 |               0     |         0.15  |         0.154 |
| GF1      | A-      |       374 |      0     |            2     |           1     |               0.003 |                   0 |             0     |                 0 |               0.003 |         0.241 |         0.235 |
| GF3      | A+      |      1068 |      0     |            3     |           0     |               0     |                   0 |             0     |                 0 |               0     |         0.114 |         0.126 |
| GF3      | A-      |       362 |      0     |            3     |           0     |               0     |                   0 |             0     |                 0 |               0     |         0.193 |         0.251 |

## 4. Locked spec (verbatim copy of MOTION_BASELINE_SPEC_LOCKED_v1.md)

> The text below was written by the stopped v1 attempt and locked before any new-baseline result was computed.
> Paths in it refer to the v1 layout. Its section 1 numbers equal the regenerated values above.

# MOTION_BASELINE_AUDIT (P2, review_followup3, 2026-10-14)

- **Status:** POST-HOC re-analysis of existing P1-A rollouts. CPU only, `nice -n 19`. No model inference, no GPU, no existing file modified.
- **Order of work:**
  1. CM definition check (step 1).
  2. Exact reproduction of C1 (step 2).
  3. This specification. It was written and its sha256 locked in `spec_lock.json` **before** any new-baseline evaluation result was computed.
  4. Unit tests.
  5. Results, reported in `MOTION_BASELINE_RESULTS.md`. This file is not edited after the lock.
- **Code:** `/root/VLA/autovla_misalignment_poc/scripts/review_followup3/`
  - `p2_kinematics.py`
  - `p2_cm_audit_reproduce.py`
  - `p2_tests.py`
  - `p2_motion_baselines.py`
  - `p2_figure.py`
  - `p2_regen.sh`

## 1. What the existing "repeat-p constant-motion" (CM) baseline is [fact: code + numerical check]

### Code
In `scripts/review_followup/p1ab/p1a_gt_free_reference.py` (lines 337 and 529 of the module):

- `CM = ref[:t_p] + [p] * (10 - t_p)`. These **tokens** are decoded with the real `ActionTokenizer.decode_token_ids_to_trajectory`.
- In C1 (`c_baseline_window.py`), the unperturbed counterpart is `CM_0 = ref[:t] + [r_t] * (10 - t)`, decoded with `p1_common.rollout_np`.

### How a token is decoded
The decoder (`rollout_np` is its numpy copy) chains each token as an SE(2) increment in the frame of its own start pose:
- x_k = x_{k-1} + Rot(h_{k-1}) · disp[a];
- h_k = h_{k-1} + dyaw[a].

Here disp[a] is the mean of the token's last sub-step box corners and dyaw[a] is the heading of its last box. Repeating token p therefore repeats the **body-frame SE(2) increment, yaw included**.

### Numerical check
Run by `p2_cm_audit_reproduce.py` on all 5,553 perturbed units, using the stored GPU decoder output. Files: `cm_definition_check.csv`, `cm_definition_summary.json`.

| check | result |
|---|---|
| Body-frame increment of every CM suffix segment vs (disp[p], dyaw[p]) | max error 7.5e-6 m and 1.2e-6 rad, i.e. exact up to the stored 1e-6 rounding |
| Global-frame displacement repeat (would be 0 for a global velocity repeat) | median deviation 0.52 m over turning units, max 7.58 m, so **not** a global repeat |
| Share of units whose perturbed token turns (\|dyaw[p]\| > 0.5°) | 83.9% |
| \|dyaw[p]\| distribution | median 2.2°, p90 7.1° per 0.5 s |
| numpy SE(2) chaining (`p2_kinematics.compose`) vs stored decoder | max 1.9e-5 (coordinate round trip) |
| \|chord angle − dyaw/2\| of p | median 1.7°, p90 8.1° |

The last row is the deviation from an ideal zero-sideslip arc.

### Finding
CM is a **local (body-frame) SE(2) repeat including yaw**. It is already a discrete constant-turn-rate, constant-speed (constant body twist) extrapolation of the perturbed motion, and it keeps the token's own lateral/chord-angle component.

- A "CTRV fitted to the last token" would differ from CM only by forcing zero sideslip (median 1.7°).
- That is a renamed near-duplicate, so it is **not added** (prompt rule).
- The P1-A protocol (line 67) already describes CM as "constant speed and yaw rate in the token frame". This is confirmed.

### C1 reproduction (step 2)
- `c_baseline_window.c1_units` and `c1_tables` were re-run in-process with the original seeds.
- Every one of the 5,553 unit rows is identical (max abs difference 0).
- Every numeric cell of `c1_excess_effects.csv` (192 rows) matches to ≤ 8.9e-16.
- Details are in `c1_reproduction_check.json`.

Reproduced example (G, A−):
- model − CM endpoint +2.35 [+1.27, +3.19];
- model − Geo +4.37 [+3.32, +5.19].

<a id="MOTION_BASELINE_SPEC"></a>
## 2. MOTION_BASELINE_SPEC (fixed before any new-baseline result was computed)

### 2.1 Units, pairing and inference (identical to C1)
- **Units and families:**
  - P1-A `raw/records.jsonl`: family G (primary) and family E;
  - P1-A `raw_fixedpos/records.jsonl`: GF1 and GF3;
  - units with p == r_t are excluded, as in C1.
- **Strata:** A−, A+, all. Same paired units and same 10-pose horizon. Evaluation uses pose 9 (5 s) or the suffix k = t+1..9.
- **Model counterfactual:** perturbed = OC, unperturbed = U, from the same batch and seed (as in C1).
- **Each baseline b:**
  - perturbed input = decoded prefix poses of r[:t] + [p];
  - unperturbed input = decoded prefix poses of r[:t] + [r_t].
  - Decoding uses `p2_kinematics.tokens_to_poses`, which equals `rollout_np` to ≤ 2e-5.
- **Per-unit contrast:** Δ_model − Δ_b, where Δ = φ(perturbed) − φ(unperturbed) is computed per method. Absolute GT errors of different methods are never subtracted.
- **Estimator:** `c_baseline_window.drive_row` / `stats_core`:
  - scene-weighted (scene means first, then scenes equally weighted);
  - drive-cluster bootstrap with B = 10,000;
  - exact two-sided drive sign-flip;
  - unit-weighted and drive-equal estimates as separate columns.
- **Drive:** drive_id = `<date.time>_veh-NN` prefix of log_id. A− has 13 drives, A+ 19, all 19.
- **Seeds:** P2 seed base 20261014, incremented by 11 per row in a fixed loop order.
- **Holm family:** within each (perturbation family, stratum, metric), the four kinematic extrapolation baselines {CM, CV0, CTRV-K3, CTRA-2}, so m = 4.
  - Geo (privileged, see below) is reported but is not in the Holm family.
  - Every analysis is post-hoc. No equivalence margin was predeclared, so **equivalence is not tested**.

### 2.2 Metrics φ (kept separate, never mixed)
| id | definition | source |
|---|---|---|
| M1 `end` | endpoint displacement norm ‖X_p(9) − X_0(9)‖ (m) | C1 "endpoint change" (same φ) |
| M2 `sfx` | suffix local-motion change: mean over k = t+1..9 of ‖ℓ_k(X_p) − ℓ_k(X_0)‖ (m per 0.5 s segment), with ℓ_k = Rot(−h_{k−1})(x_k − x_{k−1}), the body increment from the method's own poses. For token rows ℓ_k = disp[token_k] exactly | C1 "suffix change" (same φ) |
| M3 `dref` | signed error change vs the reference r: ‖X_p(9) − r(9)‖ − ‖X_0(9) − r(9)‖ (m) | C1 "same-reference variant" (same φ) |
| M4 `dgt` | signed GT error change: ‖X_p(9) − GT(9)‖ − ‖X_0(9) − GT(9)‖ (m). Secondary and GT-relative; A− is GT-defined | new; within-method Δ only |
| M5 `area` | path area, i.e. time-integrated displacement: Σ_{k=t..9} ‖X_p(k) − X_0(k)‖ · 0.5 s (m·s) | new; pose t is common to all methods |

Primary display: family G, M1. M2–M5 are secondary.

### 2.3 Baselines

Common to all baselines:
- **Input:** only the decoded executed prefix poses 0..t and the origin, i.e. the action tokens r[:t] and the token at t. **No** r[t+1:], model output, GT, or evaluation suffix. Unit test T7 enforces this.
- **Frame:** decoder frame = ego frame at t = 0.
- The baselines see strictly less than the model, which also sees cameras, ego status and navigation. No development data were used. Hyper-parameters are physical priors, fixed here and not tuned.
- Pre-t0 ego history is not available in token form and is not used.

| | CM (existing) | **CV0** (new) | **CTRV-K3** (new) | **CTRA-2** (new, optional class) |
|---|---|---|---|---|
| idea | repeat the last body twist (ℓ_t, w_t) | zero-turn global constant velocity | constant body twist fitted over several past steps | constant turn-rate and constant acceleration |
| frame of continuation | body (rotates with w_t) | global (decoder) frame, no rotation | body | body |
| speed / translation estimate | ℓ_t (last segment) | v = x_t − x_{t−1} (global; x_{−1} = origin) | ℓ̄ = mean of ℓ_j, j = t−K+1..t, K = 3 (1.5 s) | s_t = ‖ℓ_t‖, chord angle β = atan2(ℓ_t) held |
| yaw | w_t (last) | none (heading held at h_t; used only for M2 body increments) | w̄ = mean of the wrapped w_j over the same window | w_t (last) |
| acceleration | 0 | 0 | 0 | a = s_t − s_{t−1} (per segment) |
| clipping | none | none | \|w̄\| ≤ 0.5 rad/seg (1.0 rad/s) | \|a\| ≤ 1.0 m/seg² (4 m/s²); \|w\| ≤ 0.5 rad/seg; 0 ≤ s_k ≤ 10 m/seg (20 m/s) |
| short-prefix fallback | – | t = 0: v = x_0 − origin | window truncated at the origin: K_eff = min(3, t+1). t = 0 gives CTRV-K3 = CM (+ clip) | t = 0: a = 0, so CTRA-2 = CM (+ clip) |
| low speed | n/a | n/a | n/a: yaw is the decoded box heading (a state), not differentiated from positions, so no low-speed yaw noise | if s_k reaches 0 the vehicle stops: no further translation or rotation |
| yaw wrap | – | – | each increment wrapped to (−π, π] before averaging | wrapped |
| ω → 0 limit | straight line along Rot(h_t)ℓ_t | (is the ω = 0 case) | constant body velocity = global CV along Rot(h_t)ℓ̄. Equals CV0 when K_eff = 1 and w_t = 0 (test T5) | CA along a straight line; a = 0 gives CV |
| why distinct from CM | – | removes the yaw-rate continuation of the perturbed token's dyaw | the perturbation enters with weight 1/K_eff (smoothing over past steps) | the perturbation also changes the extrapolated acceleration |
| privileged? | no | no | no | no |

- **Not added:**
  - "CTRV from the last token": CM up to a zero-sideslip constraint, so a renamed duplicate.
  - Learned non-AR baselines: not required for this gate.
- **Privileged comparator:** **Geo** = r[:t] + [p] + r[t+1:].
  - It uses the model's own **unperturbed future** tokens, which a prefix-only predictor cannot know. It is reference-informed but GT-free.
  - It is reported only as the C1 "pure coordinate accumulation" control, separately and outside the Holm family.
- Clip and fallback activation counts are computed from inputs only and reported with the results.

### 2.4 Decision rule (fixed now)
- Against baseline b in stratum S, "beyond extrapolation" holds when Δ_model − Δ_b > 0 on M1 with Holm-adjusted (m = 4) sign-flip p < 0.05 and the drive 95% CI excludes 0.
- A **general** "beyond kinematic extrapolation" statement in S requires this for **all four** kinematic baselines (CM, CV0, CTRV-K3, CTRA-2) on M1.
- A negative significant estimate is reported as "model moves less than b".
- "Not detected" is not equivalence.
- M2–M5 are reported as secondary and do not override M1.
- Results in A− (GT-defined and failure-selected) are not generalized to the population.

### 2.5 Unit tests (`p2_tests.py`, output `tests_output.txt`)
- T1: compose / body_increments round trip.
- T2: `tokens_to_poses` vs `rollout_np`.
- T3: `bl_cm` on decoded prefixes reproduces the stored decoder CM and the C1 `cm_end`.
- T4: CTRV with K = 1 and no clip ≡ CM.
- T5: ω → 0 limits (discrete and closed-form CTRV → CV; CV0 ≡ CTRV-K1 when w_t = 0).
- T6: CTRA with a = 0 ≡ CM.
- T7: no-future invariance (outputs unchanged when poses after t are replaced with garbage).
- T8: C1 reproduction flag.

