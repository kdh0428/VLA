# BASELINE_WINDOW_RECHECK: part C (extrapolation baselines and correction window)

Written 2026-10-13, review_followup2, part C.
- **POST-HOC analysis of existing data.** It ran on CPU only (`nice -n 19`). No model inference and no GPU were used. No existing file was modified.
- Script that regenerates every table and figure: `scripts/review_followup2/c_baseline_window.py`. It takes about 20 s.
- Statistics come from `scripts/review_followup2/stats_core.py`:
  - drive-cluster bootstrap (B = 10,000, seed 20261013) that keeps paired rows together;
  - exact sign-flip on drive totals;
  - TOST;
  - Holm.
- drive_id = `<date.time>_veh-NN` prefix of log_id. Drive counts:
  - A−: 13 drives (16 logs).
  - A+: 19 drives (26 logs).
  - All: 19 drives.
  - P1-B common subset (A−, t* ≤ 2): 12 drives (15 logs).
- Sign-flip assumptions: drive totals are independent across drives and symmetric about 0 under H0. This is a conditional test, not a randomization test.
- Smallest attainable two-sided p:
  - 13 drives: 2/2^13 = 2.4e-4;
  - 12 drives: 4.9e-4;
  - 19 drives: 3.8e-6.
- Labels used below: [fact] = read from code or data; [computed] = computed in this run; [not checked] = could not be checked; [interp] = interpretation.

## C1. P1-A: what each number measures, and model vs baseline excess

### C1.1 Exact meaning of the three headline numbers [fact: p1a_analyze.py, p1a_gt_free_reference.py]
Shared definitions:
- **Coordinate frame**: decoder poses chained from (0,0,0), i.e. the ego frame at t = 0, x forward and y left. AutoVLA tokens are relative motions.
- **Reference r**: the model's own unperturbed rollout, generated at B = 1 with T = 0.01 and a fixed seed, and fixed before any condition. It is GT-free.
- **Sample**: family G (GT-free, 0.10 ± 0.03 m per segment, up to 8 direction sectors), A−: 364 units / 52 scenes / 16 logs / 13 drives.
- **Estimand**: paired per-unit difference, averaged within each scene first, then 52 scenes equally weighted.
- **Position**: t_p = t*, which is inherited and GT-informed.

| number | contrast | metric and denominator | what it is NOT |
|---|---|---|---|
| **+0.571 m/seg** (P1A-1) | C1 − U. C1 = reference output at t_p plus perturbed context token p; U = unperturbed, same batch and seed | suffix_loc = mean over the 9 − t_p suffix steps of ‖disp[o_k] − disp[r_k]‖. disp is the token's end displacement in its own start frame (m per 0.5 s segment). Both rows are measured against the same r | not a GT error; not a pose deviation |
| **+4.37 m** (P1A-2) | OC − Geo. OC = output and context both perturbed. Geo = r[:t] + [p] + r[t+1:], decoded with no inference | dev9_ref = x/y L2 distance between the row's pose 9 (5 s) and r's pose 9 | includes about +0.06 m of B = 10 vs B = 1 numeric noise (O1 − Geo). The same-batch contrast OC − O1 is +4.31 m |
| **−3.50 m** (P1A-3) | OC_refR − OC. OC_refR = **repeated** sliding replacement of the most recent context position by r[k−1] at every step: 6.79 applications per unit, 4.54 of them effective | dev9_ref (same as above) | **a recovery of reference consistency, not a GT task improvement.** On GT metrics OC_refR is worse in A−: amplification +19.7 pp [+9.0, +29.6], FDE5 +1.50 m [+0.64, +2.43] (drive CI). In A+ it is better: amplification −2.6 pp, FDE5 −0.40 m |

Drive-level re-analysis of the predeclared primary family (G, A−) [computed]:
- Estimates are unchanged.
- Each contrast has exact drive sign-flip p = 2.4e-4 (the floor) and Holm m = 3 p = 7.3e-4. At log level these were 3.1e-5 / 9.2e-5.
- Drive CIs:
  - P1A-1: [+0.435, +0.682].
  - P1A-2: [+3.34, +5.18].
  - P1A-3: [−4.29, −2.59].
- TOST against the predeclared margins (0.02 m/seg, 0.25 m): p_equiv = 1. These are clear non-null results; no equivalence is claimed.

**Is A− a fixed pre-intervention criterion?** [fact] Yes, in the sense that matters here.
- `group` is copied from the equal-distance records (`ed["group"]`). These records come from exps 6–11 and existed before P1-A.
- The criterion is applied to the stored natural, unperturbed rollout. It does not depend on any P1-A condition or outcome.
- **But it is GT-defined and failure-selected:** A− = scenes whose natural rollout amplifies against GT. t_p = t* is also GT-informed.
- So A− is a fixed, pre-intervention stratum, but not a GT-free or random sample.

**Are Geo and CM matched to the model?** [fact + computed over all 5,553 perturbed units in G, E, GF1 and GF3; `c1_matching_checks.json`]

| aspect | result |
|---|---|
| Initial state | Same scene and same prefix tokens r[:t_p] in OC, Geo and CM (100%) |
| First realized perturbation | Token p at t_p in OC, Geo and CM (100%). Pose at t_p is identical in OC, Geo and CM (max difference 0.0 m) |
| Coordinate transform | Geo and CM are decoded with the real `ActionTokenizer` decoder, in the same frame |
| Evaluation horizon | Same 10 poses for all, evaluated at pose 9 |
| CM_0 (needed below) | Not stored in the raw data. Rebuilt here with `p1_common.rollout_np`; its error against the decoder is ≤ 1.9e-5 m on CM_p |
| Model's unperturbed counterpart U | U equals r token-for-token in 87.2% of units. U is used as the model's own counterfactual (same batch and seed) |

- **Matching problem in the original OC − CM contrast:**
  - OC − CM (A− −0.59 m [−1.47, +0.44], p = 0.23) compares OC's deviation from r with CM_p's deviation from r.
  - The unperturbed CM_0 is itself already 4.69 m from r at pose 9 in A−. That is CM's own extrapolation error, not a perturbation effect.
  - So the original OC − CM mixes the perturbation response with the baseline's poor prediction. The prompt warns against exactly this.

### C1.2 Perturbed − unperturbed change for each predictor, and excess (family G, primary) [computed; `c1_excess_effects.csv`]
Definitions:
- Endpoint change = ‖X_p(pose 9) − X_0(pose 9)‖.
- Suffix change = mean over k > t of ‖disp[X_p,k] − disp[X_0,k]‖.
- Unperturbed counterparts: model OC vs U; Geo vs r; CM_p vs CM_0 = r[:t] + [r_t]×(10 − t).

The model's endpoint change splits into two parts. This uses the exact identities OC suffix ≡ C1 suffix (100%) and O1 suffix ≡ U suffix:
- coordinate accumulation = ‖OC9 − C1_9‖;
- local-motion (context) = ‖C1_9 − U9‖.

Table cells give the scene-weighted estimate with drive bootstrap 95% CI; p = exact drive sign-flip; Holm is over the 4 key excess rows per family × group (post-hoc).

| | A− (13 drives) | A+ (19) | all (19) |
|---|---|---|---|
| model endpoint change (m) | 4.91 [3.82, 5.71] | 2.24 [1.86, 2.54] | 2.91 [2.34, 3.29] |
| Geo endpoint change = pure coordinate accumulation (m) | 0.53 [0.43, 0.61] | 0.60 [0.54, 0.64] | 0.58 [0.53, 0.62] |
| CM endpoint change (m) | 2.56 [2.34, 2.80] | 2.76 [2.61, 2.89] | 2.71 [2.59, 2.81] |
| **excess model − Geo, endpoint** | **+4.37 [+3.32, +5.19]**, p = 2.4e-4, Holm 9.8e-4 | **+1.64 [+1.29, +1.93]**, p = 3.8e-6, Holm 1.5e-5 | **+2.32 [+1.79, +2.69]**, p = 3.8e-6 |
| **excess model − CM, endpoint** | **+2.35 [+1.27, +3.19]**, p = 7.3e-4, Holm 9.8e-4 | **−0.52 [−0.82, −0.26]**, p = 0.0019, Holm 0.0019 | **+0.20 [−0.32, +0.56]**, p = 0.51 (not detected; equivalence not tested) |
| model suffix local-motion change (m/seg); Geo = 0 | +0.58 [0.44, 0.70], p = 2.4e-4 | +0.23 [0.19, 0.26] | +0.32 [0.26, 0.36] |
| **excess model − CM, suffix** (CM = d0 = 0.101) | **+0.48 [+0.34, +0.59]**, p = 2.4e-4 | **+0.13 [+0.09, +0.16]**, p = 1.9e-5 | **+0.21 [+0.15, +0.26]**, p = 3.8e-6 |
| model coordinate-accumulation part − Geo (m) | +0.020 [−0.015, +0.065], p = 0.31 | +0.011 [−0.001, +0.023] | +0.013 [0.000, +0.024] |
| model local-motion part ‖C1_9 − U9‖ (m) | 4.80 [3.68, 5.64] | 2.02 [1.60, 2.35] | 2.71 [2.12, 3.11] |
| original OC − CM (dev9_ref; CM_0 mismatch not removed) | −0.59 [−1.47, +0.44], p = 0.23 | −4.69 [−5.35, −3.92] | −3.66 [−4.21, −3.04] |
| same-reference variant: Δdev_ref model − Δdev_ref CM | +4.02 [+2.93, +4.78] | +1.21 [+0.80, +1.53] | +1.91 [+1.29, +2.35] |

- The other weightings agree:
  - unit-weighted model − CM endpoint: A− +2.43, A+ −0.49, all +0.24;
  - drive-equal: A− +2.41, A+ −0.56, all +0.01.
- Equivalence: no margin was predeclared for any of these post-hoc excess contrasts, so **equivalence was not tested**. CIs only.

Secondary families (same direction pattern) [computed]:
- **E** (GT-anchored, 313 A− units):
  - model − Geo: A− +5.63 [+4.37, +6.81].
  - model − CM: A− +1.89 [+0.27, +3.26], p = 0.046; A+ −0.94 [−1.28, −0.68]; all −0.24 [−0.87, +0.26].
- **GF1** (fixed t_p = 1, GT-free):
  - model − Geo: A− +3.78 [+3.06, +4.67].
  - model − CM: A− +1.52 [+0.48, +2.61], p = 0.0125; A+ −0.62; all −0.09 [−0.54, +0.27].
- **GF3** (fixed t_p = 3):
  - model − Geo: A− +2.70 [+2.05, +3.61].
  - model − CM: A− +1.38 [+0.72, +2.38], p = 2.4e-4; A+ −0.41; all +0.04 [−0.29, +0.30].
- In every family the suffix excess over CM is positive in all groups.
- In every family the model's coordinate-accumulation part equals Geo's within ±0.07 m (CIs include 0).

Interpretation [interp]:
- **Beyond coordinate accumulation: yes, in A−, A+ and all.**
  - The model's coordinate-accumulation part equals Geo's.
  - The entire excess comes from the context-driven change of later local actions.
- **Beyond constant-motion extrapolation: the answer depends on the stratum.**
  - **A−:** once CM's own unperturbed extrapolation error is removed (perturbed − unperturbed for both), the model moves its endpoint about 2.35 m more than CM. The earlier reading "OC ≈ CM in A−" was produced by the mismatched comparison.
  - **A+:** the model moves about 0.5 m *less* than CM.
  - **Pooled:** no difference detected, and equivalence not tested.
  - The suffix local-motion change exceeds CM's d0 in every stratum. Part of it is in directions other than the perturbation direction; follow = 1.14 d0 in A− (original P1-A).
- **Stated limits** [interp]:
  - CM is one particular extrapolation (repeat p). A turn-rate or constant-velocity baseline fitted on development data was not available.
  - A− is failure-selected and GT-defined, so "excess over CM in A−" should not be generalized to the population. The pooled estimate is null.
  - A claim of "self-amplification beyond extrapolation" can be kept only as "in the A− (failure) stratum, against the repeat-p CM baseline". It must not be stated in general.

## C2. P1-B and R8R9 on one basis

### C2.1 Horizons: are they equal? [fact + computed from data]
- **Total-horizon contrasts (pose 9) do NOT have equal post-release horizons.**
  - Release pose = t* + s + w.
  - In the common subset (A−, t* ≤ 2, mean t* = 0.79), the mean free poses after release are s1: 6.21 / 5.21 / 4.21 (w = 1/2/3) and s3: 4.21 / 3.21 / 2.21.
  - So **s = 3 always has 2 fewer free poses.**
  - In R9 the free horizon also shrinks with w: win1 5.80 down to win4 2.90 poses (all A−). So the "length effect" is also a post-release-horizon effect.
- **Exposure also differs** for persistent windows. Corrected forwards are s1 7.21 vs s3 5.21; corrected (step, position) pairs for w = 3 are 18.6 vs 12.6. Transient windows have equal exposure (w forwards).
- **Per-pose data exist** (`err_gt_curve` in P1-B units.parquet), so I re-analysed on a **common post-release horizon N**:
  - The evaluation pose is rel + N, with the same N for early and late.
  - The same units are used for all w and both starts, which requires t* ≤ 3 − N: N = 1 gives the full common 326 units / 12 drives; N = 2 gives 277 units / 11 drives; N = 3 gives 116 units / 8 drives.
  - Remaining confound, which cannot be removed: with equal post-release horizons, the **absolute pose differs** by 2 steps. An equal post-release horizon and an equal absolute pose cannot both hold for different starts.
  - I therefore report three versions: the raw difference, a difference-in-differences against Normal at the same pose, and a recovery fraction normalized by Normal − Full at the same pose.

### C2.2 early (s = 1) − late (s = 3), persistent GT, A−, t* ≤ 2 [computed; `c2_position_length.csv`]
Total horizon, binary amplification. This is the predeclared primary, re-analysed at drive level; ±10 pp margin predeclared:

| w | estimate [drive 95% CI] | 90% CI | sign-flip p (Holm m = 3) | TOST p_lower / p_upper → p_equiv (Holm) | difference | equivalence ±10 pp |
|---|---|---|---|---|---|---|
| 1 | +10.7 pp [−0.7, +20.6] | [+0.9, +18.9] | 0.16 (0.49) | 0.003 / 0.54 → 0.54 (0.54) | not detected | **not established** |
| 2 | +4.3 pp [−2.6, +9.7] | [−1.5, +8.7] | 0.30 (0.59) | 0.003 / 0.032 → 0.032 (**0.064**) | not detected | holds unadjusted (90% CI inside); **not after Holm m = 3** |
| 3 | +0.5 pp [−4.5, +3.1] | [−3.7, +2.8] | 0.84 (0.84) | → 7e-4 (0.002) | not detected | **established** (also after Holm) |

- Change from the log-level analysis: the conclusion is the same for w = 1 and w = 3.
- **For w = 2, equivalence was claimed by the uncorrected 90% CI. At drive level it holds only uncorrected (p_equiv = 0.032); Holm over the three w gives 0.064.**
- Only 8–9 of 12 drives are informative.
- Total-horizon FDE5 (±0.5 m predeclared, secondary): w = 1 +0.47 [−0.04, +1.02]; w = 2 −0.34 [−0.67, +0.12]; w = 3 **−0.81 [−1.11, −0.46]**, p = 5e-4, Holm 0.0015 (early is better). No equivalence at any w.
- Transient windows (equal exposure), amplification: w = 1 +8.0 [−0.8, +15.4], equivalence not established; w = 2 +1.9, p_equiv Holm 0.019; w = 3 −2.1, p_equiv Holm 0.009.

Common post-release horizon (POST-HOC; the ±10 pp and ±0.5 m margins are applied post-hoc), persistent GT:

| metric | N = 1 (326 units, 12 drives) w = 1 / 2 / 3 | N = 2 (277, 11) |
|---|---|---|
| exceed-3 m diff-in-diff (pp) | +5.2 [−5.2, +10.5] / +12.9 [+0.7, +19.5] / +4.1 [−8.1, +10.2]; p 0.37 / 0.09 / 0.46; equivalence not established for any w (p_equiv Holm 0.16 / 0.66 / 0.16) | +10.7 / +6.9 / −9.5; all p ≥ 0.11; equivalence not established |
| GT error diff-in-diff, (win − normal at same pose), early − late (m) | +0.36 / +0.55 / +0.71, all p ≤ 1e-3 (late reduces error vs Normal more) | +0.62 / +0.74 / +0.85 (= the original exp-10 extension) |
| raw GT error at rel + N, early − late (m) | −1.11 / −1.22 / −1.36, p ≤ 5e-4 (early lower: evaluated 2 poses earlier) | −1.12 / −1.31 / −1.51 |
| recovery of GT error at rel + 1, same-pose (Normal − win)/(Normal − Full) | s1 89.7 / 97.4 / 99.2% vs s3 53.2 / 66.1 / 72.1% | – |

- Transient windows, N = 1, exceed-3 m: +3.9 / +4.9 / +2.1 pp, equivalent within ±10 pp after Holm (0.041 / 0.041 / 0.037; post-hoc margin).

[interp]
- **None of the analyses shows early corrections reducing *binary* amplification more than late ones.**
- On continuous error, the sign depends on how the pose-time confound is handled:
  - raw error and same-pose recovery favour early;
  - difference-in-differences favours late.
- So no "early-window specialness" claim and no "position does not matter" claim can be supported.
- The defensible statement: "at the predeclared binary endpoint and total horizon, early − late is within ±10 pp for w = 3 (and for w = 2 only uncorrected). w = 1 is unresolved."
- **Do not generalize w = 2 or w = 3 equivalence to w = 1, to s = 2 or 4, to A+, or to other horizons.**

### C2.3 Separating position from length (`fig_position_vs_length.png`)
- **A. Position at fixed w, total horizon**: amplification level vs s. It falls slightly or stays flat with later s. Late starts have 2 fewer free poses.
- **B. early − late per w**: total horizon vs common post-release horizon N = 1 / 2, with the ±10 pp band.
- **C. Length at fixed s = 1**: recovery fraction vs w (P1-B persistent and transient, R9).
- **D. Position × length at equal post-release horizon N = 1** (continuous same-pose recovery). It falls steeply with s.
- Length effect [computed]: large and monotone in every dataset (panel C, and C2.4 below).
- Position effect: small or ambiguous on the binary endpoint (A, B). Large but confounded with absolute pose on the continuous endpoint (D).

### C2.4 "90% recovery": exact denominators and drive-cluster CIs [computed; `c2_recovery_fractions.csv`]
- Fraction f = (Normal − window) / (Normal − Full).
- Denominator: Normal and Full from **the same run and the same units**:
  - P1-B: full_gt;
  - R9: gt_history in the R9 run;
  - R8: gt_history in the R8 run.
- Ratio of drive totals; drive bootstrap 95% CI (B = 10,000). "P(f ≥ .9)" = bootstrap share.
- Metric: GT amplification at pose 9, scene-weighted (unit-weighted in brackets where it differs).

R9 (recovery re-run of exp 9), all A− (365 units / 52 scenes / 13 drives). Normal 45.1%, gt_history 3.4% (scene-weighted); unit-weighted 46.6% / 4.1%.

| row | intervention | corrected positions | corrected forwards / pairs per unit | free poses after release | f [95% CI] | P(f ≥ .9) |
|---|---|---|---|---|---|---|
| win1 | persistent window, start t*+1 | 1.00 | 6.80 / 6.80 | 5.80 | 46.7% [29.6, 62.1] (unit 47.1) | 0.00 |
| win2 | persistent | 1.98 | 6.80 / 12.6 | 4.82 | 76.6% [62.7, 88.0] (76.1) | 0.01 |
| win3 | persistent | 2.96 | 6.80 / 17.4 | 3.84 | **90.1% [81.3, 99.0]** (unit 89.7 [79.5, 98.9]) | 0.56 (unit 0.50) |
| win4 | persistent | 3.90 | 6.80 / 21.3 | 2.90 | 95.8% [93.3, 100] (96.1) | 1.00 |

- FDE5 fractions: 57.5 / 82.1 / 92.5 / 96.5%.
- **Do not write "2–4 steps give 90%".**
  - w = 2 gives 77% (CI upper bound 88%).
  - w = 3 gives 90% but its CI spans 81–99%; it is about 50/50 whether f ≥ 0.9.
  - w = 4 is the first length whose whole CI is above 90%.

R8 Recent-GT (the "one previous token gives 93%" claim), all A−:
- f = 93.6% [90.3, 99.8] scene-weighted; **92.9% [89.4, 99.1] unit-weighted.**
- Denominator: R8 Normal 46.9% − R8 gt_history 4.4% (unit).
- **Recent-GT is not a single 1-token correction.** It is a repeated replacement of the most recent context position by GT at **every** generation step: **6.80 applications per unit** (positions t*+1..8, one per forward), of which **4.30 are effective** (GT token ≠ own token).
- On t* ≤ 2 it is 93.1% scene-weighted (7.21 applications, 4.57 effective).

P1-B (A−, t* ≤ 2: 326 units / 46 scenes / 12 drives). Normal 48.3%, full_gt 3.6%.

| | s = 1 | s = 2 | s = 3 | s = 4 |
|---|---|---|---|---|
| persistent w1 | 48.3 [27.6, 65.6] | 62.9 [48.7, 74.2] | 72.1 [54.8, 82.7] | 74.8 [59.9, 82.3] |
| persistent w2 | 77.0 [64.5, 86.9] | 80.1 [66.2, 89.4] | 86.6 [71.3, 94.7] | 87.9 [74.0, 94.7] |
| persistent w3 | 89.0 [79.6, 98.2] | 92.4 [87.7, 100] | 90.1 [79.2, 95.9] | 92.1 [83.6, 96.8] |
| transient w1 | 38.0 [21.2, 55.0] | 52.8 | 55.8 | 54.1 |
| transient w2 | 55.0 [41.3, 67.5] | 69.8 | 59.4 | 61.3 |
| transient w3 | 78.3 [64.0, 86.7] | 75.5 | 73.6 | 72.4 |

- Recent-GT (repeated; 7.21 applications per unit, 4.64 effective): 91.1% [87.8, 96.2].
- Persistent windows hold w positions and are visible in 8 − t* − s forwards: s1 7.21, s2 6.21, s3 5.21, s4 4.21.
- Transient windows: w forwards, one position each.
- Effective pairs, persistent s1: 6.16 / 10.20 / 12.58 for w = 1 / 2 / 3.
- The FDE5 version reverses the position ordering: persistent w3 is s1 92.2%, s4 62.1%.
- On the same basis (R9 restricted to t* ≤ 2), R9 win1–4 = 46.2 / 77.6 / 89.3 / 95.4%. This is consistent with P1-B s1.

R8R9 nature [fact]:
- 220/220 rates reproduced exactly (difference 0.0 pp).
- This is a **reproducibility audit**: the same scripts, GPU, seed and units were re-run after the original numbers were known.
- It is **not** an independent confirmation and adds no new independent samples. All R9/R8 fractions above come from the same 13 A− drives as the original.

## What could not be checked / candidates for supplementation
1. **No common horizon with equal absolute pose and equal post-release horizon.** This is structurally impossible for different starts. The 10-token horizon limits N to ≤ 3, and only t* ≤ 3 − N units qualify. A longer post-release horizon (the exp-10 20-token extension) is not on disk. **Candidate for supplementation** if a position claim is needed.
2. **R9 length contrasts are confounded with post-release horizon** (free poses fall with w). R9 has per-pose trajectories (`trajectory_pred`), but no equal-horizon length contrast was predeclared and I did not construct one. It is possible from the data but not done.
3. **No turn-rate or constant-velocity baseline fitted on development data** exists. Only the repeat-p CM baseline (pre-fixed, no GT) exists. Excess over "extrapolation" is therefore excess over this one CM. This could be computed post-hoc from codebook geometry without a GPU, but parameter selection would need development data. Not done.
4. **CM_0 was not stored** and was rebuilt with the numpy decoder copy (≤ 1.9e-5 m error vs the real decoder on CM_p). The real decoder was not run (no GPU).
5. **Window-trace verification**: in the P1-B main run it remains at smoke level (P1-B analysis §1). I used the logged counts only.
6. No predeclared margin exists for any C1 excess contrast or for the C2 common-horizon contrasts. Their equivalence is either "not tested" (C1) or post-hoc (C2).
7. P1-B A+ / pooled were not re-analysed at drive level here.

## Files (all in this directory)
- `BASELINE_WINDOW_RECHECK.md`: this report.
- `c1_excess_effects.csv`: 192 rows = 4 families × 3 groups × 16 metrics. Columns include drive CI, sign-flip CI, p, Holm, three weightings and log-level p.
- `c1_predeclared_and_gt_drive.csv`:
  - P1A-1/2/3 at drive level, with TOST against the predeclared margins;
  - original OC − CM;
  - GT metrics, kept separate;
  - OC_refR applications per unit.
- `c1_units_counterfactual.parquet`: per-unit counterfactual changes.
- `c1_matching_checks.json`: matching checks.
- `c2_position_length.csv`: total-horizon and common-horizon early − late per w, with difference test, TOST and Holm.
- `c2_recovery_fractions.csv`: fractions, denominators, drive CIs, P(f ≥ .9), applications per unit, free poses.
- `c2_levels_p1b.csv`: P1-B levels per condition.
- Figures: `fig_position_vs_length.png`, `fig_p1a_excess.png`.
- `c_run_meta.json`: inputs and their sha256, seed, B.
- Script: `/root/VLA/autovla_misalignment_poc/scripts/review_followup2/c_baseline_window.py`
  - Run: `nice -n 19 /root/miniforge3/envs/autovla/bin/python c_baseline_window.py`
