# B. SpatialVLA success rate: direct effects vs interactions, equivalence (review_followup2, 2026-10-13)

**This is a post-hoc analysis of existing data** from experiments 34, 35 and P1-D.
- No GPU was used and no model was run. CPU only, `nice -n 19`.
- Existing files were not modified.
- The source experiments were preregistered, but this analysis was not. Some rows re-derive a preregistered judgment (P1-D S1–S4) or the P0-B judgment for exp 34; these are labelled.
- The ±5 pp margin is a **post-hoc sensitivity margin proposed by this review**. It is not a field standard. ±10 pp is the original margin.
- All differences are in percentage points (pp), never relative %.

Files in this directory:
- `DIRECT_EFFECTS.csv` (819 rows): experiment × config × contrast × task level × clustering.
- `B4_TRAJECTORY_VS_SUCCESS.csv`, `INTERPRETATION.csv`, `condition_definitions.csv`, `condition_record_check.csv`.
- `episodes.csv`, `pairs.csv`, `b_stats_meta.json`.
- Figures: `fig_direct_effects.png`, `fig_tcp_vs_success.png`.

Code is in `scripts/review_followup2/`:
- `b_extract.py`, `b_check_conditions.py`, `b_direct_effects.py`, `b_figures.py`.
- Regenerate everything with `bash b_regenerate.sh`.

## 0. Units, clusters, estimand, inference

**Units (intention to treat).** Every recorded episode of the assigned protocol is used. Success is taken as recorded (done at step 80). Nothing is excluded.
- All episodes are 80 steps.
- P1-D `restore_ok` = 100%.
- Episode counts: exp 34: 720. exp 35: 3,360. P1-D: 4,800.

**Pairing.** All conditions and configs of a unit (task, seed/unit) move together. Interactions are formed per unit before resampling.

**Estimand (unchanged from the originals).**
- Within a task: the unit-weighted mean paired difference, which equals the difference of success rates.
- Pooled: the task-equal mean of the two task means.
- Each task is reported first, then pooled.

**Clusterings.** The cluster ids come from A2 (`A_spatialvla_states/run_20261013/state_manifest.csv`).

| clustering | coke | move_near |
|---|---|---|
| `primary` = initial_state_cluster_id | every seed is its own cluster (40 / 120 / 120) | 29 (exp 34), 53 (exp 35), 120 (P1-D) |
| `physical` (conservative) | as primary | **10 layouts**, the same 10 in every experiment |
| `episode` | original unit, used only to reproduce the original judgments | |

- For P1-D, primary = episode.
- Pooled cluster counts G: exp 34 69/50, exp 35 173/130 (r1e4, r4e1; seeds 0–119) and 69/50 (seeds 0–39), P1-D 240/130.

**Inference** (`stats_core`; details in the `b_direct_effects.py` docstring):
- **Difference test:** two-sided sign-flip on cluster totals.
  - Exact enumeration if G ≤ 26; otherwise Monte Carlo with 100,000 draws, p = (b+1)/(R+1).
  - Assumption: under H0 the cluster totals are independent and symmetric about 0. Conditions are deterministic interventions, not randomised assignments, so this is a conditional test, not a randomisation test.
- **Primary 95% and 90% CIs:** sign-flip test inversion.
  - Exact for G ≤ 26 (move_near physical, 10 layouts). Otherwise a Monte Carlo inversion with one fixed sign matrix (R = 20,000, seed 20261013 + 99).
  - The task-stratified cluster bootstrap percentile CI (B = 10,000) is reported alongside in `boot_ci*`.
  - No CI came out unbounded (`ci_unbounded` = 0 rows). The smallest attainable two-sided p with 10 layouts is 2⁻⁹ ≈ 0.002.
- **TOST:** `stats_core.tost(method="signflip")` gives `p_lower` (H0 θ ≤ −m) and `p_upper` (H0 θ ≥ +m), with p_equiv = max.
  - Run at m = 10 pp and m = 5 pp.
  - The equivalence column uses Holm-adjusted p_equiv within the stated family. The unadjusted p_equiv and "90% CI inside ±m" are also given.
- **Holm families:** one per family × task level × clustering. The difference p is Holm-adjusted within the same family.
  - P1-D S1–S4: m = 4.
  - P1-D core direct: {δR, δC} × {r1e4, r4e1}, m = 4.
  - exp 35 core direct, seeds 0–119: m = 4.
  - exp 34 closed loop: {δR, δC} × {perp_left, opposite}, m = 4. P0-B's family was m = 6 because it included the 2 token contrasts.
  - exp 35 core interactions r4e1 − r1e4: m = 2 (preregistered main contrast).
  - P1-D r4e1 − r1e4: m = 2 (post-hoc, cross-axis; the protocol said not to make this comparison).
  - Appendix families are listed in the `family` column.
- **Judgments:** difference from 0 and equivalence are separate columns (`difference_judgment`, `equivalence_judgment_m10/m5`, `label_m*`). Both can hold at once. "Uncertain" means neither holds.
- **Compatibility check:** F−C rows are mirrors of C−F rows. The estimate and the 95%/90% CIs were checked to reverse exactly (assertion in code). The TOST sides also reverse: exactly for exact tests, and within Monte Carlo error otherwise.

## B1. What each condition generates, puts in context and executes (from code, verified on the records)

Full table: `condition_definitions.csv`. Record check: `condition_record_check.csv`, which covers every chunk generated inside the intervention schedule.

| condition | context step-1 translation token (decoding steps after it) | executed step-1 translation | record check |
|---|---|---|---|
| N | g | g | exec = g, ctx = g: 100% |
| F | p = perturb_token(g, 0.3, dir) | p | exec ≠ g, ctx ≠ g: 100% (P1-D: exec = p, ctx = p: 960/960 per arm) |
| C | g | p | ctx = g: 100%, exec ≠ g: 100% (exp 34 perp_left: 1/1,280 chunks with p = g) |
| R | p | **g** | exec = g, ctx = p: 100% |
| FD / CD (P1-D r1e4 only) | p / g | p, **plus** offset (1 − w0)Δ at each injection step | offset ≠ 0 in 960/960 injections |
| null (P1-D) | g (shadow uses p) | g | bit-identical to N in 320/320 episodes |
| gen / natural_gen | model's own `predict_action` | own | numeric-path rerun, not a context intervention |

Schedule:
- exp 34/35: every chunk generated at t ∈ [8, 24). That is 16, 8 and 4 chunks for r1, r2 and r4.
- P1-D: chunks generated at t = 8, 12, 16, 20 (4 in every config).

**Does R execute the unperturbed action while only the context is perturbed?**
- **Yes for the step-1 translation token.** It is always g.
- **But R is not a "no execution change" control.** The tokens generated after p (step-1 rotation/gripper, and chunk steps 2–4) are executed whenever the executor uses them:
  - r1e4: the ensemble takes 43% of its weight from age ≥ 1 predictions.
  - r4e1: steps 2–4 are executed directly.
  - r2e1: step 2 is executed.
  - r1e1: only step-1 rotation and gripper reach execution.
- P1-D measured the context-attributable executed translation per episode: 0 / 0.10 / 0.32 / 0.044 m for r1e1 / r2e1 / r4e1 / r1e4 (B4).
- So δR measures the success effect of a context perturbation **including** its leakage into execution through the generated suffix.

**F vs C.**
- F and C execute the same kind of perturbation (p at step 1). They differ only in the context used to generate the rest of the chunk.
- The executed step-1 token is p in both, by construction and in the records.
- Because later states differ, later chunks differ, so the realised executed dose is not identical per episode.

**Dose matching (P1-D design B) was a predefined control, not post-hoc selection.**
- The FD/CD offset (1 − w0)·(phys(p) − phys(g)) was computed online from the model's own g and p at every injection. It was fixed in the protocol committed before the main run (b212637, design B).
- No rollouts were selected on achieved dose or outcome.
- The dose matching is approximate per episode, because states at the injection steps differ. P1-D reports 0.230–0.231 m/ep in every arm.
- Within-config contrasts here are intention-to-treat by assigned protocol. No contrast conditions on achieved dose, survival or early termination; there was no early termination.

## B2/B3. Core direct effects (pp; 95% CI [90% CI]; sign-flip inversion)

Sign conventions:
- δR = S(R) − S(N).
- δC = S(C) − S(F); + means correction benefit.
- F − C = −δC.

Columns:
- `p_eq10` and `p_eq5` are the Holm-adjusted TOST p values. ✓ means equivalence holds after Holm (p < .05).
- No core direct effect is different from 0 after Holm under the primary or physical clustering.
- Columns marked "phys" use the physical clustering (move_near 10 layouts). Coke is identical under both clusterings.

### P1-D (n = 240; primary = episode; restore verified)

| config | contrast | coke | move_near (primary) | move_near (phys) | pooled (primary) | pooled (phys) | p_eq10 prim / phys (pooled) | p_eq5 prim / phys (pooled) |
|---|---|---|---|---|---|---|---|---|
| r1e4 | δR | 0.0 [−6.2, 6.2] ✓10 | +1.7 [−5.4, 8.8] ✓10 | +1.7 [−8.8, 12.8] ✗ | +0.8 [−3.8, 5.5] | +0.8 [−4.9, 6.5] | 0.000 ✓ / 0.002 ✓ | 0.16 / 0.30 |
| r1e4 | δC | +4.2 [−2.0, 10.9] | +0.8 [−6.8, 8.5] ✓10 | +0.8 [−2.6, 4.3]* ✓10 | +2.5 [−2.6, 7.6] | +2.5 [−1.0, 6.5] | 0.006 ✓ / 0.002 ✓ | 0.27 / 0.31 |
| r4e1 | δR | +1.7 [−8.3, 11.5] | +0.8 [−8.3, 10.1] | +0.8 [−8.1, 10.0] | +1.2 [−5.6, 8.0] | +1.2 [−5.1, 7.7] | 0.006 ✓ / 0.010 ✓ | 0.27 / 0.31 |
| r4e1 | δC | −4.2 [−14.3, 5.8] | +4.2 [−5.0, 13.3] | +4.2 [−10.5, 18.8] | −0.0 [−6.7, 6.8] | −0.0 [−8.6, 7.9] | 0.006 ✓ / 0.014 ✓ | 0.23 / 0.31 |

\* In this cell only 3 of 10 layouts have a nonzero total: the other 7 cancel within the layout. The sign-flip CI is narrower than the primary CI because of that cancellation. With 10 clusters the between-layout variance is itself very imprecise, so do not read this as stronger evidence.

Task-level equivalence (Holm, ±10):
- coke: holds only for r1e4 δR.
- move_near: holds only for r1e4 under primary (δR and δC). Under physical it holds only for r1e4 δC; **r1e4 δR loses it** (Holm 0.139).
- r4e1: no task-level equivalence under either clustering. Every r4e1 task CI crosses ±10.

### exp 35 (r1e4, r4e1: seeds 0–119, n = 240)

| config | contrast | coke (120 cl) | move_near primary (53) | move_near phys (10) | pooled primary | pooled phys | p_diff prim (raw/Holm) | p_eq10 prim / phys |
|---|---|---|---|---|---|---|---|---|
| r1e4 | δR | −7.5 [−15.0, 0.0] | +5.8 [−10.0, 21.1] | +5.8 [−16.3, 22.0] | −0.8 [−9.6, 7.5] | −0.8 [−11.6, 7.6] | 0.93 / 0.93 | 0.082 / 0.20 |
| r1e4 | δC | +0.8 [−9.7, 11.3] | −15.0 [−29.1, 0.0] | −15.0 [−38.8, 8.0] | −7.1 [−15.9, 2.1] | −7.1 [−18.3, 5.5] | 0.15 / 0.54 | 0.37 / 0.47 |
| r4e1 | δR | −5.8 [−14.5, 2.9] | −6.7 [−18.9, 6.2] | −6.7 [−29.4, 10.7] | **−6.2 [−13.6, 1.1]** | −6.2 [−16.0, 2.7] | **0.136** / 0.54 | 0.37 / 0.47 |
| r4e1 | δC | +1.7 [−8.7, 12.1] | −10.8 [−25.4, 4.5] | −10.8 [−29.5, 8.3] | −4.6 [−13.5, 4.8] | −4.6 [−13.8, 6.0] | 0.38 / 0.77 | 0.37 / 0.41 |

Comparison with the episode-level (original-unit) analysis:
- r4e1 δR −6.2 pp: episode-level sign-flip p = 0.041 (original McNemar p = 0.040). Under the cluster analysis it is p = 0.136 (primary) and 0.256 (physical), **unadjusted**. The original "significant reverse effect at r4e1" does not survive initial-state clustering.
- r1e4 δC −7.1 (that is, F−C +7.1, "corrected < feedback"): p = 0.061 at episode level, 0.150 primary, 0.346 physical.
  - move_near alone: −15.0, episode p = 0.002 (Holm 0.01), primary p = 0.058, physical p = 0.28.
  - This move_near result also disappears.
- r1e4 δR pooled: episode-level TOST ±10 p_equiv = 0.001 (Holm 0.003). It would have been "equivalent". Under primary it is 0.021 raw / 0.082 Holm; under physical 0.049 / 0.198. **The equivalence is lost after clustering (rule 4).**
- No exp 35 core direct effect is equivalent at ±10 or ±5 after Holm under either clustering.

### exp 34 (official r1e4, seeds 0–39, n = 80)

| direction | contrast | coke (40) | move_near primary (29) | move_near phys (10) | pooled primary | pooled phys | p_eq10 Holm prim / phys |
|---|---|---|---|---|---|---|---|
| perp_left | δR | +5.0 [−5.6, 15.0] | +5.0 [−11.8, 21.1] | +5.0 [−20.0, 26.7] | +5.0 [−4.7, 14.3] | +5.0 [−5.9, 15.4] | 0.34 / 0.38 |
| perp_left | δC | 0.0 [−6.2, 6.2] ✓10 | +10.0 [−5.9, 25.0] | +10.0 [−7.1, 33.3] | +5.0 [−3.0, 13.2] | +5.0 [−3.6, 13.3] | 0.34 / 0.38 |
| opposite | δR | −10.0 [−22.2, 0.0] | +7.5 [−14.3, 28.6] | +7.5 [−23.1, 33.3] | −1.2 [−13.5, 10.5] | −1.2 [−16.2, 11.4] | 0.34 / 0.38 |
| opposite | δC | +7.5 [−13.3, 28.6] | −10.0 [−30.0, 6.7] | −10.0 [−30.8, 5.6] | −1.2 [−15.0, 12.1] | −1.2 [−13.8, 12.5] | 0.34 / 0.33 |

- **The P0-B judgment is reproduced:** ±10 pp equivalence is not established for any of the 4 pooled closed-loop contrasts (episode-level p_equiv 0.051–0.134). Clustering does not change it.
- The only equivalence is coke perp_left δC: 0.0, Holm p_equiv 0.001 under both clusterings. ±5 pp is not established there (0.57).
- The opposite direction shows task effects in opposite directions that cancel in the pooled mean. coke and move_near have opposite signs for both δR (−10.0 vs +7.5) and δC (+7.5 vs −10.0). None is significant.

### Interactions (pooled; per-task values are in the CSV)

| interaction | primary | physical | ±10 Holm p prim / phys | ±5 Holm prim / phys | original |
|---|---|---|---|---|---|
| P1-D S1: δR, r4e1 − r1e1 | −0.8 [−9.3, 7.7] | −0.8 [−9.2, 6.6] | 0.037 ✓ / 0.046 ✓ | 0.52 / 0.48 | −0.8, equivalent ±10 |
| P1-D S2: δC, r1e4(design B) − r1e1 | +0.8 [−7.1, 8.8] | +0.8 [−8.0, 9.2] | 0.037 ✓ / 0.046 ✓ | 0.52 / 0.48 | −0.8 (F−C sign), equivalent |
| P1-D S3: δC, r4e1 − r1e1 | −0.4 [−9.5, 8.6] | −0.4 [−11.6, 9.2] | 0.037 ✓ / 0.047 ✓ | 0.52 / 0.48 | +0.4 (F−C sign), equivalent |
| P1-D S4: δR, r1e4 − r1e1 | −1.2 [−7.6, 5.2] | −1.2 [−7.8, 4.2] | 0.014 ✓ / 0.026 ✓ | 0.52 / 0.48 | −1.3, equivalent |
| P1-D δR, r4e1 − r1e4 (post-hoc, cross-axis) | +0.4 [−7.5, 8.3] | +0.4 [−7.3, 8.3] | 0.016 ✓ / 0.021 ✓ | 0.25 / 0.24 | – |
| P1-D δC, r4e1 − r1e4 (post-hoc) | −2.5 [−10.8, 5.8] | −2.5 [−12.0, 5.8] | 0.039 ✓ / 0.061 ✗ | 0.28 / 0.30 | – |
| exp 35 δR, r4e1 − r1e4 (preregistered) | −5.4 [−17.3, 7.0] | −5.4 [−18.4, 9.7] | 0.25 / 0.26 | 0.67 / 0.63 | −5.4, n.s. |
| exp 35 δC, r4e1 − r1e4 | +2.5 [−8.9, 14.1] | +2.5 [−8.7, 12.5] | 0.20 / 0.14 | 0.67 / 0.63 | +2.5 (C−F), n.s. |

- **The P1-D S1–S4 ±10 pp equivalence survives initial-state clustering and Holm over the 4-member family.** It holds under primary (= episode) and under the conservative physical clustering, but only just under physical (Holm 0.026–0.047).
- **At ±5 pp, none of S1–S4 holds** (Holm ≈ 0.5).
- Per task, no S1–S4 interaction is equivalent at ±10 after Holm. coke S3 −9.2 [−22.8, 4.8] and move_near S3 +8.3 have opposite signs, so the pooled S3 near 0 partly reflects cancellation.
- **Small interactions do not imply "no intervention effect".** The direct effects must be read from the direct-effect rows.

**Executed-perturbation (positive-control) rows**, all clusterings in the appendix CSV, show that task outcome does respond to the executed perturbation:
- exp 35 r1e4 C−N: −22.9 [−34.9, −11.4], Holm p = 0.001.
- P1-D r4e1 C−N: −11.7 [−18.3, −5.0], Holm p = 0.009.
- exp 34 opposite C−N: −21.2, Holm p = 0.041 (primary); physical Holm p = 0.094.

## B4. Generated / executed / physical trajectory change vs success (same units; pooled, primary clusters)

All definitions are in the `b_extract.py` docstring.
- **Actual TCP mean distance:** 100·mean over t = 8..79 of ‖p_a(t) − p_b(t)‖. p(t) is the TCP position after env.step at control step t. 72 steps, denominator 72, unit cm.
- **Time-integrated error:** the sum over the same 72 steps, in cm·step.
- **Command accumulation:** ‖Σ_{t=0..79} a_a(t)[:3] − Σ a_b(t)[:3]‖. This is the commanded world-vector delta in m. It is not the actual TCP.
- **Generated suffix change:** the fraction of the 11 tokens after the step-1 translation token that differ, plus the translation-token distance over chunk steps 2–4 (m).
  - P1-D: the a-episode chunk vs its shadow counterfactual-context chunk at t = 8, 12, 16, 20 (same observation).
  - exp 34/35: the a vs b chunk at t = 8, prefix-identical pairs only (94–100% of pairs).

| row | n | suffix tokens changed | ctx-attributable exec. translation (m/ep) | command accum. (m) | TCP mean (cm) | TCP area (cm·step) | success effect (pp) [95% CI] | discordance |
|---|---|---|---|---|---|---|---|---|
| exp 34 r1e4/opp R−N | 80 | 58% | – | 0.232 | not recorded | not recorded | −1.2 [−13.5, 10.5] | 21% |
| exp 34 r1e4/opp C−F | 80 | 58% | – | 0.388 | not recorded | not recorded | −1.2 [−15.0, 12.1] | 34% |
| exp 34 natural_gen − N | 80 | 6% | – | 0.140 | – | – | +8.8 [0.0, 17.9] | 14% |
| exp 35 r1e4 R−N | 240 | 57% | – | 0.233 | 4.06 | 292 | −0.8 [−9.6, 7.5] | 19% |
| exp 35 r1e4 C−F | 240 | 57% | – | 0.301 | 5.58 | 402 | −7.1 [−15.9, 2.1] | 30% |
| exp 35 r4e1 R−N | 240 | 57% | – | 0.349 | 5.90 | 425 | −6.2 [−13.6, 1.1] | 20% |
| exp 35 r4e1 C−F | 240 | 57% | – | 0.379 | 6.90 | 497 | −4.6 [−13.5, 4.8] | 28% |
| exp 35 natural_gen − N (r1e4 / r4e1; seeds 0–39) | 80 | 7% / 3% | – | 0.160 / 0.174 | 2.78 / 2.53 | 200 / 182 | +7.5 [0.0, 15.0] / −2.5 [−11.6, 6.9] | 7.5% / 12.5% |
| P1-D r1e4 R−N | 240 | 51% | 0.044 | 0.153 | 2.67 | 193 | +0.8 [−3.8, 5.5] | 13% |
| P1-D r1e4 C−F | 240 | 51% | 0.044 | 0.182 | 3.07 | 221 | +2.5 [−2.6, 7.6] | 16% |
| P1-D r1e4 CD−FD (design B) | 240 | 54% | 0.044 | 0.240 | 4.59 | 330 | +1.2 [−4.9, 7.4] | 23% |
| P1-D r4e1 R−N | 240 | 52% | 0.318 | 0.353 | 5.95 | 428 | +1.2 [−5.6, 8.0] | 28% |
| P1-D r4e1 C−F | 240 | 55% | 0.325 | 0.356 | 6.24 | 449 | −0.0 [−6.7, 6.8] | 28% |
| P1-D r1e1 R−N | 240 | 49% | 0.000 | 0.162 | 2.98 | 215 | +2.1 [−3.2, 7.3] | 17% |
| P1-D gen − N (r1e1 / r4e1; units 0–39, branch at t = 8) | 80 | – | 0 | 0.095 / 0.161 | 1.24 / 2.00 | 90 / 144 | +7.5 [2.3, 13.3] / 0.0 [−7.1, 7.1] | 7.5% / 10% |
| P1-D null − N (all configs) | 80 | (shadow only) | – | 0 | 0 | 0 | 0 | 0% |

- Same-subset comparison rows (R−N and C−F on the rerun units) are in the CSV. For example, P1-D r4e1 R−N on units 0–39: TCP 6.35 cm vs rerun 2.00 cm; success −1.2 vs 0.0 pp.
- **Generated-suffix change is large and almost config-independent.** Roughly 50–58% of post-perturbation tokens change. It is about 10× the numeric rerun level (3–7%).
- Executed leakage and actual TCP distance scale with the execution structure:
  - r4e1 > r2e1 > r1e1 ≈ r1e4.
  - Context contrasts exceed the rerun noise in TCP: 2.7–6.9 cm vs 1.2–2.8 cm.
- The success direct effects stay near 0 with CIs of about ±5–10 pp (P1-D) or ±8–14 pp (exp 34/35).
- **The rerun mean effect and the paired discordance are reported separately here.**
  - natural_gen moves success by +7.5 to +8.8 pp with 7.5–14% discordance in r1 configs.
  - Discordance is **not** subtracted from any effect, and is **not** used as a "noise band".

## Interpretation rules (prompt section B), per case

Full list: `INTERPRETATION.csv` (primary and physical, per task and pooled).

| case | rule |
|---|---|
| P1-D r1e4 pooled, δR and δC (both clusterings) | **R2:** direct-effect equivalence holds at ±10 pp (Holm). The success effect may be bounded *for this config/margin only*. ±5 pp is NOT established. |
| P1-D r4e1 pooled, δR and δC (both clusterings) | **R2** at ±10 (Holm), not at ±5. Per task: **R3** (CIs about 18–29 pp wide, crossing ±10). |
| P1-D r1e4 coke δR; move_near δR (primary) and δC | R2 (±10 only). |
| P1-D r1e4 move_near δR under physical clustering | **R4:** equivalence lost after clustering into 10 layouts. **R3**. |
| P1-D r1e4 coke δC | R3. |
| P1-D S1–S4 interactions (pooled) | Interaction equivalence ±10 holds and survives clustering. Because the matching pooled direct effects are also equivalent (R2), the claim may cover both. Without the direct rows it would be R1 only ("the difference between configs is within ±10 pp"). |
| exp 35 r1e4 and r4e1, all task levels and both clusterings | **R3:** generation and trajectory deviation is confirmed; the success effect is uncertain. Pooled r1e4 δR: **R4** (episode-level ±10 equivalence lost). r4e1 δR −6.2: the episode-level significance is also lost (p 0.041 → 0.136 / 0.256). |
| exp 35 interactions r4e1 − r1e4 | Neither difference nor equivalence ("uncertain"). Not even R1. |
| exp 34 r1e4 (both directions, pooled and per task) | **R3.** Exception: coke perp_left δC, R2 at ±10. |

## Caveats

1. **exp 34/35 state drift; no restore verification** (A2 FINDINGS):
   - Resets were `reset(seed)` without reconfigure, so the state depends on history.
   - Drift reached 20–40× tolerance for rolling objects in move_near; all coke rows are flagged "physical state uncertain beyond tol".
   - No state or restore hash was recorded.
   - Pre-branch divergence: in exp 35, 107 of 2,400 condition-vs-natural pairs diverged before t = 8 (P1-D §0.3). Here, 94–100% of pairs had identical actions at t < 8.
   - Condition pairs share a cluster but are not bit-paired.
   - P1-D restore was verified: state and image hashes at every replayed step, `restore_ok` 3,840/3,840, null ≡ N 320/320.
2. **Only 2 tasks; move_near has only 10 physical layouts.**
   - Physical-clustering CIs rest on 10 clusters. Sign-flip inversion is exact but depends on a symmetry assumption. With 3–8 informative layouts the between-layout variance is poorly estimated, and some physical CIs are narrower than primary ones (see the * row).
   - P1-D move_near is physically **not** independent of exp 34/35: they share the same 10 layouts.
3. **The P1-D r4e1 − r1e4 interaction is a post-hoc cross-axis comparison.** The P1-D protocol excluded it. Its families are labelled.
4. **The ±10 pp equivalence of P1-D is a statement about the pooled two-task mean.** Opposite-signed task effects partly cancel: S3 coke −9.2 vs move_near +8.3, and r4e1 δC −4.2 vs +4.2. It does not bound per-task effects at ±10.
5. **P0-B's exp 34 family had m = 6** because it included the token contrasts. Here the closed-loop equivalence family is m = 4, which is less conservative, and the result is still "not established".
6. Sign-flip p values are conditional tests under symmetry, not randomisation tests. Monte Carlo p values carry error of about 0.001.
