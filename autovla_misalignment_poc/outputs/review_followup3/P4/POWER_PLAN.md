# POWER_PLAN — P4 SpatialVLA new physical states (CP2, planning only)

**Status: planning.**
- No confirmatory data exist.
- Existing SpatialVLA data were used **only** as planning inputs (discordance, cross-contrast correlation, within-state structure). Nothing here tests a P4 hypothesis.
- Tag: manipulation / SpatialVLA-4B / closed loop / SimplerEnv Google Robot.

**Code (CPU, `nice -n 19`)** in `scripts/review_followup3/cp2/`:
- `p4_planning_data.py`
- `p4_power_sim.py` (seed 20261014, 2,000 simulations per cell)
- `p4_layout_geometry.py`
- `stage3_cost_rates.py`
- `stage3_cost_plan.py`

**Outputs:**
- `planning/` and `power/` (this directory);
- `../STAGE3_COST_RATES.csv`, `../STAGE3_COST_PLAN.csv`.

**Relation to v1.** The stopped v1 attempt (`_stopped_v1_20261014/P4/`) was reviewed and then regenerated from scratch. Changes:
- scenario naming (`true0`; a CSV value "null" is read as missing by pandas);
- per-layout ICC table;
- cost rates read from logs;
- budget table.

## 1. Planning data (`planning/planning_params.csv`, `planning_within_state.csv`, `planning_corr.json`)

**Source.** Unit-level P1-D outcomes (restore-verified run; same arms and injection schedule as proposed): 120 units per task, 8 outcomes per unit = {N, F, C, R} × {r1e4, r4e1}.

**Paired discordance p_d** (share of units where the two arms disagree):

| task | config | δR (R−N) | δC (C−F) |
|---|---|---|---|
| coke | r1e4 | 0.117 | 0.142 |
| coke | r4e1 | 0.300 | 0.308 |
| move_near | r1e4 | 0.150 | 0.175 |
| move_near | r4e1 | 0.258 | 0.258 |

Exp 35 discordance is 0.17–0.34; it enters only through the stress scenario.

**Cross-contrast correlation within a unit:** |r| ≤ 0.22 (P1-D). It is kept in the simulation by resampling whole unit vectors.

**Within-state (layout) correlation, move_near** (units nested in the 10 layouts):
- ICC(1) of the paired differences = 0–0.075 in P1-D;
- 0.18–0.35 in exp 35, where duplicate seeds made identical trajectories.

**Design consequence.** SpatialVLA decoding is deterministic (C5a/C5b reruns are bitwise identical). A repeated rollout of the same state therefore adds no information, and several instructions per layout add correlated units (design effect 1 + (k − 1)·ICC). The proposed design is **one unit per new physical state**, so within-state correlation does not enter the primary analysis.

## 2. Simulation model and test

**Generative model.**
- Resample P1-D unit 4-vectors (δR r1e4, δC r1e4, δR r4e1, δC r4e1) with replacement.
- Apply a random joint sign per state, so the true mean is 0 and the discordance and correlation are kept.
- Then impose a scenario:

| scenario | definition |
|---|---|
| true0 | true effect 0 |
| small_conc | +2.5 pp on every contrast, both tasks |
| small_opp | +3 pp coke / −3 pp move_near (task × intervention 6 pp) |
| het_state | mean 0, ±10 pp state-specific effects |
| stress35 | discordance raised to 0.35 |
| boundary5 | +5 pp everywhere: the true effect sits on the ±5 margin |

**Test.** The confirmatory test is TOST (α = 0.05) on state-level paired differences with the sign-flip statistic, in its large-n normal form.

**Validation** (`power/power_validation.csv`): against a 5,000-flip Monte Carlo sign-flip at n = 120, the maximum |Δp| = 0.026 and the decision agreement is 98–98.5 %. Disagreements occur only at p ≈ 0.05.

**Families:**
- raw (per test);
- H4 = Holm within a task (all 4 must pass);
- **H8 = Holm over the 8 per-task tests (proposed primary).**

## 3. Required independent states per task for 80 % power (`power/power_required_n.csv`)

Columns are criteria (all-8 = all 8 per-task tests equivalent under Holm m = 8; "min per-test" = every single test reaches 80 %, unadjusted).

| scenario | ±10 pp, all-8 | ±10 pp, min per-test | ±7.5 pp, all-8 | ±5 pp, all-8 | ±5 pp, min per-test |
|---|---|---|---|---|---|
| true 0 | **400** | 300 | 700 | **1,500** | 1,200 |
| small concordant +2.5 pp | 700 | 400 | 1,500 | > 3,000 (power 0.28 at 3,000) | > 3,000 |
| task-opposite ±3 pp | 800 | 500 | 2,000 | > 3,000 (0.04) | > 3,000 |
| heterogeneous ±10 pp | 500 | 400 | 1,000 | 2,500 | 1,500 |
| discordance 0.35 | 600 | 400 | 1,000 | 2,500 | 1,500 |
| true effect +5 pp | 1,500 | 1,000 | never | never (≈ α by construction) | never |

**Holm impact.** Requiring all 8 tests under Holm instead of each test unadjusted raises n by about 1/3:
- ±10 pp: 300 → 400;
- ±5 pp: 1,200 → 1,500 (true 0).

**Prompt's rule of thumb** (`power/power_analytic_approx.csv`, n ≈ (1.645 + 1.282)² · p_d / ε²): 100–265 states per contrast for ±10 pp and 400–1,057 for ±5 pp, unadjusted. The simulation, which includes Holm and the worst contrast, needs more. The rule of thumb is not used.

**95 % CI half-width per contrast (true 0):**

| states per task | half-width |
|---|---|
| 100 | 6.6–10.9 pp |
| 400 | 3.3–5.4 pp |
| 500 | 3.0–4.9 pp |
| 800 | 2.4–3.9 pp |
| 1,500 | 1.7–2.8 pp |

The r4e1 contrasts are the wide ones.

**Task × intervention difference (±10 pp, unadjusted):**
- power 0.67–1.00 at 400–800 states under true 0;
- only 0.25–0.63 under the task-opposite ±3 pp scenario.

Interaction equivalence is therefore weakly powered, and the protocol keeps it secondary.

## 4. Cost (`../STAGE3_COST_PLAN.csv`, option O3)

**Episodes per design:**
- 8 episodes per state per task;
- + 5 % of states with 2 control episodes (U0′, U2);
- + a separate pilot of 16 states per task.

**Measured rates** (`../STAGE3_COST_RATES.csv`):

| label | rate | source |
|---|---|---|
| optimistic | 5.46 episodes/min | P1 SpatialVLA audit, 4 SAPIEN workers |
| conservative | 1.96 episodes/min | P1-D main run, 8 workers, GPU shared with an external process |

The P1 audit (amendment A4) showed that 8 workers thrash the page cache, so P4 uses 4.

| target (all-8 Holm, 80 %) | states / task | episodes | GPU h (5.46 / 1.96 ep/min) |
|---|---|---|---|
| ±10 pp, true 0 | 400 | 6,736 | 20.6 – 57.4 |
| ±10 pp, heterogeneous (**proposed n = 500**) | 500 | 8,356 | **25.5 – 71.2** |
| ±10 pp, discordance 0.35 | 600 | 9,976 | 30.5 – 85.0 |
| ±7.5 pp, true 0 | 700 | 11,596 | 35.4 – 98.8 |
| ±5 pp, true 0 | 1,500 | 24,556 | 75.0 – 209.2 |

**What a 48-GPU-h budget buys** (`power/power_budget48h.csv`):

*At 5.46 episodes/min: 800 states per task.*

| scenario | ±10 pp | ±7.5 pp | ±5 pp |
|---|---|---|---|
| true 0 | 1.00 | 0.92 | 0.11 |
| heterogeneous | 0.99 | 0.67 | – |
| stress (discordance 0.35) | 0.99 | 0.59 | – |

*At 1.96 episodes/min: 300 states per task.*

| scenario | ±10 pp |
|---|---|
| true 0 | 0.57 |
| heterogeneous | 0.14 |

## 5. Decision proposed for CP2 (human approval required before any P4 data)

1. **±5 pp per task is infeasible within 48 GPU h.**
   - It needs ≥ 1,500 states per task (75–209 GPU h).
   - Its power collapses if the true effect is ≥ 2.5 pp, because equivalence at ±5 cannot be shown when the true effect is close to the margin.
2. **Proposed narrowed confirmatory target: ±10 pp per task, Holm m = 8, n = 500 SimplerEnv variant states per task.**
   - Power 0.95 (true 0) / 0.80 (heterogeneous) / 0.73 (stress).
   - 25.5–71 GPU h depending on throughput.
   - If the pilot throughput is < 2.9 episodes/min, the 48-h budget is exceeded. The human then chooses between n = 400 and a larger budget, on throughput alone, before any confirmatory outcome.
3. **±7.5 and ±5 pp** are reported only as CI half-widths (descriptive precision), not as equivalence claims.
4. **What P4 does not resolve.** Even if P4 succeeds:
   - "only two tasks" remains;
   - the real-to-sim correspondence of variant states is not validated;
   - per-task ±5 pp stays unresolved.
