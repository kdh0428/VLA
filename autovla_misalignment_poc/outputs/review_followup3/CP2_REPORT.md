# CP2 REPORT — review_followup3 (2026-10-14)

> **Historical CP2 submission (2026-10-14, d70d19c).**
> - Decisions: `CP2_DECISION.md` (444299f).
> - Executed: A5 re-verification PASS 20/20 and O0 gate PASS (fae513c).
> - Final state: `STATUS.md`, `CLAIM_EVIDENCE_MATRIX.md`, `REVIEW_COVERAGE.md` (2026-10-15).

**Checkpoint.** CP2 = Stage 0–2 complete, before any Stage-3 GPU run. **No Stage-3 GPU work has been run.** This round used the CPU only (`nice -n 19`), modified no existing result, and committed nothing.

**Scope.** The manuscript is a driving-VLA study. Its main evidence is AutoVLA in open-loop 5 s plan evaluation; manipulation (SpatialVLA, OpenVLA) is complementary evidence.

**Files submitted** (all under `outputs/review_followup3/` unless noted):

| file | content |
|---|---|
| `STATUS.md` | every task with status, reason, claim impact and commit |
| `/root/VLA/paper_quantitative_package/numbers.json` + `numbers_sources.csv` | 2,117 machine-merged rows from P0 / P1 / P2, each with row ID, value, CI, p, status, tags and source file/key |
| `CLAIM_EVIDENCE_MATRIX.md` | draft; claims tagged by domain / model / evaluation mode |
| `REVIEW_COVERAGE.md` | draft verdicts for M1–M7, m1, m2, m4 |
| `P3/MATCHED_REVERSE_PROTOCOL.md` (+ `MATCHED_REVERSE_RESULTS.md/.csv` = NOT_RUN) | P3 plan |
| `P4/SPATIAL_STATE_EXPANSION_PROTOCOL.md`, `P4/POWER_PLAN.md` (+ `P4/planning/`, `P4/power/`) | P4 plan |
| `P5/DISJOINT_REPLICATION_PROTOCOL.md` (+ `DISJOINT_REPLICATION_RESULTS.md/.csv` = NOT_RUN) | P5 plan |
| `STAGE3_COST_RATES.csv`, `STAGE3_COST_PLAN.csv` | measured rates and cost plan |
| `RUN_COMMANDS.md` | commands to reproduce |

## 1. What is done

**Stage 0 / P1 AutoVLA: PASS** (protocol 0ef9334, results 47efb95).
- Paths: H7 (Recent-GT / Full-GT), H8 (Reverse-N), H9 (windows), P1A (self-reference, coordinate accumulation, rollout-saving path).
- Controls c1–c5 under greedy and T = 0.01: 0 failures.
- AutoVLA numbers on these paths change status from PROVISIONAL to **VERIFIED_MACHINERY_MANIFEST_SCOPE**. Caveat: the 21 manifest units do not retro-validate runs outside the manifest.
- Still PROVISIONAL: DM (H32 / exp 32) and the P1-B start-position script.

**P0-A … E: done** (e9907fc). Main points:
- −4.1 / +3.6 pp is an evaluation-population estimand.
- 52/2,747 is an evaluation-set frequency.
- +4.37 m now has one authoritative row: [+3.34, +5.17], Holm 7.3e-4.
- R9 has 13 drives in every window.
- P1-C cannot reach Holm < .05 under 6 day–vehicle components; this is a resolution limit.
- There are 0 day-and-vehicle-disjoint logs.
- OpenVLA is descriptive.

**P2-A/B: done** (5b158f2).
- CM is already a body-frame constant-turn-rate repeat.
- The model exceeds every kinematic control and the learned non-AR motion extrapolator **only in A−**. In A+ and over all scenes it is not distinguished.
- The P2 inputs are on the verified P1A path, including the rollout-saving path.

**P1 SpatialVLA: PENDING_DECISION.** Recorded status: "PASS on actions/state; strict C2 FAIL (bookkeeping) — pending human decision". See §2.

**P1 Impromptu / OpenVLA: NOT_RUN** (spec only). §7.1 and §7.2 rely on machinery that has not been parity-verified.

## 2. Decision needed: A5 (SpatialVLA C2 null arm)

**What happened.**
- In the 20-unit audit, the null arm U2 (null (g, g) through the intervention path) produced bitwise-identical results to the untouched N run in every unit (20/20):
  - float64 executed actions;
  - every simulator fingerprint component;
  - exec/ctx tokens;
  - TCP.
- The strict pre-fixed rule nevertheless fails in 0/20 units. The failing component is the **executor fingerprint**, which also hashes the `cf` dict. P1-D's null mode fills that dict with the shadow decode **by design**, and the dict never reaches the executed action (`Executor.step` uses only `chunks` and the sticky state).
- The A5 check is post hoc, written after data were seen, and labelled so. Under it, all 20 units pass.

**Options for the human.**

| option | what it means | cost | status afterwards |
|---|---|---|---|
| (a) accept A5 as the C2 criterion | the post-hoc label is kept in every document | 0 GPU h | "PASS (A5 post-hoc criterion)"; P1-D numbers leave PENDING_DECISION_A5 |
| (b) keep the strict rule | fix the fingerprint to hash the policy state and `cf` separately, then re-run U2 + U0 on the same 20 units | ≈ 40 episodes, < 0.5 GPU h | strict PASS or FAIL decided by the re-run |

Option (b) is **recommended**, because it closes the issue without a post-hoc criterion.

**In both cases** the 20-unit audit does not retro-validate exp 34/35. Exp 35 numbers stay NOT_VERIFIED.

**Until decided:**
- §7.3, §7.4, Table 5 and App. D keep the review_followup2 wording;
- SpatialVLA P1-D rows are `PENDING_DECISION_A5` in `numbers.json`;
- SpatialVLA P1 is **not** marked PASS.

## 3. Stage-3 options and costs (`STAGE3_COST_PLAN.csv`)

GPU = RTX 5090 (GPU 1), shared with an external process that holds 13–26 GB. Each range runs from the optimistic to the conservative **measured** rate (`STAGE3_COST_RATES.csv`). Wall time is longer when the GPU is occupied.

| option | content | rollouts / units | GPU h | review items it would move |
|---|---|---|---|---|
| **O0** gate extension | add H32 (DM) and the P1-B start-position path to the AutoVLA Stage-0 manifest, c1–c5 | 72 path-unit-decodings | **0.4 – 0.8** | DM and early-window rows leave PROVISIONAL (M6 coverage, M7 completeness). No item becomes fully resolved. |
| **A5(b)** | SpatialVLA null-arm re-run with split hashes | ≈ 40 episodes | **< 0.5** | M6 SpatialVLA part → PASS (strict) or FAIL |
| **O1** P3 AutoVLA matched Reverse | P3 gate; frozen source bank; one-shot 2 × 3 factorial (primary); repeated-schedule block (secondary); P1-C 2,436 units + dev 1,408 units | ≈ 84,600 decoded 10-token plans | **2.3 – 6.0** | **M2: unresolved → resolved for AutoVLA**. The Impromptu part stays unresolved. |
| **O2** P3 Impromptu add-on | Impromptu no-op harness + gate; same factorial on the 300 exp 33 scenes (2,700 units) | ≈ 8,100 unit-passes | **11 – 22** | M2 Impromptu part → resolved within Impromptu; M6 Impromptu path verified. Effect sizes stay non-comparable across models. |
| **O3** P4 SpatialVLA variant states | **proposed:** ±10 pp per task, Holm m = 8, n = 500 states per task | 8,356 episodes | **25.5 – 71** | **M4 new-state part: unresolved → resolved at ±10 pp**. "Two tasks only" remains; ±5 pp per task stays unresolved. |
| O3-alt | ±10 pp, n = 400 per task / n = 600 per task | 6,736 / 9,976 episodes | 20.6 – 57.4 / 30.5 – 85 | as O3; lower / higher robustness |
| O3-±5 | ±5 pp per task | 24,556 episodes (1,500 states per task) | 75 – 209 | **not feasible within 48 GPU h**; unattainable if the true effect is ≥ 2.5 pp |
| O4 held-out-drive replication (not P5) | P1-C design on the 53 unused test logs (17 drives, sharing days and vehicles) | 3,654 natural scenes + ≈ 6,300 mechanism units | ≈ 4.9 + image fetch (images local for 4/53 logs) | **none**: not day/vehicle-disjoint, so M5/M1 external validation stays unresolved |
| P5 | day/vehicle-disjoint replication | – | – | impossible: 0 disjoint logs |

**Recommended package (for approval).** O0 + A5(b) + O1 + O3 (n = 500):
- ≈ 28.4 GPU h at the optimistic rates (A5(b) counted as ≈ 0.2–0.4 h: 40 episodes + one model load);
- ≈ 78.5 GPU h at the conservative rates.

O2 is optional. With O2 added, the totals are ≈ 40–100 GPU h.

**P4 throughput rule.** The 2 × 16-state pilot measures throughput first. If the approved n would exceed the budget, the human chooses n = 400 or a larger budget. That choice depends on throughput only, before any confirmatory outcome exists.

## 4. Review coverage now vs after the recommended Stage 3

| item | now (CP2) | after O0 + A5(b) + O1 + O3, if successful |
|---|---|---|
| M1 | partially resolved; population beyond the evaluation set unresolved — addressed by narrowing (no disjoint logs) | unchanged |
| M2 | **unresolved — addressed by narrowing** | resolved for AutoVLA; Impromptu still unresolved unless O2 runs |
| M3 | resolved (claim narrowed to A−) | unchanged |
| M4 | partially resolved; new-state part unresolved — addressed by narrowing | resolved at ±10 pp per task on SimplerEnv variant states; "two tasks only" and per-task ±5 pp remain |
| M5 | partially resolved (dependence); external validation unresolved — addressed by narrowing | unchanged |
| M6 | partially resolved (AutoVLA PASS; SpatialVLA pending A5; Impromptu / OpenVLA spec only; H32 / P1-B not gated) | more paths verified; Impromptu / OpenVLA still unverified unless O2 runs |
| M7 | resolved where unit-level data exist (DM / P1-B rows PROVISIONAL) | DM / P1-B rows verified (O0) |
| m1, m2, m4 | resolved (m4 as a descriptive re-analysis) | unchanged |

## 5. Text that could not be produced at CP2

- Final manuscript numbers for the P1-D SpatialVLA rows: on hold until A5.
- Any P3, P4 or P5 result: NOT_RUN.
- DM and P1-B rows as final: PROVISIONAL until O0.
- Exp 5 / exp 10 per-unit numbers: NOT_AVAILABLE (no raw output on disk).
