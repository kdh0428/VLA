# DISJOINT_REPLICATION_PROTOCOL — P5 (CP2; cannot be executed with the available data)

**Status: NOT_RUN — BLOCKED (data).** The protocol is fixed below so that it can be used unchanged if day- and vehicle-disjoint logs become available.

**Tag:** driving / AutoVLA / open-loop 5 s plan.

## 0. Run conditions (all three are required; condition 1 fails)

1. **Disjoint logs exist.** The P0-D inventory must find logs disjoint in day AND vehicle from development, P1-C and the checkpoint's training split.
   - Result (`P0/DISJOINT_LOG_INVENTORY*.csv`, rows `P0D.DISJ.*` in `numbers.json`): **0 logs**.
   - The non-training nuPlan pool on record is OpenScene test = 147 logs over 9 days.
   - Development uses 8 of those days and P1-C uses 8, and together they cover every test day.
   - 25 logs (7 drives) are disjoint from **training** in day and vehicle, but all of them share days and vehicles with development or P1-C.
   - navmini / warmup_test_e2e add no candidate. private_test_e2e has no public log list. nuScenes train is part of the training mix.
2. **AutoVLA P1 parity passed.** Satisfied: Stage-0 gate PASS for H7, H8 and H9, commit 47efb95.
3. **CP2 approval.** Not given.

## 1. Fixed design (to be used verbatim if condition 1 is ever met)

**Code, checkpoint and decoding.** Identical to the P1-verified paths:
- H7: RN;
- H8: RV;
- H32: DM, which must first be added to the gate (option O0);
- `p1_common.py`, T = 0.01, CRN seeds as in P1-C;
- AutoVLA checkpoint and environment as in `RUN_COMMANDS.md`.

**Contrasts.**
- Primary: RN = Recent-GT − Normal and RV = Reverse@emb − Full, defined exactly as in P1-C.
- Secondary: DM.
- Metric: `Y_severe`.

**Scene selection** (fixed before any outcome):
- run the natural pass on **all** eligible scenes of the disjoint logs, or on a preregistered random sample;
- define A− / A+ from the Normal natural evaluation;
- F = all eligible A− (π = 1); S = A+ sampled m per log segment (π = m/N);
- estimate the stratum effects and the Hajek P effect **in the same cohort**;
- never select failure scenes only.

**Inference.**
- Drive clusters and day–vehicle clusters, both reported.
- State the number of independent day–vehicle groups and the exact-p resolution (2/2^G).
- Report against a preregistered precision target. A non-significant result is not "no effect".

**Naming.**
- If training overlap cannot be ruled out, call it "held-out drive" and nothing stronger.
- Logs from another city or dataset are a **domain-shift** evaluation, not a replication.

## 2. Consequence while NOT_RUN

The manuscript keeps its sentence: "no external (day/vehicle-disjoint) validation".

- P1-C may be called "new-drive" (0 shared drives), but not day- or vehicle-disjoint.
- Whether the checkpoint was model-selected on navtest cannot be determined.

**What is not P5.** Option O4 in `../STAGE3_COST_PLAN.csv` re-runs the P1-C design on the 53 test logs not used by development or P1-C (17 drives; they share days and vehicles). It is **not** P5 and would not resolve M5:
- ≈ 4.9 GPU h;
- images are present locally for only 4 of the 53 logs.
