# MATCHED_REVERSE_PROTOCOL — P3 (DRAFT for CP2; not frozen, not run)

> **Final status (2026-10-15): NOT RUN by CP2 decision** (`../CP2_DECISION.md`, 444299f). This draft is kept as the protocol to use if P3 is ever approved. M2 stays "unresolved — addressed by narrowing".

**Status: DRAFT.** This file is submitted at CP2 for approval.
- Nothing has been run. No P3 code exists yet.
- Freeze steps: after approval, (1) fix this file and commit it before any P3 output exists, then (2) write the code, run its no-op gate (§7), and freeze the source bank (§3).
- Any change after P3 data exist is a CP3 amendment.

**Scope.**
- Tag: driving / **AutoVLA** / open-loop 5 s plan (navtest).
- Impromptu is an optional add-on (§11).
- The design identifies effects **within one model**. Raw effect differences between models are never interpreted as architecture effects.

**What it answers (review M2).** Today the AutoVLA "Reverse-N" is a different treatment from the Impromptu "reverse" (`P1/INTERVENTION_SPEC.md` M-A1, §5):
- AutoVLA Reverse-N: GT history plus the **separate Normal branch's** token, injected repeatedly.
- Impromptu reverse: GT history plus the row's **own** waypoint, chosen dynamically.

P3 separates three factors inside AutoVLA with a frozen, matched factorial:
- older-prefix source;
- latest-block source;
- source provenance.

## 1. Definitions (notation of INTERVENTION_SPEC §1.2)

**Unit and steps.**
- Unit = (scene, forced token f at t*), as in exps 7–9 / P1-C.
- Generation steps k = t*+1 … 9.
- The context at step k is the set of positions t*+1 … k−1.
- Executed output = prefix + [f] + the row's own sampled tokens. Context edits never change already-emitted tokens.

**Anchor a (single, fixed by rule).**
- a = t*+3. Older prefix = position t*+1; latest block = position t*+2 = a−1.
- Sensitivity anchor: a = t*+4 (older prefix = t*+1 … t*+2).
- Units with a > 9 are excluded by rule before any outcome is seen: t* = 7 for the primary anchor, t* ≥ 6 for the sensitivity anchor. Their counts are reported per stratum.

**Factor O, older prefix (positions t*+1 … a−2).**

| level | values |
|---|---|
| GT | the GT tokens (the existing GT→token projection) |
| NAT | the **matched natural rollout**: the frozen Normal-branch tokens at the same positions, from the source bank, same length; no deletion |

**Factor L, latest block (position a−1).**

| level | value |
|---|---|
| REF | the GT token gt[a−1] |
| NSRC | the frozen **Normal-source** token c^N_{a−1} from the bank |
| CSRC | the frozen **corrected-branch-source** token c^F_{a−1} from the bank. The corrected branch is the Full-GT-history row (GT at every context position, every step); c^F_{a−1} is its own sampled token at step a−1. |

**Cells, one-shot single-anchor primary block (6 cells).**
- At steps k < a, every cell's context is its own continuation. Under common random numbers these tokens equal the Normal tokens, which the bank checks.
- At step a, positions t*+1 … a−1 are overwritten with (O, L).
- For k > a, the overwritten positions **persist** at the same values (fixed absolute positions, as in exp 9 windows), and positions ≥ a hold the row's own outputs.

| cell | reads as |
|---|---|
| (GT, REF) | one-shot Full |
| (GT, NSRC) | **matched Reverse** (one-shot, frozen Normal source) |
| (GT, CSRC) | GT history + corrected-branch's own token (the AutoVLA analogue of the Impromptu own-branch provenance) |
| (NAT, NSRC) | identical in value to Normal. Built from the bank, so it doubles as a value-identity leak check (§6). |
| (NAT, REF) | one-shot Recent-GT |
| (NAT, CSRC) | natural older + corrected-branch token |

**Other rows in the same batch:**
- untouched **Normal**;
- the existing repeated **Full-GT-history** row, as the reference for the bank;
- no-op/self-override rows of §7.

**Secondary block: repeated schedule.**
- The same 2 × 3 matrix applied at **every** step k ≥ t*+2, sliding: older = positions t*+1 … k−2, latest = k−1.
- Sources come from the frozen bank. This reproduces the existing RN/RV schedule with frozen sources.
- Reported separately and never pooled with the primary block.

**Fixed inside the model.**
- Observation (video), language prompt, stub, t* and f.
- Non-target positions.
- Sampling: T = 0.01 with the experiments' CRN, re-seeded per row with the same key across cells; greedy is reported separately.
- Horizon: 10 poses / 5 s.
- Representation: token ids. A layer-0 embedding patch ≡ token substitution, verified by C4 for H8.

**Dynamic own-branch feedback.**
- A row that re-inserts its **own** previous token at every step is not a fixed factorial value. It is run only as a separate, clearly labelled feedback block (exploratory) and is not part of any family.
- Deleting the older prefix (shortening the sequence) changes positions and length. It is a separate treatment and is **not** run.

## 2. Population, manifest and strata (chosen without outcomes)

**Primary: P1-C analysed set.**
- 64 eval logs / 24 drives; 353 scenes / 2,436 units.
- Frame and selection were preregistered (b212637, amendments b216920 and 9b3d4af) and fixed before any intervention outcome.
- Two parts:
  - **Failure stratum F:** all eligible natural A− (105 scenes / 748 units / 18 drives, π = 1).
  - **S:** the A+ sample, 4 per log segment, with Hajek weights. F ∪ S = P (the evaluation-population estimand of P0-A).
- These scenes are **re-used with new inference**: the P3 rows are new decodes. The stored P1-C rollouts are not reused as P3 arms.

**Secondary: dev mechanism set.**
- 208 scenes / 1,408 units, 13 A− drives.
- Labelled **development**: these scenes were used to design the original experiments.

**Fixed exclusions:**
- a > 9 (§1);
- harness errors, tracked per cell in the common cohort. A unit enters a contrast only if both of its cells are valid. Missing ≠ 0.

## 3. Frozen source bank (built before any factorial cell)

**Pass A, per unit.** Rows Normal and Full-GT-history are run with the experiment CRN. For every step the bank stores:
- the sampled token ids;
- the decoded motion (disp, dyaw) of each token;
- the context-prefix hash.

**Freezing.**
- Bank file: `P3/bank/source_bank.parquet` + sha256 + `bank_manifest.json` (unit → scene → log → drive → day–vehicle).
- After the bank is frozen, no source value can be re-selected from a factorial cell's outcome.

**Checks.**
- The Pass-A Normal row equals the untouched Normal row of Pass B (same harness, same batch shape) at every step.
- Full-GT-history equals the existing H7/H9 `gt_history` definition on the Stage-0 manifest units.

## 4. Outcomes and contrasts

**Metric.**
- Primary: `Y_severe` = (not A+) ∧ FDE5 > 3 m on the executed 10-pose plan. Same `label_A`, same 5 s horizon as P1-C.
- Secondary: suffix-token change vs Normal; endpoint displacement ‖X9 − X9^N‖ (m).

**Estimator.**
- Scene-weighted paired difference: unit differences are averaged within the scene first, then scenes are weighted equally (F) or by Hajek weights (P).
- These are the P1-C estimators, imported unchanged.

**Primary contrasts, F stratum, one-shot block (family P3-F, Holm m = 4, α = 0.05 two-sided).**

| id | contrast | question |
|---|---|---|
| C1 | (GT, NSRC) − (GT, REF) | latest-block source effect inside a GT older prefix (matched Reverse vs Full) |
| C2 | (NAT, REF) − (NAT, NSRC) | latest-block source effect inside a natural older prefix (one-shot Recent-GT vs Normal) |
| INT | [(GT, NSRC) − (GT, REF)] − [(NAT, NSRC) − (NAT, REF)] | does the older-prefix source change the latest-block effect? |
| C4 | (GT, CSRC) − (GT, NSRC) | provenance/value: corrected-branch source vs Normal source |

C4 is **not** called a pure provenance effect. With it, report:
- the share of units where c^F_{a−1} = c^N_{a−1};
- the distribution of |disp(c^F) − disp(c^N)| and |dyaw(c^F) − dyaw(c^N)|;
- C4 restricted to units whose source values differ.

**Secondary contrasts (each its own family, Holm within).**
- P3-P: the same 4 contrasts on the P estimand (Hajek).
- P3-S: the S stratum.
- P3-REP: the repeated-schedule block.
- P3-a4: the sensitivity anchor.
- P3-DEV: the dev set.
- Descriptive, all cells: the older-prefix main effect, i.e. GT vs NAT averaged over L.

## 5. Inference and clustering

**Primary.**
- Drive clusters: bootstrap B = 10,000 (seed 20261015, percentile) and exact sign-flip on drive totals.
- F stratum: 18 drives, exact floor 7.6e-6. P: 24 drives.

**Sensitivity.**
- Day–vehicle (10/16 groups), vehicle, day, and connected day–vehicle components. P1-C components: G = 6, exact floor 0.031, Holm floor 0.094 (P0-D).
- These are reported as inference limits, not used to replace the primary.

**Expected precision** (planning; derived from the repeated-schedule P1-C F effects):
- Bootstrap CI half-widths are ≈ 4.4 pp for RN F and RV F (`P0A.P1C.RN.F`, `P0A.P1C.RV.F`), so the SE of one paired contrast ≈ 2.2–2.5 pp.
- MDE at 80 % power (Holm worst case α/4) ≈ 8–9 pp for C1, C2 and C4.
- MDE ≈ 12 pp for INT (SE ×√2).
- One-shot single-anchor effects are expected to be **smaller** than the repeated-schedule effects. A non-significant one-shot contrast is reported with its CI and **not** called absence of an effect (no equivalence margin is preregistered).
- The sample is fixed by the frame: there is no adding of units and no early stopping on significance.

## 6. Leak checks (must pass; failure → stop and CP1-style report)

1. **Value identity.**
   - (NAT, NSRC) must equal the untouched Normal row: bitwise at B = 1; within T_equiv in batch, as in Stage-0 c2_token.
   - Wherever two cells have identical token values at every edited position (c^F = c^N, or c^N = gt), their outputs must be identical.
   - Labels/provenance that differ while the values are equal must not change the outcome. Any difference is an implementation leak.
2. **Bank immutability.** The bank sha256 is asserted before every pass.
3. **Pre-anchor identity.** Before step a, every cell's tokens equal the Normal tokens (CRN).

## 7. No-op / cache gate for the new P3 path (before the factorial runs)

The P1-verified machinery is reused:
- `p1_common.py` prefix cache + span forward;
- the fixed-absolute-position persistent override of `temporal_feedback_window.py:189-197` (H9) for the one-shot block;
- the sliding override of `action_history_causal.py:210-212` (H7) for the repeated block;
- the Stage-0 harness `scripts/review_followup3/p1/p1b_harness.py` / `p1b_judge.py`, extended with a path `P3`.

**Gate.**
- Controls c1–c5 of `PROTOCOL_P1.md` on the 9 Stage-0 manifest scenes (`stage0_manifest.json`).
- Greedy and t001 decodings.
- The same tolerance file: `pilot_c5_20261014/tolerances.json`, T_exact = 0, T_equiv = 0.75.
- C4 (full recompute of the final edited sequence) is mandatory for every cell type.

**On failure:** stop and report at CP1. No large run starts before a PASS.

## 8. Saved per unit × cell × step

- Source/target token ids and positions.
- Decoded source/target motion and their difference.
- Context-prefix hash; input / position / logits hashes.
- Bank ids + bank sha.
- No-op parity result.
- CRN key.
- Sampled tokens, decoded 10-pose plan, `label_A`, FDE5, `Y_severe`.
- Unit → scene → log → drive → day–vehicle mapping.

## 9. Cost (from `STAGE3_COST_PLAN.csv`, option O1; measured rates in `STAGE3_COST_RATES.csv`)

| block | units | decoded plans (rows × units) | GPU h (optimistic – conservative) |
|---|---|---|---|
| P3 no-op gate (9 scenes, 2 decodings) + c5 run2 | 18 path-unit-decodings | – | 0.15 – 0.37 |
| Source bank, P1-C / dev | 2,436 / 1,408 | 2 × units = 4,872 / 2,816 | 0.46 – 1.12 / 0.27 – 0.64 |
| One-shot factorial, P1-C / dev | 2,436 / 1,408 | ≈ 10 × units = 24,360 / 14,080 | 0.34 – 1.12 / 0.20 – 0.64 |
| Repeated-schedule block, P1-C / dev | 2,436 / 1,408 | ≈ 10 × units = 24,360 / 14,080 | 0.34 – 1.12 / 0.20 – 0.64 |
| Model loads (8 × 3 min) | – | – | 0.40 |
| **Total O1** | – | ≈ 84,600 decoded plans | **≈ 2.3 – 6.0 GPU h** on the RTX 5090 (plus the wait for ≥ 14 GB free; an external process holds 13–26 GB) |

How the range was derived:
- Optimistic: per-unit cost at the measured P1-A B = 10 rate (120 units/min) and the H9 B = 17 rate (88 units/min, P1-C).
- Conservative: the H7 B = 1 rate (36 units/min, P1-C).
- Disk: < 1 GB expected (the P1-C H8 records with 93 rows were 233 MB).
- Implementation (not GPU): about 1 day for code, tests, and the bank.

## 10. What P3 will and will not show

- **Will:** within AutoVLA, whether the latest-block source effect depends on the older-prefix source (INT), and how much of "Reverse" is source value vs provenance (C4 + value differences). All under a frozen, matched source bank and one-shot timing.
- **Will not:**
  - make the AutoVLA and Impromptu effect sizes comparable;
  - retro-validate the earlier repeated-schedule runs;
  - produce external validation.

## 11. Optional add-on: Impromptu VLA 3B (Tier 3 extra)

**Same factorial, in waypoint text.**
- Older prefix: GT waypoints vs frozen natural waypoints.
- Latest block: GT / frozen Normal-branch / frozen corrected-branch waypoint.
- Text rounding to 0.01 m is applied identically in every cell.
- A tokenization round-trip check is required.

**Prerequisite.** An Impromptu P1 no-op harness (PROTOCOL_P1 §9.4):
- a self-override row (own waypoint written back);
- the token-id round-trip check;
- B = 1 vs batch.

This matters because exp 33 raw no-op parity is poor (M-I1: natural_rep = natural in 237/2,728).

**Scenes:** the 300 exp 33 `exp_cde` scenes (2,700 units).

**Cost (O2):** ≈ 11 – 22 GPU h. The exp 33 rate is approximate (file mtimes); the harness/gate estimate is ≈ 1–2 h.

**If not run:** the Impromptu part of M2 stays "unresolved — addressed by narrowing".

## 12. Results file

`MATCHED_REVERSE_RESULTS.md` / `.csv` are written now with status **NOT_RUN**. Only an approved run replaces them.
