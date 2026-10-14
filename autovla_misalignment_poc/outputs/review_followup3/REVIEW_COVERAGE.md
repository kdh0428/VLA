# REVIEW_COVERAGE — CP2 draft (review_followup3, 2026-10-14)

The verdicts follow the prompt's review-item table.

- **Tier 3 items not run.** P3 and P4 were not run, and P5 is blocked by data. Every review item that depends on them is recorded as **"unresolved — addressed by narrowing"**, never "resolved".
- **"Required limitation sentence"** is the text that must remain in the manuscript even where the work succeeded. It comes from the prompt's "남는 한계" (remaining limitation) column; additions from this round are marked (+).
- **Row IDs** refer to `paper_quantitative_package/numbers.json`.

| item | work | verdict | what was done / evidence rows | required limitation sentence (manuscript) |
|---|---|---|---|---|
| **M1** failure-stratum conditional effects, population uncertainty, A− frequency | P0-A (done), P5 (NOT_RUN, no data) | **partially resolved** (population beyond the evaluation set: **unresolved — addressed by narrowing**) | Done: selection flow 2,747 → 52 A− (all A− selected, π = 1) / 156 A+; per-contrast denominators; RN / RV / DM by F, S, P with CIs; prevalence with drive / day–vehicle CIs; the P = F + S decomposition is verified (same frame, same estimand). Rows: `P0A.FLOW.*`, `P0A.P1C.*`, `P0A.DEV.*`, `P0A.PREV.*`, `P0A.DECOMP.*`. RN/RV: VERIFIED (manifest scope); DM: PROVISIONAL. | "The A− frequency is a frequency in the evaluation set, not in the deployment distribution." (+) "The −4.1/+3.6 pp effects are evaluation-population estimands over the 2,191 eligible scenes of the P1-C logs." |
| **M2** Reverse treatment is not matched across models | P3 (NOT_RUN) | **unresolved — addressed by narrowing** | Only the specification work exists: `P1/INTERVENTION_SPEC.md` §0 M-A1 and §5 document that AutoVLA Reverse-N = Normal-branch token injection, while Impromptu reverse = own-branch waypoint re-insertion (dynamic). The manuscript wording is narrowed (claim X1). Protocol draft: `P3/MATCHED_REVERSE_PROTOCOL.md`. | "Effect sizes cannot be compared across models." "Without the Impromptu run, the Impromptu Reverse remains unmatched." (+) "Interventions were matched in spirit rather than identical across models." |
| **M3** CM baseline too narrow | P2-A, P2-B (done) | **resolved** (the claim is narrowed to A−) | CM was shown to be a body-frame SE(2) repeat. Added CV0, CTRV-K3, CTRA-2 (Geo privileged) and a learned non-AR motion extrapolator (2 input variants × 3 seeds, drive-disjoint training). A−: exceeds all six comparators. A+ / all: not distinguished. Rows: `P2A.excess.*`, `P2B.excess.*`, `P2.verdict.*`, `P2B.coverage.*`. Machinery VERIFIED (P1A path incl. rollout saving). | "The extrapolator is not a non-AR version of the VLA." (+) "The excess over motion extrapolation holds only in failure-selected (A−) scenes." |
| **M4** task-level evidence | P0-B (done), P4 (NOT_RUN) | **partially resolved**; new-physical-state part **unresolved — addressed by narrowing** | Full task × config × contrast table (exp 35 and P1-D kept separate; primary and physical clustering; ±10 / ±5 pp TOST; equal-task means in a separate table; heterogeneity; weighting sensitivity). Rows: `P0B.*`. P1-D numbers are **PENDING_DECISION_A5**. Protocol and power plan for new SimplerEnv variant states: `P4/`. | "Only two tasks." (+) "move_near has 10 physical layouts; per-task success effects remain uncertain (±10 pp equivalence established for 3 of 8 P1-D task-level contrasts)." |
| **M5** small number of clusters, higher-level dependence | P0-D (done), P5 (NOT_RUN — no disjoint logs) | **partially resolved** (dependence sensitivity); external replication **unresolved — addressed by narrowing** | Drive / day–vehicle / vehicle / day / connected-component axes with exact p floors and leave-one-group-out (no sign flips); CGM auxiliary. Inventory: 0 day-and-vehicle-disjoint logs. Rows: `P0D-000`…`P0D-053`, `P0D.DISJ.*`. | "Without day/vehicle-disjoint logs there is no external validation." (+) "Under day × vehicle component clustering (6 groups) the exact test cannot reach Holm < .05 for the new-drive contrasts." |
| **M6** implementation specification, no-op | P1 (AutoVLA DONE-PASS; SpatialVLA PENDING_DECISION; Impromptu / OpenVLA spec only) | **partially resolved** | Spec table: `P1/INTERVENTION_SPEC.md`. AutoVLA gate PASS: H7 / H8 / H9 / P1A, c1–c5, greedy + T = 0.01, 0 failures (`P1.AV.*`). SpatialVLA restore audit: PASS on actions and state; strict C2 FAIL (bookkeeping hash) — **pending human decision (A5)** (`P1.SV.*`). Not gated: H32 (DM), the P1-B start-position script, Impromptu and OpenVLA. | "Past runs outside the verification manifest are not retro-validated." (+) "The Impromptu and OpenVLA intervention paths were not parity-verified." |
| **M7** missing numbers | P0-A/B/C (done) | **resolved** for rows with unit-level data; **NR kept** where none exists | +4.37 m fixed to one authoritative row (predeclared P1A-2, [+3.34, +5.17], Holm 7.3e-4) with the post-hoc V3 row kept separate. Document rounding errors listed. NR cells filled from unit-level data. Rows: `P0C.plus437.*`, `P0C.DOC.*`, `P0A.*`, `P0B.*`. DM / P1-B rows remain PROVISIONAL. | "Rows without unit-level data stay NR" (exp 5, exp 10 summaries). |
| **m1** weighting of eq. (8) | P0-C | **resolved** | Scene-weighted recomputation with joint numerator/denominator resampling. No near-zero denominators in any replicate. The old unit-weighted values are kept side by side. Rows: `P0C.R9.rec.*.NEW_scene_drive`, `P0C.R9.rec.*.OLD_exp9`. | — |
| **m2** Figure 2 n | P0-C | **resolved** | The same 365 units / 52 scenes / 13 drives / 10 day–vehicles for every window length (`P0C.R9.counts.*`). Not inferred from P1-B (12 drives). | — |
| **m4** OpenVLA comparison | P0-E | **resolved as a re-analysis; the claim is kept descriptive** | Initial-state clusters, task strata and equal-task weights. The episode-level intervals are kept side by side. The 92.6 % definition is fixed from code. Physical-unit (m, rad) translation; gripper binarisation recorded. Rows: `P0E.*`. The OpenVLA intervention path is not parity-verified. | "Precision is low with 10 initial states per task." |

## Summary

| verdict | items |
|---|---|
| resolved | M3, M7 (rows with unit-level data), m1, m2, m4 (as a re-analysis; descriptive) |
| partially resolved | M1, M4, M5, M6 |
| unresolved — addressed by narrowing | M2; plus the P4 part of M4, the P5 part of M1/M5, and the Impromptu/OpenVLA part of M6 |

## Stage-3 options and the items each would move

| option | items moved (if run and successful) | notes |
|---|---|---|
| O0 | M6 → H32 and P1-B paths verified (DM and early-window rows leave PROVISIONAL). M6 as a whole stays partially resolved. | – |
| O1 (P3 AutoVLA) | M2 → resolved **for AutoVLA only** | The Impromptu part stays unresolved unless O2 runs. |
| O1 + O2 | M2 → resolved within each model | Cross-model effect-size comparison stays impossible. |
| O3 (P4) | M4 new-state part → resolved at the approved precision (±10 pp per task proposed) | "Only two tasks" remains. ±5 pp per task remains unresolved under the proposed budget. |
| O4 (held-out-drive replication) | none | It adds new drives, but they are not day/vehicle-disjoint, so M5/M1 external validation stays unresolved. |
| P5 | none | Impossible with the available data. |
