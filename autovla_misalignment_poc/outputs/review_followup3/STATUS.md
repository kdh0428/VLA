# STATUS — review_followup3 at CP2 (2026-10-14)

**Scope.** Stage 0–2 are complete. No Stage-3 GPU work has been run. The CP2 files were written on the CPU only (`nice -n 19`), and no existing result file was modified. Nothing is committed yet; the "commit" column gives the commit that holds each result.

**Claim tags.** The manuscript is a **driving-VLA** study. Its main evidence is AutoVLA, open loop, 5 s plan (navtest). SpatialVLA (SimplerEnv Google Robot, closed loop) and OpenVLA (LIBERO-Spatial, closed loop) are complementary task-level evidence. They are not replications of the driving result.

**Status vocabulary** (also used in `paper_quantitative_package/numbers.json`):

| status | meaning |
|---|---|
| DONE | finished; output files exist |
| FAILED | ran and failed a pre-fixed rule |
| BLOCKED | cannot run (data/resource) |
| NOT_RUN | not run (by plan or awaiting approval) |
| PROVISIONAL | number exists, but its code path is not verified |
| PENDING_DECISION | needs a human decision before it can be used as final |

**AutoVLA number status after the gate.**
- The Stage-0 gate PASSED on all four paths (H7, H8, H9, P1A): 0 failures over the 21 manifest path-units × 2 decodings (`P1/NOOP_CACHE_AUDIT.md`, rows `P1.AV.verdict.*`).
- AutoVLA rows from P0/P2 on these paths therefore move from PROVISIONAL to **VERIFIED_MACHINERY_MANIFEST_SCOPE**.
- **Caveat that applies to every such row:** the gate verified the machinery on the 21 manifest units at HEAD 817d166 + harness 0ef9334 (RTX 5090, bf16). It does **not** retro-validate the original full runs outside the manifest.
- Rows on paths **outside** the manifest stay **PROVISIONAL**. These are H32 = exp 32 / P1-C direction-vs-magnitude (DM), and the P1-B start-position script.
- Natural-pass rows have status GATE_NOT_APPLICABLE: the selection flow, the prevalence and the inventory use no intervention machinery.

Row counts in `numbers.json` (2,117 rows):

| status | rows |
|---|---|
| VERIFIED_MACHINERY_MANIFEST_SCOPE | 1,528 |
| NOT_VERIFIED | 259 |
| AUDIT | 134 |
| GATE_NOT_APPLICABLE | 82 |
| PENDING_DECISION_A5 | 68 |
| PROVISIONAL | 46 |

## Task table

| task | status | reason | claim impact | commit |
|---|---|---|---|---|
| **Stage 0** AutoVLA no-op parity gate (P1-B, 4 paths) | **DONE — PASS** | H7 / H8 / H9 / P1A pass on all of c1–c5, greedy + T = 0.01, 0 failures. Harness/judge fixes A1–A3 changed no rule or tolerance. | AutoVLA RN / RV / Full-GT / windows / self-reference / C1 inputs → verified machinery (manifest scope). The withholding table entries for §5, Table 4, App. B, Fig. 2, §6.1–6.3 and Fig. 1 are lifted, but the manuscript must carry the manifest-scope caveat. | protocol 0ef9334; results 47efb95 |
| **P0-A** population, selection flow, conditional effects | **DONE** (post hoc) | – | RN P −4.1 pp [−6.4, −2.4], RV P +3.6 pp [+2.2, +5.5] = **evaluation-population** estimand (`P0A.P1C.RN.P`, `P0A.P1C.RV.P`). 52/2,747 = 1.89 % is the evaluation-set A− frequency, not a natural failure rate (`P0A.PREV.DEV.Aminus.2747.drive`). 60 % of RN P comes from the A+ stratum (`P0A.DECOMP.RN`). DM rows stay PROVISIONAL (H32). | e9907fc |
| **P0-B** SpatialVLA full task × config × contrast table | **DONE** (post hoc) → numbers **PENDING_DECISION (A5)** | Table complete: 32 task-level rows, equal-task means, heterogeneity, weighting. P1-D rows wait for the SpatialVLA P1 decision. Exp 35 rows are NOT_VERIFIED (no restore). | Pooled P1-D ±10 pp equivalence holds for all 4 contrasts (`P0B.P1D.*.pooled.*`). Per task, only 3 of 8 P1-D contrasts reach ±10 pp equivalence after Holm under primary clustering; ±5 pp: none. Exp 35 −6.2 pp stays deleted. | e9907fc |
| **P0-C** +4.37 m, recovery, denominators | **DONE** (post hoc) | – | Single authoritative +4.37 m row = predeclared P1A-2, drive bootstrap [+3.34, +5.17], Holm 7.3e-4 (`P0C.plus437.AUTHORITATIVE`); [3.32, 5.19] is the post-hoc V3 row. R9 = 13 drives for every window (`P0C.R9.counts.*`; Fig. 2 caption). Recovery 47/76/90/96 % (exp 9, unit-weighted) and 47/77/90/96 % (scene-weighted, `P0C.R9.rec.*.NEW_scene_drive`) are two estimands. P1-B rows stay PROVISIONAL (script not in the manifest). | e9907fc |
| **P0-D** dependence sensitivity + disjoint-log inventory | **DONE** (post hoc) | – | Effects keep their sign under every axis. For P1-C the day–vehicle component axis has G = 6, so the exact floor is 0.031 and the Holm floor 0.094: P1-C contrasts cannot reach Holm < .05 under full day × vehicle dependence (`P0D-*`). The inventory has 0 day-and-vehicle-disjoint logs (`P0D.DISJ.*`). | e9907fc |
| **P0-E** OpenVLA initial-state re-analysis | **DONE** (post hoc), numbers **NOT_VERIFIED** machinery | Unit records exist for all 800 episodes. The OpenVLA intervention path is spec-only (no parity run). | C−F 8-bin −3 pp, state-cluster CI [−12, +6] (`P0E.succ.C-F_d8`). §7.2 stays descriptive. | e9907fc |
| **P1 AutoVLA** (P1-A spec + P1-B gate) | **DONE — PASS** | See Stage 0. H32 and P1-B start-position are not in the manifest. | as Stage 0; DM (§5) and the early-window text (P1-B) stay PROVISIONAL | 0ef9334, 47efb95 |
| **P1 SpatialVLA** (P1-B/P1-C restore audit) | **PENDING_DECISION** — "PASS on actions/state; strict C2 FAIL (bookkeeping) — pending human decision" | Passed 20/20 each: C1, C5a, C5b, C3/P1-C restore; branch restore 60/60. The C2 null arm fails the strict pre-fixed rule (0/20) **only** on the executor fingerprint, which hashes the shadow-cf dict that null mode fills by design. Executed actions, every simulator component, exec/ctx tokens and TCP are bitwise equal (20/20; A5 post-hoc policy-state check). | Until the human decides, §7.3, §7.4, Table 5 and App. D keep their current (review_followup2) wording, and P1-D numbers are `PENDING_DECISION_A5`. **Not marked PASS.** | 47efb95 (AMENDMENT_P1_01 A4, A5) |
| **P1 Impromptu** | **NOT_RUN** (spec only) | Required only if P3 runs Impromptu. Exp 33 raw no-op parity is poor (`INTERVENTION_SPEC` M-I1). | §7.1 relies on unverified machinery: no wording stronger than "token-level effect observed; machinery not parity-verified". | – (spec in 0ef9334) |
| **P1 OpenVLA** | **NOT_RUN** (spec only; no new OpenVLA run) | – | §7.2 descriptive only | – |
| **P2-A** kinematic motion baselines | **DONE** (post hoc, spec locked before new-baseline results) | CM is a body-frame SE(2) repeat (already constant turn-rate). New controls: CV0, CTRV-K3, CTRA-2; Geo is privileged. | A− only: model > every kinematic control (`P2A.excess.G.A-.end.*`). A+ and all: not distinguished from CM / CTRA-2. §6.2 is narrowed. Status: verified machinery (P1A path incl. rollout-saving path). | 5b158f2 |
| **P2-B** learned non-AR motion extrapolator | **DONE** (protocol locked before training) | 3 training seeds; train/validation drives disjoint from every evaluation set. | A−: model > E_prefix and E_prefix_ego; the out-of-range sensitivity keeps the sign. A+ / all: not exceeded (`P2.verdict.*`). | 5b158f2 |
| **P3** matched Reverse (AutoVLA; Impromptu add-on) | **NOT_RUN** (Tier 3; awaiting CP2 approval) | Protocol draft: `P3/MATCHED_REVERSE_PROTOCOL.md`. Results: `P3/MATCHED_REVERSE_RESULTS.md` = NOT_RUN. | M2 is **unresolved — addressed by narrowing**: cross-model Reverse is not a matched treatment. | – |
| **P4** SpatialVLA new physical states | **NOT_RUN** (Tier 3; awaiting CP2 approval) | Protocol draft + `POWER_PLAN.md`. ±5 pp per task needs ≥ 1,500 states/task (≈ 75–209 GPU h), so a narrowed ±10 pp target is proposed for approval. | The new-state part of M4 is **unresolved — addressed by narrowing** | – |
| **P5** day/vehicle-disjoint replication | **NOT_RUN — BLOCKED (data)** | 0 logs are day- and vehicle-disjoint from development, P1-C and training (P0-D inventory, `P0D.DISJ.*`). | The external-validation part of M5 is **unresolved — addressed by narrowing**. The "no external (day/vehicle-disjoint) validation" sentence stays. | – (inventory in e9907fc) |
| CP2 package (this round) | DONE (uncommitted) | `numbers.json` merged by script; claim matrix, coverage, Stage-3 protocols, power and cost plans | – | not committed (as instructed) |

## Decisions needed at CP2

1. **A5 (SpatialVLA C2 null arm).** Choose (a) or (b):
   - (a) Accept the A5 policy-state criterion as the C2 pass. It is post hoc and must be labelled so. SpatialVLA P1 would then become "PASS (A5, post hoc)".
   - (b) Keep the strict rule. The fix is to hash the policy state and `cf` separately and re-run the null arm on the 20 units (≈ 40 episodes = U2 + U0 on 20 units, ≈ 7–20 min at the measured 2.0–5.5 episodes/min, + model load; < 0.5 GPU h). Until the re-run, the status stays "PASS on actions/state; strict C2 FAIL (bookkeeping)".

   Either way, the 20-unit audit does not retro-validate exp 34/35.
2. **Stage-3 selection** (costs in `CP2_REPORT.md` and `STAGE3_COST_PLAN.csv`): O0 gate extension (H32, P1-B); O1 P3 AutoVLA; O2 P3 Impromptu; O3 P4 (precision target); O4 optional held-out-drive replication (not P5).
3. **P4 precision target.** Approve or reject narrowing from ±5 pp to ±10 pp per task (Holm m = 8) **before** any P4 data exist (`P4/POWER_PLAN.md`).
