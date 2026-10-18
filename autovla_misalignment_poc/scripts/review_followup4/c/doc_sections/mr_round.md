---

## 2026-10-18 revision (review_followup4)

**Scope.** Driving / AutoVLA / open-loop 5 s plans is the main evidence. SpatialVLA (closed loop) is complementary task-level evidence. Impromptu VLA and OpenVLA are descriptive and unverified (A3; B3 not run) and are not counted as replications. No sentence speaks about VLAs in general. Sources: `autovla_misalignment_poc/outputs/review_followup4/CLAIM_CHANGES.md` (verdicts), `MANUSCRIPT_REVISION.md` in the same directory (all 64 A4 line-level replacements and every A1/A2/A3/B1/B2 change), `numbers.json` rows `RF4.*`, `tables.tex` Tables 12–14. **This section supersedes conflicting wording above.**

### A. KEEP (replaces K1, K2, K4, K5, K8, K10 above; K3, K6, K7, K9, K11, K12 stay as written)

**K1'** — driving / AutoVLA / open loop [`P0A.DEV.RN.Am`, `P0A.DEV.RV.Am`, `P0A.P1C.RN.F`, `P0A.P1C.RV.F`]
> "In AutoVLA (open-loop 5 s plans on NAVSIM navtest), repeatedly replacing the most recent action-context token with ground truth at every generation step reduced amplification in the failure stratum by 39.8 pp [29.9, 49.0] (52 scenes, 13 drives), and injecting the separate Normal branch's previous token into a ground-truth context restored it by 28.6 pp [20.9, 36.1]. In a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared with development, days and vehicles partly shared) the effects were −34.2 pp [−38.3, −29.5] and +26.8 pp [+21.9, +30.7]."

**K1b (new, B2)** — driving / AutoVLA / open loop [`RF4.B2.p1c.a1.severe.*`]
> "In a preregistered matched-prefix diagnostic with a single retained edit, placing the frozen Normal-branch token instead of ground truth in the most recent context position raised severe outcomes in the P1-C failure stratum by 25.3 pp [20.5, 29.3] behind a ground-truth older prefix and by 25.5 pp [21.6, 29.0] behind the model's own natural older prefix (100 scenes, 18 drives); the interaction was −0.2 pp [−3.8, +3.1], so in this one-shot schedule the effect tracks the value of the most recent context token rather than the presence of an older ground-truth prefix. Because the reference cells contain ground truth, this is an oracle-conditioned diagnostic, not an online correction."

**K2'** — driving / AutoVLA / open loop [`P0A.P1C.RN.P`, `P0A.P1C.RV.P`, `P0A.DECOMP.*`, `RF4.B2.p1c.a1.severe.C_*.all.P.drive`]
> "On the evaluation frame of the new-drive logs (2,191 eligible scenes, estimated from 353 sampled scenes with design weights; 24 drives), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5]; about 60% of the Recent-GT effect came from non-failure scenes. In the matched-prefix eligible domain (t* ≤ 4; 312 sampled scenes) the latest-token effect was small (+4.0 pp [+2.7, +5.7] behind the natural prefix; +3.3 pp [+1.0, +5.6] behind a ground-truth prefix, not significant after Holm). The failure category occurred in 1.9% of development evaluation scenes, an evaluation-set frequency rather than a deployment rate."

**K4'** — driving / AutoVLA / open loop [`P2A.excess.G.*.end.*`, `P2B.excess.G.*.end.*.primary`, `P2.verdict.*`, `RF4.B1.stab.*`]
> "In failure-selected scenes, this response exceeded four kinematic extrapolations — including the original constant-motion baseline, which repeats the perturbed token's body-frame motion including its yaw — and a small learned non-autoregressive motion extrapolator (+2.35 m [1.27, 3.19] over constant motion; +1.97 m [1.06, 2.87] over the learned extrapolator); in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested)."
> (B1, Methods / caption) "The Recent-GT, Full-GT and Reverse-N contrasts were unchanged in direction and nearly unchanged in size under batch-shape, attention-backend and fp32-arithmetic changes (batch shape alone changed 9% of free-run suffixes); near-zero differences between ground-truth-free self-reference conditions could not be distinguished from numerical implementation variation."

**K5'** — driving / AutoVLA / open loop [`P0C.R9.rec.win3.NEW_scene_drive`, `RF4.A2.*`]
> "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect (scene-weighted 90.1%, 95% CI 81–99; 365 units, 52 scenes, 13 drives; 2.96 corrected positions per unit, 1.90 of them differing from the model's own token); the number of corrected positions and the number of steps generated after the last added ground-truth position change together."

**K8'** — manipulation / SpatialVLA-4B / closed loop [`RF4.A1.P0B.P1D.*`]
> "In SpatialVLA closed-loop manipulation (SimplerEnv Google Robot; two tasks), context perturbations changed generated actions and the physical end-effector trajectory, while their effect on task success, averaged over the two tasks, was bounded within ±10 pp in the tested configurations (post-hoc corrected Rao–Scott Newcombe intervals, initial-state clusters, Holm); per task, ±10 pp equivalence held for 3 of 8 contrasts and ±5 pp for none, with task-level 95% CIs up to 18–25 pp wide, so task-specific effects remain uncertain."

**K10'** — manipulation / OpenVLA-7B / LIBERO-Spatial [`RF4.A1.P0E.succ.*`, `P0E.phaseC.d0_|8|_feedback`]
> "In OpenVLA-7B on LIBERO-Spatial (100 initial states; intervention path not parity-verified; descriptive), correcting the context after an x-token perturbation produced no detected change in task success (−3 pp, Newcombe 95% CI [−11.9, +5.9]; equivalence not established), while re-running the unperturbed policy alone changed 20% of outcomes; within a single step, an 8-bin x-token shift (about 2.6 mm commanded) changed at least one later component by at least the effective injected number of bins in 92.6% of 4,741 units (initial-state-cluster CI [90.8, 94.0]; translation components alone 72%)."

**K11 addition** (append to K11): "Batch shape alone changed 9% of AutoVLA free-run suffixes; all paired contrasts are therefore computed within one harness and batch, and the B2 cells run at batch size 1 on one shared prefix."

**K13 (new, Impromptu)** — driving / Impromptu VLA 3B / open loop
> "In Impromptu VLA 3B (open loop, descriptive; implementation parity not established), ground-truth context reduced and a reversed last context increased trajectory amplification by margins well above the 5–6% scene-level flip rate caused by batch and decoding-path numerics alone."

### B. DELETE or REPLACE (in addition to R1–R17)

| # | delete / replace | why | replace with |
|---|---|---|---|
| R18 | "Correcting three consecutive context steps recovered about 90%" / "a short (transient) correction has a lasting effect" / "after release the model conditions on its own tokens only" / "post-release horizon" / "re-divergence after release" | A2: GT positions are retained in every later context; no release exists | K5'; "steps after the last added ground-truth position"; "re-divergence while the corrected ground-truth positions remain in the context (exp 10, summary only)" |
| R19 | "Impromptu VLA strongly replicates …" / "the three models replicate strongly" / "no-op parity 23/300, 237/2,728" | A3: numerical floor 5–6%, path not gated; the pairs were not same-condition repeats | K13 |
| R20 | any OpenVLA per-task success CI, "no difference in task k", "C − N 95% CI excludes 0" | A1: sign-flip inversion degenerate; per-task n = 10 | K10'; per-task rows descriptive |
| R21 | "exp34 perp_left coke C − F equivalent within ±10 pp (Holm)"; "task-level CIs span up to 18–29 pp (r4e1; r1e4 6.9–21.6 pp)"; "P1-D r1e4 C − F move_near within ±5 pp" | A1 corrected intervals | "unadjusted only"; "up to 18–25 pp (r4e1; r1e4 13–20 pp)"; (delete) |
| R22 | "the Reverse effect requires a ground-truth history" / "pure provenance effect" / "the matched-prefix result is an external replication or an online correction" | B2: INT ≈ 0; value differences; existing drives; oracle cells | K1b |
| R23 | "replicated on 24 new drives" for the failure-stratum effects; "population effect −4.1 / +3.6 pp"; "across all scenes" for the 208 selected scenes | A4 denominators | K1', K2', K4' |
| R24 | "92.6% of later dimensions amplified" / "93% of the remaining dimensions move further" | A4: at least one component; bins; commanded | K10' |
| R25 | P1-A self-reference conclusions that rest on the sign of a near-zero FDE difference | B1: material under numeric settings | (delete) or "not distinguishable from numerical implementation variation" |
| R26 | "OpenVLA path verified" / OpenVLA as core evidence | B3 not run | K10' (descriptive) |
| R27 | "π₀-FAST vs π₀ shows an architecture effect"; π₀-FAST cross-step context effect "0" | not run; NA by design | (delete) |

### C. Limitation sentences added this round
1. "Interval estimates for paired success rates were corrected post hoc (Newcombe / Rao–Scott Newcombe); the original sign-flip-inversion intervals are kept in the supplement; difference-from-zero tests are unchanged."
2. "The correction-window experiments retain the corrected ground-truth positions in all later contexts; no experiment removes them."
3. "The matched-prefix factorial uses one retained edit at a single anchor on existing drives; it is an oracle-conditioned diagnostic and does not test release or repeated schedules."
4. "Numerical implementation choices (batch shape, attention backend, precision) change 9–26% of free-run suffixes without changing the main contrasts."
5. "The Impromptu VLA and OpenVLA intervention paths were not parity-verified; their results are descriptive."
