# MANUSCRIPT_REVISION — paper-ready text after the 2026-10-13 review (review_followup2)

**Why this file exists.** The repository contains no manuscript `.tex` (checked with `rg --files -g '*.tex' /root/VLA`: the only `.tex` files are `paper_quantitative_package/tables.tex`, `paper_cross_model_table.tex`, `paper_detection_table.tex`, `paper_navhard_table.tex`). The text below is a drop-in replacement for the abstract, contribution list and conclusion, plus a list of sentences to remove from any existing draft.

**Rules followed.** Every sentence uses only claims marked keep / shrink in `autovla_misalignment_poc/outputs/review_followup2/CLAIM_EVIDENCE_MATRIX.md`, with its allowed wording. Numbers come from `PAPER_NUMBERS.md` (2026-10-13). Independent units: drives for AutoVLA/Impromptu, initial-state clusters for SpatialVLA. p values are drive-level exact sign-flip with their floors; no unit-level p is used. Re-analyses of existing data are labelled post hoc in the Methods; preregistered judgments (P1-A–D, Exps. 33–35) are reported as registered.

---

## Abstract (revised)

Autoregressive vision-language-action (VLA) models generate each action conditioned on the actions they have just produced. We ask whether this self-generated action context causally shapes later actions and failures. In AutoVLA on NAVSIM, repeatedly replacing the most recent action context with ground truth at every generation step reduced trajectory amplification by 40 percentage points (pp) (95% CI 30–49; 13 independent drives; exact drive-level sign-flip p = 2.4 × 10⁻⁴, the minimum attainable), and re-inserting the model's own previous action restored it (+29 pp). Both effects replicated in a preregistered test on 24 new drives that share no drive with the development set. Without any ground-truth information, a single perturbed context token changed subsequent local actions and produced endpoint deviations 4.4 m beyond pure coordinate accumulation; this response exceeded constant-motion extrapolation only in failure-selected scenes. Correcting three consecutive context steps recovered about 90% of the full-history correction effect (95% CI 81–99%), and we found no evidence that corrections at the first deviating steps are more effective than equally long later corrections. Analogous context interventions, matched in spirit but not identical, changed subsequent action tokens in Impromptu VLA, OpenVLA and SpatialVLA. In SpatialVLA closed-loop manipulation, context perturbations changed generated actions and the physical end-effector trajectory, and longer replanning intervals made these deviations more persistent; their effect on task success, averaged over two tasks, was bounded within ±10 pp in the tested configurations, while task-specific effects remain uncertain. Deployable mitigation came from selecting among the model's own candidates and from a current-frame safety filter rather than from conditioning on external references.

*(≈ 250 words. If space is tight, drop the mitigation sentence first.)*

## Contributions (revised)

1. **Causal evidence that self-generated action context drives amplification in a driving VLA.** Repeated last-position correction lowers amplification by 39.8 pp [−49.0, −29.9] and Normal-branch re-insertion restores it by 28.6 pp [+20.9, +36.1] (13 drives, p = 2.4 × 10⁻⁴, Holm 7.3 × 10⁻⁴; leave-one-drive-out: no sign change). Repeated last-position correction recovers 93% (92.9% [89.4, 99.1]) of the full-history correction effect. A preregistered replication on 24 new drives (day/vehicle partly shared) reproduces both contrasts (failure-conditional −34.2 / +26.8 pp; population −4.1 / +3.6 pp).
2. **A ground-truth-free control (AutoVLA).** Perturbing only a context token, with the model's own unperturbed rollout as reference, changes later local actions (+0.57 m/segment) and moves the 5 s endpoint 4.37 m [3.32, 5.19] beyond pure coordinate accumulation (post-hoc excess metric; the predeclared P1A-2 row is 4.37 m [3.34, 5.17], Holm 7.3 × 10⁻⁴ — see the 2026-10-15 section). Against constant-motion extrapolation the excess is present only in failure-selected scenes (+2.35 m [1.27, 3.19]); across all scenes it is not distinguishable from extrapolation (+0.20 m, p = 0.51).
3. **Correction length rather than an early critical window.** Three consecutive corrected context steps recover about 90% (81–99%) of the full-history effect (1/2/3/4 steps: 47/77/90/96%); in a preregistered start-position test, early minus late corrections of equal length were +10.7, +4.3 and +0.5 pp for 1–3 steps, none significant.
4. **Cross-model and closed-loop scope.** The token-level context effect recurs in three further temporally or within-step autoregressive VLAs under model-specific interventions whose effect sizes we do not compare across models. In SpatialVLA, a preregistered dose-controlled experiment (4,800 episodes, state restoration verified) shows that longer replanning increases the persistence of context-induced trajectory deviations at a fixed context-perturbation budget (+214 cm·step [171, 255]; a composite configuration effect), that temporal ensembling does not absorb context effects once executed dose is matched, and that the pooled two-task success direct effects lie within ±10 pp (initial-state clusters, Holm), with task-level effects uncertain.
5. **Statistical and data audit.** We report all mechanism contrasts at the level of independent drives (AutoVLA, Impromptu) or physical initial states (SpatialVLA: move_near has only 10 physical layouts), disclose log-level but not drive-level separation of earlier held-out sets, and document protocol amendments and failed runs.

*(If the paper keeps the mitigation results: "6. Candidate selection by decoding confidence improves preregistered held-out PDMS by +0.0095 [0.0046, 0.0148] on held-out logs (not drive-disjoint); on the reactive navhard benchmark most of the gain comes from a current-frame safety filter (+0.105 of +0.118 EPDMS).")*

## Conclusion (revised paragraph)

Self-generated action context is a causal input to subsequent action generation in autoregressive VLAs. In AutoVLA, repeatedly correcting the most recent context position removes most of the amplification that follows a small first deviation, re-inserting the model's own previous action restores it, and both effects replicate on new drives; a ground-truth-free perturbation shows that the context alone changes later local actions beyond coordinate accumulation. Two stronger readings did not survive scrutiny: the response exceeds constant-motion extrapolation only in failure-selected scenes, and the first steps after a deviation are not demonstrably more important than later steps of equal length—what matters is how many consecutive context steps are corrected. Direction errors were more harmful than magnitude errors in the development drives, but this was inconclusive on new drives. Across models, interventions were matched in spirit rather than identical, so we compare directions, not magnitudes. In closed-loop manipulation with short chunks and frequent replanning, context perturbations change generated actions and the physical trajectory, but their effect on success, averaged over two tasks, was bounded within ±10 pp in the tested configurations; task-specific and smaller effects remain open, and the limited number of distinct physical initial states in the benchmark constrains how precisely they can be measured.

## Limitations paragraph (suggested additions)

- Mechanism experiments in AutoVLA were exploratory (not preregistered); the independent sample is 13 drives (10 day-vehicles), which bounds the smallest attainable exact p at 2.4 × 10⁻⁴. The one preregistered new-data replication (24 drives) shares day-vehicles with the development set.
- Recent-GT is a repeated intervention; the 93% figure is relative to the full GT-history effect. The ground-truth-free control exists only for AutoVLA, and its perturbation position and strata inherit the GT-defined failure label.
- Correction length and post-release horizon change together; absolute pose and post-release horizon cannot both be equalized within the 10-token plan.
- SpatialVLA: two tasks; move_near has 10 physical layouts shared by all experiments; ±5 pp equivalence and task-level bounds were not established. Experiments 34/35 used history-dependent resets without restore hashes; Experiment 34 recorded no end-effector trajectory.
- Held-out selection results and navhard halves are log-disjoint but not drive-disjoint.
- P1-C: the pilot agreement criterion was relaxed before evaluation data existed (amendment 1); after an out-of-memory failure of one block, that block was re-run with the same command after two other contrasts had been viewed (amendment 2; judgments unchanged and the incomplete analysis preserved).

---

## Sentences to remove or replace

| Remove (as written in earlier drafts / summaries) | Why | Replace with |
|---|---|---|
| "Replacing only the most recent self-generated action token with ground truth … removes 93% of the effect" / "a single token correction" | Recent-GT is applied at every step; 93% is a share of the GT-history effect (85% of the Normal rate) | "Last-position correction applied at every step recovered 93% of the full-history correction effect." |
| Any unit-level p: "p = 8 × 10⁻⁴²", "6.7 × 10⁻³¹", "2 × 10⁻³¹", "7 × 10⁻²⁶¹", "p = 0.0 (underflow)", "p = 6.1 × 10⁻⁴" for direction | Units within scenes/logs/drives are not independent | Drive-level exact sign-flip p with its floor (e.g. "p = 2.4 × 10⁻⁴, the minimum attainable with 13 drives") |
| "Correcting the 2–4 context steps that follow the first deviation recovers 76–96%" / "a short 2–4 step critical window" / "genuine short early window" | 2 steps = 77% (CI ≤ 88%); no early-position advantage (P1-B) | "Correcting three consecutive context steps recovered about 90% of the full-history effect (95% CI 81–99%)." + "We found no evidence that corrections at the first deviating steps are more effective than equally long later corrections." |
| "A different token with GT-like motion is as effective as the GT token" / "identity does not matter" | ±5 pp equivalence not established (TOST p = 0.09) | "A token with GT-like motion produced a small, non-significant difference from the GT token (equivalence not established)." |
| "Direction errors are more harmful than magnitude errors" (unqualified) | Inconclusive on new drives (p = 0.073) | "… in the development drives; on new drives the effect was in the same direction but inconclusive." |
| "Self-amplification beyond extrapolation" / "the response is indistinguishable from constant-motion extrapolation in failing scenes" | The old comparison mixed in the baseline's own extrapolation error; the excess holds only in A− | "In failure-selected scenes the response exceeded constant-motion extrapolation; across all scenes it was not distinguishable from extrapolation." |
| "GT-free correction fixes failures" / "reference-based correction reduces deviation by 3.5 m" presented as task improvement | It restores reference consistency but worsens GT amplification (+19.7 pp in A−) | "Reference-based correction restored consistency with the model's own rollout but did not reduce ground-truth failures." |
| "4-step replanning makes reverse increase failures (−6.25 pp, p = 0.04)" / "a small task effect appears only for reverse" | p = 0.14–0.26 after initial-state clustering; not replicated (+1.25 pp) | (delete) |
| "Ensembling protects against / absorbs feedback" | Not supported once executed dose is matched | "Temporal ensembling did not absorb context effects once executed dose was matched." |
| "Task success changes by at most 6 pp" / "the context effect on task success is indistinguishable from zero" / "no task-level effect" | Clustered CIs are ±8–14 pp (Exps. 34/35); equivalence only for pooled P1-D at ±10 pp | "… their effect on success, averaged over two tasks, was bounded within ±10 pp in the tested configurations; task-specific effects remain uncertain." |
| "Context-only perturbation moves the trajectory 0.21–0.23 m (vs rerun 0.14 m)" / "+0.116 m trajectory divergence" | Cumulative commanded action, not the actual TCP | Actual TCP: "2.7–6.0 cm mean end-effector distance (P1-D)" |
| "Replanning and ensembling cause the absorption" / "replanning interval alone controls persistence" | Ensemble half deleted; replanning effect is composite (observation refresh, context reset, execution exposure) | "Lengthening the replanning interval increased the persistence of context-induced trajectory deviations even at a fixed context-perturbation budget." |
| Cross-model row "Reverse effect: ✓ ✓ ✓" without definitions; comparing +29 vs +82 pp | Different treatments and amplification definitions | "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across columns." |
| "Pre-registered held-out" / "new held-out drives" for Exps. 21–26 | Held-out shares 22 drives (85% of scenes) with dev; navhard halves share 16 drives | "held-out logs (not drive-disjoint)" |
| "Replicated on new logs" for P1-C | Understates it: drives are disjoint | "replicated on 24 new drives (day/vehicle partly shared)" |
| "SpatialVLA move_near: 120 distinct configurations" / "seed 0–119 = 53 independent states" as independent samples | Only 10 physical layouts, shared across experiments | "120 distinct initial states including the instruction, from 10 physical layouts" |
| "R8R9 confirms Experiments 8–9" | Recovery run with known targets; a reproducibility audit | "The original values were reproduced exactly (220/220) in a recovery run." |
| "Pre-deviation entropy is identical between selected and discarded candidates" | Difference is small but its CI excludes 0 | "Pre-deviation entropy differed only slightly (−0.016) compared with after the deviation (−0.40)." |
| "Attention masks have no effect" | Non-detection, equivalence not tested | "Masking attention to the history produced no detectable change (−1.4 and −3.3 pp)." |

---

## 2026-10-15 revision (review_followup3): exact sentences to KEEP and to DELETE/REPLACE

**Scope rule for this round.** The manuscript is a **driving-VLA** study:
- The main evidence is AutoVLA, open-loop 5 s plan evaluation on navtest.
- SpatialVLA (closed loop, SimplerEnv) and OpenVLA (closed loop, LIBERO-Spatial) are complementary task-level evidence, not replications.
- Every result sentence names its domain, model and evaluation mode.
- No sentence speaks about "VLAs in general" or "autoregressive VLAs".

**Sources.**
- Numbers: `numbers.json` (row IDs in brackets).
- Verdicts: `autovla_misalignment_poc/outputs/review_followup3/CLAIM_EVIDENCE_MATRIX.md` (final).
- **This section supersedes conflicting wording above.** In particular, the abstract's opening sentence and the conclusion's first sentence are replaced (R1, R2).

### A. KEEP (use as written)

**K1** — driving / AutoVLA / open loop [`P0A.DEV.RN.Am`, `P0A.DEV.RV.Am`, `P0A.P1C.RN.F`, `P0A.P1C.RV.F`]

> "In AutoVLA (open-loop 5 s plans on NAVSIM navtest), repeatedly replacing the most recent action-context token with ground truth at every generation step reduced amplification in failure-stratum scenes by 39.8 pp [29.9, 49.0] (13 drives), and injecting the separate Normal branch's previous token into a ground-truth context restored it by 28.6 pp [20.9, 36.1]; both effects replicated in a preregistered test on 24 new drives (−34.2 and +26.8 pp)."

**K2** — driving / AutoVLA / open loop [`P0A.P1C.RN.P`, `P0A.P1C.RV.P`, `P0A.PREV.DEV.Aminus.2747.drive`]

> "On the evaluation population of the new-drive logs (2,191 eligible scenes), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5]; the failure (A−) category occurred in 1.9% of development evaluation scenes, a frequency in the evaluation set rather than in the deployment distribution."

**K3** — driving / AutoVLA / open loop, GT-free [`P0C.plus437.AUTHORITATIVE`]

> "Without ground-truth information, perturbing a single context token moved the 5 s endpoint 4.37 m [3.34, 5.17] beyond pure coordinate accumulation (13 drives; Holm-adjusted p = 7.3 × 10⁻⁴)."

If the sentence instead describes the post-hoc excess metric, quote [3.32, 5.19] with "post hoc" and Holm 9.8 × 10⁻⁴. Never mix the two rows.

**K4** — driving / AutoVLA / open loop [`P2A.excess.G.*.end.*`, `P2B.excess.G.*.end.*.primary`, `P2.verdict.*`]

> "In failure-selected scenes, this response exceeded four kinematic extrapolations — including the original constant-motion baseline, which repeats the perturbed token's body-frame motion including its yaw — and a small learned non-autoregressive motion extrapolator (e.g. +2.35 m [1.27, 3.19] over constant motion; +1.97 m [1.06, 2.87] over the learned extrapolator); across all scenes and in non-failure scenes it was not distinguishable from motion extrapolation."

**K5** — driving / AutoVLA / open loop [`P0C.R9.rec.win3.NEW_scene_drive`, `P0C.R9.counts.*`]

> "Correcting three consecutive context steps recovered about 90% (scene-weighted; 95% CI 81–99%) of the full-history correction effect (365 units, 52 scenes, 13 drives for every window length); correction length and the remaining post-release horizon change together."

**K6** — driving / AutoVLA / open loop [`P0C.P1B.*`]

> "We found no evidence that corrections at the first deviating steps are more effective than equally long later corrections."

**K7** — driving / AutoVLA / open loop [`P0A.DEV.DM.Am`, `P0A.P1C.DM.F`]

> "Direction errors in the last context token were more harmful than magnitude errors in the development drives (+9.9 pp [3.9, 21.0]); on new drives the effect had the same sign but was inconclusive (+6.2 pp, p = 0.073)."

**K8** — manipulation / SpatialVLA-4B / closed loop, SimplerEnv [`P0B.P1D.*`]

> "In SpatialVLA closed-loop manipulation (SimplerEnv Google Robot; two tasks), context perturbations changed generated actions and the physical end-effector trajectory, while their effect on task success, averaged over the two tasks, was bounded within ±10 pp in the tested configurations; per task, ±10 pp equivalence was established for 3 of 8 contrasts and ±5 pp for none, so task-specific effects remain uncertain."

**K9** — manipulation / SpatialVLA-4B / closed loop

> "Lengthening the replanning interval increased the persistence of context-induced trajectory deviations even at a fixed context-perturbation budget (a composite configuration effect); temporal ensembling did not absorb context effects once executed dose was matched."

**K10** — manipulation / OpenVLA-7B / closed loop, LIBERO-Spatial [`P0E.succ.C-F_d8`, `P0E.succ.C-F_d24`, `P0E.phaseC.d0_|8|_feedback`]

> "In OpenVLA-7B on LIBERO-Spatial (100 initial states, one episode per state and condition), correcting the context after an x-token perturbation changed task success by −3 pp (95% CI [−12, +6], initial-state clusters; 8 bins) and −5 pp ([−14, +3]; 24 bins), while re-running the unperturbed policy alone changed 20% of outcomes; within a single step, an 8-bin x-token shift moved a later dimension by at least 8 bins in 92.6% of units (initial-state-cluster CI [90.8, 94.0]). These results are descriptive."

**K11** — verification [`P1.*`]

> "No-op, self-patch, cache-invalidation and full-recompute parity of the intervention code was verified on fixed verification manifests (AutoVLA: all six intervention code paths; SpatialVLA: state restoration on 20 units); earlier runs outside these manifests are not retro-validated, and the Impromptu VLA and OpenVLA intervention paths were not parity-verified."

**K12** — cross-model

> "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across models."

### B. DELETE or REPLACE

| # | delete / replace (as written in drafts or summaries) | why | replace with |
|---|---|---|---|
| R1 | Abstract opening: "Autoregressive vision-language-action (VLA) models generate each action conditioned on the actions they have just produced." | General VLA claim | "In driving vision-language-action models that generate a plan token by token, each action is conditioned on the actions just produced; we test whether this self-generated context causally shapes later actions in AutoVLA." |
| R2 | Conclusion: "Self-generated action context is a causal input to subsequent action generation in autoregressive VLAs." | General VLA claim | "In AutoVLA (open-loop driving), self-generated action context is a causal input to subsequent action generation." |
| R3 | "Analogous context interventions … changed subsequent action tokens in Impromptu VLA, OpenVLA and SpatialVLA" when presented as replications of the driving result | Not matched; Impromptu/OpenVLA machinery not parity-verified | K12 plus "Model-specific context interventions also changed subsequent action tokens in Impromptu VLA (open-loop driving), OpenVLA and SpatialVLA (manipulation); these are complementary observations, not replications." |
| R4 | "+4.37 m [3.34, 5.18]" / "+4.37 m [3.32, 5.19], Holm 7.3 × 10⁻⁴" | 5.18 is a double-rounding error; the predeclared Holm was attached to the post-hoc CI | K3 |
| R5 | "+0.57 m/seg [0.44, 0.68]" | Double rounding | "+0.57 m/seg [0.43, 0.68]" |
| R6 | "52/2,747 (1.9%) natural failure rate" / "population effect −4.1/+3.6 pp" (unqualified) | An evaluation-set frequency and an evaluation-population estimand | K2 |
| R7 | "The constant-motion baseline ignores curvature" (reviewer premise) / adding "CTRV" as a curvature fix of CM | CM is already a body-frame SE(2) repeat with yaw (constant turn rate) | Rebuttal: "The constant-motion baseline repeats the perturbed token's body-frame translation and yaw increment, i.e. it is already a constant-turn-rate extrapolation; we additionally report zero-turn, multi-step turn-rate, turn-rate-plus-acceleration and learned non-autoregressive extrapolators (Table 9)." |
| R8 | "Beyond extrapolation" / "self-amplification beyond motion extrapolation" (unrestricted) | Holds only in A− | K4 |
| R9 | SpatialVLA exp 34 docs: "the intervention arms are identical to HF `generate`" / "generate-identical decoding" | Batch-8 chunks equal HF `generate` in only 16/20 audit units (B = 1 decode: 17/20) | "All arms decode with the same fixed batch of 8 and are therefore internally consistent; outputs are not identical to HF `generate` (bf16 batch-shape numerics)." |
| R10 | "+1.2 pp [−5.6, +8.0]" (P1-D r4e1 δR) / "−6.2 pp" (exp 35 r4e1) | Exact half-even ties | "+1.25 pp [−5.6, +8.0]" / "−6.25 pp" (the exp 35 claim stays deleted) |
| R11 | "Task-level CIs span 18–29 pp" (unscoped) | Applies to P1-D r4e1 only | "Task-level CIs span up to 18–29 pp (r4e1)" |
| R12 | "92.6% [89.8, 94.4]" without its cluster unit | That CI uses task clusters | "92.6% (task-cluster CI [89.8, 94.4]; initial-state-cluster CI [90.8, 94.0])" |
| R13 | "Recovery 47/76/90/96%" and "47/77/90/96%" used interchangeably | Unit- and scene-weighted estimands | Name the weighting; scene-weighted is primary (K5) |
| R14 | "Validated on held-out / external data" / "day- and vehicle-independent" for P1-C | No day/vehicle-disjoint logs exist; P1-C shares days and vehicles | "replicated on 24 new drives (days and vehicles partly shared); no external (day/vehicle-disjoint) validation was possible" |
| R15 | "Reverse is the same treatment in all models" / "matched Reverse" | P3 not run | K12; AutoVLA arm = "Normal-branch token injection" |
| R16 | "New physical initial states confirm the SpatialVLA result" / "±5 pp per task" | P4 not run | (delete) |
| R17 | "All past runs were verified" | Manifest scope only | K11 |

### C. Limitation sentences that must stay (prompt review table + this round)

1. (M1) "The A− frequency is a frequency in the evaluation set, not in the deployment distribution."
2. (M2) "Effect sizes cannot be compared across models; the Impromptu Reverse remains unmatched because the matched Reverse experiment was not run."
3. (M3) "The learned extrapolator is a motion model, not a non-autoregressive version of the VLA."
4. (M4) "Task-level evidence comes from only two SpatialVLA tasks; move_near has 10 physical layouts, and no new physical states were sampled."
5. (M5) "Without day/vehicle-disjoint logs there is no external validation; under day × vehicle clustering the new-drive contrasts cannot reach Holm-adjusted p < .05 with six independent groups."
6. (M6) "Past runs outside the verification manifests are not retro-validated; the Impromptu and OpenVLA intervention paths were not parity-verified."
7. (M7) "Rows without unit-level data (Experiments 5 and 10) are reported as summaries only."
8. (m4) "OpenVLA precision is low with 10 initial states per task."
