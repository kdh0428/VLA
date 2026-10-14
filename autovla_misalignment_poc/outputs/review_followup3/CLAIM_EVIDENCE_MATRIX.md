# CLAIM_EVIDENCE_MATRIX — CP2 draft (review_followup3, 2026-10-14)

**Supersedes** `outputs/review_followup2/CLAIM_EVIDENCE_MATRIX.md` (2026-10-13), which is kept unchanged.

**Row IDs** refer to `/root/VLA/paper_quantitative_package/numbers.json`. A row's `status` field states how far it may be used:

| status | use |
|---|---|
| VERIFIED_MACHINERY_MANIFEST_SCOPE | usable with the manifest-scope caveat |
| GATE_NOT_APPLICABLE | usable |
| PROVISIONAL | do not enter as final |
| PENDING_DECISION_A5 | hold final wording until A5 is decided |
| NOT_VERIFIED | descriptive only |

**Tags.** Every claim carries domain / model / evaluation mode:

| tag | meaning |
|---|---|
| **D-AV-OL** | driving / AutoVLA (Qwen2.5-VL-3B, 2,048 action tokens) / open-loop 5 s plan on navtest |
| **M-SV-CL** | manipulation / SpatialVLA-4B / closed loop, SimplerEnv Google Robot (pick_coke_can, move_near) |
| **M-OV-CL** | manipulation / OpenVLA-7B / closed loop, LIBERO-Spatial |
| **M-OV-OFF** | the same, offline within-step decode |
| **D-IM-OL** | driving / Impromptu VLA 3B / open-loop |

No claim is about "VLAs in general" or "autoregressive VLAs".

**Verdicts.**

| verdict | meaning |
|---|---|
| keep | the sentence may be used as written |
| weaken | narrow the scope or strength as stated |
| delete | do not use |
| unresolved | the evidence needed does not exist yet; the manuscript narrows the claim and states the limit |

**Manifest-scope caveat** (applies to every D-AV-OL row marked VERIFIED): the no-op/cache parity gate passed for the Recent-GT / Full-GT, Reverse-N, window and self-reference paths on a 21-unit verification manifest. The original full runs outside that manifest are not retro-validated.

## Driving (main result)

| # | claim (allowed wording, condensed) | tag | evidence rows | verdict | machinery | limits that must stay in the text |
|---|---|---|---|---|---|---|
| D1 | Repeatedly replacing the most recent action-context token with ground truth reduced amplification in failure-stratum scenes, by −39.8 pp [−49.0, −29.9] (13 drives). On 24 new drives (preregistered) the reduction was −34.2 pp [−38.3, −29.5]. | D-AV-OL | `P0A.DEV.RN.Am`, `P0A.P1C.RN.F`, `P0D-000`…`P0D-005`, `P0D-018`…`P0D-023` | **keep** | VERIFIED (H7) | Recent-GT is a **repeated sliding** correction (8 − t* applications), not a single correction. The new drives share days and vehicles with development (new-drive, not day/vehicle-disjoint). Under day–vehicle component clustering (G = 6) the exact p floor is 0.031. |
| D2 | Re-inserting the separate Normal branch's previous token into a ground-truth context restored amplification by +28.6 pp [+20.9, +36.1] (13 drives); new drives: +26.8 pp [+21.9, +30.7]. | D-AV-OL | `P0A.DEV.RV.Am`, `P0A.P1C.RV.F`, `P0D-006`…`P0D-011`, `P0D-030`… | **keep** (wording fixed) | VERIFIED (H8; Reverse-N ≡ token substitution checked by C4) | Say "**Normal-branch token injection**", never "the model's own error re-inserted" (INTERVENTION_SPEC M-A1). Not the same treatment as the Impromptu or SpatialVLA "Reverse" (see X1). |
| D3 | On the P1-C evaluation frame (2,191 eligible scenes, 24 drives), the scene-weighted effects were RN −4.1 pp [−6.4, −2.4] and RV +3.6 pp [+2.2, +5.5]. | D-AV-OL | `P0A.P1C.RN.P`, `P0A.P1C.RV.P`, `P0A.DECOMP.RN`, `P0A.DECOMP.RV` | **keep, renamed** | VERIFIED | Call it the **evaluation-population estimand**, not a population or deployment effect. 60 % of RN P comes from the A+ stratum. |
| D4 | The A− (failure-category) frequency in the development evaluation set was 52/2,747 = 1.89 % [1.16, 2.64] (drive clusters). | D-AV-OL | `P0A.PREV.DEV.Aminus.2747.drive`, `P0A.FLOW.*` | **keep, renamed** | GATE_NOT_APPLICABLE | "Evaluation-set frequency, not a deployment or natural failure rate." Never multiply it by a stratum effect. |
| D5 | Last-position correction applied at every step recovered ≈ 93 % of the full-history correction effect. | D-AV-OL | review_followup2 `CLUSTER_REANALYSIS.csv` (not merged into numbers.json) | **weaken** (unchanged since review_followup2) | VERIFIED paths (H7/H8), manifest scope | The denominator is the GT-history effect, not Normal. The correction is repeated, not single. |
| D6 | Correcting three consecutive context steps recovered ≈ 90 % (scene-weighted 90.1 %, 95 % CI 81.2–98.9) of the full-history effect. All window lengths were measured on the same 365 units / 52 scenes / 13 drives. | D-AV-OL | `P0C.R9.rec.win3.NEW_scene_drive`, `P0C.R9.rec.win3.OLD_exp9`, `P0C.R9.counts.*` | **weaken** | VERIFIED (H9) | Window length and post-release horizon change together. Quote the estimand: 47/76/90/96 % (exp 9 unit-weighted) and 47/77/90/96 % (scene-weighted) are different estimands. Fig. 2 caption: n = 365 units, 52 scenes, 13 drives (10 day–vehicles). |
| D7 | "The first 2–4 deviating steps are a special window." | D-AV-OL | `P0C.P1B.*` | **delete** (keep the no-evidence sentence) | PROVISIONAL (P1-B script not in the manifest) | Allowed: "We found no evidence that corrections at the first deviating steps are more effective than equally long later corrections." Fixed-total-horizon and fixed-post-release estimands differ. Not machinery-verified. |
| D8 | Direction errors were more harmful than magnitude errors in the development drives (+9.9 pp [+3.9, +21.0]). On new drives the effect had the same sign but was inconclusive (+6.2 pp, p = 0.073). | D-AV-OL | `P0A.DEV.DM.Am`, `P0A.P1C.DM.F`, `P0D-012`… | **weaken + unresolved** | **PROVISIONAL** (H32 not in the manifest) | Do not enter as final until the H32 path is gated (Stage-3 option O0). |
| D9 | Without ground-truth information, a perturbed context token changed later local actions and moved the endpoint by +4.37 m relative to the geometric (coordinate-accumulation) reference: predeclared P1A-2, drive bootstrap 95 % CI [+3.34, +5.17], Holm (m = 3) 7.3e-4. | D-AV-OL | `P0C.plus437.AUTHORITATIVE` (V2/AUTH); post-hoc variant `P0C.plus437.V3` | **keep (AutoVLA only)** | VERIFIED (P1A) | Use one version. The V3 numbers [3.32, 5.19] / Holm 9.8e-4 are the post-hoc excess family and must be labelled post hoc if used. The 5.18 upper bound is a double-rounding error. GT-free contrasts exist for AutoVLA only. |
| D10 | In failure-selected (A−) scenes, the endpoint response exceeded every motion-extrapolation control: CM +2.35 m [+1.27, +3.19], CV0, CTRV-K3, CTRA-2, and the learned non-AR extrapolator (+1.97 m [+1.06, +2.87]; with ego state +2.27 m). Across A+ scenes and all scenes it was not distinguishable from motion extrapolation. | D-AV-OL | `P2A.excess.G.A-.end.*`, `P2B.excess.G.A-.end.E_prefix.primary`, `P2B.excess.G.A-.end.E_prefix_ego.primary`, `P2.verdict.A-`, `P2.verdict.A+`, `P2.verdict.all` | **weaken** (stratum-restricted) | VERIFIED (P1A incl. rollout-saving path) | The general "beyond extrapolation" claim stays deleted. The extrapolator is a small motion model, **not a non-AR version of the VLA**. In family G, 8–9 % of units have a prefix out of the training range (`P2B.coverage.G.*`); excluding them keeps the sign. |
| D11 | Rebuttal fact: the original "repeat-p constant-motion" baseline is a body-frame SE(2) repeat including yaw, i.e. already a constant-turn-rate extrapolation. | D-AV-OL | `P2/MOTION_BASELINE_AUDIT.md` §1; `P2/cm_definition_summary.json` | **keep** | code check | Do not present CTRV-K3 as a curvature fix for CM. It is a different (multi-step) control. |
| D12 | Reference-based (GT-free) correction fixes failures. | D-AV-OL | review_followup2 C / P1-A OC_refR | **delete** (unchanged) | VERIFIED path | Allowed: it "restored consistency with the model's own rollout but did not reduce ground-truth failures". |
| D13 | Effects are robust to higher-level dependence. | D-AV-OL | `P0D-*` (all axes), `DEPENDENCE_leave_one_group_out.csv` | **weaken** | VERIFIED (H7/H8 rows), PROVISIONAL (DM rows) | The sign is kept on every axis and under leave-one-group-out. P1-C contrasts cannot reach Holm < .05 under day × vehicle component clustering (G = 6, Holm floor 0.094): this is an inference limit, not evidence against the effect. |
| D14 | External / out-of-distribution validation. | D-AV-OL | `P0D.DISJ.*` (0 day-and-vehicle-disjoint logs) | **unresolved — addressed by narrowing** | – | Keep "no external (day/vehicle-disjoint) validation". P1-C is "new-drive". Whether the checkpoint was model-selected on navtest cannot be determined. |
| D15 | Selection / navhard held-out results. | D-AV-OL | review_followup2 (unchanged) | **weaken (label)** | not in the gate | "held-out logs (not drive-disjoint)" |
| D16 | Impromptu VLA reproduces the token-level temporal pattern (correction, reverse, 3-step window). | D-IM-OL | exp 33 (`cross_vla_temporal_replication/RESULTS.md`); not merged | **weaken + unresolved** | **NOT_VERIFIED** (P1 not run; exp 33 raw no-op parity poor, M-I1) | Impromptu "reverse" re-inserts the row's own waypoint dynamically: it is a different treatment from AutoVLA Reverse-N. Machinery not parity-verified. Effect sizes are not comparable across models. |

## Manipulation (complementary task-level evidence, not a replication)

| # | claim | tag | evidence rows | verdict | machinery | limits |
|---|---|---|---|---|---|---|
| S1 | Context perturbations changed generated actions and the physical trajectory. Averaged over the two tasks, the effect on success was bounded within ±10 pp in the tested P1-D configurations (all 4 pooled contrasts, Holm, both clusterings). | M-SV-CL | `P0B.P1D.*.pooled.primary`, `P0B.P1D.*.pooled.physical` | **weaken** (two-task mean only) | **PENDING_DECISION_A5** | The mean equivalence is not per-task equivalence. ±5 pp is not established (pooled or per task). |
| S2 | Per-task success effects (8 contrasts per experiment, P1-D, primary clustering): ±10 pp equivalence after Holm holds for 3 of 8 (r1e4 R−N coke, r1e4 R−N move_near, r1e4 C−F move_near). The rest are uncertain, with CI widths up to 18–29 pp (r4e1). | M-SV-CL | `P0B.P1D.{r1e4,r4e1}.{R-N,C-F}.{coke,move_near}.{primary,physical}` | **unresolved — addressed by narrowing** (P4 not run) | PENDING_DECISION_A5 | Quote point estimates with CIs. move_near has only 10 physical layouts (independent physical precision is limited). Only two tasks. |
| S3 | Task × intervention heterogeneity | M-SV-CL | `P0B.het.*` | **keep as "neither detected nor excluded"** | PENDING (P1-D) / NOT_VERIFIED (exp 35) | Opposite signs cancel up to 100 % in the P1-D r4e1 C−F mean (`P0B.het.P1D.r4e1.C-F.primary`). |
| S4 | Exp 35: "reverse increases failure under 4-step replanning (−6.2 pp)" | M-SV-CL | `P0B.exp35.r4e1.R-N.pooled.primary` | **delete** (unchanged) | NOT_VERIFIED (no restore) | – |
| S5 | A longer replanning interval increased the persistence of context-induced trajectory deviations at a fixed context budget; ensembling did not absorb them at matched dose. | M-SV-CL | review_followup2 / P1-D analysis (not merged) | **keep (trajectory-level, composite setting)** | PENDING_DECISION_A5 | The replanning axis bundles observation refresh, context reset and execution exposure. |
| S6 | "Identical to HF generate" (exp 34 docs) | M-SV-CL | `P1.SV.C4_t0_report.*` (B = 1 == batch-8: 17/20; generate == batch-8: 16/20) | **delete** | audit | Arms are internally consistent (fixed batch 8), but not equal to `generate`. |
| O1 | OpenVLA: correcting the context after an x-token perturbation changed success by −3 pp (95 % CI [−12, +6], initial-state clusters, 8-bin) and −5 pp ([−14, +3], 24-bin). Re-running the unperturbed policy alone changed 20 % of outcomes. | M-OV-CL | `P0E.succ.C-F_d8`, `P0E.succ.C-F_d24`, `P0E.succ.Nrep-N` | **weaken → descriptive** | NOT_VERIFIED | 100 initial states, 10 per task (low per-task precision, `P0E.succ_task.*`). The episode-level intervals (−3 [−9, 4], −5 [−14, 4]) are kept side by side, not overwritten. |
| O2 | OpenVLA within-step dependency: an 8-bin x shift moved a later dimension by ≥ 8 bins in 92.6 % of units (initial-state CI [90.8, 94.0]). | M-OV-OFF | `P0E.phaseC.d0_|8|_feedback`, `P0E.dim.*` | **weaken** (definition stated) | NOT_VERIFIED | Unit = (frame, sign), n = 4,741. Clipped units and the gripper are excluded. In physical units the commanded downstream translation exceeds the injected one in 72 % of units. These are commanded deltas, not realised motion. |

## Cross-model

| # | claim | verdict | evidence | limit |
|---|---|---|---|---|
| X1 | "Reverse" is matched across models | **unresolved — addressed by narrowing** (P3 not run) | `P1/INTERVENTION_SPEC.md` §5 | "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across columns." Without P3, the Impromptu Reverse stays unmatched. |
| X2 | Any "VLA in general" / "autoregressive VLAs" statement | **delete** | – | Scope = driving VLA (AutoVLA); manipulation results are complementary. |

## Rows that stay PROVISIONAL or NOT_AVAILABLE

| item | status | why / what would close it |
|---|---|---|
| DM rows (`P0A.*.DM.*`, `P0A.DECOMP.DM`, `P0D-012…017`, `P0D-042…053`) | PROVISIONAL | H32 is not in the gate manifest; option O0 adds it (≈ 0.4–0.8 GPU h) |
| P1-B start-position rows (`P0C.P1B.*`) | PROVISIONAL | the start-position script is not in the gate manifest; option O0 |
| SpatialVLA P1-D rows | PENDING_DECISION_A5 | A5 decision (accept the post-hoc policy-state criterion, or re-run the null arm with separated hashes) |
| Exp 5 / exp 10 summary numbers | NOT_AVAILABLE | no per-unit raw output on disk (review_followup2 DATA_AUDIT) |
| Deployment prevalence and deployment effect | NOT_AVAILABLE | no deployment sample |
