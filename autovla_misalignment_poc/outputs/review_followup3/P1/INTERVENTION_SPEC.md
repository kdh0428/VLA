# INTERVENTION_SPEC — P1-A implementation specification (2026-10-14)

Scope: AutoVLA (primary, Stage-0 gate), SpatialVLA (P1-B/P1-C planned), Impromptu VLA and OpenVLA (**spec only**: no new run in this round, so nothing here is a parity result for them).

## Provenance

| Item | Value |
|---|---|
| Repo | `/root/VLA`, branch `autovla-experiments-11-19`, HEAD `817d166` |
| AutoVLA code | `/root/VLA/autovla`, separate repo at `ba34eed` |
| Environment | conda `autovla`: torch 2.8.0+cu128, transformers 4.49.0, model in bf16 |
| Line numbers | All `file:line` references are at HEAD 817d166. Script paths are relative to `autovla_misalignment_poc/scripts/` unless absolute. |

How this was produced:
- AutoVLA: read directly for this document.
- SpatialVLA, Impromptu, OpenVLA: from two read-only sub-audits. Their key lines were spot-checked: `exp_cde.py:42-81`, `svla_rollouts.py:124-128`, `p1d_runner.py:305-312`, `svla_offline.py:105-111`, `openvla_core.py:120-136`.
- Earlier audits this builds on: `outputs/review_followup/P0A_audit/run_20261008/{metric_spec.md, analysis.md}` and `outputs/review_followup2/DATA_AUDIT.md`. Where this document and P0A agree, P0A's code locations are re-verified against the current HEAD.

**This document describes code, not results.** Measured parity is in `NOOP_CACHE_AUDIT.*`, which is produced by the Stage-0 gate. Before that gate runs, only the control-5 pilot exists (`PROTOCOL_P1.md` §5).

---

## 0. Highlights: mismatches between the documented and the implemented interventions

Each item gives the code location, the documented wording it conflicts with, and the affected manuscript area. The manuscript areas follow the prompt's "P1 실패 시 보류" (sections withheld if P1 fails) table.

**M-A1. AutoVLA "Reverse-N" (exp 8 and P1-C H8 `reverse@emb`) is not "the model's own error re-inserted".**
- Context: the GT-history prefix at positions t*+1..k−2.
- Position k−1 gets the **input embedding of the separate Normal row's token c^N_{k−1}**, from the same batch (`prev_action_state_patching.py:90-91, 174-202`; source row index `src_of["normal"]`, `:124`).
- The reverse row's own sampled tokens never enter its own context.
- A layer-0 input patch of one position is mathematically a **token substitution**: AutoVLA adds no positional term to the embedding, because M-RoPE acts inside attention. So reverse@emb ≡ "GT history + Normal's latest token". The Stage-0 control C4 checks this equivalence numerically.
- Impromptu's `reverse` is a different treatment: GT history plus the reverse row's **own** k−1 waypoint, chosen dynamically (`cross_vla_temporal/exp_cde.py:57-58`).
- The P1-C protocol already labels the AutoVLA arm "Normal-branch token injection" (`P1C.../design_20261008/protocol.md:102,108,230`).
- Any text that calls these the same "Reverse" treatment, or a cross-model "Reverse" row, is a mismatch (M2). Affected: §5, Table 4, §7.1.

**M-A2. Recent-GT is a repeated sliding correction, not a single correction.**
- At every step k ≥ t*+2, only position k−1 holds gt[k−1]. Earlier positions revert to the row's own outputs (`action_history_causal.py:210-212`; P1-C H7; exp 32 `motion_semantics_ablation.py`).
- So it is applied 8−t* times per unit.
- Whether the GT token actually differed from the own token at each application is **not logged** in exp 7 / P1-C H7. P0A computed post hoc that the token changed 4.32 times on average for A−.
- Exp 9 `win_w` is a different schedule: GT is written into fixed absolute positions t*+1..t*+w and **stays** in the context for every later step (`temporal_feedback_window.py:189-197`). Nothing is "released" back to the model's own token at those positions.
- The only one-shot context schedule is P1-A `OC_ref1`, a one-shot reference (not GT) correction at t*+1 that stays in the context (`p1a_gt_free_reference.py:79`).
- The output-level one-shot GT forcing of exp 5 (`first_mismatch_causal.py`) is a different kind of intervention again.

**M-A3. Normal is not the same object across AutoVLA experiments.**
- Natural rollouts were produced with HF `generate` over the whole vocabulary (`full_extract.py:391-406`, `equal_distance_perturbation.py:241-244`).
- The harnesses resample only the 2,048 action rows, with their own seeds.
- They also run at different batch sizes: B = 1 (exp 7, P1-C H7), 93 (exp 8 / H8), 17 (exp 9 / H9), 6 (exp 32 / H32), 10 (P1-A).
- Measured Normal 10-token agreement is 71.6% (exp 7 vs exp 6) and 88–89% (exp 8/9/32 vs exp 7) (P0A A5).
- Each contrast in use is computed **within one batch and one harness**: RN within H7 at B = 1, RV within H8 at B = 93. So the intervention contrasts are internally paired.
- But "Normal" percentages are not interchangeable across tables. The no-op gate must also measure the batch-composition floor (pilot control c5c).

**M-A4. The prefix-cache integrity check is weak.**
- The checksum is Σ|K| of layer 0, row 0 only, with a relative tolerance of 1e-3 (`p1_common.py:255-265`; same pattern in `prev_action_state_patching.py:271,369-371` and `temporal_feedback_window.py:172,217-219`).
- It would not detect a change in other layers or rows, or a change below 1e-3 of the sum.
- Stage-0 replaces it with logits/hash parity (C3, C4).

**M-A5. The PRA "A" thresholds are documented for 3 s and applied over 5 s.**
- `vla_misalignment_poc/analysis/trajectory_metrics.py:20-31` documents the thresholds for a 6-step / 3 s horizon.
- AutoVLA passes 10 steps (5 s; `equal_distance_perturbation.py` `a_label`, `run_pra_comparison.py:76-81`).
- So v_end is the mean of steps 9–10, and STOP means < 1 m over 5 s. Already noted by P0A (F7).
- This is a wording issue, not a code bug.

**M-A6. Video M-RoPE timing input: checked, no mismatch.**
- The harness forwards never pass `second_per_grid_ts` (`p1_common.py:246-247`), so `get_rope_index` falls back to 1.0 (`modeling_qwen2_5_vl.py:1662-1665`). The natural stub was generated with `generate(**mi)`.
- The control-5 pilot recorded `get_prompt` keys = {attention_mask, input_ids, pixel_values_videos, video_grid_thw}. `second_per_grid_ts` is absent, so both paths use the same default and there is no mismatch.
- Measured for the pilot scenes: prompt_len 1,209; rope_deltas −675; video_grid_thw [[2,16,30]] × 3 cameras.

**M-S1. SpatialVLA has two different "reverse" treatments.**
- Token level: the step-4 context [S1p, R2, **N3 from the separate Normal row**] (`cross_domain_temporal/svla_offline.py:111`).
- Closed loop: execute g, context p (`svla_rollouts.py:128`, `p1d_runner.py:307`).
- In the closed-loop R and C arms, step-1 rotation and gripper tokens are generated **after** the context token. So "R executes the natural step 1" holds for the translation token only.
- Affected: §7.3/§7.4.

**M-S2. SpatialVLA exp 34/35 have no state restore.**
- Exp 34/35 reset with `env.reset(seed)` on a cached env, with no reconfigure. Measured reset drift is up to 0.26 mm and about 0.5° on move_near (review_followup2 A_spatialvla_states).
- P1-D branches by **replay**, with `get_state()` and image hashes checked per step.
- Controller integrators/targets, physics time and env RNG state are not fingerprinted. See §4.3.
- The exp 34 "trajectory divergence (m)" is a commanded-action proxy, not TCP (P0A F3).

**M-I1. Impromptu.**
- Text waypoints are rounded to 0.01 m in the context. The executed forced w1 is unrounded.
- The fast decoder re-tokenizes the context text every step.
- The no-op parity actually observed in exp 33 raw is poor: the delta-0 reference unit's `normal` row equals `natural` in 23/300 scenes; `natural_rep` equals `natural` in 237/2,728. PROTOCOL §6 of exp 33 understates this.
- Windows are persistent GT (no release); see M-A2.
- Affected: §7.1.

**M-O1. OpenVLA.**
- "reverse = feedback, corrected = natural" downstream within a step holds **by construction**: the executed token never enters the model.
- Non-action argmax ids are silently clipped to bin 254.
- Phase C natural vs Phase A executed bins agree in 82.7% of frames (batch composition).
- The 92.6% within-step dependency excludes clipped (Δ_eff = 0) units and the gripper. n = 4,741.
- Affected: §7.2.

---

## 1. AutoVLA — shared machinery (all intervention paths)

| Item | Implementation | Code |
|---|---|---|
| Model / dtype | Qwen2.5-VL-3B with 2,048 added action tokens. bf16 weights. Logits cast to float64 before softmax/sampling. | `autovla/models/autovla.py:480-483`; `p1_common.py:246-249` |
| Attention implementation / numerics | `sdpa`, from the pilot `run_meta_start.json`. Deterministic algorithms off, cudnn.benchmark off, TF32 matmul off, bf16 reduced-precision reduction **on**, no `CUBLAS_WORKSPACE_CONFIG`. Logits are bf16, so the logit grid spacing is 0.125–0.25 at the observed magnitudes; exact top-2 ties are possible and are broken by first index (greedy) or by the CRN draw (T = 0.01). | `modeling_qwen2_5_vl.py:1010-1025`; pilot |
| GT trajectory | NAVSIM local frame at the current ego pose: x forward, y left, heading in rad. 10 poses, 0.5 s apart (5 s). | `preprocess_scenes.py:95`; `autovla/navsim/navsim/agents/vla_agent.py:108-112` |
| GT → token projection | Sequential closed-loop matching. Start at (0,0,0). For each pose i, every codebook token is placed at the **previously matched token's** pose. Pick `argmin_token Σ_corners ‖corner − GT box corner‖` over a 2.0 × 4.8 m box at the last sub-step. Then move to the chosen token's pose, so the quantisation residual does not accumulate. Runs in float32. Ties go to the first index (`torch.argmin`). | `autovla/navsim/navsim/agents/autovla_agent.py:90-151` (argmin `:139-141`, state update `:146-151`), `:286-320` |
| Token → trajectory | Tokens are chained from (0,0,0). The pose after each token = the corner mean of its last sub-step box; heading = atan2(corner0 − corner3). Prediction and GT use the same frame. | `autovla/models/action_tokenizer.py:62-110`; numpy re-implementation `p1_common.py:150-162` |
| Prefix | `prompt (video tokens + text) + stub + prefix actions`. Stub = the stored natural fast stub, i.e. `full_extract` arm N `token_ids` up to the first action id (dev: `full_extract`; P1-C: `p1c_natural_pass.py` via `Planner.plan`). Prefix actions = `pred[:t*]` (exps 7–9, 32) or reference `r[:t_p]` (P1-A). | `p1_common.py:231-242`; `action_history_causal.py:184-186` |
| Action span | No begin/end special token is fed. The span is the run of action-token ids `A0+a`, starting at position P = len(prefix), which holds the action at t* (or t_p). Logits are restricted to the 2,048 action rows `[A0, A0+2048)`, i.e. renormalised over action tokens only. The natural `generate` path samples the whole vocabulary (M-A3). | `p1_common.py:279`; `action_history_causal.py:227` |
| Attention mask | 2-D all-ones `(B, P+m)`. HF builds the causal 4-D mask. All rows have equal length, so there is no padding. | `p1_common.py:276-277` |
| Position ids | M-RoPE. The prefill (cache_position[0] = 0) computes `get_rope_index` and stores `rope_deltas`. Every span forward uses `cache_position + rope_deltas`, the same on all 3 axes (text after video). `rope_deltas` is reset per scene (`llm.rope_deltas = None`). | `modeling_qwen2_5_vl.py:1831-1859`; `p1_common.py:244` |
| KV cache | One B = 1 prefill per scene, expanded to B rows (`expand(...).contiguous()`). At each step the span `[a_t*, ctx_{t*+1..k-1}]` (m = k − t* tokens) is fed on top of the cache, then `cache.crop(P)`. **Invalidation range = [P, P+m) every step; recompute start = P (the t* position).** Positions < P are never recomputed. | `p1_common.py:251-254, 268-281` |
| Already-emitted actions | The executed output is `prefix actions + [f] + own` (own = this row's sampled tokens). Context-level interventions never change the executed tokens. All AutoVLA interventions below are context-only, except exp 5 (gt) and exp 6 (forced token), which are output-level, and P1-A `O*`, which outputs p while the context reads r. | `action_history_causal.py:236`; `prev_action_state_patching.py:374`; `p1a_gt_free_reference.py:206-208` |
| Decoding / RNG | T = 0.01 (`inference.sample.temperature`). `torch.multinomial` on CPU from `softmax(logits/T)`. A `torch.Generator` is **re-seeded for every row** with `sha256(f"{seed}:{token}:{key}:{k}")[:4]`, giving common random numbers across rows and conditions. key = perturbation name (exps 7–9, 32, P1-C) or `"original"` (P1-A, scene-level CRN). Seed 0. Greedy is not used in any experiment. | `qwen2.5-vl-3B-nuplan-grpo-cot.yaml:69-74`; `p1_common.py:284-296`; `action_history_causal.py:232-235` |
| Visual cache | `llm.visual.forward` is wrapped once per process. It returns the first video embedding until `vis_box` is cleared, which happens at every scene prefill. Present in every path, untouched and hooked alike. | `p1_common.py:214-221`; `action_history_causal.py:157-164` |
| Batch composition | exp 7 / P1-C H7: B = 1, conditions run one after another on one shared prefix cache. exp 8 / H8: B = 93. exp 9 / H9: B = 17. exp 32 / H32: B = 6. P1-A: B = 10. P1-B (start position): B = len(build_rows). Reference rollouts (`p1_reference.py`): B = 1. | scripts as listed |
| Hooks present during *all* conditions | exp 7: embedding forward hook + attention wrapper on all 36 layers (pass-through unless a mask/neutral condition is active). exp 8: layer-0 forward pre-hook + 36 layer post-hooks; `PATCH.on = True` during every span forward. exp 9, P1-A, P1-B: none. | `action_history_causal.py:118-155`; `prev_action_state_patching.py:195-213, 307-311` |

### 1.1 Failure label used by every AutoVLA verdict (PRA "A" acceptance)
- `label_A` is at `/root/VLA/pra_comparison/pra_labels.py:469-488`. Kinematics come from `vla_misalignment_poc/analysis/trajectory_metrics.py:104-140`.
  - The trajectory is converted to per-step deltas (lat +right, lon +forward) by `_pos_to_delta(traj, 10)` (`run_pra_comparison.py:76-81`).
  - v_start / v_end = mean speed of the first / last two steps (DT = 0.5 s).
- Longitudinal class (`lon_class`, `pra_labels.py:357-368`), in order:
  - STOP if total distance < 1.0 m and v_end < 0.5 m/s.
  - Otherwise, if v_start < 0.5: ACCELERATE if v_end − v_start > 0.8, else MAINTAIN.
  - Otherwise by r = v_end / v_start: r ≥ 1.25 ACCELERATE; r ≤ 0.75 DECELERATE (or STOP if v_end < 0.5); else MAINTAIN.
- Lateral class (`lat_class`, `:371-374`): LEFT/RIGHT if |final lateral| > 2.0 m or |final-step heading| > 15°, else STRAIGHT. Heading is ignored when the last step is < 0.25 m (`kinematics`, `:377-384`).
- Acceptance (`accept_lon` / `accept_lat`, `:405-413`; `Config`, `:81-90`):
  - Longitudinal: the predicted class, computed from **GT v_start**, predicted v_end and predicted distance (`label_A:473`), must be in the GT class set. That set is obtained by shifting GT v_end by `max(0.5, 0.15·max(v_start, v_end))` on an 11-point grid.
  - Lateral: the predicted class must be in the GT class set obtained by shifting final lateral ±1.0 m and heading ±10° on a 5 × 5 grid.
  - A+ = lon ∧ lat; A− otherwise.
- Amplification (`analyze_action_history.py:45, 57-67`) = (not A+) ∧ FDE5 > 3.0 m. FDE5 is measured at pose 10 in the local frame.
- `Y_severe` (P1-C) is this amplification label (`analyze_p1c.py`; protocol §4).
- Thresholds: see M-A5.

### 1.2 Per-intervention table (AutoVLA)

Notation: t* = intervention position (first GT mismatch, inherited from exp 6); f = forced token at t* (`original` or distance-matched `alt*`); own = the row's own sampled tokens; k = step being generated (t*+1..9); the context at step k covers positions t*+1..k−1.

| Manuscript name / arm | Script : lines | Context at positions j = t*+1..k−1 | Source branch | Schedule | Applications per unit (changes logged?) | KV invalidation / recompute | B | Used in |
|---|---|---|---|---|---|---|---|---|
| Normal | `action_history_causal.py:213-214`; `prev_action_state_patching.py:295`; `temporal_feedback_window.py:194-195` (length 0) | own[j] | – | – | 0 | [P, P+m) each step | 1 / 93 / 17 | every contrast's reference |
| **Recent-GT** (RN = recent_gt − normal) | `action_history_causal.py:210-212` (P1-C H7 same script) | j = k−1 → gt[j]; j < k−1 → own[j] | GT | **repeated, sliding**: at every k ≥ t*+2 the last position only; earlier GT positions revert to own | 8 − t* (changes **not logged**; P0A post hoc: 4.32 actual changes per A− unit) | same | 1 | §5, Table 4, Fig 2 |
| **Full-GT / GT-history** | `action_history_causal.py:208-209`; `prev_action_state_patching.py:290-291`; `temporal_feedback_window.py` row `gt_history` (start 1, length 99) | all gt[j] | GT | persistent, growing | 8 − t* (not logged) | same | 1 / 93 / 17 | §5, Table 4, App. B |
| **Reverse-N** (RV = reverse@emb − gt_history) | `prev_action_state_patching.py:90-91` (row), `:174-202` (patch at layer-0 input, last span position), `:289-296` (base context = GT) | j < k−1 → gt[j]; j = k−1 → **embedding of Normal row's c^N_{k−1}** | **separate Normal branch in the same batch** | repeated at every k ≥ t*+2 (identity at k = t*+1, where the last position is f in every row) | 8 − t* (not logged; source/target tokens logged only in `--debug`) | same; the patch reaches the K/V of every later layer because the span is recomputed | 93 | §5, Table 4, App. B (M-A1) |
| patch_full@emb (≡ Recent-GT) | `prev_action_state_patching.py:81-82` | normal context, position k−1 embedding ← gt_history row (= gt[k−1]) | GT row | repeated | 8 − t* | same | 93 | consistency check (debug L1 = 0) |
| reverse@L / patch_full@L / identitysrc@L / patch_delta | `prev_action_state_patching.py:81-91` | hidden state of position k−1 after block L ← source row | Normal / GT / recent_gt rows | repeated | 8 − t* | same | 93 | layer analysis (exp 8) |
| Window win_w / delay / skipfirst | `temporal_feedback_window.py:58-65, 189-197` | j − t* ∈ [start, start+length) → gt[j], else own[j] | GT | **persistent fixed absolute positions** (no release) | GT present at all later steps | same | 17 | §5 window figure (Fig 2), R9 |
| Motion substitutes (DM = dir_wrong_mag_ok − dir_ok_mag_wrong) | `motion_semantics_ablation.py:54-75, 234-243` | j = k−1 → substitute s(g, o), else own | built from GT and own | repeated sliding | 8 − t* (**logged**: `substitutions`) | same | 6 | §5 DM |
| exp 5 gt / nn | `first_mismatch_causal.py:153-170` | – (output-level) | GT | **one-shot, output-level** at t*; then free HF generate | 1 | full HF generate | 1 | earlier claims |
| exp 6 forced original/alt | `equal_distance_perturbation.py:229-260` | – (output-level) | – | one-shot, output-level | 1 | HF generate (whole vocab, top_k 0, top_p 1, T 0.01) | 1 | source of t*, f, units |

### 1.3 P1-A self-reference rows (GT-free) — `review_followup/p1ab/p1a_gt_free_reference.py`

- **Reference r** (`p1_reference.py:34-98`): r[:t*+1] = stored natural pred; r[t*+1:] = harness rollout with B = 1, key "original". It is written to disk with sha256 before any condition runs. sha256 is checked at `p1a_gt_free_reference.py:106-110`.
- **Perturbation p** (`p1_common.py:165-190`): family G is GT-free. Per 45° sector of disp[p] − disp[r[t_p]], take the token with achieved distance closest to 0.10 m, within ±0.03 m. Ties go to the smaller angular error, then the smaller id.
- **Positions:** t_p = t* (inherited, GT-informed position) or fixed t_p ∈ {1, 3} (GF1/GF3). For GF, prefix = r[:t_p], which can include reference-continuation tokens when t_p > t*.
- B = 10. One batched forward per step, key "original".

| Row | Output at t_p | Context at t_p | Context j > t_p | Schedule | Code |
|---|---|---|---|---|---|
| U | r | r | own | – | `:57` |
| OC | p | p | own | – | `:58` |
| O1 | p | r | own | one-shot (context at t_p only) | `:59` |
| **C1** (context-only perturbation; input to §6.2 / Fig 1) | r | p | own | one-shot | `:60` |
| OC_ref1 | p | p | j = t_p+1 → r[j], kept | **one-shot** reference correction | `:61, 79` |
| OC_refR | p | p | j = k−1 → r[j] each step | **repeated sliding** (self-reference analogue of Recent-GT) | `:62, 79` |
| OC_refH | p | p | all r[j] | repeated, full | `:63, 79` |
| O_refH | p | r | all r[j] | repeated | `:64` |
| OC_gtR / OC_gtH | p | p | gt (sliding / all) | comparator only | `:65-66, 81-82` |

- **Applications vs actual changes** are logged per unit: `context_override_steps`, `context_override_effective_steps`, `context_override_positions` (`:172-186, 215-218`).
- **Per-unit checks** (`:204-234`): executed prefix locked; decoder origin 0; OC suffix = C1 suffix; U = reference; O1 suffix = U suffix; numpy rollout = decoder.
- **Baselines without inference** (`:221-226`): Geo = r with p composed at t_p; CM = p repeated, i.e. the body-frame local SE(2) transform repeated via the decoder, **including yaw**. This bears on P2-A.
- **Rollout-saving path (C1 input):** records store `action_idx` and `traj_xyh` rounded to 1e-6 (`:215`), produced by the real decoder `C.decode`. The pilot re-decodes the stored tokens of the manifest units and compares (`save_path` rows).

### 1.4 What the Stage-0 gate verifies for each path

See `PROTOCOL_P1.md`.
- **H7** = exp 7 / P1-C H7: Recent-GT, Full-GT.
- **H8** = exp 8 / P1-C H8: Reverse-N.
- **H9** = exp 9 / P1-C H9: windows, Full-GT.
- **P1A** = self-reference, coordinate accumulation, C1 save path.
- H32 (DM) and the P1-B start-position script use the same span machinery (`forward_spans` / identical loop). They are **not** in the Stage-0 manifest. A pass does not extend to them automatically; this is stated in the protocol.

---

## 2. SpatialVLA (SimplerEnv, Google Robot) — exp 34, exp 35, P1-D

Files:
- `cross_domain_temporal/{spatialvla_core.py (CORE), svla_rollouts.py (ROLL), svla_offline.py (OFF), action_ensemble.py}`
- `spatialvla_protection/svla_protection.py (PROT)`
- `review_followup/p1d/p1d_runner.py (P1D)`

### 2.1 Intervention arms (closed loop)
The override returns (executed token, context token) for the **step-1 translation token** g. Only token 0 of chunk step 1 is replaced, at decode position k = 0 (CORE:158-163).
- natural N: no override.
- feedback F: (p, p).
- corrected C: (p, g).
- reverse R: (g, p) (ROLL:124-128; PROT:192; P1D:305-312).
- P1D null: (g, g), through the override path. This is the existing self-patch no-op.
- P1D also has shadow-row counterfactual pairs (P1D:310-311), and feedbackD/correctedD with a post-ensemble offset (P1D:351-353).
- **Projection:**
  - No GT. The "reference" is the model's own greedy g.
  - p = `perturb_token` (ROLL:61-69): the nearest of the 4,096 translation tokens to v(g) + d·u, in normalised decoded xyz, with d = 0.3. Ties go to the lowest id (`np.argmin`).
  - Translation tokens are spherical bin centres (θ16 × φ32 × r8, adaptive Gaussian grid, clipped to [−1, 1]).
  - Physical value = 0.5(norm + 1)(q99 − q01) + q01 (CORE:178-182).
- **Frame:** fractal world_vector delta, end-effector delta control. Rotation goes euler → axis-angle; the gripper is relative and sticky (PROT:99-112).
- **Mask / positions / KV:**
  - Prefill image + prompt into a HybridCache (max P+16), copied row-wise for B rows (CORE:46-59).
  - Action tokens are fed one at a time with mask `ones(B, pos+1)` and `cache_position = [pos]` (CORE:63-69, 165-170).
  - Every planning step re-prefills; there is no KV edit across steps. The override is applied inline at k = 0, so there is no recompute. Emitted tokens are never re-decoded.
- **Decoding:** greedy argmax over the full vocabulary (CORE:156). No seeds are needed.
- **Batching:** exp 34 uses variable batches. exp 35 / P1D pad each prompt-length group to 8 rows (PROT:116-134).
- **Applications:**
  - exp 34/35: every chunk generated at t ∈ [8, 24).
  - P1D: exactly t = 8, 12, 16, 20 (assert P1D:196).
  - Changes are recoverable from the recorded exec/ctx/nat1: exp 34 had 1 no-op in 7,680 applications; exp 35 and P1D had 0.
- **Executor:** buffer of (t_gen, 4 × 7) with age < 4. Weights exp(0.8·i), newest first. Sticky gripper (PROT:78-113). A fresh Executor is built per episode.

### 2.2 Token-level (offline) arms (exp 34)
- normal, recent_ref, full_ref, win1, reverse, plus motion substitutions (OFF:27-54, 98-116).
- The reference is the greedy chunk at the same saved frame (OFF:79). Correction replaces a whole step (3 tokens).
- **reverse step-4 context = [S1p, R2, N3]**, where N3 comes from the separate Normal row (OFF:111). This mirrors AutoVLA's Reverse-N, not Impromptu's.

### 2.3 State restore (input for SpatialVLA P1-C)
- **exp 34 / 35:** `env.reset(seed=seed)` on a per-worker cached env, without reconfigure (ROLL:45-48, PROT:65).
  - The initial state depends on episode history: up to 0.26 mm / ~0.5° on move_near.
  - No state fingerprint was saved.
- **P1-D:** the root N runs from reset. At T0 = 8 it stores the float64 action prefix, a deepcopy of the executor, and rec/ever plus the hash trace (P1D:370-375). Each branch then (P1D:98-118, 244-273):
  - resets with `reset(seed, options={reconfigure: True, obj_init_options: {episode_id}})`;
  - replays the prefix;
  - compares the sha1 of `get_state()` and of the 224-px image at every step;
  - restores the executor copy.
- **P1-D results:** restore_ok 3,840/3,840; null ≡ N in 320/320 (float64 actions and state hashes over 80 steps).
- **Not covered by any existing hash:** controller targets and integrators, gripper drive targets, step counter, physics time, env RNG state, camera/lighting.
- The restore-scope table and fingerprint for the planned SpatialVLA gate are in `PROTOCOL_P1.md` §8.

## 3. Impromptu VLA 3B (spec only; no new run this round)
Files: `cross_vla_temporal/{impromptu_core.py, exp_ab.py, exp_cde.py, analyze_temporal.py}`.

- **Representation:**
  - Text waypoints "[x, y]" with Python `round(., 2)`; −0.0 is written as 0.0 (`impromptu_core.py:38-47`). Parsed with `WP_RE` (`:34`).
  - Frame: ego frame at the current pose (x forward, y left).
  - GT = future frames 1..10 of the log.
  - The executed forced w1 is unrounded, while the context copy is rounded (≤ 0.005 m mismatch).
- **Context rule** (`exp_cde.py:42-81`), at j = 2..k−1, with the forced w1 always first:
  - normal: own.
  - win_w: GT for j ≤ 1+w (persistent).
  - gt_history: all GT.
  - recent_gt: GT at j = k−1 only (sliding, repeated).
  - **reverse: own[k−1] at j = k−1, GT elsewhere. Own = the reverse row's own branch, chosen dynamically** (`:57-58`).
  - Substitutes: last slot only, RNG keyed by (token, dir, row, k) (`:59-77`).
- **KV:**
  - The prompt KV is cached once per scene.
  - At every k, each row's whole suffix is re-tokenized from text and fed after the cached prompt, with explicit `cache_position`, an all-ones mask, and implicit position ids from `rope_deltas` (`impromptu_core.py:146-224`). So invalidation = [P, end) every step.
  - Rows are grouped by suffix length, so there is no padding.
  - Exp A/B instead re-prefill with HF `generate` on left-padded rows (`:133-143`).
- **Decoding:** greedy with repetition penalty 1.05, re-implemented in the fast path. No seeds.
- **Applications vs changes:** only the substitution rows log `e_own`. win/gt/recent/reverse do not log whether GT equals own.
- **Parity data that already exists** (M-I1): `exp_cde` ref-unit `normal` vs `natural` 23/300 exact; A/B `natural_rep` vs `natural` 237/2,728.
- **What a no-op harness would need:** log token ids in `continue_batch` / `next_waypoint`; check the text round-trip; add a self-override row (own waypoint written back); run same-batch vs B = 1. This is planned only if P3 runs Impromptu.

## 4. OpenVLA (LIBERO-Spatial; spec only)
Files: `cross_vla/{openvla_core.py, openvla_rollouts.py, openvla_token_feedback.py, analyze_cross_vla.py}`.

- **Token ↔ bin:**
  - bin = clip(32000 − t − 1, 0, 254); 255 bin centres.
  - Unnormalised with q01/q99 and a mask. The gripper is binarised and inverted (`openvla_core.py:59-69, 157-163`).
  - Argmax runs over the **full vocabulary**. A non-action id is clipped to bin 254 without logging.
- **Arms** (`openvla_core.py:120-136`), with pb = clip(g + δ, 0, 254):
  - feedback: exec = context = pb.
  - corrected: exec pb, context g.
  - reverse: exec g, context pb.
  - window_w: exec pb; after dims d+1..d+w the KV is cropped to L0+d and re-fed with pb (`:90, 144-152`).
- **Phase B (closed loop):** δ ∈ {8, 24} on dim 0 at control steps 10–19.
- **Phase C (token):** dims {0, 1} × δ ∈ {±8, ±24} × arms.
- **Mask / positions / KV:**
  - The prefix is encoded once and expanded across the batch.
  - Tokens are appended one at a time with the legacy tuple cache, no attention mask and implicit positions.
  - The only KV edit is the window rewrite.
- **Decoding:** greedy, no seeds.
- **Reset:** `env.seed(0)` when the worker builds the env; `set_init_state(init_states[init])`; then 10 wait steps.
- **By construction** (M-O1): downstream reverse ≡ feedback and corrected ≡ natural within a step (19,696/19,696 rows).
- **"92.6% within-step dependency"** (`analyze_cross_vla.py:295-311`):
  - Key d0_|8|_feedback.
  - Unit = (frame, sign); n = 4,741, after 183 clipped units are dropped.
  - Event = max over dims d+1..5 of |exec_j − natural_j| ≥ Δ_eff. The gripper is excluded.
  - Pooled mean with a task-cluster bootstrap.

## 5. Cross-model "Reverse" (do not present as one treatment)

| Model / level | Context at the re-inserted position | Re-inserted value comes from | Schedule | Executed |
|---|---|---|---|---|
| AutoVLA exp 8 / P1-C H8 `reverse@emb` | GT history + latest position | **separate Normal row** (same batch), embedding ≡ token | repeated, every k ≥ t*+2 | reverse row's own samples |
| Impromptu exp 33 `reverse` | GT history + latest position | **reverse row's own** previous waypoint | repeated | own |
| SpatialVLA token level (exp 34 offline) | [S1p, R2, ·] | **separate Normal row** N3 | once (step-4 decode) | – (token-level) |
| SpatialVLA closed loop (exp 34/35, P1-D) | perturbed step-1 translation token p | the perturbation itself (exec g) | once per planning chunk in the window | g (translation only) |
| OpenVLA Phase B/C | perturbed bin pb in context | the perturbation (exec g) | once per step | g |
