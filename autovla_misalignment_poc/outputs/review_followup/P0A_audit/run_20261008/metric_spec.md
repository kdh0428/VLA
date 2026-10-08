# metric_spec — P0-A 감사 (2026-10-08)

This document describes the implementation as the code actually runs it. Every item is a 확인한 사실 with a code location, unless it is labelled otherwise. Repo HEAD is 8fd4ee2.

## A. AutoVLA (실험 4–11, 32) — 구현 명세

Repo `/root/VLA` HEAD 8fd4ee2 (branch autovla-experiments-11-19). The PoC scripts have no uncommitted changes against HEAD. AutoVLA code is in `/root/VLA/autovla`, a separate git repo at ba34eed ("checkpoint release", 2026-05-29); the agent and model code used here are clean. Codebook `autovla/codebook_cache/agent_vocab.pkl` has sha256 e6bf8eff…408e. Checkpoint `autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt` is 16,292,664,780 B (fp32 state dict). The model is built in **bf16** (`autovla/models/autovla.py:480-483`, `torch_dtype=torch.bfloat16`).

### A1. 좌표계와 GT 생성 (확인한 사실)
- **GT trajectory.** `preprocess_scenes.py:95` calls `VlaAgent.get_target_builders()[0].compute_targets(scene)` = `scene.get_future_trajectory(10)` (`autovla/navsim/navsim/agents/vla_agent.py:108-112`).
  - Frame: NAVSIM local frame at the current ego pose (origin = current ego pose; x forward, y left, heading in rad).
  - Sampling: 10 poses, 0.5 s apart, 5 s horizon. Stored as `scene["gt_trajectory"]` / `full_extract` `trajectory_gt`.
- **Trajectory → token (GT tokens).** `autovla/navsim/navsim/agents/autovla_agent.py:286-320` (`TrajectoryTargetBuilder`) → `TokenProcessor._match_agent_token` (`autovla_agent.py:88-200`).
  - Start state: (0, 0), heading 0.
  - For each GT pose i, all 2,048 codebook tokens are placed in the current state. A token's shape is the 4 corners of a 2.0 × 4.8 m box at its last sub-step (`autovla_agent.py:210-225`).
  - The chosen token is `argmin_token Σ_corners ‖token_corner − GT_box_corner‖`.
  - The state then moves to the **chosen token's** pose, not the GT pose (closed-loop tokenisation, `autovla_agent.py:146-151`). The quantisation residual therefore does not accumulate.
  - Quantisation = choice from the 2,048-token codebook. There are no separate bins.
- **Token → trajectory.** `autovla/models/action_tokenizer.py:62-110` (`decode_token_ids_to_trajectory` → `rollout`).
  - Tokens are chained from (0, 0, 0). The pose after each token = the corner mean of that token's last sub-step box.
  - The same local frame is used for prediction and GT.
- **Token motion (experiments 6 and 32).** `disp[a] = codebook[a, -1].mean(corners)` = the (forward, left) displacement of one 0.5 s segment in the token's own frame (the vehicle frame at the segment start).
  - Defined at `equal_distance_perturbation.py:147-148` and `motion_semantics_ablation.py:token_motion`.
  - For experiment 32, "direction" = atan2(dy, dx) of that segment and "magnitude" = segment length (m per 0.5 s).

### A2. 실패/증폭 판정 (확인한 사실)
- **The A label (P/R/A "A" only).** `pra_comparison/pra_labels.py:469-485` `label_A`, called through `analyze_action_history.a_eval` (`analyze_action_history.py:51-54`). P and R are not used for the A−/A+ groups.
- **Input conversion.** `run_pra_comparison._pos_to_delta(traj, 10)` (`run_pra_comparison.py:76-81`): 10 poses (x forward, y left) → per-step deltas (lateral +right, longitudinal +forward).
- **Coarse class.** `vla_misalignment_poc/analysis/trajectory_metrics.py:104-140` `action_semantics`. It is imported from `pra_labels.py:58-62`; `VLA_POC` comes first on sys.path.
  - v_start / v_end = mean speed of the first two / last two steps (DT = 0.5 s).
  - STOP: total distance < 1.0 m and v_end < 0.5.
  - Below 0.5 m/s, absolute change > 0.8 m/s.
  - Otherwise by ratio: ≥ 1.25 ACCELERATE, ≤ 0.75 DECELERATE/STOP.
  - LEFT/RIGHT: |final lateral| > 2.0 m or final-step heading > 15°.
- **Acceptance.**
  - Longitudinal: the predicted class (GT v_start, predicted v_end, predicted distance) must be in the GT class set obtained by shifting GT v_end by ±max(0.5, 0.15·max(v)) on an 11-point grid.
  - Lateral: the predicted class must be in the GT class set obtained by shifting final lateral by ±1.0 m and heading by ±10° on a 5 × 5 grid.
  - A+ = longitudinal ∧ lateral; A− = not A+.
- **Note (확인한 사실).** The trajectory_metrics thresholds are documented for HORIZON = 6 steps (3 s) (`trajectory_metrics.py:21-24`), but AutoVLA applies them to 10 steps (5 s): v_end = mean of steps 9–10, and the STOP threshold of 1 m over 5 s.
- **Amplification** (`analyze_action_history.py:45,57-67`): `amplification = (not A+) and FDE5 > 3.0 m`.
  - FDE5 = ‖pred_xy[9] − gt_xy[9]‖ in the local frame.
  - `recovery` = A+.
  - ADE5 = mean L2 over the 10 poses.
  - `downstream_err` = share of steps after t* whose token ≠ GT token.
- The same function is reused by experiments 8, 9, 11 and 32 (`from analyze_action_history import unit_metrics`) and by Impromptu (experiment 33).
- **OOD rule (condition level, not unit level).** A condition is excluded from verdicts when its mean A− or A+ Δfirst-step entropy vs Normal is > 1 nat, or when it is one of hist/recent_emb_neutral (`analyze_action_history.py:236-243`). No unit-level exclusion exists, so the analysed denominator is 365/1,043 in every condition.

### A3. 표본 정의
- **A− / A+ groups** come from the stored `full_extract` arm-N run (`equal_distance_perturbation.py:151-175`, `first_mismatch_causal.py:80-99`).
  - Exclusions: CoT present, runaway (>10 action tokens), <10 poses.
  - t* = first k with pred[k] ≠ GT[k]; no mismatch → excluded.
- **Experiment 5** = every scene with a mismatch (1,159). **Experiment 6** = all A− (52) + 3 A+ per A− at the same t* (`random.Random(0)` shuffle, 156).
- **Units** = (scene, forced token at t*): `original` + distance-matched `alt*` (5–7 per scene).
- `original_reseed` and `gt` are experiment 6 only and are **not** units of experiments 7–11 or 32 (`action_history_causal.py:198`).
- The flow is in `denominator_flow.csv`.

### A4. 개입 의사코드 (모두 확인한 사실, 코드 위치 명시)
Common harness (experiments 7, 8, 9, 11, 32): `action_history_causal.py:184-239`, `prev_action_state_patching.py:244-330`, `temporal_feedback_window.py:150-200`, `motion_semantics_ablation.py:170-250`.
```
prefix = prompt + stored fast stub (full_extract arm N) + pred[:t*]          # shared, identical for all conditions
cache  = KV(prefix)  (computed once per scene; batched scripts: expanded to B rows)
for each unit (forced token f at t*):
  own[c] = []                                                                # per-condition outputs
  for k in t*+1 .. 9:
     ctx_c = CONTEXT_RULE_c(own[c], gt, k)                                   # tokens for positions t*+1..k-1
     span  = [f] + ctx_c ; logits = LLM(span | cache) ; cache.crop(P)        # span re-encoded every step -> no stale KV
     [exp 8 only: hidden state of span[-1] at layer L replaced in row r by row s's state]
     own[c].append( multinomial(softmax(logits[action rows]/0.01), seed=sha256(f"{seed}:{token}:{pert}:{k}")) )
  executed[c] = pred[:t*] + [f] + own[c]   -> decode -> trajectory -> metrics   # GT never enters the executed output
```
CONTEXT_RULE by condition:

| 실험 / 조건 | 문맥 위치 j (t*+1 ≤ j ≤ k−1) | 폭 | 적용 횟수 / unit | 출처 branch |
|---|---|---|---|---|
| 7 normal | own[j] | 0 | 0 | – |
| 7 **recent_gt** (`action_history_causal.py:210-212`) | j = k−1 → gt[j]; j < k−1 → own_recent[j] (the row's **own** outputs, generated under the corrected context) | 1 token per forward (sliding) | at every step k ≥ t*+2 → **8−t* times** (A− mean 6.8, of which the token actually changed 4.32 times on average; computed result) | GT |
| 7 gt_history (`:208-209`) | all gt[j] | k−1−t* (grows) | every step | GT |
| 7 hist/recent_attn_mask (`:130-152`) | own; only the attention mask changes (span tokens before the query masked / only the immediately preceding one masked) | – | every step | – |
| 7 hist/recent_emb_neutral | own; input embedding replaced by the mean action embedding ([OOD]) | – | every step | – |
| 8 patch_full@L (`prev_action_state_patching.py:84-101,174-189`) | normal context; hidden of the **last span position (= position k−1)** at L ← gt_history row (same batch) | 1 position, 1 layer | every step k ≥ t*+2 (no-op at k = t*+1) | gt_history row |
| 8 **reverse@L** (`:99-100`) | gt_history context; hidden of position k−1 at L ← **normal row** (same batch, a separate Normal branch) | 1 position, 1 layer | every step k ≥ t*+2 | **separate Normal branch** (not an error the reverse branch produced itself) |
| 8 identitysrc@L | normal context; position k−1 ← recent_gt row | 1 position | every step | recent_gt branch |
| 9 **win_w** (`temporal_feedback_window.py:180-191`) | j − t* ∈ [1, w] → gt[j] (**fixed absolute positions**, kept in the context at every later step); others own | w fixed positions | the GT stays in the context for every step after its position is filled (not a single correction) | GT |
| 9 delay{s}_len{l} / skipfirst{s}_full | j − t* ∈ [s, s+l) | l | same | GT |
| 32 recent_gt (`motion_semantics_ablation.py:211-214`) | identical to 7 recent_gt | 1 sliding | 8−t* | GT |
| 32 dir_ok_mag_wrong / dir_wrong_mag_ok / mirror / random (`:56-75,211-222`) | j = k−1 → substitute s = f(g = gt[k−1], o = own[k−1]); j < k−1 → own | 1 sliding | 8−t* (logged per step in `substitutions`) | built from GT and the row's own token |
| 5 gt (`first_mismatch_causal.py:150-170`) | **one-time output-level** forcing of the GT token at position t* (enters both the executed output and the context), then free HF `generate` | 1 token, once | 1 | GT |
| 6 original/alt (`equal_distance_perturbation.py:234-260`) | forced token at t*, then free HF `generate` (whole vocabulary, top_k = 0, top_p = 1, T = 0.01; leaving the action span → invalid) | 1 token, once | 1 | – |

**Output / context / KV separation (확인한 사실)**
- In experiments 7, 8, 9 and 32, the executed output is always `pred[:t*] + [f] + own` (`action_history_causal.py:236`, `prev_action_state_patching.py:324`, `motion_semantics_ablation.py:236`). GT and substitute tokens enter **only the context**.
- The prefix KV (prompt + stub + pred[:t*]) is not changed; the scripts check this with a checksum assert, e.g. `prev_action_state_patching.py:316-318`.
- The post-t* span is recomputed every step, so no K/V of a replaced token remains.
- Experiment 8 debug (summary.json `sanity.debug`, 4 samples):
  - K/V at or before the patch layer: difference 0.
  - K/V after the patch layer: min difference 3.8 (K) / 17.0 (V).
  - Patching changes no other positions or other rows (0.0).
  - patch@emb ≡ recent_gt (logits L1 = 0).
  - patch@L35 ≡ gt_history (L1 = 0).
- **Experiments 5 and 6 are output-level interventions** (the forced token is part of the executed trajectory). They differ from the context-only corrections of experiments 7–9 and 32.

### A5. Normal 재현과 수치 경로 (계산한 결과 + summary)
| 비교 | Normal 10-token 완전 일치율 | 출처 |
|---|---|---|
| exp 7 (B = 1, own loop over the 2,048 action rows) vs exp 6 stored (HF generate) | 71.6% (A− 64.7%, A+ 74.0%) | raw (recomputed) = summary |
| exp 8 (B = 93) vs exp 7 records at the time of the run | 88.6% (gt_history 92.9%) | summary only (exp 8 records absent) |
| exp 9 (B = 17) vs exp 7 | 89.2% (gt_history 94.0%) | summary only |
| exp 32 (B = 6) vs exp 7 raw | 88.2% (A− 85.5%; recent_gt 89.0%) | raw |

A− Normal amplification by experiment: 47.4 (7) / 46.8 (8) / 46.6 (9) / 47.1 (11) / 47.9% (32).
- In exp 32 vs exp 7, the A− Normal amplification verdict differs in 8 of 365 units (computed result).
- The Normal row differs between experiments only because of batch composition and the numeric path (bf16, batch size), not the intervention.
- So the share of effect in each experiment must be computed from that experiment's own Normal row.

## B. SpatialVLA (실험 34, 35)
Code: `autovla_misalignment_poc/scripts/cross_domain_temporal/*` (pre-registration f578eae, results f4cf886) and `spatialvla_protection/*` (pre-registration 8fc20bf, results 2b0212d). Sub-audit details are in `scripts/svla_subaudit/`.
- **Reference (experiment 34, token level).** The model's own unperturbed greedy chunk; it is **not GT** (`svla_offline.py:79`).
  - Correction replaces one whole step: translation, rotation and gripper, 3 tokens.
  - The step-1 perturbation itself is never corrected.
  - The recent span is 1 step slot per decode, applied twice (at the step-3 and step-4 decodes).
- **Reverse, token level** (`svla_offline.py:111`). The step-4 context is [S1p, R2 (reference), N3 (step 3 of the **separate Normal row**)]. The re-inserted error comes from the Normal branch, not from the reverse branch.
- **Reverse, closed loop** (`svla_rollouts.py:128`, `svla_protection.py:192`). The executed action is the natural token g, while the context holds the perturbed p. This is a different intervention from the token-level reverse.
- **Output / context / KV.** Only the context token goes into the cache, and the prompt is prefilled again at every decode, so no stale KV remains.
  - Executed action = output of `ActionEnsembler` (weights exp(0.8·i), newest first: 0.574 / 0.258 / 0.116 / 0.052), then sticky gripper (`action_ensemble.py:21-26`).
- **Token-level metrics** (normalised translation space [−1, 1], not metres).
  - inj = ‖v1 − r1‖; A = ‖Σ_{k=2..4}(v_k − r_k)‖ / inj; amplified means A ≥ 1.
  - dev4 = ‖v4 − r4‖; recovered means the translation ids of steps 2–4 all equal the reference.
  - Unit exclusion: p == g, or achieved distance outside [0.5d, 1.5d]. This drops 225 of 19,200 units, leaving 18,975.
- **Experiment 34, closed loop.** Success = the `done` flag at step 80.
  - **"Trajectory divergence (m)" = distance between cumulative sums of the commanded `world_vector`** (`analyze_svla.py:68-69,85-87`). It is not the actual TCP position; experiment 34 did not save the TCP.
  - 계산한 결과 (experiment 35 data, where both exist): the commanded sum runs about 0.84–0.95 m while the real TCP moves 0.21–0.23 m. For r1e4 reverse vs natural, commanded divergence is 0.24 m vs real final TCP difference 0.069 m (r = 0.77).
- **Experiment 35 position error** (`analyze_protection.py`). At step t, the distance between the real TCP positions (`tcp.pose`, recorded after `env.step`) of two paired conditions with the same configuration, task and seed. It is not a distance from GT.
  - Pairs: feedback vs corrected, reverse vs natural, and each condition vs natural.
  - mean_err = mean over steps 8–79; final_err = step 79.
  - Realigned = first t ≥ 24 where the error stays ≤ 2 cm for 4 consecutive steps.
  - Statistics: task-stratified paired bootstrap, McNemar, Wilcoxon, sign-flip, and Holm over **5** configurations (`analyze_protection.py:19`; Claim 17 says 4).
  - Its "trajectory divergence" (+0.116 m) is again the commanded-action proxy.
- **Exposure values** (0.426 … 0.75): no committed code computes them. Recomputed from recorded chunk ages: 0.426 / 0.31 / 0 / 0.507 / 0.660 / 0.753.

## C. Impromptu VLA (실험 33)
Code: `scripts/cross_vla_temporal/*` (pre-registration 038a6d3, analysis 698148a). Details are in `scripts/navsim_subaudit/report_navsim_impromptu.md`.
- **GT.**
  - Pose = ego2global translation + yaw from the navsim log, expressed in the current ego frame (x forward, y left).
  - Future = frames i+1..i+10; spacing 0.499–0.501 s.
  - Matches AutoVLA `trajectory_gt` within 0.0005 m (계산한 결과).
- **Context quantisation.** Text rounded to 2 decimals, absolute positions (`impromptu_core.py:38-47`). No codebook.
- **Parsing.**
  - `WP_RE` (`impromptu_core.py:34`) takes the first 10 `[x, y]` matches.
  - Natural decode with fewer than 10 waypoints → the scene goes to `errors.jsonl` (0 scenes).
  - A/B rows: a scene is excluded if any of its rows is not ok, leaving 2,728 of 2,748.
  - C/D/E rows: a parse failure is first filled with the previous waypoint (`exp_cde.py:201-204`), then the unit is excluded in every row (`analyze_temporal.py:142`). 2 events, leaving 2,398 of 2,400.
- **Interventions.**
  - recent_gt: only j = k−1, sliding, 8 applications.
  - win_w: fixed absolute span, kept in context.
  - **reverse: GT context plus the reverse branch's own k−1 waypoint.** The error comes from the reverse branch itself, unlike AutoVLA and SpatialVLA.
  - Execution stays [forced w1 + own w2..w10]: context-only.
  - KV: the prompt cache is copied at every step and the full suffix is fed again.
- **Metrics.** The same functions as AutoVLA (`a_eval`, AMP_FDE = 3 m).

## D. NAVSIM evaluator
- **navhard EPDMS** (experiments 25–27, 29, 30; `paper_navhard_table.csv`, Claims 15–16).
  - Evaluator: `/root/VLA/navsim_v2`, upstream autonomousvision/navsim at HEAD 0a380a9 (2025-10-27, v2.2-1); tracked files unmodified.
  - The 2025-09-29 human-filter fix (upstream 359c7f7, Issue #151) is an ancestor of HEAD, and the fixed code is at `navsim/evaluate/pdm_score.py:171-218`.
  - All 75 scoring runs logged `human_penalty_filter: true` and ran the navsim_v2 script. The only part not logged directly is the import path (추론).
- **Held-out PDMS** (experiments 21–24, Claim 14). Evaluator: the NAVSIM v1.1.0 copy vendored inside AutoVLA (`/root/VLA/autovla/navsim`, AutoVLA ba34eed). It has no human filter, so the fix does not apply. Its upstream base commit is 미확인.
- Neither evaluator commit is recorded anywhere in the package.
