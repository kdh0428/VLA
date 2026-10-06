# PAPER_NUMBERS — 논문용 수치 은행

## 읽는 법
- 모든 수치는 기존 결과 파일에서 옮겼습니다. 이 패키지에서 다시 계산한 값은 "(recomputed)"로 표시했고, 원 분석 스크립트와 같은 방법·같은 seed를 썼습니다.
- `[a, b]` = 95% CI입니다. 별도 표기가 없으면 cluster bootstrap 2,000회(seed 0)이고, cluster 단위는 각 항목에 적었습니다.
- p 검정: 이진 지표는 exact McNemar, 연속 지표는 Wilcoxon signed-rank입니다. navhard는 log 단위 paired sign-flip permutation입니다.
- 비교는 모두 같은 장면·단위·에피소드끼리의 paired 비교입니다.
- 경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`. 상세 표는 `_parts/*_sections.md`, 출처는 `SOURCE_MAP.md`, 불일치는 `CONSISTENCY_AUDIT.md`.
- **실험 35는 결과가 없어 어떤 claim에도 쓰지 않았습니다** (`EXP35_RESULTS_TEMPLATE.md`).
- 공통 모집단 (AutoVLA 기전 실험 6–11, 32): equal-distance set.
  - 208 장면: A− 52 장면(16 log), A+ 156 장면.
  - perturbation 단위 1,408개: A− 365, A+ 1,043.
  - A− 관련 CI의 유효 cluster 수는 16 log입니다.

---

# Claim 1. Token mismatch ≠ failure
**Evidence**
- natural 2,747 장면 중 A−(5 s 궤적 실패)는 52개(1.89%)입니다.
  - 이 52개 중 첫 token이 틀린 장면은 **17/52 (32.7%)**뿐입니다.
  - 반대로 첫 token이 틀린 233 장면 중 **216 (92.7%)**은 A+입니다. A+ 전체 기준으로는 216/2,695 = 8.0%입니다.
- 첫 불일치의 변위 크기: A− 중앙값 0.113 m, A+(불일치 있음) 0.083 m.
  - 이 크기로 A−와 A+를 가르는 AUROC는 0.60 [0.52, 0.68] (n = 1,159, log bootstrap 1,000회)입니다.
- A−에서 첫 불일치 거리와 최종 FDE의 상관: Spearman **+0.26 [−0.02, +0.49]** (n = 52, log bootstrap 1,000회). p는 계산되지 않았습니다.

**N / 검정:** 2,747 장면(28 log); 상관 CI는 log cluster bootstrap.
**Source:** `O/natural_fast_mechanism/summary.json` (`N.n_Aminus`, `N.n_Aminus_step0_token_wrong`, `N.n_Aplus_step0_token_wrong`, `N.codebook_amplification.A-.spearman_first_displacement_vs_fde`); `O/first_mismatch_causal/summary.json` (`why.d_pred_gt`).
**Manuscript-safe:** "Token-level mismatch and trajectory failure largely dissociate: only 17 of 52 failing scenes (33%) begin with a wrong first token, whereas 93% of scenes with a wrong first token still succeed."
**Overclaim:** "First-token errors never cause failure" / "the size of the first error is unrelated to the outcome". The correlation is weak, and its CI only barely includes 0.

# Claim 2. Error magnitude alone is insufficient
**Evidence**
- **Same-scene split:** 71.2% (37/52) [60.0, 82.8] of A− scenes have alternatives that split into recovery and amplification. These alternatives are matched in distance (perturbation median 0.129 m; matching error median 8.8 mm / 7.9%). Re-running the same token with only a new seed flips the outcome in **1.9% (1/52) [0.0, 6.5]**.
- **Within-scene rank correlation:** distance rank vs FDE rank **+0.09 [−0.01, +0.19]**.
- **FDE spread:** within-scene alternative FDE SD 2.77 m [2.37, 3.25], vs same-token reseed |ΔFDE| 0.11 m [0.01, 0.27]. The two statistics differ, so their ratio is descriptive only.
- **Variance decomposition** (no CI): 46.2% within scene, 53.8% between scenes, 17.0% between distance quintiles.
- **Predictors of amplification** (cross-validated AUROC): scene 0.80 [0.70, 0.85] vs distance 0.61 [0.49, 0.79].
- **Group contrast:** A− alternative amplification 45.0% [37, 55] vs A+ alternative amplification 4.8% [3, 7].

**N / test:** 52 A− scenes / 313 alternatives; 156 A+ scenes / 887 alternatives; log-cluster bootstrap 2,000.
**Source:** `O/equal_distance_perturbation/summary.json` (`Q1`, `Q2.A-`, `Q4`, `alt_abs_dist_diff_median_m`, `alt_rel_err_median`); the strict subset was re-run with the same script (recomputed: `_parts/recomputed/`), giving 64.7% [55.6, 72.4].
**Manuscript-safe:** "Matched error magnitude alone does not determine the outcome: in 71% of failing scenes, alternatives at the same distance (matched to within 8.8 mm) both recover and amplify, while re-sampling the same token flips the outcome in only 2%."
**Overclaim:**
- "Distance is irrelevant." The pooled Spearman is +0.31 [+0.05, +0.56], and amplification is 75% in the largest-distance quintile.
- "Perturbations of 8.8 mm." 8.8 mm is the matching error, not the perturbation size.

# Claim 3. Previous-action feedback causally amplifies deviation
**Evidence** (A−, 365 units, Experiment 7, one harness)
- **Correction:**
  - Normal amplification 47.4% [38, 57].
  - **Recent-GT** (only the last context token replaced by GT) 7.1% [4, 10]: Δ **−40.3 pp [−50.3, −30.0]**, p = 8.0e-42 (2 vs 149).
  - GT-history 4.1% [1, 8]: Δ −43.3 pp [−54.2, −32.1].
  - Recent-GT removes 40.3/43.3 = **93%** of the GT-history effect. Relative to the Normal rate the reduction is 85%. The ratio's CI is reported only for Experiment 8: 92.9% [89.3, 99.1].
- **Reverse (Experiment 8):** inserting the model's own last-token embedding into a GT history gives **33.4% [26.3, 43.2]**, +29.0 pp [+21.1, +38.0] vs GT-history (4.4%), p = 6.7e-31. This restores 68% of the effect.
- **Attention masks** (full set): −1.4 pp (p = 0.63) and −3.3 pp (p = 0.18). Neither has an effect.
- **One-token GT correction at the first mismatch** (Experiment 5, 52 A− scenes): recovery 17.3 → 73.1% (+55.8 pp [+35.4, +69.2], p = 4.2e-7), FDE −6.09 m [−7.38, −4.28].

**Source:** `O/action_history_causal/summary.json` (`subsets."all perturbations".A-`); `O/prev_action_state_patching/summary.json` (`groups.A-.all.rows."reverse@emb"`, `fraction_of_gt_history_effect.recent_gt`); `O/first_mismatch_causal/summary.json`.
**Manuscript-safe:** "Replacing only the most recent self-generated action token with ground truth reduces amplification from 47.4% to 7.1% (−40.3 pp [−50.3, −30.0]), 93% of the effect of a fully ground-truth history; re-inserting the model's own previous token restores it (+29.0 pp [+21.1, +38.0])."
**Overclaim:**
- "Removes 93% of failures." The 93% is a share of the GT-history effect.
- "Attention to history plays no role." On the outcome-selected amplified subset the masks have an effect: −16 to −19 pp.
- Mixing Normal rates across experiments (46.6–47.9%). Take numerator and denominator from the same experiment.

# Claim 4. No single amplification layer
**Evidence** (Experiment 8, 37 patch sites = embedding + 36 decoder layers)
- **Range across sites:** patching the previous-action state at any single site gives amplification **3.8–7.4%**, which is 93–101% of the GT-history effect.
- **Spread vs uncertainty:** the SD across the 37 sites is 1.16 pp, while the median per-site CI width is 6.34 pp. All 37 CIs share a common interval [4.24, 7.24]% (recomputed from the reported CIs).
- **Late layers:** layers L28, L31 and L35 give a small additional reduction of about 3 pp relative to the embedding patch (e.g. L31 −3.6 pp, p = 2.4e-4).

**Source:** `O/prev_action_state_patching/summary.json`, `figure_layer_patching.csv`.
**Manuscript-safe:** "The previous-action signal is not localized: patching any of the 37 sites (embedding plus 36 decoder layers) yields 3.8–7.4% amplification, with all confidence intervals overlapping."
**Overclaim:**
- "All layers are exactly equivalent." The late layers give a significant additional reduction of about 3 pp.
- "37 layers." Write 36 decoder layers + embedding.
- "L35 MLP dominance reproduced." The original report rates this a partial reproduction.

# Claim 5. Short 2–4 step critical window
**Evidence** (Experiment 9, A− 365 units)
- **Window recovery:** correcting 1/2/3/4 context steps recovers **47 [33, 60] / 76 [62, 87] / 90 [79, 99] / 96 [94, 100]%** of the full effect, on amplification, using Normal 46.6% and Full 4.1%. On FDE the figures are 58/82/92/97%.
- **Diminishing returns:** going from 4 steps to 5 steps adds −0.5 pp (p = 0.5).
- **Horizon controlled** (Experiment 10, t* = 0, free horizon N = 4, 116 units / 17 scenes): FDE recovery is 62 [50, 74] / 87 [80, 93] / 90 [83, 97] / 95 [90, 100]%.
- **Re-divergence after release** (only units with ≤ 1 m error at release): 25–29% (Normal 49.0%), with wide CIs (e.g. 3-step [2.1, 45.5]).

**Source:** `O/temporal_feedback_window/summary.json`, `O/horizon_controlled_window/summary.json`, `figure_temporal_window.csv`.
**Manuscript-safe:** "Correcting the 2–4 context steps that follow the first deviation recovers 76–96% of the full-history correction effect."
**Overclaim:**
- "A permanent fix." A 4-step correction's recovery decays from 101% to 95% as the free horizon grows (−6 pp [−10, −1]).
- Window percentages without naming the metric (amplification vs FDE).

# Claim 6. Motion > token identity / embedding
**Evidence** (Experiment 11, A− 365 units)
- **GT-like motion:** a different token with motion nearest to GT gives 8.5% amplification vs 7.9% for the GT token: **+0.5 pp [−1.9, +6.3], p = 0.85**. This is 99% [86, 105] of the Recent-GT effect.
- **Same motion, different identity:** a different token with motion nearest to the model's own gives 43.8% vs 47.1% (p = 0.18). It removes only 8% [−17, 24] of the effect.
- **Joint regression** (A−, n = 1,460, bootstrap 1,000): per 1 SD, motion distance adds +16.1 pp [+12.2, +21.5] amplification, embedding cosine +4.4 pp [−0.4, +8.2].

**Source:** `O/prev_action_identity_decomposition/summary.json` (`groups.A-.vs_recent_gt.geo_nn_gt`, `vs_normal.geo_nn_self`, `geometry_vs_embedding`).
**Manuscript-safe:** "What matters about the previous token is the motion it encodes: a different token with GT-like motion is as effective as the GT token (+0.5 pp [−1.9, +6.3]), while a different token with the model's own motion is not."
**Overclaim:**
- "Embedding plays no role." The embedding cosine effect on FDE is +0.50 m [+0.04, +0.97].
- "Random-embedding control." No such condition exists; the nearest controls are a random learned token and the mean embedding.

# Claim 7. Direction > magnitude
**Evidence** (Experiment 32, A− 365 units)
- **Direction-wrong vs magnitude-wrong** (correct magnitude/wrong direction vs correct direction/wrong magnitude): amplification **+9.0 pp [+3.1, +18.7], p = 6.1e-4**; FDE +1.21 m [+0.74, +2.00]. The direction-wrong substitute is *closer* to GT (0.318 vs 0.505 m).
- **Mirror at matched distance:** placing an error of the same size on the opposite side of GT gives 47.9 → 13.7% (**−34.2 pp [−44.0, −22.8], p = 4e-29**). The achieved distance is about 11% smaller, 0.458 vs 0.515 m.
- **A+ scenes:** magnitude-wrong 3.7% (p = 0.58 vs Normal), direction-wrong 8.6% (p = 1.1e-5).
- **Replications:**
  - Impromptu: +15.6 pp [+9.0, +24.3], p = 2e-31.
  - SpatialVLA (token level): +1.6 pp [+0.4, +2.8], p = 7e-5.

**Source:** `O/motion_semantics_ablation/row_contrasts.json`, `analysis.md` (achieved geometry); cross-model values in Claims 8 and 10.
**Manuscript-safe:** "Direction errors are more harmful than magnitude errors even when closer to ground truth (+9.0 pp [+3.1, +18.7])."
**Overclaim:**
- "Exactly matched distance" (the mirror condition is about 11% smaller).
- Claiming the effect is equally large across models (SpatialVLA shows +1.6 pp).

# Claim 8. Impromptu VLA: Strong replication (preregistered)
**Evidence** (commit 038a6d3; 2,728 scenes / 28 logs; C–E 2,398 units / 300 scenes)
- **Natural performance:** A− 33.4% [26.2, 38.0], FDE5 5.75 m.
- **Rerun noise:** batch-only rerun gives A− disagreement 5.9% [5.2, 6.9]; an A+ scene turns A− in 4.6%.
- **0.2 m perturbation:** A+ → A− in 49.5% [46.9, 52.5]; paired +24.8 pp [+19.4, +32.2]. 97.4% of perturbations grow ≥ 3×; absorption is 0.45%.
- **Correction:** amplification **63.4 → 1.3%**, −62.2 pp [−65.6, −59.1] (A− in the same row: 64.5 → 16.3%).
- **Reverse:** 0 → **81.9%** [80.4, 84.2], D10 +40.5 m [+37.3, +43.9].
- **Window:** 1/2/3/4 step = 24/39/91/95%.
- **Motion:** GT-near ≈ GT, p = 0.87.
- **Preregistered criteria:** 1, 2, 4, 5 and the task criterion pass; criterion 3 (stable/unstable branches) **fails**: 2.2% and 3.2% < 5.9%.

**Source:** `O/cross_vla_temporal_replication/analysis.json` (`AB.*`, `CDE.all_units.*`), RESULTS.md, PROTOCOL.md.
**Manuscript-safe:** "In Impromptu VLA, a different temporally autoregressive driving VLA, correcting only the previous waypoint in context reduces amplification from 63.4% to 1.3% (−62.2 pp [−65.6, −59.1]) and re-inserting the erroneous waypoint restores it to 81.9%."
**Overclaim:**
- "Failure drops to 1%." That is amplification; A− only falls to 16.3%.
- "All criteria replicated." Criterion 3 failed.
- Omitting the leak: GT context in absolute coordinates leaks GT information; reverse and motion substitution are the leak-free evidence.

# Claim 9. OpenVLA: local-only architectural control
**Evidence** (commit 6b9e8ad + amendment 6d0375a; LIBERO-Spatial 10 tasks × 10 = 100 episodes)
- **Within-step feedback is strong:** shifting the x token by 8 bins amplifies a downstream dimension in **92.6% [89.8, 94.4]** of 4,741 units (24 bins: 81.2%). Corrected gives 0%; reverse = feedback.
- **Task level:** feedback − corrected **+3 pp [−4, +9], p = 0.65** (8 bins), +5 pp (p = 0.36, 24 bins). Every condition vs natural has p ≥ 0.115.
- **Rerun control:** 87 vs 81% (−6 pp [−14, +2], p = 0.26), outcome disagreement **20%**.

**Source:** `O/cross_vla_replication/analysis.json` (`phaseC`, `vs_natural`, `natural_rep_vs_natural`).
**Manuscript-safe:** "OpenVLA, which does not condition on its previous actions, shows strong within-step token feedback (92.6%) but no task-level effect distinguishable from re-run variability (20% outcome disagreement)."
**Overclaim:**
- "OpenVLA is immune to feedback." Within-step feedback is strong.
- "No effect proven." 100 episodes cannot detect effects of about 10 pp.
- Reporting the official 84.7% as if it were in the repository (it appears only in the summary document).

# Claim 10. SpatialVLA: token-level replication, task-level partial
**Evidence** (commit f578eae; 18,975 token units; 80 closed-loop episodes; episode-cluster bootstrap)
- **Token level:**
  - Correction: amplification −8.3 pp [−9.3, −7.4], p = 7e-261; step-4 deviation −82%.
  - Reverse: +5.1 pp [+4.4, +5.8]; step-4 deviation restored to 0.0566, Normal level (0.0561).
  - Window: 1 step 86% (RESULTS says 87%), 2nd-last step only 63%.
- **Closed loop, context effect** (opposite direction, d = 0.3):
  - Feedback − Corrected **+1.25 pp [−12.5, +13.75], p = 1**.
  - Reverse − Natural **−1.25 pp, p = 1**.
  - The perp_left contrasts are also non-significant (p ≥ 0.29).
- **Closed loop, executed-action effect:** Corrected − Natural **−21.25 pp [−33.75, −7.5], p = 0.006**; Feedback − Reverse −18.75 pp, p = 0.004.
- **Rerun noise and trajectories:** numeric-path rerun disagreement 13.75%. Context-only perturbation moves the trajectory 0.21–0.23 m vs rerun 0.14 m. The recomputed difference is +0.09 m [+0.05, +0.14], p = 1.7e-5 (opposite); this test is an addition, not in the original analysis.

**Source:** `O/cross_domain_temporal_replication/analysis.json`, RESULTS.md; `spatialvla_closed_loop_paired.csv` (recomputed with the functions in `analyze_svla.py`).
**Manuscript-safe:** "In SpatialVLA, previous-action token feedback replicates at the token level (correction −82% step-4 deviation; reverse fully restores it), but in closed loop the context effect on task success is indistinguishable from zero (+1.25 pp [−12.5, +13.75]), while the executed perturbation itself costs −21 pp."
**Overclaim:**
- "Feedback has no effect in SpatialVLA." It does change trajectories.
- "Proven absent at task level." The CI is ±13 pp.
- Saying the ensemble weights older predictions more (the Experiment 34 protocol text is wrong; the actual weights are newest-first 57/26/12/5%).

# Claim 11. Feedback is general; task failure is architecture-dependent
**Evidence**
- **Token-level causal structure (correction reduces, reverse restores) in four models:**
  - AutoVLA −40.3 pp
  - Impromptu −62.2 pp
  - OpenVLA within-step 92.6%
  - SpatialVLA −8.3 pp
- **Open-loop trajectory or task effects appear only in the two driving models**, which keep a 10-step autoregressive action history.
- **The two closed-loop manipulation models show no context-induced task effect:** OpenVLA has no temporal history; SpatialVLA has a 4-step chunk, replans every step, and steps 2–4 make up 42.6% of the executed action.

**Source:** Claims 3 and 8–10; `paper_cross_model_table.csv`.
**Manuscript-safe:** "Previous-action feedback is a general token-level property of autoregressive VLA decoders; whether it becomes a task-level failure mechanism depends on how long self-generated action history persists in execution."
**Overclaim:**
- "Replanning and ensembling cause the absorption." That is not tested; it is the Experiment 35 hypothesis, and its results are not available.
- "General failure mechanism of all VLAs."
- Comparing amplification magnitudes across models (the definitions differ).

# Claim 12. Instability is weakly predictable before deviation
**Evidence** (Experiment 31, held-out 43,469 candidates, prevalence 0.058; log bootstrap 1,000)
- **Pre-deviation AUROC 0.570–0.634:** entropy 0.634 [0.607, 0.658]; hidden probes 0.611–0.631, no better than entropy.
- **Within-scene pre AUROC 0.504–0.649** (excluding candidate variance; no CI).
- **Selected vs discarded candidates** (Experiment 22): pre-deviation entropy differs by −0.016 [−0.023, −0.010], vs post-deviation −0.397 [−0.422, −0.371].

**Source:** `O/instability_detection_baselines/detection.json` (`heldout.*_pre`); `O/mechanism_selection_link/heldout/summary.json`.
**Manuscript-safe:** "Before the first deviation, amplifying rollouts are only weakly distinguishable (held-out AUROC 0.57–0.63, including hidden-state probes)."
**Overclaim:**
- "Pre-deviation entropy is identical." The difference is small, but its CI excludes 0.
- "Within-scene = chance." Not tested; medoid 0.649.
- "Before deviation." Pre includes the step that emits the deviating token.

# Claim 13. Instability is strongly detectable after deviation
**Evidence:** post-deviation pairwise disagreement **0.900 [0.890, 0.913]** (within-scene 0.898); medoid distance 0.883 [0.866, 0.901], AUPRC 0.403 (random baseline 0.058); log-likelihood 0.832; hidden probes 0.745–0.765.
**Source:** `O/instability_detection_baselines/detection.json` (`heldout.*_post`).
**Manuscript-safe:** "After the first deviation, candidate disagreement identifies amplifying rollouts with AUROC 0.90 [0.89, 0.91]."
**Overclaim:** "Disagreement is significantly better than all other detectors." No paired AUROC test was run.

# Claim 14. Confidence-based selection works open loop (and in non-reactive PDMS)
**Evidence**
- **Dev, 56 logs / 4,563 scenes** (rule committed before decoding in e6d9dff; no formal preregistration document):
  - Rank-sum A− 2.15 → 1.45%: **−0.70 pp [−1.07, −0.33]**, p = 3.8e-4 (−33% relative).
  - FDE 0.758 → 0.484 m.
- **Held-out preregistered, 513fdba; 52 logs / 4,814 scenes:**
  - PDMS 0.8893 → 0.8988: **+0.0095 [+0.0046, +0.0148]**, Wilcoxon p = 0.0034.
  - Open-loop A− 1.50 → 0.71%: −0.79 pp [−1.01, −0.55], p = 4.5e-7 (−53%).
- **Candidate count:** N = 8 reaches 86% of the N = 17 PDMS gain. Inference cost at N = 16: 1.31× (3080 Ti) / 1.28× (5090); N = 17 was not measured.

**Source:** `O/expanded_best_of_n/summary_pooled.json`, `O/pdm_score_best_of_n/heldout/summary.json`, `O/heldout_best_of_n/summary_pooled.json`, `figure_candidate_count.csv`.
**Manuscript-safe:** "Selecting among the model's own samples by a confidence rank-sum improves pre-registered held-out PDMS by +0.0095 [+0.0046, +0.0148] and halves open-loop failures (−0.79 pp)."
**Overclaim:**
- Presenting the −33% as preregistered (it is the dev set).
- "Works in closed loop." In navhard, rank-sum gives +0.020, p = 0.094, and its second-half replication fails at the boundary.

# Claim 15. The safety filter dominates navhard gains
**Evidence** (76 logs / 225 groups)
- **Filter only:** EPDMS 0.2287 → 0.3333, **+0.1046 [+0.0845, +0.1251]**, p < 1e-4, 128/9/88 groups.
- **F1** (filter + rank-sum): **+0.1183 [+0.0956, +0.1400]**.
  - The filter alone accounts for 0.1046/0.1183 = **88.4%** (often rounded to 89%).
  - Rank-sum's marginal contribution is +0.0137 [−0.0020, +0.0296], p = 0.20.
- **Drivable-area share:** 2-player Shapley gives drivable area **85.7%** of the filter-only gain (denominator 0.1046; 75.8% of the F1 gain).
- **Preregistered second half** (40 logs / 120 groups): F1 replicates, +0.1121 [+0.0819, +0.1402]; max log-lik does not, +0.0097 [−0.0037, +0.0235], p = 0.080.

**Source:** `O/safety_filter_ablation/ablation.json`, `O/safety_filter_components/components.json`, `O/navhard_full_validation/navhard_full_comparison.json`, `paper_navhard_table.csv`.
**Manuscript-safe:** "On navhard, a current-frame safety filter yields most of the gain (+0.105 of +0.118 EPDMS, 88%), mainly through the drivable-area constraint (86% of the filter gain by Shapley); confidence selection adds a non-significant +0.014."
**Overclaim:**
- "89%" (the exact value is 88.4%).
- "86% of the total improvement" (the denominator is the filter-only gain).
- Omitting that the filter and the score share PDM rules.
- Calling navhard "closed loop" (it is a two-stage pseudo closed loop).

# Claim 16. Candidate generation remains the bottleneck
**Evidence**
- **Oracle:** oracle17 EPDMS **0.4053**, +0.1767 [+0.1515, +0.2012].
- **Share of oracle gain achieved:** the best deployed method (filter + max log-lik) captures 0.1216/0.1767 = **68.9%**. That method was chosen after seeing the results; with preregistered F1 the share is **67.0%**. No CI.
- **Remaining headroom:** +0.0550 [+0.0355, +0.0749].
- **Tokens where all 17 candidates score 0:** **1,867/5,912 = 31.6%** (recomputed from `token_scores.npy`).

**Source:** `O/candidate_oracle/oracle_comparison.json`, `_parts/navhard_verify.json`.
**Manuscript-safe:** "Even an oracle choosing among the 17 candidates reaches only 0.405 EPDMS, and in 31.6% of tokens no candidate scores above zero, pointing to candidate generation as the remaining bottleneck."
**Overclaim:**
- "Deployable methods reach 69% of the oracle" without noting post-hoc method choice (F1 gives 67%).
- Treating the oracle as a true upper bound (it is an approximation that ignores two-stage weighting).
