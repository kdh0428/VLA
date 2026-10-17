# PAPER_NUMBERS — 논문용 수치 은행 (2026-10-13 개정; 2026-10-15 review_followup3 개정)

## 읽는 법
- 모든 수치는 기존 결과 파일에서 옮겼습니다. 이 패키지에서 다시 계산한 값은 "(recomputed)"로 표시했고, 원 분석 스크립트와 같은 방법·같은 seed를 썼습니다.
- `[a, b]` = 95% CI입니다. 별도 표기가 없으면 cluster bootstrap 2,000회(seed 0)이고, cluster 단위는 각 항목에 적었습니다.
- p 검정: 이진 지표는 exact McNemar, 연속 지표는 Wilcoxon signed-rank입니다. navhard는 log 단위 paired sign-flip permutation입니다.
- 비교는 모두 같은 장면·단위·에피소드끼리의 paired 비교입니다.
- 경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`. 상세 표는 `_parts/*_sections.md`, 출처는 `SOURCE_MAP.md`, 불일치는 `CONSISTENCY_AUDIT.md`.
- 실험 35(2b0212d) 결과는 Claim 11과 Claim 17에 반영했습니다. 상세는 `EXP35_RESULTS.md`에 있습니다.
- 공통 모집단 (AutoVLA 기전 실험 6–11, 32): equal-distance set.
  - 208 장면: A− 52 장면(16 log, **13 drive**), A+ 156 장면(26 log, 19 drive).
  - perturbation 단위 1,408개: A− 365, A+ 1,043.
  - A− 관련 독립 cluster 수는 16 log가 아니라 **13 drive**입니다(2026-10-13 개정).

## 2026-10-13 개정 (review_followup2 반영)
- 근거: `O/review_followup2/CLAIM_EVIDENCE_MATRIX.md`(유지·축소·삭제 판정, 허용 문장), `CLUSTER_REANALYSIS.md`, `DIRECT_EFFECTS.md`, `BASELINE_WINDOW_RECHECK.md`, `DATA_AUDIT.md`. 모두 기존 자료의 **사후** 재분석입니다.
- **p값 교체:** AutoVLA·Impromptu 기전 대비의 단위 수준 McNemar p(8e-42, 6.7e-31, 2e-31, "0", SpatialVLA 7e-261 등)는 쓰지 않습니다. drive 합에 대한 exact sign-flip p를 쓰고 하한을 함께 적습니다: 하한 2/2^G는 정보 있는(합 ≠ 0) drive 수 G로 정해집니다: 13 drive 2.4e-4, 18 drive 7.6e-6, 20 drive 1.9e-6. sign-flip은 drive 합이 H0에서 독립·0에 대칭이라는 가정 아래의 조건부 검정입니다. SpatialVLA closed loop은 과제별 초기 상태 cluster(coke = seed, move_near = primary 상태 또는 물리 배치 10개) 기준입니다.
- **개정한 claim:** 3(반복 Recent-GT, 분모, drive p, 새 drive 재현), 5(3 step ≈ 90% [81, 99]; early window 삭제), 6(GT 근접 token 동등성 미성립), 7(새 drive 미결론), 10(실행 명령 누적 vs TCP, 초기 상태 수), 11(모델별 Reverse, −6.2%p 삭제), 17(ensemble 보호 삭제, P1-D H1/H2). **추가한 claim:** 18(SpatialVLA 직접 효과), 19(GT 없는 문맥 효과), 20(외삽 초과는 실패 층에서만), 21(held-out은 drive 비분리).
- 원 수치를 남긴 claim(1, 2, 4, 8, 9, 12–16)의 단위 수준 p는 위 규칙으로 읽습니다. 원고에 쓸 때는 각 claim의 Manuscript-safe 문장과 아래 개정 메모를 따릅니다.

## 2026-10-15 개정 (review_followup3 반영)

**우선순위.** 이 문서와 `numbers.json`(2,164행, `autovla_misalignment_poc/scripts/review_followup3/cp2/cp2_numbers.py`가 생성)이 충돌하면 `numbers.json`이 우선합니다. 각 행에는 row ID, 값, 95% CI, p, status, 출처 파일과 key가 있습니다.

**검증 상태.**
- AutoVLA 여섯 경로가 no-op/cache 관문을 통과했습니다: H7, H8, H9, P1A, H32, P1B. 해당 수치의 status는 `VERIFIED_MACHINERY_MANIFEST_SCOPE`입니다.
- SpatialVLA 복원 감사도 PASS입니다(분리 fingerprint C2 20/20).
- 검증 manifest 밖의 원 run은 소급 검증되지 않았습니다. 원고에는 이 범위 문장을 함께 씁니다.
- 실험 35와 OpenVLA는 `NOT_VERIFIED`이며 기술적 수준으로만 씁니다.

**이번에 수정한 것**(`cp2_doc_fixes.py`가 원 값과 대조):
- 반올림: +1.2 → +1.25, −6.2 → −6.25(half-even 경계값은 두 자리로 표기).
- +4.37 m 행의 Holm 표기 혼동(Claim 19).
- OpenVLA 92.6%와 +3 [−4, +9]가 task 군집 CI라는 표시(Claim 9).
- 18–29%p를 r4e1로 한정(Claim 18).
- 운동 baseline 서술 갱신(Claim 20).

**새 claim과 표.**
- Claim 22: 평가 모집단 추정량과 A− 빈도.
- Claim 23: 운동 외삽 대조.
- Claim 24: 개입 machinery 검증 범위.
- `tables.tex` Table 8–11은 `numbers.json`에서 기계 생성했습니다(`cp2_tables.py`).
- 최종 판정: `review_followup3/CLAIM_EVIDENCE_MATRIX.md`.

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

# Claim 3. Previous-action context causally changes later generation (revised 2026-10-13)
**Evidence** (A−, 365 units / 52 scenes / 16 logs / **13 drives**; Experiment 7 for correction, R8 recovery run of Experiment 8 for reverse; scene-weighted estimand, drive-cluster bootstrap CI, exact drive-level sign-flip p, Holm over the AutoVLA primary family m = 3; post-hoc re-analysis of exploratory experiments)
- **Correction:**
  - Normal amplification 47.4% → **Recent-GT** 7.1% → GT-history 4.1% (unit-weighted rates, Experiment 7).
  - Recent-GT is a **repeated (sliding) context intervention**: at every generation step k ≥ t*+2 the most recent context position k−1 is replaced by GT (8 − t* applications per unit, A− mean 6.8; the token actually changes 4.3 times on average). Executed tokens are always the model's own. It is not a single one-token correction.
  - Recent − Normal **−39.8 pp [−49.0, −29.9]**, p = 2.4e-4 (the minimum attainable with 13 drives; all 13 drives have the same sign), Holm 7.3e-4. Sign-flip-inverted 95% CI [−51.5, −28.7]. Unit-weighted estimate −40.3 pp.
  - Share of the GT-history effect: **92.9% [89.4, 99.1]** (R8, unit-weighted; scene-weighted 93.6% [90.3, 99.8]). Denominator = GT-history − Normal in the same run; relative to the Normal rate the reduction is 85%.
- **Reverse (Normal-branch token injection):** in a GT-history context, the embedding at position k−1 is copied at every step from the **separate Normal row in the same batch**. Reverse − Full **+28.6 pp [+20.9, +36.1]**, p = 2.4e-4, Holm 7.3e-4 (R8 recovery run; 220/220 rates reproduced exactly — a reproducibility audit, not an independent confirmation).
- **New drives (P1-C, preregistered b212637 + amendments b216920, 9b3d4af):** 64 logs / 24 drives with **0 drives shared** with the mechanism set (10 day-vehicles shared). Failure-conditional stratum F (18 drives): RN −34.2 pp [−38.3, −29.5], RV +26.8 pp [+21.9, +30.7]; population estimand P (24 drives, Hajek-weighted): RN −4.1 pp [−6.4, −2.4], RV +3.6 pp [+2.2, +5.5]; all "replicated" under the preregistered drive-level exact sign-flip + Holm rule.
- **Leave-one-drive-out:** range −43.5 to −36.2 (RN) and +26.4 to +31.4 (RV); max p 4.9e-4; no sign change.
- **Attention masks** (Experiment 7): −1.4 pp and −3.3 pp, not detected (unit-level p 0.63 / 0.18; not re-tested at drive level; equivalence not tested).
- **One-token GT correction at the first mismatch** (Experiment 5, output-level, the only true single correction): recovery 17.3 → 73.1% (+55.8 pp [+35.4, +69.2]). Per-unit raw is missing; no cluster-level p is available.

**Source:** `O/review_followup2/A_autovla_cluster/run_20261013/CLUSTER_REANALYSIS.csv` (rows AV-P1, AV-P2), `O/review_followup2/C_baseline_window/run_20261013/c2_recovery_fractions.csv` (R8 recent_gt), `O/review_followup/P1C_new_log_replication/run_20261008/analysis/decisions.json`; original: `O/action_history_causal/summary.json`, `O/prev_action_state_patching/summary.json`.
**Manuscript-safe:** "Repeatedly replacing the most recent action context with ground truth reduced amplification by 40 pp (13 drives; replicated on 24 new drives), and re-inserting the model's own previous action restored it." / "Last-position correction applied at every step recovered 93% of the full-history correction effect."
**Overclaim:**
- "Replacing a single token once removes 93% of failures." Recent-GT is repeated at every step, and 93% is a share of the GT-history effect.
- Unit-level p values (8e-42, 6.7e-31). Report the drive-level p with its floor (2.4e-4 for 13 drives).
- "Replicated on held-out data" without saying "new drives, day-vehicles partly shared".
- "Attention to history plays no role." The masks show no detected effect on the full set; on the outcome-selected amplified subset they have −16 to −19 pp.
- Mixing Normal rates across experiments (46.6–47.9%). Take numerator and denominator from the same run.

# Claim 4. No single amplification layer
**Evidence** (Experiment 8, 37 patch sites = embedding + 36 decoder layers)
- **Range across sites:** patching the previous-action state at any single site gives amplification **3.8–7.4%**, which is 93–101% of the GT-history effect.
- **Spread vs uncertainty:** the SD across the 37 sites is 1.16 pp, while the median per-site CI width is 6.34 pp. All 37 CIs share a common interval [4.24, 7.24]% (recomputed from the reported CIs).
- **Late layers:** layers L28, L31 and L35 give a small additional reduction of about 3 pp relative to the embedding patch (e.g. L31 −3.6 pp, p = 2.4e-4).

**Source:** `O/prev_action_state_patching/summary.json`, `figure_layer_patching.csv`.
**Manuscript-safe:** "The previous-action signal is not localized: patching any of the 37 sites (embedding plus 36 decoder layers) yields 3.8–7.4% amplification, with all confidence intervals overlapping."
**2026-10-13 note:** Experiment 8's main-run per-unit raw is missing; the site range is summary-level. The R8 recovery run reproduced the reverse/GT-history rows (Claim 3). Per-site p values (e.g. L31 p = 2.4e-4) are unit-level.
**Overclaim:**
- "All layers are exactly equivalent." The late layers give a significant additional reduction of about 3 pp.
- "37 layers." Write 36 decoder layers + embedding.
- "L35 MLP dominance reproduced." The original report rates this a partial reproduction.

# Claim 5. Correction length matters; no evidence for a special early window (revised 2026-10-13)
**Evidence** (A−, 13 drives; R9 recovery run of Experiment 9 and P1-B preregistered; drive-cluster CIs)
- **Length (R9, persistent windows at fixed positions t*+1…t*+w):** recovery fraction f = (Normal − window)/(Normal − GT-history), same run and units: **1/2/3/4 steps = 46.7 [29.6, 62.1] / 76.6 [62.7, 88.0] / 90.1 [81.3, 99.0] / 95.8 [93.3, 100]%**. Bootstrap share P(f ≥ 0.9): 0.00 / 0.01 / 0.56 / 1.00. FDE fractions 57.5 / 82.1 / 92.5 / 96.5%.
  - The post-release free horizon shrinks with w (5.8 → 2.9 poses), so the length effect is partly a post-release-horizon effect.
- **Position (P1-B, preregistered, A− with t* ≤ 2: 326 units / 12 drives):** early (s = 1) − late (s = 3) at equal window length, total horizon, binary amplification: w1 **+10.7 [−0.7, +20.6]**, w2 **+4.3 [−2.6, +9.7]**, w3 **+0.5 [−4.5, +3.1]** pp; sign-flip p 0.16 / 0.30 / 0.84 (Holm 0.49 / 0.59 / 0.84). ±10 pp equivalence (predeclared margin): w3 established (Holm 0.002); w2 only unadjusted (p_equiv 0.032, Holm 0.064); w1 not established.
  - Late windows have 2 fewer free poses after release. At a common post-release horizon (post-hoc), the sign of the continuous early − late contrast depends on the handling of the absolute-pose confound (raw error and same-pose recovery favour early; difference-in-differences favours late).
- **Horizon-controlled extension (Experiment 10, summary only, no per-unit raw):** FDE recovery 62 / 87 / 90 / 95% at free horizon N = 4; re-divergence after release 25–29%.

**Source:** `O/review_followup2/C_baseline_window/run_20261013/c2_recovery_fractions.csv`, `c2_position_length.csv`, `BASELINE_WINDOW_RECHECK.md` §C2; `O/horizon_controlled_window/summary.json`.
**Manuscript-safe:** "Correcting three consecutive context steps recovered about 90% of the full-history effect (95% CI 81–99%)." / "We found no evidence that corrections at the first deviating steps are more effective than equally long later corrections."
**Overclaim (delete):**
- "Correcting 2–4 steps recovers 90%" / "a short 2–4 step critical window". Two steps give 77% (CI upper bound 88%); only four steps have the whole CI above 90%.
- "The first steps after a deviation are special." Not supported (P1-B).
- "Position does not matter." Only w = 3 equivalence is established, at the predeclared binary endpoint and total horizon; do not generalize it to w = 1, other starts, A+, or other horizons.
- "A permanent fix." Recovery decays after release (Experiment 10, summary-level).

# Claim 6. Motion matters more than embedding proximity; GT-like motion ≈ GT is not established (revised 2026-10-13)
**Evidence** (Experiment 11, A− 365 units / 13 drives; drive-cluster re-analysis, post-hoc)
- **GT-like motion:** a different token whose motion is nearest to GT vs the GT token: **+0.8 pp [−1.6, +6.6]** (scene-weighted, drive bootstrap; 90% CI [−1.4, +5.2]), sign-flip p = 0.73. **±5 pp equivalence is not established:** TOST p_lower 1.2e-4, p_upper 0.090 → p_equiv 0.090 (bootstrap TOST 0.057; unit-weighted sign-flip TOST 0.087). Drive-equal weighting: +4.7 pp, 90% CI [+0.2, +10.4], i.e. the GT-like token may be slightly worse than GT.
- **Same motion, different identity:** a different token with motion nearest to the model's own gives 43.8% vs 47.1% (p = 0.18, unit-level). It removes only 8% [−17, 24] of the effect.
- **Joint regression** (A−, n = 1,460, bootstrap 1,000): per 1 SD, motion distance +16.1 pp [+12.2, +21.5] amplification, embedding cosine +4.4 pp [−0.4, +8.2].

**Source:** `O/review_followup2/A_autovla_cluster/run_20261013/CLUSTER_REANALYSIS.csv` (row "AV-S GeoNN(GT)-RecentGT [equivalence]"); `O/prev_action_identity_decomposition/summary.json`.
**Manuscript-safe:** "A token with GT-like motion produced a small, non-significant difference from the GT token (equivalence not established), whereas a different token with the model's own motion did not reduce amplification."
**Overclaim:**
- "A token with GT-like motion is as effective as the GT token" / "token identity is irrelevant" / "invariant to identity." Equivalence is not established.
- "Embedding plays no role." The embedding cosine effect on FDE is +0.50 m [+0.04, +0.97].
- "Random-embedding control." No such condition exists.

# Claim 7. Direction > magnitude in the development drives; inconclusive on new drives (revised 2026-10-13)
**Evidence**
- **Development set (Experiment 32, A− 365 units / 13 drives; protocol af2eb10; drive re-analysis post-hoc):** direction-wrong − magnitude-wrong amplification **+9.9 pp [+3.9, +21.0]** (scene-weighted), exact drive sign-flip **p = 0.0056**, Holm 0.0056 (sign-flip-inverted CI [+3.0, +24.3]). Weighting-sensitive: unit-weighted +9.0 pp (p = 0.012), drive-equal +15.5 pp [+3.3, +27.0] (p = 0.028). LODO range +8.1 to +13.3, max p 0.011. FDE +1.22 m [+0.72, +2.14]. The direction-wrong substitute is *closer* to GT (0.318 vs 0.505 m).
- **New drives (P1-C, preregistered):** F stratum (18 drives) **+6.2 pp [+0.6, +12.1], sign-flip p = 0.073** (sign-flip CI [−0.4, +12.7]); P (24 drives) +2.6 pp [−0.4, +5.5], p = 0.10. Preregistered judgment: **inconclusive** in both families (design-stage power 0.19–0.33). The bootstrap CI excluding 0 does not override the preregistered sign-flip rule.
- **Mirror at matched distance:** 47.9 → 13.7% (achieved distance about 11% smaller).
- **Other models:** Impromptu +15.7 pp [+9.1, +24.7] (20 drives, p = 8.2e-5; waypoint-level substitution); SpatialVLA token level +1.6 pp [+0.4, +2.8] (episode clusters; not re-clustered by initial state).

**Source:** `O/review_followup2/A_autovla_cluster/run_20261013/CLUSTER_REANALYSIS.csv` (AV-P3, P1C DM F/P, IMP Direction−Magnitude); `O/motion_semantics_ablation/row_contrasts.json`.
**Manuscript-safe:** "Direction errors were more harmful than magnitude errors in the development drives; on new drives the effect was in the same direction but inconclusive."
**Overclaim:**
- "Direction > magnitude generalizes" or "replicated on new logs."
- p = 6.1e-4 or 2e-31 (unit-level).
- "Exactly matched distance" (the mirror condition is about 11% smaller).
- Claiming the effect is equally large across models.

# Claim 8. Impromptu VLA: Strong replication (preregistered)
**Evidence** (commit 038a6d3; 2,728 scenes / 28 logs; C–E 2,398 units / 300 scenes)
- **Natural performance:** A− 33.4% [26.2, 38.0], FDE5 5.75 m.
- **Rerun noise:** batch-only rerun gives A− disagreement 5.9% [5.2, 6.9]; an A+ scene turns A− in 4.6%.
- **0.2 m perturbation:** A+ → A− in 49.5% [46.9, 52.5]; paired +24.8 pp [+19.4, +32.2]. 97.4% of perturbations grow ≥ 3×; absorption is 0.45%.
- **Correction:** amplification **63.4 → 1.3%**, −62.2 pp [−65.6, −59.1] (A− in the same row: 64.5 → 16.3%).
- **Reverse:** 0 → **81.9%** [80.4, 84.2], D10 +40.5 m [+37.3, +43.9].
- **Window:** 1/2/3/4 step = 24/39/91/95%.
- **Motion:** GT-near vs GT difference not significant, p = 0.87 (not an equivalence test).
- **Drive-level re-analysis (2026-10-13, 20 drives):** recent-GT −62.2 pp [−65.4, −59.1], reverse +81.9 pp [+80.5, +83.9], GT-history −63.4 pp, all exact sign-flip p = 1.9e-6 (floor), Holm 7.6e-6; direction − magnitude +15.7 pp [+9.1, +24.7], p = 8.2e-5. The Impromptu recent-GT is also repeated at every waypoint, and its reverse re-inserts the branch's own error (a different treatment from AutoVLA's).
- **Preregistered criteria:** 1, 2, 4, 5 and the task criterion pass; criterion 3 (stable/unstable branches) **fails**: 2.2% and 3.2% < 5.9%.

**Source:** `O/cross_vla_temporal_replication/analysis.json` (`AB.*`, `CDE.all_units.*`), RESULTS.md, PROTOCOL.md.
**Manuscript-safe:** "In Impromptu VLA, a different temporally autoregressive driving VLA, correcting only the previous waypoint in context reduces amplification from 63.4% to 1.3% (−62.2 pp [−65.6, −59.1]) and re-inserting the erroneous waypoint restores it to 81.9%."
**Overclaim:**
- "Failure drops to 1%." That is amplification; A− only falls to 16.3%.
- "All criteria replicated." Criterion 3 failed.
- Omitting the leak: GT context in absolute coordinates leaks GT information; reverse and motion substitution are the leak-free evidence.
- "GT-near waypoint is equivalent to GT" (no equivalence test).
- p = 2e-31 or "0" (unit-level).

# Claim 9. OpenVLA: local-only architectural control
**Evidence** (commit 6b9e8ad + amendment 6d0375a; LIBERO-Spatial 10 tasks × 10 = 100 episodes)
- **Within-step feedback is strong:** shifting the x token by 8 bins amplifies a downstream dimension in **92.6% [89.8, 94.4]** of 4,741 units (task-cluster CI; initial-state-cluster CI [90.8, 94.0]; 24 bins: 81.2%). Corrected gives 0%; reverse = feedback.
- **Task level:** feedback − corrected **+3 pp [−4, +9], p = 0.65** (8 bins; task-cluster CI — with initial-state clusters [−6, +12], review_followup3 P0-E), +5 pp (p = 0.36, 24 bins). Every condition vs natural has p ≥ 0.115.
- **Rerun control:** 87 vs 81% (−6 pp [−14, +2], p = 0.26), outcome disagreement **20%**.

**Source:** `O/cross_vla_replication/analysis.json` (`phaseC`, `vs_natural`, `natural_rep_vs_natural`).
**Manuscript-safe:** "OpenVLA, which does not condition on its previous actions, shows strong within-step token feedback (92.6%) but no task-level effect distinguishable from re-run variability (20% outcome disagreement)."
**Overclaim:**
- "OpenVLA is immune to feedback." Within-step feedback is strong.
- "No effect proven." 100 episodes cannot detect effects of about 10 pp.
- Reporting the official 84.7% as if it were in the repository (it appears only in the summary document).

# Claim 10. SpatialVLA: token-level replication; closed-loop success effects uncertain (revised 2026-10-13)
**Evidence** (Experiment 34, commit f578eae; 18,975 token units; closed loop 80 episodes = coke 40 seeds + move_near 40 seeds = **coke 40 / move_near 29 primary initial states / 10 physical layouts**)
- **Token level** (episode-cluster sign-flip, P0-B; not re-clustered by initial state):
  - Correction (recent_ref − normal; the reference is the model's own greedy chunk, not GT): −8.3 pp [−9.3, −7.4], p ≤ 5e-6; step-4 deviation −82%.
  - Reverse (Normal-row token inserted into the reference context): +5.1 pp [+4.4, +5.8], p ≤ 5e-6; step-4 deviation restored to 0.0566 (Normal 0.0561).
  - Window: 1 step 86% (RESULTS says 87%), 2nd-last step only 63%.
- **Closed loop, direct context effects** (initial-state clusters, post-hoc B re-analysis; pooled = task-equal mean): δR = R − N and δC = C − F for perp_left and opposite are +5.0 / +5.0 / −1.2 / −1.2 pp with 95% CIs about ±10–14 pp; **±10 pp equivalence not established** for any of the four (Holm p_equiv 0.33–0.38; reproduces P0-B). In the opposite direction, coke and move_near have opposite-signed effects that cancel in the pooled mean. Judgment: **uncertain**.
- **Closed loop, executed perturbation:** opposite C − N −21.2 pp, Holm p = 0.041 (primary clusters) / 0.094 (physical).
- **Trajectories:** the "0.21–0.23 m vs rerun 0.14 m" values are **cumulative commanded actions**, not the actual TCP; Experiment 34 did not record the TCP. Generated-suffix tokens change in 58% of positions after the perturbation (vs 6% for a numeric rerun).
- **Rerun noise:** numeric-path rerun success disagreement 13.75%.

**Source:** `O/review_followup2/B_spatialvla_direct/run_20261013/DIRECT_EFFECTS.csv` (experiment exp34), `B4_TRAJECTORY_VS_SUCCESS.csv`; `O/review_followup/P0B_cluster_stats/run_20261008/paired_effects.csv` (token level); `O/cross_domain_temporal_replication/analysis.json`.
**Manuscript-safe:** "In SpatialVLA, previous-action token feedback replicates at the token level (correction −82% step-4 deviation; reverse restores it); in closed loop the context effect on task success was not detected and equivalence within ±10 pp was not established, while the executed perturbation itself reduced success by about 21 pp."
**Overclaim:**
- "Feedback has no effect in SpatialVLA" / "proven absent at task level."
- "Context perturbation moves the trajectory 0.21–0.23 m." That is commanded-action accumulation.
- p = 7e-261 (unit-level).
- Treating 80 episodes as 80 independent initial states (move_near has 29 / 10).
- Saying the ensemble weights older predictions more (actual weights newest-first 57/26/12/5%).

# Claim 11. Context feedback recurs at token level across models; task-level consequences are architecture-dependent (revised 2026-10-13)
**Evidence**
- **Token-level causal structure (correction reduces, re-inserted error restores) in four models**, with model-specific interventions and amplification definitions:
  - AutoVLA: repeated Recent-GT −39.8 pp (13 drives); Reverse = Normal-branch embedding injection +28.6 pp.
  - Impromptu: repeated recent-GT waypoint −62.2 pp (20 drives, p = 1.9e-6 = floor); Reverse = re-insertion of the branch's own erroneous waypoint +81.9 pp.
  - OpenVLA: within-step 92.6%; reverse = feedback (identical downstream tokens).
  - SpatialVLA: token-level correction −8.3 pp (reference = own greedy chunk); reverse = Normal-row token insertion.
- **Interventions are matched in spirit, not identical**; amplification definitions differ (driving: A− ∧ FDE5 > 3 m; SpatialVLA token: accumulated step-2–4 deviation ≥ injected; OpenVLA: downstream bin shift). **Effect sizes are not comparable across columns.**
- **Open-loop trajectory effects appear in the two driving models**, which keep a 10-step autoregressive action history.
- **Closed-loop manipulation:** OpenVLA shows no task effect distinguishable from rerun variability. SpatialVLA context perturbations change generated actions (50–58% of suffix tokens) and the actual TCP trajectory (2.7–6.0 cm), and longer replanning intervals make these deviations persist (P1-D H1, composite configuration effect); the success direct effect, averaged over two tasks, is bounded within ±10 pp in the tested configurations (P1-D), task-specific effects are uncertain (Claim 18).
- **Experiment 35 task effects are deleted:** the r4e1 reverse −6.25 pp (p = 0.040, episode level) is p = 0.14 / 0.26 after initial-state clustering and did not replicate in P1-D (+1.25 pp [−5.6, +8.0]).

**Source:** Claims 3, 7–10, 17, 18; `paper_cross_model_table.csv`; `O/review_followup2/CLAIM_EVIDENCE_MATRIX.md`.
**Manuscript-safe:** "Previous-action context causally shapes subsequent action tokens in all temporally autoregressive decoders we tested; whether this produces task-level failure depends on the action horizon and execution scheme." / "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across columns."
**Overclaim:**
- "Replanning and ensembling absorb the feedback." Ensemble protection is deleted; replanning governs trajectory persistence only (composite effect).
- "A small task effect appears for reverse at 4-step replanning (−6.25 pp)." Deleted.
- "Execution architecture fully explains the driving vs. manipulation difference."
- "General failure mechanism of all VLAs."
- Placing the different "Reverse" treatments in one row without per-model definitions, or comparing amplification magnitudes across models.

# Claim 12. Instability is weakly predictable before deviation
**Evidence** (Experiment 31, held-out 43,469 candidates, prevalence 0.058; log bootstrap 1,000)
- **Pre-deviation AUROC 0.570–0.634:** entropy 0.634 [0.607, 0.658]; hidden probes 0.611–0.631, no better than entropy.
- **Within-scene pre AUROC 0.504–0.649** (excluding candidate variance; no CI).
- **Selected vs discarded candidates** (Experiment 22): pre-deviation entropy differs by −0.016 [−0.023, −0.010], vs post-deviation −0.397 [−0.422, −0.371].

**Source:** `O/instability_detection_baselines/detection.json` (`heldout.*_pre`); `O/mechanism_selection_link/heldout/summary.json`.
**Manuscript-safe:** "Before the first deviation, amplifying rollouts are only weakly distinguishable (held-out AUROC 0.57–0.63, including hidden-state probes)."
**Overclaim:**
- "Pre-deviation entropy is identical." (also: held-out logs are not drive-disjoint, Claim 21) The difference is small, but its CI excludes 0.
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
- "Independent held-out drives." Held-out shares 22 drives (85% of held-out scenes) with dev; write "held-out logs (not drive-disjoint)" (Claim 21).
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
- Calling the second-half replication drive-independent (the halves share 16 drives).

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

# Claim 17. Replanning interval governs trajectory persistence of context perturbations; ensembling does not absorb them (revised 2026-10-13)
**Evidence**
- **Experiment 35** (preregistered 8fc20bf; r1e4 and r4e1 seeds 0–119; **coke 120 / move_near 53 primary / 10 physical initial states**): r4e1 − r1e4 reverse vs natural mean actual-TCP distance +1.84 cm [+1.26, +2.46] (episode-level Wilcoxon p = 1e-10, not initial-state clustered); realignment +10.9 steps. The "+0.116 m trajectory divergence" is cumulative commanded action, not TCP.
- **P1-D** (preregistered b212637; 4,800 episodes, restore verified 3,840/3,840; context-perturbation budget fixed at 4 chunks in every configuration):
  - **H1 (replanning axis):** area DiD[R − N] r4e1 − r1e1 **+213.8 cm·step [+170.6, +255.2]**, p < 1e-4 (both tasks same direction) → supported. Composite effect of replanning interval + observation refresh / context reset + execution exposure (context-attributable executed translation 0.32 vs 0 m/episode).
  - **H2 (ensemble axis, executed dose matched by a predefined online offset):** DiD +20.7 cm·step [−24.6, +68.0], p = 0.385; 90% CI inside the predefined ±100 → equivalent, and not in the absorption direction. The P0-C "ensemble attenuation" is explained by reduced executed dose.
- **Task outcome:** see Claim 18. Experiment 35's r4e1 reverse −6.25 pp is deleted (clustered p = 0.14 / 0.26; P1-D +1.25 pp [−5.6, +8.0]). Experiment 35's "+1.8 cm without the ensemble" (r1e1 F vs C) is p = 0.028 unadjusted, 0.14 after Holm over 5 configurations.

**Source:** `O/review_followup/P1D_dose_control/run_20261008/analysis.md`, `paired_effects.csv`; `O/review_followup2/B_spatialvla_direct/run_20261013/DIRECT_EFFECTS.md`; `O/spatialvla_feedback_protection_ablation/RESULTS.md`.
**Manuscript-safe:** "Lengthening the replanning interval increased the persistence of context-induced trajectory deviations even at a fixed context-perturbation budget; temporal ensembling did not absorb context effects once executed dose was matched."
**Overclaim (delete):**
- "Ensembling protects against feedback."
- "4-step replanning makes reverse increase failures (−6.25 pp)."
- "Task success changes by at most 6 pp" (Experiment 35 CIs after clustering are ±8–14 pp).
- "Replanning interval alone" (it is a composite configuration effect).
- "Removing protections turns SpatialVLA into an AutoVLA-like failure mode." / "Correction restores success."

# Claim 18. SpatialVLA: generated and physical trajectory changes vs success direct effects (new 2026-10-13; post-hoc B re-analysis)
**Evidence** (P1-D, 240 units per configuration = coke 120 seeds + move_near 120 units; clusters: coke = seed, move_near = 120 primary states or 10 physical layouts; sign-flip inversion CIs; Holm within the stated family)
- **Direct effects** (pp; δR = S(R) − S(N); δC = S(C) − S(F), + = correction benefit; F − C = −δC):

| config | contrast | coke | move_near (primary / physical CI) | pooled (primary) | ±10 pp Holm p (primary / physical) | ±5 pp Holm p |
|---|---|---|---|---|---|---|
| r1e4 (official) | δR | 0.0 [−6.2, 6.2] | +1.7 [−5.4, 8.8] / [−8.8, 12.8] | +0.8 [−3.8, 5.5] | <0.001 / 0.002 | 0.16 / 0.30 |
| r1e4 | δC | +4.2 [−2.0, 10.9] | +0.8 [−6.8, 8.5] / [−2.6, 4.3] | +2.5 [−2.6, 7.6] | 0.006 / 0.002 | 0.27 / 0.31 |
| r4e1 | δR | +1.7 [−8.3, 11.5] | +0.8 [−8.3, 10.1] / [−8.1, 10.0] | +1.25 [−5.6, 8.0] | 0.006 / 0.010 | 0.27 / 0.31 |
| r4e1 | δC | −4.2 [−14.3, 5.8] | +4.2 [−5.0, 13.3] / [−10.5, 18.8] | −0.0 [−6.7, 6.8] | 0.006 / 0.014 | 0.23 / 0.31 |

- **Pooled two-task direct effects are equivalent within ±10 pp** under both clusterings after Holm (m = 4); **±5 pp is not established**; **task-level CIs span 18–29 pp in r4e1** (r1e4: 6.9–21.6 pp; r4e1: no task-level equivalence under either clustering). No direct effect differs from 0 after Holm.
- **Interactions S1–S4** (preregistered): ±10 pp equivalence survives initial-state clustering and Holm (primary 0.014–0.037, physical 0.026–0.047); ±5 pp not established (Holm ≈ 0.5). Per task, no interaction is equivalent; coke S3 −9.2 vs move_near +8.3 partly cancel.
- **Same units, generated / executed / physical changes:** suffix tokens changed 49–55% (vs numeric rerun 3–7%); context-attributable executed translation 0.044 (r1e4) / 0.32 (r4e1) m/episode; actual TCP mean distance R − N 2.67 cm (r1e4) / 5.95 cm (r4e1) vs rerun 1.24–2.00 cm; time-integrated 193 / 428 cm·step (t = 8–79, 72 steps).
- **Executed perturbation lowers success:** C − N r4e1 −11.7 pp [−18.3, −5.0] (Holm 0.009); exp 35 r1e4 C − N −22.9 pp.
- **Experiments 34/35** (episode-level judgments): all direct effects are **uncertain** after initial-state clustering; Experiment 35's pooled r1e4 δR ±10 equivalence (episode level) is lost after clustering.
- ±5 pp is a post-hoc sensitivity margin proposed by this review, not a field standard. move_near has only 10 physical layouts, shared by Experiments 34, 35 and P1-D.

**Source:** `O/review_followup2/B_spatialvla_direct/run_20261013/DIRECT_EFFECTS.{md,csv}`, `B4_TRAJECTORY_VS_SUCCESS.csv`, `INTERPRETATION.csv`; `O/review_followup2/A_spatialvla_states/run_20261013/cluster_counts.csv`.
**Manuscript-safe:** "Context perturbations changed generated actions and the physical trajectory, while their effect on success, averaged over two tasks, was bounded within ±10 pp in the tested configurations; task-specific effects remain uncertain."
**Overclaim:** "No effect on success"; "success is invariant to context"; "bounded within ±5 pp"; per-task bounds; generalizing beyond r1e4/r4e1 and the two tasks; counting 120 move_near units as 120 independent physical states.

# Claim 19. GT-free action-context effect (AutoVLA only) (new 2026-10-13)
**Evidence** (P1-A, preregistered b212637; family G = GT-free perturbation 0.10 ± 0.03 m/segment; A− 364 units / 52 scenes / 13 drives; reference = the model's own unperturbed rollout, no GT)
- Context-only perturbation changes later local actions: suffix local motion **+0.571 m/seg [+0.435, +0.682]** (C1 − U).
- Output + context perturbation vs pure coordinate accumulation (Geo baseline, no inference): endpoint **+4.37 m [+3.32, +5.19]** (post-hoc C1 excess metric ‖OC9 − U9‖ − ‖Geo9 − r9‖; drive p = 2.4e-4; Holm 9.8e-4 over the post-hoc excess family, m = 4). The **predeclared** P1A-2 metric (dev9_ref, OC − Geo) is the authoritative row: **+4.37 m [+3.34, +5.17]**, sign-flip-inversion CI [+3.06, +5.35], Holm 7.3e-4 (P1A family, m = 3) — `numbers.json` `P0C.plus437.AUTHORITATIVE`. Do not attach the predeclared Holm to the post-hoc CI (corrected 2026-10-15). The model's coordinate-accumulation part equals Geo's (+0.020 m [−0.015, +0.065]); the excess is entirely the context-driven change of later local actions. A+ +1.64 m, all scenes +2.32 m.
- Limits: AutoVLA only (no GT-free control for Impromptu or SpatialVLA); perturbation position t_p = t* and the A−/A+ strata inherit the GT definition; the reference requires a counterfactual rollout (not deployable online).

**Source:** `O/review_followup2/C_baseline_window/run_20261013/BASELINE_WINDOW_RECHECK.md` §C1, `c1_excess_effects.csv`, `c1_predeclared_and_gt_drive.csv`; `O/review_followup/P1A_gt_free_reference/run_20261008/analysis.md`.
**Manuscript-safe:** "Without ground-truth information, a single perturbed context token changed subsequent local actions and produced endpoint deviations 4.4 m beyond pure coordinate accumulation (AutoVLA)."
**Overclaim:** "GT-free effects are general across models"; "GT-free correction fixes failures" — the reference-based correction (OC_refR, −3.50 m deviation from the reference) restores consistency with the model's own rollout but **worsens** GT amplification in A− (+19.7 pp [+9.0, +29.6]). Allowed instead: "Reference-based correction restored consistency with the model's own rollout but did not reduce ground-truth failures."

# Claim 20. Excess over constant-motion extrapolation only in failure-selected scenes (new 2026-10-13; general claim deleted)
**Evidence** (post-hoc C1; perturbed − unperturbed change for each predictor, so CM's own extrapolation error is removed)
- Model − CM endpoint change: **A− +2.35 m [+1.27, +3.19]** (p = 7.3e-4, Holm 9.8e-4); **A+ −0.52 m [−0.82, −0.26]**; **all scenes +0.20 m [−0.32, +0.56], p = 0.51** (not detected; equivalence not tested — no margin was predeclared).
- The original contrast OC − CM (A− −0.59 m, p = 0.23, "indistinguishable from extrapolation") mixed in CM's own unperturbed extrapolation error (4.69 m) and is withdrawn.
- Suffix local-motion change exceeds CM's d0 in every stratum (+0.48 / +0.13 / +0.21 m/seg).
- **Updated 2026-10-15 (review_followup3 P2):** CM is a body-frame SE(2) repeat of the perturbed token (yaw included), i.e. already a constant-turn-rate extrapolation. Added CV0 (zero-turn global velocity), CTRV-K3, CTRA-2 and a learned non-AR motion extrapolator (prefix / prefix + ego, 3 seeds). In A− the model exceeds all six; in A+ and all scenes it does not (see Claim 23).

**Source:** `O/review_followup2/C_baseline_window/run_20261013/c1_excess_effects.csv`, `BASELINE_WINDOW_RECHECK.md` §C1.2.
**Manuscript-safe:** "In failure-selected scenes the response exceeded constant-motion extrapolation; across all scenes it was not distinguishable from extrapolation."
**Overclaim (delete):** "self-amplification beyond extrapolation" as a general property; "indistinguishable from extrapolation in failing scenes" (the old, mismatched comparison).

# Claim 21. Held-out and navhard validations are log-disjoint, not drive-disjoint (new 2026-10-13; labelling)
**Evidence** (`O/review_followup2/A_autovla_cluster/run_20261013/overlap_matrix_drives.csv`)
- Selection dev (32 drives) and preregistered held-out (28 drives) share **22 drives**; 4,077 of 4,814 held-out scenes (85%) come from drives also in dev. The drive-disjoint subset is 7 logs / 737 scenes / 6 drives and was not re-analysed.
- navhard first and second halves share 0 logs but **16 drives**.
- P1-C eval shares **0 drives** with the AutoVLA mechanism set (10 day-vehicles shared).

**Manuscript-safe:** "held-out logs (not drive-disjoint)"; for P1-C: "new drives (day/vehicle partly shared)".
**Overclaim:** "independent held-out drives" for Experiments 21–26.

# Claim 22. Evaluation-population estimand and A− frequency (new 2026-10-15; review_followup3 P0-A, post hoc)
**Tag:** driving / AutoVLA / open-loop 5 s plan.

**Evidence** (`numbers.json` `P0A.*`; Table 8)
- **Selection flow.** 2,747 non-CoT dev natural scenes (20 drives) → **52 A−** (13 drives, 10 day–vehicles), which is **all** A− under the dev rule (π = 1; 0 A− lack a first mismatch). 156 A+ were sampled 3 per A− at the same t* (Random(0)); the replica equals the stored set (208/208).
- **A− frequency in the evaluation set** (not a deployment or natural failure rate):
  - dev: 52/2,747 = **1.89 % [1.16, 2.64]** (drive clusters);
  - P1-C: 107/5,671 = 1.89 % [1.32, 2.53].
- **Evaluation-population estimand.** Hajek, scene-weighted, over the 2,191 eligible scenes of the 64 P1-C logs:
  - RN **−4.1 pp [−6.4, −2.4]**;
  - RV **+3.6 pp [+2.2, +5.5]** (24 drives);
  - P = F + S holds exactly; 60 % of RN P comes from the A+ stratum.

**Manuscript-safe:** "On the evaluation population of the new-drive logs (2,191 eligible scenes, 24 drives), recent-context correction reduced amplification by 4.1 pp [2.4, 6.4]; the A− category occurred in 1.9 % of evaluation scenes (an evaluation-set frequency, not a deployment rate)."

**Overclaim:** "natural failure rate 52/2,747"; "population/deployment effect"; prevalence × stratum effect products.

# Claim 23. Response beyond motion extrapolation — A− only (new 2026-10-15; review_followup3 P2, post hoc; supersedes the baseline limits of Claim 20)
**Tag:** driving / AutoVLA / open-loop 5 s plan; GT-free family G (P1-A rollouts).

**Evidence** (`numbers.json` `P2A.*`, `P2B.*`, `P2.verdict.*`; Table 9)
- **CM** (repeat-p constant motion) is a body-frame SE(2) repeat of the perturbed token, yaw included: already a constant-turn-rate extrapolation. C1 reproduces exactly.
- **Endpoint Δ_model − Δ_control (m), A−:**

| control | A− | Holm p |
|---|---|---|
| CM | +2.35 [+1.27, +3.19] | 0.0015 (m = 4) |
| CV0 | +4.02 [+2.94, +4.83] | |
| CTRV-K3 | +3.02 [+1.93, +3.87] | |
| CTRA-2 | +1.43 [+0.44, +2.33] | 0.023 |
| learned non-AR extrapolator, prefix | +1.97 [+1.06, +2.87] | |
| learned non-AR extrapolator, prefix + ego | +2.27 [+1.50, +3.10] | |

  The 3 training seeds differ by ≤ 0.04 m. Excluding out-of-range units keeps the sign.
- **A+:** CM −0.52, CTRA-2 −1.56, extrapolator −1.11 / −0.56 (the model moves *less*).
- **All scenes:** CM +0.20 (p = 0.51), extrapolator −0.34 / +0.15 (n.s.).

**Manuscript-safe:** "In failure-selected scenes, AutoVLA's endpoint response to a GT-free context perturbation exceeded four kinematic extrapolations (including the original body-frame constant-motion repeat) and a small learned non-autoregressive motion extrapolator; across all scenes and in non-failure scenes it was not distinguishable from motion extrapolation."

**Overclaim:** "beyond extrapolation" without the A− restriction; calling the extrapolator a non-AR VLA; "CM ignores curvature".

# Claim 24. Intervention-machinery verification scope (new 2026-10-15; review_followup3 P1)
**Evidence** (`numbers.json` `P1.*`; Table 11)
- **AutoVLA no-op/cache gate.**
  - Controls: hook no-op, self-patch, cache invalidation, full recompute vs cached edit, self-reproducibility.
  - Decodings: greedy and T = 0.01. Tolerances: T_exact 0, T_equiv 0.75.
  - Paths: H7 / H8 / H9 / P1A (21 path-units) and H32 / P1B (18 path-units).
  - Result: PASS, 0 failures. Reverse-N ≡ token substitution (82/82 steps).
- **SpatialVLA restore audit** (20 units): PASS, with the C2 null arm settled by the split-fingerprint re-verification (20/20). The earlier strict combined-hash 0/20 and post-hoc A5 20/20 are kept on record.
- Arms equal HF `generate` only in 16/20 T0 chunks, so the exp 34 "identical to generate" wording is withdrawn.
- **Not verified:** Impromptu and OpenVLA paths; exp 34/35 restore; runs outside the manifests.

**Manuscript-safe:** "No-op, self-patch, cache and full-recompute parity of the intervention code was verified on fixed verification manifests (AutoVLA: all six intervention paths; SpatialVLA: state restoration on 20 units); earlier runs outside these manifests are not retro-validated, and the Impromptu and OpenVLA paths were not parity-verified."

