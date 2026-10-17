# COHORT_ESTIMAND_MAP — sets, denominators, estimands and definitions (review_followup4 A4)

Date: 2026-10-17. This is a documentation audit: CPU only, and no document was edited.

- Code: `scripts/review_followup4/a4_cohort_map.py`. It re-reads every count from the source tables and asserts it, and it checks that each quoted violation text really sits on the cited line.
- Outputs: `COHORT_ESTIMAND_MAP.csv` (25 rows: 8 sets, 14 numbers, 3 definitions) and `A4_WORDING_VIOLATIONS.csv` (64 file:line items with proposed replacements).

## 1. Sets: every number belongs to exactly one of these

| set | scenes | units | logs / drives / day–veh. | weighting | what it may be called |
|---|---|---|---|---|---|
| dev natural | 2,747 | 1 plan each | 28 / 20 / 14 | unweighted | A− 52 = 1.89% [1.16, 2.64] and severe 40 = 1.46% are **evaluation-set frequencies**, not deployment or natural failure rates |
| dev mechanism (selected) | 208 = 52 A− (all, π = 1) + 156 A+ (3 per A−, same t*) | A− stratum 365 | 26 / 19 / 13 (A−: 16 / 13 / 10) | selected; A− share 25% vs 1.9% | mechanism contrasts are A−-stratum (52 scenes / 13 drives) conditional effects |
| P1-C stage 1 | 5,671 | 1 plan | 64 / 24 / 16 | unweighted | A− 107 = 1.89% (evaluation set) |
| **P1-C target E** | 2,191 eligible (t* ≤ 7) | – | 64 / 24 / 16 | Hajek target | "evaluation-population (E) estimand"; A− 105/2,191 = 4.79% |
| **P1-C intervention sample** | 353 = 105 F + 248 S (m = 4 per segment) | 2,436 | 64 / 24 / 16 | F weight 1, S weight N_seg/m | P effects RN −4.1 [−6.4, −2.4], RV +3.6 [+2.2, +5.5] are weighted estimates from these 353 scenes |
| **P1-C failure stratum F** | 105 | 748 | 41 / **18** / 10 | scene-weighted | RN −34.2, RV +26.8, DM +6.2: **18 drives**, never "24 (independent) failure drives" |
| motion-baseline "all" | **208 selected** (= dev mechanism set) | P1-A units | 26 / 19 / 13 | scene-weighted over the selected mix | "all 208 selected mechanism scenes", not 2,747 / 2,191 / population. The P2 label "E" means the learned extrapolator and is unrelated to target E |

## 2. Required facts

- **RN ≈ 60% from A+.** P = F + S holds exactly (frame 2,191).
  - RN: F contributes 39.7%, S (A+) 60.3%. RV: 35.3% / 64.7%. DM: 11.3% / 88.7% (`POPULATION_decomposition.csv`).
  - The −4.1 pp evaluation-population effect is therefore mostly a change in non-failure scenes. It is not recovery of originally failing scenes.
- **OpenVLA 92.6%.**
  - Unit: (saved frame, sign of δ). n = 4,741, i.e. 2,462 frames × 2. 183 fully clipped units are dropped, and 30 units have an effective shift below 8 bins.
  - Definition: amplified = max over the downstream dimensions y, z, roll, pitch, yaw (gripper excluded) of |bin − natural bin| ≥ the effective injected |x shift| in bins. This means **at least one** later component reaches the threshold, not all.
  - Bins mix m and rad. Translation components alone: 72.3% (bins) / 72.4% (commanded m). Rotation components alone: 85.9%.
  - CI: task clusters [89.8, 94.4]; initial-state clusters [90.8, 94.0].
  - This is not task failure and not realised motion.
- **Physical units, kept separate.** Values are commanded per-step deltas through the OSC_POSE input scaling, not realised TCP motion.
  - 8 bins = x 2.64 mm, y 2.41 mm, z 2.93 mm.
  - 8 bins = roll 3.31 mrad, pitch 6.02 mrad, yaw 5.18 mrad.
  - The gripper is binary (bin changed in 4.7% of units).
- **Commanded vs actual TCP.**
  - SpatialVLA metre-scale numbers are cumulative commanded actions: exp34 0.21–0.23 m (no TCP recorded) and exp35 +0.116 m.
  - Actual TCP numbers are centimetre-scale: P1-D R−N 2.67 / 5.95 cm (re-run 1.24 / 2.00); exp35 +1.84 cm.
  - OpenVLA has no TCP at all.
- **Opposite per-task point estimates ≠ heterogeneity.** P1-D r4e1 C−F is −4.2 / +4.2 pp, with heterogeneity −8.3 [−21.7, +5.0] (p = 0.23). exp35 r1e4 R−N is −7.5 / +5.8, with heterogeneity CI [−30.2, +3.2]. exp34 was not tested.
  - The "cancellation share" is arithmetic on point estimates, so neither heterogeneity nor cancellation is established.
- **Drive-disjoint ≠ day/vehicle-disjoint.**
  - P1-C shares 0 drives with the dev mechanism set but 10 day–vehicle combinations. It has 6 connected day–vehicle components, so the exact Holm floor is 0.094. No day-and-vehicle-disjoint log exists.
  - The exp 21 held-out set shares 22 drives (85% of scenes), and the navhard halves share 16 drives.
- **A− and severe.**
  - A− = NOT `pra_labels.label_A`, which is the conjunction of a longitudinal and a lateral class check.
    - Kinematics: DT = 0.5 s; v_start is the mean speed over the first 2 steps and v_end over the last 2.
    - Longitudinal class: STOP if the total path is < 1.0 m and v_end < 0.5 m/s. If v_start < 0.5 m/s: ACCELERATE iff Δv > 0.8 m/s, otherwise MAINTAIN. Otherwise ratio ≥ 1.25 → ACCELERATE, ≤ 0.75 → DECELERATE/STOP, else MAINTAIN.
    - Longitudinal acceptance: the predicted class (computed with the GT start speed) must be reachable within GT v_end ± max(0.5 m/s, 15%).
    - Lateral class: LEFT/RIGHT if the final lateral offset is > 2.0 m or the final-step heading is > 15° (heading is ignored if the last step is < 0.25 m). Lateral acceptance uses a tolerance of ±1.0 m / ±10°.
    - An invalid plan counts as A− (in P1-C: amplification 0).
  - Severe = amplification = A− ∧ FDE5 > 3.0 m, where FDE5 is measured at pose 10.
  - **Horizon caveat.** The rule is applied to the **10-pose / 5 s plan** (`_pos_to_delta(traj, 10)` in `analyze_action_history.a_eval`), although `trajectory_metrics.py` documents these thresholds for a **3 s horizon** (`HORIZON = 6`; "total path length below this over 3 s"). Only the P/R/A comparison has a 3 s sensitivity run (`common_3s_horizon`). The original definition is preserved, and no document currently states this.
  - The GT-free `severe_ref{2,3,4}` (P1-A/B) is a different outcome: dev9_ref (pose index 9 = 5 s endpoint, measured against the model's own reference) > 2 / 3 / 4 m.
- **Non-significance is not equivalence, and a verified path is not a validated estimate.** The `VERIFIED_MACHINERY_MANIFEST_SCOPE` status certifies an intervention code path. It does not certify the estimand or the interval (see A1).

## 3. Wording violations (do not edit here; full text and replacements in `A4_WORDING_VIOLATIONS.csv`)

Scanned: `/root/VLA/EXPERIMENT_SUMMARY.md`, `/root/VLA/paper_quantitative_package/*.md` + `tables.tex`, `/root/VLA/review_followup3/*.md`. Every item's text was verified on its line.


| file:line | rule | current text | proposed replacement (short) |
|---|---|---|---|
| EXPERIMENT_SUMMARY.md:11 | A1 CI validity | p·CI는 cluster sign-flip(역산 CI) | p는 cluster sign-flip(δ = 0, 유효), CI는 사후 수정(Rao–Scott design-effect Newcombe 10; 과제 평균은 MOVER) — review_followup4 A1 |
| EXPERIMENT_SUMMARY.md:26 | set/denominator; F = 105 scenes / 18 drives; evaluation-population | 새 24 drive**(P1-C)에서 재현: 실패 조건부 RN −34.2%p | 새 drive(P1-C; 0 drive 공유, day–vehicle 10 공유)에서 재현: 실패 층 F(105 장면 / 18 drive) RN −34.2%p, RV +26.8%p; 평가 집합 E 추정량(적격 2,191 장면, 선정 353 장면 가중, 24 drive) −4.1 / +3.6%p — RN의 약 60%는 A+ 층 기여 |
| EXPERIMENT_SUMMARY.md:34 | motion-baseline all = 208 selected; non-significance != equivalence | 전체에서는 +0.20 m(p = 0.51)로 외삽과 구별되지 않습니다 | 선정 기전 집합 전체(208 장면 = A− 52 + A+ 156)에서는 +0.20 m [−0.32, +0.56](p = 0.51)로 차이가 검출되지 않았습니다(동등성 미검정) |
| EXPERIMENT_SUMMARY.md:36 | OpenVLA 92.6% exact definition; units | 나머지 차원이 93%에서 그 이상 움직이고 | x token을 ±8 bin(명령 약 2.6 mm/step) 바꾸면 단위의 92.6%(4,741 단위; 초기 상태 군집 CI [90.8, 94.0])에서 하위 성분(y, z, roll, pitch, yaw; gripper 제외) 중 적어도 하나의 bin 변화가 실제 주입 bin 이상이었고(translation 성분만으로는 72%) |
| EXPERIMENT_SUMMARY.md:38 | non-significance != equivalence; 20% rerun disagreement is not a detection threshold | closed-loop 기저 변동(자연 재실행끼리도 결과 20% 불일치) 안에 있었습니다 | LIBERO 과제 성공률 효과는 검출되지 않았고(C − F 8-bin −3%p [−11.9, +5.9], 동등성 미성립), 자연 재실행끼리도 결과가 20% 달랐습니다 |
| EXPERIMENT_SUMMARY.md:88 | OpenVLA 92.6% exact definition | 하위 차원 93% 증폭 | x token ±8 bin 교란 → 하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위 92.6% |
| EXPERIMENT_SUMMARY.md:157 | evaluation set vs population | 모집단 A− 1.24 → 2.60%로 악화 | dev 자연 평가 집합(2,747 장면: A− 52 / A+ 2,695)으로 재가중한 A− 빈도 1.24 → 2.60%로 악화(배포 빈도 아님) |
| EXPERIMENT_SUMMARY.md:211 | OpenVLA 92.6% exact definition; units | 하위 차원이 δ 이상 움직이는 비율 **92.6% [89.8, 94.4]** | 하위 성분(y..yaw; gripper 제외) 중 적어도 하나의 \|bin 변화\| ≥ 실제 주입 bin(clip 후)인 단위 비율 92.6% (4,741 단위 = 2,462 프레임 × 부호; task CI [89.8, 94.4], 초기 상태 CI [90.8, 94.0]); bin은 m·rad 혼합 |
| EXPERIMENT_SUMMARY.md:280 | opposite point estimates != heterogeneity | pooled 평균에서 상쇄됩니다 | 과제별 점추정의 부호가 반대(δR −10.0 vs +7.5%p, δC +7.5 vs −10.0%p)지만 과제 간 차이는 검정하지 않았고 과제별 CI가 넓어, 이질성이나 상쇄가 확인된 것은 아닙니다 |
| EXPERIMENT_SUMMARY.md:361 | motion-baseline all = 208 selected | 전체 +0.20 m [−0.32, +0.56] (p = 0.51) | 선정 208 장면 전체(A− 52 + A+ 156) +0.20 m [−0.32, +0.56] (p = 0.51; 동등성 미검정) |
| EXPERIMENT_SUMMARY.md:364 | F = 105 scenes / 18 drives; P = 353 weighted | RN F −34.2 [−38.3, −29.5] / P −4.1 | RN F(105 장면 / 18 drive) −34.2 [−38.3, −29.5] / P(E 2,191 적격 장면, 353 선정 장면 가중, 24 drive) −4.1 |
| EXPERIMENT_SUMMARY.md:421 | evaluation-population | **모집단 추정량.** | **평가 집합(E) 추정량.** |
| EXPERIMENT_SUMMARY.md:472 | opposite point estimates != heterogeneity | 과제 효과가 반대 부호로 평균에서 100% 상쇄됩니다 | P1-D r4e1 C−F의 과제별 점추정은 −4.2 / +4.2%p로 부호가 반대지만, 과제 간 차이 −8.3%p [−21.7, +5.0](p = 0.23)는 0을 포함하므로 이질성·상쇄는 확인되지 않았습니다 |
| EXPERIMENT_SUMMARY.md:488 | motion-baseline all = 208 selected | \| 전체 \| CM +0.20 (n.s.) | \| 선정 208 장면 전체 (A− 52 + A+ 156) \| CM +0.20 (n.s.; 동등성 미검정) |
| EXPERIMENT_SUMMARY.md:490 | all = 208; non-significance != equivalence | A+와 전체에서는 운동 외삽과 구별되지 않습니다 | A+와 선정 208 장면 전체에서는 운동 외삽과의 차이가 검출되지 않았습니다(동등성 미검정) |
| EXPERIMENT_SUMMARY.md:542 | F = 18 drives | 새 24 drive에서 재현 | 새 drive의 실패 층(105 장면 / 18 drive)에서 재현 |
| EXPERIMENT_SUMMARY.md:567 | A-/severe exact formula; 3 s thresholds applied at 5 s | 평가 horizon은 5 s(10 token)이며 | 평가 horizon은 5 s(10 token)입니다. A−는 P/R/A 행동 범주 판정(pra_labels.label_A: 종방향 v_end/v_start 1.25/0.75, 저속 0.5 m/s·가속 0.8 m/s, STOP = 경로 < 1.0 m & v_end < 0.5 m/s, 허용 max(0.5 m/s, 15%); 횡방향 최종 횡변위 > 2.0 m 또는 마지막 step heading > 15°, 허용 ±1 m/±10°)이고, 증폭(severe) = A... |
| EXPERIMENT_SUMMARY.md:575 | opposite point estimates != heterogeneity | 과제별로 반대 부호 효과가 일부 상쇄됩니다 | 과제별 점추정의 부호가 일부 반대지만 과제 간 이질성은 검출되지도 배제되지도 않았습니다 |
| EXPERIMENT_SUMMARY.md:47 | A1 CI validity | 과제별로는 CI 폭 최대 18–29%p(P1-D r4e1)로 불확실 | 과제별로는 CI 폭 최대 18–25%p(P1-D r4e1; r1e4 13–20%p; A1 사후 수정 구간)로 불확실 |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:11 | F = 105 scenes / 18 drives; drive- vs day/vehicle-disjoint; cohort of the 40 pp | Both effects replicated in a preregistered test on 24 new drives that share no drive with the development set | '...reduced trajectory amplification in failure-stratum (A−) scenes by 40 pp (52 scenes, 13 drives; ...)' and 'Both effects replicated in a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared with the development set, b... |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:17 | F = 18 drives; evaluation-population; 60% A+ | A preregistered replication on 24 new drives (day/vehicle partly shared) reproduces both contrasts (failure-conditional −34.2 / +26.8 pp; po | A preregistered replication on new drives (days/vehicles partly shared) reproduces both contrasts: failure stratum (105 scenes, 18 drives) −34.2 / +26.8 pp; evaluation-frame estimand over 2,191 eligible scenes (353 sampled, weighted; 24 drives) −4.1 / +3.6 ... |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:18 | motion-baseline all = 208; non-significance != equivalence | across all scenes it is not distinguishable from extrapolation (+0.20 m, p = 0.51) | across the 208 selected mechanism scenes (52 A− + 156 A+; not the natural mix) no difference from extrapolation was detected (+0.20 m [−0.32, +0.56], p = 0.51; equivalence not tested) |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:49 | motion-baseline all = 208 | across all scenes it was not distinguishable from extrapolation | ...; across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested). |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:58 | F = 18 drives | "replicated on 24 new drives (day/vehicle partly shared)" | "replicated on new drives (failure stratum 105 scenes / 18 drives; evaluation frame 24 drives; days and vehicles partly shared)" |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:83 | F = 18 drives | both effects replicated in a preregistered test on 24 new drives (−34.2 and +26.8 pp) | both effects replicated in a preregistered test on new drives (failure stratum, 105 scenes / 18 drives; −34.2 and +26.8 pp; days and vehicles partly shared) |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:87 | P1-C intervention sample 353 weighted; 60% A+ | On the evaluation population of the new-drive logs (2,191 eligible scenes) | On the evaluation frame of the new-drive logs (2,191 eligible scenes; estimated from 353 sampled scenes with design weights, 24 drives), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5] (about 60% of the RN effect from non-failur... |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:97 | all = 208; non-significance | across all scenes and in non-failure scenes it was not distinguishable from motion extrapolation | in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested) |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:121 | OpenVLA 92.6% exact definition; units | moved a later dimension by at least 8 bins in 92.6% of units | an 8-bin x-token shift (≈ 2.6 mm commanded) changed at least one later component (y, z, roll, pitch, yaw; gripper excluded) by at least the effective injected number of bins in 92.6% of 4,741 units (initial-state-cluster CI [90.8, 94.0]; translation compone... |
| paper_quantitative_package/MANUSCRIPT_REVISION.md:148 | F = 18 drives | "replicated on 24 new drives (days and vehicles partly shared) | "replicated on new drives (failure stratum 105 scenes / 18 drives; days and vehicles partly shared); no external (day/vehicle-disjoint) validation was possible" |
| paper_quantitative_package/tables.tex:37 | A-/severe exact formula; 3 s thresholds at 5 s | Amplification = A$-$ and FDE$_{5\,s}>3$\,m. | Amplification = A$-$ and FDE$_{5\,s}>3$\,m, where A$-$ = failure of the coarse action-category rule (longitudinal class from start/end speed with tolerance $\max(0.5$\,m/s$, 15\%)$; lateral class from final offset $>2$\,m or final heading $>15^\circ$, toler... |
| paper_quantitative_package/tables.tex:51 | drive- vs day/vehicle-disjoint; F cohort | 0 drives shared with this set | New drives (P1-C, pre-registered; failure stratum 105 scenes / 18 drives; 0 drives but 10 day--vehicle combinations shared with this set): ... |
| paper_quantitative_package/tables.tex:69 | OpenVLA 92.6% exact definition | Normal amplification & 47.4 [38, 57] & 63.4 [60.6, 66.7] & within-step 92.6 | OpenVLA cell: 'perturbed within-step dependency: $\ge$1 later component (y..yaw, gripper excl.) shifted $\ge$ injected bins, 92.6 (state CI [90.8, 94.0]); bins, not physical units' (or move it to its own row) |
| paper_quantitative_package/tables.tex:174 | A1 CI validity | pp with sign-flip-inversion 95\% CI on initial-state clu | pp with post-hoc corrected 95% CI (Rao--Scott design-effect Newcombe; MOVER for the two-task mean) on initial-state clusters; original sign-flip-inversion CIs in the supplement |
| paper_quantitative_package/tables.tex:187 | A1 CI validity | task-level CIs span up to 18--29 pp (r4e1; r1e4 6.9--21.6 pp) | task-level CIs span up to 18--25 pp (r4e1; r1e4 13--20 pp; post-hoc corrected) |
| paper_quantitative_package/tables.tex:230 | motion-baseline all = 208 selected | all & +0.20 [$-$0.33, +0.56] | row label 'all (208 selected = 52 A$-$ + 156 A+)'; caption: '\emph{all} is the selected mechanism set, not the natural set'; also rename column 'E' (learned extrapolator) to avoid confusion with P1-C target E |
| paper_quantitative_package/tables.tex:240 | A1 CI validity | pp, sign-flip-inversion 95\% CI on initial-state clusters | pp, post-hoc corrected 95% CI (Rao--Scott design-effect Newcombe) on initial-state clusters |
| paper_quantitative_package/PAPER_NUMBERS.md:85 | evaluation-population; 353 sampled | population estimand P (24 drives, Hajek-weighted) | evaluation-frame estimand P (2,191 eligible scenes; 353 sampled, Hajek-weighted; 24 drives) |
| paper_quantitative_package/PAPER_NUMBERS.md:91 | F = 18 drives | (13 drives; replicated on 24 new drives) | (A− stratum, 52 scenes / 13 drives; replicated in the P1-C failure stratum, 105 scenes / 18 new drives) |
| paper_quantitative_package/PAPER_NUMBERS.md:180 | OpenVLA 92.6% exact definition | amplifies a downstream dimension in **92.6% [89.8, 94.4]** | changes at least one downstream component (y..yaw; gripper excluded) by ≥ the effective injected bin shift in 92.6% [89.8, 94.4] (bins; translation-only 72%) |
| paper_quantitative_package/PAPER_NUMBERS.md:185 | non-significance != equivalence | no task-level effect distinguishable from re-run variability | no task-level effect was detected (C−F −3 pp [−11.9, +5.9]; equivalence not established), while re-running alone changed 20% of outcomes |
| paper_quantitative_package/PAPER_NUMBERS.md:197 | opposite point estimates != heterogeneity | opposite-signed effects that cancel in the pooled mean | opposite-signed task point estimates (heterogeneity not tested; task CIs wide), so the pooled mean may hide task differences |
| paper_quantitative_package/PAPER_NUMBERS.md:220 | non-significance != equivalence | OpenVLA shows no task effect distinguishable from rerun variability | OpenVLA showed no detectable task effect (equivalence not established) |
| paper_quantitative_package/PAPER_NUMBERS.md:328 | opposite point estimates != heterogeneity | coke S3 −9.2 vs move_near +8.3 partly cancel | coke S3 −9.2 vs move_near +8.3 (opposite point estimates; heterogeneity not established) |
| paper_quantitative_package/PAPER_NUMBERS.md:350 | motion-baseline all = 208 | **all scenes +0.20 m [−0.32, +0.56], p = 0.51** | **all 208 selected mechanism scenes (52 A− + 156 A+) +0.20 m [−0.32, +0.56], p = 0.51** |
| paper_quantitative_package/PAPER_NUMBERS.md:356 | motion-baseline all = 208 | across all scenes it was not distinguishable from extrapolation | across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested) |
| paper_quantitative_package/PAPER_NUMBERS.md:405 | motion-baseline all = 208 | across all scenes and in non-failure scenes it was not disting | in non-failure scenes and across the 208 selected mechanism scenes no excess was detected |
| paper_quantitative_package/PAPER_EVIDENCE.md:558 | OpenVLA 92.6% exact definition | 하위 차원 증폭이 92.6% | 하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위가 92.6% |
| paper_quantitative_package/PAPER_EVIDENCE.md:716 | non-significance != equivalence | closed-loop task 효과가 없다는 것은 | closed-loop task 효과가 검출되지 않았다는 것은(동등성 미검정) |
| paper_quantitative_package/PAPER_EVIDENCE.md:802 | evaluation set vs population | 모집단 A− 1.24 → 2.60%로 악화 | dev 평가 집합(52/2,695) 재가중 A− 1.24 → 2.60%로 악화 |
| paper_quantitative_package/PAPER_EVIDENCE.md:817 | evaluation set vs population | 모집단 가중 52/2,695 | dev 평가 집합 가중 52/2,695 |
| paper_quantitative_package/EXP35_RESULTS.md:18 | non-significance != equivalence | task 수준에서는 교정 효과가 없고 | task 수준에서는 교정 효과가 검출되지 않았고(동등성 미검정; 군집 후 불확실) |
| paper_quantitative_package/EXP35_RESULTS.md:56 | commanded delta vs actual TCP | 궤적 발산 | '실행 명령 누적 끝점 거리 (commanded, TCP 아님)' |
| paper_quantitative_package/EXP35_RESULTS.md:36 | commanded delta vs actual TCP | trajectory_divergence F 대 C / R 대 N (m) | commanded_cumulative_divergence F 대 C / R 대 N (m; TCP 아님) |
| paper_quantitative_package/CONSISTENCY_AUDIT.md:47 | non-significance != equivalence; heterogeneity from per-task significance | 결론(문맥 효과 0, 실행 효과 있음) | 결론(문맥 효과 검출 안 됨, 실행 효과 있음). 과제별 점추정 차이(coke −37.5 vs move_near −2.5%p)는 이질성 검정 없이 '집중'으로 쓰지 않음 |
| paper_quantitative_package/CONSISTENCY_AUDIT.md:75 | non-significance != equivalence | → 효과 없음 ✔ | → 효과 검출 안 됨(동등성 미검정) |
| paper_quantitative_package/CONSISTENCY_AUDIT.md:457 | stale; contradicts review_followup3 P2 | 단 A−에서는 등속 외삽 baseline과 구별되지 않는다 | 단 등속 외삽 초과는 A−(실패 선택 층)에서만 성립한다(P2) |
| paper_quantitative_package/SOURCE_MAP.md:291 | evaluation set vs population | 상시 적용 모집단 1.24 → 2.60% | 상시 적용 dev 평가 집합 재가중 A− 1.24 → 2.60% |
| review_followup3/CLAIM_EVIDENCE_MATRIX.md:16 | verified path != statistical validity | \| VERIFIED_MACHINERY_MANIFEST_SCOPE \| usable with the manifest-scope caveat \| | \| VERIFIED_MACHINERY_MANIFEST_SCOPE \| intervention code path verified on the manifest; NOT a certification of the estimand or interval (see review_followup4 A1) \| |
| review_followup3/CLAIM_EVIDENCE_MATRIX.md:47 | F = 18 drives | On 24 new drives (preregistered) the reduction was −34.2 pp | In the P1-C failure stratum (preregistered; 105 scenes, 18 new drives) the reduction was −34.2 pp |
| review_followup3/CLAIM_EVIDENCE_MATRIX.md:49 | intervention sample 353 weighted | On the P1-C evaluation frame (2,191 eligible scenes, 24 drives) | On the P1-C evaluation frame (2,191 eligible scenes; 353 sampled, design-weighted; 24 drives) |
| review_followup3/CLAIM_EVIDENCE_MATRIX.md:56 | all = 208; non-significance | Across A+ scenes and all scenes it was not distinguishable from motion extrapolation | Across A+ scenes and the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested) |
| review_followup3/CLAIM_EVIDENCE_MATRIX.md:70 | opposite point estimates != heterogeneity | Opposite signs cancel up to 100 % | Opposite-signed task point estimates (P1-D r4e1 C−F −4.2 / +4.2 pp; heterogeneity −8.3 pp [−21.7, +5.0]); heterogeneity neither detected nor excluded |
| review_followup3/CLAIM_EVIDENCE_MATRIX.md:75 | OpenVLA 92.6% exact definition | moved a later dimension by ≥ 8 bins in 92.6 % of units | changed at least one later component (y..yaw; gripper excluded) by ≥ the effective injected shift (8 bins unless clipped) in 92.6 % of units |
| review_followup3/CP2_REPORT.md:45 | all = 208; non-significance | In A+ and over all scenes it is not distinguished | In A+ and over the 208 selected mechanism scenes no excess was detected (equivalence not tested). |

Counts by rule: OpenVLA 92.6% exact definition: 8; non-significance != equivalence: 7; F = 18 drives: 7; opposite point estimates != heterogeneity: 6; A1 CI validity: 5; motion-baseline all = 208: 5; all = 208: 4; evaluation set vs population: 4; motion-baseline all = 208 selected: 4; F = 105 scenes / 18 drives: 2; evaluation-population: 2; A-/severe exact formula: 2; commanded delta vs actual TCP: 2; set/denominator: 1; P1-C intervention sample 353 weighted: 1; drive- vs day/vehicle-disjoint: 1; stale: 1; verified path != statistical validity: 1; intervention sample 353 weighted: 1
