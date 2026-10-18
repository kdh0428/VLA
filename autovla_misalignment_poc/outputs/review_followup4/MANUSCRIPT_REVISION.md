# MANUSCRIPT_REVISION — review_followup4 round (2026-10-18)

**No manuscript `.tex` exists in the repository.** `rg --files -g '*.tex' /root/VLA` finds only `paper_quantitative_package/{tables.tex, paper_cross_model_table.tex, paper_detection_table.tex, paper_navhard_table.tex}` (and third-party LaTeX in vendored packages). So no manuscript was edited. This file gives (1) sentence-level replacements for every A4 wording violation, (2) the A1/A2/A3/B1/B2/B3-driven sentence changes, and (3) the machine-generated tables. The paper-ready KEEP / DELETE text is in `paper_quantitative_package/MANUSCRIPT_REVISION.md`, section "2026-10-18 revision (review_followup4)"; the verdicts are in `CLAIM_CHANGES.md`.

**What was edited in the repository** (manuscript-facing documents only; result files under `outputs/review_followup*/` untouched):
- `/root/VLA/EXPERIMENT_SUMMARY.md`: new §6g (Korean) and in-place corrections, each line marked "(2026-10-18 수정)".
- `/root/VLA/paper_quantitative_package/`: `PAPER_NUMBERS.md` (2026-10-18 section, Claims 25–27, in-place fixes), `MANUSCRIPT_REVISION.md` (banner + 2026-10-18 revision section, in-place fixes), `README.md` (note), `PAPER_EVIDENCE.md`, `EXP35_RESULTS.md`, `CONSISTENCY_AUDIT.md`, `SOURCE_MAP.md` (in-place fixes), `tables.tex` (hand-written lines: end-of-line comment "corrected 2026-10-18"; machine block regenerated), `numbers.json` / `numbers_sources.csv` (+1,532 RF4 rows).
- Not edited: `/root/VLA/review_followup3/*.md` (historical round reports; 7 A4 items listed below with their replacements, superseded by `CLAIM_CHANGES.md`), `CONCLUSIONS.md` (outside the allowed set; its window sentences l. 213–214, 217, 374 are covered by A2 rule R18 below), code docstrings.
- Every replaced text is logged in `DOC_EDITS_APPLIED.csv` (file, line, old, new, source).

## 1. A4 wording violations (64 items; file:line refers to the files before this round's edits)

Status column: **applied** = replaced in place (line marked); **c_tables** = applied by the table generator; **not edited** = historical report, use the replacement when quoting.

| # | file:line | rule | current text | replacement | status |
|---|---|---|---|---|---|
| 0 | `EXPERIMENT_SUMMARY.md:11` | A1 CI validity | p·CI는 cluster sign-flip(역산 CI) | p는 cluster sign-flip(δ = 0에서 유효)이고 CI는 사후 수정 구간(Rao–Scott design-effect Newcombe 10, 과제 평균은 MOVER; 원 sign-flip 역산 CI는 이진 쌍 자료의 경계에서 무효, review_followup4 A1) | applied |
| 1 | `EXPERIMENT_SUMMARY.md:26` | set/denominator; F = 105 scenes / 18 drives; evaluation-population | 새 24 drive**(P1-C)에서 재현: 실패 조건부 RN −34.2%p | 기전 실험과 drive가 겹치지 않는 **새 drive**(P1-C; 0 drive 공유, day–vehicle 10 공유)에서 재현 [사전 등록]: 실패 층 F(105 장면 / 18 drive) RN −34.2%p, RV +26.8%p; 평가 집합 E 추정량(적격 2,191 장면, 선정 353 장면 가중, 24 drive) −4.1 / +3.6%p. RN 평균 효과의 약 60%는 A+(비실패) 층에서 오므로 원래 실패 장면의 복구로만 설명하지 않습니다. | applied |
| 2 | `EXPERIMENT_SUMMARY.md:34` | motion-baseline all = 208 selected; non-significance != equivalence | 전체에서는 +0.20 m(p = 0.51)로 외삽과 구별되지 않습니다 | 선정 기전 집합 전체(208 장면 = A− 52 + A+ 156)에서는 +0.20 m [−0.32, +0.56](p = 0.51)로 차이가 검출되지 않았습니다(동등성 미검정) | applied |
| 3 | `EXPERIMENT_SUMMARY.md:36` | OpenVLA 92.6% exact definition; units | 나머지 차원이 93%에서 그 이상 움직이고 | 한 step 안에서 x token을 ±8 bin(명령 약 2.6 mm/step) 바꾸면 단위의 92.6%(4,741 단위; 초기 상태 군집 CI [90.8, 94.0])에서 하위 성분(y, z, roll, pitch, yaw; gripper 제외) 중 적어도 하나의 bin 변화가 실제 주입 bin 이상이었고(translation 성분만으로는 72%; 명령값이며 실현 운동·과제 실패가 아님) | applied |
| 4 | `EXPERIMENT_SUMMARY.md:38` | non-significance != equivalence; 20% rerun disagreement is not a detection threshold | closed-loop 기저 변동(자연 재실행끼리도 결과 20% 불일치) 안에 있었습니다 | LIBERO 과제 성공률 효과는 검출되지 않았고(C − F 8-bin −3%p [−11.9, +5.9], Newcombe 10; 동등성 미성립), 자연 재실행끼리도 결과가 20% 달랐습니다(이 20%를 효과 검출 임계값으로 쓰지 않음) | applied |
| 5 | `EXPERIMENT_SUMMARY.md:88` | OpenVLA 92.6% exact definition | 하위 차원 93% 증폭 | 하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위 92.6%(기술적) | applied |
| 6 | `EXPERIMENT_SUMMARY.md:157` | evaluation set vs population | 모집단 A− 1.24 → 2.60%로 악화 | dev 자연 평가 집합(2,747 장면: A− 52 / A+ 2,695)으로 재가중한 A− 빈도 1.24 → 2.60%로 악화(배포 빈도 아님) | applied |
| 7 | `EXPERIMENT_SUMMARY.md:211` | OpenVLA 92.6% exact definition; units | 하위 차원이 δ 이상 움직이는 비율 **92.6% [89.8, 94.4]** | 하위 성분(y..yaw; gripper 제외) 중 적어도 하나의 bin 변화 절댓값이 실제 주입 bin(clip 후) 이상인 단위 비율(4,741 단위 = 2,462 프레임 × 부호; bin은 m·rad 혼합) **92.6% [89.8, 94.4]** | applied |
| 8 | `EXPERIMENT_SUMMARY.md:280` | opposite point estimates != heterogeneity | pooled 평균에서 상쇄됩니다 | 지만 과제 간 차이는 검정하지 않았고 과제별 CI가 넓어, 이질성이나 상쇄가 확인된 것은 아닙니다 | applied |
| 9 | `EXPERIMENT_SUMMARY.md:361` | motion-baseline all = 208 selected | 전체 +0.20 m [−0.32, +0.56] (p = 0.51) | 선정 208 장면 전체(A− 52 + A+ 156) +0.20 m [−0.32, +0.56] (p = 0.51; 동등성 미검정) | applied |
| 10 | `EXPERIMENT_SUMMARY.md:364` | F = 105 scenes / 18 drives; P = 353 weighted | RN F −34.2 [−38.3, −29.5] / P −4.1 | RN F(105 장면 / 18 drive) −34.2 [−38.3, −29.5] / P(E 2,191 적격 장면, 353 선정 장면 가중, 24 drive) −4.1 | applied |
| 11 | `EXPERIMENT_SUMMARY.md:421` | evaluation-population | **모집단 추정량.** | **평가 집합(E) 추정량.** (이전 표기 '모집단 추정량'; P는 선정 353 장면의 설계 가중 추정) | applied |
| 12 | `EXPERIMENT_SUMMARY.md:472` | opposite point estimates != heterogeneity | 과제 효과가 반대 부호로 평균에서 100% 상쇄됩니다 | 과제별 점추정이 −4.2 / +4.2%p로 부호가 반대지만, 과제 간 차이 −8.3%p [−21.7, +5.0](p = 0.23)는 0을 포함하므로 이질성·상쇄는 확인되지 않았습니다 | applied |
| 13 | `EXPERIMENT_SUMMARY.md:488` | motion-baseline all = 208 selected | \| 전체 \| CM +0.20 (n.s.) | \| 선정 208 장면 전체 (A− 52 + A+ 156) \| CM +0.20 (n.s.; 동등성 미검정) | applied |
| 14 | `EXPERIMENT_SUMMARY.md:490` | all = 208; non-significance != equivalence | A+와 전체에서는 운동 외삽과 구별되지 않습니다 | A+와 선정 208 장면 전체에서는 운동 외삽과의 차이가 검출되지 않았습니다(동등성 미검정) | applied |
| 15 | `EXPERIMENT_SUMMARY.md:542` | F = 18 drives | 새 24 drive에서 재현 | 새 drive의 실패 층(105 장면 / 18 drive)에서 재현 | applied |
| 16 | `EXPERIMENT_SUMMARY.md:567` | A-/severe exact formula; 3 s thresholds applied at 5 s | 평가 horizon은 5 s(10 token)이며 | 평가 horizon은 5 s(10 token)이며, 그 이상은 모델 분포 밖입니다. A−는 P/R/A 행동 범주 판정(pra_labels.label_A: 종방향 v_end/v_start 1.25/0.75, 저속 0.5 m/s·가속 0.8 m/s, STOP = 경로 < 1.0 m & v_end < 0.5 m/s, 허용 max(0.5 m/s, 15%); 횡방향 최종 횡변위 > 2.0 m 또는 마지막 step heading > 15°, 허용 ±1 m/±10°)이고, 증폭(severe) = A− ∧ FDE5 > 3 m입니다. 이 임계값은 3 s horizon(HORIZON = 6)용으로 문서화되었지만 10 pose(5 s) 계획에 그대로 적용됐습니다(원 정의 보존; 3 s 민감도는 P/R/A 비교에만 있음; review_followup4 A4). | applied |
| 17 | `EXPERIMENT_SUMMARY.md:575` | opposite point estimates != heterogeneity | 과제별로 반대 부호 효과가 일부 상쇄됩니다 | 과제별 점추정의 부호가 일부 반대지만 과제 간 이질성은 검출되지도 배제되지도 않았습니다 | applied |
| 18 | `EXPERIMENT_SUMMARY.md:47` | A1 CI validity | 과제별로는 CI 폭 최대 18–29%p(P1-D r4e1)로 불확실 | 과제별로는 CI 폭 최대 18–25%p(P1-D r4e1; r1e4 13–20%p; A1 사후 수정 구간)로 불확실**합니다(2026-10-15 판의 '18–29%p, r1e4 6.9–21.6%p'는 경계에서 무효인 sign-flip 역산 CI 기준) | applied |
| 19 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:11` | F = 105 scenes / 18 drives; drive- vs day/vehicle-disjoint; cohort of the 40 pp | Both effects replicated in a preregistered test on 24 new drives that share no drive with the development set | Both effects replicated in a preregistered test on new drives (failure stratum: 105 scenes, 18 drives; no drive shared with the development set, but days and vehicles partly shared). | applied |
| 20 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:17` | F = 18 drives; evaluation-population; 60% A+ | A preregistered replication on 24 new drives (day/vehicle partly shared) reproduces both contrasts (failure-conditional −34.2 / +26.8 pp; population −4.1 / +3.6 pp) | A preregistered replication on new drives (days/vehicles partly shared) reproduces both contrasts: failure stratum (105 scenes, 18 drives) −34.2 / +26.8 pp; evaluation-frame estimand over 2,191 eligible scenes (353 sampled, weighted; 24 drives) −4.1 / +3.6 pp, about 60% of the RN effect coming from the non-failure (A+) stratum | applied |
| 21 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:18` | motion-baseline all = 208; non-significance != equivalence | across all scenes it is not distinguishable from extrapolation (+0.20 m, p = 0.51) | across the 208 selected mechanism scenes (52 A− + 156 A+; not the natural mix) no difference from extrapolation was detected (+0.20 m [−0.32, +0.56], p = 0.51; equivalence not tested) | applied |
| 22 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:49` | motion-baseline all = 208 | across all scenes it was not distinguishable from extrapolation | across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested) | applied |
| 23 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:58` | F = 18 drives | "replicated on 24 new drives (day/vehicle partly shared)" | "replicated on new drives (failure stratum 105 scenes / 18 drives; evaluation frame 24 drives; days and vehicles partly shared)" | applied |
| 24 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:83` | F = 18 drives | both effects replicated in a preregistered test on 24 new drives (−34.2 and +26.8 pp) | both effects replicated in a preregistered test on new drives (failure stratum, 105 scenes / 18 drives; −34.2 and +26.8 pp; days and vehicles partly shared) | applied |
| 25 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:87` | P1-C intervention sample 353 weighted; 60% A+ | On the evaluation population of the new-drive logs (2,191 eligible scenes) | On the evaluation frame of the new-drive logs (2,191 eligible scenes; estimated from 353 sampled scenes with design weights, 24 drives), the corresponding effects were −4.1 pp [−6.4, −2.4] and +3.6 pp [+2.2, +5.5] (about 60% of the RN effect from non-failure scenes); | applied |
| 26 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:97` | all = 208; non-significance | across all scenes and in non-failure scenes it was not distinguishable from motion extrapolation | in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested) | applied |
| 27 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:121` | OpenVLA 92.6% exact definition; units | moved a later dimension by at least 8 bins in 92.6% of units | an 8-bin x-token shift (≈ 2.6 mm commanded) changed at least one later component (y, z, roll, pitch, yaw; gripper excluded) by at least the effective injected number of bins in 92.6% of 4,741 units (initial-state-cluster CI [90.8, 94.0]; translation components alone 72%) | applied |
| 28 | `paper_quantitative_package/MANUSCRIPT_REVISION.md:148` | F = 18 drives | "replicated on 24 new drives (days and vehicles partly shared) | "replicated on new drives (failure stratum 105 scenes / 18 drives; days and vehicles partly shared); no external (day/vehicle-disjoint) validation was possible" | applied |
| 29 | `paper_quantitative_package/tables.tex:37` | A-/severe exact formula; 3 s thresholds at 5 s | Amplification = A$-$ and FDE$_{5\,s}>3$\,m. | Amplification = A$-$ and FDE$_{5\,s}>3$\,m, where A$-$ = failure of the coarse action-category rule (longitudinal class from start/end speed with tolerance $\max(0.5$\,m/s$, 15\%)$; lateral class from final offset $>2$\,m or final heading $>15^\circ$, tolerance $\pm1$\,m/$\pm10^\circ$) applied to the 5\,s plan with thresholds originally specified for a 3\,s horizon. | applied |
| 30 | `paper_quantitative_package/tables.tex:51` | drive- vs day/vehicle-disjoint; F cohort | 0 drives shared with this set | New drives (P1-C, pre-registered; failure stratum 105 scenes / 18 drives; 0 drives but 10 day--vehicle combinations shared with this set): RN $-34.2$ [$-38.3$, $-29.5$] | applied |
| 31 | `paper_quantitative_package/tables.tex:69` | OpenVLA 92.6% exact definition | Normal amplification & 47.4 [38, 57] & 63.4 [60.6, 66.7] & within-step 92.6 | within-step, $\ge$1 later component $\ge$ injected bins: 92.6 [89.8, 94.4] (task CI; state CI [90.8, 94.0]; bins, not physical units) | applied |
| 32 | `paper_quantitative_package/tables.tex:174` | A1 CI validity | pp with sign-flip-inversion 95\% CI on initial-state clu | pp with original sign-flip-inversion 95\% CI on initial-state clusters (invalid at the boundary for paired binary data; post-hoc corrected Rao--Scott Newcombe CIs in Table~\ref{tab:rf4_ci}) | applied |
| 33 | `paper_quantitative_package/tables.tex:187` | A1 CI validity | task-level CIs span up to 18--29 pp (r4e1; r1e4 6.9--21.6 pp) | task-level CIs span up to 18--25 pp (r4e1; r1e4 13--20 pp; post-hoc corrected intervals, Table~\ref{tab:rf4_ci}) | applied |
| 34 | `paper_quantitative_package/tables.tex:230` | motion-baseline all = 208 selected | all & +0.20 [$-$0.33, +0.56] | row label 'all (208 selected = 52 A$-$ + 156 A+)'; caption: '\emph{all} is the selected mechanism set, not the natural set'; also rename column 'E' (learned extrapolator) to avoid confusion with P1-C target E | c_tables |
| 35 | `paper_quantitative_package/tables.tex:240` | A1 CI validity | pp, sign-flip-inversion 95\% CI on initial-state clusters | pp, post-hoc corrected 95% CI (Rao--Scott design-effect Newcombe) on initial-state clusters | c_tables |
| 36 | `paper_quantitative_package/PAPER_NUMBERS.md:85` | evaluation-population; 353 sampled | population estimand P (24 drives, Hajek-weighted) | evaluation-frame estimand P (2,191 eligible scenes; 353 sampled, Hajek-weighted; 24 drives) | applied |
| 37 | `paper_quantitative_package/PAPER_NUMBERS.md:91` | F = 18 drives | (13 drives; replicated on 24 new drives) | (A− stratum, 52 scenes / 13 drives; replicated in the P1-C failure stratum, 105 scenes / 18 new drives) | applied |
| 38 | `paper_quantitative_package/PAPER_NUMBERS.md:180` | OpenVLA 92.6% exact definition | amplifies a downstream dimension in **92.6% [89.8, 94.4]** | changes at least one downstream component (y..yaw; gripper excluded) by at least the effective injected bin shift (bins mix m and rad; translation components alone 72%) in **92.6% [89.8, 94.4]** | applied |
| 39 | `paper_quantitative_package/PAPER_NUMBERS.md:185` | non-significance != equivalence | no task-level effect distinguishable from re-run variability | no detected task-level effect (C − F −3 pp [−11.9, +5.9]; equivalence not established), while re-running alone changed 20% of outcomes | applied |
| 40 | `paper_quantitative_package/PAPER_NUMBERS.md:197` | opposite point estimates != heterogeneity | opposite-signed effects that cancel in the pooled mean | opposite-signed task point estimates (heterogeneity not tested; task CIs wide), so the pooled mean may hide task differences | applied |
| 41 | `paper_quantitative_package/PAPER_NUMBERS.md:220` | non-significance != equivalence | OpenVLA shows no task effect distinguishable from rerun variability | OpenVLA showed no detectable task effect (equivalence not established) | applied |
| 42 | `paper_quantitative_package/PAPER_NUMBERS.md:328` | opposite point estimates != heterogeneity | coke S3 −9.2 vs move_near +8.3 partly cancel | coke S3 −9.2 vs move_near +8.3 (opposite point estimates; heterogeneity not established) | applied |
| 43 | `paper_quantitative_package/PAPER_NUMBERS.md:350` | motion-baseline all = 208 | **all scenes +0.20 m [−0.32, +0.56], p = 0.51** | **all 208 selected mechanism scenes (52 A− + 156 A+) +0.20 m [−0.32, +0.56], p = 0.51** | applied |
| 44 | `paper_quantitative_package/PAPER_NUMBERS.md:356` | motion-baseline all = 208 | across all scenes it was not distinguishable from extrapolation | across the 208 selected mechanism scenes no difference from extrapolation was detected (equivalence not tested) | applied |
| 45 | `paper_quantitative_package/PAPER_NUMBERS.md:405` | motion-baseline all = 208 | across all scenes and in non-failure scenes it was not disting | in non-failure scenes and across the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested) | applied |
| 46 | `paper_quantitative_package/PAPER_EVIDENCE.md:558` | OpenVLA 92.6% exact definition | 하위 차원 증폭이 92.6% | 하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위가 92.6% | applied |
| 47 | `paper_quantitative_package/PAPER_EVIDENCE.md:716` | non-significance != equivalence | closed-loop task 효과가 없다는 것은 | closed-loop task 효과가 검출되지 않았다는 것은(동등성 미검정) | applied |
| 48 | `paper_quantitative_package/PAPER_EVIDENCE.md:802` | evaluation set vs population | 모집단 A− 1.24 → 2.60%로 악화 | dev 평가 집합(52/2,695) 재가중 A− 1.24 → 2.60%로 악화 | applied |
| 49 | `paper_quantitative_package/PAPER_EVIDENCE.md:817` | evaluation set vs population | 모집단 가중 52/2,695 | dev 평가 집합 가중 52/2,695 | applied |
| 50 | `paper_quantitative_package/EXP35_RESULTS.md:18` | non-significance != equivalence | task 수준에서는 교정 효과가 없고 | task 수준에서는 교정 효과가 검출되지 않았고(동등성 미검정; 군집 후 불확실) | applied |
| 51 | `paper_quantitative_package/EXP35_RESULTS.md:56` | commanded delta vs actual TCP | 궤적 발산 | 실행 명령 누적 끝점 거리 (commanded, TCP 아님) | applied |
| 52 | `paper_quantitative_package/EXP35_RESULTS.md:36` | commanded delta vs actual TCP | trajectory_divergence F 대 C / R 대 N (m) | commanded_cumulative_divergence F 대 C / R 대 N (m; TCP 아님) | applied |
| 53 | `paper_quantitative_package/CONSISTENCY_AUDIT.md:47` | non-significance != equivalence; heterogeneity from per-task significance | 결론(문맥 효과 0, 실행 효과 있음) | 결론(문맥 효과 검출 안 됨, 실행 효과 있음)은 바뀌지 않지만, 과제별 점추정 차이(coke −37.5 vs move_near −2.5%p)는 이질성 검정 없이 '집중'이나 이질성으로 쓰지 않습니다. | applied |
| 54 | `paper_quantitative_package/CONSISTENCY_AUDIT.md:75` | non-significance != equivalence | → 효과 없음 ✔ | → 효과 검출 안 됨(동등성 미검정) | applied |
| 55 | `paper_quantitative_package/CONSISTENCY_AUDIT.md:457` | stale; contradicts review_followup3 P2 | 단 A−에서는 등속 외삽 baseline과 구별되지 않는다 | 단 등속 외삽 초과는 A−(실패 선택 층)에서만 성립한다(review_followup3 P2) | applied |
| 56 | `paper_quantitative_package/SOURCE_MAP.md:291` | evaluation set vs population | 상시 적용 모집단 1.24 → 2.60% | 상시 적용 dev 평가 집합 재가중 A− 1.24 → 2.60% | applied |
| 57 | `review_followup3/CLAIM_EVIDENCE_MATRIX.md:16` | verified path != statistical validity | \| VERIFIED_MACHINERY_MANIFEST_SCOPE \| usable with the manifest-scope caveat \| | \| VERIFIED_MACHINERY_MANIFEST_SCOPE \| intervention code path verified on the manifest; NOT a certification of the estimand or interval (see review_followup4 A1) \| | not edited |
| 58 | `review_followup3/CLAIM_EVIDENCE_MATRIX.md:47` | F = 18 drives | On 24 new drives (preregistered) the reduction was −34.2 pp | In the P1-C failure stratum (preregistered; 105 scenes, 18 new drives) the reduction was −34.2 pp | not edited |
| 59 | `review_followup3/CLAIM_EVIDENCE_MATRIX.md:49` | intervention sample 353 weighted | On the P1-C evaluation frame (2,191 eligible scenes, 24 drives) | On the P1-C evaluation frame (2,191 eligible scenes; 353 sampled, design-weighted; 24 drives) | not edited |
| 60 | `review_followup3/CLAIM_EVIDENCE_MATRIX.md:56` | all = 208; non-significance | Across A+ scenes and all scenes it was not distinguishable from motion extrapolation | Across A+ scenes and the 208 selected mechanism scenes no excess over motion extrapolation was detected (equivalence not tested) | not edited |
| 61 | `review_followup3/CLAIM_EVIDENCE_MATRIX.md:70` | opposite point estimates != heterogeneity | Opposite signs cancel up to 100 % | Opposite-signed task point estimates (P1-D r4e1 C−F −4.2 / +4.2 pp; heterogeneity −8.3 pp [−21.7, +5.0]); heterogeneity neither detected nor excluded | not edited |
| 62 | `review_followup3/CLAIM_EVIDENCE_MATRIX.md:75` | OpenVLA 92.6% exact definition | moved a later dimension by ≥ 8 bins in 92.6 % of units | changed at least one later component (y..yaw; gripper excluded) by ≥ the effective injected shift (8 bins unless clipped) in 92.6 % of units | not edited |
| 63 | `review_followup3/CP2_REPORT.md:45` | all = 208; non-significance | In A+ and over all scenes it is not distinguished | In A+ and over the 208 selected mechanism scenes no excess was detected (equivalence not tested). | not edited |

## 2. Sentence changes driven by A1 / A2 / A3 / B1 / B2 / B3

### A1 — paired success-rate intervals (post-hoc correction; originals kept)

| replace | with | rows |
|---|---|---|
| "OpenVLA C − N (8 bins): 95% CI [−16.7, 0.0] excludes 0" | "C − N 8 bins −8 pp, Newcombe 95% CI [−17.1, +1.0] (p = 0.115); not different from 0" | `RF4.A1.P0E.succ.C-N_d8` |
| any OpenVLA per-task CI, incl. "[0, 0]" / "no difference in task 5" | (delete) or "descriptive; 10 states per task" with the Newcombe interval | `RF4.A1.P0E.succ_task.*` |
| "R − N_rep (8 bins) equivalent within ±10 pp (unadjusted)" | "equivalence within ±10 pp not established (p = 0.052)" | `RF4.A1.P0E.succ.R-Nrep_d8` |
| "sign-flip-inversion 95% CI" (SpatialVLA Tables 7 / 10) | "post-hoc corrected Rao–Scott design-effect Newcombe 95% CI (MOVER for the two-task mean); original sign-flip-inversion CI in the supplement" | Table 13 |
| "task-level CIs span up to 18–29 pp (r4e1; r1e4 6.9–21.6 pp)" | "task-level CIs span up to 18–25 pp (r4e1; r1e4 13–20 pp)" | `RF4.A1.P0B.P1D.*` |
| "exp34 r1e4/perp_left coke C − F: equivalent within ±10 pp (Holm)" | "equivalent within ±10 pp only before multiplicity adjustment (corrected 95% CI [−11.3, +11.3], Holm p 0.15)" | `RF4.A1.chg.*` |
| "P1-D r2e1 − r1e1 C − F interaction (pooled, physical) equivalent within ±10 pp (Holm)" | "equivalence not established (Holm p 0.051)" | `RF4.A1.chg.*` |
| "P1-D r1e4 C − F move_near (physical) within ±5 pp; CI [−2.6, +4.3]" | (delete); corrected CI [−6.9, +8.6] | `RF4.A1.P0B.P1D.r1e4.C-F.move_near.physical` |
| "exp35 r1e4 C − F move_near: CI [−29.1, 0.0]" | "not different from 0 after Holm (p = 0.058); the corrected interval [−29.1, −0.3] barely excludes 0 — a boundary disagreement, not evidence of an effect" | `RF4.A1.P0B.exp35.r1e4.C-F.move_near.primary` |
| Methods: "p and CI by cluster sign-flip (inverted CI)" | "p by cluster sign-flip at δ = 0 (exact); CIs by Newcombe 10 (independent states) or Rao–Scott design-effect Newcombe (repeated states), chosen by a coverage rule fixed before recomputation (post hoc)" | – |

### A2 — correction window (GT persists; no release)

| replace | with |
|---|---|
| "Correcting three consecutive context steps recovered about 90% …" | "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect." + qualifiers (scene-weighted 90.1% [81.2, 98.9]; 365 units / 52 scenes / 13 drives; denominator Normal − GT-history = 41.7 pp; 2.96 GT positions per unit, 1.90 differing from own; mean 3.8 steps after the last added GT position) |
| "post-release (free) horizon", "free post-release steps", "after release", "고정 해제 후 horizon" | "steps after the last added GT position (GT retained)" |
| "교정을 멈추면 서서히 재발산", "해제 후 감쇠", "해제 후 재발산 25–29%", "re-divergence after release" | "re-divergence while the w corrected GT positions remain in the context (exp 10, summary only)" |
| "after a 3-step correction the GT information is removed and the effect persists"; "later generation reads only the model's own tokens"; "a short transient correction demonstrated a lasting recovery" | (delete) |
| Impromptu "교정 해제 후 재발산은 거의 없었습니다" | (delete) |
| (optional) | "When each of the three ground-truth tokens was shown only once, as the most recent context position, and then replaced by the model's own token, the recovery was 78% (95% CI 64–87; post-hoc comparison, 326 units, 12 drives)." |

### A3 — Impromptu

| replace | with |
|---|---|
| "Impromptu VLA strongly replicates the token-level temporal pattern" / "Strong replication" / "the three models replicate strongly" | "In Impromptu VLA 3B (open loop, descriptive; implementation parity not established), ground-truth context reduced and a reversed last context increased trajectory amplification by margins well above the 5–6% scene-level flip rate caused by batch and decoding-path numerics alone." |
| "no-op parity 23/300 / 237/2,728"; "comparisons are like for like" | "23/300 compares two decoding implementations and batches; 237/2,728 compares batch compositions; the same path and batch reproduce bit for bit (108/108)" |
| win1 −15.4 pp; direction − magnitude +15.6 pp; random − direction-ok +17.3 pp called "replicated" | "descriptive only (near the numerical floor; condition-correlated batching untested)" |

### B1 — numerical sensitivity

| add / replace | text |
|---|---|
| add (Methods / caption) | "Batch shape alone changed 9% of AutoVLA free-run suffixes and attention backend or fp32 arithmetic about 25%; the Recent-GT, Full-GT and Reverse-N contrasts were unchanged in direction and nearly unchanged in size." |
| replace any P1-A self-reference conclusion that rests on the sign of a near-zero FDE difference | "not distinguishable from numerical implementation variation (B1)" |

### B2 — matched prefix (new round)

| add / replace | text |
|---|---|
| add (Results) | K1b and the B2 part of K2' (`paper_quantitative_package/MANUSCRIPT_REVISION.md`, 2026-10-18 section); Table 12 |
| delete | "the effect requires a ground-truth history"; "pure provenance"; "online correction / deployment improvement"; "external replication"; "the correction was released"; pooling with the repeated runs |
| add (Limitations) | "The matched-prefix factorial uses one retained edit at a single anchor on existing drives; it is an oracle-conditioned diagnostic and does not test release or repeated schedules." |

### B3 — OpenVLA

| replace | with |
|---|---|
| "OpenVLA replicates …" / OpenVLA as core or verified evidence | "descriptive; intervention path not parity-verified" (K10') |
| "effects within the 20% rerun noise" used as a threshold | "no detected task-success effect (−3 pp [−11.9, +5.9]; equivalence not established); re-running alone changed 20% of outcomes" |

## 3. Machine-generated tables

- `paper_quantitative_package/numbers.json` — new rows `RF4.*` (1,532): `RF4.B2.*` (1,176: B2 contrasts and denominators, status `VERIFIED_IN_RUN_ALL_UNITS`, analysis tag preregistered / oracle-conditioned), `RF4.B1.*` (47: materiality), `RF4.A1.*` (213 corrected CI rows + 36 judgment changes; status NOT_VERIFIED for OpenVLA and exp 35, VERIFIED for P1-D), `RF4.A4.*` (25: cohort map, AUDIT), `RF4.A2.*`, `RF4.A3.*` (Impromptu, NOT_VERIFIED), `RF4.B3.OPENVLA_GATE` (NOT_RUN), `RF4.FEAS.pi0fast` (FEASIBILITY_ONLY). 781 existing rows carry `rf4_note`.
- `paper_quantitative_package/tables.tex`
  - between `% >>> BEGIN machine-generated review_followup3 (cp2_tables.py) >>>` and its END: Tables 8–11 regenerated with corrected captions / labels (P = evaluation frame from 353 sampled scenes, F = 18 drives; Table 9 "all (208 selected)", learned extrapolator renamed L; Table 10 marked "original sign-flip-inversion CI … corrected CIs in Table 13"; Table 11 B1 floor). Values unchanged.
  - new block `% >>> BEGIN machine-generated review_followup4 (c_tables.py) >>>`: **Table 12** B2 matched-prefix factorial (F / E4 / dev), **Table 13** post-hoc corrected CIs (SpatialVLA P1-D per task, OpenVLA state level, next to the original intervals), **Table 14** B1 numerical sensitivity.
- Regenerate: `scripts/review_followup4/c/c_numbers.py`, then `c_tables.py`.
