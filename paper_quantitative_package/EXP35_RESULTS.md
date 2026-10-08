# Experiment 35 — results (SpatialVLA feedback-protection ablation)

**출처**
- 사전 등록: commit **8fc20bf** (`O/spatialvla_feedback_protection_ablation/PROTOCOL.md`).
- 분석 스크립트: `analyze_protection.py` (commit 44266eb, 결과를 보기 전).
- 결과: commit **2b0212d**.
- 원 보고서: `O/spatialvla_feedback_protection_ablation/RESULTS.md`.
- 통계 출처: seed 0–39 전 설정은 `analysis.json`, r1e4·r4e1의 seed 0–119 합동은 `pooled_seed0_119/analysis.json`.

**통계 방법**
- CI: 과제로 층화한 paired bootstrap, 2,000회, seed 0.
- p: 성공률은 exact McNemar, 궤적은 Wilcoxon.
- 보조 설정(r1e2, r1e1, r2e2, r2e1)은 Holm 보정.
- 설정 간 대비 차이: sign-flip permutation 10,000회.

**판정: Partial support.** 재계획 간격을 늘리면 문맥 feedback의 궤적 오차가 커지고 오래 남습니다. 그러나 task 수준에서는 교정 효과가 없고, reverse 효과만 4 step 재계획에서 작게 나타납니다.

## Protocol notes
- **실제 실행 가중치** (ActionEnsembler, temp −0.8): 현재 chunk step 1 57%, 이전 chunk step 2 26%, step 3 12%, step 4 5%. 실험 34 문서의 반대 설명은 오류입니다.
- **ensemble을 보호 장치로 가정하지 않았습니다.** 측정된 translation feedback 노출(실행 action 중 step ≥ 2 예측의 가중치, `ages` 기록에서 계산)은 다음과 같습니다.

  | 설정 | r1e4 | r1e2 | r1e1 | r2e1 | r2e2 | r4e1 |
  |---|---|---|---|---|---|---|
  | 노출 | 0.426 | 0.31 | 0 | 0.50 | 0.655 | 0.75 |

- **설정 동치:** r4e4 ≡ r4e1, r2e4 ≡ r2e2 (4-step chunk라서).
- **seed 추가 규칙 발동:** r4e1의 C − F와 N − R가 둘 다 +1.2%p였고 p = 1이었습니다. 그래서 r1e4·r4e1을 seed 40–119로 확장했습니다(1,280 에피소드; PROTOCOL의 "+640"은 계산 표기 오류).
- **CI 방법 차이:** 실험 34는 층화하지 않은 bootstrap을 썼으므로 CI 방법이 다릅니다.

## 최종 표 (seed 0–39; 각 80 에피소드)

Feedback − Corrected < 0과 Reverse − Natural < 0이 가설 방향입니다.

| configuration | replan_interval | ensemble_size | feedback_exposure | natural_success | feedback_success | corrected_success | reverse_success | feedback_minus_corrected [CI], p | reverse_minus_natural [CI], p | trajectory_divergence F 대 C / R 대 N (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| r1e4 | 1 | 4 | 0.43 | 86.2 | 65.0 | 70.0 | 86.2 | −5.0 [−16.2, +6.2], 0.52 | 0.0 [−10.0, +8.8], 1 | 0.302 / 0.241 |
| r1e2 | 1 | 2 | 0.31 | 85.0 | 70.0 | 68.8 | 86.2 | +1.2 [−10.0, +13.8], 1 (Holm 1) | +1.2 [−7.5, +10.0], 1 (Holm 1) | 0.282 / 0.200 |
| r1e1 | 1 | 1 | 0 | 85.0 | 67.5 | 68.8 | 80.0 | −1.2 [−13.8, +11.2], 1 (Holm 1) | −5.0 [−12.5, +2.5], 0.34 (Holm 1) | 0.420 / 0.190 |
| r2e2 | 2 | 2 | 0.66 | 82.5 | 77.5 | 77.5 | 83.8 | 0.0 [−10.0, +10.0], 1 (Holm 1) | +1.2 [−10.0, +12.5], 1 (Holm 1) | 0.243 / 0.293 |
| r2e1 | 2 | 1 | 0.50 | 87.5 | 76.2 | 68.8 | 85.0 | +7.5 [−5.0, +18.8], 0.31 (Holm 1) | −2.5 [−11.2, +6.2], 0.79 (Holm 1) | 0.380 / 0.276 |
| r4e1 | 4 | 1 | 0.75 | 82.5 | 73.8 | 75.0 | 81.2 | −1.2 [−12.5, +10.0], 1 | −1.2 [−11.2, +7.5], 1 | 0.396 / 0.344 |
| 재실행 noise (natural_gen) | | | | r1e4 성공 불일치 7.5%, r4e1 12.5% | | | | | | 0.160 / 0.174 (gen 대 N) |

## 주 분석 — seed 0–119 합동 (각 240 에피소드)

| configuration | natural | feedback | corrected | reverse | feedback_minus_corrected [CI], p | reverse_minus_natural [CI], p |
|---|---|---|---|---|---|---|
| r1e4 | 82.9 | 67.1 | 60.0 | 82.1 | +7.1 [+0.4, +14.2], 0.060 (45 vs 28) | −0.8 [−6.2, +4.6], 0.88 |
| r4e1 | 81.7 | 69.6 | 65.0 | 75.4 | +4.6 [−2.5, +11.7], 0.22 | **−6.2 [−12.1, −1.2], 0.040** (16 vs 31) |
| r4e1 − r1e4 (상호작용) | | | | | −2.5 [−12.1, +7.1], sign-flip p = 0.66 | −5.4 [−13.3, +2.1], sign-flip p = 0.20 |

## 오차 지속 (r4e1 − r1e4, 합동, paired Wilcoxon)

| 대비 | 평균 위치 오차 | 궤적 발산 | 최종 오차 | 재정렬 시점 |
|---|---|---|---|---|
| reverse 대 natural | +1.84 cm [+1.26, +2.46], p = 1e-10 | +0.116 m [+0.075, +0.155], p = 7e-11 | +2.66 cm, p = 2e-6 | +10.9 step, p = 1e-6 |
| feedback 대 corrected | +1.32 cm [+0.66, +1.92], p = 2e-4 | +0.078 m [+0.017, +0.135], p = 9e-6 | +1.37 cm, p = 0.002 | +10.6 step, p = 3e-7 |

seed 0–39 보조 설정의 r1e4 대비 차이:
- r2e1: feedback 대 corrected 재정렬 +13.3 step [+6.3, +20.1], p = 5e-4.
- r1e1: feedback 대 corrected 평균 오차 +1.76 cm, p = 0.028.
- r1e2, r2e2: 유의한 차이 없음.

반감기: 모든 설정에서 오차 곡선이 t = 24 값의 절반 아래로 내려가지 않았습니다. 교란 구간이 끝난 뒤에도 오차는 계속 커집니다.

## 사전 등록 기준

| 기준 | 판정 |
|---|---|
| 1. 공식 설정에서 task 효과 없음 | 충족 (C − F p = 0.06, 가설과 반대 방향; N − R p = 0.88) |
| 2. 보호를 줄인 설정에서 Feedback − Corrected 효과가 유의하게 커짐 | **미충족** |
| 3. reverse가 natural보다 실패를 유의하게 늘림 | r4e1에서만 충족 (p = 0.04), 상호작용은 n.s. |
| 4. 보호를 줄일수록 오차 지속 증가 | 충족 (재계획 축) |

→ **Partial support.**

## 논문 문장
- **써도 되는 문장:** "Lengthening the replanning interval from 1 to 4 steps makes context-induced trajectory deviations larger and longer-lived (reverse vs. natural: +1.8 cm mean position error, p = 1e-10; realignment +11 steps), but task success changes little: only the reverse condition at 4-step replanning reaches −6.2 pp [−12.1, −1.2] (p = 0.04), and the difference from official execution is not significant."
- **과장이 되는 문장:**
  - "Temporal ensembling protects against feedback."
  - "Removing protections makes SpatialVLA fail like AutoVLA."
  - "Context correction restores success." 결과는 반대 방향이었습니다.
