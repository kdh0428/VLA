# Experiment 35 — results template (결과 없음, 실행 중)

- 사전 등록: commit **8fc20bf** (`outputs/spatialvla_feedback_protection_ablation/PROTOCOL.md`).
- 원 데이터: `outputs/spatialvla_feedback_protection_ablation/rollouts/episodes.jsonl`. 분석 스크립트: `scripts/spatialvla_protection/analyze_protection.py`.
- **이 파일의 모든 결과 칸은 비어 있어야 합니다.** 실행이 끝나고 사전 등록된 분석을 돌리기 전에는 값을 채우지 않습니다.

## 실행 계획 (사전 등록 그대로)

| configuration | replan_interval | ensemble_size | 조건 | 에피소드 |
|---|---|---|---|---|
| r1e4 (공식) | 1 | 4 | natural / feedback / corrected / reverse + natural_gen | 5 × 80 |
| r1e2 | 1 | 2 | natural / feedback / corrected / reverse | 4 × 80 |
| r1e1 | 1 | 1 (없음) | 〃 | 4 × 80 |
| r2e2 | 2 | 2 (= 가능한 최대) | 〃 | 4 × 80 |
| r2e1 | 2 | 1 | 〃 | 4 × 80 |
| r4e1 | 4 | 1 (= ensemble-4, 각 step을 덮는 예측이 1개뿐) | 〃 + natural_gen | 5 × 80 |

- 계획된 에피소드는 총 **2,080개**입니다.
- 과제: SimplerEnv `google_robot_pick_coke_can`, `google_robot_move_near`, seed 0–39.
- 교란: d = 0.3, opposite, control step 8–23에 생성되는 chunk의 step-1 translation token.

## Protocol notes

1. **실제 실행 가중치** (`ActionEnsembler`, temp −0.8, i = 0이 가장 오래된 예측, w ∝ exp(0.8·i)):

   | 예측 | 가중치 |
   |---|---|
   | 현재 chunk의 step 1 | 57% |
   | 1 step 전 chunk의 step 2 | 26% |
   | 2 step 전 chunk의 step 3 | 12% |
   | 3 step 전 chunk의 step 4 | 5% |

   실험 34 PROTOCOL.md §0의 "오래된 예측일수록 가중치가 크다"는 설명은 틀렸습니다. 결과는 공식 코드를 썼으므로 영향이 없습니다.
2. **ensemble을 단순한 보호 장치로 가정하지 않습니다.** 매 step 재계획에 ensemble이 없으면 step 2–4가 실행되지 않습니다. ensemble은 feedback의 영향을 받은 step 2–4를 실행에 노출하는 통로(공식 설정에서 43%)이면서, 그 오차를 평균하는 장치이기도 합니다.
3. **현재 가설:** "Replanning and temporal ensembling jointly modulate how much feedback-affected future action is exposed to execution."
4. **feedback_exposure 정의 (사전 등록된 설정에서 계산되는 구조량이며, 결과가 아님):** 실행 action 중 문맥 교란 이후에 생성된 token(step-1 rotation·gripper, step 2–4)이 차지하는 가중치. translation 기준으로는 step 2–4 예측의 가중치 합입니다.

   | 설정 | translation feedback_exposure |
   |---|---|
   | r1e4 | 0.43 |
   | r1e2 | 1 step 전 chunk의 step 2 가중치 1/(1+e^0.8) ≈ 0.31 |
   | r1e1 | 0 (rotation·gripper만 노출) |
   | r2e1 | 실행 step의 1/2이 step 2 |
   | r2e2 | 짝수 step은 [step 1, 전 chunk의 step 3], 홀수 step은 [step 2, 전 chunk의 step 4] |
   | r4e1 | 실행 step의 3/4이 step 2–4 |

   이 값은 RESULTS 작성 때 analyze 스크립트의 실제 `ages`/`preds` 기록으로 확인한 뒤 채웁니다.
5. 사전 등록된 seed 추가 규칙(r4e1의 C − F와 N − R가 둘 다 > 0인데 하나라도 p ≥ 0.05이면 r1e4·r4e1을 seed 40–119로 확장)이 적용되면 이 표에 합동 결과 열을 추가합니다.

## 최종 표 schema (빈 칸 유지)

| configuration | replan_interval | ensemble_size | natural_success | feedback_success | corrected_success | reverse_success | feedback_minus_corrected | reverse_minus_natural | trajectory_divergence | feedback_exposure | CI | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1e4 | 1 | 4 | | | | | | | | | | |
| r1e2 | 1 | 2 | | | | | | | | | | |
| r1e1 | 1 | 1 | | | | | | | | | | |
| r2e2 | 2 | 2 | | | | | | | | | | |
| r2e1 | 2 | 1 | | | | | | | | | | |
| r4e1 | 4 | 1 | | | | | | | | | | |

열 정의:
- 성공률: x/80 (%).
- `feedback_minus_corrected`: 성공률 차이(%p). paired bootstrap 95% CI(과제 층화, 2,000회, seed 0), exact McNemar p, 보조 설정은 Holm 보정 p도 표기.
- `reverse_minus_natural`: 같은 방식.
- `trajectory_divergence`: feedback 대 corrected, reverse 대 natural의 실행 world_vector 누적합 끝점 차이(m). Wilcoxon.

추가로 채울 표:
- r4e1 대 r1e4의 대비 차이(bootstrap CI, sign-flip permutation).
- 오차 지속: 평균 위치 오차, 반감기, 재정렬 시점(τ = 2 cm, 4 step 연속).
- natural_gen 재실행 불일치(r1e4, r4e1).
- 과제별 결과.
- 실패 유형.
- 실험 34와의 비교(A).

## 논문에서 채울 위치

- `tables.tex` Table 3 (cross-model)의 SpatialVLA 행 각주.
- `PAPER_NUMBERS.md` Claim 11.
- `paper_cross_model_table.csv`의 "replanning frequency / temporal ensemble" 행 설명.
