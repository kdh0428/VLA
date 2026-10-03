# 실험 28 (P1): 다른 autoregressive VLA에서의 기전 재현 — OpenVLA-7B / LIBERO

작성: 2026-10-03, 실행 전. 결과를 보고 정의·threshold를 바꾸지 않습니다.

## 모델·환경과 AutoVLA와의 대응

- 모델: `openvla/openvla-7b-finetuned-libero-spatial` (bf16, RTX 5090만 사용), LIBERO-Spatial 10 과제 × 초기 상태 0–9 = 100 에피소드.
  공식 평가 절차(run_libero_eval.py: 10 step 대기, 최대 220 step, 180도 회전, JPEG·Lanczos·0.9 center crop, gripper 정규화·반전)를
  TensorFlow 없이 재구현했고, 첫 관측에서 공식 `predict_action`과 같은 action을 내는 것을 확인했습니다.
- OpenVLA는 control step마다 action token 7개(x, y, z, roll, pitch, yaw, gripper)를 **차원 순서로 autoregressive하게** 생성하고,
  다음 control step의 입력에는 이전 action token이 들어가지 않습니다(관측 + 지시문만).
  따라서 AutoVLA의 "직전 action token → 다음 action token" feedback에 대응하는 것은 **한 step 안의 차원 간 조건화**이고,
  시간축의 영향은 환경(상태)을 통해서만 전달됩니다. 이 차이는 결과 해석에 그대로 반영합니다.

## 개입 (모두 greedy 디코딩, 개입 차원 d의 원래 token bin g, 교란 bin p = clip(g + δ, 0, 254))

| 이름 | 실행 token(d) | 이후 차원이 보는 token(d) | AutoVLA 대응 |
|---|---|---|---|
| natural | g | g | 자연 디코딩 |
| feedback | p | p | 작은 첫 action 편차 (자연 feedback) |
| corrected | p | g | previous-action correction (실행은 교란, 조건은 정상) |
| reverse | g | p | 정상 실행에 잘못된 previous action 삽입 |
| window_w (w = 1–4) | p | 다음 w개 차원은 g, 그 이후는 p | 1/2/3/4-step correction window |

## Phase A — 자연 rollout (closed loop)

100 에피소드 자연 실행. 성공 여부, step별 token·엔트로피·margin 기록, 5 step마다 전처리 이미지(PNG) 저장.

## Phase B — closed-loop 교란

- 개입 차원 d = 0 (x 이동, 각 step의 첫 action token), control step 10–19 (10 step, 대기 step 제외) 동안 매 step 적용.
- |δ| ∈ {8, 24} bin, 부호는 초기 상태 번호가 짝수면 +, 홀수면 −. 조건 = {feedback, corrected, reverse} × |δ| = 6, 각 100 에피소드.
- 교란 구간의 매 step에서 같은 관측의 자연 디코딩을 함께 계산해, step 내 하위 차원 편차를 정확히 측정합니다.
- 지표: 과제 성공률(자연 대비), 자연 성공 에피소드 중 교란 후 실패 비율, 하위 차원(1–5) bin 편차.
- 통계: 에피소드 단위 paired 비교(McNemar exact), 과제 단위 cluster bootstrap 95% CI(2,000회, seed 0), paired permutation(과제 단위 sign-flip, 20,000회).

## Phase C — step 내 token feedback (시뮬레이션 없음)

- Phase A에서 저장한 관측(5 step마다) 전부에 대해: d ∈ {0, 1}, δ ∈ {±8, ±24}, 개입 = {natural, feedback, corrected, reverse, window_1..4}.
- 지표 (하위 차원 j > d, gripper 제외 j ≤ 5):
  - 하위 편차 D = Σ_j |bin_j − natural bin_j| (실행 token 기준)
  - **증폭**: max_j |Δbin_j| ≥ |δ| (주입한 편차만큼 이상 하위 차원이 움직임)
  - **회복**: 모든 하위 차원 Δbin = 0
  - 차원 d+1의 엔트로피·greedy 확률 변화
- 질문 대응:
  1. 작은 첫 편차가 항상 실패로 이어지는가 → Phase B feedback의 실패 비율 (< 100%면 "항상은 아님")
  2. 같은 크기 교란에서 안정/불안정 분기 → Phase B 에피소드별 성공/실패 분기, Phase C 증폭/회복 분포
  3. previous-action correction 효과 → feedback vs corrected (Phase B 성공률, Phase C 증폭률)
  4. reverse intervention → reverse vs natural (실패율, 하위 편차)
  5. critical window → window_1..4 vs corrected / feedback (Phase C)

## 수정 (2026-10-03 19:40, Phase B 일부를 본 뒤·최종 분석 전에 추가)

Phase B 중간 점검에서 교란 조건이 성공하고 자연 실행이 실패한 에피소드가 있어, closed loop에서 batch 구성만 달라져도(bf16 수치 차이)
궤적이 갈라질 수 있음을 확인했습니다. 이를 측정하는 대조군을 추가합니다. 기존 정의·조건·threshold는 바꾸지 않습니다.
- **Phase A-rep**: 같은 100 에피소드를 자연 디코딩으로 다시 실행하되 worker 3개(다른 batch 구성)로 돌립니다.
- 보고: 자연 대 자연-rep의 성공 불일치율과 성공률 차이를 "교란 없는 기저 변동"으로 함께 보고하고,
  각 교란 조건의 자연 대비 차이를 이 기저 변동과 비교합니다(자연-rep 대비 paired 비교도 함께 계산).
