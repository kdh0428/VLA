# 실험 28 (P1): 다른 autoregressive VLA에서의 기전 재현 — OpenVLA-7B / LIBERO-Spatial

프로토콜: `PROTOCOL.md` (커밋 6b9e8ad, 실행 전; 자연 재실행 대조군 수정 6d0375a는 Phase B 일부를 본 뒤·최종 분석 전에 추가).
모델 `openvla/openvla-7b-finetuned-libero-spatial`, RTX 5090만 사용, LIBERO 10 과제 × 초기 상태 0–9. 공식 평가 절차를 TensorFlow 없이 재구현했고
첫 관측에서 공식 `predict_action`과 같은 action을 냈습니다. 분석: `analysis.json` (`scripts/cross_vla/analyze_cross_vla.py`).

## 구조적 차이 (해석의 전제)

OpenVLA는 control step마다 action token 7개(x, y, z, roll, pitch, yaw, gripper)를 차원 순서로 autoregressive하게 생성하고,
**다음 step의 입력에는 이전 action token이 들어가지 않습니다**. AutoVLA의 "직전 action token → 다음 action token" feedback에 대응하는 것은
한 step 안의 차원 간 조건화뿐이고, 시간축 영향은 환경 상태를 통해서만 전달됩니다.

## 1. step 안의 token feedback (Phase C, 2,462 프레임, 시뮬레이션 없음)

개입 차원 d의 token을 δ bin 바꾸고, 이후 차원(≤ 5, gripper 제외)의 변화를 같은 관측의 자연 디코딩과 비교했습니다. 괄호는 과제 단위 bootstrap 95% CI.

| 개입 (d = 0, x) | 하위 차원 편차 합 D (bin) | 증폭 (하위 차원이 δ 이상 움직임) | 회복 (하위 변화 0) | d+1 엔트로피 변화 |
|---|---:|---|---:|---:|
| feedback, \|δ\| = 8 | 87.7 | **92.6% [89.8, 94.4]** | 1.5% | +0.087 |
| feedback, \|δ\| = 24 | 106.5 | 81.2% [77.2, 83.9] | 0.2% | +0.113 |
| corrected (실행은 교란, 문맥은 정상) | 0 | 0% | 100% | 0 |
| reverse (실행은 정상, 문맥만 교란) | feedback과 동일 | 동일 | 동일 | 동일 |
| window 1 / 2 / 3 / 4 (\|δ\| = 8) | 62.0 / 36.7 / 23.6 / 8.5 | 84.5 / 66.9 / 53.4 / 26.8% | 5.5 / 19.5 / 30.7 / 60.1% | – |

d = 1(y) 교란도 같은 양상입니다(feedback 증폭 87.4% / 75.3%, window_3에서 28.0% / 21.1%).

- **step 안의 token feedback은 OpenVLA에서도 강하게 나타납니다.** 작은 첫 token 편차(8 bin)가 같은 step의 나머지 차원을 93%에서 그 이상 움직입니다.
- **하위 차원을 결정하는 것은 실행된 token이 아니라 문맥에 들어간 이전 token**입니다: reverse(정상 실행 + 잘못된 문맥)는 feedback과
  하위 token이 완전히 같고, corrected(교란 실행 + 정상 문맥)는 하위 변화가 0입니다. AutoVLA의 "직전 token identity가 다음 token을 조건화"와 같은 구조입니다.
- **window**: 교정한 차원 수만큼 노출된 차원이 줄어 증폭이 단계적으로 감소하지만, 교정 뒤 교란을 다시 넣으면 남은 차원은 거의 그대로 발산합니다
  (window_1 후에도 84.5%). AutoVLA처럼 짧은 교정이 이후 발산을 막는 "지속 효과"는 차원 축에서 관찰되지 않았습니다.

## 2. closed-loop 과제 결과 (Phase A/B, 100 에피소드 × 조건, 교란은 control step 10–19에 매 step)

| 조건 | 성공률 | vs 자연 | vs 자연 재실행 | 자연 성공 중 실패 비율 |
|---|---:|---|---|---:|
| 자연 | 87% | – | – | – |
| **자연 재실행 (대조군, batch 구성만 다름)** | **81%** | −6%p, 결과 불일치 20% (p 0.26) | – | 15% |
| feedback \|δ\| 8 / 24 | 82 / 84% | −5 / −3%p (n.s.) | +1 / +3%p | 13 / 13% |
| corrected \|δ\| 8 / 24 | 79 / 79% | −8 / −8%p (n.s.) | −2 / −2%p | 16 / 18% |
| reverse \|δ\| 8 / 24 | 84 / 79% | −3 / −8%p (n.s.) | +3 / −2%p | 9 / 17% |

과제 단위 bootstrap CI와 과제 단위 permutation, McNemar 모두 어떤 교란 조건도 유의하지 않습니다(p ≥ 0.12).

- **작은 첫 action 편차가 항상 실패로 이어지지는 않습니다.** 자연 성공 에피소드 중 교란 후 실패는 9–18%이고, 이는 교란 없이 다시 돌렸을 때의
  실패 비율(15%)과 같은 수준입니다. 같은 크기의 교란에서 76개 에피소드는 회복하고 11개는 실패해 분기가 있지만, 이 분기는 교란 없이도 생기는
  closed-loop 수치 변동(bf16, batch 구성)과 구별되지 않고, 교란 구간의 step 내 하위 편차 크기로도 예측되지 않습니다(실패 57.9 vs 회복 68.3 bin, δ 24).
- **previous-action correction과 reverse intervention의 과제 수준 효과는 검출되지 않습니다**(feedback − corrected +3 ~ +5%p, n.s.).

## 결론

> AutoVLA에서 발견한 "직전 token identity가 다음 token을 조건화하는" 국소 autoregressive feedback은 OpenVLA의 step 안(차원 간)에서 그대로 나타납니다.
> 그러나 OpenVLA는 이전 action을 다음 step에 다시 넣지 않으므로 그 feedback이 시간축으로 누적되지 않고, LIBERO closed loop는 step 수준의 token 오차를
> 흡수합니다. 이 규모(100 에피소드)에서는 과제 실패에 대한 효과가 closed-loop 기저 변동 안에 있습니다.

즉 증폭 기전의 **token 수준 구성요소는 모델 간에 일반적**이지만, 그것이 **궤적 실패로 이어지려면 AutoVLA처럼 여러 시점의 action을 한 번에
autoregressive하게 생성하는 구조(시간축 action history)가 필요**하다는 해석과 일치합니다.

한계: LIBERO-Spatial 한 suite, 100 에피소드; 교란은 x(또는 y) 한 차원·10 step; π0-FAST처럼 여러 시점의 action chunk를 autoregressive하게
생성하는 모델에서는 시간축 feedback을 직접 시험할 수 있으나 이번에는 다루지 않았습니다.
