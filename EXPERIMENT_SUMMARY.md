# Driving VLA 실패 기전 연구 — 전체 실험 총정리

최종 갱신: 2026-10-05 (실험 33 추가). 브랜치 `autovla-experiments-11-19`. 실험별 상세 보고서는 각 `outputs/<실험>/` 디렉토리에 있고,
누적 결론 문서는 [CONCLUSIONS.md](CONCLUSIONS.md), 새 서버 인계는 [HANDOFF.md](HANDOFF.md)입니다.

- 주 대상: **AutoVLA** (Qwen2.5-VL-3B, action codebook 2,048 token, NAVSIM/nuPlan). 비교: ORION (Bench2Drive, 실험 1), **OpenVLA-7B** (LIBERO, 실험 28).
- 통계: 별도 표시가 없으면 [ ]는 log(또는 clip·과제) 단위 cluster bootstrap 95% CI, 이진 지표 McNemar, 연속 지표 Wilcoxon,
  navhard는 log 단위 paired sign-flip permutation과 그룹 단위 Wilcoxon을 함께 보고했습니다.
- 재현성 규칙: held-out·검증 실험은 결과를 보기 전에 프로토콜/사전 등록을 커밋했고(커밋 해시 표기), threshold와 규칙은 결과를 보고 바꾸지 않았습니다.
  기존 결과 파일은 덮어쓰지 않았습니다. 실험 25 이후의 모든 새 GPU 작업은 RTX 5090만 사용했습니다.

---

## 1. 한눈에 보는 결론

1. **실패는 perception보다 결정·실행 결합 단계에서 나고**, CoT 텍스트는 action과 느슨하게 결합되어 있습니다(실험 1–4).
2. **작은 첫 action-token 편차가 궤적 실패로 증폭되는 기전은 1-step autoregressive feedback**입니다: 직전에 생성한 action token이 residual stream을 통해
   다음 token을 조건화하고, 직전 token 1개만 GT로 바꿔도 효과의 93%가 사라집니다(실험 5–8). 첫 이탈 직후 2–4 step이 임계 window입니다(실험 9–10).
3. 그 직전 token에서 결과를 결정하는 것은 embedding이 아니라 **motion**, 그중에서도 **방향**입니다. 같은 크기 오차라도 자기 이탈과 같은 방향이면 증폭되고
   반대 방향이면 대부분 회복됩니다(실험 11, 32).
4. **다른 VLA(OpenVLA-7B)에서도 token 수준 feedback은 재현됩니다**: 한 step 안에서 첫 token을 8 bin 바꾸면 나머지 차원이 93%에서 그 이상 움직이고,
   이를 결정하는 것은 문맥에 들어간 이전 token입니다. 그러나 OpenVLA는 이전 action을 다음 step에 넣지 않아 feedback이 시간축으로 누적되지 않고,
   LIBERO 과제 실패에 대한 효과는 closed-loop 기저 변동(자연 재실행끼리도 결과 20% 불일치) 안에 있었습니다(실험 28).
5. **시간 순서가 있는 action chunk를 autoregressive하게 생성하는 다른 VLA(Impromptu VLA 3B)에서는 시간축 증폭 기전까지 재현됩니다**(사전 등록 판정 Strong):
   직전 waypoint 하나만 GT로 바꾸면 증폭 63.4 → 1.3%(98% 제거), 잘못된 직전 waypoint를 다시 넣으면 0 → 81.9%로 복원, 1/2/3/4-step 교정이 효과의
   24/39/91/95%(AutoVLA 47/76/90/96%), motion이 GT와 가까우면 identity와 무관, 방향 오답이 크기 오답보다 해롭습니다. 다만 Impromptu는 작은 편차를
   거의 흡수하지 못해(0.2 m 편차의 97%가 3배 이상 커짐) 궤적 수준의 안정 분기는 재현되지 않았습니다(실험 33).
6. **불안정 rollout은 이탈 전에는 예측되지 않습니다**: 엔트로피·likelihood·margin·후보 합의·hidden-state probe 모두 이탈 전 AUROC 0.57–0.63(장면 내 0.50–0.65),
   이탈 후 0.75–0.90입니다(실험 22, 31).
7. **GT 없는 완화**: 참조로 조건화를 교정하면 정상 장면이 손해를 봐 순이득이 없고(실험 12–17), 모델 자신의 후보 중 확신도로 고르면 open-loop 실패율 −33%,
   사전 등록 held-out PDMS +0.0095로 유의합니다(실험 20–21).
8. **반응형 2단계 평가(NAVSIM v2 navhard 전체 225 그룹)에서는 확신도 선택의 이득이 작고 불안정**하며(+0.02, max log-lik은 사전 등록 절반에서 미재현),
   **현재 frame 안전 필터가 이득의 대부분**을 만듭니다(EPDMS 0.229 → 0.333, +0.105; 선택기를 더해 0.347–0.350). 필터 이득의 86%는 주행 가능 영역 제약이고,
   진행도를 희생해 얻은 점수가 아닙니다(실험 25–27, 29).
9. **남은 병목은 후보 생성**입니다: 17개 후보 중 oracle을 골라도 EPDMS 0.405이고, token의 31%는 안전한 후보가 하나도 없습니다(실험 30).
10. 연속 closed-loop(CARLA) 평가는 checkpoint·렌더링·디스크 제약으로 수행하지 못했습니다(P6).

---

## 2. 실험 목록

| # | 실험 | 핵심 결과 | 보고서 |
|---|---|---|---|
| 1 | P/R/A 실패 분해 (ORION, AutoVLA) | 실패는 perception보다 결정·실행 결합 단계 | `pra_comparison/outputs/PRA_COMPARISON.md` |
| 2 | CoT 인과 개입 | 선언 방향을 뒤집어도 실행 반전은 +6.4%p, CoT는 느슨하게 결합 | `outputs/cot_intervention/` |
| 3 | Fast vs CoT (무편향 500 장면) | CoT 강제 시 A+ −5.0%p | `outputs/fast_vs_cot_unbiased/` |
| 4 | Natural/Fast 기전 재검증 | late action formation, L35 MLP 지배 재현 | `outputs/natural_fast_mechanism/` |
| 5 | 첫 mismatch 인과 | 첫 token 하나 교정으로 A− 장면 A+ 17 → 73% | `outputs/first_mismatch_causal/` |
| 6 | 등거리 perturbation | 같은 크기 편차가 장면에 따라 증폭(45%)/회복으로 갈림 | `outputs/equal_distance_perturbation/` |
| 7 | Action history 인과 | 직전 token 1개 교정이 효과의 93% (증폭 47 → 7%) | `outputs/action_history_causal/` |
| 8 | Layer-wise state patching | 특정 layer가 아니라 residual 전체가 운반 | `outputs/prev_action_state_patching/` |
| 9 | Temporal feedback window | 2–4 step 교정으로 효과의 90–96% | `outputs/temporal_feedback_window/` |
| 10 | Horizon-controlled window | horizon 착시가 아님, 단 해제 후 감쇠 | `outputs/horizon_controlled_window/` |
| 11 | 직전 token identity 분해 | embedding이 아니라 motion(codebook 기하)이 결정 | `outputs/prev_action_identity_decomposition/` |
| 12–17 | GT 없는 참조로 조건화 교정 | 모델이 참조를 66–89% 따라가 정상 장면 손해, 배포 조건 순이득 없음 | `outputs/*reference_stabilization*/` |
| 18–19 | Best-of-N 선택 | 조건화 대신 확신도 선택이 실패 장면을 구함 | `outputs/best_of_n_selection*/` |
| 20 | 표본 확대 (새 log 56개) | rank-sum: A− 2.15 → 1.45%, FDE −36% | `outputs/expanded_best_of_n/` |
| 21–24 | 사전 등록 held-out 52 log | PDMS +0.0095, 안전 필터 F1 +0.032 | `outputs/selection_validation/` |
| – | 3080 Ti 결과 5090 재실행 | 결론 동일 | `outputs/gpu5090_reanalysis/` |
| 25 | NAVSIM v2 navhard 절반 | F1 EPDMS +0.125 | `outputs/navhard_eval/` |
| 26 | navhard 나머지 절반 사전 등록 + 전체 | F1 재현(+0.112), max log-lik 미재현 | `outputs/navhard_full_validation/` |
| 27 | 안전 필터 ablation | F1 개선의 ~89%가 필터 자체 | `outputs/safety_filter_ablation/` |
| 28 (P1) | **다른 VLA(OpenVLA)에서 기전 재현** | step 안의 token feedback은 재현(첫 token 8 bin 교란 → 하위 차원 93% 증폭, 문맥 token이 결정), 과제 실패 효과는 기저 변동 안 | `outputs/cross_vla_replication/` |
| 29 (P2) | 안전 필터 구성요소 | 주행 가능 영역 제약이 86% | `outputs/safety_filter_components/` |
| 30 (P3) | 후보 oracle 상한 | oracle 0.405 vs 최선 0.350, 병목은 생성 | `outputs/candidate_oracle/` |
| 31 (P4) | 불안정 탐지 baseline | 이탈 전 AUROC 0.57–0.63(hidden probe 포함), 이탈 후 0.75–0.90 → 이탈 전에는 예측 불가 | `outputs/instability_detection_baselines/` |
| 32 (P5) | previous-action motion semantics | 방향 오차가 크기 오차보다 해로움 | `outputs/motion_semantics_ablation/` |
| 33 | **시간축 action-chunk VLA(Impromptu VLA 3B)에서 기전 재현** | correction 98%, reverse로 복원, 약 3 step window, 방향 > 크기 → Strong replication (궤적 수준 안정 분기는 미재현) | `outputs/cross_vla_temporal_replication/` |
| P6 | 연속 closed-loop (CARLA) | 현재 환경에서 수행 불가 (checkpoint·렌더링·디스크) | `outputs/continuous_closed_loop/STATUS.md` |

(경로의 `outputs/`는 `autovla_misalignment_poc/outputs/`입니다.)

---

## 3. 실패 기전 (실험 1–11, 32)

### 3.1 어디서 깨지는가 (실험 1–4)

| A− 발생 단계 | ORION | AutoVLA |
|---|---:|---:|
| perception (A− ∧ P−) | 6.0% | 25.8% |
| reasoning (A− ∧ P+R−) | 78.2% | 50.0% |
| interface (A− ∧ P+R+) | 15.9% | 24.2% |

- 두 모델 모두 말한 결정과 실제 행동의 결합이 느슨합니다(행동은 맞고 선언이 틀린 P+R−A+가 최대 그룹, 38–42%).
- CoT 개입(159 장면): 선언 방향을 뒤집어도 실행 반전 +6.4%p [+2.7, +9.7]; 방향과 무관한 문장 재작성은 A+를 +17.0%p 바꿉니다.
- 무편향 500 장면: CoT 강제 시 A+ 98.8 → 93.8% (−5.0%p, p = 5e-6), ADE +0.50 m.
- perception 정보는 실패군에서도 유지(AUROC 0.81–0.90), action은 마지막 layer(L32–35, MLP 지배)에서 형성됩니다.

### 3.2 작은 첫 편차가 증폭되는 기전 (실험 5–10)

| 실험 | 결과 |
|---|---|
| 5. 첫 mismatch 1-token 교정 | A− 장면 A+ 17.3 → 73.1% (+55.8%p), FDE 8.18 → 2.09 m |
| 6. 같은 거리(8.8 mm) 대안 token | A− 장면 증폭 45.0% vs A+ 장면 4.8%; A− 장면의 71%에서 증폭·회복이 공존 → 장면 맥락이 결정 |
| 7. 조건화 교정 | Normal 47.4% → Recent-GT(직전 1개만 GT) 7.1%, GT-history 4.1%; attention mask는 효과 없음 |
| 8. layer patching (37 layer) | 어느 layer에서 patch해도 3.8–7.4%; 역방향 patch(자기 token embedding 1개 삽입)로 4.4 → 33.4% 복귀 |
| 9. 교정 window | 1-step 47%, 2-step 76%, 3-step 90%, 4-step 96% (효과 대비) |
| 10. horizon 통제 | 같은 free horizon에서도 1-step 62% vs 4-step 95% 회복; 해제 후 25–29% 재발산 |

→ **직전에 생성한 action token의 identity가 residual stream을 통해 다음 action을 조건화하는 1-step autoregressive feedback**이 증폭을 만듭니다.
첫 이탈 직후 2–4 step이 임계 window이지만, 교정을 멈추면 서서히 재발산합니다.

### 3.3 그 identity의 무엇이 중요한가 (실험 11, 32)

| 직전 token := (A− 365 단위) | 증폭 | Recent-GT 효과 대비 |
|---|---:|---:|
| 자기 token | 47.1–47.9% | – |
| GT | 7.7–7.9% | 100% |
| GT와 motion이 가장 비슷한 다른 token (실험 11) | 8.5% | 99% |
| 자기와 motion이 비슷한 다른 token (실험 11) | 43.8% | 8% |
| **방향 정답 + 크기 오답** (실험 32) | 17.3% | 76% |
| **크기 정답 + 방향 오답** (실험 32, GT와의 거리는 더 작음) | 26.3% | 54% |
| **같은 거리 오차를 GT 반대편에** (실험 32) | 13.7% | 85% |
| 크기만 맞춘 무작위 (실험 32) | 19.2% | 71% |

- embedding 근접은 motion을 고정하면 효과가 없고(실험 11), **motion 중에서도 방향(횡방향·heading)이 크기보다 중요**합니다:
  방향 오답 − 크기 오답 = 증폭 +9.0%p [+3.1, +18.7], FDE +1.21 m (A−); 정상 장면에서도 방향 오답은 증폭 4.3 → 8.6%, 크기 오답은 3.7%로 무해.
- 같은 크기의 오차라도 모델 자신의 이탈과 반대 방향이면 증폭이 47.9 → 13.7%로 줄어듭니다 → **자기 이탈 방향을 확인해 주는 직전 action이 feedback을 자기 강화**합니다.

---

## 4. GT 없는 완화: 조건화는 실패, 선택은 통함 (실험 12–24)

- **조건화 교정**(이전 frame 계획, CTRA, 과거 계획 합의, PDM-Closed 궤적을 context에 넣음): 모델이 참조 motion을 66–89% 따라가
  효과가 참조 품질로 결정되고, 정상 장면이 손해를 봅니다. 교정 시점 없이 상시 적용하면 모집단 A− 1.24 → 2.60%로 악화(실험 15).
- **선택**(모델 자신의 후보 17개 = T 0.01 자연 + T 1.0 샘플 16개, context 불변):

| 데이터 | 규칙 | 결과 |
|---|---|---|
| 새 log 56개 (실험 20, 4,563 장면) | rank-sum (엔트로피 순위 + log-lik 순위) | A− 2.15 → 1.45% (−0.70%p [−1.07, −0.33], p = 4e-4), FDE 0.76 → 0.48 m |
| 사전 등록 held-out 52 log (실험 21, 4,814 장면) | rank-sum | PDMS 0.8893 → 0.8988 (+0.0095 [+0.0046, +0.0148]), 충돌 0.56 → 0.23% |
| 같은 held-out (실험 24) | rank-sum + 현재 frame 안전 필터(F1) | PDMS 0.9209 (+0.0315 [+0.0234, +0.0410]), 충돌 0.10% |

- 실험 22: 선택된 후보와 버려진 후보는 **이탈 전 엔트로피가 같고(차이 −0.016, 장면 내 AUROC 0.51) 이탈 후에만 갈라집니다(−0.40)**.
  rank-sum은 장면 안에서 증폭 rollout을 AUROC 0.85–0.87로 구별합니다.
- 실험 23: 후보 8개에서 이득의 85%, 12–16에서 포화; N 16 추론 비용 1.2–1.3배(batch 디코딩).
- 5090 재실행: 3080 Ti로 돌린 모든 run을 5090에서 다시 해도 held-out rank-sum +0.0099, F1 +0.0316으로 같습니다.

---

## 5. NAVSIM v2 navhard 2단계 pseudo closed-loop (실험 25–27, 29–30)

전체 navhard_two_stage: 76 log, 225 그룹(원본 450 + 합성 5,462 장면). 실험 26의 두 번째 절반은 디코딩 전에 사전 등록(96a595c).

| 조건 | EPDMS | Δ vs 선택 없음 [95% CI] | 그룹 개선/악화 |
|---|---:|---|---|
| 선택 없음 | 0.229 | – | – |
| rank-sum | 0.248 | +0.020 [+0.003, +0.037], perm p 0.094 | 92 / 73 |
| max log-lik | 0.249 | +0.020 [+0.008, +0.033] (두 번째 절반에서 미재현) | 68 / 56 |
| 충돌 제약만 | 0.241 | +0.012 [+0.005, +0.020] | 33 / 6 |
| 주행 가능 영역 제약만 | 0.315 | +0.087 [+0.066, +0.106] | 117 / 6 |
| **안전 필터만** (충돌 + DA, 자연 계획이 통과하면 유지) | **0.333** | **+0.105 [+0.085, +0.125]** | **128 / 9** |
| 필터 + 무작위 | 0.328 | +0.099 | 154 / 53 |
| 필터 + rank-sum (F1) | 0.347 | +0.118 [+0.096, +0.140] | 147 / 51 |
| 필터 + max log-lik | 0.350 | +0.122 [+0.101, +0.142] | 141 / 33 |
| oracle (17개 중 최고, 분석용 상한) | 0.405 | +0.177 [+0.152, +0.201] | 180 / 18 |

- **재현성(실험 26)**: F1은 사전 등록한 두 번째 절반에서 +0.112 [+0.082, +0.140], p 5e-5로 재현. max log-lik은 +0.010 [−0.004, +0.024]로 미재현.
- **분해(실험 27)**: F1 개선 +0.118 중 필터 자체 +0.105(89%), 선택기 추가분 rank-sum +0.014 [−0.002, +0.030](n.s.), max log-lik +0.017 [+0.003, +0.031].
  통과 후보 중 무작위 선택은 필터만보다 나을 게 없고(−0.005) 승차감만 나빠집니다.
- **구성요소(실험 29)**: 두 제약의 Shapley 몫은 주행 가능 영역 86%, 충돌 14%. 점수 이득의 99–104%가 곱셈 항(안전 위반 감소)이고 진행도 항은 −0.0001 ~ −0.0055 →
  진행도를 희생해 얻은 점수가 아닙니다. 다만 필터가 바꾼 장면에서는 4초 계획 도달 거리가 2.5–2.9 m 짧아 약간 보수적입니다.
- **oracle(실험 30)**: 최선 방법은 oracle 이득의 69%를 얻었고 남은 선택 여지는 +0.055. oracle 자체가 0.405이고 token의 31%는 17개 후보가 모두 0점 →
  **주된 병목은 후보 생성**입니다(oracle은 미래를 보는 분석용 상한, 배포 방법 아님).
- 한계: 안전 필터와 채점이 같은 PDM 규칙을 써서 이 지표에서 필터 이득이 유리하게 측정됩니다. 2단계 pseudo closed-loop이며 연속 시뮬레이션이 아닙니다.

---

## 6. 다른 VLA에서의 재현: OpenVLA-7B / LIBERO (실험 28, P1)

모델 `openvla-7b-finetuned-libero-spatial`(RTX 5090), LIBERO-Spatial 10 과제 × 100 에피소드. 공식 평가를 재구현해 자연 성공률 87%(공식 보고 84.7%).
프로토콜 6b9e8ad(실행 전), 자연 재실행 대조군 추가 6d0375a(Phase B 일부를 본 뒤·최종 분석 전). 상세: `outputs/cross_vla_replication/RESULTS.md`.

**구조 차이**: OpenVLA는 step마다 7개 action token(x, y, z, roll, pitch, yaw, gripper)을 차원 순서로 생성하고, 이전 step의 action을 입력에 넣지 않습니다.
AutoVLA의 "직전 action token → 다음 token" feedback에 대응하는 것은 step 안의 차원 간 조건화뿐입니다.

| 질문 | 결과 |
|---|---|
| step 안 token feedback (Phase C, 2,462 프레임) | x token 8 bin 교란 → 하위 차원이 δ 이상 움직이는 비율 **92.6% [89.8, 94.4]**, 24 bin 81.2%; y 교란도 75–87% |
| 무엇이 하위 token을 결정하나 | **문맥에 들어간 이전 token**: reverse(정상 실행 + 잘못된 문맥) = feedback과 하위 token 동일, corrected(교란 실행 + 정상 문맥) = 변화 0 |
| 교정 window (1–4개 차원) | 노출 차원 수에 비례해 감소(93 → 85 → 67 → 53 → 27%); 교정 후 교란을 다시 넣으면 남은 차원은 거의 그대로 발산 (AutoVLA의 지속 효과는 없음) |
| 1. 작은 첫 편차가 항상 실패로? | 아니오. 자연 성공 중 교란 후 실패 9–18% = 교란 없는 재실행의 실패 15%와 같은 수준 |
| 2. 같은 크기에서 안정/불안정 분기? | 분기는 있으나(76 회복 / 11 실패) closed-loop 수치 변동과 구별되지 않고, step 내 하위 편차 크기로 예측되지 않음 |
| 3. previous-action correction | 과제 수준 효과 검출 안 됨 (feedback − corrected +3 ~ +5%p, n.s.) |
| 4. reverse intervention | step 안에서는 feedback과 같은 하위 발산; 과제 성공률 차이 −3 ~ −8%p vs 자연, −2 ~ +3%p vs 재실행 (n.s.) |
| 대조군 | 같은 정책을 batch 구성만 바꿔 다시 돌려도 성공 81% vs 87%, 결과 불일치 20% |

→ **"직전 token identity가 다음 token을 조건화하는" 국소 autoregressive feedback은 모델 간에 일반적**이지만, 그것이 **궤적·과제 실패로 증폭되려면
AutoVLA처럼 여러 시점의 action을 한 번에 autoregressive하게 생성하는 구조(시간축 action history)가 필요**하다는 해석과 일치합니다.
한계: LIBERO-Spatial 한 suite·100 에피소드, 교란은 한 차원·10 step. π0-FAST처럼 action chunk를 autoregressive하게 생성하는 모델은 다루지 못했습니다(시간축 chunk 모델은 실험 33에서 Impromptu VLA로 검증).

---

## 6b. 시간축 action-chunk VLA에서의 재현: Impromptu VLA 3B (실험 33)

사전 등록 038a6d3(본 실행 전). RTX 5090, navtest PoC 2,748 장면(28 log). 상세: `outputs/cross_vla_temporal_replication/RESULTS.md`.

**모델 선정(코드 수준 확인)**: Impromptu VLA 3B(`aaaaaap/ImpromptuVLAModel/3B_AD`, Qwen2.5-VL-3B)는 미래 waypoint 10개(0.5 s 간격, 5 s)를 시간순 텍스트
`[x, y]`로 autoregressive하게 생성해 4개 조건(다중 token AR 생성, 앞 action → 뒤 action conditioning, chunk 안 시간 순서, 공개 checkpoint·code)을 모두 만족합니다.
π0-FAST는 DCT 주파수 계수 token이라 시간 순서가 없어 제외, SpatialVLA는 LIBERO checkpoint 비공개·SimplerEnv(Vulkan) 불가로 제외했습니다.
AutoVLA는 0.5 s마다 상대 motion codebook token 1개, Impromptu는 절대 위치를 digit token 약 12개로 쓰는 차이가 있고, Impromptu의 자연 성능은
A− 33.4%, FDE5 5.75 m로 AutoVLA보다 훨씬 낮습니다. 자연 재실행(batch 구성만 다름)의 결과 불일치는 5.9%입니다.

| 실험 | Impromptu VLA 3B | AutoVLA |
|---|---|---|
| A. 작은 초기 편차(0.2 m) | 기준 A+ 장면의 49.5%가 A−로 전환(재실행 기저 4.6%); 궤적 수준에서는 97%가 3배 이상 커짐(흡수 0.5%) | 상당수 흡수 (A+ 장면 증폭 4.8%) |
| B. 같은 크기 분기 | 사전 등록 기준(궤적 수준 안정 r ≤ 1 공존) 2.2–3.2% < 재실행 5.9% → **미충족**; 탐색적으로 task 수준 통과/증폭 공존 87–93%, 횡·대각 교란이 전후보다 약 2배 증폭 | 같은 크기 편차가 장면에 따라 갈림 |
| C. 직전 action 교정 (문맥만) | recent_gt 증폭 63.4 → **1.3%**(−62.2%p [−65.6, −59.1]); gt_history 0% | 47 → 7% (93%) |
| C. 교정 window 1/2/3/4 step | 효과의 24 / 39 / **91** / 95% | 47 / 76 / 90 / 96% |
| D. reverse (교정 문맥 + 자기 오차 재삽입) | 증폭 0 → **81.9%** (+81.9%p [+80.4, +84.2]), 발산 +40.5 m | 4.4 → 33.4% |
| E. motion semantics | GT와 5 cm 다른 값 = GT와 같은 효과(p = 0.87); 방향 오답 − 크기 오답 증폭 +15.6%p(p = 2e-31), 오차 크기를 맞춘 첫 대체 시점에서도 +0.29 m(p = 2e-99, 탐색적) | motion > embedding, 방향 > 크기 (+9.0%p) |

사전 등록 기준: 1 correction **충족**, 2 reverse **충족**, 3 궤적 수준 분기 **미충족**, 4 짧은 window **충족**, 5 방향 > 크기 **충족**,
task 수준(교란이 기저 변동보다 큰 실패를 만들고 교정이 실패를 줄임) **충족** → **Strong replication**.
제한: 여기서 task 실패는 AutoVLA와 같은 open-loop 궤적 실패이고, Impromptu는 절대 위치를 쓰므로 GT 문맥 교정에 GT 정보 누출이 섞여 있습니다
(정보를 더하지 않는 reverse와 motion 대체가 이를 배제하는 근거). 교정 해제 후 재발산은 AutoVLA와 달리 거의 없었습니다.

→ **AutoVLA의 오류 증폭은 특정 모델의 특성이 아니라 시간 순서가 있는 action chunk를 autoregressive하게 생성하는 구조에서 반복되는 failure mechanism**으로
보입니다. 시간축 action history가 없는 OpenVLA(실험 28)에서는 궤적·과제 실패로 증폭되지 않았고, 증폭의 정도(작은 편차의 흡수)는 action 표현과 모델 정확도에 따라 크게 다릅니다.

---

## 7. 불안정 rollout 탐지: 이탈 전에 예측 가능한가 (실험 31, P4)

실험 20–24의 5090 후보(dev 56 log 42,095 이탈 후보 / held-out 52 log 43,469, 증폭 유병률 5.8%)를 teacher forcing해 margin과 hidden state(layer 18, 36)를 얻고,
probe는 dev에서만 학습해 held-out에 한 번 적용했습니다. 프로토콜 bf9aa21. 상세: `outputs/instability_detection_baselines/RESULTS.md`.

| held-out 탐지기 | pre AUROC [95% CI] | pre 장면 내 | post AUROC [95% CI] | post 장면 내 | post AUPRC |
|---|---|---:|---|---:|---:|
| entropy | 0.634 [0.607, 0.658] | 0.516 | 0.823 [0.809, 0.842] | 0.786 | 0.215 |
| log-likelihood | 0.570 [0.549, 0.587] | 0.596 | 0.832 [0.820, 0.846] | 0.811 | 0.246 |
| probability margin | 0.589 [0.561, 0.615] | 0.504 | 0.769 [0.755, 0.786] | 0.716 | 0.132 |
| 후보 궤적 분산 | 0.582 [0.566, 0.598] | – | 0.797 [0.786, 0.810] | – | 0.157 |
| pairwise disagreement | 0.620 [0.604, 0.637] | 0.539 | **0.900 [0.890, 0.913]** | 0.898 | 0.381 |
| medoid 거리 | 0.630 [0.610, 0.652] | 0.649 | 0.883 [0.866, 0.901] | **0.904** | **0.403** |
| hidden logistic / ridge / MLP probe | 0.61–0.63 | 0.54–0.55 | 0.745–0.765 | 0.74–0.75 | 0.18–0.20 |

→ **불안정 rollout은 이탈 전에는 예측되지 않고**(모델 내부 표현을 써도 엔트로피보다 낫지 않음), **이탈 이후 feedback이 진행되면서 구별됩니다.**
이탈 후에는 후보 간 불일치가 가장 강한 신호이고 확신도가 그다음입니다. 선택이 통하는 이유는 이탈 후 자기 강화 구간을 보기 때문이며,
이탈을 미리 막는 trigger로는 쓸 수 없습니다.

---

## 8. 연속 closed-loop (P6) — 수행 불가

공개된 AutoVLA checkpoint는 NAVSIM용(`AutoVLA_PDMS_89.ckpt`) 하나뿐이고 CARLA/Bench2Drive용은 없습니다. 컨테이너에는 CARLA에 필요한
NVIDIA EGL/Vulkan 라이브러리가 없고(LIBERO도 Mesa 소프트웨어 렌더링으로 실행), 디스크도 약 20 GB 부족합니다. nuPlan closed-loop는 카메라를
렌더링하지 않아 카메라 기반 AutoVLA를 돌릴 수 없습니다. 필요한 조건은 `outputs/continuous_closed_loop/STATUS.md`에 정리했습니다.

---

## 9. 종합 그림

```
perception (대체로 보존)
   │
   ▼
결정/CoT 텍스트 ──(느슨한 결합)──▶ action token 생성 (마지막 layer에서 형성)
                                        │ 첫 이탈 t*
                                        ▼
            직전 action token의 motion(특히 방향)이 residual을 통해 다음 token을 조건화
            → 자기 이탈 방향을 확인하는 1-step feedback이 자기 강화 (장면 맥락이 증폭 여부를 좌우)
                                        │
          oracle 교정: 2–4 step이면 90–95% 차단 (해제 후 감쇠)
          GT 없는 조건화 교정: 정상 장면 손해로 순이득 없음
          배포 가능한 완화: 후보 선택 (open-loop·PDMS에서 유의)
                         + 현재 frame 안전 필터 (반응형 navhard에서 이득의 대부분, DA 제약이 핵심)
          남은 병목: 후보 생성 (oracle 0.405, token의 1/3은 안전한 후보가 없음)
```

- 실패의 축은 "무엇을 보았는가"가 아니라 **"직전에 무엇을 출력했는가"**이고, 그 정보의 핵심은 진행 방향입니다.
- token 수준 feedback은 다른 autoregressive VLA(OpenVLA)에서도 나타나지만, 궤적 실패로의 증폭은 시간축 action history를 autoregressive하게 생성하는 구조에서 생깁니다. 그런 구조의 다른 모델(Impromptu VLA)에서는 교정·reverse·짧은 window·방향 효과까지 재현되었습니다(실험 33).
- 불안정성은 이탈 전에는 보이지 않고 이탈 후에만 보이므로, 실용적인 완화는 "이탈을 미리 막기"가 아니라 **여러 후보를 만들고 이탈 후 신호와 현재 frame 안전 규칙으로 거르기**입니다.

---

## 10. 한계

- AutoVLA 기전 실험(5–11, 32)은 equal-distance set(208 장면, A− 52 장면)에 기반해 log cluster CI가 넓고, 단일 seed·T 0.01입니다.
- 평가 horizon은 5 s(10 token)이며, 그 이상은 모델 분포 밖입니다.
- navhard 결과는 2단계 pseudo closed-loop이고, 안전 필터와 채점이 같은 규칙을 씁니다. 연속 closed-loop(P6)는 수행하지 못했습니다.
- GPU·batch 구성이 달라지면 경계 장면의 디코딩이 bf16 수치 차이로 갈릴 수 있어, 모든 비교는 같은 실행 안의 쌍대 비교로 했습니다.
- OpenVLA 재현(실험 28)은 LIBERO-Spatial 한 suite, 100 에피소드 규모이고, OpenVLA는 시간축 action history가 없어 step 내 차원 간 feedback만 직접 비교할 수 있습니다.
- Impromptu 재현(실험 33)은 open-loop 궤적 실패 기준이고, 절대 위치 표현 때문에 GT 문맥 교정에 GT 정보 누출이 섞여 있습니다. 궤적 수준 안정 분기(사전 등록 기준 3)는 재현되지 않았습니다.

---

## 11. 재현과 데이터

- 환경·자산: `tools/download_autovla_assets.sh`, `tools/setup_autovla_env.sh`; OpenVLA/LIBERO는 `/root/VLA/openvla/env.sh`(autovla env + target dir + Mesa EGL).
- navhard: `tools/navhard_half2_pipeline.sh`, `tools/navhard_score_groups.sh`, split 설정 `autovla_misalignment_poc/configs/navsim_v2/`.
- ablation·구성요소·oracle: `tools/navhard_ablation_pipeline.sh`, `tools/navhard_components_candidates_pipeline.sh`, `tools/paper_jobs_scheduler.sh`.
- Impromptu VLA: `autovla_misalignment_poc/scripts/cross_vla_temporal/` (`impromptu_core.py`, `exp_ab.py`, `exp_cde.py`, `analyze_temporal.py`), checkpoint `/root/VLA/impromptu/3B_AD`; 실험 33 디스크 정리 기록 `outputs/cross_vla_temporal_replication/DISK_CLEANUP.md`.
- OpenVLA: `autovla_misalignment_poc/scripts/cross_vla/` (`openvla_core.py`, `openvla_rollouts.py`, `openvla_token_feedback.py`, `analyze_cross_vla.py`).
- 원시 rollout·hidden state·submission은 용량 때문에 저장소에 넣지 않았고, 위 스크립트로 재생성됩니다.
- 디스크 정리 기록: 사용하지 않는 카메라(CAM_B0/L0/L2/R0/R2, 약 19 GB)를 삭제했습니다. 원 전처리(preprocess_scenes)를 처음부터 다시 하려면 shard 0–5를 다시 받아야 합니다.
