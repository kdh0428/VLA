# 실험 34 결과: 다른 backbone·domain에서의 previous-action temporal feedback 재현 (SpatialVLA-4B, SimplerEnv)

사전 등록: `PROTOCOL.md` (commit f578eae, 본 실행 전). 분석: `scripts/cross_domain_temporal/analyze_svla.py` → `analysis.json`.
RTX 5090만 사용. 정의·threshold·판정 기준은 결과를 본 뒤 바꾸지 않았습니다.

**판정: Partial replication.**
- **Token 수준: 사전 등록 Strong 조건 (1)–(3)을 모두 충족합니다.** 직전 action을 교정하면 이후 step의 증폭이 크게 줄고, 오차를 품은 직전 action만 다시 넣으면 그 효과가 그대로 돌아옵니다.
- **Task 수준: 재현되지 않았습니다.** closed loop에서 문맥만 교정해도 실패가 줄지 않았고, 문맥에만 교란을 넣어도 실패가 늘지 않았습니다(자연 재실행 변동 범위 안).
- 궤적에는 영향이 있습니다. 문맥에만 넣은 교란도 궤적을 재실행 변동보다 크게 바꿉니다(0.21–0.23 m vs 0.14 m).

## 실행 중 변경·이슈 (기록)
- **Closed loop 재시작:** 첫 실행에서 시뮬레이터 8개가 CPU thread를 과점유했습니다(load 135, GPU 유휴). worker당 thread 수를 제한하고(`LP_NUM_THREADS=2` 등) 같은 설정으로 재시작했습니다. 이미 끝난 에피소드는 유지하고 이어서 실행했으며, 실험 정의는 바뀌지 않았습니다.
- **Token 수준 중단 1회:** 1,396번째 프레임에서 모델이 translation 자리에 translation 범위 밖 token(265345)을 생성해 motion 대체 함수가 중단됐습니다.
  - 범위 밖 id는 tokenizer의 decode와 같은 방식으로 clip해 처리했고, 남은 204 프레임을 이어서 실행했습니다.
  - 분석도 같은 clip 규칙을 씁니다.
  - 정의 변경은 없지만, 사전 등록에 없던 처리이므로 기록합니다.

## 1. 기본 성능과 자연 재실행 변동 (Experiment A)

| 조건 | 성공률 (95% CI) | natural과 성공 불일치 | 실행 궤적 끝점 발산 (m) |
|---|---|---|---|
| natural (step-wise 디코더) | 85.0% [77.5, 92.5] (coke 92.5%, move near 77.5%) | – | – |
| natural_mix (개입 행과 같은 batch) | 85.0% | 0.0% | 0.003 [0.000, 0.007] |
| natural_gen (모델 자체 `predict_action`) | 93.8% [87.5, 98.8] | **13.8%** | **0.140 [0.100, 0.185]** |

- 공식 보고치(coke 0.86, move near 0.78)와 비슷합니다.
- batch 구성은 결과에 영향이 없습니다. 반면 수치 경로(bf16, `generate` vs step-wise)만 달라도 에피소드의 13.8%에서 성공 여부가 뒤집히고, 궤적이 0.14 m 갈라집니다.
- natural_gen의 성공률이 +8.8 %p 높지만 유의하지 않습니다(McNemar p = 0.065).
- 같은 관측에서 `generate`와 step-wise chunk가 일치한 비율은 92.9%입니다(token 수준 수치 noise).
- **noise floor: 성공 불일치 13.8%, 궤적 발산 0.14 m.**

## 2. Closed-loop 개입 (B·C·D의 task 수준; d = 0.3, control step 8–23)

| 조건 | 성공률 | − natural (95% CI, McNemar p) | 불일치 | 궤적 발산 (m) |
|---|---|---|---|---|
| feedback_perp_left | 86.2% | +1.2 %p [−8.8, +11.2], p = 1 | 23.8% | 0.422 [0.380, 0.471] |
| corrected_perp_left | 91.2% | +6.2 %p [−1.2, +13.8], p = 0.23 | 13.8% | 0.416 [0.370, 0.463] |
| reverse_perp_left | 90.0% | +5.0 %p [−3.8, +13.8], p = 0.39 | 15.0% | 0.209 [0.171, 0.252] |
| feedback_opposite | 65.0% | **−20.0 %p [−31.2, −8.8], p = 0.002** | 32.5% | 0.437 [0.378, 0.500] |
| corrected_opposite | 63.7% | **−21.2 %p [−33.8, −7.5], p = 0.006** | 43.8% | 0.423 [0.370, 0.482] |
| reverse_opposite | 83.8% | −1.2 %p [−11.2, +8.8], p = 1 | 21.2% | 0.232 [0.184, 0.285] |

사전 등록한 대비:
- **corrected vs feedback (교정 효과):**
  - perp_left: +5.0 %p [−1.2, +12.5], p = 0.29
  - opposite: −1.2 %p [−13.8, +12.5], p = 1
  - 두 방향 모두 교정 효과가 없습니다.
- **reverse vs corrected:**
  - perp_left: −1.2 %p, p = 1
  - opposite: +20.0 %p [+7.5, +31.2], p = 0.004
  - opposite에서는 reverse가 오히려 성공률이 높습니다. 즉 문맥 교란을 다시 넣어도 실패가 늘지 않았습니다.

해석:
- task 실패는 opposite 방향에서만 생겼고(−20 %p), 원인은 **실행된 교란 action 자체**입니다. 그 action을 실행하면 문맥을 교정해도 실패하고(corrected), 문맥에만 넣으면 실패하지 않습니다(reverse).
- perp_left 교란은 실행해도 task에 영향이 없습니다.
- 문맥에만 넣은 교란(reverse)의 궤적 발산은 0.21–0.23 m로, CI가 재실행 noise(0.10–0.19)와 거의 겹치지 않습니다. feedback은 이후 실행 action을 실제로 바꾸지만, 성공 여부는 바꾸지 못합니다.
- 반대로 문맥 교정은 궤적 발산을 거의 줄이지 못합니다(0.42 → 0.42). closed loop의 궤적 편차는 대부분 실행된 action에서 옵니다.

## 3. Token 수준 (B·C·D·E·F): 1,600 프레임, 18,975 단위

교란 주입 step 1 다음 step 2는 항상 p를 문맥으로 봅니다. 자연 chunk R과 비교한 translation token 변경률은 step 2 55.6%, step 3 36.0%, step 4 27.9%입니다. 즉 직전 action → 다음 action feedback이 존재합니다.

### C·D. 교정과 reverse

| 행 | 증폭률 (A ≥ 1) | 회복률 | D (step 2–4) | step-4 편차 |
|---|---|---|---|---|
| normal | 22.1% [19.5, 24.7] | 36.7% | 0.191 | 0.056 |
| recent_ref (직전 step만 R) | 13.7% | 37.5% | 0.115 | 0.016 |
| full_ref (step 2·3 모두 R) | 13.5% | 37.6% | 0.109 | 0.010 |
| win1 (step 2만 R, step 3 재생성) | 15.4% | 37.4% | 0.126 | 0.027 |
| **reverse** (step 2 = R + normal의 오차 step 3) | 18.6% | 36.8% | 0.155 | **0.057** |

- **교정 효과:**
  - recent_ref − normal: 증폭 −8.3 %p [−9.3, −7.4] (McNemar p ≈ 1e−260), step-4 편차 −0.040.
  - full_ref − normal: 증폭 −8.5 %p, D −43%, step-4 편차 −82%.
  - normal에서 증폭된 4,185 단위만 보면 증폭이 100% → 53%로 줄어듭니다.
- **reverse:**
  - reverse − full_ref: 증폭 +5.1 %p [+4.4, +5.8], step-4 편차 +0.047 [+0.042, +0.052] (Wilcoxon p ≈ 0).
  - step-4 편차가 normal 수준(0.057 vs 0.056)으로 **완전히 복원**됩니다.
  - 증폭 단위만 보면 +23.6 %p이고, step-4 편차는 0.204로 normal과 같습니다.
- 교정이 증폭을 절반까지만 줄이는 이유: 교정할 수 없는 부분이 남기 때문입니다. step 1의 교란 자체가 문맥에 남아 있고, step 2는 그 교란만 보고 생성되므로 교정할 수 없습니다. 교정 가능한 step 4의 편차는 82% 제거됩니다.

### E. Temporal window

chunk가 4 step이라 step 4 기준 교정 위치는 step 3(1 step 전)과 step 2(2 step 전)뿐입니다. full_ref 효과 대비 step-4 편차 회복률:

| 교정 위치 | 회복률 |
|---|---|
| step 3만 (1 step 전, recent_ref) | 87% |
| step 2만 (2 step 전, win1) | 63% |
| 둘 다 | 100% |

- win1 − full_ref: 증폭 +1.9 %p (p ≈ 6e−68), step-4 편차 +0.017.
- 직전 step이 효과를 가장 많이 전달하고, 2 step 전도 기여합니다. window는 ≥ 1–2 step으로, chunk 길이 때문에 그 이상은 측정할 수 없습니다.

### F. Motion semantics

대상: 마지막 문맥 step의 translation token만 대체한 10,542 단위.

| 대체 | 증폭률 | D | 기준(R)과의 거리 |
|---|---|---|---|
| ref_trans (R의 token) | 30.1% | 0.264 | 0 |
| near_ref (한 bin 옆) | 34.4% | 0.297 | 0.021 |
| dir_ok_mag_wrong | 34.4% | 0.318 | 0.175 |
| dir_wrong_mag_ok | 36.0% | 0.335 | 0.120 |
| random_mag_matched | 43.4% | 0.439 | 0.284 |
| (normal) | 36.8% | 0.320 | – |

- **방향 > 크기:**
  - dir_wrong_mag_ok − dir_ok_mag_wrong: 증폭 +1.6 %p (p = 7e−5), D +0.017 (p = 7e−6), step-4 편차 +0.013 (p = 6e−17).
  - 방향이 틀린 대체가 R에 **더 가까운데도**(0.120 vs 0.175) 편차를 더 만듭니다. 방향의 효과는 유의하지만 크기는 작습니다.
- **motion > identity가 아닙니다:**
  - near_ref − ref_trans: 증폭 +4.3 %p, step-4 편차 +0.016. 거리가 0.021밖에 안 되는 한 bin 차이가, 거리 0.175인 크기 오류 효과(+0.029)의 절반을 넘습니다.
  - Impromptu에서는 near-GT ≈ GT였는데, SpatialVLA는 token identity에 민감합니다.
  - rot·grip token도 기여합니다(ref_trans 18.1% vs recent_ref 13.7%, 전체 단위 기준).

### B. 같은 크기 교란의 갈림 (stable / unstable branch)

| | d = 0.15 | d = 0.30 |
|---|---|---|
| 같은 프레임에서 회복 방향과 증폭 방향 공존 | 25.1% [21.8, 28.7] | 15.3% [12.8, 17.9] |
| log A 분산 중 프레임 간 비율 | 60% | 64% |
| 방향별 증폭률 | 23–32% | 14–21% |

- 전체 단위의 36.7%는 step 2–4 token이 R과 완전히 같습니다(교란 흡수). 22.1%는 증폭됩니다.
- **안정·불안정 branch가 모두 존재하고**, 어느 쪽인지는 주로 상태(프레임)가 정합니다.
- Impromptu(0.2 m 편차의 97%가 3배 이상 성장)와 달리, AutoVLA처럼 흡수되는 branch가 있습니다.

## 4. 사전 등록 기준 판정

| 기준 | Token 수준 | Closed loop (task) |
|---|---|---|
| (1) 교정이 증폭을 유의하게 감소 | ✓ (−8.5 %p, step-4 편차 −82%) | ✗ (corrected vs feedback: +5.0 / −1.2 %p, n.s.) |
| (2) reverse가 증폭을 다시 증가 | ✓ (step-4 편차 완전 복원) | ✗ (문맥만 교란해도 성공률 변화가 noise 이내) |
| (3) 효과 > 자연 재실행 변동 | ✓ (결정적 paired 대비; 수치 noise 7.1% chunk 불일치보다 큼) | 궤적만 ✓ (0.21–0.23 m > 0.14 m), 성공 ✗ |
| temporal window | ✓ (1 step 87%, 2 step 63%) | – |

token/action 수준 feedback은 명확합니다. 하지만 task/trajectory 실패로 이어지는지는 사전 등록 기준에서 충족되지 않았으므로 **Partial replication**입니다.

## 5. 비교표

| | AutoVLA | Impromptu VLA | **SpatialVLA (실험 34)** |
|---|---|---|---|
| Backbone | Qwen2.5-VL 3B | Qwen2.5-VL 3B | **PaliGemma2 3B (Gemma2) + Ego3D** |
| Domain | 자율주행 | 자율주행 | **로봇 조작 (SimplerEnv Google Robot)** |
| Action representation | codebook | absolute waypoint text | 구면 bin translation + rotation + gripper token, 4-step chunk |
| Temporal AR | ✓ | ✓ | ✓ (chunk 안, 매 step 재계획 + ensemble) |
| Correction effect | ✓ | ✓ | token ✓ (step-4 편차 −82%), task ✗ |
| Reverse effect | ✓ | ✓ | token ✓ (완전 복원), task ✗ |
| Critical window | 2–4 | ~3 | ≥ 1–2 (chunk 길이 한계; 1 step 87%) |
| Direction > magnitude | ✓ | ✓ | 약한 ✓ (+1.6 %p; identity 민감, near-ref ≠ ref) |
| Stable / unstable absorption | ✓ | ✗ | ✓ (흡수 37%, 증폭 22%, 같은 상태에서 공존 15–25%) |

## 6. 결론: 일반 메커니즘인가, driving/Qwen 특이 현상인가

- **token 수준 메커니즘은 일반적입니다.** "직전 action이 다음 action을 조건화하고, 그 오차를 교정하면 효과가 사라지고, 다시 넣으면 돌아온다"는 인과 구조가 비-Qwen backbone(Gemma2)과 다른 domain(로봇 조작), 다른 action 표현(구면 bin token)에서 재현됐습니다. 따라서 driving이나 Qwen에만 있는 현상은 아닙니다.
- **실패 메커니즘이 되는지는 배포 구조에 달려 있어, 일반적이라고 말할 수 없습니다.**
  - AutoVLA·Impromptu에서는 feedback이 trajectory와 plan 실패로 이어졌습니다.
  - SpatialVLA에서는 feedback이 4-step chunk 안에 갇힙니다. 매 control step마다 새 관측으로 재계획하고, 4개 chunk를 ensemble로 평균하기 때문입니다.
  - 그래서 문맥 오차는 궤적을 바꾸지만(0.21 m) task 성공은 바꾸지 못했습니다. 교란으로 인한 실패는 전부 실행된 action 자체에서 왔습니다.
  - OpenVLA(실험 28: 같은 step 안의 token feedback은 있으나 task 효과는 없음)와 같은 양상입니다.
- 정리: previous-action temporal feedback은 **temporally autoregressive VLA 디코더의 일반적인 token 수준 인과 구조**입니다. 이것이 **task 실패의 원인**이 되는 조건은 driving/Qwen이라는 점이 아니라, feedback이 닿는 거리(chunk·horizon 길이)와 재계획·ensemble 같은 흡수 구조입니다. 현재 증거로는 "일반적 실패 메커니즘"이 아니라 "일반적 메커니즘 + 구조 의존적 실패"까지만 주장할 수 있습니다.

## 한계
- 과제 2개, 에피소드 80개라 closed loop 검정력이 제한적입니다. 성공률 차이 약 10 %p 미만은 검출하기 어렵습니다.
- Closed-loop 개입은 d = 0.3, 방향 2개, control step 8–23만 다뤘습니다.
- chunk가 4 step이라 window를 2 step보다 길게 측정할 수 없습니다.
- token 수준 증폭 정의(A ≥ 1)는 사전 등록대로 정규화 translation 공간에서만 계산했습니다(rotation·gripper 제외).
- 범위 밖 translation token은 clip해서 처리했습니다(위 "실행 중 변경·이슈" 참조).

## 파일
- `closed_loop/episodes.jsonl`: 720 에피소드의 step별 exec/ctx token과 실행 action. `closed_loop/frames/`: natural 관측(4 step마다).
- `token_level/units.jsonl`: 프레임별 단위와 행별 chunk.
- `analysis.json`: 모든 통계.
- 스크립트: `scripts/cross_domain_temporal/` (`spatialvla_core.py`, `spatialvla_policy.py`, `svla_rollouts.py`, `svla_offline.py`, `analyze_svla.py`).
