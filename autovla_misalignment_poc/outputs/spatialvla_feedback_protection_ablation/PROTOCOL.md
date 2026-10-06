# 실험 35: SpatialVLA feedback 보호 구조 ablation — 사전 등록

작성: 2026-10-06, 본 실행 전. 그 전에 실행한 것은 smoke test(20 에피소드, seed 0–1), 실행기 동등성 검사, batch 불변성 검사뿐입니다.
실험 34의 결과·파일은 읽기만 하고 수정하지 않습니다. 새 결과는 이 디렉토리에만 저장하고, RTX 5090만 사용합니다.
정의·threshold·판정 기준은 결과를 본 뒤 바꾸지 않습니다.

## 가설

SpatialVLA의 짧은 4-step chunk, 매 step 재계획, 최근 4개 chunk ensemble이 autoregressive feedback 오차를 흡수한다.

## 0. 실행 전에 확인된 구조적 사실 (가설 해석에 필요)

1. **실험 34 PROTOCOL.md §0의 ensemble 설명이 틀렸습니다.**
   - 실험 34 문서에는 "오래된 예측일수록 가중치가 크다, 실행 action의 대부분은 chunk 뒤쪽 step(2–4)에서 온다"고 적었습니다.
   - 코드를 확인하니 `ActionEnsembler`의 weight는 exp(−temp·i)이고, temp = −0.8, i = 0이 가장 오래된 예측입니다. 따라서 **가장 최근 예측의 가중치가 가장 큽니다.**
   - 4개 예측이 모두 있을 때 가중치는 다음과 같습니다(합 1).

     | 예측 | 가중치 |
     |---|---|
     | 현재 chunk의 step 1 | 0.57 |
     | 1 step 전 chunk의 step 2 | 0.26 |
     | 2 step 전 chunk의 step 3 | 0.12 |
     | 3 step 전 chunk의 step 4 | 0.05 |

   - 결과적으로 공식 실행 action의 43%가 앞 step 문맥에 조건화된 step 2–4에서 옵니다.
   - 실험 34의 결과와 분석은 실제 공식 실행 코드를 그대로 썼으므로 영향이 없고, 설명 문장만 틀렸습니다. 실험 34 파일은 고치지 않고 여기에 기록합니다.
2. **ensemble은 feedback의 차단 장치이면서 통로입니다.**
   - 문맥 교란은 step-1 translation 뒤에 생성되는 token(step 1의 rotation·gripper, step 2–4 전체)을 바꿉니다.
   - 매 step 재계획에 ensemble이 없으면(r1e1) step 1만 실행되므로, feedback은 **step-1 rotation·gripper(같은 step 안의 feedback)로만** 실행에 도달합니다.
   - ensemble은 step 2–4를 실행에 섞어 넣고(43%), 동시에 그 오차를 다른 chunk의 예측과 평균합니다.
3. **4-step chunk에서는 일부 설정이 서로 같아집니다.**
   - 4 step마다 재계획하면 각 step을 덮는 예측이 1개뿐이라 ensemble-4 = 없음입니다(r4e4 ≡ r4e1). 따라서 사용자 요청의 D3와 D4는 같은 실행이고, r4e1로 한 번만 실행합니다.
   - 2 step마다 재계획하면 덮는 예측이 최대 2개라 ensemble-4 = ensemble-2입니다(r2e4 ≡ r2e2).
4. **batch 구성의 수치 영향을 없앴습니다.**
   - batch 크기가 달라지면 같은 관측에서 생성되는 chunk가 4/40에서 달라졌습니다(bf16 kernel 선택). 고정된 batch 크기 안에서 동행 행이 바뀌는 것은 0/40이었습니다.
   - 그래서 이번 실행은 prompt 길이 group마다 batch를 8행으로 채워 디코딩합니다(`decode_fixed`). 각 에피소드가 자기 관측과 개입만의 결정적 함수가 되고, 조건 간 차이는 개입에서만 생깁니다.
   - 실험 34는 batch 크기가 가변이었으므로, 공식 설정 재현(A)은 에피소드 단위로 완전히 같지는 않고 분포 수준에서 비교합니다.
   - 수치 경로 재실행 noise는 `natural_gen`(모델 자체 `predict_action`)으로 따로 측정합니다.
5. **실행기 동등성:** `Executor(K=1, E=4)`가 실험 34 `PolicyState`(공식 adapter)와 같은 action을 냄을 확인했습니다. 기록된 실험 34 action과의 최대 차이는 5e−6(기록 반올림)입니다.

## 1. 과제·seed·개입 (실험 34와 동일)

- **과제·seed:** SimplerEnv `google_robot_pick_coke_can`, `google_robot_move_near`, seed 0–39(과제당 40). 80 step, 성공 = 마지막 step의 `done`.
- **개입 대상:** control step 8–23에 **생성되는** 모든 chunk의 step-1 translation token g.
- **교란:** p = 정규화 공간에서 v(g) + 0.3·u에 가장 가까운 token, u = opposite(주 분석 조건).
- **조건:**

  | 조건 | 실행 step 1 | 문맥 step 1 |
  |---|---|---|
  | natural | 자유 디코딩 | 자유 디코딩 |
  | feedback | p | p |
  | corrected | p | g |
  | reverse | g | p |

- 재계획 간격 K가 크면 window 안의 개입 chunk 수가 줄어듭니다(K = 1: 16개, K = 2: 8개, K = 4: 4개). 그래서 feedback의 주효과는 설정 간에 직접 비교하지 않고, 설정 **안의** 대비(feedback − corrected, reverse − natural)를 비교합니다.

## 2. 실행 설정 `r{K}e{E}`

K step마다 새 관측으로 chunk를 생성합니다. step t의 실행 action은 t를 덮는(0 ≤ t − t_gen < 4) 최신 E개 chunk의 예측을 공식 가중치 규칙(temp −0.8, 최신 우선)으로 평균한 값이고, 그 뒤 sticky gripper를 그대로 적용합니다.

| 설정 | 재계획 | ensemble | 대응 (사용자 요청) |
|---|---|---|---|
| r1e4 | 매 step | 4 (공식) | A, B1/B4, C1, D1 |
| r1e2 | 매 step | 2 | B3 |
| r1e1 | 매 step | 없음 | B2, D2 |
| r2e2 | 2 step | 가능한 최대 2 | C2 (ensemble 유지) |
| r2e1 | 2 step | 없음 | C2 (ensemble 없음) |
| r4e1 | 4 step | 없음 (= 4) | C3, D3 ≡ D4 |

- 각 설정에서 natural, feedback, corrected, reverse를 80 에피소드씩 돌립니다.
- r1e4와 r4e1에서는 natural_gen도 80 에피소드씩 돌려 수치 경로 재실행 noise로 씁니다.
- 총 2,080 에피소드입니다.

## 3. Experiment F (fresh observation) — 생략, 근거

결정적 greedy 디코딩에서는 같은 관측으로 다시 생성하면 같은 chunk가 나옵니다. 그래서 "관측은 고정하고 재계획만" 하는 조건은 이미 실행한 chunk의 step 1을 다시 실행하는 action repeat가 되어, 공식 정책 의미를 크게 깨뜨립니다.
대신 구조적으로 분리합니다. 매 step 재계획 + ensemble 없음(r1e1)에서는 sequence reset만으로 step 2–4 문맥 feedback이 실행에서 빠집니다(관측과 무관). 반면 실행된 오차의 회복(새 관측의 기여)은 E의 오차 지속 곡선으로 봅니다. 이 한계는 RESULTS.md에 적습니다.

## 4. 측정값

**에피소드 수준**
- 성공.
- 실패 유형:
  - coke: 최종 `is_grasped` / `lifted_object`, 한 번이라도 grasp했는지.
  - move near: `moved_wrong_obj`, `near_tgt_obj`, `all_obj_keep_height`.

**궤적 지표 (paired natural = 같은 설정·과제·seed의 natural)**
- 실험 34 궤적 발산: 실행 world_vector 누적합 끝점의 차이(m).
- TCP 위치 오차 ‖p_cond(t) − p_nat(t)‖: 최대값, 최종값(t = 79), 평균(t ∈ [8, 80)).
- TCP 방향 차이: quaternion 각도.
- translation action 차이 ‖a_cond(t) − a_nat(t)‖.

**인과 오차 곡선**
- feedback 대 corrected: ‖p_F(t) − p_C(t)‖. 실행 교란은 같고 문맥만 다르므로, 순수 문맥 feedback이 만든 위치 차이입니다.
- reverse 대 natural: ‖p_R(t) − p_N(t)‖.
- 두 곡선을 t = 8…79마다 기록합니다.

**chunk·ensemble 지표**
- chunk 불일치: 같은 t에 두 조건이 모두 chunk를 생성했을 때 translation token이 다른 비율.
- ensemble 분산: 실행 step에서 평균에 들어간 예측 world_vector들의 평균 제곱 편차(E = 1이면 0).

**회복 지표**
- 재정렬 시점: window가 끝난 t = 24 이후, 위치 오차 ≤ 2 cm가 4 step 연속 유지되는 첫 t.
- 재정렬률: t = 79까지 재정렬된 에피소드 비율.
- 반감기: 평균 오차 곡선이 t = 24의 값의 절반 이하가 되는 첫 경과 step. 같은 정의를 에피소드별로도 계산해 중앙값을 냅니다.
- 위치 오차는 대비마다 정의합니다: feedback vs corrected, reverse vs natural, 그리고 각 조건 vs natural.

## 5. 통계

- 같은 (과제, seed)를 조건 간 paired로 비교합니다.
- 성공: exact McNemar. 차이의 95% CI는 과제로 층화한 에피소드 paired bootstrap(2,000회, seed 0).
- 연속 지표: paired Wilcoxon.
- 설정 간 대비 차이(예: (C − F)_{r4e1} − (C − F)_{r1e4}): 에피소드 paired bootstrap CI와, 에피소드 안에서 두 설정의 대비를 서로 바꾸는 sign-flip permutation(10,000회).
- 과제별 결과도 보고합니다.
- **주 검정:**
  - r4e1에서 C − F > 0, 그리고 N − R > 0.
  - r4e1 대 r1e4의 대비 차이.
- **보조 검정:** 나머지 설정(r1e2, r1e1, r2e2, r2e1)의 C − F, N − R. 설정 5개에 대해 Holm 보정 p를 함께 보고합니다.
- **seed 추가 규칙 (사전 고정):**
  - 조건: seed 0–39 결과에서 r4e1의 C − F와 N − R가 둘 다 가설 방향(> 0)인데, 둘 중 하나라도 McNemar p ≥ 0.05인 경우.
  - 이때만 r1e4와 r4e1의 natural / feedback / corrected / reverse를 seed 40–119로 확장(+640 에피소드)하고, 두 설정의 주 검정은 seed 0–119 합동 결과로 판정합니다.
  - seed 0–39 결과도 함께 보고합니다.
  - 그 밖의 경우에는 seed를 추가하지 않습니다.
  - 교란 크기·방향·window는 어떤 경우에도 바꾸지 않습니다.

## 6. 판정 기준 (사용자 기준 그대로)

**Strong support** — 아래 네 조건을 모두 만족할 때.
1. 공식 r1e4에서 corrected·reverse의 task 효과가 없음(C − F, N − R 모두 p ≥ 0.05).
2. 보호를 줄인 설정(r1e1, r1e2, r2e1, r2e2, r4e1) 중 하나 이상에서 C − F가 유의(Holm p < 0.05)하고, r1e4보다 유의하게 큼(대비 차이 CI가 0을 제외).
3. 그 설정에서 reverse가 natural보다 실패를 유의하게 늘림(N − R > 0, p < 0.05).
4. 보호를 줄일수록 오차 지속이 커짐. F 대 C, R 대 N의 평균 위치 오차가 r1e4보다 유의하게 크고(Wilcoxon p < 0.05), 반감기나 재정렬 시점이 더 김.

→ "Frequent replanning and temporal ensembling suppress autoregressive feedback amplification before it reaches task failure."

**Partial support** — 성공률 대비는 유의하지 않지만, 보호를 줄인 설정에서 기준 4(궤적 발산·회복 시간의 악화)가 명확히 성립할 때.

**No support** — 보호를 줄여도 task 수준 교정·reverse 효과가 없고, 궤적 지속도 커지지 않을 때.
→ "SpatialVLA's task-level robustness cannot be explained primarily by replanning or temporal ensembling."

추가 해석 규칙(판정을 바꾸지 않음):
- ensemble이 feedback을 막는지는 r1e4 대 r1e2 대 r1e1, r2e2 대 r2e1의 대비로 봅니다.
- 재계획이 막는지는 r1e1 대 r2e1 대 r4e1의 대비로 봅니다.
- 어느 쪽이 더 기여하는지는 대비 크기로 비교합니다.
