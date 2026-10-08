# P1-D: 실행 설정과 교란량의 통제 — 사전 등록 초안 (DRAFT, 승인 전)

- 작성: 2026-10-08. 계획: `/root/VLA/VLA_experiment_plan_20261008.md` §2, §9. 선행: 실험 35 (`outputs/spatialvla_feedback_protection_ablation/`), P0-C (`outputs/review_followup/P0C_exp35_fixed_axis/run_20261008/`), P0-A 감사.
- **상태: 본 실행 전.** 이 문서 작성 전에 실행한 것은 smoke test 3회(총 19 episode, 새 unit만 사용, 결과 지표는 분석하지 않음), 초기 상태 검사(CPU), 실험 35 raw의 정밀도 분석뿐이다. 승인 후 본 실행 전에 이 파일과 코드를 commit해야 사전 등록이 된다(이 agent는 commit하지 않았다).
- 표기: **[확인]** 실행·코드로 직접 확인, **[계산]** 이 준비 작업에서 계산한 값, **[미확인]**, **[가설]**.
- 코드: `scripts/review_followup/p1d/` — `p1d_runner.py`(실행기), `analyze_p1d.py`(분석), `precision_analysis.py`, `check_initial_states.py`, `build_units.py`. 기존 파일은 수정하지 않았고 `svla_protection.Executor / decode_fixed`, `spatialvla_core`, `svla_rollouts.perturb_token`을 import만 한다.

## 0. 준비 중 확인한 사실 (설계에 반영)

1. **move_near의 seed는 초기 상태를 거의 새로 만들지 않는다 [확인 + 계산].**
   - `reset(seed)`에서 episode_id = `RandomState(seed).randint(60)`, 배경 overlay = `RandomState(seed).choice(4)`이며 같은 첫 난수에서 나온다(520/520 seed 일치). 그래서 seed로는 60×4 = 240개 구성 중 결합된 일부만 나온다.
   - 실험 34/35의 seed 0–119는 move_near에서 **53개의 서로 다른 구성**뿐이다(seed 0–39는 29개). 중복 seed 쌍 67개 중 62개는 궤적이 같았고, 67개 모두 성공 여부가 같았다(r1e4 natural). 따라서 실험 35/P0-C의 move_near 표본은 명목 수(120/40)보다 작고, 그 CI는 좁게 계산됐다. 이 사실은 P0-B/P0-C에 전달해야 한다(이 작업에서 기존 결과는 수정하지 않음).
   - P1-D는 episode_id를 직접 지정한다(`build_units.py`). move_near unit 120개 = 실험 34/35에서 쓰지 않은 (episode, overlay) 구성, overlay당 30개(`default_rng(20261008)`). pick_coke_can은 위치가 연속 균등이라 seed 200–319의 120개가 모두 서로 다른 새 초기 상태다(520/520 고유).
2. **reset은 `reconfigure=True`일 때만 비트 단위로 재현된다 [확인].** 기본 reset은 같은 process에서 직전 episode가 같은 장면을 썼을 때 상태가 ~2e-6만큼 다르고 이후 궤적이 갈라진다(move_near seed 0). `reconfigure=True`이면 이전 이력과 무관하게 같은 seed에서 같은 상태 hash가 나온다(6/6 seed). `env.step`은 같은 action에 대해 결정적이다.
3. **실험 35에는 개입 전(t < 8) 분기가 있었다 [계산].** F/C/R 대 같은 설정 natural 2,400쌍 중 107쌍(28개 seed)에서 개입 전에 chunk token이 이미 달랐고, 실행 action·TCP까지 달라진 쌍은 coke 69/1,200, move_near 14/1,200이었다(최대 TCP 차이 6.7 cm, t < 8). 위 2의 이력 의존 reset 또는 decode 비결정성이 원인 후보이며, 어느 쪽인지는 **[미확인]**이다. 현재 GPU에서 batch 동반 행을 바꿔 같은 관측을 24회 디코딩하면 결과는 모두 같았다.
4. 위 1–3 때문에 P1-D는 **공통 prefix를 재연으로 복원**한다(§2.3). seed만으로 짝을 맞추지 않는다.

## 1. 질문과 가설

질문: 문맥(context) 교란이 궤적·과제에 미치는 효과를 실행 구조(재계획 간격 K, ensemble 수 E)가 조절하는가? 이때 (a) 문맥 교란 횟수와 (b) 실제로 실행된 교란량을 각각 고정한다.

- **H1 (재계획 축, 설계 A).** 문맥 예산을 고정하면(같은 4회의 generation call, 같은 action index) reverse − natural의 실제 TCP 편차 면적은 r4e1에서 r1e1보다 크다. 근거: r4e1은 문맥에 조건화된 step 2–4의 translation을 그대로 실행하고(노출 12 step), r1e1은 같은 step의 rot·grip만 실행한다(translation 노출 0). 직접 실행 dose는 두 설정 모두 0이다.
- **H2 (ensemble 축, 설계 B).** 직접 실행 dose를 맞추면(주입 step마다 실행 translation 변화 = Δ) feedback − corrected의 TCP 편차 면적은 r1e4가 r1e1보다 작다(ensemble이 문맥에서 생긴 차이를 흡수). 근거: P0-C에서 E 1 → 2가 F−C 면적을 줄였다(−128 cm·step, Holm 0.05). 다만 그 비교에서는 교란 chunk가 16개였고 dose도 달랐다. 방향은 사전에 지정하지만 검정은 양측으로 한다.
- **성공률(부차).** H1/H2와 같은 대비의 성공률 DiD. 정밀도 분석(§4) 결과, 성공률 상호작용을 primary로 삼기에는 표본이 부족하다. 따라서 성공률은 사전 지정한 의미 있는 효과 범위에 대한 등가(TOST)·추정 보고를 주목적으로 한다.

## 2. 설계

### 2.1 실행 설정 (같은 checkpoint `spatialvla-4b-224-sft-fractal`, bf16, RTX 5090, transformers 4.47 target dir)

| 설정 | 재계획 | ensemble | 축 | 주입 step의 age-0 가중치 w0 |
|---|---|---|---|---|
| r1e1 | 매 step | 없음 | 두 축의 기준 | 1.0 |
| r2e1 | 2 step | 없음 | 재계획 | 1.0 |
| r4e1 | 4 step | 없음(≡ e4) | 재계획 | 1.0 |
| r1e4 | 매 step | 4 (공식) | ensemble | 0.574 |

r1e2와 r2e2는 제외한다. 이유: 비용, 그리고 P0-C에서 E 2 → 4의 차이가 없었다(+1.7 cm·step). 설정 간 비교는 한 축씩만 한다(r1e1–r2e1–r4e1, r1e1–r1e4). r4e1 − r1e4 같은 교차축 비교는 하지 않는다.

### 2.2 개입

- **주입 시점:** 모든 설정에서 t ∈ {8, 12, 16, 20}에 생성되는 chunk 4개. 네 설정 모두 이 시점에 chunk를 생성한다. 다른 chunk는 건드리지 않는다.
- **교란:** step-1 translation token g → p = `perturb_token(g, 0.3, opposite)`. 실험 34/35와 같은 규칙이다. |v(g)| < 0.3이면 교란이 원점을 넘는 "방향 반전"이 되며, 주입마다 `flip` 표시로 기록한다.
- **조건:**

  | 조건 | 실행 step-1 | 문맥 | 추가 |
  |---|---|---|---|
  | N natural | g | g | — |
  | F feedback | p | p | — |
  | C corrected | p | g | — |
  | R reverse | g | p | — |
  | FD feedbackD | p | p | 주입 step에서 실행 world vector에 (1 − w0)·Δ 더함, Δ = phys(p) − phys(g) |
  | CD correctedD | p | g | 위와 같음 |
  | null | g | g | 개입 경로와 shadow decode를 그대로 거침 (no-op 검증) |
  | gen | — | — | t ≥ 8부터 모델 자체 `predict_action` (수치 경로 재실행 noise) |

- **설계 A (문맥 예산 고정):** 네 설정 × {N, F, C, R}.
  - generation call 수(4)와 바뀌는 index(step-1 translation)가 같다.
  - 실행 노출은 설정마다 다르며 따로 측정한다(§3).
  - E = 1 설정에서는 직접 실행 dose도 같다. r1e4의 F/C 직접 dose는 0.574배다.
- **설계 B (실행 섭동량 통제):** r1e4 × {FD, CD}.
  - 주입 step의 실행 translation = (같은 상태의 무교란 ensemble) + Δ가 되어, E = 1 설정의 F/C와 직접 dose가 같아진다.
  - E = 1에서는 FD ≡ F, CD ≡ C이므로 설계 B의 E = 1 arm은 설계 A의 F/C를 공유한다(같은 episode).
  - 문맥 수정 횟수는 이 방법으로는 달라지지 않는다(4회 그대로).
  - 실행 action은 더 이상 모델 token에서 나오지 않는다. 실험자가 주입한 물리적 offset이며, 물리적으로 실행 가능하다.
  - 대안이던 "r1e4의 교란 chunk 수를 1/w0 ≈ 7개로 늘리는" 방식은 주입 시점·간격이 바뀌어 버렸다(기각).
- **dose와 노출은 완벽히 맞지 않는다.** 주입 시점의 상태가 설정마다 달라(t < 8에서 이미 실행이 다름) g, Δ, 반전 여부가 달라진다. 달성 dose를 episode·주입 단위로 보고하며, dose로 사후 정규화한 효과는 결론에 쓰지 않는다.

### 2.3 공통 초기 상태와 전체 상태 복원

- 평가 unit = (task, unit). 같은 unit은 모든 설정·조건에서 같은 `reset(seed, reconfigure=True[, episode_id])` 상태에서 시작한다.
- 각 (task, unit, 설정) group은 N을 먼저 실행한다.
- t = 8에서 N의 executor(chunk buffer = ensemble buffer, sticky gripper 상태)를 deep copy하고, t < 8의 정확한 float64 action을 저장한다.
- 다른 조건은 다음 순서로 진행한다.
  1. 같은 seed로 reset하고 N의 action을 재연한다(decode 없음).
  2. 재연한 매 step에서 전체 simulator 상태 hash(`env.get_state()`)와 관측 이미지 hash가 N과 같은지 확인한다(`restore.ok`).
  3. executor 사본을 복원하고 t = 8부터 closed loop로 진행한다.
- 따라서 controller·gripper target(get_state에 포함되지 않음)도 재시뮬레이션으로 복원된다. 관측 시점, 종료 상태(80 step 고정)도 같다.
- greedy decoding이라 RNG는 쓰지 않는다. decode는 실험 35와 같이 prompt 길이 group마다 8행으로 고정한다.

### 2.4 관측 갱신과 history reset의 분리 — 불가, 한계로 명시

- SpatialVLA의 "history"는 chunk 안의 token prefix다. 이것은 generation call마다 초기화되고, 실행 가능한 정책에서는 매 call이 새 관측을 쓴다.
- 2×2 분리의 각 칸은 다음과 같다.
  - **갱신 + reset:** 매 step 재계획(r1e1)이다.
  - **둘 다 없음:** r4e1의 step 2–4다.
  - **reset만:** r4e1에서 같은 관측의 chunk를 문맥 g로 다시 디코딩한 것이다. 이는 바로 C/N 조건이고, 이미 설계 안에 있다.
  - **갱신만(새 관측 + 이전 교란 문맥 유지):** 새 관측의 chunk에 이전 token을 강제 prefix로 넣어야 한다. chunk의 step-1 slot이 현재 시점의 action이라는 의미가 깨지므로 학습 분포 밖이다.
  - 같은 관측으로 다시 계획하는 방식은 결정적 greedy decoding에서 같은 chunk를 반복하게 될 뿐이다(실험 35 §3).
- **결론:** 갱신과 reset을 물리적으로 실행 가능한 정책 안에서 분리하지 않는다. P1-D의 재계획 축 결론은 "관측 갱신 + 문맥 reset의 복합 실행 설정"의 효과로 제한한다.
- shadow decode(§3)는 실제 상태에서 문맥만 바꾼 counterfactual chunk를 계산한다. 이것은 실행되지 않는 분석이므로 closed-loop 결과와 따로 표시한다. baseline 관측의 강제 replay 분석은 수행하지 않는다.

### 2.5 새 과제 (추가하지 않음)

- **[가설]** 문맥 오차가 과제 실패로 번지는 것은 노출 구간이 접촉·gripper 결정 시점과 겹칠 때뿐이다. 근거는 다음과 같다.
  - P0-C에서 F−C 효과가 coke의 grasp 실패·놓침(sticky onset 변화)에 집중됐다.
  - move_near에서는 궤적 효과의 방향이 반대였다.
- 이 가설을 검정하려면 주입 시점을 과제의 접촉 단계에 맞춘 설계, 또는 접촉 시점이 window 안에 들어오는 과제(예: drawer open/close: articulated 접촉, 더 긴 horizon)가 필요하다.
- 현재 SpatialVLA의 drawer 과제 기저 성공률과 접촉 시점 분포를 모른다 **[미확인]**. 또 과제를 추가하면 비용이 1.5배가 된다.
- 그래서 P1-D 본 실행에는 넣지 않는다. 대신 접촉 시점 대 주입 시점(first_contact_t, first_grasp_t)을 기록해 이 가설의 탐색 자료로 쓴다. 과제 추가는 별도 승인 사항이다.

## 3. 측정 (episode / step / 주입 단위; `episodes.jsonl`)

- **step:**
  - 실제 TCP pose(위치·quaternion), gripper closedness
  - 손가락 link와 접촉한 물체(impulse > 1e-6), info flag(is_grasped, lifted_object, moved_correct_obj, near_tgt_obj 등)
  - 실행 action(float64), ensemble 예측·age, 상태 hash
- **generation call:**
  - 실행/문맥 token, g, counterfactual 문맥 chunk(shadow row, 같은 fixed batch)
- **주입:**
  - g, p, 달성 정규화 거리, |v(g)|, 반전 여부, Δ_phys, w0, offset
  - 직접 실행 dose = w0·Δ (+offset)
- **노출 (설정 간에 다르며, 측정만 함):**
  - (i) 구조적 가중치 노출: 주입 chunk에서 온 age ≥ 1 예측의 실행 가중치 합. smoke 확인값은 r1e1 0, r2e1 4, r4e1 12, r1e4 1.70.
  - (ii) 문맥 기인 실행 편차: 실제 action과, 주입 chunk를 counterfactual 문맥 chunk로 바꾼 action의 차이(translation m, rotation rad, gripper). 둘 다 같은 상태, sticky 적용 전이다.
- **episode:**
  - 성공(마지막 step done)
  - 실패 유형 flag
  - 첫 grasp·첫 접촉 시점
  - sticky onset 수
  - overhead: wall time, decode time, generation call 수
- **paired 지표 (P0-C와 같은 정의):**
  - TCP 편차 면적(t = 8–79 합, cm·step) — **primary**
  - 평균·최대·최종 편차
  - 창 이후 면적
  - command proxy: 실행 world vector 누적합의 차이. P0-A에 따라 **실제 TCP가 아니라고 표시**한다.
  - 재정렬 시간: t ≥ 24 이후 편차 ≤ 2 cm가 4 step 지속되는 첫 시점. 끝까지 재정렬되지 않으면 80으로 censor한다.
  - 첫 grasp/접촉 시점 이동, ever-grasped 차이, sticky onset 차이

## 4. 표본 크기 — 정밀도 분석 [계산]

- `precision_analysis.py`: 실험 35 raw를 사용했다. per-episode SD는 task 층화 bootstrap(2,000회)의 90% 상한이다. move_near의 중복 구성은 제거했다(`precision_dedup/`). 정규근사를 사용했다.
- 실험 35는 r1/r2 설정에서 교란 chunk가 16/8개였다(P1-D는 4개). r4e1은 P1-D와 같은 주입 시점이다(8, 12, 16, 20). 따라서 r1 설정의 SD는 보수적(큰 쪽) 추정일 가능성이 있다.

| 추정량 (P1-D 대응) | SD/episode (90% 상한) | n = 160 CI 반폭 | n = 240 CI 반폭 | 검정력 (n = 240) |
|---|---|---|---|---|
| 성공 R−N, 한 설정 | 0.46–0.50 | ±7.6 %p | ±6.2 %p | 10 %p: 0.88 (α 0.05) |
| 성공 DiD R−N r4e1 − r1e1 (S1) | 0.61 | ±9.4 %p | ±7.7 %p | 10 %p: 0.72 (α 0.05), 5 %p: 0.25 |
| 성공 DiD F−C (S2/S3) | 0.88–0.93 | ±14 %p | ±11–12 %p | 10 %p: ≈ 0.4 |
| 면적 DiD R−N r4e1 − r1e1 (H1) | 349 cm·step | ±54 | ±44 | 100 cm·step: 0.97 (α 0.0125) |
| 면적 DiD F−C r1e4 − r1e1 (H2) | 443 cm·step | ±69 | ±56 | 100 cm·step: 0.84 (α 0.0125) |

- 성공률 DiD를 ±5 %p 정밀도로 추정하려면 R−N은 약 570, F−C는 약 1,200 episode가 필요하다. 그래서 성공률 상호작용은 primary가 될 수 없다.
- **선택: unit 120개/task, 설정·조건당 n = 240 episode.**
  - H1/H2(궤적)에서 의미 있는 효과 100 cm·step에 대한 검정력이 충분하다(0.84–0.97).
  - 성공률 S1은 등가 판정이 가능하다. 90% CI 반폭이 약 6.5 %p이므로, 추정치가 |Δ| ≤ 약 3.5 %p이면 ±10 %p 등가를 선언할 수 있다.
  - F−C 성공률 DiD는 n = 240에서도 ±11 %p라 결론을 내리기 어렵다. 이 점을 사전에 명시한다.

## 5. 통계 계획

- **평가 단위:** episode = (task, unit). 같은 episode를 조건 간·설정 간 paired로 비교한다. 두 task를 같은 가중치로 층화한다.
- **추정:** task별 평균의 평균. 95% CI는 task 층화 episode bootstrap(2,000회, seed 0)이다. 한 episode의 모든 설정·조건을 함께 재추출한다.
- **검정:**
  - 설정 내 성공 대비: exact McNemar
  - 성공 DiD: sign-flip permutation (10,000회)
  - 연속 대비·DiD: sign-flip permutation(평균)을 주 검정으로 하고, Wilcoxon은 함께 보고한다.
- **Family와 보정:**
  - **Primary (Holm, 2개):**
    - H1 = 면적 DiD[R−N] r4e1 − r1e1 (설계 A)
    - H2 = 면적 DiD[F−C] r1e4(FD−CD) − r1e1(F−C) (설계 B)
  - **Secondary 성공 (Holm, 4개):**
    - S1 DiD[R−N] r4e1 − r1e1
    - S2 DiD[FD−CD vs F−C] r1e4 − r1e1
    - S3 DiD[F−C] r4e1 − r1e1
    - S4 DiD[R−N] r1e4 − r1e1
  - **Secondary 궤적 (Holm, 6개):**
    - T1 DiD[R−N] r2e1 − r1e1
    - T2 DiD[F−C] r4e1 − r1e1
    - T3 DiD[R−N] r1e4 − r1e1
    - T4 DiD[F−C] r1e4 − r1e1 (설계 A: dose 불일치, B와 대조용)
    - T5 최대 편차 DiD[R−N] r4e1 − r1e1
    - T6 재정렬 시간 DiD[R−N] r4e1 − r1e1
  - 그 밖의 모든 표(설정 내 대비, task별 값, 접촉·gripper 사건, 노출·dose)는 탐색적이며 보정하지 않는다.
- **의미 있는 효과 범위 (결과 전에 고정):**
  - 성공률: |Δ| ≥ 10 %p이면 의미 있음, < 5 %p이면 무시할 수준. 등가 판정은 90% CI가 ±10 %p 안일 때다. 참고로 실험 35 r1 설정에서 실행 교란 자체의 효과(N − F)가 15–21 %p였다.
  - 궤적: |면적 DiD| ≥ 100 cm·step(t = 8–79 평균 약 1.4 cm)이면 의미 있음. 실험 35 재실행 noise 면적은 180–200 cm·step이었다.
  - 이번 실행의 gen − N, null − N 기준선을 함께 보고한다. gen과 null은 unit 0–39에서만 측정한다.
- **판정 문구 (사전 고정):**
  - H1이 Holm p < 0.05이고 추정치 ≥ 100 → "문맥 예산을 고정해도 재계획 간격이 문맥 효과의 궤적 지속을 키운다(복합 설정 효과)".
  - p < 0.05이지만 < 100 → "통계적으로는 검출되나 사전 정의한 의미 범위 미만".
  - H2도 같은 구조로 판정하되, 방향이 음수일 때만 "ensemble이 흡수"라고 쓴다.
  - 성공률이 등가 범위 안이면 "과제 수준 효과는 ±10 %p 안(trajectory-only)". 등가도 유의도 아니면 "판별 불가".
- **제외 규칙:**
  - `restore.ok = False`인 branch는 primary 분석에서 제외한다. 이는 기술적 실패이며, 조건별 분모와 무제외 민감도를 함께 보고한다.
  - parser 실패는 없다(token decoding이 고정됨).
  - 반전 교란 episode는 제외하지 않는다. 반전 비율을 설정·조건별로 보고하고, 주입 4회 중 반전이 0회인 episode의 하위집합 분석을 탐색적으로 낸다.

## 6. 실행 순서와 종료 규칙

1. **Block 1** (unit 0–39/task):
   - 네 설정 × {N, F, C, R, null}
   - r1e4 × {FD, CD}
   - r1e1·r4e1 × gen
   - 총 1,920 episode.
   - **기술 gate:** verification 필드만 본다. restore_ok 100%, null ≡ N ≥ 95%, prefix 일치 100%. 결과 지표는 열람하지 않는다. gate를 통과하지 못하면 중단하고 보고한다.
2. **Block 2** (unit 40–119/task): 네 설정 × {N, F, C, R} + r1e4 × {FD, CD}, 총 2,880 episode.
3. 고정 n이다. 중간 효능 분석이나 결과에 따른 seed 추가는 하지 않는다. 기술적 중단 시 이미 끝난 unit만으로 분석하되, 계획 대비 n을 명시한다.

## 7. 명령과 예상 시간

```bash
# 승인 후, 이 파일·코드 commit 다음에 실행
bash /root/VLA/autovla_misalignment_poc/outputs/review_followup/P1D_dose_control/run_20261008/launch.sh
```

(`launch.sh`가 GPU 여유 메모리 13 GB 이상인지 확인하고, block 1 → gate → block 2 → 분석을 차례로 수행한다.)

- 총 4,800 episode. 실험 35의 처리량(8 worker, 2.7 ep/min)을 기준으로 하면 약 30시간이다.
- branch는 72 step이라 실제로는 25–35시간으로 추정한다.
- smoke(2 worker): 8 episode에 3.5분, 7 episode에 4.0분이었다.
- block 1은 약 12시간, block 2는 약 18시간이다.
- RTX 5090에 SpatialVLA가 약 12 GB를 쓴다. 다른 작업이 GPU를 함께 쓰면 OOM 위험이 있다(smoke 중 2회 OOM으로 시작 실패, 재시도 후 정상).

## 8. Smoke test 결과 (새 unit만, 결과 지표는 분석하지 않음) [확인]

| 검사 | 결과 |
|---|---|
| 재연 복원 (상태 hash + 이미지 hash, t < 8) | 13/13 branch 일치 |
| null ≡ N (t = 0–79 action·상태 hash 비트 일치) | 2/2 (r4e1 coke, r1e4 move_near) |
| N 재실행 (다른 process·worker 수·동반 행) ≡ N | 2/2 |
| 주입 시점 | 모든 설정에서 정확히 8, 12, 16, 20 (4회) |
| 구조적 노출 | r1e1 0, r2e1 4, r4e1 12, r1e4 1.70 |
| 직접 실행 dose (4 주입 합) | E = 1의 F/C: 0.22–0.25 m; r1e4 FD/CD: 0.22–0.24 m(offset 적용, Δ 크기와 일치); r1e4 null/R: 0 |
| move_near episode_id·overlay 지정 | assert 통과 |

## 9. 남은 불확실성

- 실험 35 개입 전 분기의 원인 **[미확인]**: reset 이력 또는 decode 비결정성. P1-D는 재연 복원으로 이 문제를 우회한다.
- "opposite 0.3"의 방향 반전(실험 35에서 27–55%)은 그대로 남는다. 반전이 없는 교란(예: 크기를 |v|로 제한)으로 바꿀지는 승인 사항이다.
- gen/null 기준선은 unit 0–39에서만 측정한다.

## Lead 결정 (2026-10-08, 사전 등록 commit 직전, 본 실행 결과를 보기 전)
1. 이 protocol과 코드를 commit해 사전 등록으로 고정한다.
2. 주 분석은 궤적(H1, H2; Holm 2)으로 하고, 성공률은 보조 family와 동등성 판정으로 둔다. N = 과제당 120 단위, arm당 240 episode, 고정 n이며 결과에 따른 확장은 없다.
3. r1e2와 r2e2는 제외한다(P0-C에서 E2와 E4의 차이가 없었다).
4. 실험 34/35와 비교할 수 있도록 "opposite 0.3"을 유지한다. 원점을 지나 방향이 뒤집히는 교란(reversal)은 기록된 reversal flag로 사전 지정 민감도 분석에서 따로 본다.
5. 새 과제는 추가하지 않는다. 접촉 시점은 탐색용으로 기록한다.
6. move_near는 seed 대신 episode id를 직접 지정한다.
7. move_near 중복 seed 문제와 실험 35의 개입 전 발산은 P0-B/P0-C와 실험 34/35 해석에 대한 정정 사항으로 따로 기록한다(이 실험의 판정과는 무관).
8. GPU는 RTX 5090 하나다. 실행 순서는 P1-A/B/R8R9 → P1-C → P1-D이다.
