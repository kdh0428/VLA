# Cross-model replication — 정량 근거 정리 (논문용)

경로는 모두 `/root/VLA/autovla_misalignment_poc/` 기준입니다. 이 문서는 기존 결과만 사용했고, 새 실험은 하지 않았습니다.

- "(recomputed: …)" 표시: raw 데이터에서 **원 분석 스크립트와 같은 방법**으로 이 패키지에서 다시 계산한 값입니다. 계산 script는 `outputs/paper_quantitative_package/_parts/scripts/`에 있습니다.
- 그 밖의 값: `analysis.json`(JSON key 표기)이나 RESULTS.md에서 그대로 옮겼습니다.
- 출처 상세: `source_map_crossmodel.md`. 충돌·불일치: `conflicts_crossmodel.md`.

AutoVLA 참조값 (모두 open-loop, navtest PoC, log 단위 cluster bootstrap 2,000회, McNemar):

| 항목 | 값 | 출처 |
|---|---|---|
| 직전 token 교정 (A− 365 단위 / 52 장면) | Normal 47.4% [38, 57] → Recent-GT 7.1% [4, 10]<br>Δ −40.3%p [−50.3, −30.0], p = 8e-42 | `outputs/action_history_causal/ACTION_HISTORY_CAUSAL.md` |
| reverse@emb | GT-history 4.4% → 33.4% [26.3, 43.2]<br>+29.0%p [+21.1, +38.0], p = 6.7e-31 | `outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` |
| 교정 window 1/2/3/4 step | full 효과의 47 / 76 / 90 / 96% | `outputs/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` |
| 방향 오답 − 크기 오답 | +9.0%p [+3.1, +18.7], p = 6e-4 | `outputs/motion_semantics_ablation/RESULTS.md` |
| 같은 거리 대안 token의 증폭 | A− 장면 45.0% vs A+ 장면 4.8% | `outputs/equal_distance_perturbation/EQUAL_DISTANCE.md` |

---

## A. Impromptu VLA 3B (실험 33) — 시간축 action-chunk 모델, **Strong replication**

### A.1 설정
- 사전 등록 038a6d3, 결과 698148a. RTX 5090.
- 모델: `aaaaaap/ImpromptuVLAModel/3B_AD` (Qwen2.5-VL-3B 전체 fine-tune).
- 출력: 미래 10개 waypoint(0.5–5 s)를 **현재 ego 좌표의 절대 위치 텍스트** `[x, y]`로 시간순 autoregressive 생성합니다(waypoint당 digit token 약 12개). greedy 디코딩에 repetition penalty 1.05를 적용했습니다.
- 장면:
  - navtest PoC 28 log의 2,748 장면 중 해석 가능한 **2,728 장면** (`AB.n_scenes`, `AB.n_logs` = 28)
  - C–E는 무작위 300 장면 × 8 방향 = **2,398 단위** (2,400 중 2 단위 parse 실패; `CDE.n_units`)
- 통계: log 단위 cluster bootstrap 95% CI (2,000회, seed 0), McNemar exact, Wilcoxon (`scripts/cross_vla_temporal/analyze_temporal.py`).
- 모든 지표는 **open-loop 궤적 지표**입니다. A−는 P/R/A accept 실패이고, 증폭은 A− ∧ FDE5 > 3 m입니다.

### A.2 자연 성능과 재실행 변동 (`analysis.json` `AB.natural`, `AB.rerun_variability`)
- 자연 성능: A− **33.4% [26.2, 38.0]**, 증폭 31.5% [24.4, 36.2], FDE5 **5.75 m [5.17, 6.27]** (N = 2,728)
- batch 구성만 바꾼 재실행(natural vs natural_rep):
  - A− 불일치 **5.9% [5.2, 6.9]**, 증폭 불일치 5.9% [5.1, 6.7]
  - 최종 waypoint 이동 0.90 m [0.84, 0.96]
  - A− 비율 차이 +0.18%p [−0.74, +1.08] (McNemar 83 vs 78, p = 0.75)
- 자연 A+ 장면이 재실행만으로 A−가 되는 비율: **4.6% [3.7, 5.5]** (N = 1,818)

### A.3 0.2 m 첫 waypoint 교란 (Experiment A; `AB.A.m0.2`)
- 단위: 2,728 장면 × 8 방향 = 21,824. 기준은 pert_m0(δ = 0, 같은 prefix 경로)입니다.

| 지표 | 값 | N |
|---|---|---|
| 기준 A+ → 교란 후 A− | **49.5% [46.9, 52.5]** (재실행 기저 4.6%의 약 10배) | 14,424 |
| 교란 후 A− − 기준 A− (paired) | **+24.8%p [+19.4, +32.2]**, McNemar 7,138 vs 1,731, p ≈ 0 | 21,824 |
| r ≥ 3 (궤적 3배 이상 증폭) | **97.4% [96.9, 97.9]**, r 중앙값 27.6 | 21,824 |
| r ≤ 1 (흡수) | **0.45% [0.34, 0.60]** | 21,824 |
| (0.5 m) A+ → A− / paired | 58.6% [55.9, 61.4] / +31.1%p [+25.4, +39.1] | |

### A.4 같은 크기 교란의 안정·불안정 분기 (Experiment B, 사전 등록 기준 3; `AB.B`)
- 같은 장면에 안정 분기(A+ ∧ r ≤ 1)와 불안정 분기(증폭 ∨ r ≥ 3)가 공존하는 비율:
  - 0.2 m: **2.2% [1.5, 3.2]**
  - 0.5 m: **3.2% [2.4, 3.9]**
- 두 값 모두 재실행 불일치 5.9%보다 작으므로 **기준 3은 미충족(FAIL)**입니다.
- log r 분산 중 장면 간 비율(ICC): 0.32 (0.2 m), 0.21 (0.5 m)
- (탐색적) task 수준에서 A+ 방향과 증폭 방향이 같은 장면에 공존하는 비율은 86.9–92.6%입니다 (`exploratory_task_level_branches.json`, 사전 등록 밖).

### A.5 직전 action 교정과 window (Experiment C; `CDE.all_units`)
- 실행 궤적은 [강제 w̃_1, 자기 생성 w_2..w_10]으로 고정하고, **문맥만** 교정했습니다.

| 행 | 증폭 [95% CI] | A− | normal 대비 Δ증폭 [95% CI], McNemar p | full 효과 대비 (증폭) [95% CI] |
|---|---|---|---|---|
| normal | **63.4% [60.6, 66.7]** | 64.5% | – | – |
| win1 | 48.0% [42.8, 52.2] | 48.8% | −15.4%p [−22.0, −9.5], 7e-24 | 24.3% [15.6, 33.8] |
| win2 | 38.9% [33.7, 43.2] | 40.3% | −24.6%p [−31.4, −18.2], 2e-69 | 38.7% [29.7, 47.8] |
| win3 | 6.0% [3.0, 8.2] | 9.5% | −57.4%p [−62.4, −54.0], ≈0 | **90.5% [87.0, 95.4]** |
| win4 | 3.2% [0.9, 5.2] | 6.3% | −60.2%p [−64.4, −57.4], ≈0 | 94.9% [91.8, 98.6] |
| gt_history | **0.0%** | 0.0% | −63.4%p [−66.7, −60.6], ≈0 | 100% |
| recent_gt (직전 1개만 GT) | **1.3% [0.8, 1.8]** | **16.3%** | **−62.2%p [−65.6, −59.1]**, ≈0 (기록값 0.0, underflow) | 98.0% [97.2, 98.8] |

- **63.4 → 1.3%의 절대 효과: −62.2%p [−65.6, −59.1]** (N = 2,398, log 28개 cluster).
  - 1.3%는 *증폭률*입니다. 같은 행의 A−는 64.5 → 16.3%로, Δ −48.1%p [−52.3, −44.4], p = 8e-244입니다.
  - 절대 위치 표현이라 GT 문맥에는 GT 정보 누출이 섞입니다 (conflicts M2).
- window 기준 사전 등록 기준 4(단조 증가, win4 이내 80% 이상)는 **충족**입니다: 24 → 39 → 91 → 95%.

### A.6 reverse (Experiment D; `CDE.all_units.reverse_vs_gt_history`)
- gt_history 문맥 끝에 자기 생성(오차를 품은) waypoint를 다시 넣었습니다.
- 증폭: **0.0 → 81.9% [80.4, 84.2]**, Δ +81.9%p [+80.4, +84.2] (McNemar 1,964 vs 0, p ≈ 0)
- 최종 발산 D10: **+40.5 m [+37.3, +43.9]** (Wilcoxon p ≈ 0), FDE5 +39.9 m [+37.0, +43.1]
- reverse − normal: 증폭 +18.5%p [+15.6, +21.7], p = 2e-78

### A.7 motion semantics (Experiment E; `CDE.all_units.motion`)
- **GT 근처 값 ≈ GT** (near_gt − recent_gt): 증폭 −0.08%p [−0.60, +0.30], **p = 0.87**
- **방향 오답 − 크기 오답**:
  - 증폭 **+15.6%p [+9.0, +24.3]**, p = 2.0e-31
  - FDE +14.5 m [+12.7, +15.8]
  - D10 +18.4 m [+15.9, +20.1], p = 1.8e-268
- (탐색적) 오차 크기 e를 맞춘 첫 대체 시점(waypoint 3)에서도 방향 오답이 다음 waypoint 오차를 **+0.29 m [+0.25, +0.33]** 더 키웁니다 (Wilcoxon p = 2.0e-99, N = 2,398; `exploratory_matched_first_substitution.json`).

### A.8 사전 등록 기준 (PROTOCOL §5)

| 기준 | 결과 |
|---|---|
| 1 correction | PASS (−62.2 / −63.4%p) |
| 2 reverse | PASS (+81.9%p, +40.5 m) |
| 3 branch > 재실행 변동 | **FAIL** (2.2 / 3.2% < 5.9%) |
| 4 window | PASS |
| 5 방향 > 크기 | PASS |
| task 수준 | PASS (A− +24.8%p vs 기저 5.9%, 교정이 증폭 감소) |

→ **Strong replication**. 단서는 세 가지입니다: open-loop 궤적 실패라는 점, 궤적 수준의 흡수가 없다는 점, GT 정보 누출입니다.

---

## B. OpenVLA-7B / LIBERO-Spatial (실험 28) — **구조적 대조군(architectural control)**

### B.1 설정
- 사전 등록 6b9e8ad. 재실행 대조군 amendment 6d0375a는 Phase B 일부를 본 뒤·최종 분석 전에 추가됐습니다. 결과 commit 69402c6.
- 모델: `openvla/openvla-7b-finetuned-libero-spatial` (bf16, RTX 5090).
- 과제: LIBERO-Spatial **10 과제 × 초기 상태 0–9 = 100 에피소드**.
- 구조: step마다 7개 token(x, y, z, roll, pitch, yaw, gripper)을 차원 순서로 생성하고, **다음 step 입력에 이전 action이 없습니다**. 따라서 feedback은 step 안의 차원 간 조건화뿐입니다.
- 통계 (`scripts/cross_vla/analyze_cross_vla.py`):
  - 에피소드 paired McNemar exact
  - 과제 단위 cluster bootstrap (2,000, seed 0)
  - 과제 단위 exact sign-flip permutation (2^10)
  - Phase C는 과제 단위 bootstrap **1,000회**

### B.2 기본 성능과 재실행
- 자연 성공: **87/100 = 87%** (`natural_success`). 과제별 10, 10, 10, 9, 7, 9, 9, 7, 9, 7.
- 공식 보고 84.7%는 `EXPERIMENT_SUMMARY.md` §6에만 있습니다 (conflicts m2).
- 재실행(natural_rep, batch 구성만 다름): **81/100**
  - 차이 −6%p [−14, +2], McNemar 7 vs 13, p = 0.26, perm p = 0.30
  - **결과 불일치 20/100 = 20%**
  - 자연 성공 87개 중 재실행에서 실패한 비율 13/87 = 15%

### B.3 step 안 token feedback (Phase C, 2,462 프레임, 시뮬레이션 없음; `phaseC`)

| 개입 (d = 0, x) | N 단위 | 증폭 (하위 차원 \|Δbin\| ≥ \|δ\|) [95% CI] | 하위 편차 D (bin) | 회복 |
|---|---|---|---|---|
| feedback, \|δ\| = 8 | 4,741 | **92.6% [89.8, 94.4]** | 87.7 | 1.5% |
| feedback, \|δ\| = 24 | 4,741 | **81.2% [77.2, 84.0]** | 106.5 | 0.2% |
| corrected (교란 실행 + 정상 문맥) | 4,741 | 0% | 0 | 100% |
| reverse (정상 실행 + 교란 문맥) | 4,741 | 92.6% (feedback과 같음) | 87.7 | 1.5% |
| d = 1 (y), \|δ\| = 8 / 24 | 4,869 | 87.4% [85.4, 89.2] / 75.3% [72.4, 77.8] | 64.2 / 83.9 | 4.2 / 1.3% |
| 차원 축 "window" 1/2/3/4 (x, 8) | 4,741 | 84.5 / 66.9 / 53.4 / 26.8% | 62.0 / 36.7 / 23.6 / 8.5 | 5.6 / 19.5 / 30.7 / 60.1% |

- 하위 token은 **문맥에 들어간 token이 결정**합니다. 이 결과는 결정적 greedy 디코딩에서 정의상 그렇게 됩니다: reverse = feedback, corrected = 0.
- closed loop 교란 구간 안에서 같은 관측의 자연 디코딩과 비교해도 같은 결과입니다 (`closed_loop_window_token_stats`): feedback_d8 증폭 79.3%, reverse_d8 76.2%, corrected 0% (각 1,000 step).
- 차원 축 window는 시간축 window가 아니므로 AutoVLA·Impromptu의 window와 비교할 수 없습니다.

### B.4 과제 수준 (Phase A/B, 교란 control step 10–19, 각 100 에피소드)

| 대비 | 성공률 a vs b | Δ [과제 bootstrap 95% CI] | McNemar (a only / b only), p | 과제 perm p | 불일치 |
|---|---|---|---|---|---|
| feedback_d8 − natural | 82 vs 87 | −5%p [−13, +2] | 6/11, 0.33 | 0.39 | 17% |
| feedback_d24 − natural | 84 vs 87 | −3%p [−14, +8] | 8/11, 0.65 | 0.74 | 19% |
| corrected_d8 − natural | 79 vs 87 | −8%p [−15, 0] | 6/14, **0.115** | 0.14 | 20% |
| corrected_d24 − natural | 79 vs 87 | −8%p [−17, +1] | 8/16, 0.15 | 0.21 | 24% |
| reverse_d8 − natural | 84 vs 87 | −3%p [−9, +3] | 5/8, 0.58 | 0.56 | 13% |
| reverse_d24 − natural | 79 vs 87 | −8%p [−15, 0] | 7/15, 0.13 | 0.14 | 22% |
| **feedback − corrected** (d8 / d24) | 82 vs 79 / 84 vs 79 | +3%p [−4, +9] / +5%p [−4, +14] | 11/8, 0.65 / 12/7, 0.36 | 0.59 / 0.43 | – |
| natural_rep − natural (대조군) | 81 vs 87 | −6%p [−14, +2] | 7/13, 0.26 | 0.30 | **20%** |
| 각 조건 − natural_rep | – | −2 … +3%p | p ≥ 0.63 | ≥ 0.45 | – |

- 자연 성공 87개 중 교란 후 실패 비율은 9.2–18.4%입니다. 교란 없는 재실행의 실패 비율은 15%입니다.

### B.5 OpenVLA가 구조적 대조군이라는 수치 근거
1. **step 안 feedback은 강합니다.** x token 8 bin 교란에서 하위 차원 증폭이 92.6%입니다. 이는 AutoVLA·Impromptu의 문맥 의존성과 같은 "문맥 token이 다음 token을 결정"하는 구조입니다.
2. **시간축으로는 전달되지 않습니다.**
   - 모든 교란 조건이 natural 대비 −3 ~ −8%p이고, McNemar p ≥ 0.115, 과제 permutation p ≥ 0.14입니다.
   - feedback − corrected는 +3 / +5%p이고 CI가 0을 포함합니다.
   - 이 크기는 **교란 없는 재실행의 차이(−6%p, 불일치 20%)와 같은 범위**입니다.
3. 시간축 증폭이 없는 모델에서는 token 수준 효과가 크더라도(92.6%) 과제 수준 효과가 측정되지 않습니다. 이는 "시간축 action history가 증폭의 조건"이라는 해석과 맞습니다. 다만 검정력에 한계가 있습니다(100 에피소드, 과제 10개).

---

## C. SpatialVLA-4B / SimplerEnv (실험 34) — **Partial replication** (token Strong, task 없음)

### C.1 Architecture (코드 확인)
- 실험 정보: 사전 등록 f578eae, 결과 f4cf886. RTX 5090.
- checkpoint: `IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal`
- backbone: PaliGemma2-3B (Gemma2) + SigLIP + Ego3D(ZoeDepth)
- 출력: **4-step chunk**, step당 **3 token**:
  - translation: 구면 bin θ16 × φ32 × r8 = 4,096
  - rotation: 16³
  - gripper: 2
- 디코딩: prefix 뒤 suffix를 causal greedy로 생성합니다. step k가 step < k의 token을 attend합니다.
- 실행 방식 (공식 adapter `spatialvla_policy.PolicyState` 그대로):
  - **매 control step 재계획**
  - 실행 action = 최근 4개 chunk에서 현재 step에 해당하는 예측의 `ActionEnsembler(4, −0.8)` 가중 평균
  - sticky gripper (10회 반복)
- **ensemble 가중치** (recomputed: `svla_token_step_change.py`, 규칙 exp(0.8·i)/Σ, index 0 = 가장 오래된 예측):

  | 예측 | 가중치 |
  |---|---|
  | 현재 chunk step 1 | **0.574** |
  | 1 step 전 chunk step 2 | **0.258** |
  | 2 step 전 chunk step 3 | **0.116** |
  | 3 step 전 chunk step 4 | **0.052** |

  - 앞 step 문맥에 조건화된 step 2–4가 실행 action에서 차지하는 몫은 **42.6%**입니다.
- **정정 기록**:
  - 실험 34 PROTOCOL.md §0의 "오래된 예측일수록 가중치 큼 … 실행 action의 대부분은 step 2–4에서 온다"는 **틀렸습니다**. 실제로는 최신 예측이 가장 크고, step 2–4의 몫은 42.6%입니다.
  - 실험 34는 공식 코드를 그대로 실행했으므로 결과에는 영향이 없습니다.
  - 같은 정정이 실험 35 사전 등록(`outputs/spatialvla_feedback_protection_ablation/PROTOCOL.md` §0-1, commit 8fc20bf)에 기록돼 있습니다. 실험 34 파일은 수정하지 않았습니다.

### C.2 Token 수준 (오프라인, `token_level/units.jsonl`, `analysis.json` `token_level`)
- 규모: 1,600 프레임(80 에피소드의 natural rollout에서 4 step마다), **18,975 단위** (프레임 × d ∈ {0.15, 0.30} × 6 방향, 달성 거리 필터 후).
- 통계: 에피소드(task, seed) cluster bootstrap (2,000, seed 0), McNemar, Wilcoxon.
- 수치 noise: 같은 관측에서 `generate`와 step-wise 디코더의 chunk가 일치한 비율은 92.9%입니다 (불일치 7.1%).

| 지표 | 값 |
|---|---|
| step 2/3/4 translation token 변경률 (normal vs R) | **55.6 / 36.0 / 27.9%** (10,542 / 6,824 / 5,286 of 18,975; recomputed: `svla_token_step_change.py`; 원본에 CI 없음) |
| 증폭 (A ≥ 1) normal | **22.1% [19.5, 24.7]** |
| 흡수 (step 2–4 token = R) | **36.7% [33.1, 40.2]** |
| 같은 프레임에서 흡수·증폭 방향 공존 | d 0.15: **25.1% [21.8, 28.7]**<br>d 0.30: **15.3% [12.8, 17.9]** (N = 1,600 프레임) |
| log A 분산 중 프레임 간 비율 | 60% / 64% |

교정·reverse·window (전체 18,975 단위):

| 행 | 증폭 [95% CI] | 흡수 | D | step-4 편차 [95% CI] |
|---|---|---|---|---|
| normal | 22.1% [19.5, 24.7] | 36.7% | 0.191 | 0.0561 [0.0499, 0.0626] |
| recent_ref | 13.7% [11.8, 15.8] | 37.5% | 0.115 | 0.0162 [0.0138, 0.0191] |
| full_ref | 13.5% [11.7, 15.4] | 37.6% | 0.109 | 0.0098 [0.0078, 0.0122] |
| win1 | 15.4% [13.4, 17.6] | 37.4% | 0.126 | 0.0271 [0.0234, 0.0315] |
| reverse | 18.6% [16.3, 20.9] | 36.8% | 0.155 | **0.0566** [0.0504, 0.0632] |

- **교정 효과**:
  - recent_ref − normal: 증폭 **−8.3%p [−9.3, −7.4]**, p = 6.7e-261. step-4 편차 −0.040 [−0.045, −0.035]
  - full_ref − normal: 증폭 −8.5%p [−9.6, −7.5], p = 2.0e-273. step-4 편차 **−0.046 [−0.051, −0.041] (−82%)**, D −43%
  - normal에서 증폭된 4,185 단위만 보면 증폭이 100% → 52.9% (full_ref)로 줄어듭니다.
- **reverse 효과**:
  - reverse − full_ref: 증폭 **+5.1%p [+4.4, +5.8]** (1,180 vs 213, p = 1.9e-162). step-4 편차 **+0.047 [+0.042, +0.052]** (Wilcoxon p ≈ 0)
  - reverse의 step-4 편차는 0.0566으로, normal 0.0561 수준까지 완전히 복원됩니다.
  - 증폭된 단위만 보면 reverse − full_ref = +23.6%p [+21.3, +26.0]입니다.
- **window**: full_ref의 step-4 편차 감소분을 기준으로 한 회복률입니다.
  - 1 step 전(recent_ref) **86.3%**. RESULTS에는 87%로 적혀 있습니다 (conflicts m4).
  - 2 step 전만(win1) **62.6%**
  - win1 − full_ref: 증폭 +1.9%p [+1.5, +2.3], p = 6.3e-68
  - chunk가 4 step이라 3 step 이상 전의 window는 측정할 수 없습니다.
- **방향 vs 크기** (10,542 motion 단위):
  - dir_wrong_mag_ok − dir_ok_mag_wrong: 증폭 **+1.6%p [+0.4, +2.8]**, p = 7.3e-5. step-4 편차 +0.013 [+0.009, +0.017], p = 5.5e-17
  - 방향 오답 대체는 R과의 거리가 0.120으로, 크기 오답(0.175)보다 R에 더 가깝습니다.
  - near_ref − ref_trans: 증폭 +4.3%p [+3.2, +5.6], p = 4.2e-31. token identity에 민감하다는 뜻이며, Impromptu의 near_gt ≈ GT와 다릅니다.
- **token 수준 사전 등록 기준**: (1) 교정 PASS, (2) reverse PASS, (3) 자연 변동보다 큼 PASS, window PASS.

### C.3 Closed loop (SimplerEnv, `closed_loop/episodes.jsonl`, 80 에피소드 × 9 조건)
- 과제·평가: pick_coke_can과 move_near, 환경 seed 0–39, 80 step, 성공 = 마지막 step의 `done`.
- 개입: control step 8–23 동안 생성되는 모든 chunk에 적용합니다. step-1 translation token을 d = 0.3, perp_left 또는 opposite 방향으로 바꿉니다.
  - feedback: 실행 p, 문맥 p
  - corrected: 실행 p, 문맥 g
  - reverse: 실행 g, 문맥 p
- 아래 표는 모두 recomputed입니다 (`_parts/scripts/svla_closed_loop_paired.py`). analyze_svla.py의 `boot`(에피소드 cluster, 층화 없음, 2,000회, seed 0)와 `paired`(McNemar exact, Wilcoxon)를 **import해서 그대로** 썼습니다. 전체 합계 값은 `analysis.json` `closed_loop.*`와 완전히 같습니다.
- 궤적 발산 = 실행된 world_vector 누적합의 끝점 간 L2 거리입니다.

**전체 80 에피소드 (natural = 68/80 = 85.0% [77.5, 92.5])**

| 조건 | 성공 n/N (%) | Δ vs natural [95% CI] | McNemar (cond only / natural only), p | 성공 불일치 | 끝점 발산 vs natural (m) [95% CI] |
|---|---|---|---|---|---|
| natural_mix | 68/80 (85.00%) | +0.00 [0, 0] | 0/0, 1 | 0.00% | 0.003 [0.000, 0.007] |
| natural_gen (재실행) | 75/80 (93.75%) | +8.75 [+1.25, +16.25] | 9/2, 0.065 | **13.75%** | **0.140 [0.100, 0.185]** |
| feedback_perp_left | 69/80 (86.25%) | +1.25 [−8.75, +11.25] | 10/9, 1 | 23.75% | 0.422 [0.380, 0.471] |
| corrected_perp_left | 73/80 (91.25%) | +6.25 [−1.25, +13.75] | 8/3, 0.23 | 13.75% | 0.416 [0.370, 0.464] |
| reverse_perp_left | 72/80 (90.00%) | +5.00 [−3.75, +13.75] | 8/4, 0.39 | 15.00% | 0.209 [0.171, 0.252] |
| feedback_opposite | 52/80 (65.00%) | **−20.00 [−31.25, −8.75]** | 5/21, **0.0025** | 32.50% | 0.437 [0.378, 0.500] |
| corrected_opposite | 51/80 (63.75%) | **−21.25 [−33.75, −7.50]** | 9/26, **0.0060** | 43.75% | 0.423 [0.370, 0.482] |
| reverse_opposite | 67/80 (83.75%) | −1.25 [−11.25, +8.75] | 8/9, 1 | 21.25% | 0.232 [0.184, 0.285] |

**과제별** (각 40 에피소드; natural: coke 37/40 = 92.5%, move near 31/40 = 77.5%)

| 조건 | coke: 성공, Δ [CI], p | move near: 성공, Δ [CI], p |
|---|---|---|
| natural_gen | 39/40, +5.0 [0.0, +12.5], 0.50; 불일치 5.0% | 36/40, +12.5 [−2.5, +27.5], 0.18; 불일치 22.5% |
| feedback_perp_left | 38/40, +2.5 [−10.0, +12.5], 1 | 31/40, 0.0 [−17.5, +17.5], 1 |
| corrected_perp_left | 38/40, +2.5 [−5.0, +12.5], 1 | 35/40, +10.0 [−5.0, +22.5], 0.29 |
| reverse_perp_left | 39/40, +5.0 [−2.5, +15.0], 0.63 | 33/40, +5.0 [−7.5, +20.0], 0.73 |
| feedback_opposite | 22/40, **−37.5 [−52.5, −22.5], 6.1e-5** | 30/40, −2.5 [−20.0, +12.5], 1 |
| corrected_opposite | 25/40, **−30.0 [−47.5, −12.5], 0.004** | 26/40, −12.5 [−35.0, +10.0], 0.36 |
| reverse_opposite | 33/40, −10.0 [−22.5, 0.0], 0.22 | 34/40, +7.5 [−10.0, +22.5], 0.55 |

- opposite 교란으로 생긴 실패는 주로 pick_coke_can에서 나옵니다. RESULTS에는 없는 분해이고, 과제별 검정력은 낮습니다.

### C.4 효과 분리: "실행 action 효과" vs "문맥 feedback 효과" (전체 80 에피소드, recomputed)

| 효과 유형 | 대비 | 성공 a vs b | Δ [95% CI] | McNemar a/b, p | 불일치 | a–b 끝점 거리 (m) | Δ(발산 vs natural) [CI], Wilcoxon p |
|---|---|---|---|---|---|---|---|
| **문맥 feedback** | **Feedback(opp) − Corrected(opp)** | 52 vs 51 | **+1.25 [−12.50, +13.75]** | 14/13, **1** | 33.75% | 0.388 [0.305, 0.484] | +0.013 [−0.052, +0.079], 0.73 |
| **문맥 feedback** | **Reverse(opp) − Natural** | 67 vs 68 | **−1.25 [−11.25, +8.75]** | 8/9, **1** | 21.25% | 0.232 [0.184, 0.285] | – |
| **문맥 feedback** | **Feedback(left) − Corrected(left)** | 69 vs 73 | **−5.00 [−12.50, +1.25]** | 2/6, **0.29** | 10.00% | 0.227 [0.182, 0.276] | +0.006 [−0.029, +0.041], 0.55 |
| **문맥 feedback** | **Reverse(left) − Natural** | 72 vs 68 | **+5.00 [−3.75, +13.75]** | 8/4, **0.39** | 15.00% | 0.209 [0.171, 0.252] | – |
| 실행 action | Corrected(opp) − Natural | 51 vs 68 | **−21.25 [−33.75, −7.50]** | 9/26, **0.006** | 43.75% | 0.423 [0.370, 0.482] | – |
| 실행 action | Feedback(opp) − Reverse(opp) | 52 vs 67 | **−18.75 [−31.25, −7.50]** | 5/20, **0.004** | 31.25% | 0.445 [0.378, 0.521] | +0.204 [+0.140, +0.270], 2.0e-8 |
| 실행 action | Corrected(left) − Natural | 73 vs 68 | +6.25 [−1.25, +13.75] | 8/3, 0.23 | 13.75% | 0.416 [0.370, 0.464] | – |
| 실행 action | Feedback(left) − Reverse(left) | 69 vs 72 | −3.75 [−12.50, +5.00] | 5/8, 0.58 | 16.25% | 0.398 [0.356, 0.446] | +0.212 [+0.162, +0.269], 3.0e-10 |
| (사전 등록) | Corrected − Feedback (opp / left) | | −1.25 [−13.75, +12.50] p = 1 / +5.00 [−1.25, +12.50] p = 0.29 | | | | |
| (사전 등록) | Reverse − Corrected (opp / left) | | **+20.00 [+7.50, +31.25] p = 0.0037** / −1.25 [−8.75, +6.25] p = 1 | | | | |

- 이 네 대비는 analyze_svla.py에 정의된 것과 같은 방법입니다. Corrected−Feedback과 Reverse−Corrected는 `analysis.json`의 `corrected_vs_feedback_*`, `reverse_vs_corrected_*`와 같고, Reverse−Natural은 `vs_natural.reverse_*`와 같습니다.
- Feedback−Corrected는 Corrected−Feedback의 부호만 바꾼 값입니다.
- 추가한 대비 (원 분석에 없음; 같은 지표, 같은 `paired` Wilcoxon과 에피소드 bootstrap): 문맥만 교란했을 때의 궤적 발산 − 재실행 발산
  - reverse_opposite **+0.093 m [+0.047, +0.141]**, p = 1.7e-5
  - reverse_perp_left **+0.070 m [+0.017, +0.121]**, p = 5.0e-4
  - 즉 문맥 feedback은 **궤적**을 재실행 noise보다 유의하게 바꿉니다.
- 과제별 문맥 feedback 대비 (Feedback − Corrected, Reverse − Natural)는 coke와 move near 모두 p ≥ 0.22로 유의한 것이 없습니다 (`spatialvla_closed_loop_paired.csv`).

해석:
- **실행 action 효과는 큽니다.** opposite 방향에서 교란 action을 실행하면, 문맥을 교정하든 안 하든 성공률이 약 20%p 떨어집니다.
- **문맥 feedback 효과는 성공률에서는 0과 구별되지 않습니다.**
  - 네 대비가 모두 |Δ| ≤ 5%p, p ≥ 0.29입니다.
  - CI 폭은 ±10–13%p로, 약 10%p 미만의 효과는 검출할 수 없습니다.
  - 재실행 불일치는 13.75%입니다.
- **문맥 feedback은 궤적에는 영향을 줍니다.** reverse의 발산은 0.21–0.23 m로, 재실행 0.14 m보다 큽니다.
- **closed-loop 사전 등록 기준**: (1) 교정 FAIL, (2) reverse FAIL → **Partial replication**.

### C.5 Cross-model 요약 (AutoVLA / Impromptu / OpenVLA / SpatialVLA)
- **token 수준 인과 구조는 네 모델 모두에서 관찰됩니다.** 문맥 token이 다음 token을 결정하고, 교정하면 효과가 줄고, 다시 넣으면 돌아옵니다.
- **open-loop 궤적·task 효과로 이어진 것은 driving 두 모델뿐입니다.**
  - AutoVLA: −40.3%p
  - Impromptu: −62.2%p
- **closed loop에서 문맥 feedback의 task 효과는 검출되지 않았습니다.**
  - OpenVLA: feedback − corrected +3 / +5%p, p ≥ 0.36. 시간축 history가 없습니다.
  - SpatialVLA: |Δ| ≤ 5%p, p ≥ 0.29. 4-step chunk, 매 step 재계획, ensemble에서 step 2–4의 몫은 42.6%입니다.
- 두 closed-loop 모델 모두 재실행 noise가 큽니다(불일치 20%, 13.75%).
- 이 결과는 "기전은 일반적이지만, task 실패로 이어지는지는 구조에 따라 다르다"는 해석을 지지합니다. 반대로 closed-loop task 효과가 없다는 것은 검정력 한계(N = 80–100) 안에서의 결론입니다.
