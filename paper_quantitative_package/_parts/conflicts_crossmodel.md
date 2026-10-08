# Cross-model 충돌·불일치 기록 (실험 28 / 33 / 34 + AutoVLA 참조값)

경로 기준: `/root/VLA/autovla_misalignment_poc/` (별도 표기 없으면). 원본 파일은 하나도 수정하지 않았습니다.
등급: **CRITICAL** = 논문 수치가 틀어짐 / **MAJOR** = 해석·비교 가능성에 영향 / **MINOR** = 반올림·문서 표현·subset 차이.

CRITICAL 등급 항목은 없습니다. raw 데이터에서 다시 계산한 값(SpatialVLA closed loop 전체, token step 변경률, ensemble 가중치)은 모두 `analysis.json`·RESULTS.md와 소수점 반올림 수준까지 일치했습니다.

---

## MAJOR

### M1. SpatialVLA ensemble 가중치 설명 오류 (실험 34 PROTOCOL.md §0)
- **실험 34 문서 기술**: `outputs/cross_domain_temporal_replication/PROTOCOL.md` §0 "ActionEnsembler(temp −0.8; 오래된 예측일수록 가중치 큼) … 실행 action의 대부분은 앞 step에 조건화된 chunk의 뒤쪽 step(2–4)에서 옵니다."
- **코드 (진실)**: `scripts/cross_domain_temporal/action_ensemble.py`에서 weights = exp(−temp·i), temp = −0.8입니다. 쌓는 순서는 `zip(range(n−1,−1,−1), history)`이고 index 0이 가장 오래된 chunk입니다. 따라서 **가장 최근 예측의 가중치가 가장 큽니다**.
  - 다시 계산한 값 (`_parts/scripts/svla_token_step_change.py`): 현재 chunk step 1 **0.5741**, 1 step 전 chunk step 2 **0.2579**, 2 step 전 chunk step 3 **0.1159**, 3 step 전 chunk step 4 **0.0521**.
  - step 2–4의 합은 **42.6%**로, "대부분"이 아니라 절반 미만입니다.
- **영향**: 실험 34는 공식 adapter 코드(`spatialvla_policy.PolicyState`, `ActionEnsembler(4, −0.8)`)를 그대로 썼으므로 결과에는 영향이 없습니다. 틀린 것은 설명 문장뿐입니다. 다만 논문에서 "ensemble이 feedback을 흡수한다"고 해석할 때는 42.6%라는 정확한 값을 써야 합니다.
- **이미 기록된 곳**: `outputs/spatialvla_feedback_protection_ablation/PROTOCOL.md` §0-1 (실험 35 사전 등록, commit 8fc20bf)에 같은 정정과 가중치 0.57/0.26/0.12/0.05, 43%가 적혀 있습니다. 실험 34 파일은 수정하지 않았습니다.
- **추가 사항 (MINOR)**: `action_ensemble.py`의 주석 "if temp > 0, more recent predictions get exponentially *less* weight"는 temp > 0일 때만 맞는 설명입니다. temp < 0을 쓰는 이 설정에서 오해를 부른 원인으로 보입니다.

### M2. Impromptu "63.4 → 1.3%"는 *증폭률*이지 *실패율(A−)*이 아님
- `outputs/cross_vla_temporal_replication/analysis.json`
  - `CDE.all_units.means.normal.amp` = 0.6343, `…recent_gt.amp` = 0.0125 (N = 2,398 단위)
  - 차이 `CDE.all_units.vs_normal.recent_gt.amp` = −0.6218 [−0.6557, −0.5913]. McNemar p는 0.0으로 기록돼 있는데, 이는 underflow입니다.
- 같은 행의 **A−는 64.5% → 16.3%** (`…recent_gt.a_minus` = 0.1635, 차이 −48.1%p [−52.3, −44.4], p = 8e-244)로, 1.3%가 아닙니다.
- RESULTS.md §5D에는 "remove cause → failure 63 → 0–1%"라고 적혀 있습니다. `EXPERIMENT_SUMMARY.md` §1-5와 §6b는 "증폭 63.4 → 1.3%"로 정확하게 썼습니다. "failure"라는 표현은 **증폭 정의(A− ∧ FDE5 > 3 m)에 한정**해야 합니다.
- 해석상 제약 두 가지:
  - gt_history(0.0%)와 recent_gt에는 절대 위치 표현 때문에 **GT 정보 누출**이 섞입니다 (RESULTS §8).
  - 그래서 GT 정보 없이 원인만 확인하는 근거는 reverse(+81.9%p)와 near_gt ≈ recent_gt(p = 0.87)입니다.
- 절대 효과: −62.2%p [−65.6, −59.1]. log-cluster bootstrap(28 log, 2,000회, seed 0)으로 구한 값입니다.

### M3. AutoVLA "reverse"는 다른 개입 방식
- AutoVLA 4.4 → 33.4%는 실험 8 (`outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` l.10, l.333)의 **reverse@emb**입니다. GT-history 문맥에 직전 위치의 Normal *embedding*을 삽입하는 residual patch입니다.
- Impromptu와 SpatialVLA의 reverse는 **token/텍스트 문맥 자체**를 바꿉니다. Impromptu는 gt_history 문맥에 자기 waypoint를 넣고, SpatialVLA는 step 2 = R, step 3 = normal로 둡니다.
- 비교표의 "Reverse" 행은 정성적으로만 비교할 수 있습니다. 기준선도 다릅니다: AutoVLA는 gt_history 4.4% (실험 8 harness), Impromptu는 0.0%, SpatialVLA는 full_ref 13.5%입니다.

### M4. "Amplification" 정의가 모델마다 다름 (직접 비교 불가)
- AutoVLA·Impromptu: A− ∧ FDE5 > 3 m. 궤적 수준이고 open-loop입니다.
- OpenVLA: 하위 차원 중 하나가 |δ| bin 이상 움직이면 증폭입니다. step 안에서 정의됩니다.
- SpatialVLA: ‖Σ_{k=2..4}(v_k − r_k)‖ / ‖v_1 − r_1‖ ≥ 1. 정규화 translation 공간 기준입니다.
- 비교표에는 수치를 함께 적되, 열 간 크기 비교는 금지라고 caption에 명시했습니다.

---

## MINOR

### m1. SpatialVLA natural 85.0% vs task별 92.5% / 77.5% — population/subset 차이 (충돌 아님)
- 다시 계산한 값 (`_parts/scripts/svla_closed_loop_paired.py`):
  - 합계 68/80 = 85.0%
  - pick_coke_can 37/40 = 92.5%
  - move_near 31/40 = 77.5%
- 원본 값:
  - `analysis.json` `closed_loop.success.natural` = 0.85 [0.775, 0.925]
  - RESULTS §1은 "85.0% [77.5, 92.5] (coke 92.5%, move near 77.5%)"로 세 값을 모두 적었습니다.
  - `EXPERIMENT_SUMMARY.md` §6c는 task별 값(92.5%, 77.5%)만 적었습니다.
- **주의**: 85.0%의 CI 하한·상한(77.5, 92.5)이 우연히 task별 성공률과 같은 숫자입니다. 둘을 혼동하지 말아야 합니다.

### m2. OpenVLA 공식 성공률 84.7% vs 재현 87%
- 공식 84.7%는 **`EXPERIMENT_SUMMARY.md` §6에만** 있습니다. 실험 28의 RESULTS.md, PROTOCOL.md, analysis.json에는 없습니다. 외부 논문 값이며 repo 안에서 검증할 수 없습니다.
- 재현값:
  - `outputs/cross_vla_replication/analysis.json` `natural_success.rate` = 0.87 (87/100)
  - 같은 정책을 batch 구성만 바꿔 다시 돌린 natural_rep = 81/100 (`natural_rep_vs_natural.rate_a` = 0.81)
- 즉 재현값 자체가 81–87% 범위로 흔들립니다(결과 불일치 20%). 87%와 84.7%의 차이(+2.3%p)는 이 재실행 변동 안에 있습니다. 진짜 충돌이라기보다 **noise 범위 안의 차이**입니다.

### m3. SpatialVLA natural_gen: bootstrap CI와 McNemar의 판단이 다름
- `closed_loop.vs_natural.natural_gen`: +8.75%p, CI [+1.25, +16.25]로 0을 제외합니다. 반면 McNemar 9 vs 2, p = 0.0654입니다.
- RESULTS §1은 McNemar 기준으로 "유의하지 않습니다"라고 썼습니다. 논문에는 두 값을 모두 적어야 합니다.

### m4. SpatialVLA 1-step window 87% vs 다시 계산한 86.3%
- `token_level.all.means.*.dev4`로 계산: (0.05613 − 0.01624) / (0.05613 − 0.00984) = **86.3%** (recomputed: `build_figure_cross_model.py`)
- RESULTS §3E와 `EXPERIMENT_SUMMARY.md`는 "87%"라고 적었습니다. 반올림·전사 차이로 보입니다. 2-step-only 62.6%는 RESULTS의 "63%"와 맞습니다.

### m5. analyze_svla.py docstring과 코드의 bootstrap 방식이 다름
- docstring에는 "episode bootstrap CI (2,000, seed 0, **stratified by task**)"라고 적혀 있습니다.
- 실제 `boot()`는 (task, seed) cluster를 층화 없이 복원추출합니다(`rng.choice(cl)`, `random.Random(0)`을 호출마다 새로 만듦).
- PROTOCOL §5에는 "에피소드 cluster bootstrap"이라고만 적혀 있습니다. 보고된 CI는 모두 **층화하지 않은** 방식으로 계산됐습니다. 이 패키지의 재계산도 같은 함수를 import해서 썼습니다.

### m6. OpenVLA 통계 구현과 PROTOCOL의 차이
- **permutation**: PROTOCOL은 "과제 단위 sign-flip, 20,000회"입니다. 코드(`analyze_cross_vla.perm_tasks`)는 2^10 = 1,024개 부호 조합을 **모두 열거하는 exact 검정**입니다. 결과적으로 더 엄밀하지만 문서와 다릅니다.
- **Phase C CI**: `boot_tasks(c_, 1000)`로 1,000회 resample입니다. PROTOCOL의 2,000회는 Phase B 통계에 대한 명시입니다.
- **p 하한 표현**: RESULTS는 "p ≥ 0.12"라고 썼지만, 실제 최소 McNemar p는 corrected_d8의 0.1153입니다 (`vs_natural.corrected_d8.p_mcnemar`). "p ≥ 0.115"가 정확합니다.
- **amendment 시각**: PROTOCOL 본문에는 "2026-10-03 19:40"이라고 적혀 있지만, commit 6d0375a의 시각은 19:32 +0000입니다(시간대 또는 표기 차이로 추정). 이 amendment는 Phase B 일부를 본 뒤에 추가됐고, 문서에도 그렇게 공개돼 있습니다.

### m7. AutoVLA Normal 증폭률이 실험마다 다름 (harness 실행 차이)
- 실험 7 (`action_history_causal`): 47.4% [38, 57]
- 실험 8 (`prev_action_state_patching`): 46.8%
- 실험 9 (`temporal_feedback_window`): 46.6%
- 세 실험 모두 같은 365 A− 단위를 썼습니다. window 비율 47/76/90/96%는 실험 9의 46.6%와 4.1% 기준입니다. 비교표의 correction 행은 실험 7을 썼습니다.

### m8. AutoVLA 자연 성능의 모집단 차이
- **A− 1.9%**: 52/2,747 (`outputs/natural_reference_stabilization/NATURAL_REFERENCE.md` l.14, "자연 실패 장면 52 + 나머지 2,695"). 장면 수 2,747과 Impromptu PROTOCOL의 2,748은 1개 차이가 납니다.
- 같은 문서의 모집단 가중 normal 행은 1.24%이고, `fast_vs_cot_unbiased/FAST_VS_COT.md`의 500 장면 subset에서는 A+ 98.8%, 즉 A− 1.2%입니다.
- **FDE5 0.66 m**는 500 장면 subset 값입니다.
- 실험 33 RESULTS §2의 "A− 1.9%, FDE5 0.66 m"는 서로 다른 모집단의 값을 한 칸에 섞어 놓은 것입니다. subset 차이이며 충돌은 아닙니다.

### m9. Impromptu task-level 비교에서 기준선 정의가 다름
- **교란 후 A+ → A− 49.5%**: 분모는 **pert_m0가 A+**인 단위입니다 (`AB.A.m0.2.flip_to_a_minus_given_m0_ok`, N = 14,424 = 1,803 장면 × 8).
- **재실행 기저 4.6%**: 분모는 **natural이 A+**인 장면입니다 (`AB.A.baseline_flip_natural_ok_to_rep_a_minus`, N = 1,818).
- 두 기준선이 다르지만, 둘 다 RESULTS §4–5에 기준선으로 공개돼 있습니다. paired 기준 효과(+24.8%p [+19.4, +32.2])는 pert_m0 대비 값입니다.

### m10. AutoVLA 장면 수준 matched-magnitude 수치의 의미
- `EXPERIMENT_SUMMARY.md` §3.2의 "같은 거리(8.8 mm) 대안 token"에서 8.8 mm는 교란의 크기가 아닙니다. 대안 token과 원래 token의 GT 오차 거리 차이의 **중앙값**입니다 (`equal_distance_perturbation/EQUAL_DISTANCE.md` l.3).
- 비교표의 "Perturbation size" 행에는 그 의미대로 적었습니다.

### m11. SpatialVLA closed-loop task별 이질성 (RESULTS에 없던 값, 충돌 아님)
- 다시 계산한 값 (`spatialvla_closed_loop_paired.csv`): opposite 교란의 실패는 **pick_coke_can에 집중**됩니다.
  - pick_coke_can: feedback −37.5%p [−52.5, −22.5], p = 6.1e-5 / corrected −30.0%p, p = 0.004
  - move_near: feedback −2.5%p, p = 1 / corrected −12.5%p, p = 0.36
- RESULTS에는 합계 값만 있습니다. task별 검정력은 40 에피소드로 낮습니다.

### m12. 실험 33 RESULTS의 SpatialVLA 제외 사유는 이후 실험으로 무효화됨
- 실험 33 RESULTS §1과 PROTOCOL §0은 SpatialVLA를 "LIBERO checkpoint 비공개, Vulkan 없음"을 이유로 제외했습니다.
- 실험 34에서 fractal checkpoint와 Mesa lavapipe로 SpatialVLA를 실행했습니다. 시점 차이일 뿐 충돌은 아닙니다.

### m13. 반올림
- 실험 34 RESULTS의 corrected_opposite "63.7%"는 51/80 = 63.75%입니다. 63.8로 쓰는 것이 일반적입니다.
- 이 패키지의 CSV에는 63.75로 적었습니다.
