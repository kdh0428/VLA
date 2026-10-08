# P0-B 군집 통계 재분석 — 분석 프로토콜

작성: 2026-10-08, 새 통계량을 하나도 계산하기 전. 근거 계획: `/root/VLA/VLA_experiment_plan_20261008.md` §2, §4(P0-B).

## 0. 성격 (사전 등록 아님)

- 이 분석은 **이미 결과를 본 자료의 재분석**이다. 원 분석의 unit-level 효과·CI·p(예: AutoVLA Recent−Normal −40.3 pp, p = 8e-42;
  SpatialVLA closed-loop Feedback−Corrected +1.25 pp [−12.5, +13.75])는 작성자가 이미 알고 있다. 따라서 사전 등록이라고 부르지 않는다.
- 이 문서가 고정하는 것은 "새 군집 통계(장면 가중, log cluster, LOO, sign-flip)를 계산하기 전에" 정한 estimand·family·방법·margin이다.
  작성 전 확인한 것은 raw 파일의 구조(필드 이름, 행 수, 조건 이름, cluster 수)와 원 분석 스크립트뿐이다.
- 새 모델 추론·GPU 사용 없음. 지표는 원 분석 스크립트의 함수를 import해서 그대로 쓰고 재정의하지 않는다
  (`analyze_action_history.unit_metrics`/`a_eval`, 증폭 = A− ∧ FDE5 > 3 m; Impromptu `analyze_temporal.metrics`;
  SpatialVLA `analyze_svla.token_level`의 A = ‖Σ_{k=2..4}(v_k−r_k)‖/‖v_1−r_1‖, 증폭 = A ≥ 1, translation table은 processor에서 재구성 — weight 미사용).

## 1. 자료와 단위

| 모델 | 실험 | raw | 반복 단위 | 1차 군집(평균 단위) | 독립 cluster |
|---|---|---|---|---|---|
| AutoVLA | 7 `action_history_causal/units.jsonl` | 1,408 단위 | (장면 token, perturbation) | 장면 | navtest log (A− 16) |
| AutoVLA | 11 `prev_action_identity_decomposition/units.jsonl` | 1,408 | 〃 | 장면 | log |
| AutoVLA | 32 `motion_semantics_ablation/records.jsonl` (지표는 `unit_metrics`로 재계산) | 1,408 | 〃 | 장면 | log |
| AutoVLA | 6 `equal_distance_perturbation/rows.jsonl` | 1,824 조건 행 | (장면, 대안) | 장면 | log |
| Impromptu | 33 `cross_vla_temporal_replication/exp_cde/records.jsonl` | 300 장면 × 8 방향 | (장면, 방향) | 장면 | log (28) |
| SpatialVLA | 34 closed loop `closed_loop/episodes.jsonl` | 720 = 9 조건 × 80 | episode (task, seed) | — | episode, task 층화 |
| SpatialVLA | 34 token `token_level/units.jsonl` | 1,600 frame, 단위 (frame, d, u) | 단위 | episode | episode (task, seed), task 층화 |

- 실험 5, 8, 9, 10: 패키지는 per-unit raw가 없다고 기록. 재분석 전에 디렉토리와 파일시스템에서 jsonl 존재 여부를 검증하고,
  없으면 "raw unavailable"과 필요한 자료(단위별 조건 결과 + log/scene ID)를 적는다. summary에서 값을 추정해 채우지 않는다.
- AutoVLA **Reverse−Full**(reverse@emb − GT-history)은 실험 8에만 있다 → 실험 8 raw가 없으면 primary family의 이 항목은 "계산 불가"로 남긴다.

## 2. Join과 제외

- AutoVLA: 실험 7/11/32/6을 공통 unit ID `(token, pert)`로 join. 각 실험 안에서 대비 두 조건이 모두 있는 단위만 paired 분석에 쓰며,
  (a) 한 조건 결측, (b) 지표 None, (c) 실험 간 unit 집합 불일치, (d) 그룹(A−/A+) 또는 log 표기 불일치를 `exclusion_reason`으로 먼저 보고한다.
  OOD 판정(첫 step entropy가 Normal보다 > 1 nat 상승; 원 스크립트 규칙)은 조건 수준 규칙이며 primary 조건(normal, recent_gt, gt_history,
  dir_*)에는 원 분석에서 적용되지 않았다. 재분석도 단위를 이 규칙으로 빼지 않는다(무제외가 주 분석).
- 실험 32에서 대체 token이 기록되지 않은 단위(`substitutions` 비어 있음)가 있으면 따로 세고, 주 분석에는 원 분석처럼 포함하며
  해당 단위를 뺀 민감도를 함께 보고한다.
- Impromptu: 원 스크립트대로 parse 실패 단위와 기준 단위(dir = None)를 제외하고, 그 수를 보고한다.
- SpatialVLA closed loop: (task, seed)로 모든 조건을 join, 9 조건이 모두 있는 episode만 쓴다. token level: 원 스크립트의 모든 단위
  (달성 거리 범위 밖 단위는 이미 실행 시 제외)를 그대로 쓰고, episode ID = (task, seed).

## 3. Primary family와 다중 비교

결과 지표는 모두 **증폭(이진)** 이다. 보조 지표는 AutoVLA FDE5(m).

**AutoVLA primary family (m = 3, Holm):** A− 그룹.
1. Recent−Normal: `recent_gt − normal`, 실험 7.
2. Reverse−Full: `reverse@emb − gt_history`, 실험 8 → raw 없으면 p = 1로 두고(검정하지 않은 것으로 취급) family 크기는 3을 유지한다(보수적).
3. 방향 대조: `dir_wrong_mag_ok − dir_ok_mag_wrong`, 실험 32(PROTOCOL.md가 방향 vs 크기를 핵심 질문으로 지정).

**SpatialVLA 실험 34 primary family (m = 6, Holm):**
1–2. closed loop Feedback−Corrected, 방향 opposite / perp_left 각각 (`feedback_<dir> − corrected_<dir>`, 성공률).
3–4. closed loop Reverse−Natural, opposite / perp_left (`reverse_<dir> − natural`, 성공률).
5. token correction: `recent_ref − normal` (증폭).
6. token reverse: `reverse − full_ref` (증폭).
(논문 Claim 10은 opposite를 대표로 쓰지만, 방향 선택이 사후적이지 않도록 두 방향을 모두 family에 넣는다.)

**보조(secondary, 다중 보정 없음, 탐색적으로 표기):** 실험 11 Recent−Normal(실험 7의 harness 재현), 실험 11 `geo_nn_gt − recent_gt`
(동등성 질문), AutoVLA 세 대비의 FDE5, A+ 그룹 같은 대비, 실험 6 대안 증폭률(A−, A+)과 A−−A+ 차, Impromptu(실험 33)
`recent_gt − normal`, `gt_history − normal`, `reverse − gt_history`, `dir_wrong_mag_ok − dir_ok_mag_wrong`, SpatialVLA token 방향 대조.
Impromptu 4개는 자체 Holm(m = 4)도 병기한다. 탐색 지표를 primary로 승격하지 않는다.

## 4. Estimand와 가중

장면(또는 episode) s의 단위 차이 d_su = Y_su(조건 a) − Y_su(조건 b).

- **주 estimand (scene-weighted):** θ_S = (1/N_S) Σ_s d̄_s, d̄_s = 장면 안 단위 평균. 즉 같은 장면의 섭동을 평균한 뒤 장면을 동등 가중.
  log는 추론에서 cluster로만 쓰인다(장면이 많은 log는 그만큼 가중).
- **민감도 1 (unit-weighted):** θ_U = 모든 단위 평균 (원 분석의 점추정과 같음).
- **민감도 2 (log-equal):** θ_L = (1/G) Σ_g mean_{s∈g} d̄_s.
- SpatialVLA closed loop: θ = task별 episode 평균 차이를 task 균등 평균(과제당 40 episode라 episode 평균과 같음). 과제별 값도 보고.
- SpatialVLA token: **주** = episode-weighted(episode 안 단위 평균 → task 안 episode 평균 → task 균등 평균). 민감도 = unit-weighted(원 분석).

## 5. 추론

- **log block bootstrap CI:** log를 복원 추출(G개), 선택된 log의 장면 전부를 넣어 같은 estimator를 다시 계산. B = 10,000, seed 20261008,
  percentile 95% CI. 모든 가중 방식에 같은 resample index를 쓴다.
- **sign-flip 검정 (주 p값):** log 요약 S_g(주 estimand: log 안 장면 효과의 합; log-equal: log 평균)에 부호 ε_g ∈ {±1}를 곱한
  T* = Σ ε_g S_g / N_S(또는 평균)의 분포에서 양측 p = P(|T*| ≥ |T_obs|). G ≤ 20이면 2^G 조합 **전수 열거**(AutoVLA A− 16 log → 65,536),
  G > 20이면 Monte Carlo 200,000회(+1 보정). 가정: H0에서 각 log 요약이 0에 대해 **대칭**이고 log끼리 **독립**(log 간 교환가능성이 아니라
  log 안 부호 대칭). 처치가 무작위 배정된 것이 아니므로(조건은 같은 입력에 대한 결정적 개입) 정확한 randomization inference라 부르지 않는다.
  최소 달성 p = 2/2^G(16 log: 3.05e-5).
- **cluster-robust t (보조):** 선형화 분산, df = G − 1.
- **원 방법 재현:** unit-level McNemar exact (원 분석의 p)와 원 log bootstrap(2,000, 원 seed)을 같은 표에 둔다.
- **leave-one-log-out:** AutoVLA A− primary 대비마다 16 log를 하나씩 빼고 θ_S, bootstrap CI(B = 2,000), 전수 sign-flip p를 다시 계산.
  Impromptu는 28 log LOO(점추정과 Monte Carlo sign-flip 20,000).
  "단일 log 의존" 판정: 어떤 log 하나를 뺐을 때 (i) 부호가 바뀌거나, (ii) CI가 0을 포함하게 되거나, (iii) sign-flip p가 0.05를 넘으면 의존으로 본다.
  또 각 log의 영향도 = θ_S − θ_S(−g)를 보고한다.
- **SpatialVLA:** episode bootstrap — task 안에서 seed(episode)를 복원 추출, episode의 모든 조건을 함께 묶음, B = 10,000, seed 20261008.
  closed loop 검정 = task 층화 episode sign-flip(Monte Carlo 200,000) + exact McNemar(episode 독립 가정; 원 방법). token level =
  episode 요약의 task 층화 sign-flip(Monte Carlo 200,000); 단위는 episode 안 반복 관측. 과제별 효과도 같은 방법으로 보고.
- 다중 비교: family 안 Holm, 주 p값(sign-flip) 기준. α = 0.05.

## 6. 효과 부재 진술과 동등성 margin (새 수치를 보기 전에 고정)

"효과 없음/동등"을 말하는 경우에만 쓴다. 판정 = 90% CI(log/episode bootstrap)가 ±margin 안에 있으면 "실용적 동등", 아니면
"비유의하지만 동등성 미확인"으로 쓴다. 비유의성을 0으로 쓰지 않는다.

- **SpatialVLA closed-loop 성공률: ±10 pp.** 근거: "문맥 feedback이 task 실패를 일으킨다"는 주장에 실질적 의미가 있으려면 최소 10 episode 중
  1개 정도의 성공 변화가 필요하다고 본다(로봇 조작 벤치마크에서 방법 간 비교가 보통 10 pp 단위로 해석됨). 이 margin은 원 CI 폭과 무관하게 정했으며,
  원 CI(±13 pp)를 이미 알고 있다는 사실은 여기 공개한다.
- **AutoVLA/Impromptu 증폭률 차이(동등성 질문, 예: 실험 11 geo_nn_gt ≈ recent_gt): ±5 pp 절대.** 근거: 장면 20개 중 1개 수준의 증폭 변화보다
  작은 차이는 "기전상 같은 효과"라는 진술을 뒤집지 않는 수준으로 본다.
- **SpatialVLA token 증폭률: ±2 pp 절대.** 근거: token 단위 증폭 기준율이 낮은(수십 % 미만) 지표에서 2 pp는 단위 50개 중 1개 차이.

## 7. 산출물

`manifest.json`(이 파일의 작성 시각·sha256 포함), `units.parquet`, `condition_summary.csv`, `paired_effects.csv`, `loo_logs.csv`,
`comparison_with_original.csv`, `analysis.md`(한국어; 확인한 사실/계산한 결과/미확인/가설 구분). 기존 파일은 수정하지 않는다.
