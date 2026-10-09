# P1-B 결과: 같은 교정 예산에서 시작 위치 이동 (AutoVLA)

작성: 2026-10-09.
- 사전 등록: `protocol.md`(commit b212637, 실행 전).
- 실행: 2026-10-08 RTX 5090, `run_all_main.sh` stage 4, 38.1분.
- 분석: CPU만.
- 이 파일은 상위 세션이 agent 출력을 그대로 저장한 것이다.

표기:
- [확인한 사실] / [계산한 결과] / [미확인] / [가설].
- 기본은 장면 가중이다.
- [ ] = log block bootstrap 95% CI.
- p = log sign-flip. primary는 15 log, 2^15 전수 열거다.
- "early − late"의 양수 = early 교정 후 증폭이 더 많이 남음.

## 0. 요약

- **H_early(같은 예산에서 early가 증폭을 더 줄인다)는 어느 w에서도 지지되지 않는다.** [계산한 결과]
  - primary 세 대비: P_gt s = 1 − s = 3, A−, t* ≤ 2, 46 장면 / 326 단위 / 15 log.
  - 점추정은 모두 0 이상이다. 즉 early가 같거나 더 나쁘다.
  - Holm p는 모두 > 0.3이다.
- 사전 등록 동등성(±10 pp, 90% CI)
  - **w = 2와 w = 3은 동등성이 성립한다.**
  - **w = 1은 동등성이 성립하지 않는다**(+10.7 pp, 90% CI [+1.4, +18.6]). 점추정은 late가 더 낫다는 쪽이지만, 결론은 불확정이다.
- **길이 효과는 크다.** s = 1 persistent에서 Normal→Full 효과의 회복률은 다음과 같다(집계 수준). [계산한 결과]
  - w = 1: 48%.
  - w = 2: 77%.
  - w = 3: 89%.
- **지표 의존성(보조)** [계산한 결과]
  - 연속 FDE5에서는 w = 3에서 early가 유리하다: −0.81 m [−1.12, −0.43], p < 1e-3.
  - 이진 증폭과 연속 오차가 다른 답을 준다. 위치 효과가 "없다"고 쓰지 말고, "이진 증폭에서는 ±10 pp 안(w = 2, 3)"으로 제한해 써야 한다.
- 노출이 같은 transient window(주 보조)에서도 early 우위는 없다. [계산한 결과]
- self-reference는 no-op이 아니다. E:original 단위의 9.1%에서 normal ≠ reference다. [계산한 결과]
- **plan §11 대응:** "길이는 중요하지만 위치 차이는 작음 → 짧은 교정의 효율"을 지지한다(이진 증폭 기준). "early 특수성"은 지지되지 않는다. [가설: 주장 대응]

## 1. 실행·분모·검증

| 항목 | 값 |
|---|---|
| 단위 | 1,408 단위 / 208 장면(A− 365 / 52 / 16, A+ 1,043 / 156 / 26). runtime 실패 0(`raw/errors.jsonl` 0행) [확인한 사실] |
| primary 공통 부분집합 | A−, t* ≤ 2: **326 단위 / 46 장면 / 15 log**. 사전 등록 값과 같다. 모든 조건이 같은 단위다. missing pairs 0, NaN drop 0 [확인한 사실] |
| A+ 공통 부분집합 | 920 / 138 / 25 [확인한 사실] |
| truncated window | t* ≤ 2 부분집합에서 0. t* > 2 단위에서 3,020행. total-horizon 대비에서는 쓰지 않음 [확인한 사실] |
| 제외 | 결과 기반 제외 없음 [확인한 사실] |
| output_prefix_locked (전 53 행) | 1.000 (1,408 단위) [확인한 사실] |
| abort 조건 | reference sha256 통과. 잠금 100%. cache assert 없음. 장면 실패 0 → 해당 없음 [확인한 사실] |
| 재시도·병합 | 없음 [확인한 사실] |
| 교정 예산(A− 공통, 단위 평균) | persistent 교정 forward: s1 7.21, s2 6.21, s3 5.21, s4 4.21. 즉 late window는 더 적은 step에서만 보인다. transient: 모든 s에서 w forward. 실효 교정 쌍: P_s1_w1 5.76 vs P_s3_w1 4.43 [계산한 결과] |

**cluster 주의:** primary의 15 log는 12 drive다(전체 A− 16 log = 13 drive). drive 수준 군집은 쓰지 않았다. [확인한 사실]

**trace 검증 범위** [확인한 사실 / 미확인]
- main run의 `trace_verification.json`에는 n_units와 prefix 잠금만 있다.
- window가 정의대로 적용됐는지(persistent는 이후 모든 step, transient는 k = j+1에서만)는 smoke 4단위에서만 점검됐다.
- main run에서는 `raw/intervention_trace.jsonl`과 `intervention_trace.jsonl`(첫 24 단위)에 (step, position) 단위로 기록돼 있다. 이번 분석에서 전수 대조는 하지 않았다.
- 예산 집계(위 표)는 정의와 일치한다.

### 1.1 self-reference 비-no-op 점검 [계산한 결과]
- E:original 단위(f = r[t*])는 208개다.
- 그중 19개(9.1%; A− 9/52 = 17.3%, A+ 6.4%)에서 이번 run의 `normal`이 B = 1 reference와 다르다.
  - dev9_ref 평균 0.23 m, 최대 16.3 m.
- original 단위의 self 행 중 실효 교정 쌍 > 0인 행은 4.3%다.
- 따라서 f = r일 때도 self 교정은 완전한 no-op가 아니다.
- protocol §2.4가 예고한 대로, 모든 self 효과는 이 run의 `normal` 대비로만 계산했다.
- f ≠ r 단위에서는 normal 자체가 reference에서 크게 벗어나 있다. A− 공통에서 normal dev9_ref는 6.14 m다. 이는 섭동 f를 넣었으므로 당연하다.

## 2. Primary family (A−, t* ≤ 2, persistent GT, amplification, Holm m = 3)

| w | early − late 장면 가중 [95% CI] | 90% CI | sign-flip p | Holm p | cluster-t p | unit 가중 | log 균등 | 정보 log | LOO (15회) | ±10 pp TOST | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **+10.7 pp [−0.3, +20.1]** | [+1.4, +18.6] | 0.116 | 0.349 | 0.057 | +9.8 [−0.6, +19.4] | +11.1 [−0.1, +24.3] | 11 | +6.9 ~ +13.4; p 0.041–0.23 | 불성립 | H_early 기각 방향; 검출 안 됨; 동등성 미확인 |
| 2 | **+4.3 pp [−2.1, +9.8]** | [−1.1, +8.9] | 0.285 | 0.570 | 0.163 | +4.0 [−2.4, +9.8] | +5.1 [−2.7, +15.6] | 10 | +1.9 ~ +5.4; p 0.15–0.57 | **성립** | early ≈ late (±10 pp 안) |
| 3 | **+0.5 pp [−4.5, +3.1]** | [−3.7, +2.8] | 0.836 | 0.836 | 0.796 | +0.3 [−4.9, +3.0] | −2.2 [−5.1, +0.4] | 8 | −1.6 ~ +1.2; p 0.42–1.0 | **성립** | early ≈ late (±10 pp 안) |

[계산한 결과]
- 수준(장면 가중 GT 증폭, A− 공통): normal 48.3%, full_gt 3.6%, recent_gt 7.6%.
  - P_gt s1: w1 26.7 / w2 13.9 / w3 8.5%.
  - P_gt s3: w1 16.1 / w2 9.6 / w3 8.0%.
- w = 1:
  - LOO 한 경우(`2021.08.30.14.54.34_veh-40` 제거)에 비보정 p = 0.041이다. 방향은 "late가 더 낫다"이고, 다른 14회는 p > 0.08이다.
  - 사전 등록 판정은 Holm 기준이다. "검출 안 됨, 동등성 미확인"으로 둔다.
  - "late가 더 낫다"고 주장할 근거도 아니다.
- 사전 결정 규칙: 음수이고 Holm p < 0.05이면 H_early 지지다. 어느 w도 충족하지 않는다.
- 동등성 margin ±10 pp는 남은 증폭 수준(8–27%)에 비해 넓다. "±10 pp 안"이라는 것은 "차이가 없다"보다 훨씬 약한 진술이다. [가설: 해석 주의]

## 3. 사전 등록 보조 분석 (비보정, 표시: 보조)

### 3.1 total-horizon 고정, persistent GT, A− (`P1B_secondary_totalH_P_gt_A-`)

| 대비 | amplification | FDE5 (m) |
|---|---|---|
| s1 − s2, w1 | +6.5 pp [+0.7, +12.6], p = 0.080 | +0.14 [−0.21, +0.59], p = 0.43 |
| s1 − s3, w1 | (primary) | +0.47 [−0.00, +0.99], p = 0.11 |
| s1 − s4, w1 | +11.9 pp [+0.8, +20.6], p = 0.085 | −0.08 [−0.70, +0.51], p = 0.77 |
| s1 − s2, w2 | +1.4 pp, p = 0.42 | −0.06, p = 0.63 |
| s1 − s3, w2 | (primary) | −0.34 [−0.65, +0.12], p = 0.18 |
| s1 − s4, w2 | +4.9 pp, p = 0.26 | **−0.91 [−1.33, −0.34], p = 0.009** |
| s1 − s2, w3 | +1.5 pp, p = 0.19 | −0.32 [−0.50, −0.09], p = 0.019 |
| s1 − s3, w3 | (primary) | **−0.81 [−1.12, −0.43], p = 3e-4** |
| s1 − s4, w3 | +1.4 pp, p = 0.61 | **−1.45 [−1.92, −0.86], p = 3e-4** |

- amp2 / amp4도 증폭과 같은 양상이다. w = 1에서는 early가 같거나 나쁘고, w ≥ 2에서는 ±10 pp 안이다. [계산한 결과]

### 3.2 transient GT (노출 동일, 주 보조), A−

| w | s1 − s3 amplification | s1 − s3 FDE5 |
|---|---|---|
| 1 | +8.0 pp [−0.4, +14.8], p = 0.12 (TOST 불성립) | +0.36 [−0.06, +0.75], p = 0.14 |
| 2 | +1.9 pp [−4.7, +6.8], p = 0.50 (TOST 성립) | −0.38 [−0.76, +0.08], p = 0.16 |
| 3 | −2.1 pp [−5.8, +2.4], p = 0.49 (TOST 성립) | −0.63 [−1.05, −0.09], p = 0.042 |

[계산한 결과] 노출을 맞춰도 이진 증폭에서 early 우위는 없다. persistent에서 late window가 더 적게 노출되는 교란은 결론을 바꾸지 않는다.

### 3.3 start × length 지도 (A−, 각 window − normal, GT amplification)
- 이 표는 Normal→Full 효과 대비 집계 회복률이다: (normal − window)/(normal − full). `condition_summary.csv` 수준에서 계산했고, 사전 등록 §5의 집계 지표다. 그림은 `figures/start_length_map_gt_Aminus.png`다.

| | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| persistent w1 | 48% | 63% | 72% | 75% |
| persistent w2 | 77% | 80% | 87% | 88% |
| persistent w3 | 89% | 92% | 90% | 92% |
| transient w1 | 38% | 53% | 56% | 54% |
| transient w2 | 55% | 70% | 59% | 61% |
| transient w3 | 78% | 76% | 74% | 72% |

- 모든 window − normal 대비는 유의하다(persistent −21.6 ~ −41.3 pp, transient −17.0 ~ −35.0 pp; p ≤ 0.007). [계산한 결과]
- 같은 지도를 FDE5 기준으로 보면 순서가 반대다(early 우위). [계산한 결과]
  - persistent w3: s1 92%, s2 86%, s3 75%, s4 62%.
  - persistent w2: s1 81%, s4 62%.
- 즉 위치가 늦을수록 이진 증폭 회복률은 같거나 높다. 반면 연속 endpoint 오차 회복률은 낮다. [계산한 결과]
- 해석: late 교정은 FDE를 3 m 아래로 끌어내리는 데는 충분하다. 그러나 그 전에 쌓인 편차를 되돌리지 못해 평균 오차는 더 남는 것으로 보인다. 확인하지 않았다. [가설]

### 3.4 self-reference family (A−, P_self / T_self; 지표 dev9_ref, severe_ref, GT 증폭)
- GT 증폭: P_self와 T_self 모두 모든 s1 − s_late에서 ±10 pp 안(TOST 성립)이다. self 교정은 증폭을 줄이지 않는다. [계산한 결과]
  - 수준: normal 48.3%, full_self 65.8%, recent_self 64.6%. P_self 창은 54–64%다.
  - 즉 reference로 되돌리면 A−에서 GT 증폭이 오히려 오른다. P1-A와 같은 현상이다.
- dev9_ref: persistent에서 early가 reference에 더 가깝다. [계산한 결과]
  - w3 s1 − s3: −0.53 m [−1.12, −0.14], p = 0.014.
  - s1 − s4: −1.05 m, p = 0.002.
  - w = 1: −0.05 m, p = 0.88.
- GT reference 결과와 self reference 결과는 섞지 않았다.

### 3.5 A+와 pooled (P_gt / T_gt, s1 − s3)
- A+ (920 / 138 / 25), 증폭: 모든 w에서 |차이| ≤ 0.7 pp, TOST 성립. [계산한 결과]
- A+ FDE5: P w2 −0.17 m (p = 0.002), w3 −0.23 m (p < 1e-4). early 우위다.
- pooled (1,246 / 184 / 25), 증폭: P w1 +2.4 pp (p = 0.10), w2 +0.7, w3 +0.1. 모두 TOST 성립.
- pooled FDE5: w3 −0.38 m (p < 1e-4).

### 3.6 해제 후 free horizon 고정 (실험 10 확장, 별도 보고, total-horizon 결과와 섞지 않음)
- 정의: rel = t* + s + w, 평가 pose = rel + N. window 효과 = (window − normal), 같은 pose에서 계산한다. 대비 = s1 − s3.
- N = 2 (A−, t* ≤ 1: 277 단위 / 39 장면 / 14 log), GT target [계산한 결과]
  - 연속 오차 효과의 early − late: P_gt +0.62 / +0.74 / +0.85 m (w1/2/3, p ≤ 3e-4). T_gt +0.52 / +0.51 / +0.65 m (p ≤ 0.009).
  - 즉 같은 해제 후 거리에서 late window가 normal 대비 오차를 더 크게 줄인다.
  - exceed3 비율 차이는 모두 비유의다(|차이| ≤ 11 pp).
- **해석 주의** [가설]
  - late window의 평가 pose는 2 step 늦다. 그 pose에서는 normal 오차 자체가 더 크다.
  - 따라서 m 단위 차이에는 pose 시점 교란이 섞인다.
  - 이 대비는 사전 등록 그대로 보고하지만, early/late 우열의 증거로 쓰지 않는다.
- N = 3 민감도 (t* = 0: 116 단위 / 17 장면 / 9 log): P_gt +0.98 / +0.89 / +0.94 m (p ≤ 0.012). 방향이 같다. [계산한 결과]
- 재발산 (`free_horizon_rediverge.csv`, 해제 시 GT 오차 ≤ 1 m인 stab 단위 중 rel + N에서 > τ) [계산한 결과]
  - P_gt, N = 2:
    - rediverge3: 0–1.7%.
    - rediverge2: 1.0–11.1%.
    - stab 분모: s1w1 230, s1w2 220, s1w3 201, s2 176–137, s3 117–101 / 277.
  - T_gt, N = 2: rediverge3 0–4.3%.
  - self target: rediverge3 0–1.8%.
  - 해제 후 재발산은 드물다. 다만 late window일수록 stab 분모가 작다.
- 20-token 연장은 하지 않았다(lead decision 3). 해제 후 더 긴 horizon의 재발산은 검증하지 않았다. [확인한 사실]

## 4. 가설별 판정

| 가설 | 사전 결정 규칙 | 결과 | 판정 |
|---|---|---|---|
| H_early | s1 − s3 < 0, Holm < 0.05 (w별) | w1 +10.7 pp (Holm 0.35), w2 +4.3 (0.57), w3 +0.5 (0.84) | **지지 안 됨(모든 w)** [계산한 결과] |
| H_null (위치 차이 작음) | 90% CI ⊂ ±10 pp | w2 [−1.1, +8.9], w3 [−3.7, +2.8] 성립; w1 [+1.4, +18.6] 불성립 | **w = 2, 3에서 성립. w = 1은 미확인** [계산한 결과] |
| 길이 효과 | (지도, 보조) | s1에서 48 → 77 → 89%, 모든 window − normal 유의 | 길이가 중요함 [계산한 결과] |

## 5. plan §11 주장 대응

| §11 행 | 이번 결과 | 대응 |
|---|---|---|
| 같은 예산에서 early가 일관되게 강함 → 초기 위치 특수성 | 이진 증폭에서 early 우위 없음. w1은 반대 방향 점추정 | **지지 안 됨.** "첫 step이 특별하다"는 주장은 쓰지 않는다 |
| 길이는 중요하지만 위치 차이는 작음 → 짧은 교정의 효율 | 길이 효과 큼. w = 2, 3에서 ±10 pp 동등성 | **지지(이진 증폭 기준).** 단 연속 FDE5에서는 w ≥ 2에서 early가 0.3–1.4 m 유리(보조). "위치 무관"이 아니라 "임계 증폭률 기준 위치 차이 작음"으로 쓴다 |

[가설: 주장 문장 선택.] 원래 실험 9의 해석과 일치한다: "가장 이른 step이 특별하지 않고 첫 몇 step의 누적량이 결정"(`outputs/temporal_feedback_window`).

## 6. 한계
1. t*와 A−/A+는 GT 정의를 상속했다. 공통 부분집합(t* ≤ 2)은 t*가 이른 장면만 포함한다. [확인한 사실]
2. primary의 cluster는 15 log = 12 drive다. sign-flip p 하한은 2/2^15다. w1의 정보 있는 log는 11개, w3은 8개뿐이라 검정력이 낮다. [확인한 사실]
3. persistent window는 시작 위치와 노출 step 수가 교란된다. transient가 이를 보완하지만 결론은 같았다. [확인한 사실 + 계산한 결과]
4. 이진 증폭(FDE5 > 3 m ∧ A−)과 연속 FDE5가 다른 방향을 보인다. primary는 이진 지표로 사전 등록됐다. [계산한 결과]
5. free horizon 대비에는 평가 pose 시점 교란이 있다. 최대 N = 3이다. [가설 + 확인한 사실]
6. bf16 B = 53 수치 경로라 실험 9(B = 17)와는 질적으로만 비교한다. self 교정은 f = r에서도 9%의 단위에서 no-op이 아니다. [확인한 사실]
7. 같은 16 log 안의 결과다. 새 log 재현은 P1-C의 일이다. [미확인]
8. 이 문서는 결과를 본 뒤 작성했다. §3.3의 집계 회복률은 사전 등록 §5의 항목인데 스크립트가 출력하지 않았다. 그래서 `condition_summary.csv` 수준값으로 산술 계산했다. §1.1 점검은 `units.parquet`에서 계산한 기술 통계다. [확인한 사실]

## 7. 파일
- `protocol.md`, `manifest_prerun.json`, `manifest.json`(HEAD b212637).
- `units.parquet`, `free_horizon_units.parquet`, `condition_summary.csv`, `paired_effects.csv`, `loo_logs.csv`, `free_horizon_rediverge.csv`, `deviation_curves.csv`, `trace_verification.json`, `intervention_trace.jsonl`.
- raw는 `raw/`.
- 그림: `figures/start_length_map_gt_Aminus.png`(A− 공통 부분집합, P_gt / T_gt의 s × w 증폭 수준 + normal / full_gt 기준선).
