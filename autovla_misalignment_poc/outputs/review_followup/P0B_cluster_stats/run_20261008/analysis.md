# P0-B 군집 통계 재분석 — 결과

작성: 2026-10-08. 프로토콜: `protocol.md`(14:01:22 UTC 작성, 첫 결과 파일 14:06:57 UTC; `manifest.json`). 스크립트: `p0b_analysis.py`, 로그 `run.log`.
**이 분석은 사전 등록이 아니다.** 원 분석의 unit-level 결과를 이미 본 상태에서 한 재분석이다. 새 모델 추론과 GPU는 쓰지 않았다.
(이 파일은 분석 agent가 작성한 내용을 상위 세션이 그대로 저장한 것이다. agent 환경에서 보고서 파일 쓰기가 막혔다.)

## 0. 요약

- **결론이 바뀌는 primary 효과는 없다.** AutoVLA Recent−Normal과 방향 대조, SpatialVLA token correction/reverse는 장면 가중·log cluster 기준에서도 유지된다. 이 판정은 Holm 보정 뒤에도 같다.
- **p값은 크게 약해진다.** 원 unit-level p는 365개 단위를 독립으로 셌다. 이번 재분석은 독립 cluster 16개(A− log)를 기준으로 한다.
  - Recent−Normal: 8.0e-42 → 1.2e-4.
  - 방향 대조: 6.1e-4 → 5.6e-3, Holm 보정 후 0.011.
  - 극소 p값은 원고에 쓰지 않아야 한다.
- **단일 log에 의존하는 primary 효과는 없다.** 16 log를 하나씩 빼도 부호는 바뀌지 않았다. CI는 0을 포함하지 않았고, sign-flip p는 0.05를 넘지 않았다.
- **가중 방식에 따라 결론이 달라지는 보조 진술이 하나 있다.** 실험 11의 "GT와 motion이 비슷한 token ≈ GT token" 동등성이다.
  - unit 가중에서는 ±5 pp 동등성이 성립한다.
  - 장면 가중과 log 균등 가중에서는 성립하지 않는다(log 균등 +3.9 pp, 90% CI [+0.2, +8.4]).
- **SpatialVLA closed-loop의 문맥 효과는 계속 비유의하다.** 그러나 ±10 pp 동등성도 성립하지 않는다. 따라서 "효과 없음"이 아니라 "검출되지 않음, 동등성 미확인"으로 써야 한다.
- **AutoVLA primary의 Reverse−Full은 계산할 수 없다.** 이 대비는 실험 8에만 있고, 실험 8에는 per-unit raw가 없다.

## 1. 확인한 사실

### 1.1 Raw 존재 여부

**확인한 사실:** 실험 5/8/9/10은 per-unit raw가 없다.
- 각 디렉토리에는 summary/report/figure만 있다.
- 파일시스템 전체에서 해당 이름의 `*.jsonl`을 검색했지만 나오지 않았다.
- summary.json에도 log별 값이 없다.

따라서 아래 항목은 **raw unavailable**이다.

| 실험 | 영향 받는 수치 | 필요한 것 |
|---|---|---|
| 5 first-mismatch | 1-token 교정 +55.8 pp 등 | 장면별 조건 결과(token, log, group, 조건별 trajectory 또는 recovery/FDE) |
| 8 state patching | **AutoVLA primary Reverse−Full**(reverse@emb − GT-history, 원 +29.0 pp [+21.1, +38.0], p = 6.7e-31), layer 37개 | 단위별 (token, pert, log) × 93 조건 결과 |
| 9 temporal window | window 1–4 회복률 | 단위별 × 17 조건 결과 |
| 10 horizon-controlled | t* = 0 window, 재발산 | 단위별 × window × N 결과(보고서가 인용한 `errors.jsonl`도 없음) |

- 복구하려면 GPU harness를 다시 실행해야 한다. 이번 작업 범위 밖이다.
- 원 summary 값은 `comparison_with_original.csv`의 AV-P2 행에 "RAW UNAVAILABLE"로만 남겼다.

### 1.2 Join과 제외

출처: `join_report.json`.

**AutoVLA 실험 6/7/11/32**
- 네 실험을 `(장면 token, perturbation)`으로 join했다.
- 실험 7/11/32는 각각 1,408 단위이고, 1,408개 모두 세 실험에 공통으로 있다.
- 실험 6과 비교한 결과:
  - 누락 단위 0, 초과 단위 0.
  - log 또는 group 불일치 0.
- 규모:
  - A−: 365 단위, 52 장면, 16 log.
  - A+: 1,043 단위, 156 장면, 26 log.
- paired 누락은 모든 대비에서 0이다.
- primary 조건에는 OOD 규칙이 적용되지 않으므로 단위 제외도 0이다.
- 실험 32에서 대체 기록이 없는 단위는 0개다. 따라서 제외 민감도는 주 분석과 같다.
- 실험 6의 A− 대안 1개는 invalid output이다. 원 분석처럼 증폭 False로 포함했다.

**Impromptu 실험 33 C–E**
- 사용한 단위: 2,398개(300 장면, 28 log).
- 원 규칙에 따라 제외한 단위:
  - 기준 단위(dir = None) 300개.
  - parse 실패 2개. 이 2개는 `units.parquet`에 `exclusion_reason`으로 남겼다.

**SpatialVLA 실험 34**
- closed loop: 80 episode(과제당 40개)가 9 조건 모두에서 짝을 이룬다.
- token level: 18,975 단위, 1,600 frame, 80 episode.
- 방향 대조는 e > 0인 10,542 단위만 쓴다. 원 규칙과 같다.

**검증:** 이번 재분석의 unit 가중 점추정은 원 분석 값과 **모든 대비에서 정확히 일치**한다(`comparison_with_original.csv`, `unit_estimate_matches = True`). unit-level McNemar p도 같다. 즉 join과 지표 재구성은 원 분석과 같은 자료를 쓴다.

### 1.3 독립 cluster 수와 반복 단위 수

| 대상 | 독립 cluster | 중간 단위 | 반복 단위 |
|---|---|---|---|
| AutoVLA A− (실험 6/7/11/32) | 16 log | 52 장면 | 365 (장면당 평균 7.0) |
| AutoVLA A+ | 26 log | 156 장면 | 1,043 |
| Impromptu C–E | 28 log | 300 장면 | 2,398 |
| SpatialVLA closed loop | 80 episode (2 과제 × 40 seed) | – | 80 (episode = 단위) |
| SpatialVLA token | 80 episode | 1,600 frame | 18,975 (방향 대조 10,542) |

AutoVLA A−의 장면과 단위는 log 사이에 고르게 퍼져 있지 않다.
- 한 log(`2021.09.16.19.27.01_veh-45_01749_03230`)가 52 장면 중 15개(29%), 365 단위 중 104개를 차지한다.

## 2. 계산한 결과

수치 읽는 법:
- 기본 수치는 장면 가중(primary)이다.
- [ ] = log block bootstrap 95% CI(B = 10,000, percentile).
- p = log 요약값의 sign-flip 검정(16 log는 2^16개 부호 조합 전수 열거). Holm은 family 안 보정값이다.

### 2.1 AutoVLA primary family (A−, Holm m = 3)

| 대비 | 원 분석 (unit 가중) | 장면 가중 (주) | unit 가중 | log 균등 | 주 p (sign-flip) | Holm p | cluster-t p (df 15) |
|---|---|---|---|---|---|---|---|
| Recent−Normal (실험 7) | −40.3 pp [−50.3, −30.0], McNemar p = 8.0e-42 | **−39.8 pp [−50.0, −28.8]** | −40.3 [−50.6, −29.7] | −43.0 [−56.7, −29.3] | 1.2e-4 | 3.7e-4 | 2e-6 |
| Reverse−Full (실험 8) | +29.0 pp [+21.1, +38.0], p = 6.7e-31 | **raw unavailable** | – | – | – | (p = 1로 family에 포함) | – |
| Direction−Magnitude (실험 32) | +9.0 pp [+3.1, +18.7], McNemar p = 6.1e-4 | **+9.9 pp [+3.8, +20.4]** | +9.0 [+3.2, +19.4] | +12.9 [+2.7, +22.8] | 5.6e-3 | 0.011 | 0.027 |

보조 지표 FDE5:
- Recent−Normal: −4.06 m [−4.73, −3.20], p = 3.1e-5. 원 분석은 −4.20 m, Wilcoxon p = 4.5e-53.
- Direction−Magnitude: +1.22 m [+0.71, +2.10], p = 1.8e-4. 원 분석은 +1.21 m, p = 9.8e-9.

log별 부호:
- Recent−Normal: 16 log 중 14개가 음수, 2개가 0, 양수는 0개.
  - 효과가 0인 log는 부호 정보를 주지 않는다. 그래서 정보가 있는 log는 14개이고, 전수 sign-flip p의 하한은 2/2^14 = 1.2e-4다. 관측 p가 바로 이 하한이다.
- Direction−Magnitude: 11개 양수, 2개 음수, 3개 0.
  - 음수 log: `2021.09.16.14.39.34_veh-42` −33 pp, `2021.05.25.14.24.08_veh-25_00934` −6 pp.

### 2.2 Leave-one-log-out

출처: `loo_logs.csv`. 각 LOO마다 bootstrap B = 2,000을 쓰고, 15 log 전수 sign-flip을 했다.

| 대비 | 전체 | LOO 범위 | LOO 최대 p | CI가 0 포함 | 부호 반전 | 영향이 가장 큰 log (제거 시) |
|---|---|---|---|---|---|---|
| Recent−Normal (실험 7) | −39.8 pp | −43.5 ~ −36.2 | 2.4e-4 | 0/16 | 0 | `2021.06.28.18.03.27_veh-14` (−43.5), `2021.09.16.19.27.01_veh-45` (−36.2) |
| Recent−Normal (실험 11 harness) | −38.8 | −42.8 ~ −35.2 | 2.4e-4 | 0/16 | 0 | 같은 두 log |
| Direction−Magnitude (실험 32) | +9.9 pp | +8.1 ~ +13.3 | 0.011 | 0/16 | 0 | `2021.09.09.18.29.25_veh-39` (+8.1, p = 0.011); `2021.09.16.19.27.01_veh-45` 제거 시 +13.3 |
| Recent−Normal FDE | −4.06 m | −4.20 ~ −3.74 | 6.1e-5 | 0/16 | 0 | `veh-45_01749` (−3.74) |
| Direction−Magnitude FDE | +1.22 m | +1.06 ~ +1.55 | 3.7e-4 | 0/16 | 0 | `veh-45_01749` (+1.55), `2021.09.29.19.02.14_veh-28` (+1.06) |

**판정:** protocol §5 기준으로 단일 log에 의존하는 primary 효과는 없다.
- 방향 대조에서 영향이 가장 큰 log는 `veh-45_01749`이고, 그 방향은 예상과 반대다.
  - 이 log는 15 장면을 차지하는데, 효과가 거의 0이다(+1.5 pp).
  - 그래서 이 log를 빼면 효과가 오히려 커진다(+13.3).
  - unit 가중에서 log 균등 가중으로 바꿀 때 +9.0 → +12.9로 커지는 것도 같은 이유다.

### 2.3 가중 민감도

| 대비 | 장면 | unit | log 균등 |
|---|---|---|---|
| Recent−Normal | −39.8 | −40.3 | −43.0 |
| Direction−Magnitude | +9.9 | +9.0 | +12.9 |
| GT-history−Normal (실험 7, 보조) | −42.5 | −43.3 | −45.4 |
| Mirror−Normal (실험 32, 보조) | −33.6 | −34.2 | −35.4 |
| geo_nn_gt − recent_gt (실험 11, 동등성) | +0.8 [90% CI −1.4, +5.2] | +0.5 [−1.6, +5.0] | +3.9 [+0.2, +8.4] |

- 방향(부호)과 유의성은 모든 primary 대비에서 가중 방식과 무관하다.
- 실험 11 동등성 진술(margin ±5 pp, protocol §6)은 판정이 갈린다.
  - unit 가중에서는 "동등"이다.
  - 장면 가중과 log 균등 가중에서는 "동등성 미확인"이다.
  - log 균등 가중에서는 90% CI가 0을 넘는다(GT 근접 token이 GT보다 약간 나쁨).
  - 따라서 "GT-like motion is as effective as the GT token"은 동등성 주장으로 쓰지 말아야 한다. "차이가 작고 비유의"(장면 가중 +0.8 pp [−1.6, +6.6], p = 0.73)로 낮춰 써야 한다.

### 2.4 AutoVLA 보조 (A+, 실험 6)

**A+ 그룹(26 log)**

| 대비 | 원 분석 | 재분석 (장면 가중) |
|---|---|---|
| Recent−Normal | −3.5 pp, McNemar p = 9.3e-9 | −3.4 pp [−5.6, −1.0], sign-flip p = 0.022 |
| Direction−Magnitude | +4.9 pp, p = 3.7e-7 | +4.5 pp [+1.8, +7.3], p = 0.011 |

둘 다 여전히 0.05 아래지만, 증거는 수 자릿수 약해진다.

**실험 6 대안 증폭률**

| 그룹 | 원 분석 | 장면 가중 | log 균등 |
|---|---|---|---|
| A− | 45.0% | 43.8% [34.9, 53.1] | 47.2% |
| A+ | 4.8% | 4.7% [2.7, 6.7] | 3.0% |

A− − A+ 차이: +39.1 pp [+29.8, +49.7]. 26 log를 공동으로 재표집한 값이다.

### 2.5 Impromptu (실험 33, 28 log, 보조 family Holm m = 4)

| 대비 | 원 분석 | 장면 가중 | log 균등 | sign-flip p | LOO 범위 |
|---|---|---|---|---|---|
| Recent−Normal | −62.2 [−65.6, −59.1] | −62.2 [−65.4, −59.0] | −62.7 | ≤ 5e-6 (MC 하한) | −63.2 ~ −61.6 |
| GT-history−Normal | −63.4 | −63.4 [−66.6, −60.5] | −63.6 | ≤ 5e-6 | −64.4 ~ −62.9 |
| Reverse−Full | +81.9 [+80.4, +84.2] | +81.9 [+80.3, +84.1] | +83.3 | ≤ 5e-6 | +81.6 ~ +82.4 |
| Direction−Magnitude | +15.6 [+9.0, +24.2] | +15.7 [+9.3, +24.2] | +20.8 | 1.0e-5 | +14.5 ~ +18.8 |

- 원 결론은 그대로다.
- p는 원래 "0"(underflow)이었다. 이번에는 Monte Carlo 200,000회의 하한이다. 28 log의 전수 sign-flip이라면 하한이 2/2^28이다.

### 2.6 SpatialVLA (실험 34, primary family Holm m = 6)

방법: 과제 안에서 episode를 재표집하고, 그 episode의 모든 조건을 함께 묶었다.

**Closed loop (성공률, 80 episode)**

| 대비 | 원 분석 (비층화 bootstrap) | 재분석 (과제 층화) | sign-flip p | Holm p | ±10 pp 동등성 |
|---|---|---|---|---|---|
| Feedback−Corrected opposite | +1.25 pp [−12.5, +13.75], p = 1 | +1.25 [−11.25, +13.75] | 1.0 | 1.0 | 미확인 (90% CI [−10.0, +12.5]) |
| Reverse−Natural opposite | −1.25 pp, p = 1 | −1.25 [−11.25, +8.75] | 1.0 | 1.0 | 미확인 ([−10.0, +7.5]) |
| Feedback−Corrected perp_left | −5.0 pp, p = 0.29 | −5.0 [−12.5, +1.25] | 0.29 | 1.0 | 미확인 |
| Reverse−Natural perp_left | +5.0 pp, p = 0.39 | +5.0 [−3.75, +13.75] | 0.39 | 1.0 | 미확인 |

과제별 결과(모두 비유의):

| 대비 | move_near | pick_coke_can |
|---|---|---|
| Feedback−Corrected opposite | +10.0 pp [−5, +25], p = 0.34 | −7.5 [−27.5, +12.5], p = 0.63 |
| Reverse−Natural opposite | +7.5, p = 0.55 | −10.0 [−22.5, 0.0], p = 0.22 |

- 두 과제의 효과 부호가 서로 반대다. pooled 값이 0에 가까운 이유 중 하나다.
- 과제 안 40 episode로는 이를 구별할 수 없다.
- 층화 때문에 CI 하한이 1.25 pp 좁아졌을 뿐, 결론은 같다.
  - 비유의다.
  - 동등성도 미확인이다. "효과 없음"이라고 쓸 수 없다.

**Token level (증폭 A ≥ 1)**

| 대비 | 원 분석 | episode 가중 (주) | unit 가중 | sign-flip p | Holm p |
|---|---|---|---|---|---|
| Correction (recent_ref − normal) | −8.3 pp [−9.3, −7.4], McNemar p = 7e-261 | −8.3 [−9.3, −7.4] | −8.3 [−9.3, −7.4] | ≤ 5e-6 | 3e-5 |
| Reverse (reverse − full_ref) | +5.1 [+4.4, +5.8], p = 2e-162 | +5.1 [+4.4, +5.8] | +5.1 | ≤ 5e-6 | 3e-5 |
| Direction−Magnitude (보조) | +1.6 [+0.4, +2.8], p = 7e-5 | +1.9 [+0.7, +3.1] | +1.6 | 0.003 | – |

과제별 효과:
- Correction: move_near −9.4 pp, pick_coke_can −7.2 pp.
- Reverse: move_near +6.0 pp, pick_coke_can +4.2 pp.

두 과제 모두 CI가 0을 포함하지 않는다. 다만 방향 대조는 과제별 unit 가중에서 pick_coke_can의 95% CI가 0을 포함한다([−0.3, +3.0]).

### 2.7 원 분석과의 판정 대조

출처: `comparison_with_original.csv`.

| 대비 | 원 p | 새 p (sign-flip, 장면/episode 가중) | 원 판정 | 새 판정 | 변화 |
|---|---|---|---|---|---|
| AutoVLA Recent−Normal | 8.0e-42 | 1.2e-4 (Holm 3.7e-4) | 유의 | 유의 | p 약 10^37배 증가, 판정 동일 |
| AutoVLA Direction−Magnitude | 6.1e-4 | 5.6e-3 (Holm 0.011; log 균등 0.027) | 유의 | 유의 | p 약 9–44배 증가 |
| AutoVLA Reverse−Full | 6.7e-31 | 계산 불가 | 유의 | raw 없음 | 재검정 불가 |
| AutoVLA A+ Recent−Normal | 9.3e-9 | 0.022 | 유의 | 유의(경계 근처) | 약화 |
| 실험 11 geo_nn_gt ≈ recent_gt | 0.85 | 0.73 | "같음"으로 서술 | 동등성은 unit 가중에서만 | **진술 수정 필요** |
| Impromptu 4 대비 | 0–2e-31 | ≤ 1e-5 | 유의 | 유의 | 판정 동일 |
| SpatialVLA closed loop 4 대비 | 0.29–1 | 0.29–1 | 비유의 | 비유의, 동등성 미확인 | 판정 동일 |
| SpatialVLA token correction/reverse | 7e-261 / 2e-162 | ≤ 5e-6 | 유의 | 유의 | 판정 동일 |

## 3. CI와 p 판정이 달라지는 이유 (가정 설명)

1. **원 unit-level McNemar/Wilcoxon**
   - 가정: A− 365 단위가 서로 독립이다.
   - 실제 구조: 단위는 52 장면 안에 반복되고, 장면은 16 log 안에 묶인다.
   - 같은 장면의 섭동끼리는 결과가 강하게 상관한다. Claim 2: FDE 분산의 54%가 장면 사이에서 나온다.
   - 그래서 유효 표본이 365보다 훨씬 작은데 p는 365를 기준으로 계산된다. 이것이 8e-42 같은 극소 p의 원인이다.
2. **원 log bootstrap CI**
   - log cluster를 이미 반영했다. 그래서 새 CI와 폭이 거의 같다.
   - 점추정만 unit 가중에서 장면 가중으로 바뀌었다(차이 1 pp 이내).
   - 원 분석의 CI와 p는 서로 다른 독립 단위를 가정했다. CI는 log 기준이고, p는 unit 기준이다.
3. **새 sign-flip**
   - 가정: H0에서 각 log 요약값의 분포가 0을 중심으로 대칭이고, log끼리 독립이다.
   - 조건은 무작위 배정이 아니라 같은 입력에 대한 결정적 개입이다. 따라서 이 검정은 정확한 randomization inference가 아니다. 대칭성 가정 아래의 조건부 검정이다.
   - p의 하한은 2/2^(정보 있는 log 수)다. AutoVLA A−에서는 1.2e-4–3.1e-5이므로, 그보다 작은 p는 원리적으로 나올 수 없다.
4. **16 cluster에서의 한계**
   - percentile bootstrap은 cluster 수가 적으면 CI가 약간 좁아지는 경향이 있다(가설 수준의 일반론).
   - 그래서 cluster-t(df 15)를 함께 보고했다. 방향 대조에서 cluster-t p는 0.027이다. sign-flip p(0.0056)보다 크지만 여전히 0.05 미만이다.
5. **SpatialVLA closed loop**
   - episode 하나가 곧 단위다. 그래서 unit-level McNemar와 episode sign-flip이 사실상 같은 검정이 되고 p도 같다.
   - 원 bootstrap은 docstring과 달리 과제 층화를 하지 않았다(SOURCE_MAP m5). 이번에 층화했지만 결과는 거의 같다.

## 4. 미확인

- 실험 5/8/9/10의 per-unit raw. 특히 AutoVLA primary인 **Reverse−Full의 cluster 재분석은 할 수 없다**.
  - 원 +29.0 pp [+21.1, +38.0]의 CI는 log bootstrap이므로 폭은 유지될 가능성이 크다.
  - 그러나 p = 6.7e-31은 unit-level이다. 군집 기준 p는 알 수 없다.
  - 이를 확인하려면 실험 8의 reverse@emb와 GT-history 조건을 per-unit 기록과 함께 재실행해야 한다.
- 16 log 밖으로의 일반화. LOO는 이 16 log 안에서 단일 log 의존이 없다는 것만 보인다. 새 log에서도 재현되는지는 P1-C에서 확인해야 한다.
- log보다 상위의 의존성. 같은 차량·날짜에서 나온 log끼리(예: `2021.05.25.14.24.08_veh-25`의 두 구간, `2021.09.29.14.44.26_veh-28`의 두 구간) 상관이 있으면 독립 cluster 수는 16보다 작다. 이번 분석은 이를 검사하지 않았다.
- SpatialVLA token level의 frame 수준 가중. episode 안에서 frame별 단위 수가 달라도 episode 평균으로만 묶었다.

## 5. 가설

- 방향 대조의 효과 크기가 가중 방식에 따라 +9.0~+12.9 pp로 달라진다. 이는 장면이 가장 많은 log(`veh-45_01749`, 15 장면)에서 방향 효과가 거의 없기 때문이다. 이 log의 장면 특성(예: 저속·직진 위주라 방향 오차가 증폭으로 이어지기 어려움)이 원인일 수 있지만, 확인하지 않았다.
- SpatialVLA closed loop에서 두 과제의 부호가 반대인 것은 과제별 실패 경로가 달라서일 수 있다. 다만 과제당 40 episode는 이 차이를 검출하기에 부족하다.

## 6. 원고 수정 권고 (재분석에 근거)

- p값 표기:
  - AutoVLA Recent−Normal은 "p = 8e-42" 대신 "log-level sign-flip p = 1.2e-4 (16 logs; exact)"로 쓰고, CI는 그대로 쓴다.
  - 방향 대조는 "p = 6e-4" 대신 "p = 0.006 (Holm 0.011)"로 쓴다.
- "As effective as the GT token"(Claim 6)은 동등성이 아니라 "작고 비유의한 차이"로 쓴다.
- SpatialVLA closed loop은 "indistinguishable from zero" 뒤에 "equivalence within ±10 pp not established"를 덧붙인다.
- AutoVLA Reverse−Full의 p는 unit-level이라고 명시한다. 또는 재실행 전까지 CI만 보고한다.

## 7. 파일

- `protocol.md`: 분석 전 작성.
- `manifest.json`: 시각, sha256, 입력·설정, raw unavailable 목록.
- `p0b_analysis.py`, `run.log`, `join_report.json`.
- `units.parquet`: 252,048행, long format. 열: model, experiment, unit_id, scene_id, log_id, task_id, episode_id, seed, group, perturbation, condition, amplification, fde5, success, exclusion_reason, note.
- `condition_summary.csv`: 조건별 수준. 장면/unit/log 가중, cluster CI.
- `paired_effects.csv`: 대비 × 가중. 열: estimate, 95/90% CI, sign-flip p와 방법, cluster-t p, unit-level p, Holm p, n_clusters/n_scenes/n_units, 동등성 판정.
- `loo_logs.csv`: AutoVLA 5 대비 × 16 log, Impromptu 4 대비 × 28 log.
- `comparison_with_original.csv`: 원 점추정·CI·p와 새 값 대조.
