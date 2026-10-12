# A3 군집 재분석: AutoVLA·Impromptu primary 대비를 drive 단위로 (2026-10-13)

**성격: 기존 자료의 사후(POST-HOC) 분석이다.** 사전 등록이 아니다. 원 결과(unit-level p, P0-B log 결과, P1-C 결과)를 모두 본 뒤 수행했다. 모델 추론·GPU는 쓰지 않았다(CPU, `nice -n 19`). 기존 파일은 수정하지 않았다. P1-C 행은 사전 등록된 drive 수준 분석을 독립 코드로 다시 계산한 확인용이다.

- 표: `CLUSTER_REANALYSIS.csv`(대비당 1행, 넓은 형식), `cluster_reanalysis_long.csv`(대비 × cluster 수준 × 가중), `leave_one_drive_out.csv`, `per_drive_effects.csv`, `p1c_subsets_drive_level.csv`, `verification_vs_reported.csv`.
- 그림: `fig_log_to_drive_effects.png`. CSV에서 `a3_figure.py`로 생성.
- 코드: `scripts/review_followup2/a3_cluster_reanalysis.py`, 통계 모듈 `stats_core.py`, 검증 `test_stats_core.py`(출력은 `test_stats_core_output.txt`).
- 표기: [확인한 사실] / [계산한 결과] / [미확인] / [가설].

## 0. 요약

- **판정이 바뀐 대비는 없다.** [계산한 결과]
  - 대상: AutoVLA primary family 3개, 실험 11 동등성 질문, Impromptu 4개, P1-C 6개.
  - 독립 단위를 log에서 drive로 올리고 원 primary family 안에서 Holm 보정을 해도 판정은 그대로다.
- **AutoVLA A−의 독립 단위는 13 drive다.** 16 log, 52 장면, 365 단위가 여기에 들어 있다.
  - 정확 sign-flip p의 하한은 2/2^13 = 2.4e-4다.
  - Recent−Normal: p 1.2e-4(log) → 2.4e-4(drive). Holm 후 7.3e-4.
  - Reverse−Full: p 6.1e-5 → 2.4e-4. Holm 후 7.3e-4.
  - 두 대비 모두 관측 p가 바로 이 하한이다. 즉 13 drive 모두에서 효과 부호가 같다. 13 drive로 얻을 수 있는 가장 강한 증거이며, 그보다 더 작은 p는 나올 수 없다.
  - Direction−Magnitude: p 0.0056(log와 drive가 같음). Holm 0.0056.
- **95% CI는 거의 바뀌지 않는다.**
  - P0-B의 log bootstrap CI는 이미 log를 cluster로 썼다. 같은 drive에 속한 log는 3쌍뿐이다.
  - 예: Recent−Normal log [−50.0, −28.8] → drive [−49.0, −29.9] pp.
  - sign-flip 검정을 뒤집어 얻은 95% CI는 조금 더 넓다: [−51.5, −28.7].
- **Leave-one-drive-out**: AutoVLA primary 3개와 Impromptu 4개 모두에서 부호 반전 0, CI가 0을 포함한 경우 0이다.
  - 최대 p: Recent−Normal·Reverse−Full 4.9e-4, Direction−Magnitude 0.011.
- **실험 11 "GT와 motion이 가장 가까운 token ≈ GT"는 여전히 동등성이 성립하지 않는다(판정: 불확실).**
  - drive 기준 +0.8 pp, 95% CI [−1.6, +6.6], 90% CI [−1.4, +5.2].
  - TOST(±5 pp, drive sign-flip): p_lower 1.2e-4, p_upper 0.090 → p_equiv 0.090.
  - bootstrap TOST p_equiv는 0.057이다.
  - unit 가중에서도 sign-flip TOST p_equiv는 0.087이다. P0-B는 unit 가중 90% bootstrap CI만으로 "동등"이라고 판정했는데, 이 판정은 sign-flip TOST에서는 유지되지 않는다.
- **Impromptu 실험 33의 독립 단위는 20 drive다**(28 log). p는 4e-6~2e-5(MC 하한) → 1.9e-6(전수 하한 2/2^20). Direction−Magnitude만 8.2e-5다. 판정은 같다.
- **P1-C는 사전 등록 결과를 완전히 재현했다.**
  - 6개 대비 모두 θ, 전수 p, Holm p, bootstrap CI(같은 seed), sign-flip 90% CI, 판정이 `decisions.json`과 일치한다.
  - RN과 RV는 F·P 모두 "replicated", DM은 F·P 모두 "inconclusive"다.

## 1. 방법 (왜 이 방법인가)

**계층과 짝짓기를 보존했다.**
- 계층: drive(`<date.time>_veh-NN`) → log segment → 장면(token) → 섭동 단위(original/alt_k 또는 방향).
- 대비는 원 실험(또는 P1-C block) 안에서만 계산한다. 따라서 각 대비는 자기 실험의 Normal을 쓴다(P0-A F6).
- 단위마다 두 조건을 짝지었다(`stats_core.pair_rows`). 짝이 빠진 단위는 0개다. AutoVLA primary 조건은 원 분석에서도 제외가 없었다.
- Impromptu는 원 규칙대로 parse 실패 2단위를 뺐다.

**Estimand는 그대로 두었다.**
- 주 estimand는 장면 가중이다. 장면 안 단위 평균을 내고, 장면을 동등하게 가중한다(P0-B·P1-C의 primary).
- P1-C 모집단(P) estimand는 Hajek 가중 w = 1/π를 쓴다.
- 다음 둘은 **서로 다른 추정 대상**이므로 별도 열에 둔다.
  - unit 가중: 원 분석의 점추정.
  - drive 균등 가중: drive 안 장면 가중 비율을 drive끼리 동등하게 평균.
- 비교를 위해 log 균등 가중도 long 표에 둔다.

**추론은 drive 단위로 한다.**
- **95% CI(주):** drive cluster bootstrap. drive를 복원 추출하고, 뽑힌 drive의 log·장면·단위·두 조건을 통째로 넣는다. B = 10,000, percentile.
  - 하위 계층은 따로 재표집하지 않는다. 최상위 cluster의 합이 하위 변동을 모두 담기 때문에, 모든 계층을 독립으로 재표집하면 변동을 이중으로 센다.
- **p:** drive 합 T_g에 대한 sign-flip.
  - G ≤ 26이면 2^G 전수 열거(meet-in-the-middle)다. 여기서는 모든 drive 수준 검정이 전수다(13/19/20/18/24 drive).
  - G > 26이면 Monte Carlo(R = 200,000, seed 기록, p = (b+1)/(R+1))다. 64 log처럼 log 수준에서만 쓰였다.
- **sign-flip의 가정:**
  - H0에서 drive 합끼리 독립이고, 각각 0에 대해 대칭이다.
  - 조건은 같은 입력에 대한 결정적 개입이고 무작위 배정이 아니다. 따라서 이것은 randomization test가 아니다. 대칭성 가정 아래의 조건부 검정이다.
  - drive 안 모든 관측의 부호가 함께 바뀐다(drive 합에 부호를 곱함).
- **보조 지표:** sign-flip을 뒤집어 얻은 95%·90% CI, cluster-robust t(df = G − 1).
- **Holm 보정은 원 family 안에서만 한다.**
  - AutoVLA primary: m = 3.
  - Impromptu: m = 4.
  - P1-C F: m = 3, α = .05.
  - P1-C P: m = 3, α = .025.
  - 실험 11과 FDE5·A+ 대비는 family 밖 보조 대비라서 보정하지 않았다.
- **동등성 질문(실험 11, margin ±5 pp, P0-B §6에서 정한 값):** cluster TOST.
  - p_lower, p_upper, p_equiv = max(p_lower, p_upper)를 보고한다.
  - 90% CI와 95% CI를 함께 보고한다.
  - "0과의 차이"와 "동등성"은 서로 배타적이지 않은 별도 판정으로 둔다(`stats_core.judge`).
- **검증** [계산한 결과]: 새 코드로 P0-B의 log 수준 장면 가중 추정과 전수 p를 다시 계산했다(`verification_vs_reported.csv`).
  - 모든 대비에서 일치한다. 같은 seed를 쓴 bootstrap CI도 소수 5자리까지 같다.
  - 예외: P0-B가 MC로 계산한 26–28 log 대비는 MC 오차 범위 안에서 일치한다. 이 경우 이번 26 log 계산은 전수다.

## 2. 결과표 (장면 가중, amplification, pp)

n = 단위/장면/log/drive(정보 있는 drive). CI는 drive bootstrap 95%이고, [sf]는 sign-flip 검정을 뒤집어 얻은 95% CI다.

| 대비 | n | 원 추정 [95% CI], 원 p (단위 독립 가정) | log 기준: 추정, p, Holm | drive 기준 95% CI / [sf] | drive p (전수; 하한) | drive Holm | 판정 변화 |
|---|---|---|---|---|---|---|---|
| AV-P1 Recent−Normal (실험 7) | 365/52/16/13(13) | −40.3 [−50.3, −30.0], 8e-42 | −39.8, 1.2e-4, 2.4e-4 | [−49.0, −29.9] / [−51.5, −28.7] | 2.4e-4 (2.4e-4) | 7.3e-4 | 없음 |
| AV-P2 Reverse−Full (R8, reverse@emb) | 365/52/16/13(13) | +29.0 [+21.1, +38.0], 6.7e-31 | +28.6, 6.1e-5, 1.8e-4 | [+20.9, +36.1] / [+19.2, +37.9] | 2.4e-4 (2.4e-4) | 7.3e-4 | 없음 |
| AV-P3 Direction−Magnitude (실험 32) | 365/52/16/13(13) | +9.0 [+3.1, +18.7], 6.1e-4 | +9.9, 0.0056, 0.0056 | [+3.9, +21.0] / [+3.0, +24.3] | 0.0056 (2.4e-4) | 0.0056 | 없음 |
| 실험 11 GeoNN(GT)−RecentGT (동등성 ±5) | 365/52/16/13(7) | +0.5 [−1.9, +6.3], 0.85 | +0.8, 0.73 | [−1.6, +6.6] / [−1.9, +8.3]; 90% [−1.4, +5.2] | 0.73 (0.016) | – | 없음: 동등성 미성립(TOST p_equiv 0.090) → "불확실" |
| IMP Recent−Normal | 2398/300/28/20(20) | −62.2 [−65.6, −59.1], "0" | −62.2, 5e-6 (MC), 2e-5 | [−65.4, −59.1] / [−66.0, −58.7] | 1.9e-6 (1.9e-6) | 7.6e-6 | 없음 |
| IMP GT-history−Normal | 2398/300/28/20(20) | −63.4 [−66.7, −60.6], "0" | −63.4, 5e-6, 2e-5 | [−66.5, −60.5] / [−67.1, −60.2] | 1.9e-6 | 7.6e-6 | 없음 |
| IMP Reverse−Full (자기 오차 재삽입) | 2398/300/28/20(20) | +81.9 [+80.4, +84.2], "0" | +81.9, 5e-6, 2e-5 | [+80.5, +83.9] / [+80.3, +84.3] | 1.9e-6 | 7.6e-6 | 없음 |
| IMP Direction−Magnitude | 2398/300/28/20(20) | +15.6 [+9.0, +24.2], 2e-31 | +15.7, 1e-5 (MC), 2e-5 | [+9.1, +24.7] / [+8.9, +27.0] | 8.2e-5 | 8.2e-5 | 없음 |
| P1-C RN F | 748/105/41/18(18) | −34.2 [−38.3, −29.5], 7.6e-6 (사전 등록, drive) | −34.2, 5e-6 (MC) | [−38.3, −29.5] / [−38.7, −28.2] | 7.6e-6 (7.6e-6) | 2.3e-5 | replicated(같음) |
| P1-C RV F | 748/105/41/18(16) | +26.8 [+21.9, +30.7], 3.1e-5 | +26.8 | [+21.9, +30.7] / [+20.5, +31.1] | 3.1e-5 | 6.1e-5 | replicated(같음) |
| P1-C DM F | 748/105/41/18(12) | +6.2 [+0.6, +12.1], 0.073 | +6.2, 0.061 | [+0.6, +12.1] / [−0.4, +12.7] | 0.073 | 0.073 | inconclusive(같음) |
| P1-C RN P | 2436/353/64/24(21) | −4.1 [−6.4, −2.4], 5.7e-6 | −4.1 | [−6.4, −2.4] / [−6.6, −2.1] | 5.7e-6 | 1.1e-5 | replicated(같음) |
| P1-C RV P | 2436/353/64/24(21) | +3.6 [+2.2, +5.5], 9.5e-7 | +3.6 | [+2.2, +5.5] / [+2.0, +5.7] | 9.5e-7 | 2.9e-6 | replicated(같음) |
| P1-C DM P | 2436/353/64/24(22) | +2.6 [−0.4, +5.5], 0.10 | +2.6, 0.061 | [−0.4, +5.5] / [−0.6, +5.9] | 0.10 | 0.10 | inconclusive(같음) |

### 2.1 가중 방식별 추정 (drive cluster, 서로 다른 estimand)

| 대비 | unit 가중 (p) | 장면 가중, 주 (p) | drive 균등 [95% CI] (p) |
|---|---|---|---|
| AV-P1 Recent−Normal | −40.3 (2.4e-4) | −39.8 (2.4e-4) | −46.6 [−58.6, −35.0] (2.4e-4) |
| AV-P2 Reverse−Full | +29.0 (2.4e-4) | +28.6 (2.4e-4) | +32.9 [+23.3, +43.0] (2.4e-4) |
| AV-P3 Direction−Magnitude | +9.0 (0.012) | +9.9 (0.0056) | +15.5 [+3.3, +27.0] (0.028) |
| 실험 11 GeoNN−RecentGT | +0.5 (0.89; TOST p_equiv 0.087) | +0.8 (0.73; TOST 0.090) | +4.7 [−0.3, +11.6] (0.17; TOST 0.48) |
| IMP Direction−Magnitude | +15.6 (8.2e-5) | +15.7 (8.2e-5) | +22.9 [+15.6, +30.3] (7.6e-6) |
| P1-C DM F | +5.2 (0.13) | +6.2 (0.073) | +4.7 [+0.6, +9.7] (0.058) |
| P1-C DM P | +2.8 (0.074) | +2.6 (0.10) | +2.4 [−0.5, +5.2] (0.11) |

나머지 대비의 가중별 값은 CSV의 `drive_estimate_unit` / `drive_estimate_drive_equal` 열에 있다.
- 방향 대조는 가중 방식에 민감하다(+9.0 ~ +15.5 pp). 장면이 가장 많은 drive(`2021.09.16.19.27.01_veh-45`, A− 15 장면)에서 효과가 거의 0이기 때문이다. 이 원인은 P0-B와 같다.
- 실험 11의 동등성은 어느 가중에서도 sign-flip TOST로 성립하지 않는다.
  - drive 균등 가중에서는 90% CI [+0.2, +10.4]가 0을 넘고 margin도 넘는다.
  - 즉 GT 근접 token이 GT보다 약간 나쁠 수 있다.

### 2.2 Leave-one-drive-out (장면 가중, B = 2,000, 전수 sign-flip)

| 대비 | LODO 범위 | 최대 p | CI가 0 포함 | 부호 반전 | 영향이 가장 큰 drive |
|---|---|---|---|---|---|
| AV-P1 Recent−Normal | −43.5 ~ −36.2 | 4.9e-4 | 0/13 | 0 | 2021.06.28.18.03.27_veh-14 |
| AV-P2 Reverse−Full | +26.4 ~ +31.4 | 4.9e-4 | 0/13 | 0 | 2021.06.28.18.03.27_veh-14 |
| AV-P3 Direction−Magnitude | +8.1 ~ +13.3 | 0.011 | 0/13 | 0 | 2021.09.16.19.27.01_veh-45 |
| IMP 4 대비 | 범위 폭 ≤ 4.3 pp | ≤ 1.6e-4 | 0/20 | 0 | – |
| P1-C RN/RV F·P | – | ≤ 6.1e-5 | 0 | 0 | – |
| P1-C DM F | +4.1 ~ +7.5 | 0.15 | 4/18 | 0 | 2021.09.16.19.49.00_veh-42 |
| P1-C DM P | +1.9 ~ +3.8 | 0.20 | 22/24 | 0 | – |

P0-B의 leave-one-log-out은 같은 drive에 속한 두 log를 따로 뺐다. 이번 LODO가 그 두 log를 함께 빼는 검사다.

### 2.3 P1-C 사전 지정 민감도 부분집합을 drive 수준에서 다시 계산 (`p1c_subsets_drive_level.csv`)

- **S1(navhard 출처가 아닌 log)**: F는 7 drive, P는 13 drive.
  - RN F: −34.5 pp, p = 0.016(하한 0.016). RV F: +27.0, p = 0.016. DM F: +5.5, p = 0.41.
  - RN P: −5.2, p = 0.002. RV P: +4.5, p = 0.001. DM P: +6.0, p = 0.011.
- **S2(day-vehicle 비중첩)**: F는 4 drive, P는 7 drive.
  - F 대비는 전수 p의 하한이 0.125다. sign-flip으로는 유의성을 원리적으로 판정할 수 없고, sign-flip 역산 CI는 무한하다.
  - P 대비 p: RN 0.0625, RV 0.031, DM 0.031.
- **해석** [가설 수준 주의]: S1·S2의 DM P는 양의 효과를 보이고 CI가 0을 넘는다. 그러나 이것은 부분집합에 대한 사후 분석이다. 사전 등록된 전체 집합의 판정은 "inconclusive"이며, 부분집합 결과로 이 판정을 바꾸지 않는다.

## 3. 작은 cluster 수에 따른 주의

- **AutoVLA A−는 13 drive뿐이다.** [계산한 결과]
  - percentile bootstrap은 cluster가 적으면 CI가 좁게 나오는 경향이 있다. `test_stats_core`의 모의 실험에서 경계 조건(G = 13, 기저율 8%, 효과 = margin) 95% coverage는 0.86이었다.
  - 그래서 sign-flip 역산 CI를 함께 둔다. 이 CI는 모의 실험 조건들에서 coverage 0.87–1.0이고, 대체로 bootstrap보다 넓다.
- **전수 sign-flip p에는 하한이 있다.** 하한은 2/2^(정보 있는 drive 수)다.
  - A− 13 drive: 2.4e-4. A+ 19 drive: 정보 있는 drive 9–14개라 하한은 0.0039–1.2e-4다. Impromptu 20 drive: 1.9e-6.
  - 이 하한보다 작은 p를 원고에 쓰면 안 된다.
- **이산적인 이진 결과에서는 sign-flip이 보수적이었다.** 모의 실험에서 G = 13, 24일 때 기저율 8%의 귀무 상황에서 기각률은 0.013–0.023이었다.
- **p와 CI가 어긋나는 경우가 있다.** P1-C DM F의 drive bootstrap 95% CI는 [+0.6, +12.1]로 0을 제외한다. 반면 sign-flip p는 0.073이고, 역산 CI [−0.4, +12.7]는 0을 포함한다.
  - 사전 등록 규칙은 sign-flip을 판정 기준으로 쓴다. 따라서 판정은 "inconclusive"다.
  - 원고에서는 bootstrap CI만 인용해 "효과 있음"으로 쓰지 않는다.
- **역산 CI가 불안정할 수 있다.** 정보 없는(합이 0인) drive가 있고 G가 작으면 sign-flip 역산 CI가 구간 형태가 아닐 수 있다. 예: S2 RN P의 상한이 0. 이런 경우는 bootstrap CI를 함께 본다.
- **13 drive는 서로 완전히 독립이 아닐 수 있다.**
  - A−의 13 drive는 10 day-vehicle에 속한다.
  - 같은 날·같은 차량의 drive 사이 상관(날씨, 경로, 센서 보정)은 이번 분석에서 다루지 않았다.
  - P1-C의 S3(day-vehicle cluster, 16개)는 원 분석에 이미 있다(RN F p = 0.002 등). [미확인: PoC 집합의 day-vehicle 수준 재분석은 하지 않았다]

## 4. 개입 정의 (모델 간 같은 이름, 다른 처치)

| 이름 | AutoVLA (실험 7/8/32, P1-C) | Impromptu (실험 33) | 미래 GT 사용 | 정상 counterfactual 참조 |
|---|---|---|---|---|
| Recent-GT | 문맥만 바꾼다. 매 생성 step k ≥ t*+2에서 위치 k−1을 GT로 바꾸는 **반복 개입**이다(8−t*회, A− 평균 6.8회). | 직전 waypoint를 GT로 바꾼다(절대 좌표, GT 정보 누출 포함). | 예 | 아니오 |
| Reverse | reverse@emb: GT-history 문맥에서 위치 k−1의 embedding을 **같은 batch의 별도 Normal 행**에서 복사한다. | 교정 문맥에 branch **자신의** k−1 오차를 다시 넣는다. | 예(기저 문맥) | AutoVLA: 예(Normal 행) / Impromptu: 아니오 |
| Direction / Magnitude | 위치 k−1에 대체값을 넣는다. GT와 자기 token으로 만든 방향 오답/크기 오답을 매 step 넣는다. | 같은 개념, waypoint 수준. | 예 | 아니오 |

같은 이름의 Reverse 대비라도 AutoVLA와 Impromptu에서는 다른 처치다. 효과 크기(+28.6 vs +81.9 pp)를 같은 처치의 모델 간 차이로 비교하면 안 된다.

## 5. 원고에 쓸 수 있는 문장 (제안)

- "Clustered by drive (13 independent drives in the A− set), Recent-GT context correction lowered amplification by 39.8 pp (95% drive-bootstrap CI −49.0 to −29.9; exact sign-flip p = 2.4e-4, the minimum attainable with 13 drives; Holm-adjusted 7.3e-4)."
- 위 문장은 사후 분석 결과로 표기한다. 단위 수준 p(8e-42, 6.7e-31, 2e-31, "0")는 쓰지 않는다.
- "GT와 motion이 가장 가까운 token은 GT와 차이가 작고 비유의했다(+0.8 pp, 95% CI −1.6 to +6.6). 다만 ±5 pp 동등성은 성립하지 않았다(TOST p = 0.09)."
