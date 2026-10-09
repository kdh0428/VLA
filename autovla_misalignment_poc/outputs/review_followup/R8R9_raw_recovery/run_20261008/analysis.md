# R8R9 결과: 실험 8(state patching)·실험 9(temporal window) raw 복구 재실행

작성: 2026-10-09.
- protocol: `protocol.md`(commit b212637, 실행 전).
- 실행: 2026-10-08 17:11–18:19 UTC, RTX 5090. 원 스크립트를 수정 없이 실행했다(sha256: `logs/original_script_sha256.txt`).
- 분석: CPU만.
- 이 파일은 상위 세션이 agent 출력을 그대로 저장한 것이다.

**성격: 이것은 복구(recovery)다. 확인(confirmation)이 아니다.**
- 원 summary 수치를 이미 알고 있는 상태에서 같은 분석을 재실행해 per-unit raw를 되살렸다.
- 그 위에 P0-B cluster 통계를 적용했다.
- 아래 p값은 "이미 보고된 분석의 복구 raw에 대한 cluster 통계"로만 인용해야 한다.

표기:
- [확인한 사실] / [계산한 결과] / [미확인] / [가설].
- 기본은 장면 가중이다.
- [ ] = log block bootstrap 95% CI(B = 10,000).
- p = log sign-flip. A−는 16 log 전수, A+와 pooled는 26 log MC 200,000.

## 0. 요약
- **재현은 정확하다.** [계산한 결과]
  - 비교 행 220개(R8 186, R9 34; 그룹 × 조건)의 unit 가중 증폭률이 원 `summary.json`과 차이 0.0 pp로 같다.
  - summary.json 전 필드를 대조했다. 부동소수 반올림(≤ 4.4e-16)과 Wilcoxon p 2개를 빼면 같다.
- **AutoVLA primary Reverse−Full(A−, reverse@emb − gt_history)** [계산한 결과]
  - 장면 가중 **+28.6 pp [+20.4, +37.0]**.
  - **log sign-flip p = 6.1e-5**(정보 있는 15 log의 하한 2/2^15). cluster-t p = 4.0e-6.
  - unit 가중 +29.0 pp [+21.2, +37.6]로, 원 +29.0 pp [+21.1, +38.0]과 같다.
  - LOO 16회 모두 CI가 0을 포함하지 않는다.
- **갱신된 AutoVLA primary family Holm(m = 3)**: Recent−Normal **2.4e-4**, Reverse−Full **1.8e-4**, Direction−Magnitude **5.6e-3**. [계산한 결과]
  - 이전 P0-B에서는 Reverse−Full을 p = 1로 넣어 Direction−Magnitude Holm이 0.011이었다.
- **window 1–4(A−)** [계산한 결과]
  - win_w − normal: −19.5 / −32.0 / −37.6 / −39.9 pp, 모두 p ≤ 9.2e-4.
  - win3 − gt_history: +4.2 pp, p = 0.12(정보 있는 log 4개).
  - win4 − gt_history: +1.8 pp, p = 0.5(정보 있는 log 2개).
- 90% 기준 critical window는 가중 방식에 따라 다르다. [계산한 결과]
  - unit 가중(원 규칙): 4 step.
  - 장면 가중: 3 step.
  - 80% 기준: 둘 다 3 step.

## 1. 실행·분모·재현 점검

| 항목 | 값 |
|---|---|
| exp 8 (93 행, B = 93) | 208 장면 ok, 1,408 단위, unit 실패 0. `raw/exp8/errors.jsonl` 빈 파일 [확인한 사실] |
| exp 8 sanity (`--limit 4 --max-alts 1 --debug`) | 4 장면 / 8 단위 ok [확인한 사실] |
| exp 9 (17 행, B = 17) | 208 장면 ok, 1,408 단위, 실패 0 [확인한 사실] |
| 재시도·병합 | 없음(launcher log에 OOM 없음) [확인한 사실] |
| 분모 | A− 365 단위 / 52 장면 / 16 log, A+ 1,043 / 156 / 26 (R8·R9 동일). 결과 기반 제외 없음. paired missing 0 [확인한 사실] |
| raw sha256 | `logs/raw_sha256.txt` [확인한 사실] |

### 1.1 원 summary와의 비교 (`comparison_with_original_summary.csv`, protocol §3)
- **정확 재현 기준(모든 비율이 1e-9 안에서 같음)을 충족한다.** [계산한 결과]
  - 220/220 행에서 diff_pp = 0.0이다.
  - 예: R8 A− normal 46.85%, gt_history 4.38%, reverse@emb 33.42%, recent_gt 7.40%. R9 A− win1–4는 26.58 / 14.25 / 8.49 / 5.75%.
- 원 분석 스크립트를 새 디렉토리에 돌린 summary.json을 원 파일과 필드별로 대조했다. [계산한 결과]
  - exp 8 summary: 77,888 leaf 중 617개가 다르고, 최대 차는 4.4e-16(합산 순서 반올림)이다.
  - exp 8 layer_results: 29,995 leaf 중 250개, 최대 2.2e-16이다.
  - exp 9 summary: 34,739 leaf 중 743개가 다르다. 그중 741개는 ≤ 1e-15다.
  - 나머지 2개는 "original token only" 부분집합의 Wilcoxon p다: win6 5.2e-7 vs 8.6e-9, win7 8.2e-4 vs 2.1e-4. 원인은 차이 ≈ 0인 쌍의 반올림이 Wilcoxon의 0-차이 처리를 바꾼 것으로 보인다. [가설]
  - Markdown 보고서의 표는 숫자까지 같다. 원 보고서의 narrative 문단만 없다.
- exp-7 raw(`outputs/action_history_causal/records.jsonl`)와 token이 같은 비율 [계산한 결과]
  - R8 normal 88.6%, recent_gt 88.6%, gt_history 92.9%.
  - R9 normal 89.2%, gt_history 94.0%.
  - 원래 보고된 88.6%(exp 8)·89.2%(exp 9)와 같다.
  - 다만 P0-A U5에 따르면 원 비교 대상이던 exp-7 raw는 사라졌다. 지금 파일이 그 원본과 같은 파일인지는 확인하지 않았다. [미확인]
- 기기 조건: driver 610.43, RTX 5090. 원 run_meta도 RTX 5090이다. 결과가 bit 단위로 같으므로 driver·library 변화의 영향은 이 지표에서 관측되지 않았다. [확인한 사실 + 계산한 결과]

## 2. R8 primary: Reverse−Full (A−, amplification)

| 가중 | 추정 [95% CI] | 90% CI | sign-flip p | cluster-t p |
|---|---|---|---|---|
| 장면 (주) | **+28.6 pp [+20.4, +37.0]** | [+21.7, +35.5] | **6.1e-5** | 4.0e-6 |
| unit | +29.0 pp [+21.2, +37.6] | – | 6.1e-5 | – |
| log 균등 | +30.9 pp [+20.5, +42.4] | – | 6.1e-5 | – |

[계산한 결과]
- 정보 있는 log는 15/16이다. p는 전수 sign-flip의 하한(2/2^15)과 같다.
- LOO 16회: 추정 +26.4 ~ +31.4 pp, CI 하한 최소 +18.9 pp, 모든 p ≤ 1.2e-4.
  - 영향이 큰 log: `2021.09.16.19.27.01_veh-45_01749_03230` 제거 시 +26.4, `2021.06.28.18.03.27_veh-14` 제거 시 +31.4.
- FDE5(보조): +3.59 m [+2.68, +4.25], p = 3.1e-5.
- 해석 주의(P0-A F1) [확인한 사실]: reverse@emb는 위치 k−1의 embedding을 reverse branch 자신의 출력이 아니라 **별도 Normal 행**에서 복사한다.

### 2.1 갱신된 AutoVLA primary family (P0-B, Holm m = 3)

| 대비 | 출처 | 장면 가중 | sign-flip p | Holm p (갱신) | 이전 Holm (Reverse−Full p = 1 자리표시) |
|---|---|---|---|---|---|
| Recent−Normal (실험 7) | P0-B | −39.8 pp [−50.0, −28.8] | 1.2e-4 | **2.4e-4** | 3.7e-4 |
| Reverse−Full (실험 8) | 이번 복구 | +28.6 pp [+20.4, +37.0] | 6.1e-5 | **1.8e-4** | (raw 없음) |
| Direction−Magnitude (실험 32) | P0-B | +9.9 pp [+3.8, +20.4] | 5.6e-3 | **5.6e-3** | 0.011 |

[계산한 결과] `p1_stats.holm`으로 계산했다. 세 primary 대비 모두 Holm 후 p < 0.01이다.

## 3. R8 보조 (비보정)

| 대비 | A− amplification | A− FDE5 | A+ amplification | pooled amplification |
|---|---|---|---|---|
| recent_gt − normal | −39.2 pp [−49.2, −28.0], p = 6.1e-5 | −3.96 m | −3.7 pp, p = 0.017 | −12.6 pp |
| gt_history − normal | −41.9 pp [−52.7, −29.7], p = 6.1e-5 | −4.48 m | −3.5 pp, p = 0.040 | −13.1 pp |
| reverse@emb − gt_history | (primary) | +3.59 m | +2.9 pp [+1.1, +4.8], p = 0.020 | +9.3 pp |
| reverse@L0 / L4 / L8 / L12 − gt | +34.0 / +34.1 / +34.9 / +33.7 pp | +3.66 ~ +3.77 m | +2.5 ~ +3.1 pp | +10.4 ~ +10.8 pp |
| reverse@L16 / L20 / L24 / L28 − gt | +35.2 / +35.9 / +36.3 / +36.7 pp | +3.70 ~ +4.05 m | +2.4 ~ +2.9 pp | +10.6 ~ +11.3 pp |
| reverse@L35 − gt | +41.9 pp (= normal − gt, reverse@L35 ≡ normal) | +4.48 m | +3.5 pp | +13.1 pp |

[계산한 결과]
- A− layer 대비는 모두 p ≤ 1.2e-4다.
- A+는 정보 있는 log가 9–10개뿐이라 p = 0.015–0.041이다.
- 원 결론 "embedding(직전 token identity)만으로 Normal−GT 간격의 약 68%가 재현되고, 깊은 layer일수록 나머지가 채워진다"는 cluster 기준에서도 유지된다(28.6/41.9 = 68%).

## 4. R9: window 1–4 cluster 통계 (A−, 365 단위 / 52 장면 / 16 log)

| 대비 | amplification [95% CI] | sign-flip p | 정보 log | cluster-t p | FDE5 (m) | FDE p |
|---|---|---|---|---|---|---|
| win1 − normal | −19.5 pp [−26.7, −11.3] | 9.2e-4 | 15 | 1.4e-4 | −2.59 [−3.15, −1.90] | 6.1e-5 |
| win2 − normal | −32.0 pp [−41.5, −20.5] | 2.4e-4 | 14 | 4.8e-5 | −3.71 | 3.1e-5 |
| win3 − normal | −37.6 pp [−49.7, −25.0] | 1.2e-4 | 14 | 3.3e-5 | −4.17 | 3.1e-5 |
| win4 − normal | −39.9 pp [−50.8, −28.3] | 1.2e-4 | 14 | 4.7e-6 | −4.36 | 3.1e-5 |
| gt_history − normal | −41.7 pp [−52.6, −29.7] | 1.2e-4 | 14 | 5.9e-6 | −4.51 | 3.1e-5 |
| win1 − gt_history | +22.2 pp [+13.9, +30.5] | 4.9e-4 | 12 | 4.5e-5 | +1.92 [+1.21, +2.43] | 1.2e-4 |
| win2 − gt_history | +9.7 pp [+4.9, +15.0] | 0.002 | 10 | 6.5e-4 | +0.81 [+0.49, +1.08] | 9.2e-5 |
| win3 − gt_history | +4.2 pp [+0.4, +6.5] | 0.12 | 4 | 0.013 | +0.34 [+0.09, +0.49] | 0.012 |
| win4 − gt_history | +1.8 pp [0.0, +3.0] | 0.50 | 2 | 0.062 | +0.16 [+0.01, +0.27] | 0.091 |
| delay2_len1 / delay3_len1 / delay4_len1 − normal | −26.9 / −29.8 / −30.1 pp | ≤ 4.9e-4 | 13–15 | – | −2.67 / −2.93 / −2.37 | ≤ 1.2e-4 |
| delay2_len2 / delay3_len2 − normal | −34.5 / −35.7 pp | 1.2e-4 | 14 | – | −3.59 / −3.26 | 3.1e-5 |

[계산한 결과]
- win3과 win4의 gt_history 대비는 정보 있는 log가 4개와 2개뿐이다. sign-flip p 하한이 각각 0.125와 0.5라 검정력이 거의 없다.
  - 사전 등록 동등성 margin이 없다.
  - 따라서 "win4 ≈ Full"은 "차이 미검출(sign-flip), 동등성 미확인"으로 쓴다.
  - 같은 대비의 cluster-t p는 0.013 / 0.062, FDE5 p는 0.012 / 0.091이다. 작은 잔여 차이가 있을 가능성을 배제하지 못한다.
- 원 unit-level p와의 대조: win1 − normal은 McNemar 2.3e-14에서 sign-flip 9.2e-4로, win2는 9.2e-30에서 2.4e-4로 바뀐다. 판정은 같고 p만 크게 약해진다.
- A+ (1,043 / 156 / 26): win1–4 − normal −2.3 / −3.0 / −2.7 / −3.3 pp (p = 0.016–0.039). FDE −0.75 ~ −1.14 m (p ≤ 6e-4).
- pooled: win1–4 − normal −6.6 / −10.2 / −11.4 / −12.4 pp (p ≤ 1.9e-4).

### 4.1 회복 비율과 critical window (집계 수준, 원 규칙 f ≥ 0.9 / 0.8 함께)

| window | unit 가중 f (원 방식) | 장면 가중 f |
|---|---|---|
| win1 | 47.1% | 46.7% |
| win2 | 76.1% | 76.6% |
| win3 | **89.7%** | **90.1%** |
| win4 | 96.1% | 95.8% |

[계산한 결과]
- f = (normal − win)/(normal − gt_history)이고 수준값에서 계산했다.
- unit 가중: normal 46.6%, gt_history 4.1%.
- 장면 가중: normal 45.1%, gt_history 3.4%.
- 판정:
  - **f ≥ 0.9 기준:** unit 가중에서 최소 w = 4다. 원 보고와 같고, 원 log bootstrap 분포는 w3 50% / w4 50%다. 장면 가중에서는 w = 3이다.
  - **f ≥ 0.8 기준:** 두 가중 모두 w = 3이다.
  - **FDE5 f ≥ 0.9 (원 보고):** w = 3(bootstrap 84%).
- 따라서 "90% 회복에 3–4 step"이라는 범위 진술은 유지된다. 한 정수로 고정하면 가중 방식에 민감하다.
- 이 critical-window bootstrap 분포는 원 분석 스크립트가 복구 raw에서 다시 계산한 것이다(`raw/exp9/TEMPORAL_FEEDBACK_WINDOW.md` §4). 원 값과 같다.

## 5. predeclared 항목 중 생성되지 않은 것 [확인한 사실]
- R9 대비와 R8 보조의 LOO. `r8r9_build_units.py`는 R8 primary에만 LOO를 만든다. 커밋된 스크립트로 만들 수 없어 계산하지 않았다.
- R9 대비에는 Holm family가 사전 등록되지 않았다. 위 p는 모두 비보정이다.

## 6. plan §11 주장 대응

| 주장 | 이번 결과 | 대응 |
|---|---|---|
| (기존 AutoVLA primary) 직전 token identity의 1-step feedback이 증폭을 만든다 | Reverse−Full이 cluster 기준 유의(Holm 1.8e-4), 3 primary 모두 Holm < 0.01 | 같은 16 log 안에서 log-cluster 통계로 유지된다. 실패 선택 집합 안의 결과다 |
| 길이는 중요하지만 위치 차이는 작음 → 짧은 교정의 효율 | window 1–4의 단조 회복(47 → 96%), 3–4 step에서 90% | 지지(P1-B와 함께). critical window를 정수 하나로 쓰지 않는다 |
| cluster/held-out에서 효과 약화 → 범위 축소 | cluster 통계에서 p는 약해졌지만 판정은 같다. held-out(새 log)은 이 작업의 범위가 아니다 | 새 log 일반화는 P1-C가 판단한다. 이 복구는 그 행의 증거가 아니다 |

[가설: 주장 대응.] 이 결과는 복구이므로 "재현 확인" 증거로 쓰지 않는다. 같은 스크립트·GPU·seed를 다시 실행해 bit 단위로 같은 값을 얻은 것이고, 독립 표본이 아니다.

## 7. 한계
1. 복구다. 원 수치를 본 뒤 같은 분석을 재실행했다. [확인한 사실]
2. 16 A− log는 13 drive다. drive 수준 군집은 쓰지 않았다. p는 sign-flip 하한(2/2^15 – 2/2^16)에 있다. [확인한 사실]
3. 실패 선택 집합(A− = 증폭 장면)과 GT 정의 t*를 상속했다. 일반 모집단 효과가 아니다. [확인한 사실]
4. reverse@emb는 Normal 행에서 embedding을 복사하는 정의다(P0-A F1). "자기 출력 피드백"을 직접 재현한 것과는 다르다. [확인한 사실]
5. window 대비 중 win3, win4 − gt_history는 정보 있는 log가 매우 적다. [계산한 결과]
6. exp-7 token 일치율의 비교 대상 파일이 원본과 같은지는 확인하지 않았다. [미확인]
7. §2.1의 Holm 갱신과 §4.1의 회복 비율은 산출물에 대한 산술이다. 새 지표는 만들지 않았다. [확인한 사실]

## 8. 파일
- `protocol.md`, `manifest_prerun.json`, `manifest.json`.
- `units.parquet`, `paired_effects.csv`, `loo_logs.csv`, `condition_summary.csv`, `comparison_with_original_summary.csv`.
- `raw/exp8/`, `raw/exp9/`: records, 원 분석 출력, figures.
- `logs/`: exp8.log, exp8_sanity.log, exp9.log, analyze_exp*.log, build_units.log, original_script_sha256.txt, raw_sha256.txt, done.
