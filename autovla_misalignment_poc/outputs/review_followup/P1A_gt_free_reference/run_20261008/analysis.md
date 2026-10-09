# P1-A 결과: GT 없는 reference와 좌표·외삽 대조 (AutoVLA)

작성: 2026-10-09.
- 사전 등록: `protocol.md`(commit b212637, 실행 전).
- 실행: 2026-10-08 RTX 5090, `run_all_main.sh` stage 1–3.
- 분석: CPU에서만 했다.
- 이 파일: 분석 agent 환경에서 보고서 파일 쓰기가 막혀, 상위 세션이 그대로 저장한다.

표기:
- **[확인한 사실]** = 파일·코드·trace에서 읽은 것.
- **[계산한 결과]** = 이번 실행 산출물, 또는 그 산출물에 대한 단순 산술.
- **[미확인]** = 확인하지 않았거나 할 수 없는 것.
- **[가설]** = 해석·추정.

수치 읽는 법:
- 기본은 장면 가중(primary).
- [ ] = log block bootstrap 95% CI(B = 10,000).
- p = log 요약값 sign-flip(16 log는 2^16 전수 열거).
- Holm = primary family(m = 3) 안 보정.
- 출처: `paired_effects.csv`, `condition_summary.csv`, `loo_logs.csv`, `deviation_curves.csv`, `trace_verification.json`, `units.parquet`(primary), `fixedpos/`(사전 등록 보조 arm).

## 0. 요약

- **primary 세 가설은 모두 사전 등록 방향으로 유의하다(Holm p = 9.2e-5, 각각).** [계산한 결과]
  - P1A-1 (C1 − U, suffix_loc): **+0.571 m/segment [+0.430, +0.677]**.
  - P1A-2 (OC − Geo, dev9_ref): **+4.37 m [+3.34, +5.13]**.
  - P1A-3 (OC_refR − OC, dev9_ref): **−3.50 m [−4.26, −2.60]**.
  - 다음 민감도에서 모두 부호와 유의성이 같다: unit 가중, log 균등 가중, leave-one-log-out, rank ≤ 100, 8-sector 장면.
- **GT 정보 없이도, 문맥 하나의 변화가 이후 local action을 바꾼다.** 그 결과 endpoint 편차가 기하 baseline(Geo)보다 훨씬 크다. [계산한 결과]
  - 출력 섭동 + reference 문맥(O1)은 Geo와 사실상 같다(+0.06 m).
  - 따라서 효과는 문맥 경로에서 나온다.
- **A−에서 OC는 constant-motion(CM) baseline과 구별되지 않는다.**
  - 보조 대비 −0.59 m [−1.46, +0.41], p = 0.21, 동등성 미확인. [계산한 결과]
  - A+에서는 OC가 CM보다 훨씬 작다(−4.69 m).
  - 따라서 A−에서 "섭동 방향을 계속 따라가는 운동 prior"보다 큰 특수 자기강화라고 주장할 근거는 이번 결과에 없다. [가설]
- **GT-free 교정(OC_refR)은 reference와의 편차를 줄이지만, GT 오차는 줄이지 않는다.**
  - A−에서 GT 증폭은 오히려 +19.7 pp [+8.4, +29.5] 늘고, FDE5도 +1.50 m 나빠진다(보조). [계산한 결과]
  - A− reference 자체가 GT 기준 실패 궤적이기 때문으로 보인다(U의 GT 증폭 67%). [가설]
- **검증**
  - 다음 점검은 2,861 단위 전부 100%다: prefix 잠금, decoder 원점, prefix pose, O1≡U suffix, OC≡C1 suffix.
  - **U = reference 비율은 0.889**다. protocol §7의 5% 보고 기준을 넘는다.
  - 원인은 208 장면 중 23개(11.1%)에서 B = 10 batch 수치 경로가 B = 1 reference와 갈라진 것이다.
  - primary 판정에는 영향이 없다(§3). [확인한 사실 + 계산한 결과]
- **plan §11 대응** [가설: 주장 대응]
  - "GT 없이 local suffix 변화·성장 차이 유지" 행을 지지한다: oracle 정보만으로 설명되지 않는 action-context 효과.
  - A−에서는 "기하/외삽 baseline과 유사" 행의 외삽(CM) 부분도 함께 해당한다.
  - 따라서 특수 자기강화 주장은 축소해야 한다.

## 1. 실행·분모·제외

| 항목 | 값 |
|---|---|
| reference | 208 장면, `reference/references.jsonl`, sha256 검사 통과(abort 없음) [확인한 사실] |
| primary 단위(family G) | A− 364 단위 / 52 장면 / 16 log; A+ 1,089 / 156 / 26; 계 1,453 [확인한 사실] |
| family E(보조, GT-anchored) | A− 365(E:original 52는 p = r이라 대비에서 제외 → 313), A+ 1,043(→ 887) [확인한 사실] |
| runtime 실패 | `raw/errors.jsonl`, `raw_fixedpos/errors.jsonl` 모두 0행 [확인한 사실] |
| 재시도·병합 | 없음. launcher log에 OOM 기록 없음. `*.log.attempt1`은 첫 시도 log 사본 [확인한 사실] |
| 조건별 분모 | 모든 G 조건이 A− 364 / 52 / 16. paired join missing 0, NaN drop 0(primary·G 보조 전부) [확인한 사실] |
| growth NaN(d0 < 0.02 m) | G 0건. E는 A− 11, A+ 27 단위(log2 growth 대비에서만 제외) [확인한 사실] |
| OOD flag(첫 step entropy 변화 > 1 nat) | 해당 조건 없음. 최대 평균 Δentropy 0.22 nat [계산한 결과] |
| 섭동 크기·개연성(G, A−) | achieved 0.101 ± 0.013 m(0.072–0.130); rank 중앙값 56(1–1,671); log p 평균 −11.9; Δheading 평균 −0.004 rad [계산한 결과] |

**cluster 주의:** [확인한 사실: 개수 / 미확인: 영향 크기]
- A−의 16 "log"는 13개 drive(같은 날짜·차량)에서 나왔다. A+의 26 log는 19 drive다.
- 이 protocol은 drive 수준 군집을 쓰지 않는다.
- drive 안에 상관이 있으면 독립 cluster는 16개보다 적다. 그러면 p와 CI가 낙관적일 수 있다.

## 2. Primary family (family G, A−, Holm m = 3)

| 대비 | 장면 가중 [95% CI] | 90% CI | sign-flip p | Holm p | cluster-t p | unit 가중 | log 균등 | LOO (16회) | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| P1A-1 C1 − U, suffix_loc (m/seg) | **+0.571 [+0.430, +0.677]** | [+0.452, +0.662] | 3.1e-5 | 9.2e-5 | 1.5e-7 | +0.575 [+0.432, +0.684] | +0.540 [+0.411, +0.675] | +0.521 ~ +0.601; CI 0 포함 0/16; 최대 p 6.1e-5 | H1 지지 |
| P1A-2 OC − Geo, dev9_ref (m) | **+4.37 [+3.34, +5.13]** | [+3.52, +5.01] | 3.1e-5 | 9.2e-5 | 7.1e-8 | +4.41 [+3.38, +5.18] | +4.19 [+3.28, +5.14] | +4.04 ~ +4.63; 0/16; 6.1e-5 | H2 지지 |
| P1A-3 OC_refR − OC, dev9_ref (m) | **−3.50 [−4.26, −2.60]** | [−4.12, −2.75] | 6.1e-5 | 9.2e-5 | 3.9e-7 | −3.57 [−4.34, −2.69] | −3.33 [−4.25, −2.44] | −3.68 ~ −3.22; 0/16; 1.2e-4 | H3 지지 |

[계산한 결과]
- 세 대비 모두 16 log가 정보 있는 log다. 그래서 sign-flip p 하한은 2/2^16 = 3.1e-5다. P1A-1과 P1A-2의 p는 이 하한이다.
- 결정 규칙(protocol §5): 두측 Holm p < 0.05이고 예측 방향이면 지지한다. 세 대비 모두 충족한다.
- 동등성(TOST)은 null을 주장할 때만 쓰기로 사전 등록했으므로 적용하지 않았다(`equivalent_TOST90` = False).
- 해석 보조:
  - C1의 suffix token mismatch는 U보다 +83.2 pp, local heading은 +0.031 rad 크다.
  - 0.10 m/segment 섭동 하나가 평균 0.57 m/segment의 suffix local 변화로 이어진다.
  - follow는 OC·C1 모두 1.14 d0다. 즉 이후 local action이 섭동 방향으로 평균 섭동 크기 이상 이동한다.

### 2.1 사전 등록 민감도 (primary 대비, 비보정, `P1A_sensitivity_*`)

| 민감도 | 단위 / 장면 / log | P1A-1 | P1A-2 | P1A-3 |
|---|---|---|---|---|
| rank(p \| prefix) ≤ 100 | 214 / 51 / 16 | +0.508 [+0.394, +0.585], p = 3.1e-5 | +3.84 [+2.96, +4.58], p = 3.1e-5 | −3.12 [−3.73, −2.40], p = 6.1e-5 |
| 8 sector 모두 있는 장면 | 168 / 21 / 10 | +0.536 [+0.384, +0.731], p = 0.0020 | +4.31 [+3.09, +5.72], p = 0.0020 | −3.70 [−5.00, −2.63], p = 0.0020 |

[계산한 결과]
- 개연성 높은 token만 써도 결론은 같다.
- 방향 지지가 완전한 장면만 써도 결론은 같다.
- 8-sector 집합의 p = 0.0020은 정보 있는 log 10개일 때의 하한 2/2^10이다.

## 3. 검증 점검 (`trace_verification.json`)

| 점검 | primary (2,861 단위) | fixed-position (2,900 단위) |
|---|---|---|
| output_prefix_locked | 1.000 | 1.000 |
| decoder_origin_zero | 1.000 | 1.000 |
| prefix_pose_err_zero | 1.000 | 1.000 |
| O1_suffix_equals_U | 1.000 | 1.000 |
| OC_suffix_equals_C1 | 1.000 | 1.000 |
| numpy rollout vs decoder 최대 오차 | 1.7e-5 m | 2.2e-5 m |
| prefix log-prob vs reference run 최대 차 | 1.5e-6 | 1.4e-6 |
| **U_equals_reference** | **0.889** | **0.857** |

[확인한 사실] protocol §7의 abort 조건 (a)–(d)는 하나도 발동하지 않았다.
- sha256 통과.
- 잠금·원점·pose 100%.
- cache assert 없음.
- 장면 실패 0.

### 3.1 U_equals_reference = 0.889의 의미

**무엇인가** [확인한 사실 + 계산한 결과]
- U는 이번 실행에서 B = 10 batch로 다시 생성한 무섭동 rollout이다.
- reference는 같은 seed key로 B = 1에서 미리 생성해 고정한 token열이다.
- 둘이 다른 단위는 11.1%다(G·E 모두).
- 이 불일치는 **장면 단위로 결정적이다.**
  - 208 장면 중 23개(11.1%)에서는 모든 단위가 U ≠ reference다.
  - 나머지 185 장면에서는 모든 단위가 U = reference다.
- 원인: T = 0.01·bf16에서 batch 크기에 따른 수치 경로 차이가 near-tie를 뒤집은 것으로 보인다. [가설]
- 크기:
  - U의 dev9_ref 장면 가중 평균은 A− 0.088 m다.
  - 불일치 단위만 보면 평균 0.90 m, 중앙값 0.30 m, 최대 4.09 m다.

**protocol 규칙** [확인한 사실]
- §7의 규칙: 5% 초과 시 비율을 보고한다. 분석은 고정 reference를 그대로 쓰고, U − reference를 noise floor로 보인다.
- abort 조건이 아니다. 그대로 따랐다.

**추론에 미치는 영향** [계산한 결과]
- **P1A-1(C1 − U)과 P1A-3(OC_refR − OC):** 두 조건 모두 같은 batch·seed로 생성했다. U ≠ reference가 양쪽에 같이 들어가 paired 차이에서 상쇄되는 구조다. 영향이 없다.
- **P1A-2(OC − Geo):** Geo는 reference suffix r[t_p+1:]를 쓰고, OC는 이번 batch에서 생성된다. 그래서 불일치 장면에서는 OC − Geo에 U − reference noise가 섞인다.
  - noise floor = 사전 등록 보조 대비 **O1 − Geo**(O1 suffix ≡ U suffix): A− **+0.063 m [0.000, +0.112]**, p = 0.25.
  - 같은 batch끼리 비교한 **OC − O1 = +4.31 m [+3.27, +5.08]**, p = 3.1e-5. primary 크기의 98.6%다.
- 결론: U ≠ reference는 primary 판정을 바꾸지 않는다. Geo 기반 대비의 절대 크기만 약 0.06 m 부풀린다.

**fixed-position arm에서는 더 크다** [계산한 결과]
- U ≠ reference: GF1 A− 29.9%, GF3 A− 13.3%.
- O1 − Geo: GF1 +0.73 m [+0.20, +1.27], GF3 +0.35 m [+0.03, +0.68].
- protocol §2.6이 예고한 대로다. t_p < t*이면 U가 stored HF-generate rollout과 다른 경로로 생성되기 때문이다.
- 이 arm에서는 OC − O1을 함께 읽어야 한다(§4.4).

**하지 않은 분석** [미확인]
- U = reference인 185 장면만으로 primary를 다시 계산하는 분석은 하지 않았다. 사전 등록되지 않았기 때문이다.
- 대신 사전 등록된 O1 − Geo와 OC − O1이 같은 질문에 답한다.

### 3.2 구조 동일성의 함의 [확인한 사실 + 가설]
- OC ≡ C1 suffix, O1 ≡ U suffix가 100%다. AutoVLA의 생성이 문맥에만 의존한다는 smoke 발견을 전 단위에서 확인한 것이다.
- 이 모델에서는 다음이 성립한다.
  - "출력 섭동 + reference 문맥"이 곧 기하 baseline이다. O1 dev9 0.596 m ≈ Geo 0.534 m이고, 차이는 noise floor 수준이다.
  - OC − C1은 같은 suffix에서 p와 r[t_p]를 합성한 차이다(dev9 +0.11 m).
- 따라서 endpoint 편차의 대부분은 **문맥이 바뀐 뒤 생성된 suffix**에서 나온다.

## 4. 사전 등록 보조 분석 (비보정, 탐색적)

### 4.1 family G, A− (`P1A_secondary_G_A-`)

| 대비 | 지표 | 장면 가중 [95% CI] | p |
|---|---|---|---|
| O1 − Geo (기대 0) | dev9_ref | +0.063 [0.000, +0.112] | 0.25 |
| **OC − CM** | dev9_ref | **−0.59 [−1.46, +0.41]** (90% [−1.30, +0.20]) | 0.21 |
| OC − O1 | dev9_ref | +4.31 [+3.27, +5.08] | 3.1e-5 |
| C1 − U | dev9_ref | +4.70 [+3.60, +5.48] | 3.1e-5 |
| C1 − U | suffix_yaw (rad) | +0.031 [+0.022, +0.039] | 3.1e-5 |
| C1 − U | suffix_mismatch | +0.832 [+0.782, +0.877] | 3.1e-5 |
| OC_ref1 − OC (one-shot) | dev9_ref | −2.27 [−3.07, −1.31] | 5e-4 |
| OC_refH − OC | dev9_ref | −3.84 [−4.60, −2.90] | 3.1e-5 |
| O_refH − OC | dev9_ref | −4.37 [−5.14, −3.34] | 3.1e-5 |
| OC_gtR − OC (GT arm) | dev9_ref | +2.07 [+1.06, +3.07] | 0.003 |
| OC_gtR − OC_refR | dev9_ref | +5.56 [+4.10, +6.85] | 1e-4 |
| severe_ref3 OC − Geo | rate | +49.8 pp [+35.7, +62.2] | 1e-4 |
| severe_ref3 OC_refR − OC | rate | −38.3 pp [−48.5, −27.6] | 1e-4 |
| ref_amp3 OC − Geo | rate | +28.2 pp [+19.2, +35.5] | 5e-4 |
| GT amplification OC − U | rate | −23.1 pp [−35.8, −8.1] | 0.002 |
| **GT amplification OC_refR − OC** | rate | **+19.7 pp [+8.4, +29.5]** | 0.002 |
| GT amplification OC_gtR − OC | rate | −38.3 pp [−48.5, −27.4] | 1e-4 |
| **GT FDE5 OC_refR − OC** | m | **+1.50 [+0.73, +2.32]** | 0.002 |
| log2 growth OC − Geo | log2 | +2.83 [+2.45, +3.19] | 3.1e-5 |
| log2 growth OC − CM | log2 | −0.62 [−1.01, −0.29] | 0.008 |

수준(장면 가중, A−) [계산한 결과]
- dev9_ref: U 0.09, Geo 0.53, O_refH 0.54, O1 0.60, OC_refH 1.07, OC_refR 1.40, OC_ref1 2.63, C1 4.79, OC 4.90, CM 5.49, OC_gtR 6.97, OC_gtH 7.43 m.
- GT 증폭률: U 67%, OC 44%, OC_refR 64%, OC_gtR 6%.

해석 [가설]
- **외삽 baseline**
  - A−에서 OC endpoint 편차는 섭동 local motion을 끝까지 반복한 CM과 통계적으로 구별되지 않는다.
  - 성장률(log2)로는 OC가 CM보다 작다(−0.62).
  - A+에서는 OC가 CM보다 4.7 m 작다(§4.2).
  - 즉 A− 장면에서 모델은 바뀐 문맥을 "등속 외삽에 가까운 정도"로 따라간다. 이것이 운동 prior를 넘는 증폭인지는 이 설계로 판별되지 않는다.
  - OC − CM에는 사전 등록 동등성 margin이 없다. 90% CI도 ±0.25 m보다 넓다. 따라서 "OC ≈ CM"이 아니라 "차이 미검출, 동등성 미확인"으로 쓴다.
- **reference 편차 ≠ GT 오차**
  - GT-free 반복 교정은 모델 자신의 무섭동 rollout으로 되돌리는 교정이다.
  - A−에서는 그 rollout이 GT 기준 실패이므로 GT 오차가 커진다.
  - 반대로 섭동(OC)은 A− GT 증폭을 −23 pp 줄인다. 섭동이 실패 궤적에서 벗어나게 한 경우다.
  - 따라서 P1A-3은 "자기 일관성 교정이 문맥 효과를 되돌린다"는 증거다. "GT 없이 실패를 고친다"는 증거가 아니다.
- **one-shot vs 반복**(별개 arm)
  - dev9_ref 감소: one-shot −2.27 < 반복 sliding −3.50 < 전체 history −3.84.
  - O_refH(출력만 섭동, 문맥 전부 reference)는 Geo 수준(0.54 m)으로 돌아간다.
- **편차 곡선**(`figures/deviation_curves_G_Aminus.png`)
  - C1·OC는 t_p 직후부터 Geo와 갈라진다. offset 2에서 0.61–0.71 m vs 0.20 m다.
  - offset 9에서는 5.9–6.0 m까지 커진다. CM은 7.5 m다.
  - OC_refR은 offset 2 이후 기울기가 꺾여 1.58 m에서 끝난다.

### 4.2 A+와 pooled (family G)

| 대비 | A+ (1,089 / 156 / 26) | pooled (1,453 / 208 / 26) |
|---|---|---|
| P1A-1 C1 − U | +0.214 [+0.179, +0.248] | +0.303 [+0.249, +0.341] |
| P1A-2 OC − Geo | +1.64 [+1.30, +1.91] | +2.32 [+1.81, +2.66] |
| P1A-3 OC_refR − OC | −1.12 [−1.35, −0.85] | −1.72 [−2.02, −1.27] |
| OC − CM | **−4.69 [−5.48, −3.75]** | −3.66 [−4.35, −2.96] |
| GT amplification OC_refR − OC | −2.6 pp [−3.8, −1.4] | +3.0 pp [−0.1, +5.2], p = 0.12 |
| GT FDE5 OC_refR − OC | −0.40 m [−0.61, −0.25] | +0.07 [−0.23, +0.30], p = 0.61 |

[계산한 결과]
- A+에서도 세 primary 대비의 방향은 같다. 크기는 A−의 1/3–1/2이다.
- A+의 sign-flip은 26 log Monte Carlo이고, 표시 p는 모두 < 1e-4다.
- A+에서는 GT-free 교정이 GT 오차도 줄인다(−0.40 m). A−와 반대다.
- A+ reference가 대체로 성공 궤적이기 때문으로 보인다. [가설]

### 4.3 family E (GT-anchored 등거리 섭동, 연결용)
- A− primary 세 방향이 같고, 크기는 G보다 크다. 모두 p ≤ 3.1e-5. [계산한 결과]
  - C1 − U +0.742 [+0.567, +0.909].
  - OC − Geo +5.62 [+4.30, +6.81].
  - OC_refR − OC −4.45 [−5.49, −3.35].
- OC − CM: A− −0.72 [−1.85, +0.63], p = 0.30.
- E 섭동은 GT 대비 거리로 골랐으므로 GT-free가 아니다. 연결용으로만 쓴다.

### 4.4 GT-free 고정 위치 arm (t_p ∈ {1, 3}, 사전 등록 보조, `fixedpos/`)

| 대비 (A−, 52 장면 / 16 log) | GF1 (374 단위) | GF3 (362 단위) |
|---|---|---|
| P1A-1 C1 − U | +0.387 [+0.322, +0.493], p ≤ 1e-4 | +0.422 [+0.324, +0.579], p ≤ 1e-4 |
| P1A-2 OC − Geo | +3.89 [+3.07, +4.76] | +2.89 [+2.02, +3.73] |
| P1A-3 OC_refR − OC | −3.00 [−3.87, −2.16] | −2.16 [−2.88, −1.40] |
| O1 − Geo (noise floor) | +0.73 [+0.20, +1.27] | +0.35 [+0.03, +0.68] |
| OC − O1 (noise 무관) | +3.16 [+2.53, +3.95] | +2.54 [+1.90, +3.42] |
| OC − CM | −0.60 [−1.33, +0.32], p = 0.19 | +0.11 [−0.75, +1.09], p = 0.83 |
| GT amp OC_refR − OC | +13.7 pp [−1.2, +27.2], p = 0.13 | +11.5 pp [−2.8, +23.7], p = 0.26 |

[계산한 결과]
- 위치를 GT 없이 고정해도 세 primary 방향과 유의성(비보정)이 유지된다.
- GF1에서는 OC − Geo의 약 0.7 m가 noise다. noise와 무관한 OC − O1도 크다.
- A+에서도 같은 방향이다(예: GF1 A+ P1A-2 +1.58 [+1.25, +1.84], GF3 A+ +1.07 [+0.80, +1.31]).
- A−에서 OC − CM을 구별하지 못하는 것도 같다.
- `fixedpos/loo_logs.csv`는 비어 있다. primary family가 아니어서 스크립트가 LOO를 만들지 않는다. [확인한 사실]

### 4.5 진단 (검정 없음)
- 섭동 균형은 §1에 있다.
- G A−에서 natural-prediction 빈도 중앙값은 13회(0–238)다.
- 개연성이 낮은 token도 섞여 있지만, rank ≤ 100 민감도에서 결론은 같다. [계산한 결과]

## 5. 가설별 판정

| 가설 | 사전 결정 규칙 | 결과 | 판정 |
|---|---|---|---|
| H1 문맥만의 효과 | P1A-1 > 0, Holm < 0.05 | +0.571 m/seg, Holm 9.2e-5 | **지지** [계산한 결과] |
| H2 기하 이상 | P1A-2 > 0, Holm < 0.05 | +4.37 m, Holm 9.2e-5 | **지지** [계산한 결과] |
| H3 GT-free 교정 | P1A-3 < 0, Holm < 0.05 | −3.50 m, Holm 9.2e-5 | **지지**(reference 편차 기준). GT 오차 기준으로는 A−에서 악화(보조) [계산한 결과] |
| 해석 규칙(OC ≈ Geo 또는 OC ≈ CM이면 축소) | — | OC ≫ Geo. A−에서 OC − CM 미검출(동등성 미확인). A+에서는 OC ≪ CM | **부분 해당: 자기강화 주장 축소** [가설] |

## 6. plan §11 주장 대응

| §11 행 | 이번 결과 | 대응 |
|---|---|---|
| GT 없이 local suffix 변화·성장 차이 유지 → oracle 정보만으로 설명되지 않는 action-context 효과 | P1A-1·2가 GT-free reference·섭동·고정 위치에서 유지 | **지지.** 단 위치 t*와 A−/A+ 층은 GT에서 왔다 |
| 기하/외삽 baseline과 유사 → 운동 prior·기하 일관성 중심 해석, 특수 자기강화 축소 | Geo와는 다르다. A−에서 CM과 구별되지 않는다 | **A−에 해당.** "문맥이 섭동 방향 운동을 지속시킨다(운동 prior와 양립)"로 쓴다. "등속 외삽보다 큰 자기증폭"은 쓰지 않는다 |
| (P1A-3) | GT-free 교정은 reference로 되돌리지만 A− GT 실패는 고치지 않는다 | "실패 교정"이 아니라 "문맥 일관성 회복"으로 서술한다 |

[가설: 주장 문장 선택은 해석이다.]

## 7. 한계
1. t_p = t*와 A−/A+ 층은 GT 정의를 상속했다. reference, family G 섭동, 고정 위치 arm은 GT-free다. 그러나 장면 선택과 primary 위치는 GT-informed다. [확인한 사실]
2. 16 log는 실제로 13 drive다. drive 수준 군집은 사전 등록되지 않았다. sign-flip p 하한은 3.1e-5이고, p들은 그 하한에 있다. [확인한 사실]
3. 결과는 같은 16 log 안의 것이다. 새 log 일반화는 P1-C의 일이다. [미확인]
4. U ≠ reference(장면의 11%)는 B = 10 대 B = 1 수치 경로 차이다. Geo 기반 대비에 약 0.06 m의 noise floor를 더한다. 고정 위치 arm에서는 최대 0.73 m다. [계산한 결과]
5. family G token은 모델 기준 개연성이 낮을 수 있다(rank 중앙값 56, 최대 1,671). 개연성을 맞춘 설계는 별도 사전 등록이 필요하다. [확인한 사실]
6. reference 대비 편차는 GT 오차가 아니다. A−에서 두 지표는 반대로 움직인다. [계산한 결과]
7. CM은 섭동 token을 반복하는 강한 외삽이다. OC ≈ CM이 곧 등속 prior라는 뜻은 아니다. 이를 가르려면 다른 외삽 baseline이 필요하다. [가설]
8. seed 1개, T = 0.01로 사실상 결정적이다. 변동은 장면·섭동 사이에서만 나온다. [확인한 사실]
9. 이 문서는 결과를 본 뒤 작성했다. §3.1의 장면 단위 분해와 §1의 균형 요약은 사전 등록 산출물에 대한 기술 통계이고, 새 검정은 아니다. [확인한 사실]

## 8. 파일
- 사전 등록: `protocol.md`, `manifest_prerun.json`.
- 사후: `manifest.json`(HEAD b212637, script sha256).
- primary: `units.parquet`, `condition_summary.csv`, `paired_effects.csv`, `loo_logs.csv`, `deviation_curves.csv`, `trace_verification.json`, `intervention_trace.jsonl`. raw는 `raw/`, `reference/`.
- 고정 위치 arm: `fixedpos/`. raw는 `raw_fixedpos/`.
- 그림: `figures/deviation_curves_G_Aminus.png`(A− family G, 조건별 reference 대비 편차; `deviation_curves.csv`에서 작성).
