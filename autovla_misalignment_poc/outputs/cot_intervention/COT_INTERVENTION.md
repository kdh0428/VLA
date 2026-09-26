# AutoVLA — CoT가 action 생성의 인과적 원인인가?

GPU1 (NVIDIA GeForce RTX 5090), 장면 159개 × 조건 7개. 이미지·ego state·prompt·Scene/Critical Object 텍스트는 고정하고 추론/결정 텍스트만 바꾼 뒤, action token은 원래 decoding 설정(do_sample, T=0.01)으로 자유 생성했습니다. 기존 결과는 수정하지 않았습니다.

## 1. 결론

**판정: weakly coupled / potentially post-hoc**

**선언한 방향**은 action을 거의 움직이지 못하고, **CoT 텍스트의 형태**(모델이 스스로 쓴 긴 추론이냐, 짧게 재작성한 추론이냐)는 방향과 무관하게 action을 크게 흔듭니다.

방향 효과 — 같은 형식으로 재작성하고 선언 방향만 반대로 한 두 조건의 쌍대 차이:

- 실행 행동이 반대 방향으로 간 비율 (반대 CoT − 교정 CoT): **+6.4%p [+2.7, +9.7]**; 결정 문구만 바꾼 경우 +1.9%p [-2.6, +7.6]
- A+ (교정 CoT − 반대 CoT): +7.6%p [+1.6, +13.0]; 결정 문구만 +2.5%p [-1.7, +5.8]
- 종료 속도가 반대 방향으로 이동한 양: +0.33 m/s [+0.10, +0.52]
- GT action 10개의 log-likelihood (교정 − 반대, teacher-forced): +0.39 nats [-0.24, +0.92]; 결정 문구만 +0.29 nats [+0.07, +0.53]
- 첫 action token 변화율의 재작성 대조군 대비 초과분: +0.6%p [-4.9, +6.7]

재작성 효과 — 방향과 무관:

- 결정은 그대로 두고 추론만 템플릿으로 바꾼 `template_original`의 A+ 변화: +17.0%p [+10.3, +26.5], 첫 action token 변화 22.0%
- P+R−A−에서 교정 CoT의 A+ 회복은 `original` 대비 +32.8%p이지만, 같은 형식의 재작성 대조군 대비로는 +10.9%p입니다. 회복의 대부분은 방향이 아니라 재작성 자체에서 옵니다.

판정 기준: 방향 효과(반대 방향 이동 비율 차이) ≥ +50%p → causally coupled; < +15%p이면서 첫 토큰 초과 변화 < 15%p → weakly coupled; 그 외 partially coupled.

**기준선 변경에 대한 공개**: 실행 전에는 `original` 대비 변화로 판정하도록 정했습니다. 그러나 대상 장면이 저장된 forced run의 *실패*로 선택되었기 때문에, 근소한 차이의 action token이 어떤 프리픽스 변화에도 다시 뽑히면서 방향과 무관하게 `original`에서 멀어집니다(평균으로의 회귀). 118/159 장면을 본 뒤 이를 확인하고, 형식을 고정한 반대-교정 쌍대 비교로 바꿨습니다. 성공으로 선택된 대조군(P+R+A+)은 모든 조건에서 A+가 유지되어 이 설명과 일치합니다. 참고로 원래 기준의 수치는 반대 방향 이동 -4.8%p, 결정 문구만 -2.5%p입니다(음수는 이 회귀 효과 때문).

## 2. Natural fast-thinking vs forced CoT (분리 보고)

같은 장면에서 모델 고유의 두 모드를 그대로 비교한 것입니다(개입 없음).

| 대상 | n | A+ natural (CoT 없음) | A+ forced CoT | 첫 토큰 동일 | 종방향 클래스 동일 | 궤적 차이 L2 (m) |
|---|---:|---:|---:|---:|---:|---:|
| P+R-A- | 64 | 73.4% | 18.8% | 71.9% | 46.9% | 2.03 |
| P+R+A- | 31 | 87.1% | 19.4% | 64.5% | 38.7% | 2.20 |
| P+R+A+ (control) | 64 | 100.0% | 100.0% | 90.6% | 90.6% | 0.39 |
| ALL | 159 | 86.8% | 51.6% | 78.0% | 62.9% | 1.40 |

대상 샘플은 forced-CoT 라벨로 골랐으므로 natural 모드의 A+가 높게 나오는 것은 선택 효과를 포함합니다.

## 3. Forced-CoT 개입 결과 (전체 장면)

변화율은 모두 같은 장면의 `original` 대비. [ ]는 log 단위 cluster bootstrap 95% CI.

| 조건 | 설명 | 첫 토큰 변화 | step-0 argmax 변화 | A+ (5 s) | 선언 행동을 따름 | 반대 방향으로 이동 | 궤적 변화 (m) | ADE / FDE 5 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `original` | 모델 고유 강제 CoT | 0.0% | 0.0% | 51.6% [42, 62] | 37.1% | 23.9% [18, 31] | 0.00 | 1.72 / 4.59 |
| `template_original` | 추론 재작성, 결정은 원래대로 (재작성 대조군) | 22.0% | 22.0% | 68.6% [61, 80] | 42.1% | 14.5% [9, 21] | 1.12 | 1.31 / 3.49 |
| `corrected_decision` | 결정 문구만 GT로 교체 | 3.8% | 4.4% | 57.9% [50, 69] | 41.5% | 19.5% [13, 27] | 0.48 | 1.60 / 4.28 |
| `corrected_full` | 추론+결정을 GT에 맞게 재작성 | 22.6% | 22.0% | 74.2% [67, 84] | 53.5% | 13.8% [7, 21] | 1.16 | 1.26 / 3.34 |
| `counter_decision` | 결정 문구만 반대 행동으로 교체 | 6.3% | 6.9% | 55.3% [46, 68] | 8.8% | 21.4% [16, 27] | 0.71 | 1.65 / 4.40 |
| `counter_full` | 추론+결정을 반대 행동으로 재작성 | 22.9% | 22.9% | 67.5% [59, 79] | 7.6% | 19.1% [11, 26] | 1.07 | 1.57 / 4.21 |

### Teacher-forced 분포 변화 (compounding 제거)

모든 조건을 **같은 action token 열**로 채점했습니다. dGT logp = GT action 10개의 log-likelihood 합의 `original` 대비 변화, JS = original이 생성한 token을 강제했을 때 step별 action 분포의 JS divergence 평균.

| 조건 | ΔGT logp (10 step 합) | ΔGT logp step 0 | ΔGT margin step 0 | JS 평균 | JS step 0 | 종료 속도 변화: 반대 방향 (m/s) | 종료 속도 변화: GT 방향 (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original` | +0.00 [+0.00, +0.00] | 0.000 | 0.000 | 0.0000 | 0.0000 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| `template_original` | +0.84 [+0.30, +1.48] | 0.112 | 0.055 | 0.0584 | 0.0956 | -0.25 [-0.54, +0.11] | +0.62 [+0.42, +0.82] |
| `corrected_decision` | +0.10 [-0.01, +0.24] | 0.015 | -0.050 | 0.0166 | 0.0068 | -0.11 [-0.32, +0.05] | +0.19 [+0.05, +0.41] |
| `corrected_full` | +0.70 [+0.26, +1.19] | 0.126 | 0.132 | 0.0632 | 0.1059 | -0.23 [-0.50, +0.02] | +0.66 [+0.46, +0.91] |
| `counter_decision` | -0.19 [-0.40, +0.06] | 0.004 | 0.031 | 0.0266 | 0.0134 | +0.02 [-0.08, +0.16] | +0.27 [+0.14, +0.42] |
| `counter_full` | +0.36 [-0.32, +1.18] | 0.044 | -0.240 | 0.0654 | 0.0958 | +0.08 [-0.24, +0.30] | +0.51 [+0.22, +0.86] |

## 4. 방향 효과 vs 재작성 효과 (장면별 쌍대 차이)

값 = 뒤 조건 − 앞 조건의 장면별 차이 평균, [ ] log 단위 cluster bootstrap 95% CI. 확률 지표는 비율 차이(0.10 = 10%p).

| 대조 | P+R-A- | P+R+A- | P+R+A+ (control) | ALL |
|---|---:|---:|---:|---:|
| towards_cf: counter_full - corrected_full | +0.113 [+0.04, +0.17] | +0.097 [+0.00, +0.19] | +0.000 [-0.05, +0.04] | +0.064 [+0.03, +0.10] |
| towards_cf: counter_decision - corrected_decision | +0.031 [-0.09, +0.20] | +0.000 [+0.00, +0.00] | +0.016 [+0.00, +0.05] | +0.019 [-0.03, +0.08] |
| A5: corrected_full - counter_full | +0.161 [+0.05, +0.27] | +0.065 [-0.05, +0.20] | +0.000 [-0.05, +0.04] | +0.076 [+0.02, +0.13] |
| A5: corrected_decision - counter_decision | +0.078 [-0.01, +0.24] | -0.097 [-0.27, +0.03] | +0.031 [+0.00, +0.07] | +0.025 [-0.02, +0.06] |
| dv_toward_cf (m/s): counter_full - corrected_full | +0.697 [+0.26, +1.13] | +0.305 [-0.16, +0.67] | -0.013 [-0.18, +0.16] | +0.330 [+0.10, +0.52] |
| dv_toward_cf (m/s): counter_decision - corrected_decision | +0.211 [-0.26, +0.95] | +0.275 [+0.04, +0.66] | -0.021 [-0.10, +0.05] | +0.130 [-0.08, +0.42] |
| GT logp 10 steps: corrected_full - counter_full | +0.343 [-1.01, +1.14] | +1.243 [+0.32, +2.14] | +0.014 [-0.41, +0.34] | +0.387 [-0.24, +0.92] |
| GT logp 10 steps: corrected_decision - counter_decision | +0.906 [+0.56, +1.36] | -0.549 [-1.07, +0.02] | +0.078 [-0.22, +0.42] | +0.289 [+0.07, +0.53] |
| GT logp step0: corrected_full - counter_full | +0.018 [-0.59, +0.41] | +0.354 [-0.21, +0.80] | +0.018 [-0.19, +0.20] | +0.084 [-0.17, +0.29] |
| argmax0 changed: counter_full - template_original | -0.016 [-0.11, +0.06] | +0.032 [-0.06, +0.17] | +0.016 [-0.06, +0.10] | +0.006 [-0.05, +0.07] |
| A5: template_original - original (rewrite only) | +0.219 [+0.08, +0.44] | +0.452 [+0.29, +0.67] | -0.016 [-0.05, +0.00] | +0.170 [+0.10, +0.26] |
| A5: corrected_full - template_original | +0.109 [+0.00, +0.19] | +0.065 [+0.00, +0.14] | +0.000 [+0.00, +0.00] | +0.057 [+0.01, +0.10] |
| towards_cf: counter_full - template_original | +0.065 [-0.05, +0.14] | +0.161 [+0.04, +0.26] | +0.000 [-0.05, +0.04] | +0.057 [+0.00, +0.10] |

## P+R-A-: 틀린 CoT를 교정하면 A−가 A+로 복구되는가?

n = 64

| 조건 | A+ (5 s) | A+ (3 s) | 선언 행동을 따름 | 반대 방향으로 이동 | step-0 argmax 변화 | ΔGT logp | ADE 5 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original` | 18.8% [7, 32] | 62.5% | 18.8% | 43.8% | 0.0% | 0.00 | 2.63 |
| `template_original` | 40.6% [30, 56] | 73.4% | 18.8% | 28.1% | 23.4% | 1.07 | 2.22 |
| `corrected_decision` | 29.7% [19, 45] | 68.8% | 21.9% | 34.4% | 9.4% | 0.30 | 2.46 |
| `corrected_full` | 51.6% [42, 65] | 76.6% | 40.6% | 23.4% | 28.1% | 1.03 | 1.93 |
| `counter_decision` | 21.9% [13, 33] | 65.6% | 15.6% | 37.5% | 10.9% | -0.61 | 2.75 |
| `counter_full` | 37.1% [26, 56] | 66.1% | 14.5% | 32.3% | 22.6% | 0.81 | 2.60 |

쌍대 비교 (McNemar, gain = original에서 실패→조건에서 성공):

- A5 corrected_full vs original: gain 23, loss 2, p = 1.94e-05 (n = 64)
- A5 corrected_decision vs original: gain 14, loss 7, p = 0.189 (n = 64)
- A5 template_original vs original: gain 18, loss 4, p = 0.00434 (n = 64)
- towards_cf counter_full vs original: gain 6, loss 12, p = 0.238 (n = 62)
- towards_cf counter_decision vs original: gain 3, loss 7, p = 0.344 (n = 64)
- towards_cf counter_full vs template_original: gain 7, loss 3, p = 0.344 (n = 62)
- A5 corrected_full vs counter_full: gain 15, loss 5, p = 0.0414 (n = 62)
- A5 corrected_decision vs counter_decision: gain 13, loss 8, p = 0.383 (n = 64)
- towards_cf counter_full vs corrected_full: gain 8, loss 1, p = 0.0391 (n = 62)
- towards_cf counter_decision vs corrected_decision: gain 9, loss 7, p = 0.804 (n = 64)

## P+R+A-: 이미 올바른 CoT를 반대로 바꾸면 action도 따라 바뀌는가?

n = 31

| 조건 | A+ (5 s) | A+ (3 s) | 선언 행동을 따름 | 반대 방향으로 이동 | step-0 argmax 변화 | ΔGT logp | ADE 5 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original` | 19.4% [0, 41] | 77.4% | 16.1% | 29.0% | 0.0% | 0.00 | 2.39 |
| `template_original` | 64.5% [50, 81] | 90.3% | 58.1% | 9.7% | 25.8% | 1.21 | 1.20 |
| `corrected_decision` | 29.0% [10, 47] | 80.6% | 22.6% | 25.8% | 3.2% | -0.16 | 2.20 |
| `corrected_full` | 71.0% [57, 85] | 87.1% | 51.6% | 16.1% | 25.8% | 1.23 | 1.41 |
| `counter_decision` | 38.7% [23, 54] | 80.6% | 9.7% | 25.8% | 9.7% | 0.39 | 1.83 |
| `counter_full` | 64.5% [48, 79] | 80.6% | 6.5% | 25.8% | 29.0% | -0.01 | 1.65 |

쌍대 비교 (McNemar, gain = original에서 실패→조건에서 성공):

- A5 corrected_full vs original: gain 16, loss 0, p = 3.05e-05 (n = 31)
- A5 corrected_decision vs original: gain 3, loss 0, p = 0.25 (n = 31)
- A5 template_original vs original: gain 15, loss 1, p = 0.000519 (n = 31)
- towards_cf counter_full vs original: gain 2, loss 3, p = 1 (n = 31)
- towards_cf counter_decision vs original: gain 0, loss 1, p = 1 (n = 31)
- towards_cf counter_full vs template_original: gain 5, loss 0, p = 0.0625 (n = 31)
- A5 corrected_full vs counter_full: gain 4, loss 2, p = 0.688 (n = 31)
- A5 corrected_decision vs counter_decision: gain 1, loss 4, p = 0.375 (n = 31)
- towards_cf counter_full vs corrected_full: gain 3, loss 0, p = 0.25 (n = 31)
- towards_cf counter_decision vs corrected_decision: gain 0, loss 0, p = 1 (n = 31)

## P+R+A+ (control): 행동이 맞았던 장면에서도 반대 CoT를 따라가는가? (대조군)

n = 64

| 조건 | A+ (5 s) | A+ (3 s) | 선언 행동을 따름 | 반대 방향으로 이동 | step-0 argmax 변화 | ΔGT logp | ADE 5 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original` | 100.0% [100, 100] | 100.0% | 65.6% | 1.6% | 0.0% | 0.00 | 0.49 |
| `template_original` | 98.4% [95, 100] | 100.0% | 57.8% | 3.1% | 18.8% | 0.44 | 0.46 |
| `corrected_decision` | 100.0% [100, 100] | 100.0% | 70.3% | 1.6% | 0.0% | 0.03 | 0.45 |
| `corrected_full` | 98.4% [95, 100] | 100.0% | 67.2% | 3.1% | 14.1% | 0.11 | 0.53 |
| `counter_decision` | 96.9% [93, 100] | 100.0% | 1.6% | 3.1% | 1.6% | -0.04 | 0.46 |
| `counter_full` | 98.4% [96, 100] | 96.9% | 1.6% | 3.1% | 20.3% | 0.10 | 0.53 |

쌍대 비교 (McNemar, gain = original에서 실패→조건에서 성공):

- A5 corrected_full vs original: gain 0, loss 1, p = 1 (n = 64)
- A5 corrected_decision vs original: gain 0, loss 0, p = 1 (n = 64)
- A5 template_original vs original: gain 0, loss 1, p = 1 (n = 64)
- towards_cf counter_full vs original: gain 1, loss 0, p = 1 (n = 64)
- towards_cf counter_decision vs original: gain 1, loss 0, p = 1 (n = 64)
- towards_cf counter_full vs template_original: gain 1, loss 1, p = 1 (n = 64)
- A5 corrected_full vs counter_full: gain 1, loss 1, p = 1 (n = 64)
- A5 corrected_decision vs counter_decision: gain 2, loss 0, p = 0.5 (n = 64)
- towards_cf counter_full vs corrected_full: gain 1, loss 1, p = 1 (n = 64)
- towards_cf counter_decision vs corrected_decision: gain 1, loss 0, p = 1 (n = 64)

## 7. 실험 타당성 점검

- 편집 성공률: natural_nocot 100%, original 100%, template_original 100%, corrected_decision 100%, corrected_full 100%, counter_decision 100%, counter_full 100%
- 편집 후 CoT에서 다시 파싱한 선언 행동이 목표와 일치: corrected_full 100%, corrected_decision 100%, counter_full 100%, counter_decision 100%
- 저장된 run 재현(10개 action token 완전 일치): forced 54%, natural 74%. CoT 전체를 한 번에 prefill하는 것과 원래의 증분 생성은 수치가 조금 달라 후반 토큰이 갈라지므로, 모든 비교는 저장 결과가 아니라 **같은 harness에서 다시 돌린 `original`** 기준입니다.
- `template_original`은 추론을 재작성하되 결정은 그대로 두는 대조군입니다. 이 조건의 변화량이 '텍스트를 바꾸기만 해도 생기는' 바닥값입니다.

## 8. 한계

- 교정/반대 CoT는 템플릿 문장입니다. 원래 CoT와 문체가 달라 분포 밖(OOD) 입력일 수 있고, `*_decision` 조건은 그 영향을 줄이기 위한 최소 편집입니다.
- 반대 CoT는 장면 설명과 모순될 수 있습니다(예: 적색등 설명 뒤 가속 결정). 요청하신 'plausible reasoning'은 일반론 수준입니다.
- 표본은 P+R−A− / P+R+A−가 작습니다. CI를 함께 보십시오.
- Natural 모드는 CoT가 없으므로 CoT 개입 실험 자체가 불가능합니다. natural 결과는 forced 결과와 섞지 않았습니다.

## 9. 파일

`/root/VLA/autovla_misalignment_poc/outputs/cot_intervention/`: `records.jsonl`(조건별 프리픽스·생성 토큰·궤적), `logits/`(조건×10 step×2048 자유 생성 logits), `teacher_forced/`(강제 채점), `rows.jsonl`(장면×조건 지표), `summary.json`. 스크립트: `scripts/cot_intervention.py`, `scripts/cot_teacher_forced.py`, `scripts/analyze_cot_intervention.py`, `scripts/make_cot_report.py`.
