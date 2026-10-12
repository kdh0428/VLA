# CLAIM_EVIDENCE_MATRIX (2026-10-13)

## 읽는 법
- **독립 표본 수:**
  - AutoVLA는 drive 수다(A− 13 drive, P1-C eval 24 drive).
  - SpatialVLA는 고유 초기 상태 수다. coke는 seed마다 고유하고, move_near는 물리 배치 10개 / 지시문 포함 상태 29–120개다.
- **CI와 p:**
  - CI는 cluster 기준이다.
  - p는 drive 또는 상태 수준 exact sign-flip이다. 13 drive일 때 가능한 최소 p는 2.4e-4다.
- **근거 위치:** `outputs/review_followup2/`의 A/B/C 결과, 그리고 `outputs/review_followup/`의 P0–P1이다.
- **판정 표기:** 유지 = 그대로 쓸 수 있음 / 축소 = 범위나 강도를 줄여서 씀 / 삭제 = 쓰지 않음.

| 주장 | 근거 실험 | 독립 표본 | 효과·CI | 제한 | 판정 | 허용되는 원고 문장 |
|---|---|---|---|---|---|---|
| **직전 action 문맥이 이후 생성을 인과적으로 바꾼다 (Recent−Normal, Reverse−Full)** | 실험 7, R8R9(복구), P1-C(새 drive, 사전 등록) | PoC 13 drive / P1-C 24 drive (drive 분리, day-vehicle 일부 공유) | PoC RN −39.8%p [−49.0, −29.9], RV +28.6%p [+20.9, +36.1] (p = 2.4e-4). P1-C RN F −34.2%p, RV F +26.8%p, P −4.1 / +3.7%p (재현) | Recent-GT는 **매 step 직전 위치를 GT로 바꾸는 반복 개입**(단위당 6.8회, 실효 4.3회). Reverse는 AutoVLA에서 별도 Normal 행의 embedding을 삽입하는 개입. 증폭 = A− ∧ FDE5 > 3 m | **유지** | "Repeatedly replacing the most recent action context with ground truth reduced amplification by 40 pp (13 drives; replicated on 24 new drives), and re-inserting the model's own previous action restored it." |
| **"직전 token 하나로 93% 제거"** | 실험 7/8 | 13 drive | 92.9% [89.4, 99.1] of the GT-history effect | 분모는 GT-history 효과다(Normal 대비로는 85%). 단발 교정이 아니라 반복 교정이다 | **축소** | "Last-position correction applied at every step recovered 93% of the full-history correction effect." |
| **GT 없는 action-context 효과** | P1-A (사전 등록) | 13 drive (A−), 19 drive (A+) | 문맥만 바꿔도 suffix local motion +0.57 m/seg. 기하 baseline 대비 +4.37 m [+3.32, +5.19] (Holm 9.8e-4). 모든 층에서 기하 baseline 초과 | AutoVLA에만 적용. 교란 위치와 A−/A+ 층은 GT 정의를 상속. Impromptu·SpatialVLA에는 GT-free 대조 없음 | **유지 (AutoVLA 한정)** | "Without ground-truth information, a single perturbed context token changed subsequent local actions and produced endpoint deviations 4.4 m beyond pure coordinate accumulation (AutoVLA)." |
| **등속 외삽을 넘는 초과증폭** | P1-A + C1 재분석 | 13 / 19 drive | 모델 − CM: A− +2.35 m [+1.27, +3.19]; A+ −0.52 m [−0.82, −0.26]; 전체 +0.20 m (p = 0.51) | 원래 비교(−0.59 m)는 CM 자체의 외삽 오차가 섞인 잘못된 대비였다. 초과는 실패 선택 층에서만 나타난다 | **축소 (일반 주장 삭제)** | "In failure-selected scenes the response exceeded constant-motion extrapolation; across all scenes it was not distinguishable from extrapolation." |
| **GT 없는 교정이 실패를 고친다** | P1-A OC_refR | 13 drive | reference 편차 −3.50 m. 그러나 GT 증폭은 +19.7%p 악화 | reference와의 일관성 회복이며, 과제 실패 개선이 아니다 | **삭제** | (쓰지 않는다. 대신) "Reference-based correction restored consistency with the model's own rollout but did not reduce ground-truth failures." |
| **early critical window (초기 위치 특수성)** | P1-B (사전 등록), C2 | 12 drive | early − late: w1 +10.7%p, w2 +4.3%p, w3 +0.5%p. 차이는 모두 비유의 | 늦은 window는 교정 해제 후 horizon이 2 step 짧다. 같은 해제 후 horizon에서는 처리 방식에 따라 부호가 바뀐다 | **삭제** ("초기 2–4 step이 특별하다") | "We found no evidence that corrections at the first deviating steps are more effective than equally long later corrections." |
| **짧은 교정 길이의 효율** | R9(복구), P1-B | 13 drive | 회복률 1/2/3/4 step = 47 / 77 / 90 [81, 99] / 96% (분모: 같은 run의 Normal−Full) | 교정 길이와 해제 후 남은 horizon이 함께 변한다(5.8 → 2.9 pose). "2–4 step이면 90%"는 틀린 표현 | **축소** | "Correcting three consecutive context steps recovered about 90% of the full-history effect (95% CI 81–99%)." |
| **방향 > 크기 일반화** | 실험 32 (PoC), P1-C (새 drive) | 13 / 18 drive | PoC +9.9%p [+3.9, +21.0], p = 0.0056. P1-C F +6.2%p, p = 0.073 (사전 등록 판정: 결론 안 남). drive 균등 가중 시 +15.5%p | 새 drive에서 미결론이고 검정력이 낮다 | **축소** | "Direction errors were more harmful than magnitude errors in the development drives; on new drives the effect was in the same direction but inconclusive." |
| **ensemble 보호 / 재계획의 역할 (SpatialVLA)** | 실험 35, P0-C, P1-D (사전 등록) | P1-D 고유 상태 coke 120 / move_near 120(지시문) 또는 10(물리) | 재계획: 문맥 예산을 고정해도 R−N 궤적 면적 +214 cm·step (H1 지지, 복합 설정 효과). ensemble: 실행 교란량을 맞추면 +21, ±100 동등. P0-C의 ensemble 감쇠는 실행 교란량 차이로 설명됨 | 재계획 축은 관측 갱신·문맥 reset·실행 노출이 함께 바뀐 복합 효과. move_near 물리 배치가 10개뿐 | **재계획: 유지 (궤적 한정, 복합 설정). ensemble 보호: 삭제** | "Lengthening the replanning interval increased the persistence of context-induced trajectory deviations even at a fixed context-perturbation budget; temporal ensembling did not absorb context effects once executed dose was matched." |
| **생성·물리 궤적 변화 vs 성공률 직접 효과** | P1-D, B 재분석 | 위와 같음 | 생성 suffix 50–58% 변화, TCP 2.7–6.0 cm. 성공률 직접 효과: δR·δC pooled r1e4/r4e1이 ±10%p 동등(두 군집 기준, Holm). ±5%p 불성립. 과제별 CI 18–29%p. 실행 교란은 성공률을 낮춤(C−N −11.7 ~ −22.9%p) | 두 과제 평균. 과제별로는 불확실. 실험 34/35 결과는 상태 군집 후 모두 불확실. 실험 35 r4e1 −6.2%p는 군집 후 비유의(p = 0.14–0.26)이고 P1-D에서 재현되지 않음 | **축소** | "Context perturbations changed generated actions and the physical trajectory, while their effect on success, averaged over two tasks, was bounded within ±10 pp in the tested configurations; task-specific effects remain uncertain." |
| **실험 35 "4 step 재계획에서 reverse가 실패를 늘린다 (−6.2%p)"** | 실험 35 | 120 coke + move_near 53(지시문)/10(물리) | −6.2%p, 초기 상태 군집 후 p = 0.14 / 0.26. P1-D에서 +1.2%p [−5.6, +8.0] | 군집 전 p = 0.04는 단위를 과대 계산한 결과 | **삭제** | (쓰지 않는다) |
| **모델 간 개입의 비교 가능성** | 실험 7/8, 33, 34 | – | – | Reverse 정의가 모델마다 다르다(embedding 삽입 / 자기 waypoint 재삽입 / 정상 실행 + 문맥만 교란). 증폭 정의도 다르다(A−∧FDE / 하위 차원 bin 이동 / 정규화 translation 누적) | **축소** | "Interventions were matched in spirit rather than identical across models; effect sizes are not comparable across columns." |
| **"GT motion 근접 token ≈ GT token" (identity 불변)** | 실험 11 | 13 drive | +0.8%p [−1.6, +6.6], TOST p = 0.09 | 동등성 불성립 | **축소** | "A token with GT-like motion produced a small, non-significant difference from the GT token (equivalence not established)." |
| **선택·navhard의 held-out 검증** | 실험 21–26 | dev/held-out drive 22개 공유 | (기존 수치 유지) | 새 log 검증이지 새 drive 검증이 아니다 | **축소 (표기)** | "held-out logs (not drive-disjoint)" |

## 원자료가 없어 확인하지 못한 항목
- 실험 5와 10: per-unit raw 없음. 1-token 교정 +55.8%p와 horizon-controlled 재발산은 summary로만 뒷받침된다.
- 실험 34·35: 물체 초기 상태를 기록하지 않았다. reset 방식 때문에 물체 위치가 drift했고, 조건 쌍의 bit 단위 복원도 검증하지 않았다. 실험 34는 실제 TCP 기록도 없다.
- P1-B: window가 정의대로 적용됐는지는 smoke 수준 trace로만 검증했다.
- AutoVLA 20-token 연장 자료와 turn-rate baseline: 없음.
