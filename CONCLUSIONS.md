# Driving VLA의 perception-to-action misalignment 및 action-token 오류 증폭 — 결론 정리

대상 모델: **AutoVLA** (Qwen2.5-VL-3B 기반, nuPlan/navsim), **ORION** (Bench2Drive).
모든 실험은 저장된 inference 결과 또는 GPU1 재추론으로 수행했고, 기존 결과 파일은 수정하지 않았습니다.
[ ]는 log/clip 단위 cluster bootstrap 95% CI이며, 이진 지표는 McNemar, 연속 지표는 Wilcoxon 쌍대 검정입니다.

---

## 0. 한 줄 요약

1. 실패는 **perception 단계보다 결정·실행 결합 단계**에서 주로 발생합니다.
2. **CoT 텍스트는 action의 원인이라기보다 느슨하게 결합**되어 있고, CoT를 강제하면 오히려 성능이 떨어집니다.
3. 작은 action-token 편차가 궤적 실패로 증폭되는 기전은 **직전에 생성한 action 토큰의 identity가 residual stream을 통해 다음 action을 조건화하는 1-step autoregressive feedback**입니다. 특정 layer나 긴 history attention이 아닙니다. 그 identity 중 결과를 결정하는 것은 **토큰이 뜻하는 motion(codebook 기하)**이며, embedding 벡터의 근접성이 아닙니다.
4. 첫 mismatch 직후 **2–4 step**의 feedback만 안정화해도 이후 발산의 대부분(90–95%)이 사라집니다. 다만 교정을 멈추면 효과가 서서히 감쇠하므로 영구적 안정화는 아닙니다.
5. 모델은 **조건화된 motion을 66–89% 그대로 따라갑니다.** 그래서 GT 없이 쓸 수 있는 참조로 조건화를 교정하면 참조 품질이 전부를 결정하고, 배포 조건에서는 정상 장면의 손해 때문에 순이득이 없습니다(실험 12–16).
6. 대신 **모델 자신이 샘플한 후보(T 1.0, 16개) 중 디코딩 확신도로 하나를 고르면**, GT·외부 참조·교정 시점 없이 실패율과 궤적 오차가 모두 줄어듭니다. 이전에 쓰지 않은 log 56개(장면 4,563개)에서 규칙을 고정한 채 확인했을 때, 엔트로피 순위 + log-likelihood 순위의 합으로 고르면 A− 비율 2.15 → 1.45%(−33%, p = 4e-4), FDE 0.76 → 0.48 m(−36%)였습니다(실험 18–20).
7. 이 선택은 **실제 주행 지표로도 이어지고, 기전과 직접 연결됩니다**(실험 21–24, 사전 등록한 held-out log 52개에서 확인). NAVSIM PDM Score +0.0095, at-fault 충돌 0.56 → 0.23%이고, 현재 frame 정보만 쓰는 안전 필터를 더하면 PDMS +0.032, 충돌 0.10%입니다. 선택된 후보와 버려진 후보는 첫 이탈 **전** 엔트로피가 같고 **이탈 후에만** 갈라지므로(0.57 vs 0.97), 확신도는 feedback으로 불안정해지는 rollout을 골라내는 신호입니다.

---

## 1. 실험 목록

| # | 실험 | 질문 | 결과 경로 |
|---|---|---|---|
| 1 | P/R/A 실패 분해 | 실패가 perception / reasoning / interface 중 어디서 나는가 | `pra_comparison/outputs/PRA_COMPARISON.md` |
| 2 | CoT 인과 개입 | CoT가 action을 인과적으로 결정하는가 | `autovla_misalignment_poc/outputs/cot_intervention/` |
| 3 | Fast vs CoT (무편향) | CoT 강제가 성능을 올리는가 | `.../outputs/fast_vs_cot_unbiased/` |
| 4 | Natural/Fast 기전 재검증 | 기존 메커니즘 결과가 궤적 실패 정의에서도 성립하는가 | `.../outputs/natural_fast_mechanism/` |
| 5 | 첫 mismatch 인과 | 첫 토큰 하나가 궤적 실패의 원인인가 | `.../outputs/first_mismatch_causal/` |
| 6 | 등거리 perturbation | 같은 크기의 편차는 같은 결과를 내는가 | `.../outputs/equal_distance_perturbation/` |
| 7 | Action history 인과 | 증폭을 유지하는 것은 무엇인가 | `.../outputs/action_history_causal/` |
| 8 | Layer-wise state patching | 어느 layer가 그 효과를 운반하는가 | `.../outputs/prev_action_state_patching/` |
| 9 | Temporal feedback window | 몇 step을 안정화해야 하는가 | `.../outputs/temporal_feedback_window/` |
| 10 | Horizon-controlled window | 그 window가 horizon 교란 때문은 아닌가 | `.../outputs/horizon_controlled_window/` |
| 11 | 직전 토큰 identity 분해 | identity 중 motion인가 embedding인가 | `.../outputs/prev_action_identity_decomposition/` |
| 12 | 비-oracle 참조 안정화 | GT 없이(CTRA, 이전 frame 계획) 교정할 수 있는가 | `.../outputs/reference_stabilization/` |
| 13 | Receding-horizon 재계획 | 재계획이 feedback 발산을 끊는가 (pseudo closed loop) | `.../outputs/receding_horizon_replanning/` |
| 14 | 합의 참조 | 여러 과거 계획의 합의가 더 좋은 참조인가 | `.../outputs/consensus_reference_stabilization/` |
| 15 | 배포 조건 안정화 | 교정 시점 없이 자연 디코딩에 적용하면 | `.../outputs/natural_reference_stabilization/` |
| 16 | PDM-Closed 참조 | 규칙 기반 planner 궤적이 좋은 참조인가 | `.../outputs/pdm_reference_stabilization/` |
| 17 | seed·온도 강건성 | 실험 12의 효과가 샘플링에 강건한가 | `.../outputs/robustness_reference_stabilization/` |
| 18 | Best-of-N 선택 (N 8, T 0.7) | 조건화 대신 선택하면 | `.../outputs/best_of_n_selection/` |
| 19 | Best-of-N 선택 (N 16–32, T 1.0–1.3) | 후보 다양성을 늘리면 | `.../outputs/best_of_n_selection_n16_T1/` |
| 20 | 표본 확대 (새 log 56개) | 선택 효과가 새 데이터에서 재현되는가 | `.../outputs/expanded_best_of_n/` |
| 21 | PDM Score | open-loop 개선이 주행 품질 지표로 이어지는가 | `.../outputs/pdm_score_best_of_n/`, 보고서 `.../outputs/selection_validation/` |
| 22 | 기전 ↔ 선택 연결 | 선택이 feedback으로 불안정한 rollout을 거르는가 | `.../outputs/mechanism_selection_link/` |
| 23 | 후보 수 곡선 | 몇 개에서 포화되는가, 비용은 | `.../outputs/candidate_count_curve/` |
| 24 | 현재 frame 안전 필터 | 필터 + rank-sum | `.../outputs/safety_filter/` |
| 25 | NAVSIM v2 navhard pseudo closed-loop | 선택·필터가 2단계 반응형 평가에서도 유효한가 | `.../outputs/navhard_eval/` |
| – | 3080 Ti 결과의 5090 재실행 | GPU 혼용이 결론을 바꾸는가 | `.../outputs/gpu5090_reanalysis/` |

---

## 2. 실패 분해: 어디서 깨지는가 (실험 1)

두 모델에 **동일한 P/R/A 라벨링 함수**(`pra_comparison/pra_labels.py`)를 적용했습니다.

| A− 발생 단계 | 정의 | ORION | AutoVLA |
|---|---|---:|---:|
| perception | A− ∧ P− | 6.0% [3.0, 9.9] | 25.8% [17.4, 32.7] |
| reasoning | A− ∧ P+R− | 78.2% [70.1, 83.1] | 50.0% [41.3, 55.4] |
| interface | A− ∧ P+R+ | 15.9% [11.7, 22.3] | 24.2% [17.4, 34.6] |

- **ORION**: 결정 단계에서 깨집니다. perception이 맞은 A− 중 순수 결정 오류 55.1%, 순수 interface 20.6%.
- **AutoVLA**: 결정 오류와 interface 이탈이 겹칩니다. 순수 결정 오류는 13.7%뿐이고, **선언과 실행이 어긋난 경우가 86.3%**입니다.
- 두 모델 모두 **말한 결정과 실제 행동의 결합이 느슨합니다**. 행동은 맞는데 선언한 결정이 GT와 다른 P+R−A+가 ORION 38.5%, AutoVLA 42.4%로 가장 큰 그룹입니다. reasoning 텍스트를 action의 설명으로 읽으면 안 됩니다.
- 실패의 **양**은 다릅니다. A− 비율: ORION 19.8%, AutoVLA(강제 CoT) 5.2%, AutoVLA(자연 생성) 1.9%.

---

## 3. CoT는 action의 원인이 아니다 (실험 2, 3)

**개입 실험 (159 장면, 이미지·prompt 고정, 추론/결정 텍스트만 변경): 판정 = weakly coupled / post-hoc**

- 선언 방향을 반대로 뒤집어도 실행이 반대로 가는 비율은 **+6.4%p [+2.7, +9.7]**에 그칩니다.
- 반면 방향과 무관한 **텍스트 재작성**만으로 A+가 +17.0%p [+10.3, +26.5] 변하고 첫 action token이 22.0% 바뀝니다.
- 즉 action은 CoT가 *말한 내용*보다 CoT의 *형태*에 더 반응합니다.
- 기준선 변경을 보고서에 공개했습니다(실패로 선택된 샘플의 평균 회귀 때문에 `original` 대비 비교를 반대-교정 쌍대 비교로 교체).

**무편향 비교 (도시 층화 무작위 500 장면, 결과를 보지 않고 추출)**

| 조건 | A+ (5 s) | ADE (m) | FDE (m) | 첫 토큰 entropy |
|---|---:|---:|---:|---:|
| Natural | 98.8% [97.8, 99.8] | 0.29 | 0.66 | 0.224 |
| Forced Fast | 98.8% [97.8, 99.8] | 0.29 | 0.65 | 0.224 |
| Forced CoT | 93.8% [90.3, 96.3] | 0.79 | 1.88 | 0.409 |

**CoT를 강제하면 성능이 떨어집니다**: ΔA+ −5.0%p [−8.0, −2.9] (p=4.7e-06), ΔADE +0.50 m, entropy +0.185 nats. 이전 결과에서 관찰된 "CoT 실패"의 상당 부분은 CoT 자체가 만든 것입니다.

---

## 4. 실패 정의 정정과 기전 재검증 (실험 4)

기존 메커니즘 결과는 **natural run + "step-0 token ID ≠ GT"** 정의에서 나온 것이었습니다. 궤적 기준으로 다시 보면:

- natural 궤적 실패 52건 중 step-0 토큰이 틀린 것은 **17건뿐**이고, 토큰이 틀렸는데 궤적은 맞은 경우가 216건입니다. **두 실패 정의는 대부분 다른 샘플**입니다.
- 재현된 것: perception 정보는 실패군에서도 유지(AUROC 0.81–0.90), **late action formation**(commitment L34–35, 최종 margin 격차의 87%가 L32–35에서 생성), **L35 MLP 지배**(attention 대비 7.8–21.5배), **오류 증폭**(FDE/첫 오차 58배, 전파율 5%→73%).
- 부분 재현: small codebook mismatch(오답 토큰의 NN 순위 중앙값이 8 → 15로 커짐).

---

## 5. 첫 mismatch 하나가 궤적 실패를 만든다 (실험 5)

첫 불일치 step t*에서 **토큰 하나만** GT로 교정하고 이후는 자유 생성:

| 지표 | A− original | A− GT 1-token 교정 | A+ (step-matched) |
|---|---:|---:|---:|
| A+ (5 s) | 17.3% | **73.1%** (+55.8%p, p=4.2e-07) | 99.0% → 98.7% (변화 없음) |
| FDE 5 s (m) | 8.18 | 2.09 (−6.09 m) | 2.01 → 0.75 |
| 오차 증가 속도 (m/step) | 1.035 | 0.246 | 0.239 → 0.083 |

A−의 궤적 실패는 **경로 의존적**입니다. 첫 토큰 하나를 되돌리면 대부분 회복됩니다. A+는 애초에 교정할 것이 없습니다.

---

## 6. 결과를 가르는 것은 편차의 크기가 아니라 장면이다 (실험 6)

같은 거리(중앙값 상대차 7.9%, 절대 8.8 mm)의 대안 토큰을 강제했을 때:

- A− 장면: alt 전체 증폭률 **45.0%**, recovery 49.2% — **같은 크기의 편차가 장면 안에서도 회복과 증폭으로 갈립니다**(A− 장면의 71%에서 두 결과가 공존).
- A+ 장면: 같은 편차의 증폭률은 **4.8%**뿐입니다.
- 방향별 차이(34.9%–60.0%)는 장면 간 차이보다 작습니다. **거리도 방향도 아닌 장면 맥락이 결과를 지배합니다.**

---

## 7. 증폭을 유지하는 것은 직전 토큰의 identity다 (실험 7)

첫 mismatch 이후 조건화 방식만 바꾸고 실행 action은 모델 출력으로 둔 실험(A− 365 / A+ 1043 단위):

| 조건 | A− 증폭률 |
|---|---:|
| Normal AR | 47.4% |
| Action-history attention mask | 46.0% (효과 없음) |
| Recent-action attention mask | 44.1% (효과 없음) |
| **Recent-GT (직전 1개만 GT)** | **7.1%** |
| **GT-history (전부 GT)** | **4.1%** |

**직전 토큰 1개만 교정해도 전체 효과의 93%가 나옵니다.** attention mask는 효과가 없으므로, 기전은 긴 history 참조가 아니라 **생성 위치 자신의 입력 토큰**입니다.

---

## 8. 특정 layer가 아니라 residual stream 전체가 운반한다 (실험 8)

직전 action 위치의 hidden state를 layer별로 GT-history state로 교체(37개 layer 전수):

- 증폭률은 **어느 layer에서 patch해도 3.8–7.4%** (GT-history 효과의 93–101%).
- 구조적 동치를 수치로 확인: `patch@embedding` ≡ Recent-GT, `patch@L35` ≡ GT-history (1408/1408 토큰 일치, logits L1 = 0). 직전 action 위치가 곧 다음 action의 생성 위치이기 때문입니다.
- **역방향(reverse) patch**: GT 문맥에 자기 생성 토큰의 embedding 하나만 넣어도 증폭이 4.4% → **33.4%**로 복귀(+29.0%p, Normal−GT 간격의 68%). identity만으로 충분조건에 근접합니다.
- layer에 따라 달라지는 부분은 **과거 GT 문맥의 통합**이며 주로 **L17–L33**(L19, L30에서 계단식)에서 들어옵니다. 이 성분은 GT 재정렬(+25%p)과 entropy(−0.49 nat)를 크게 바꾸지만 증폭은 약 3%p만 바꿉니다.
- KV cache 검증: patch layer 이하 K/V 차이 0, 이후 layer에서 K 3.8 / V 17 이상, logits L1 ≥ 2.4e3. 거짓 음성 없음.

**결론: 오류를 유지하는 정보는 "직전에 무엇을 생성했는가"라는 토큰 identity이고, 이는 embedding에서 주입되어 residual로 끝까지 운반됩니다. 중간 layer가 이를 선택적으로 증폭·차단하지 않습니다.**

---

## 9. 임계 시간 window: 2–4 step (실험 9, 10)

**(9) 10-token horizon에서의 window 길이** — 교정 step 수를 늘려가며(A− 365 단위):

| 교정 window | A− 증폭률 | Full 효과 대비 |
|---|---:|---:|
| Normal AR | 46.6% | – |
| 1-step | 26.6% | 47% |
| 2-step | 14.2% | 76% |
| 3-step | 8.5% | 90% |
| 4-step | 5.8% | 96% |
| Full GT-history | 4.1% | 100% |

- 5-step부터 추가 이득이 유의하지 않습니다(p=0.5).
- **가장 이른 step이 특별하지는 않습니다.** 교정 종료 위치를 맞춘 비교에서 t*+1 추가(−4.9%p)와 t*+2 추가(−4.9%p)의 기여가 같습니다. 누적량이 결정합니다.

**(10) Horizon 교란 통제** — 모든 window에 교정 종료 후 **동일한 free step 수**를 부여(t*=0, N=4):

| window | A− FDE 회복 | A+ FDE 회복 | free horizon N을 늘릴 때 (A−) |
|---|---:|---:|---|
| 1-step | 62% [50, 74] | 71% | 83% (N=1) → 49% (N=7) |
| 2-step | 87% [80, 93] | 87% | 98% → 79% (N=6) |
| 3-step | **90% [83, 97]** | 93% | 98% → 87% (N=5) |
| 4-step | **95% [90, 100]** | 89% | 101% → 95% (N=4) |

- **horizon 교란만으로 생긴 착시가 아닙니다.** 같은 2 s를 줘도 1-step(62%)과 4-step(95%)의 차이가 큽니다.
- 그러나 **감쇠합니다**. 모든 window에서 N 증가에 따른 회복률 감소의 CI가 0을 포함하지 않습니다(3-step −11%p [−19, −4]). 교정 지속 대비 초과 오차도 3-step 기준 0.03 → 0.75 m로 누적됩니다.
- 해제 후 재발산: 3–4 step 교정에서 25–29% (Normal 49%, Full 12%).
- **5 s 이후는 분포 밖이라 검증 불가**: AutoVLA는 10번째 action 뒤 `\n</answer>`로 답을 닫으려 하며 action 토큰 확률이 ≈0입니다(1049/1049 단위).

**수정된 claim**: *AutoVLA has a genuine short early window (the first ~2–4 autoregressive transitions after a deviation) in which stabilizing the previous-action feedback removes most of the subsequent 1.5–2 s divergence even at an equal free-generation horizon; the benefit is not a horizon artefact, but it decays after release, so the window delays rather than permanently prevents re-divergence.*

---

## 9b. identity 중 결과를 가르는 것은 motion이다 (실험 11)

직전 action 토큰 하나를 조건별 대체 토큰으로 바꾸고 이후는 자유 생성했습니다(실험 7–8과 같은 harness, A− 365 / A+ 1043 단위).
사전 분석: action embedding은 `lm_head`와 tied이고, codebook 기하를 선형으로 거의 담지 않습니다(CV R² ≤ 0.09, 거리 순위상관 0.06). 410개 토큰은 embedding이 사실상 동일한 미학습 군집입니다.

| 직전 토큰 := | A− 증폭률 | Recent-GT 효과 대비 | A+ 증폭률 |
|---|---:|---:|---:|
| 자기 토큰 (Normal) | 47.1% | – | 4.2% |
| GT (Recent-GT) | 7.9% | 100% | 0.7% |
| **GT와 motion이 가장 비슷한 다른 토큰** | **8.5%** | **99%** | **0.8%** |
| **자기와 motion이 가장 비슷한 다른 토큰** | **43.8%** (p = 0.18) | 8% | 7.2% |
| 무작위 토큰 / 평균 embedding | 51.5 / 52.1% | – | **43.2 / 40.5%** |

- motion과 embedding 근접을 공동 회귀하면(두 변수 상관 −0.04), 대체 토큰이 motion으로 GT에서 1 SD 멀어질 때 A− FDE +1.99 m [+1.52, +2.70], 증폭 +16.1%p이고, embedding 근접은 motion을 고정하면 효과가 없습니다(전체 +0.03 m [−0.13, +0.19]).
- motion 정보를 없애면 정상 장면(A+)도 4% → 40%대로 무너집니다. 직전 action의 motion은 오류를 증폭하는 경로이자 정상 주행을 유지하는 운동학적 prior이므로, 조건화를 끊는 방식은 해결책이 아닙니다.
- **mitigation 함의**: 안정화에 정확한 GT 토큰은 필요 없고, GT와 motion이 같으면 충분합니다. 다만 실험 12–16에서 보듯 비-oracle 참조는 GT보다 훨씬 부정확해, 외부 planner 궤적을 조건화하는 것은 오히려 해롭습니다(§9c).
- 한계: embedding 조건의 조작 강도가 약하고(Δcos 약 ±0.1), motion 거리는 codebook 48-d L2 정의입니다.

## 9c. GT 없는 mitigation: 조건화는 실패하고, 선택은 통한다 (실험 12–19)

**조건화 교정 (context에 참조 motion을 넣음)**

| 실험 | 참조 / 조건 | 실패 장면 | 정상 장면 | 판정 |
|---|---|---|---|---|
| 12 | 이전 frame 계획, oracle 시점 | A− 증폭 47.1 → 33.2% (GT 효과의 34%) | 4.2 → 3.2% | 부분 효과 |
| 12 | CTRA 운동학 외삽 | 효과 없음 / 악화 | 4.2 → 9–16% | 해로움 |
| 14 | 0.5–1.5 s 전 세 계획의 medoid 합의 | 이전 계획보다 −7.1%p 추가 | FDE +0.28 m | 트레이드오프 |
| 16 | PDM-Closed 궤적 | 47.1 → 31.0% (FDE 거의 불변) | **4.2 → 20.1%** | 해로움 |
| 17 | 이전 계획, seed 0–2 × T 0.01/0.5 | 6개 실행 모두 45–48 → 31–33% | 해 없음 | 강건 |
| 15 | 이전 계획, **교정 시점 없이 상시 적용** | 자연 실패 A− 65 → 33% | A− 0 → 2%, FDE +0.8 m | **모집단 1.24 → 2.60%로 악화** |

- 모델은 조건화된 참조의 motion으로 실행 토큰을 66–89% 끌어갑니다(GT, 이전 계획, PDM 모두). 그래서 효과는 참조↔GT motion 거리로 결정되고(가까운 1/3: −30.6%p, 먼 1/3: +13.1%p), 정상 장면에서는 모델 자신이 어떤 비-oracle 참조보다 정확해 교정이 손해가 됩니다.
- 모델 엔트로피로 켤 장면을 고르는 trigger는 실패 장면을 AUROC 0.88로 구별하지만 교정 위치보다 뒤를 봐야 하고, 인과적 신호(첫 step, AUROC 0.70)로는 모집단 순이득이 유의하지 않습니다(−0.18%p [−0.53, +0.19]).

**재계획 (실험 13, pseudo closed loop)**: 2–4 step마다 로그 관측으로 재계획하면 실패 장면의 5 s 실패가 59.6 → 23.5%로 줄지만 로그 카메라를 통한 누출이 섞여 있고, 자기 실행 속도를 되먹이면 8 s 오차가 +2.5–5.5 m 커지는 상태 수준 feedback이 생기며, 재계획 안에서 이전 계획으로 묶으면 회복을 막습니다.

**선택 (모델 자신의 후보 중 고름, context 불변)**

| 설정 | 규칙 | 실패 장면 Δ A− | 모집단 Δ A− | 모집단 Δ FDE5 |
|---|---|---|---|---|
| N 8, T 0.7 | 최소 엔트로피 | −11.5%p | +0.11%p (n.s.) | −0.17 m |
| N 16, T 1.0, seed 0–4 | 최소 엔트로피 | −21 ~ −33%p | 5 seed 합산 −0.41%p [−0.81, +0.09] | **−0.16 m [−0.27, −0.02]** |
| N 32, T 1.0 | 최소 엔트로피 | −36.5%p | −0.36%p (n.s.) | −0.18 m |
| N 32, T 1.3 | 최소 엔트로피 | −30.8%p | +0.41%p (n.s.) | +0.01 m |
| N 16, T 1.0, seed 0–4 | 최대 log-likelihood | −6 ~ −14%p | 5 seed 합산 −0.39%p [−0.77, +0.02] | **−0.16 m [−0.24, −0.08]** |
| N 16, T 1.0, seed 0–4 | 순위 합 (엔트로피 + log-lik) | −15 ~ −21%p | 5 seed 합산 −0.42%p [−0.85, +0.09]; held-out seed 3에서 유의 | **−0.18 m [−0.28, −0.06]** |

- 선택은 조건화와 달리 정상 장면을 해치지 않습니다(모든 후보가 모델 분포 안에 있음). 이득의 상한은 후보 다양성이 정합니다(oracle 선택의 잔여 실패: N 8 T 0.7에서 29%, N 16 T 1.0에서 12%). T 1.3은 정상 장면 후보까지 퍼뜨려 역효과가 납니다.
- 엔트로피는 10 step 전체로 계산해야 합니다(처음 2–4 step만 쓰면 모집단 실패율이 오히려 증가).
- 최소 엔트로피는 실패 장면을 가장 많이 구하지만 설정에 민감하고(N·T 곡선 10개 중 모집단 유의 2개), 최대 log-likelihood는 효과가 작지만 거의 모든 설정에서 FDE를 유의하게 줄입니다. 원래 28개 log(자연 실패 52개)로는 실패율 감소가 경계선이었습니다.
- **표본 확대 (실험 20)**: 이전에 쓰지 않은 navtest log 56개(장면 4,563개, 자연 실패 98개)에 규칙을 고정한 채 적용했습니다.

| 규칙 | 모집단 A− (자연 2.15%) | Δ A− [95% CI], McNemar p | FDE5 (자연 0.758 m) |
|---|---:|---|---:|
| **순위 합** | **1.45%** | **−0.70%p [−1.07, −0.33], 4e-4** | **0.484 m** |
| **최대 log-likelihood** | **1.47%** | **−0.68%p [−0.97, −0.39], 3e-6** | 0.557 m |
| 최소 엔트로피 | 1.88% | −0.26%p [−0.71, +0.19], 0.28 | 0.565 m |
| oracle (상한) | 0.24% | −1.91%p | 0.191 m |

  **순위 합과 최대 log-likelihood가 새 데이터에서 실패율을 약 1/3, 궤적 오차를 26–36% 유의하게 줄였습니다.** 최소 엔트로피는 실패 장면 구제율이 가장 높지만(59%) 정상 장면에서 새 실패를 만들어 상쇄됩니다.

## 9d. 검증: 주행 품질, 기전, 비용, 안전 필터 (실험 21–24)

개발 셋(새 log 56개, 4,563 장면)에서 규칙을 정하고, 결과를 보기 전에 사전 등록(`outputs/heldout_best_of_n/PREREGISTRATION.md`, 커밋 513fdba)한 뒤 held-out(log 52개, 4,814 장면)에서 한 번 평가했습니다. 상세: `autovla_misalignment_poc/outputs/selection_validation/SELECTION_VALIDATION.md`.

| held-out | PDMS | at-fault 충돌 | 주행 가능 영역 이탈 | TTC 위반 | open-loop A− |
|---|---:|---:|---:|---:|---:|
| 자연 계획 | 0.8893 | 0.56% | 4.36% | 2.14% | 1.50% |
| rank-sum | 0.8988 (+0.0095 [+0.0046, +0.0148]) | 0.23% | 3.61% | 1.54% | 0.71% |
| rank-sum + 현재 frame 안전 필터(F1) | **0.9209 (+0.0315 [+0.0234, +0.0410])** | **0.10%** | **1.23%** | 1.33% | 1.08% |

- **주행 품질(실험 21)**: rank-sum은 개발 셋(+0.016)과 held-out(+0.0095) 모두에서 PDMS를 올리고 충돌을 절반 이하로 줄이며 진행도를 떨어뜨리지 않습니다. 자연 실패 장면에서는 PDMS +0.11–0.14입니다.
- **기전 연결(실험 22)**: 선택 vs 버림의 이탈 **전** 엔트로피 차이는 −0.016(장면 내 증폭 AUROC 0.51 = 우연), 이탈 **후**는 −0.40(held-out)입니다. 버려진 후보는 오차 증가가 1.8배 빠르고 GT 재정렬률이 낮으며(0.24 vs 0.39) 증폭률이 3–6배 높습니다. rank-sum은 장면 안에서 증폭 rollout을 AUROC 0.85–0.87로 구별합니다. 확신도는 장면 난이도가 아니라 **이탈 이후 feedback의 자기 강화**를 포착합니다.
- **후보 수(실험 23)**: N = 8에서 held-out PDMS 이득의 85%, 12–16에서 포화. 후보를 한 batch로 디코딩하면 N = 16의 추론 비용은 1.3배입니다.
- **안전 필터(실험 24)**: 현재 객체의 등속 외삽과 지도만으로 충돌·주행 가능 영역 이탈이 예측된 후보를 지우면 PDMS 이득이 2–3배가 됩니다. 다만 open-loop A−(사람 궤적과의 거리)는 rank-sum 단독보다 덜 줄어, 필터는 선택을 "사람처럼"보다 "규칙 준수" 쪽으로 옮깁니다. 필터와 평가가 같은 채점 규칙을 쓰는 점도 감안해야 합니다.
- 한계: PDMS는 비반응형 시뮬레이션입니다(반응형 2단계 평가는 §9e).
- **5090 재실행**: 3080 Ti로 디코딩한 모든 run(dev 17 log, held-out 22 log, 실험 14·15·18·19)을 5090에서 다시 돌려 5090 결과만으로 재분석했습니다. held-out rank-sum +0.0099 [+0.0050, +0.0154], F1 +0.0316 [+0.0235, +0.0411]로 혼합 GPU 결과와 같고, 기전·후보 수 결론도 같습니다. 개별 경계 장면의 자연 계획은 GPU마다 다르지만 선택의 개선 폭은 같습니다. 상세: `autovla_misalignment_poc/outputs/gpu5090_reanalysis/GPU5090_REANALYSIS.md`.

## 9e. NAVSIM v2 navhard 2단계 pseudo closed-loop (실험 25)

우선순위 1·2가 모두 긍정적이어서 navhard_two_stage의 절반(36 log, 105 그룹, 원본 210 + 합성 2,702 장면)을 공식 v2 채점으로 평가했습니다. 규칙·필터는 dev에서 고정한 그대로입니다. 상세: `autovla_misalignment_poc/outputs/navhard_eval/NAVHARD_CLOSED_LOOP.md`.

| 규칙 | EPDMS | Δ vs 자연 [95% CI, log-cluster] | 그룹 개선/악화 |
|---|---:|---|---|
| 등속 궤적 | 0.130 | – | – |
| 자연 계획 | 0.239 | – | – |
| rank-sum | 0.258 | +0.019 [−0.007, +0.050] | 41 / 38 |
| max log-lik | 0.271 | **+0.032 [+0.012, +0.057]** | 36 / 27 |
| min entropy | 0.235 | −0.005 [−0.034, +0.022] | 43 / 36 |
| **F1 (현재 frame 안전 필터 + rank-sum)** | **0.365** | **+0.125 [+0.090, +0.164]** | 70 / 27 |

- 안전 필터는 반응형 2단계 평가에서도 EPDMS를 52% 올립니다(1단계 0.72 → 0.81, 2단계 0.33 → 0.45).
- 확신도 선택만으로는 이득이 작고, 분포 밖 합성 장면에서는 엔트로피 성분이 도움이 안 됩니다(rank-sum은 CI가 0 포함, max log-lik만 유의).
- 한계: 필터와 채점이 같은 PDM 규칙을 씁니다. navhard 절반이며, 실제 연속 closed-loop 시뮬레이션은 아닙니다.

---

## 10. 종합 그림

```
perception (대체로 보존)
      │
      ▼
결정/CoT 텍스트 ──(느슨한 결합, 방향 효과 +6.4%p)──▶ action token
      │                                                  │
      │                                    첫 mismatch (t*)
      │                                                  │
      ▼                                                  ▼
장면 맥락이 증폭 여부를 지배          직전 토큰 identity가 residual로
(같은 거리 편차: A− 45% vs A+ 5%)     다음 action을 조건화 (1-step feedback)
                                                         │
                                          2–4 step 안정화 시 90–95% 차단
                                          (단, 해제 후 서서히 재발산)
```

- 실패의 축은 "무엇을 보았는가"가 아니라 **"직전에 무엇을 출력했는가"**입니다.
- oracle 교정으로는 2–4 step 안정화가 궤적 실패의 대부분을 막지만, GT 없는 참조로 조건화를 교정하면 모델이 그 참조를 따라가 정상 장면이 손해를 봅니다(§9c). 배포 가능한 순이득은 **조건화가 아니라 선택**에서 나왔습니다: 모델 자신의 다양한 후보 중 디코딩 확신도(엔트로피 순위 + log-likelihood 순위)가 가장 높은 계획을 고르면, 새 log 56개에서 실패율 −33%, FDE −36%였습니다(open-loop 기준).

---

## 11. 한계

- **표본**: A− 장면은 52개(그중 t*=0은 17개)입니다. log cluster CI가 넓습니다.
- **seed와 decoding**: 실험 1–16은 단일 seed, T = 0.01입니다. 실험 12의 핵심 효과는 seed 0–2 × T 0.01/0.5에서 재현되었고(실험 17), 실험 19는 3 seed를 합산했습니다.
- **평가 horizon 5 s**. 모델의 계획 길이가 10 token이라 그 이상은 분포 밖입니다. 실험 13의 8 s 재계획은 카메라가 로그를 따르는 pseudo closed loop이며, 진짜 closed-loop 평가(NAVSIM v2 pseudo-simulation 등)는 하지 않았습니다.
- **GPU**: 실험 14, 15, 18, 19(seed 0–2)는 RTX 3080 Ti, 나머지는 RTX 5090입니다. 실패 장면은 모델이 확신하지 못하는 경계 장면이라 GPU 간 부동소수 차이만으로도 결과가 바뀝니다(3080 Ti에서 5090 자연 실패 장면의 A− 재현 65%). 모든 비교는 같은 실행 안의 쌍대 비교입니다.
- **교정 정의**: 조건화 context만 GT로 두고 실행 action은 모델 출력입니다. 실제 주행에서 궤적을 교정하는 것과는 다릅니다.
- **수치 재현성**: 조건을 batch로 묶어 계산하므로 batch 크기가 다른 이전 실험과 토큰 재현율이 87–94%입니다. 모든 비교는 같은 batch 안의 쌍대 비교입니다.
- **ORION**은 P/R/A 비교에만 사용했고, 기전 실험(7–10)은 AutoVLA에서만 수행했습니다.

---

## 12. 재현

각 실험 디렉토리의 스크립트를 순서대로 실행합니다(GPU가 필요한 실험은 `CUDA_VISIBLE_DEVICES=1`).

```bash
cd autovla_misalignment_poc
CUDA_VISIBLE_DEVICES=1 python scripts/full_extract.py              # 전체 추출 (arm N/C)
python scripts/make_natural_fast_report.py                          # 기전 재검증 (GPU 불필요)
CUDA_VISIBLE_DEVICES=1 python scripts/first_mismatch_causal.py      # 실험 5
CUDA_VISIBLE_DEVICES=1 python scripts/equal_distance_perturbation.py # 실험 6
CUDA_VISIBLE_DEVICES=1 python scripts/action_history_causal.py      # 실험 7
CUDA_VISIBLE_DEVICES=1 python scripts/prev_action_state_patching.py # 실험 8
CUDA_VISIBLE_DEVICES=1 python scripts/temporal_feedback_window.py   # 실험 9
CUDA_VISIBLE_DEVICES=1 python scripts/horizon_controlled_window.py  # 실험 10
CUDA_VISIBLE_DEVICES=1 python scripts/prev_action_identity_decomposition.py  # 실험 11
# 실험 12-19: bash ../tools/regenerate_autovla.sh (단계와 인자는 스크립트 머리말)
python scripts/analyze_<실험명>.py                                   # 분석·figure·보고서
```

원시 rollout 데이터(`records.jsonl`, `units.jsonl`, tensor `.npz`)는 용량 때문에 저장소에 포함하지 않았습니다. 위 스크립트로 재생성됩니다.
