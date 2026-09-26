# Driving VLA의 perception-to-action misalignment 및 action-token 오류 증폭 — 결론 정리

대상 모델: **AutoVLA** (Qwen2.5-VL-3B 기반, nuPlan/navsim), **ORION** (Bench2Drive).
모든 실험은 저장된 inference 결과 또는 GPU1 재추론으로 수행했고, 기존 결과 파일은 수정하지 않았습니다.
[ ]는 log/clip 단위 cluster bootstrap 95% CI이며, 이진 지표는 McNemar, 연속 지표는 Wilcoxon 쌍대 검정입니다.

---

## 0. 한 줄 요약

1. 실패는 **perception 단계보다 결정·실행 결합 단계**에서 주로 발생합니다.
2. **CoT 텍스트는 action의 원인이라기보다 느슨하게 결합**되어 있고, CoT를 강제하면 오히려 성능이 떨어집니다.
3. 작은 action-token 편차가 궤적 실패로 증폭되는 기전은 **직전에 생성한 action 토큰의 identity가 residual stream을 통해 다음 action을 조건화하는 1-step autoregressive feedback**입니다. 특정 layer나 긴 history attention이 아닙니다.
4. 첫 mismatch 직후 **2–4 step**의 feedback만 안정화해도 이후 발산의 대부분(90–95%)이 사라집니다. 다만 교정을 멈추면 효과가 서서히 감쇠하므로 영구적 안정화는 아닙니다.

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
- 이는 mitigation 설계에 직접 쓰입니다: 첫 mismatch 검출 후 **2–4 step 동안만** 직전 action 조건화를 안정화(재계획, 앙상블, 또는 conditioning 보정)하면 궤적 실패의 대부분을 막을 수 있습니다. 다만 그 이후로도 주기적 재안정화가 필요합니다.

---

## 11. 한계

- **표본**: A− 장면은 52개(그중 t*=0은 17개)입니다. log cluster CI가 넓습니다.
- **단일 seed, 거의 greedy decoding**(T=0.01). 샘플링 다양성에 따른 변동은 다루지 않았습니다.
- **평가 horizon 5 s**. 모델의 계획 길이가 10 token이라 그 이상은 분포 밖이며, 장기 안정성은 closed-loop 재계획 실험이 필요합니다.
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
python scripts/analyze_<실험명>.py                                   # 분석·figure·보고서
```

원시 rollout 데이터(`records.jsonl`, `units.jsonl`, tensor `.npz`)는 용량 때문에 저장소에 포함하지 않았습니다. 위 스크립트로 재생성됩니다.
