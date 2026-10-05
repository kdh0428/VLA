# 실험 33: autoregressive action-chunk VLA에서의 previous-action feedback 재현 — Impromptu VLA 3B

사전 등록: `PROTOCOL.md` (커밋 038a6d3, 본 실행 전; 실행 세부 §6 포함). 디스크 정리 기록: `DISK_CLEANUP.md`.
분석: `analysis.json` (`scripts/cross_vla_temporal/analyze_temporal.py`), 탐색적 분석 `exploratory_*.json`(사전 등록 밖, 그렇게 표시).
모든 추론은 RTX 5090. 기존 AutoVLA/navhard 결과·문서는 수정하지 않았습니다.

## 1. 사용 모델과 architecture

**Impromptu VLA 3B** (`aaaaaap/ImpromptuVLAModel/3B_AD`, "3B Base+Impromptu"; Qwen2.5-VL-3B 전체 fine-tune, bf16 7.1 GB).

| 선정 조건 (코드 수준 확인) | Impromptu | 근거 |
|---|---|---|
| action을 여러 token/step으로 autoregressive 생성 | O | Qwen2.5-VL causal LM `generate`; 답 = `<PLANNING>…: [x1, y1], …, [x10, y10]</PLANNING>` |
| 앞 action token이 뒤 token을 conditioning | O | causal attention; 아래 C·D 개입으로 실제 conditioning을 직접 확인 |
| chunk 안 시간 순서 | O | 미래 10개 waypoint, 0.5 s 간격(0.5–5.0 s), 왼쪽부터 시간순 |
| 공개 checkpoint·code | O | 공개 weight, 공개 data-engine(`loaders/pipelines`)의 프롬프트·답 형식 |

제외한 후보: **π0-FAST**(DCT 주파수 계수를 BPE로 압축한 token이라 chunk 안 시간 순서가 없음), **SpatialVLA**(조건은 충족하나 LIBERO checkpoint 비공개,
SimplerEnv는 이 컨테이너에 없는 Vulkan 필요), **OpenVLA**(실험 28; step 안 7차원만 생성, 시간축 action history 없음).

입력은 현재 frame의 CAM_F0/R0/L0(이미지 ≤ 262,144 px)과 과거 1.5 s의 4개 상태(위치·가속도·속도). 출력 waypoint는 **현재 ego 좌표의 절대 위치**를
여러 digit token으로 씁니다. 디코딩은 greedy(+ checkpoint의 repetition penalty 1.05).

## 2. AutoVLA와의 action generation 구조 비교

| | AutoVLA | Impromptu VLA |
|---|---|---|
| backbone | Qwen2.5-VL-3B | Qwen2.5-VL-3B |
| action 단위 | 0.5 s마다 상대 motion 1개 = codebook token 1개(2,048개) | 0.5 s마다 절대 위치 `[x, y]` = digit token 약 12개 |
| chunk | 10 step(5 s), 시간순 autoregressive | 10 waypoint(5 s), 시간순 autoregressive |
| 앞 action → 뒤 action | 직전 motion token이 다음 token을 조건화 | 앞 waypoint 텍스트(위치)가 다음 waypoint를 조건화 |
| 자연 성능(navtest PoC) | A− 1.9%, FDE5 0.66 m | **A− 33.4%, 증폭 31.5%, FDE5 5.75 m** (2,728 장면) |

차이: (i) Impromptu는 절대 위치를 쓰므로 문맥에 GT waypoint를 넣으면 GT 궤적 정보 자체가 들어갑니다(AutoVLA의 GT motion token보다 강한 정보 누출).
(ii) Impromptu는 자연 상태에서도 훨씬 부정확합니다.

## 3. Intervention protocol

- **A·B** (2,748 장면 중 해석 가능한 2,728): 첫 waypoint(0.5 s)를 자연 값 + δ로 강제(|δ| = 0.2, 0.5 m × 방향 8개), 나머지 9개 자유 생성.
  기준 `pert_m0`(δ = 0, 같은 경로)과 비교. r = 최종 발산 / |δ|.
- **C·D·E** (무작위 300 장면 × 방향 8 = 2,398 단위, |δ| = 0.5 m): waypoint를 하나씩 생성하며 **문맥만** 교정·대체. 실행 궤적은 항상
  [강제 w̃_1, 그 행이 생성한 w_2..w_10]. 행 정의는 `PROTOCOL.md` §3(AutoVLA 실험 7–11, 32와 같은 구조).
- 지표: A−(P/R/A), 증폭(A− and FDE5 > 3 m), FDE5, D10(같은 행 δ = 0 대비 최종 발산). log 단위 bootstrap 95% CI, McNemar / Wilcoxon.

## 4. Natural rerun variability

| 비교 | A− 불일치 | 증폭 불일치 | 최종 waypoint 이동 |
|---|---:|---:|---:|
| natural vs natural_rep (batch 구성만 다름) | 5.9% [5.2, 6.9] | 5.9% [5.1, 6.7] | 0.90 m [0.84, 0.96] |
| natural vs pert_m0 (prefix 재encoding 경로) | 5.2% [4.4, 6.5] | 5.2% [4.5, 6.1] | 0.74 m [0.70, 0.80] |

natural과 natural_rep의 A− 비율 차이는 +0.18%p(McNemar 83 vs 78, p = 0.75)로 방향성 없는 수치 변동입니다.
자연 A+ 장면이 재실행만으로 A−가 되는 비율은 4.6% [3.7, 5.5]입니다.

## 5. Experiment A–E 결과

### A. 작은 초기 편차 ≠ 실패?

| | \|δ\| = 0.2 m | \|δ\| = 0.5 m |
|---|---|---|
| 기준(δ = 0)이 A+인 장면에서 교란 후 A− | **49.5% [46.9, 52.5]** | 58.6% [55.9, 61.4] |
| 교란 후 A− − 기준 A− (paired) | +24.8%p [+19.4, +32.2], McNemar p ≈ 0 | +31.1%p [+25.4, +39.1], p ≈ 0 |
| 최종 발산비 r 중앙값 | 27.6 (0.2 m → 약 5.5 m) | 15.3 |
| r ≤ 1 (흡수) / r ≥ 3 (궤적 증폭) | 0.45% / 97.4% | 0.60% / 96.8% |

→ task 수준에서는 **작은 편차가 항상 실패로 가지는 않습니다**(0.2 m 교란 후에도 약 절반은 A+ 유지). 그러나 실패로의 전환(49.5%)은 재실행 기저(4.6%)의
10배이고, **궤적 수준에서는 거의 모든 편차가 3배 이상 커집니다**(97%). AutoVLA(A+ 장면 증폭 4.8%, 같은 크기 편차의 상당수가 흡수)보다 훨씬 취약합니다.

### B. matched-magnitude 안정/불안정 분기

| | \|δ\| = 0.2 m | \|δ\| = 0.5 m |
|---|---|---|
| **사전 등록 기준**: 안정(A+ and r ≤ 1)과 불안정(증폭 or r ≥ 3)이 같은 장면에 공존 | **2.2% [1.5, 3.2]** | **3.2% [2.4, 3.9]** |
| log r 분산 중 장면 간 비율(ICC) | 0.32 | 0.21 |
| 방향별 증폭률 (전방 0° / 후방 180° / 횡·대각) | 34% / 36% / 60–71% | 36% / 39% / 61–82% |
| (탐색적) task 수준: 같은 장면에서 A+ 방향과 증폭 방향이 공존 | 86.9% [84.8, 89.0] | 92.6% [90.7, 94.2] |

→ 사전 등록 기준(궤적 수준 "안정" = r ≤ 1)의 공존 비율은 재실행 불일치(5.9%)보다 **낮아 기준 3은 충족되지 않습니다**: 궤적 수준에서 안정 분기가 거의 없기 때문입니다.
다만 분산의 70–80%가 장면 안(방향)에 있고, task 수준에서는 같은 크기의 교란이 방향에 따라 통과/증폭으로 갈리는 장면이 87–93%입니다(탐색적).
횡·대각 방향 교란이 전후 방향보다 약 2배 증폭됩니다.

### C. Previous-action correction (문맥만 교정; 2,398 단위)

| 문맥 | 증폭 [95% CI] | A− | FDE5 | D10 | normal 대비 증폭 (McNemar) |
|---|---|---:|---:|---:|---|
| normal | 63.4% [60.6, 66.7] | 64.5% | 10.49 m | 8.80 m | – |
| win1 | 48.0% [42.8, 52.2] | 48.8% | 7.90 | 6.50 | −15.4%p, p = 7e-24 |
| win2 | 38.9% [33.7, 43.2] | 40.3% | 5.77 | 4.69 | −24.6%p, p = 2e-69 |
| **win3** | **6.0% [3.0, 8.2]** | 9.5% | 2.06 | 0.84 | −57.4%p, p ≈ 0 |
| win4 | 3.2% [0.9, 5.2] | 6.3% | 1.32 | 0.37 | −60.2%p, p ≈ 0 |
| gt_history (full) | 0.0% | 0.0% | 0.06 | 0.01 | −63.4%p, p ≈ 0 |
| **recent_gt (직전 1개만 GT)** | **1.3% [0.8, 1.8]** | 16.3% | 0.94 | 1.12 | **−62.2%p [−65.6, −59.1]**, p ≈ 0 |

full 효과 대비 비율(증폭 / D10): win1 24% / 26%, win2 39% / 47%, **win3 91% / 91%**, win4 95% / 96%, recent_gt 98% / 87%.
normal이 증폭된 1,521 단위에서도 같은 양상입니다(win1 57%, win2 58%, win3 93%, win4 96%).

### D. Reverse intervention

| 문맥 | 증폭 | D10 | FDE5 |
|---|---:|---:|---:|
| gt_history (교정된 문맥) | 0.0% | 0.01 m | 0.06 m |
| **gt_history + 자기 생성(오차) waypoint를 마지막에 재삽입** | **81.9% [80.4, 84.2]** | **40.5 m** | 39.95 m |
| normal (원래 feedback) | 63.4% | 8.80 m | 10.49 m |

reverse − gt_history: 증폭 +81.9%p [+80.4, +84.2](McNemar p ≈ 0), D10 +40.5 m [+37.3, +43.9](Wilcoxon p ≈ 0).
→ **원인을 제거하면 실패가 사라지고(remove cause → failure 63 → 0–1%), 원인을 다시 넣으면 실패가 돌아옵니다(reinsert → 82%).**
reverse가 normal보다도 큰 이유: 자기 궤적은 이미 GT에서 멀어져 있어, GT 문맥 끝에 자기 waypoint를 붙이면 비정상적으로 큰 암묵적 motion(점프)이
생기고 모델이 그 motion을 외삽합니다 — 이것 자체가 "직전 motion이 다음 waypoint를 결정한다"는 해석과 일치합니다.

### E. Motion semantics (마지막 문맥 waypoint 대체)

| 대체 | 증폭 | FDE5 | D10 |
|---|---:|---:|---:|
| GT (recent_gt) | 1.3% | 0.94 | 1.12 |
| **GT에서 5 cm 떨어진 다른 값 (motion ≈ GT)** | 1.2% | 0.98 | 1.17 |
| 방향 정답 + 크기 오답 | 64.2% | 12.26 | 8.92 |
| 크기 정답 + 방향 오답 | 79.9% | 26.78 | 27.36 |
| 크기만 맞춘 무작위 방향 | 81.5% | 58.00 | 84.20 |

- near_gt − recent_gt: 증폭 −0.1%p(p = 0.87) → **정확한 token identity가 아니라 motion이 GT와 가까우면 충분**합니다(AutoVLA 실험 11과 같음).
- 방향 오답 − 크기 오답: 증폭 +15.6%p(p = 2e-31), FDE +14.5 m, D10 +18.4 m(p ≈ 0) → **방향이 크기보다 중요**(AutoVLA 실험 32와 같음).
- 한계와 보완: 사전 등록 정의는 대체 크기 e를 "그 행 자신의 오차"로 두어, 행마다 궤적이 갈라진 뒤에는 e가 달라집니다(평균 e: 방향 오답 행 9.0 m, 크기 오답 행 5.6 m).
  **첫 대체 시점(waypoint 3)에서는 모든 행의 문맥이 같아 e가 정확히 같으므로**(탐색적), 그 시점의 즉각 반응을 비교하면 방향 오답이 다음 waypoint 오차를
  +0.29 m [+0.25, +0.33] 더 키웁니다(Wilcoxon p = 2e-99, 2,398 단위) — e를 맞춰도 방향 > 크기.

## 6. AutoVLA와 공통으로 재현된 현상

1. **직전 action 문맥이 다음 action을 결정**: 직전 1개만 GT로 바꿔도 증폭의 98% 제거(AutoVLA 93%).
2. **reverse로 실패 복원**: 교정된 문맥에 자기 오차 action을 다시 넣으면 증폭 0 → 82%(AutoVLA 4.4 → 33.4%).
3. **짧은 temporal critical window**: 1/2/3/4-step 교정이 full 효과의 24/39/91/95%(AutoVLA 47/76/90/96%). 둘 다 약 3 step에서 90%.
4. **motion이 identity보다 중요**: GT와 motion이 가까운 다른 값이면 GT와 같은 효과(AutoVLA 99%).
5. **방향 > 크기**: 크기를 맞추고 방향을 틀리면 더 증폭(AutoVLA +9.0%p, Impromptu +15.6%p; e를 맞춘 첫 시점에서도 성립).
6. **같은 크기의 편차가 방향·장면에 따라 통과/실패로 갈림**(task 수준, 탐색적; AutoVLA 실험 6의 "장면 맥락이 결정"과 유사하나 Impromptu에서는 방향 효과가 더 큼).

## 7. 재현되지 않은 현상

1. **궤적 수준의 흡수(안정 분기)**: AutoVLA에서는 같은 크기 편차의 상당수가 흡수되었지만, Impromptu는 0.2 m 편차의 97%가 최종적으로 3배 이상 커지고
   흡수(r ≤ 1)는 0.5%뿐입니다. 사전 등록 기준 3(궤적 수준 안정·불안정 공존 > 재실행 불일치)은 **충족되지 않았습니다**(2.2–3.2% < 5.9%).
2. **교정 해제 후 감쇠**: AutoVLA는 교정을 멈추면 서서히 재발산했지만, Impromptu의 win3·win4는 해제 후에도 D10 0.4–0.8 m로 안정적으로 유지됩니다(아래 §8 참고).

## 8. architecture 차이로 설명 가능한 부분

- **절대 위치 표현**: GT waypoint를 문맥에 넣으면 모델이 GT 궤적의 위치·추세를 그대로 받아(gt_history FDE 0.06 m) 해제 후에도 그 추세를 이어 갑니다.
  교정 효과의 일부는 "feedback 차단"이 아니라 "GT 정보 제공"입니다. 그래서 정보를 더하지 않고 원인만 다시 넣는 **reverse**와, GT 정보가 없는
  **motion 대체(E)**가 핵심 증거입니다 — 둘 다 같은 결론(직전 motion이 다음 waypoint를 결정)을 줍니다.
- **편차가 거의 항상 커지는 이유**: 절대 위치를 생성하는 모델은 앞 waypoint 간 차이(암묵적 motion)를 외삽합니다. 첫 waypoint의 0.2 m 편차는 곧 첫 step motion의
  오차이고, 이후 waypoint가 그 motion을 따라 누적되어 발산합니다. AutoVLA는 상대 motion token이라 한 step의 편차가 위치 오프셋으로 남을 뿐 motion 외삽은
  다음 token 선택에서만 일어납니다.
- **낮은 자연 성능**: 자연 A− 33%로 경계 장면이 많아, 작은 교란이 실패로 넘어가기 쉽습니다.

## 9. 통계적 유의성 / CI 요약

- C(correction): win1–4, gt_history, recent_gt 모두 normal 대비 증폭·D10·FDE 감소 p < 1e-23, log bootstrap CI가 0을 제외.
- D(reverse): reverse − gt_history 증폭 +81.9%p [+80.4, +84.2], D10 +40.5 m [+37.3, +43.9], p ≈ 0.
- A(task 수준): 교란 − 기준 A− +24.8%p [+19.4, +32.2](0.2 m), +31.1%p [+25.4, +39.1](0.5 m), 재실행 불일치 5.9%를 크게 넘음.
- E(motion): 방향 오답 − 크기 오답 증폭 +15.6%p(p = 2e-31); e를 맞춘 첫 시점 +0.29 m(p = 2e-99, 탐색적); near_gt ≈ GT(p = 0.87).
- B(분기, 사전 등록 기준): 공존 2.2% [1.5, 3.2] / 3.2% [2.4, 3.9] < 재실행 불일치 5.9% → 기준 미충족.

## 10. 최종 판정

사전 등록 기준(PROTOCOL.md §5)별 결과:

| 기준 | 결과 |
|---|---|
| 1. correction이 증폭·발산을 유의하게 줄임 | **충족** (recent_gt −62.2%p, gt_history −63.4%p) |
| 2. reverse가 발산을 유의하게 늘림 | **충족** (+81.9%p, +40.5 m) |
| 3. matched-magnitude 안정/불안정 분기 > 재실행 변동 | **미충족** (궤적 수준 안정 분기가 거의 없음; task 수준 공존은 탐색적으로 87–93%) |
| 4. 짧은 window (win1..4 단조 증가, win4 이내 80% 이상) | **충족** (24 → 39 → 91 → 95%) |
| 5. 방향 > 크기 | **충족** (+15.6%p; e 맞춘 첫 시점에서도 +0.29 m) |
| task 수준: 교란이 재실행 변동보다 큰 실패를 만들고 correction이 그 실패를 줄임 | **충족** (A− +25–31%p vs 기저 5.9%; 증폭 63.4 → 1.3%(recent_gt)) |

**판정: Strong replication** — 사전 등록 정의(기준 1·2 성립 + task 수준에서 교란이 기저 변동보다 큰 실패를 만들고 correction이 그 실패를 유의하게 줄임)를 충족합니다.
단, 다음 제한을 함께 명시합니다: (a) 여기서 "task 실패"는 AutoVLA와 같은 open-loop 궤적 실패(A−/증폭)이며 closed-loop 주행 실패가 아닙니다.
(b) 기준 3(궤적 수준 안정 분기)은 충족되지 않았습니다 — Impromptu에서는 작은 편차가 흡수되지 않고 거의 항상 커집니다.
(c) 절대 위치 표현 때문에 GT 문맥 교정에는 GT 정보 누출이 섞여 있으며, 이를 배제하는 근거는 reverse와 motion 대체 결과입니다.

### 질문: AutoVLA의 catastrophic error amplification은 특정 모델의 특성인가, autoregressive action-chunk generation의 일반적 failure mechanism인가?

**시간 순서가 있는 action chunk를 autoregressive하게 생성하는 구조에서 반복되는 일반적인 failure mechanism으로 보입니다.** 표현 방식(상대 motion codebook
token vs 절대 위치 digit token)과 학습 데이터·성능이 다른 두 driving VLA(AutoVLA, Impromptu VLA)에서 같은 인과 구조가 재현되었습니다:
직전 action 문맥을 고치면 증폭이 사라지고(93–98%), 잘못된 직전 action을 다시 넣으면 돌아오며, 약 3 step의 짧은 window가 있고, motion(특히 방향)이 token identity보다
중요합니다. 반대로 실험 28의 OpenVLA처럼 **시간축 action history를 autoregressive하게 생성하지 않는 구조**에서는 token 수준 feedback이 한 step 안에 머물고
궤적·과제 실패로 증폭되지 않았습니다. 증폭의 **정도**(작은 편차가 얼마나 흡수되는가)는 모델에 따라 크게 다릅니다 — AutoVLA는 상당수를 흡수하지만 Impromptu는
거의 흡수하지 못합니다 — 이는 action 표현(절대 위치 외삽)과 모델 정확도에 좌우되는 부분으로 보입니다.
