# P1-C 설계 분석 (CPU 전용; GPU·추론·다운로드 없음)

(설계 agent가 작성한 내용을 상위 세션이 저장했다. agent 환경에서 보고서 파일 쓰기가 막혔다.)
라벨: [확인한 사실] 파일에서 읽음 / [계산한 결과] 여기서 계산 / [미확인] 확인하지 않음 / [가설] 해석.

## 1. Split [계산한 결과]

**사용하지 않은 navtest log는 없다 [확인한 사실].** navtest 136개 log 모두 이전 출력에 등장한다(`outputs/`에서 log 이름과 token을 스캔).
- PoC 28개: 모든 기전 실험에 사용.
- shard 6–17의 56개, shard 18–31의 52개: 선택·탐지·PDM·navhard에만 쓰였고, 교란·기전 실험에는 쓰지 않았다.

**cluster는 log segment가 아니라 drive(`<date.time>_veh-NN`) 단위여야 한다.**
- PoC가 아닌 segment 108개 중 44개가 PoC segment와 같은 drive에 속한다.
- P0-B가 "16 log"로 센 A− cluster는 실제로는 13개 drive다.

| 역할 | log | drive | 비고 |
|---|---|---|---|
| dev_discovery (PoC) | 28 | 20 | 기존 결과. 설정은 모두 여기서 고정 |
| eval (shard 6–31, dev와 drive 분리) | 64 | 24 (day-vehicle 16) | 5,671 장면 |
| pilot_harness_check | 2 | – | dev와 drive가 인접. 배관 점검용이며 분석하지 않음 |
| excluded_drive_adjacent_to_dev | 42 | – | 사용 안 함 |
| reserve_non_navtest_not_used | 11 | – | 전부 Boston. 사용 안 함 |

- **발견 단계** = 기존 PoC 결과(실험 6–9, 32)와 P0-A/P0-B. 조건, 단위 구성, 결과 지표, estimand, 분석 방법을 모두 여기서 고정한다.
- **평가 단계** = eval 64 log. 새로 조정하는 설정이 없으므로 별도 튜닝 집합은 필요 없다. pilot은 배관 점검만 한다.
- **shard 18–31만 쓰지 않는 이유:** 기전 질문에 대해서는 shard 6–17과 18–31 모두 똑같이 미사용이다. 한쪽만 쓰면 cluster가 약 12–13 drive로 절반이 된다.
- 계획용 라벨(기존 5090 natural 행) 기준 eval의 eligible A−(t* ≤ 7)는 111 장면, 20 drive이고 특정 drive에 몰려 있다(한 drive 21개, 세 drive 12–13개). eligible A+는 2,090 장면이다.

## 2. 누출 점검

**통과:**
- eval log 중 기전 실험에 쓰인 것: 0개.
- eval drive 중 dev와 겹치는 것: 0개(설계상). pilot drive도 eval과 겹치지 않는다.
- PoC에서 파생된 frame 장면 중 eval log에 속하는 것: 0개.
- 합성(synthetic) 장면은 쓰지 않는다.

**주의 표시:**
- **navhard 출처:** eval 64개 중 34개가 navhard 합성 장면의 출처 log다. navhard에서는 선택 규칙만 조정했고 기전 설정은 조정하지 않았다. 민감도 분석용으로 navhard와 무관한 30 log(13 drive) 부분집합을 둔다.
- **같은 날·같은 차량:** eval 64개 중 41개가 dev와 같은 날·같은 차량이다(day-vehicle 16개 중 10개, eval A− 113 장면 중 69개). 이들을 빼면 drive가 7개만 남아 너무 적다. 그래서 겹치지 않는 부분집합 분석과 day-vehicle 단위 cluster 분석을 민감도로 사전 지정한다.
- **natural 라벨은 이미 본 적이 있다:** eval 장면의 natural(교란 없는) 라벨은 실험 20–24에서 이미 확인했다. 이것은 P1-C의 결과 지표가 아니라 baseline 공변량이다. strata는 1단계 natural pass를 새로 돌려 다시 정한다. A− 라벨의 약 18%가 바뀔 것으로 예상한다(3080 Ti와 5090의 A− 일치가 50개 중 41개).

## 3. 모집단 표집

- **frame:** 5,671 장면.
- **eligible:** 유효한 natural 계획이 있고 t* ≤ 7인 장면. 계획용 라벨 기준 2,201 장면(39%)이다. 이탈이 전혀 없는 장면은 3,257(57%), t*가 8–9인 장면은 211(4%)이다.
- **F stratum:** eligible A− 장면 전부(π = 1).
- **S stratum:** log segment마다 eligible A+ 장면 m = 4개를 단순 무작위 추출(token sha256 순서). π = m_seg / N_S,seg.
- **estimand:**
  - θ_F: 실패 조건부 재현.
  - θ_P: F ∪ S를 Hajek 가중으로 합친 값(첫 natural 이탈이 있다는 조건부).
  - θ_S와 Δ = θ_F − θ_S로 실패 선택 효과와 모집단 효과를 분리한다.
- A−는 eligible 장면의 약 5%다. 그래서 dev 효과 크기를 가정하면 모집단의 Recent−Normal은 약 −5%p, F에서는 −40%p가 된다.

## 4. 시뮬레이션 모델

- **효과:** P0-B의 단위별 paired 차이(실험 7 Recent−Normal, 실험 32 Direction−Magnitude; 1,408 단위).
- **cluster:** 실제 eval 24 drive와 segment별 실제 eligible F/S 개수를 썼다. 두 cluster 모델을 둔다.
  - drive 모델: drive당 dev log 1개(보수적).
  - segment 모델: segment당 dev log 1개.
- **A+ 재가중:** dev A+는 t*를 A−에 맞춰 뽑았으므로 eval의 t* 분포로 재가중했다(t* = 0/1/2/3/4–7에 0.58/0.58/1.30/2.72/3.69).
- **Reverse−Full:** raw가 없어 Recent−Normal 형태의 noise를 요약 CI 폭 비율로 축소해 썼다(A− 0.81, A+ 0.82). 중심은 A− +29.0%p, A+ +3.1%p다. [가설 수준의 근사]
- **가설 시나리오:** H0, MME, dev 효과, dev 효과의 절반. 검정력과 CI 폭은 시뮬레이션한 H0 분포로 보정했다.
- **sign-flip 보정 점검(1,000회):** 명목 5%에서 F 1종 오류 4.1–6.5%, P 1종 오류 6.4–10.7%.
- **한계:**
  - 이질성 추정이 dev 13 drive에서만 나온다.
  - 효과 평균을 옮기면 단위 수준에서 불가능한 값이 생길 수 있다.
  - Reverse−Full은 근사다.
  - 실제 strata는 계획용 라벨과 다를 것이다.

### 검정력 표 (2,000회, 가용 drive 24개 전부, 장면 가중, 보정 후, Holm 첫 단계 기준; 범위는 보수적 drive 모델 / segment 모델)

| Estimand | 대비 | dev 효과 | MME | 95% 반폭 | dev 효과에서 검정력 | ½ dev에서 | MME에서 |
|---|---|---|---|---|---|---|---|
| F | Recent−Normal | −39.8 pp | −10 | ±17.2 / ±13.3 | 0.99 / 1.00 | 0.48 / 0.72 | 0.12 / 0.21 |
| F | Reverse−Full (근사) | +29.0 | +10 | ±13.9 / ±10.7 | 0.96 / 1.00 | 0.40 / 0.62 | 0.19 / 0.33 |
| F | Direction−Magnitude | +9.9 | +5 | ±12.6 / ±10.1 | 0.19 / 0.33 | 0.05 / 0.07 | 0.05 / 0.07 |
| P | Recent−Normal | ≈ −5.1 | −2 | ±2.2 / ±1.7 | 1.00 | 0.38 / 0.55 | 0.22 / 0.30 |
| P | Reverse−Full (근사) | ≈ +4.3 | +2 | ±1.8 / ±1.4 | 1.00 | 0.40 / 0.59 | 0.33 / 0.48 |
| P | Direction−Magnitude | ≈ +4.8 | +2 | ±6.2 / ±4.1 | 0.16 / 0.40 | 0.05 / 0.11 | 0.04 / 0.09 |

## 5. MME 근거 (새 결과 없이 고정)

- **F, Recent−Normal −10%p와 Reverse−Full +10%p:**
  - 원래 증폭됐을 실패(기준 약 47%)의 약 1/5이다.
  - Normal과 Full의 차이(약 43%p)의 약 1/4이다.
  - dev leave-one-log-out 변동 폭보다 작다.
- **F, Direction−Magnitude +5%p:** 교란 20개당 증폭 1개가 늘어나는 크기이고, dev leave-one-log-out 하한(+8.1)의 절반이다.
- **P, 세 대비 모두 ±2%p:**
  - 기준 6–7% 대비 약 30% 상대 변화다.
  - 이 프로젝트에서 의미 있다고 본 실험 20의 선택 이득이 첫 이탈 장면 기준 약 −1.7%p에 해당한다.
- **목표 반폭:** ≤ 0.61·|MME|이면 Holm 첫 단계에서 검정력 80%가 된다. F의 모든 대비와 P의 Direction−Magnitude는 이 목표에 도달할 수 없다.

## 6. 선택한 N과 이유

- **N:**
  - 64 log / 24 drive 전부.
  - eligible A− 장면 전부(약 111개).
  - log segment마다 A+ 4 장면(248개, 가중치 1/π).
  - 장면마다 거리 매칭 단위 전부.
  - 합계 약 359 장면, 2,425 단위.
- **N은 고른 값이 아니라 가용성으로 정해진다.** 정밀도는 drive 수가 결정한다.
  - F Recent−Normal 반폭은 drive 8/12/16/20/24개일 때 ±30/24/21/18/17%p다.
  - 목표(±6%p)는 가상으로 drive 48개를 써도 도달하지 못한다(±12%p).
- **m = 4:** m을 16으로 늘려도 보수적 모델에서 P Recent−Normal 반폭이 ±2.2에서 ±2.1%p로만 줄어든다. drive 간 분산이 지배적이기 때문이다.
- **장면당 단위 전부:** 장면당 3개만 써도 정밀도는 같다. 하지만 전부 쓰면 dev와 같은 단위 정의를 유지할 수 있고, GPU 비용은 약 0.8시간 늘어난다.
- **이 설계로 보일 수 있는 것:** dev 크기의 Recent−Normal과 Reverse−Full 효과가 실패 선택 집합 밖에서도 유지되는지(두 estimand 모두), 그리고 실패 집합과 모집단의 차이.
- **보일 수 없는 것:** MME 크기의 효과는 확립할 수 없다. Direction 대조는 결론이 나지 않을 가능성이 크다.
- **추론 방법:**
  - 주 p값은 drive 수준 exact sign-flip이다.
  - 보정 점검에서 F의 1종 오류는 4–6.5%, P는 명목 5%에서 최대 10.7%였다. 그래서 family P는 α = 0.025를 쓴다.
  - cluster-t와 percentile 구간은 실제 포함률이 낮다(1종 오류 8–14%).
- **N을 바꿀 수 있는 것은 독립 drive를 더 확보하는 것뿐이다.**
  - navtest 밖 11개 log를 더해도 최대 약 4 drive이고, 모두 Boston이다. NAVSIM이 이 log들을 제외한 이유는 [미확인].
  - trainval/navtrain log는 학습 오염 가능성이 있고 [가설], 수백 GB 다운로드가 필요하다.
  - 권고: 확장하지 않고, P1-C를 가용성으로 크기가 정해진 재현 실험으로 보고한다.

## 7. GPU와 데이터 예산

측정 출처는 `run_meta.json`과 run log다.
- `bon_20.log`: 779 장면 21.4분.
- `full_extract`: 6.26 s/장면.
- ED: 208 장면 6.05분.
- 실험 7 / 8 / 9 / 32: 1,408 단위당 19.5 / 20.5 / 6.4 / 5.4분.
- 모델 로드 약 1분(P0-A 소규모 재실행, 57–65 s).

예상 시간:

| 단계 | 시간 |
|---|---|
| 1단계 natural pass | 2.6시간 (5,671 × 1.65 s, 17행 기준 측정이라 1행 pass의 상한) |
| 거리 매칭 단위 생성 | 0.17시간 |
| 기전 스크립트 4개 | 1.5시간 |
| 모델 로드 | 약 1.0시간 (shard를 묶을 때) – 2.6시간 (shard별) |
| pilot | 약 0.5시간 |
| **합계** | **약 5–7.5시간, 최악 약 15시간** (다운로드 시간 미포함) |

- 영상 데이터: 실험 20–24 이후 카메라 영상을 지웠으므로 shard 6–31을 다시 받아야 한다.
  - 전송량은 약 100 GB(shard 크기 [미확인])이고, shard 하나씩 스트리밍한다.
  - eval log의 CAM_F0/L1/R1만 남기면 총 20.2 GB, 한 번에 최대 2.24 GB다.
- 현재 남은 디스크는 9.0 GB다.
- shard별 영상 예산은 `work/shard_budget.json`에 있다.

## 8. 승인이 필요한 결정

- **D1. 평가 집합:** dev와 drive가 겹치지 않는 64 log 전부를 쓴다. navhard 출처 log와 같은 날·같은 차량 log도 포함하고, 각각 민감도 분석으로 따로 본다.
- **D2. 모집단 범위:** 첫 natural 이탈이 t* ≤ 7인 장면으로 한정한다. 이탈이 없는 57%는 비율로만 보고한다. 이들에 교란을 주입하는 stratum은 새로운 개입이므로 넣지 않는다.
- **D3. 주 가중 방식:** P0-B처럼 장면 가중으로 한다. drive 균등 가중이 더 정밀하고 보정도 낫지만(F Recent−Normal ±13.1 vs ±17.2) 다른 estimand이므로 데이터를 보기 전에 정해야 한다.
- **D4. Reverse:** 실험 8의 reverse@emb를 유지하고 "Normal-branch token injection"으로 표시한다. 자기 참조 4조건 설계는 P1-A에 둔다.
- **D5. Direction 대조:** 계획서 §8대로 두 primary family에 모두 넣고 검정력 부족을 감수할지, 보조로 내릴지 정해야 한다.
- **D6. family α:** F는 0.05 + Holm, P는 0.025 + Holm.
- **D7. 실행 전 필요한 코드 변경:**
  - `full_extract`의 arm-N 형식으로 쓰는 natural 전용 1단계 writer. 또는 `full_extract.py`를 그대로 쓰되 비용이 약 4배.
  - `equal_distance_perturbation.py`에 token 목록을 직접 지정하는 옵션(현재 규칙은 A− 전부 + t*를 맞춘 A+ 3개).
  - 3-카메라 스트리밍 driver(경로 가드 포함).
  - 고정된 분석 스크립트.
  - 이후 pilot 점검 P1을 통과해야 한다: Planner natural pass와 `full_extract` arm N의 일치 ≥ 85%.
- **D8. 다운로드:** 약 100 GB 스트리밍.
- **D9. GPU 예산:** 약 5–7.5시간. 예상의 1.5배를 넘으면 기술적으로 중단하고 점검한다.

## 9. 미확인

- shard 크기와 다운로드 대역폭.
- 1행 natural pass의 실제 소요 시간.
- 기전 스크립트에 카메라 3개면 충분한지. teacher forcing에서는 됐고, pilot 점검 P3에서 확인한다.
- Reverse−Full의 분산 구조.
- 실제 효과의 drive 내 상관.
- NAVSIM이 navtest 밖 11개 log를 제외한 이유.
