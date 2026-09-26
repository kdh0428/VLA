# ORION vs AutoVLA — P / R / A failure decomposition

저장된 inference 결과만 사용 (GPU 미사용). 두 모델에 **동일한 라벨링 함수**를 적용했습니다. 재현: `python run_pra_comparison.py && python make_report.py` (`/root/VLA/pra_comparison`).

## 1. 결론

**3단계 분류로는 두 모델 모두 reasoning이 최대 원천이지만, 실패가 일어나는 방식은 다릅니다.** 실제 행동이 틀린(A−) 샘플 중 perception은 맞고 reasoning이 틀린 비율은 ORION 78.2% [70.1, 83.1], AutoVLA 50.0% [41.3, 55.4]이며, §3의 8개 민감도 설정 모두에서 최대입니다.

- **ORION — 결정 단계에서 발생.** perception이 맞은 A−(판정 가능 481건) 중 결정이 틀렸고 planner가 그 결정을 그대로 실행한 순수 결정 오류 265건(55.1%), 결정은 맞았는데 실행이 달라진 순수 interface 99건(20.6%), 둘 다 117건(24.3%). 정지 중 'keep'이 미해결인 100건은 제외.
- **AutoVLA — 결정 오류와 interface 이탈이 겹침.** perception이 맞은 A−(판정 가능 95건) 중 순수 결정 오류는 13건(13.7%)뿐이고, 실행이 선언과 다르게 나간 경우가 82건(86.3%; 순수 interface 31, 결정도 틀림 51)입니다. CoT가 말한 결정과 실제 action token의 결합이 약하며, 3단계 분류의 reasoning 비율에는 이 복합 실패가 들어 있습니다.
- **P+R+A− (결정은 맞았는데 다르게 행동)**: ORION 98건 (A− 중 15.9% [11.7, 22.3]), AutoVLA 31건 (A− 중 24.2% [17.4, 34.6]).
- **Perception (P−)**: ORION 6.0% [3.0, 9.9], AutoVLA 25.8% [17.4, 32.7]. 대부분 hallucination에서 오며, miss만 보면 ORION 1.9%, AutoVLA 10.9%로 떨어집니다.
- 두 모델의 차이는 **실패의 양**입니다: A− 비율 ORION 19.8%, AutoVLA(강제 CoT) 5.2%, AutoVLA(자연 생성) 1.9%.
- 별도로, 두 모델 모두 **말한 결정과 실제 행동의 결합이 느슨합니다**: 행동은 맞는데 선언한 결정이 GT와 다른 P+R−A+가 ORION 38.5%, AutoVLA 42.4%로 모든 그룹 중 가장 큽니다. reasoning 텍스트를 action의 설명으로 읽으면 안 된다는 뜻입니다.

## 2. 요청하신 핵심 표

비율은 P/R/A가 모두 판정 가능한 샘플 기준, 괄호는 clip(ORION) / log(AutoVLA) 단위 cluster bootstrap 95% CI.

| Group | ORION | AutoVLA |
|---|---:|---:|
| P-R-A- | 24 (0.8% [0.2, 1.4]) | 25 (1.0% [0.4, 1.5]) |
| P+R-A- | 483 (15.4% [8.5, 23.4]) | 64 (2.6% [1.8, 3.2]) |
| P+R+A- | 98 (3.1% [2.0, 4.6]) | 31 (1.3% [0.9, 1.8]) |
| P+R+A+ | 1148 (36.7% [27.5, 45.9]) | 824 (33.3% [28.9, 39.1]) |
| 판정 가능 / 전체 | 3129 / 3132 | 2472 / 2604 |

### A− 샘플의 발생 단계

| 단계 | 정의 | ORION | AutoVLA |
|---|---|---:|---:|
| perception | A− ∧ P− | 37 (6.0% [3.0, 9.9]) | 33 (25.8% [17.4, 32.7]) |
| reasoning | A− ∧ P+R− | 483 (78.2% [70.1, 83.1]) | 64 (50.0% [41.3, 55.4]) |
| interface | A− ∧ P+R+ | 98 (15.9% [11.7, 22.3]) | 31 (24.2% [17.4, 34.6]) |
| 합계 A− | | 618 | 128 |

### reasoning 실패(P+R−A−)의 내용

R 기준별 실패 건수 (한 샘플이 여러 기준에 걸릴 수 있음).

| R 기준 | ORION | AutoVLA |
|---|---:|---:|
| R_d 종방향 결정이 GT와 다름 | 462 | 56 |
| R_d 횡방향 결정이 GT와 다름 | 60 | 16 |
| R_c 위험 하 감속 판단 실패 | 47 | 3 |
| R_c 정지 중인데 이동 중이라 서술 | 13 | 1 |
| R_c 이동 중인데 정지라 서술 | 0 | 1 |
| R_a 가까운 선행차 미언급 | 4 | 1 |
| R_a 보행자 위험 미언급 | 3 | 4 |
| R_b 없는 차량 근거 | 5 | 2 |
| R_b 없는 보행자 근거 | 0 | 0 |
| **결정(R_d)만 틀림** | 417 / 483 (86.3%) | 54 / 64 (84.4%) |
| **실행 종방향 클래스 = 선언 결정** | 304 / 383 (79.4%) | 23 / 64 (35.9%) |

두 모델 모두 reasoning 실패는 객체 인식·hallucination이 아니라 **결정 기준(R_d)**에서 옵니다. 그러나 실행이 그 틀린 결정을 따르는지는 모델마다 다릅니다:

- **ORION**: P+R−A−의 79.4%에서 실행 궤적이 틀린 선언 결정을 그대로 따릅니다. interface는 말한 대로 실행했고, 오류는 결정 단계에서 이미 발생했습니다.
- **AutoVLA**: P+R−A−에서 실행이 선언을 따르는 비율은 35.9%뿐입니다. 결정이 틀린 데다 실행도 그 결정과 다르게 나간 복합 실패가 다수이므로, reasoning 비율만 보고 interface 문제가 작다고 해석하면 안 됩니다.

### 결정 정오 × 실행이 선언을 따르는가 (P+ ∧ A−)

선언과 실행을 같은 accept 함수로 판정하므로 '결정이 맞고 실행이 따름'은 A−가 될 수 없습니다. 종방향은 선언 클래스(정지 중 'keep'은 해소된 클래스)와 실행 클래스를, 횡방향은 선언이 있을 때만 비교합니다.

| 셀 | 의미 | ORION | AutoVLA |
|---|---|---:|---:|
| ok/diverges | 결정 맞음 · 실행이 다름 → 순수 interface | 99 (20.6%) | 31 (32.6%) |
| wrong/follows | 결정 틀림 · 실행이 따름 → 순수 결정 오류 | 265 (55.1%) | 13 (13.7%) |
| wrong/diverges | 결정 틀림 · 실행도 다름 → 복합 | 117 (24.3%) | 51 (53.7%) |
| ok/follows | 결정 맞음 · 실행이 따름 (불가능해야 함) | 0 (0.0%) | 0 (0.0%) |
| unresolved | 정지 중 'keep' 미해결 | 100 | 0 |

## 3. 민감도 — 결론이 라벨 선택에 얼마나 의존하는가

A− 중 perception / reasoning / interface 비율.

| 설정 | ORION P / R / I | ORION P+R+A− | AutoVLA P / R / I | AutoVLA P+R+A− |
|---|---|---:|---|---:|
| 기준 (공통 기준, 속도비례 허용오차) | 6.0% / 78.2% / 15.9% | 98 | 25.8% / 50.0% / 24.2% | 31 |
| ORION에 신호등 GT 반영 (AutoVLA는 신호등 GT 없음 → 동일) | 17.5% / 69.1% / 13.4% | 83 | 25.8% / 50.0% / 24.2% | 31 |
| 허용오차 0 (경계 flip도 실패로 계산) | 6.7% / 80.7% / 12.6% | 120 | 23.5% / 54.9% / 21.5% | 74 |
| P를 miss만으로 판정 (hallucination 제외) | 1.9% / 80.6% / 17.5% | 108 | 10.9% / 60.2% / 28.9% | 37 |
| 정지 중 'keep'의 모호한 경우를 R+로 (상한) | 6.0% / 65.7% / 28.3% | 175 | 25.8% / 50.0% / 24.2% | 31 |
| 신호등 반영 + 모호한 정지 'keep' R+ | 17.5% / 56.6% / 25.9% | 160 | 25.8% / 50.0% / 24.2% | 31 |
| 선언 STOP/DECELERATE 상호 인정 | 6.1% / 77.0% / 16.9% | 103 | 27.5% / 49.2% / 23.3% | 28 |
| AutoVLA도 3 s horizon으로 평가 | 6.0% / 78.2% / 15.9% | 98 | 27.0% / 54.1% / 18.9% | 14 |

reasoning이 최대 단계라는 결론은 모든 설정에서 유지됩니다. 가장 크게 움직이는 것은 ORION의 interface 비율로, 정지 상태에서 선언한 'keep'을 어떻게 읽느냐(§5 R_d)에 달려 있습니다 — 모호한 경우를 R+로 두면 상한값이 됩니다.

## 4. 전체 8개 그룹

| Group | ORION | AutoVLA |
|---|---:|---:|
| P+R+A+ | 1148 (36.7%) | 824 (33.3%) |
| P+R+A- | 98 (3.1%) | 31 (1.3%) |
| P+R-A+ | 1206 (38.5%) | 1048 (42.4%) |
| P+R-A- | 483 (15.4%) | 64 (2.6%) |
| P-R+A+ | 58 (1.9%) | 122 (4.9%) |
| P-R+A- | 13 (0.4%) | 8 (0.3%) |
| P-R-A+ | 99 (3.2%) | 350 (14.2%) |
| P-R-A- | 24 (0.8%) | 25 (1.0%) |
| 판정 불가 | 3 {'P-R?A?': 1, 'P+R?A?': 2} | 132 {'P?R-A+': 14, 'P+R?A+': 86, 'P?R+A+': 8, 'P-R?A+': 16, 'P+R?A-': 7, 'P-R?A-': 1} |

## 5. 라벨 정의 (두 모델 공통)

**A (action)** — 계획 궤적의 의미적 행동. 두 모델 궤적을 같은 좌표계(+right, +forward, 0.5 s)로 변환해 ORION의 `trajectory_metrics.action_semantics` 하나로 STOP/DECELERATE/MAINTAIN/ACCELERATE 및 STRAIGHT/LEFT/RIGHT를 구합니다. action token ID는 사용하지 않고, L2 임계값도 쓰지 않습니다.
- 실행 행동의 종방향 클래스는 **GT 시작 속도(실제 현재 속도) 기준**으로 계산합니다.
- GT 궤적에서 허용오차 안에 도달 가능한 클래스면 정답: 속도 max(0.5 m/s, 15%), 횡방향 1 m / 10°. 정지 중 0.25 m 미만 이동의 heading은 무시합니다(pose jitter).
- horizon: ORION 3 s, AutoVLA 5 s(모델 고유). 3 s 통일은 §3 민감도.

**P (perception)** — 모델의 *장면 설명* 텍스트(ORION: scene description + critical objects, AutoVLA: Scene Description + Critical Object Description) vs GT 장면. 공통 변수: 차선 내 선행차량, 전방 보행자/자전거 위험.
- miss: GT에 엄격 영역 내 존재하는데 설명에서 인지하지 않음 / hallucination: 구체적으로 주장했는데 GT에 완화 영역 내에도 없음.
- 일반론·가정형 언급("vehicles can proceed", "watch for any pedestrians")은 주장으로 보지 않습니다.
- 신호등은 ORION만 GT가 있어 공통 기준에서 제외, §3에서 반영.

**R (reasoning)** — 모델의 *추론/결정* 텍스트. 네 기준을 모두 만족해야 R+.
- R_a 중요 객체: GT 위험(선행차 ≤15 m, 보행자 위험, [신호등 설정] 적색등)을 추론에서 언급
- R_b hallucination: 존재하지 않는 위험을 근거로 주장하지 않음 (신호등 설정: 적색인데 녹색이라 주장하지 않음)
- R_c 상황 판단: 위험 하에 GT가 감속하면 결정도 감속/일치, ego 자신의 운동 상태를 틀리게 서술하지 않음
- R_d 최종 결정: 선언 결정을 **A와 똑같은 accept 함수**로 판정. 따라서 선언 = 실행이면 R_d와 A 판정이 항상 같고, P+R+A−는 '말한 것과 다르게 행동한' 경우만 셉니다 (위반 0건 자동 검증).
- 정지 중 선언한 'keep'은 단어만으로 해석 불가: 진행 논리가 있으면 진행, 정지 의도가 있으면 정지, 둘 다 없으면 미해결(기준은 R−, 상한은 §3).
- 자유 텍스트 GT reasoning은 쓰지 않습니다(AutoVLA에 없음). ORION도 같은 GT 장면 규칙으로 판정합니다.

## 6. AutoVLA 비교 조건

AutoVLA의 자연 생성(arm N)은 2,748개 중 1개만 CoT를 내므로 P/R을 볼 텍스트가 없습니다. 그래서 P/R/A는 **강제 CoT(arm C) 한 번의 생성**에서 CoT와 그 생성의 궤적을 함께 사용했습니다 (arm C의 CoT와 arm N의 action을 짝짓지 않음). max_length 도달·action 토큰 폭주 샘플 {'truncated_or_runaway': 143, 'short_trajectory': 1}은 제외했습니다. arm C는 반사실 조건이며 A− 비율이 자연 생성보다 높습니다 (5.2% vs 1.9%).

## 7. 라벨 검증

규칙 기반 라벨이므로 각 버전마다 무작위 표본을 원문과 대조해 읽고 규칙을 고쳤습니다. 발견해 수정한 오류:
- 정지 GT에서 고정 ±1 m/s 허용오차가 출발(0→8 m/s)까지 정답 처리 → 속도비례 허용오차
- 선언과 실행을 다른 규칙으로 판정해 선언=실행인데 R+A−가 발생 → 단일 accept 함수
- 예측 궤적 자신의 첫 속도 기준 클래스 → GT 시작 속도 기준
- ORION이 정지선에서 "keep … until closer to the intersection"(=계속 진행)을 'keep=정지 유지'로 인정 → 추론 텍스트로 해소
- "the traffic light ahead … vehicles can proceed", "a pedestrian crossing"(횡단보도)이 객체 주장으로 오인 → 일반론·가정형 차단
- AutoVLA가 ego를 "the vehicle"로 부르는 것을 타 차량 언급으로 오인 → 제거
- 정지 중 sub-mm pose jitter의 heading이 GT를 STOP+RIGHT로 만듦 → 0.25 m 미만 무시

최종 버전 표본 판독 결과: ORION P+R+A− 8건 중 7–8건, AutoVLA P+R+A− 8건 중 8건이 실제 '말한 것과 다르게 행동'한 사례(AutoVLA 중 3건은 정지 vs 1.4 m/s 감속 수준의 경미한 차이). P− 판정은 텍스트에서 객체를 뽑는 가장 약한 부분으로, ORION·AutoVLA 모두 5건 중 3–4건이 실제 오류였습니다.

## 8. 한계

- R은 규칙 기반입니다. LLM judge나 사람 라벨이 아니며, 위 판독은 소표본입니다.
- 공통 기준은 신호등을 볼 수 없습니다. ORION의 적색등 관련 reasoning 오류("currently green")는 신호등 설정에서만 잡히며, AutoVLA의 신호등 언급은 GT가 없어 검증 불가입니다.
- ORION GT(Chat-B2D)는 *critical* 객체만 나열하므로, 좌표로 주장된 객체는 20 m 이내에서만 hallucination으로 판정합니다.
- ORION P+R+A−는 clip 집중도가 높습니다(적색등 대기 장면). CI가 cluster bootstrap인 이유입니다.
- ORION 궤적은 QA 샘플링 때문에 실행마다 달라집니다(이번 추출의 frame별 재시드 run 사용).

## 9. 다음 단계용 subset

`outputs/subsets/<MODEL>/<group>.json` — 파일명 규칙 `+`→`p`, `−`→`m` (예: `PpRpAm.json` = P+R+A−). 각 샘플에 `sample_id`, `cluster`, `tensor_file`(추출 텐서 경로), A 판정 근거, P/R 실패 사유가 들어 있어 causal/internal analysis에 바로 쓸 수 있습니다. 전체 샘플별 라벨은 `outputs/labels_<MODEL>.jsonl`.

| 파일 | ORION | AutoVLA |
|---|---:|---:|
| `PpRmAm.json` (P+R−A−) | 483 | 64 |
| `PpRpAm.json` (P+R+A−) | 98 | 31 |
| `PpRpAp.json` (P+R+A+) | 1148 | 824 |
| `PmRmAm.json` (P−R−A−) | 24 | 25 |
