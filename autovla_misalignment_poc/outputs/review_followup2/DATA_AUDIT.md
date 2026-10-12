# DATA_AUDIT — 독립 표본·개입·자료 분할·실패 기록 통합 감사 (2026-10-13)

**성격:** 기존 자료에 대한 사후(POST-HOC) 감사다. 모델 추론·GPU는 쓰지 않았다. 기존 원자료·결과 파일·옛 프로토콜은 수정하지 않았다.
이 문서는 아래 두 감사를 합치고, 실패·재시도·분할 경계·사전 등록 시점을 한곳에 정리한다. 세부 표와 코드는 각 run 디렉터리에 있다.

| 출처 | 내용 |
|---|---|
| `A_autovla_cluster/run_20261013/DATA_AUDIT_autovla.md` | AutoVLA·Impromptu manifest(867,856행), drive/log/장면 중복표 |
| `A_spatialvla_states/run_20261013/METHOD.md`, `FINDINGS.txt`, `cluster_counts.csv` | SpatialVLA 초기 상태 재생성, 3단계 cluster, 복원·drift |
| `B_spatialvla_direct/run_20261013/condition_definitions.csv`, `condition_record_check.csv` | SpatialVLA N/F/C/R 정의와 기록 대조 |
| `review_followup/P0A_audit/run_20261008/analysis.md` | 개입 정의(F1–F11)와 raw 존재 여부 |
| `review_followup/P1C_new_log_replication/design_20261008/protocol.md` (Amendment 1·2) | P1-C 수정·OOM 재시도 |
| `review_followup/P1D_dose_control/run_20261008/analysis.md` §0 | P1-D 중단된 첫 시작 |
| `git log` (저장소 `/root/VLA`) | 사전 등록 commit 시각 |

표기: [확인] 기록·코드로 직접 확인 / [계산] 이번 감사에서 계산 / [미확인] 자료가 없어 확인 불가.

---

## 1. 독립 단위의 정의

| 모델 | 독립 단위(추론 단위) | 그 아래 계층 (독립 표본으로 세지 않음) | 근거 |
|---|---|---|---|
| AutoVLA, Impromptu (open-loop) | **drive** = log 이름의 앞 두 필드 `<date.time>_veh-NN` | log segment → 장면(token, = 초기 상태 cluster) → 섭동 단위 → 조건 | P1-C `split.csv`의 drive 정의와 147행 모두 일치 [확인] |
| SpatialVLA (closed-loop) | **과제별 물리적 초기 상태 cluster** (아래 §3) | seed → episode → 조건 × 실행 설정 | A2 METHOD.md (cluster 정의·허용오차를 cluster 계산 전에 고정) |

- AutoVLA 조건 간 decode는 장면 × 조건마다 1회다(`repeat_id` = 0). 확률적 반복은 없다. 대신 장면 안 섭동 단위(평균 7개)가 새 초기 상태가 아니다. A− 365 단위는 52 장면, 16 log, **13 drive**다.
- day-vehicle(`<date>_veh-NN`)은 drive보다 상위 계층이다. 이번 재분석은 day-vehicle 상관을 모형화하지 않았다(PoC 집합 [미확인], P1-C에는 사전 지정 S3 부분집합이 있다).

## 2. AutoVLA·Impromptu: drive/log/장면 중복

### 2.1 집합 요약 [계산, `A_autovla_cluster/run_20261013/overlap_set_summary.csv`]

| 집합 | log | drive | day-vehicle | 장면 |
|---|---|---|---|---|
| PoC 기전 단위 (exp 6–11/32, R8R9, P1-A/B) | 26 | 19 | 13 | 208 |
| 그중 A− (AutoVLA primary family) | 16 | **13** | 10 | 52 |
| Impromptu 실험 33 | 28 | 20 | 14 | 300 |
| PoC dev_discovery (navtest shards 0–5) | 28 | 20 | 14 | – |
| 선택 dev (실험 20, shards 6–17) | 56 | 32 | 16 | 4,563 |
| 선택 held-out (실험 21–24, shards 18–31) | 52 | 28 | 14 | 4,814 |
| navhard 출처 log (실험 25–30) | 76 | 34 | 15 | 합성 장면 |
| navhard 1차 / 2차 절반 (실험 25 / 26) | 36 / 40 | 22 / 28 | 11 / 12 | – |
| P1-C eval | 64 | 24 | 16 | 1단계 5,671 / 분석 353 |
| P1-C pilot | 2 | 2 | 2 | 10 |

- "A−의 16 log = 13 drive"를 확인했다. 한 drive에 두 log가 들어간 경우가 3쌍이다(`2021.05.25.14.24.08_veh-25`, `2021.06.03.13.55.17_veh-35`, `2021.09.29.14.44.26_veh-28`).
- A− 장면은 drive 사이에 고르지 않다. `2021.09.16.19.27.01_veh-45` 한 drive가 52 장면 중 15개를 차지한다(방향 대조가 가중 방식에 민감한 원인).
- Impromptu 300 장면과 AutoVLA 기전 208 장면은 같은 drive들에서 왔고, 장면 18개를 공유한다(A−와는 4개).

### 2.2 공유 drive 수 [계산, `overlap_matrix_drives.csv`]

| | PoC 기전 (19) | A− (13) | 선택 dev (32) | 선택 held-out (28) | navhard 출처 (34) | P1-C eval (24) | P1-C pilot (2) |
|---|---|---|---|---|---|---|---|
| PoC 기전 | 19 | 13 | 13 | 11 | 14 | **0** | 2 |
| 선택 dev | 13 | 10 | 32 | **22** | 27 | 19 | 2 |
| 선택 held-out | 11 | 8 | 22 | 28 | 24 | 17 | 2 |
| navhard 출처 | 14 | 11 | 27 | 24 | 34 | 19 | 2 |
| P1-C eval | **0** | 0 | 19 | 17 | 19 | 24 | **0** |

### 2.3 판정

1. **P1-C eval은 PoC 기전 집합과 drive가 겹치지 않는다(0/24).** Impromptu 집합, dev_discovery, pilot과도 0개다. 따라서 P1-C의 Recent−Normal·Reverse−Full 재현은 **"새 drive" 재현**이다. 다만 **day-vehicle 10개를 공유**한다(drive 분리, day/vehicle 비분리). day-vehicle까지 분리한 S2 부분집합은 F 4 drive / P 7 drive뿐이다(사후 재계산은 `CLUSTER_REANALYSIS.md` §2.3).
2. **P1-C eval 64 log는 모두 선택 실험(dev 32, held-out 32 log)에서 natural plan으로 decode된 log이고, 34 log(19 drive)는 navhard 출처다.** 이 자료들은 섭동·문맥 개입에 쓰인 적이 없고 자연 결과만 알려진 상태였다(P1-C leakage_check §5와 같다).
3. **선택 실험의 dev/held-out은 log 수준에서만 분리됐고 drive 수준에서는 분리되지 않았다(새 발견).** 공유 log 0, 공유 drive 22(held-out 28 drive 중 22개), held-out 4,814 장면 중 4,077개(85%)가 dev와 같은 drive다. navhard 1·2차 절반도 공유 log 0, 공유 drive 16이다.
   - 따라서 사전 등록 held-out(513fdba)과 navhard 2차 절반(96a595c)의 재현은 **"새 log, 같은 drive 포함" 재현**이다. 원고 표기: "held-out logs (not drive-disjoint)".
   - drive-disjoint 부분집합은 held-out 7 log / 737 장면 / 6 drive뿐이다. 선택 규칙 효과의 재계산은 **수행하지 않았다(검증 미수행; 기전 논문의 핵심 주장 밖, GAP_DECISION #7).**
4. 선택 dev/held-out은 PoC 기전 집합과 drive를 13/11개 공유한다(장면 공유 0). 선택 규칙 설계와 기전 결과 사이의 정보 흐름은 감사 대상이 아니다 [미확인].
5. P1-C pilot 2 log는 PoC 기전 집합과 drive 2개를 공유한다. pilot은 harness 점검에만 썼고 효과 분석에는 쓰지 않았다 [확인].
6. 같은 drive 안 시간상 가까운 장면을 무작위로 train/test에 나눈 분할은 **기전 결과에는 없다**(무작위 분할을 하지 않음). **선택 결과의 held-out에는 사실상 해당한다**(3번).

## 3. SpatialVLA: 실험 × 과제별 고유 초기 상태 수

정의(METHOD.md, cluster 계산 전에 고정; RNG seed·실험·조건·실행 설정·`reconfigure` 여부·episode_id label은 **키가 아니다**):
- **primary** (`initial_state_cluster_id`) = 과제 + 목표 지정(지시문, source/target 물체) + asset id(물체 모델·scale, robot `urdf_version`, background overlay, scene) + 연속 상태(모든 동적 actor의 pose·속도, 로봇 root pose, joint qpos/qvel).
- **scene** = primary에서 지시문을 뺀 것.
- **physical** = 물리적으로 의미 있는 내용만(과제 family, 물체 모델·scale, 연속 상태). 시각용 `urdf_version`(visual mesh 경로만 다름; collision·inertia·joint 동일)과 지시문을 뺐다.
- 허용오차: 위치·quaternion·qpos 1e-4, 선속도 1e-3 m/s, 각속도·qvel 1e-2 rad/s. 근거는 float32 저장 정밀도(ulp ≤ 2.4e-7)와 환경 자체의 정지 판정 기준이다. cluster 결과를 본 뒤 바꾸지 않았다.

| 과제 | 실험 (seed / unit) | episode | primary | scene | physical |
|---|---|---|---|---|---|
| pick_coke_can | 실험 34 (0–39) | 360 | 40 | 40 | 40 |
| pick_coke_can | 실험 35 (0–119) | 1,680 | 120 | 120 | 120 |
| pick_coke_can | P1-D (200–319) | 2,400 | 120 | 120 | 120 |
| pick_coke_can | 세 실험 합집합 | 4,440 | 240 | 240 | 240 |
| move_near | 실험 34 (0–39) | 360 | 29 | 24 | **10** |
| move_near | 실험 35 (0–119) | 1,680 | 53 | 35 | **10** |
| move_near | 실험 35 0–39 / 40–119 | 1,040 / 640 | 29 / 47 | 24 / 34 | 10 / 10 |
| move_near | P1-D (120 units) | 2,400 | 120 | 40 | **10** |
| move_near | 세 실험 합집합 | 4,440 | 173 | 40 | **10** |

- **"move_near seed 0–119 = 서로 다른 초기 상태 53개"는 primary 수준에서 정확히 맞다**(재생성한 pose·asset으로 확인; seed 0–39는 29개). 그러나 **물리적 물체 배치는 10개뿐**이다. episode_id가 물체 triplet(5) × source/target 쌍(6, 지시문만 바꿈) × xy 배치(2)를 부호화하기 때문이다.
- **실험 34, 35, P1-D의 move_near는 모두 같은 10개 물리 배치를 쓴다.** P1-D의 "120 distinct configurations, 유효 n = 명목 n"은 primary 수준에서만 성립한다. 물리적으로 P1-D move_near는 실험 34/35와 독립이 아니다.
- coke는 seed마다 고유 상태다. 실험 34 ⊂ 실험 35(40개 공유), P1-D와는 겹치지 않는다(가장 가까운 두 coke 위치 사이 xy 1.2 mm).
- 같은 초기 상태의 반복(seed가 달라도 같은 cluster, 또는 같은 unit의 여러 조건·설정)은 stochastic 변동 추정에는 쓰되 **새 초기 상태 표본으로 세지 않았다.** B 재분석은 primary와 physical 두 clustering을 모두 보고한다.

### 3.1 restore / hash 정의

| 개념 | 목적 | 내용 |
|---|---|---|
| `initial_state_cluster_id` | "같은 물리적 시작 문제인가" | 위 정의, 허용오차 안 일치 |
| `complete_restore_hash` | "bit 단위로 이어서 실행할 수 있는가" | simulator 내부 상태(`get_state`: actor pose/속도, articulation qpos/qvel) + 제어기 target·gripper 상태(get_state 밖) + policy chunk/ensemble buffer·sticky-gripper 상태 + reset RNG(`RandomState(seed)`). decode는 greedy라 policy RNG는 없다 |

- **P1-D:** episode마다 `init.state_hash`(get_state float64 sha1)와 `img_hash`를 기록했다. 재생성한 hash가 120/120 unit(과제별)에서 일치하고, in-run `restore.ok` 3,840/3,840, null ≡ N 320/320이다. 제어기·gripper target은 같은 action prefix 재시뮬레이션으로 복원했고 executor는 T0에서 deep-copy했다. 제어기 내부를 직접 hash하지는 않았다 [확인].
- **실험 34:** state hash 기록 없음. 저장된 t = 0 frame이 재생성 관측과 coke 34/40, move_near 38/40에서 pixel 단위로 일치한다. 불일치는 물체 영역의 작은 차이(최대 2–19/255)이며 아래 drift로 설명된다. 실험 34에는 실제 TCP 기록도 없다.
- **실험 35:** 물체 상태 기록 없음. 지시문과 TCP `pose0`만 일치한다. 물체 상태는 seed → reset 코드 경로로 추정했고 run 기록으로 검증하지 않았다.
- **조건 쌍 복원:** 실험 34/35의 조건 쌍은 같은 cluster를 공유하지만 bit 단위로 복원 검증되지 않았다. P1-D만 복원 검증됐다.

### 3.2 drift (실험 34/35식 `reset(seed)`, reconfigure 없음)

- move_near: 6–9%의 reset이 bit 단위로 다르다. 허용오차를 넘는 것은 굴러가는 물체뿐이다(사과 최대 20×, 오렌지 최대 40× 허용오차; 최대 0.26 mm, quaternion 차 4.0e-3 ≈ 0.5°).
- coke: stepped episode 뒤에만 drift가 생긴다(27/102 reset이 다르고 7개가 허용오차 초과; 최대 4.5e-5 m).
- 규칙대로 해당 행을 "physical state uncertain beyond tol"로 표시했다: coke 실험 34/35 전부, move_near 30/40(실험 34)·95/120(실험 35).
- cluster 배정은 모호하지 않다(최대 drift 5.5× < 최소 cluster 간 거리의 절반 17.4×/2; move_near 같은 키 cluster끼리는 ≥ 2,000× 떨어짐).
- 실험 35의 분기 전 발산(2,400쌍 중 107쌍이 t < 8에서 발산; P1-D §0.3)의 그럴듯한 원인이지만 증명되지 않았다 [미확인].

## 4. 실패·재시도·누락·프로토콜 변경 (결과에 따라 제외된 것이 있는가)

| 실험 | 기록 | 처리 | 결과 기반 제외/변경 |
|---|---|---|---|
| **P1-C H8** (reverse@emb, 93행 batch) | 본 실행 1차에서 2,436 단위 **전부** CUDA OOM(외부 프로세스가 RTX 5090 메모리 점유). 실패 출력은 `mech/_failed_attempt1_prev_action_state_patching/errors.jsonl`(2,436행, records 0)에 보존. launcher의 오류율 정지 규칙이 1단계에만 적용돼 run이 멈추지 않았다 | **Amendment 2**(commit 9b3d4af, 2026-10-10 08:26 UTC): 선택된 353 장면 영상을 다시 받아 H8만 같은 명령으로 재실행(attempt 2, 3,866 s, errors 0). 같은 동결 분석 스크립트(sha256 60345ded…)로 재분석 | 없음. 다만 **H8 없이 돌린 첫 분석(파일 시각 08:24 UTC)에서 RN·DM 결과가 열람된 뒤** amendment 2가 commit됐다. 첫 분석은 `analysis_incomplete_noH8/`에 보존돼 있고 그 RN·DM 판정은 최종본과 같다. family {RN, RV, DM}, Holm m = 3은 그대로다 |
| **P1-C pilot 점검 P1** | 원 기준(Planner vs 저장된 arm N 10-token 일치 ≥ 85%) 미달: 74.6%(97/130) | **Amendment 1**(b216920, 2026-10-08 19:19 UTC, 평가 자료 생성 전): arm N 자기 재현성(76.2%)이 85%에 못 미쳐, 기준을 "arm-N 재실행보다 5%p 넘게 낮지 않음 + stub ≥ 95% + eligibility ≥ 90%"로 변경. 결정 규칙의 주 90% CI를 sign-flip 역산 CI로 명시. 분석 스크립트 고정 | 평가 자료가 없을 때의 변경. 그러나 **pilot 기준을 pilot 결과를 본 뒤 완화한 것**이므로 원고에 보고해야 한다. GPU 예산 재승인(e1a08f1, 10-09) |
| P1-C 1단계 / 기타 block | 1단계 오류율 0, H7/H9/H32 harness 제외 0 장면, flagged log 0. 선택: 5,671 → eligible 2,191 → F 105 + S 248 = 353 장면, 2,436 단위 | – | 없음 |
| **P1-D 중단된 첫 시작** | 2026-10-10 08:24:17 block 1 시작, model·env 초기화(08:25:38)까지만, episode 0개(`episodes.jsonl` 0 byte) | P1-C H8 재실행에 GPU를 비우려고 상위 세션이 의도적으로 중단. `rollouts/_aborted_start_20261010/`로 옮기고 10:05에 같은 인자로 재시작 | 없음(결과 0개). A2 cluster 계산에도 포함하지 않았다 |
| P1-D 본 실행 | 4,800 episode, `restore_ok` 100%, 제외 0. 실행 41시간(예상 25–35시간) | – | 없음 |
| R8R9 | 실패·재시도 없음, errors.jsonl 비어 있음, 220/220 정확 재현 | 원 수치를 아는 상태의 **복구 실행(재현성 감사)**. 같은 script·GPU·seed·단위 | 없음. 독립 확인 실험이 아니며 독립 표본을 더하지 않는다 |
| P1-A / P1-B | 본 실행 errors 0. smoke 단계 OOM 기록(`p1a_fixedpos_oom_attempt*.log`) | smoke만 재시도 | 없음 |
| 실험 35 seed 확장 | r1e4·r4e1을 seed 40–119로 확장 | 사전 등록(8fc20bf) 규칙에 따른 확장 | 규칙 기반. 단 확장분은 move_near primary 47 / physical 10 상태만 더했다(새 물리 배치 0) |
| 실험 34 natural_mix | 프로토콜은 "같은 batch"였으나 별도 block으로 실행됨(P0-A S4, 가설) | – | 약한 대조 |
| 실험 6 | A− 대안 1개 invalid output | 원 규칙대로 증폭 False로 포함 | 없음 |
| 실험 33 Impromptu | parse 실패 2단위 | 원 규칙대로 제외(결과값이 아니라 parse 상태) | 없음 |
| 실험 5 / 8 / 9 / 10 원 run | per-unit raw 없음(P0-A U1–U4) | 8·9는 R8R9로 복구. 5·10은 여전히 summary만 있음 | 실험 5의 1-token 교정 +55.8%p, 실험 10의 horizon-controlled 재발산은 cluster 재검정 불가 |

## 5. 개입 정의 (모델별; 같은 이름이 같은 처치가 아니다)

| 이름 | AutoVLA (exp 7/8/9/32, R8R9, P1-C) | Impromptu (exp 33) | SpatialVLA token level (exp 34 offline) | SpatialVLA closed loop (exp 34/35, P1-D) |
|---|---|---|---|---|
| **Recent-GT / recent_ref** | 문맥만 변경. 매 생성 step k ≥ t*+2에서 위치 k−1을 GT로 바꾸는 **반복(sliding) 개입**: 단위당 8−t*회(A− 평균 6.8회, 실제 token 변경 평균 4.3회). 실행 출력은 모델 자신의 token | 생성하는 waypoint마다 직전 문맥 waypoint를 GT로 바꿈(역시 매 step 반복; 절대 좌표라 GT 정보 누출 포함) | 직전 step token을 모델 자신의 greedy reference chunk 값으로(GT 아님) | – |
| **GT-history / full** | 이전 action token 전부를 GT로 | 전부 GT | 전부 reference | – |
| **win_w** (exp 9, P1-B, P1-C H9) | 고정 위치 t*+s..t*+s+w−1을 GT로 바꾸고 이후에도 유지(persistent). P1-B transient는 w개 forward에만 적용 | 첫 w개 waypoint를 GT로 | 1–2 step 전 교정(chunk 4 step 한계) | – |
| **Reverse** | reverse@emb: GT-history 문맥에서 위치 k−1의 embedding을 **같은 batch의 별도 Normal 행**에서 복사(Normal-branch token injection). 정상 counterfactual 참조 = 예 | 교정 문맥에 branch **자신의** k−1 waypoint를 다시 넣음. counterfactual 참조 = 아니오 | Normal 행의 token을 reference 문맥에 삽입 | **R**: 정상 step-1 translation token(g)을 실행하고 문맥에만 교란 token(p)을 넣음. 단 p 뒤에 생성된 token(step-1 회전·gripper, chunk step 2–4)은 실행 구조에 따라 실행된다(문맥 기인 실행 translation 0 / 0.10 / 0.32 / 0.044 m/ep: r1e1 / r2e1 / r4e1 / r1e4) |
| **Direction / Magnitude** | k−1에 GT와 자기 token으로 만든 방향 오답·크기 오답 대체값을 매 step 넣음 | 같은 개념, waypoint 수준 | 같은 개념, token 수준 | – |
| **geo_nn_gt** (exp 11) | k−1 = GT를 뺀 token 중 GT와 motion이 가장 가까운 token | near_gt: GT + 5 cm 무작위 방향 | near_ref: 한 bin 옆 token | – |
| **N / F / C** | – | – | – | **N** 정상(문맥 g, 실행 g). **F** 교란 실행 + 교란 문맥(p, p). **C** 교란 실행 + 정상 문맥(문맥 g, 실행 p). 교란 = step-1 translation token, `opposite`, d = 0.3; 실험 34/35는 t ∈ [8, 24)의 모든 chunk(r1/r2/r4 = 16/8/4 chunk), P1-D는 t = 8, 12, 16, 20의 4 chunk(모든 설정 동일) |
| **FD / CD** (P1-D r1e4) | – | – | – | F/C에 ensemble로 희석되는 몫 (1 − w0)Δ를 매 주입 step 실행에 더해 직접 실행 dose를 r1e1과 맞춤. 프로토콜(b212637, design B)에서 사전 고정, 모델 자신의 g·p로 online 계산; 결과로 rollout을 고르지 않음 |
| **P1-A G/E 계열** (AutoVLA) | GT 없는 model-reference 교란·교정. reference = 모델 자신의 무교란 rollout(B = 1, T = 0.01, 고정 seed). OC_refR = 직전 문맥 위치를 r[k−1]로 매 step 바꾸는 반복 개입(단위당 6.79회, 실효 4.54회). 교란 위치 t_p = t*와 A−/A+ 층은 GT 정의를 상속 | – | – | – |

- 미래 GT 사용: AutoVLA/Impromptu의 Recent-GT·GT-history·win·Direction/Magnitude·geo_nn_gt는 예, P1-A G 계열은 아니오(단 정상 counterfactual reference 필요 → 온라인 배포 불가). SpatialVLA closed loop은 GT를 쓰지 않는다(정상 token g = 모델 자신의 출력).
- **"Reverse"는 모델마다 다른 처치다.** 모델 간 비교표에서 같은 행으로 놓을 때는 처치 정의를 각 열에 적고, 효과 크기(+28.6 vs +81.9%p 등)를 같은 처치의 모델 간 차이로 비교하지 않는다. 증폭 정의도 다르다(AutoVLA/Impromptu: A− ∧ FDE5 > 3 m; SpatialVLA token: 누적 step 2–4 translation 편차 ≥ 주입 편차; OpenVLA: 하위 차원 bin 이동).

## 6. 분할 경계

| 분할 | 범위 | 분리 수준 | 비고 |
|---|---|---|---|
| PoC 기전 dev | navtest shards 0–5, 28 log / 20 drive (기전 단위 26 log / 19 drive, A− 16 log / 13 drive) | – | 모든 기전 설정·임계값이 여기서 정해짐 |
| 선택 dev / held-out | shards 6–17 / 18–31 (56 / 52 log) | log 분리, **drive 22개 공유** | held-out 장면의 85%가 dev와 같은 drive |
| navhard 1차 / 2차 절반 | 36 / 40 log | log 분리, **drive 16개 공유** | – |
| P1-C eval | shards 6–31의 64 log / 24 drive (dev와 같은 drive인 42 log 제외, 2 log는 pilot) | **drive 분리(0 공유)**, day-vehicle 10개 공유 | 선택·navhard 자료와는 겹치지만 그 자료는 개입에 쓰이지 않음 |
| SpatialVLA | 실험 34 ⊂ 실험 35 (coke·move_near primary); P1-D coke는 새 상태, P1-D move_near는 새 primary 상태지만 **같은 10개 물리 배치** | 물리 배치 수준 비분리 | P1-D move_near를 실험 34/35와 독립 표본으로 세지 않는다 |

## 7. 사전 등록 commit과 결과 열람 시점

모든 시각은 commit 시각(UTC, `git log`)이다. "사후"는 원 결과를 본 뒤 계획·수행된 분석이다.

| 실험 / 분석 | 사전 등록·고정 commit | 결과 commit / 열람 | 성격 |
|---|---|---|---|
| 실험 4–11 (AutoVLA 기전) | 없음(script와 결과를 함께 commit: 1541f30 09-26, bf15767 09-29) | 같은 commit | **탐색(발견) 실험.** 사전 등록 증거 없음(P0-A U6) |
| 실험 20 (선택 dev) | 규칙 commit e6d9dff 09-29 09:26 (정식 사전 등록 문서 없음) | f4cedfd 09-29 12:19 | 준-사전 |
| 실험 21–24 (선택 held-out) | 513fdba 09-29 20:12 | e3bc8db 09-29 21:48 | 사전 등록. 단 held-out은 drive 비분리(§2.3) |
| 실험 26 (navhard 2차 절반) | 96a595c 10-02 14:08 | ad7991c 10-02 17:56 | 사전 등록. drive 16개 공유 |
| 실험 27, 29, 30, 31 | 19c0130 / 0cffc15 / bf9aa21 (각 실행 전) | f98cb10 / 0ab33a4 / 86087e0 / 69402c6 | 사전 프로토콜 |
| 실험 32 (방향 > 크기) | af2eb10 10-03 15:10 | aee41e5 10-03 19:30 | 사전 프로토콜. 같은 PoC 단위(실험 7 결과를 본 뒤 설계) |
| 실험 28 (OpenVLA) | 6b9e8ad 10-03 14:42; 수정 6d0375a 19:32 (Phase B 일부 열람 후, 최종 분석 전) | 69402c6 10-03 23:55 | 사전 등록 + 결과 일부 열람 후 대조군 추가 |
| 실험 33 (Impromptu) | 038a6d3 10-04 16:51 | 698148a 10-05 01:11 | 사전 등록 |
| 실험 34 (SpatialVLA) | f578eae 10-05 12:55 | f4cf886 10-06 06:53 | 사전 등록 |
| 실험 35 (SpatialVLA ablation) | 8fc20bf 10-06 08:01; 분석 script 44266eb 10-06 10:28 (결과 분석 전) | 2b0212d 10-08 03:02 | 사전 등록. 실험 34 결과를 본 뒤 설계 |
| P0-A/B/C | (P0-B protocol 14:01 UTC, 첫 결과 14:06) | e10374a 10-08 14:14 | **사후** |
| P1-A, P1-B, R8R9, P1-C 설계, P1-D | b212637 10-08 15:24 (P0 결과를 본 뒤, 모든 본 실행 전) | P1-A/B/R8R9 08d54f3 10-09 13:14; P1-D 60509f6 10-12 21:08 | 사전 등록. **R8R9는 복구 실행(확인 실험 아님)** |
| P1-C | b212637 + Amendment 1 b216920 (10-08 19:19, 평가 자료 전) + Amendment 2 9b3d4af (10-10 08:26, **RN·DM 결과 열람 후**) | 1216812 10-10 10:06 | 사전 등록 + 두 수정(§4) |
| review_followup2 A–C (이 감사 포함) | 없음 | 589ded1 10-12 22:57 | **사후.** P1-C 행만 사전 등록 분석의 독립 재계산 |

- 이번 A–C 분석을 commit해도 기존 자료 분석이 사전 등록 실험으로 바뀌지 않는다. 사전 등록된 판정(P1-A–D, 실험 33–35)은 원래 판정대로 보고하고, drive/초기 상태 군집 재분석 결과는 별도 열로 둔다.

## 8. 검증 불가 항목

- 실험 5·10의 per-unit raw(1-token 교정 +55.8%p, horizon-controlled 재발산): summary로만 뒷받침된다.
- 실험 34·35의 실행 중 물체 상태(기록 없음, reset drift), 실험 34의 실제 TCP.
- P1-B window 적용의 본 실행 trace: smoke 수준 검증만 있다.
- AutoVLA 20-token 연장 자료, turn-rate baseline: 없다.
- 선택 결과의 drive-disjoint 부분집합 재계산: 수행하지 않았다.
- PoC 집합의 day-vehicle 수준 재분석: 수행하지 않았다.
