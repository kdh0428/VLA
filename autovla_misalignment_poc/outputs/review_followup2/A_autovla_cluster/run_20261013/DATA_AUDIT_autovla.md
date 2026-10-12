# DATA_AUDIT (AutoVLA·Impromptu): 독립 표본·개입·자료 분할 감사 (A1·A2, 2026-10-13)

**성격: 기존 파일의 사후(POST-HOC) 감사다.** 모델 추론·GPU 없음. 기존 파일은 수정하지 않았다.
표기: [확인한 사실] / [계산한 결과] / [미확인] / [가설].

- 코드: `scripts/review_followup2/a1_manifest.py`, `a2_overlap.py`
- 산출물:
  - `manifest_autovla_impromptu.parquet`, `manifest_columns.csv`, `manifest_keys.json`, `manifest_counts_by_experiment.csv`
  - `overlap_set_summary.csv`, `overlap_pairwise.csv`, `overlap_matrix_{drives,logs,scenes}.csv`
  - `drive_membership.csv`, `Aminus_log_to_drive.csv`, `overlap_checks.json`

## 1. A1 manifest

### 1.1 규모 [계산한 결과]

`manifest_autovla_impromptu.parquet`는 867,856행이다.
- 실험 13개: exp06/07/11/32/33, R8, R9, P1-A, P1-B, P1-C H7/H8/H9/H32.
- 범위: drive 44, log 92, 장면(= 초기 상태 cluster) 843.
- 1행 = (실험, 단위 pair_id, 개입, 결과 지표).

### 1.2 유일키와 열

- **유일키:** (experiment, pair_id, intervention, outcome_name, repeat_id). 중복 0행(`manifest_keys.json`).
- **필수 열:** 프롬프트 A1의 열을 모두 두었고, 해당 없는 값은 null이다. 열 정의, 누락률, 고유값 수는 `manifest_columns.csv`에 있다.
- **누락률:**
  - `simulator_seed` 1.0: 시뮬레이터가 없다(open-loop).
  - `complete_restore_hash` 1.0: 복원할 시뮬레이터·제어기 상태가 없다. 기록된 프레임에 대한 forward pass다.
  - `policy_rng_seed` 0.142: exp 6/7/11/32/33 행이다. P0-B units 표에 seed가 저장되지 않았다. 추정해서 채우지 않았다.
  - `block` 0.789: P1-C 이외 행이다.
  - `group_or_stratum`, `t_star` 0.066: Impromptu 행이다.
  - `outcome_value` 1e-6: invalid 출력의 FDE5다.

### 1.3 initial_state_cluster_id와 복원 검증

- **`initial_state_cluster_id`** = `navtest:<scene token>`. 물리적 초기 상태는 기록된 센서 프레임과 ego 상태다. 같은 장면이면 섭동·조건·모델이 달라도 같은 cluster다.
- **driving open-loop 실험에서의 복원 검증**
  - 시뮬레이터 상태 hash로 하지 않는다. 같은 코드·checkpoint·입력으로 다시 실행해서 검증한다.
  - R8R9: 220/220 행이 정확히 재현됐다 [확인한 사실, R8R9 analysis.md].
  - Normal의 harness 간 일치율: P0-A F6 기준 71.6–89.2% [확인한 사실].
- **반복 decode 없음:** 장면 × 조건마다 decode는 1회다(`repeat_id` = 0). 확률적 반복이 없으므로, 반복을 독립 초기 상태로 셀 위험은 driving 쪽에는 없다.
  - 대신 장면 안 섭동 단위(평균 7개)가 새 초기 상태가 아니다. 이것이 핵심이다. A−는 365 단위지만 장면은 52개다.

### 1.4 실험별 고유 수 [계산한 결과, `manifest_counts_by_experiment.csv`]

| 실험 | pair(단위) | 장면 = 초기 상태 | log | drive |
|---|---|---|---|---|
| exp07 / exp11 / exp32 / R8 / R9 / P1-B | 1,408 | 208 | 26 | 19 |
| exp06 | 1,824 (조건 행) | 208 | 26 | 19 |
| P1-A | 2,861 | 208 | 26 | 19 |
| exp33 Impromptu | 2,398 | 300 | 28 | 20 |
| P1-C (block마다) | 2,436 | 353 | 64 | 24 |

## 2. A2 drive와 초기 상태, 중복표

### 2.1 drive 정의

- **정의:** drive_id = log 이름의 앞 두 필드 `<date.time>_veh-NN`. [확인한 사실]
- 다음 두 정의와 같은지 행 단위로 확인했다(`overlap_checks.json`).
  - P1-C `split.csv`의 `drive`: 147행 모두 일치.
  - P1-C units의 `drive_id`: 일치.

### 2.2 "A−의 16 log = 13 drive" 확인 [계산한 결과, `Aminus_log_to_drive.csv`]

- 확인된다. 두 log가 한 drive에 속하는 경우가 3쌍이다.
  - `2021.05.25.14.24.08_veh-25` (00934_01067, 04059_04203)
  - `2021.06.03.13.55.17_veh-35` (00789_00999, 02866_03582)
  - `2021.09.29.14.44.26_veh-28` (00238_00320, 01059_01191)
- 다른 집합의 독립 단위 수:
  - A+ 26 log = 19 drive.
  - PoC 기전 집합 전체: 26 log = 19 drive.
  - Impromptu 실험 33: 28 log = 20 drive. PoC frame_index의 28 log 전체이고, AutoVLA와 같은 drive들이다.
- 장면 겹침: Impromptu 300 장면과 AutoVLA 기전 208 장면이 공유하는 장면은 18개다(A−와는 4개). 같은 log의 서로 다른 장면을 썼다.
- A− 장면은 drive 사이에 고르게 퍼져 있지 않다.
  - `2021.09.16.19.27.01_veh-45` 한 drive가 52 장면 중 15개를 차지한다.
  - `2021.06.28.18.03.27_veh-14`는 9개를 차지한다.

### 2.3 집합 요약 [계산한 결과, `overlap_set_summary.csv`]

| 집합 | log | drive | day-vehicle | 장면 |
|---|---|---|---|---|
| PoC 기전 단위 (exp 6–11/32, R8R9, P1-A/B) | 26 | 19 | 13 | 208 |
| 그중 A− | 16 | 13 | 10 | 52 |
| Impromptu 실험 33 | 28 | 20 | 14 | 300 |
| PoC dev_discovery (shards 0–5) | 28 | 20 | 14 | – |
| 선택 dev (expanded_best_of_n, shards 6–17) | 56 | 32 | 16 | 4,563 |
| 선택 held-out (heldout_best_of_n, shards 18–31) | 52 | 28 | 14 | 4,814 |
| navhard 출처 log | 76 | 34 | 15 | (합성 장면) |
| navhard 1차 절반 (실험 25) / 2차 절반 (실험 26) | 36 / 40 | 22 / 28 | 11 / 12 | – |
| P1-C eval (전체 1단계 / 분석 장면) | 64 / 64 | 24 / 24 | 16 | 5,671 / 353 |
| P1-C pilot | 2 | 2 | 2 | 10 |

### 2.4 공유 drive 수 (`overlap_matrix_drives.csv`; log·장면은 `overlap_matrix_logs/scenes.csv`)

| | PoC 기전 | A− | Impromptu | 선택 dev | 선택 held-out | navhard 출처 | P1-C eval | P1-C pilot |
|---|---|---|---|---|---|---|---|---|
| PoC 기전 (19) | 19 | 13 | 19 | 13 | 11 | 14 | **0** | 2 |
| 선택 dev (32) | 13 | 10 | 13 | 32 | **22** | 27 | 19 | 2 |
| 선택 held-out (28) | 11 | 8 | 11 | 22 | 28 | 24 | 17 | 2 |
| navhard 출처 (34) | 14 | 11 | 15 | 27 | 24 | 34 | 19 | 2 |
| P1-C eval (24) | **0** | 0 | 0 | 19 | 17 | 19 | 24 | **0** |

### 2.5 판정

1. **P1-C eval은 PoC 기전 집합과 drive가 겹치지 않는다(0/24).** 설계대로이고 이번에 확인했다. [계산한 결과]
   - Impromptu 집합, PoC dev_discovery 28 log, pilot과도 drive가 0개 겹친다.
   - 따라서 P1-C의 RN·RV 재현은 **"새 drive" 재현**이다. "새 log"보다 강한 의미다.
   - 다만 day-vehicle은 10개를 공유한다(2021.05.25 veh-25/35, 06.03 veh-35, 06.28 veh-14/26/38, 09.16 veh-42/45, 09.29 veh-28, 10.06 veh-52). 즉 drive는 분리됐지만 day/vehicle은 분리되지 않았다.
   - day-vehicle까지 분리한 부분집합 S2는 F 4 drive, P 7 drive뿐이다. 이 부분집합의 drive 수준 결과는 `CLUSTER_REANALYSIS.md` §2.3에 있다(사후 재계산).
2. **P1-C eval과 선택·navhard 자료는 겹친다(기전 실험이 아니다).** [계산한 결과]
   - P1-C eval 64 log는 모두 선택 실험(dev 32, held-out 32)에서 decode된 log다.
   - 장면 수준: P1-C 1단계 5,671 장면이 모두 선택 실험에서 natural plan으로 decode됐다. 분석된 353 장면도 모두 포함된다(dev 193, held-out 160).
   - 34 log(19 drive)는 navhard 출처 log다.
   - 이 자료들은 섭동·문맥 개입에 쓰이지 않았다. 자연(무개입) 결과만 알려진 상태다. 이 점은 P1-C leakage_check §5와 같다.
3. **선택 실험의 dev/held-out 분할은 log 수준에서만 분리됐고, drive 수준에서는 분리되지 않았다.** [계산한 결과, 새 발견]
   - 공유 log는 0개다.
   - 공유 drive는 22개다: held-out 28 drive 중 22개, day-vehicle 12개.
   - held-out 52 log 중 45개, held-out 4,814 장면 중 4,077개(85%)가 dev 선택 자료와 같은 drive에 속한다.
   - 실험 25/26의 navhard 두 절반도 공유 log는 0개지만 공유 drive는 16개다.
   - 따라서 사전 등록된 held-out(513fdba)과 navhard 2차 절반(96a595c)의 재현은 **"새 log" 재현이지 "새 drive" 재현이 아니다**. 같은 drive 안의 시간상 가까운 장면이 dev와 held-out 양쪽에 들어갔다.
   - drive-disjoint 부분집합은 held-out 7 log(737 장면, 6 drive)뿐이다. 이 부분집합으로 선택 규칙 효과를 다시 계산하려면 선택 분석 코드를 다시 돌려야 한다. 이번 범위(A: 기전 대비) 밖이라 **수행하지 않았다(검증 미수행).**
   - 원고에서는 선택 결과의 held-out을 "log-disjoint, not drive-disjoint"로 표기해야 한다.
4. **선택 dev/held-out과 PoC 기전 집합은 drive를 13/11개 공유한다.** PoC 장면은 0개 공유한다. 선택 규칙의 설계와 기전 결과 사이의 정보 흐름은 이 감사의 대상이 아니다. [미확인]
5. **P1-C pilot 2 log는 PoC 기전 집합과 drive 2개를 공유한다.** 설계상 dev 인접 drive를 pilot으로 돌렸다. pilot은 harness 점검에만 썼고 효과 분석에는 쓰지 않았다(P1-C protocol §6). [확인한 사실]

## 3. 실패·재시도·누락 (결과에 따라 제외된 것이 있는가)

| 실험 | 기록 | 처리 | 결과 기반 제외 |
|---|---|---|---|
| P1-C H8 (prev_action_state_patching, 93행 batch) | 본 실행 1차에서 2,436 단위 **전부** CUDA OOM이 났다. 외부 프로세스가 RTX 5090 메모리를 점유했다. 실패 출력은 `mech/_failed_attempt1_prev_action_state_patching/errors.jsonl`(2,436행, records 0)에 보존됐다. launcher의 오류율 정지 규칙이 1단계에만 적용돼 run이 멈추지 않았다. | **수정 2(9b3d4af, 2026-10-10)**: 선택된 353 장면의 영상을 다시 받아 H8만 같은 명령으로 재실행했다(attempt 2, 3,866 s, errors 0). 같은 동결 스크립트(sha256 60345ded…)로 분석을 다시 실행했다. H8 없이 돌린 첫 분석(RN·DM 결과가 열람됨)은 `analysis_incomplete_noH8/`에 보존돼 있고, 그 RN·DM 판정은 최종본과 같다. | 없음. 재실행 사유는 기술적 실패이고, family {RN, RV, DM}과 Holm m = 3은 그대로다. [확인한 사실] |
| P1-C 1단계 / 기타 block | 1단계 오류율 0(전 log). H7/H9/H32의 harness 제외는 0 장면. flagged log 0. 선택: 5,671 → eligible 2,191(no mismatch 3,268, t* > 7 212) → F 105 + S 248 = 353 장면, 2,436 단위. | – | 없음 [계산한 결과, denominators.csv] |
| P1-C pilot 점검 P1 | 85% 기준 미달(74.6%). | 수정 1(b216920)에서 기준을 arm-N 자기 재현성(76.2%)에 연동했다. 평가 자료 생성 전이다. | 해당 없음(평가 자료 없음) |
| R8R9 | 실패·재시도 없음. errors.jsonl 비어 있음. 220/220 정확 재현. | 원 수치를 아는 상태의 **복구 실행**이다. 확인 실험이 아니다. | 없음 |
| P1-A / P1-B | 본 실행 errors 0. `*.log.attempt1`은 첫 시도 log 사본이다(P1-A analysis.md). smoke 단계에 OOM 기록이 있다(`P1A .../smoke/p1a_fixedpos_oom_attempt*.log`). | smoke만 재시도했다. | 없음 [확인한 사실] |
| 실험 6 | A− 대안 1개가 invalid output이다. | 원 규칙대로 증폭 False로 포함했다. | 없음 |
| 실험 33 Impromptu | parse 실패 2단위. | 원 규칙대로 제외했다(기준 단위 300개도 정의상 제외). | 결과값이 아니라 parse 상태로 정한 제외다 |
| 실험 5/8/9/10 원 run | per-unit raw가 없다(P0-B). | 8·9는 R8R9로 복구했다. 5·10은 여전히 raw가 없다. | – |

## 4. 개입 정의 (GT 사용 여부와 counterfactual 참조)

| 개입 | 정의 | 미래 GT 사용 | 정상 counterfactual rollout 참조 | 온라인 배포 가능 |
|---|---|---|---|---|
| AutoVLA Recent-GT (실험 7/32, P1-C H7) | 문맥만 바꾼다. 매 step k ≥ t*+2에서 위치 k−1을 GT로 바꾼다. 한 번의 교정이 아니라 **반복 개입**이다(8−t*회, A− 평균 6.8회, 실제 token 변경은 평균 4.32회; P0-A F2). 실행 출력은 모델 자신의 token이다. | 예 | 아니오 | 아니오 |
| GT-history | 이전 action token 전부를 GT로 바꾼다. | 예 | 아니오 | 아니오 |
| win_w (실험 9, P1-C H9) | 고정 위치 t*+1..t*+w를 GT로 바꾸고 이후에도 유지한다. | 예 | 아니오 | 아니오 |
| AutoVLA reverse@emb (실험 8, P1-C H8) | GT-history 문맥에서 위치 k−1 embedding을 **같은 batch의 별도 Normal 행**에서 복사한다. Normal-branch token injection이다. | 예 | **예** | 아니오 |
| Impromptu reverse (실험 33) | 교정 문맥에 branch **자신의** k−1 waypoint를 다시 넣는다. | 예 | 아니오 | 아니오 |
| SpatialVLA closed-loop reverse | 정상 action을 실행하고 문맥만 교란한다. 참고용이며 B 파트 대상이다. | – | – | – |
| dir_wrong_mag_ok / dir_ok_mag_wrong | k−1에 GT와 자기 token으로 만든 대체값을 매 step 넣는다(방향 오답/크기 오답). | 예 | 아니오 | 아니오 |
| geo_nn_gt (실험 11) | k−1 = GT를 뺀 token 중 GT와 motion이 가장 가까운 token. | 예 | 아니오 | 아니오 |
| P1-A G/E 계열 | GT 없는 model-reference 교란과 교정이다. 정의는 P1-A protocol에 있다. | **아니오** | 예(모델 자신의 reference branch) | 아니오(reference가 필요) |

**"Reverse"는 모델마다 다른 처치다.** 모델 간 표에서 같은 행으로 비교하지 않는다.

## 5. 분할 경계 (요약)

- **PoC 기전 dev:** navtest shards 0–5, 28 log / 20 drive.
  - 기전 단위는 26 log / 19 drive에 있다. A−는 16 log / 13 drive다.
  - 모든 기전 설정이 이 집합에서 정해졌다.
- **P1-C eval:** shards 6–31의 64 log / 24 drive.
  - dev와 같은 drive인 42 log는 제외했고, 2 log는 pilot으로 돌렸다.
  - drive 수준으로 분리됐지만 day-vehicle은 10개를 공유한다.
- **선택 dev/held-out:** shards 6–17 / 18–31. log는 분리됐지만 **drive 22개를 공유한다.**
- **navhard 1차/2차 절반:** log는 분리됐지만 drive 16개를 공유한다.
- **같은 drive 안 장면의 train/test 분할:** 기전 결과에는 해당하지 않는다(무작위 분할을 하지 않음). 선택 결과의 held-out에는 사실상 해당한다(§2.5-3).
