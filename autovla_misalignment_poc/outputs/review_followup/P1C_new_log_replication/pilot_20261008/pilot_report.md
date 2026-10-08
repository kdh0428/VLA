# P1-C pilot 보고서 (2026-10-08)

- 사전 등록: commit b212637 (`design_20261008/protocol.md`, `analysis.md` §6–8, 결정 D7).
- 범위: `pilot_harness_check` log 2개만 사용. 하네스 점검 전용이며 효과는 보지 않았다. 분석 스크립트는 effect 값을 가린 상태(`--mask-effects`)로 실행 여부만 확인했다.
  - `2021.09.29.19.02.14_veh-28_02911_03005` (shard 16, 22 장면)
  - `2021.10.06.08.16.17_veh-52_01590_01725` (shard 20, 30 장면)
- P1 확인에는 dev(PoC) 장면만 썼다. eval log는 내려받지도 처리하지도 않았다.
- GPU는 RTX 5090만 썼다(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, nice 19). P1-A/B 실행기(run_all_main)가 18:19:39Z에 끝난 뒤에 시작했다.
- git commit은 하지 않았다.
- 라벨: [확인한 사실] / [계산한 결과] / [미확인] / [가설].

## 0. 결론

**P1이 실패했다. 사전 등록 규칙(§6)에 따라 본 실행을 막아 두었다.** `PILOT_OK` 파일을 쓰지 않았으므로 실행기가 시작을 거부한다.

| 점검 | 결과 | 판정 |
|---|---|---|
| P1: Planner natural vs `full_extract` arm N, 10-token 전체 일치 | 30 장면 중 22 (73.3%). 130 장면으로 늘리면 97/130 = 74.6% (Wilson 95% 0.665–0.813) | **실패** (기준 ≥ 85%) |
| P1: stub id가 다른 비율 | 0/130 (0%) | 통과 (기준 ≤ 5%) |
| P2 (protocol): H8 debug | patch@emb와 recent_gt의 logits L1 = 0, 0. patch@L35와 gt_history의 logits L1 = 0, 0. selfpatch JS = 0. prefix-KV checksum assert 위반 0건 | 통과 |
| P2 (사용자): 기록 완결성과 조건별 분모 | 선택 10 장면, 10 장면 / 76 unit 생성. H7, H8, H9, H32 모두 76/76 unit, 오류 0 | 통과 |
| P3 (protocol): 끝에서 끝까지 실행 | 다운로드 → 3-카메라 추출 → 1단계 → 선택 → prune → ED unit → H7/H8/H9/H32 → 정리, 모두 무오류 | 통과 |
| P3 (사용자): 카메라 3개로 충분한가 | 아래 세 근거 참고 | 통과 |
| ED wrapper parity | dev 3 장면에서 alternatives 3/3 동일, 조건 행 27/27 동일(저장된 dev ED와 bit 단위로 같음) | 통과 |
| 분석 스크립트 | 합성 31/31, dev 재현 4/4 통과. pilot 출력에서 masked로 실행됨 | 통과 |

P3(카메라) 근거:
- 모델 코드가 front, left, right 3개 카메라만 읽는다: `camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']`.
- PoC 5 장면을 CAM_F0/L1/R1만 있는 symlink tree로 돌린 결과가 원래 디렉터리로 돌린 결과와 같았다. token id 5/5 동일, entropy 차이 0.
- P1 비교 자체가 3-카메라 디렉터리로 돌린 결과를 8-카메라 시기의 `full_extract` 기록과 비교한 것이다.

## 1. P1 실패 분석 (dev PoC 장면 130개, [계산한 결과])

`checks/p1_detail.json`에 저장했다. 같은 장면을 두 가지 절차로 다시 디코딩했다.
- **Planner:** 사전 등록된 1단계 절차.
- **armN_generate:** `full_extract` pass 1을 그대로 다시 구현한 것. `p1c_natural_pass_armN.py`. HF `generate` 한 번으로 stub과 action을 함께 생성한다. 장면별 seed를 쓴다.

| 비교 | 10-token 전체 | 첫 token | t* | A± 라벨 | eligible+stratum |
|---|---|---|---|---|---|
| Planner vs 저장된 arm N | 74.6% [.665, .813] | 96.9% | 89.2% | 98.5% | 93.1% |
| armN_generate vs 저장된 arm N | 76.2% [.681, .827] | 96.9% | 91.5% | 98.5% | 93.8% |
| Planner vs armN_generate | 82.3% [.748, .879] | 100% | 96.2% | 98.5% | 95.4% |

해석 [가설, 근거 있음]:
- **arm N 절차를 그대로 다시 실행해도 저장된 arm N과 76%만 일치한다.** 따라서 85% 기준은 어떤 하네스로도 달성할 수 없는 값이다.
- 원인은 T = 0.01 샘플링에서 거의 동점인 logit이 RNG에 좌우된다는 점이다. `full_extract`는 run 전체에 seed를 한 번만 주었고, 장면 사이에 arm C도 돌렸다. 그래서 장면 하나만 떼어 같은 RNG stream을 재현할 수 없다.
- 이를 뒷받침하는 사실 [확인한 사실]:
  - stub은 100% 같다.
  - 같은 접두부(stub과 pred)에서 출발하면 ED 연속은 저장값과 bit 단위로 같다(27/27).
  - Planner의 반복 실행은 결정적이다(P3 5/5 동일).
- 같은 범위의 기존 값: P0-A A5에서 dev 하네스 간 Normal 일치는 71.6–89.2%였다.
- 사전 등록 §3의 "Stub = Planner … equals the arm-N procedure"는 stub 수준에서는 맞고, 10-token 수준에서는 위와 같은 비율로만 맞는다.

## 2. 프로토콜 개정이 필요한 사항 (평가 데이터를 보기 전에 날짜를 적은 개정 commit 필요)

1. **[필수] P1 기준.** 현재 기준(≥ 85%)으로는 본 실행이 불가능하다. 선택지:
   - **A (권고).** 1단계 절차는 Planner로 유지한다(사전 등록 절차, 기전 스크립트와 같은 span 재계산 계열). P1 기준을 "arm N 자기 재현성 기준"으로 바꾼다. 예:
     - Planner와 저장된 arm N의 일치가 armN_generate와 저장된 arm N의 일치보다 5 pp 넘게 낮지 않을 것 (관측 74.6 vs 76.2)
     - stub id 일치 ≥ 95% (관측 100%)
     - eligible+stratum 일치 ≥ 90% (관측 93.1%)
     - 개정 시 관측값을 함께 기록한다.
   - **B.** 1단계를 `p1c_natural_pass_armN.py`로 바꾼다. 저장된 arm N과의 일치는 76.2%로 나아지지 않는다. 그리고 runaway와 CoT 판정이 `full_extract` 규칙을 따르게 된다. 이것도 개정이 필요하다.
   - 개정을 commit한 뒤, lead가 `pilot_20261008/PILOT_OK`를 직접 쓴다. 내용에는 개정 commit hash와 근거(`checks.json`, `checks/p1_detail.json`)를 적는다. 실행기는 이 파일 없이는 시작하지 않는다.
2. **[필수] GPU 예산.** 승인 범위는 5–7.5 h인데, 현재 GPU 공유 상태에서 추정치는 9.35 h다(§4). 둘 중 하나가 필요하다.
   - 재승인을 받는다. 그러면 1.5배 중단선은 14.0 h가 된다(`gpu_estimate.json`).
   - GPU가 비어 있을 때 실행한다. 그 경우 추정치는 약 5.5 h다(dev 실행 속도 기준).
3. **[해석 확정 필요] 결정 규칙 §9 "90% CI".** 분석 스크립트는 sign-flip test-inversion 90% CI를 쓴다. sign-flip p가 유의성을 결정하기 때문이다. bootstrap 90% CI를 쓴 판정도 같은 행에 함께 기록한다. 이 선택을 개정에 명시해야 한다.
4. **[명시 권고] 실행 순서.** 모든 shard의 1단계 → 선택 → prune을 먼저 끝내고, 그다음에 단위 생성과 기전 블록 4개를 선택된 장면 전체(약 359 장면)에 한 번만 실행한다.
   - 이렇게 하면 모델 로드가 26×6회에서 26+5회로 준다.
   - 선택된 장면의 이미지(장면당 12장)만 남긴다.
   - estimand와 설계는 바뀌지 않는다.
   - §10의 "shard별 오류율 > 5%" 중단은 1단계에 적용한다. 기전 블록의 장면 오류는 §7의 제외 규칙으로 집계한다.
5. **[명시 권고] 세부 정의.**
   - S4 flag의 분모는 "eligible 장면 + 1단계 오류 장면"이다. 1단계 오류 장면은 eligible 여부를 알 수 없기 때문이다.
   - Q3 Δ는 sign-flip 없이 bootstrap과 cluster-t로만 보고한다. 프로토콜은 CI만 요구한다.
   - Planner 절차에서는 runaway가 구조적으로 0이다(action token을 정확히 10개만 생성).
   - 1단계 key는 `"0:{token}"`이다. stub seed가 실험 20의 seed와 같다.

## 3. 구현 (새 파일만 추가, 기존 스크립트 무수정)

모든 파일은 `/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c/`에 있다. dev 스크립트의 sha256은 설계 manifest와 같다(`manifest.json` → `dev_scripts_unchanged_vs_design_manifest`).

| 파일 | 역할 |
|---|---|
| `p1c_common.py` | split, eligibility(t* ≤ 7), F/S 층화, sha256 순서(`P1C-20261008|token`), m = 4, π와 w |
| `p1c_natural_pass.py` | D7a. 1단계 writer. `_planner.Planner.plan`을 수정 없이 쓰고, stub id만 `generate` 출력에서 기록한다. `full_extract` arm-N 형식에 stage-1 라벨을 더한다. 재개 가능 |
| `p1c_natural_pass_armN.py` | 개정 선택지 B용 대안(`full_extract` arm N 재구현). 현재 실행기에는 연결하지 않았다 |
| `p1c_select.py` | stage-1만 사용해 log별 F 전부와 S 4개를 뽑는다. 분모, 선택 token, 필요한 이미지 목록을 쓴다. 덮어쓰기를 거부한다 |
| `p1c_build_units.py` | D7b. ED wrapper. `ED.select_alternatives`와 `ED.a_label`을 그대로 쓰고, 장면별 GPU loop는 원본을 그대로 복사했다. token 목록(선택 파일)을 입력으로 받는다. `--check-dev`로 parity를 시험한다 |
| `p1c_driver.sh` | D7c. 스트리밍 driver(아래 가드 참고) |
| `p1c_disk.py` | 모든 삭제를 담당한다(prune, remove, clean-tmp) |
| `p1c_info.py` | shard 순서, log 목록, 예산 계산 |
| `analyze_p1c.py` | D7d. 고정된 분석 스크립트. sha256 `60345dedfc9d3aeb14d21ec74590c6bb9a00a386b8962754f615be58b69cfc94` |
| `test_analyze_p1c.py` | 합성 데이터 테스트 |
| `test_dev_p1c.py` | dev 데이터 테스트 |
| `p1c_pilot.sh`, `p1c_pilot_checks.py` | pilot 실행과 점검 |
| `p1c_main_launcher.sh` | 본 실행기 |

`p1c_driver.sh`의 가드:
- 경로는 모두 literal이거나 `${VAR:?}`이고, prefix를 검사한다. `cd`를 쓰지 않는다.
- 삭제는 `p1c_disk.py`만 한다. 삭제하기 전에 다음을 모두 확인한다.
  - realpath가 예상 경로와 같다.
  - split.csv의 role이 eval 또는 pilot이다.
  - PoC 보호 목록에 없다.
  - fetch ledger에 있다(이 파이프라인이 옮겨 온 log).
- 삭제할 때마다 `DISK_CLEANUP.md`에 먼저 기록한다.
- 디스크 여유를 확인한다: 시작 전에 3 GB + 1.25×예산, 진행 중에는 watchdog이 3 GB 미만을 감시한다.

`analyze_p1c.py`가 구현하는 것:
- 추정량: θ_F, θ_S, θ_P(Hajek), Δ = θ_F − θ_S. 장면 가중이 주 분석이고, unit 가중과 drive 균등 가중은 민감도다.
- 검정: drive 수준 exact sign-flip(meet-in-the-middle, 2^24 경우를 0.0초에 계산). Holm은 F family α 0.05, P family α 0.025.
- 구간: drive block bootstrap(B = 10,000), cluster-t, test-inversion CI.
- 민감도 부분집합:
  - S1: navhard와 무관한 log
  - S2: dev와 날짜·차량이 겹치지 않는 log
  - S3: day-vehicle 단위 cluster
  - S4: flag된 log 제외
  - 공통 포함 집합
- 보조 분석:
  - FDE5 대비, recovery 대비
  - window 수준, gap fraction f(w), win4 동등성 검정, f(3) ≥ 0.75 점검
  - drive별, log별, 도시별 효과와 forest plot
  - H32 보조 조건
  - OOD entropy 규칙(보고만 하고 적용하지 않음)
- 원시 행은 plan §2와 protocol §11의 필드를 모두 담아 `units.parquet`에 저장한다.

테스트 결과:
- **합성 데이터, 31/31 통과.**
  - exact == brute force
  - H0에서 1종 오류 0.049 (α .05), 0.0085 (α .0083)
  - Hajek 편향 +0.0001
  - CI 경계에서 p = 0.050
  - 합성 효과의 결정이 맞게 나온다(RN, RV는 replicated, DM은 inconclusive)
  - S3의 cluster 수가 16 이하
  - mask가 작동한다
- **dev 데이터, 4/4 통과.** θ_F와 θ_S가 P0-B 값과 소수점 6자리까지 같다.
  - RN A−: −0.397894
  - RN A+: −0.033527
  - DM A−: +0.098520
  - DM A+: +0.045024
  - dev A− cluster는 13 drive로, 설계 문서와 같다.

## 4. 측정한 실행 시간과 갱신한 추정 ([확인한 사실]: 측정값, [계산한 결과]: 추정)

조건: 외부 프로세스가 5090에 약 14–15 GB를 차지하고 있었고, pilot 단계 동안 GPU 사용률은 약 94%였다.

| 단계 | 측정값 | 본 실행 추정 |
|---|---|---|
| 다운로드와 추출(전체 shard 스트리밍) | shard 16: 36 s (3.25 GB 전송), shard 20: 112 s | 26 shard 약 0.6 h (GPU 아님) |
| 1단계 natural | 3.08 s/장면, shard당 로드 오버헤드 약 78 s | 5,671 장면 → 5.41 h |
| ED unit 생성 | 5.37 s/장면 | 359 장면 → 0.55 h |
| H7 action_history_causal | 1.57 s/unit | 약 2,728 unit(장면당 7.6) → 1.19 h |
| H8 prev_action_state_patching | 1.62 s/unit | 1.23 h |
| H9 temporal_feedback_window | 0.69 s/unit | 0.52 h |
| H32 motion_semantics_ablation | 0.47 s/unit | 0.36 h |
| 2단계 모델 로드 | 약 75 s/회 | 0.1 h |
| **합계** | | **9.35 GPU h** (1.5배 중단선 14.0 h). GPU가 비어 있으면 약 5.5 h |

- pilot에서 쓴 GPU wall time은 약 0.6 h였다(P1 확장 130 장면 포함, `gpu_time.jsonl`).
- 기록 크기: H8이 unit당 약 96 KB이므로 본 실행에서 약 0.26 GB, 나머지는 합계 약 0.1 GB다.

## 5. 디스크

- 실행 전 여유 공간: 9.1 GB.
- pilot 이미지: 3-카메라, 2 log, 약 0.29 GB. prune 후 0.023 GB를 유지했고, 끝난 뒤 모두 삭제했다(`DISK_CLEANUP.md` 6행).
- 관측된 최소 여유 공간: 8.69 GB (18:23Z). 이때는 P1-A/B 출력도 늘어나고 있었다.
- 실제 추출량은 641 kB/frame 예산의 약 1.17배였다. 그래서 driver의 사전 확인 배수를 1.25로 올렸다.
- 본 실행 예상 최대 사용량: 가장 큰 shard 20의 eval 이미지 약 2.6 GB + 유지 이미지 약 0.9 GB + 출력 약 0.4 GB → 최소 여유 약 5 GB 이상이 예상된다(3 GB 하한보다 높다).
- 현재 상태: `sensor_blobs/test`는 원래의 dev log 32개, 여유 공간 8.6 GB.

## 6. pilot 분모 (stage-1, 효과 아님)

| log | 장면 | eligible | F | S eligible | S 표본 | 부적격 사유 |
|---|---|---|---|---|---|---|
| veh-28_02911_03005 | 22 | 5 | 1 | 4 | 4 | no_mismatch 16, t* > 7 1 |
| veh-52_01590_01725 | 30 | 8 | 1 | 7 | 4 | no_mismatch 22 |

- 기전 블록 4개 × 76 unit이 모두 생성되었고, 오류는 0이다.
- 조건별 분모: H7 7조건, H8 93행, H9 17행, H32 6조건, 모두 unit 76개다(`checks.json`).

## 7. 본 실행 명령 (개정과 `PILOT_OK`, 그리고 `analyze_p1c.py`를 포함한 p1c 스크립트 commit 이후에만)

```bash
nohup bash /root/VLA/autovla_misalignment_poc/scripts/review_followup/p1c/p1c_main_launcher.sh \
  > /root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/run_20261008_launcher.out 2>&1 &
```

- 실행기는 시작 전에 다음을 확인한다.
  - `pilot_20261008/PILOT_OK`와 `gpu_estimate.json`이 있다.
  - p1c 스크립트가 git에 commit되어 있고 HEAD와 차이가 없다.
- shard는 eval_order_key 순서로 하나씩 처리한다(6 25 27 29 9 12 16 21 7 23 13 15 11 20 26 18 17 31 28 14 19 10 8 24 30 22).
- 모델을 로드하기 전마다 다음을 기다린다: GPU 여유 ≥ 13 GB, RAM ≥ 6 GB, P1-A/B 실행기 없음.
- 재개할 수 있다. 끝난 단계는 건너뛰고, 부분 출력은 덮어쓰지 않는다. 동시에 두 개가 실행되지 않도록 flock을 건다.
- 중단 코드: 10 디스크, 11 1단계 오류율 > 5%, 12 GPU 1.5배 초과, 13 gate 미충족.
- 끝나면 분석을 한 번 실행한다: `python analyze_p1c.py --run .../run_20261008 --out .../run_20261008/analysis`.

## 8. 산출물

- `checks.json`: P1–P3, parity, 완결성, 실행 시간.
- `checks/p1_detail.json`: 130 장면 일치표.
- `gpu_estimate.json`, `gpu_time.jsonl`, `manifest.json`.
- `logs/`: driver, pilot, 단계별 로그, 디스크 샘플, 합성·dev 테스트 로그.
- `analysis_masked/`: effect를 가린 실행 확인용 출력.
- `stage1/`, `selection/`, `units/`, `mech/`: pilot 원시 기록.
- `../DISK_CLEANUP.md`.
- 참고: `../work_scene_index.json`은 navtest 장면→log 색인 캐시다.
