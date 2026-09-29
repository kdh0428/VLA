# 인계 문서 — 새 서버에서 이어서 실험하기

마지막 갱신: 2026-09-29 (실험 20까지). 결론은 [CONCLUSIONS.md](CONCLUSIONS.md), 재현 기록은 [autovla_misalignment_poc/REPRODUCTION.md](autovla_misalignment_poc/REPRODUCTION.md).

## 1. 지금 어디까지 왔나

| # | 실험 | 상태 | 보고서 |
|---|---|---|---|
| 1–10 | 원 PoC (P/R/A 분해 ~ horizon-controlled window) | 완료, 새 서버에서 핵심 수치 재현 확인 | `CONCLUSIONS.md` §1–9 |
| 11 | 직전 토큰 identity 분해 (motion vs embedding) | 완료 | `autovla_misalignment_poc/outputs/prev_action_identity_decomposition/PREV_ACTION_IDENTITY.md` |
| 12 | 비-oracle 참조 안정화 (CTRA, 이전 frame 계획) | 완료 | `.../reference_stabilization/REFERENCE_STABILIZATION.md` |
| 13 | receding-horizon 재계획 (로그 관측, pseudo closed loop) | 완료 | `.../receding_horizon_replanning/RECEDING_HORIZON.md` |
| 14 | 합의 참조 안정화 (0.5/1.0/1.5 s 전 계획의 motion 합의) | 완료 (3080 Ti) | `.../consensus_reference_stabilization/CONSENSUS_REFERENCE.md` |
| 15 | 배포 조건 안정화 (교정 시점 없음, 자연 디코딩) | 완료 (3080 Ti) | `.../natural_reference_stabilization/NATURAL_REFERENCE.md` |
| 16 | PDM-Closed 참조 | 완료 | `.../pdm_reference_stabilization/PDM_REFERENCE.md` |
| 17 | seed·온도 강건성 | 완료 | `.../robustness_reference_stabilization/ROBUSTNESS.md` |
| 18 | best-of-N 선택 (N 8, T 0.7) | 완료 (3080 Ti) | `.../best_of_n_selection/BEST_OF_N.md` |
| 19 | best-of-N 선택 (N 4–32, T 0.7–1.3, 5 seed) | 완료 (두 GPU) | `.../best_of_n_selection_n16_T1/BEST_OF_N_16.md` |
| 20 | 표본 확대: 새 navtest log 56개(4,563 장면) | 완료 (두 GPU, shard 스트리밍) | `.../expanded_best_of_n/EXPANDED_BEST_OF_N.md` |

한 줄 요약: 실패는 결정·실행 결합에서 나고, 작은 action-token 편차는 **직전 토큰이 뜻하는 motion**이 다음 토큰을 조건화하는 1-step feedback으로 증폭됩니다. 모델은 조건화된 motion을 66–89% 따라가므로, GT 없는 참조로 조건화를 교정하면 참조 품질이 전부를 결정하고 배포 조건에서는 순이득이 없습니다(실험 12–16). **배포 가능한 순이득은 선택에서 나왔습니다**: 모델 자신의 후보 16개(T 1.0) 중 엔트로피 순위 + log-likelihood 순위가 가장 좋은 계획을 고르면, 이전에 쓰지 않은 log 56개(4,563 장면)에서 실패율 2.15 → 1.45%(−33%, p = 4e-4), FDE 0.76 → 0.48 m(실험 20, open-loop).

## 2. 새 서버 준비 (순서대로)

전제: Linux, NVIDIA GPU(24 GB 이상 권장; 12 GB에서도 batch ≤ 15행 실험은 동작), 디스크 약 80 GB, 인터넷(HuggingFace, S3, GitHub).
**모든 경로가 `/root/VLA` 절대경로로 하드코딩**되어 있으므로 저장소를 반드시 `/root/VLA`에 둡니다.

```bash
git clone https://github.com/kdh0428/VLA.git /root/VLA
cd /root/VLA
bash tools/download_autovla_assets.sh   # AutoVLA repo(커밋 고정) + 체크포인트 16 GB + Qwen 7.5 GB + navtest 메타 + 카메라 shard 0-5 (23 GB) + 지도 1 GB. 약 30분
bash tools/setup_autovla_env.sh         # Miniforge + conda env `autovla` (python 3.9, torch 2.8.0+cu128). 약 15분
bash tools/regenerate_autovla.sh        # 원시 결과 재생성 (약 9-10시간, 단계 설명은 스크립트 머리말)
```

`regenerate_autovla.sh`는 단계별 실행이 가능합니다(`STAGES="ed ah" bash tools/regenerate_autovla.sh`). 새 실험은 대부분 `ed`(equal-distance 단위 정의)까지만 있으면 됩니다: `pre → full → ed` (약 5시간, 대부분 `full`).

## 3. 반드시 알아야 할 함정

| 증상 | 원인 | 대처 (스크립트에 반영됨) |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=1`인데 느린 GPU에서 돔 | torch 기본 GPU 순서가 속도순(FASTEST_FIRST) | env `autovla`에 `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| "pinned to GPU 1" 로 종료 | 모든 GPU 실험이 `CUDA_VISIBLE_DEVICES=1`을 강제 | 다른 구성이면 `VLA_ANY_GPU=1` 설정 후 원하는 번호 사용. 수치가 소폭 달라지므로 **한 실험의 모든 조건은 같은 GPU에서** |
| 모델 로드 중 exit 137 | fp32 체크포인트 16 GB를 RAM에 통째로 올림 | 전 스크립트 `torch.load(..., mmap=True)` |
| 장면 로드 `TypeError: stat ... NoneType` | nuPlan 지도 경로 미설정 | env에 `NUPLAN_MAPS_ROOT`, 지도는 다운로드 스크립트가 받음 |
| `libGL.so.1` 없음 | 컨테이너에 GL 없음 | opencv-python-headless |
| torch 2.4 설치 시 5090 미지원 | 원본 requirements | torch 2.8.0+cu128 (원 PoC 실행 버전과 동일) |
| 디스크 부족 | full_extract 텐서가 두 arm이면 56 GB | `--tensor-arms N` (18 GB). arm C 텐서가 필요한 `cot_intervention` 재현은 불가 |
| 12 GB GPU에서 모델 로드 OOM (공유 GPU) | `resize_token_embeddings`의 mean-resizing이 1.2 GB 일시 할당 | `VLA_LOW_MEM=1` (그 행은 체크포인트가 덮어쓰므로 가중치 동일). `VLA_BUILD_ON_CPU=1`은 RAM 14 GB 한도에서 죽음 |
| 같은 실험이 GPU마다 조금 다른 수치 | 실패 장면은 경계 장면이라 부동소수 차이에 민감 | 한 실험의 모든 조건은 한 GPU·한 batch에서. 5090과 3080 Ti 결과를 섞어 비교하지 말 것 |
| `Planner` 쓰는 스크립트에서 상대경로 깨짐 | `_planner.Planner`가 `os.chdir(/root/VLA/autovla)` | 경로는 절대경로로 |
| 재실행 시 "records.jsonl exists; refusing" | 결과 덮어쓰기 방지 | 출력 폴더를 지우거나 `--output` 지정 |

## 4. 코드 지도 (새로 추가된 것)

```
autovla_misalignment_poc/scripts/
  _planner.py                        공용: 모델 로드, 한 번의 계획(stub 생성 + 10 action, context 치환 지원),
                                     궤적<->토큰(GT와 같은 routine), CTRA 외삽, SE2 변환
  build_frame_scenes.py              ED 장면별 이전(-1..-3)/미래(+2..+14) frame 장면 + 8 s 로그 pose -> outputs/frame_index/index.json
  prev_action_identity_decomposition.py / analyze_prev_action_identity.py / make_identity_figures.py   실험 11
  reference_stabilization.py / analyze_reference_stabilization.py    실험 12, 14, 16, 17 (--sources, --windows, --temperature, --seed)
  receding_horizon_replanning.py / analyze_receding_horizon.py       실험 13
  natural_reference_stabilization.py / analyze_natural_reference_stabilization.py   실험 15 (perturbation 없는 자연 디코딩)
  pdm_reference.py                   PDM-Closed 궤적 (navsim MetricCacheProcessor 설정, CPU)
  best_of_n_selection.py / analyze_best_of_n.py / pool_best_of_n.py  실험 18-19 (후보 샘플링 + 선택 규칙 + seed 합산)
  expanded_best_of_n.py / analyze_expanded_best_of_n.py             실험 20 (새 장면: stub 생성 + 후보 디코딩, 모집단 직접 추정)
tools/stream_expanded_best_of_n.sh                                  실험 20 shard 스트리밍 (받기 -> 전처리 -> 디코딩 -> 이미지 삭제)
  check_full_extract_repro.py        재생성 결과를 git HEAD 수치와 비교
tools/
  download_autovla_assets.sh, setup_autovla_env.sh, regenerate_autovla.sh
```

실험 7–14는 모두 같은 harness입니다: equal-distance 단위(A− 52 장면/365 단위, A+ 156 장면/1043 단위), prompt + fast stub + t* 이전 action의 prefix KV cache, 이후 매 step span 재계산, 모든 조건을 한 batch로 같은 seed 샘플링(T = 0.01). 새 조건을 추가할 때는 `reference_stabilization.py`나 `prev_action_identity_decomposition.py`의 행(row) 정의를 확장하는 것이 가장 빠릅니다.

## 5. 다음 실험 후보 (우선순위 순)

1. **best-of-N 선택의 진짜 closed-loop 검증**: 실험 19의 확신도 기반 선택(순위 합 또는 log-likelihood 권장)이 유일한 배포형 순이득입니다. NAVSIM v2 pseudo-simulation(또는 nuPlan 시뮬레이터)에서 PDMS로 평가해야 실제 주행 이득을 말할 수 있습니다. 실험 13처럼 로그 카메라를 쓰는 재계획은 누출이 있어 대체가 안 됩니다.
2. **선택 신호 개선**: oracle 선택은 실패율을 0.24%까지 낮추므로(실험 20) 여지가 큽니다. 저장된 `outputs/expanded_best_of_n/gpu*/records.jsonl`(후보별 step 엔트로피·log-prob·궤적; 저장소에는 없음, 재생성 필요)로 CPU에서 새 규칙을 먼저 평가하고, 반드시 새 log나 새 seed로 확인하세요(규칙을 같은 데이터로 고르면 과적합). 조기 가지치기(앞쪽 step만으로 선택)는 통하지 않았습니다.
3. **선택 규칙 개선**: 엔트로피와 log-likelihood의 결합, 후보 간 합의와의 결합, 선택 후 재계획. 모두 저장된 `records.jsonl`로 CPU에서 먼저 평가할 수 있습니다.
4. **표본 확대 (계속)**: shard 6–17은 처리했습니다(`tools/stream_expanded_best_of_n.sh <GPU> <shard...>`, 이미지를 처리 후 지워 디스크 15 GB로 가능). shard 18–31(navtest log 약 60개)이 남아 있습니다.
5. **ORION에서 재현**: 실험 7–19는 AutoVLA에서만 했습니다(ORION 모델·데이터 약 60 GB 필요).

**하지 않아도 되는 것(이미 음성)**: 외부 planner(PDM-Closed)나 CTRA 궤적으로 조건화 교정, 오래된 계획의 합의로 조건화, 엔트로피 trigger로 조건화 교정 켜기, 재계획 안에서 이전 계획으로 조건화 — 모두 정상 장면을 해쳐 순손해였습니다.
