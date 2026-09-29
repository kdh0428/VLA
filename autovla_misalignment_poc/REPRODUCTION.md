# 새 머신에서의 재현 기록 (2026-09-26 ~ 09-29)

저장소를 새 컨테이너(RTX 5090 + RTX 3080 Ti, RAM 한도 14.4 GB, 디스크 101 GB)에 clone해 환경을 다시 만들고, 기존 결과가 그대로 재현되는지 확인했습니다.

## 환경 구축

```bash
bash tools/download_autovla_assets.sh   # 체크포인트, Qwen2.5-VL-3B, navtest 메타데이터, 카메라 shard 0-5 (PoC 28개 log 전부 포함)
bash tools/setup_autovla_env.sh         # conda env `autovla`
# 추가로 nuPlan 지도(nuplan-maps-v1.1.zip)를 autovla/dataset/nuplan/maps 에 풀어야 함
python autovla_misalignment_poc/scripts/preprocess_scenes.py   # 장면 2748개
```

| 항목 | 원본 requirements | 이 환경 | 이유 |
|---|---|---|---|
| torch | 2.4.0 | **2.8.0+cu128** | 2.4는 RTX 5090(sm_120) 미지원. 원 PoC 실행 기록과 같은 버전 |
| flash-attn / waymo / autoawq | 설치 | 미설치 | PoC는 eager attention, nuPlan만 사용 |
| opencv-python | 4.9.0.80 | opencv-python-headless 4.9.0.80 | 컨테이너에 libGL 없음 |
| urllib3 | – | <1.27 | botocore 1.37 요구 |
| env var | – | `CUDA_DEVICE_ORDER=PCI_BUS_ID` | 미설정 시 torch가 GPU를 속도순으로 매겨 `CUDA_VISIBLE_DEVICES=1`이 3080 Ti를 가리킴 |
| env var | – | `NUPLAN_MAPS_ROOT`, `NUPLAN_MAP_VERSION` | navsim 장면 로드에 지도 필요 |

코드 변경은 재현에 영향이 없는 것만 했습니다.

- 모든 스크립트의 체크포인트 로드에 `mmap=True`: fp32 16 GB 체크포인트를 RAM 14.4 GB 한도 안에서 로드(값은 동일).
- `full_extract.py`: GPU 고정을 `CUDA_VISIBLE_DEVICES=1`로 통일(원 머신에서는 torch 기준 0번이 5090이었음), 디스크 절약용 `--tensor-arms` 옵션 추가(기본값은 기존과 동일, 이번 실행은 `N`만 저장).
- `pra_comparison/run_pra_comparison.py`: 텐서를 저장하지 않은 arm의 `tensor_file`이 비어 있을 때 처리.
- `reference_stabilization.py`만 `CUDA_VISIBLE_DEVICES=0`(3080 Ti)도 허용합니다. 한 실행의 모든 조건은 같은 GPU에서 한 batch로 계산됩니다.

## 재현 결과

`scripts/check_full_extract_repro.py`가 git HEAD의 수치와 개수 단위로 비교합니다.

| 대상 | 결과 |
|---|---|
| `full_extract` (4.8 h) | **전부 일치**: 2748 장면, arm N fork 2747/2748, arm N n 2747 / A− 52 / step-0 오류 17·216, arm C n 2604 / A− 136 / 58·430 |
| P/R/A (AutoVLA) | **전부 일치**: 제외 143 + 1, 8개 그룹 개수, arm N A− 비율 1.892% (ORION 기록은 저장소에 없어 비교하지 않음) |
| `action_history_causal` | **전부 일치**: 모든 조건의 증폭률·FDE (Normal 47.4%, Recent-GT 7.1%, GT-history 4.1% …), 저장된 ED 토큰 재현율 71.59% |
| `equal_distance_perturbation` | Q4(교차검증 로지스틱 AUROC)만 최대 0.009 차이(solver 수치), 나머지 동일 |

재생성한 결과 파일은 기존 커밋 버전과 수치가 같으므로 저장소에는 기존 파일을 유지했습니다.

## 재현하지 못한 것

- `cot_intervention` 및 natural_fast_mechanism의 arm C 분석: arm C 텐서(약 40 GB)를 디스크 부족으로 저장하지 않음.
- ORION 전체: 모델·데이터를 받지 않음.
