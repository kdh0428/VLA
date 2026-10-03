# P6: 연속 closed-loop 평가 — 현재 환경에서 수행 불가 (2026-10-03 확인)

요청: Bench2Drive / CARLA 같은 연속 재계획 환경에서 (1) 원래 AutoVLA, (2) 안전 필터만, (3) 안전 필터 + 확신도 선택기 비교.

## 확인한 사항

| 요건 | 상태 |
|---|---|
| CARLA용 AutoVLA checkpoint | **없음.** 공개 저장소 `Zewei-Zhou/AutoVLA`에는 NAVSIM checkpoint(`AutoVLA_PDMS_89.ckpt`)만 있습니다(파일 목록: `.gitattributes`, `AutoVLA_PDMS_89.ckpt`, `README.md`). 논문은 CARLA 결과를 보고하지만 그 checkpoint는 공개되지 않았습니다. |
| CARLA 렌더링 | 컨테이너에 NVIDIA EGL/Vulkan 라이브러리(libEGL_nvidia, libnvidia-gl)가 없습니다(LIBERO도 Mesa 소프트웨어 렌더링으로만 실행). CARLA 0.9.15는 GPU Vulkan 렌더링이 필요해 소프트웨어 렌더링으로는 실용적인 속도가 나오지 않습니다. |
| 디스크 | CARLA 0.9.15 + Bench2Drive 자산 약 20–25 GB가 필요하지만 여유는 약 7 GB입니다. |
| 대안: nuPlan closed-loop | nuPlan 시뮬레이터는 카메라를 렌더링하지 않아 카메라 기반 AutoVLA를 연속 실행할 수 없습니다. NAVSIM v2의 2단계 pseudo closed-loop가 바로 이 제약을 우회하려고 만든 평가입니다. |

## 결론과 필요한 것

이 환경에서는 의미 있는 연속 closed-loop 평가를 할 수 없습니다. NAVSIM checkpoint를 CARLA에 그대로 쓰면 sim-to-real 차이와 카메라 배치 차이 때문에
원래 AutoVLA 자체의 성능이 무너져, 필터 효과를 해석할 수 없습니다. 수행하려면 다음이 필요합니다.
1. CARLA로 학습한 AutoVLA checkpoint (저자 공개 또는 Bench2Drive 데이터로 재학습: 3B 모델 SFT/RFT, 수십 GPU-day 규모).
2. NVIDIA 그래픽 드라이버 라이브러리(EGL/Vulkan)가 노출된 컨테이너, 디스크 약 30 GB 추가.
3. 그 뒤 비교 조건 3개 × Bench2Drive 220 route. 안전 필터는 CARLA의 현재 객체·지도로 같은 규칙(등속 외삽 충돌, 주행 가능 영역)을 구현.

NAVSIM v2 결과와는 별도 평가로 취급해야 한다는 요청 사항은 그대로 유효합니다.
