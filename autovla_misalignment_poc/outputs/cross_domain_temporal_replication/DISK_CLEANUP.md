# 디스크 정리 기록 (cross-domain temporal replication 준비, 2026-10-05 12:33:05)

정리 전: 사용 92G, 여유 5.5G — 필요량: SpatialVLA sft-fractal 8.1 GB + SimplerEnv·의존성·출력 약 2 GB

허용 범주 1–4(cache, 임시, 재생성 가능한 intermediate)를 모두 지워도 약 6.9 GB에 그쳐, 사용자에게 확인 후(2026-10-05) 아래를 삭제했습니다.

| path | size | 삭제 이유 | 재생성 방법 |
|---|---|---|---|
| /root/VLA/openvla/openvla-7b-finetuned-libero-spatial | 15G | 공개 checkpoint(유일본 아님). 실험 28(OpenVLA) 완료, 결과는 outputs/cross_vla_replication에 보존. 사용자가 이 항목 삭제를 선택 | `huggingface_hub.snapshot_download('openvla/openvla-7b-finetuned-libero-spatial', local_dir='/root/VLA/openvla/openvla-7b-finetuned-libero-spatial')` (약 3분) |
정리 후: 사용 78G, 여유 20G (파일시스템 반영 지연 후 확인)
