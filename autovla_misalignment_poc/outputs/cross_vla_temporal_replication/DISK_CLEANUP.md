# 디스크 정리 기록 (cross-VLA temporal replication 준비, 2026-10-04 16:21:34)

정리 전: 사용 91G, 여유 6.7G

| 순위 | path | size | 삭제 가능한 이유 | 재생성 방법 |
|---|---|---|---|---|
| 1 | /root/.cache/pip | 402M | pip 다운로드 cache | pip install 시 자동 재다운로드 |
| 1 | /root/miniforge3/pkgs (conda clean) | 656M | conda 패키지 tarball cache, 설치된 env와 무관 | conda install 시 재다운로드 |
| 1 | /root/.cache/mesa_shader_cache | 900K | Mesa shader cache | 실행 시 자동 생성 |
| 2 | scratchpad/ov_dbg | 3.2M | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/ov_dbg2 | 5.5K | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/ov_dbg3 | 3.2M | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/ov_smoke | 137K | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/cv_test | 12M | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/flags_test | 5.2M | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/flags_smoke | 2.0K | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/tf_test | 5.5K | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/stf_test.sh | 3.0K | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 2 | scratchpad/shardtmp | 512 | 디버그·smoke test 임시 출력 (최종 결과 아님) | 해당 스크립트 재실행 |
| 3 | /root/VLA/navhard/exp/metric_cache_half | 598M | navhard 첫 절반 metric cache. 모든 채점 완료(공식 CSV·group score 보존) | scripts/navhard_metric_cache.py (CPU 약 10분) |
| 3 | /root/VLA/navhard/exp/metric_cache_half2 | 639M | navhard 두 번째 절반 metric cache. 채점 완료 | NAVHARD_SUBSET=.../half2/subset scripts/navhard_metric_cache.py --cache .../metric_cache_half2 |
| 6 | /root/VLA/navhard/navhard_two_stage/sensor_blobs (36 log, navhard 첫 절반 합성 장면 CAM_F0/L1/R1) | 5.5G | 실험 25–30의 디코딩·채점 완료(decode records, 공식 CSV, group score는 outputs에 보존). 1–3순위 정리 후 여유 8.0 GB로 모델 7.5 GB를 받기에 여유분 부족 | navsim_v2.2_navhard_two_stage_{curr,hist}_sensors.tar.gz에서 outputs/navhard_eval/subset/synthetic_files.txt 목록만 추출 (약 8분) |

주: outputs/ 안의 intermediate(teacher_forced hidden state, OpenVLA 프레임)는 '기존 outputs 삭제 금지' 규칙에 따라 후보에서 제외했습니다.
정리 후: 사용 84G, 여유 14G (정리 전 여유 6.7G)
