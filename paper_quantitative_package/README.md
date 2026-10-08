# Paper quantitative evidence package

위치: `/root/VLA/paper_quantitative_package/` (원래 `autovla_misalignment_poc/outputs/` 아래에서 옮김). 문서 안의 `O/`는 `/root/VLA/autovla_misalignment_poc/outputs/`입니다.

이 패키지는 기존 실험 결과만 써서 논문용 수치·표·그림 데이터를 정리한 것입니다.
- 새 실험이나 GPU inference는 하지 않았습니다.
- 기존 실험 디렉토리는 수정하지 않았습니다.
- 재계산은 원 분석 스크립트와 같은 방법·seed로만 했고, 해당 값에는 "(recomputed)"를 붙였습니다. 재계산 스크립트는 `_parts/scripts/`에 있습니다.
- 실험 35 결과(2b0212d)는 `EXP35_RESULTS.md`, PAPER_NUMBERS Claim 11·17, tables.tex Table 6에 반영했습니다.

| 파일 | 내용 |
|---|---|
| `PAPER_NUMBERS.md` | **핵심 문서.** claim 17개 각각에 대해 핵심 수치, N, CI, p, 출처, 논문에 써도 되는 문장, 과장이 되는 문장을 정리 |
| `PAPER_EVIDENCE.md` | 요청 §2–6 전체 정량 근거: AutoVLA A–F, Impromptu, OpenVLA, SpatialVLA, 탐지, 선택, navhard |
| `SOURCE_MAP.md` | 실험 번호 → 디렉토리 색인, 결과별 출처·commit·N·seed·GPU·검정·스크립트 |
| `CONSISTENCY_AUDIT.md` | 요약 문서와 원 결과의 불일치(CRITICAL 0, MAJOR / MINOR 목록과 상세) |
| `tables.tex` | 본문 Table 1–6: 모델 구조, AutoVLA 인과 개입, cross-model, 탐지, 완화, 실험 35 실행 구조 ablation |
| `paper_cross_model_table.{csv,tex}` | 4개 모델 전체 비교표(23행) |
| `paper_detection_table.{csv,tex}` | 실험 31 held-out 탐지 전체 표 |
| `paper_navhard_table.{csv,tex}` | navhard 76 log / 225 그룹 전체 표 |
| `spatialvla_closed_loop_paired.csv` | 실험 34 closed loop 80 에피소드 paired 표, 과제별 값, 실행 효과와 문맥 효과의 분리 |
| `figure_*.csv` | 그림을 다시 그리는 데 필요한 summary point와 CI: equal_distance, causal_intervention, layer_patching, temporal_window, motion_semantics, cross_model, detection_pre_post, candidate_count, navhard_methods |
| `EXP35_RESULTS.md` | 실험 35 결과표(설정 6개, 합동 주 분석, 오차 지속, 판정) |
| `_parts/` | 영역별 원문 추출, 출처, 불일치 문서, 재계산 스크립트와 결과 |

LaTeX 표는 `booktabs`, `graphicx`가 필요합니다. 이 서버에는 LaTeX가 없어 컴파일 확인은 하지 못했고, 괄호와 환경의 짝만 검사했습니다.
