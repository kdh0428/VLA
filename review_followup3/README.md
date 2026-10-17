# review_followup3 — 핵심 보고서 (2026-10-14/15, CODING_AGENT_PROMPT_20261014_v2)

이 폴더에는 사람이 읽을 핵심 보고서만 모아 두었다. 2026-10-17에 `autovla_misalignment_poc/outputs/review_followup3/`에서 옮겼다.

- **원 데이터·CSV·스크립트 출력은 원래 위치에 그대로 있다:** `autovla_misalignment_poc/outputs/review_followup3/`(P0–P5, STAGE3_COST_*.csv).
- **아래 문서 안의 상대 경로**(`P0/…csv` 등)는 그 원래 디렉토리를 기준으로 한다.
- **재생성 스크립트**(`scripts/review_followup3/p0_regen.sh` 등)를 다시 돌리면 보고서가 원래 위치에 새로 쓰인다.

| 파일 | 내용 |
|---|---|
| `STATUS.md` | 작업별 최종 상태(DONE / NOT_RUN 등)와 그 상태가 claim에 미치는 영향 |
| `REVIEW_COVERAGE.md` | 리뷰 항목 M1–M7, m1, m2, m4의 해결 여부와 원고에 남길 한계 문장 |
| `CLAIM_EVIDENCE_MATRIX.md` | claim별 유지/약화/삭제 판정(도메인·모델·평가 방식 태그 포함) |
| `CP2_REPORT.md`, `CP2_DECISION.md` | Stage 3 선택지와 비용, 그리고 사용자 결정 |
| `RUN_COMMANDS.md` | 재현 명령(CPU와 GPU를 나눠 정리) |
| `P0/` | 모집단·선택 흐름, SpatialVLA 과제별 효과, 수치 정리(+4.37 m), window 가중, 의존성, OpenVLA 재분석 |
| `P1/` | 개입 구현 명세, no-op parity 결과(AutoVLA 6개 경로 PASS), SpatialVLA 상태 복원 결과 |
| `P2/` | CM 정의 감사, 운동 baseline 결과, learned extrapolator 결과 |

원고 문장 수정안(유지 K1–K12 / 삭제·교체 R1–R17)과 수치는 `paper_quantitative_package/`의 `MANUSCRIPT_REVISION.md`, `numbers.json`, `tables.tex`에 있다.
