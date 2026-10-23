# v04 원고 서버 적용 결과 — 2026-10-23 v01

입력은 `CVPR_V04_WITH_AGENT_PROMPT_20261023_v01.zip`(sha256 3721ab26…)입니다. 그 안의 `CVPR_INTEGRATED_REVISION_20261023_v04.zip`은 51c255ca…로, 요청서의 해시와 일치합니다. 입력 ZIP 자체는 수정하지 않았습니다.

## 단계별 상태

| 작업 | 상태 | 내용 |
| --- | --- | --- |
| A. 원고 반영 | PASS | 적용 경로는 `/root/VLA/v04_apply/work/CVPR_INTEGRATED_REVISION_20261023_v04`입니다. 이전 작업 폴더(`integrated_v03_work`, `svla_wc_work`, `vla0_work`, `vla0_cl_work` 등)와 기존 결과는 그대로 두었습니다. `/root/VLA`와 그 상위에 `AGENTS.md`는 없습니다. |
| B. CPU 검증 | PASS | `verify_20261023_v04.py`는 종료 코드 0, 25,957개 검사 통과입니다. `verify_baseline_20261023_v04.py`는 종료 코드 0, 668개 검사 통과입니다. 원고 수정 후 두 검증기를 다시 실행해도 동일하게 통과했습니다. |
| C. 문장·범위 점검 | PASS(2문장 수정) | 잠금 조건(아래 표)을 확인했고, 본문 두 곳에 "별도 cross-step 연구" 표현을 추가했습니다. diff는 `server_qa/diffs/`에 있습니다. |
| D. PDF 빌드 | PASS | clean pdfLaTeX + BibTeX 빌드입니다. 본문은 총 10쪽(본문 8쪽, LLM 명세와 참고문헌 9쪽부터)이고, 보충자료는 19쪽(참고문헌 19쪽)입니다. 오류·미정의 참조·overfull은 모두 0이고, 공식 CVPR 2027 스타일 해시와 일치합니다. 전 페이지를 렌더링해 확인했습니다. |
| E. 정리 | PASS | 이 README, 변경 감사 기록, 새 해시 목록, 결과 ZIP을 만들었습니다. |

새 GPU 실험은 하지 않았습니다(GPU 사용 0분). 모델 추가, 재학습, 성공률 측정도 없었습니다.

## 실행 명령과 종료 코드
- `python verify_20261023_v04.py`: 종료 코드 0, 25,957개 검사, 결과 `qa/RAW_RECOMPUTATION_20261023_v04.json`
  - 입력에 들어 있던 결과 파일과 비교하면 일부 값이 마지막 자리 부동소수점에서만 다릅니다(최대 5.6e-17). 서버 NumPy의 합산 순서 차이로 보이며, 보고 정밀도의 모든 수치는 같습니다.
  - 로그: `server_qa/verify_20261023_v04*.log`
- `python verify_baseline_20261023_v04.py`: 종료 코드 0, 668개 검사. 결과 JSON은 입력과 바이트 단위로 같습니다.
- `python build_20261023_v04.py`: 종료 코드 0. 결과는 `qa/BUILD_QA_20261023_v04.json`, `server_qa/build_20261023_v04.log`입니다.
  - 빌드 도구: pdfTeX 1.40.29와 BibTeX(TeX Live 2026)를 공식 CTAN net installer로 scratchpad에 최소 설치해 사용했습니다(scheme-infraonly + collection-latex + 필요한 패키지).
  - 기존 데이터·가중치는 삭제하거나 이동하지 않았습니다.
- 사용한 Python 환경은 `autovla` env의 python 3.9입니다(NumPy 1.23.4, Pillow 11.3.0, pypdf 6.21.1).

## 서버에서 재실행한 것과 동봉 기록만 확인한 것
- **서버에서 재실행:** 두 CPU 검증기(원시 토큰·pose·bootstrap·그림 CSV 재계산), pdfLaTeX/BibTeX 빌드, 렌더링 시각 점검, 입력 해시 대조(433개 중 검증기가 다시 쓴 1개 파일만 차이).
- **동봉 기록만 확인:** 기록된 logit 차이(0.125, 0.5, 0.0), collection-time parity, GPU 실행 시간(62.8분). 이 값들은 새 GPU 추론으로 다시 인증한 것이 아닙니다.

## 잠금 조건 점검(C)

| 조건 | 결과 |
| --- | --- |
| 주 결과: 69관측 / 34 layouts / 2과제, Δ=0.048378 [0.025522, 0.072148], 정규화 거리(pp 아님) | 본문·표·보충자료 모두 일치 |
| D_S = a3·a4의 6개 운동 성분(12개 성분) 평균 절대 차이, VLA-0의 30개 정수 성분 거리와 구분 | §3.4에 명시됨 |
| O/L에 같은 sampled chunk의 대응 행동을 넣고, 각 위치 gripper는 자연값 유지 | 명시됨 |
| O에서 a2를 자연값으로 고정 → 전체 전파 효과가 아님 | 명시됨 |
| 매칭 donor 69/105, 거리 비율 0.8–1.25 | 명시됨 |
| 서랍: 35관측 / 15 reset scenarios / 7 pose groups, 0.144881 [0.090537, 0.209987], 자세군 민감도 [0.075, 0.236]는 별도 분석, 두 과제 결과와 합치지 않음 | 일치 |
| 서랍의 collection-time 비교 불가, R logit 차이 최대 0.5(bitwise 아님) | 명시됨 |
| cross-step 성공률 연구와 within-call 명령 수준 연구 구분 | 초록과 Discussion에 "별도 cross-step 연구"를 추가해 보강(diff 참조). 결론은 이미 구분돼 있었음 |
| 모델 선택: 조사한 구현에 한정, FAST는 autoregressive이면서 시간 비국소적, WorldVLA는 제3자 fork와 revision 명시 | 일치 |
| `pre-specified` 유지(`pre-registered`는 부정문에서만 사용), 원시 프로토콜은 수정 안 함 | 일치 |
| 새 방법·성능 향상·backbone 인과효과를 주장하지 않음 | 일치 |

## 입력 대비 변경 사항

`server_qa/CHANGE_AUDIT_vs_input_20261023_v01.json`에 정리했으며, 변경 파일은 14개, 추가·삭제 파일은 없습니다.
- **TeX 2개**
  - `paper/paper_body.tex`: 초록과 Discussion에 "separate cross-step" 표현을 추가했습니다.
  - `paper/llm_usage_statement_20261023_v04.tex`: Claude Code 사용 사실을 추가했습니다.
- **빌드 산출물**: PDF 4개(`paper/main.pdf`, `paper/supplementary.pdf` 및 최상위 사본 2개), aux·log·build console, `qa/BUILD_QA_20261023_v04.json`.
- **검증기 결과**: `qa/RAW_RECOMPUTATION_20261023_v04.json`(마지막 자리 부동소수점 차이만 있음).
- 실험 원시 기록, 프로토콜, source 후보, snapshot은 바뀌지 않았습니다.

새 해시 목록은 `server_qa/SHA256SUMS_APPLIED_20261023_v01.txt`(434개 파일)입니다.

## 저자 확인이 필요한 사항
1. **LLM Usage Statement**
   - 이 서버의 실행 기록(이번 세션과 2026-09-26 이후 작업 메모)에서 Anthropic Claude Code가 실험·평가·분석 코드의 구현과 실행(모델 추론 harness, 시뮬레이터 rollout 포함), CPU 검증, 그림 생성, 서지 확인, 원고 수정 초안에 쓰인 것을 확인해 그 사실을 추가했습니다.
   - ChatGPT/Codex 문장은 v04 그대로 두었습니다.
   - 이 서버 밖에서 쓴 도구와 이전 기간의 전체 사용 내역은 확인할 수 없습니다. 저자가 최종 완전성을 확인해야 합니다.
2. **빌드 환경:** 저자 환경의 pdfLaTeX 버전과 이번 서버(TeX Live 2026)가 다를 수 있습니다. 제출 전 8쪽 경계를 한 번 더 확인해 주세요.
3. **제출 정보:** CVPR 2027 paper ID와 제출 시스템 정보는 확인하지 않았습니다.

## 결과 ZIP에서 뺀 입력 파일
`raw_packages/`의 원시 ZIP 3개(약 50 MB)는 넣지 않았습니다. 파일 이름과 해시는 `server_qa/OMITTED_INPUT_FILES_20261023_v01.json`에 있습니다. 입력 v04 ZIP 안과 서버의 적용 경로에 그대로 있습니다.
