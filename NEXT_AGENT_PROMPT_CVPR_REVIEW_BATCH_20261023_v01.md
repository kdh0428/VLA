이 서버에는 이전 서버의 작업을 git으로 옮긴다. /root/VLA가 아직 없으면 `git clone https://github.com/kdh0428/VLA.git /root/VLA && cd /root/VLA && git checkout autovla-experiments-11-19`로 받는다(경로가 /root/VLA로 하드코딩돼 있으므로 반드시 이 위치). git에 없는 weights·conda env·SimplerEnv·렌더러는 인수인계 §0.5대로 먼저 준비한다. 이전 코딩 에이전트가 `/root/VLA/CODING_AGENT_CVPR_REVIEW_BATCH_EXECUTION_20261023_v01.md`(A–F 리뷰 대응 통합 실행)를 진행하다가 중단했다. 이 작업을 이어서 끝까지 완료해 줘.

먼저 다음 두 파일을 끝까지 읽고 시작해:
1. `/root/VLA/HANDOFF_CVPR_REVIEW_BATCH_20261023_v01.md` — 인수인계(완료/미완료 상태, 경로, 재개 명령, 주의사항)
2. `/root/VLA/CODING_AGENT_CVPR_REVIEW_BATCH_EXECUTION_20261023_v01.md` — 실행 지침(충돌 시 이 파일이 우선)

진행 방식:
- 새 서버 환경부터 확인한다: GPU 종류·순서·VRAM, RAM, 디스크 여유, conda env `autovla`, `/root/VLA/simpler/env.sh`의 경로와 CUDA_VISIBLE_DEVICES, Vulkan 렌더러. 결과를 `v05_work/status/ENV_NEW_SERVER_<날짜>.json`에 기록한다. 하드웨어나 경로가 바뀌었으면 인수인계 §3·§4의 규칙을 따른다. 이전 서버의 부분 결과와 새 서버 결과를 섞지 않고, 새 out 디렉터리에서 validation부터 다시 하며, 변경은 protocol Amendment로 기록한다.
- 이미 끝난 A·E·F1 자료(`v05_work/A_materials`, `autovla_E`, `lit`)는 다시 만들지 말고 재사용한다.
- 남은 일을 실행 프롬프트의 우선순위대로 한다.
  - B: VLA-0 penalty 1.05/1.0의 나머지 → D(bridge) → B4(donor follow-up).
  - C: SpatialVLA full episode. `c_run.py`는 검증된 적이 없으니 smoke → val gate → main 순서로 한다. 자원이 남으면 r4e1 비교도 한다.
  - 그 다음 분석 스크립트를 작성하고, raw record로부터 표와 그림을 만든다.
  - F2–F3: 본문 8쪽 v05 검토용 원고, 전체 보충자료, PDF 빌드와 렌더 QA, CVPR_OVERLEAF_20261023_v05.zip을 만든다.
  - REVIEW_AFTER_RESULTS_20261023_v01.md, README_RESULT_20261023_v01.md, TASK_STATUS_20261023_v01.json을 쓴다.
- 최종 산출물은 `/root/VLA/` 아래 새 폴더 하나에 `CVPR_REVIEW_BATCH_RESULT_20261023_v01.zip` 하나로 묶는다.
- 단계마다 나에게 확인받지 말고 끝까지 진행해. 도구 하나가 거절돼도 가능한 다른 작업은 계속한다. 내가 "중단"이라고 말할 때만 멈춘다.
- 규칙:
  - 커밋하지 않는다. v04와 기존 원시 기록·checkpoint·weights는 수정·삭제하지 않는다.
  - GPU는 지정 GPU(이전 서버 기준 physical GPU 1, RTX 5090, CUDA_DEVICE_ORDER=PCI_BUS_ID)만 쓰고 GPU 분을 ledger에 기록한다.
  - 실행하지 않은 검사를 PASS로 표시하지 않는다. 사후 분석은 표시한다. "pre-registered"가 아니라 "pre-specified"라고 쓴다.
  - 내가 만든 임시 파일은 정리한다. 비어 있을 수 있는 변수로 cd/rm/mv를 하지 않는다.
- 작업 문서와 최종 보고는 한국어, 논문 본문은 영어로 쓴다. 마지막 보고에는 실제 GPU 실행 범위, 독립 unit 수, 통과한 gate, 미실행 항목, 결과 ZIP 위치를 짧게 적어 줘.

참고: 인수인계 §1에 적힌 것처럼 `CVPR_OVERLEAF_20261023_v04.zip`과 첨부 리뷰 텍스트는 서버에 없다. 내가 새로 올려 주면 그것을 쓰고, 없으면 v04 서버 적용본을 기준으로 진행하고 README에 "미제공"으로 기록해.
