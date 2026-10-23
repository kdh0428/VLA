# 인수인계: CODING_AGENT_CVPR_REVIEW_BATCH_EXECUTION_20261023_v01.md 이어서 실행
작성: 2026-10-23 (서버 이전 직전, 사용자 요청으로 실행 중단 후 작성)
대상: 새 서버에서 이 작업을 이어받는 코딩 에이전트

## 0. 한 줄 요약
실행 프롬프트 `/root/VLA/CODING_AGENT_CVPR_REVIEW_BATCH_EXECUTION_20261023_v01.md`(A–F)를 실행하던 중 사용자 지시로 중단했다.
A·E·F1(문헌)은 자료 수집이 끝났다. B(VLA-0)는 validation PASS, 본 실행 17/149 관측에서 중단됐다. C(SpatialVLA full episode)는 protocol·manifest 고정 직후 validation 단계에서 중단됐다(완료 episode 0, **러너가 end-to-end로 검증된 적 없음**). D·B4·C 본 실행·분석·v05 원고·최종 ZIP은 미실행이다.
모든 작업물은 `/root/VLA/v05_work/` 아래에 있다. v04 기준본과 원시 기록은 변경하지 않았다. 커밋도 없다.

## 0.5 새 서버 준비 (git clone으로 받는 경우)
`git clone https://github.com/kdh0428/VLA.git /root/VLA && cd /root/VLA && git checkout autovla-experiments-11-19`
경로는 모두 `/root/VLA` 기준 절대경로로 하드코딩돼 있으므로 반드시 `/root/VLA`에 clone한다.

**git에 포함된 것** (커밋 "handoff: CVPR review batch v05 …"에서 `git add -f`로 강제 추가):
- 프롬프트·인수인계: 이 파일, `NEXT_AGENT_PROMPT_CVPR_REVIEW_BATCH_20261023_v01.md`, `CODING_AGENT_CVPR_REVIEW_BATCH_EXECUTION_20261023_v01.md`
- `v05_work/` 전체(protocol·lock·러너·manifest·survey·A/E/lit 자료·B 부분 기록·환경 spec `v05_work/status/env_specs/`)
- `v04_apply/` 전체(v04 원고 TeX·그림·verifier·raw_packages zip — verifier가 raw_packages를 읽음)
- `vla0_work/`의 scripts, run(pilot records), obs(npz 162개), hf_6c62a51(config·tokenizer·dataset_stats.pkl; **weights 제외**), vla0 repo 소스(b78db19)와 tarball
- SpatialVLA within-call 자료 전체: `svla_wc_work/`, `integrated_v03_work/`(scripts·manifest·protocol·raw records·verifier·paper_v03)
- AutoVLA: 미추적 스크립트(py/md; `autovla_misalignment_poc/scripts/` 아래). Fig. 2 토큰 예시용 `autovla_misalignment_poc/outputs/review_followup4/B/B2/p1c_main/{cells,bank}.jsonl`은 이미 이전 commit에 들어 있음. 미추적 AutoVLA 출력 이미지·프레임(약 370 MB)은 남은 작업에 불필요해 제외(이전 서버에만 있음)
- `simpler/env.sh`

**git에 없는 것 → 새 서버에서 다시 준비** (크기는 이전 서버 기준):
| 항목 | 이전 경로 | 다시 얻는 법 |
|---|---|---|
| conda env `autovla` | /root/miniforge3/envs/autovla | `v05_work/status/env_specs/conda_autovla_env_export_20261023.yml`, `pip_freeze_autovla_20261023.txt`로 재구성(torch 2.8.0+cu128, transformers 4.49.0) |
| SimplerEnv용 transformers 4.47 등 target site | /root/VLA/simpler/site, /root/VLA/spatialvla/site (약 0.7 GB) | `target_site_dists_simpler_spatialvla_20261023.txt` 목록대로 `pip install --target` |
| SimplerEnv + ManiSkill2_real2sim(assets 포함) | /root/VLA/simpler/SimplerEnv | `simplerenv_commit_20261023.txt`의 commit으로 clone(submodule 포함) |
| Mesa lavapipe (software Vulkan) | /root/VLA/openvla/mesa (309 MB) | 이전 서버에서 rsync 권장. NVIDIA Vulkan을 쓰면 렌더러 변경을 protocol Amendment로 기록하고, 렌더 결과가 달라질 수 있으므로 기존 수치와 비교하지 않는다 |
| SpatialVLA checkpoint | /root/VLA/spatialvla/spatialvla-4b-224-sft-fractal (7.7 GB) | HF `IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal`. 파일 sha256을 `env_specs/spatialvla_ckpt_fingerprint_from_v02_main.json`과 대조 |
| VLA-0 weights | (RAM 디스크에서만 사용) | `v05_work/vla0_B/scripts/download_vla0_weights_to_shm.sh` (HF ankgoyal/vla0-libero@6c62a511, sha 2021480f…/2fd9786a…) |
| TeX Live | scratchpad(이전 세션 전용) | §5의 패키지 목록으로 재설치 |
| AutoVLA 데이터·checkpoint·기타 raw | /root/VLA/autovla*, datasets | 남은 작업(B, C, D, F)에는 불필요(E는 완료). 필요하면 rsync |

`simpler/`는 git에 올릴 수 없다(`simpler/site/sapien.libs/libsapien.so` 209 MB > GitHub 100 MB 한도, `SimplerEnv`는 중첩 git repo). 대신 `v05_work/status/env_specs/setup_simpler_spatialvla_new_server.sh`가 기록된 commit·버전으로 SimplerEnv, target site 패키지, SpatialVLA checkpoint를 다시 설치한다(이전 서버 SimplerEnv에는 로컬 수정이 없었음). git에는 `simpler/env.sh`만 들어 있다. 렌더러를 NVIDIA Vulkan으로 바꾸면 env.sh의 `VK_ICD_FILENAMES`/`LD_LIBRARY_PATH`를 고치고 protocol Amendment로 기록한다.

가능하면 `rsync -a /root/VLA/{spatialvla,simpler,openvla/mesa} <새서버>:/root/VLA/`로 직접 옮기는 편이 재설치보다 빠르고 재현성 위험도 작다.

## 1. 기준 자료 위치 (서버에 실제로 있는 것)
| 용도 | 경로 |
|---|---|
| 실행 프롬프트(우선 지침) | `/root/VLA/CODING_AGENT_CVPR_REVIEW_BATCH_EXECUTION_20261023_v01.md` |
| v04 서버 적용본(원고 기준; 덮어쓰기 금지) | `/root/VLA/v04_apply/work/CVPR_INTEGRATED_REVISION_20261023_v04/` (paper/, verifier, qa, raw_packages) |
| v04 서버 적용 결과 ZIP | `/root/VLA/CVPR_V04_SERVER_APPLY_RESULT_20261023_v01/CVPR_V04_SERVER_APPLY_RESULT_20261023_v01.zip` (sha 2162c0b2…) |
| v05 작업 디렉터리 | `/root/VLA/v05_work/` |
| VLA-0 기존 원시 기록(재사용 패널) | `/root/VLA/vla0_work/run/pilot/records.jsonl`(149 OK / 150), 관측 `vla0_work/obs/*.npz`, adapter `vla0_work/scripts/vla0_adapter.py` |
| SpatialVLA within-call 기존 자료 | `/root/VLA/svla_wc_work`, `/root/VLA/integrated_v03_work/{svla_v02,svla_drawer}` |
| SpatialVLA cross-step(P1-D) 코드 | `/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1d/`, `.../spatialvla_protection/svla_protection.py` |
| AutoVLA 원시 기록 | `/root/VLA/autovla_misalignment_poc/outputs/` (review_followup4/5 등) |

서버에 없는 입력: `CVPR_OVERLEAF_20261023_v04.zip`(프롬프트에 sha 26a829bf…가 적혀 있음)과 첨부 리뷰 `붙여넣은 텍스트 (1)(20261023-103344).txt`, `CVPR_REVIEW_RESPONSE_AND_BATCH_PLAN_20261023_v01.md`는 이 서버에 업로드된 적이 없다. 사용자가 주면 v04 서버 적용본과 diff를 떠서 변경 목록을 기록한다. 주지 않으면 서버 적용본을 기준으로 진행하고 README에 "미제공"으로 적는다.

## 2. 작업별 상태 (TASK_STATUS 초안)
| 작업 | 상태 | 근거/산출물 |
|---|---|---|
| A1 기존 근거 가시화(PDMS·FDE 임계값·비용·extrapolator·estimand 표) | 자료 DONE, 원고 반영 미실행 | `v05_work/A_materials/` (A1_driving_table, A1_fde_threshold, A2_compute_cost, A3_learned_extrapolator.md, A4_inference_table, A5_arvla_citation_check, RV 재계산 스크립트·검증) |
| A2 원고 명확화 수정(명칭, 표 재배치, Fig.2 토큰 예시 등) | 미실행 | v05 원고 사본만 있음: `v05_work/manuscript/paper_v05/`(v04 paper/ 복사, 빌드 산출물 제거) |
| B VLA-0 penalty 1.05/1.0 | PARTIAL(17/149, 중단) | `v05_work/vla0_B/` |
| B4 donor follow-up | 미실행 | 같은 러너 phase `donor` |
| C SpatialVLA full episode | 미실행(gate 전 중단) | `v05_work/svla_C/` |
| C 부가 r4e1 | 미실행 | 러너 phase `exec_r4e1` |
| D bridge-fixed/regenerated | 미실행 | B 러너 phase `bridge` |
| E1 AutoVLA 폐루프 | BLOCKED(확인 완료) | `v05_work/autovla_E/E1_CLOSED_LOOP_READINESS_20261023.json` |
| E2 day/vehicle 독립 자료 | BLOCKED_NO_DISJOINT_DATA(확인 완료) + 기술적 leave-one-component-out | `v05_work/autovla_E/` |
| F1 관련 연구 | 자료 DONE, 원고 반영 미실행 | `v05_work/lit/` (references_new_20261023.bib, LIT_STATUS_20261023.csv, related_work_paragraph_20261023.tex) |
| F2–F3 분석·v05 원고·PDF·Overleaf ZIP·최종 ZIP | 미실행 | — |

### 이미 얻은 핵심 사실(재확인 불필요, 근거 파일 있음)
- VLA-0 checkpoint `generation_config.json`: repetition_penalty = **1.05**, temperature 1e-6. native `QwenActor.forward`는 penalty를 `model.generation_config`에서 읽는다. HF RepetitionPenalty는 prompt를 포함한 전체 input_ids(강제 prefix 포함)에 적용된다. 이 점은 해석에 기록해야 한다.
- 공식 SpatialVLA SimplerEnv adapter(DelinQu/SimplerEnv-OpenVLA@ccfe3809 `spatialvla_model.py`, 사본 `v05_work/svla_C/reference/`)의 google_robot 설정: 매 step predict_action, ActionEnsembler(horizon 4, temp −0.8), sticky gripper 10회 → **r1e4**. `svla_protection.Executor(1,4)`와 산술이 같다.
- layout inventory(CPU survey, `v05_work/svla_C/manifest/survey_*.json`):
  - move_near는 registry 전체에 물리 layout이 **10개뿐**이다(80 seeds → coarse fp 10).
  - coke는 seeds 6000–6039가 모두 고유하고, 기존 survey 154개와 겹치지 않는다.
  - close/open drawer는 60 seeds에서 robot pose group **9**개, (pose, drawer) 26개다.
- E1의 blocker:
  - 카메라를 다시 렌더링하는 폐루프 시뮬레이터(CARLA)가 없다.
  - nuPlan closed-loop은 카메라를 렌더링하지 않는다.
  - NAVSIM v2 navhard 합성 이미지는 삭제됐다(재다운로드 약 14 GB 필요).
  - HF에는 AutoVLA_PDMS_89 하나만 있다.
- E2: navtest 147 logs 밖의 자료가 0개다. leave-one-component-out 결과(F의 C_NAT 0.232–0.272, C_GT 0.222–0.277; E4는 C_GT 4/6 component 양수)는 기술적 요약이며 외부 검증이 아니다.
- AR-VLA(`hu2026arvla`)는 references.bib에만 있고 v04 본문에서는 인용되지 않는다. 새 bib 키: yu2026ac2vla, xiong2026cacvla, zhong2026acotvla, yin2025deepthinkvla(v3 읽음), lu2026tacitvla, li2026thinkact5hz, renz2025simlingo.
- A 자료 주의점:
  - C_NAT E4의 severe_τ는 τ에 단조적이지 않다.
  - development RN FDE는 exp7 −4.06, R8 재실행 −3.96이다.
  - RV는 PDMS로 채점된 적이 없다.
  - P1-C H8 소요시간이 두 로그에서 다르다(63.3 vs 10.6분).

## 3. B/D (VLA-0) — 이어가는 법
- protocol(출력 전 고정): `v05_work/vla0_B/PROTOCOL_B_D_VLA0_20261023_v01.md`, lock `PROTOCOL_LOCK_B_D_20261023.txt`(protocol·러너·adapter sha256, 12:12:18Z).
- 러너: `v05_work/vla0_B/scripts/vla0_v05_run.py`. phase 순서는 validate → main → bridge(D) → donor(B4)이고, obs_id 단위로 resume된다. 누적 GPU 상한은 main 120 / D 135 / B4 225 분이다(process wall 기준).
- 현재 기록: `run/validation.jsonl` 8/8 PASS, `run/main.jsonl` 17 관측(마지막 pil_libero_goal_t3_i2). 17개 모두 p=1.05 replay가 기존 기록과 token-exact다. 사용 GPU는 약 11분이다(B process; C 로드 약 2분 별도).
- weights는 RAM 디스크(/dev/shm)에서 로드한 뒤 삭제하도록 설정돼 있다(`--release-shm-weights`, 14 GB RAM 제약). 새 서버에서는 먼저 다운로드한다:
  `bash /root/VLA/v05_work/vla0_B/scripts/download_vla0_weights_to_shm.sh` (먼저 `mkdir -p /dev/shm/vla0/model_last && cp /root/VLA/vla0_work/hf_6c62a51/model_last/* /dev/shm/vla0/model_last/`. 스크립트가 sha256를 출력하며 2021480f… / 2fd9786a…와 일치해야 한다)
- **하드웨어가 바뀌면**: 이전 17개(RTX 5090)와 새 하드웨어 결과를 섞지 않는다. 새 out 디렉터리(예: `run_newserver_<날짜>`)로 validate부터 다시 실행하고, 기존 `run/`은 "aborted partial (server 1)"로 보존·보고한다. 같은 RTX 5090 환경이면 `--out run` 그대로 resume해도 된다.
- 실행 예:
  `cd /root/VLA/v05_work/vla0_B && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<5090 index> TOKENIZERS_PARALLELISM=false && /root/miniforge3/envs/autovla/bin/python scripts/vla0_v05_run.py --out run --release-shm-weights > logs_run_resume.txt 2>&1`
- 속도: main은 관측당 약 19 s(8 generate)로 149개 ≈ 47분이다. D는 30 관측으로 약 3분, B4는 관측당 sample batch(8개) 수에 따라 달라진다.
- 분석 스크립트는 아직 없다. 프로토콜 정의대로 새로 작성한다: response D = 같은 penalty SELF 대비 6 dims |Δ|/1000 평균, task-cluster bootstrap 10,000 seed 20261023, penalty 간 paired 차이, matched·same-token-set 부분집합, B4 donor 특성 표, D의 bridge-regen − bridge-fixed. 기존 분석은 `vla0_work/scripts/vla0_analyze.py`, `vla0_posthoc_supplement.py`를 참고한다.

## 4. C (SpatialVLA full episode) — 이어가는 법
- protocol: `v05_work/svla_C/PROTOCOL_C_SVLA_FULL_EPISODE_20261023_v01.md`, lock `PROTOCOL_LOCK_C_20261023.txt`(12:22:06Z). manifest `manifest/C_manifest_20261023_v01.json`(60 units; SELF validation은 task당 처음 2개), 생성 스크립트 `scripts/build_manifest_C.py`.
- 러너: `scripts/c_run.py`(svla_wc_core.py v02 사본, sha da39b618…). phase는 `val`(6 units: N, SELF, O, L) → `main`(54 units: N, O, L) → 선택 `exec_r4e1`(task당 5 units)이다. unit 단위로 resume된다(units.jsonl의 unit_complete).
- **주의: c_run.py는 첫 episode가 끝나기 전에 중단돼 end-to-end로 검증된 적이 없다.** 먼저 다음을 한다.
  1. 별도 out 디렉터리에서 coke 1 unit 정도의 smoke를 돌린다. 버그를 고치면 protocol의 Amendment로 기록하고 lock을 다시 만든다. 이 단계는 개입 결과를 보기 전이어야 한다. smoke 결과는 분석에서 제외한다.
  2. val phase gate를 확인한다: anchor parity, SELF episode의 action·state hash가 N과 같은지, branch restore_ok, step 10 executed command 차이 = ensemble weight × a3 차이.
  3. 그 다음 main을 실행한다.
- `run/val/`에는 빈 jsonl과 run_meta만 있다(결과 없음). 새로 시작할 때는 새 out 디렉터리를 쓴다.
- 실행 예: `source /root/VLA/simpler/env.sh && cd /root/VLA/v05_work/svla_C && python scripts/c_run.py --phase val --out run/val_<날짜> --workers 6 --cap-min 60`
- 환경 의존:
  - `svla_wc_core.SVLA`는 `torch.cuda.device_count()==1`이고 장치 이름에 "5090"이 있어야 한다는 assert를 가진다. GPU가 다르면 assert를 바꾸고, 하드웨어 변경을 protocol에 기록한 뒤 parity gate로 재확인한다.
  - `simpler/env.sh`는 CUDA_VISIBLE_DEVICES=1을 하드코딩한다.
  - 렌더러는 Mesa lavapipe(software Vulkan)이다. drawer 장면은 step당 약 2.2 s로 느리다.
- 예상 비용(추정, 미검증): GPU 약 85–200분(약 20k official calls), drawer episode 113 steps.

## 5. F (원고) — 남은 일
- `v05_work/manuscript/paper_v05/`에서 작업한다(v04 paper/ 사본). 빌드: main.tex / supplementary.tex, pdfLaTeX + bibtex, CVPR 2027 style(`cvpr.sty` 해시는 v04 FORMAT_PROVENANCE.json과 대조).
- TeX Live(이전 서버에서는 scratchpad에 설치했으므로 새 서버에는 없음): CTAN net installer(install-tl, scheme-infraonly) + collection-latex + 패키지 amsfonts amsmath babel-english booktabs caption cleveref cm-super colortbl enumitem epstopdf etoolbox fancyhdr geometry graphics helvetic hyperref lineno microtype multirow natbib oberdiek psnfss times tools url xcolor xurl (+ 필요 시 tlmgr로 추가). 용량 약 220 MB.
- 그림 재생성은 `/root/miniforge3/envs/autovla/bin/python`을 쓴다(base python3에는 matplotlib이 없다). PyMuPDF/pypdf는 렌더 QA에 필요하며, 없으면 별도 target 디렉터리에 pip install한다.
- Fig. 2: `paper_body.tex`는 `fig2_design.pdf`를 참조하고, 이는 `make_design_figure.py` 출력(figures/matched_prefix_design.pdf)의 복사본이다. 네 cell 실제 토큰 예시는 `autovla_misalignment_poc/outputs/review_followup4/B/` 아래 B2 matched-prefix 레코드에서 가져온다(미확인).
- 본문 8쪽을 유지한다. LLM Usage Statement는 보존하고 도구 정보만 갱신한다. 보충자료 A–U는 누락하지 않는다. v05는 검토용 candidate로 표시한다. REVIEW_AFTER_RESULTS_20261023_v01.md를 작성한다. 최종 ZIP은 `CVPR_REVIEW_BATCH_RESULT_20261023_v01.zip`(프롬프트 §4 구성)을 `/root/VLA/` 아래 새 폴더 하나에 둔다.

## 6. 환경·운영 주의 (이전 서버 기준; 새 서버에서 재확인)
- conda env `autovla`(/root/miniforge3/envs/autovla). SpatialVLA/SimplerEnv는 `source /root/VLA/simpler/env.sh`(transformers 4.47 site), VLA-0는 같은 env(transformers 4.49, torch 2.8 cu128)를 쓴다.
- GPU: physical GPU 1 = RTX 5090(32 GB), GPU 0 = 3080 Ti. 항상 `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 쓴다. RAM은 14 GB여서 /dev/shm weights와 시뮬레이터 worker 수에 주의한다(이전에 VLA-0 + SpatialVLA 동시 실행 시 사용 RAM 약 4 GB 확인). 디스크는 이전 서버에서 여유 약 1.2 GB였다.
- shell은 zsh다. 공백으로 구분한 변수는 word-split되지 않는다. 비어 있을 수 있는 변수로 cd/rm/mv를 하지 않는다(과거 사고 있음). `rm`은 리터럴 절대경로로만 쓴다.
- 금지: 커밋, 기존 결과·원시 기록 덮어쓰기, checkpoint·weights·raw 삭제, 실행하지 않은 검사를 PASS로 표시하는 것, 사후 분석 미표기. "pre-registered" 대신 "pre-specified"를 쓴다. GPU 분은 ledger에 기록한다.
- 사용자는 단계마다 확인받지 않고 끝까지 진행하길 원한다("중단하라 한 적 없다"). 단, 사용자가 명시적으로 "중단"하면 즉시 멈춘다. 보고는 한국어로 한다.

## 7. 정리 대기 파일 (사용자 요청: 안 보는 프롬프트·zip 삭제 — 권한 거부로 미삭제)
아래는 git 미추적이고 내용이 다른 곳에 보존돼 있거나 대체된 파일이다. 사용자가 직접 삭제하거나 이전 대상에서 제외하면 된다.
- 프롬프트: `CODING_AGENT_CVPR_INTEGRATED_EXECUTION_20261023_v02.md`, `CODING_AGENT_PROMPT_20261014_v2.md`, `CODING_AGENT_SPATIALVLA_WITHIN_CALL_20261022_v01.md`, `VLA_experiment_plan_20261008.md`, `VLA_next_experiments_prompt_20261013.md`
- zip:
  - `CVPR_V04_WITH_AGENT_PROMPT_20261023_v01.zip`(76 MB, v04_apply/work로 적용 완료)
  - `CVPR_PAPER_REVISION_20261021_v02.zip`(45 MB, 이전 입력)
  - `CVPR_20261019_source.zip`
  - `CVPR_SPATIALVLA_WITHIN_CALL_RESULT_20261022_v01/…zip`(18 MB, v04 raw_packages에 같은 sha 9ca819ca… 사본 있음)
  - `CVPR_INTEGRATED_REVISION_20261023_v03/…zip`(18 MB, v04로 대체)
- 기타: `vla (1).pdf`(10-20 원고 PDF, 대체됨)
- 유지: 현재 프롬프트, `CVPR_V04_SERVER_APPLY_RESULT_20261023_v01/` ZIP(v04 기준 결과), git 추적 reproducibility zip들, v04 work 안의 raw_packages.
