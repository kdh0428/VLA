# PROTOCOL C — SpatialVLA same within-call intervention, full episodes (pre-specified, 2026-10-23)
신규 개입 출력 생성 전에 고정. hash는 PROTOCOL_LOCK_C_20261023.txt.

## 모델·경로
- SpatialVLA-4B-224-sft-fractal, unnorm fractal20220817_data/0.1.0, transformers 4.47.0 (기존 site).
- decoding: `svla_wc_core.py`(v02 최종본, 변경 없음). 매 policy call = official `generate` 동등 경로(greedy, max_new_tokens 256, B=1) + fresh generate cache(A3: transformers 4.47의 `self._cache` 재사용 제거; 공식 코드의 문자 그대로의 수치 경로와 다른 점으로 기록).
- 개입 call의 conditioning continuation: adapter `forced()` (HybridCache P+256 prefill + 1 token/forward), 기존 within-call 연구와 동일.

## 실행 설정 (공식 확인)
- 공식 SimplerEnv SpatialVLA adapter: DelinQu/SimplerEnv-OpenVLA@ccfe3809 `simpler_env/policies/spatialvla/spatialvla_model.py` 직접 확인(2026-10-23):
  google_robot: 매 env step마다 predict_action, ActionEnsembler(pred_action_horizon = action_chunk_size = 4, temp −0.8, 최신 예측 가중 최대), sticky gripper 10회, euler→axis-angle, 224×224 INTER_AREA. → 이 연구 표기 r1e4. 기존 `svla_protection.Executor(K=1,E=4)`와 산술 동일.
- 부가 비교 r4e1(4 step마다 새 chunk, chunk를 그대로 실행): 주 실험 완료 후 자원이 남을 때만, task당 manifest 순서 처음 5 units. 이 설정에서 anchor는 env step 8(= policy call 2)로 고정해 과제 진행 단계를 맞춘다(주 실험의 policy call 8 = env step 8).
- horizon: 각 env의 공식 TimeLimit(truncation)까지 실행(coke/move_near 80, drawer 113 예상; 실제 값 기록). early stopping 없음. success = 공식 SimplerEnv 규칙(마지막 step의 done/success), 보조로 ever-success와 최초 success step 기록. 모든 arm 동일.
- settling step 없음(SimplerEnv reset 직후 step 0이 policy call 0). env step = policy call index (r1e4). token generation index는 a1..a4 × 3 tokens + EOS.

## 표본 (manifest/C_manifest_20261023_v01.json, CPU survey만으로 생성)
- coke(google_robot_pick_coke_can): seeds 6000–6019, 20 독립 layout(coarse fp가 기존 survey 154개와 겹치지 않음).
- move_near: registry 전체에 물리 layout 10개만 존재(survey 80 seeds의 coarse fp 10개). layout당 서로 다른 instruction 2개 state(seed 순 처음) = 20 states / 10 독립 layout. layout은 기존 연구에서 사용된 것과 동일(재사용 명시).
- 세 번째 과제 close drawer(google_robot_close_drawer): SpatialVLA 공식 SimplerEnv 평가 과제, 학습 불필요, 기존 open-drawer within-call 자료와 과제 분리. seed 6000–6059 survey에서 (robot pose, target drawer) 고유 조합 처음 20개; robot pose group 9개 → cluster = pose group. 출력·효과를 보고 고르지 않음.
- SELF full-episode validation: task당 manifest 처음 2 units.

## 개입 (anchor = zero-based policy call 8, env step 8)
- 공통 natural 궤적 N을 reset(seed, reconfigure=True)부터 실행. anchor 직전 executor(chunk buffer, ensemble buffer, sticky gripper 상태) deep copy와 N의 float64 action 기록.
- 다른 arm: 같은 seed로 reset 후 N의 action t<8을 replay(디코딩 없음), 매 step state hash·image hash를 N과 비교(restore_ok), executor 복원 후 anchor부터 자기 관측으로 폐루프.
- anchor call: 자연 chunk a1..a4(12 tokens) = official greedy. emitted prefix a1/a2는 모든 arm에서 자연값으로 실행.
  - SELF: conditioning = 자연 a1,a2 → continuation a3,a4 (자연과 token/EOS 동일해야 함).
  - O: conditioning a1 = donor a1(translation·rotation; gripper 자연과 동일 조건), a2 자연.
  - L: conditioning a1 자연, a2 = donor a2.
  - 실행 chunk = 자연 a1,a2 + arm continuation a3,a4 → 공식 unnormalize → executor. 편집 conditioning token을 실행 prefix에 덮어쓰지 않음.
- donor: anchor 관측에서 official T=0.7(top_p 1, top_k 0) sample 최대 64개, seed = sha256("svla_v05_C_src:"+unit_id+":"+j)[:4] LE; v02 규칙 그대로 첫 joint-valid & distance-matched(a1,a2 type-valid·EOS 없음, gripper token 동일, d_O>0, d_L>0, 0.8 ≤ d_L/d_O ≤ 1.25; d = 정규화 6-D motion 평균 |Δ|). suffix 반응·성공 여부 사용 안 함. 없으면 SOURCE_MISSING → O/L 미실행(N은 보고).
- anchor 이후 arm은 개입 없이 official decoding.

## 결측·gate 규칙
- anchor 자연 출력이 invalid(12 type-valid tokens 아님 또는 EOS 위치 이상) → NATURAL_INVALID_AT_ANCHOR, arm 미실행.
- anchor parity(adapter natural 13 tokens == official, SELF forced a3a4+EOS == official) 실패 → PARITY_FAIL, arm 미실행·분리.
- branch restore_ok 실패 → 해당 branch 분리(RESTORE_FAIL), 분석 제외·보고.
- arm continuation이 type-invalid/EOS 포함 → ARM_INVALID, 분리 보고.
- 비 anchor call의 invalid 출력은 공식 adapter와 동일하게 처리(decode 가능한 범위), 횟수 기록.
- SELF validation 실패(SELF full episode의 action·state hash가 N과 다름) → 결정성 gate FAIL: 원인 수정 전 결과를 primary로 쓰지 않음.

## 분석 (사전 고정)
- unit = (task, state); cluster = 독립 layout(coke·move_near) 또는 robot pose group(close drawer). task별 보고가 primary; 3 task 평균은 task별 결과 옆 기술적 요약.
- decoding: D_S(arm) = anchor call a3,a4의 정규화 6-D motion 평균 |arm − N|; L−O.
- executed command: anchor 이후 env action 차이(‖Δworld‖, ‖Δrot‖), 최초 command 차이 step, chunk 8의 a3/a4 전달 가중치(official ensemble weight), delivered dose = weight × (a3/a4 unnormalized world-vector 차이).
- motion: TCP 위치 paired 거리 |arm − N| at env steps 12, 16, 24, 40, 최종; 최초 운동 차이 step(TCP 위치 차 > 1e-6 m).
- success: L−N, O−N, L−O (공식 success); cluster bootstrap 10,000(seed 20261023, percentile 95%), cluster 수 ≤ 1이면 NA. 추가로 discordant pair 수.
- 연결 도표: unit별 D_S → command dose → TCP 거리 → success 변화.
- coverage: attempted → natural valid → parity pass → donor found → arm valid → restore ok → completed, task/state별.
- 기존 cross-step success와 pooling하지 않음. equivalence 검정은 하지 않음(±10pp 등 margin 상속 없음). 비유의를 invariance로 쓰지 않음.

## GPU 상한(이번 세션 설정)
- C main ≤ 200 GPU-min(모델 적재 process wall time 기준), r4e1 부가 ≤ 40 GPU-min. 초과 시 manifest 순서상 진행 중 unit까지 완료 후 PARTIAL. GPU 1(RTX 5090), B 실험과 동시 실행 가능(각 process B=1; 재현성은 SELF/replay gate로 확인).

## 실행 세부 (lock 전 추가)
- 실행 순서: task 간 round-robin(manifest 순서). phase val = SELF validation units 6개(N, SELF, O, L); gate 통과 후 phase main = 나머지 54 units(N, O, L).
- unit 내부 arm 순서 고정: N → SELF(val만) → O → L. 각 policy call은 B=1, 같은 model process에서 worker 순서대로 직렬 디코딩.
- 디코딩 불가 출력은 직전 decoded chunk를 재사용(기록: decode_error), chunk가 4 actions 미만이면 마지막 action으로 padding(기록: padded).
- 안전 상한 400 steps(TimeLimit이 먼저 적용되는 것이 기대값).
