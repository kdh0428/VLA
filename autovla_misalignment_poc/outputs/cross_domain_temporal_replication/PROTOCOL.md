# 실험 34: 다른 backbone·domain에서의 previous-action temporal feedback 재현 — 사전 등록

작성: 2026-10-05, 본 실행 전(모델·환경 smoke test만 실행: SimplerEnv 4 episode × 3 조건, 디코더 동등성 6 관측).
결과를 보고 정의·threshold·판정 기준을 바꾸지 않습니다. 새 결과는 이 디렉토리에만 저장하고, RTX 5090만 사용합니다.

## 0. 모델 선정 (코드 수준 확인)

| 후보 | backbone | action 표현 | chunk | 시간축 AR dependency | 이전 action 입력 위치 | 개입 위치 | benchmark | 판정 |
|---|---|---|---|---|---|---|---|---|
| AR-VLA (`insait-institute/AR-VLA-lerobot`, ART) | ResNet + transformer(ACT 변형); generalist는 별도 | 연속/k-means token, step마다 1개 | 1 step씩 streaming | decoder 입력이 **관측 proprio state**(`state_input_proj(observation.state)`), 자기 action token을 다시 넣지 않음 | 환경 상태 | state KV cache | PushT/ALOHA, SimplerEnv(generalist) | 제외: 앞 action이 아니라 관측 상태를 conditioning |
| VQ-VLA (`VQ-VLA/vq-vla-weight`) | OpenVLA (Llama-2) | causal-conv residual VQ: chunk 5 step → latent 1개 → residual code 4개 | 5 | **없음** — 4 token이 시간 step이 아니라 residual 단계(coarse→fine) | – | – | LIBERO-90 | 제외: chunk 안 시간 순서 없음 |
| WorldVLA / RynnVLA-002 (`Alibaba-DAMO-Academy/WorldVLA`) | Chameleon-7B | step당 7 token(256 bin), chunk 5–10 | 5 | **차단** — `generate_att_mask*`가 각 action 블록(10004…15004)이 이전 action 블록을 보지 못하게 함(학습·추론 `att_mask=True`) | – | – | LIBERO | 제외: 이전 action conditioning을 설계상 제거(오류 전파 방지 목적) |
| π0-FAST | PaliGemma | DCT 계수 BPE | 10–50 | 주파수 계수 순서(시간 순서 아님) | – | – | LIBERO | 제외(실험 33) |
| **SpatialVLA-4B** (`IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal`) | **PaliGemma2-3B (Gemma2) + SigLIP + Ego3D(ZoeDepth)** | step당 3 token: translation(구면 θ16 × φ32 × r8 = 4,096), rotation(16³), gripper(2) | **4 step** | **있음** — prefix 뒤 suffix를 causal로 greedy 생성; step k의 token이 step < k의 token을 attend(추론 mask `_update_causal_mask`, `generate`) | 같은 chunk의 앞 step token(문맥) | 문맥 token 교체(실행 token과 분리), KV cache | SimplerEnv Google Robot(fractal) | **채택** |

SpatialVLA 세부: transformers 4.47(모델 코드 요구), bf16 8.1 GB GPU, 디스크 7.6 GB. SimplerEnv(ManiSkill2_real2sim, SAPIEN 2.2.2)는 Mesa lavapipe
소프트웨어 Vulkan으로 실행(NVIDIA Vulkan 없음). 추론 모드 mask가 입력 전체를 양방향으로 보므로, 강제 token은 프롬프트와 함께 넣지 않고
`generate`처럼 prefill 후 cache로 하나씩 넣는 디코더(`spatialvla_core.decode_step` / `decode_chunks`)를 쓰며, 6개 관측에서 `generate`와 token이 완전히 같음을 확인했습니다.

공식 실행 방식(DelinQu/SimplerEnv-OpenVLA `spatialvla_model.py`, google_robot): 매 control step chunk 예측, 실행 action = 최근 4개 chunk의
해당 step 예측을 ActionEnsembler(temp −0.8; 오래된 예측일수록 가중치 큼)로 평균, sticky gripper. 따라서 실행 action의 대부분은 앞 step에
조건화된 chunk의 뒤쪽 step(2–4)에서 옵니다.

AutoVLA·Impromptu와의 차이: chunk가 4 step으로 짧아 첫 step 편차 뒤 교정할 수 있는 문맥 위치가 step 2·3뿐이고(window 1–2 step),
greedy 결정적 디코딩이라 "문맥에서 교란 제거/재삽입"은 token 수준에서는 자명하게 정해지므로 그 효과는 closed loop에서 측정합니다.

## 1. 과제와 에피소드

SimplerEnv Google Robot `google_robot_pick_coke_can`, `google_robot_move_near`, 환경 seed 0–39(과제당 40, 총 80 episode), 80 step,
성공 = 마지막 step의 `done`(공식 평가와 동일). cluster = 에피소드(과제, seed).

## 2. Experiment A — 자연 재실행 변동 (closed loop)

`natural`(step-wise 디코더, 자연 행끼리 batch), `natural_mix`(같은 디코더, 개입 조건 행들과 같은 batch), `natural_gen`(모델 자체 `predict_action`,
batch 1). 지표: 성공 불일치율, 성공률 차이(McNemar), 실행 action 궤적 발산(누적 world_vector 끝점 차이). 이 값이 noise floor입니다.

## 3. Closed-loop 개입 (Experiment B·C·D의 task 수준)

control step 8–23에 생성되는 모든 chunk에서 step-1 translation token g를 교란 token p로 바꿉니다. p = 정규화 공간에서 v(g) + d·u에 가장 가까운
translation token, d = 0.3, u ∈ {perp_left(수평면 좌측 90°), opposite(반대 방향)}.

| 조건 | 실행 step-1 | 문맥 step-1 (step 2–4 생성 시) |
|---|---|---|
| feedback | p | p |
| corrected (원인 제거) | p | g |
| reverse (원인 재삽입) | g | p |

지표: 성공률, 자연(natural) 대비 paired 차이, 실행 궤적 발산. 9 조건(natural, natural_mix, natural_gen, 3 모드 × 2 방향) × 80 episode.

## 4. Token 수준 실험 (B·C·D·E·F) — natural rollout의 4 step마다 저장한 관측(약 1,600 프레임)

단위 = (프레임, d, u), d ∈ {0.15, 0.30}, u ∈ {along, opposite, perp_left, perp_right, up, down}; 달성 거리가 [0.5d, 1.5d] 밖이면 제외.
기준 chunk R = 교란 없는 greedy chunk. 교란 행: step-1 translation = p(문맥·실행 모두), step-1의 rot·grip과 step 2–4는 생성.

| 행 | step k(=3, 4) 생성 시 문맥의 step 2..k−1 |
|---|---|
| normal | 자기 생성 |
| recent_ref | 마지막(k−1)만 R, 나머지 자기 생성 |
| full_ref | 모두 R (step 2·3) |
| win1 | step 2만 R, step 3은 자기 생성 (교정 window 1 step) |
| **reverse** | step 4 생성 시 [step 2 = R, step 3 = **normal 행이 생성한(오차를 품은) step 3**] — 교정된 문맥에 잘못된 직전 action만 재삽입 |
| motion: ref_trans / near_ref / dir_ok_mag_wrong / dir_wrong_mag_ok / random_mag_matched | 마지막 문맥 step의 translation token만 대체(rot·grip은 자기 생성): R의 token / R과 φ 또는 θ 한 칸 다른 token / R과 같은 (θ, φ), r을 바꿔 거리 ≈ e / R과 같은 r, (θ, φ)를 바꿔 거리 ≈ e / R과 같은 r의 무작위 (θ, φ) (e = 자기 token과 R의 거리) |

지표(정규화 translation 공간): 하위 발산 D = Σ_{k=2..4} ||v_k − r_k||, 증폭비 A = ||Σ_{k=2..4}(v_k − r_k)|| / ||v_1 − r_1||(주입 대비 추가 누적 편차),
**증폭 = A ≥ 1**, **회복 = step 2–4 translation token이 R과 모두 같음**, 누적 끝점 편차 E4 = ||Σ_{k=1..4}(v_k − r_k)||.
B: 같은 프레임·같은 d에서 회복 방향과 증폭 방향의 공존 비율, 방향별 증폭률, 분산 중 프레임 간/프레임 안 비율.
token 수준 기저 변동: 같은 관측의 `generate` vs step-wise 디코더 chunk 불일치율.

## 5. 통계

에피소드 cluster bootstrap 95% CI(2,000, seed 0); 이진 paired McNemar, 연속 paired Wilcoxon; 모든 효과를 자연 재실행 변동과 직접 비교.

## 6. 성공 판정 (사용자 기준을 그대로 고정)

- **Strong replication**: (1) correction이 하위 증폭을 유의하게 줄이고(token: recent_ref/full_ref vs normal; closed loop: corrected vs feedback),
  (2) reverse 삽입이 증폭을 다시 늘리며(token: reverse vs full_ref, step 4 기준; closed loop: reverse vs corrected 또는 natural),
  (3) 효과가 자연 재실행 변동보다 큼. temporal window(win1 vs full_ref 차이)가 나오면 추가 증거.
- **Partial replication**: token/action 수준 feedback은 있으나 task/trajectory 실패까지 이어지는 효과는 불확실.
- **No replication**: correction/reverse 모두 자연 변동과 구별되지 않음.
