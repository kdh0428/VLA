# Driving VLA 실패 기전 연구 — 전체 실험 총정리

최종 갱신: 2026-10-18 (review_followup4 반영: §6g 신설; A1–A4·B1·B2·B3 근거의 사실 정정은 해당 줄에 '(2026-10-18 수정)'으로 표시). 이전 갱신 2026-10-13 (review_followup2 사후 재분석 반영: drive·초기 상태 군집, SpatialVLA 직접 효과, 외삽 baseline·교정 window 재점검; §1·§3·§6b–§6e·§9·§10 개정). (2026-10-18 수정)
이전 갱신 2026-10-12 (후속 검증 P0–P1 §6e, 논문 수치 패키지 `paper_quantitative_package/`). 브랜치 `autovla-experiments-11-19`. 실험별 상세 보고서는 각 `outputs/<실험>/` 디렉토리에 있고,
누적 결론 문서는 [CONCLUSIONS.md](CONCLUSIONS.md), 새 서버 인계는 [HANDOFF.md](HANDOFF.md)입니다.
주장별 유지·축소·삭제 판정은 `autovla_misalignment_poc/outputs/review_followup2/CLAIM_EVIDENCE_MATRIX.md`, 새 실험 미수행 근거는 같은 디렉터리의 `GAP_DECISION.md`, 자료 감사는 `DATA_AUDIT.md`를 따릅니다.

- 주 대상: **AutoVLA** (Qwen2.5-VL-3B, action codebook 2,048 token, NAVSIM/nuPlan). 비교: ORION (Bench2Drive, 실험 1), **OpenVLA-7B** (LIBERO, 실험 28), **Impromptu VLA 3B** (navtest, 실험 33), **SpatialVLA-4B** (SimplerEnv 로봇 조작, 실험 34).
- 통계(2026-10-13 개정):
  - AutoVLA·Impromptu 기전 대비의 독립 단위는 **drive**입니다(A− 16 log = 13 drive, Impromptu 28 log = 20 drive, P1-C eval 24 drive). [ ]는 drive cluster bootstrap 95% CI, p는 **drive 합에 대한 exact sign-flip**입니다. 13 drive에서 가능한 최소 p는 2/2^13 = 2.4e-4, 20 drive에서는 1.9e-6입니다. 단위 수준 McNemar p(8e-42, 6.7e-31, 2e-31, 7e-261 등)는 표본을 과대 계산한 값이라 쓰지 않습니다.
  - SpatialVLA closed loop의 독립 단위는 **과제별 초기 상태 cluster**입니다(coke는 seed마다 고유; move_near는 지시문 포함 primary 29/53/120개, **물리 배치 10개**). p는 cluster sign-flip(δ = 0에서 유효)이고 CI는 사후 수정 구간(Rao–Scott design-effect Newcombe 10, 과제 평균은 MOVER; 원 sign-flip 역산 CI는 이진 쌍 자료의 경계에서 무효, review_followup4 A1)이고 primary와 physical 두 clustering을 함께 보고합니다. (2026-10-18 수정)
  - sign-flip은 drive(또는 cluster) 합이 H0에서 독립·0에 대칭이라는 가정 아래의 조건부 검정이며, 무작위 배정 검정이 아닙니다.
  - 그 밖의 표(실험 1–4, 12–31 등)는 원 표기(log·clip·과제 단위 cluster bootstrap, McNemar/Wilcoxon, navhard log 단위 sign-flip)를 그대로 둡니다.
- 재현성 규칙: held-out·검증 실험은 결과를 보기 전에 프로토콜/사전 등록을 커밋했고(커밋 해시 표기), threshold와 규칙은 결과를 보고 바꾸지 않았습니다.
  예외와 시점은 `review_followup2/DATA_AUDIT.md` §4·§7에 정리했습니다: AutoVLA 기전 실험 4–11은 사전 등록이 없는 탐색 실험이고, P1-C는 pilot 기준 완화(수정 1, 평가 자료 전)와 H8 OOM 재실행(수정 2, RN·DM 결과 열람 후)이 있었습니다.
  기존 결과 파일은 덮어쓰지 않았습니다. 실험 25 이후의 모든 새 GPU 작업은 RTX 5090만 사용했습니다.
- 표기: **[사전 등록]** = 결과 전 commit된 판정, **[사후]** = 결과를 본 뒤의 재분석(review_followup2 A–C 포함).

---

## 1. 한눈에 보는 결론

1. **실패는 perception보다 결정·실행 결합 단계에서 나고**, CoT 텍스트는 action과 느슨하게 결합되어 있습니다(실험 1–4).
2. **직전 action 문맥이 이후 생성을 인과적으로 바꿉니다**(AutoVLA, 실험 7·8은 탐색 실험; R8R9 복구; P1-C 새 drive [사전 등록]).
   - **매 생성 step 직전 위치를 GT로 바꾸는 반복 개입**(Recent-GT; 단위당 평균 6.8회 적용, 실제 token 변경 4.3회)이 증폭을 **−39.8%p [−49.0, −29.9]** 줄이고, 자기 이전 action(별도 Normal 행의 embedding)을 다시 넣으면 **+28.6%p [+20.9, +36.1]** 복원됩니다(13 drive, 두 대비 모두 exact sign-flip p = 2.4e-4 = 13 drive의 최소값, Holm 7.3e-4) [사후 drive 재분석].
   - 기전 실험과 drive가 겹치지 않는 **새 drive**(P1-C; 0 drive 공유, day–vehicle 10 공유)에서 재현 [사전 등록]: 실패 층 F(105 장면 / 18 drive) RN −34.2%p, RV +26.8%p; 평가 집합 E 추정량(적격 2,191 장면, 선정 353 장면 가중, 24 drive) −4.1 / +3.6%p. RN 평균 효과의 약 60%는 A+(비실패) 층에서 오므로 원래 실패 장면의 복구로만 설명하지 않습니다. (2026-10-18 수정)
   - "직전 token 1개로 93%"는 **반복 교정이 full GT-history 교정 효과의 92.9% [89.4, 99.1]**를 회복한다는 뜻입니다(분모 = GT-history 효과; Normal 증폭률 대비로는 85%). 한 번의 1-token 교정이 아닙니다.
   - 교정 길이: 고정된 3개 문맥 위치를 GT로 바꾸고 이후 문맥에도 그대로 유지하면(persistent GT, 해제 없음) full-history 효과의 **약 90% [81, 99]**를 회복합니다(1/2/3/4 step = 47 / 77 / 90 / 96%, scene 가중; 실험 9 원 unit 가중은 47 / 76 / 90 / 96%; R9, 13 drive). "2–4 step이면 90%"는 틀린 표현입니다. 교정 위치 수와 마지막 GT 위치 이후 생성 step 수(3개일 때 평균 3.8)가 함께 변합니다. 단위당 GT 위치 2.96개 중 자기 token과 다른 것은 1.90개입니다(review_followup4 A2). (2026-10-18 수정)
   - **초기 위치 특수성(early critical window)은 근거가 없습니다**: 같은 길이의 교정을 첫 이탈 직후와 2 step 뒤에 넣은 차이는 w = 1/2/3에서 +10.7 / +4.3 / +0.5%p로 모두 비유의입니다(P1-B [사전 등록], 12 drive).
3. 그 직전 token에서 embedding 근접보다 **motion**이 중요하고(실험 11), 방향 오답이 크기 오답보다 해롭습니다(실험 32).
   - GT와 motion이 가장 가까운 다른 token은 GT token과 차이가 작고 비유의합니다(+0.8%p [−1.6, +6.6]). 그러나 **±5%p 동등성은 성립하지 않습니다**(TOST p = 0.09). "identity와 무관/GT와 같다"로 쓰지 않습니다.
   - 방향 − 크기: 개발 13 drive에서 +9.9%p [+3.9, +21.0], p = 0.0056이지만, **새 drive(P1-C)에서는 같은 방향 +6.2%p, p = 0.073으로 결론이 나지 않았습니다**(사전 등록 판정 inconclusive). 일반화 주장은 하지 않습니다.
3b. **GT 없는 action-context 효과(AutoVLA 한정, P1-A [사전 등록])**: GT 정보 없이 문맥 token 하나만 교란해도 이후 local action이 +0.57 m/seg 변하고, 순수 좌표 누적(기하 baseline)보다 endpoint가 +4.37 m [+3.32, +5.19] 더 움직입니다(13 drive; **사후** C1 excess 지표, Holm m = 4 9.8e-4). 사전 정의 P1A-2 지표의 authoritative 행은 +4.37 m [+3.34, +5.17], Holm m = 3 7.3e-4이며 두 행을 섞지 않습니다 (2026-10-15 수정).
   - 등속(constant-motion) 외삽 대비 초과 변화는 **실패 선택 층(A−)에서만** +2.35 m [+1.27, +3.19]이고, A+에서는 −0.52 m, 선정 기전 집합 전체(208 장면 = A− 52 + A+ 156)에서는 +0.20 m [−0.32, +0.56](p = 0.51)로 차이가 검출되지 않았습니다(동등성 미검정) [사후 C1]. "외삽을 넘는 일반적 자기증폭"은 삭제합니다. (2026-10-18 수정)
   - GT 없는 reference 교정(−3.50 m)은 모델 자신의 rollout과의 **일관성 회복**이며 GT 기준 실패를 줄이지 않습니다(A− 증폭 +19.7%p 악화).
4. **OpenVLA-7B(조작, LIBERO-Spatial)에서도 step 안 token 수준 의존성이 관찰됩니다**(기술적: 개입 경로 parity 미검증, B3 미실행, 핵심 증거에서 제외): 한 step 안에서 x token을 ±8 bin(명령 약 2.6 mm/step) 바꾸면 단위의 92.6%(4,741 단위; 초기 상태 군집 CI [90.8, 94.0])에서 하위 성분(y, z, roll, pitch, yaw; gripper 제외) 중 적어도 하나의 bin 변화가 실제 주입 bin 이상이었고(translation 성분만으로는 72%; 명령값이며 실현 운동·과제 실패가 아님), (2026-10-18 수정)
   이를 결정하는 것은 문맥에 들어간 이전 token입니다. 그러나 OpenVLA는 이전 action을 다음 step에 넣지 않아 feedback이 시간축으로 누적되지 않고,
   LIBERO 과제 성공률 효과는 검출되지 않았고(C − F 8-bin −3%p [−11.9, +5.9], Newcombe 10; 동등성 미성립), 자연 재실행끼리도 결과가 20% 달랐습니다(이 20%를 효과 검출 임계값으로 쓰지 않음)(실험 28). (2026-10-18 수정)
5. **시간 순서가 있는 action chunk를 autoregressive하게 생성하는 다른 VLA(Impromptu VLA 3B)에서도 문맥 교정·재삽입 효과가 관찰됩니다**(기술적·미검증: 사전 등록 판정은 Strong이었지만 batch·decoding 경로 수치만으로 장면 수준 판정이 5–6% 바뀌고 경로 parity가 검증되지 않아 강한 독립 재현으로 세지 않습니다, review_followup4 A3; 실험 33): (2026-10-18 수정)
   매 waypoint 생성마다 직전 문맥 waypoint를 GT로 바꾸면 증폭 63.4 → 1.3%(−62.2%p [−65.4, −59.1], 20 drive, exact sign-flip p = 1.9e-6 = 최소값), 자기 오차 waypoint를 다시 넣으면 0 → 81.9%.
   - Impromptu의 Reverse는 **자기 branch의 오차를 다시 넣는** 처치이고, AutoVLA의 Reverse(별도 Normal 행 embedding 삽입)와 다른 처치입니다. 증폭 정의·효과 크기는 모델 간에 비교하지 않습니다.
   - 절대 좌표 표현이라 GT 문맥 교정에는 GT 정보 누출이 섞입니다. GT 근접 waypoint ≈ GT(p = 0.87)도 동등성 검정이 아닙니다.
   - Impromptu는 작은 편차를 거의 흡수하지 못해(0.2 m 편차의 97%가 3배 이상 커짐) 궤적 수준의 안정 분기는 재현되지 않았습니다.
5b. **SpatialVLA-4B(Gemma2, 로봇 조작)에서는 token 수준 문맥 효과가 재현되고, 생성 action과 실제 TCP 궤적도 바뀌지만, 성공률 효과는 작은 범위로만 제한할 수 있습니다**(실험 34 Partial, 실험 35, P1-D [사전 등록]; 직접 효과 재분석 [사후]).
   - token 수준: 직전 action 교정으로 step-4 편차 −82%, 교란 token 재삽입으로 복원(episode cluster p ≤ 5e-6; 초기 상태 군집 재분석은 하지 않음).
   - closed loop: 문맥 교란은 생성 suffix token의 50–58%를 바꾸고 실제 TCP를 평균 2.7–6.0 cm 이동시킵니다(P1-D·실험 35; 실험 34의 "0.21–0.23 m"는 실제 TCP가 아니라 **실행 명령의 누적값**이며 실험 34에는 TCP 기록이 없습니다).
   - 성공률 직접 효과(δR = R − N, δC = C − F): P1-D 공식 r1e4와 r4e1에서 **두 과제 평균(pooled)**은 ±10%p 동등(초기 상태 두 clustering, Holm)이지만 ±5%p 동등은 성립하지 않고, **과제별로는 CI 폭 최대 18–25%p(P1-D r4e1; r1e4 13–20%p; A1 사후 수정 구간)로 불확실**합니다(2026-10-15 판의 '18–29%p, r1e4 6.9–21.6%p'는 경계에서 무효인 sign-flip 역산 CI 기준). 실험 34/35 결과는 초기 상태 군집 후 모두 불확실입니다. (2026-10-18 수정)
   - 실행된 교란 자체는 성공률을 낮춥니다(C − N −11.7 ~ −22.9%p). 실패는 주로 실행된 action에서 옵니다.
5c. **실행 구조 (실험 35 → P1-D).**
   - 재계획 간격을 늘리면 문맥 교란이 만든 궤적 편차가 더 오래 남습니다. 문맥 교란 예산을 고정해도 R − N 궤적 면적 DiD +214 cm·step [+171, +255](P1-D H1 지지). 단 관측 갱신·문맥 reset·실행 노출이 함께 바뀌는 **복합 설정 효과**입니다.
   - **실험 35의 "4 step 재계획에서 reverse가 실패를 늘린다(−6.25%p, p = 0.04)"는 삭제합니다**: 초기 상태 군집 후 p = 0.14(primary) / 0.26(physical)이고 P1-D에서 +1.25%p [−5.6, +8.0]로 재현되지 않았습니다.
   - **"ensemble이 보호/흡수한다"는 주장도 삭제합니다**: 실행 교란량을 맞추면 ensemble 축 DiD +21 cm·step [−25, +68]로 사전 정의 ±100 범위 안이고 방향도 흡수 쪽이 아닙니다(P1-D H2). P0-C의 ensemble 감쇠는 실행 교란량 차이로 설명됩니다.
6. **불안정 rollout은 이탈 전에는 약하게만 예측됩니다**: 엔트로피·likelihood·margin·후보 합의·hidden-state probe 모두 이탈 전 AUROC 0.57–0.63(장면 내 0.50–0.65),
   이탈 후 0.75–0.90입니다(실험 22, 31).
7. **GT 없는 완화**: 참조로 조건화를 교정하면 정상 장면이 손해를 봐 순이득이 없고(실험 12–17), 모델 자신의 후보 중 확신도로 고르면 open-loop 실패율 −33%,
   사전 등록 held-out PDMS +0.0095로 유의합니다(실험 20–21). 단 held-out은 dev와 **log만 분리되고 drive 22개를 공유**합니다(held-out 장면의 85%; "held-out logs, not drive-disjoint").
8. **반응형 2단계 평가(NAVSIM v2 navhard 전체 225 그룹)에서는 확신도 선택의 이득이 작고 불안정**하며(+0.02, max log-lik은 사전 등록 절반에서 미재현),
   **현재 frame 안전 필터가 이득의 대부분**을 만듭니다(EPDMS 0.229 → 0.333, +0.105; 선택기를 더해 0.347–0.350). 필터 이득의 86%는 주행 가능 영역 제약이고,
   진행도를 희생해 얻은 점수가 아닙니다(실험 25–27, 29). navhard 두 절반도 log만 분리되고 drive 16개를 공유합니다.
9. **남은 병목은 후보 생성**입니다: 17개 후보 중 oracle을 골라도 EPDMS 0.405이고, token의 31%는 안전한 후보가 하나도 없습니다(실험 30).
10. 연속 closed-loop(CARLA) 평가는 checkpoint·렌더링·디스크 제약으로 수행하지 못했습니다(P6).

---

## 2. 실험 목록

| # | 실험 | 핵심 결과 | 보고서 |
|---|---|---|---|
| 1 | P/R/A 실패 분해 (ORION, AutoVLA) | 실패는 perception보다 결정·실행 결합 단계 | `pra_comparison/outputs/PRA_COMPARISON.md` |
| 2 | CoT 인과 개입 | 선언 방향을 뒤집어도 실행 반전은 +6.4%p, CoT는 느슨하게 결합 | `outputs/cot_intervention/` |
| 3 | Fast vs CoT (무편향 500 장면) | CoT 강제 시 A+ −5.0%p | `outputs/fast_vs_cot_unbiased/` |
| 4 | Natural/Fast 기전 재검증 | late action formation, L35 MLP 지배 재현 | `outputs/natural_fast_mechanism/` |
| 5 | 첫 mismatch 인과 | 첫 token 하나 교정으로 A− 장면 A+ 17 → 73% | `outputs/first_mismatch_causal/` |
| 6 | 등거리 perturbation | 같은 크기 편차가 장면에 따라 증폭(45%)/회복으로 갈림 | `outputs/equal_distance_perturbation/` |
| 7 | Action history 인과 | 매 step 직전 위치를 GT로 바꾸는 반복 교정이 증폭 47 → 7% (GT-history 효과의 93%; drive p = 2.4e-4) | `outputs/action_history_causal/` |
| 8 | Layer-wise state patching | 특정 layer가 아니라 residual 전체가 운반 (원 raw 없음, R8로 복구) | `outputs/prev_action_state_patching/` |
| 9 | Temporal feedback window | 1/2/3/4 step 교정이 효과의 47/77/90/96% (scene 가중; 원 unit 가중 47/76/90/96; 3 step ≈ 90% [81, 99]; 원 raw 없음, R9로 복구) | `outputs/temporal_feedback_window/` |
| 10 | Horizon-controlled window | horizon 착시가 아님; 이전 표기 '해제 후 감쇠'는 w개 GT 위치가 문맥에 남아 있는 동안의 재발산(summary 수준) | `outputs/horizon_controlled_window/` (2026-10-18 수정) |
| 11 | 직전 token identity 분해 | embedding이 아니라 motion(codebook 기하)이 결정 | `outputs/prev_action_identity_decomposition/` |
| 12–17 | GT 없는 참조로 조건화 교정 | 모델이 참조를 66–89% 따라가 정상 장면 손해, 배포 조건 순이득 없음 | `outputs/*reference_stabilization*/` |
| 18–19 | Best-of-N 선택 | 조건화 대신 확신도 선택이 실패 장면을 구함 | `outputs/best_of_n_selection*/` |
| 20 | 표본 확대 (새 log 56개) | rank-sum: A− 2.15 → 1.45%, FDE −36% | `outputs/expanded_best_of_n/` |
| 21–24 | 사전 등록 held-out 52 log (log 분리, drive 22개 dev와 공유) | PDMS +0.0095, 안전 필터 F1 +0.032 | `outputs/selection_validation/` |
| – | 3080 Ti 결과 5090 재실행 | 결론 동일 | `outputs/gpu5090_reanalysis/` |
| 25 | NAVSIM v2 navhard 절반 | F1 EPDMS +0.125 | `outputs/navhard_eval/` |
| 26 | navhard 나머지 절반 사전 등록 + 전체 | F1 재현(+0.112), max log-lik 미재현 | `outputs/navhard_full_validation/` |
| 27 | 안전 필터 ablation | F1 개선의 ~89%가 필터 자체 | `outputs/safety_filter_ablation/` |
| 28 (P1) | **다른 VLA(OpenVLA)에서 기전 재현** | step 안의 token feedback은 재현(첫 token 8 bin 교란 → 하위 성분 중 적어도 하나가 주입 bin 이상 변한 단위 92.6%(기술적), 문맥 token이 결정), 과제 실패 효과는 기저 변동 안 | `outputs/cross_vla_replication/` (2026-10-18 수정) |
| 29 (P2) | 안전 필터 구성요소 | 주행 가능 영역 제약이 86% | `outputs/safety_filter_components/` |
| 30 (P3) | 후보 oracle 상한 | oracle 0.405 vs 최선 0.350, 병목은 생성 | `outputs/candidate_oracle/` |
| 31 (P4) | 불안정 탐지 baseline | 이탈 전 AUROC 0.57–0.63(hidden probe 포함), 이탈 후 0.75–0.90 → 이탈 전에는 예측 불가 | `outputs/instability_detection_baselines/` |
| 32 (P5) | previous-action motion semantics | 방향 오차가 크기 오차보다 해로움 | `outputs/motion_semantics_ablation/` |
| 33 | **시간축 action-chunk VLA(Impromptu VLA 3B)에서 기전 재현** | correction 98%, 자기 오차 재삽입(Impromptu식 reverse)으로 복원, 약 3 step window, 방향 > 크기 → 사전 등록 판정 Strong이었으나 기술적·미검증으로 낮춤(수치 바닥 5–6%, 경로 parity 미검증; 궤적 수준 안정 분기는 미재현) | `outputs/cross_vla_temporal_replication/` (2026-10-18 수정) |
| 34 | **다른 backbone·domain(SpatialVLA-4B, Gemma2, SimplerEnv 로봇 조작)에서 기전 재현** | token: 교정 step-4 편차 −82%, reverse 완전 복원, 1-step 86% (RESULTS 표기 87%); closed loop: 문맥 교정·교란의 성공률 효과는 검출되지 않음(초기 상태 군집 후 불확실, 동등성 미성립) → Partial replication | `outputs/cross_domain_temporal_replication/` |
| 35 | **SpatialVLA 실행 구조 ablation (재계획 간격 × ensemble)** | 4 step 재계획에서 문맥 교란의 실제 TCP 궤적 편차 증가(+1.8 cm)·재정렬 지연; task 효과는 초기 상태 군집 후 모두 불확실(원 보고 reverse −6.2%p는 군집 후 p = 0.14, P1-D 미재현 → 삭제), 교정 효과 없음, ensemble 보호 효과 없음 → Partial support | `outputs/spatialvla_feedback_protection_ablation/` |
| P6 | 연속 closed-loop (CARLA) | 현재 환경에서 수행 불가 (checkpoint·렌더링·디스크) | `outputs/continuous_closed_loop/STATUS.md` |

(경로의 `outputs/`는 `autovla_misalignment_poc/outputs/`입니다.)

---

## 3. 실패 기전 (실험 1–11, 32)

### 3.1 어디서 깨지는가 (실험 1–4)

| A− 발생 단계 | ORION | AutoVLA |
|---|---:|---:|
| perception (A− ∧ P−) | 6.0% | 25.8% |
| reasoning (A− ∧ P+R−) | 78.2% | 50.0% |
| interface (A− ∧ P+R+) | 15.9% | 24.2% |

- 두 모델 모두 말한 결정과 실제 행동의 결합이 느슨합니다(행동은 맞고 선언이 틀린 P+R−A+가 최대 그룹, 38–42%).
- CoT 개입(159 장면): 선언 방향을 뒤집어도 실행 반전 +6.4%p [+2.7, +9.7]; 방향과 무관한 문장 재작성은 A+를 +17.0%p 바꿉니다.
- 무편향 500 장면: CoT 강제 시 A+ 98.8 → 93.8% (−5.0%p, p = 5e-6), ADE +0.50 m.
- perception 정보는 실패군에서도 유지(AUROC 0.81–0.90), action은 마지막 layer(L32–35, MLP 지배)에서 형성됩니다.

### 3.2 작은 첫 편차가 증폭되는 기전 (실험 5–10)

실험 5–11은 사전 등록이 없는 탐색 실험입니다. 아래 p는 2026-10-13 drive 군집 재분석 값입니다(13 drive, exact sign-flip, 최소 p = 2.4e-4) [사후].

| 실험 | 결과 |
|---|---|
| 5. 첫 mismatch 1-token 교정 (출력 수준의 유일한 단발 교정) | A− 장면 A+ 17.3 → 73.1% (+55.8%p), FDE 8.18 → 2.09 m — per-unit raw 없음, cluster 재검정 불가 |
| 6. 같은 크기 대안 token (perturbation 중앙값 0.129 m, 대안 간 거리 매칭 오차 중앙값 8.8 mm / 7.9%) | A− 장면 증폭 45.0% vs A+ 장면 4.8%; A− 장면의 71%에서 증폭·회복이 공존 → 장면 맥락이 결정 |
| 7. 조건화 교정 (문맥만, 실행 출력은 모델 자신) | Normal 47.4% → Recent-GT 7.1%, GT-history 4.1%. Recent-GT는 **매 생성 step 직전 위치를 GT로 바꾸는 반복 개입**(단위당 6.8회, 실효 4.3회). Recent − Normal **−39.8%p [−49.0, −29.9]**(장면 가중), p = 2.4e-4, Holm 7.3e-4. attention mask 두 종: −1.4 / −3.3%p, 효과 검출 안 됨(동등성 미검정) |
| 8. layer patching (embedding + 36 decoder layer) | 어느 site에서 patch해도 3.8–7.4%(원 summary; 원 raw 없음). Reverse(GT-history 문맥에 **별도 Normal 행**의 위치 k−1 embedding 삽입): R8 복구 run에서 Reverse − Full **+28.6%p [+20.9, +36.1]**, p = 2.4e-4, Holm 7.3e-4 |
| 9. 교정 길이 (persistent window, 고정 위치) | 1/2/3/4 step = **47 [30, 62] / 77 [63, 88] / 90 [81, 99] / 96 [93, 100]%** of full-history 효과(R9, 분모 = 같은 run의 Normal − GT-history; drive CI). 3 step의 CI가 81–99%이고 f ≥ 0.9일 확률은 약 50%; 전체 CI가 90% 위인 것은 4 step부터. 마지막 GT 위치 이후 생성 step(GT 유지; 이전 표기 '해제 후 free horizon')도 5.8 → 2.9 pose로 함께 줄어듭니다 (2026-10-18 수정) |
| 10. horizon 통제 | 같은 free horizon에서도 1-step 62% vs 4-step 95% 회복; w개 GT 위치가 문맥에 남아 있는 동안 25–29% 재발산(이전 표기 '해제 후') — per-unit raw 없음 (2026-10-18 수정) |
| P1-B. 시작 위치 [사전 등록] | 같은 길이 w의 교정을 s = 1(첫 이탈 직후) vs s = 3에 넣은 early − late: w1 +10.7 [−0.7, +20.6], w2 +4.3 [−2.6, +9.7], w3 +0.5 [−4.5, +3.1]%p, 모두 비유의(12 drive). ±10%p 동등성은 w3만 Holm 후 성립(w2는 보정 전만). 마지막 GT 위치 이후 step 수를 맞추면 연속 지표의 부호가 처리 방식에 따라 바뀝니다 [사후 C2] (2026-10-18 수정) |

→ **직전 action 문맥이 다음 action 생성을 인과적으로 조건화**하고, 그 정보는 embedding에서 주입돼 residual로 운반됩니다(실험 8; layer별 동치는 summary 수준).
교정 길이 효과는 크지만(1 → 3 step: 47 → 90%), **첫 이탈 직후 step이 특별하다는 근거는 없습니다**(P1-B). w개 GT 위치를 문맥에 유지한 채 새 GT 추가를 멈추면 일부 재발산합니다(실험 10, summary 수준). GT를 문맥에서 제거하는 해제 실험은 없습니다(review_followup4 A2). (2026-10-18 수정)

### 3.3 그 identity의 무엇이 중요한가 (실험 11, 32)

| 직전 token := (A− 365 단위) | 증폭 | Recent-GT 효과 대비 |
|---|---:|---:|
| 자기 token | 47.1–47.9% | – |
| GT | 7.7–7.9% | 100% |
| GT와 motion이 가장 비슷한 다른 token (실험 11) | 8.5% | 99% (GT 대비 +0.8%p [−1.6, +6.6]; ±5%p 동등성 미성립, TOST p = 0.09) |
| 자기와 motion이 비슷한 다른 token (실험 11) | 43.8% | 8% |
| **방향 정답 + 크기 오답** (실험 32) | 17.3% | 76% |
| **크기 정답 + 방향 오답** (실험 32, GT와의 거리는 더 작음) | 26.3% | 54% |
| **같은 거리 오차를 GT 반대편에** (실험 32) | 13.7% | 85% |
| 크기만 맞춘 무작위 (실험 32) | 19.2% | 71% |

- embedding 근접은 motion을 고정하면 효과가 검출되지 않고(실험 11), GT 근접 motion token은 GT token과 차이가 작고 비유의합니다. 그러나 **동등성은 성립하지 않으므로**(drive 균등 가중에서는 90% CI [+0.2, +10.4]로 GT 근접 token이 약간 나쁠 수 있음) "identity와 무관/GT와 같다"로 쓰지 않습니다 [사후 drive 재분석].
- **방향 오답 − 크기 오답** = 증폭 +9.9%p [+3.9, +21.0](장면 가중, 13 drive), p = 0.0056, Holm 0.0056, FDE +1.22 m (A−); 가중 방식에 민감합니다(unit +9.0, drive 균등 +15.5%p). 정상 장면에서도 방향 오답은 증폭 4.3 → 8.6%, 크기 오답은 3.7%.
- **새 drive(P1-C, 18 drive)에서는 +6.2%p [+0.6, +12.1], sign-flip p = 0.073으로 사전 등록 판정 "결론 안 남"**입니다(bootstrap CI만 보고 "효과 있음"으로 쓰지 않음). 따라서 방향 > 크기는 **개발 drive에서의 결과**로 한정합니다.
- 같은 크기의 오차라도 모델 자신의 이탈과 반대 방향이면 증폭이 47.9 → 13.7%로 줄어듭니다(달성 거리는 약 11% 작음) → 자기 이탈 방향을 확인해 주는 직전 action이 feedback을 강화한다는 해석과 일치합니다(개발 drive).

---

## 4. GT 없는 완화: 조건화는 실패, 선택은 통함 (실험 12–24)

- **조건화 교정**(이전 frame 계획, CTRA, 과거 계획 합의, PDM-Closed 궤적을 context에 넣음): 모델이 참조 motion을 66–89% 따라가
  효과가 참조 품질로 결정되고, 정상 장면이 손해를 봅니다. 교정 시점 없이 상시 적용하면 dev 자연 평가 집합(2,747 장면: A− 52 / A+ 2,695)으로 재가중한 A− 빈도 1.24 → 2.60%로 악화(배포 빈도 아님)(실험 15). (2026-10-18 수정)
- **선택**(모델 자신의 후보 17개 = T 0.01 자연 + T 1.0 샘플 16개, context 불변):

| 데이터 | 규칙 | 결과 |
|---|---|---|
| 새 log 56개 (실험 20, 4,563 장면; PoC 기전 집합과 drive 13개 공유) | rank-sum (엔트로피 순위 + log-lik 순위) | A− 2.15 → 1.45% (−0.70%p [−1.07, −0.33], p = 4e-4), FDE 0.76 → 0.48 m |
| 사전 등록 held-out 52 log (실험 21, 4,814 장면; **log 분리, dev와 drive 22개 공유**, drive-disjoint 부분은 7 log / 6 drive뿐이고 재계산하지 않음) | rank-sum | PDMS 0.8893 → 0.8988 (+0.0095 [+0.0046, +0.0148]), 충돌 0.56 → 0.23% |
| 같은 held-out (실험 24) | rank-sum + 현재 frame 안전 필터(F1) | PDMS 0.9209 (+0.0315 [+0.0234, +0.0410]), 충돌 0.10% |

- 실험 22: 선택된 후보와 버려진 후보는 **이탈 전 엔트로피 차이가 작고(−0.016 [−0.023, −0.010], CI는 0을 제외; 장면 내 AUROC 0.51; 동등성은 검정하지 않음) 이탈 후에 크게 갈라집니다(−0.40)**.
  rank-sum은 장면 안에서 증폭 rollout을 AUROC 0.85–0.87로 구별합니다.
- 실험 23: 후보 8개에서 이득의 85%, 12–16에서 포화; N 16 추론 비용 1.2–1.3배(batch 디코딩).
- 5090 재실행: 3080 Ti로 돌린 모든 run을 5090에서 다시 해도 held-out rank-sum +0.0099, F1 +0.0316으로 같습니다.

---

## 5. NAVSIM v2 navhard 2단계 pseudo closed-loop (실험 25–27, 29–30)

전체 navhard_two_stage: 76 log, 225 그룹(원본 450 + 합성 5,462 장면). 실험 26의 두 번째 절반은 디코딩 전에 사전 등록(96a595c).

| 조건 | EPDMS | Δ vs 선택 없음 [95% CI] | 그룹 개선/악화 |
|---|---:|---|---|
| 선택 없음 | 0.229 | – | – |
| rank-sum | 0.248 | +0.020 [+0.003, +0.037], perm p 0.094 | 92 / 73 |
| max log-lik | 0.249 | +0.020 [+0.008, +0.033] (두 번째 절반에서 미재현) | 68 / 56 |
| 충돌 제약만 | 0.241 | +0.012 [+0.005, +0.020] | 33 / 6 |
| 주행 가능 영역 제약만 | 0.315 | +0.087 [+0.066, +0.106] | 117 / 6 |
| **안전 필터만** (충돌 + DA, 자연 계획이 통과하면 유지) | **0.333** | **+0.105 [+0.085, +0.125]** | **128 / 9** |
| 필터 + 무작위 | 0.328 | +0.099 | 154 / 53 |
| 필터 + rank-sum (F1) | 0.347 | +0.118 [+0.096, +0.140] | 147 / 51 |
| 필터 + max log-lik | 0.350 | +0.122 [+0.101, +0.142] | 141 / 33 |
| oracle (17개 중 최고, 분석용 상한) | 0.405 | +0.177 [+0.152, +0.201] | 180 / 18 |

- **재현성(실험 26)**: (두 절반은 log만 분리되고 drive 16개를 공유하므로 "새 log" 재현입니다.) F1은 사전 등록한 두 번째 절반에서 +0.112 [+0.082, +0.140], p 5e-5로 재현. max log-lik은 +0.010 [−0.004, +0.024]로 미재현.
- **분해(실험 27)**: F1 개선 +0.118 중 필터 자체 +0.105(89%), 선택기 추가분 rank-sum +0.014 [−0.002, +0.030](n.s.), max log-lik +0.017 [+0.003, +0.031].
  통과 후보 중 무작위 선택은 필터만보다 나을 게 없고(−0.005) 승차감만 나빠집니다.
- **구성요소(실험 29)**: 두 제약의 Shapley 몫은 주행 가능 영역 86%, 충돌 14%. 점수 이득의 99–104%가 곱셈 항(안전 위반 감소)이고 진행도 항은 −0.0001 ~ −0.0055 →
  진행도를 희생해 얻은 점수가 아닙니다. 다만 필터가 바꾼 장면에서는 4초 계획 도달 거리가 2.5–2.9 m 짧아 약간 보수적입니다.
- **oracle(실험 30)**: 최선 방법은 oracle 이득의 69%를 얻었고 남은 선택 여지는 +0.055. oracle 자체가 0.405이고 token의 31%는 17개 후보가 모두 0점 →
  **주된 병목은 후보 생성**입니다(oracle은 미래를 보는 분석용 상한, 배포 방법 아님).
- 한계: 안전 필터와 채점이 같은 PDM 규칙을 써서 이 지표에서 필터 이득이 유리하게 측정됩니다. 2단계 pseudo closed-loop이며 연속 시뮬레이션이 아닙니다.

---

## 6. 다른 VLA에서의 재현: OpenVLA-7B / LIBERO (실험 28, P1)

모델 `openvla-7b-finetuned-libero-spatial`(RTX 5090), LIBERO-Spatial 10 과제 × 100 에피소드. 공식 평가를 재구현해 자연 성공률 87%(공식 보고 84.7%).
프로토콜 6b9e8ad(실행 전), 자연 재실행 대조군 추가 6d0375a(Phase B 일부를 본 뒤·최종 분석 전). 상세: `outputs/cross_vla_replication/RESULTS.md`.

**구조 차이**: OpenVLA는 step마다 7개 action token(x, y, z, roll, pitch, yaw, gripper)을 차원 순서로 생성하고, 이전 step의 action을 입력에 넣지 않습니다.
AutoVLA의 "직전 action token → 다음 token" feedback에 대응하는 것은 step 안의 차원 간 조건화뿐입니다.

| 질문 | 결과 |
|---|---|
| step 안 token feedback (Phase C, 2,462 프레임) | x token 8 bin 교란 → 하위 성분(y..yaw; gripper 제외) 중 적어도 하나의 bin 변화 절댓값이 실제 주입 bin(clip 후) 이상인 단위 비율(4,741 단위 = 2,462 프레임 × 부호; bin은 m·rad 혼합) **92.6% [89.8, 94.4]**(task 군집 CI; 초기 상태 군집 CI [90.8, 94.0], review_followup3 P0-E), 24 bin 81.2%; y 교란도 75–87% (2026-10-18 수정) |
| 무엇이 하위 token을 결정하나 | **문맥에 들어간 이전 token**: reverse(정상 실행 + 잘못된 문맥) = feedback과 하위 token 동일, corrected(교란 실행 + 정상 문맥) = 변화 0 |
| 교정 window (1–4개 차원) | 노출 차원 수에 비례해 감소(93 → 85 → 67 → 53 → 27%); 교정 후 교란을 다시 넣으면 남은 차원은 거의 그대로 발산 (AutoVLA의 지속 효과는 없음) |
| 1. 작은 첫 편차가 항상 실패로? | 아니오. 자연 성공 중 교란 후 실패 9–18% = 교란 없는 재실행의 실패 15%와 같은 수준 |
| 2. 같은 크기에서 안정/불안정 분기? | 분기는 있으나(76 회복 / 11 실패) closed-loop 수치 변동과 구별되지 않고, step 내 하위 편차 크기로 예측되지 않음 |
| 3. previous-action correction | 과제 수준 효과 검출 안 됨 (feedback − corrected +3 ~ +5%p, n.s.) |
| 4. reverse intervention | step 안에서는 feedback과 같은 하위 발산; 과제 성공률 차이 −3 ~ −8%p vs 자연, −2 ~ +3%p vs 재실행 (n.s.) |
| 대조군 | 같은 정책을 batch 구성만 바꿔 다시 돌려도 성공 81% vs 87%, 결과 불일치 20% |

→ **"직전 token identity가 다음 token을 조건화하는" 국소 autoregressive feedback은 모델 간에 일반적**이지만, 그것이 **궤적·과제 실패로 증폭되려면
AutoVLA처럼 여러 시점의 action을 한 번에 autoregressive하게 생성하는 구조(시간축 action history)가 필요**하다는 해석과 일치합니다.
한계: LIBERO-Spatial 한 suite·100 에피소드, 교란은 한 차원·10 step. π0-FAST처럼 action chunk를 autoregressive하게 생성하는 모델은 다루지 못했습니다(시간축 chunk 모델은 실험 33에서 Impromptu VLA로 검증).

---

## 6b. 시간축 action-chunk VLA에서의 재현: Impromptu VLA 3B (실험 33)

사전 등록 038a6d3(본 실행 전). RTX 5090, navtest PoC 2,748 장면(28 log). 상세: `outputs/cross_vla_temporal_replication/RESULTS.md`.

**모델 선정(코드 수준 확인)**: Impromptu VLA 3B(`aaaaaap/ImpromptuVLAModel/3B_AD`, Qwen2.5-VL-3B)는 미래 waypoint 10개(0.5 s 간격, 5 s)를 시간순 텍스트
`[x, y]`로 autoregressive하게 생성해 4개 조건(다중 token AR 생성, 앞 action → 뒤 action conditioning, chunk 안 시간 순서, 공개 checkpoint·code)을 모두 만족합니다.
π0-FAST는 DCT 주파수 계수 token이라 시간 순서가 없어 제외, SpatialVLA는 LIBERO checkpoint 비공개·SimplerEnv(Vulkan) 불가로 제외했습니다(이후 실험 34에서 fractal checkpoint와 Mesa 소프트웨어 Vulkan으로 SimplerEnv 실행이 가능해져 검증).
AutoVLA는 0.5 s마다 상대 motion codebook token 1개, Impromptu는 절대 위치를 digit token 약 12개로 쓰는 차이가 있고, Impromptu의 자연 성능은
A− 33.4%, FDE5 5.75 m로 AutoVLA보다 훨씬 낮습니다. 자연 재실행(batch 구성만 다름)의 결과 불일치는 5.9%입니다.

| 실험 | Impromptu VLA 3B | AutoVLA |
|---|---|---|
| A. 작은 초기 편차(0.2 m) | 기준 A+ 장면의 49.5%가 A−로 전환(재실행 기저 4.6%); 궤적 수준에서는 97%가 3배 이상 커짐(흡수 0.5%) | 상당수 흡수 (A+ 장면 증폭 4.8%) |
| B. 같은 크기 분기 | 사전 등록 기준(궤적 수준 안정 r ≤ 1 공존) 2.2–3.2% < 재실행 5.9% → **미충족**; 탐색적으로 task 수준 통과/증폭 공존 87–93%, 횡·대각 교란이 전후보다 약 2배 증폭 | 같은 크기 편차가 장면에 따라 갈림 |
| C. 직전 action 교정 (문맥만, 매 waypoint 생성마다 직전 문맥 위치를 GT로 = 반복 개입) | recent_gt 증폭 63.4 → **1.3%**(−62.2%p [−65.4, −59.1], 20 drive, exact sign-flip p = 1.9e-6 = 최소값, Holm 7.6e-6); gt_history 0% | 47 → 7% (반복 교정, GT-history 효과의 93%; −39.8%p, 13 drive p = 2.4e-4) |
| C. 교정 window 1/2/3/4 step | 효과의 24 / 39 / **91** / 95% | 47 / 77 / 90 [81, 99] / 96% |
| D. reverse — **모델마다 다른 처치** | 교정 문맥에 **자기 branch의 오차 waypoint 재삽입**: 증폭 0 → **81.9%** (+81.9%p [+80.5, +83.9], p = 1.9e-6), 발산 +40.5 m | **별도 Normal 행의 embedding 삽입**(reverse@emb): 4.4 → 33.4% (R8: +28.6%p, p = 2.4e-4) |
| E. motion semantics | GT와 5 cm 다른 값 vs GT: 차이 비유의(p = 0.87; 동등성 검정 아님); 방향 오답 − 크기 오답 증폭 +15.7%p [+9.1, +24.7](20 drive, p = 8.2e-5), 오차 크기를 맞춘 첫 대체 시점에서도 +0.29 m(탐색적, 단위 수준 p라 인용하지 않음) | motion > embedding, 방향 > 크기 (+9.9%p, 개발 drive; 새 drive에서 미결론) |

사전 등록 기준: 1 correction **충족**, 2 reverse **충족**, 3 궤적 수준 분기 **미충족**, 4 짧은 window **충족**, 5 방향 > 크기 **충족**,
task 수준(교란이 기저 변동보다 큰 실패를 만들고 교정이 실패를 줄임) **충족** → **Strong replication**(사전 등록 판정, 기록 보존). review_followup4 A3 이후: 기술적·미검증 — 같은 경로·batch는 bit 단위로 재현되지만 batch 크기·padding·prefill 분할·decoder 구현만 바꿔도 장면 수준 A−/증폭 판정이 5–6% 바뀌고, 경로별 no-op gate가 없습니다. 큰 대비(−63.4%p, +81.9%p)만 바닥과 함께 기술합니다. (2026-10-18 수정)
제한: 여기서 task 실패는 AutoVLA와 같은 open-loop 궤적 실패이고, Impromptu는 절대 위치를 쓰므로 GT 문맥 교정에 GT 정보 누출이 섞여 있습니다
(정보를 더하지 않는 reverse와 motion 대체가 이를 배제하는 근거). (삭제: Impromptu window도 GT가 이후 문맥에 유지되므로 '해제 후' 비교가 아닙니다, review_followup4 A2/A3.) (2026-10-18 수정)

→ 문맥 교정이 증폭을 줄이고 오차 문맥이 증폭을 되살리는 패턴은 시간 순서가 있는 action chunk를 autoregressive하게 생성하는 두 driving 모델에서 반복됩니다.
다만 개입은 **취지가 맞춰진 것이지 동일하지 않고**(Reverse 정의, 좌표 표현, GT 누출 여부가 다름), 증폭 정의도 모델마다 달라 **효과 크기를 열끼리 비교하지 않습니다**.
Impromptu에는 GT 없는 대조(P1-A에 해당)가 없으므로 GT-free 결론은 AutoVLA에 한정합니다. 시간축 action history가 없는 OpenVLA(실험 28)에서는 궤적·과제 실패로 증폭되지 않았고, 작은 편차의 흡수 정도는 action 표현과 모델 정확도에 따라 크게 다릅니다.

## 6c. 다른 backbone·domain에서의 재현: SpatialVLA-4B / SimplerEnv (실험 34)

사전 등록 f578eae(본 실행 전), 결과 f4cf886. RTX 5090. 상세: `outputs/cross_domain_temporal_replication/RESULTS.md`.

**모델 선정(코드 수준 확인)**: 비-Qwen·로봇 조작·시간축 AR·공개 checkpoint·개입 가능 조건을 모두 만족하는 것은 SpatialVLA-4B
(`IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal`, PaliGemma2/Gemma2 + Ego3D)뿐이었습니다. 4-step chunk를 step당 3 token(구면 bin translation 4,096 /
rotation / gripper)으로 causal greedy 생성하며, step k가 앞 step token을 attend합니다. 제외: AR-VLA(decoder가 자기 action이 아니라 관측 proprio state를 conditioning),
VQ-VLA(chunk 5 step을 residual VQ 4단계로 압축 — 시간 순서 없음), WorldVLA(attention mask가 이전 action 블록을 차단), π0-FAST(DCT 계수).
SimplerEnv Google Robot `pick_coke_can`·`move_near`, seed 0–39(80 에피소드), 공식 실행 방식(매 step 재계획, 최근 4 chunk ensemble, sticky gripper).
자연 성공률 coke 92.5%·move near 77.5%(공식 0.86/0.78). 자연 재실행(수치 경로만 다름) 성공 불일치 **13.8%**, 궤적 발산 0.14 m.

| 실험 | SpatialVLA-4B (token 수준, 18,975 단위) | SpatialVLA-4B (closed loop, d = 0.3) |
|---|---|---|
| B. 작은 교란 | step 2/3/4 translation token 변경 56/36/28%; 증폭(A ≥ 1) 22.1%, 완전 흡수 36.7%; 같은 상태에서 흡수·증폭 방향 공존 15–25% | opposite 교란 실행 −20.0%p (원 episode 수준 p = 0.002; 초기 상태 군집 후 C − N −21.2%p, Holm p = 0.041 primary / 0.094 physical), perp_left 영향 검출 안 됨 |
| C. 직전 action 교정 (문맥만) | 증폭 22.1 → 13.5%, step-4 편차 −82% (episode cluster p ≤ 5e-6) | corrected − feedback(δC): +5.0 / −1.2%p, 불확실(동등성 미성립); 실행 명령 누적 차 0.42 → 0.42 m(실제 TCP 아님) |
| D. reverse (SpatialVLA closed loop: 정상 action 실행 + 문맥만 교란) | step-4 편차 0.010 → 0.057 (normal 0.056 수준; token 수준 reverse는 Normal 행 token 삽입) | 문맥에만 교란(δR): 성공률 +5.0 / −1.2%p, 불확실; **실행 명령 누적** 차 0.21–0.23 m (재실행 0.14 m) — 실험 34는 실제 TCP를 기록하지 않음 |
| E. window | 1 step 전 교정 87%, 2 step 전 63% (chunk 4 step이라 그 이상 측정 불가) | – |
| F. motion semantics | 방향 오답 − 크기 오답 +1.6%p (방향 오답이 기준에 더 가까운데도); 한 bin 옆 token ≠ 기준(+4.3%p) → identity 민감 | – |

사전 등록 기준: token 수준 (1) 교정 **충족**, (2) reverse **충족**, (3) 자연 변동보다 큼 **충족**, window **충족**;
closed loop task 수준 교정·reverse **미충족** → **Partial replication**. opposite 교란의 실패는 문맥을 교정해도 그대로(corrected −21.2%p)여서
실행된 action 자체가 원인입니다. 실행 중 범위 밖 translation token 1건은 tokenizer decode처럼 clip해 처리했습니다(RESULTS.md에 기록).

**2026-10-13 초기 상태 군집 재분석 [사후]** (`review_followup2/DIRECT_EFFECTS.md`, `DATA_AUDIT.md` §3):
- 실험 34의 독립 초기 상태는 coke 40, move_near **primary 29 / 물리 배치 10**입니다(seed 40개가 아님).
- 4개 pooled closed-loop 직접 효과(δR, δC × perp_left, opposite) 모두 ±10%p 동등성이 성립하지 않습니다(P0-B 판정 재현, Holm p_equiv 0.33–0.38). 판정: **불확실**("효과 없음" 아님).
- opposite 방향에서는 coke와 move_near의 효과 부호가 반대(δR −10.0 vs +7.5%p, δC +7.5 vs −10.0%p)지만 과제 간 차이는 검정하지 않았고 과제별 CI가 넓어, 이질성이나 상쇄가 확인된 것은 아닙니다(모두 비유의). (2026-10-18 수정)
- 실험 34/35는 reset이 이력에 의존하고(복원 hash 없음), 조건 쌍이 bit 단위로 복원 검증되지 않았습니다.

| | AutoVLA | Impromptu VLA | SpatialVLA |
|---|---|---|---|
| Backbone | Qwen2.5-VL 3B | Qwen2.5-VL 3B | PaliGemma2 3B (Gemma2) |
| Domain | 자율주행 | 자율주행 | 로봇 조작 |
| Action representation | codebook | absolute waypoint text | 구면 bin token, 4-step chunk |
| Temporal AR | ✓ | ✓ | ✓ |
| Correction (문맥 교정) | ✓ (매 step 직전 위치 := GT, 반복) | ✓ (매 waypoint 직전 위치 := GT, 반복; GT 누출 포함) | token ✓ (reference = 자기 greedy chunk) / task: 불확실 |
| Reverse — **처치 정의가 다름** | 별도 Normal 행 embedding 삽입 ✓ | 자기 branch 오차 재삽입 ✓ | token: Normal 행 token 삽입 ✓ / closed loop: 정상 실행 + 문맥만 교란, task 불확실 |
| 교정 길이 (share of full effect) | 1/2/3/4 step 47/77/90/96%; 3 step ≈ 90% [81, 99] | 1/2/3/4 step 24/39/91/95% | 1 step 86%, 2nd-last 63% (chunk 한계) |
| 초기 위치 특수성 | 근거 없음 (P1-B) | 미검정 | 미검정 |
| Direction > magnitude | 개발 drive ✓ (+9.9%p) / 새 drive 미결론 (+6.2%p, p = 0.073) | ✓ (+15.7%p) | 약한 ✓ (+1.6%p, token) |
| Stable / unstable absorption | ✓ | ✗ | ✓ |

증폭 정의도 모델마다 다르므로(AutoVLA·Impromptu: A− ∧ FDE5 > 3 m; SpatialVLA token: 누적 step 2–4 편차 ≥ 주입 편차) **효과 크기를 열끼리 비교하지 않습니다**. 개입은 취지가 맞춰졌을 뿐 동일하지 않습니다.

→ 직전 action 문맥이 이후 생성을 바꾸는 token 수준 효과는 SpatialVLA(조작)에서도 관찰됐습니다. 모델별 개입이 달라 일반 VLA에 대한 주장으로 쓰지 않습니다. (2026-10-18 수정)
그러나 **task 실패 기전이 되는지는 구조 의존적**입니다. feedback이 짧은 chunk(4 step) 안에 갇히고 매 step 새 관측으로 재계획되는 SpatialVLA에서는
생성 action과 실제 궤적은 바뀌지만 성공률 효과는 두 과제 평균 ±10%p 이내로만 제한되고(P1-D, 검증한 설정), 과제별 효과는 불확실합니다(OpenVLA 실험 28과 비슷한 양상).


## 6d. 실행 구조 ablation: SpatialVLA 재계획 간격 × ensemble (실험 35)

- 사전 등록 8fc20bf, 분석 스크립트 44266eb(결과 보기 전), 결과 2b0212d. RTX 5090.
- 상세: `outputs/spatialvla_feedback_protection_ablation/RESULTS.md`, `paper_quantitative_package/EXP35_RESULTS.md`.
- 설계: 실험 34와 같은 과제·교란(opposite, d = 0.3, step 8–23)을 6개 실행 설정 × 4조건으로 돌렸습니다. 각 80 에피소드(seed 0–39)입니다.
- seed 추가: 사전 등록 규칙이 발동해 r1e4와 r4e1을 seed 40–119로 확장했고, 두 설정은 합동 240 에피소드로 판정했습니다.
- 실행 가중치 정정: ActionEnsembler의 가중치는 최신 예측이 가장 큽니다(57/26/12/5%). 실험 34 문서의 반대 설명은 오류이며, 결과에는 영향이 없습니다.
- 초기 상태: seed 0–119는 coke 120개, move_near primary 53개 / **물리 배치 10개**입니다. reset이 이력에 의존해(reconfigure 없음) 물체 상태가 허용오차 이상 drift한 행이 있고, 조건 쌍의 복원은 검증되지 않았습니다(`review_followup2/DATA_AUDIT.md` §3).

| 설정 (재계획 / ensemble / 노출) | N | F | C | R | F − C | R − N |
|---|---|---|---|---|---|---|
| r1e4 (1 / 4 / 0.43, 공식, 240) | 82.9 | 67.1 | 60.0 | 82.1 | +7.1 [+0.4, +14.2], p = 0.06 | −0.8, p = 0.88 |
| r1e2 (1 / 2 / 0.31) | 85.0 | 70.0 | 68.75 | 86.25 | +1.25, Holm 1 | +1.25, Holm 1 |
| r1e1 (1 / 없음 / 0) | 85.0 | 67.5 | 68.75 | 80.0 | −1.25, Holm 1 | −5.0, Holm 1 |
| r2e2 (2 / 2 / 0.66) | 82.5 | 77.5 | 77.5 | 83.75 | 0.0, Holm 1 | +1.25, Holm 1 |
| r2e1 (2 / 없음 / 0.50) | 87.5 | 76.25 | 68.75 | 85.0 | +7.5, Holm 1 | −2.5, Holm 1 |
| r4e1 (4 / 없음 = 4 / 0.75, 240) | 81.7 | 69.6 | 65.0 | 75.4 | +4.6, p = 0.22 | −6.25 [−12.1, −1.2], p = 0.040 (episode 수준; **군집 후 비유의, 삭제**) |

(위 표의 p와 CI는 원 episode 수준 분석입니다. 1.25%p 배수의 정확한 반올림 경계값은 두 자리로 적었습니다 (2026-10-15 수정). move_near seed 0–119는 primary 53 / 물리 배치 10개의 초기 상태뿐이라 표본을 과대 계산합니다.)

**초기 상태 군집 재분석 [사후]** (r1e4·r4e1, seed 0–119; `review_followup2/DIRECT_EFFECTS.md`):

| 설정 | δR = R − N, pooled [95% CI] (primary / physical p) | δC = C − F, pooled [95% CI] | ±10%p 동등성 (Holm, primary / physical) |
|---|---|---|---|
| r1e4 | −0.8 [−9.6, 7.5] (0.93) | −7.1 [−15.9, 2.1] (0.15 / 0.35) | 0.082 / 0.20 — 불성립 (episode 수준에서 성립하던 동등성이 군집 후 사라짐) |
| r4e1 | −6.25 [−13.6, 1.1] (**0.136 / 0.256**, 보정 전) | −4.6 [−13.5, 4.8] | 0.37 / 0.47 — 불성립 |

- r4e1 − r1e4 상호작용(사전 등록 주 대비): δR −5.4 [−17.3, 7.0], δC +2.5 [−8.9, 14.1]; 차이도 동등성도 성립하지 않음(불확실).
- move_near r1e4 δC −15.0%p는 episode 수준 p = 0.002였지만 군집 후 0.058(primary) / 0.28(physical)입니다.
- 실행 교란 양성 대조: r1e4 C − N −22.9 [−34.9, −11.4], Holm p = 0.001 → 성공률 지표는 실행된 교란에는 반응합니다.

궤적 지속 (r4e1 − r1e4, 합동; 실제 TCP 위치 거리, episode 수준 Wilcoxon p):
- reverse 대 natural: 평균 위치 오차 +1.84 cm [+1.26, +2.46], p = 1e-10. 실행 명령 누적 차 +0.116 m(실제 TCP 아님), p = 7e-11. 재정렬 +10.9 step, p = 1e-6.
- feedback 대 corrected: +1.32 cm, p = 2e-4.
- r2e1: 재정렬 +13 step, p = 5e-4.
- r1e1(ensemble 없음): feedback 대 corrected +1.76 cm, 보정 전 p = 0.028, 5개 설정 Holm 후 0.14(P0-A F4) → 비유의.
- 모든 설정에서 교란 뒤 오차가 줄지 않았습니다(반감기 없음).
- 이 궤적 p들은 초기 상태 군집을 반영하지 않습니다. 같은 방향의 효과가 P1-D(복원 검증, 교란 예산 고정)에서 군집 단위로 확인됐습니다(§6e H1).

사전 등록 기준(원 판정, 그대로 보고):
- 1 (공식 설정에서 효과 없음): 충족
- 2 (보호를 줄이면 교정 효과 증가): **미충족** (방향 반대)
- 3 (reverse가 실패 증가): r4e1에서만 충족 (상호작용 n.s.) — **군집 재분석 후 r4e1 효과도 비유의, P1-D 미재현**
- 4 (오차 지속 증가): 재계획 축에서 충족
- → **Partial support**

→ 자기 생성 action이 실행에 오래 남을수록(재계획 간격) 문맥 교란이 만든 궤적 편차가 더 오래 남습니다. ensemble이 문맥 효과를 흡수한다는 근거는 없습니다(실행 교란량을 맞춘 P1-D H2). 성공률 효과는 실험 35 자료만으로는 군집 후 불확실하고, P1-D에서 두 과제 평균 ±10%p 이내로 제한됩니다. 따라서 driving 모델과의 차이를 실행 구조만으로 설명할 수는 없습니다.


## 6e. 후속 검증 실험 (VLA_experiment_plan_20261008 P0–P1, review_followup2 A–C)

종합: `autovla_misalignment_poc/outputs/review_followup/REVIEW_FOLLOWUP_SUMMARY.md`(P0–P1, 2026-10-12)와 `autovla_misalignment_poc/outputs/review_followup2/`(A–C 사후 재분석, 2026-10-13).
P0와 review_followup2 A–C는 **사후 재분석**이고, P1-A–D는 **사전 등록**이다(b212637; P1-C는 수정 b216920·9b3d4af 포함). R8R9는 원 수치를 아는 상태의 **복구 실행(재현성 감사)**이며 확인 실험이 아니다.

| 실험 | 결과 (2026-10-13 기준) | 판정 |
|---|---|---|
| **P0-A 감사** [사후] | Recent-GT는 매 step 직전 위치를 교정하는 **반복 개입**(단위당 6.8회, 실효 4.3회). Reverse는 모델마다 다른 처치. SpatialVLA 실험 34/35의 m 단위 "궤적 발산"은 **실행 명령의 누적값**(실험 35에서 실제 TCP의 약 4배). 실험 5/8/9/10은 per-unit raw 없음 | 서술 정정 |
| **P0-B → A3 군집 통계** [사후] | 독립 단위를 log(16)에서 **drive(13)**로 올려도 판정 변화 없음. Recent−Normal −39.8%p [−49.0, −29.9], Reverse−Full +28.6%p [+20.9, +36.1], 둘 다 p = 2.4e-4(13 drive 최소값, Holm 7.3e-4); Direction−Magnitude +9.9%p [+3.9, +21.0], p = 0.0056. leave-one-drive-out에서 부호 반전·CI 0 포함 없음. 실험 11 GT 근접 token: +0.8%p [−1.6, +6.6], ±5%p 동등성 **미성립**(TOST p = 0.090) | p 표기를 drive 수준으로 교체 |
| **P1-A GT 없는 reference** [사전 등록] + C1 [사후] | 문맥만 교란해도 suffix local motion +0.57 m/seg [+0.43, +0.68]; 기하 baseline 대비 endpoint +4.37 m [+3.34, +5.17] (2026-10-15 수정); reference 교정 −3.50 m (drive p = 2.4e-4, Holm 7.3e-4). 원 비교 "A−에서 등속 외삽(CM)과 구별되지 않음(−0.59 m)"은 CM 자체의 외삽 오차(4.69 m)가 섞인 잘못된 대비였다. 교란 전후 변화량으로 다시 비교하면 모델 − CM = A− **+2.35 m [+1.27, +3.19]**, A+ −0.52 m [−0.82, −0.26], 선정 208 장면 전체(A− 52 + A+ 156) +0.20 m [−0.32, +0.56] (p = 0.51; 동등성 미검정) | GT-free 문맥 효과 유지(AutoVLA 한정). 외삽 초과는 실패 선택 층에서만 → 일반 주장 삭제. reference 교정은 GT 실패를 줄이지 않음(A− 증폭 +19.7%p) (2026-10-18 수정) |
| **P1-B 시작 위치** [사전 등록] + C2 [사후] | early − late (같은 w): +10.7 / +4.3 / +0.5%p, 모두 비유의(12 drive). ±10%p 동등성: w3 성립(Holm 0.002), **w2는 보정 전만(Holm 0.064)**, w1 불성립. 늦은 window는 해제 후 horizon이 2 pose 짧고, 같은 해제 후 horizon에서는 연속 지표 부호가 처리 방식에 따라 바뀜 | early window 특수성 근거 없음. 위치 효과 크기는 불확실 |
| **R8R9 복구** | 실험 8·9 원 수치 220/220 재현(재현성 감사). 교정 길이 회복률 1/2/3/4 step = 47 / 77 / **90 [81, 99]** / 96% (분모: 같은 run의 Normal − GT-history; drive CI). 반복 Recent-GT = GT-history 효과의 92.9% [89.4, 99.1] | "3 step ≈ 90%"로 표기. "2–4 step 90%"는 틀림 |
| **P1-C 새 drive** [사전 등록] | 64 log / 24 drive, 기전 집합과 drive 0 공유(day-vehicle 10 공유). RN F(105 장면 / 18 drive) −34.2 [−38.3, −29.5] / P(E 2,191 적격 장면, 353 선정 장면 가중, 24 drive) −4.1; RV F +26.8 [+21.9, +30.7] / P +3.6 — replicated(drive exact sign-flip, Holm). DM F +6.2 [+0.6, +12.1], sign-flip p = 0.073 / P +2.6, p = 0.10 — **inconclusive**. 독립 재계산이 사전 등록 결과를 완전히 재현 | 핵심 기전의 **새 drive** 재현. 방향 > 크기는 미결론 (2026-10-18 수정) |
| **P1-C 실행 기록** | 수정 1: pilot 기준 85%를 arm-N 자기 재현성 기준으로 완화(평가 자료 전). 수정 2: H8 2,436 단위 전부 OOM → 같은 명령 재실행; H8 없는 첫 분석에서 RN·DM이 열람된 뒤 commit, 첫 분석은 `analysis_incomplete_noH8/`에 보존, RN·DM 판정 동일 | 결과 기반 제외 없음. 원고에 보고 |
| **P1-D SpatialVLA 교란량 통제** [사전 등록] + B [사후] | H1: 문맥 예산 고정, R − N 궤적 면적 DiD r4e1 − r1e1 **+214 cm·step [+171, +255]**(복합 설정 효과). H2: 실행 dose를 맞추면 ensemble 축 DiD +21 [−25, +68], ±100 동등(흡수 방향 아님). 성공률 상호작용 S1–S4: ±10%p 동등, **초기 상태 군집(primary·physical)과 Holm 후에도 유지**, ±5%p는 불성립. 직접 효과 δR·δC pooled r1e4/r4e1: ±10%p 동등(두 clustering, Holm), ±5%p 불성립, **과제별 CI 18–29%p로 불확실**. 실험 35 r4e1 reverse −6.25%p는 미재현(+1.25 [−5.6, +8.0]). move_near 120 unit은 물리적으로 실험 34/35와 같은 10개 배치 | 생성·물리 궤적 변화는 확인, 성공률 효과는 두 과제 평균 ±10%p 이내(검증한 설정)로 제한, 과제별 불확실 |

→ 유지되는 것: 문맥 교정이 증폭을 줄이고 오차 문맥이 증폭을 되살리는 AutoVLA 기전(drive 군집 통계, **새 drive** 재현), GT 정보 없는 문맥 효과(AutoVLA, 순수 좌표 누적 초과), 교정 길이 효과(3 step ≈ 90%).
축소·삭제: 외삽을 넘는 일반적 자기증폭(실패 층에서만), early window 특수성(근거 없음), 방향 > 크기의 일반화(새 drive 미결론), GT 근접 token의 동등성(미성립), 실험 35 −6.25%p와 ensemble 보호(삭제), SpatialVLA 성공률 효과는 "두 과제 평균 ±10%p 이내"로만.
새 GPU 실험은 하지 않았다(`review_followup2/GAP_DECISION.md`: 독립 drive를 더 확보할 수 없고, move_near 물리 배치는 10개로 고정).

---

## 6f. review_followup3 (2026-10-15): 수치 복원, 개입 machinery 검증, 운동 baseline

전부 기존 자료의 **사후** 재분석입니다. 새 GPU 실행은 검증용뿐입니다: AutoVLA no-op/cache parity 관문, SpatialVLA 복원 감사, A5 재검증, O0 관문 확장. 효과를 추정하는 새 실험(P3, P4, P5)은 **하지 않았습니다**. P3와 P4는 CP2 결정으로 미실행이고, P5는 disjoint log가 없어 미실행입니다.

- 결과 위치: `autovla_misalignment_poc/outputs/review_followup3/`. 최종 판정은 `STATUS.md`, `CLAIM_EVIDENCE_MATRIX.md`, `REVIEW_COVERAGE.md`에 있습니다.
- 수치 은행: `paper_quantitative_package/numbers.json`. 2,164행이며 스크립트로 병합했고 손으로 입력한 수치는 없습니다.
- commit: 0ef9334, e9907fc, 5b158f2, 47efb95, d70d19c, 444299f, fae513c.
- 태그: 주행 / AutoVLA / open-loop 5 s plan이 주 결과입니다. 조작 결과는 보완 증거입니다.

**P1 — 개입 machinery 검증 (정의는 `P1/INTERVENTION_SPEC.md`)**

AutoVLA no-op/cache 관문은 **6개 경로 모두 PASS**이고 실패는 0건입니다.
- 대조 c1–c5: hook no-op, self-patch, cache 무효화, full recompute, 자체 재현.
- 디코딩은 greedy와 T = 0.01이고, 허용 오차는 사전 고정값입니다(T_exact 0, T_equiv 0.75 logit).

| 단계 | 경로 | 범위 |
|---|---|---|
| Stage 0 | H7 Recent-GT/Full-GT, H8 Reverse-N, H9 window, P1A self-reference·좌표 누적·rollout 저장 | 21 path-unit |
| O0 확장 | H32 실험 32 DM, P1B 시작 위치 | 18 path-unit; 저장된 raw와 54/54·477/477 일치 |

- Reverse-N이 token 치환과 같다는 것을 C4가 확인했습니다(82/82 step).
- **범위:** 검증 manifest 밖의 원 run은 소급 검증하지 않았습니다.
- B = 1과 실험 batch 사이에서 free-run suffix가 0–22% 행에서 달라집니다. 이것은 batch 구성에 따른 bf16 수치 한계입니다. 그래서 모든 대비는 같은 harness, 같은 batch 안에서 쌍대로 계산합니다.

SpatialVLA 복원 감사도 **PASS**입니다(20 단위 = 2 과제 × 5 상태 × r1e4/r4e1).

| 대조 | 결과 |
|---|---|
| C1, C3/P1-C 복원, C5a, C5b | 각 20/20, bit 단위 동일 |
| F/C/R branch 복원 | 60/60 |
| C2 null arm (분리 fingerprint 재검증, AMENDMENT_P1_02) | 20/20 |

- C2는 policy-state hash, 실행 action, 시뮬레이터 상태를 비교했습니다.
- 처음의 엄격 기준(장부 기록 dict까지 포함한 hash)은 0/20이었고, A5 사후 검사는 20/20이었습니다. 둘 다 기록으로 남깁니다.
- B = 1 decode와 batch-8 chunk는 20개 중 17개가 같았고, HF generate와 batch-8은 16개가 같았습니다. 따라서 실험 34 문서의 **"generate와 동일"이라는 표현은 틀렸습니다.** arm들은 고정 batch 8 안에서만 서로 일관됩니다.
- 실험 34/35는 복원이 검증되지 않았습니다. Impromptu와 OpenVLA 경로는 parity를 검증하지 않았습니다.

**P0-A — 모집단·선택 흐름**

- **선택 흐름.** dev 자연 평가 2,748 기록에서 CoT stub 1개를 빼면 **2,747 장면**(20 drive)입니다.
  - 이 중 A− 52(13 drive / 10 day–vehicle), severe 40입니다.
  - 첫 mismatch가 있는 장면은 1,159개이고, mismatch 없는 A−는 0개입니다. 따라서 52는 **이 규칙에서 A− 전부**(π = 1)입니다.
  - A+ 156은 같은 t*에서 A− 하나당 3개씩 `Random(0)`으로 뽑았습니다. 재현하면 208/208이 동일합니다.
- **A− 빈도 = 평가 세트 안의 빈도이며, 배치 분포나 자연 실패율이 아닙니다.**
  - dev: 52/2,747 = 1.89% [1.16, 2.64] (drive 군집)
  - P1-C: 107/5,671 = 1.89% [1.32, 2.53]
  - P1-C 적격 frame: 105/2,191 = 4.79% [3.43, 6.21]
- **평가 집합(E) 추정량.** (이전 표기 '모집단 추정량'; P는 선정 353 장면의 설계 가중 추정) RN P −4.1%p [−6.4, −2.4], RV P +3.6%p [+2.2, +5.5]입니다(P1-C 사전 등록 판정을 정확히 재현). (2026-10-18 수정)
  - 대상은 P1-C 64 log의 적격 2,191 장면에 대한 scene 가중 Hajek 추정량입니다. 이름은 **evaluation-population estimand**입니다.
  - P = F + S 분해가 성립합니다. RN 효과의 60%는 A+(S) 층에서 옵니다.
  - 다른 코호트의 효과와 빈도를 곱해 배치 효과를 만들지 않습니다.

**P0-C — 수치 정합**

- **+4.37 m의 authoritative 행**은 사전 정의 P1A-2(OC − Geo, dev9_ref, A− 364 단위 / 52 장면 / 13 drive)입니다.
  - drive bootstrap [+3.34, +5.17], sign-flip-inversion [+3.06, +5.35], p = 2.4e-4, Holm(m = 3) 7.3e-4.
  - [+3.32, +5.19] / Holm 9.8e-4는 **사후** C1 excess 지표이며 별도 행입니다.
  - 근처에 있는 "p = 7.3e-4"는 endpoint 전용 p가 아닙니다. 사전 정의 P1A family의 Holm 값(교정 대비 P1A-3도 같은 값)입니다.
- **R9.** 모든 window 길이가 같은 365 단위 / 52 장면 / **13 drive** / 10 day–vehicle입니다(Figure 2 caption n). P1-B의 12 drive로 추정한 값이 아닙니다.
- **회복률.** scene 가중 46.7 / 76.6 / 90.1 [81.2, 98.9] / 95.8%와 실험 9 원 unit 가중 47.1 / 76.1 / 89.7 / 96.1%는 서로 다른 추정량입니다.
  - 분자와 분모를 같은 resample에서 계산했고, 분모가 0 이하이거나 5%p 미만인 경우는 0건입니다.
  - 마지막 GT 위치 이후 step 평균(GT 유지; 이전 표기 '해제 후 free step')이 5.8 / 4.8 / 3.8 / 2.9로 함께 줄어듭니다. (2026-10-18 수정)
- **P1-B**(12 drive, 326 단위)는 고정 전체 horizon(E_total)과 마지막 GT 위치 이후 고정 horizon(E_post; GT 유지)을 별도 추정량으로 보고합니다. (2026-10-18 수정)
- **문서 반올림·표기 수정**(2026-10-15, `cp2_doc_fixes.py`가 원 값과 대조):

| 수정 전 | 수정 후 | 이유 |
|---|---|---|
| 5.18 | 5.17 | 이중 반올림 |
| [+0.44, +0.68] | [+0.43, +0.68] | 이중 반올림 |
| +1.2 | +1.25 | half-even 경계값 |
| −6.2 | −6.25 | half-even 경계값 |
| 실험 35 표의 68.8 / 86.2 / 83.8 / 76.2 | 68.75 / 86.25 / 83.75 / 76.25 | half-even 경계값 |

  - +4.37 m 행의 Holm 표기 혼동(사전 정의 Holm을 사후 CI에 붙인 것)을 바로잡았습니다.
  - "18–29%p"는 P1-D r4e1에 한정된다고 표시했습니다(r1e4는 6.9–21.6%p).
  - OpenVLA 92.6%의 CI는 task 군집 CI라고 표시했습니다.

**P0-D — 상위 의존성·disjoint log**

- AutoVLA A− RN/RV는 day 축(8 그룹)에서 p = 0.0078, Holm 0.023입니다. DM은 day 축에서 p = 0.039입니다.
- P1-C는 day–vehicle 연결 성분이 6개뿐입니다. exact p 하한은 0.031, Holm 하한은 0.094이므로 Holm < .05에 도달할 수 없습니다. 이는 추론 해상도의 한계이며, 효과가 없다는 증거가 아닙니다.
- leave-one-group-out에서 부호가 바뀐 경우는 없습니다.
- **disjoint log 없음.** OpenScene test는 147 log, 9일뿐이고, dev와 P1-C가 이 9일을 모두 덮습니다. 그래서 dev, P1-C, 학습 데이터와 day·vehicle이 모두 겹치지 않는 log는 **0개**입니다.
  - 학습 데이터하고만 disjoint한 log는 25개 있지만, 모두 dev나 P1-C와 day·vehicle을 공유합니다.
  - P5는 NOT_RUN이고, 원고의 외부 검증 부재 문장을 유지합니다.

**P0-E — OpenVLA 재분석** (조작 / OpenVLA-7B / LIBERO-Spatial closed loop; machinery 미검증이므로 기술적 수준)

- 초기 상태 군집으로 다시 계산하면 C − F는 8-bin −3%p [−12, +6](p = 0.65), 24-bin −5%p [−14, +3](p = 0.36), Holm 0.72입니다. 기존 episode 구간은 나란히 둡니다.
- 자연 재실행끼리도 결과가 20% 다릅니다.
- step 안 92.6%는 단위 = (프레임, 부호), n = 4,741이고, clip된 단위와 gripper를 제외한 값입니다. 초기 상태 군집 CI는 [90.8, 94.0]입니다.
- 물리 단위로 보면 ±8 bin x 교란은 약 2.6 mm의 명령에 해당합니다. 하위 translation 명령이 주입량 이상인 비율은 72%입니다(명령값이며 실현된 운동이 아님).

**P0-B — SpatialVLA 과제별 표** (조작 / SpatialVLA-4B / SimplerEnv closed loop)

- 두 과제 평균은 P1-D 4개 대비 모두 ±10%p 동등입니다.
- 과제별로는 8개 중 3개만 Holm 후 ±10%p 동등입니다(primary): r1e4 R−N coke, r1e4 R−N move_near, r1e4 C−F move_near.
- ±5%p 동등은 하나도 성립하지 않습니다.
- task × intervention 이질성은 검출되지도, 배제되지도 않았습니다. P1-D r4e1 C−F에서는 과제별 점추정이 −4.2 / +4.2%p로 부호가 반대지만, 과제 간 차이 −8.3%p [−21.7, +5.0](p = 0.23)는 0을 포함하므로 이질성·상쇄는 확인되지 않았습니다. (2026-10-18 수정)

**P2 — 운동 외삽 대조** (주행 / AutoVLA / open-loop, P1-A rollout, GT-free family G)

- **CM 정의.** 기존 repeat-p constant-motion은 global 속도 반복이 아닙니다. 교란 token을 yaw까지 포함해 **body-frame SE(2)로 반복**하므로 이미 일정 회전율 외삽입니다. "CM이 곡률을 무시한다"는 리뷰 전제는 이 구현에는 맞지 않습니다. C1은 정확히 재현됩니다.
- 추가한 대조:
  - CV0: 회전 0, global 등속
  - CTRV-K3: 최근 3 step 평균 회전율·속도
  - CTRA-2: 일정 회전율과 가속도
  - 학습된 소형 non-AR 운동 외삽기: MLP, prefix 입력과 prefix + ego 입력, seed 3개. 평가 drive와 분리된 학습 drive로 학습했습니다. VLA의 non-AR 버전이 아닙니다.
- 결과 (Δ_model − Δ_대조, endpoint 변위, m):

| 층 | 결과 |
|---|---|
| A− | 모든 대조를 초과: CM +2.35 [+1.27, +3.19], CV0 +4.02, CTRV-K3 +3.02, CTRA-2 +1.43 [+0.44, +2.33], 외삽기 +1.97 [+1.06, +2.87] / +2.27 [+1.50, +3.10]. 범위 밖 단위를 빼도 부호 유지 |
| A+ | 모델 < CM (−0.52), < CTRA-2 (−1.56), < 외삽기 (−1.11 / −0.56) |
| 선정 208 장면 전체 (A− 52 + A+ 156) | CM +0.20 (n.s.; 동등성 미검정), 외삽기 −0.34 / +0.15 (n.s.) (2026-10-18 수정) |

→ **"운동 외삽을 넘는 반응"은 A−(실패 선택 층)에서만 말할 수 있습니다.** A+와 선정 208 장면 전체에서는 운동 외삽과의 차이가 검출되지 않았습니다(동등성 미검정). (2026-10-18 수정)

**판정 요약.** 리뷰 항목별 판정은 다음과 같습니다(근거 행은 `REVIEW_COVERAGE.md`).

| 판정 | 항목 |
|---|---|
| 해결 | M3, M7, m1, m2, m4(기술적) |
| 부분 해결 | M1, M4, M5, M6 |
| 미해결 — 주장 축소로 대응 | M2, M4의 새 상태 부분, M1/M5의 외부 검증, Impromptu/OpenVLA machinery |


## 6g. review_followup4 (2026-10-18 수정): 통계 구간 정정, window 의미, 수치 민감도, matched-prefix 진단

AUTOVLA_NEXT_STEPS_20261018 실행 결과입니다. 주 모델은 AutoVLA이고, SpatialVLA·OpenVLA는 서로 다른 개입·실행 수준의 보완 증거입니다.

- 결과 위치: `autovla_misalignment_poc/outputs/review_followup4/`. 최종 상태 `STATUS.md`, 주장 변경 `CLAIM_CHANGES.md`, 원고 문장 `MANUSCRIPT_REVISION.md`, 재현 명령 `RUN_COMMANDS.md`, 재현 ZIP `reproducibility_minimal.zip`과 재계산 결과 `REPRO_CHECK.md`, 파일별 hash `SOURCE_MANIFEST.csv`.
- commit: 5523ce3(A2/A3), 2a31ad7(A1/A4), 552b3ac(B1 결과, B2 프로토콜 사전 등록), 28f1bfd(FEASIBILITY), c7d1bba(B2 결과). 이번 마무리 문서는 commit하지 않았습니다.
- **역사 기록 보존.** review_followup3 CP2에서 P3/P4를 실행하지 않기로 한 결정은 그대로 둡니다. B2는 별도 프로토콜을 가진 새 round이며 P3를 소급 DONE으로 바꾸지 않습니다.
- 원고 `.tex`는 저장소에 없습니다. 문장별 수정안과 기계 생성 표(`paper_quantitative_package/tables.tex` Table 12–14)만 제공합니다.
- 계산: A1·A2·A4·A5·C는 CPU만. GPU(RTX 5090)는 A3 Impromptu parity 소규모 실행(약 3분), B1(18.3분), B2 gate·pilot·본 실행(dev 52.9분, P1-C 약 86분)뿐입니다. 새 효과 실험은 B2 하나입니다.

**A1 — 성공률 CI 퇴화 (사후 수정, 원 분석 보존)**

- 원인: `stats_core.signflip_ci`와 sign-flip TOST는 이동된 cluster 합이 0에 대칭이라고 가정합니다. 차이가 {−1, 0, 1}이고 0이 많은 쌍 이진 자료에서는 δ = 0에서만 성립하므로, 모든 차이가 0이면 구간이 동점 허용오차 ±2.4958e−13으로 붕괴합니다(bit 단위 재현). δ = 0의 sign-flip / exact McNemar p와 Holm family는 유효하고 그대로입니다.
- 영향: OpenVLA 과제별 150행 중 11행이 폭 0, 75행이 한쪽 끝 0 고정; SpatialVLA 819행 중 37행이 끝 0 고정.
- 방법 선택(실제 자료 재계산 전에 규칙 고정): 독립 쌍은 Newcombe 10(최소 coverage 0.951), 군집 반복은 Rao–Scott design-effect Newcombe(과제 평균·상호작용은 MOVER). 두 방법 모두 폭 0 구간을 내지 않습니다.
- 판정 변경 36행(미러·clustering 중복 포함):
  - OpenVLA C − N 8-bin "95% CI가 0을 제외"는 동점 artefact였습니다: [−16.7, −7e−14] → [−17.1, +1.0](p = 0.115와 일치). R − N_rep의 보정 전 ±10%p 동등은 사라졌습니다(p_eq 0.049 → 0.052).
  - OpenVLA 과제별 CI(10 상태)는 인용하지 않습니다. 새 구간은 ±0.18–0.28로 정보가 없고 기술적으로만 씁니다.
  - SpatialVLA 실험 34 r1e4/perp_left coke C − F "±10%p 동등(Holm)" → 보정 전만(Holm 0.147). P1-D r2e1 − r1e1 C − F 상호작용(pooled, physical) Holm 0.045 → 0.051로 동등 불성립. P1-D r1e4 C − F move_near physical의 ±5%p(보정 전) 동등과 [−2.6, +4.3]은 artefact였습니다([−6.9, +8.6]).
  - **유지:** P1-D pooled ±10%p 동등(4개 대비, 두 clustering, Holm), 과제별 3/8, S1–S4 pooled ±10%p.
  - 과제별 CI 폭: "18–29%p(r4e1; r1e4 6.9–21.6%p)" → **18–25%p(r4e1; r1e4 13–20%p)**.
- 비유의는 동등성이 아니며, 폭 0 구간은 같음의 근거가 아닙니다. AutoVLA 쪽 같은 구성의 6개 행(P1-C "below MME" 규칙 포함)은 CI 끝이 아니라 p로 읽습니다.

**A2 — 교정 window의 의미 (GT 유지, 해제 없음)**

- 코드·실제 forward trace·저장 기록 모두에서, window는 고정 위치 t*+1 … t*+w에 GT를 쓰고 **이후 모든 생성 step의 문맥에 그대로 유지**합니다(gate trace 4,136/4,136 전이에서 유지, 제거 0건; 실험 9 저장 기록 23,936/23,936행이 규칙과 일치). window가 끝나면 새 GT 위치가 추가되지 않을 뿐, 기존 GT는 제거되지 않습니다.
- 허용 문장: "Three fixed context positions replaced with ground truth and retained in later contexts recovered about 90% of the full-history correction effect." 함께 보고: scene 가중 90.1% [81.2, 98.9], A− 365 단위 / 52 장면 / 13 drive, 분모 = 같은 run의 Normal − GT-history(41.7%p), 단위당 GT 위치 2.96개(자기 token과 다른 것 1.90개), 마지막 GT 위치 이후 생성 step 평균 3.8.
- P1-B의 "한 번만 보이고 바로 자기 token으로 교체되는" GT(transient)는 78% [64, 87], 유지되는 GT는 89% [79.5, 98.0], 차이 +4.8%p [+2.9, +7.8](사후, 326 단위 / 12 drive; 유지와 노출 횟수가 섞임).
- 삭제: "3 step 교정 후 GT 정보를 제거해도 효과 유지", "이후 생성은 자기 token만 참고", "짧은 일시적 교정으로 지속 회복 입증", "해제 후 horizon/감쇠/재발산" 표현(→ "마지막 GT 위치 이후 step", "GT가 남아 있는 동안의 재발산"). 별도 GT 제거 실험 없이 해제 결과를 만들지 않았습니다.

**A3 — Impromptu 재현성 원인 (기술적·미검증으로 낮춤)**

- 같은 경로·같은 batch는 bit 단위로 재현됩니다(exp_cde 재실행 108/108행, B = 1 반복 4/4). batch 크기, left padding, prefill/incremental 분할, decoder 구현 중 하나만 바꿔도 출력이 달라집니다. 원인은 bf16 수치 차이가 greedy digit 디코딩의 근소 동점에서 0.01 m 끝자리를 뒤집고, 모델이 그 값을 다시 문맥으로 증폭하는 것입니다.
- 문서의 "no-op parity" 23/300, 237/2,728은 같은 조건 반복이 아니었습니다(구현 차이, batch 구성 차이). 장면 수준 A−/증폭 판정은 수치만으로 5–6% 바뀝니다.
- 결론: Impromptu는 **기술적 보완 증거**(경로 parity 미검증)이며 강한 독립 재현으로 세지 않습니다. 바닥을 크게 넘는 대비(교정 −63.4%p, reverse +81.9%p)만 바닥과 함께 기술하고, 작은 대비(win1 −15.4%p, 방향 − 크기 +15.6%p 등)는 재현이라 부르지 않습니다.

**A4 — 분모·지표·범위**

| 집합 | 장면 | 쓸 수 있는 이름 |
|---|---|---|
| dev 자연 평가 | 2,747 | A− 52 = 1.89%는 평가 집합 빈도(배포 실패율 아님) |
| dev 기전 집합 | 208 선정(A− 52 + A+ 156; A− 비율 25%) | 기전 대비는 A− 층(52 장면 / 13 drive) 조건부 |
| P1-C 평가 대상 E | 적격 2,191 | 평가 집합(E) 추정량 |
| P1-C 개입 표본 | 선정 353(F 105 + S 248), 설계 가중 | RN −4.1 / RV +3.6%p는 이 353 장면의 가중 추정 |
| P1-C 실패 층 F | 105 장면 / **18 drive** | "24 independent failure drives"라고 쓰지 않음 |
| 운동 baseline "전체" | 208 선정 | 2,747·2,191·모집단 아님 |

- RN 평가 집합 효과의 약 60%는 A+(비실패) 층 기여입니다. OpenVLA 92.6%는 "하위 성분 중 **적어도 하나**가 주입 bin 이상" 지표이고, bin은 m·rad 혼합(8 bin = x 2.64 mm, yaw 5.18 mrad 등), 명령값이며 실현 운동·과제 실패가 아닙니다. SpatialVLA의 m 단위 값은 명령 누적, 실제 TCP는 cm 단위입니다. 과제별 반대 부호 점추정만으로 이질성·상쇄를 확정하지 않습니다. P1-C는 drive-disjoint이지 day/vehicle-disjoint가 아닙니다.
- A−/severe 식: A− = NOT `pra_labels.label_A`(종·횡방향 범주 규칙), severe = A− ∧ FDE5 > 3 m(pose 10). 이 임계값은 3 s horizon용으로 문서화됐지만 5 s 계획에 적용됐습니다(원 정의 보존).
- 64개 문장 위반 중 57개를 이 문서와 `paper_quantitative_package`에 줄 단위로 고쳤고(해당 줄에 "(2026-10-18 수정)"), 나머지 7개는 역사 보고서(`review_followup3/CLAIM_EVIDENCE_MATRIX.md`, `CP2_REPORT.md`)라 고치지 않고 `MANUSCRIPT_REVISION.md`에 수정안만 적었습니다. 전체 목록: `outputs/review_followup4/DOC_EDITS_APPLIED.csv`.

**B1 — AutoVLA 수치 민감도 (GPU 18.3분, 56 dev 장면)**

| 설정 | suffix 변경 (row-run) | severe 판정 변경 |
|---|---|---|
| 같은 process 반복 | 0 (bit 단위 동일) | 0 |
| batch 형태만 | 8.9% | 0.26% |
| eager attention | 26.4% | 1.2% |
| fp32 연산 | 25.0% | 0.7% |

- RN / Full-GT / RV 대비는 어느 설정에서도 material하지 않았습니다(severe 변화 ≤ 0.036, FDE5 대비 변화 ≤ 0.16 m vs 효과 1.5–1.9 m).
- P1-A self-reference 대비(OC − U, OC_refR − OC, OC_refH − OC)의 0 근처 FDE5 차이는 수치 설정에 따라 부호가 바뀌었습니다 → "수치 구현 변동과 구별되지 않음"으로 제한합니다. +4.37 m와 A− 외삽 초과는 바닥보다 훨씬 커서 유지합니다.
- 정상 수치 변동을 개입 효과에서 빼지 않고 옆에 보고합니다.

**B2 — AutoVLA matched-prefix factorial (새 효과 실험; 사전 등록 552b3ac, 결과 c7d1bba)**

- 설계: 하나의 편집된 prefix(이전 prefix = GT 또는 matched natural; 최신 block = GT / frozen Normal-source / frozen corrected-source)를 만들고 **이후에도 고정 위치에 유지**한 채 suffix를 B = 1로 생성. 모든 cell이 같은 관측·지시·anchor·source bank·디코딩. identity 검사 100%(P1-C 4,160, dev 2,630 unit-anchor), P1-C 입력 검사 2,160/2,160.
- 실패 층 F(P1-C 100 장면 / 711 단위 / 18 drive; severe, Holm):

| 대비 | F | E4 (Hajek, 312 장면 / 24 drive) |
|---|---|---|
| C_GT = GT_NSRC − GT_REF | **+25.3%p** [+20.5, +29.3] | +3.3%p [+1.0, +5.6], Holm 0.051 (α 0.025에서 비유의) |
| C_NAT = NAT_NSRC − NAT_REF | **+25.5%p** [+21.6, +29.0] | **+4.0%p** [+2.7, +5.7], Holm 6.5e-5 |
| INT = C_GT − C_NAT | −0.2%p [−3.8, +3.1] | −0.7%p [−2.2, +0.8] |
| C_PROV = GT_CSRC − GT_NSRC | **−15.8%p** [−20.1, −10.2] | −2.2%p [−4.2, −0.2], Holm 0.13 |

- 해석: 이 단발(one-shot) 일정에서 효과는 **최신 문맥 token의 값**을 따르며, 이전 GT prefix의 존재에 의존하지 않습니다(INT ≈ 0; 이전 값이 실제로 다른 단위에서도 −1.7%p [−5.4, +1.9]). 실패 층 효과는 크고 E4 평균 효과는 작습니다. E4는 t* ≤ 4 영역이며 원래 2,191 장면 추정량이 아닙니다. dev 코호트(A− 49 장면 / 13 drive)도 같은 패턴입니다(C_GT +28.5, C_NAT +28.4, INT +0.1%p).
- 범위: GT cell이 있으므로 **oracle-conditioned diagnostic**입니다(online 복구·배포 개선 아님). 편집은 유지되므로 해제 실험이 아니고, 기존 drive의 새 개입 arm이므로 외부 검증이 아닙니다. 반복 Recent-GT / Reverse-N 결과와 섞지 않습니다. C_PROV는 값의 차이이지 순수 provenance 효과가 아닙니다. 민감도 anchor(n_old = 2)의 P 수준 INT +2.7%p(p = 0.0046)는 사전 등록 검정이 아닌 민감도 결과입니다.

**B3 — OpenVLA 최소 경로 검증: NOT_RUN** (사용자 결정 2026-10-17; LIBERO-Spatial checkpoint가 디스크에 없음; 여유 디스크 약 2.3–3 GB). OpenVLA는 기술적·미검증으로 두고 핵심 증거에서 제외합니다. 20% 재실행 불일치를 "효과 검출 임계값 20%p"로 쓰지 않습니다(`OPENVLA_GATE.md`).

**§6 — π₀-FAST 적합성(feasibility only).** 공식 `pi0_fast_libero` checkpoint는 공개 bucket에 있으나 이 장비에서는 디스크(약 20–25 GB 필요)와 RTX 5090(sm_120)의 JAX 동작 미확인 때문에 지금은 불가합니다. π₀-FAST는 문맥에 action history가 없어 step 간 action-context 개입은 NA입니다(측정된 0이 아님). WorldVLA의 mask 적용/미적용 checkpoint 쌍은 확인되지 않았습니다(`FEASIBILITY.md`).

**재현 ZIP (A5).** `reproducibility_minimal.zip` 71.8 MB(이미지·checkpoint·raw logit 제외; 코드, protocol, manifest, judge 로그, 단위 수준 표, B2 cell·bank, 고정된 공개 revision·checksum `PINS.csv`, 누락 목록 `MISSING.md`). 압축을 푼 사본에서 `regen.sh`로 핵심 표(A1_corrected_CIs, A1_coverage_selection, B2_dev, B2_p1c, P0A_population, P2_motion_baselines)를 실제로 다시 계산해 원본과 비교했습니다: 파일 22개 중 byte 동일 20개, 값 동일 2개, 불일치·실패 0개(`REPRO_CHECK.md`).

**주장 변경 요약** (`CLAIM_CHANGES.md`):
- 유지(좁힘): 실패 층 반복 Recent-GT/Normal-branch 주입 효과(dev 13 drive, P1-C F 18 drive), B2 최신 token 값 효과(INT ≈ 0, oracle-conditioned), 평가 집합 효과 −4.1 / +3.6%p(353 선정 가중, RN 60% A+)와 B2 E4 +4.0 / +3.3%p, GT 없는 +4.37 m와 A−에서만의 외삽 초과, SpatialVLA 생성·TCP 변화와 두 과제 평균 ±10%p(수정 구간).
- 좁힘: 90% → "GT로 바꾸고 유지한 3개 고정 위치"; P1-A self-reference 0 근처 FDE 차이는 수치 바닥과 구별 불가.
- 낮춤: Impromptu, OpenVLA → 기술적·미검증, 핵심 증거 제외.
- 삭제: 해제·일시 교정·자기 token만 참조 주장, "GT history가 필요", 순수 provenance, online 교정, 외부 재현, OpenVLA 과제별 결론과 C − N "CI 0 제외", 일반 VLA·보편 원인·모델 간 구조 순위·π₀ vs π₀-FAST 구조 효과.

---

## 7. 불안정 rollout 탐지: 이탈 전에 예측 가능한가 (실험 31, P4)

실험 20–24의 5090 후보(dev 56 log 42,095 이탈 후보 / held-out 52 log 43,469, 증폭 유병률 5.8%)를 teacher forcing해 margin과 hidden state(layer 18, 36)를 얻고,
probe는 dev에서만 학습해 held-out에 한 번 적용했습니다. 프로토콜 bf9aa21. 상세: `outputs/instability_detection_baselines/RESULTS.md`.

| held-out 탐지기 | pre AUROC [95% CI] | pre 장면 내 | post AUROC [95% CI] | post 장면 내 | post AUPRC |
|---|---|---:|---|---:|---:|
| entropy | 0.634 [0.607, 0.658] | 0.516 | 0.823 [0.809, 0.842] | 0.786 | 0.215 |
| log-likelihood | 0.570 [0.549, 0.587] | 0.596 | 0.832 [0.820, 0.846] | 0.811 | 0.246 |
| probability margin | 0.589 [0.561, 0.615] | 0.504 | 0.769 [0.755, 0.786] | 0.716 | 0.132 |
| 후보 궤적 분산 | 0.582 [0.566, 0.598] | – | 0.797 [0.786, 0.810] | – | 0.157 |
| pairwise disagreement | 0.620 [0.604, 0.637] | 0.539 | **0.900 [0.890, 0.913]** | 0.898 | 0.381 |
| medoid 거리 | 0.630 [0.610, 0.652] | 0.649 | 0.883 [0.866, 0.901] | **0.904** | **0.403** |
| hidden logistic / ridge / MLP probe | 0.61–0.63 | 0.54–0.55 | 0.745–0.765 | 0.74–0.75 | 0.18–0.20 |

→ **불안정 rollout은 이탈 전에는 예측되지 않고**(모델 내부 표현을 써도 엔트로피보다 낫지 않음), **이탈 이후 feedback이 진행되면서 구별됩니다.**
이탈 후에는 후보 간 불일치가 가장 강한 신호이고 확신도가 그다음입니다. 선택이 통하는 이유는 이탈 후 자기 강화 구간을 보기 때문이며,
이탈을 미리 막는 trigger로는 쓸 수 없습니다.

---

## 8. 연속 closed-loop (P6) — 수행 불가

공개된 AutoVLA checkpoint는 NAVSIM용(`AutoVLA_PDMS_89.ckpt`) 하나뿐이고 CARLA/Bench2Drive용은 없습니다. 컨테이너에는 CARLA에 필요한
NVIDIA EGL/Vulkan 라이브러리가 없고(LIBERO도 Mesa 소프트웨어 렌더링으로 실행), 디스크도 약 20 GB 부족합니다. nuPlan closed-loop는 카메라를
렌더링하지 않아 카메라 기반 AutoVLA를 돌릴 수 없습니다. 필요한 조건은 `outputs/continuous_closed_loop/STATUS.md`에 정리했습니다.

---

## 9. 종합 그림

(2026-10-13 개정: review_followup2 판정 반영. 이전 그림의 "2–4 step이면 90–95% 차단", "방향이 핵심", "구조가 효과를 흡수"는 축소·삭제됐습니다.)

```
perception (대체로 보존)
   │
   ▼
결정/CoT 텍스트 ──(느슨한 결합)──▶ action token 생성 (마지막 layer에서 형성)
                                        │ 첫 이탈 t*
                                        ▼
            직전 action 문맥(특히 motion)이 residual을 통해 다음 token을 조건화
            → 매 step 문맥을 GT로 교정하면 증폭 −40%p, 별도 Normal branch token을 넣으면 +29%p (13 drive; 새 drive의 실패 층(105 장면 / 18 drive)에서 재현) (2026-10-18 수정)
            → GT 없이 문맥 하나만 교란해도 이후 local action이 바뀜 (AutoVLA; 외삽 초과는 실패 층에서만)
            (장면 맥락이 증폭 여부를 좌우; 방향 > 크기는 개발 drive에서만, 새 drive 미결론)
                                        │
          oracle 문맥 교정: 연속 3 step ≈ 90% [81, 99] 회복 (첫 step이 특별하다는 근거 없음; GT 위치는 이후 문맥에 유지, 해제 실험 없음) (2026-10-18 수정)
          GT 없는 조건화 교정: 정상 장면 손해로 순이득 없음; reference 교정은 일관성 회복이지 실패 감소가 아님
          배포 가능한 완화: 후보 선택 (open-loop·PDMS에서 유의; held-out은 log 분리·drive 비분리)
                         + 현재 frame 안전 필터 (반응형 navhard에서 이득의 대부분, DA 제약이 핵심)
          남은 병목: 후보 생성 (oracle 0.405, token의 1/3은 안전한 후보가 없음)
```

- 실패의 축은 "무엇을 보았는가"보다 **"직전에 무엇을 출력했는가"**입니다. 직전 action의 motion이 embedding 근접보다 중요하다는 것까지는 지지되지만, 방향이 핵심이라는 주장은 개발 drive에 한정됩니다.
- 모델별로 다른 개입에서 token 수준 문맥 효과가 관찰됐습니다(SpatialVLA는 복원 검증; OpenVLA·Impromptu는 기술적·미검증으로 강한 독립 재현으로 세지 않음). 궤적 수준 증폭은 시간축 action history를 생성하는 두 driving 모델에서 나타났습니다(Impromptu, 실험 33). 다만 모델마다 Reverse 정의와 증폭 정의가 달라 효과 크기는 비교하지 않습니다. (2026-10-18 수정)
- SpatialVLA(로봇 조작)에서는 문맥 교란이 생성 action과 실제 TCP 궤적을 바꾸고, 재계획 간격이 길수록 그 편차가 오래 남습니다(P1-D, 복합 설정 효과). 성공률 효과는 두 과제 평균 ±10%p 이내로만 제한되고(검증한 r1e4·r4e1), 과제별 효과와 ±5%p 수준은 불확실합니다. "구조가 효과를 흡수한다", "ensemble이 보호한다", "4 step 재계획에서 reverse가 실패를 늘린다(−6.2%p)"는 근거가 없습니다.
- 불안정성은 이탈 전에는 약하게만 보이고(AUROC 0.57–0.63) 이탈 후에 강하게 보이므로, 실용적인 완화는 "이탈을 미리 막기"가 아니라 **여러 후보를 만들고 이탈 후 신호와 현재 frame 안전 규칙으로 거르기**입니다.

---

## 10. 한계

- **독립 표본 수가 작습니다.** AutoVLA 기전 실험(5–11, 32)의 A−는 52 장면이지만 독립 단위는 **13 drive**(10 day-vehicle)이고, exact sign-flip p는 2.4e-4보다 작을 수 없습니다. 단일 seed·T 0.01입니다. day / vehicle / day-vehicle 의존성은 review_followup3 P0-D에서 민감도 분석했습니다: 모든 축에서 부호가 유지되지만, P1-C는 day–vehicle 연결 성분이 6개뿐이라 exact 검정의 Holm 하한이 0.094입니다 (2026-10-15 수정).
- **탐색 실험과 사전 등록의 구분.** 실험 4–11은 사전 등록이 없고, R8R9는 원 수치를 아는 상태의 복구 실행입니다. 새 자료에서의 확인은 P1-C(24 drive, day-vehicle 10개 공유)뿐이며, 방향 > 크기는 거기서 결론이 나지 않았습니다(사전 설계 검정력 0.19–0.33).
- **분할 경계.** 선택 실험의 held-out과 navhard 2차 절반은 log만 분리되고 drive를 공유합니다(22개 / 16개). drive-disjoint 재계산은 하지 않았습니다.
- **개입 정의.** Recent-GT는 매 step 반복 개입이고, Reverse는 모델마다 다른 처치입니다. GT 없는 대조(P1-A)는 AutoVLA에만 있고, 교란 위치와 A−/A+ 층은 GT 정의를 상속합니다. 기존 repeat-p constant-motion(CM)은 body-frame SE(2) 반복(이미 일정 회전율 외삽)임을 확인했고, CV0·CTRV-K3·CTRA-2와 학습된 non-AR 운동 외삽기를 추가했습니다; 외삽 초과는 A−에서만 성립합니다(review_followup3 P2) (2026-10-15 수정).
- **교정 위치·길이.** 교정 위치 수와 마지막 GT 위치 이후 step 수가 함께 변하고(GT는 이후 문맥에 유지되며 해제 실험은 없음), 절대 pose와 그 step 수를 동시에 맞출 수 없습니다(10 token 한계). 실험 5·10의 per-unit raw가 없어 1-token 교정과 재발산은 summary 수준입니다. (2026-10-18 수정)
- 평가 horizon은 5 s(10 token)이며, 그 이상은 모델 분포 밖입니다. A−는 P/R/A 행동 범주 판정(pra_labels.label_A: 종방향 v_end/v_start 1.25/0.75, 저속 0.5 m/s·가속 0.8 m/s, STOP = 경로 < 1.0 m & v_end < 0.5 m/s, 허용 max(0.5 m/s, 15%); 횡방향 최종 횡변위 > 2.0 m 또는 마지막 step heading > 15°, 허용 ±1 m/±10°)이고, 증폭(severe) = A− ∧ FDE5 > 3 m입니다. 이 임계값은 3 s horizon(HORIZON = 6)용으로 문서화되었지만 10 pose(5 s) 계획에 그대로 적용됐습니다(원 정의 보존; 3 s 민감도는 P/R/A 비교에만 있음; review_followup4 A4). (2026-10-18 수정)
- navhard 결과는 2단계 pseudo closed-loop이고, 안전 필터와 채점이 같은 규칙을 씁니다. 연속 closed-loop(P6)는 수행하지 못했습니다.
- GPU·batch 구성이 달라지면 경계 장면의 디코딩이 bf16 수치 차이로 갈릴 수 있어, 모든 비교는 같은 실행 안의 쌍대 비교로 했습니다.
- OpenVLA 재현(실험 28)은 LIBERO-Spatial 한 suite, 100 에피소드 규모이고, OpenVLA는 시간축 action history가 없어 step 내 차원 간 feedback만 직접 비교할 수 있습니다.
- Impromptu 재현(실험 33)은 open-loop 궤적 실패 기준이고, 절대 위치 표현 때문에 GT 문맥 교정에 GT 정보 누출이 섞여 있습니다. 궤적 수준 안정 분기(사전 등록 기준 3)는 재현되지 않았습니다. GT 없는 대조는 없습니다.
- **SpatialVLA 초기 상태.** 과제가 2개뿐이고, move_near는 물리 배치가 **10개**뿐이며 실험 34·35·P1-D가 모두 같은 10개를 씁니다(지시문 포함 primary 상태는 29/53/120). 따라서 move_near의 독립 표본은 명목 seed 수보다 훨씬 작고, 과제별 성공률 CI가 넓습니다(P1-D 18–29%p). 과제별 ±5%p를 검증하려면 과제당 약 1,500개의 새 물리 상태(75–209 GPU h)가 필요합니다. 새 SimplerEnv 변형 상태 실험(P4, ±10%p 목표)은 계획만 하고 실행하지 않았습니다(review_followup3 CP2 결정) (2026-10-15 수정).
- 실험 34/35는 reset이 이력에 의존하고(reconfigure 없음, 굴러가는 물체 최대 40× 허용오차 drift) 복원 hash가 없으며, 실험 34에는 실제 TCP 기록이 없습니다(m 단위 "궤적 발산"은 실행 명령 누적). closed-loop 교란은 d = 0.3·방향 2개·step 8–23만 다뤘고, chunk가 4 step이라 window를 2 step보다 길게 볼 수 없습니다.
- P1-D의 재계획 축 효과(H1)는 관측 갱신·문맥 reset·실행 노출이 함께 바뀐 복합 설정 효과이고, 관측 갱신과 sequence reset의 분리는 실행 가능한 정책 안에서 하지 못했습니다.
- ±5%p는 이번 검토가 제안한 사후 민감도 범위이며 분야의 공인 기준이 아닙니다. ±10%p 동등성은 두 과제 평균에 대한 진술이고, 과제별 점추정의 부호가 일부 반대지만 과제 간 이질성은 검출되지도 배제되지도 않았습니다. (2026-10-18 수정)

---

## 11. 재현과 데이터

- 환경·자산: `tools/download_autovla_assets.sh`, `tools/setup_autovla_env.sh`; OpenVLA/LIBERO는 `/root/VLA/openvla/env.sh`(autovla env + target dir + Mesa EGL).
- navhard: `tools/navhard_half2_pipeline.sh`, `tools/navhard_score_groups.sh`, split 설정 `autovla_misalignment_poc/configs/navsim_v2/`.
- ablation·구성요소·oracle: `tools/navhard_ablation_pipeline.sh`, `tools/navhard_components_candidates_pipeline.sh`, `tools/paper_jobs_scheduler.sh`.
- Impromptu VLA: `autovla_misalignment_poc/scripts/cross_vla_temporal/` (`impromptu_core.py`, `exp_ab.py`, `exp_cde.py`, `analyze_temporal.py`), checkpoint `/root/VLA/impromptu/3B_AD`; 실험 33 디스크 정리 기록 `outputs/cross_vla_temporal_replication/DISK_CLEANUP.md`.
- SpatialVLA: `autovla_misalignment_poc/scripts/cross_domain_temporal/` (`spatialvla_core.py`, `spatialvla_policy.py`, `svla_rollouts.py`, `svla_offline.py`, `analyze_svla.py`), 환경 `/root/VLA/simpler/env.sh`(transformers 4.47 target dir + SimplerEnv + Mesa lavapipe Vulkan), checkpoint `/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal`; 실험 34 디스크 정리 기록 `outputs/cross_domain_temporal_replication/DISK_CLEANUP.md`(OpenVLA LIBERO checkpoint 15 GB 삭제, `snapshot_download`로 재생성).
- OpenVLA: `autovla_misalignment_poc/scripts/cross_vla/` (`openvla_core.py`, `openvla_rollouts.py`, `openvla_token_feedback.py`, `analyze_cross_vla.py`).
- review_followup2 사후 재분석(drive·초기 상태 군집, 직접 효과, baseline·window)의 재생성 명령: `autovla_misalignment_poc/outputs/review_followup2/RUN_COMMANDS.md`(CPU만, 모델 재실행 불필요).
- 원시 rollout·hidden state·submission은 용량 때문에 저장소에 넣지 않았고, 위 스크립트로 재생성됩니다.
- 디스크 정리 기록: 사용하지 않는 카메라(CAM_B0/L0/L2/R0/R2, 약 19 GB)를 삭제했습니다. 원 전처리(preprocess_scenes)를 처음부터 다시 하려면 shard 0–5를 다시 받아야 합니다.
