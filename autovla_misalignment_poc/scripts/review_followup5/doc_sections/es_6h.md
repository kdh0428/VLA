## 6h. review_followup5 (2026-10-18 수정, review_followup5): AutoVLA 구간 재검토, B2 상위 군집, GT-free 수치 재추론

실행 지시: `CODING_AGENT_PROMPT_FOLLOWUP5.md`. 결과 `autovla_misalignment_poc/outputs/review_followup5/`(STATUS, CLAIM_CHANGES, MANUSCRIPT_REVISION, RUN_COMMANDS, REPRO_CHECK, INPUT_HASHES). 프로토콜은 결과 전에 로컬 파일로 고정했습니다(`FOLLOWUP5_PROTOCOL.md`, sha256 543ff14e…; 공개 사전 등록 아님). commit하지 않았습니다.

**A — AutoVLA 구간·TOST·MME (CPU, 사후 수정).** 이진 결과의 sign-flip 역산 CI와 이동 sign-flip TOST는 동점 때문에 무효이고, δ = 0 sign-flip p(부호 대칭 귀무가설 아래 열거 정확)는 유효합니다. 실제 군집 구조로 coverage를 먼저 확인했더니 **보고해 온 drive bootstrap 95% CI도 과소 포함**(최소 0.83–0.91)이었습니다. 선택: dev A−(13 drive)·P1-C F(18 drive)는 cluster-t, **P1-C 평가 집합 P(가중, 24 drive)·day–vehicle·7 drive 이하 부분집합은 NR**(어느 방법도 기준 미달).
- 결론 유지: dev RN −39.8%p cluster-t [−50.9, −28.7], RV +28.6 [+20.1, +37.2]; P1-C F RN −34.2 [−39.0, −29.4], RV +26.8 [+22.1, +31.5]. "replicated" 판정은 δ = 0 p와 부호에 근거하므로 그대로입니다.
- 바뀐 것: P1-C F DM +6.2%p의 구간이 0을 포함([−0.1, +12.4]; 판정은 원래도 inconclusive). **P 수준 −4.1 / +3.6%p는 점추정과 p(5.7e-6, 9.5e-7)만 보고하고 CI는 검증되지 않음(NR)**. P1-C window win4 − gt_history ±5%p 동등은 F만 유지, S는 철회. 실험 11 GT 근접 token ±5%p: 원 sign-flip TOST(p 0.09)는 무효이고 사후 cluster-t TOST로는 ±5%p 안(p 0.027, 보정 전, 사후 margin)이지만 확증적 동등성으로 쓰지 않습니다. P1-B ±10%p 판정(w3 동등, w2 보정 전만)은 그대로입니다. "below MME" 판정은 원래도 없었습니다.

**A — B2 상위 군집 (CPU).** P1-C 312 장면의 day–vehicle 이분 그래프는 연결 성분 6개(dev ∪ P1-C는 5개, day–vehicle 10개 공유)입니다. F에서 C_GT / C_NAT는 6개 성분 모두 같은 부호이고 성분 검정 p = 0.031(= 최소값), Holm 0.125(4개 family의 최소 가능값 0.125 > 0.05)입니다. drive·day–vehicle 수준에서는 C_GT, C_NAT, C_PROV(F)와 C_NAT(P)가 Holm 후 유지됩니다. 성분 수준 비유의는 추론 해상도의 한계이며 효과 부재가 아닙니다.
- 문장 수정: primary anchor의 older factor는 **한 위치**(t*+1)입니다. INT −0.2%p [−3.8, +3.1]는 "효과가 같다/오래된 문맥과 독립"이 아니라 차이 추정치와 불확실성으로 보고하고, older-prefix 주효과(F +2.0 ~ +4.8%p)를 함께 씁니다. 사전 margin이 없으므로 동등성으로 승격하지 않습니다.
- Impromptu의 "5–6% 수치 바닥보다 크므로 강건" 문장도 고칩니다. 관측된 판정 불일치율은 조건별 수치 편향의 상한이 아니므로 기술적·parity 미확립으로만 씁니다.

**B — GT-free 핵심 대비의 수치 재추론 (GPU, RTX 5090 51.8분).** P1-A G family 전체(1,453 단위 / 선정 208 장면; A− 364 / 52 / 13 drive)를 S0(bf16·SDPA·B = 10), SE(eager attention), SF(fp32 연산)로 같은 harness에서 다시 생성했습니다. S0 재실행은 원 raw와 14,530/14,530 행이 같은 token이었습니다.
- A− endpoint 초과(Δ_model − Δ_baseline, m): Geo +4.37 → SE +4.06 / SF +4.45, CM +2.35 → +2.04 / +2.43, CV0 +4.02 → +3.71 / +4.10, CTRV-K3 +3.02 → +2.71 / +3.10, CTRA-2 +1.43 → +1.12 / +1.51, 학습 외삽기 +1.97 → +1.65 / +2.04. **사전 고정 family 12개 검정 모두 Holm 후 유지**(Holm p ≤ 0.009).
- 설정 간 직접 차이(A−): SE − S0 −0.31 m [−0.91, +0.40], SF − S0 +0.08 m [−0.39, +0.52]. 사전 기준으로 material하지 않지만, A−에서 최대 약 0.9 m의 설정 차이는 배제하지 못합니다(동등성 검정 아님).
- A+와 선정 208 장면 전체의 결론(일반적 외삽 초과 없음)도 SE·SF에서 같았습니다. GF1/GF3, E family, 0 근처 self-reference FDE 차이는 다시 돌리지 않았습니다. t*와 층은 GT 정의를 상속하므로 전체 절차가 GT-free online은 아닙니다.

**C — 독립 날짜·차량 자료: `BLOCKED_NO_DISJOINT_DATA`.** navtest 147 log, navhard two-stage 2,731 장면은 모두 dev ∪ P1-C 성분과 날짜·차량을 공유합니다. navmini/warmup 중 날짜·차량이 겹치지 않는 13 log는 모두 학습 자료(OpenScene trainval)입니다. private_test_hard(메타데이터 179 kB만 스트리밍)는 log·차량·도시가 익명이고 GT가 비공개라 판정할 수 없습니다. 검정력 계획: 실패 층 효과가 절반이어도 새 성분 10개(약 30 drive) 이상이면 Holm 기준 검정력 0.82–0.94이고, 성분 6개로는 Holm 유의가 불가능합니다.

**D — GT 유지 대 제거 (선택):** NOT_RUN, 프로토콜 초안만 있습니다.
