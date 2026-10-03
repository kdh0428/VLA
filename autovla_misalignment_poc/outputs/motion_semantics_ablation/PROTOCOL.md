# 실험 32 (P5): previous-action motion semantics ablation — 실행 전 작성 (2026-10-03)

실험 11과 같은 harness·데이터(equal-distance set, 208 장면 / 1,408 단위, A− 52 장면 / 365 단위), RTX 5090.
t*+2부터 매 step 직전 context token을 행마다 바꿉니다. token motion m = 코드북 끝 pose의 (dx, dy) (ActionTokenizer.rollout과 동일),
GT token g, 행 자신의 token o, 자기 오차 e = |m_o − m_g|.

| 행 | 직전 token | 무엇을 맞추고 무엇을 틀리게 하나 |
|---|---|---|
| normal | o (모델 생성) | – |
| recent_gt | g | 방향·크기 모두 정답 |
| dir_ok_mag_wrong | 방향 = g, 속도 오차 = e | 방향 정답, 크기 오답 |
| dir_wrong_mag_ok | 속도 = g, 변위 오차 = e | 크기 정답, 방향 오답 |
| mirror_same_dist | 2 m_g − m_o 최근접 (g 반대편, 같은 거리 e) | 같은 크기의 오차, 반대 방향 |
| random_mag_matched | 속도 ±5%(최소 0.05 m) 안의 무작위 token | 크기만 정답, 방향 무작위 |

dir_ok_mag_wrong·dir_wrong_mag_ok·mirror_same_dist는 GT로부터의 끝점 거리를 자기 token의 오차 e와 맞춰, 오차의 "크기"가 아니라
"종류"(방향 vs 속도)를 비교합니다. 실제 달성된 Δ방향·Δ크기·거리를 대체마다 기록해 확인합니다.

지표 (실험 11과 동일 정의, analyze_action_history): 증폭률(A− and FDE5 > 3 m), 회복률(P/R/A 5 s 승인), ADE, FDE,
이후 step 엔트로피와 log-likelihood. A− 단위가 주 분석, A+는 보조.
통계: log 단위 bootstrap 95% CI, 단위 짝 비교(McNemar / Wilcoxon), recent_gt 효과 대비 비율.
핵심 질문: previous-action feedback에서 중요한 정보가 방향인가, 크기인가, 둘의 조합인가.
