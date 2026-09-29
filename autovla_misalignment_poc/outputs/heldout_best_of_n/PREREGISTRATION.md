# Held-out 평가 사전 등록 (2026-09-29, held-out 결과를 보기 전에 작성)

**데이터**: OpenScene navtest 카메라 shard 18–31의 navtest log(PoC 28 log, 개발 셋 56 log와 겹치지 않음). shard 18–25는 RTX 5090(`gpu1`), 26–31은 RTX 3080 Ti(`gpu0`)에서 디코딩. 후보 = 자연 계획(T 0.01) + T 1.0 샘플 16개, seed 0. 총 4,814 장면.

**고정된 규칙 (개발 셋 `outputs/expanded_best_of_n`에서 결정, 이후 변경 없음)**

| 역할 | 규칙 | 정의 |
|---|---|---|
| 기준 | normal | 자연 계획(후보 0) |
| **주 규칙** | **ranksum** | 후보 17개 중 (평균 엔트로피 순위 + 합산 log-prob 순위 역순)이 최소 |
| 보조 | max_loglik, min_entropy | 실험 18–20 정의 그대로 |
| **주 필터** | **F1** | 현재 frame 정보만으로(등속 외삽 + 지도) 충돌 또는 주행 가능 영역 이탈이 예측된 후보 제거 → 남은 후보 중 ranksum, 모두 제거되면 전체 ranksum |
| 보조 필터 | F2, F3 | F1 + 역주행 / F1 + TTC (`scripts/safety_filter_select.py`) |
| 후보 수 | ranksum at N = 1, 2, 4, 8, 12, 16, 17 | 자연 계획 + 무작위 N−1개 샘플(장면당 5회 평균); PDMS는 앞쪽 부분집합 |

F1을 주 필터로 고른 근거(개발 셋): F1–F3의 PDMS 차이(0.9293–0.9309)는 CI 안이며, F1이 가장 단순하고 충돌(0.15%)과 open-loop FDE(0.507 m)가 가장 좋았음.

**주 지표**: PDMS(navsim 기본 채점 설정), 충돌·주행 가능 영역·TTC·승차감 위반율, 진행도; open-loop A−(P/R/A 5 s), 강한 실패(A− ∧ FDE5 > 3 m), FDE5. 모든 CI는 log 단위 cluster bootstrap 95%, 쌍대 검정은 McNemar(이진)/Wilcoxon(연속).

**기전 분석**: `scripts/mechanism_selection_link.py`를 그대로 적용(선택 vs 버림의 이탈 전/후 엔트로피, 오차 증가, 재정렬, 증폭; 증폭 AUROC).

**판정 기준**: ranksum과 F1의 ΔPDMS 95% CI가 0을 배제하면 개발 셋 결과가 재현된 것으로 본다. held-out은 한 번만 평가하며, 결과를 본 뒤 규칙을 바꾸지 않는다.
