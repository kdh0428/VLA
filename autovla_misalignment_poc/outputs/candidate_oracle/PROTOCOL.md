# 실험 30 (P3): 후보 oracle 상한 — 채점 전 작성 (2026-10-03)

전체 navhard(76 log, 225 그룹), 새 디코딩 없이 기존 후보 17개(후보 0 = T 0.01 자연, 1–16 = T 1.0 샘플).

1. 후보 k(0–16)를 모든 장면에 쓴 submission을 공식 2단계 파이프라인으로 채점 → token별 공식 점수(CSV `score`). k = 0은 `normal` 재사용.
2. **oracle17**: token마다 17개 중 공식 token 점수가 가장 높은 후보(동률이면 낮은 번호). **oracle16**: 샘플 16개(1–16) 중에서만.
   oracle 선택 submission을 다시 공식 채점(2단계 가중치와 two-frame comfort가 선택에 따라 다시 계산됨).
3. 비교: no selection, rank-sum, max log-lik, safety filter only, safety + rank-sum, safety + max log-lik, oracle16, oracle17.
4. 해석: oracle − 최선 방법 = 선택의 여지(selection gap), oracle 자체의 절대 수준 = 생성의 한계(generation ceiling).
   oracle은 GT 미래를 쓰는 분석용 상한이며 배포 방법이 아닙니다.
통계: P2와 동일(log bootstrap, log permutation, 그룹 Wilcoxon).
