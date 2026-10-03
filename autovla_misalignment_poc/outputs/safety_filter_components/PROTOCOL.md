# 실험 29 (P2): 안전 필터 구성요소 ablation — 채점 전 작성 (2026-10-03)

전체 navhard(76 log, 225 그룹), 실험 25–26의 RTX 5090 디코딩과 현재 frame 플래그를 그대로 사용(새 GPU 작업 없음). threshold는 dev 고정값(플래그 ≥ 1).

| # | 조건 | 통과 기준 | 통과 후보 중 선택 | 출처 |
|---|---|---|---|---|
| 1 | no filter | – | 후보 0 (자연) | `normal` 재사용 |
| 2 | collision-only | no_at_fault_collisions ≥ 1 | 후보 0이 통과하면 0, 아니면 통과한 첫 후보, 없으면 0 | 신규 `comp_collision` |
| 3 | drivable-area-only | drivable_area_compliance ≥ 1 | 같음 | 신규 `comp_dac` |
| 4 | collision + drivable-area | 둘 다 | 같음 | 실험 27 `filter_only` 재사용 |
| 5 | full filter + rank-sum | 둘 다 | rank-sum, 없으면 전체 rank-sum | `F1` 재사용 |
| 6 | full filter + max log-lik | 둘 다 | max log-lik, 없으면 전체 max log-lik | `F1_maxll` 재사용 |

대비: 2 − 1 (충돌 제약 단독), 3 − 1 (주행 가능 영역 제약 단독), 4 − 2 (충돌 위에 DA 추가), 4 − 3 (DA 위에 충돌 추가), 4 − 1, 5 − 4, 6 − 4.
지표: EPDMS, 원본/합성 충돌, 원본/합성 DAC 위반, 원본/합성 진행도, 합성 승차감, 개선/악화/동일 그룹.
진행도 희생 여부: 진행도 차이와 CI, EPDMS 이득을 곱셈 항(NC·DAC·DDC·TLC)과 가중 항(EP·TTC·LK·HC·EC)으로 나눈 Shapley 분해(token 단위, 공식 CSV).
통계: log 단위 bootstrap 95% CI(2,000, seed 0), log 단위 paired sign-flip permutation(20,000, seed 0), 그룹 단위 Wilcoxon.
