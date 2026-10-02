# 안전 필터 ablation 프로토콜 (실험 27)

작성: 2026-10-02, 새 조건을 채점하기 전. 목적: 전체 navhard(76 log, 225 그룹)에서 F1의 EPDMS 개선(+0.118)을
안전 필터 자체의 몫과 확신도 선택기의 추가 몫으로 나눕니다. 필터 threshold와 선택 규칙은 dev에서 고정한 그대로이며
새로 튜닝하지 않습니다.

## 데이터와 재사용

- 디코딩: 실험 25–26의 RTX 5090 디코딩 (`navhard_eval/decode_5090`, `navhard_full_validation/half2/decode_5090`), 읽기 전용.
  새 GPU 작업은 없습니다(선택과 채점은 CPU).
- 안전 플래그: 같은 실험의 `safety_flags/flags.jsonl` (현재 frame만 사용). 필터 = `VARIANTS["F1"]`:
  no_at_fault_collisions ≥ 1 이고 drivable_area_compliance ≥ 1 인 후보만 남김.
- 이미 채점된 조건은 `navhard_full_validation/half{1,2}/group_scores`를 그대로 씁니다(같은 picks, 같은 채점 코드).

## 조건 (후보 0 = T 0.01 자연 계획, 1–16 = T 1.0 샘플)

| # | 이름 | 생존 후보가 있을 때 | 생존 후보가 없을 때 | 출처 |
|---|---|---|---|---|
| 1 | `normal` | 후보 0 | 후보 0 | 재사용 |
| 2 | `ranksum` | (필터 없음) 전체 rank-sum | – | 재사용 |
| 3 | `filter_only` | 후보 0이 통과하면 후보 0, 아니면 통과한 후보 중 가장 앞 번호(샘플은 i.i.d.이므로 확신도와 무관한 선택) | 후보 0 | 신규 |
| 4 | `filter_random` | 통과 후보 중 균등 무작위 | 전체 중 균등 무작위 | 신규, seed 0/1/2 (`random.Random(f"{seed}:{token}")`), 그룹 점수는 3 seed 평균 |
| 5 | `filter_maxll` | 통과 후보 중 합산 log-prob 최대 | 전체 max log-lik | 재사용 (= `F1_maxll`) |
| 6 | `filter_ranksum` | 통과 후보 중 rank-sum | 전체 rank-sum | 재사용 (= `F1`) |

보조 조건(대체 규칙의 몫을 분리하기 위해): `filter_ranksum_natfb`, `filter_maxll_natfb` = 5·6과 같지만 생존 후보가 없으면 후보 0.

## 대비 (A − B, 그룹 단위 짝)

- 주: `filter_only − normal`, `filter_ranksum − filter_only`, `filter_maxll − filter_only`
- 보조: `filter_random − filter_only`, `filter_ranksum − filter_random`, `filter_maxll − filter_random`,
  `filter_ranksum_natfb − filter_only`(생존 후보 안에서의 선택기 몫), `filter_ranksum − filter_ranksum_natfb`(대체 규칙의 몫),
  같은 두 개를 max log-lik으로, `ranksum − normal`, `filter_ranksum − normal`.
- 지표: EPDMS, 1·2단계 점수, at-fault 충돌, 주행 가능 영역 위반, 진행도, history·extended comfort, 개선/악화/동일 그룹 수.
- 통계: log 단위 bootstrap 95% CI(2,000회, seed 0), log 단위 paired sign-flip permutation(양측, 20,000회, seed 0),
  그룹 단위 Wilcoxon signed-rank. 스크립트 `scripts/analyze_navhard_ablation.py`.
