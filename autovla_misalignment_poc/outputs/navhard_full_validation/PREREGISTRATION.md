# 사전 등록: navhard 나머지 절반 검증 (실험 26)

작성: 2026-10-02, navhard 두 번째 절반의 디코딩·채점 전. 이 파일을 커밋한 뒤에 두 번째 절반 결과를 생성합니다.
규칙·threshold는 dev(navtest shard 6–17)에서 고정한 값 그대로이며, 중간 결과를 보고 바꾸지 않습니다.

## 데이터

- 첫 절반 (이미 보고, 실험 25): navhard_two_stage 76 log 중 36 log, 105 그룹 (`../navhard_eval/subset/chosen_logs.json`).
- **두 번째 절반 (이번 검증)**: 나머지 40 log 전부, 120 그룹, 1단계 240 + 합성 2,760 장면
  (`half2/subset/second_half_logs.json`, split `navhard_half2`). 두 절반은 log가 겹치지 않습니다.
- 전체 navhard: 76 log, 225 그룹 = 두 절반의 그룹을 합친 것(공식 EPDMS가 그룹 평균이므로 합쳐도 같은 정의).

## 고정된 절차

- 디코딩: RTX 5090만 사용(`CUDA_VISIBLE_DEVICES=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), `expanded_best_of_n.py` 기본값
  (T 0.01 자연 계획 + T 1.0 샘플 16개 = 후보 17개), 첫 절반과 같은 코드.
- 안전 플래그: `navhard_safety_flags.py` (현재 객체 등속 외삽, 현재 신호 유지, 지도; human_penalty_filter 끔; 미래 정보 없음).
- 채점: 공식 `run_pdm_score_from_submission.py` + 그룹 점수 저장 래퍼 `navhard_group_scores.py`.

## 비교 조건 (선택 규칙)

| 이름 | 정의 |
|---|---|
| `normal` | 선택 없음 (후보 0 = T 0.01 자연 계획) |
| `ranksum` | rank(평균 엔트로피) + rank(−합산 log-prob) 최소 (`analyze_best_of_n.pick`) |
| `max_loglik` | 합산 log-prob 최대 |
| `F1` | 플래그 no_at_fault_collisions ≥ 1 이고 drivable_area_compliance ≥ 1 인 후보 중 rank-sum; 없으면 전체 rank-sum (`safety_filter_select.VARIANTS["F1"]`) |
| `F1_maxll` | 같은 F1 생존 후보 중 합산 log-prob 최대; 없으면 전체 max_loglik (이번에 새로 정의, 첫 절반 결과를 보기 전에 고정) |

## 지표와 검정

- 주 지표: EPDMS. 보조: 1단계(원본)·2단계(합성) 점수, at-fault 충돌, 주행 가능 영역 위반, 진행도, history comfort,
  extended comfort, TTC, 주행 방향 (공식 가중 평균; 그룹 값은 orig·prev 두 부분의 평균).
- 각 규칙 vs `normal`: 평균 차이, log 단위 bootstrap 95% CI(2,000회, seed 0),
  **주 검정 = log 단위 paired sign-flip permutation(양측, 20,000회, seed 0)**, 보조 = 그룹 단위 Wilcoxon signed-rank,
  개선/악화/동일 그룹 수와 1단계 장면 수. 스크립트: `scripts/analyze_navhard_full.py`.

## 재현 판정 (두 번째 절반, 각각 α = 0.05, 다중 비교 보정 없음 — 가설 2개를 미리 지정)

- **H1 (F1)**: EPDMS Δ(F1 − normal) > 0 이고 bootstrap 95% CI 하한 > 0 이며 log permutation p < 0.05.
- **H2 (max_loglik)**: 같은 기준으로 Δ(max_loglik − normal) > 0.
- rank-sum, F1_maxll은 기술적으로 보고합니다(판정 대상 아님). 첫 절반에서의 효과 크기(F1 +0.125, max_loglik +0.032)와의
  일치 여부는 CI가 첫 절반 추정치를 포함하는지로 함께 보고합니다.
- 마지막으로 전체 navhard(225 그룹)에서 같은 표를 보고합니다.

첫 절반은 같은 디코딩 결과(`../navhard_eval/decode_5090`, 읽기 전용)를 이번 절차로 다시 채점해 `half1/`에 둡니다(F1_maxll과 세부 지표 추가). 기존 `navhard_eval/` 결과는 수정하지 않습니다.
