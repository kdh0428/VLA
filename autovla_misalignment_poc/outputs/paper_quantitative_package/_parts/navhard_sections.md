# NAVSIM v2 navhard 2단계 pseudo closed-loop (실험 25, 26, 27, 29, 30) — 논문용 정량 근거

작성: 2026-10-06. 새 실험·채점 없이 기존 결과 파일에서만 뽑았습니다. 모든 평균·Δ·CI·p·그룹 수는 아래 분석 JSON 키에서
그대로 복사했습니다(재계산하지 않음). 직접 계산한 값은 이미 보고된 차이들의 **비율**(89%, 86%, 69% 등)과 token 행렬의 개수뿐이며
"(recomputed: `_parts/scripts/navhard_verify.py`)"로 표시했습니다. 그룹 점수 파일에서 다시 구한 225 그룹 평균은 네 JSON의
값과 모두 일치합니다(차이 ≤ 6e-17, `_parts/navhard_verify.json` → `json_minus_recomputed_mean`).

## 0. 공통 설정

| 항목 | 값 |
|---|---|
| 모델 | AutoVLA, `AutoVLA_PDMS_89.ckpt` (`scripts/_planner.py`의 `CKPT`) |
| 벤치마크 | NAVSIM v2 `navhard_two_stage`, 공식 `run_pdm_score_from_submission.py` + 그룹 점수 래퍼 `scripts/navhard_group_scores.py` |
| 규모 | 76 log, 225 그룹 = 첫 절반 36 log/105 그룹(split `navhard_half`) + 두 번째 절반 40 log/120 그룹(`navhard_half2`); 1단계 원본 450 장면(210+240) + 2단계 합성 5,462 장면(2,702+2,760) = token 5,912개 |
| 후보 | 장면당 17개: 후보 0 = T 0.01 자연 계획, 후보 1–16 = T 1.0 샘플 (`scripts/expanded_best_of_n.py`, `--seed 0`, token별 sha256 seed) |
| GPU | 디코딩 RTX 5090 (`CUDA_VISIBLE_DEVICES=1`, PCI_BUS_ID). 실험 27, 29, 30은 새 GPU 작업 없음(선택·채점 CPU) |
| 지표 | EPDMS = 그룹 평균, 그룹 점수 = (s1(orig)·s2(orig) + s1(prev)·s2(prev)) / 2 (`scripts/analyze_navhard_full.py` docstring) |
| 신뢰구간 | log-cluster bootstrap 95% percentile CI, 2,000회, seed 0 (`random.Random(0)`) |
| 주 검정 | log 단위 paired sign-flip permutation, 양측, 20,000회, seed 0 (`np.random.default_rng(0)`); 최솟값 1/20,001 = 5.0e-5 |
| 보조 검정 | 그룹 단위 Wilcoxon signed-rank (0 차이 제외) |
| 개선/악화/동일 | 그룹 차이 > 1e-9 / < −1e-9 / 나머지 |
| 다중 비교 보정 | 없음 (사전 등록 가설은 H1, H2 두 개) |
| 안전 필터 F1 | 현재 frame 플래그 `no_at_fault_collisions ≥ 1` 그리고 `drivable_area_compliance ≥ 1` (`safety_filter_select.VARIANTS["F1"]`; 현재 객체 등속 외삽·현재 신호 유지·지도, human_penalty_filter 끔, 미래 정보 없음) |

## 1. 전체 navhard 표 (76 log, 225 그룹, 모두 같은 225 그룹에서 선택 없음과 짝지은 차이)

| 조건 (코드 이름) | 실험 | EPDMS | Δ vs 선택 없음 | 95% CI | p (log perm) | p (그룹 Wilcoxon) | 그룹 +/−/= |
|---|---|---:|---:|---|---:|---:|---|
| 선택 없음 (`normal`) | 26 | 0.2287 | – | – | – | – | – |
| rank-sum (`ranksum`) | 26 | 0.2482 | +0.0196 | [+0.0028, +0.0372] | 0.094 | 0.082 | 92/73/60 |
| max log-lik (`max_loglik`) | 26 | 0.2486 | +0.0199 | [+0.0080, +0.0325] | 0.0016 | 0.107 | 68/56/101 |
| 충돌 제약만 (`collision_only`) | 29 | 0.2407 | +0.0121 | [+0.0054, +0.0198] | 0.0024 | 5.8e-6 | 33/6/186 |
| 주행 가능 영역 제약만 (`dac_only`) | 29 | 0.3154 | +0.0868 | [+0.0664, +0.1059] | 5e-5 (하한) | 4.4e-20 | 117/6/102 |
| 안전 필터만 (`filter_only` = `collision_dac`) | 27 | 0.3333 | +0.1046 | [+0.0845, +0.1251] | 5e-5 (하한) | 1.3e-21 | 128/9/88 |
| 안전 필터 + 무작위 (`filter_random`, seed 0/1/2 평균) | 27 | 0.3279 | +0.0992 | [+0.0790, +0.1188] | 5e-5 (하한) | 8.0e-14 | 154/53/18 |
| 안전 필터 + rank-sum (`F1`, 사전 등록 주 필터) | 26 | 0.3470 | +0.1183 | [+0.0956, +0.1400] | 5e-5 (하한) | 1.5e-15 | 147/51/27 |
| 안전 필터 + max log-lik (`F1_maxll`) | 26 | 0.3503 | +0.1216 | [+0.1007, +0.1423] | 5e-5 (하한) | 5.1e-19 | 141/33/51 |
| oracle, 샘플 16개 중 (`oracle16`)† | 30 | 0.4034 | +0.1747 | [+0.1474, +0.1999] | 5e-5 (하한) | 3.3e-25 | 180/22/23 |
| oracle, 후보 17개 중 (`oracle17`)† | 30 | 0.4053 | +0.1767 | [+0.1515, +0.2012] | 5e-5 (하한) | 1.4e-25 | 180/18/27 |

† 채점에 쓰이는 미래를 보는 분석용 상한이며 배포 방법이 아닙니다. token별 최고점 선택이라 2단계 가중치와 two-frame comfort를 고려하지 않는 근사 상한입니다(그래서 일부 지표에서 oracle16 > oracle17).

출처: `outputs/navhard_full_validation/navhard_full_comparison.json` → `pooled.rules.<rule>.epdms`, `groups_better_worse_tied`;
`outputs/safety_filter_ablation/ablation.json` → `full.contrasts['filter_only - normal' | 'filter_random - normal']`;
`outputs/safety_filter_components/components.json` → `full.contrasts['collision_only - no_filter' | 'dac_only - no_filter']`;
`outputs/candidate_oracle/oracle_comparison.json` → `full.contrasts['oracle17 - no_selection' | 'oracle16 - no_selection']`.
같은 내용의 CSV/TeX: `paper_navhard_table.csv`, `paper_navhard_table.tex`, 그림용 `figure_navhard_methods.csv`.
`F1`은 실험 27 `filter_ranksum - normal`과 수치가 같고(+0.11831, 같은 CI·p), `F1_maxll`은 `filter_maxll - normal`과 같습니다.

보조 조건(표 밖, 실험 27): 생존 후보가 없을 때 자연 계획으로 대체하는 `filter_ranksum_natfb` 0.3503, `filter_maxll_natfb` 0.3497.
이들의 "vs 선택 없음" CI·p는 **보고되지 않았습니다**(JSON에는 `− filter_only`, `filter_* − *_natfb` 대비만 있음).

### 1.1 세부 지표 (전체 225 그룹, 위반율 %; `RESULTS.md` 실험 26·27·29·30 표와 각 JSON `full.means`)

| 조건 | 1단계 / 2단계 점수 | 충돌 s1 / s2 | DAC 위반 s1 / s2 | 진행도 s1 / s2 | 2단계 history comfort |
|---|---|---|---|---|---:|
| 선택 없음 | 0.738 / 0.311 | 1.56 / 22.18 | 16.67 / 41.01 | 0.832 / 0.844 | 0.952 |
| rank-sum | 0.773 / 0.323 | 1.33 / 20.54 | 12.67 / 40.61 | 0.825 / 0.843 | 0.948 |
| max log-lik | 0.769 / 0.317 | 1.11 / 21.16 | 13.56 / 41.65 | 0.828 / 0.842 | 0.948 |
| 충돌 제약만 | 0.740 / 0.328 | 0.89 / 16.35 | 16.89 / 40.47 | 0.832 / 0.843 | 0.932 |
| DA 제약만 | 0.827 / 0.384 | 2.22 / 21.30 | 5.56 / 25.47 | 0.829 / 0.840 | 0.944 |
| 안전 필터만 | 0.834 / 0.403 | 1.33 / 17.62 | 5.56 / 27.11 | 0.832 / 0.840 | 0.937 |
| 필터 + 무작위 | 0.819 / 0.403 | 1.56 / 17.76 | 5.56 / 26.63 | 0.829 / 0.840 | 0.899 |
| F1 | 0.830 / 0.423 | 1.33 / 16.31 | 5.56 / 26.35 | 0.824 / 0.830 | 0.937 |
| F1 + max log-lik | 0.835 / 0.423 | 1.11 / 16.42 | 5.56 / 26.75 | 0.826 / 0.831 | 0.937 |
| oracle17 | 0.867 / 0.468 | 0.44 / 16.92 | 5.78 / 28.71 | 0.854 / 0.866 | 0.934 |

## 2. 두 번째 절반 사전 등록 재현 (실험 26)

사전 등록: `outputs/navhard_full_validation/PREREGISTRATION.md`, 커밋 **96a595c** (2026-10-02 14:08, 두 번째 절반 디코딩 전).
결과 커밋 **ad7991c** (17:56). 분석 스크립트 `scripts/analyze_navhard_full.py`는 사전 등록 커밋에 들어 있었고, 이후 변경은 표 출력
서식 한 줄(`+ 0.0`으로 −0.00 표시 제거)뿐입니다(`git diff 96a595c ad7991c`).
판정 규칙(각 α = 0.05): **Δ > 0 이고 bootstrap CI 하한 > 0 이며 log permutation p < 0.05**. 판정 대상은 H1(F1), H2(max_loglik) 두 개뿐이고
rank-sum, F1_maxll은 "기술적 보고"로 미리 지정되었습니다.

두 번째 절반: 40 log, 120 그룹(1단계 240 + 합성 2,760, 디코딩 3,000/3,000 성공).
출처: `navhard_full_comparison.json` → `half2.rules.<rule>.epdms`.

| 규칙 | 두 번째 절반 Δ | 95% CI | p (log perm) | p (Wilcoxon) | 그룹 +/−/= | 첫 절반 Δ (CI 안?) | 사전 등록 판정 |
|---|---:|---|---:|---:|---|---|---|
| F1 (H1) | +0.1121 | [+0.0819, +0.1402] | 5e-5 (하한) | 3.2e-10 | 77/24/19 | +0.1254 (안) | **통과 — 재현** |
| max log-lik (H2) | +0.0097 | [−0.0037, +0.0235] | 0.080 | 0.52 | 32/29/59 | +0.0317 (밖) | **실패 — 재현 안 됨** (CI 하한 < 0, p ≥ 0.05) |
| rank-sum (판정 대상 아님) | +0.0203 | [−0.00002, +0.0402] | 0.108 | 0.075 | 51/35/34 | +0.0187 (안) | 판정 대상 아님. 같은 기준을 적용하면 기준 미충족(CI 하한 −1.5e-5, p 0.11) |
| F1 + max log-lik (판정 대상 아님) | +0.1116 | [+0.0846, +0.1364] | 5e-5 (하한) | 3.1e-11 | 73/15/32 | +0.1331 (안) | 판정 대상 아님. 같은 기준이면 충족 |

선택 없음 EPDMS: 첫 절반 0.2393, 두 번째 절반 0.2194.
첫 절반(실험 25 디코딩을 실험 26 절차로 재채점, 36 log/105 그룹)의 값: rank-sum +0.0187 [−0.0072, +0.0503] p 0.35/0.49; max log-lik +0.0317
[+0.0117, +0.0572] p 0.0072/0.12; F1 +0.1254 [+0.0900, +0.1638] p 2.5e-4/5.1e-7 (`half1.rules`). 실험 25 원보고서(`navhard_eval/NAVHARD_CLOSED_LOOP.md`,
`navhard_comparison.json`)는 같은 Δ·CI를 보고하지만 p 값은 보고하지 않았습니다.

해석 메모: 전체 225 그룹 max log-lik +0.0199 [+0.0080, +0.0325], p 0.0016은 효과를 처음 발견한 첫 절반을 포함하므로 독립 검증이 아닙니다(RESULTS.md 결론 2와 동일).
"확신도만으로 고르는 규칙은 closed-loop에서 작고 불안정한 이득(+0.02)"이 원 보고의 결론입니다.

## 3. 안전 필터 분해 (실험 27 프로토콜 19c0130 → 결과 f98cb10; 실험 29 프로토콜 0cffc15 → 결과 0ab33a4)

### 3.1 필터 vs 선택기 (실험 27, 225 그룹; `ablation.json` → `full.contrasts`)

| 대비 | Δ EPDMS | 95% CI | p (perm / Wilcoxon) | 그룹 +/−/= | 첫 절반 / 두 번째 절반 |
|---|---:|---|---|---|---|
| F1 총 이득 = filter_ranksum − normal | +0.1183 | [+0.0956, +0.1400] | 5e-5 / 1.5e-15 | 147/51/27 | +0.1254 / +0.1121 |
| 안전 필터만 = filter_only − normal | +0.1046 | [+0.0845, +0.1251] | 5e-5 / 1.3e-21 | 128/9/88 | +0.1129 / +0.0974 |
| rank 한계 기여 = filter_ranksum − filter_only | +0.0137 | [−0.0020, +0.0296] | 0.20 / 0.17 | 99/88/38 | +0.0125 / +0.0147 |
| maxlog 한계 기여 = filter_maxll − filter_only | +0.0170 | [+0.0029, +0.0313] | 0.0018 / 0.13 | 79/74/72 | +0.0202 / +0.0142 |
| filter_random − filter_only | −0.0054 | [−0.0215, +0.0110] | 0.80 / 0.11 | 92/110/23 | −0.0039 / −0.0068 |
| filter_ranksum − filter_random | +0.0191 | [+0.0069, +0.0305] | 0.077 / 2.5e-4 | 130/75/20 | |
| filter_maxll − filter_random | +0.0224 | [+0.0107, +0.0347] | 0.0059 / 2.2e-4 | 122/81/22 | |
| filter_ranksum_natfb − filter_only | +0.0170 | [+0.0031, +0.0316] | 0.0024 / 0.077 | 100/80/45 | |
| filter_ranksum − filter_ranksum_natfb | −0.0033 | [−0.0108, +0.0021] | 0.20 / 0.44 | 12/17/196 | |

**"안전 필터 몫 89%" 도출** (recomputed: `_parts/scripts/navhard_verify.py`):
- 식: (filter_only − normal) / (filter_ranksum − normal) = 0.104610 / 0.118313 = **0.8842 → 88.4%**.
- 보고서의 "89%"는 반올림된 값으로 나눈 0.105 / 0.118 = 0.8898에서 나옵니다. 정확한 값은 88.4%(반올림 88%)입니다.
- 보수: rank 몫 = 0.013703 / 0.118313 = 11.6% (보고서 "11%"; 정확값 반올림은 12%). 두 항의 합은 정확히 F1 총 이득입니다(잔차 0).
- 같은 방식의 max log-lik 쪽: 필터 몫 0.104610 / 0.121644 = 86.0%, maxlog 몫 0.017035 / 0.121644 = 14.0% (보고서 "14%"와 일치).
- 주의: 이것은 두 조건 순서의 단순 분할(필터 먼저)이며 Shapley가 아닙니다. 순서를 바꾼 "선택기만" 경로(rank-sum only − normal = +0.0196)를 쓰면 필터 몫은 (0.1183 − 0.0196)/0.1183 = 83.4%가 됩니다(참고용 파생 비율, 보고서에 없음).

### 3.2 충돌 vs 주행 가능 영역 제약 (실험 29, `components.json` → `full.contrasts`)

| 대비 | Δ EPDMS | 95% CI | p (perm / Wilcoxon) | 그룹 +/−/= | 첫 / 두 번째 절반 |
|---|---:|---|---|---|---|
| collision_only − no_filter | +0.0121 | [+0.0054, +0.0198] | 0.0024 / 5.8e-6 | 33/6/186 | +0.0090 / +0.0147 |
| dac_only − no_filter | +0.0868 | [+0.0664, +0.1059] | 5e-5 / 4.4e-20 | 117/6/102 | +0.0923 / +0.0820 |
| collision_dac − no_filter | +0.1046 | [+0.0845, +0.1251] | 5e-5 / 1.3e-21 | 128/9/88 | +0.1129 / +0.0974 |
| collision_dac − collision_only (DA 추가분) | +0.0926 | [+0.0708, +0.1138] | 5e-5 / 1.0e-19 | 115/7/103 | +0.1039 / +0.0826 |
| collision_dac − dac_only (충돌 추가분) | +0.0178 | [+0.0100, +0.0276] | 2e-4 / 7.0e-7 | 37/6/182 | +0.0206 / +0.0154 |

**"DA 기여 86%" 도출** (2인 Shapley, 두 순서 평균; recomputed: `navhard_verify.py`):
- Shapley_DA = ½[(dac_only − none) + (both − collision_only)] = ½(0.086848 + 0.092558) = **0.089673**
- Shapley_collision = ½[(collision_only − none) + (both − dac_only)] = ½(0.012051 + 0.017821) = **0.014936**
- 분모 = both − none = 0.104610 (안전 필터만의 이득; F1 총 이득이 아님). DA 몫 = 0.089673 / 0.104610 = **85.7% → 86%** (재현됨), 충돌 몫 14.3%.
- 상호작용 = both − dac_only − collision_only (모두 none 기준) = 0.005770 (보고서 "+0.006"); Shapley 값은 상호작용을 반씩 이미 포함하므로 0.0897 + 0.0149 = 0.1046이고, 상호작용을 따로 더하면 안 됩니다.
- Shapley 값 자체에는 CI·p가 **보고되지 않았습니다**(구성 대비들의 CI만 있음).
- 다른 분모를 쓰면: DA 단독 한계 0.0868 / 0.1046 = 83.0%, DA Shapley / F1 총 이득 = 0.0897 / 0.1183 = 75.8%(참고용 파생 비율).

### 3.3 진행도와 계획 길이 (실험 29)

| 항목 | 값 | 출처 |
|---|---|---|
| 진행도 변화, 안전 필터만 − 선택 없음 | s1 −0.0006 [−0.0026, +0.0014], s2 −0.0046 [−0.0124, +0.0030] (둘 다 CI가 0 포함) | `components.json` `full.contrasts['collision_dac - no_filter']` `stage1/2:ego_progress` |
| 진행도 변화, F1 − 선택 없음 | s1 −0.0081 [−0.0146, −0.0021], s2 −0.0148 [−0.0242, −0.0056] | `navhard_full_comparison.md` pooled 세부표 / `ablation.json` `filter_ranksum - normal` |
| 진행도 변화, 선택기 추가분 (F1 − 필터만) | s1 −0.0075 [−0.0141, −0.0020], s2 −0.0102 [−0.0179, −0.0028] | `ablation.json` `filter_ranksum - filter_only` |
| token 단위 Shapley, 곱셈 항 몫 (안전 필터만) | s1 0.0976/0.0953 = 102.4%, s2 0.1139/0.1145 = 99.5% | `score_decomposition.json` `conds.collision_dac` |
| 곱셈 항 몫 (F1 / F1+maxlog) | s1 103.6% / 101.5%, s2 100.1% / 98.7% | 같은 파일 `full_ranksum`, `full_maxll` |
| 진행도(EP) 항 | 필터만 s1 −0.0001 [−0.0004, +0.0002], s2 −0.0021 [−0.0027, −0.0017]; F1 s1 −0.0025, s2 −0.0055 | 같은 파일 `ego_progress` |
| 4초 계획 끝점 도달 거리(전체 평균) | 선택 없음 19.13 m → 필터만 18.54 m (−0.59 m, −3.1%), F1 18.17 m (−0.96 m), F1+maxlog 18.36 m | `plan_reach.json` `mean_plan_reach_4s_m` |
| 바뀐 장면에서의 도달 거리 변화 | 충돌만 −2.65 m (504 token), DA만 −2.47 m (1,070), 필터만 −2.88 m (1,213), F1 −1.17 m (4,852), F1+maxlog −1.75 m (2,610) | `plan_reach.json` `mean_change_in_changed_m`, `n_changed` |

계획 도달 거리에는 CI·검정이 **보고되지 않았고**, 이를 만든 스크립트가 저장소에서 확인되지 않습니다(conflicts 참고).
"이득의 99–104%가 곱셈 항"은 세 필터 조건(필터만, F1, F1+maxlog)의 6개 값 98.7–103.6%를 반올림한 범위입니다. 충돌 제약만은 83%/90%로 이 범위 밖입니다.

## 4. oracle 상한 (실험 30, 프로토콜 0cffc15 → 결과 86087e0)

| 항목 | 값 | 출처 |
|---|---|---|
| oracle17 EPDMS | 0.4053 (첫 절반 0.4478 / 두 번째 0.3681) | `oracle_comparison.json` `full.means.oracle17.epdms` |
| oracle16 EPDMS | 0.4034 | `full.means.oracle16.epdms` |
| oracle17 − 선택 없음 | +0.1767 [+0.1515, +0.2012], p 5e-5 / 1.4e-25, 180/18/27 | `full.contrasts['oracle17 - no_selection']` |
| 남은 여지 oracle17 − F1+maxlog | **+0.0550 [+0.0355, +0.0749]**, p 5e-5 / 2.1e-12, 153/36/36 | `full.contrasts['oracle17 - filter_maxll']` |
| 남은 여지 oracle17 − F1 | +0.0584 [+0.0383, +0.0786], p 1e-4 / 1.6e-11, 154/42/29 | `full.contrasts['oracle17 - filter_ranksum']` |
| 17개 후보가 모두 0점인 token | **1,867 / 5,912 = 31.6%** (첫 절반 915/2,912 = 31.4%, 두 번째 952/3,000 = 31.7%) | recomputed: `navhard_verify.py` ← `candidate_oracle/half*/oracle/token_scores.npy` |
| 자연 후보가 0점인 token | 52.5% | 같음 |
| token 평균 점수: 자연 / 17개 중 최고 / 16개 중 최고 | 0.369 / 0.556 / 0.555 | 같음 (보고서 0.37 / 0.56) |
| oracle17이 자연 후보를 고른 token | 43.9% | 같음 |

**"배포 방법이 oracle 이득의 69%" 도출** (recomputed: `navhard_verify.py`):
- 식: (F1+maxlog − 선택 없음) / (oracle17 − 선택 없음) = 0.121644 / 0.176669 = **0.6885 → 69%** (재현됨). 남은 여지 0.055025 = 0.176669 − 0.121644(잔차 0).
- 분자에 쓴 "최선 방법"은 F1+maxlog이며, 이것은 사전 등록에서 판정 대상이 아닌 기술적 조건이었고 전체 225 그룹을 보고 "최선"으로 고른 것입니다.
  사전 등록 주 필터 F1을 쓰면 0.118313 / 0.176669 = **67.0%**, 안전 필터만이면 0.104610 / 0.176669 = 59.2%입니다.
- 절반별: F1+maxlog 첫 절반 0.1331 / 0.2086 = 63.8%, 두 번째 0.1116 / 0.1488 = 75.0% (파생 비율, CI 없음). 69%에는 CI가 **보고되지 않았습니다**.
- 생성 한계: oracle 자체 0.405, token의 약 31–32%는 어떤 후보도 0점 초과가 아님 → 보고서 결론 "주된 병목은 후보 생성".
  주의: 안전 필터의 "모두 걸림" 비율(현재 frame 플래그 기준) 27.5%와 oracle의 "모두 0점"(채점 미래 기준) 31.6%는 서로 다른 양입니다.

## 5. 논문 문장 후보 (수치는 위 표 그대로)

- 전체 navhard(76 log, 225 그룹)에서 현재 frame 안전 필터는 EPDMS를 0.229 → 0.333으로 올렸고(+0.105 [+0.085, +0.125], log permutation p < 1e-4, 128/9/88 그룹), rank-sum 선택기를 더한 F1은 0.347(+0.118 [+0.096, +0.140])입니다.
- F1 이득의 88%(0.1046/0.1183)는 필터만으로 얻어지고, rank-sum 선택기의 추가분 +0.014 [−0.002, +0.030]은 유의하지 않습니다(p 0.20).
- 필터 이득의 86%(2인 Shapley)는 주행 가능 영역 제약에서 옵니다.
- 사전 등록한 두 번째 절반(40 log, 120 그룹)에서 F1은 재현되었고(+0.112 [+0.082, +0.140], p < 1e-4), max log-lik은 재현되지 않았습니다(+0.010 [−0.004, +0.024], p 0.08).
- oracle17은 0.405이며 F1+max log-lik은 oracle 이득의 69%(F1 기준 67%)를 얻었습니다. token의 31.6%는 17개 후보가 모두 0점입니다.
- 한계(모든 문장에 동반): 필터와 채점이 같은 PDM 규칙을 쓰고, 2단계 pseudo closed-loop이며 연속 closed-loop이 아니고, 다중 비교 보정이 없습니다.
