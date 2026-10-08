# AutoVLA 기전 실험 — 논문용 정량 근거 (섹션 A–F)

모델: AutoVLA (Qwen2.5-VL-3B backbone, action codebook 2,048 token, natural fast-thinking 경로). 벤치마크: NAVSIM/nuPlan navtest PoC 28 log, 장면 2,747개(arm N).
경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`, `S/` = `/root/VLA/autovla_misalignment_poc/scripts/`, `P/` = `/root/VLA/paper_quantitative_package/`.
표기 규칙
- `[a, b]`는 원 분석 스크립트가 계산한 **log 단위 cluster bootstrap 95% percentile CI**입니다(별도 표시 없으면 2,000회, seed 0, 재표본 단위 = navtest log). 이진 지표의 p는 **McNemar exact(binomial) 검정**, 연속 지표의 p는 **Wilcoxon signed-rank**(단위별 쌍대)입니다.
- "not reported" = 원 결과에 없고, 원 스크립트의 같은 방법으로 재계산할 수 없거나 재계산하지 않은 값.
- "(recomputed: …)" = 이 패키지에서 기존 값으로부터 산술/개수만 다시 구한 값. 새 통계 검정은 하나도 추가하지 않았습니다.
- 증폭(amplification) 정의(실험 6–11, 32 공통, 실행 전 고정): **A− (P/R/A coarse-action 5 s 판정 실패) AND FDE(5 s) > 3.0 m**. recovery = P/R/A 5 s 판정 통과(A+). 출처: `S/analyze_equal_distance.py` L5–8, `S/analyze_action_history.py` `AMP_FDE = 3.0`.
- 실험 6–11, 32의 분석 단위: equal-distance set = 장면 208개(A− 52, A+ 156) / (장면, perturbation) 단위 1,408개(A− 365, A+ 1,043). A− 52 장면은 **16개 log**, A+ 156 장면은 26개 log(A+ log가 A− log를 포함, 전체 26 log)에서 옵니다 (recomputed: `P/_parts/scripts/build_autovla_csvs.py`, `O/equal_distance_perturbation/records.jsonl`의 distinct `log` 개수). 따라서 A− 관련 CI의 유효 cluster 수는 16입니다.

---

## A. mismatch ≠ failure (실험 4 natural_fast_mechanism, 실험 5 first_mismatch_causal)

### A.1 natural 모집단과 A+/A− (실험 4)
출처: `O/natural_fast_mechanism/summary.json` 키 `N.n`, `N.n_Aminus`, `N.n_Aminus_step0_token_wrong`, `N.n_Aplus_step0_token_wrong`; 보고서 `O/natural_fast_mechanism/NATURAL_FAST_MECHANISM.md` §1.

| 항목 | 값 | 출처 키 |
|---|---|---|
| natural N (arm N, full extraction) | 2,747 장면 (28 log) | `N.n` |
| A− (궤적 coarse action 5 s 실패) | 52 / 2,747 = **1.89%** | `N.n_Aminus` |
| A+ | 2,695 / 2,747 = 98.11% | 2747 − 52 (recomputed) |
| A− 중 **step-0(첫) token 오답** | **17 / 52 = 32.7%** | `N.n_Aminus_step0_token_wrong` |
| A− 중 어떤 step이든 token 불일치 존재 | 52 / 52 = 100% | `O/first_mismatch_causal/summary.json` `n.A-` = 52 ("첫 불일치가 있는 장면 전부") |
| **step-0 token 오답인데 A+** | **216 / 2,695 = 8.0%** (A+ 대비); step-0 오답 233건 중 216건 = **92.7%**가 A+ | `N.n_Aplus_step0_token_wrong`; 233 = `decomposition.token_step0.n_fail` |
| A+ 중 어떤 step이든 token 불일치 존재 | 1,107 / 2,695 = 41.1% | `O/first_mismatch_causal/summary.json` `n.A+` |
| 재현 확인 run(gpu1 natural run, residual만 저장) | n 2,747, A− 46, A− 중 step-0 오답 18(39.1%), A+ 중 step-0 오답 207 | `G.*` 키 (다른 run, 다른 population — conflicts 참조) |

비율 계산은 (recomputed: 위 정수의 나눗셈). 결론: **token-ID 실패(step-0 오답 233건)와 궤적 실패(A− 52건)는 대부분 다른 샘플**(교집합 17건)입니다.

### A.2 첫 불일치(first mismatch)의 거리
출처: `O/natural_fast_mechanism/summary.json` `N.codebook_amplification.{A-, A+ with a token mismatch}`; `O/first_mismatch_causal/summary.json` `why.d_pred_gt`.

| 지표 | A− (n=52) | A+ with token mismatch (n=1,107) | 비고 |
|---|---:|---:|---|
| 오답 token–GT token 변위 거리, **중앙값** | **0.113 m** (p75 0.189) | **0.083 m** (p75 0.136) | `wrong_token_displacement_m.median` |
| 무작위 token 쌍 대비 백분위 | 0.5% | 0.3% | 무작위 쌍 중앙값 2.48 m, p10 0.64 m |
| GT 기준 오답 token의 NN 순위 중앙값 | 15 (순위≤10: 44%) | 8 (59%) | `nn_rank_of_wrong_token` |
| 첫 불일치 step의 pose 오차 | 0.103 m | 0.087 m | `pose_error_at_first_mismatch_m` (다른 거리 정의) |
| A− vs A+ 판별 AUROC (d_pred_gt) | 0.60 [0.52, 0.68] (n=1,159) | | `O/first_mismatch_causal/summary.json` `why.d_pred_gt` (AUROC CI: log bootstrap 1,000회 seed 2) |

### A.3 첫 불일치 거리 vs 최종 FDE 상관
출처: `N.codebook_amplification.<group>.spearman_first_displacement_vs_fde` (스크립트 `S/natural_fast_mechanism.py` `_spearman_ci`: log cluster bootstrap 1,000회, `random.Random(3)`).

| 그룹 | Spearman ρ | 95% CI | p | n |
|---|---:|---|---|---:|
| A− | **+0.26** | [−0.02, +0.49] | not reported (스크립트가 p를 계산하지 않음) | 52 |
| A+ with token mismatch | +0.40 | [+0.34, +0.49] | not reported | 1,107 |
| step-0 오답(구 정의) | +0.31 | [+0.18, +0.44] | not reported | 233 |
| (참고) 첫 거리 vs ADE, A− | +0.43 | [+0.20, +0.62] | not reported | 52 |

→ A−에서 첫 편차 크기와 최종 FDE의 상관은 약하고 CI가 0을 포함합니다.

### A.4 하류 증폭 (A− vs A+)
출처: `N.codebook_amplification`, 보고서 §3.5–3.6 표.

| 지표 | A− | A+ with mismatch | 비고 |
|---|---:|---:|---|
| FDE / 첫 step 오차 (중앙값 비) | **58.0×** | **6.5×** | `fde_over_first_step_error_median` |
| ADE / FDE (중앙값, 저장된 natural run) | 2.27 / 6.63 m | 0.28 / 0.66 m | `ade_m`, `fde_m` = **median** (`S/natural_fast_mechanism.py` L422) |
| 오답 전파: P(오답 \| 깨끗한 prefix) → P(오답 \| 직전 오답) | 5.3% → 72.9% (13.8×), 그룹 풀링 | | 보고서 §3.5 "전파(전체)" 행; `verdicts.error_amplification.propagation_ratio` = 13.83 |
| 첫 불일치 이후 downstream token error (재생성 original) | 94.6% [88.4, 99.1] | 72.4% [68.6, 74.7] (step-matched) / 70.5% (all) | `O/first_mismatch_causal/summary.json` `A-.original.downstream_err`, `A+ (step-matched).original.downstream_err` |
| pose 오차 곡선 평균 (step 0→9, m) | 0.08, 0.24, 0.52, 0.99, 1.67, 2.56, 3.75, 5.18, 6.71, 8.50 | 0.05, 0.10, 0.17, 0.27, 0.40, 0.56, 0.75, 0.97, 1.21, 1.48 | `pose_error_curve_mean` |

### A.5 첫 불일치 timestep(t*) 분포
출처: `O/first_mismatch_causal/summary.json` `t_star_hist`.

| t* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A− (n=52) | 17 | 22 | 7 | 2 | 1 | 2 | 0 | 1 | 0 | 0 |
| A+ (n=1,107) | 216 | 249 | 175 | 96 | 79 | 74 | 55 | 55 | 56 | 52 |

A− 중앙값 t* = 1, A+ = 2; AUROC(t*) 0.31 [0.26, 0.37] (`why.t_star`). A−는 이르게 이탈하므로 A+ 비교는 t* 분포를 맞춘 가중 비교(step-matched, 유효 n = 944)도 함께 보고됩니다.

### A.6 첫 불일치 교정 (실험 5)
설계: t*에서 token 하나만 (a) original, (b) GT token, (c) GT의 최근접(NN) token으로 강제, 이후 자유 생성. GPU RTX 5090, T 0.01, seed 0. 출처: `O/first_mismatch_causal/summary.json` 키 `A-.{original,gt,nn}`, `A+ (step-matched).*`, `<group>.contrasts.*`, `interaction.*`; 보고서 `O/first_mismatch_causal/FIRST_MISMATCH_CAUSAL.md`. 스크립트 `S/first_mismatch_causal.py`, `S/analyze_first_mismatch_causal.py`(REPS 2000, seed 0, McNemar binomtest, Wilcoxon).

| 지표 | A− original | A− GT 1-token | A− NN token | A+ step-matched original | A+ GT | A+ NN |
|---|---:|---:|---:|---:|---:|---:|
| A+ recovery (5 s) | 17.3% [7.7, 33.3] (9/52) | **73.1% [63.6, 84.4]** (38/52) | 69.2% [57.1, 80.7] (36/52) | 99.0% [98.2, 99.6] | 98.7% [97.3, 99.6] | 95.9% [94.1, 98.2] |
| downstream token error | 94.6% [88.4, 99.1] | 43.5% [30.5, 54.8] | 85.9% [73.7, 93.0] | 72.4% [68.6, 74.7] | 29.9% [25.9, 33.7] | 62.3% [59.5, 64.8] |
| ADE 5 s (m) | 2.91 [2.34, 3.28] | 0.68 [0.39, 0.94] | 1.74 [1.07, 2.47] | 0.83 [0.69, 0.96] | 0.27 [0.21, 0.33] | 0.71 [0.59, 0.82] |
| FDE 5 s (m) | 8.18 [6.39, 9.48] | **2.09 [1.15, 3.01]** | 4.93 [3.05, 6.58] | 2.01 [1.66, 2.28] | 0.75 [0.55, 0.92] | 1.83 [1.47, 2.12] |

정수 분자는 (recomputed: 비율 × 52). 쌍대 대비(A−, n = 52 장면):

| 대비 | ΔA+ recovery | Δdownstream error | ΔADE (m) | ΔFDE (m) |
|---|---|---|---|---|
| GT − original | **+55.8%p [+35.4, +69.2]**, McNemar p = 4.2e-07 (32↑/3↓) | −51.0%p [−66.2, −36.6], p = 4.7e-08 | −2.23 [−2.65, −1.64], p = 1.0e-08 | **−6.09 [−7.38, −4.28]**, p = 2.2e-08 |
| NN − original | +51.9%p [+31.0, +64.4], p = 1.4e-06 (30↑/3↓) | −8.7%p [−21.9, −1.5], p = 0.0042 | −1.17 [−1.76, −0.27], p = 3.2e-05 | −3.26 [−4.61, −1.30], p = 3.8e-05 |
| GT − NN | +3.8%p [−8.7, +18.2], p = 0.80 (9↑/7↓) | −42.3%p [−51.4, −30.8], p = 2.7e-07 | −1.06 [−1.87, −0.45], p = 1.5e-04 | −2.84 [−4.55, −1.27], p = 5.6e-04 |

A+ (step-matched)에서 GT − original: ΔA+ −0.3%p [−0.9, +0.2], p = 0.81; ΔFDE −1.26 m [−1.51, −0.97], p = 4.3e-96. A− vs A+ 교호작용(GT 교정 효과 차이, step-matched): ΔA+ 차이 **+56.1%p [+34.0, +69.9]**, ΔFDE 차이 −4.84 m [−6.16, −2.98], bootstrap p ≈ 0 (`interaction."vs A+ step-matched"`, `p_boot`, seed 1).
해석에 필요한 점: NN token(GT와 0.035 m, original은 0.084 m; `nn_fragility`)도 A+ recovery를 GT와 통계적으로 구분되지 않게 회복시키지만(+3.8%p, p = 0.80), downstream token error는 GT보다 42%p 높습니다 → 궤적 회복에 정확한 token identity는 필요 없습니다(실험 11과 일관).

---

## B. Equal-distance perturbation (실험 6, 핵심)

설계: A− 52 장면 + A− 장면당 **같은 t***의 A+ 3개(156). t*에 원래 token 또는 GT로부터 **같은 거리**(원래 오답의 거리와 상대오차 허용 0.1/0.2/0.35 단계, 절대 5 mm, 최소 5개)의 대안 token을 강제. original_reseed = 같은 token, 다른 sampling seed(noise floor). RTX 5090, T 0.01, seed 0. 출처: `O/equal_distance_perturbation/summary.json`(키 `n_scenes`, `alts_per_scene`, `alt_rel_err_median`, `alt_abs_dist_diff_median_m`, `table`, `Q1`–`Q5`), 보고서 `EQUAL_DISTANCE.md`; strict 부분집합(전체 codebook fill로 들어온 대안 제외)은 `EQUAL_DISTANCE_strict.md`. **strict의 summary json은 출력 디렉토리에 없어** 원 스크립트를 사본에서 `STRICT=1`로 재실행해 얻었습니다 (recomputed: `P/_parts/scripts/rerun_equal_distance_analysis.sh` → `P/_parts/recomputed/equal_distance_summary_strict_rerun.json`; 재실행 MD는 원 MD와 Q4 AUROC 소수 둘째 자리만 다름 — REPRODUCTION.md에 기록된 solver 수치차와 동일).

### B.1 설계 수치

| 항목 | full set | strict |
|---|---|---|
| 장면 | 208 (A− 52 / A+ 156), log 26 (A− 16) | 동일 장면; 대안이 남은 장면 A− 51 / A+ 149 |
| perturbation 단위(original + 대안) | A− 52 + 313 = **365**, A+ 156 + 887 = **1,043** (총 1,408) | 대안 A− 292 / A+ 771 |
| 장면당 대안 수 | A− 6.0 / A+ 5.7 | (strict MD 머리말은 full 값을 그대로 인쇄 — conflicts) |
| perturbation 거리 (A− 대안) | **중앙값 0.129 m, 평균 0.179 m** [0.139, 0.225] | 중앙값 0.136 m |
| perturbation 거리 (A+ 대안) | 중앙값 0.087 m, 평균 0.120 m [0.102, 0.140] | 중앙값 0.095 m |
| 원래 오답 token 거리 | A− 중앙값 0.113 / 평균 0.161 m; A+ 0.077 / 0.109 m | |
| **거리 매칭 오차** | 대안–원래 거리 상대차 **중앙값 7.9%**, 절대차 **중앙값 8.8 mm** | |
| 허용오차 단계 사용(장면) | 0.1: 69, 0.2: 41, 0.35: 98 | |

출처 키: `table.<g>.<cond>.dist.{median,mean,ci95}`, `alt_rel_err_median` = 0.0792, `alt_abs_dist_diff_median_m` = 0.00883, `rel_used`.

### B.2 결과 (그룹 × 조건)

| 그룹 | 조건 | n | recovery | amplification | FDE (m) |
|---|---|---:|---:|---:|---:|
| A− | original | 52 | 17.3% [8, 33] | 67.3% [51, 80] | 8.18 [6.39, 9.48] |
| A− | original_reseed | 52 | 15.4% [7, 30] | 69.2% [54, 81] | 8.26 [6.55, 9.51] |
| A− | GT token | 52 | 73.1% [64, 84] | 17.3% [9, 23] | 2.09 [1.15, 3.01] |
| A− | **대안 (all)** | **313** | **49.2% [41, 56]** | **45.0% [37, 55]** | **6.19 [5.04, 7.12]** |
| A− | 대안 (strict) | 292 | 48.3% [40, 55] | 46.2% [39, 56] | 6.39 [5.26, 7.36] |
| A− | 대안, 원래 token과 같은 방향 sector | 39 | 38.5% | 53.8% [39, 64] | 6.47 |
| A− | 대안, 다른 방향 | 274 | 50.7% | 43.8% [36, 54] | 6.15 |
| A+ | original | 156 | 98.7% [96, 100] | 0.0% | 1.80 [1.30, 2.18] |
| A+ | **대안 (all)** | **887** | **93.6% [91, 96]** | **4.8% [3, 7]** | **2.35 [1.86, 2.72]** |
| A+ | 대안 (strict) | 771 | 93.1% [91, 96] | 5.1% [3, 7] | 2.48 [1.95, 2.86] |

A− 대안의 방향 sector별 증폭은 34.9–60.0% (sector당 n 33–48), A+는 1.9–8.4% (`table.<g>."alt: <sector>"`).
주의: A− 대안과 A+ 대안은 각자 자기 원래 오차 거리에 맞춰져 있어 **그룹 간 거리가 같지 않습니다**(중앙값 0.129 vs 0.087 m). 그룹 간 직접 비교는 아래 B.4의 거리 5분위 표로 보완합니다.

### B.3 장면 안에서 결과가 갈리는가, noise floor

| 지표 (A−, 장면 단위 n = 52) | full | strict (n = 51) |
|---|---|---|
| **같은 거리 대안들이 recovery와 amplification으로 갈린 장면** | **71.2% (37/52) [60.0, 82.8]** | 64.7% (33/51) [55.6, 72.4] |
| 대안 전부 recovery | 15.4% (8/52) [7.7, 27.0] | 19.6% |
| 대안 전부 amplification | 9.6% (5/52) [2.7, 16.3] | 11.8% |
| 장면 내 대안 FDE SD (장면 평균) | **2.77 m [2.37, 3.25]** | 2.77 m [2.40, 3.25] |
| 장면 내 대안 FDE 범위 (장면 중앙값) | 5.25 m [4.53, 10.05] | 5.26 m |
| **noise floor: 같은 token reseed의 \|ΔFDE\| (장면 평균)** | **0.11 m [0.01, 0.27]** | 0.11 m |
| **noise floor: 같은 token reseed의 결과 뒤집힘** | **1.9% (1/52) [0.0, 6.5]** | 2.0% |
| (A+) 갈린 장면 / reseed 뒤집힘 | 19.9% [13.0, 25.6] / 0.6% (1/156) | 18.8% (28/149) / 0.7% |

출처: `Q2.<g>.scenes_with_mixed_alt_outcomes` 등; 정수 분자는 (recomputed: 비율 × n). "seed-noise FDE **SD**"는 원 분석이 계산하지 않았습니다(**not reported**); 보고된 것은 original vs original_reseed의 평균 절대차(0.11 m)입니다.

### B.4 거리와 결과: 상관, 분산 분해, 거리 5분위

상관 (`Q1`, log cluster bootstrap 2,000회, p는 스크립트가 계산하지 않음 → not reported):

| 그룹 | Spearman(거리, FDE) 풀링 | **장면 내 순위 상관** (scene-demeaned ranks) | n (조건 행) |
|---|---|---|---:|
| A− | +0.31 [+0.05, +0.56] | **+0.09 [−0.01, +0.19]** | 364 |
| A+ | +0.47 [+0.37, +0.56] | +0.10 [+0.06, +0.15] | 1,043 |
| ALL | +0.46 [+0.36, +0.56] | +0.09 [+0.05, +0.15] | 1,407 |

(A− n = 364: FDE 결측 1행 제외.) strict: A− 장면 내 +0.10 [−0.01, +0.22].

FDE 분산 분해 (`Q2.<g>.share_*`, CI 없음; 장면 간 + 장면 내 = 100%, 거리 구간 간 비율은 별도 분해):

| 그룹 | 거리 5분위 구간 간 | 장면 간 | 장면 내(같은 장면, 다른 대안) |
|---|---:|---:|---:|
| A− | 17.0% | 53.8% | **46.2%** |
| A+ | 11.2% | 35.5% | 64.5% |
| ALL | 13.1% | 54.9% | 45.1% |

거리 5분위(original + 대안, `Q2.<g>.buckets`, CI 없음):

| 분위 | A− 거리 범위 (m) | A− n | A− 증폭 | A− FDE 평균 (SD) | A+ 거리 범위 (m) | A+ n | A+ 증폭 | A+ FDE 평균 (SD) |
|---|---|---:|---:|---|---|---:|---:|---|
| 0 | 0.027–0.068 | 73 | 50.7% | 5.74 (4.84) | 0.014–0.058 | 209 | 1.4% | 1.18 (2.09) |
| 1 | 0.068–0.105 | 73 | 31.5% | 4.37 (4.07) | 0.058–0.076 | 208 | 5.8% | 1.66 (3.23) |
| 2 | 0.106–0.147 | 72 | 29.2% | 4.37 (3.54) | 0.076–0.107 | 209 | 2.4% | 1.65 (1.76) |
| 3 | 0.148–0.256 | 73 | 53.4% | 7.44 (5.78) | 0.108–0.165 | 209 | 4.3% | 2.71 (3.20) |
| 4 | 0.261–0.669 | 73 | 75.3% | 10.45 (6.48) | 0.166–0.635 | 208 | 6.7% | 4.14 (4.02) |

무엇이 증폭을 예측하는가 (`Q4`, log GroupKFold 교차검증 AUROC, CI = log bootstrap 1,000회): A− 대안에서 거리만 0.61 [0.49, 0.79], 거리+방향 0.62 [0.56, 0.72], **장면(leave-one-out 장면 증폭률) 0.80 [0.70, 0.85]**, 장면+거리+방향 0.80 [0.73, 0.84] (재실행에서 둘째 자리 ±0.01 변동).

### B.5 "같은 크기의 오차만으로는 결과가 정해지지 않는다"를 지지하는 최소 수치
1. **장면 내 혼합**: A− 장면의 71.2% (37/52) [60.0, 82.8]에서 같은 거리(매칭 오차 중앙값 8.8 mm, 7.9%) 대안들이 recovery와 amplification으로 갈림 — 같은 token reseed의 결과 뒤집힘은 1.9% (1/52) [0, 6.5]에 불과. (strict에서도 64.7% [55.6, 72.4].)
2. **장면 내 FDE 산포 vs noise**: 같은 장면의 대안 FDE SD 2.77 m [2.37, 3.25] vs 같은 token reseed |ΔFDE| 0.11 m [0.01, 0.27] (약 25배; 서로 다른 통계량이므로 비율은 서술적).
3. **장면 내 거리 순위와 FDE 순위의 상관 ≈ 0**: A− +0.09 [−0.01, +0.19].
4. **분산의 46.2%가 장면 내**, 거리 구간 간 설명분은 17.0%.
5. **거리가 겹치는 구간에서 그룹별 결과가 전혀 다름**: 최소 거리 5분위(A− 0.027–0.068 m vs A+ 0.014–0.058 m) 증폭 50.7% vs 1.4% (CI 없음). 대안 전체 증폭 A− 45.0% [37, 55] vs A+ 4.8% [3, 7].
6. **예측력**: 장면 AUROC 0.80 > 거리 0.61.
반대 방향 근거(주장 한정 필요): 거리가 결과와 무관하지는 않습니다 — A− 풀링 Spearman +0.31 [+0.05, +0.56], 최대 거리 분위 증폭 75.3%. 따라서 논문 문구는 "matched error magnitude alone does not determine outcome"까지가 지지되며, "distance is irrelevant"는 지지되지 않습니다.

---

## C. Causal feedback (실험 7 action_history_causal, 실험 8 reverse patch)

설계(실험 7): equal-distance set의 같은 (장면, perturbation) 단위에서 t*까지 prefix와 강제 token을 고정하고, t* 이후 **조건화 문맥만** 바꿈(실행 action은 항상 모델 출력). 7개 조건을 한 harness에서 생성(batch 7행), RTX 5090, T 0.01, seed = sha256(`0:token:perturbation:k`). 출처: `O/action_history_causal/summary.json` 키 `subsets."all perturbations".<g>.conditions.<cond>` 및 `.vs_normal.<cond>`; 보고서 `ACTION_HISTORY_CAUSAL.md`. 스크립트 `S/action_history_causal.py`, `S/analyze_action_history.py`(REPS 2000, `cboot` seed 0, McNemar binomtest, Wilcoxon).

### C.1 조건별 수준 (all perturbations)

A− (단위 365, 장면 52, log 16):

| 조건 | amplification | recovery | ADE (m) | FDE (m) | downstream token error | 판정 사용 |
|---|---:|---:|---:|---:|---:|---|
| Normal AR | **47.4% [38, 57]** | 44.7% [37, 52] | 2.45 [2.06, 2.75] | 6.54 [5.36, 7.45] | 91.6% [88, 94] | 기준 |
| Action-history attention mask | 46.0% [38, 59] | 47.7% [35, 56] | 2.28 [1.90, 2.63] | 5.96 [4.83, 6.96] | 92.8% [89, 95] | 예 |
| Recent-action attention mask | 44.1% [35, 54] | 46.6% [39, 55] | 2.33 [1.96, 2.65] | 6.17 [4.99, 7.21] | 91.4% [88, 94] | 예 |
| [OOD] Action-history embedding neutralised | 66.6% [53, 82] | 33.4% [18, 47] | 9.41 | 25.86 | 98.5% | 아니오 (첫 step entropy +1.94 nat) |
| [OOD] Recent-action embedding neutralised | 52.3% [44, 61] | 44.4% [36, 52] | 4.37 | 10.45 | 97.2% | 아니오 |
| **GT-history** (t* 이후 문맥 전부 GT) | **4.1% [1, 8]** | 93.2% [87, 97] | 0.91 [0.68, 1.19] | 1.80 [1.28, 2.54] | 36.3% [33, 42] | 예 |
| **Recent-GT** (직전 1 token만 GT) | **7.1% [4, 10]** | 89.6% [85, 93] | 1.07 [0.81, 1.35] | 2.34 [1.59, 3.17] | 61.4% [57, 68] | 예 |

A+ (단위 1,043, 장면 156): Normal 4.2% [2, 6], attention mask 4.6 / 4.8%, GT-history 0.6% [0, 2], Recent-GT 0.7% [0, 2]; FDE 2.25 / 1.00 / 1.05 m.
Reverse (실험 8, 아래 C.3): GT-history rollout의 직전 위치에 자기 생성 token embedding 하나 삽입 → A− 33.4% [26.3, 43.2].

### C.2 쌍대 대비와 상대 효과

| 대비 (A−, n = 365) | Δamplification | p (McNemar, 불일치 쌍) | ΔFDE (m) | Δdownstream error | 출처 |
|---|---|---|---|---|---|
| Recent-GT − Normal | **−40.3%p [−50.3, −30.0]** | 8.0e-42 (2 vs 149) | −4.20 [−4.90, −3.39], p = 4.5e-53 | −30.2%p [−35.3, −23.1] | `vs_normal.recent_gt` |
| GT-history − Normal | **−43.3%p [−54.2, −32.1]** | 2.2e-46 | −4.73 [−5.52, −3.72], p = 1.4e-55 | −55.4%p [−59.2, −49.3] | `vs_normal.gt_history` |
| Action-history attention mask − Normal | −1.4%p [−8.5, +7.4] | 0.63 | −0.57 [−0.99, −0.10], p = 0.0029 | +1.2%p | `vs_normal.hist_attn_mask` |
| Recent-action attention mask − Normal | −3.3%p [−9.7, +2.9] | 0.18 | −0.36 [−0.73, −0.07], p = 0.076 | −0.3%p | `vs_normal.recent_attn_mask` |
| Reverse@emb − GT-history (실험 8) | **+29.0%p [+21.1, +38.0]** | 6.7e-31 | +3.67 [+2.83, +4.29], p = 1.2e-54 | | `O/prev_action_state_patching/summary.json` `groups.A-.all.rows."reverse@emb".vs_base` |

상대 효과 (모두 증폭률 기준, 명시 공식):
- **"93% removed"의 정확한 출처**: Recent-GT가 GT-history 효과 중 차지하는 비율 = (Normal − Recent-GT) / (Normal − GT-history) = (47.4 − 7.1) / (47.4 − 4.1) = 40.3 / 43.3 = **93.0%** (unrounded 0.47397, 0.07123, 0.04110 → 93.04%; recomputed: `build_autovla_csvs.py`, 보고된 평균의 비). 실험 7 자체는 이 비율의 CI를 **보고하지 않음**.
  - 같은 1,408 단위를 다른 batch(93행)로 재디코딩한 실험 8에서 같은 비율이 계산·보고됨: (46.8 − 7.4) / (46.8 − 4.4) = **92.9% [89.3, 99.1]** (`O/prev_action_state_patching/summary.json` `groups.A-.all.fraction_of_gt_history_effect.recent_gt`, log cluster bootstrap 2,000회).
  - 실험 11(batch 9행): GT-history 효과 / Recent-GT 효과 = 1.098 [1.021, 1.169] (`O/prev_action_identity_decomposition/summary.json` `groups.A-.frac_of_recent_gt_effect.gt_history.amplification`) → 역수 = Recent-GT 비율 **91.1% [85.5, 97.9]** (recomputed: 1/x; percentile CI는 단조변환에 대해 불변).
  - 주의: "93%가 사라진다"는 Normal 증폭률의 93%가 아니라 **GT-history가 만드는 최대 감소분의 93%**입니다. Normal 증폭률 대비 상대 감소는 (47.4 − 7.1) / 47.4 = **85.0%** (recomputed).
- Normal − GT-history: 43.3%p (분모 자체, 100%).
- 실험 7 attention mask의 상대 효과: (47.4 − 46.0) / 43.3 = 3.2%, (47.4 − 44.1) / 43.3 = 7.6% (recomputed; 둘 다 n.s.).
- **Reverse − GT-history(복원 비율)** = (Reverse − GT-history) / (Normal − GT-history) = (33.4 − 4.4) / (46.8 − 4.4) = 29.0 / 42.4 = **68.4%** (실험 8 수치; recomputed; CI not reported). 더 깊은 layer의 Normal state를 넣을수록 L0–L28 38.6–41.4%, L35 46.8%(= Normal)로 복원.

### C.3 47.1 vs 47.4 등 기준값 차이의 원인 (같은 population)
A− Normal AR 증폭률은 모두 **같은 365 단위(52 장면)**에서 측정됐고, 차이는 batch 구성에 따른 bf16 수치 경로 차이입니다(이전 실험 token 재현율 86.5–89.2%).

| 실험 | batch 행 수 | Normal | GT-history | Recent-GT | Normal token 재현율(vs 실험 7) | 출처 |
|---|---:|---:|---:|---:|---:|---|
| 7 action_history | 7 | 47.4% | 4.1% | 7.1% | – | `O/action_history_causal/summary.json` |
| 8 state patching | 93 | 46.8% | 4.4% | 7.4% | 88.6% | `O/prev_action_state_patching/summary.json` `sanity.reproduces_action_history_causal_normal` |
| 9 temporal window | 17 | 46.6% | 4.1% | – | 89.2% | `O/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` §10 |
| 11 identity | 9 | 47.1% | 4.1% | 7.9% | 87.6% | `O/prev_action_identity_decomposition/summary.json` `reproduces_action_history_tokens` |
| 32 motion semantics | 6 | 47.9% | – | 7.7% | 88–89% (RESULTS.md) | `O/motion_semantics_ablation/summary.json` |

→ 서로 다른 population이 아니라 **재디코딩 수치 잡음**(±0.7%p 범위)이며, 각 실험의 모든 비교는 같은 batch 안의 쌍대 비교입니다. 상대 효과를 인용할 때는 반드시 같은 실험의 분자·분모를 사용해야 합니다(예: 실험 9의 47%를 실험 7의 Normal 47.4%와 섞지 말 것).

### C.4 부분집합 (같은 harness, 실험 7)
- original token only (A− 52 단위): Normal 67.3% [45, 86] → Recent-GT 3.8% [0, 11] (−63.5%p, p = 2.3e-10), GT-history 1.9% (−65.4%p, p = 1.2e-10). Attention mask −11.5%p [−25.5, +5.4], p = 0.11.
- previously amplified perturbations (equal-distance run에서 증폭됐던 A− 175 단위, 45 장면): Normal 94.3% → Recent-GT 13.1% (−81.1%p), GT-history 8.0% (−86.3%p), **attention mask −16.0%p [−23.7, −6.2], p = 4.3e-06 / −19.4%p [−28.1, −10.1], p = 5.7e-08** (결과로 선택된 부분집합 — conflicts C-2 참조).

---

## D. Layer/state patching (실험 8)

설계: 직전 action 위치의 residual state를 layer별로 GT-history rollout의 state로 교체(full replacement), α-delta patch, reverse patch. 조건 93개를 한 batch로 계산. **patch 위치 37개 = embedding 출력(emb) + decoder block L0–L35 출력**(Qwen2.5-VL-3B 언어모델 decoder block 36개). 출처: `O/prev_action_state_patching/summary.json` `groups.<g>.all.rows.<name>.level`, `vs_base`, `fraction_of_gt_history_effect`; 보고서 `PREV_ACTION_STATE_PATCHING.md` §3–5. 스크립트 `S/prev_action_state_patching.py`, `S/analyze_prev_action_state_patching.py`(`Boot` log cluster 2,000회, seed 0; McNemar/Wilcoxon). 전체 layer 표: `P/figure_layer_patching.csv`.

| 항목 (A−, 365 단위) | amplification | GT-history 효과 대비 | 비고 |
|---|---:|---:|---|
| Normal AR | 46.8% [36.8, 56.7] | – | |
| GT-history (= patch@L35, 1408/1408 동일) | 4.4% [1.6, 8.0] | 100% | |
| Recent-GT (= **patch@emb**, 1408/1408 동일) | 7.4% [4.2, 10.2] | **93% [89, 99]** | 구조적 동치 |
| patch_full 37개 위치 **최소** | **3.8%** (L31) [1.4, 7.2] | | |
| patch_full 37개 위치 **최대** | **7.4%** (emb, L5, L13) | | |
| GT-history 효과 대비 범위 | | **93–101%** | `fraction_of_gt_history_effect` |
| Reverse@emb (GT 문맥 + 자기 token embedding) | **33.4% [26.3, 43.2]** | 복원 68.4% (recomputed) | vs GT-history +29.0%p [+21.1, +38.0], p = 6.7e-31 |
| Reverse@L0–L28 | 38.1–41.4% | | |
| Reverse@L35 (= Normal) | 46.8% | | |
| Delta patch α = 0.5, 대표 10개 layer | 6.6% (L35) – 14.8% (L16) | | α = 0.25: 28.5–34.2% |
| attention-mask 조건 | 실험 8에는 없음 → 실험 7 C.1 (46.0%, 44.1%; n.s.) | | |

A+ (1,043 단위): Normal 4.5%, patch_full 37개 위치 0.48–0.86%.

"단일 layer가 유일하게 인과적이지 않다"를 보여주는 요약(모두 보고된 37개 점추정과 CI에서만 계산; recomputed: `build_autovla_csvs.py`):
- A− patch_full 증폭률 범위 3.84–7.40% (폭 **3.56%p**), 37개 점추정의 SD **1.16%p** — 반면 개별 layer CI 폭의 중앙값은 **6.34%p**, 그리고 Normal−GT-history 효과는 42.5%p.
- 37개 CI가 모두 겹치는 공통 구간이 존재: [4.24, 7.24]% (max CI_low = 4.24, min CI_high = 7.24).
- 모든 위치가 GT-history 효과의 92.9–101.3%를 달성.
- 다만 "어느 layer나 완전히 같다"는 아님: 내부 layer − Recent-GT(=emb) 쌍대 차이는 L28 −3.3%p [−4.6, −0.8] p = 4.9e-4, L31 −3.6%p p = 2.4e-4, L35 −3.0%p p = 0.0034로 **후반 layer에서 약 3%p의 작은 유의한 추가 감소**가 있음(보고서 §5 "내부 layer − Recent-GT"). 이 추가분은 GT 재정렬(+25%p)·entropy(−0.49 nat)에서 크고, 보고서는 이를 과거 GT 문맥 통합(L17–L33)으로 해석.

---

## E. Temporal window (실험 9) 와 horizon 통제 (실험 10)

### E.1 실험 9: 교정 window 길이 (A−, all perturbations, 365 단위 / 52 장면)
교정 정의: context 위치 t*+1 … t*+w를 GT token으로(실행 action은 모델 출력). 17개 조건 한 batch, RTX 5090, T 0.01. 출처: `O/temporal_feedback_window/summary.json` `subsets."all perturbations".A-.rows.<cond>.{level,vs_normal}`, `.fraction.<metric>.<cond>`; 보고서 `TEMPORAL_FEEDBACK_WINDOW.md` §3–4. 스크립트 `S/temporal_feedback_window.py`, `S/analyze_temporal_feedback_window.py`.
**Full 효과 대비 비율 공식**: f(w) = (Normal − win_w) / (Normal − GT-history), 같은 실험 안의 값(Normal 46.6%, Full 4.1%), CI = 단위 재표본 log cluster bootstrap.

| 조건 | amplification | 절대 감소 vs Normal | f(w) 증폭 | ADE (m) | FDE (m) | f(w) FDE |
|---|---:|---|---:|---:|---:|---:|
| Normal | 46.6% [36.2, 56.9] | – | – | 2.43 [2.05, 2.73] | 6.45 [5.26, 7.43] | – |
| 1-step | 26.6% [20.5, 34.8] | −20.0%p [−27.4, −12.3], p = 2.3e-14 | **47% [33, 60]** = (46.6−26.6)/(46.6−4.1) | 1.42 | 3.74 [2.84, 4.70] | 58% [50, 68] |
| 2-step | 14.2% [9.0, 20.7] | −32.3%p [−42.5, −20.5], p = 9.2e-30 | **76% [62, 87]** | 1.09 | 2.62 [1.95, 3.59] | 82% [75, 88] |
| 3-step | 8.5% [3.6, 13.4] | −38.1%p [−50.8, −24.8], p = 1.0e-40 | **90% [79, 99]** | 0.98 | 2.15 [1.55, 2.91] | 92% [87, 98] |
| 4-step | 5.8% [2.5, 8.8] | −40.8%p [−52.2, −29.2], p = 1.1e-43 | **96% [94, 100]** | 0.93 | 1.94 [1.49, 2.62] | 97% [94, 100] |
| 5-step | 5.2% | −41.4%p; vs 4-step −0.5%p, p = 0.5 | 97% | 0.93 | 1.90 | 98% |
| Full (GT-history) | 4.1% [1.4, 7.7] | −42.5%p [−53.4, −30.6], p = 1.7e-45 | 100% | 0.91 | 1.80 [1.30, 2.55] | 100% |

FDE 절대 감소 vs Normal: −2.71, −3.82, −4.30, −4.51, −4.65 m (1/2/3/4-step/Full). A+ (1,043): 증폭 4.2 → 1.8 / 1.0 / 1.2 / 0.7 / 0.6%; FDE 2.24 → 1.48 / 1.22 / 1.13 / 1.07 / 1.01 m, f(w) FDE 62 / 83 / 90 / 95%.
90% 최소 window(점추정, bootstrap 분포): 증폭 4 step (3: 50%, 4: 50%), FDE 3 step (3: 84%) (`subsets.*.A-.critical_window`).
끝 위치를 맞춘 비교(horizon 동일): t*+1 추가 −4.9%p [−8.2, −0.9] p = 0.011, t*+2 추가 −4.9%p [−9.2, −2.5] p = 0.004 (`O/temporal_feedback_window/early_step_contribution.json`).

### E.2 실험 10: horizon 통제 (같은 free horizon)
설계: 20 action token까지 생성, 교정 해제 pose r_w = t*+w+1 이후 **정확히 N free step**에서 평가, Normal·Full은 같은 pose 구간의 쌍대 기준값. 주 분석은 학습된 10-token 안(t* = 0: A− 116 단위 / 17 장면, A+ 336 단위 / 50 장면; t* ≤ 1: A− 277 / 39, A+ 772 / 116). 제외 1 장면(A+, 미래 GT 부족). 출처: `O/horizon_controlled_window/summary.json` `analyses.in10."t*=0".<g>.N.<N>.windows.<w>.{level,fraction,rediverge_given_stab,vs_full}`, `decay_test.json`; 보고서 `HORIZON_CONTROLLED_WINDOW.md`. 스크립트 `S/horizon_controlled_window.py`, `S/analyze_horizon_controlled_window.py`.
공식: (Normal_same − win) / (Normal_same − Full_same), 같은 pose 구간.

A−, t* = 0, N = 4 (2 s):

| window | FDE@release+4 (m) | Normal/Full 같은 구간 | **FDE 회복** | 증폭(FDE > 3 m) 회복 | ΔFDE vs Full | 해제 후 재발산* | Full 재발산* |
|---|---:|---|---:|---:|---|---:|---:|
| Normal | 3.12 [1.97, 4.63] | – | – | – | +1.66 [+1.15, +2.18] | 49.0% [29.3, 64.5] | – |
| 1-step | 2.75 [1.67, 3.99] | 4.24 / 1.82 | **62% [50, 74]** | 34% [26, 52] | +0.92 [+0.57, +1.20], p = 8.9e-13 | 40.2% [14.3, 61.0] (n = 87) | 16.1% |
| 2-step | 2.62 | 5.60 / 2.18 | **87% [80, 93]** | (`P/figure_temporal_window.csv`) | +0.44 [+0.20, +0.72] | 24.7% [7.4, 56.1] | |
| 3-step | 3.05 | 7.13 / 2.60 | **90% [83, 97]** | | +0.45 [+0.11, +0.87] | 25.0% [2.1, 45.5] | |
| 4-step | 3.34 | 8.77 / 3.04 | **95% [90, 100]** | | +0.30 [+0.00, +0.54], p = 0.0011 | 28.8% [4.8, 44.2] | 12.1% [0.0, 24.2] |

*해제 시점 오차 ≤ 1 m 단위 중 N step 뒤 FDE > 2 m 비율.
A+, t* = 0, N = 4: FDE 회복 71 / 87 / 93 / 89%. t* ≤ 1, N = 3: A− 72 / 90 / 95 / 97%, A+ 66 / 88 / 94 / 98%.
감쇠 검정(window 고정, N 증가; `decay_test.json`): 2-step 98 → 79% (N 1→6) −19%p [−31, −8]; 3-step 98 → 87% (N 1→5) −11%p [−19, −4]; 4-step 101 → 95% (N 1→4) −6%p [−10, −1].
"증폭"을 같은 free horizon에서 보는 이진 지표: 짧은 구간이라 원 분석은 FDE > 3 m(amp) / > 2 m(amp2) / > 1 m을 모두 보고하고 주 판정은 FDE 비율에 둠; amp2 기준 90% window 2 step(bootstrap [2, 99]).

---

## F. Motion semantics (실험 11, 실험 32)

공통: 직전 context token 하나를 조건별로 교체(k ≥ t*+2), 실행 action은 각 조건 자신의 출력, equal-distance set 365 / 1,043 단위. 출처: `O/prev_action_identity_decomposition/summary.json` (실험 11; 스크립트 `S/prev_action_identity_decomposition.py`, `S/analyze_prev_action_identity.py`), `O/motion_semantics_ablation/summary.json`, `row_contrasts.json` (실험 32; `S/motion_semantics_ablation.py`, `S/analyze_motion_semantics.py`). "Recent-GT 효과 대비" = (Normal − row) / (Normal − Recent-GT), CI = log cluster bootstrap 2,000회 (`frac_effect`, seed 7).

### F.1 수준 (A−, 365 단위 / 52 장면)

| 실험 | 직전 token := | amplification | FDE (m) | Recent-GT 효과 대비 | vs Normal p (McNemar) |
|---|---|---:|---:|---:|---|
| 11 | 자기 token (Normal) | 47.1% [37.3, 56.9] | 6.51 [5.25, 7.47] | – | – |
| 11 | GT token (Recent-GT) | 7.9% [4.3, 11.5] | 2.34 [1.60, 3.15] | 100% | 6.5e-42 |
| 11 | GT-history | 4.1% [1.3, 7.7] | 1.78 [1.28, 2.51] | 110% [102, 117] | 4.4e-46 |
| 11 | **GT와 motion 최근접 다른 token** | **8.5% [4.4, 14.9]** | 2.45 [1.62, 3.33] | **99% [86, 105]** | 2.6e-41 |
| 11 | GT와 embedding 최근접 다른 token | 7.7% [3.7, 13.6] | 2.47 [1.83, 3.50] | 101% [86, 109] | 6.1e-38 |
| 11 | **자기와 motion 최근접 다른 token** | **43.8% [35.4, 53.2]** | 6.39 [5.06, 7.31] | 8% [−17, 24] | **0.18** |
| 11 | 자기와 embedding 최근접 다른 token | 50.1% [41.4, 62.6] | 7.04 [5.98, 8.23] | −8% | 0.27 |
| 11 | 무작위 (학습된) token | 51.5% [43.4, 59.2] | 9.05 [7.49, 10.24] | −11% | 0.23 |
| 11 | [OOD] 평균 action embedding | 52.1% [42.7, 60.7] | 9.76 [8.22, 10.68] | −13% | 0.15 |
| 32 | 자기 token (Normal) | 47.9% [37.9, 58.0] | 6.49 [5.25, 7.46] | – | – |
| 32 | GT token (Recent-GT) | 7.7% [4.2, 11.5] | 2.33 [1.60, 3.16] | 100% | 4.2e-43 |
| 32 | **방향 정답 + 크기 오답** | **17.3% [10.8, 23.0]** | 3.63 [2.66, 4.27] | **76% [65, 91]** | 3.5e-26 |
| 32 | **크기 정답 + 방향 오답** | **26.3% [21.6, 34.2]** | 4.84 [4.09, 5.70] | **54% [33, 64]** | 2.7e-13 |
| 32 | **GT 반대편, 같은 거리** | **13.7% [9.1, 18.8]** | 3.13 [2.27, 3.80] | **85% [76, 91]** | 4.0e-29 |
| 32 | 크기만 맞춘 무작위 | 19.2% [14.7, 25.3] | 3.81 [2.90, 4.50] | 71% [56, 79] | 3.8e-19 |

"random embedding" 조건은 두 실험 모두에 없음: 실험 11의 '무작위 token'(학습된 token 중 무작위, embedding은 실제 token의 것)과 '[OOD] 평균 embedding'이 가장 가까운 조건입니다.
A+ (1,043): 실험 11 Normal 4.2%, GT-motion 최근접 0.8%, 자기-motion 최근접 7.2%, 무작위 43.2%, 평균 embedding 40.5%; 실험 32 Normal 4.3%, 방향정답·크기오답 3.7% (vs Normal p = 0.58), 방향오답·크기정답 8.6% (p = 1.1e-05), 반대편 3.2% (p = 0.15), 크기만 맞춘 무작위 11.6%.
달성 기하(실험 32 A−, `substitution_geometry`): 자기 오차 e ≈ 0.50 m; 방향정답·크기오답 d(sub, GT) 0.505 m, Δ방향 0.5°; 방향오답·크기정답 d(sub, GT) **0.318 m**(더 작음), Δ방향 17.3°; 반대편 d(sub, GT) 0.458 m (e = 0.515 m), Δ방향 12.1°.

### F.2 쌍대 대비

| 대비 | 그룹 | Δamplification | p | ΔFDE (m) | 출처 |
|---|---|---|---|---|---|
| **방향오답 − 크기오답** (실험 32) | A− | **+9.0%p [+3.1, +18.7]** | McNemar 6.1e-04 (61 vs 28) | **+1.21 [+0.74, +2.00]**, Wilcoxon 9.8e-09 | `row_contrasts.json` "A- dir_wrong_mag_ok-dir_ok_mag_wrong" |
| 방향오답 − 크기오답 | A+ | +4.9%p [+2.1, +8.0] | 3.7e-07 | +0.54 [+0.27, +0.77] | 동 A+ |
| **반대편 같은 거리 − 자기 token** (실험 32) | A− | **−34.2%p [−44.0, −22.8]** | 4.0e-29 (10 vs 135) | −3.35 [−4.04, −2.63], p = 1.3e-35 | "A- mirror_same_dist-normal" |
| 반대편 − 자기 | A+ | −1.2%p [−2.7, +0.6] | 0.15 | −0.61 [−0.86, −0.34] | |
| 방향오답 − 반대편 | A− | +12.6%p [+7.0, +21.1] | 4.7e-07 | +1.71 [+1.33, +2.30] | |
| **GT-motion 최근접 − GT token** (실험 11) | A− | **+0.5%p [−1.9, +6.3]** | **0.85** (15 vs 13) | +0.11 [−0.07, +0.25], Wilcoxon p = 0.043 | `groups.A-.vs_recent_gt.geo_nn_gt` |
| GT-motion 최근접 − GT token | A+ | +0.1%p [−1.1, +1.0] | 1.0 | +0.26 [+0.13, +0.34], p = 3.9e-18 | `groups.A+.vs_recent_gt.geo_nn_gt` |
| 자기-motion 최근접 − 자기 token (실험 11) | A− | −3.3%p [−10.8, +5.4] | 0.18 | −0.12 [−0.60, +0.38] | `groups.A-.vs_normal.geo_nn_self` |
| 크기만 맞춘 무작위 − 크기오답 (실험 32) | A− / A+ | +1.9%p [−2.9, +9.5], p = 0.49 / **+7.9%p [+4.5, +10.6]**, p = 1.7e-12 | | | |

기하 vs embedding 공동 회귀(실험 11, 네 최근접 대체 조건 풀링, 표준화 계수, log bootstrap 1,000회; `geometry_vs_embedding`):

| 집단 | n | ΔFDE / Δgeo 1 SD | ΔFDE / Δcos 1 SD | Δ증폭 / Δgeo 1 SD | Δ증폭 / Δcos 1 SD |
|---|---:|---|---|---|---|
| A− | 1,460 | **+1.99 m [+1.52, +2.70]** | +0.50 m [+0.04, +0.97] | **+16.1%p [+12.2, +21.5]** | +4.4%p [−0.4, +8.2] |
| 전체 | 5,632 | +1.00 m [+0.73, +1.31] | +0.03 m [−0.13, +0.19] | +7.8%p [+5.9, +10.3] | −0.1%p [−1.5, +1.2] |

### F.3 최소 근거 표시
**"motion > token identity / embedding"** (실험 11):
1. GT-motion 최근접 다른 token: 8.5% vs GT token 7.9% → 차이 +0.5%p [−1.9, +6.3], p = 0.85; Recent-GT 효과의 99% [86, 105].
2. 자기-motion 최근접 다른 token(identity만 변경): 43.8% vs 자기 47.1% → p = 0.18; Recent-GT 효과의 8% [−17, 24].
3. 공동 회귀: motion 1 SD당 A− 증폭 +16.1%p [+12.2, +21.5], embedding cos 1 SD당 +4.4%p [−0.4, +8.2](전체 −0.1%p).
(주의: 'GT embedding 최근접' 조건 단독 7.7%는 이 token이 motion으로도 GT에 가까워진 탓(0.62 vs 0.84)이라 조건별 표만으로는 embedding 가설을 판정할 수 없다고 보고서가 명시.)

**"direction > magnitude"** (실험 32):
1. 방향오답(크기정답) − 크기오답(방향정답): A− +9.0%p [+3.1, +18.7], p = 6.1e-04; FDE +1.21 m [+0.74, +2.00] — 방향오답 행의 GT 거리가 더 작은데도(0.318 vs 0.505 m).
2. A+에서 크기오답 3.7%(Normal 4.3%와 차이 없음, p = 0.58) vs 방향오답 8.6%(p = 1.1e-05).
3. 같은 크기 오차를 GT 반대편에: 47.9 → 13.7% (−34.2%p [−44.0, −22.8]), Recent-GT 효과의 85% [76, 91] — 단, 반대편 행의 달성 거리 0.458 m가 자기 오차 0.515 m보다 약 11% 작다는 점을 함께 기술해야 함.
