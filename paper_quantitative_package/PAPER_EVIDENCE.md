# PAPER_EVIDENCE — 논문 본문용 정량 근거 전체 (요청 §2–§6)

이 문서는 영역별 추출 결과를 그대로 합친 것입니다. 원 파일은 `_parts/`에 있습니다.
- §A–F (AutoVLA): `autovla_sections.md`
- 실험 33, 28, 34 상세: `crossmodel_sections.md`
- 탐지·선택: `detection_selection_sections.md`
- navhard: `navhard_sections.md`

표·그림용 CSV/TeX는 패키지 최상위에 있습니다.

---

## AutoVLA 기전 실험 — 논문용 정량 근거 (섹션 A–F)

모델: AutoVLA (Qwen2.5-VL-3B backbone, action codebook 2,048 token, natural fast-thinking 경로). 벤치마크: NAVSIM/nuPlan navtest PoC 28 log, 장면 2,747개(arm N).
경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`, `S/` = `/root/VLA/autovla_misalignment_poc/scripts/`, `P/` = `/root/VLA/paper_quantitative_package/`.
표기 규칙
- `[a, b]`는 원 분석 스크립트가 계산한 **log 단위 cluster bootstrap 95% percentile CI**입니다(별도 표시 없으면 2,000회, seed 0, 재표본 단위 = navtest log). 이진 지표의 p는 **McNemar exact(binomial) 검정**, 연속 지표의 p는 **Wilcoxon signed-rank**(단위별 쌍대)입니다.
- "not reported" = 원 결과에 없고, 원 스크립트의 같은 방법으로 재계산할 수 없거나 재계산하지 않은 값.
- "(recomputed: …)" = 이 패키지에서 기존 값으로부터 산술/개수만 다시 구한 값. 새 통계 검정은 하나도 추가하지 않았습니다.
- 증폭(amplification) 정의(실험 6–11, 32 공통, 실행 전 고정): **A− (P/R/A coarse-action 5 s 판정 실패) AND FDE(5 s) > 3.0 m**. recovery = P/R/A 5 s 판정 통과(A+). 출처: `S/analyze_equal_distance.py` L5–8, `S/analyze_action_history.py` `AMP_FDE = 3.0`.
- 실험 6–11, 32의 분석 단위: equal-distance set = 장면 208개(A− 52, A+ 156) / (장면, perturbation) 단위 1,408개(A− 365, A+ 1,043). A− 52 장면은 **16개 log**, A+ 156 장면은 26개 log(A+ log가 A− log를 포함, 전체 26 log)에서 옵니다 (recomputed: `P/_parts/scripts/build_autovla_csvs.py`, `O/equal_distance_perturbation/records.jsonl`의 distinct `log` 개수). 따라서 A− 관련 CI의 유효 cluster 수는 16입니다.

---

### A. mismatch ≠ failure (실험 4 natural_fast_mechanism, 실험 5 first_mismatch_causal)

#### A.1 natural 모집단과 A+/A− (실험 4)
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

#### A.2 첫 불일치(first mismatch)의 거리
출처: `O/natural_fast_mechanism/summary.json` `N.codebook_amplification.{A-, A+ with a token mismatch}`; `O/first_mismatch_causal/summary.json` `why.d_pred_gt`.

| 지표 | A− (n=52) | A+ with token mismatch (n=1,107) | 비고 |
|---|---:|---:|---|
| 오답 token–GT token 변위 거리, **중앙값** | **0.113 m** (p75 0.189) | **0.083 m** (p75 0.136) | `wrong_token_displacement_m.median` |
| 무작위 token 쌍 대비 백분위 | 0.5% | 0.3% | 무작위 쌍 중앙값 2.48 m, p10 0.64 m |
| GT 기준 오답 token의 NN 순위 중앙값 | 15 (순위≤10: 44%) | 8 (59%) | `nn_rank_of_wrong_token` |
| 첫 불일치 step의 pose 오차 | 0.103 m | 0.087 m | `pose_error_at_first_mismatch_m` (다른 거리 정의) |
| A− vs A+ 판별 AUROC (d_pred_gt) | 0.60 [0.52, 0.68] (n=1,159) | | `O/first_mismatch_causal/summary.json` `why.d_pred_gt` (AUROC CI: log bootstrap 1,000회 seed 2) |

#### A.3 첫 불일치 거리 vs 최종 FDE 상관
출처: `N.codebook_amplification.<group>.spearman_first_displacement_vs_fde` (스크립트 `S/natural_fast_mechanism.py` `_spearman_ci`: log cluster bootstrap 1,000회, `random.Random(3)`).

| 그룹 | Spearman ρ | 95% CI | p | n |
|---|---:|---|---|---:|
| A− | **+0.26** | [−0.02, +0.49] | not reported (스크립트가 p를 계산하지 않음) | 52 |
| A+ with token mismatch | +0.40 | [+0.34, +0.49] | not reported | 1,107 |
| step-0 오답(구 정의) | +0.31 | [+0.18, +0.44] | not reported | 233 |
| (참고) 첫 거리 vs ADE, A− | +0.43 | [+0.20, +0.62] | not reported | 52 |

→ A−에서 첫 편차 크기와 최종 FDE의 상관은 약하고 CI가 0을 포함합니다.

#### A.4 하류 증폭 (A− vs A+)
출처: `N.codebook_amplification`, 보고서 §3.5–3.6 표.

| 지표 | A− | A+ with mismatch | 비고 |
|---|---:|---:|---|
| FDE / 첫 step 오차 (중앙값 비) | **58.0×** | **6.5×** | `fde_over_first_step_error_median` |
| ADE / FDE (중앙값, 저장된 natural run) | 2.27 / 6.63 m | 0.28 / 0.66 m | `ade_m`, `fde_m` = **median** (`S/natural_fast_mechanism.py` L422) |
| 오답 전파: P(오답 \| 깨끗한 prefix) → P(오답 \| 직전 오답) | 5.3% → 72.9% (13.8×), 그룹 풀링 | | 보고서 §3.5 "전파(전체)" 행; `verdicts.error_amplification.propagation_ratio` = 13.83 |
| 첫 불일치 이후 downstream token error (재생성 original) | 94.6% [88.4, 99.1] | 72.4% [68.6, 74.7] (step-matched) / 70.5% (all) | `O/first_mismatch_causal/summary.json` `A-.original.downstream_err`, `A+ (step-matched).original.downstream_err` |
| pose 오차 곡선 평균 (step 0→9, m) | 0.08, 0.24, 0.52, 0.99, 1.67, 2.56, 3.75, 5.18, 6.71, 8.50 | 0.05, 0.10, 0.17, 0.27, 0.40, 0.56, 0.75, 0.97, 1.21, 1.48 | `pose_error_curve_mean` |

#### A.5 첫 불일치 timestep(t*) 분포
출처: `O/first_mismatch_causal/summary.json` `t_star_hist`.

| t* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A− (n=52) | 17 | 22 | 7 | 2 | 1 | 2 | 0 | 1 | 0 | 0 |
| A+ (n=1,107) | 216 | 249 | 175 | 96 | 79 | 74 | 55 | 55 | 56 | 52 |

A− 중앙값 t* = 1, A+ = 2; AUROC(t*) 0.31 [0.26, 0.37] (`why.t_star`). A−는 이르게 이탈하므로 A+ 비교는 t* 분포를 맞춘 가중 비교(step-matched, 유효 n = 944)도 함께 보고됩니다.

#### A.6 첫 불일치 교정 (실험 5)
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

### B. Equal-distance perturbation (실험 6, 핵심)

설계: A− 52 장면 + A− 장면당 **같은 t***의 A+ 3개(156). t*에 원래 token 또는 GT로부터 **같은 거리**(원래 오답의 거리와 상대오차 허용 0.1/0.2/0.35 단계, 절대 5 mm, 최소 5개)의 대안 token을 강제. original_reseed = 같은 token, 다른 sampling seed(noise floor). RTX 5090, T 0.01, seed 0. 출처: `O/equal_distance_perturbation/summary.json`(키 `n_scenes`, `alts_per_scene`, `alt_rel_err_median`, `alt_abs_dist_diff_median_m`, `table`, `Q1`–`Q5`), 보고서 `EQUAL_DISTANCE.md`; strict 부분집합(전체 codebook fill로 들어온 대안 제외)은 `EQUAL_DISTANCE_strict.md`. **strict의 summary json은 출력 디렉토리에 없어** 원 스크립트를 사본에서 `STRICT=1`로 재실행해 얻었습니다 (recomputed: `P/_parts/scripts/rerun_equal_distance_analysis.sh` → `P/_parts/recomputed/equal_distance_summary_strict_rerun.json`; 재실행 MD는 원 MD와 Q4 AUROC 소수 둘째 자리만 다름 — REPRODUCTION.md에 기록된 solver 수치차와 동일).

#### B.1 설계 수치

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

#### B.2 결과 (그룹 × 조건)

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

#### B.3 장면 안에서 결과가 갈리는가, noise floor

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

#### B.4 거리와 결과: 상관, 분산 분해, 거리 5분위

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

#### B.5 "같은 크기의 오차만으로는 결과가 정해지지 않는다"를 지지하는 최소 수치
1. **장면 내 혼합**: A− 장면의 71.2% (37/52) [60.0, 82.8]에서 같은 거리(매칭 오차 중앙값 8.8 mm, 7.9%) 대안들이 recovery와 amplification으로 갈림 — 같은 token reseed의 결과 뒤집힘은 1.9% (1/52) [0, 6.5]에 불과. (strict에서도 64.7% [55.6, 72.4].)
2. **장면 내 FDE 산포 vs noise**: 같은 장면의 대안 FDE SD 2.77 m [2.37, 3.25] vs 같은 token reseed |ΔFDE| 0.11 m [0.01, 0.27] (약 25배; 서로 다른 통계량이므로 비율은 서술적).
3. **장면 내 거리 순위와 FDE 순위의 상관 ≈ 0**: A− +0.09 [−0.01, +0.19].
4. **분산의 46.2%가 장면 내**, 거리 구간 간 설명분은 17.0%.
5. **거리가 겹치는 구간에서 그룹별 결과가 전혀 다름**: 최소 거리 5분위(A− 0.027–0.068 m vs A+ 0.014–0.058 m) 증폭 50.7% vs 1.4% (CI 없음). 대안 전체 증폭 A− 45.0% [37, 55] vs A+ 4.8% [3, 7].
6. **예측력**: 장면 AUROC 0.80 > 거리 0.61.
반대 방향 근거(주장 한정 필요): 거리가 결과와 무관하지는 않습니다 — A− 풀링 Spearman +0.31 [+0.05, +0.56], 최대 거리 분위 증폭 75.3%. 따라서 논문 문구는 "matched error magnitude alone does not determine outcome"까지가 지지되며, "distance is irrelevant"는 지지되지 않습니다.

---

### C. Causal feedback (실험 7 action_history_causal, 실험 8 reverse patch)

설계(실험 7): equal-distance set의 같은 (장면, perturbation) 단위에서 t*까지 prefix와 강제 token을 고정하고, t* 이후 **조건화 문맥만** 바꿈(실행 action은 항상 모델 출력). 7개 조건을 한 harness에서 생성(batch 7행), RTX 5090, T 0.01, seed = sha256(`0:token:perturbation:k`). 출처: `O/action_history_causal/summary.json` 키 `subsets."all perturbations".<g>.conditions.<cond>` 및 `.vs_normal.<cond>`; 보고서 `ACTION_HISTORY_CAUSAL.md`. 스크립트 `S/action_history_causal.py`, `S/analyze_action_history.py`(REPS 2000, `cboot` seed 0, McNemar binomtest, Wilcoxon).

#### C.1 조건별 수준 (all perturbations)

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

#### C.2 쌍대 대비와 상대 효과

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

#### C.3 47.1 vs 47.4 등 기준값 차이의 원인 (같은 population)
A− Normal AR 증폭률은 모두 **같은 365 단위(52 장면)**에서 측정됐고, 차이는 batch 구성에 따른 bf16 수치 경로 차이입니다(이전 실험 token 재현율 86.5–89.2%).

| 실험 | batch 행 수 | Normal | GT-history | Recent-GT | Normal token 재현율(vs 실험 7) | 출처 |
|---|---:|---:|---:|---:|---:|---|
| 7 action_history | 7 | 47.4% | 4.1% | 7.1% | – | `O/action_history_causal/summary.json` |
| 8 state patching | 93 | 46.8% | 4.4% | 7.4% | 88.6% | `O/prev_action_state_patching/summary.json` `sanity.reproduces_action_history_causal_normal` |
| 9 temporal window | 17 | 46.6% | 4.1% | – | 89.2% | `O/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` §10 |
| 11 identity | 9 | 47.1% | 4.1% | 7.9% | 87.6% | `O/prev_action_identity_decomposition/summary.json` `reproduces_action_history_tokens` |
| 32 motion semantics | 6 | 47.9% | – | 7.7% | 88–89% (RESULTS.md) | `O/motion_semantics_ablation/summary.json` |

→ 서로 다른 population이 아니라 **재디코딩 수치 잡음**(±0.7%p 범위)이며, 각 실험의 모든 비교는 같은 batch 안의 쌍대 비교입니다. 상대 효과를 인용할 때는 반드시 같은 실험의 분자·분모를 사용해야 합니다(예: 실험 9의 47%를 실험 7의 Normal 47.4%와 섞지 말 것).

#### C.4 부분집합 (같은 harness, 실험 7)
- original token only (A− 52 단위): Normal 67.3% [45, 86] → Recent-GT 3.8% [0, 11] (−63.5%p, p = 2.3e-10), GT-history 1.9% (−65.4%p, p = 1.2e-10). Attention mask −11.5%p [−25.5, +5.4], p = 0.11.
- previously amplified perturbations (equal-distance run에서 증폭됐던 A− 175 단위, 45 장면): Normal 94.3% → Recent-GT 13.1% (−81.1%p), GT-history 8.0% (−86.3%p), **attention mask −16.0%p [−23.7, −6.2], p = 4.3e-06 / −19.4%p [−28.1, −10.1], p = 5.7e-08** (결과로 선택된 부분집합 — conflicts C-2 참조).

---

### D. Layer/state patching (실험 8)

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

### E. Temporal window (실험 9) 와 horizon 통제 (실험 10)

#### E.1 실험 9: 교정 window 길이 (A−, all perturbations, 365 단위 / 52 장면)
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

#### E.2 실험 10: horizon 통제 (같은 free horizon)
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

### F. Motion semantics (실험 11, 실험 32)

공통: 직전 context token 하나를 조건별로 교체(k ≥ t*+2), 실행 action은 각 조건 자신의 출력, equal-distance set 365 / 1,043 단위. 출처: `O/prev_action_identity_decomposition/summary.json` (실험 11; 스크립트 `S/prev_action_identity_decomposition.py`, `S/analyze_prev_action_identity.py`), `O/motion_semantics_ablation/summary.json`, `row_contrasts.json` (실험 32; `S/motion_semantics_ablation.py`, `S/analyze_motion_semantics.py`). "Recent-GT 효과 대비" = (Normal − row) / (Normal − Recent-GT), CI = log cluster bootstrap 2,000회 (`frac_effect`, seed 7).

#### F.1 수준 (A−, 365 단위 / 52 장면)

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

#### F.2 쌍대 대비

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

#### F.3 최소 근거 표시
**"motion > token identity / embedding"** (실험 11):
1. GT-motion 최근접 다른 token: 8.5% vs GT token 7.9% → 차이 +0.5%p [−1.9, +6.3], p = 0.85; Recent-GT 효과의 99% [86, 105].
2. 자기-motion 최근접 다른 token(identity만 변경): 43.8% vs 자기 47.1% → p = 0.18; Recent-GT 효과의 8% [−17, 24].
3. 공동 회귀: motion 1 SD당 A− 증폭 +16.1%p [+12.2, +21.5], embedding cos 1 SD당 +4.4%p [−0.4, +8.2](전체 −0.1%p).
(주의: 'GT embedding 최근접' 조건 단독 7.7%는 이 token이 motion으로도 GT에 가까워진 탓(0.62 vs 0.84)이라 조건별 표만으로는 embedding 가설을 판정할 수 없다고 보고서가 명시.)

**"direction > magnitude"** (실험 32):
1. 방향오답(크기정답) − 크기오답(방향정답): A− +9.0%p [+3.1, +18.7], p = 6.1e-04; FDE +1.21 m [+0.74, +2.00] — 방향오답 행의 GT 거리가 더 작은데도(0.318 vs 0.505 m).
2. A+에서 크기오답 3.7%(Normal 4.3%와 차이 없음, p = 0.58) vs 방향오답 8.6%(p = 1.1e-05).
3. 같은 크기 오차를 GT 반대편에: 47.9 → 13.7% (−34.2%p [−44.0, −22.8]), Recent-GT 효과의 85% [76, 91] — 단, 반대편 행의 달성 거리 0.458 m가 자기 오차 0.515 m보다 약 11% 작다는 점을 함께 기술해야 함.

---

## Cross-model replication — 정량 근거 정리 (논문용)

경로는 모두 `/root/VLA/autovla_misalignment_poc/` 기준입니다. 이 문서는 기존 결과만 사용했고, 새 실험은 하지 않았습니다.

- "(recomputed: …)" 표시: raw 데이터에서 **원 분석 스크립트와 같은 방법**으로 이 패키지에서 다시 계산한 값입니다. 계산 script는 `paper_quantitative_package/_parts/scripts/`에 있습니다.
- 그 밖의 값: `analysis.json`(JSON key 표기)이나 RESULTS.md에서 그대로 옮겼습니다.
- 출처 상세: `source_map_crossmodel.md`. 충돌·불일치: `conflicts_crossmodel.md`.

AutoVLA 참조값 (모두 open-loop, navtest PoC, log 단위 cluster bootstrap 2,000회, McNemar):

| 항목 | 값 | 출처 |
|---|---|---|
| 직전 token 교정 (A− 365 단위 / 52 장면) | Normal 47.4% [38, 57] → Recent-GT 7.1% [4, 10]<br>Δ −40.3%p [−50.3, −30.0], p = 8e-42 | `outputs/action_history_causal/ACTION_HISTORY_CAUSAL.md` |
| reverse@emb | GT-history 4.4% → 33.4% [26.3, 43.2]<br>+29.0%p [+21.1, +38.0], p = 6.7e-31 | `outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` |
| 교정 window 1/2/3/4 step | full 효과의 47 / 76 / 90 / 96% | `outputs/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` |
| 방향 오답 − 크기 오답 | +9.0%p [+3.1, +18.7], p = 6e-4 | `outputs/motion_semantics_ablation/RESULTS.md` |
| 같은 거리 대안 token의 증폭 | A− 장면 45.0% vs A+ 장면 4.8% | `outputs/equal_distance_perturbation/EQUAL_DISTANCE.md` |

---

### A. Impromptu VLA 3B (실험 33) — 시간축 action-chunk 모델, **Strong replication**

#### A.1 설정
- 사전 등록 038a6d3, 결과 698148a. RTX 5090.
- 모델: `aaaaaap/ImpromptuVLAModel/3B_AD` (Qwen2.5-VL-3B 전체 fine-tune).
- 출력: 미래 10개 waypoint(0.5–5 s)를 **현재 ego 좌표의 절대 위치 텍스트** `[x, y]`로 시간순 autoregressive 생성합니다(waypoint당 digit token 약 12개). greedy 디코딩에 repetition penalty 1.05를 적용했습니다.
- 장면:
  - navtest PoC 28 log의 2,748 장면 중 해석 가능한 **2,728 장면** (`AB.n_scenes`, `AB.n_logs` = 28)
  - C–E는 무작위 300 장면 × 8 방향 = **2,398 단위** (2,400 중 2 단위 parse 실패; `CDE.n_units`)
- 통계: log 단위 cluster bootstrap 95% CI (2,000회, seed 0), McNemar exact, Wilcoxon (`scripts/cross_vla_temporal/analyze_temporal.py`).
- 모든 지표는 **open-loop 궤적 지표**입니다. A−는 P/R/A accept 실패이고, 증폭은 A− ∧ FDE5 > 3 m입니다.

#### A.2 자연 성능과 재실행 변동 (`analysis.json` `AB.natural`, `AB.rerun_variability`)
- 자연 성능: A− **33.4% [26.2, 38.0]**, 증폭 31.5% [24.4, 36.2], FDE5 **5.75 m [5.17, 6.27]** (N = 2,728)
- batch 구성만 바꾼 재실행(natural vs natural_rep):
  - A− 불일치 **5.9% [5.2, 6.9]**, 증폭 불일치 5.9% [5.1, 6.7]
  - 최종 waypoint 이동 0.90 m [0.84, 0.96]
  - A− 비율 차이 +0.18%p [−0.74, +1.08] (McNemar 83 vs 78, p = 0.75)
- 자연 A+ 장면이 재실행만으로 A−가 되는 비율: **4.6% [3.7, 5.5]** (N = 1,818)

#### A.3 0.2 m 첫 waypoint 교란 (Experiment A; `AB.A.m0.2`)
- 단위: 2,728 장면 × 8 방향 = 21,824. 기준은 pert_m0(δ = 0, 같은 prefix 경로)입니다.

| 지표 | 값 | N |
|---|---|---|
| 기준 A+ → 교란 후 A− | **49.5% [46.9, 52.5]** (재실행 기저 4.6%의 약 10배) | 14,424 |
| 교란 후 A− − 기준 A− (paired) | **+24.8%p [+19.4, +32.2]**, McNemar 7,138 vs 1,731, p ≈ 0 | 21,824 |
| r ≥ 3 (궤적 3배 이상 증폭) | **97.4% [96.9, 97.9]**, r 중앙값 27.6 | 21,824 |
| r ≤ 1 (흡수) | **0.45% [0.34, 0.60]** | 21,824 |
| (0.5 m) A+ → A− / paired | 58.6% [55.9, 61.4] / +31.1%p [+25.4, +39.1] | |

#### A.4 같은 크기 교란의 안정·불안정 분기 (Experiment B, 사전 등록 기준 3; `AB.B`)
- 같은 장면에 안정 분기(A+ ∧ r ≤ 1)와 불안정 분기(증폭 ∨ r ≥ 3)가 공존하는 비율:
  - 0.2 m: **2.2% [1.5, 3.2]**
  - 0.5 m: **3.2% [2.4, 3.9]**
- 두 값 모두 재실행 불일치 5.9%보다 작으므로 **기준 3은 미충족(FAIL)**입니다.
- log r 분산 중 장면 간 비율(ICC): 0.32 (0.2 m), 0.21 (0.5 m)
- (탐색적) task 수준에서 A+ 방향과 증폭 방향이 같은 장면에 공존하는 비율은 86.9–92.6%입니다 (`exploratory_task_level_branches.json`, 사전 등록 밖).

#### A.5 직전 action 교정과 window (Experiment C; `CDE.all_units`)
- 실행 궤적은 [강제 w̃_1, 자기 생성 w_2..w_10]으로 고정하고, **문맥만** 교정했습니다.

| 행 | 증폭 [95% CI] | A− | normal 대비 Δ증폭 [95% CI], McNemar p | full 효과 대비 (증폭) [95% CI] |
|---|---|---|---|---|
| normal | **63.4% [60.6, 66.7]** | 64.5% | – | – |
| win1 | 48.0% [42.8, 52.2] | 48.8% | −15.4%p [−22.0, −9.5], 7e-24 | 24.3% [15.6, 33.8] |
| win2 | 38.9% [33.7, 43.2] | 40.3% | −24.6%p [−31.4, −18.2], 2e-69 | 38.7% [29.7, 47.8] |
| win3 | 6.0% [3.0, 8.2] | 9.5% | −57.4%p [−62.4, −54.0], ≈0 | **90.5% [87.0, 95.4]** |
| win4 | 3.2% [0.9, 5.2] | 6.3% | −60.2%p [−64.4, −57.4], ≈0 | 94.9% [91.8, 98.6] |
| gt_history | **0.0%** | 0.0% | −63.4%p [−66.7, −60.6], ≈0 | 100% |
| recent_gt (직전 1개만 GT) | **1.3% [0.8, 1.8]** | **16.3%** | **−62.2%p [−65.6, −59.1]**, ≈0 (기록값 0.0, underflow) | 98.0% [97.2, 98.8] |

- **63.4 → 1.3%의 절대 효과: −62.2%p [−65.6, −59.1]** (N = 2,398, log 28개 cluster).
  - 1.3%는 *증폭률*입니다. 같은 행의 A−는 64.5 → 16.3%로, Δ −48.1%p [−52.3, −44.4], p = 8e-244입니다.
  - 절대 위치 표현이라 GT 문맥에는 GT 정보 누출이 섞입니다 (conflicts M2).
- window 기준 사전 등록 기준 4(단조 증가, win4 이내 80% 이상)는 **충족**입니다: 24 → 39 → 91 → 95%.

#### A.6 reverse (Experiment D; `CDE.all_units.reverse_vs_gt_history`)
- gt_history 문맥 끝에 자기 생성(오차를 품은) waypoint를 다시 넣었습니다.
- 증폭: **0.0 → 81.9% [80.4, 84.2]**, Δ +81.9%p [+80.4, +84.2] (McNemar 1,964 vs 0, p ≈ 0)
- 최종 발산 D10: **+40.5 m [+37.3, +43.9]** (Wilcoxon p ≈ 0), FDE5 +39.9 m [+37.0, +43.1]
- reverse − normal: 증폭 +18.5%p [+15.6, +21.7], p = 2e-78

#### A.7 motion semantics (Experiment E; `CDE.all_units.motion`)
- **GT 근처 값 ≈ GT** (near_gt − recent_gt): 증폭 −0.08%p [−0.60, +0.30], **p = 0.87**
- **방향 오답 − 크기 오답**:
  - 증폭 **+15.6%p [+9.0, +24.3]**, p = 2.0e-31
  - FDE +14.5 m [+12.7, +15.8]
  - D10 +18.4 m [+15.9, +20.1], p = 1.8e-268
- (탐색적) 오차 크기 e를 맞춘 첫 대체 시점(waypoint 3)에서도 방향 오답이 다음 waypoint 오차를 **+0.29 m [+0.25, +0.33]** 더 키웁니다 (Wilcoxon p = 2.0e-99, N = 2,398; `exploratory_matched_first_substitution.json`).

#### A.8 사전 등록 기준 (PROTOCOL §5)

| 기준 | 결과 |
|---|---|
| 1 correction | PASS (−62.2 / −63.4%p) |
| 2 reverse | PASS (+81.9%p, +40.5 m) |
| 3 branch > 재실행 변동 | **FAIL** (2.2 / 3.2% < 5.9%) |
| 4 window | PASS |
| 5 방향 > 크기 | PASS |
| task 수준 | PASS (A− +24.8%p vs 기저 5.9%, 교정이 증폭 감소) |

→ **Strong replication**. 단서는 세 가지입니다: open-loop 궤적 실패라는 점, 궤적 수준의 흡수가 없다는 점, GT 정보 누출입니다.

---

### B. OpenVLA-7B / LIBERO-Spatial (실험 28) — **구조적 대조군(architectural control)**

#### B.1 설정
- 사전 등록 6b9e8ad. 재실행 대조군 amendment 6d0375a는 Phase B 일부를 본 뒤·최종 분석 전에 추가됐습니다. 결과 commit 69402c6.
- 모델: `openvla/openvla-7b-finetuned-libero-spatial` (bf16, RTX 5090).
- 과제: LIBERO-Spatial **10 과제 × 초기 상태 0–9 = 100 에피소드**.
- 구조: step마다 7개 token(x, y, z, roll, pitch, yaw, gripper)을 차원 순서로 생성하고, **다음 step 입력에 이전 action이 없습니다**. 따라서 feedback은 step 안의 차원 간 조건화뿐입니다.
- 통계 (`scripts/cross_vla/analyze_cross_vla.py`):
  - 에피소드 paired McNemar exact
  - 과제 단위 cluster bootstrap (2,000, seed 0)
  - 과제 단위 exact sign-flip permutation (2^10)
  - Phase C는 과제 단위 bootstrap **1,000회**

#### B.2 기본 성능과 재실행
- 자연 성공: **87/100 = 87%** (`natural_success`). 과제별 10, 10, 10, 9, 7, 9, 9, 7, 9, 7.
- 공식 보고 84.7%는 `EXPERIMENT_SUMMARY.md` §6에만 있습니다 (conflicts m2).
- 재실행(natural_rep, batch 구성만 다름): **81/100**
  - 차이 −6%p [−14, +2], McNemar 7 vs 13, p = 0.26, perm p = 0.30
  - **결과 불일치 20/100 = 20%**
  - 자연 성공 87개 중 재실행에서 실패한 비율 13/87 = 15%

#### B.3 step 안 token feedback (Phase C, 2,462 프레임, 시뮬레이션 없음; `phaseC`)

| 개입 (d = 0, x) | N 단위 | 증폭 (하위 차원 \|Δbin\| ≥ \|δ\|) [95% CI] | 하위 편차 D (bin) | 회복 |
|---|---|---|---|---|
| feedback, \|δ\| = 8 | 4,741 | **92.6% [89.8, 94.4]** | 87.7 | 1.5% |
| feedback, \|δ\| = 24 | 4,741 | **81.2% [77.2, 84.0]** | 106.5 | 0.2% |
| corrected (교란 실행 + 정상 문맥) | 4,741 | 0% | 0 | 100% |
| reverse (정상 실행 + 교란 문맥) | 4,741 | 92.6% (feedback과 같음) | 87.7 | 1.5% |
| d = 1 (y), \|δ\| = 8 / 24 | 4,869 | 87.4% [85.4, 89.2] / 75.3% [72.4, 77.8] | 64.2 / 83.9 | 4.2 / 1.3% |
| 차원 축 "window" 1/2/3/4 (x, 8) | 4,741 | 84.5 / 66.9 / 53.4 / 26.8% | 62.0 / 36.7 / 23.6 / 8.5 | 5.6 / 19.5 / 30.7 / 60.1% |

- 하위 token은 **문맥에 들어간 token이 결정**합니다. 이 결과는 결정적 greedy 디코딩에서 정의상 그렇게 됩니다: reverse = feedback, corrected = 0.
- closed loop 교란 구간 안에서 같은 관측의 자연 디코딩과 비교해도 같은 결과입니다 (`closed_loop_window_token_stats`): feedback_d8 증폭 79.3%, reverse_d8 76.2%, corrected 0% (각 1,000 step).
- 차원 축 window는 시간축 window가 아니므로 AutoVLA·Impromptu의 window와 비교할 수 없습니다.

#### B.4 과제 수준 (Phase A/B, 교란 control step 10–19, 각 100 에피소드)

| 대비 | 성공률 a vs b | Δ [과제 bootstrap 95% CI] | McNemar (a only / b only), p | 과제 perm p | 불일치 |
|---|---|---|---|---|---|
| feedback_d8 − natural | 82 vs 87 | −5%p [−13, +2] | 6/11, 0.33 | 0.39 | 17% |
| feedback_d24 − natural | 84 vs 87 | −3%p [−14, +8] | 8/11, 0.65 | 0.74 | 19% |
| corrected_d8 − natural | 79 vs 87 | −8%p [−15, 0] | 6/14, **0.115** | 0.14 | 20% |
| corrected_d24 − natural | 79 vs 87 | −8%p [−17, +1] | 8/16, 0.15 | 0.21 | 24% |
| reverse_d8 − natural | 84 vs 87 | −3%p [−9, +3] | 5/8, 0.58 | 0.56 | 13% |
| reverse_d24 − natural | 79 vs 87 | −8%p [−15, 0] | 7/15, 0.13 | 0.14 | 22% |
| **feedback − corrected** (d8 / d24) | 82 vs 79 / 84 vs 79 | +3%p [−4, +9] / +5%p [−4, +14] | 11/8, 0.65 / 12/7, 0.36 | 0.59 / 0.43 | – |
| natural_rep − natural (대조군) | 81 vs 87 | −6%p [−14, +2] | 7/13, 0.26 | 0.30 | **20%** |
| 각 조건 − natural_rep | – | −2 … +3%p | p ≥ 0.63 | ≥ 0.45 | – |

- 자연 성공 87개 중 교란 후 실패 비율은 9.2–18.4%입니다. 교란 없는 재실행의 실패 비율은 15%입니다.

#### B.5 OpenVLA가 구조적 대조군이라는 수치 근거
1. **step 안 feedback은 강합니다.** x token 8 bin 교란에서 하위 차원 증폭이 92.6%입니다. 이는 AutoVLA·Impromptu의 문맥 의존성과 같은 "문맥 token이 다음 token을 결정"하는 구조입니다.
2. **시간축으로는 전달되지 않습니다.**
   - 모든 교란 조건이 natural 대비 −3 ~ −8%p이고, McNemar p ≥ 0.115, 과제 permutation p ≥ 0.14입니다.
   - feedback − corrected는 +3 / +5%p이고 CI가 0을 포함합니다.
   - 이 크기는 **교란 없는 재실행의 차이(−6%p, 불일치 20%)와 같은 범위**입니다.
3. 시간축 증폭이 없는 모델에서는 token 수준 효과가 크더라도(92.6%) 과제 수준 효과가 측정되지 않습니다. 이는 "시간축 action history가 증폭의 조건"이라는 해석과 맞습니다. 다만 검정력에 한계가 있습니다(100 에피소드, 과제 10개).

---

### C. SpatialVLA-4B / SimplerEnv (실험 34) — **Partial replication** (token Strong, task 없음)

#### C.1 Architecture (코드 확인)
- 실험 정보: 사전 등록 f578eae, 결과 f4cf886. RTX 5090.
- checkpoint: `IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal`
- backbone: PaliGemma2-3B (Gemma2) + SigLIP + Ego3D(ZoeDepth)
- 출력: **4-step chunk**, step당 **3 token**:
  - translation: 구면 bin θ16 × φ32 × r8 = 4,096
  - rotation: 16³
  - gripper: 2
- 디코딩: prefix 뒤 suffix를 causal greedy로 생성합니다. step k가 step < k의 token을 attend합니다.
- 실행 방식 (공식 adapter `spatialvla_policy.PolicyState` 그대로):
  - **매 control step 재계획**
  - 실행 action = 최근 4개 chunk에서 현재 step에 해당하는 예측의 `ActionEnsembler(4, −0.8)` 가중 평균
  - sticky gripper (10회 반복)
- **ensemble 가중치** (recomputed: `svla_token_step_change.py`, 규칙 exp(0.8·i)/Σ, index 0 = 가장 오래된 예측):

  | 예측 | 가중치 |
  |---|---|
  | 현재 chunk step 1 | **0.574** |
  | 1 step 전 chunk step 2 | **0.258** |
  | 2 step 전 chunk step 3 | **0.116** |
  | 3 step 전 chunk step 4 | **0.052** |

  - 앞 step 문맥에 조건화된 step 2–4가 실행 action에서 차지하는 몫은 **42.6%**입니다.
- **정정 기록**:
  - 실험 34 PROTOCOL.md §0의 "오래된 예측일수록 가중치 큼 … 실행 action의 대부분은 step 2–4에서 온다"는 **틀렸습니다**. 실제로는 최신 예측이 가장 크고, step 2–4의 몫은 42.6%입니다.
  - 실험 34는 공식 코드를 그대로 실행했으므로 결과에는 영향이 없습니다.
  - 같은 정정이 실험 35 사전 등록(`outputs/spatialvla_feedback_protection_ablation/PROTOCOL.md` §0-1, commit 8fc20bf)에 기록돼 있습니다. 실험 34 파일은 수정하지 않았습니다.

#### C.2 Token 수준 (오프라인, `token_level/units.jsonl`, `analysis.json` `token_level`)
- 규모: 1,600 프레임(80 에피소드의 natural rollout에서 4 step마다), **18,975 단위** (프레임 × d ∈ {0.15, 0.30} × 6 방향, 달성 거리 필터 후).
- 통계: 에피소드(task, seed) cluster bootstrap (2,000, seed 0), McNemar, Wilcoxon.
- 수치 noise: 같은 관측에서 `generate`와 step-wise 디코더의 chunk가 일치한 비율은 92.9%입니다 (불일치 7.1%).

| 지표 | 값 |
|---|---|
| step 2/3/4 translation token 변경률 (normal vs R) | **55.6 / 36.0 / 27.9%** (10,542 / 6,824 / 5,286 of 18,975; recomputed: `svla_token_step_change.py`; 원본에 CI 없음) |
| 증폭 (A ≥ 1) normal | **22.1% [19.5, 24.7]** |
| 흡수 (step 2–4 token = R) | **36.7% [33.1, 40.2]** |
| 같은 프레임에서 흡수·증폭 방향 공존 | d 0.15: **25.1% [21.8, 28.7]**<br>d 0.30: **15.3% [12.8, 17.9]** (N = 1,600 프레임) |
| log A 분산 중 프레임 간 비율 | 60% / 64% |

교정·reverse·window (전체 18,975 단위):

| 행 | 증폭 [95% CI] | 흡수 | D | step-4 편차 [95% CI] |
|---|---|---|---|---|
| normal | 22.1% [19.5, 24.7] | 36.7% | 0.191 | 0.0561 [0.0499, 0.0626] |
| recent_ref | 13.7% [11.8, 15.8] | 37.5% | 0.115 | 0.0162 [0.0138, 0.0191] |
| full_ref | 13.5% [11.7, 15.4] | 37.6% | 0.109 | 0.0098 [0.0078, 0.0122] |
| win1 | 15.4% [13.4, 17.6] | 37.4% | 0.126 | 0.0271 [0.0234, 0.0315] |
| reverse | 18.6% [16.3, 20.9] | 36.8% | 0.155 | **0.0566** [0.0504, 0.0632] |

- **교정 효과**:
  - recent_ref − normal: 증폭 **−8.3%p [−9.3, −7.4]**, p = 6.7e-261. step-4 편차 −0.040 [−0.045, −0.035]
  - full_ref − normal: 증폭 −8.5%p [−9.6, −7.5], p = 2.0e-273. step-4 편차 **−0.046 [−0.051, −0.041] (−82%)**, D −43%
  - normal에서 증폭된 4,185 단위만 보면 증폭이 100% → 52.9% (full_ref)로 줄어듭니다.
- **reverse 효과**:
  - reverse − full_ref: 증폭 **+5.1%p [+4.4, +5.8]** (1,180 vs 213, p = 1.9e-162). step-4 편차 **+0.047 [+0.042, +0.052]** (Wilcoxon p ≈ 0)
  - reverse의 step-4 편차는 0.0566으로, normal 0.0561 수준까지 완전히 복원됩니다.
  - 증폭된 단위만 보면 reverse − full_ref = +23.6%p [+21.3, +26.0]입니다.
- **window**: full_ref의 step-4 편차 감소분을 기준으로 한 회복률입니다.
  - 1 step 전(recent_ref) **86.3%**. RESULTS에는 87%로 적혀 있습니다 (conflicts m4).
  - 2 step 전만(win1) **62.6%**
  - win1 − full_ref: 증폭 +1.9%p [+1.5, +2.3], p = 6.3e-68
  - chunk가 4 step이라 3 step 이상 전의 window는 측정할 수 없습니다.
- **방향 vs 크기** (10,542 motion 단위):
  - dir_wrong_mag_ok − dir_ok_mag_wrong: 증폭 **+1.6%p [+0.4, +2.8]**, p = 7.3e-5. step-4 편차 +0.013 [+0.009, +0.017], p = 5.5e-17
  - 방향 오답 대체는 R과의 거리가 0.120으로, 크기 오답(0.175)보다 R에 더 가깝습니다.
  - near_ref − ref_trans: 증폭 +4.3%p [+3.2, +5.6], p = 4.2e-31. token identity에 민감하다는 뜻이며, Impromptu의 near_gt ≈ GT와 다릅니다.
- **token 수준 사전 등록 기준**: (1) 교정 PASS, (2) reverse PASS, (3) 자연 변동보다 큼 PASS, window PASS.

#### C.3 Closed loop (SimplerEnv, `closed_loop/episodes.jsonl`, 80 에피소드 × 9 조건)
- 과제·평가: pick_coke_can과 move_near, 환경 seed 0–39, 80 step, 성공 = 마지막 step의 `done`.
- 개입: control step 8–23 동안 생성되는 모든 chunk에 적용합니다. step-1 translation token을 d = 0.3, perp_left 또는 opposite 방향으로 바꿉니다.
  - feedback: 실행 p, 문맥 p
  - corrected: 실행 p, 문맥 g
  - reverse: 실행 g, 문맥 p
- 아래 표는 모두 recomputed입니다 (`_parts/scripts/svla_closed_loop_paired.py`). analyze_svla.py의 `boot`(에피소드 cluster, 층화 없음, 2,000회, seed 0)와 `paired`(McNemar exact, Wilcoxon)를 **import해서 그대로** 썼습니다. 전체 합계 값은 `analysis.json` `closed_loop.*`와 완전히 같습니다.
- 궤적 발산 = 실행된 world_vector 누적합의 끝점 간 L2 거리입니다.

**전체 80 에피소드 (natural = 68/80 = 85.0% [77.5, 92.5])**

| 조건 | 성공 n/N (%) | Δ vs natural [95% CI] | McNemar (cond only / natural only), p | 성공 불일치 | 끝점 발산 vs natural (m) [95% CI] |
|---|---|---|---|---|---|
| natural_mix | 68/80 (85.00%) | +0.00 [0, 0] | 0/0, 1 | 0.00% | 0.003 [0.000, 0.007] |
| natural_gen (재실행) | 75/80 (93.75%) | +8.75 [+1.25, +16.25] | 9/2, 0.065 | **13.75%** | **0.140 [0.100, 0.185]** |
| feedback_perp_left | 69/80 (86.25%) | +1.25 [−8.75, +11.25] | 10/9, 1 | 23.75% | 0.422 [0.380, 0.471] |
| corrected_perp_left | 73/80 (91.25%) | +6.25 [−1.25, +13.75] | 8/3, 0.23 | 13.75% | 0.416 [0.370, 0.464] |
| reverse_perp_left | 72/80 (90.00%) | +5.00 [−3.75, +13.75] | 8/4, 0.39 | 15.00% | 0.209 [0.171, 0.252] |
| feedback_opposite | 52/80 (65.00%) | **−20.00 [−31.25, −8.75]** | 5/21, **0.0025** | 32.50% | 0.437 [0.378, 0.500] |
| corrected_opposite | 51/80 (63.75%) | **−21.25 [−33.75, −7.50]** | 9/26, **0.0060** | 43.75% | 0.423 [0.370, 0.482] |
| reverse_opposite | 67/80 (83.75%) | −1.25 [−11.25, +8.75] | 8/9, 1 | 21.25% | 0.232 [0.184, 0.285] |

**과제별** (각 40 에피소드; natural: coke 37/40 = 92.5%, move near 31/40 = 77.5%)

| 조건 | coke: 성공, Δ [CI], p | move near: 성공, Δ [CI], p |
|---|---|---|
| natural_gen | 39/40, +5.0 [0.0, +12.5], 0.50; 불일치 5.0% | 36/40, +12.5 [−2.5, +27.5], 0.18; 불일치 22.5% |
| feedback_perp_left | 38/40, +2.5 [−10.0, +12.5], 1 | 31/40, 0.0 [−17.5, +17.5], 1 |
| corrected_perp_left | 38/40, +2.5 [−5.0, +12.5], 1 | 35/40, +10.0 [−5.0, +22.5], 0.29 |
| reverse_perp_left | 39/40, +5.0 [−2.5, +15.0], 0.63 | 33/40, +5.0 [−7.5, +20.0], 0.73 |
| feedback_opposite | 22/40, **−37.5 [−52.5, −22.5], 6.1e-5** | 30/40, −2.5 [−20.0, +12.5], 1 |
| corrected_opposite | 25/40, **−30.0 [−47.5, −12.5], 0.004** | 26/40, −12.5 [−35.0, +10.0], 0.36 |
| reverse_opposite | 33/40, −10.0 [−22.5, 0.0], 0.22 | 34/40, +7.5 [−10.0, +22.5], 0.55 |

- opposite 교란으로 생긴 실패는 주로 pick_coke_can에서 나옵니다. RESULTS에는 없는 분해이고, 과제별 검정력은 낮습니다.

#### C.4 효과 분리: "실행 action 효과" vs "문맥 feedback 효과" (전체 80 에피소드, recomputed)

| 효과 유형 | 대비 | 성공 a vs b | Δ [95% CI] | McNemar a/b, p | 불일치 | a–b 끝점 거리 (m) | Δ(발산 vs natural) [CI], Wilcoxon p |
|---|---|---|---|---|---|---|---|
| **문맥 feedback** | **Feedback(opp) − Corrected(opp)** | 52 vs 51 | **+1.25 [−12.50, +13.75]** | 14/13, **1** | 33.75% | 0.388 [0.305, 0.484] | +0.013 [−0.052, +0.079], 0.73 |
| **문맥 feedback** | **Reverse(opp) − Natural** | 67 vs 68 | **−1.25 [−11.25, +8.75]** | 8/9, **1** | 21.25% | 0.232 [0.184, 0.285] | – |
| **문맥 feedback** | **Feedback(left) − Corrected(left)** | 69 vs 73 | **−5.00 [−12.50, +1.25]** | 2/6, **0.29** | 10.00% | 0.227 [0.182, 0.276] | +0.006 [−0.029, +0.041], 0.55 |
| **문맥 feedback** | **Reverse(left) − Natural** | 72 vs 68 | **+5.00 [−3.75, +13.75]** | 8/4, **0.39** | 15.00% | 0.209 [0.171, 0.252] | – |
| 실행 action | Corrected(opp) − Natural | 51 vs 68 | **−21.25 [−33.75, −7.50]** | 9/26, **0.006** | 43.75% | 0.423 [0.370, 0.482] | – |
| 실행 action | Feedback(opp) − Reverse(opp) | 52 vs 67 | **−18.75 [−31.25, −7.50]** | 5/20, **0.004** | 31.25% | 0.445 [0.378, 0.521] | +0.204 [+0.140, +0.270], 2.0e-8 |
| 실행 action | Corrected(left) − Natural | 73 vs 68 | +6.25 [−1.25, +13.75] | 8/3, 0.23 | 13.75% | 0.416 [0.370, 0.464] | – |
| 실행 action | Feedback(left) − Reverse(left) | 69 vs 72 | −3.75 [−12.50, +5.00] | 5/8, 0.58 | 16.25% | 0.398 [0.356, 0.446] | +0.212 [+0.162, +0.269], 3.0e-10 |
| (사전 등록) | Corrected − Feedback (opp / left) | | −1.25 [−13.75, +12.50] p = 1 / +5.00 [−1.25, +12.50] p = 0.29 | | | | |
| (사전 등록) | Reverse − Corrected (opp / left) | | **+20.00 [+7.50, +31.25] p = 0.0037** / −1.25 [−8.75, +6.25] p = 1 | | | | |

- 이 네 대비는 analyze_svla.py에 정의된 것과 같은 방법입니다. Corrected−Feedback과 Reverse−Corrected는 `analysis.json`의 `corrected_vs_feedback_*`, `reverse_vs_corrected_*`와 같고, Reverse−Natural은 `vs_natural.reverse_*`와 같습니다.
- Feedback−Corrected는 Corrected−Feedback의 부호만 바꾼 값입니다.
- 추가한 대비 (원 분석에 없음; 같은 지표, 같은 `paired` Wilcoxon과 에피소드 bootstrap): 문맥만 교란했을 때의 궤적 발산 − 재실행 발산
  - reverse_opposite **+0.093 m [+0.047, +0.141]**, p = 1.7e-5
  - reverse_perp_left **+0.070 m [+0.017, +0.121]**, p = 5.0e-4
  - 즉 문맥 feedback은 **궤적**을 재실행 noise보다 유의하게 바꿉니다.
- 과제별 문맥 feedback 대비 (Feedback − Corrected, Reverse − Natural)는 coke와 move near 모두 p ≥ 0.22로 유의한 것이 없습니다 (`spatialvla_closed_loop_paired.csv`).

해석:
- **실행 action 효과는 큽니다.** opposite 방향에서 교란 action을 실행하면, 문맥을 교정하든 안 하든 성공률이 약 20%p 떨어집니다.
- **문맥 feedback 효과는 성공률에서는 0과 구별되지 않습니다.**
  - 네 대비가 모두 |Δ| ≤ 5%p, p ≥ 0.29입니다.
  - CI 폭은 ±10–13%p로, 약 10%p 미만의 효과는 검출할 수 없습니다.
  - 재실행 불일치는 13.75%입니다.
- **문맥 feedback은 궤적에는 영향을 줍니다.** reverse의 발산은 0.21–0.23 m로, 재실행 0.14 m보다 큽니다.
- **closed-loop 사전 등록 기준**: (1) 교정 FAIL, (2) reverse FAIL → **Partial replication**.

#### C.5 Cross-model 요약 (AutoVLA / Impromptu / OpenVLA / SpatialVLA)
- **token 수준 인과 구조는 네 모델 모두에서 관찰됩니다.** 문맥 token이 다음 token을 결정하고, 교정하면 효과가 줄고, 다시 넣으면 돌아옵니다.
- **open-loop 궤적·task 효과로 이어진 것은 driving 두 모델뿐입니다.**
  - AutoVLA: −40.3%p
  - Impromptu: −62.2%p
- **closed loop에서 문맥 feedback의 task 효과는 검출되지 않았습니다.**
  - OpenVLA: feedback − corrected +3 / +5%p, p ≥ 0.36. 시간축 history가 없습니다.
  - SpatialVLA: |Δ| ≤ 5%p, p ≥ 0.29. 4-step chunk, 매 step 재계획, ensemble에서 step 2–4의 몫은 42.6%입니다.
- 두 closed-loop 모델 모두 재실행 noise가 큽니다(불일치 20%, 13.75%).
- 이 결과는 "기전은 일반적이지만, task 실패로 이어지는지는 구조에 따라 다르다"는 해석을 지지합니다. 반대로 closed-loop task 효과가 없다는 것은 검정력 한계(N = 80–100) 안에서의 결론입니다.

---

## 불안정 rollout 탐지 및 후보 선택·완화 — 논문용 정량 근거 (실험 12–24, 31/P4)

작성 2026-10-06. 모든 수치는 `autovla_misalignment_poc/outputs/<실험>/` 아래 기존 보고서·summary JSON에서 그대로 옮겼습니다.
재계산한 값은 "(recomputed: <script>)"로 표시했고, 헬퍼 스크립트는 `_parts/scripts/`에 있습니다. 새 실험·새 threshold·새 지표는 없습니다.
경로의 `outputs/`는 `/root/VLA/autovla_misalignment_poc/outputs/`입니다. 보고서에 없고 같은 방법으로 재계산할 수도 없는 값은 "not reported"로 적었습니다.

공통 통계 방법(각 스크립트에서 확인):
- 95% CI = **log 단위 cluster bootstrap**, percentile 2.5/97.5. 반복 횟수는 스크립트마다 다릅니다: 선택·PDMS·후보 수 2,000회(`random.Random(0)`), P4 탐지 1,000회, 실험 22 AUROC 500회.
- 쌍대 검정: 이진 지표는 McNemar(정확 이항검정, `binomtest`), 연속 지표는 Wilcoxon signed-rank. 기준은 언제나 같은 장면·같은 실행 안의 자연 계획(후보 0, T = 0.01)입니다.
- A− = P/R/A 5 s 궤적 실패(`pra_labels`), "강한 실패/증폭" = A− ∧ FDE5 > 3 m.

---

### 1. 불안정 rollout 탐지: 이탈 전에는 예측되지 않는다 (실험 31/P4, 참고로 실험 22)

**설계** (`outputs/instability_detection_baselines/PROTOCOL.md`, 실행 전 커밋 bf9aa21; probe 격자 축소는 probe 결과 전 커밋 9b91b2d).
데이터는 실험 20–24의 **5090 전용** 디코딩 후보입니다(새 샘플링 없음). dev = `expanded_best_of_n/gpu1` + `expanded_best_of_n_5090/gpu1`(56 log, 4,563 장면),
held-out = `heldout_best_of_n/gpu1` + `heldout_best_of_n_5090/gpu1`(52 log, 4,814 장면). 단위는 장면당 후보 17개(자연 계획 + T 1.0 샘플 16개) 중
GT token에서 t* < 9에 처음 벗어나는 후보, 라벨은 증폭입니다. pre = t*를 내는 step까지(포함), post = t*+1..9.
heuristic 탐지기는 부호를 미리 고정했고, hidden-state probe(layer 18/36 후보, 세 probe 모두 layer 18 선택)는 dev에서만 학습해 held-out에 한 번 적용했습니다.
margin과 hidden state는 같은 seed로 기록된 token을 teacher forcing해 얻었습니다(RTX 5090; 기록된 log-prob와의 차이 중앙값 0.03, 오류 0 — RESULTS.md 기재).

- held-out N = **43,469** 이탈 후보, 증폭 **2,538** (유병률 0.0584 = 무작위 탐지기의 AUPRC), 양쪽 클래스가 있는 장면 1,091개.
- dev N = **42,095**, 증폭 2,885 (유병률 0.0685), 혼합 장면 1,093개.

**표 1. held-out 탐지 성능** (`outputs/instability_detection_baselines/detection.json`, 키 `heldout.<detector>_{pre,post}`; 파일 `paper_detection_table.csv/.tex`)

| 탐지기 (json 키) | pre AUROC [95% CI] | pre 장면 내 | post AUROC [95% CI] | post 장면 내 | post AUPRC [95% CI] | dev pre / post AUROC |
|---|---|---:|---|---:|---|---|
| entropy | 0.634 [0.607, 0.658] | 0.516 | 0.823 [0.809, 0.842] | 0.786 | 0.215 [0.194, 0.243] | 0.608 / 0.779 |
| log-likelihood (`loglik`) | 0.570 [0.549, 0.587] | 0.596 | 0.832 [0.820, 0.846] | 0.811 | 0.246 [0.225, 0.270] | 0.536 / 0.788 |
| margin | 0.589 [0.561, 0.615] | 0.504 | 0.769 [0.755, 0.786] | 0.716 | 0.132 [0.119, 0.147] | 0.566 / 0.734 |
| candidate variance (`cand_variance`, 장면 단위) | 0.582 [0.566, 0.598] | (0.432)† | 0.797 [0.786, 0.810] | (0.431)† | 0.157 [0.139, 0.181] | 0.558 / 0.777 |
| pairwise disagreement (`disagreement`) | 0.620 [0.604, 0.637] | 0.539 | **0.900 [0.890, 0.913]** | 0.898 | 0.381 [0.350, 0.419] | 0.584 / 0.861 |
| medoid distance (`medoid_dist`) | 0.630 [0.610, 0.652] | **0.649** | 0.883 [0.866, 0.901] | **0.904** | **0.403 [0.368, 0.440]** | 0.583 / 0.826 |
| hidden logistic (`probe_logreg`, L18, C = 1e-3) | 0.631 [0.611, 0.651] | 0.549 | 0.765 [0.747, 0.786] | 0.754 | 0.198 [0.175, 0.231] | CV 0.632 / 0.705 |
| hidden ridge (`probe_ridge`, L18, α = 1e4) | 0.611 [0.591, 0.630] | 0.539 | 0.745 [0.726, 0.768] | 0.745 | 0.182 [0.161, 0.212] | CV 0.613 / 0.687 |
| hidden MLP (`probe_mlp`, L18, α = 1e-2) | 0.622 [0.601, 0.643] | 0.549 | 0.749 [0.726, 0.772] | 0.736 | 0.175 [0.150, 0.207] | CV 0.615 / 0.720 |

† 장면 안에서 거의 상수인 점수라 장면 내 AUROC는 의미가 없습니다(RESULTS.md도 괄호로 표시).
probe의 dev 값은 dev log-grouped 3-fold CV AUROC(선택용)이며, probe의 dev pooled/장면 내 AUROC는 **not reported**입니다.
장면 내 AUROC의 CI와 탐지기 간 쌍대 AUROC 검정(DeLong 등)은 **not reported**입니다(모든 탐지기가 같은 후보에서 평가된 쌍대 설계이지만 검정은 하지 않았음).

**해석.**
1. 이탈 전 held-out pooled AUROC는 **0.570–0.634**, 장면 내 0.504–0.649(후보 분산 제외)입니다. 모델 내부 표현(hidden probe 0.611–0.631)도 엔트로피(0.634)보다 낫지 않습니다.
2. 이탈 후에는 **0.745–0.900**으로 올라갑니다. 후보 간 불일치(disagreement 0.900, medoid 0.883; 장면 내 0.898/0.904)가 가장 강하고,
   확신도(log-lik 0.832, entropy 0.823)가 다음, hidden probe(0.745–0.765)가 가장 낮습니다.
3. 주의: "pre"는 이탈 token(t*)을 내는 step까지 포함하므로 엄밀히는 "이탈 시점까지"입니다. 이 정의는 pre 신호에 오히려 유리하므로 "이탈 전에는 예측되지 않는다"는 결론을 보수적으로 만듭니다.
4. "장면 내 0.50–0.65 = 우연 수준"이라는 표현에는 검정이 없습니다(CI·p not reported). medoid 거리의 pre 장면 내 0.649는 0.5보다 분명히 높으므로 "약한 신호"로 쓰는 것이 정확합니다.

**실험 22(기전 ↔ 선택 연결)와의 관계.** `outputs/mechanism_selection_link/heldout_5090/summary.json`의 `auroc_amplification.pre_dev_entropy`(0.6335, 장면 내 0.5164)와
`post_dev_entropy`(0.8232, 0.7858)는 P4의 `entropy_pre/post`와 **점추정이 완전히 같습니다**(같은 5090 후보). CI만 [0.608, 0.658] vs [0.607, 0.658]로 약간 다른데,
실험 22는 bootstrap 500회, P4는 1,000회이기 때문입니다. 실험 22의 원래(혼합 GPU) 값은 `mechanism_selection_link/heldout/summary.json`: 이탈 후보 43,517, 증폭 2,550,
pre_dev_entropy 0.631 / 장면 내 0.513 — 모집단(GPU) 차이이며 충돌이 아닙니다.

**표 2. 선택된 후보 vs 같은 장면에서 버려진 후보 (실험 22, 장면 단위 쌍대 차이, log bootstrap 2,000회; p는 not reported)**

| 지표 | dev 혼합 (`mechanism_selection_link/dev`) | held-out 혼합 (`.../heldout`) | held-out 5090 (`.../heldout_5090`) |
|---|---|---|---|
| 이탈 전 엔트로피 선택/버림, 차이 [CI] (n 장면) | 0.518 / 0.542, −0.024 [−0.032, −0.016] (1,419) | 0.509 / 0.525, −0.016 [−0.023, −0.010] (1,464) | 0.509 / 0.525, −0.016 [−0.023, −0.009] (1,471) |
| 이탈 후 엔트로피 | 0.589 / 0.978, −0.389 [−0.411, −0.368] | 0.570 / 0.966, −0.397 [−0.422, −0.371] | 0.570 / 0.966, −0.396 [−0.420, −0.373] |
| 증폭률 (n = 전체 장면) | 1.2% / 3.9%, −2.7%p [−3.1, −2.3] (4,563) | 0.5% / 3.3%, −2.8%p [−3.1, −2.4] (4,814) | 0.5% / 3.3%, −2.8%p [−3.2, −2.5] (4,814) |
| FDE5 (m) | 0.48 / 1.29, −0.81 [−0.87, −0.75] | 0.39 / 1.14, −0.75 [−0.82, −0.69] | 0.39 / 1.13, −0.74 [−0.82, −0.68] |
| rank-sum 점수의 증폭 AUROC, 장면 내 | 0.851 (1,114 장면) | 0.867 (1,096) | 0.869 (1,114) |

이탈 전 엔트로피 차이는 이탈 후 차이의 약 4–6%로 작지만 **CI가 0을 배제**합니다. 따라서 "이탈 전 엔트로피가 같다"(CONCLUSIONS·EXPERIMENT_SUMMARY의 표현)가 아니라
"이탈 전 차이는 작고(−0.016), 이탈 후에 크게 벌어진다(−0.40)"로 써야 합니다(`conflicts_detsel.md` M1).

---

### 2. GT 없는 완화 1: 조건화 교정(reference stabilization)은 순이득이 없다 (실험 12–17, 음성 결과)

공통: equal-distance 집합(A− 52 장면 / 365 단위, A+ 156 장면 / 1,043 단위), 교정 시점 t*는 oracle(실험 15 제외), T = 0.01, seed 0, CI는 log cluster bootstrap.

| 실험 (출력) | GPU | 참조 / 조건 | 실패(A−) 단위 | 정상(A+) 단위 | 판정 |
|---|---|---|---|---|---|
| 12 (`reference_stabilization`) | 5090 | 이전 frame 계획, 4 step (`prev_w4`) | 증폭 47.1 → 33.2%, −14.0%p [−22.3, −0.5], McNemar p = 7e-6, GT 효과의 34% [3, 53] | 4.1 → 3.2%, −1.0%p [−3.0, +2.6] | 부분 효과 |
| 12 | 5090 | CTRA 운동학 외삽, 전체 (`kin_all`) | 57.8%, +10.7%p [+1.1, +16.6] | 16.0%, +11.9%p [+6.5, +15.7] | 해로움 |
| 13 (`receding_horizon_replanning`) | – | 2–4 step 재계획, 실행 상태 되먹임 | A− 5 s 실패 59.6 → 23.5% (−35.3%p [−50.0, −14.3]); 로그 관측 누출 섞임, 8 s 오차 +2.5–5.5 m | – | 혼재 |
| 14 (`consensus_reference_stabilization`) | 3080 Ti | 0.5/1.0/1.5 s 전 세 계획의 medoid 합의 (`consmed_all`) | 27.9% (prev_all 35.1% 대비 −7.1%p [−12.9, −2.7]) | FDE +0.28 m [+0.01, +0.58] | 트레이드오프 |
| 16 (`pdm_reference_stabilization`) | 5090 | PDM-Closed 궤적 (`pdm_all`) | 47.1 → 31.0% (FDE 6.13 m, 거의 불변) | **4.2 → 20.1%** (p = 2e-31) | 해로움 |
| 17 (`robustness_reference_stabilization`) | 5090 | prev_w4, seed 0–2 × T 0.01/0.5 | 6개 실행 모두 44.7–48.5 → 31.2–33.4% (GT 효과의 30–37%) | 해 없음 | 강건 (oracle 시점) |
| **15** (`natural_reference_stabilization`) | 3080 Ti | prev_w4, **교정 시점 없이 상시** | 자연 실패 52 장면 A− 65.4 → 32.7% (−32.7%p [−48.1, −9.7], p = 0.002) | 정상 297 장면 A− 0 → 2.0% (p = 0.03), FDE +0.80 m | **모집단 A− 1.24 → 2.60%로 악화** |

- 모델은 조건화된 참조 motion으로 실행 token을 66–89% 끌어갑니다(`PDM_REFERENCE.md`). 효과는 참조↔GT motion 거리로 결정됩니다(실험 12 w4 삼분위: 가까움 −30.6%p, 멂 +13.1%p).
- 실험 15의 엔트로피 trigger: 인과적으로 쓸 수 있는 첫 step 엔트로피(A− vs 정상 AUROC 0.70)로 고른 정책은 모집단 A− −0.18%p [−0.53, +0.19](n.s.). AUROC 0.88–0.89 신호는 교정 위치보다 뒤를 봐야 얻어집니다.
- 실험 15의 5090 재실행(`natural_reference_stabilization_5090/analysis_console.txt`): 자연 실패 69.2 → 32.7%, 모집단 1.31 → 2.27%. 수치는 다르지만(경계 장면의 GPU 수치 차이) 결론(모집단 악화)은 같습니다.
  실험 14의 5090 재실행: consmed_all 28.8%(Normal 47.1%) — 같은 결론. EXPERIMENT_SUMMARY·CONCLUSIONS는 3080 Ti 원 실행 값을 인용합니다.

---

### 3. GT 없는 완화 2: 모델 자신의 후보 중 확신도로 선택 (실험 18–24)

후보 = 자연 계획(T 0.01) 1개 + T 1.0 샘플 16개(seed 0), 한 batch 디코딩. 주 규칙 **rank-sum** = argmin(평균 엔트로피 순위 + 합산 log-prob 역순위)
(`scripts/analyze_best_of_n.py::pick`). rank-sum은 실험 19의 결과를 본 뒤 정의했고(스크립트 docstring에 명시) 커밋 e6d9dff(09:26 UTC)로 고정한 뒤
실험 20 디코딩(첫 shard 10:05)을 시작했습니다. 실험 20은 형식적 사전 등록 문서는 없지만 규칙은 데이터 전에 커밋되어 있었습니다.

#### 3.1 실험 18–19 (PoC 28 log, 실패 52 + 정상 297 장면, 모집단 가중 52/2,695)

- N 16 T 1.0, 5 seed 합산(`pool_best_of_n.py`): rank-sum 모집단 ΔA− −0.42%p [−0.85, +0.09](n.s.), ΔFDE5 −0.178 m [−0.276, −0.059] (자연 A− 1.51%, FDE 0.582 m) — `BEST_OF_N_16.md`.
- 실패 장면 52개만으로는 실패율 감소가 경계선이었고, 이것이 실험 20(표본 확대)의 동기입니다.

#### 3.2 Dev 56 log (실험 20, `outputs/expanded_best_of_n/`)

navtest shard 6–17, PoC와 겹치지 않음. **4,563 장면, 56 log, 자연 A− 98건**. GPU: 5090 39 log(3,294 장면) + 3080 Ti 17 log(1,269 장면) = "혼합 GPU".
출처 `expanded_best_of_n/summary_pooled.json` (`rules.<rule>.{a_minus,failure,fde5}`), ADE는 아래 표시.

| 규칙 | A− (건수/4,563) | ΔA− [95% CI] | McNemar (규칙만 A− / 자연만 A−), p | 강한 실패 | ADE5 (m) | ΔADE5 [CI], Wilcoxon p | FDE5 (m) | ΔFDE5 [CI], Wilcoxon p | 자연 실패 구제율 |
|---|---|---|---|---:|---:|---|---:|---|---:|
| 자연 계획 | **2.15%** (98) | – | – | 1.80% | **0.313** | – | **0.758** | – | – |
| **rank-sum** | **1.45%** (66) | **−0.70%p [−1.07, −0.33]** | 23 / 55, **p = 3.8e-4** | 1.18% | **0.213** | −0.100 [−0.115, −0.087], 5.6e-75 | **0.484** | −0.274 [−0.314, −0.237], 1.1e-73 | 56.1% |
| max log-lik | 1.47% (67) | −0.68%p [−0.97, −0.39] | 7 / 38, 3.1e-6 | 1.29% | 0.240 | −0.073 [−0.084, −0.062], 4.5e-65 | 0.557 | −0.201 [−0.235, −0.171], 1.2e-64 | 38.8% |
| min entropy | 1.88% (86) | −0.26%p [−0.71, +0.19] | 46 / 58, 0.28 | 1.62% | 0.245 | −0.068 [−0.088, −0.048] | 0.565 | −0.193 [−0.245, −0.140] | 59.2% |
| oracle (FDE 최소, 비배포) | 0.24% (11) | −1.91%p [−2.50, −1.34] | 4 / 91 | 0.11% | 0.119 | −0.194 | 0.191 | −0.567 | 92.9% |

ADE 열은 기존 보고서에 없어 `analyze_expanded_best_of_n.py`와 같은 방법(같은 `pick`·`unit_metrics`·bootstrap seed 0·2,000회·Wilcoxon)으로 재계산했습니다
(recomputed: `_parts/scripts/recompute_ade_expanded.py` → `_parts/recomputed_ade_dev_mixed.json`). 같은 실행에서 A−/FDE5/CI/McNemar가 `summary_pooled.json`과 소수점까지 일치해 방법 동일성을 확인했습니다.
ADE 점추정은 `candidate_count_curve/dev/summary.json`의 `curve.1.ade` = 0.3132, `curve.17.ade` = 0.2128과도 같습니다.

- 상대 감소: A− −32.6%(−0.70/2.15; "−33%"의 근거), 강한 실패 −34%(1.80 → 1.18%), ADE −32%, FDE −36%.
- GPU별(`expanded_best_of_n/gpu{1,0}/summary.json`): 5090 부분 ΔA− −0.85%p [−1.32, −0.40](p = 5e-4), **3080 Ti 부분 −0.32%p [−0.83, +0.13](n.s., p = 0.45)**. FDE 감소는 두 부분 모두 유의(−0.27/−0.28 m).
- 5090 전용 재실행(`gpu5090_reanalysis/dev_5090_summary_pooled.json`): A− 2.15 → 1.42%, ΔA− −0.72%p [−1.09, −0.36], McNemar 23/56 p = 2.6e-4, FDE 0.754 → 0.484 m.
- dev PDMS(실험 21, `pdm_score_best_of_n/dev/summary.json`): 0.8951 → 0.9113, +0.0161 [+0.0114, +0.0206], Wilcoxon p = 1.4e-6. 학습형 선택기(Ridge·로지스틱, 특징 12개, log-grouped 5-fold CV)는 ΔA− −0.55%p / −0.26%p로 rank-sum(−0.70%p)보다 약해 버렸습니다(`learned_selector/dev_cv.json`, held-out 미적용 = held-out "not run").

#### 3.3 Held-out 52 log (실험 21–24, 사전 등록 `heldout_best_of_n/PREREGISTRATION.md`, 커밋 **513fdba** 2026-09-29 20:12 UTC)

navtest shard 18–31, PoC·dev와 겹치지 않음. **4,814 장면, 52 log, 자연 A− 72건**. GPU: 5090 30 log(2,823 장면) + 3080 Ti 22 log(1,991 장면).
사전 등록 판정 기준: rank-sum과 F1의 ΔPDMS 95% CI가 0을 배제 → **둘 다 충족**. PDMS = navsim 기본 채점(4 s, 비반응형).
출처: `pdm_score_best_of_n/heldout/summary.json`, `.../heldout_filter/summary.json` (`subsets.all.<rule>`), open-loop `heldout_best_of_n/summary_pooled.json`.

| 규칙 | PDMS | ΔPDMS [95% CI] | Wilcoxon p | at-fault 충돌 (건수/4,814) | DA 이탈 | TTC 위반 | 진행도 | open-loop A− (건수) |
|---|---:|---|---:|---|---:|---:|---:|---|
| 자연 계획 | **0.8893** | – | – | 0.56% (27) | 4.36% | 2.14% | 0.814 | 1.50% (72) |
| **rank-sum** (주 규칙) | **0.8988** | **+0.0095 [+0.0046, +0.0148]** | **0.0034** | **0.23% (11)** | 3.61% | 1.54% | 0.821 | 0.71% (34) |
| max log-lik | 0.8977 | +0.0084 [+0.0048, +0.0122] | 0.0035 | 0.25% (12) | 3.70% | 1.64% | 0.820 | 0.77% |
| min entropy | 0.8982 | +0.0088 [+0.0040, +0.0142] | 0.047 | 0.39% (19) | 3.57% | 1.68% | 0.819 | 0.89% |
| **F1 = 안전 필터 + rank-sum** (주 필터) | **0.9209** | **+0.0315 [+0.0234, +0.0410]** | **5.0e-17** | **0.10% (5)** | 1.23% | 1.33% | 0.840 | 1.08% |
| F2 (+ 역주행) | 0.9206 | +0.0312 [+0.0229, +0.0409] | 2.1e-16 | 0.10% (5) | 1.25% | – | 0.840 | 1.10% |
| F3 (+ TTC) | 0.9197 | +0.0303 [+0.0220, +0.0399] | 1.3e-14 | 0.15% (7) | 1.33% | – | 0.838 | 1.47% |
| oracle (FDE 최소, 비배포) | 0.9101 | +0.0207 [+0.0143, +0.0281] | 1.0e-9 | 0.17% (8) | 2.58% | 1.20% | 0.831 | 0.17% |

- 충돌률 차이에 대한 쌍대 검정·CI는 **not reported**입니다(PDMS 하위 지표는 평균 위반율만 보고). 건수는 위반율 × 4,814로 복원한 값입니다.
- open-loop(held-out, `summary_pooled.json`): rank-sum A− 1.50 → 0.71%, **−0.79%p [−1.01, −0.55]**, McNemar 10/48, p = 4.5e-7 (상대 −53%);
  FDE5 0.624 → 0.394 m (−0.230 [−0.274, −0.195]); ADE5 0.262 → 0.181 m (−0.081 [−0.096, −0.068], Wilcoxon p = 5.2e-80; recomputed: `recompute_ade_expanded.py` → `recomputed_ade_heldout_mixed.json`, `candidate_count_curve/heldout` curve.1/17 ade와 일치).
- F1 open-loop A− 1.08%(−0.42%p [−0.73, −0.12], `safety_filter/heldout/open_loop_summary.json`): 필터는 PDMS를 크게 올리지만 사람 궤적과의 거리(A−)는 rank-sum 단독보다 덜 줄입니다.
  모든 후보가 제거된 장면 2.7%(F1_all_removed 0.0268)는 전체 rank-sum으로 돌아갑니다. 필터와 PDMS 채점이 같은 규칙(관측만 다름)을 써서 이 지표에 유리합니다.
- 자연 실패 장면(held-out 72개, `subsets.natural_failures`): rank-sum PDMS 0.592 → 0.703 (+0.111 [+0.029, +0.209]), F1 0.797 (+0.205 [+0.122, +0.311]). dev 98개: rank-sum +0.139 [+0.082, +0.192].
- dev(참고): F1 0.9301, +0.0350 [+0.0276, +0.0421]; 충돌 0.77% (35) → rank-sum 0.37% (17) → F1 0.15% (7).

#### 3.4 후보 수 곡선 (실험 23, `candidate_count_curve/`, `pdm_score_best_of_n/*_ncurve/`; 파일 `figure_candidate_count.csv`)

N = 자연 계획을 포함한 후보 수(N = 1은 자연 계획만, N = 17은 전체). open-loop는 장면당 무작위 부분집합 5개 평균, PDMS는 앞쪽 부분집합(자연 + 샘플 1..N−1).
포화 비율 = ΔN / ΔN=17 (단순 비율, derived).

| N | held-out A− | ΔA− [CI] | held-out ΔPDMS [CI] | PDMS 포화 | A− 포화 | dev ΔPDMS | 추론 비용 3080 Ti | 추론 비용 5090 |
|---:|---:|---|---|---:|---:|---|---:|---:|
| 1 | 1.50% | – | – | 0 | 0 | – | 1.00 | 1.00 |
| 2 | 1.37% | −0.13%p [−0.34, +0.08] | +0.0024 [−0.0002, +0.0051] | 0.25 | 0.16 | +0.0028 | not measured | not measured |
| 4 | 1.08% | −0.42%p [−0.67, −0.18] | +0.0060 [+0.0021, +0.0103] | 0.63 | 0.53 | +0.0077 | 1.03 | 1.28 |
| 8 | 0.92% | −0.57%p [−0.82, −0.32] | +0.0081 [+0.0040, +0.0125] | **0.86** | 0.73 | +0.0128 | 1.16 | 1.28 |
| 12 | 0.70% | −0.79%p [−1.02, −0.56] | +0.0084 [+0.0041, +0.0130] | 0.88 | 1.00 | +0.0146 | not measured | not measured |
| 16 | 0.69% | −0.81%p [−1.02, −0.58] | +0.0092 [+0.0047, +0.0140] | 0.97 | 1.02 | +0.0165 | **1.31** | 1.28 |
| 17 | 0.71% | −0.79%p [−1.01, −0.55] | +0.0095 [+0.0046, +0.0148] | 1.00 | 1.00 | +0.0161 | not measured | not measured |

- 추론 비용은 같은 100개 PoC 장면에서 계획 전체(영상 인코딩 + 프롬프트 + stub + action 디코딩)의 실측 시간 비율입니다. N = 17(실제 배포 설정)은 측정하지 않았습니다.
  3080 Ti: N 1 3.1/3.3분(평균 3.2), N 4 3.3, N 8 3.7, N 16 4.2분 — 보고서 값은 `SELECTION_VALIDATION.md`, 원 log는 저장소 밖(scratchpad `timing_N*.log`)에만 있습니다.
  5090: `gpu5090_reanalysis/timing5090_N{1,4,8,16}.log` — N 1 1.8분(1.10 s/장면), N 4/8/16 각 2.3분(1.36 s/장면) → 경과 시간 기준 1.28×, 장면당 1.24×(보고서는 "1.2×").
- N > 17: PoC 28 log 실험 19에만 있습니다(N 32 = 자연 + 32 샘플 = 33 후보, 5090): rank-sum 모집단 A− 1.31 → 0.76%(−0.55%p, CI not reported), 실패 장면 −28.8%p [−40.2, −14.3].
  같은 설정의 N 16(17 후보, 3080 Ti)은 −0.70%p로, 16 → 32에서 추가 이득이 보이지 않습니다. 다른 모집단·단일 seed·다른 GPU라 dev/held-out 곡선과 직접 합치면 안 됩니다.
- 5090 전용 held-out 곡선: N 8 ΔPDMS +0.0086(N 17 이득의 87%), N 16 +0.0096, N 17 +0.0099.

#### 3.5 3080 Ti vs 5090 (`gpu5090_reanalysis/GPU5090_REANALYSIS.md`)

"3080 Ti 결과"라고 부르는 실험 20–24의 원 실행은 실제로는 **혼합 GPU**(dev 5090 39 + 3080 Ti 17 log, held-out 5090 30 + 3080 Ti 22 log)이고,
재실행은 3080 Ti 부분만 5090에서 다시 디코딩해 **5090 전용**으로 만든 것입니다(5090 부분은 같은 파일 재사용).

| 결과 | 혼합 GPU (원 실행, 사전 등록 대상) | 5090 전용 (재실행) | 차이 |
|---|---|---|---|
| dev rank-sum ΔA− | −0.70%p [−1.07, −0.33], p = 3.8e-4 (2.15 → 1.45%) | −0.72%p [−1.09, −0.36], p = 2.6e-4 (2.15 → 1.42%) | 소수점 둘째 자리 |
| dev rank-sum ΔPDMS | +0.0161 [+0.0114, +0.0206] (0.8951 → 0.9113) | +0.0149 [+0.0105, +0.0190] (0.8961 → 0.9110) | 0.0012 |
| held-out rank-sum ΔPDMS | **+0.0095 [+0.0046, +0.0148]**, p = 0.0034 | +0.0099 [+0.0050, +0.0154], p = 0.0037 | 0.0004 |
| held-out F1 ΔPDMS | **+0.0315 [+0.0234, +0.0410]** | +0.0316 [+0.0235, +0.0411] | 0.0001 |
| held-out 충돌 자연 / rank-sum / F1 | 0.56 / 0.23 / 0.10% | 0.56 / 0.23 / 0.10% | 없음 |
| held-out open-loop A− 자연 → rank-sum | 1.50 → 0.71% (72 → 34건) | 1.52 → 0.71% (73 → 34건), −0.81%p [−1.02, −0.59] | 자연 실패 1건 |
| held-out 이탈 후보 / 증폭 (실험 22) | 43,517 / 2,550 | 43,469 / 2,538 (P4가 사용) | 모집단 차이 |
| 실험 15 모집단 A− (자연 → prev_w4) | 3080 Ti 단독: 1.24 → 2.60% | 1.31 → 2.27% | 같은 방향 |

- **정본**: EXPERIMENT_SUMMARY.md §4와 CONCLUSIONS.md §9d는 혼합 GPU 원 실행(사전 등록 513fdba 대상) 값을 주 결과로 쓰고 5090 재실행을 "결론 동일"의 강건성 확인으로 씁니다.
  실험 31(P4)은 5090 전용 후보를 씁니다. 논문에서는 held-out 주 결과를 혼합 GPU(사전 등록 그대로)로, 5090 전용을 재현성 보조로 보고하는 것이 출처와 일치합니다.
- 수치 차이는 경계 장면의 bf16 수치 차이로 greedy 경로가 갈리는 데서 오며(PoC 실패 52 장면의 자연 A−: 3080 Ti 63–65% vs 5090 67–71%), 쌍대 개선 폭은 두 GPU에서 같습니다.

---

### 4. 논문에 쓸 수 있는 문장과 범위

- "불안정 rollout은 이탈 시점까지의 신호로는 거의 예측되지 않는다(held-out pooled AUROC 0.57–0.63; hidden probe 0.61–0.63). 이탈 이후에는 후보 간 불일치가 AUROC 0.90으로 구별한다." — 실험 31 held-out, N = 43,469.
- "확신도 rank-sum 선택은 사전 등록 held-out 52 log에서 PDMS를 +0.0095 [+0.0046, +0.0148] 올렸고, 현재 frame 안전 필터와 결합하면 +0.0315 [+0.0234, +0.0410]이다(비반응형 PDMS)." — 실험 21/24.
- "dev 56 log에서 rank-sum은 open-loop A−를 2.15 → 1.45%(−0.70%p [−1.07, −0.33], McNemar p = 4e-4, 상대 −33%)로 줄였다." — 실험 20, 규칙은 데이터 전 커밋, 형식적 사전 등록은 아님.
- 범위: 모두 open-loop 5 s 또는 비반응형 PDMS(NAVSIM v1)이며, 반응형 navhard(실험 25–30)와 연속 closed-loop는 이 절의 범위가 아닙니다. seed 0 한 번의 후보 샘플입니다.

---

## NAVSIM v2 navhard 2단계 pseudo closed-loop (실험 25, 26, 27, 29, 30) — 논문용 정량 근거

작성: 2026-10-06. 새 실험·채점 없이 기존 결과 파일에서만 뽑았습니다. 모든 평균·Δ·CI·p·그룹 수는 아래 분석 JSON 키에서
그대로 복사했습니다(재계산하지 않음). 직접 계산한 값은 이미 보고된 차이들의 **비율**(89%, 86%, 69% 등)과 token 행렬의 개수뿐이며
"(recomputed: `_parts/scripts/navhard_verify.py`)"로 표시했습니다. 그룹 점수 파일에서 다시 구한 225 그룹 평균은 네 JSON의
값과 모두 일치합니다(차이 ≤ 6e-17, `_parts/navhard_verify.json` → `json_minus_recomputed_mean`).

### 0. 공통 설정

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

### 1. 전체 navhard 표 (76 log, 225 그룹, 모두 같은 225 그룹에서 선택 없음과 짝지은 차이)

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

#### 1.1 세부 지표 (전체 225 그룹, 위반율 %; `RESULTS.md` 실험 26·27·29·30 표와 각 JSON `full.means`)

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

### 2. 두 번째 절반 사전 등록 재현 (실험 26)

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

### 3. 안전 필터 분해 (실험 27 프로토콜 19c0130 → 결과 f98cb10; 실험 29 프로토콜 0cffc15 → 결과 0ab33a4)

#### 3.1 필터 vs 선택기 (실험 27, 225 그룹; `ablation.json` → `full.contrasts`)

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

#### 3.2 충돌 vs 주행 가능 영역 제약 (실험 29, `components.json` → `full.contrasts`)

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

#### 3.3 진행도와 계획 길이 (실험 29)

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

### 4. oracle 상한 (실험 30, 프로토콜 0cffc15 → 결과 86087e0)

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

### 5. 논문 문장 후보 (수치는 위 표 그대로)

- 전체 navhard(76 log, 225 그룹)에서 현재 frame 안전 필터는 EPDMS를 0.229 → 0.333으로 올렸고(+0.105 [+0.085, +0.125], log permutation p < 1e-4, 128/9/88 그룹), rank-sum 선택기를 더한 F1은 0.347(+0.118 [+0.096, +0.140])입니다.
- F1 이득의 88%(0.1046/0.1183)는 필터만으로 얻어지고, rank-sum 선택기의 추가분 +0.014 [−0.002, +0.030]은 유의하지 않습니다(p 0.20).
- 필터 이득의 86%(2인 Shapley)는 주행 가능 영역 제약에서 옵니다.
- 사전 등록한 두 번째 절반(40 log, 120 그룹)에서 F1은 재현되었고(+0.112 [+0.082, +0.140], p < 1e-4), max log-lik은 재현되지 않았습니다(+0.010 [−0.004, +0.024], p 0.08).
- oracle17은 0.405이며 F1+max log-lik은 oracle 이득의 69%(F1 기준 67%)를 얻었습니다. token의 31.6%는 17개 후보가 모두 0점입니다.
- 한계(모든 문장에 동반): 필터와 채점이 같은 PDM 규칙을 쓰고, 2단계 pseudo closed-loop이며 연속 closed-loop이 아니고, 다중 비교 보정이 없습니다.

---

