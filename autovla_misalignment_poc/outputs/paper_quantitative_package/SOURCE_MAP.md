# SOURCE_MAP — 논문 수치 출처 지도

이 문서는 논문에 쓰는 모든 수치를 원 결과 파일까지 거슬러 갈 수 있게 정리한 것입니다.
- 각 항목에 원 결과 파일, raw data, 사전 등록·분석 commit, N, seed, GPU, 지표 정의, 검정, cluster 단위, 통계를 만든 스크립트를 적었습니다.
- 아래 네 절은 각 영역 담당 추출 결과를 그대로 합친 것입니다.
- 경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`, `S/` = `/root/VLA/autovla_misalignment_poc/scripts/`.
- 이 패키지에서 재계산에 쓴 스크립트는 `_parts/scripts/`, 재계산 결과는 `_parts/recomputed*`, `_parts/navhard_verify.json`에 있습니다.

## 실험 번호 → 디렉토리 빠른 색인

| 실험 | 이름 | 모델 / 벤치마크 | 디렉토리 (`O/`) | 사전 등록 | 결과 commit |
|---|---|---|---|---|---|
| 4 | natural mismatch vs failure | AutoVLA / navtest PoC 28 log | `natural_fast_mechanism` | 아니오 | 1541f30 |
| 5 | first-mismatch correction | 〃 | `first_mismatch_causal` | 아니오 | 1541f30 |
| 6 | equal-distance perturbation | 〃 (208 장면) | `equal_distance_perturbation` | 아니오 | 1541f30 |
| 7 | action-history causal | 〃 | `action_history_causal` | 아니오 | 1541f30 |
| 8 | layer/state patching | 〃 | `prev_action_state_patching` | 아니오 | 1541f30 |
| 9 | temporal window | 〃 | `temporal_feedback_window` | 아니오 | 1541f30 |
| 10 | horizon-controlled window | 〃 | `horizon_controlled_window` | 아니오 | 1541f30 |
| 11 | identity vs motion | 〃 | `prev_action_identity_decomposition` | 아니오 | bf15767 |
| 12–17 | reference stabilization (음성 결과) | 〃 | `reference_stabilization*`, `receding_horizon_replanning`, `consensus_*`, `pdm_reference_stabilization`, `robustness_*`, `natural_reference_stabilization*` | 아니오 | bf15767 |
| 18–19 | best-of-N | 〃 | `best_of_n_selection*` | 아니오 | e6d9dff |
| 20 | dev 56 log selection | AutoVLA / navtest shard 6–17 | `expanded_best_of_n` | 규칙만 디코딩 전 commit (e6d9dff) | f4cedfd |
| 21–24 | held-out PDMS, 기전 연결, 후보 수, 안전 필터 | AutoVLA / navtest shard 18–31 | `heldout_best_of_n`, `pdm_score_best_of_n`, `mechanism_selection_link`, `candidate_count_curve`, `safety_filter` | **513fdba** | e3bc8db |
| 25 | navhard 첫 절반 | AutoVLA / NAVSIM v2 navhard | `navhard_eval` | 아니오 | 1b91b0f |
| 26 | navhard 두 번째 절반 + 전체 | 〃 | `navhard_full_validation` | **96a595c** | ad7991c |
| 27 | 안전 필터 ablation | 〃 | `safety_filter_ablation` | 19c0130 | f98cb10 |
| 28 | OpenVLA cross-VLA | OpenVLA-7B / LIBERO-Spatial | `cross_vla_replication` | **6b9e8ad** (+6d0375a) | 69402c6 |
| 29 | 안전 필터 구성요소 | AutoVLA / navhard | `safety_filter_components` | 0cffc15 | 0ab33a4 |
| 30 | candidate oracle | 〃 | `candidate_oracle` | 0cffc15 | 86087e0 |
| 31 | 불안정 탐지 baseline | AutoVLA / held-out | `instability_detection_baselines` | bf9aa21 (+9b91b2d) | 69402c6 |
| 32 | motion semantics | AutoVLA / navtest PoC | `motion_semantics_ablation` | af2eb10 | aee41e5 |
| 33 | Impromptu temporal replication | Impromptu VLA 3B / navtest PoC | `cross_vla_temporal_replication` | **038a6d3** | 698148a |
| 34 | SpatialVLA cross-domain | SpatialVLA-4B / SimplerEnv | `cross_domain_temporal_replication` | **f578eae** | f4cf886 |
| 35 | SpatialVLA 보호 구조 ablation | 〃 | `spatialvla_feedback_protection_ablation` | **8fc20bf** | 실행 중 (결과 없음) |

## 기록해 둘 정정 (원본 파일은 수정하지 않음)

- **실험 34 PROTOCOL.md §0의 ensemble 설명이 틀렸습니다.**
  - 원문: "오래된 예측일수록 가중치 큼, 실행 action의 대부분은 step 2–4에서 온다".
  - 실제(`S/cross_domain_temporal/action_ensemble.py`, temp −0.8): 최신 예측 가중치가 가장 큽니다. 순서대로 현재 chunk step 1이 0.574, 이전 chunk step 2가 0.258, step 3이 0.116, step 4가 0.052입니다. step 2–4의 합은 42.6%입니다.
  - 결과에는 영향이 없습니다. 공식 코드를 그대로 실행했기 때문입니다.
  - 같은 정정이 실험 35 PROTOCOL.md §0-1(8fc20bf)에 기록돼 있습니다.

---


# A. AutoVLA 기전 (실험 4–11, 32)

## AutoVLA 기전 실험 — Source map

경로 약어: `O/` = `/root/VLA/autovla_misalignment_poc/outputs/`, `S/` = `/root/VLA/autovla_misalignment_poc/scripts/`. 저장소 `/root/VLA`, 브랜치 `autovla-experiments-11-19`.
공통 사항
- 모델: AutoVLA, backbone Qwen2.5-VL-3B (언어모델 decoder block 36개), action codebook 2,048 token, natural fast-thinking 경로(실험 2–3 제외). 체크포인트 fp32, `torch.load(mmap=True)`.
- 벤치마크: NAVSIM/nuPlan navtest, PoC 28 log, 전처리 장면 2,748개(arm N fork 성공 2,747개; `autovla_misalignment_poc/REPRODUCTION.md`).
- 디코딩: T = 0.01, top_p 1.0, max_length 2048 (`run_meta.json` `gen_conf`/`temperature`). 단일 seed.
- GPU: 모든 기전 실험의 `run_meta.json` `gpu` = "NVIDIA GeForce RTX 5090", `cuda_visible_devices` = "1"(환경변수 `CUDA_DEVICE_ORDER=PCI_BUS_ID`로 5090 지정). 3080 Ti로 돈 기전 실험은 없음 → `O/gpu5090_reanalysis/`에는 실험 5–11의 재실행이 없고(그 디렉토리는 실험 18–24용), 기전 수치는 5090 원본뿐.
- raw data(`*.jsonl`)는 `.gitignore`(`**/*.jsonl`)로 저장소에 없음, 로컬 디스크에만 존재.
- 사전등록: 실험 32만 실행 전 프로토콜 커밋이 있음(af2eb10). 실험 4–11은 스크립트·결과가 같은 커밋(1541f30, 2026-09-26 13:29 UTC / bf15767, 2026-09-29 08:45 UTC)으로 처음 들어와 **git상 사전등록 증거 없음**("not preregistered"). 보고서 일부가 "실행 전 고정"을 주장하나(예: 실험 4 규칙, 증폭 임계 3.0 m) 커밋으로 검증 불가.
- 통계 공통: log 단위 cluster bootstrap 95% percentile CI(재표본 단위 = navtest log), McNemar exact(binomtest), Wilcoxon signed-rank. equal-distance set의 cluster 수: A− 16 log, 전체 26 log (recomputed: `build_autovla_csvs.py`).

---

#### S1. 실험 4 — Natural/Fast 기전 재검증 (mismatch ≠ failure 기반 수치)
- 모델/벤치: 위 공통; arm N(full extraction, 층별 attn/mlp 성분 저장), 비교 run G(gpu1 natural run), C(Forced-CoT).
- 보고서: `O/natural_fast_mechanism/NATURAL_FAST_MECHANISM.md`
- raw: 저장된 tensor(`O/full_extract/`, arm N), `O/gpu1/`; 이 디렉토리에는 `summary.json`만.
- 사전등록: not preregistered (규칙이 스크립트 상단에 "실행 전 고정"이라고 서술, git 증거 없음). 분석 커밋: 1541f30.
- N: 2,747 장면(arm N), A− 52(16 log), A+ 2,695; G: A− 46; C: 2,604, A− 136.
- seed: 해당 없음(GPU 미사용, 저장 tensor 분석). GPU: 미사용(원 추출은 5090).
- metric: A− = P/R/A coarse-action accept 규칙 5 s 실패(token ID 미사용); 오답 token 변위 = codebook 변위 거리; FDE/첫 오차 = 중앙값 비.
- 검정: AUROC/차이 = log cluster bootstrap 1,000회(`REPS = 1000`), Spearman CI = `_spearman_ci`(log bootstrap 1,000회, `random.Random(3)`), p 미계산.
- 스크립트: `S/natural_fast_mechanism.py` (보고서 생성 `S/make_natural_fast_report.py`). 키: `N.n`, `N.n_Aminus`, `N.n_Aminus_step0_token_wrong`, `N.n_Aplus_step0_token_wrong`, `N.codebook_amplification.*`, `N.verdicts.*`.

#### S2. 실험 5 — 첫 mismatch 1-token 교정
- 보고서: `O/first_mismatch_causal/FIRST_MISMATCH_CAUSAL.md`; 요약 `summary.json`; `run_meta.json`(n_ok 1159, seed 0, 5090, 15.7분).
- raw: records 미보관(디렉토리에 jsonl 없음) → CI 재계산 불가.
- 사전등록: not preregistered. 분석 커밋 1541f30 (실행 스크립트 bf15767에서 GPU 핀 옵션만 수정).
- N: A− 52 장면, A+ 1,107 장면(step-matched 유효 n 944), 총 1,159; 단위 = 장면.
- seed 0; GPU RTX 5090.
- metric: A+ recovery(P/R/A 5 s), downstream token error(t* 이후 GT 불일치 비율), ADE/FDE 5 s, 오차 증가 속도.
- 검정: log cluster bootstrap 2,000회 seed 0(가중 평균 `wmean_boot`), 교호작용 bootstrap seed 1(`p_boot`), McNemar binomtest, Wilcoxon; AUROC CI 1,000회 seed 2.
- 스크립트: `S/first_mismatch_causal.py`, `S/analyze_first_mismatch_causal.py`. 키: `A-.{original,gt,nn}`, `A-.contrasts."gt - original"`, `interaction."vs A+ step-matched"`, `t_star_hist`, `reproduction`, `nn_fragility`.

#### S3. 실험 6 — Equal-distance perturbation
- 보고서: `O/equal_distance_perturbation/EQUAL_DISTANCE.md`, `EQUAL_DISTANCE_strict.md`(STRICT=1); `summary.json`; `run_meta.json`(208 장면, seed 0, rel_levels 0.1/0.2/0.35, abs_tol 0.005 m, min_alts 5, A+ 3/A−, 5090, 6.1분).
- raw: `O/equal_distance_perturbation/records.jsonl`(208), `rows.jsonl`(1,824 = 조건 행), `run.log`.
- strict summary json은 디렉토리에 없음 → `O/paper_quantitative_package/_parts/recomputed/equal_distance_summary_strict_rerun.json` (recomputed: `_parts/scripts/rerun_equal_distance_analysis.sh`, 원 스크립트 무수정, 사본 디렉토리에서 실행). 같은 재실행으로 full summary가 Q4 AUROC(±0.01)를 제외하고 동일함을 확인.
- 사전등록: not preregistered (증폭 임계 3.0 m "사전 고정"은 스크립트 docstring 서술). 분석 커밋 1541f30.
- N: 장면 208 (A− 52 / A+ 156, log 16 / 26), 조건 행 1,824, perturbation 단위(original+대안) A− 365 / A+ 1,043.
- seed 0 (original_reseed는 다른 seed); GPU RTX 5090.
- metric: amplification = A− ∧ FDE5 > 3 m; recovery = P/R/A 5 s A+; 거리 = 첫 불일치 token의 GT 대비 codebook 변위(m).
- 검정: `boot`/`boot_stat` log cluster bootstrap 2,000회 seed 0(Q4는 1,000회), Wilcoxon(Q3, Q5), GroupKFold 로지스틱 AUROC(Q4). Spearman/장면 내 상관 p 미계산.
- 스크립트: `S/equal_distance_perturbation.py`, `S/analyze_equal_distance.py`. 키: `n_scenes`, `alt_rel_err_median`, `alt_abs_dist_diff_median_m`, `table`, `Q1`–`Q5`.

#### S4. 실험 7 — Action history 인과 (Normal / Recent-GT / GT-history / attention mask / embedding 중립화)
- 보고서: `O/action_history_causal/ACTION_HISTORY_CAUSAL.md`; `summary.json`; `run_meta.json`(1,408 perturbation, 7 조건, seed 0, 5090, 19.5분, blocked attention calls 688,104).
- raw: `records.jsonl`(208), `units.jsonl`(1,408).
- 사전등록: not preregistered. 분석 커밋 1541f30; REPRODUCTION.md: 새 서버에서 "모든 조건의 증폭률·FDE 전부 일치".
- N: A− 365 단위 / 52 장면, A+ 1,043 / 156; 부분집합 previously amplified A− 175/45, A+ 43/32; original token only 52/156.
- seed: sha256(`0:token:perturbation:k`) per step; GPU RTX 5090.
- metric: 위 증폭 정의; 판정은 OOD 아닌 조건만(첫 step entropy가 Normal보다 >1 nat 오르면 OOD).
- 검정: `cboot` log cluster 2,000회 seed 0, McNemar binomtest, Wilcoxon; A−/A+ 교호작용 bootstrap seed 3.
- 스크립트: `S/action_history_causal.py`, `S/analyze_action_history.py`. 키: `subsets."all perturbations".A-.conditions.*`, `.vs_normal.*`, `subsets.*.interaction`.
- "93%" = (Normal − Recent-GT)/(Normal − GT-history) 계산은 이 실험 summary에 없음 → 패키지에서 점추정만 재계산, CI는 실험 8 키 `groups.A-.all.fraction_of_gt_history_effect.recent_gt` 사용.

#### S5. 실험 8 — Layer-wise previous-action state patching (+ reverse)
- 보고서: `O/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md`; `summary.json`, `layer_results.json`, `narrative.json`, `sanity/debug_checks.json`, `figures/`.
- raw: jsonl 없음(디렉토리에 records 미보관).
- 사전등록: not preregistered. 분석 커밋 1541f30.
- N: A− 365 / 52, A+ 1,043 / 156; 조건 93행(patch 위치 37 = emb + L0–L35, delta α 3 × 10 layer, selfpatch 3, identitysrc 10, reverse 10, 기준 3).
- seed 0 공식 동일; GPU RTX 5090 (20.5분).
- metric: 실험 7과 동일(`from analyze_action_history import unit_metrics, AMP_FDE`).
- 검정: `Boot` log cluster 2,000회 seed 0(scene cluster 보조 seed 1), McNemar, Wilcoxon.
- 스크립트: `S/prev_action_state_patching.py`, `S/analyze_prev_action_state_patching.py`. 키: `groups.<g>.all.rows.<name>.{level,vs_base}`, `groups.<g>.all.fraction_of_gt_history_effect`, `sanity.*`.

#### S6. 실험 9 — Temporal feedback window
- 보고서: `O/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md`; `summary.json`, `early_step_contribution.json`, `narrative.json`.
- raw: jsonl 없음.
- 사전등록: not preregistered. 분석 커밋 1541f30.
- N: A− 365 / 52, A+ 1,043 / 156 (t* 분포 A− {0: 116, 1: 161, 2: 49, 3: 13, 4: 6, 5: 12, 7: 8}); 17 조건.
- seed 0 공식; GPU RTX 5090 (6.4분).
- metric: f(w) = (Normal − win_w)/(Normal − GT-history); critical window = f ≥ 0.9 최소 w, bootstrap 분포.
- 검정: `Boot` log cluster 2,000회, McNemar, Wilcoxon; 끝 위치 맞춘 비교 seed 11.
- 스크립트: `S/temporal_feedback_window.py`, `S/analyze_temporal_feedback_window.py`. 키: `subsets.<sub>.<g>.rows.<cond>.{level,vs_normal}`, `.fraction`, `.critical_window`.

#### S7. 실험 10 — Horizon-controlled window
- 보고서: `O/horizon_controlled_window/HORIZON_CONTROLLED_WINDOW.md`; `summary.json`, `decay_test.json`, `narrative.json`.
- raw: jsonl 없음 (오류 기록은 보고서에 인용된 `errors.jsonl`, 디렉토리에는 없음).
- 사전등록: not preregistered. 분석 커밋 1541f30.
- N: t* ≤ 1 장면 A− 39 / A+ 116(1 제외), 단위 A− 277 / A+ 772; t* = 0: A− 116 단위 / 17 장면, A+ 336 / 50.
- seed 0 공식; 20 token 생성; GPU RTX 5090 (7.7분).
- metric: 해제 후 정확히 N free step의 FDE/ADE, Normal·Full 같은 pose 구간; amp(>3 m), amp2(>2 m); 재발산 = 해제 시 ≤1 m → N step 뒤 >2 m.
- 검정: `Boot` log cluster 2,000회, McNemar, Wilcoxon; decay = 쌍대 bootstrap 차이.
- 스크립트: `S/horizon_controlled_window.py`, `S/analyze_horizon_controlled_window.py`. 키: `analyses.in10."t*=0".<g>.N.<N>.windows.<w>.*`, `horizon_trend_tstar0`.

#### S8. 실험 11 — 직전 token identity 분해 (motion vs embedding)
- 보고서: `O/prev_action_identity_decomposition/PREV_ACTION_IDENTITY.md`; `summary.json`, `analysis_console.txt`, `figures/`.
- raw: `records.jsonl`(1,408), `units.jsonl`(1,408), `errors.jsonl`(0).
- 사전등록: not preregistered. 실행 2026-09-27(보고서), 스크립트·결과 커밋 bf15767 (2026-09-29).
- N: A− 365 / 52, A+ 1,043 / 156; 9 조건; 미학습 token 410개 제외(cos ≥ 0.999 군집).
- seed 0; GPU RTX 5090, torch 2.8.0+cu128 (4.3분).
- metric: 실험 7과 동일; motion 거리 = codebook 48-d L2; embedding 근접 = cos.
- 검정: `cboot` 2,000회, McNemar, Wilcoxon, `frac_effect`(seed 7), 공동 회귀 log bootstrap 1,000회.
- 스크립트: `S/prev_action_identity_decomposition.py`, `S/analyze_prev_action_identity.py`, `S/make_identity_figures.py`. 키: `groups.<g>.{conditions,vs_normal,vs_recent_gt,frac_of_recent_gt_effect}`, `geometry_vs_embedding`.

#### S9. 실험 32 — Previous-action motion semantics (방향 vs 크기)
- 프로토콜: `O/motion_semantics_ablation/PROTOCOL.md` (실행 전 작성 2026-10-03).
- 결과: `RESULTS.md`, `analysis.md`, `summary.json`, `row_contrasts.json`, `run_meta.json`.
- raw: `records.jsonl`(1,408), `errors.jsonl`(0).
- **사전등록 커밋: af2eb10** (2026-10-03 15:10:07 UTC; PROTOCOL.md + `S/motion_semantics_ablation.py`), records 작성 19:28 → 실행 전 커밋 확인. 결과·분석 커밋: aee41e5 (19:30:04 UTC). 분석 스크립트 `S/analyze_motion_semantics.py`는 결과와 함께 aee41e5에 처음 커밋(사전등록 대상 아님; 지표·통계는 실험 11 함수 재사용).
- N: A− 365 / 52, A+ 1,043 / 156; 6 조건.
- seed 0; GPU RTX 5090 (5.4분).
- metric: token motion = codebook 끝 pose (dx, dy); 자기 오차 e = |m_o − m_g|; 증폭 정의 동일.
- 검정: `cboot` 2,000회, McNemar, Wilcoxon, `frac_effect`.
- 스크립트: `S/motion_semantics_ablation.py`, `S/analyze_motion_semantics.py`. 키: `groups.<g>.{conditions,vs_normal,vs_recent_gt,frac_of_recent_gt_effect,substitution_geometry}`, `row_contrasts.json` 각 키.

#### S10. 실험 2 — CoT 인과 개입 (배경)
- 보고서 `O/cot_intervention/COT_INTERVENTION.md`; `summary.json`; run_meta: 159 장면 × 7 조건, seed 0, 5090.
- 사전등록: not preregistered(보고서가 실행 전 기준과 기준선 변경을 공개). 커밋 1541f30. arm C tensor 미저장으로 새 서버에서 재현 불가(REPRODUCTION.md).
- 핵심: 반대 CoT − 교정 CoT 실행 반전 +6.4%p [+2.7, +9.7]; template 재작성 A+ +17.0%p [+10.3, +26.5].
- 스크립트: `S/cot_intervention.py`, `S/analyze_cot_intervention.py`.

#### S11. 실험 3 — Fast vs CoT (무편향 500 장면, 배경)
- 보고서 `O/fast_vs_cot_unbiased/FAST_VS_COT.md`; run_meta: 500 장면, seed 20260915, 5090, 49분.
- 사전등록: not preregistered. 커밋 1541f30.
- 핵심: CoT − natural ΔA+ −5.0%p [−8.0, −2.9], McNemar p = 4.65e-06 (3↑/28↓); ΔADE +0.50 m [+0.36, +0.70].
- 스크립트: `S/fast_vs_cot_unbiased.py`, `S/analyze_fast_vs_cot.py`.

#### S12. 5090 재분석 / mechanism–selection link (참고)
- `O/gpu5090_reanalysis/GPU5090_REANALYSIS.md` (커밋 1b91b0f): 실험 18–24(best-of-N)만 5090 재실행. 기전 관련 언급: 원래 28 log의 "실패 단위 52개"에서 T = 0.01 자연 계획 A−가 3080 Ti 63–65% vs 5090 67–71%(best-of-N harness). 실험 5–11 수치에는 영향 없음(원래 5090).
- `O/mechanism_selection_link/{dev,heldout}{,_5090}/summary.json` (커밋 e3bc8db, 1b91b0f): 선택 vs 버림 rollout의 이탈 전/후 엔트로피(5090 held-out 이탈 전 −0.016 [−0.023, −0.009], 이탈 후 −0.396 [−0.420, −0.373]). 기전 섹션 A–F 표에는 사용하지 않음.

#### S13. 패키지 내 재계산 산출물
- `O/paper_quantitative_package/_parts/scripts/build_autovla_csvs.py`: summary json → CSV 복사, distinct log 개수, layer 점추정의 min/max/SD/CI 폭 중앙값, 실험 7 비율 점추정.
- `O/paper_quantitative_package/_parts/scripts/rerun_equal_distance_analysis.sh`: 원 `analyze_equal_distance.py` 재실행(full + STRICT=1) → `_parts/recomputed/`.

---

# B. Cross-model (실험 28, 33, 34)

## Source map — cross-model replication

Root: `/root/VLA/autovla_misalignment_poc/` (paths below relative to it). Commits from `git -C /root/VLA log --oneline -- <path>`.
All model runs used RTX 5090 (per each RESULTS/PROTOCOL; AutoVLA exps 6–9 "GPU1" = RTX 5090 per TEMPORAL_FEEDBACK_WINDOW.md §2).
"recomputed" = reproduced in this package from raw data with the original script's functions/method.

### Common to each experiment

| Exp | Name | Model | Benchmark | RESULTS | Raw data | Prereg commit | Analysis/results commit | Seeds | Script |
|---|---|---|---|---|---|---|---|---|---|
| 6 | Equal-distance perturbation | AutoVLA (Qwen2.5-VL-3B, natural fast) | NAVSIM navtest PoC | `outputs/equal_distance_perturbation/EQUAL_DISTANCE.md` | `rows.jsonl`, `records.jsonl` | none (discovery) | 1541f30 | sampling T 0.01, fixed seeds | (PoC scripts) |
| 7 | Action-history causal | AutoVLA | navtest PoC | `outputs/action_history_causal/ACTION_HISTORY_CAUSAL.md` | `units.jsonl`, `records.jsonl` | none | 1541f30 | seed = sha256(token:pert:k) | — |
| 8 | Prev-action state patching | AutoVLA | navtest PoC | `outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` | `layer_results.json`, `summary.json` | none | 1541f30 | same harness | — |
| 9 | Temporal feedback window | AutoVLA | navtest PoC | `outputs/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` | `summary.json` | none | 1541f30 | same harness | — |
| 11 | Prev-action identity | AutoVLA | navtest PoC | `outputs/prev_action_identity_decomposition/PREV_ACTION_IDENTITY.md` | `units.jsonl` | none | bf15767 | — | — |
| 32 | Motion semantics | AutoVLA | navtest PoC | `outputs/motion_semantics_ablation/RESULTS.md` | `records.jsonl`, `row_contrasts.json` | af2eb10 (protocol) | aee41e5 | — | — |
| 28 | Cross-VLA (P1) | OpenVLA-7B `openvla/openvla-7b-finetuned-libero-spatial` (bf16) | LIBERO-Spatial 10 tasks × inits 0–9 | `outputs/cross_vla_replication/RESULTS.md` | `rollouts/phaseA,phaseA_rep,phaseB/episodes.jsonl` (100/100/600 lines; untracked in git), `token_feedback/token_feedback.jsonl` (2,462 lines) | 6b9e8ad; amendment 6d0375a (after part of Phase B, before final analysis) | 69402c6 | init states 0–9; greedy; bootstrap seed 0 | `scripts/cross_vla/analyze_cross_vla.py` |
| 33 | Temporal action-chunk replication | Impromptu VLA 3B `aaaaaap/ImpromptuVLAModel/3B_AD` (Qwen2.5-VL-3B) | navtest PoC, 28 logs, 2,748 scenes (2,728 parsable) | `outputs/cross_vla_temporal_replication/RESULTS.md` | `exp_ab/records.jsonl`, `exp_cde/records.jsonl` (+`selected_scenes.json`) | 038a6d3 | 698148a | greedy (+rep. penalty 1.05); C–E scene sample seed 0; bootstrap seed 0 | `scripts/cross_vla_temporal/analyze_temporal.py` |
| 34 | Cross-domain temporal replication | SpatialVLA-4B `IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal` | SimplerEnv Google Robot pick_coke_can, move_near; env seeds 0–39 | `outputs/cross_domain_temporal_replication/RESULTS.md` | `closed_loop/episodes.jsonl` (720 episodes = 9 conds × 80), `token_level/units.jsonl` (1,600 frames) | f578eae | f4cf886 | env seeds 0–39; greedy; bootstrap seed 0 | `scripts/cross_domain_temporal/analyze_svla.py` |
| 35 (record only) | Feedback-protection ablation | SpatialVLA-4B | SimplerEnv | (running; not used) | — | 8fc20bf (PROTOCOL.md records ensemble-weight correction) | — | — | — |

### Statistical methods (read from scripts)

| Exp | CI | Cluster unit | Binary test | Continuous test | Notes |
|---|---|---|---|---|---|
| AutoVLA 6–11, 32 | log-cluster bootstrap 95%, 2,000 | log | McNemar | Wilcoxon | as stated in each report |
| 28 closed loop | task-cluster bootstrap, 2,000, `random.Random(0)`, percentile | task (10) | McNemar exact (binomtest) over (task, init) | — | + exact task-level sign-flip permutation (2^10 enumerations; protocol said 20,000) |
| 28 Phase C | task-cluster bootstrap, **1,000** | task | — | — | means over (frame, sign) units; units with clipped δ_eff = 0 dropped |
| 33 | log-cluster bootstrap, 2,000, `random.Random(0)`, order-statistic | log (28) | McNemar exact over units | Wilcoxon over units | window fraction: (normal − row)/(normal − gt_history), log bootstrap |
| 34 closed loop | episode-cluster bootstrap, 2,000, `random.Random(0)` re-seeded per call, **unstratified** (docstring says stratified — see conflicts m5) | episode (task, seed); 80 | McNemar exact | Wilcoxon | divergence = ‖Σ executed world_vector endpoint(cond) − (natural)‖ |
| 34 token | same bootstrap over units clustered by episode (task, seed) | episode | McNemar | Wilcoxon | |

### Key results → source

| # | Result | Value | N | Metric definition | Test / cluster | Source path + JSON key |
|---|---|---|---|---|---|---|
| O1 | OpenVLA natural success | 87/100 = 87% | 100 eps | LIBERO success | — | `outputs/cross_vla_replication/analysis.json` `natural_success.rate` |
| O2 | OpenVLA official success | 84.7% | — | — | — | `/root/VLA/EXPERIMENT_SUMMARY.md` §6 only (not in exp 28 files) |
| O3 | OpenVLA rerun | 81/100; −6 pp [−14, +2]; disagreement 20/100; McNemar p 0.263; perm p 0.305 | 100 | — | McNemar / task boot | `natural_rep_vs_natural` |
| O4 | OpenVLA within-step amp, x, 8 bins | 92.58% [89.81, 94.35]; D 87.73 bins; recovered 1.5% | 4,741 units / 2,462 frames | max_j \|Δbin_j\| ≥ \|δ\|, j=d+1..5 | task boot 1,000 | `phaseC["d0_\|8\|_feedback"]` |
| O5 | x, 24 bins | 81.16% [77.16, 83.95] | 4,741 | same | same | `phaseC["d0_\|24\|_feedback"]` |
| O6 | y, 8 / 24 bins | 87.37% [85.36, 89.15] / 75.33% [72.38, 77.82] | 4,869 | same | same | `phaseC["d1_\|8\|_feedback"]`, `["d1_\|24\|_feedback"]` |
| O7 | corrected / reverse token | 0% (rec 100%) / identical to feedback | 4,741 | same | — | `phaseC["d0_\|8\|_corrected"]`, `["d0_\|8\|_reverse"]` |
| O8 | Dimension "window" 1–4 (x, 8) | amp 84.5 / 66.9 / 53.4 / 26.8% | 4,741 | same | task boot | `phaseC["d0_\|8\|_window_k"]` |
| O9 | Task: feedback − corrected | +3 pp [−4, +9] p 0.648 (8); +5 pp [−4, +14] p 0.359 (24) | 100 | success | McNemar / task boot / perm 0.594, 0.426 | `feedback_vs_corrected_d8`, `_d24` |
| O10 | Task: reverse − natural | −3 pp [−9, +3] p 0.581 (8); −8 pp [−15, 0] p 0.134 (24) | 100 | success | same; perm 0.563, 0.141 | `reverse_vs_natural_d8`, `_d24` |
| O11 | Task: each condition − natural | −3 … −8 pp, min McNemar p 0.115 (corrected_d8) | 100 | success | same | `vs_natural.*` |
| O12 | Fail given natural success | 9.2–18.4% vs rerun 15% (13/87) | 87 | — | — | `fail_given_natural_success`; rerun share from `natural_rep_vs_natural.b_only`/87 |
| I1 | Impromptu natural | A− 33.36% [26.20, 38.04]; amp 31.49%; FDE5 5.748 m [5.174, 6.273] | 2,728 scenes / 28 logs | P/R/A A−; amp = A− ∧ FDE5 > 3 m | log boot | `AB.natural.*` |
| I2 | Rerun disagreement | A− 5.90% [5.20, 6.85]; amp 5.87%; final wp shift 0.899 m [0.842, 0.956]; A−-rate diff +0.18 pp, McNemar 83 vs 78 p 0.753 | 2,728 | — | log boot | `AB.rerun_variability.*` |
| I3 | Rerun A+→A− | 4.57% [3.74, 5.52] | 1,818 | — | log boot | `AB.A.baseline_flip_natural_ok_to_rep_a_minus` |
| I4 | 0.2 m: A+→A− | 49.49% [46.91, 52.50] | 14,424 units | given pert_m0 A+ | log boot | `AB.A.m0.2.flip_to_a_minus_given_m0_ok` |
| I5 | 0.2 m: paired A− vs pert_m0 | +24.78 pp [+19.44, +32.16], McNemar 7,138 vs 1,731 p≈0 | 21,824 | — | McNemar / log boot | `AB.A.m0.2.vs_m0_a_minus` |
| I6 | 0.2 m: r≥3 / r≤1; median r | 97.39% [96.91, 97.89] / 0.45% [0.34, 0.60]; 27.56 | 21,824 | r = D10/\|δ\| vs pert_m0 | log boot | `AB.A.m0.2.r_ge_3`, `r_le_1`, `r_median` |
| I7 | Prereg coexistence (crit. 3) | 2.24% [1.54, 3.21] (0.2), 3.15% [2.36, 3.88] (0.5) < 5.9% → FAIL | 2,728 | stable (A+ ∧ r≤1) & unstable (amp ∨ r≥3) same scene | log boot | `AB.B.m0.2.coexist_share`, `m0.5` |
| I8 | Correction | normal 63.43% [60.59, 66.67] → recent_gt 1.25% [0.81, 1.77]; Δ −62.18 pp [−65.57, −59.13]; gt_history 0.0% | 2,398 units / 300 scenes / 28 logs | amp | McNemar p 0.0 (underflow) / log boot | `CDE.all_units.means.*.amp`, `vs_normal.recent_gt.amp` |
| I9 | Window 1/2/3/4 (frac of full, amp) | 24.3 [15.6, 33.8] / 38.7 [29.7, 47.8] / 90.5 [87.0, 95.4] / 94.9 [91.8, 98.6]; recent_gt 98.0 | 2,398 | (normal−row)/(normal−gt_history) | log boot | `CDE.all_units.window_frac_of_full.*.amp` |
| I10 | Reverse | 0.0 → 81.90% [80.37, 84.19]; Δamp +81.9 pp (McNemar 1,964 vs 0); ΔD10 +40.52 m [+37.32, +43.86] Wilcoxon p≈0 | 2,398 | — | — | `CDE.all_units.reverse_vs_gt_history.amp`, `.d10` |
| I11 | near_gt − recent_gt | −0.08 pp [−0.60, +0.30], p 0.875 | 2,398 | amp | McNemar | `CDE.all_units.motion["near_gt - recent_gt"].amp` |
| I12 | direction − magnitude | +15.64 pp [+8.99, +24.25] p 2.0e-31; ΔFDE +14.52 m; ΔD10 +18.44 m [+15.90, +20.09] p 1.8e-268 | 2,398 | — | McNemar / Wilcoxon | `CDE.all_units.motion["dir_wrong_mag_ok - dir_ok_mag_wrong"]` |
| I13 | matched-e first sub (exploratory) | +0.294 m [+0.253, +0.326], p 2.0e-99 | 2,398 | next-wp error at w3 | Wilcoxon | `exploratory_matched_first_substitution.json` `dir_wrong_minus_dir_ok` |
| S1 | SpatialVLA natural | 68/80 = 85.0% [77.5, 92.5]; coke 37/40, move near 31/40 | 80 | success = done at last step | episode boot | `outputs/cross_domain_temporal_replication/analysis.json` `closed_loop.success.natural`; per task recomputed |
| S2 | Rerun natural_gen | 75/80 = 93.75%; Δ +8.75 pp [+1.25, +16.25]; McNemar 9 vs 2 p 0.0654; disagreement 13.75%; divergence 0.1396 m [0.1002, 0.1855] | 80 | — | McNemar / episode boot | `closed_loop.vs_natural.natural_gen`, `discordant_vs_natural.natural_gen`, `final_traj_divergence_vs_natural_m.natural_gen` |
| S3 | Closed loop paired table | see `spatialvla_closed_loop_paired.csv` | 80 | — | same | recomputed: `_parts/scripts/svla_closed_loop_paired.py` (identical to `closed_loop.*`) |
| S4 | Token step-2/3/4 change | 10,542 / 6,824 / 5,286 of 18,975 = 55.56 / 35.96 / 27.86% | 18,975 | normal-row translation token ≠ R | descriptive (no CI in original) | recomputed: `_parts/scripts/svla_token_step_change.py` |
| S5 | Token normal amp / absorption | 22.06% [19.47, 24.70] / 36.67% [33.07, 40.18] | 18,975 units / 1,600 frames / 80 eps | A ≥ 1 / steps 2–4 tokens = R | episode boot | `token_level.all.means.normal.amp`, `.rec` |
| S6 | Correction | recent_ref − normal amp −8.31 pp [−9.31, −7.38] p 6.7e-261; full_ref −8.55 pp [−9.62, −7.54] p 2.0e-273; dev4 −0.0462 [−0.0514, −0.0413] (−82%) | 18,975 | — | McNemar / Wilcoxon | `token_level.all.vs_normal.{recent_ref,full_ref}` |
| S7 | Reverse | reverse − full_ref amp +5.10 pp [+4.39, +5.84] (1,180 vs 213) p 1.9e-162; dev4 +0.0468 [+0.0418, +0.0519] p≈0; reverse dev4 0.0566 vs normal 0.0561 | 18,975 | — | McNemar / Wilcoxon | `token_level.all.reverse_vs_full_ref` |
| S8 | Window | recent_ref 86.3%, win1 62.6% of full_ref dev4 reduction; win1 − full_ref amp +1.88 pp [+1.49, +2.32] p 6.3e-68 | 18,975 | (normal − row)/(normal − full_ref) on dev4 | derived; CI not reported | `token_level.all.means.*.dev4`, `win1_vs_full_ref` |
| S9 | Coexistence | d 0.15: 25.1% [21.8, 28.7]; d 0.30: 15.3% [12.8, 17.9] | 1,600 frames | same frame/d: ≥1 recovered & ≥1 amplified | episode boot | `token_level.B.d0.15/d0.3.coexist_recovered_and_amplified` |
| S10 | Direction vs magnitude | +1.61 pp [+0.40, +2.84] p 7.3e-5; dev4 +0.0126 [+0.0088, +0.0171] p 5.5e-17; near_ref − ref_trans +4.31 pp [+3.17, +5.63] p 4.2e-31 | 10,542 motion units | — | McNemar / Wilcoxon | `token_level.motion` |
| S11 | Ensemble weights | 0.5741 / 0.2579 / 0.1159 / 0.0521 (newest-first); steps 2–4 42.6% | — | exp(0.8 i)/Σ | analytic | `scripts/cross_domain_temporal/action_ensemble.py`; recomputed `svla_token_step_change.py`; also exp 35 `PROTOCOL.md` §0-1 |
| A1 | AutoVLA correction | 47.4% [38, 57] → 7.1% [4, 10]; Δ −40.3 pp [−50.3, −30.0] p 8e-42 | 365 A− units / 52 scenes | amp = A− ∧ FDE > 3 m | McNemar / log boot | `outputs/action_history_causal/ACTION_HISTORY_CAUSAL.md` (A− table) |
| A2 | AutoVLA reverse@emb | 4.4% → 33.4% [26.3, 43.2]; +29.0 pp [+21.1, +38.0] p 6.7e-31 | 365 | — | McNemar | `outputs/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md` l.10, l.333 |
| A3 | AutoVLA window | 46.6 → 26.6/14.2/8.5/5.8, full 4.1 → 47/76/90/96% | 365 | — | log boot | `outputs/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md` §1 |
| A4 | AutoVLA direction > magnitude | +9.0 pp [+3.1, +18.7] p 6e-4; FDE +1.21 m | A− 365 | — | McNemar | `outputs/motion_semantics_ablation/RESULTS.md` l.26 |
| A5 | AutoVLA matched-distance | A− scenes amp 45.0% [37, 55] (313) vs A+ 4.8% [3, 7] (887) | — | — | log boot | `outputs/equal_distance_perturbation/EQUAL_DISTANCE.md` l.12, l.26 |
| A6 | AutoVLA natural | A− 52/2,747 = 1.9%; FDE5 0.66 m [0.48, 0.87] (500-scene subset) | — | — | — | `outputs/natural_reference_stabilization/NATURAL_REFERENCE.md` l.14; `outputs/fast_vs_cot_unbiased/FAST_VS_COT.md` l.9 |

---

# C. 탐지·선택 (실험 12–24, 31)

## Source map — 탐지·선택 결과 (실험 12–24, 31/P4)

경로 기준: `O = /root/VLA/autovla_misalignment_poc/outputs`, `S = /root/VLA/autovla_misalignment_poc/scripts`. 커밋은 `git -C /root/VLA log --oneline -- <path>`로 확인했습니다.
모델은 모두 **AutoVLA** (Qwen2.5-VL-3B, checkpoint `AutoVLA_PDMS_89.ckpt`, action codebook 2,048 token, 10 token = 5 s), torch 2.8.0+cu128.
벤치마크는 OpenScene/NAVSIM **navtest** (nuPlan), open-loop 5 s P/R/A 라벨(`/root/VLA/pra_comparison/pra_labels.py`)과 NAVSIM v1 비반응형 PDM Score.
공통 정의: A− = P/R/A 5 s 궤적 실패; 증폭(amplification, "강한 실패") = A− ∧ FDE5 > 3 m (`S/analyze_action_history.py`, `AMP_FDE = 3.0`).

| 주요 결과 | 실험 # / 이름 | 데이터 (N) | seed / T | GPU | 지표 정의 | 검정 / CI | cluster 단위 | 보고서 | 원자료 (raw) | 요약 JSON (키) | 사전 등록 커밋 | 분석·결과 커밋 | 스크립트 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out 탐지 AUROC pre/post (표 1) | 31 (P4) 불안정 rollout 탐지 baseline | held-out 52 log, 4,814 장면 → 이탈 후보 43,469 (증폭 2,538); dev 56 log → 42,095 (2,885) | 후보 seed 0, T 1.0 ×16 + T 0.01 ×1; teacher forcing 같은 seed | RTX 5090 (디코딩·teacher forcing 모두) | 후보 단위 pooled AUROC/AUPRC, 장면 내 AUROC = 양쪽 클래스 장면 평균; pre = t*까지(포함), post = t*+1..9 | log bootstrap 1,000회 percentile; 탐지기 간 검정 없음 | log | `O/instability_detection_baselines/RESULTS.md`, `detection.md` | `O/expanded_best_of_n/gpu1`, `O/expanded_best_of_n_5090/gpu1`, `O/heldout_best_of_n/gpu1`, `O/heldout_best_of_n_5090/gpu1` (`records.jsonl`, git 미추적); `O/instability_detection_baselines/teacher_forced/*/{steps.jsonl,hidden_*_L{18,36}.f16}` | `O/instability_detection_baselines/detection.json` → `heldout.<det>_{pre,post}.{auroc,auroc_ci95,auprc,within_scene_auroc,n,prevalence}`; probe는 `layer,hparam,dev_cv_auroc` | `PROTOCOL.md` bf9aa21 (2026-10-03 15:10, 실행 전); 격자 축소 9b91b2d (21:48, probe 결과 전) | 69402c6 (결과); 스크립트 af2eb10, 9b91b2d | `S/teacher_force_candidates.py`, `S/analyze_instability_detection.py` |
| 선택 vs 버림 이탈 전/후 엔트로피, rank-sum 장면 내 증폭 AUROC | 22 기전 ↔ 선택 연결 | dev 4,563 장면 / 77,571 후보; held-out 4,814 / 81,838 | seed 0 | 혼합 (`dev`, `heldout`) / 5090 (`*_5090`) | 장면 단위 쌍대 차이(선택 − 버림 평균); 후보 단위 AUROC | log bootstrap 2,000회(차이), 500회(AUROC); p 없음 | log | `O/selection_validation/SELECTION_VALIDATION.md` §2 | 위 best-of-N records | `O/mechanism_selection_link/{dev,heldout,dev_5090,heldout_5090}/summary.json` → `selected_vs_discarded.all_scenes.<metric>.{selected,discarded,diff,ci95,n}`, `auroc_amplification.<sig>` | 스크립트 고정 513fdba | e3bc8db (혼합), 1b91b0f (5090) | `S/mechanism_selection_link.py` |
| dev A− 2.15 → 1.45%, FDE 0.758 → 0.484 m | 20 표본 확대 (expanded best-of-N) | navtest shard 6–17, 56 log, 4,563 장면, 자연 A− 98 | seed 0, 후보 17 (T 0.01 + T 1.0 ×16) | 혼합: 5090 39 log / 3,294 장면 (`gpu1`), 3080 Ti 17 log / 1,269 장면 (`gpu0`) | 모집단 비율·평균, 자연 계획 대비 쌍대 | McNemar(정확), Wilcoxon, log bootstrap 2,000회 | log | `O/expanded_best_of_n/EXPANDED_BEST_OF_N.md` | `O/expanded_best_of_n/gpu{0,1}/records.jsonl` | `O/expanded_best_of_n/summary_pooled.json` → `rules.ranksum.a_minus.{mean,diff,ci95,mcnemar}`, `rules.ranksum.fde5` | 형식 문서 없음; 규칙은 e6d9dff (09:26 UTC, 디코딩 전)에 커밋 | f4cedfd | `S/expanded_best_of_n.py`, `S/analyze_expanded_best_of_n.py`, `S/analyze_best_of_n.py::pick` |
| dev ADE 0.313 → 0.213 m (recomputed) | 20 | 같음 | 같음 | 혼합 | ADE5 = 10 waypoint 평균 L2 (`unit_metrics.ade5`) | Wilcoxon, log bootstrap 2,000회 (원 스크립트와 동일) | log | – (보고서에 없음) | 같음 | `_parts/recomputed_ade_dev_mixed.json` → `rules.ranksum.ade5`; 교차 확인 `O/candidate_count_curve/dev/summary.json` → `curve.{1,17}.ade` | – | 이 패키지 (커밋 안 함) | `_parts/scripts/recompute_ade_expanded.py` |
| held-out PDMS +0.0095, 충돌 0.56 → 0.23% | 21 PDM Score (사전 등록 held-out) | navtest shard 18–31, 52 log, 4,814 장면, 자연 A− 72 | seed 0 | 혼합: 5090 30 log / 2,823 장면, 3080 Ti 22 log / 1,991 장면 | navsim 기본 PDMS (4 s, progress 5, TTC 5, comfort 2), 하위 지표 위반율 | Wilcoxon, log bootstrap 2,000회 (충돌 검정 없음) | log | `O/selection_validation/SELECTION_VALIDATION.md` §1 | `O/pdm_score_best_of_n/heldout/pdm_scores.jsonl` (미추적), `O/heldout_best_of_n/gpu{0,1}/records.jsonl` | `O/pdm_score_best_of_n/heldout/summary.json` → `subsets.all.ranksum.{pdms,d_pdms,d_pdms_ci95,wilcoxon_p,viol_no_at_fault_collisions}` | **513fdba** (2026-09-29 20:12:38; 디코딩 종료 20:11:19, 분석 전) | e3bc8db (21:48) | `S/pdm_score_candidates.py`, `S/analyze_pdm_scores.py` |
| held-out F1 PDMS +0.0315, 충돌 0.10% | 24 현재 frame 안전 필터 | 같음 | 같음 | 혼합 | F1 = 등속 외삽 + 지도로 충돌·DA 이탈 예측 후보 제거 후 rank-sum (전부 제거 시 rank-sum) | 같음 | log | 같은 보고서 §4 | `O/safety_filter/heldout/{flags.jsonl,picks.json}` (미추적) | `O/pdm_score_best_of_n/heldout_filter/summary.json` → `subsets.all.F1.*`; open-loop `O/safety_filter/heldout/open_loop_summary.json` → `open_loop.F1` | 513fdba (F1을 주 필터로 명시) | e3bc8db (safety_filter 디렉토리 자체는 git 미추적) | `S/safety_filter_flags.py`, `S/safety_filter_select.py` |
| held-out open-loop A− 1.50 → 0.71% | 21 | 같음 | 같음 | 혼합 | A− | McNemar, log bootstrap 2,000회 | log | 같은 보고서 §1 | 같음 | `O/heldout_best_of_n/summary_pooled.json` → `rules.ranksum.a_minus` | 513fdba | e3bc8db | `S/analyze_expanded_best_of_n.py` |
| 후보 수 곡선 (N 1–17), 포화 | 23 후보 수 | dev 4,563 / held-out 4,814 장면 | 부분집합 seed = 장면 token, 장면당 5개 | 혼합 / 5090 (`*_5090`) | rank-sum, N = 자연 포함 후보 수; open-loop는 무작위 부분집합 평균, PDMS는 앞쪽 부분집합 | log bootstrap 2,000회; PDMS Wilcoxon | log | 같은 보고서 §3 | 위 records | `O/candidate_count_curve/{dev,heldout,*_5090}/summary.json` → `curve.<N>`; `O/pdm_score_best_of_n/*_ncurve/summary.json` → `subsets.all.ranksum_N<N>` | 513fdba (N 목록 명시) | e3bc8db, 1b91b0f | `S/candidate_count_curve.py` |
| 추론 비용 1.03/1.16/1.31× (3080 Ti), 1.2–1.28× (5090) | 23 | 같은 100 PoC 장면 (`navtest_poc`) | – | 3080 Ti / 5090 | 계획 전체 경과 시간 비율 (N = 자연 포함 후보 수, `--n N−1`) | 없음 (1회 측정) | – | `SELECTION_VALIDATION.md` §3, `GPU5090_REANALYSIS.md` | 3080 Ti: scratchpad `timing_N*.log`만 (저장소 밖); 5090: `O/gpu5090_reanalysis/timing5090_N{1,4,8,16}.log` | – | – | 1b91b0f (5090 log) | `S/expanded_best_of_n.py` (timing.sh, scratchpad) |
| 5090 재실행 동일성 | – 3080 Ti → 5090 재분석 | dev 3080 Ti 부분 17 log / 1,269 장면, held-out 22 log / 1,991 장면 재디코딩 | 같은 seed | 5090 | 위와 같음 | 위와 같음 | log | `O/gpu5090_reanalysis/GPU5090_REANALYSIS.md` | `O/expanded_best_of_n_5090/gpu1`, `O/heldout_best_of_n_5090/gpu1` | `O/gpu5090_reanalysis/{dev,heldout}_5090_summary_pooled.json`; `O/pdm_score_best_of_n/{dev,heldout}_5090{,_filter,_ncurve}/summary.json` | – (사후 재현) | 1b91b0f | 위 스크립트 동일 |
| 학습형 선택기 dev CV (ΔA− −0.55 / −0.26%p) | 21 (개발 단계 대안) | dev 4,563 장면 | – | 혼합 | log-grouped 5-fold CV | log bootstrap, McNemar | log | `SELECTION_VALIDATION.md` (데이터와 절차) | `O/learned_selector/dev_features.npz` | `O/learned_selector/dev_cv.json` → `ridge_fde`, `logit_aplus` | 스크립트 513fdba | 결과 파일 git 미추적 | `S/learned_selector.py`, `S/selector_features.py` |
| N 16/32 PoC 선택 (모집단 가중) | 18–19 best-of-N | PoC 28 log, 실패 52 + 정상 297 장면 (가중 52/2,695) | seed 0–4 (N16 T1.0), 나머지 seed 0 | 3080 Ti (N4/N8/N16 seed 0–2, 4), 5090 (N32, T0.85/1.15, seed 3, `*_5090`) | 실패 장면 ΔA−, 모집단 가중 ΔA−/ΔFDE | McNemar, Wilcoxon, log bootstrap 2,000회 | log | `O/best_of_n_selection/BEST_OF_N.md`, `O/best_of_n_selection_n16_T1/BEST_OF_N_16.md` | `O/best_of_n_selection*/records.jsonl` | `O/best_of_n_selection*/summary.json` → `population_a_minus_rate.*`; `analysis_console.txt` | 없음 (rank-sum은 사후 정의, docstring 명시) | bf15767, e6d9dff | `S/best_of_n_selection.py`, `S/analyze_best_of_n.py`, `S/pool_best_of_n.py` |
| 조건화 교정 prev_w4 47.1 → 33.2% | 12 비-oracle 참조 안정화 | equal-distance A− 52 장면 / 365 단위, A+ 156 / 1,043 | seed 0, T 0.01 | 5090 | 증폭률, 교정 시점 t* oracle | McNemar(단위), log bootstrap | log | `O/reference_stabilization/REFERENCE_STABILIZATION.md` | `records.jsonl` | `summary.json`, `analysis_console.txt` | 없음 | bf15767 | `S/reference_stabilization.py`, `S/analyze_reference_stabilization.py` |
| 재계획 59.6 → 23.5% | 13 receding-horizon | equal-distance 208 장면 (A− 52, A+ 156), 계획 6,943개 | seed 0, T 0.01 | 5090 | 5 s/8 s 실패 | log bootstrap | log | `O/receding_horizon_replanning/RECEDING_HORIZON.md` | `records.jsonl` | `summary.json` | 없음 | bf15767 | `S/receding_horizon_replanning.py`, `S/analyze_receding_horizon.py` |
| 합의 참조 27.9% (+0.28 m A+) | 14 합의 참조 | 207/208 장면, A− 365 / A+ 1,037 | seed 0, T 0.01 | 3080 Ti (원), 5090 (`_5090`) | 증폭률, A+ FDE | McNemar, log bootstrap | log | `O/consensus_reference_stabilization/CONSENSUS_REFERENCE.md` | `records.jsonl` | `summary.json`, `analysis_console.txt` | 없음 | bf15767; 5090 1b91b0f | `S/reference_stabilization.py --sources ...` |
| 상시 적용 모집단 1.24 → 2.60% | 15 배포 조건 안정화 | 자연 실패 52 + 무작위 정상 297 장면 (349/352 성공), 가중 52/2,695 | seed 0, T 0.01 | 3080 Ti (원), 5090 (`_5090`: 1.31 → 2.27%) | A−, FDE5, 모집단 가중 A− | McNemar, Wilcoxon, log bootstrap | log | `O/natural_reference_stabilization/NATURAL_REFERENCE.md` | `records.jsonl` | `summary.json`, `analysis_console.txt` | 없음 | bf15767; 5090 1b91b0f | `S/natural_reference_stabilization.py`, `S/analyze_natural_reference_stabilization.py` |
| PDM-Closed 참조 A+ 4.2 → 20.1% | 16 PDM-Closed 참조 | A− 365 / A+ 1,043 단위, 489 장면 | seed 0, T 0.01 | 5090 | 증폭률 | McNemar, log bootstrap | log | `O/pdm_reference_stabilization/PDM_REFERENCE.md` | `records.jsonl` | `summary.json` | 없음 | bf15767 | `S/pdm_reference.py`, `S/reference_stabilization.py` |
| seed·T 강건성 31–33% | 17 | 같은 equal-distance 집합 | seed 0–2 × T 0.01/0.5 | 5090 | 증폭률 | McNemar | log | `O/robustness_reference_stabilization/ROBUSTNESS.md` | `T*_seed*/records.jsonl` | `T*_seed*/summary.json` | 없음 | bf15767 | `S/reference_stabilization.py` |

비고
- `probe60/`(60 장면 hidden-state 추출, 2026-09-26, 커밋 1541f30)은 초기 PoC의 perception probe 자료이며 실험 22/31의 탐지 표와 무관합니다(`run_meta.json`만 있음).
- 실험 20–24 records, PDM 점수 jsonl, safety_filter·learned_selector 출력은 `.gitignore`(`autovla_misalignment_poc/**`, `**/*.jsonl`) 때문에 저장소에 없고, 요약 JSON과 보고서만 커밋되어 있습니다.
- 이 패키지의 헬퍼: `_parts/scripts/recompute_ade_expanded.py`(ADE 재계산), `_parts/scripts/build_detsel_tables.py`(CSV/TeX 생성; 기존 JSON 값만 옮기고 포화 비율만 계산).

---

# D. navhard (실험 25–27, 29, 30)

## Source map — navhard (실험 25, 26, 27, 29, 30)

경로는 `/root/VLA/autovla_misalignment_poc/` 기준(tools는 `/root/VLA/tools/`). 커밋은 `git -C /root/VLA log --oneline -- <path>`로 확인.
공통: 모델 AutoVLA `AutoVLA_PDMS_89.ckpt`; 벤치마크 NAVSIM v2 `navhard_two_stage`(공식 `run_pdm_score_from_submission.py`);
지표 EPDMS(그룹 평균, 그룹 = (s1·s2)(orig)와 (s1·s2)(prev)의 평균); 후보 17개(T 0.01 자연 + T 1.0 ×16, `scripts/expanded_best_of_n.py` `--seed 0`);
CI = log-cluster bootstrap 2,000회 seed 0; 주 검정 = log 단위 paired sign-flip permutation 양측 20,000회 seed 0; 보조 = 그룹 Wilcoxon; cluster unit = log;
디코딩 GPU = RTX 5090 (`CUDA_VISIBLE_DEVICES=1`, PCI_BUS_ID).

| 핵심 결과 | 실험 | 이름 | N log / 그룹 | 보고서 | 원자료 (raw) | 사전 등록 / 프로토콜 커밋 | 결과(분석) 커밋 | 분석 스크립트 | 검정 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| 첫 절반 F1 +0.125, max log-lik +0.032 (CI만) | 25 | navhard 절반 pseudo closed-loop | 36 / 105 (1단계 210 + 합성 2,702) | `outputs/navhard_eval/NAVHARD_CLOSED_LOOP.md` | `outputs/navhard_eval/decode_5090/records.jsonl`, `group_scores/g_*.json`, `safety_flags/flags.jsonl`, `filter/picks.json`, `subset/groups.json` | 없음 (dev에서 고정한 규칙; 사전 등록 문서 없음) | 1b91b0f | `scripts/analyze_navhard.py` → `navhard_comparison.json` | bootstrap CI만 (p 미보고) | GPU 5090; 채점 `tools/navhard_score_groups.sh` |
| 두 번째 절반 재현 판정 (H1 F1 통과, H2 max log-lik 실패) | 26 | navhard 나머지 절반 사전 등록 검증 | 40 / 120 (1단계 240 + 합성 2,760) | `outputs/navhard_full_validation/RESULTS.md`, `navhard_full_comparison.md` | `half2/decode_5090/`, `half2/group_scores/g_{normal,ranksum,max_loglik,F1,F1_maxll}.json`, `half2/subset/groups.json`, `half2/safety_flags/` | **96a595c** (`PREREGISTRATION.md`) | **ad7991c** | `scripts/analyze_navhard_full.py` → `navhard_full_comparison.json` `half2.rules.*` | perm(주) + Wilcoxon + bootstrap | 파이프라인 `tools/navhard_half2_pipeline.sh` |
| 전체 navhard: 선택 없음 0.2287, rank-sum, max log-lik, F1 0.3470, F1+maxlog 0.3503 | 26 | 전체 navhard | 76 / 225 | 같음 | `half1/group_scores/` (실험 25 디코딩 재채점) + `half2/group_scores/` | 96a595c | ad7991c | 같음 → `pooled.rules.*` | 같음 | 전체 표에는 효과를 발견한 첫 절반 포함 |
| 필터만 +0.105, 필터+무작위, rank/maxlog 한계 기여, "89%" | 27 | 안전 필터 ablation | 76 / 225 | `outputs/safety_filter_ablation/RESULTS.md`, `ablation.md` | `half{1,2}/group_scores/g_{filter_only,filter_random_s0..2,filter_*_natfb}.json`, `half{1,2}/picks/`; 재사용 `navhard_full_validation/half{1,2}/group_scores/` | **19c0130** (`PROTOCOL.md`) | **f98cb10** | `scripts/analyze_navhard_ablation.py` (19c0130에 커밋) → `ablation.json` `full.*` | 같음 | 새 GPU 작업 없음; `tools/navhard_ablation_pipeline.sh`; 무작위 seed 0/1/2 `random.Random(f"{seed}:{token}")` |
| 충돌만 +0.012, DA만 +0.087, Shapley DA 86%, 곱셈 항 99–104%, 계획 도달 거리 | 29 | 안전 필터 구성요소 (P2) | 76 / 225 (token: 1단계 450, 2단계 5,462) | `outputs/safety_filter_components/RESULTS.md`, `components.md` | `half{1,2}/group_scores/g_comp_{collision,dac}.json`, `half{1,2}/picks/component_picks.json`, 공식 CSV(점수 분해 입력) | **0cffc15** (`PROTOCOL.md`) | **0ab33a4** | `scripts/analyze_navhard_conditions.py` → `components.json`; `scripts/navhard_score_decomposition.py` → `score_decomposition.json` (둘 다 af2eb10, 프로토콜 후·결과 전); `plan_reach.json`은 생성 스크립트 미확인 | 같음; 점수 분해는 token 단위, log bootstrap CI | `tools/navhard_components_candidates_pipeline.sh`; 새 GPU 없음 |
| oracle17 0.405, 남은 여지 +0.055, "69%", token 31% 모두 0점 | 30 | 후보 oracle 상한 (P3) | 76 / 225 (token 5,912) | `outputs/candidate_oracle/RESULTS.md`, `oracle_comparison.md` | `half{1,2}/group_scores/g_cand01..16.json`, `g_oracle16/17.json`, `half{1,2}/oracle/token_scores.npy`, `oracle_picks.json` | **0cffc15** (`PROTOCOL.md`) | **86087e0** | `scripts/navhard_oracle_picks.py`, `scripts/analyze_navhard_conditions.py` → `oracle_comparison.json` | 같음 | oracle은 채점 미래 사용, 배포 불가; 새 디코딩 없음 |
| 파생 비율 (88.4%, 85.7%, 68.9%, 67.0%, 31.6%) | 27/29/30 | 이 패키지에서 계산 | – | `_parts/navhard_sections.md` | 위 JSON + `token_scores.npy` | – | 커밋 안 함 | `_parts/scripts/navhard_verify.py` → `_parts/navhard_verify.json` | 비율만, 새 검정 없음 | – |
| 논문 표 | 26/27/29/30 | – | 76 / 225 | `paper_navhard_table.{csv,tex}`, `figure_navhard_methods.csv` | 위 JSON | – | – | `_parts/scripts/navhard_build_tables.py` (JSON 값 복사만) | – | – |

### 각 결과의 정확한 JSON 키

| 값 | 파일 | 키 |
|---|---|---|
| 선택 없음 0.2287 | `outputs/navhard_full_validation/navhard_full_comparison.json` | `pooled.rules.normal.epdms.mean` |
| rank-sum +0.0196 | 같음 | `pooled.rules.ranksum.epdms` |
| max log-lik +0.0199 | 같음 | `pooled.rules.max_loglik.epdms` |
| F1 +0.1183 | 같음 | `pooled.rules.F1.epdms` (= `ablation.json` `full.contrasts['filter_ranksum - normal']`) |
| F1+maxlog +0.1216 | 같음 | `pooled.rules.F1_maxll.epdms` |
| 두 번째 절반 F1 / max log-lik / rank-sum | 같음 | `half2.rules.{F1,max_loglik,ranksum}.epdms` |
| 필터만 +0.1046 | `outputs/safety_filter_ablation/ablation.json` | `full.contrasts['filter_only - normal'].epdms` |
| 필터+무작위 +0.0992 | 같음 | `full.contrasts['filter_random - normal'].epdms` |
| rank 한계 +0.0137 / maxlog 한계 +0.0170 | 같음 | `full.contrasts['filter_ranksum - filter_only' / 'filter_maxll - filter_only']` |
| 충돌만 +0.0121 / DA만 +0.0868 | `outputs/safety_filter_components/components.json` | `full.contrasts['collision_only - no_filter' / 'dac_only - no_filter']` |
| DA 추가분 +0.0926 / 충돌 추가분 +0.0178 | 같음 | `full.contrasts['collision_dac - collision_only' / 'collision_dac - dac_only']` |
| 곱셈 항·진행도 항 | `outputs/safety_filter_components/score_decomposition.json` | `conds.<cond>.stage{1,2}.{total,multiplicative,ego_progress}` |
| 계획 도달 거리 | `outputs/safety_filter_components/plan_reach.json` | `<cond>.mean_plan_reach_4s_m`, `mean_change_in_changed_m`, `n_changed` |
| oracle17 / oracle16 | `outputs/candidate_oracle/oracle_comparison.json` | `full.means.oracle17.epdms`, `full.contrasts['oracle17 - no_selection']` |
| 남은 여지 +0.0550 | 같음 | `full.contrasts['oracle17 - filter_maxll']` |
| 모두 0점 token 31.6% | `outputs/candidate_oracle/half{1,2}/oracle/token_scores.npy` | `(M.max(1) == 0).mean()` (navhard_verify.py) |
| 필터 통계 (자연 계획 걸림 47.6/48.4%, 모두 걸림 27.6/27.4%) | `outputs/navhard_eval/filter/filter_stats.json`, `outputs/navhard_full_validation/half2/filter/filter_stats.json` | `F1_natural_removed`, `F1_all_removed`, `F1_n_survivors` |

---
