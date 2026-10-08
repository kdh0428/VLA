# AutoVLA 기전 실험 — Source map

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

### S1. 실험 4 — Natural/Fast 기전 재검증 (mismatch ≠ failure 기반 수치)
- 모델/벤치: 위 공통; arm N(full extraction, 층별 attn/mlp 성분 저장), 비교 run G(gpu1 natural run), C(Forced-CoT).
- 보고서: `O/natural_fast_mechanism/NATURAL_FAST_MECHANISM.md`
- raw: 저장된 tensor(`O/full_extract/`, arm N), `O/gpu1/`; 이 디렉토리에는 `summary.json`만.
- 사전등록: not preregistered (규칙이 스크립트 상단에 "실행 전 고정"이라고 서술, git 증거 없음). 분석 커밋: 1541f30.
- N: 2,747 장면(arm N), A− 52(16 log), A+ 2,695; G: A− 46; C: 2,604, A− 136.
- seed: 해당 없음(GPU 미사용, 저장 tensor 분석). GPU: 미사용(원 추출은 5090).
- metric: A− = P/R/A coarse-action accept 규칙 5 s 실패(token ID 미사용); 오답 token 변위 = codebook 변위 거리; FDE/첫 오차 = 중앙값 비.
- 검정: AUROC/차이 = log cluster bootstrap 1,000회(`REPS = 1000`), Spearman CI = `_spearman_ci`(log bootstrap 1,000회, `random.Random(3)`), p 미계산.
- 스크립트: `S/natural_fast_mechanism.py` (보고서 생성 `S/make_natural_fast_report.py`). 키: `N.n`, `N.n_Aminus`, `N.n_Aminus_step0_token_wrong`, `N.n_Aplus_step0_token_wrong`, `N.codebook_amplification.*`, `N.verdicts.*`.

### S2. 실험 5 — 첫 mismatch 1-token 교정
- 보고서: `O/first_mismatch_causal/FIRST_MISMATCH_CAUSAL.md`; 요약 `summary.json`; `run_meta.json`(n_ok 1159, seed 0, 5090, 15.7분).
- raw: records 미보관(디렉토리에 jsonl 없음) → CI 재계산 불가.
- 사전등록: not preregistered. 분석 커밋 1541f30 (실행 스크립트 bf15767에서 GPU 핀 옵션만 수정).
- N: A− 52 장면, A+ 1,107 장면(step-matched 유효 n 944), 총 1,159; 단위 = 장면.
- seed 0; GPU RTX 5090.
- metric: A+ recovery(P/R/A 5 s), downstream token error(t* 이후 GT 불일치 비율), ADE/FDE 5 s, 오차 증가 속도.
- 검정: log cluster bootstrap 2,000회 seed 0(가중 평균 `wmean_boot`), 교호작용 bootstrap seed 1(`p_boot`), McNemar binomtest, Wilcoxon; AUROC CI 1,000회 seed 2.
- 스크립트: `S/first_mismatch_causal.py`, `S/analyze_first_mismatch_causal.py`. 키: `A-.{original,gt,nn}`, `A-.contrasts."gt - original"`, `interaction."vs A+ step-matched"`, `t_star_hist`, `reproduction`, `nn_fragility`.

### S3. 실험 6 — Equal-distance perturbation
- 보고서: `O/equal_distance_perturbation/EQUAL_DISTANCE.md`, `EQUAL_DISTANCE_strict.md`(STRICT=1); `summary.json`; `run_meta.json`(208 장면, seed 0, rel_levels 0.1/0.2/0.35, abs_tol 0.005 m, min_alts 5, A+ 3/A−, 5090, 6.1분).
- raw: `O/equal_distance_perturbation/records.jsonl`(208), `rows.jsonl`(1,824 = 조건 행), `run.log`.
- strict summary json은 디렉토리에 없음 → `/root/VLA/paper_quantitative_package/_parts/recomputed/equal_distance_summary_strict_rerun.json` (recomputed: `_parts/scripts/rerun_equal_distance_analysis.sh`, 원 스크립트 무수정, 사본 디렉토리에서 실행). 같은 재실행으로 full summary가 Q4 AUROC(±0.01)를 제외하고 동일함을 확인.
- 사전등록: not preregistered (증폭 임계 3.0 m "사전 고정"은 스크립트 docstring 서술). 분석 커밋 1541f30.
- N: 장면 208 (A− 52 / A+ 156, log 16 / 26), 조건 행 1,824, perturbation 단위(original+대안) A− 365 / A+ 1,043.
- seed 0 (original_reseed는 다른 seed); GPU RTX 5090.
- metric: amplification = A− ∧ FDE5 > 3 m; recovery = P/R/A 5 s A+; 거리 = 첫 불일치 token의 GT 대비 codebook 변위(m).
- 검정: `boot`/`boot_stat` log cluster bootstrap 2,000회 seed 0(Q4는 1,000회), Wilcoxon(Q3, Q5), GroupKFold 로지스틱 AUROC(Q4). Spearman/장면 내 상관 p 미계산.
- 스크립트: `S/equal_distance_perturbation.py`, `S/analyze_equal_distance.py`. 키: `n_scenes`, `alt_rel_err_median`, `alt_abs_dist_diff_median_m`, `table`, `Q1`–`Q5`.

### S4. 실험 7 — Action history 인과 (Normal / Recent-GT / GT-history / attention mask / embedding 중립화)
- 보고서: `O/action_history_causal/ACTION_HISTORY_CAUSAL.md`; `summary.json`; `run_meta.json`(1,408 perturbation, 7 조건, seed 0, 5090, 19.5분, blocked attention calls 688,104).
- raw: `records.jsonl`(208), `units.jsonl`(1,408).
- 사전등록: not preregistered. 분석 커밋 1541f30; REPRODUCTION.md: 새 서버에서 "모든 조건의 증폭률·FDE 전부 일치".
- N: A− 365 단위 / 52 장면, A+ 1,043 / 156; 부분집합 previously amplified A− 175/45, A+ 43/32; original token only 52/156.
- seed: sha256(`0:token:perturbation:k`) per step; GPU RTX 5090.
- metric: 위 증폭 정의; 판정은 OOD 아닌 조건만(첫 step entropy가 Normal보다 >1 nat 오르면 OOD).
- 검정: `cboot` log cluster 2,000회 seed 0, McNemar binomtest, Wilcoxon; A−/A+ 교호작용 bootstrap seed 3.
- 스크립트: `S/action_history_causal.py`, `S/analyze_action_history.py`. 키: `subsets."all perturbations".A-.conditions.*`, `.vs_normal.*`, `subsets.*.interaction`.
- "93%" = (Normal − Recent-GT)/(Normal − GT-history) 계산은 이 실험 summary에 없음 → 패키지에서 점추정만 재계산, CI는 실험 8 키 `groups.A-.all.fraction_of_gt_history_effect.recent_gt` 사용.

### S5. 실험 8 — Layer-wise previous-action state patching (+ reverse)
- 보고서: `O/prev_action_state_patching/PREV_ACTION_STATE_PATCHING.md`; `summary.json`, `layer_results.json`, `narrative.json`, `sanity/debug_checks.json`, `figures/`.
- raw: jsonl 없음(디렉토리에 records 미보관).
- 사전등록: not preregistered. 분석 커밋 1541f30.
- N: A− 365 / 52, A+ 1,043 / 156; 조건 93행(patch 위치 37 = emb + L0–L35, delta α 3 × 10 layer, selfpatch 3, identitysrc 10, reverse 10, 기준 3).
- seed 0 공식 동일; GPU RTX 5090 (20.5분).
- metric: 실험 7과 동일(`from analyze_action_history import unit_metrics, AMP_FDE`).
- 검정: `Boot` log cluster 2,000회 seed 0(scene cluster 보조 seed 1), McNemar, Wilcoxon.
- 스크립트: `S/prev_action_state_patching.py`, `S/analyze_prev_action_state_patching.py`. 키: `groups.<g>.all.rows.<name>.{level,vs_base}`, `groups.<g>.all.fraction_of_gt_history_effect`, `sanity.*`.

### S6. 실험 9 — Temporal feedback window
- 보고서: `O/temporal_feedback_window/TEMPORAL_FEEDBACK_WINDOW.md`; `summary.json`, `early_step_contribution.json`, `narrative.json`.
- raw: jsonl 없음.
- 사전등록: not preregistered. 분석 커밋 1541f30.
- N: A− 365 / 52, A+ 1,043 / 156 (t* 분포 A− {0: 116, 1: 161, 2: 49, 3: 13, 4: 6, 5: 12, 7: 8}); 17 조건.
- seed 0 공식; GPU RTX 5090 (6.4분).
- metric: f(w) = (Normal − win_w)/(Normal − GT-history); critical window = f ≥ 0.9 최소 w, bootstrap 분포.
- 검정: `Boot` log cluster 2,000회, McNemar, Wilcoxon; 끝 위치 맞춘 비교 seed 11.
- 스크립트: `S/temporal_feedback_window.py`, `S/analyze_temporal_feedback_window.py`. 키: `subsets.<sub>.<g>.rows.<cond>.{level,vs_normal}`, `.fraction`, `.critical_window`.

### S7. 실험 10 — Horizon-controlled window
- 보고서: `O/horizon_controlled_window/HORIZON_CONTROLLED_WINDOW.md`; `summary.json`, `decay_test.json`, `narrative.json`.
- raw: jsonl 없음 (오류 기록은 보고서에 인용된 `errors.jsonl`, 디렉토리에는 없음).
- 사전등록: not preregistered. 분석 커밋 1541f30.
- N: t* ≤ 1 장면 A− 39 / A+ 116(1 제외), 단위 A− 277 / A+ 772; t* = 0: A− 116 단위 / 17 장면, A+ 336 / 50.
- seed 0 공식; 20 token 생성; GPU RTX 5090 (7.7분).
- metric: 해제 후 정확히 N free step의 FDE/ADE, Normal·Full 같은 pose 구간; amp(>3 m), amp2(>2 m); 재발산 = 해제 시 ≤1 m → N step 뒤 >2 m.
- 검정: `Boot` log cluster 2,000회, McNemar, Wilcoxon; decay = 쌍대 bootstrap 차이.
- 스크립트: `S/horizon_controlled_window.py`, `S/analyze_horizon_controlled_window.py`. 키: `analyses.in10."t*=0".<g>.N.<N>.windows.<w>.*`, `horizon_trend_tstar0`.

### S8. 실험 11 — 직전 token identity 분해 (motion vs embedding)
- 보고서: `O/prev_action_identity_decomposition/PREV_ACTION_IDENTITY.md`; `summary.json`, `analysis_console.txt`, `figures/`.
- raw: `records.jsonl`(1,408), `units.jsonl`(1,408), `errors.jsonl`(0).
- 사전등록: not preregistered. 실행 2026-09-27(보고서), 스크립트·결과 커밋 bf15767 (2026-09-29).
- N: A− 365 / 52, A+ 1,043 / 156; 9 조건; 미학습 token 410개 제외(cos ≥ 0.999 군집).
- seed 0; GPU RTX 5090, torch 2.8.0+cu128 (4.3분).
- metric: 실험 7과 동일; motion 거리 = codebook 48-d L2; embedding 근접 = cos.
- 검정: `cboot` 2,000회, McNemar, Wilcoxon, `frac_effect`(seed 7), 공동 회귀 log bootstrap 1,000회.
- 스크립트: `S/prev_action_identity_decomposition.py`, `S/analyze_prev_action_identity.py`, `S/make_identity_figures.py`. 키: `groups.<g>.{conditions,vs_normal,vs_recent_gt,frac_of_recent_gt_effect}`, `geometry_vs_embedding`.

### S9. 실험 32 — Previous-action motion semantics (방향 vs 크기)
- 프로토콜: `O/motion_semantics_ablation/PROTOCOL.md` (실행 전 작성 2026-10-03).
- 결과: `RESULTS.md`, `analysis.md`, `summary.json`, `row_contrasts.json`, `run_meta.json`.
- raw: `records.jsonl`(1,408), `errors.jsonl`(0).
- **사전등록 커밋: af2eb10** (2026-10-03 15:10:07 UTC; PROTOCOL.md + `S/motion_semantics_ablation.py`), records 작성 19:28 → 실행 전 커밋 확인. 결과·분석 커밋: aee41e5 (19:30:04 UTC). 분석 스크립트 `S/analyze_motion_semantics.py`는 결과와 함께 aee41e5에 처음 커밋(사전등록 대상 아님; 지표·통계는 실험 11 함수 재사용).
- N: A− 365 / 52, A+ 1,043 / 156; 6 조건.
- seed 0; GPU RTX 5090 (5.4분).
- metric: token motion = codebook 끝 pose (dx, dy); 자기 오차 e = |m_o − m_g|; 증폭 정의 동일.
- 검정: `cboot` 2,000회, McNemar, Wilcoxon, `frac_effect`.
- 스크립트: `S/motion_semantics_ablation.py`, `S/analyze_motion_semantics.py`. 키: `groups.<g>.{conditions,vs_normal,vs_recent_gt,frac_of_recent_gt_effect,substitution_geometry}`, `row_contrasts.json` 각 키.

### S10. 실험 2 — CoT 인과 개입 (배경)
- 보고서 `O/cot_intervention/COT_INTERVENTION.md`; `summary.json`; run_meta: 159 장면 × 7 조건, seed 0, 5090.
- 사전등록: not preregistered(보고서가 실행 전 기준과 기준선 변경을 공개). 커밋 1541f30. arm C tensor 미저장으로 새 서버에서 재현 불가(REPRODUCTION.md).
- 핵심: 반대 CoT − 교정 CoT 실행 반전 +6.4%p [+2.7, +9.7]; template 재작성 A+ +17.0%p [+10.3, +26.5].
- 스크립트: `S/cot_intervention.py`, `S/analyze_cot_intervention.py`.

### S11. 실험 3 — Fast vs CoT (무편향 500 장면, 배경)
- 보고서 `O/fast_vs_cot_unbiased/FAST_VS_COT.md`; run_meta: 500 장면, seed 20260915, 5090, 49분.
- 사전등록: not preregistered. 커밋 1541f30.
- 핵심: CoT − natural ΔA+ −5.0%p [−8.0, −2.9], McNemar p = 4.65e-06 (3↑/28↓); ΔADE +0.50 m [+0.36, +0.70].
- 스크립트: `S/fast_vs_cot_unbiased.py`, `S/analyze_fast_vs_cot.py`.

### S12. 5090 재분석 / mechanism–selection link (참고)
- `O/gpu5090_reanalysis/GPU5090_REANALYSIS.md` (커밋 1b91b0f): 실험 18–24(best-of-N)만 5090 재실행. 기전 관련 언급: 원래 28 log의 "실패 단위 52개"에서 T = 0.01 자연 계획 A−가 3080 Ti 63–65% vs 5090 67–71%(best-of-N harness). 실험 5–11 수치에는 영향 없음(원래 5090).
- `O/mechanism_selection_link/{dev,heldout}{,_5090}/summary.json` (커밋 e3bc8db, 1b91b0f): 선택 vs 버림 rollout의 이탈 전/후 엔트로피(5090 held-out 이탈 전 −0.016 [−0.023, −0.009], 이탈 후 −0.396 [−0.420, −0.373]). 기전 섹션 A–F 표에는 사용하지 않음.

### S13. 패키지 내 재계산 산출물
- `/root/VLA/paper_quantitative_package/_parts/scripts/build_autovla_csvs.py`: summary json → CSV 복사, distinct log 개수, layer 점추정의 min/max/SD/CI 폭 중앙값, 실험 7 비율 점추정.
- `/root/VLA/paper_quantitative_package/_parts/scripts/rerun_equal_distance_analysis.sh`: 원 `analyze_equal_distance.py` 재실행(full + STRICT=1) → `_parts/recomputed/`.
