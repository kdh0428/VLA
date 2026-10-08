# Source map — navhard (실험 25, 26, 27, 29, 30)

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

## 각 결과의 정확한 JSON 키

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
