# RUN_COMMANDS — review_followup2 표·그림을 원자료에서 다시 만드는 명령 (2026-10-13)

- 모든 단계는 **CPU만** 쓴다(모델 추론·GPU 없음). 전체 모델 재실행은 필요 없다.
- 기존 결과 파일은 읽기만 하고, 출력은 각 `run_20261013/` 디렉터리에 쓴다. 다시 실행하면 그 디렉터리의 파일을 덮어쓴다(원자료는 건드리지 않음).
- 코드: `/root/VLA/autovla_misalignment_poc/scripts/review_followup2/` (commit **589ded1**, 2026-10-12 22:57 UTC). 통계 모듈은 모두 `stats_core.py`를 쓴다.
- 경로 약어: `S=/root/VLA/autovla_misalignment_poc/scripts/review_followup2`, `O=/root/VLA/autovla_misalignment_poc/outputs`.
- 런타임은 실제 실행 기록(메타 JSON, 로그, 파일 시각)에서 옮겼다. 기록이 없는 것은 "미기록"으로 적었다.

## 0. 환경

| 용도 | 환경 | 명령 |
|---|---|---|
| A1–A3, B, C, 테스트 | `autovla` conda env (Python 3.9.23, numpy/pandas/pyarrow/matplotlib/pytest 8.4.2) | `PY=/root/miniforge3/envs/autovla/bin/python` |
| A2 SpatialVLA 초기 상태 재생성 | SimplerEnv(ManiSkill2_real2sim, SAPIEN 2.2.2) + Mesa lavapipe software Vulkan, autovla env 위 | `source /root/VLA/simpler/env.sh; export CUDA_VISIBLE_DEVICES=""` (env.sh가 GPU 1을 지정하므로 CPU 전용으로 덮어씀) |

모든 명령은 `nice -n 19`로 실행했다. 셸 안전 규칙: 디렉터리 이동은 `cd "$S" || exit 1`처럼 존재 확인 후에만 한다.

## 1. 원자료 (입력)

| 단계 | 입력 파일 | 원자료를 만든 commit |
|---|---|---|
| A1/A2/A3 | `O/review_followup/P0B_cluster_stats/run_20261008/units.parquet` (exp 6/7/11/32/33), `O/review_followup/R8R9_raw_recovery/run_20261008/units.parquet`, `O/review_followup/P1A_gt_free_reference/run_20261008/units.parquet`, `O/review_followup/P1B_start_position/run_20261008/units.parquet`, `O/review_followup/P1C_new_log_replication/run_20261008/analysis/units.parquet`, `.../design_20261008/split.csv`, 선택·navhard 실험의 scene/log 목록 (`a2_overlap.py` 머리말) | e10374a (P0-B), 08d54f3 (P1-A/B, R8R9), 1216812 (P1-C) |
| A2 SpatialVLA | `O/cross_domain_temporal_replication/closed_loop/episodes.jsonl`·`frames/` (exp 34), `O/spatialvla_feedback_protection_ablation/rollouts{,_ext}/episodes.jsonl` (exp 35), `O/review_followup/P1D_dose_control/run_20261008/{units.json, rollouts/main/episodes.jsonl}` | f4cf886, 2b0212d, 60509f6 |
| B | 위 SpatialVLA episode 파일 + `O/review_followup2/A_spatialvla_states/run_20261013/state_manifest.csv` + `O/review_followup/P0C_exp35_fixed_axis/run_20261008/token_table.npz` | 위와 같음 + 589ded1 |
| C | `O/review_followup/P1A_gt_free_reference/run_20261008/{raw,raw_fixedpos}/records.jsonl`, `units.parquet`, `fixedpos/units.parquet`, `O/review_followup/P1B_start_position/run_20261008/units.parquet`, `O/review_followup/R8R9_raw_recovery/run_20261008/units.parquet`, `O/equal_distance_perturbation/records.jsonl`, `/root/VLA/autovla/codebook_cache/agent_vocab.pkl` (sha256은 `c_run_meta.json`) | 08d54f3, 1541f30/bf15767 |

일부 raw 디렉터리(`P1A .../raw/`, `P1B .../raw/`, `R8R9 .../raw/`, P1-C `mech/`, SpatialVLA `frames/`)는 용량 때문에 git에 없고 디스크에만 있다(`git status`의 untracked 항목).

## 2. 명령

### A. AutoVLA·Impromptu 감사와 drive 군집 재분석 → `O/review_followup2/A_autovla_cluster/run_20261013/`

```bash
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup2
PY=/root/miniforge3/envs/autovla/bin/python
cd "$S" || exit 1
nice -n 19 "$PY" a1_manifest.py            # manifest_autovla_impromptu.parquet (867,856행), manifest_columns.csv, manifest_keys.json, manifest_counts_by_experiment.csv
nice -n 19 "$PY" a2_overlap.py             # overlap_set_summary.csv, overlap_pairwise.csv, overlap_matrix_{drives,logs,scenes}.csv, drive_membership.csv, Aminus_log_to_drive.csv, overlap_checks.json
nice -n 19 "$PY" a3_cluster_reanalysis.py  # CLUSTER_REANALYSIS.csv, cluster_reanalysis_long.csv, leave_one_drive_out.csv, per_drive_effects.csv, p1c_subsets_drive_level.csv, verification_vs_reported.csv, a3_run_meta.json
nice -n 19 "$PY" a3_figure.py              # fig_log_to_drive_effects.png (CLUSTER_REANALYSIS.csv에서만 생성)
nice -n 19 "$PY" -m pytest -v -s test_stats_core.py > "$O/review_followup2/A_autovla_cluster/run_20261013/test_stats_core_output.txt"   # 9 tests, 54 s
```

- seed: drive bootstrap 20261013 (B = 10,000, LODO B = 2,000), P0-B log 재현용 seed "20261008 + 100 i / 700 + 10 i", P1-C 재현용 20261008. sign-flip은 G ≤ 26이면 전수(meet-in-the-middle), 그 이상이면 MC 200,000(`a3_run_meta.json`). 테스트 RNG seed 13.
- 런타임: a3 시작 2026-10-12 21:57:17 UTC(`a3_run_meta.json`); a1/a2/a3 개별 소요 시간 미기록. pytest 54 s.
- 검증: `verification_vs_reported.csv`가 P0-B log 수준 추정·전수 p와 모든 대비에서 일치해야 한다. P1-C 6개 대비는 `decisions.json`과 일치해야 한다.

### A2. SpatialVLA 초기 상태 cluster → `O/review_followup2/A_spatialvla_states/run_20261013/`

```bash
source /root/VLA/simpler/env.sh; export CUDA_VISIBLE_DEVICES=""
S=/root/VLA/autovla_misalignment_poc/scripts/review_followup2
RUN=/root/VLA/autovla_misalignment_poc/outputs/review_followup2/A_spatialvla_states/run_20261013
cd "$S" || exit 1
mkdir -p "$RUN/raw"
for T in google_robot_pick_coke_can google_robot_move_near; do
  nice -n 19 python svla_reset_states.py --task "$T" --out "$RUN/raw"                         # raw_<task>.jsonl: canon + repeat + drift (약 1–2분/과제)
  nice -n 19 python svla_reset_states.py --task "$T" --out "$RUN/raw" --fresh-check <seeds>   # fresh_<task>.jsonl: 새 프로세스 canonical reset (7개/과제)
done
nice -n 19 python svla_reset_states.py --task google_robot_move_near --out "$RUN/raw" --mixed 600     # mixed_drift.jsonl: 두 과제 env를 한 프로세스에서 무작위 순서 (과제당 300)
nice -n 19 python svla_reset_states.py --task google_robot_move_near --out "$RUN/raw" --stepped 200   # stepped_drift.jsonl: exp 35 episode 재생 후 reset (약 23분)
nice -n 19 python cluster_states.py --run "$RUN"   # state_manifest.csv, cluster_counts.csv, cluster_sizes.csv, cluster_overlap.csv, drift_summary.csv, determinism.json, verification_summary.json, p1d_episode_index.json
```

- 허용오차와 cluster 정의는 `METHOD.md`(cluster 계산 전 작성)에 고정돼 있고 `cluster_states.py` 상단 상수와 같다. 바꾸지 않는다.
- `--fresh-check`의 정확한 seed 목록, `--mixed`/`--stepped`의 정확한 인자는 실행 기록에 남아 있지 않다. 위 값은 `raw/fresh_*.jsonl`(과제당 7행), `drift_summary.csv`(mixed 과제당 300, stepped 98 + 102)에서 역산한 것이다. 결과 파일의 행 수로 확인할 수 있다.
- 런타임(파일 시각): raw 21:45–21:46, fresh 21:47, mixed 21:48–21:49, stepped 21:51–22:14 UTC; cluster_states 22:15–22:17.
- reset RNG = `RandomState(seed)`; policy는 돌리지 않는다.

### B. SpatialVLA 성공률 직접 효과 → `O/review_followup2/B_spatialvla_direct/run_20261013/`

```bash
bash /root/VLA/autovla_misalignment_poc/scripts/review_followup2/b_regenerate.sh
# = b_extract.py (episodes.csv, pairs.csv; ~10 s) → b_check_conditions.py (condition_record_check.csv; ~1 min)
#   → b_direct_effects.py (DIRECT_EFFECTS.csv 819행, B4_TRAJECTORY_VS_SUCCESS.csv, INTERPRETATION.csv, b_stats_meta.json; ~12 min)
#   → b_figures.py (fig_direct_effects.png, fig_tcp_vs_success.png)
# 후처리만 다시: nice -n 19 $PY b_direct_effects.py --post   (_raw_rows.pkl 사용; 72 s)
```

- seed 20261013; bootstrap B = 10,000; sign-flip MC 100,000 (G > 26); MC 역산 CI R = 20,000 (seed 20261013 + 99); 전수 기준 G ≤ 26; margin ±10 / ±5 pp (`b_stats_meta.json`).
- 전제: A2의 `state_manifest.csv`가 먼저 있어야 한다.
- 주의: run 디렉터리의 `b_direct_effects.log`(22:48 UTC)는 **첫 시도의 기록**이며 "TOST side reversal failed exp34 r1e4/opposite F-C" assertion으로 끝난다(MC 오차 허용폭 문제). 현재 CSV(22:50)와 meta(22:51, runtime 72 s)는 그 뒤 `--post` 재실행의 산출물이다. 코드의 허용폭은 현재 `6·sqrt(p(1−p)/MC) + 2e-5`(MC) / 1e-9(전수)다. 다시 실행해 assertion이 통과하는지 확인할 것.

### C. 외삽 baseline·교정 window → `O/review_followup2/C_baseline_window/run_20261013/`

```bash
cd /root/VLA/autovla_misalignment_poc/scripts/review_followup2 || exit 1
nice -n 19 /root/miniforge3/envs/autovla/bin/python c_baseline_window.py
# c1_excess_effects.csv (192행), c1_predeclared_and_gt_drive.csv, c1_units_counterfactual.parquet, c1_matching_checks.json,
# c2_position_length.csv, c2_recovery_fractions.csv, c2_levels_p1b.csv, fig_position_vs_length.png, fig_p1a_excess.png, c_run_meta.json
```

- seed 20261013, bootstrap B = 10,000, 전수 drive sign-flip. 런타임 20.6 s (`c_run_meta.json`, 2026-10-12 22:11:43 UTC). 입력 sha256은 `c_run_meta.json`에 있다.
- CM_0(무교란 constant-motion)은 원자료에 없어 `p1_common.rollout_np`(numpy decoder 사본)로 다시 만든다(실제 decoder 대비 오차 ≤ 1.9e-5 m).

## 3. 주요 표·그림 ↔ 산출물

| 표·그림 (문서) | 파일 | 명령 |
|---|---|---|
| drive 군집 재분석 표, LODO (`CLUSTER_REANALYSIS.md` §2) | `CLUSTER_REANALYSIS.csv`, `leave_one_drive_out.csv` | A3 |
| log → drive 효과량 그림 | `fig_log_to_drive_effects.png` | a3_figure |
| drive/log/장면 중복표 (`DATA_AUDIT.md` §2) | `overlap_matrix_*.csv`, `overlap_set_summary.csv` | A2 overlap |
| SpatialVLA 고유 초기 상태 수 (`DATA_AUDIT.md` §3) | `cluster_counts.csv`, `state_manifest.csv` | A2 SpatialVLA |
| task/config별 직접 효과·동등성·상호작용 그림 | `fig_direct_effects.png` | B |
| TCP 궤적 지표 vs 성공률 그림 | `fig_tcp_vs_success.png`, `B4_TRAJECTORY_VS_SUCCESS.csv` | B |
| P1-A 모델 − 외삽 초과 변화 그림 | `fig_p1a_excess.png`, `c1_excess_effects.csv` | C |
| 교정 위치 vs 길이 그림, 90% 회복 분모 | `fig_position_vs_length.png`, `c2_position_length.csv`, `c2_recovery_fractions.csv` | C |
| 통계 코드 검증 (동일 쌍 → 0, 조건 뒤집기 → 부호·CI 반전, cluster 복제 → CI 불변, 경계 모의, 직접 효과 vs 상호작용) | `test_stats_core_output.txt` | pytest |

## 4. 원 실험(GPU) 재실행이 필요한 경우만

이 문서의 분석은 GPU가 필요 없다. 원 rollout 자체를 다시 만들어야 할 때의 진입점은 다음과 같다(이번에는 실행하지 않음).
- AutoVLA 기전 실험 5–11: `/root/VLA/CONCLUSIONS.md` §12의 명령. R8R9 복구 실행: `scripts/review_followup/` (P1-A/B/R8R9 protocol, b212637).
- P1-C: `bash scripts/review_followup/p1c/p1c_driver.sh` (amendment 1·2 포함), 분석 `scripts/review_followup/p1c/analyze_p1c.py` (sha256 60345ded…).
- P1-D: `O/review_followup/P1D_dose_control/run_20261008/launch.sh`, 분석 `scripts/review_followup/p1d/analyze_p1d.py` (sha256 bb58d1f2…).
- SpatialVLA 실험 34/35: `scripts/cross_domain_temporal/svla_rollouts.py`, `scripts/spatialvla_protection/svla_protection.py` (환경 `/root/VLA/simpler/env.sh`, RTX 5090).
