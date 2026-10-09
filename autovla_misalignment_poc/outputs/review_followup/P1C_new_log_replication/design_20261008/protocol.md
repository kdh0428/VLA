# P1-C protocol: new-log replication of the AutoVLA action-context mechanism (preregistration DRAFT)

- Status: **DRAFT, not yet committed or approved.** Written 2026-10-08, before any P1-C inference.
- Plan: `/root/VLA/VLA_experiment_plan_20261008.md` §2 and §8.
- Design inputs:
  - `split.csv`, `leakage_check.md`
  - `power_simulation.csv/.png`, `analysis.md`
  - P0-A (`../../P0A_audit/run_20261008/metric_spec.md`)
  - P0-B (`../../P0B_cluster_stats/run_20261008/`)
- Every setting below comes from existing dev results: PoC experiments 6–9 and 32, and the P0-A/P0-B definitions. **No setting may be changed after evaluation data exist.** A change needs a dated amendment in this file, committed before the affected run, and the change is reported in the paper.
- Labels: [확인한 사실] / [계산한 결과] / [미확인] / [가설].

## 1. Questions

- **Q1 (failure-conditional replication).** In scenes from new drives where the natural plan fails (A−), do the dev mechanism contrasts hold?
  - Recent-GT context correction reduces amplification.
  - Re-inserting the Normal branch's previous token into a GT-history context restores amplification.
  - A direction error is more harmful than an equal-distance magnitude error.
- **Q2 (population effect).** Among all scenes from new drives with a natural first token deviation, regardless of natural success or failure, how large are the same contrasts?
- **Q3 (failure selection).** How much larger is the effect in the failure stratum than in the success stratum? This separates a failure-selected effect from a population effect.

These questions concern open-loop, single-model behaviour only (AutoVLA, `AutoVLA_PDMS_89.ckpt`). The limitation that the two driving models share a backbone remains. Nothing here licenses closed-loop claims.

## 2. Data split ([계산한 결과], `split.csv`)

| Role | Logs (segments) | Drives | Use |
|---|---|---|---|
| `dev_discovery` | 28 (PoC, shards 0–5) | 20 | Existing results only. All settings were fixed here. |
| `eval` | 64 (shards 6–31) | 24 (16 day-vehicles) | P1-C evaluation. 5,671 navtest scenes. |
| `pilot_harness_check` | 2 | 2 | Harness/stream checks only. Never analysed for effects. |
| `excluded_drive_adjacent_to_dev` | 42 | – | Same drive as a dev log. Not used. |
| `reserve_non_navtest_not_used` | 11 | – | Outside navtest. Not used. |

- Eval logs have never been used in any perturbation or mechanism experiment. They were used only for selection, detection and PDM experiments (exp 20–31), which decoded natural and sampled plans.
- **Cluster = drive** (`<date.time>_veh-NN`).
- Sensitivity sets:
  - (S1) non-navhard-source logs (30 logs, 13 drives).
  - (S2) day-vehicle-disjoint logs (23 logs, 7 drives).
  - (S3) clustering at day-vehicle level (16 clusters).

## 3. Population, strata, sampling

### Stage 1: natural pass (outcome-blind to every perturbation)

- Decode every navtest scene of the 64 eval logs once in the evaluation harness:
  - natural fast path, T = 0.01, seed 0, RTX 5090 (GPU 1), bf16, batch 1.
  - Stub = the `_planner.Planner` stem + free decoding to the first action token. This equals the arm-N procedure (§6, check P1).
- Store, per scene:
  - token ids
  - 10 action tokens
  - trajectory
  - GT tokens (the same tokenisation routine)
  - t* = first k with pred[k] ≠ GT[k]
  - `a_eval` label (A+/A−)
  - FDE5
- Define strata from these labels only. The existing planning labels (`work/frame_labels_5090.json`) are reported for agreement only.

### Eligibility E

- Natural plan has 10 poses.
- No CoT stub.
- No runaway (more than 10 action tokens).
- t* exists and **t* ≤ 7**: at least one Recent-GT application (8−t* ≥ 1), and dev had t* ≤ 7.

Scenes that are not eligible are counted per log by reason and not perturbed:

- No mismatch: expected ≈57%.
- t* ∈ {8, 9}: expected ≈3%.
- Invalid.

### Strata (within E)

| Stratum | Definition | Inclusion | Expected size (planning labels) |
|---|---|---|---|
| F (failure) | natural A− | all, π = 1 | 111 scenes in 20 drives |
| S (success) | natural A+ | per log segment, simple random sample without replacement of m_seg = min(**m = 4**, N_S,seg); π = m_seg / N_S,seg | 248 of 2,090 |

- **S sampling order.** Within each segment, sort S scenes by `sha256("P1C-20261008|" + scene_token)` and take the first m_seg. This is fixed before stage 1 and is independent of outcomes.
- **Population weights.** w = 1/π. F scenes have w = 1. S scenes have w = N_S,seg / m_seg.
- **Out of scope** (reported as shares, not perturbed):
  - No-mismatch scenes. The mechanism is defined at the first natural deviation; an injected-perturbation design for them belongs to P1-A/B.
  - t* > 7.
  - The population estimand is therefore **conditional on a natural first token deviation at t* ≤ 7**. The paper must say so.

### Units

- Per scene: `original` (forced natural token at t*) plus distance-matched `alt_*` tokens, using the unchanged dev rule:
  - 8 direction sectors, one per sector.
  - Tolerance max(5 mm, rel·d(GT, pred)), with rel = 0.10, widened to 0.20/0.35 only if fewer than 5 sectors qualify.
- All units are kept (dev ≈ 7 per scene).
- `original_reseed` and `gt` (exp 6 only) are not units.

## 4. Conditions

- Each block uses its own harness, unchanged: same script version, conditions, batch composition and seed rule as dev.
- Each contrast uses its own block's Normal (P0-A F6: Normal differs between harnesses).
- Per-step sampling seed: `sha256(f"{seed}:{token}:{pert}:{k}")`, T = 0.01, seed 0.

| Block (dev script) | Conditions used | P0-A definition |
|---|---|---|
| H7 `action_history_causal.py` | `normal`, `recent_gt`, `gt_history` | recent_gt = sliding replacement of position k−1 by GT, at every step k ≥ t*+2 (8−t* times). Not a single correction. |
| H8 `prev_action_state_patching.py` | `gt_history`, `reverse@emb` | GT-history context. At position k−1 the embedding comes from the **separate Normal row in the same batch**, not from the reverse branch's own output. |
| H9 `temporal_feedback_window.py` | `normal`, `win1`, `win3`, `win4`, `gt_history` | win_w = GT at fixed positions t*+1..t*+w, kept in the context afterwards. |
| H32 `motion_semantics_ablation.py` | `dir_wrong_mag_ok`, `dir_ok_mag_wrong` (+ `normal`, `recent_gt`, `mirror_same_dist` secondary) | Sliding substitution at k−1, built from GT and the row's own token. |

- The executed output is always `pred[:t*] + [f] + own`. Corrections enter only the context.
- Every other condition these scripts compute by default (attention masks, layer sweep, delays) is run because the scripts are unchanged. Those conditions are **exploratory only** and not reported as P1-C results.
- Reverse naming: the paper calls H8 reverse a "Normal-branch previous-token injection". The self-referential 4-condition design (plan §6) is P1-A; it does not replace this block.

## 5. Outcomes and estimands

- **Primary outcome:** amplification = (not A+) ∧ FDE5 > 3.0 m, per unit (`analyze_action_history.unit_metrics`, unchanged).
- **Secondary outcomes:** FDE5 (m) and recovery (A+).
- **Scene effect:** d_s = mean over the scene's units of (cond_a − cond_b).
- **Estimators:**
  - θ_F = Σ_{s∈F} d_s / |F| (scene-weighted, primary).
  - θ_P = Σ_{s∈F∪S} w_s d_s / Σ w_s (Hajek, scene-weighted, primary).
  - θ_S is defined the same way within S.
  - Q3 uses Δ = θ_F − θ_S.

### Primary contrasts

Two families, each with Holm FWER control:

| Family | Contrast | Predicted sign | MME (justification in analysis.md §4) |
|---|---|---|---|
| F | RN = recent_gt − normal (H7) | − | −10 pp |
| F | RV = reverse@emb − gt_history (H8) | + | +10 pp |
| F | DM = dir_wrong_mag_ok − dir_ok_mag_wrong (H32) | + | +5 pp |
| P | RN | − | −2 pp |
| P | RV | + | +2 pp |
| P | DM | + | +2 pp |

### Secondary (not in any family, reported with CIs)

- Q3 Δ for RN, RV and DM.
- Windows, in F and S: amplification for normal / win1 / win3 / win4 / gt_history, and the gap fraction f(w) = (normal − win w)/(normal − gt_history) for w = 1, 3, 4.
  - Dev expectation: 47 / 90 / 96%.
  - Prespecified secondary checks: win4 − gt_history within ±5 pp (90% CI, equivalence), and f(3) ≥ 0.75.
- FDE5 versions of every primary contrast.
- Per-drive and per-log effects (table + forest plot).
- City strata.

## 6. Checks before evaluation (pilot; failure blocks the eval run)

- **P1.** Planner natural pass vs stored `full_extract` arm N on 30 PoC scenes (images are local).
  - Agreement of the 10 action tokens is reported.
  - If < 85% full-sequence agreement, or the stub ids differ in more than 5% of scenes, stop: the harness is not equivalent to the dev arm N.
  - Reference values: other dev harnesses agreed on Normal 71.6–89.2% (P0-A A5), and GPU swaps changed t* in 3% of scenes [계산한 결과].
  - Because the threshold is a judgement call, the observed value is reported whatever it is.
- **P2.** H8 debug sanity on 2 scenes must reproduce:
  - patch@emb ≡ recent_gt (logits L1 = 0)
  - patch@L35 ≡ gt_history
  - prefix-KV checksum unchanged
- **P3.** End-to-end on the 2 pilot logs: 3-camera streaming, natural pass, ED units, all 4 blocks.
  - Record per-unit runtimes.
  - Record peak disk.
- Pilot effects are **not looked at**. The analysis script runs on pilot output only to test that it executes, with effect columns masked.

## 7. Exclusions (complete list)

- **Scene level:** not eligible (§3); image/feature load error; harness exception.
  - Counted per log and reason.
  - If more than 5% of a log's eligible scenes fail, the log is flagged and S4 (without flagged logs) is reported.
- **Unit level:** none in the primary analysis (dev had none).
  - Invalid/parse/runaway outputs count as amplification = 0 and recovery = 0. This is the dev rule for invalid output.
  - Sensitivity: common inclusion set without any unit that is invalid in any condition of that block.
- **Condition level:** the dev OOD rule (Δ first-step entropy > 1 nat) is reported, never applied to primary conditions.
- **No exclusion may depend on a perturbation outcome.**

## 8. Inference

- **Point estimates:** scene-weighted (primary); unit-weighted and drive-equal (sensitivity).
- **p-values:** drive-level sign-flip on drive totals T_g = Σ_{s∈g} w_s d_s.
  - Exact enumeration: 2^G with G ≤ 24 clusters, about 1.7e7 sign patterns.
  - Holm within family.
  - Assumption: under H0 each drive total is symmetric about 0 and drives are independent. This is not randomization inference.
- **Calibration by simulation** (`work/signflip_calibration.csv`, conservative cluster model):
  - Family F: type I is 4–6.5% at nominal 5%.
  - Family P: type I is up to 9–11% at nominal 5% (HT-weighted totals are skewed).
  - **Family P therefore uses a nominal family α of 0.025.** The Holm first step is then .0083.
  - Simulated type I at α = .0167 was 2.8–6.5%. At .0083 it was not simulated separately [미확인]; it is expected to be lower.
- **CIs:** 95% drive block bootstrap (B = 10,000, percentile, strata kept inside drives) and cluster-t (df = G−1), both reported.
  - With about 20 clusters both under-cover in simulation, so the sign-flip p governs significance.
  - Test-inversion sign-flip CIs are given for the primary contrasts.
- **Sensitivity:** S1–S4, weighting variants, and day-vehicle clustering. Report every one, whichever way it falls.

## 9. Decision rules (fixed now)

- **Replicated:** the contrast's Holm-adjusted p < family α and the sign matches the prediction.
- **Not replicated, below MME:** the 90% CI lies inside (−|MME|, +|MME|).
- **Inconclusive:** anything else. Write it as "not detected / not established". Never write "no effect".
- **Expected limitation (from the simulation, before data):**
  - F-family RN and RV have power ≥ 0.96 for dev-sized effects. Power is only 0.12–0.33 at the MME and 0.40–0.72 at half the dev effect (Holm first step, calibrated; range = drive vs segment cluster model).
  - F-family DM has 0.19–0.33 power at the dev effect (0.37–0.57 at unadjusted α = .05, sign-flip check).
  - P-family RN and RV have power ≈ 1.0 at the dev effect and 0.22–0.48 at the MME. P-family DM has 0.16–0.40 at the dev effect.
  - So a non-significant DM result is the expected outcome. It is "inconclusive", not evidence against the direction effect.

## 10. Stopping

- **Fixed design:** all 64 eval logs, one pass, no interim effect analysis, no optional stopping, and no adding logs after looking.
- **Technical pauses, no look at effects:**
  - pilot check failure (fix, re-pilot on the same pilot logs, document)
  - harness error rate > 5% of eligible scenes in a shard
  - free disk < 3 GB
  - GPU-time overrun > 1.5× the estimate (analysis.md §6)
  - Resume after a pause only with an amendment entry.
- If the run cannot be completed, report it as incomplete, listing the logs processed. Do not analyse a subset as if it were the planned design.
- **Analysis code** (`analyze_p1c.py`, to be written) is frozen and its hash recorded here **before** stage 2 output exists.

## 11. Records (plan §2)

- `manifest.json` records:
  - checkpoint size/hash: `AutoVLA_PDMS_89.ckpt` 16,292,664,780 B
  - codebook sha256 e6bf8eff…
  - autovla commit ba34eed
  - repo commit
  - script sha256
  - GPU, torch 2.8.0+cu128, bf16
  - batch per block
  - seeds
  - the shard list and the S-sample token list with π
- Raw rows: `log_id, drive_id, scene_id, stratum, pi, weight, t_star, perturbation, condition, block, seed, decode_j, amplification, fde5, recovery, parser_status, exclusion_reason`, plus all plan §2 fields (null where not applicable).
- Denominator flow per log: scenes → eligible → F/S → sampled → units → analysed.

## Lead 결정 (2026-10-08, 사전 등록 commit 직전, 평가 데이터를 보기 전)
- D1: dev와 drive가 겹치지 않는 64 log 전부를 쓴다. navhard 출처 log와 같은 날·같은 차량 log는 사전 지정 민감도 분석으로 따로 본다.
- D2: 모집단은 t* ≤ 7인 장면이다. 이탈이 없는 장면은 비율로만 보고한다.
- D3: 주 분석은 장면 가중이다(P0-B와 같음). drive 균등 가중은 민감도 분석이다.
- D4: 실험 8의 reverse@emb를 유지하고 "Normal-branch token injection"으로 표기한다.
- D5: Direction−Magnitude는 계획서 §8에 따라 두 primary family에 그대로 둔다. 검정력이 부족하다는 점을 미리 밝힌다.
- D6: family F는 α = 0.05 + Holm, family P는 α = 0.025 + Holm이다. 주 p값은 drive 수준 exact sign-flip이다.
- D7: 필요한 코드(natural 전용 1단계 writer, token 목록 옵션, 3-카메라 스트리밍 driver, 고정된 분석 스크립트)는 새 파일로 구현한다. 구현이 끝나면 분석 스크립트를 평가 데이터를 보기 전에 추가로 commit한다. pilot 점검 P1–P3 통과를 본 실행의 조건으로 한다.
- D8–D9: 사용자 승인(2026-10-08). 약 100 GB를 shard 단위로 스트리밍하고, GPU 약 5–7.5시간이다. 예상의 1.5배를 넘으면 기술적으로 중단하고 점검한다.

## Amendment 1 (2026-10-08, pilot 이후·평가 데이터 생성 전; 평가 결과를 본 적 없음)
- **pilot 점검 P1 기준 변경**
  - 원 기준: Planner natural pass와 저장된 full_extract arm N의 10-token 완전 일치 ≥ 85%.
  - pilot 결과: 74.6%(97/130 PoC 장면).
  - arm N 절차 자체를 다시 돌려도 76.2%(99/130)로 85%에 도달하지 못한다. 원인은 T = 0.01 샘플링에서 근소한 logit 차이를 난수열이 가르는 것인데, full_extract는 run 전체에 seed를 한 번만 넣어서 장면 단위로 재현할 수 없다. P0-A에서 dev harness 간 일치율도 71.6–89.2%였다.
  - 즉 85%는 arm N의 자기 재현성보다 높게 잡힌 기준이었다.
  - 새 기준은 아래 셋을 모두 만족하는 것이다.
    1. Planner의 일치율이 arm-N 재실행의 일치율보다 5%p 넘게 낮지 않다. 관측값은 74.6 vs 76.2로 충족한다.
    2. stub 일치 ≥ 95%. 관측값 100%.
    3. eligibility+stratum 일치 ≥ 90%. 관측값 93.1%.
  - 결과 지표, estimand, 분석 방법은 바꾸지 않는다.
- **결정 규칙의 90% CI:** sign-flip 검정을 뒤집어 얻은 90% CI를 주 CI로 쓴다. bootstrap 기준 판정도 함께 적는다.
- **실행 순서와 정의**(설계는 그대로이고 명시만 한다):
  - 1단계는 shard 단위로 돌리고, 선택된 장면의 영상만 남긴다. 단위 생성과 4개 기전 block은 선택된 장면 전체에 대해 한 번씩 돌린다.
  - shard당 오류율 > 5%면 멈추는 규칙은 1단계에 적용한다.
  - 민감도 부분집합 S4(flag된 log)의 분모에는 1단계 실패도 넣는다.
  - Q3의 Δ는 bootstrap과 cluster-t 구간만 보고한다.
  - Planner 절차에서 runaway는 항상 0이다.
  - 1단계 seed key는 "0:{token}"이다.
- **GPU 예산:** 측정 기반 재추정치는 9.35 GPU 시간이다(외부 프로세스와 GPU 공유). 사용자 재승인 여부를 아래에 기록한다.
- **분석 스크립트 고정:** `scripts/review_followup/p1c/analyze_p1c.py`, sha256 60345dedfc9d3aeb14d21ec74590c6bb9a00a386b8962754f615be58b69cfc94.
- GPU budget re-approved by the user on 2026-10-09: 9.35 GPU h (1.5x pause line 14 h); P1-C runs before P1-D.
