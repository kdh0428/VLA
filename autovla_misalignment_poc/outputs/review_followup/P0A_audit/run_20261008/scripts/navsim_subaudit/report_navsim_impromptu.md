# P0-A sub-report: NAVSIM evaluator provenance + Impromptu (exp 33) parsing/GT/intervention

## (a) NAVSIM

### Code copies (확인한 사실)
| copy | git | commit / date | dirty | version | used by |
|---|---|---|---|---|---|
| /root/VLA/autovla/navsim | vendored inside AutoVLA repo (origin ucla-mobility/AutoVLA), no own .git | AutoVLA ba34eed (2026-05-29 "checkpoint release"); navsim subtree last touched ba34eed | tracked files clean (only untracked Qwen/dataset) | setup.py version 1.1.0 (NAVSIM v1) | default `import navsim` in conda env autovla (editable install `__editable__.navsim-1.1.0.pth` -> /root/VLA/autovla/navsim/navsim); open-loop PoC scripts; held-out PDMS (exp 21-24: S/pdm_score_candidates.py:36-37 inserts autovla/navsim, :53 default_scoring_parameters.yaml, :73 pdm_score) |
| /root/VLA/navsim_v2 | origin autonomousvision/navsim, branch main | HEAD 0a380a9 (2025-10-27 "Revise highlights and changelog in README.md"), `git describe` v2.2-1-g0a380a9; cloned 2026-09-29 21:48:55 UTC (.git/logs/HEAD) | no tracked modification (`git diff HEAD` empty); 4 untracked split yamls navhard_half{,2} | NAVSIM v2.2+ | navhard exp 25-27, 29, 30 (official run_pdm_score_from_submission.py, via S/navhard_group_scores.py:14-15; navhard_submission.py:38 / navhard_metric_cache.py:67 / navhard_safety_flags.py:124 assert "navsim_v2" in navsim.__file__) |
| /root/VLA/navhard | not a repo (inside /root/VLA) | data + exp outputs (75 scoring runs under navhard/exp/*/<ts>/) | - | - | scoring outputs |
| `git ls-remote origin main` (2026-10-08) | | 0a380a9 = still upstream main HEAD → no later upstream evaluator commit exists |

### 2025-09-29 human-filter fix (확인한 사실)
- Upstream commit 359c7f7 "Fix human filter bug (Issue #151)" (2025-09-29, navsim/evaluate/pdm_score.py, +README changelog) is an ancestor of the local HEAD 0a380a9, and the working-tree file navsim_v2/navsim/evaluate/pdm_score.py is unmodified (mtime = clone time). The fixed code is at navsim_v2/navsim/evaluate/pdm_score.py:171-218 (skip_columns incl. pdm_score; recompute multiplicative_metrics_prod and weighted_metrics when any metric is overridden by the human filter).
- The fix matters for the final score: run_pdm_score_from_submission.py compute_final_scores (line 146 ff.) computes score = multiplicative_metrics_prod × weighted average of weighted_metrics, i.e. exactly the columns the fix recomputes.
- All 75 navhard scoring runs (navhard/exp/*/2026.09.30.14.59.28 … 2026.10.03.18.58.44) have `human_penalty_filter: true` in code/hydra/config.yaml, and all 75 log.txt reference /root/VLA/navsim_v2/navsim/planning/script/run_pdm_score_from_submission.py. All runs post-date the clone (2026-09-29 21:48 UTC).
- 계산한 결과/추론: the script cannot run against the v1.1 copy (v1.1 has `SceneFrameType` commented out, autovla/navsim/navsim/common/enums.py:3, and no human filter at all), so the scoring imports were v2 → the evaluator used for every navhard EPDMS includes the 2025-09-29 fix. (Direct import-path log of `navsim.evaluate.pdm_score` per run not recorded → this last link is inferred, not logged; the asserts in the submission/metric-cache scripts confirm v2 for those steps.)
- NAVSIM v1 PDMS (exp 21-24 held-out, Claim 14: PDMS +0.0095, F1 +0.0315, collisions): uses the vendored v1.1 evaluator, which has no human-penalty filter (grep 'human' in autovla/navsim/navsim/evaluate/pdm_score.py: no hit) → the 2025-09-29 fix is not applicable (확인한 사실). Upstream base commit of the vendored v1.1 copy: 미확인 (no git metadata of its own).
- Note (확인한 사실): the current-frame safety flags (S/navhard_safety_flags.py:78) deliberately use `PDMScorerConfig(human_penalty_filter=False)`; this is the filter feature, not the evaluator.
- Provenance gap (확인한 사실): none of SOURCE_MAP.md / PAPER_EVIDENCE.md / PAPER_NUMBERS.md records the navsim_v2 commit (0a380a9) or the v1.1 vendored source; they only say "공식 run_pdm_score_from_submission.py".

### Paper numbers depending on the v2 evaluator
paper_navhard_table.csv (all 11 rows: normal 0.2287, ranksum, max_loglik, collision_only, dac_only, filter_only 0.3333, filter_random, F1 0.3470, F1_maxll, oracle16, oracle17 0.4053), PAPER_NUMBERS Claim 15 (+0.1046, 88%, Shapley 86%), Claim on oracle (0.405, 31.6% zero tokens), exp 25 half-1 (0.239 → 0.365), exp 26 half-2, exp 27/29/30, navhard_verify.json. v1 evaluator: Claim 14 PDMS numbers (+0.0095 [+0.0046,+0.0148], F1 +0.0315, collisions 0.56→0.23%, 0.10%), candidate-count PDMS curve.

## (b) Impromptu exp 33 (scripts commit 038a6d3 prereg; analysis 698148a; all clean vs HEAD)

### Parsing (확인한 사실)
- WP_RE = `\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]` (impromptu_core.py:34); parse_wps = all matches (:50-51); first 10 used.
- Natural: HF generate greedy, rep. penalty 1.05, max_new_tokens 200 (:133-143); <10 waypoints → RuntimeError → scene to errors.jsonl (exp_ab.py:48-50, exp_cde.py:180-182). errors.jsonl empty in both (0 scenes).
- A/B rows: ok = 10 waypoints (exp_ab.py:61-62); analysis drops the whole scene if any row not ok (analyze_temporal.py:76). Raw: 2,748 scenes, 20 excluded → 2,728 (matches RESULTS). Not-ok rows only in perturbation rows: m0.2_d45 4, m0.2_d135 5, m0.2_d270 1, m0.2_d225 1, m0.5_d135 6, m0.5_d90 2, m0.5_d315 1 (계산한 결과).
- C/D/E: one waypoint per step via FastWaypointDecoder (stop at first token containing ']', max 20 tokens; :184-226). Parse failure → value imputed by copying previous own waypoint (or w1) and logged in parse_fail (exp_cde.py:201-204); analysis excludes the unit (all rows) if it or the scene's δ=0 reference unit has any parse_fail (analyze_temporal.py:142). Raw: 300 scenes / 28 logs, 2,400 units, 2 parse-fail events (row normal 1, dir_wrong_mag_ok 1) → 2 units excluded → 2,398 (matches RESULTS). Ref-unit failures 0 (계산한 결과).

### GT generation / coordinates (확인한 사실)
- Pose = (ego2global_translation x,y, yaw of ego2global_rotation) from navsim log pkl (impromptu_core.py:61-63), same definition as NAVSIM EgoStatus (autovla/navsim/navsim/common/dataclasses.py:389-395). Ego frame of the current frame, x forward, y left (:66-68). Future = frames i+1..i+10 (:99-100); history statuses frames i-3..i (:78).
- 계산한 결과: all 2,748 PoC scenes have i ≥ 3 and i+10 in range (no silent negative-index wrap); frame spacing 0.4993–0.5006 s. Impromptu GT equals AutoVLA full_extract trajectory_gt[:10, :2] within 0.0005 m (3-decimal rounding) for all 2,748 scenes.
- Text quantisation: fmt_wp python round(·,2), -0.0→0.0 (:38-42); context prefix " [x, y]," per waypoint (:45-47). Stored GT/ctx in records rounded to 3 decimals (exp_cde.py:197, 205); metrics use the 3-decimal GT.
- No codebook / token quantisation (text digits); waypoints are ABSOLUTE positions, so a GT context waypoint also resets accumulated position (RESULTS.md:175 acknowledges GT-information leakage).

### Interventions (exp_cde.py context(), :116-155) (확인한 사실)
- Executed/output trajectory = [forced w1] + row's own generated w2..w10 for every row (exp_cde.py:208); only the text context changes (context-only). Forced w1 = natural w1 + 0.5 m × direction, kept in all rows' context (:120, :187).
- KV: prompt (images+question+ANSWER_HEAD) KV computed once per scene (:183, impromptu_core.py:159-166); for each step and row the whole assistant suffix (context waypoints as text) is re-tokenised and re-fed on a fresh copy of the prompt cache (impromptu_core.py:195-204) → no stale KV of replaced waypoints; context and output strictly separated.
- recent_gt: at step k (k=3..10) only j = k−1 is GT (:129-130); sliding, re-applied at every step → 8 applications per unit, width 1 waypoint per forward; earlier positions keep the row's own outputs.
- win_w: waypoints j ∈ [2, 1+w] are GT at every later step (:125-126) → a fixed absolute span (first w waypoints after w1), re-applied at each step k > j (not a single one-time correction). For k ≤ w+2 it is identical to gt_history.
- gt_history: all j=2..k−1 GT (:127-128).
- reverse: j=2..k−2 GT, j = k−1 = own[reverse][k−1], i.e. the error generated by the REVERSE branch itself under its GT-history context (own is per-row, :193/:200-202) — NOT imported from the Normal branch. At k=3 reverse context = normal context (w1 + own w2).
- motion substitutions (near_gt, dir_ok_mag_wrong, dir_wrong_mag_ok, random_mag_matched) replace only j = k−1 (:133-151), deterministic RNG sha256(token, dir, row, k).
- Numeric path (확인한 사실): natural uses HF generate (batched, left padding), C/D/E rows use the custom FastWaypointDecoder (manual repetition penalty) — normal row ≠ natural path; PROTOCOL.md:86-87 reports 0.01–0.05 m last-digit differences. A/B baseline variability measured via natural_rep (batch composition), 5.9 % A− discordance (RESULTS.md:49).
- Metrics: a_eval / AMP_FDE imported from analyze_action_history (analyze_temporal.py:24, 31-34) — same P/R/A + FDE5 > 3 m as AutoVLA; d10 = final-waypoint distance to the δ=0 reference unit of the same row.

### 미확인 / 가설
- 미확인: per-run logged module path of navsim.evaluate.pdm_score in navhard scoring (only inferred).
- 미확인: upstream commit of the AutoVLA-vendored navsim v1.1.
- 가설: because Impromptu context waypoints are absolute positions, recent_gt/win corrections inject position (not only motion) information; reverse/motion rows partially address this (RESULTS acknowledges).
