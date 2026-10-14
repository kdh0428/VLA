# AMENDMENT P1-01: harness and judge logic fixes before the Stage-0 gate (2026-10-14)

The protocol is frozen at commit 0ef9334.

**Unchanged:** tolerances (T_exact = 0, T_equiv = 0.75), pass/fail rules, the manifest, the controls, and all model and intervention code. Only comparison bookkeeping was fixed.

**What was seen before the fixes.** The controls 1–4 code had never executed, so I ran a smoke test to catch crashes:
- scope: 1 pilot scene (c05f89f644505f4c, alt0 + G:G_faster), all 4 paths, greedy only;
- output: scratchpad only, not part of the gate;
- result: no crash;
- what the summary showed: c1, c2 and c5a were bitwise 0; c3a, c3b and c4 had a max |Δlogit| ≤ 0.375; one flip in P1A c4. The two logic errors below also showed up.

The gate is run from scratch afterwards on the full manifest.

## A1 — `p1b_harness.py` `compare()`: C4 token substitution of the H8 emb-patch rows
- **Bug:** in C4, `reverse@emb` and `patch_full@emb` are evaluated by token substitution (PROTOCOL §1, C4). Their input ids therefore differ from the cached-edit path by design, at the last span position. `compare()` compares logits only on steps with equal input hashes, so only step t*+1 was compared.
- **Fix:** a step counts as having equal input iff:
  - (a) the spans are equal except the last token,
  - (b) the cached path's last token equals the recorded substitution target, and
  - (c) the full path's last token equals the recorded source token.
  - The source is the Normal row's c_{k−1} for `reverse@emb` and gt[k−1] for `patch_full@emb`.
- **Recording:** each step's verification is saved as `substitution_steps_verified`. All other rows and controls are unaffected.

## A2 — `p1b_judge.py`: "effective override in a self-patch" applied to the wrong control
- **Bug:** the rule (PROTOCOL §6, exact class) refers to the self-patch override. The judge also applied it to `c2_hook`. In that control, `override_effective_b` reports the rows' *ordinary experiment* context overrides (e.g. GT in `gt_history`), which are present in both the untouched and the hooked run. The hook-level self-patch is checked by logits and tokens instead.
- **Fix:** the rule is applied to `c2_token` only.

## A3 — `p1b_judge.py` report formatting (made after run1 finished, before the final judge ran)
- **Why:** the judge's markdown listed only the verdict per path and the failures. It now also writes the per control × path × decoding summary table (rows, failures, max |Δlogit|, flips, suffix-equal rows).
- **Also changed:** the report-only rows (c4_free, c5c_free) now get their logit maxima and flip counts filled in, where before they showed 0.
- **Unchanged:** pass/fail logic, classes and tolerances.
- **Data seen at the time:** a preliminary judge of run1 had been run; all paths PASS with 0 failures.

## A4 — SpatialVLA gate: workers 8 → 4 (resource problem, not a result)
- **What happened:** the first gate_run1 attempt used 8 SAPIEN workers (PROTOCOL §8.4, as in P1-D). It thrashed the page cache: io.pressure ≈ 60%, ≈ 130 MB/s of re-reads by my own 9 processes, workers at ≈ 2% CPU. It completed **0 episodes in 30 min** (2 C4-at-T0 rows written).
- **Action:** aborted and kept as `svla_gate/run1_aborted_8workers_iothrash/` and `gate_run1_aborted_8workers.log`. No episode data had been produced, so nothing about the outcome was seen.
- **Rerun:** with 4 workers (the pilot ran at 1.4 episodes/min with 4 workers).
- **Why this does not change results:** `decode_fixed` pads every prompt-length group to exactly 8 rows independently of how many workers are active (`svla_protection.py:119-134`). The executed chunks therefore do not depend on the number of workers.
- **Unchanged:** arms, units, tolerances and pass rules.
- **Cost:** about 2× the wall time of §8.4.

## A5 — SpatialVLA C2 (null arm): executor fingerprint includes the shadow-cf bookkeeping (post hoc; data seen)
- **What was seen:** after 32 of 160 gate episodes, all four `nullfull` (U2) arms had executed float64 actions and every simulator fingerprint component bitwise equal to N over 80 steps. Only my executor fingerprint differed, from t = 8 (the first injection step) on.
- **Cause:** the fingerprint (`svla_noop_restore.exec_fp`) hashes `ex.cf`, the counterfactual-chunk dict that P1-D's null mode fills with the shadow (g, p) decode by design (`p1d_runner.py:310-311, 324-327`). `Executor.step` uses only `ex.chunks` and the sticky state, so `cf` never reaches the executed action. The difference is therefore expected and is not a difference in the executed computation.
- **Strict verdict:** the pre-fixed bitwise rule, which includes the executor fingerprint, is **failed** by the null arm. It is reported unchanged as `pass_strict_prefixed_rule = False`.
- **Added check, labelled post hoc:** `pass_policy_state_A5` is true iff all of the following hold:
  - the only differing component is the executor fingerprint;
  - actions are bytes-equal at every step;
  - the exec/ctx tokens are equal at every generation. The chunks (`tokens_to_actions` of the exec tokens) and the sticky state (a function of the chunks) are therefore equal;
  - the TCP difference is 0.
- **Lead's decision:** whether U2 counts as passing is for the lead. Tolerances are unchanged.
- **For future runs:** hash policy state and `cf` separately.
