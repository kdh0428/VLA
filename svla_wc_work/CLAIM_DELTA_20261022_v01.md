# CLAIM_DELTA_20261022_v01 — proposed changes to manuscript v02 (`paper/paper_body.tex`); v02 itself is not edited

## What the new evidence supports
- **Path verification.** The within-call intervention path is verified for a policy with a different backbone family (SpatialVLA-4B: SigLIP + Gemma 2 language model). Self-patch was token-identical in 117/117 observations, with bitwise-equal continuation logits.
- **Direction.** Under that path, editing the latest preceding action changes the continuation more often and by more than editing the older action. This holds in both tasks, at both continuation steps, and in both translation and rotation:
  - all joint-valid observations: Δ = 0.070 [0.044, 0.100] (64 observations, 32 layouts, 2 tasks);
  - the latest edit changes a3–a4 tokens in 89% of observations, the older edit in 27%.
- **Weak primary analysis.** The pre-specified distance-matched estimate (0.084) has 7 observations, and its two-task interval is NA. This must be stated wherever the result is used.

## Proposed edits (location → text)

1. **Abstract, last two sentences.** Current text:

   > "AutoVLA and VLA-0 share a Qwen2.5-VL-3B backbone; SpatialVLA (Gemma 2 language model) is tested with a different, cross-step intervention. Position-specific within-call responses are therefore shown for two policies, action representations, and domains within one backbone family."

   Proposed replacement:

   > "AutoVLA and VLA-0 share a Qwen2.5-VL-3B backbone. In SpatialVLA (Gemma 2 language model), a token-identical self-patch verifies the same within-call intervention, and editing the latest preceding action again changes the continuation more than editing the older one in two SimplerEnv tasks (normalized difference 0.070, exploratory layout-bootstrap interval [0.044, 0.100]; the pre-specified distance-matched subset has only 7 observations)."

2. **§3 (Interventions).** Add a paragraph after §3.3, "SpatialVLA: within-call action edits". SpatialVLA emits 4 actions of 3 tokens (translation, rotation, gripper):
   - the first two actions are kept as the emitted prefix;
   - the continuation (a3–a4) is decoded after forcing a copy of a1–a2 in which a1 (older, O) or a2 (latest, L) is replaced by a whole action from one of four T = 0.7 samples;
   - the replacement keeps the natural gripper token;
   - the response is the mean absolute difference of the 6 normalized motion components of a3–a4 from the self-patch R.
3. **Table 1.** Add a row: "SpatialVLA within-call pilot | i | 105 observations (64 analyzable) | 2 tasks; 35 panel layouts (25 new coke, 10 reused move_near), 32 analyzable (23 coke, 9 move_near) | Early natural-rollout observations, open loop".
4. **§5 (Level i).** Add a subsection "SpatialVLA: the latest action dominates in a second backbone family", with the denominators table and the all-joint-valid Δ. Include:
   - the matched result and its NA interval;
   - the 89% vs 27% change rates;
   - D(OL, L) = 0.043, with "the older edit is not irrelevant";
   - J reported only as descriptive.
5. **Discussion / Limitations.** Replace "the position-specific within-call result is untested across backbone families; SpatialVLA adds a SigLIP/Gemma 2 policy, but with a cross-step rather than within-call intervention" with:

   > "the position-specific within-call result now spans two backbone families (Qwen2.5-VL in AutoVLA and VLA-0; Gemma 2 in SpatialVLA), but the SpatialVLA estimate comes from two tasks, a primary distance-matched subset of only 7 observations, and donor distances that are larger for the latest edit on average; it does not cover motion or success for SpatialVLA within-call edits."
6. **Supplement.** Add a section with:
   - the path table (`FEASIBILITY_20261022_v01.md`);
   - the A1/A3 amendments, including the `generate` cache-reuse finding;
   - the source rejection counts (297/420 zero distance);
   - the figures `analysis/fig_L_vs_O_by_task`, `fig_a3_a4` and `fig_donor_distance`.

## Unchanged
- **AutoVLA higher-level clusters.** The fixed splits have 6 day–vehicle components; the union and the 147 logs have 5.
- **SpatialVLA cross-step success.** The ±10 pp bound applies only to the two-task means.
- **VLA-0 numbers and closed-loop results.**
- **No claim of** physical motion, success, task-population or backbone-population generality for the new result.

## Remaining limits specific to this result
- Two tasks, and move_near has only 10 layouts (reused). Observations are within-layout repeats (calls 0, 4, 8), so the effective unit is the layout.
- Matching was rarely satisfied because the sampled donors seldom changed both positions by similar amounts; 297/420 samples reproduced a natural action. A larger or differently designed donor pool would be needed for a well-powered matched estimate.
- Normalized Euler differences are not geodesic rotation distances. The responses are command-level, not motion.
