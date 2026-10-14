# MATCHED_REVERSE_RESULTS — P3

**Status: NOT_RUN** (2026-10-14, CP2).

**Why.**
- P3 is Tier 3. It needs new AutoVLA inference: the frozen source bank, a one-shot 2 × 3 factorial, and a repeated-schedule block.
- Per the prompt, P3 starts only after P1 passes **and CP2 approval**. CP2 approval has not been given.
- No P3 code, bank or rollout exists. A re-analysis of the existing results cannot create the missing arms: there is no frozen matched natural older prefix, and no corrected-branch-source latest block.

**Protocol draft:** `MATCHED_REVERSE_PROTOCOL.md`.

**Cost if approved:**
- AutoVLA: ≈ 2.3–6.0 GPU h.
- Impromptu add-on: ≈ 11–22 GPU h.
- Source: `../STAGE3_COST_PLAN.csv`, options O1 and O2.

**Claim impact while NOT_RUN.** Review M2 is **unresolved — addressed by narrowing**:
- the cross-model "Reverse" is described as matched in spirit, not identical;
- effect sizes are not compared across models;
- the AutoVLA arm is called "Normal-branch token injection";
- the Impromptu Reverse stays unmatched.
