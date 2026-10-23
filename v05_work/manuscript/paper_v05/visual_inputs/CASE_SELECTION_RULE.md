# CASE_SELECTION_RULE (post-hoc deterministic rule; NOT a preregistration)

Written before any case image or per-scene ranking was produced. Cases are illustrations, not a representative
sample of any population and not independent test data.

## A. AutoVLA driving cases (V1)

1. **Candidates.** B2 primary cohort, stratum F, n_old = 1 (anchor a = t*+3). Every unit whose NAT_REF and NAT_NSRC
   cells both exist and whose C_NAT row in `unit_contrasts.parquet` is `valid`. Units excluded for visual availability
   (no CAM_F0 file) are counted separately; they stay in the group counts but cannot be selected.
2. **Unit metric.** d_unit = FDE5(NAT_REF) − FDE5(NAT_NSRC) (m; treatment = NAT_REF, the arm with the GT value in the
   latest context position). s_unit = severe(NAT_REF) − severe(NAT_NSRC) (severe = A− AND FDE5 > 3 m at 5 s, the
   existing definition). Scene metric = unweighted mean over the scene's valid units (d_scene, s_scene).
3. **Groups.** improved: s_scene < 0; unchanged-severe: s_scene = 0 (the severe judgment did not change on average;
   continuous FDE and trajectories may differ); worsened: s_scene > 0.
4. **Scene choice per group.** Sort scenes by |d_scene − median(d_scene in group)|, tie-break by scene id (ascending).
   Take the first. The second is the next scene from a **different drive**; if none, the next scene. At most 2.
5. **Unit within a scene.** Among the scene's valid units whose s_unit has the group's sign (any unit when no unit
   has it), take the unit with |d_unit − median(d_unit of those units)| smallest; tie-break by unit id.
6. **Main-text suggestion.** The first-ranked case of each group. Empty groups are reported as empty; no other cohort
   or anchor is used to fill them.
7. **Reported.** Group counts and shares, percentile of each selected d_scene among all candidate scenes, and the
   full candidate manifest (CASE_CANDIDATES.csv) with selection flags.

## B. SpatialVLA cases (V4)

1. **Candidates.** P1-D, config r4e1 (largest structural context exposure, H1 focal config), both tasks, paired
   units (task, seed) where N, R, F and C episodes all exist with restore_ok.
2. **Contrasts.** R−N and C−F. Unit metric = TCP translation deviation area, t = 8..79 (cm·step); success change
   = success(R) − success(N) (resp. C − F) ∈ {−1, 0, +1}.
3. **Groups** by success change (−1 worse, 0 same, +1 better). Per group: unit with area closest to the group median,
   tie-break (task, seed); the second must have a different physical state (`init.state_hash`) and task if possible.
   At most 2 per group per contrast.
4. Success rates and TCP summaries are always computed from all eligible episodes, not from the selected cases.

## Changes
None at writing time. Any later change is appended here with the reason.
