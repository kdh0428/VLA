# Visual-evidence revision — 2026-10-19

The manuscript now uses 8 main-content pages, followed by 1 references-only page. The supplementary material is 8 pages. The main paper has 8 figures and 2 tables. All additions derive from existing outputs; no new GPU experiment was performed.

## Incorporated evidence

1. Three original front-camera frames with paired matched-prefix plans: an improved scene, an unchanged-severe-rate scene, and the sole worsened scene. All five supplied cases and their actual metadata/arrays are retained in the source. Cases are selected illustrations, not population representatives.
2. Repeated and matched-prefix effect comparison: large failure-selected effects versus smaller weighted E/E4 effects, with the latest permitted approximate intervals and explicit point-only rows.
3. All matched-prefix C_NAT scene effects, weighted accepted-stratum effects, and within-drive summaries. Shared days/vehicles prevent reading the ticks as independent replication.
4. Fixed-support time-resolved motion audit: common F sample 342 units/49 scenes; accepted sample 1,032 units/147 scenes; union 1,374 units/196 scenes. The displayed 0–2.5 s curves do not shrink the sample over time. At 2.5 s, F's CM excess is 0.87 m, while mean GT distance is lower for the perturbed plan (3.04 vs 3.76 m). Response size is not an accuracy certificate.
5. R9 retained-window widths, with clear distinction from repeated correction, release, and task success. Width-8 saturation partly reflects the finite plan horizon.
6. Actual SpatialVLA TCP differences, three measured C–F TCP examples spanning success outcomes, and corrected pooled success intervals. Task-specific cancellation and ten reused move_near layouts remain explicit. No robot RGB is invented or substituted.

## Updated statistical interpretation

The new joint-cell check confirms that interaction differences are formed within unit before aggregation. Its new-drive F coverage falls to 88.6%, so the displayed F interaction interval is withdrawn. The point estimate and p value remain unchanged and are not evidence of equality.

Finite-frame E/E4 simulations yield 92.8–100% coverage for a nominal 95% procedure; some new-drive checks fall below 90%. The paper labels retained intervals approximate, shows E RN and E4 C_GT/interaction only under the stated frame interpretation, and conservatively keeps E RV/E4 C_NAT point-only. Full inputs and policy are supplied; no nominal-coverage validation or independent day/vehicle validation is claimed.

## Verification and remaining limits

- Compact inputs reproduce all matched-prefix aggregate contrasts and case FDE/prefix checks.
- Common time support and the response/excess definitions are checked directly from supplied rows.
- Pooled success contrasts reproduce the supplied paired outcomes; plots use measured TCP.
- Existing headline effects and zero-null p values are unchanged.
- The original JPEG bytes have not been compared with model-preprocessed tensors; there is no inferred camera projection.
- No new day-and-vehicle-disjoint data, matched learned non-AR VLA, real-robot evaluation, or longer driving horizon is added.
- The target-year template and actual Paper ID still require submission-time checks.

`BUILD_QA.json` records page counts, compilation checks, visual review, and clean-source reproduction. `evidence/VISUAL_NUMERICAL_CHECKS.json` records the data consistency checks.
