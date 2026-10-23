# DISJOINT_FEASIBILITY — independent day-and-vehicle validation data (CVPR next validation, 2026-10-19)

**Status: `BLOCKED_NO_NEW_DISJOINT_DATA`** (C: `BLOCKED_BY_DATA`; no replication protocol or runner was prepared, and no
existing cohort was re-run as a substitute). New GPU use: 0.

## Delta since review_followup5 C_DISJOINT (`DISJOINT_DELTA_INVENTORY.csv`)

Every file on this machine newer than `review_followup5/C_DISJOINT/overlap_audit.json` (2026-10-18 15:09 UTC) was listed
(`find -newer`, outputs and scripts of this project excluded): two prompt files, the CVPR source ZIP (LaTeX, figures,
evidence tables; no scene data), the updated EXPERIMENT_SUMMARY and two git FETCH_HEAD timestamps. **No new data or
metadata file and no explicitly provided other collection exists.** The previously inspected sources are unchanged and
remain ineligible (navtest / navhard share all days and vehicles; 13 day-and-vehicle-disjoint navmini logs are training
data; private_test_hard is anonymised and GT is withheld).

## Independence rule applied

A new cohort is independent only if its whole day set and its whole vehicle set are disjoint from those of dev ∪ P1-C
(days 2021.05.25 … 2021.10.06 and 13 vehicles in 5 connected components; `review_followup5/C_DISJOINT/overlap_audit.json`).
Unidentifiable day or vehicle counts as not disjoint. New logs or new day–vehicle combinations alone are not enough.

## What would be needed

1. Logs with identifiable recording day and vehicle (log name or metadata), both outside the dev ∪ P1-C sets, and not in
   OpenScene trainval (training data of the checkpoint); checkpoint-selection overlap would still be UNKNOWN.
2. The 8 front/side camera streams, ego status and history in the NAVSIM/OpenScene 2 Hz format the AutoVLA checkpoint
   uses, plus GT future trajectories (A− and severe need GT) and maps for preprocessing.
3. Enough independent components: under the planning model of the author-query Q2, half of the failure-stratum effect at
   α/4 needs roughly 10 components of 3 drives for 50–95% power depending on the within-component correlation; the exact
   p floor (2/2^G) is a resolution limit, not power.
4. A data licence allowing this use and local download; none was attempted (no accounts, no large downloads).
