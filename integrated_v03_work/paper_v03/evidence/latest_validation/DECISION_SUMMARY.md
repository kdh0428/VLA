# DECISION_SUMMARY — CVPR next validation (2026-10-19)

**New GPU use: 0.** Questions 1–2 were answered by CPU work on existing data (plan locked before the simulation:
`STATISTICAL_CHECK_PLAN.md`, sha256 d2b9c42b…; post hoc). Question 3 needs new data, which does not exist here.
Nothing committed or pushed; no existing input, output, protocol or raw was changed.

| # | question | answer | basis | next step |
|---|---|---|---|---|
| 1 | Interval for the weighted eligible frames E / E4 | **APPROXIMATE_ONLY** | finite-population (frame) design D3: cluster-t 95% coverage 0.928–1.00 for all five E/E4 contrasts; superpopulation (new drives) D2: 0.886–0.940 — **E_RV and E4 C_NAT fall below 0.90 (WITHHOLD for new drives)**, E_RN 0.908, E4 C_GT 0.927, E4 INT 0.912; the drive bootstrap is lower everywhere | report E/E4 cluster-t intervals only as approximate eligible-frame intervals (or withhold); do not present them as intervals for new drives |
| 2 | Does the INT interval reflect the joint outcome and the cluster structure? | **APPROXIMATE_ONLY** | code check: INT is formed per unit from the four cells of the same unit, then scene → drive (correct joint structure). Simulation with joint four-cell label swaps: E4 INT cluster-t 0.912–0.994 (approximate); F INT superpopulation 0.886–0.937 (**WITHHOLD** for new drives). Day–vehicle / component dependence is not reflected (drive clusters only) | keep INT as an estimate with an approximate (E4) or withheld (F, new drives) interval; never as "no interaction" / equality; no post-hoc margin |
| 3 | Day-and-vehicle-disjoint validation data | **BLOCKED** (`BLOCKED_NO_NEW_DISJOINT_DATA`; C: `BLOCKED_BY_DATA`) | delta inventory: no new data or metadata since review_followup5; known sources unchanged and ineligible | needs logs with identifiable day + vehicle outside dev ∪ P1-C, not in OpenScene trainval, with cameras + GT; then a C_NAT-first replication protocol |

## Details for question 1

| contrast | D3 finite-population min / max | D2 new drives min / max | decision (plan §8) |
|---|---|---|---|
| E RN (−4.1 pp) | 0.937 / 1.000 | 0.908 / 0.940 | APPROXIMATE_ONLY |
| E RV (+3.6 pp) | 0.953 / 0.995 | 0.886 / 0.960 | WITHHOLD for new drives; frame interval approximate |
| E4 C_NAT (+4.0 pp) | 0.940 / 0.995 | 0.890 / 0.947 | WITHHOLD for new drives; frame interval approximate |
| E4 C_GT (+3.3 pp) | 0.930 / 1.000 | 0.927 / 0.938 | APPROXIMATE_ONLY |
| E4 INT (−0.7 pp) | 0.928 / 0.994 | 0.912 / 0.931 | APPROXIMATE_ONLY |
| F RN (−34.2 pp) | – (census) | 0.915 / 0.937 | APPROXIMATE_ONLY |
| F C_NAT (+25.5 pp) | – | 0.936 / 0.943 | near nominal here; with the fixed-scene check (min 0.931) → approximate |
| F INT (−0.2 pp) | – | 0.886 / 0.937 | WITHHOLD for new drives |

R = 5,000 (cluster-t) / 1,000 (bootstrap) per scenario; 4 effect levels (zero, ± half, observed) × 2 couplings × designs;
bias ≤ 0.002, no zero-width or failed intervals (`SIM_RESULTS.csv`, `SIM_SUMMARY.csv`). The lowest coverage occurs at the
observed effect scale under D2, where drive-to-drive effect heterogeneity with 18–24 drives widens the true sampling
distribution beyond the cluster-t approximation.

## No result-changing error found

All estimates and p values are unchanged; only interval status changes (`PAPER_READY_NUMBERS.csv`, 9 rows).
