# CLAIM_EVIDENCE_20261023_v03

This file covers the claims added or changed in v03. All claims carried over from v02 are unchanged; the v02 raw-data verifier passes 668/668 checks against the v03 TeX (`qa/V02_VERIFIER_ON_V03.txt`).

- **Levels:** (i) decoding response, (ii) executed motion, (iii) task outcome.
- **Units:** layout = a physical reset state within a task. Observations are generation calls 2/6/10 (follow-up) or 0/4/8 (discovery) within a layout.

| Claim (where) | Evidence / data | Unit and n | Estimate [95% interval] | Limits stated |
| --- | --- | --- | --- | --- |
| SpatialVLA within-call path is valid (§5.4, Supp. P) | Self-patch R raw tokens + EOS = natural; adapter = official; official = collection-time output | calls: 12 validation; 105 discovery; 4 smoke + 105 follow-up; 44 drawer | All pass (drawer: the collection comparison is NA by amendment) | Fresh cache per call is part of the evaluation path. The R continuation logits are bitwise equal in validation, discovery and follow-up; in the drawer task they differ by ≤ 0.5 with no token change |
| Library cache reuse changed outputs (Supp. P) | v01 attempt 1: record-time vs collection-time outputs; diagnostic runs | observations | 7/105 output differences; 4/105 R failures; consistent after a fresh cache | Cause of 3 residual cases not isolated; not attributed wholly to bf16; not a method claim |
| Discovery pilot: exploratory, same direction (§5.4, Table 5, Supp. P) | v01 raw records; standalone decoder | 64 obs / 32 layouts (secondary); 7 obs / 7 layouts (matched) | 0.070 [0.044, 0.100] (secondary); 0.084, interval n/a (primary) | Secondary only: L donors farther on average, so distance confounding cannot be excluded |
| **Follow-up with pre-registered distance-matched donors: latest > older (§5.4, abstract, Table 5)** | v02 raw records, standalone analyzer + independent verifier (ALL_MATCH) | **69 obs / 34 layouts / 2 tasks** (coke 55/25 new layouts; move_near 14/9 reused layouts) | **Δ = 0.048 [0.026, 0.072]**; coke 0.042 [0.015, 0.070]; move_near 0.054 [0.018, 0.093] | Command level, not executed. Intermediate action fixed. Tasks reused, so not a task-generalization test. Normalized action units. Coverage 69/105 within 64 samples |
| Donor distance balanced in follow-up (§5.4, Supp. Q) | d_L and d_O of the selected donors | 69 obs | mean 0.054 vs 0.053; ratio quartiles 0.93/1.02/1.16 | Matching is on mean absolute 6-D distance only |
| Older position not irrelevant (§5.4, Supp. Q) | Direct D(OL, L) | 69 obs | 0.021 [0.012, 0.030] | J = −0.024 [−0.040, −0.011] is descriptive only |
| Token-change rates (§5.4) | arm vs R tokens | 69 obs | L 56/69; O 27/69 | — |
| Secondary components (Supp. Q, Table S14) | a3/a4, translation/rotation, same-change-type subset | 69 (48) obs | L > O at each step and in each component; same-type Δ 0.048 [0.019, 0.079] | Not used to change the primary |
| Open-drawer supplementary task (§5.4, Supp. R) | drawer raw records (recovered observations; amendment A1) | 35 obs / 15 layouts / 1 task | Δ = 0.145 [0.091, 0.210] | Separate task; not pooled. Observations recovered after an interrupted collection |
| Pattern observed in a second backbone family, without a causal backbone effect (abstract, Limitations) | Qwen2.5-VL (AutoVLA, VLA-0) and SigLIP/Gemma 2 (SpatialVLA) | 3 policies | — | Policies differ in data, representation and domain, so the backbone is not isolated |
| AutoVLA contrast ≠ VLA-0/SpatialVLA direct L/O (Discussion, Supp. S) | design comparison table | — | — | Different estimands, metrics and units |
| Model selection (Limitations, Supp. T) | official code and model cards at recorded revisions; local checks | 4 surveyed policies | pi0-FAST / OpenVLA / OFT: no within-call time channel; WorldVLA: masked (null control) | Surveyed only; not counted as experiments |

## Independent units and counts reported separately
- **Task types:** 2 (coke, move_near), plus 1 supplementary (open drawer).
- **Physical layouts:**
  - discovery: 35 in the panel (25 coke + 10 move_near);
  - follow-up: 35 in the panel, 34 analyzed (25 new coke + 9 of the 10 reused move_near);
  - drawer: 16 observed, 15 analyzed.
- **Observations:** discovery 105 (64 analyzed); follow-up 105 (69 analyzed); drawer 44 (35 analyzed).
- **Inputs:** 0 of the 105 follow-up input images are identical to any discovery input.
