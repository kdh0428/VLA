# DISJOINT_REPLICATION_RESULTS — P5

**Status: NOT_RUN** (2026-10-14, CP2).

**Reason.** No day- and vehicle-disjoint logs exist in the available nuPlan/OpenScene data relative to development, P1-C and the AutoVLA training split:
- 0 logs (P0-D inventory, `../P0/DISJOINT_LOG_INVENTORY_summary.csv`);
- `numbers.json` rows `P0D.DISJ.04`, `.05` and `.07`–`.09` all show 0 logs;
- `P0D.DISJ.06` = 25 logs that are disjoint from training only, and all of them share days and vehicles with development or P1-C.

The other two run conditions:
- AutoVLA parity is satisfied: Stage-0 PASS.
- CP2 approval: pending.

**Claim impact.**
- The external-validation part of M5 is **unresolved — addressed by narrowing**.
- The population part of M1 beyond the evaluation set is **unresolved — addressed by narrowing**.
- The manuscript keeps "no external (day/vehicle-disjoint) validation".
