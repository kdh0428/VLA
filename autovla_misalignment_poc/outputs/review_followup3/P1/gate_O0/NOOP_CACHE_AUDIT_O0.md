# NOOP_CACHE_AUDIT (Stage-0 gate)

Tolerances: T_exact = 0.0, T_equiv = 0.75 (from /root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/pilot_c5_20261014/tolerances.json).

| path | verdict |
|---|---|
| H32 | PASS |
| P1B | PASS |

| control | path | decoding | class | rows | fail | max abs dlogit | flips (all explained unless failed) | suffix-equal rows |
|---|---|---|---|---|---|---|---|---|
| c2_token | H32 | greedy | equiv | 54 | 0 | 0 | 0 | 54 |
| c2_token | H32 | t001 | equiv | 54 | 0 | 0 | 0 | 54 |
| c2_token | P1B | greedy | equiv | 477 | 0 | 0 | 0 | 477 |
| c2_token | P1B | t001 | equiv | 477 | 0 | 0 | 0 | 477 |
| c3a_incremental | H32 | greedy | equiv | 9 | 0 | 0.4062 | 0 | 9 |
| c3a_incremental | H32 | t001 | equiv | 9 | 0 | 0.4062 | 0 | 9 |
| c3a_incremental | P1B | greedy | equiv | 54 | 0 | 0.4062 | 4 | 54 |
| c3a_incremental | P1B | t001 | equiv | 54 | 0 | 0.4062 | 4 | 54 |
| c3b_deep_invalidate | H32 | greedy | equiv | 54 | 0 | 0.625 | 5 | 54 |
| c3b_deep_invalidate | H32 | t001 | equiv | 54 | 0 | 0.625 | 5 | 54 |
| c3b_deep_invalidate | P1B | greedy | equiv | 90 | 0 | 0.375 | 11 | 90 |
| c3b_deep_invalidate | P1B | t001 | equiv | 90 | 0 | 0.375 | 8 | 90 |
| c4_free | H32 | greedy | report | 54 | 0 | 0.4375 | 8 | 46 |
| c4_free | H32 | t001 | report | 54 | 0 | 0.4375 | 8 | 49 |
| c4_free | P1B | greedy | report | 90 | 0 | 0.5312 | 16 | 74 |
| c4_free | P1B | t001 | report | 90 | 0 | 0.5312 | 13 | 86 |
| c4_tf | H32 | greedy | equiv | 54 | 0 | 0.4375 | 9 | 54 |
| c4_tf | H32 | t001 | equiv | 54 | 0 | 0.4375 | 9 | 54 |
| c4_tf | P1B | greedy | equiv | 90 | 0 | 0.5312 | 16 | 90 |
| c4_tf | P1B | t001 | equiv | 90 | 0 | 0.5312 | 13 | 90 |
| c5a | H32 | greedy | exact | 54 | 0 | 0 | 0 | 54 |
| c5a | H32 | t001 | exact | 54 | 0 | 0 | 0 | 54 |
| c5a | P1B | greedy | exact | 477 | 0 | 0 | 0 | 477 |
| c5a | P1B | t001 | exact | 477 | 0 | 0 | 0 | 477 |
| c5b | H32 | greedy | exact | 108 | 0 | 0 | 0 | 0 |
| c5b | H32 | t001 | exact | 108 | 0 | 0 | 0 | 0 |
| c5b | P1B | greedy | exact | 567 | 0 | 0 | 0 | 0 |
| c5b | P1B | t001 | exact | 567 | 0 | 0 | 0 | 0 |
| c5c_free | H32 | greedy | report | 54 | 0 | 0.5 | 3 | 51 |
| c5c_free | H32 | t001 | report | 54 | 0 | 0.5 | 4 | 48 |
| c5c_free | P1B | greedy | report | 90 | 0 | 0.5312 | 9 | 82 |
| c5c_free | P1B | t001 | report | 90 | 0 | 0.5312 | 9 | 78 |
| c5c_tf | H32 | greedy | equiv | 54 | 0 | 0.5 | 4 | 54 |
| c5c_tf | H32 | t001 | equiv | 54 | 0 | 0.5 | 6 | 54 |
| c5c_tf | P1B | greedy | equiv | 90 | 0 | 0.5312 | 10 | 90 |
| c5c_tf | P1B | t001 | equiv | 90 | 0 | 0.5312 | 10 | 90 |
| c5d | H32 | t001 | report | 54 | 0 | 0 | 0 | 0 |
| c5d | P1B | t001 | report | 477 | 0 | 0 | 0 | 0 |

Failures (0):

