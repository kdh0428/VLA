# OPENVLA_GATE — B3, OpenVLA minimal path verification (review_followup4)

**Status: NOT_RUN** (written 2026-10-18). No OpenVLA model was loaded and nothing ran on a GPU for B3.

## Why it was not run

1. **User decision, 2026-10-17.** The lead decided not to run B3 in this round. OpenVLA is therefore **excluded from the core evidence** and stays **descriptive / unverified**, as allowed by AUTOVLA_NEXT_STEPS_20261018 §4 B3 ("핵심 증거에서 제외하기로 명시하면 대규모 추가 실행 없이 unverified 상태를 유지할 수 있다").
2. **Checkpoint not on disk.** The LIBERO-Spatial checkpoint used by exp 28 (`openvla/openvla-7b-finetuned-libero-spatial`, revision `962318cec55ac10993ff0f5f43eda9a270b4c873`, pinned by the HF cache `refs/main` and the HF API on 2026-10-18) was deleted during the exp 34 disk cleanup (`outputs/cross_domain_temporal_replication/DISK_CLEANUP.md`). Only `refs/main` is left in `~/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-spatial/` (1.5 kB).
3. **Disk.** About 2.3–3 GB were free on `/root` during this round. The checkpoint needs about 15 GB.

## What B3 would have checked (not done; listed so the gap is explicit)

| check (AUTOVLA_NEXT_STEPS §4 B3) | status |
|---|---|
| exact LIBERO-Spatial checkpoint revision and action normaliser restored | revision pinned (above); normaliser config copy exists at `outputs/review_followup3/P0/_external/openvla-7b-finetuned-libero-spatial_config.json`; **not executed** |
| untouched / hook-no-op / same-token self-patch / cached vs full recompute, same observation / prompt / RNG / batch | NOT_RUN |
| frequency and effect of silent clipping of non-action argmax to bin 254 | NOT_RUN |
| action-token mask (would change the policy; old and new paths kept apart) | NOT_RUN |
| diagnosis of the 20 % outcome disagreement between unperturbed reruns (state restore scope, controller state, RNG, observation fingerprint) | NOT_RUN |

## Consequences for the claims

- Every OpenVLA number stays **descriptive** (`numbers.json` status NOT_VERIFIED; `RF4.B3.OPENVLA_GATE` = NOT_RUN).
- OpenVLA is **not counted as an independent replication** and is not part of the core evidence (CLAIM_CHANGES.md, section 5 item "lower").
- The A1 corrected state-level intervals (Newcombe 10) are reported as descriptive statistics of the stored 100-state × 8-condition records only. Per-task intervals (10 states) are uninformative and must not support any per-task conclusion.
- The 20 % rerun disagreement is a property of the unverified closed-loop path. It is **not** used as an "effect detection threshold of 20 pp".
- The within-step 92.6 % figure keeps its exact definition (at least one downstream component reaches the effective injected bin shift; commanded bins, not realised motion, not task failure).

## If B3 is run later

Needs ≈ 15 GB of free disk for the checkpoint plus the LIBERO environment (`/root/VLA/openvla/env.sh`). Reuse `scripts/cross_vla/openvla_core.py` unchanged for the "old path"; any action-token mask goes into a separate "new path" and must not overwrite exp 28 results.
