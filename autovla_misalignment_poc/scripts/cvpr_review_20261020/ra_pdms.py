#!/usr/bin/env python
"""Addendum 4 (post_hoc): official NAVSIM v1 PDM Score (non-reactive) of STORED AutoVLA plans, CPU only.
Reuses scripts/pdm_score_candidates.py (_init/_score: in-memory metric cache identical to MetricCacheProcessor,
default scoring parameters). No policy inference. Plans = stored emitted action indices decoded with the codebook.
Scored: B2 cells GT_REF / GT_NSRC / NAT_REF / NAT_NSRC (n_old = 1) and P1-C H7 normal / recent_gt.
"""
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/root/VLA/autovla/dataset/nuplan/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
import pdm_score_candidates as PS  # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/cvpr_review_and_contribution_20261020/data"
B2 = f"{POC}/outputs/review_followup4/B/B2/p1c_main/cells.jsonl"
P1C = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008/mech/action_history_causal/records.jsonl"


def jobs():
    picks, logs = {}, {}
    for line in open(P1C):
        r = json.loads(line)
        logs[r["token"]] = r["log"]
        for pn, pv in r["perturbations"].items():
            for c in ("normal", "recent_gt"):
                picks.setdefault(r["token"], {})[f"P1C|{r['token']}|{pn}|{c}"] = (0, pv["conditions"][c]["action_idx"])
    for line in open(B2):
        x = json.loads(line)
        if x["n_old"] != 1 or x["cell"] not in ("GT_REF", "GT_NSRC", "NAT_REF", "NAT_NSRC"):
            continue
        picks.setdefault(x["scene"], {})[f"B2|{x['unit_id']}|{x['cell']}"] = (0, x["acts"])
    return [(t, logs[t], p) for t, p in picks.items()]


def main():
    J = jobs()
    dst = f"{OUT}/pdms_raw.jsonl"
    done = {json.loads(l)["token"] for l in open(dst)} if os.path.exists(dst) else set()
    J = [j for j in J if j[0] not in done]
    logs = sorted({j[1] for j in J})
    print(f"[plan] {len(J)} scenes, {sum(len(j[2]) for j in J)} plans, {len(logs)} logs", flush=True)
    t0 = time.time(); n = 0
    with mp.get_context("spawn").Pool(int(os.environ.get("WORKERS", "6")), initializer=PS._init, initargs=(logs,)) as pool, open(dst, "a") as f:
        for res in pool.imap_unordered(PS._score, J, chunksize=2):
            f.write(json.dumps(res) + "\n"); f.flush(); n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"[prog] {n}/{len(J)} {el / n:.1f}s/scene eta {(len(J) - n) * el / n / 60:.1f} min", flush=True)
    print(f"[done] {n} scenes {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
