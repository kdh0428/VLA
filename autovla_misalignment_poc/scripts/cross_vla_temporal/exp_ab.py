#!/usr/bin/env python
"""
Experiment A/B (outputs/cross_vla_temporal_replication/PROTOCOL.md): small first-waypoint deviation and matched-magnitude
branches in Impromptu VLA 3B. Per navtest PoC scene:
  natural       greedy decode, batch of one
  natural_rep   the same, decoded inside the batch with the 16 perturbation rows (batch-composition variability)
  pert_m0       first waypoint forced to the natural w1 itself (delta = 0, same re-encoded prefix path): divergence reference
  pert_m{m}_d{a}  first waypoint forced to natural w1 + delta (|delta| = m in {0.2, 0.5} m, direction a in 0..315 deg),
                waypoints 2..10 generated freely
Writes <out>/records.jsonl (resumable). RTX 5090 only.

  CUDA_VISIBLE_DEVICES=1 python exp_ab.py <out_dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from impromptu_core import Impromptu, parse_wps, prefix_text, scene_inputs

SCENES = "/root/VLA/autovla/dataset/nuplan/navtest_poc"
MAGS, DIRS = (0.2, 0.5), tuple(range(0, 360, 45))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "records.jsonl")
    done = {json.loads(l)["token"] for l in open(dst)} if os.path.exists(dst) else set()
    toks = [t for t in sorted(f[:-5] for f in os.listdir(SCENES)) if t not in done]
    if a.limit:
        toks = toks[:a.limit]
    print(f"[plan] {len(toks)} scenes ({len(done)} done)", flush=True)
    m = Impromptu(); t0 = time.time()
    with open(dst, "a") as fo, open(os.path.join(a.out, "errors.jsonl"), "a") as fe:
        for n, t in enumerate(toks, 1):
            try:
                log = json.load(open(os.path.join(SCENES, f"{t}.json")))["front_camera_paths"][0].split("/")[0]
                q, imgs, fut = scene_inputs(log, t)
                nat = parse_wps(m.continue_batch(q, imgs, [""])[0])[:10]
                if len(nat) < 10:
                    raise RuntimeError(f"natural decode gave {len(nat)} waypoints")
                rows = [("natural_rep", None), ("pert_m0", tuple(nat[0]))]     # pert_m0: delta = 0 through the same prefix path
                for mg in MAGS:
                    for ang in DIRS:
                        d = (mg * math.cos(math.radians(ang)), mg * math.sin(math.radians(ang)))
                        rows.append((f"pert_m{mg}_d{ang}", (nat[0][0] + d[0], nat[0][1] + d[1])))
                prefixes = ["" if w1 is None else prefix_text([w1]) for _, w1 in rows]
                outs = m.continue_batch(q, imgs, prefixes)
                res = {}
                for (name, w1), o in zip(rows, outs):
                    w = parse_wps(o)
                    traj = (w[:10] if w1 is None else [w1] + w[:9])
                    res[name] = {"traj": np.round(traj, 3).tolist(), "ok": len(traj) == 10}
                fo.write(json.dumps({"token": t, "log": log, "gt": np.round(fut, 3).tolist(), "natural": nat, "rows": res}) + "\n")
                fo.flush()
            except Exception as exc:
                fe.write(json.dumps({"token": t, "error": repr(exc)}) + "\n"); fe.flush()
            if n % 50 == 0:
                el = time.time() - t0
                print(f"[prog] {n}/{len(toks)} {el / n:.2f}s/scene eta {(len(toks) - n) * el / n / 60:.0f} min", flush=True)
    print(f"[done] {len(toks)} scenes in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
