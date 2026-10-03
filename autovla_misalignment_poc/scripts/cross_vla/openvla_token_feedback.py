#!/usr/bin/env python
"""
Phase C of the cross-VLA replication (PROTOCOL.md): within-step token feedback in OpenVLA, no simulation.

For every frame saved in phase A (every 5th control step of the 100 natural episodes): interventions at dimension
d in {0, 1} with delta in {-24, -8, +8, +24} bins and modes natural / feedback / corrected / reverse / window_1..4,
all decoded from one shared prefix. Writes one JSON line per frame with the executed bins of every variant and the
entropy / greedy probability at d+1.

  source /root/VLA/openvla/env.sh
  python openvla_token_feedback.py <phaseA_dir> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from PIL import Image

from openvla_core import OpenVLA

DIMS, DELTAS = (0, 1), (-24, -8, 8, 24)
MODES = ["feedback", "corrected", "reverse", "window_1", "window_2", "window_3", "window_4"]

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, "token_feedback.jsonl")
    done = {json.loads(l)["frame"] for l in open(dst)} if os.path.exists(dst) else set()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    frames = sorted(os.path.join(d, f) for d in os.listdir(os.path.join(src, "frames"))
                    for f in os.listdir(os.path.join(src, "frames", d)) if f.endswith(".png"))
    frames = [f for f in frames if f not in done]
    variants = [None] + [{"dim": d, "delta": dl, "mode": m} for d in DIMS for dl in DELTAS for m in MODES]
    print(f"[plan] {len(frames)} frames, {len(variants)} variants each", flush=True)
    vla = OpenVLA(); t0 = time.time()
    with open(dst, "a") as fo:
        for n, rel in enumerate(frames, 1):
            task = int(rel.split("/")[0].split("_")[0][1:])
            img = Image.open(os.path.join(src, "frames", rel)).convert("RGB")
            r = vla.decode_batch([img], suite.get_task(task).language, variants)
            fo.write(json.dumps({"frame": rel, "task": task, "variants": [v and f"{v['dim']}:{v['delta']}:{v['mode']}" or "natural" for v in variants],
                                 "exec": r["exec_bins"].tolist(), "ctx": r["ctx_bins"].tolist(),
                                 "ent": np.round(r["entropy"], 4).tolist(), "pmax": np.round(r["pmax"], 4).tolist()}) + "\n")
            if n % 200 == 0:
                fo.flush(); el = time.time() - t0
                print(f"[prog] {n}/{len(frames)} {el / n:.2f}s/frame", flush=True)
    print(f"[done] {len(frames)} frames in {(time.time() - t0) / 60:.1f} min", flush=True)
