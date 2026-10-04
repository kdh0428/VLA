#!/usr/bin/env python
"""
Experiment C/D/E (PROTOCOL.md section 3): conditioning-only interventions in Impromptu VLA 3B, waypoint by waypoint.

Unit = (scene, direction) with the first waypoint forced to natural w1 + 0.5 m in direction a (a = 0..315 deg).
The executed trajectory of every row is [forced w1, the row's own generated w2..w10]; only the context used to generate
the next waypoint is changed (rows: normal, win1-4, gt_history, recent_gt, reverse, near_gt, dir_ok_mag_wrong,
dir_wrong_mag_ok, random_mag_matched). 300 scenes drawn with seed 0 from the navtest PoC scenes. RTX 5090 only.

  CUDA_VISIBLE_DEVICES=1 python exp_cde.py <out_dir> [--n-scenes 300] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time

import numpy as np

from impromptu_core import FastWaypointDecoder, Impromptu, parse_wps, prefix_text, scene_inputs

SCENES = "/root/VLA/autovla/dataset/nuplan/navtest_poc"
MAG, DIRS = 0.5, tuple(range(0, 360, 45))
ROWS = ["normal", "win1", "win2", "win3", "win4", "gt_history", "recent_gt", "reverse",
        "near_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched"]
SUB = {"near_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched"}


def rng_for(*key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(":".join(map(str, key)).encode()).digest()[:4], "little"))


def rot(v, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def context(row, k, w1, own, gt, key):
    """Context waypoints c_1..c_{k-1} (1-based k = waypoint being generated) and the substitution log."""
    c = [np.asarray(w1, float)]
    log = None
    for j in range(2, k):
        o, g = np.asarray(own[j], float), np.asarray(gt[j - 1], float)       # gt list is 0-based (g_j = gt[j-1])
        last = j == k - 1
        if row == "normal":
            v = o
        elif row.startswith("win"):
            v = g if j <= 1 + int(row[3:]) else o
        elif row == "gt_history":
            v = g
        elif row == "recent_gt":
            v = g if last else o
        elif row == "reverse":
            v = o if last else g
        elif row in SUB and last:
            prev = c[-1]; mg = g - prev; nm = np.linalg.norm(mg); e = float(np.linalg.norm(o - g))
            r = rng_for(*key, row, k)
            if row == "near_gt":
                a = r.uniform(0, 2 * math.pi); v = g + 0.05 * np.array([math.cos(a), math.sin(a)])
            elif nm < 1e-3:                                                     # standing still: direction undefined
                v = g
            elif row == "dir_ok_mag_wrong":
                u = mg / nm; sgn = 1.0 if np.dot(o - g, u) >= 0 else -1.0
                v = prev + u * max(nm + sgn * e, 0.0)
            elif row == "dir_wrong_mag_ok":
                th = 2 * math.asin(min(1.0, e / (2 * nm))); cr = mg[0] * (o - g)[1] - mg[1] * (o - g)[0]
                v = prev + rot(mg, th if cr >= 0 else -th)
            else:                                                               # random_mag_matched
                v = prev + rot(mg, r.uniform(0, 2 * math.pi)) * (1 + r.uniform(-0.05, 0.05))
            log = {"k": k, "e_own": round(e, 4), "d_sub_gt": round(float(np.linalg.norm(v - g)), 4),
                   "dmag": round(float(np.linalg.norm(v - prev) - nm), 4),
                   "dphi_deg": (round(float(abs(math.degrees(math.atan2(mg[0] * (v - prev)[1] - mg[1] * (v - prev)[0],
                                                                        float(np.dot(mg, v - prev)))))), 3) if nm >= 1e-3 else None)}
        else:
            v = o
        c.append(v)
    return c, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n-scenes", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allt = sorted(f[:-5] for f in os.listdir(SCENES))
    sel = sorted(random.Random(0).sample(allt, a.n_scenes))
    dst = os.path.join(a.out, "records.jsonl")
    done = {json.loads(l)["token"] for l in open(dst)} if os.path.exists(dst) else set()
    toks = [t for t in sel if t not in done]
    if a.limit:
        toks = toks[:a.limit]
    json.dump(sel, open(os.path.join(a.out, "selected_scenes.json"), "w"))
    print(f"[plan] {len(toks)} scenes ({len(done)} done), {len(DIRS)} units x {len(ROWS)} rows", flush=True)
    m = Impromptu(); fd = FastWaypointDecoder(m); t0 = time.time()
    with open(dst, "a") as fo, open(os.path.join(a.out, "errors.jsonl"), "a") as fe:
        for n, t in enumerate(toks, 1):
            try:
                log = json.load(open(os.path.join(SCENES, f"{t}.json")))["front_camera_paths"][0].split("/")[0]
                q, imgs, fut = scene_inputs(log, t); gt = fut.tolist()
                nat = parse_wps(m.continue_batch(q, imgs, [""])[0])[:10]
                if len(nat) < 10:
                    raise RuntimeError("natural decode incomplete")
                fd.set_scene(q, imgs)                           # prompt KV cache shared by all rows of the scene
                units = []
                for ang in (None,) + DIRS:                      # None: delta = 0 reference unit (same path, divergence baseline)
                    mg = 0.0 if ang is None else MAG; ar = 0.0 if ang is None else math.radians(ang)
                    w1 = (nat[0][0] + mg * math.cos(ar), nat[0][1] + mg * math.sin(ar))
                    units.append({"dir": ang, "w1": w1, "own": {r: {} for r in ROWS}, "subs": {r: [] for r in SUB}, "ctx": {r: [] for r in ROWS}})
                for k in range(2, 11):
                    prefixes, idx = [], []
                    for ui, u in enumerate(units):
                        for r in ROWS:
                            c, lg = context(r, k, u["w1"], u["own"][r], gt, (t, u["dir"]))
                            if lg:
                                u["subs"][r].append(lg)
                            if k == 10:
                                u["ctx"][r] = np.round(c, 3).tolist()
                            prefixes.append(prefix_text(c)); idx.append((ui, r))
                    outs = fd.next_waypoint(prefixes)
                    for (ui, r), o in zip(idx, outs):
                        w = parse_wps(o)
                        units[ui]["own"][r][k] = list(w[0]) if w else list(units[ui]["own"][r].get(k - 1, units[ui]["w1"]))
                        if not w:
                            units[ui].setdefault("parse_fail", []).append([r, k])
                rec = {"token": t, "log": log, "gt": np.round(fut, 3).tolist(), "natural": nat, "units": []}
                for u in units:
                    rec["units"].append({"dir": u["dir"], "w1": list(u["w1"]), "parse_fail": u.get("parse_fail", []),
                                         "rows": {r: [list(u["w1"])] + [u["own"][r][k] for k in range(2, 11)] for r in ROWS},
                                         "subs": u["subs"], "ctx_k10": u["ctx"]})
                fo.write(json.dumps(rec) + "\n"); fo.flush()
            except Exception as exc:
                fe.write(json.dumps({"token": t, "error": repr(exc)}) + "\n"); fe.flush()
            el = time.time() - t0
            if n % 10 == 0:
                print(f"[prog] {n}/{len(toks)} {el / n:.1f}s/scene eta {(len(toks) - n) * el / n / 60:.0f} min", flush=True)
    print(f"[done] {len(toks)} scenes in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
