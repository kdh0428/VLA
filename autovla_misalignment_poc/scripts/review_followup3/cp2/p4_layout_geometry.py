#!/usr/bin/env python
"""P4 move_near variant-layout geometry pre-check (CP2, 2026-10-14). PLANNING ONLY: pure numpy, no simulator, no policy.

Question: how many geometrically valid, distinct 3-object move_near layouts exist under the candidate sampling rules of
SPATIAL_STATE_EXPANSION_PROTOCOL.md, before any physics settle check? Everything here is read from SimplerEnv code/data:
  benchmark layouts   MoveNearGoogleBakedTexInSceneEnvV1._xy_config_per_triplet (2 xy configs x 5 triplets)
                      ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/move_near_in_scene.py
  object quats/bboxes obj_init_quat_dict (same file) and data/custom/info_pick_custom_baked_tex_v1.json
  'near target' rule  evaluate(): dist < half_xy_diag(src) + half_xy_diag(tgt) + 0.10 (bbox rotated by init quat)
  region R_bench      axis-aligned box of the 6 benchmark move_near positions: x [-0.33, -0.13], y [0.04, 0.34]
  region R_simpler    SimplerEnv's own object sampling rectangle on the same scene (GraspSingleInSceneEnv
                      _initialize_actors uniform([-0.35, -0.02], [-0.12, 0.42]); = coke-can visual-matching grid range)

Writes outputs/review_followup3/P4/planning/layout_geometry.csv (+ layout_geometry_meta.json).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python p4_layout_geometry.py
"""
import ast
import itertools
import json
import os
import re

import numpy as np
import pandas as pd

SIMPLER = "/root/VLA/simpler/SimplerEnv/ManiSkill2_real2sim"
SRC = f"{SIMPLER}/mani_skill2_real2sim/envs/custom_scenes/move_near_in_scene.py"
INFO = f"{SIMPLER}/data/custom/info_pick_custom_baked_tex_v1.json"
OUT = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P4/planning"
N_CAND = 200_000
SEED = 20261014
DEDUP_M = 0.03          # a layout is new only if some object moves >= 3 cm vs every benchmark layout (>> 0.26 mm drift)
# initial source-target distance must exceed the near-target threshold by at least the smallest margin found in the
# benchmark layouts themselves (computed below, not chosen)


def euler_sxyz(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def parse_v1():
    """read triplets, xy configs and euler quats of the V1 (google_robot_move_near) env from the source file."""
    txt = open(SRC).read()
    cls = txt[txt.index("class MoveNearGoogleBakedTexInSceneEnvV1"):]
    cls = cls[:cls.index("def _load_model")]
    trip = ast.literal_eval(re.search(r"self\.triplets = (\[.*?\n        \])", cls, re.S).group(1))
    xy = ast.literal_eval(re.search(r"self\._xy_config_per_triplet = (\[.*?\n        \])", cls, re.S).group(1))
    quat = {}
    for name, body in re.findall(r'"(\w+)": (euler2quat\([^)]*\)|\[[^\]]*\])', cls):
        if body.startswith("euler2quat"):
            args = [eval(x.strip(), {"np": np}) for x in body[len("euler2quat("):-1].split(",")]
            quat[name] = euler_sxyz(*args)
        else:
            quat[name] = np.eye(3)
    return trip, xy, quat


def half_diag(name, info, rot):
    bb = info[name]["bbox"]
    ext = np.array(bb["max"]) - np.array(bb["min"])
    wext = np.abs(rot) @ ext                       # extents of the rotated box (as in evaluate(): quat2mat @ bbox)
    return float(np.linalg.norm(wext[:2]) / 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    trip, xy_cfgs, quat = parse_v1()
    info = json.load(open(INFO))
    hd = {n: half_diag(n, info, quat[n]) for t in trip for n in t}
    bench = np.array(xy_cfgs)                      # 2 x 3 x 2
    pts = bench.reshape(-1, 2)
    bench_min_pair = min(np.linalg.norm(a - b) for c in bench for a, b in itertools.combinations(c, 2))
    bench_margin = min(np.linalg.norm(np.array(c[si]) - np.array(c[gi])) - (hd[t[si]] + hd[t[gi]] + 0.10)
                       for c in xy_cfgs for t in trip for si, gi in ((0, 1), (0, 2), (1, 2)))
    regions = {"R_bench": ([pts[:, 0].min(), pts[:, 1].min()], [pts[:, 0].max(), pts[:, 1].max()]),
               "R_simpler": ([-0.35, -0.02], [-0.12, 0.42])}
    rng = np.random.default_rng(SEED)
    rows = []
    for rname, (lo, hi) in regions.items():
        cand = rng.uniform(lo, hi, size=(N_CAND, 3, 2))
        pair = np.stack([np.linalg.norm(cand[:, i] - cand[:, j], axis=1) for i, j in ((0, 1), (0, 2), (1, 2))], 1)
        # distance to every benchmark layout (same slot order; slot = object index in the triplet)
        dev = np.stack([np.abs(cand - b[None]).max(axis=(1, 2)) for b in bench], 1).min(1)
        for dmin in (bench_min_pair, 0.20, 0.15):
            ok_pair = pair.min(1) >= dmin - 1e-9
            for ti, t in enumerate(trip):
                # all 6 (source, target) instructions of the layout must need movement: dist > near thr + margin
                need = np.ones(N_CAND, bool)
                for (si, gi), k in zip(((0, 1), (0, 2), (1, 2)), range(3)):
                    thr = hd[t[si]] + hd[t[gi]] + 0.10 + bench_margin
                    need &= pair[:, k] > thr
                ok = ok_pair & need
                new = ok & (dev >= DEDUP_M)
                sel = cand[new]
                # greedy count of mutually distinct layouts (every pair differs by >= DEDUP_M in some object position)
                kept = []
                for c in sel[:20000]:
                    if all(np.abs(c - k).max() >= DEDUP_M for k in kept[-3000:]):
                        kept.append(c)
                    if len(kept) >= 3000:
                        break
                spread = float(np.sqrt(sel.reshape(len(sel), -1).var(0).sum())) if len(sel) > 1 else 0.0
                rows.append({"region": rname, "x_range": f"[{lo[0]}, {hi[0]}]", "y_range": f"[{lo[1]}, {hi[1]}]",
                             "pair_min_m": round(dmin, 4), "triplet": ti, "objects": "|".join(t),
                             "n_candidates": N_CAND, "accept_pair": float(ok_pair.mean()),
                             "accept_pair_and_need_move": float(ok.mean()),
                             "accept_new_vs_benchmark": float(new.mean()),
                             "median_dev_from_benchmark_m": float(np.median(dev[ok])) if ok.any() else np.nan,
                             "n_accepted_of_candidates": int(new.sum()),
                             "n_mutually_distinct_greedy_cap3000": len(kept), "layout_spread_m": spread})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "layout_geometry.csv"), index=False, float_format="%.5f")
    json.dump({"source": SRC, "info": INFO, "triplets": trip, "xy_configs": xy_cfgs,
               "half_xy_diag_m": hd, "benchmark_min_pairwise_m": bench_min_pair, "regions": regions,
               "dedup_m": DEDUP_M, "benchmark_min_near_margin_m": bench_margin, "n_candidates": N_CAND, "seed": SEED},
              open(os.path.join(OUT, "layout_geometry_meta.json"), "w"), indent=1, default=float)
    print(json.dumps({"half_xy_diag_m": hd, "benchmark_min_pairwise_m": bench_min_pair,
                      "benchmark_min_near_margin_m": bench_margin}, indent=1))
    print(df.groupby(["region", "pair_min_m"])[["accept_pair", "accept_pair_and_need_move", "accept_new_vs_benchmark",
                                                 "n_mutually_distinct_greedy_cap3000"]].agg(["min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
