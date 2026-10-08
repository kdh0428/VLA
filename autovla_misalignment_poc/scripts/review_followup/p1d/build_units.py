#!/usr/bin/env python
"""P1-D evaluation units (initial states), fixed before any P1-D rollout.

move_near: reset(seed) draws episode_id = RandomState(seed).randint(60) and overlay = OVERLAYS[RandomState(seed).choice(4)]
from the *same* first draw (checked on 520 seeds: 520/520), so seeds reach only a coupled subset of the 60 x 4
(episode, overlay) configurations (seeds 0-119 of experiments 34/35 cover 53 distinct ones; duplicate seeds give identical
trajectories). P1-D therefore passes episode_id explicitly (obj_init_options) and picks, for each wanted overlay, an unused
seed >= 1000 whose first draw yields that overlay. Units = configurations never used in experiments 34/35 (seeds 0-119),
30 per overlay, drawn with numpy default_rng(20261008).
pick_coke_can: object xy is drawn uniformly (continuous), seeds 200-319 give 120 distinct initial states (checked) not used
in experiments 34/35.

  python build_units.py --out RUN/units.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

OVERLAYS = ["", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-task", type=int, default=120)
    ap.add_argument("--coke-seed0", type=int, default=200)
    a = ap.parse_args()
    used = {(int(np.random.RandomState(s).randint(60)), int(np.random.RandomState(s).choice(4))) for s in range(120)}
    rng = np.random.default_rng(20261008)
    per = a.n_per_task // 4
    seed_pool = {u: [] for u in range(4)}
    s = 1000
    while min(len(v) for v in seed_pool.values()) < per:
        seed_pool[int(np.random.RandomState(s).choice(4))].append(s); s += 1
    mn = []
    for u in range(4):
        free = [e for e in range(60) if (e, u) not in used]
        pick = sorted(rng.choice(free, size=per, replace=False).tolist())
        for e, sd in zip(pick, seed_pool[u]):
            mn.append({"seed": sd, "episode_id": int(e), "overlay": OVERLAYS[u]})
    out = {"google_robot_move_near": {str(r["seed"]): {"episode_id": r["episode_id"], "overlay": r["overlay"]} for r in mn},
           "google_robot_pick_coke_can": {str(s): {} for s in range(a.coke_seed0, a.coke_seed0 + a.n_per_task)},
           "excluded_move_near_configs_from_exp34_35": sorted([list(x) for x in used]),
           "note": "move_near: reset(seed, options={'obj_init_options': {'episode_id': id}, 'reconfigure': True})"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"move_near units {len(mn)} (excluded {len(used)} exp34/35 configs); seeds {mn[0]['seed']}..{max(r['seed'] for r in mn)}")


if __name__ == "__main__":
    main()
