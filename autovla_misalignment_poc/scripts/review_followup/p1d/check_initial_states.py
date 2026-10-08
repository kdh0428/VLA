#!/usr/bin/env python
"""P1-D: initial-state identity across seeds (CPU only, no policy). For each task and seed, reset the SimplerEnv env and
record the episode configuration (move_near episode_id, urdf/overlay version, object poses, robot qpos) and a hash of the
full simulator state (env.get_state()). Answers: (1) is reset(seed) reproducible (same hash on a second reset, also after
other seeds were run in between)? (2) which seeds share an identical initial state (finite episode sets)?

  source /root/VLA/simpler/env.sh
  nice -n 19 python check_initial_states.py --out DIR --seeds 0-119,200-439
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

TASKS = ("google_robot_pick_coke_can", "google_robot_move_near")


def parse(spec):
    out = []
    for part in spec.split(","):
        a, b = map(int, part.split("-")); out += list(range(a, b + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-119,200-439")
    ap.add_argument("--recheck", type=int, default=12, help="re-reset this many seeds at the end and compare hashes")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import simpler_env
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for task in TASKS:
        env = simpler_env.make(task)
        u = env.unwrapped

        def sig(seed):
            obs, info = env.reset(seed=seed)
            st = np.asarray(u.get_state(), dtype=np.float64)
            h = hashlib.sha1(np.round(st, 6).tobytes()).hexdigest()[:16]
            hx = hashlib.sha1(st.tobytes()).hexdigest()[:16]
            objs = []
            for act in u._scene.get_all_actors():
                if act.name not in ("arena", "ground", ""):
                    objs.append([act.name] + [round(float(x), 4) for x in act.pose.p])
            return {"task": task, "seed": seed, "state_hash6": h, "state_hash_exact": hx,
                    "episode_id": int(info["episode_id"]) if "episode_id" in info else None,
                    "urdf_version": getattr(u, "urdf_version", None),
                    "instr": env.get_language_instruction(), "objs": objs,
                    "robot_qpos": [round(float(x), 4) for x in u.agent.robot.get_qpos()]}
        seeds = parse(a.seeds)
        first = {}
        for s in seeds:
            r = sig(s); rows.append(r); first[s] = r["state_hash_exact"]
        rec = []
        for s in seeds[:: max(1, len(seeds) // a.recheck)][: a.recheck]:
            rec.append(sig(s)["state_hash_exact"] == first[s])
        print(json.dumps({"task": task, "recheck_identical": f"{sum(rec)}/{len(rec)}"}), flush=True)
        rows.append({"task": task, "recheck_identical": int(sum(rec)), "recheck_n": len(rec)})
        env.close()
    with open(os.path.join(a.out, "initial_states.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # summary
    summ = {}
    for task in TASKS:
        rs = [r for r in rows if r.get("task") == task and "seed" in r]
        by = {}
        for r in rs:
            by.setdefault(r["state_hash6"], []).append(r["seed"])
        old = {r["state_hash6"] for r in rs if r["seed"] < 200}
        new = [r for r in rs if r["seed"] >= 200]
        summ[task] = {"n_seeds": len(rs), "n_distinct_states": len(by),
                      "n_seeds_0_119_distinct": len({r["state_hash6"] for r in rs if r["seed"] < 120}),
                      "new_seeds_colliding_with_0_119": sum(r["state_hash6"] in old for r in new),
                      "n_new": len(new)}
    json.dump(summ, open(os.path.join(a.out, "initial_states_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
