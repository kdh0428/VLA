#!/usr/bin/env python
"""SpatialVLA P1-B/P1-C manifest (PROTOCOL_P1.md §8.1), CPU only, never reads an outcome.
Units come from the P1-D unit list (outputs/review_followup/P1D_dose_control/run_20261008/units.json).
Per task: order = sha256("P1-SVLA-20261014|" + task + "|" + seed); take the first 5 (move_near: first 5 with distinct
episode_id = distinct physical layout). Configs r1e4, r4e1 for every unit -> 20 units. Pilot = first 2 per task."""
import hashlib, json, os, sys
U = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1D_dose_control/run_20261008/units.json"
out = sys.argv[1]
if os.path.exists(out):
    raise SystemExit("exists")
units = json.load(open(U))
res = []
for task in ("google_robot_pick_coke_can", "google_robot_move_near"):
    keys = sorted(units[task], key=lambda s: hashlib.sha256(f"P1-SVLA-20261014|{task}|{s}".encode()).hexdigest())
    seen, picked = set(), []
    for s in keys:
        v = units[task][s]
        eid = v.get("episode_id")
        if eid is not None and eid in seen:
            continue
        seen.add(eid); picked.append((int(s), v))
        if len(picked) == 5:
            break
    for i, (s, v) in enumerate(picked):
        res.append({"task": task, "seed": s, "episode_id": v.get("episode_id"), "overlay": v.get("overlay"),
                    "configs": ["r1e4", "r4e1"], "pilot": i < 2, "rank": i})
json.dump({"rule": __doc__, "units_source": U, "units": res}, open(out, "w"), indent=1)
for r in res: print(r)
