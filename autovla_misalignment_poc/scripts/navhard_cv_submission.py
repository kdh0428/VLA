#!/usr/bin/env python
"""
Constant-velocity submission for the navhard subset: validates the official two-stage scoring path
(metric caches, split configs, submission format) before any model decoding, and gives a trivial baseline.
Pose k (k = 1..8, 0.5 s) = (|v| * 0.5k, 0, 0) in the ego frame.

  PYTHONPATH=/root/VLA/navsim_v2 python scripts/navhard_cv_submission.py <out.pkl>
"""
import json
import pickle
import sys

import numpy as np

SUBSET = "/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset"
JS = "/root/VLA/autovla/dataset/nuplan/navhard_half_json"

if __name__ == "__main__":
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    ts = TrajectorySampling(num_poses=8, interval_length=0.5)
    s1 = json.load(open(f"{SUBSET}/stage1_tokens.json")); s2 = json.load(open(f"{SUBSET}/synthetic_tokens.json"))
    def traj(t):
        v = float(np.hypot(*json.load(open(f"{JS}/{t}.json"))["velocity"]))
        return Trajectory(np.array([[v * 0.5 * k, 0.0, 0.0] for k in range(1, 9)], np.float32), ts)
    sub = {"first_stage_predictions": [{t: traj(t) for t in s1}], "second_stage_predictions": [{t: traj(t) for t in s2}]}
    pickle.dump(sub, open(sys.argv[1], "wb"))
    print(f"stage-1 {len(s1)}, stage-2 {len(s2)} -> {sys.argv[1]}")
