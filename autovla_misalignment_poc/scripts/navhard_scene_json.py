#!/usr/bin/env python
"""
AutoVLA scene JSONs for the navhard subset (stage 1 originals + stage 2 synthetic scenes; CPU only).

Same fields the AutoVLA feature builder / prompt read, built like preprocess_scenes.py / VlaAgent:
  velocity, acceleration      ego-frame vectors of the current (last history) frame
  instruction                 argmax(driving_command) -> turn left / keep forward / turn right / unknown
  front/left/right camera     CAM_F0 / CAM_L1 / CAM_R1 of the 4 history frames (absolute paths)
The prompt uses no history or future trajectory; gt_trajectory is a zero placeholder the feature
builder requires. Stage-1 JSONs also keep the logged 4 s future (gt_future_4s) for reference.

Output: <out>/<token>.json, keyed by the token the NAVSIM v2 submission expects (stage-1 token /
synthetic initial_token).
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

SUBSET = os.environ.get("NAVHARD_SUBSET", "/root/VLA/autovla_misalignment_poc/outputs/navhard_eval/subset")   # other navhard subsets via env
LOGS = "/root/VLA/autovla/dataset/nuplan/navsim_logs/test"
ORIG_SENS = "/root/VLA/autovla/dataset/nuplan/sensor_blobs/test"          # PoC logs
ORIG_SENS_NAVHARD = "/root/VLA/navhard/original_sensor_blobs"            # other logs (kept apart from streamed shards)
SYN = "/root/VLA/navhard/navhard_two_stage/synthetic_scene_pickles"
SYN_SENS = "/root/VLA/navhard/navhard_two_stage/sensor_blobs"
OUT = os.environ.get("NAVHARD_JSON", "/root/VLA/autovla/dataset/nuplan/navhard_half_json")
CMD = {0: "turn left", 1: "keep forward", 2: "turn right", 3: "unknown"}
CAMS = {"front_camera_paths": "CAM_F0", "left_camera_paths": "CAM_L1", "right_camera_paths": "CAM_R1"}


def record(token, vel, acc, cmd, paths, extra=None):
    r = {"token": token, "dataset_name": "nuplan", "cot_output": [], "velocity": [float(vel[0]), float(vel[1])],
         "acceleration": [float(acc[0]), float(acc[1])], "instruction": CMD[int(np.argmax(cmd))],
         "gt_trajectory": np.zeros((10, 3)).tolist(), "his_trajectory": np.zeros((4, 3)).tolist(), **paths}
    if extra:
        r.update(extra)
    return r


def rel_or(root, rel):
    """root/rel if it exists, else the navhard originals dir (PoC logs lack some navhard frames)."""
    p = os.path.join(root, rel)
    return p if os.path.exists(p) else os.path.join(ORIG_SENS_NAVHARD, rel)


def rel_pose(p, o):
    c, s = np.cos(o[2]), np.sin(o[2]); dx, dy = p[0] - o[0], p[1] - o[1]
    return [c * dx + s * dy, -s * dx + c * dy, float(np.arctan2(np.sin(p[2] - o[2]), np.cos(p[2] - o[2])))]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    groups = json.load(open(f"{SUBSET}/groups.json"))
    s1 = set(json.load(open(f"{SUBSET}/stage1_tokens.json")))
    init2scene = json.load(open(f"{SUBSET}/synthetic_initial_to_scene_token.json"))
    from pyquaternion import Quaternion
    import yaml
    poc = set(yaml.safe_load(open("/root/VLA/autovla_misalignment_poc/configs/scene_filter_navtest_subset.yaml"))["log_names"])
    n1 = n2 = 0
    missing = []
    for lg in sorted({g["log"] for g in groups}):
        frames = pickle.load(open(f"{LOGS}/{lg}.pkl", "rb"))
        pos = {f["token"]: i for i, f in enumerate(frames)}
        for t in [t for t in s1 if t in pos]:
            i = pos[t]; cur = frames[i]
            root = ORIG_SENS if lg in poc else ORIG_SENS_NAVHARD
            pick = lambda rel: rel_or(root, rel)   # noqa: E731
            paths = {k: [pick(frames[j]["cams"][c]["data_path"]) for j in range(i - 3, i + 1)] for k, c in CAMS.items()}
            missing += [p for v in paths.values() for p in v if not os.path.exists(p)]
            pose = lambda f: [f["ego2global_translation"][0], f["ego2global_translation"][1], Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0]]  # noqa: E731
            fut = [rel_pose(pose(frames[j]), pose(cur)) for j in range(i + 1, min(len(frames), i + 9))]
            ds = cur["ego_dynamic_state"]
            json.dump(record(t, ds[:2], ds[2:], cur["driving_command"], paths, {"stage": 1, "log": lg, "gt_future_4s": fut}),
                      open(f"{OUT}/{t}.json", "w"))
            n1 += 1
    for t in json.load(open(f"{SUBSET}/synthetic_tokens.json")):
        d = pickle.load(open(f"{SYN}/{init2scene[t]}.pkl", "rb"))
        fr = d["frames"]
        paths = {k: [os.path.join(SYN_SENS, f["camera_dict"][c.lower()]["data_path"]) for f in fr[-4:]] for k, c in CAMS.items()}
        missing += [p for v in paths.values() for p in v if not os.path.exists(p)]
        es = fr[-1]["ego_status"]
        json.dump(record(t, es["ego_velocity"], es["ego_acceleration"], es["driving_command"], paths,
                         {"stage": 2, "log": d["scene_metadata"]["log_name"], "scene_token": init2scene[t]}), open(f"{OUT}/{t}.json", "w"))
        n2 += 1
    print(f"stage-1 {n1}, synthetic {n2}, image paths missing on disk: {len(missing)}")
    json.dump(missing, open(f"{SUBSET}/missing_images.json", "w"))


if __name__ == "__main__":
    main()
