#!/usr/bin/env python
"""
Scene JSONs for the frames around each equal-distance scene, plus logged ego poses (CPU only).

For every scene f0 of the equal-distance set (208 tokens) this collects, from the navsim log
pickles, the previous frames (f0 - 1 .. f0 - 3, for previous-plan references) and the even future frames
f0 + 2 .. f0 + 14 (replanning points), and preprocesses them with preprocess_scenes.py into
dataset/nuplan/navtest_poc_frames (frame_interval 1, has_route off; a frame is only available if
it has 3 history and 10 future frames in the log). It also stores the logged global ego pose and
dynamic state of frames f0 .. f0 + 16 (8 s) for evaluation beyond the 5 s scene horizon.

Output: outputs/frame_index/index.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import yaml
from pyquaternion import Quaternion

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOVLA_DIR = "/root/VLA/autovla"
LOGS = os.path.join(AUTOVLA_DIR, "dataset/nuplan/navsim_logs/test")
EVAL_STEPS = 16
FUTURE = list(range(2, 15, 2))
PAST = [1, 2, 3]


def pose_of(fr):
    t = fr["ego2global_translation"]
    return [float(t[0]), float(t[1]), float(Quaternion(*fr["ego2global_rotation"]).yaw_pitch_roll[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--out", default=os.path.join(POC_DIR, "outputs/frame_index"))
    ap.add_argument("--scenes-out", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc_frames"))
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--units", default=None, help="JSON list of {token, log, group} used instead of --ed-records")
    ap.add_argument("--index-name", default="index.json")
    ap.add_argument("--no-future", action="store_true", help="only past frames (no replanning frames)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ed = json.load(open(args.units)) if args.units else [json.loads(l) for l in open(args.ed_records)]
    if args.no_future:
        global FUTURE
        FUTURE = []
    by_log = {}
    for r in ed:
        by_log.setdefault(r["log"], []).append(r)

    index, need = {}, set()
    for log, rs in sorted(by_log.items()):
        frames = pickle.load(open(os.path.join(LOGS, f"{log}.pkl"), "rb"))
        pos = {fr["token"]: i for i, fr in enumerate(frames)}
        n = len(frames)
        ok_scene = lambda j: 3 <= j and j + 10 < n          # noqa: E731  3 history + 10 future frames
        for r in rs:
            i = pos[r["token"]]
            ent = {"log": log, "group": r["group"], "idx": i, "n_frames": n,
                   "prev": frames[i - 1]["token"] if ok_scene(i - 1) else None,
                   "past": {str(k): frames[i - k]["token"] for k in PAST if ok_scene(i - k)},
                   "future": {str(k): frames[i + k]["token"] for k in FUTURE if ok_scene(i + k)},
                   "log_pose": [pose_of(frames[j]) for j in range(i, min(n, i + EVAL_STEPS + 1))],
                   "log_dyn": [list(map(float, frames[j]["ego_dynamic_state"])) for j in range(i, min(n, i + EVAL_STEPS + 1))]}
            index[r["token"]] = ent
            if ent["prev"]:
                need.add(ent["prev"])
            need.update(ent["future"].values())
            need.update(ent["past"].values())

    json.dump(index, open(os.path.join(args.out, args.index_name), "w"))
    n_prev = sum(e["prev"] is not None for e in index.values())
    n_full = sum(len(e["log_pose"]) == EVAL_STEPS + 1 for e in index.values())
    n_fut = {k: sum(k in e["future"] for e in index.values()) for k in map(str, FUTURE)}
    print(f"[index] scenes {len(index)}, prev frame available {n_prev}, 8 s log horizon {n_full}, "
          f"future frames available {n_fut}; frames to preprocess {len(need)}", flush=True)

    if os.path.isdir(args.scenes_out):
        have0 = {f[:-5] for f in os.listdir(args.scenes_out)}
        todo = sorted(need - have0)
        print(f"[index] {len(need & have0)} frames already preprocessed, {len(todo)} to go", flush=True)
    else:
        todo = sorted(need)
    flt = {"_target_": "navsim.common.dataclasses.SceneFilter", "_convert_": "all",
           "num_history_frames": 4, "num_future_frames": 10, "frame_interval": 1, "has_route": False,
           "max_scenes": None, "log_names": sorted(by_log), "tokens": todo}
    fpath = os.path.join(args.out, "scene_filter_frames.yaml" if not args.units else
                         f"scene_filter_{os.path.splitext(args.index_name)[0]}.yaml")
    yaml.safe_dump(flt, open(fpath, "w"), sort_keys=False)
    if args.skip_preprocess or not todo:
        return
    subprocess.run([sys.executable, os.path.join(POC_DIR, "scripts/preprocess_scenes.py"), "--scene-filter", fpath,
                    "--out", args.scenes_out, "--anno-out", os.path.join(args.out, "annotations")], check=True)
    have = {f[:-5] for f in os.listdir(args.scenes_out)}
    print(f"[done] preprocessed {len(need & have)}/{len(need)} frames -> {args.scenes_out}")


if __name__ == "__main__":
    main()
