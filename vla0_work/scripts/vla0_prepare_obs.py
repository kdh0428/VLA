#!/usr/bin/env python
"""CPU: render LIBERO initial observations exactly as the native VLA-0 eval does before its first model call:
OffScreenRenderEnv(bddl, 256x256, camera_depths=True), env.seed(7), set_init_state(init_states[i]), 10 dummy steps
[0]*6+[-1] (frame_skip=10), then the observation. Saves raw agentview/wrist images (simulator orientation) + metadata.
Run with `source /root/VLA/openvla/env.sh` (LIBERO + robosuite + Mesa EGL)."""
import argparse, hashlib, json, os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITES = ["libero_spatial", "libero_object", "libero_goal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = json.load(open(a.manifest))
    os.makedirs(a.out, exist_ok=True)
    bd = benchmark.get_benchmark_dict()
    envs = {}
    for x in M["observations"]:
        dst = os.path.join(a.out, f"{x['obs_id']}.npz")
        if os.path.exists(dst):
            continue
        ts = bd[x["suite"]]()
        task = ts.get_task(x["task_index"])
        assert task.name == x["task_name"], (task.name, x["task_name"])
        init_states = ts.get_task_init_states(x["task_index"])
        key = (x["suite"], x["task_index"])
        if key not in envs:
            for e in envs.values():
                e.close()
            envs.clear()
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True)
            envs[key] = env
        env = envs[key]
        env.seed(7)
        env.reset()
        obs = env.set_init_state(init_states[x["init_state_index"]])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        ag, wr = obs["agentview_image"], obs["robot0_eye_in_hand_image"]
        np.savez_compressed(dst, agentview=ag, wrist=wr)
        print(x["obs_id"], task.language, hashlib.sha256(ag.tobytes()).hexdigest()[:12], flush=True)
    for e in envs.values():
        e.close()


if __name__ == "__main__":
    main()
