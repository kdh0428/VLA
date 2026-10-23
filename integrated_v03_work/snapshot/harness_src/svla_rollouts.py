#!/usr/bin/env python
"""
Closed-loop SpatialVLA rollouts in SimplerEnv (Google Robot) with chunk interventions (PROTOCOL.md, experiments A-D closed loop).

K simulator processes step episodes in lockstep; all rows are decoded in one batched greedy pass per control step
(spatialvla_core.decode_chunks, identical tokens to generate). Conditions:
  natural                 free decoding
  natural_gen             the model's own predict_action (generate, batch of one) — numeric-path rerun
  feedback_{dir}          step-1 translation token := p(g, d, dir) for chunks generated at control steps in [8, 24)
  corrected_{dir}         executed step-1 = p, steps 2-4 generated with the natural step-1 token in context
  reverse_{dir}           executed step-1 = natural, steps 2-4 generated after p
Execution as the official adapter (ensemble temp -0.8 over the last 4 chunks, sticky gripper), 80 steps, success = done at
the last step. Writes <out>/episodes.jsonl (resumable) and, for --save-frames, every 4th observation as PNG.

  source /root/VLA/simpler/env.sh
  python svla_rollouts.py --out DIR --conditions natural,feedback_perp_left,... --seeds 0-39 [--save-frames]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

TASKS = ("google_robot_pick_coke_can", "google_robot_move_near")
WIN = (8, 24)
D_CLOSED = 0.3


def worker(conn):
    for k, v in (("LP_NUM_THREADS", "2"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")):
        os.environ[k] = v                                # avoid thread oversubscription (8 simulators, software Vulkan)
    import cv2 as cv
    cv.setNumThreads(1)
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    envs = {}
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            task, seed = arg
            if task not in envs:
                envs[task] = simpler_env.make(task)
            env = envs[task]
            obs, _ = env.reset(seed=seed)
            img = cv.resize(get_image_from_maniskill2_obs_dict(env, obs), (224, 224), interpolation=cv.INTER_AREA)
            conn.send((img, env.get_language_instruction()))
        elif cmd == "step":
            task, action = arg
            env = envs[task]
            obs, _, done, trunc, info = env.step(action)
            img = cv.resize(get_image_from_maniskill2_obs_dict(env, obs), (224, 224), interpolation=cv.INTER_AREA)
            conn.send((img, bool(done), bool(trunc)))
        elif cmd == "close":
            conn.send(None); return


def perturb_token(T, g, d, direction):
    """Translation token whose normalised vector is closest to v(g) + d * u (u from the natural motion direction)."""
    v = T["norm"][T["id2row"][g]]
    vh = v / max(np.linalg.norm(v), 1e-6)
    xy = np.array([v[0], v[1], 0.0]); xy = xy / np.linalg.norm(xy) if np.linalg.norm(xy) > 1e-6 else np.array([1.0, 0, 0])
    u = {"along": vh, "opposite": -vh, "perp_left": np.array([-xy[1], xy[0], 0.0]), "perp_right": np.array([xy[1], -xy[0], 0.0]),
         "up": np.array([0, 0, 1.0]), "down": np.array([0, 0, -1.0])}[direction]
    j = int(np.argmin(np.linalg.norm(T["norm"] - (v + d * u), axis=1)))
    return int(T["ids"][j]), float(np.linalg.norm(T["norm"][j] - v)), u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-frames", action="store_true")
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-"))
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "episodes.jsonl")
    done = {(r["task"], r["seed"], r["cond"]) for r in map(json.loads, open(dst))} if os.path.exists(dst) else set()
    queue = [(t, s, c) for c in a.conditions.split(",") for t in a.tasks.split(",") for s in range(s0, s1 + 1) if (t, s, c) not in done]
    print(f"[plan] {len(queue)} episodes", flush=True)
    from spatialvla_core import SpatialVLA, decode_chunks, translation_table
    from spatialvla_policy import PolicyState
    m = SpatialVLA(); T = translation_table(m)
    ctx = mp.get_context("spawn")
    pipes = []
    for _ in range(a.workers):
        p_, c_ = ctx.Pipe(); pr = ctx.Process(target=worker, args=(c_,), daemon=True); pr.start(); pipes.append(p_)
    out = open(dst, "a"); t0 = time.time(); n = 0
    active = {}

    def start(w):
        task, seed, cond = queue.pop(0)
        pipes[w].send(("reset", (task, seed))); img, instr = pipes[w].recv()
        if a.save_frames:
            os.makedirs(os.path.join(a.out, "frames", f"{task}_s{seed}"), exist_ok=True)
        active[w] = {"task": task, "seed": seed, "cond": cond, "img": img, "instr": instr, "t": 0, "pol": PolicyState(),
                     "rec": [], "success": False}
    for w in range(a.workers):
        if queue:
            start(w)
    while active:
        ws = sorted(active)
        gen_ws = [w for w in ws if active[w]["cond"] == "natural_gen"]
        dec_ws = [w for w in ws if active[w]["cond"] != "natural_gen"]
        chunks = {}
        if dec_ws:
            inps = [m.inputs(active[w]["img"], active[w]["instr"]) for w in dec_ws]
            specs = []
            for w in dec_ws:
                ep = active[w]; c = ep["cond"]
                if c.startswith("natural") or not (WIN[0] <= ep["t"] < WIN[1]):
                    specs.append(None)
                else:
                    mode, direction = c.split("_", 1)
                    specs.append((mode, direction))

            def override(r, g):
                sp = specs[r]
                if sp is None:
                    return None
                p, dist, _ = perturb_token(T, g, D_CLOSED, sp[1])
                return {"feedback": (p, p), "corrected": (p, g), "reverse": (g, p)}[sp[0]]
            exe, cx, nat1 = decode_chunks(m, inps, list(range(len(dec_ws))), override)
            for r, w in enumerate(dec_ws):
                chunks[w] = (exe[r], cx[r], int(nat1[r]))
        for w in gen_ws:
            ids, _ = m.generate_reference(active[w]["img"], active[w]["instr"])
            chunks[w] = (np.asarray(ids), np.asarray(ids), int(ids[0, 0]))
        for w in ws:
            ep = active[w]; exe, cx, g = chunks[w]
            acts = m.tokens_to_actions(exe)
            env_action, ens = ep["pol"].to_env(acts)
            ep["rec"].append({"t": ep["t"], "exec": exe.tolist(), "ctx": cx.tolist() if not np.array_equal(exe, cx) else None,
                              "nat1": g, "action": np.round(env_action, 5).tolist()})
            if a.save_frames and ep["t"] % 4 == 0:
                import cv2 as cv
                cv.imwrite(os.path.join(a.out, "frames", f"{ep['task']}_s{ep['seed']}", f"{ep['t']:03d}.png"), ep["img"][:, :, ::-1])
            pipes[w].send(("step", (ep["task"], env_action)))
        for w in ws:
            ep = active[w]
            img, dn, tr = pipes[w].recv(); ep["img"] = img; ep["t"] += 1; ep["success"] = dn
            if tr or ep["t"] >= 80:
                out.write(json.dumps({"task": ep["task"], "seed": ep["seed"], "cond": ep["cond"], "instr": ep["instr"],
                                      "success": bool(ep["success"]), "steps": ep["t"], "rec": ep["rec"]}) + "\n"); out.flush()
                n += 1; del active[w]
                if queue:
                    start(w)
    for p_ in pipes:
        p_.send(("close", None)); p_.recv()
    print(f"[done] {n} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
