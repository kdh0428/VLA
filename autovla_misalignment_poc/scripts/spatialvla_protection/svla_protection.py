#!/usr/bin/env python
"""
SpatialVLA feedback-protection ablation (outputs/spatialvla_feedback_protection_ablation/PROTOCOL.md). RTX 5090 only.

Execution configurations "r{K}e{E}": a new 4-step chunk is generated (from a fresh observation) every K control steps;
the executed action at step t is the exponentially weighted average (official ActionEnsembler rule, temp -0.8, newest
prediction weighted most) of the predictions for step t from the newest E chunks that cover t (0 <= t - t_gen < 4).
r1e4 is the official SpatialVLA SimplerEnv adapter (identical to experiment 34's PolicyState). The sticky-gripper rule
is applied unchanged after averaging.

Interventions (as experiment 34) apply to every chunk *generated* at control steps 8..23: step-1 translation token g is
replaced by p(g, d = 0.3, opposite) in execution and/or in the decoding context of steps 2-4:
  feedback  (exec p, ctx p)   corrected (exec p, ctx g)   reverse (exec g, ctx p)
natural = free decoding (step-wise decoder); natural_gen = the model's own predict_action (numeric-path rerun).
Decoding uses a fixed batch of 8 rows per prompt-length group (decode_fixed), so every row is a deterministic function
of its own observation and intervention: conditions differ only through the intervention.

Conditions are "<config>:<cond>". Writes <out>/episodes.jsonl (resumable): per step executed env action, the predictions
averaged at that step (world vector), TCP pose, generated chunk tokens; final scalar env info and ever-true flags.

  source /root/VLA/simpler/env.sh
  python svla_protection.py --out DIR --conditions r1e4:natural,r4e1:feedback,... --seeds 0-39 --workers 8
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
DIRECTION = "opposite"
HORIZON = 4
TEMP = -0.8


def worker(conn):
    for k, v in (("LP_NUM_THREADS", "2"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")):
        os.environ[k] = v
    import cv2 as cv
    cv.setNumThreads(1)
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    envs = {}

    def pose(env):
        p = env.unwrapped.tcp.pose
        return [float(x) for x in p.p] + [float(x) for x in p.q]

    def scal(info):
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in info.items()
                if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating))}
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            task, seed = arg
            if task not in envs:
                envs[task] = simpler_env.make(task)
            env = envs[task]
            obs, _ = env.reset(seed=seed)
            img = cv.resize(get_image_from_maniskill2_obs_dict(env, obs), (224, 224), interpolation=cv.INTER_AREA)
            conn.send((img, env.get_language_instruction(), pose(env)))
        elif cmd == "step":
            task, action = arg
            env = envs[task]
            obs, _, done, trunc, info = env.step(action)
            img = cv.resize(get_image_from_maniskill2_obs_dict(env, obs), (224, 224), interpolation=cv.INTER_AREA)
            conn.send((img, bool(done), bool(trunc), pose(env), scal(info)))
        elif cmd == "close":
            conn.send(None); return


class Executor:
    """Replan every K steps, average the newest E covering predictions (official weights), sticky gripper."""

    def __init__(self, K, E):
        self.K, self.E = K, E
        self.chunks = []                       # (t_gen, (4, 7) actions)
        self.sticky_on, self.repeat, self.sticky_action, self.prev_grip = False, 0, 0.0, None

    def plans_at(self, t):
        return t % self.K == 0

    def add(self, t, acts):
        self.chunks.append((t, np.asarray(acts, dtype=np.float64)))
        self.chunks = [(g, c) for g, c in self.chunks if t - g < HORIZON]

    def step(self, t):
        from transforms3d.euler import euler2axangle
        cov = sorted([(g, c[t - g]) for g, c in self.chunks if 0 <= t - g < HORIZON], key=lambda x: x[0])[-self.E:]
        preds = np.stack([p for _, p in cov])                     # oldest -> newest
        w = np.exp(-TEMP * np.arange(len(preds))); w = w / w.sum()
        a = (w[:, None] * preds).sum(0)
        world = np.array(a[:3]); roll, pitch, yaw = a[3:6]
        ax, ang = euler2axangle(roll, pitch, yaw); rot = ax * ang
        cur = np.array(a[6:7])
        if self.prev_grip is None:
            rel = np.array([0.0]); self.prev_grip = cur
        else:
            rel = self.prev_grip - cur
        if np.abs(rel) > 0.5 and not self.sticky_on:
            self.sticky_on, self.sticky_action, self.prev_grip = True, rel, cur
        if self.sticky_on:
            self.repeat += 1; rel = self.sticky_action
        if self.repeat == 10:
            self.sticky_on, self.repeat, self.sticky_action = False, 0, 0.0
        env_action = np.concatenate([world, rot, np.asarray(rel, dtype=np.float64).reshape(1)])
        return env_action, preds[:, :3], [int(t - g) for g, _ in cov]


FIXED_B = 8


def decode_fixed(m, inps, override, decode_chunks):
    """decode_chunks with every prompt-length group padded to FIXED_B rows (copies of its first row, no intervention).
    A row's greedy chunk then does not depend on how many other workers plan at the same step or on their content
    (checked: chunk changes with batch size for 4/40 rows, never with companions at a fixed batch size)."""
    n = len(inps)
    exe = np.zeros((n, 4, 3), dtype=np.int64); ctx = np.zeros((n, 4, 3), dtype=np.int64); nat1 = np.zeros(n, dtype=np.int64)
    groups = {}
    for i, x in enumerate(inps):
        groups.setdefault(int(x["input_ids"].shape[1]), []).append(i)
    for idx in groups.values():
        for s0 in range(0, len(idx), FIXED_B):
            part = idx[s0:s0 + FIXED_B]
            ins = [inps[i] for i in part] + [inps[part[0]]] * (FIXED_B - len(part))
            e, c, g = decode_chunks(m, ins, list(range(FIXED_B)), lambda r, gg: override(part[r], gg) if r < len(part) else None)
            exe[part], ctx[part], nat1[part] = e[:len(part)], c[:len(part)], g[:len(part)]
    return exe, ctx, nat1


def parse_cfg(cfg):
    k, e = cfg[1:].split("e")
    return int(k), int(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-"))
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "episodes.jsonl")
    done = {(r["task"], r["seed"], r["cond"]) for r in map(json.loads, open(dst))} if os.path.exists(dst) else set()
    queue = [(t, s, c) for c in a.conditions.split(",") for t in a.tasks.split(",") for s in range(s0, s1 + 1) if (t, s, c) not in done]
    print(f"[plan] {len(queue)} episodes", flush=True)
    from spatialvla_core import SpatialVLA, decode_chunks, translation_table
    from svla_rollouts import perturb_token
    m = SpatialVLA(); T = translation_table(m)
    ctx = mp.get_context("spawn")
    pipes = []
    for _ in range(a.workers):
        p_, c_ = ctx.Pipe(); pr = ctx.Process(target=worker, args=(c_,), daemon=True); pr.start(); pipes.append(p_)
    out = open(dst, "a"); t0 = time.time(); n = 0
    active = {}

    def start(w):
        task, seed, cond = queue.pop(0)
        pipes[w].send(("reset", (task, seed))); img, instr, ps = pipes[w].recv()
        cfg, mode = cond.split(":")
        active[w] = {"task": task, "seed": seed, "cond": cond, "mode": mode, "img": img, "instr": instr, "t": 0,
                     "ex": Executor(*parse_cfg(cfg)), "rec": [], "pose0": ps, "success": False, "final_info": {}, "ever": {}}
    for w in range(a.workers):
        if queue:
            start(w)
    while active:
        ws = sorted(active)
        plan = [w for w in ws if active[w]["ex"].plans_at(active[w]["t"])]
        gen_ws = [w for w in plan if active[w]["mode"] == "natural_gen"]
        dec_ws = [w for w in plan if active[w]["mode"] != "natural_gen"]
        chunks = {}
        if dec_ws:
            inps = [m.inputs(active[w]["img"], active[w]["instr"]) for w in dec_ws]
            specs = []
            for w in dec_ws:
                ep = active[w]
                specs.append(None if ep["mode"] == "natural" or not (WIN[0] <= ep["t"] < WIN[1]) else ep["mode"])

            def override(r, g):
                if specs[r] is None:
                    return None
                p, _, _ = perturb_token(T, g, D_CLOSED, DIRECTION)
                return {"feedback": (p, p), "corrected": (p, g), "reverse": (g, p)}[specs[r]]
            exe, cx, nat1 = decode_fixed(m, inps, override, decode_chunks)
            for r, w in enumerate(dec_ws):
                chunks[w] = (exe[r], cx[r], int(nat1[r]))
        for w in gen_ws:
            ids, _ = m.generate_reference(active[w]["img"], active[w]["instr"])
            ids = np.asarray(ids); chunks[w] = (ids, ids, int(ids[0, 0]))
        for w in ws:
            ep = active[w]; rec = {"t": ep["t"]}
            if w in chunks:
                exe, cx, g = chunks[w]
                ep["ex"].add(ep["t"], m.tokens_to_actions(exe))
                rec.update({"exec": exe.tolist(), "ctx": cx.tolist() if not np.array_equal(exe, cx) else None, "nat1": g})
            env_action, preds, ages = ep["ex"].step(ep["t"])
            rec.update({"action": np.round(env_action, 5).tolist(), "preds": np.round(preds, 5).tolist(), "ages": ages})
            ep["rec"].append(rec)
            pipes[w].send(("step", (ep["task"], env_action)))
        for w in ws:
            ep = active[w]
            img, dn, tr, ps, info = pipes[w].recv()
            ep["img"] = img; ep["rec"][-1]["pose"] = [round(x, 5) for x in ps]; ep["t"] += 1; ep["success"] = dn
            ep["final_info"] = info
            for k, v in info.items():
                if isinstance(v, bool):
                    ep["ever"][k] = ep["ever"].get(k, False) or v
            if tr or ep["t"] >= 80:
                out.write(json.dumps({"task": ep["task"], "seed": ep["seed"], "cond": ep["cond"], "instr": ep["instr"],
                                      "success": bool(ep["success"]), "steps": ep["t"], "pose0": ep["pose0"],
                                      "final_info": ep["final_info"], "ever": ep["ever"], "rec": ep["rec"]}) + "\n"); out.flush()
                n += 1; del active[w]
                if n % 40 == 0:
                    el = time.time() - t0
                    print(f"[prog] {n} episodes {el / 60:.1f} min ({n / el * 60:.2f}/min)", flush=True)
                if queue:
                    start(w)
    for p_ in pipes:
        p_.send(("close", None)); p_.recv()
    print(f"[done] {n} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
