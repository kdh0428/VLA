#!/usr/bin/env python
"""
Closed-loop OpenVLA rollouts on LIBERO-Spatial with token interventions (outputs/cross_vla_replication/PROTOCOL.md).

K simulator processes (Mesa EGL software rendering) step episodes of the same task in lockstep; the main process
decodes all of them in one batch on the RTX 5090. Episodes are appended to <out>/episodes.jsonl (resumable).

  source /root/VLA/openvla/env.sh
  python openvla_rollouts.py --phase A --out .../rollouts/phaseA          # natural, saves every 5th preprocessed frame
  python openvla_rollouts.py --phase B --out .../rollouts/phaseB          # feedback / corrected / reverse x |delta| 8, 24
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

SUITE, MAX_STEPS, WAIT = "libero_spatial", 220, 10
WIN = (10, 20)                    # control steps with the intervention (phase B)
BDDL = "/root/VLA/openvla/LIBERO/libero/libero/bddl_files"


def worker(conn, lp_threads):
    os.environ["LP_NUM_THREADS"] = str(lp_threads)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    from openvla_preproc import preprocess
    suite = benchmark.get_benchmark_dict()[SUITE]()
    env, cur = None, None
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            task_id, init = arg
            if cur != task_id:
                if env is not None:
                    env.close()
                t = suite.get_task(task_id)
                env = OffScreenRenderEnv(bddl_file_name=os.path.join(BDDL, t.problem_folder, t.bddl_file),
                                         camera_heights=256, camera_widths=256, camera_names=["agentview"])
                env.seed(0); cur = task_id
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(task_id)[init])
            for _ in range(WAIT):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            conn.send((np.asarray(preprocess(obs["agentview_image"])), False))
        elif cmd == "step":
            action, save = arg
            obs, _, done, _ = env.step(action)
            img = np.asarray(preprocess(obs["agentview_image"]))
            if save:
                Image.fromarray(img).save(save)
            conn.send((img, bool(done)))
        elif cmd == "close":
            if env is not None:
                env.close()
            conn.send(None)
            return


def conditions(phase):
    if phase == "A":
        return [{"name": "natural"}]
    return [{"name": f"{m}_d{a}", "mode": m, "abs_delta": a} for a in (8, 24) for m in ("feedback", "corrected", "reverse")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0-9")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lp-threads", type=int, default=2)
    a = ap.parse_args()
    rng_ = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))   # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "episodes.jsonl")
    done_keys = {(r["task"], r["init"], r["cond"]) for r in map(json.loads, open(dst))} if os.path.exists(dst) else set()
    from libero.libero import benchmark
    from openvla_core import OpenVLA, to_env_action
    suite = benchmark.get_benchmark_dict()[SUITE]()
    vla = OpenVLA()
    ctx = mp.get_context("spawn")
    pipes, procs = [None] * a.workers, [None] * a.workers
    COMM = (BrokenPipeError, EOFError, ConnectionResetError, OSError)

    def spawn(w):                                     # (re)start simulator worker w
        if procs[w] is not None and procs[w].is_alive():
            procs[w].kill()
        p_, c_ = ctx.Pipe(); pr = ctx.Process(target=worker, args=(c_, a.lp_threads), daemon=True); pr.start()
        pipes[w], procs[w] = p_, pr
    for w in range(a.workers):
        spawn(w)
    n_restart = 0
    out = open(dst, "a"); t_start = time.time(); n_done = 0
    for task_id in rng_(a.tasks):
        lang = suite.get_task(task_id).language
        queue = [{"task": task_id, "init": i, **c} for i in rng_(a.inits) for c in conditions(a.phase)
                 if (task_id, i, c["name"]) not in done_keys]
        active = {}                                   # worker -> episode state

        def start(w):
            nonlocal n_restart
            while queue:
                ep = queue.pop(0)
                sign = 1 if ep["init"] % 2 == 0 else -1
                ep.update(t=0, rec=[], delta=sign * ep.get("abs_delta", 0))
                if a.phase == "A":
                    os.makedirs(os.path.join(a.out, "frames", f"t{task_id}_i{ep['init']}"), exist_ok=True)
                try:
                    pipes[w].send(("reset", (task_id, ep["init"]))); ep["img"], _ = pipes[w].recv()
                    active[w] = ep
                    return
                except COMM:                          # worker died (e.g. host OOM): restart it, retry the episode
                    n_restart += 1; print(f"[restart] worker {w} at reset", flush=True)
                    queue.insert(0, {k: v for k, v in ep.items() if k not in ("t", "rec", "img")}); spawn(w)

        def lost(w):
            nonlocal n_restart
            ep = active.pop(w); n_restart += 1
            print(f"[restart] worker {w} lost task {ep['task']} init {ep['init']} {ep['name']} at t={ep['t']}", flush=True)
            queue.insert(0, {k: v for k, v in ep.items() if k not in ("t", "rec", "img")}); spawn(w)
        for w in range(a.workers):
            if queue:
                start(w)
        while active:
            ws = sorted(active)
            imgs, ivs, rows = [], [], []              # rows: (worker, is_counterfactual_natural)
            for w in ws:
                ep = active[w]
                use = "mode" in ep and WIN[0] <= ep["t"] < WIN[1]
                imgs.append(ep["img"]); ivs.append({"dim": 0, "delta": ep["delta"], "mode": ep["mode"]} if use else None)
                rows.append((w, False))
                if use:
                    imgs.append(ep["img"]); ivs.append(None); rows.append((w, True))
            from PIL import Image
            res = vla.decode_batch([Image.fromarray(x) for x in imgs], lang, ivs)
            cf = {w: k for k, (w, isn) in enumerate(rows) if isn}
            for k, (w, isn) in enumerate(rows):
                if isn:
                    continue
                ep = active[w]
                r = {"t": ep["t"], "exec": res["exec_bins"][k].tolist(), "ent": np.round(res["entropy"][k], 4).tolist(),
                     "margin": np.round(res["margin"][k], 4).tolist(), "pmax": np.round(res["pmax"][k], 4).tolist()}
                if w in cf:
                    j = cf[w]; r["ctx"] = res["ctx_bins"][k].tolist(); r["natural"] = res["exec_bins"][j].tolist()
                    r["natural_ent"] = np.round(res["entropy"][j], 4).tolist()
                ep["rec"].append(r)
                save = (os.path.join(a.out, "frames", f"t{task_id}_i{ep['init']}", f"{ep['t'] + 1:03d}.png")
                        if a.phase == "A" and (ep["t"] + 1) % 5 == 0 else None)
                try:
                    pipes[w].send(("step", (to_env_action(res["action"][k]), save)))
                except COMM:
                    lost(w)
            for w in ws:
                if w not in active:
                    start(w); continue
                ep = active[w]
                try:
                    ep["img"], dn = pipes[w].recv(); ep["t"] += 1
                except COMM:
                    lost(w); start(w); continue
                if dn or ep["t"] >= MAX_STEPS:
                    out.write(json.dumps({"task": ep["task"], "init": ep["init"], "cond": ep["name"], "delta": ep["delta"],
                                          "success": bool(dn), "steps": ep["t"], "rec": ep["rec"]}) + "\n"); out.flush()
                    n_done += 1; del active[w]
                    if queue:
                        start(w)
        el = time.time() - t_start
        print(f"[task {task_id}] episodes done {n_done}  {el / 60:.1f} min", flush=True)
    for p_ in pipes:
        try:
            p_.send(("close", None)); p_.recv()
        except COMM:
            pass
    print(f"[done] worker restarts {n_restart}", flush=True)


if __name__ == "__main__":
    main()
