#!/usr/bin/env python
"""
P1-D runner: SpatialVLA (SimplerEnv Google Robot) execution structure x dose control.
Protocol: outputs/review_followup/P1D_dose_control/<run_id>/protocol.md. RTX 5090 only (source /root/VLA/simpler/env.sh).

Execution configurations r{K}e{E} exactly as experiment 35 (svla_protection.Executor: new 4-step chunk every K steps,
official ensemble weights over the newest E covering predictions, sticky gripper). Decoding as experiment 35
(decode_fixed: every prompt-length group padded to 8 rows).

Branching / full-state restoration
  Every (task, seed, config) group runs its natural episode N from reset(seed). At the branch step T0 (= first
  injection step, 8) N's executor (chunk buffer, ensemble buffer, sticky-gripper state) is deep-copied and N's exact
  float64 actions for t < T0 are stored. Each other condition of the group resets the env with the same seed, replays
  N's actions (no decoding), checks the full simulator state hash (env.get_state()) and the observation hash against
  N's at every replayed step, restores the executor copy and continues closed loop from T0. The controller / gripper
  target state is restored by re-simulation (it is not part of get_state()). Pre-branch divergence (experiment 35
  had ~3.7% of condition-vs-natural pairs diverging before t = 8 through decode non-determinism) is therefore
  impossible by construction and every branch carries its own restoration check (`restore_ok`).

Interventions (only at the injection steps --inject, default 8,12,16,20; every configuration generates a chunk there)
  g = greedy step-1 translation token, p = perturb_token(g, 0.3, opposite) (experiment 34/35 rule).
    natural   no intervention (the group's root)
    null      override returns (g, g) through the intervention path + shadow decode    -> must equal N (no-op check)
    feedback  exec p, ctx p            corrected  exec p, ctx g           reverse  exec g, ctx p
    feedbackD / correctedD  as feedback / corrected, plus a post-ensemble world-vector offset (1 - w0) * (phys(p) -
              phys(g)) at the injection step, so the executed step-t translation equals the natural-ensemble action
              plus exactly Delta = phys(p) - phys(g) whatever E is (direct executed dose matched across E;
              identical to feedback / corrected when E = 1).
    gen       the model's own predict_action from T0 on (numeric-path re-run noise, branch at T0).
  Shadow decode: for every injected chunk the counterfactual-context chunk at the same observation is decoded in the
  same fixed batch (feedback<->corrected swap the context, reverse/null use the other of (g, p)); per executed step
  the runner records the pre-sticky ensemble action difference actual - counterfactual (= context-attributable
  executed deviation at the actual state) and the direct executed dose at each injection step.
  --inject window:8-24 reproduces experiment 35's schedule (every chunk generated in [8, 24)) for validation.

Records (episodes.jsonl, resumable): per step action (float64 exact), preds, ages, TCP pose, gripper closedness,
finger contacts, sim-state hash, info flags; per generation call tokens (exec, ctx, cf), nat1, injection record;
per episode success, final info, ever-true flags, restoration check, timing.

  python p1d_runner.py --out DIR --configs r1e1,r4e1 --modes natural,null,reverse --seeds 200-201 --workers 2
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "spatialvla_protection"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cross_domain_temporal"))
from svla_protection import Executor, decode_fixed, parse_cfg, TEMP, HORIZON  # noqa: E402

TASKS = ("google_robot_pick_coke_can", "google_robot_move_near")
D_CLOSED, DIRECTION = 0.3, "opposite"
MODES = ("natural", "null", "feedback", "corrected", "reverse", "feedbackD", "correctedD", "gen")
H = lambda x: hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- simulator worker
def worker(conn):
    for k, v in (("LP_NUM_THREADS", "2"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")):
        os.environ[k] = v
    import cv2 as cv
    cv.setNumThreads(1)
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    envs = {}

    def observe(env, obs):
        u = env.unwrapped
        img = cv.resize(get_image_from_maniskill2_obs_dict(env, obs), (224, 224), interpolation=cv.INTER_AREA)
        p = u.tcp.pose
        robot_links = {l.name for l in u.agent.robot.get_links()}
        fingers = {"link_finger_left", "link_finger_right", "link_finger_tip_left", "link_finger_tip_right"}
        touched = set()
        for c in u._scene.get_contacts():
            n0, n1 = c.actor0.name, c.actor1.name
            for a_, b_ in ((n0, n1), (n1, n0)):
                if a_ in fingers and b_ not in robot_links:
                    if sum(np.linalg.norm(pt.impulse) for pt in c.points) > 1e-6:
                        touched.add(b_)
        return img, {"pose": [float(x) for x in p.p] + [float(x) for x in p.q],
                     "grip": float(u.agent.get_gripper_closedness()), "contacts": sorted(touched),
                     "state_hash": H(np.asarray(u.get_state(), dtype=np.float64)), "img_hash": H(img)}

    def scal(info):
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in info.items()
                if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating))}
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":                         # (task, seed, replay_actions): reset, then replay a prefix
            task, seed, replay, eid = arg
            if task not in envs:
                envs[task] = simpler_env.make(task)
            env = envs[task]
            # reconfigure=True: rebuild the scene on every reset. Without it the initial state depends on the
            # previous episode run in this process (checked: move_near seed 0 differs by ~2e-6 when the same scene
            # is reused, trajectories then diverge); with it reset(seed) is bit-reproducible whatever ran before.
            opts = {"reconfigure": True}
            if eid is not None:
                opts["obj_init_options"] = {"episode_id": int(eid)}
            obs, info0 = env.reset(seed=seed, options=opts)
            info0 = dict(info0); info0["overlay"] = getattr(env.unwrapped, "urdf_version", None)
            img, ob = observe(env, obs)
            trace = [ob]
            info = {}
            for a in replay:
                obs, _, done, trunc, info = env.step(np.asarray(a, dtype=np.float64))
                img, ob = observe(env, obs); ob["info"] = scal(info); trace.append(ob)
            i0 = scal(info0); i0["overlay"] = info0["overlay"]
            conn.send((img, env.get_language_instruction(), trace, i0))
        elif cmd == "step":
            task, action = arg
            env = envs[task]
            obs, _, done, trunc, info = env.step(action)
            img, ob = observe(env, obs); ob["info"] = scal(info)
            conn.send((img, bool(done), bool(trunc), ob))
        elif cmd == "close":
            conn.send(None); return


# ----------------------------------------------------------------------------------------------- executor
class Exec2(Executor):
    """svla_protection.Executor (unchanged arithmetic for the executed action) + counterfactual bookkeeping."""

    def __init__(self, K, E):
        super().__init__(K, E)
        self.cf = {}                                # t_gen -> (4, 7) counterfactual-context chunk actions

    def ens_raw(self, t, use_cf):
        cov = sorted([(g, c[t - g]) for g, c in self.chunks if 0 <= t - g < HORIZON], key=lambda x: x[0])[-self.E:]
        preds = np.stack([(self.cf[g][t - g] if (use_cf and g in self.cf) else p) for g, p in cov])
        w = np.exp(-TEMP * np.arange(len(preds))); w = w / w.sum()
        return (w[:, None] * preds).sum(0), w, [int(t - g) for g, _ in cov]


# ----------------------------------------------------------------------------------------------- main
def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        a, b = (part.split("-") + [part])[:2]
        out += list(range(int(a), int(b) + 1))
    return out


def parse_inject(spec):
    if spec.startswith("window:"):
        a, b = map(int, spec[7:].split("-"))
        return ("window", a, b)
    return ("times", sorted(int(x) for x in spec.split(",")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True, help="e.g. r1e1,r2e1,r4e1,r1e4")
    ap.add_argument("--modes", required=True, help=",".join(MODES))
    ap.add_argument("--seeds", required=True, help="e.g. 200-319")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--inject", default="8,12,16,20")
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the condition key (re-runs of identical conditions)")
    ap.add_argument("--config-modes", default="", help="additional modes for single configs, e.g. "
                    "'r1e4:feedbackD,correctedD;r4e1:gen'")
    ap.add_argument("--units-file", default="", help="units.json from build_units.py: per-task unit seeds (+ move_near "
                    "episode_id); with it, --seeds selects units by position, e.g. 0-119")
    ap.add_argument("--extra", default="", help="extra explicit groups task:seed:cfg:mode;... (smoke tests)")
    a = ap.parse_args()
    inj = parse_inject(a.inject)
    T0 = inj[1] if inj[0] == "window" else inj[1][0]

    def is_inj(t):
        return (inj[1] <= t < inj[2]) if inj[0] == "window" else (t in inj[1])
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "episodes.jsonl")
    done = set()
    if os.path.exists(dst):
        for line in open(dst):
            r = json.loads(line); done.add((r["task"], r["seed"], r["cond"]))
    cfgs = a.configs.split(","); modes = a.modes.split(",")
    cfg_modes = {}
    for part in filter(None, a.config_modes.split(";")):
        c, ms = part.split(":"); cfg_modes[c] = ms.split(",")
        assert c in cfgs and all(m in MODES for m in cfg_modes[c]), part
    assert all(m in MODES for m in modes), modes
    for c in cfgs:
        K, _ = parse_cfg(c)
        assert all(t % K == 0 for t in (inj[1] if inj[0] == "times" else [T0])), f"{c} does not plan at every injection step"
    units = json.load(open(a.units_file)) if a.units_file else None
    eid_of = {}
    if units:
        for t in TASKS:
            for k, v in units.get(t, {}).items():
                if "episode_id" in v:
                    eid_of[(t, int(k))] = (v["episode_id"], v.get("overlay"))
    items = []
    for t in a.tasks.split(","):
        seed_list = parse_seeds(a.seeds)
        if units:
            allu = sorted(int(k) for k in units[t])
            seed_list = [allu[i] for i in seed_list]
        for s in seed_list:
            for c in cfgs:
                for m in ["natural"] + [m for m in modes if m != "natural"] + cfg_modes.get(c, []):
                    items.append((t, s, c, m))
    if a.extra:
        for g in a.extra.split(";"):
            t, s, c, m = g.split(":"); items.append((t, int(s), c, m))
    tagc = (lambda c, m: f"{c}:{m}{a.tag}")
    pending = [it for it in items if (it[0], it[1], tagc(it[2], it[3])) not in done]
    # every group with a pending branch needs its root in this run (for the T0 snapshot); a root that is already
    # recorded is re-run under the key "<cfg>:natural#rerun" (doubles as a re-run reproducibility check)
    roots = {(t, s, c) for t, s, c, _ in pending}
    queue = [it for it in pending if it[3] != "natural"]
    for (t, s, c) in roots:
        queue.append((t, s, c, "natural" if (t, s, tagc(c, "natural")) not in done else "natural#rerun"))
    queue = list(dict.fromkeys(queue))
    queue.sort(key=lambda it: (it[1], it[0], it[2], not it[3].startswith("natural")))   # each root before its branches
    print(f"[plan] {len(queue)} episodes (roots {sum(it[3] == 'natural' for it in queue)})", flush=True)
    json.dump({"argv": sys.argv, "T0": T0, "inject": a.inject, "D": D_CLOSED, "direction": DIRECTION,
               "start": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(a.out, f"run_args_{int(time.time())}.json"), "w"))

    from spatialvla_core import SpatialVLA, decode_chunks, translation_table
    from svla_rollouts import perturb_token
    import torch
    m = SpatialVLA(); T = translation_table(m)
    gpu = torch.cuda.get_device_name(0)
    ctx = mp.get_context("spawn")
    pipes = []
    for _ in range(a.workers):
        p_, c_ = ctx.Pipe(); pr = ctx.Process(target=worker, args=(c_,), daemon=True); pr.start(); pipes.append(p_)
    out = open(dst, "a"); t0 = time.time(); n = 0
    snaps = {}            # (task, seed, cfg) -> {"actions", "ex", "trace"}
    active = {}

    def start(w):
        for qi, (task, seed, cfg, mode) in enumerate(queue):
            if mode.startswith("natural") or (task, seed, cfg) in snaps:
                break
        else:
            return False
        queue.pop(qi)
        if mode.startswith("natural"):
            replay, ex, tstart = [], Exec2(*parse_cfg(cfg)), 0
        else:
            sn = snaps[(task, seed, cfg)]
            replay, ex, tstart = sn["actions"], copy.deepcopy(sn["ex"]), T0
        ts = time.time()
        eid, ov = eid_of.get((task, seed), (None, None))
        pipes[w].send(("reset", (task, seed, replay, eid))); img, instr, trace, info0 = pipes[w].recv()
        if eid is not None:
            assert info0.get("episode_id") == eid and info0.get("overlay") == ov, (task, seed, eid, ov, info0)
        ep = {"task": task, "seed": seed, "cfg": cfg, "mode": mode.split("#")[0], "cond": tagc(cfg, mode), "img": img, "instr": instr,
              "t": tstart, "ex": ex, "rec": [], "success": False, "final_info": {}, "ever": {}, "t_start": ts,
              "dec_time": 0.0, "n_gen": 0, "init": {k: trace[0][k] for k in ("pose", "state_hash", "img_hash")},
              "episode_id": info0.get("episode_id"), "overlay": info0.get("overlay"), "restore": None}
        if not mode.startswith("natural"):
            sn = snaps[(task, seed, cfg)]
            ok = all(x["state_hash"] == y["state_hash"] and x["img_hash"] == y["img_hash"] for x, y in zip(trace, sn["trace"]))
            ep["restore"] = {"ok": bool(ok and len(trace) == len(sn["trace"])), "n_steps": len(replay)}
            ep["rec"] = copy.deepcopy(sn["rec"])
            for k in ("ever",):
                ep[k] = dict(sn[k])
        active[w] = ep
        return True

    for w in range(a.workers):
        if queue:
            start(w)
    idle = []
    while active or queue:
        if not active:
            raise RuntimeError("queue blocked: branch without root")
        ws = sorted(active)
        plan = [w for w in ws if active[w]["ex"].plans_at(active[w]["t"])]
        gen_ws = [w for w in plan if active[w]["mode"] == "gen"]
        dec_ws = [w for w in plan if active[w]["mode"] != "gen"]
        chunks = {}
        if dec_ws:
            td = time.time()
            # main rows then shadow rows (counterfactual context) for injected chunks
            rows = []                                # (worker, kind) kind = "main" | "cf"
            for w in dec_ws:
                rows.append((w, "main"))
            for w in dec_ws:
                ep = active[w]
                if ep["mode"] not in ("natural",) and is_inj(ep["t"]):
                    rows.append((w, "cf"))
            inps_w = {w: m.inputs(active[w]["img"], active[w]["instr"]) for w in dec_ws}
            inps = [inps_w[w] for w, _ in rows]
            pinfo = {}

            def override(r, g):
                w, kind = rows[r]; ep = active[w]; md = ep["mode"]
                if md == "natural" or not is_inj(ep["t"]):
                    return None
                p, dist, _ = perturb_token(T, g, D_CLOSED, DIRECTION)
                pinfo[w] = (g, p, dist)
                main = {"null": (g, g), "feedback": (p, p), "corrected": (p, g), "reverse": (g, p),
                        "feedbackD": (p, p), "correctedD": (p, g)}[md]
                if kind == "main":
                    return main
                return {"null": (g, p), "feedback": (p, g), "corrected": (p, p), "reverse": (g, g),
                        "feedbackD": (p, g), "correctedD": (p, p)}[md]
            exe, cx, nat1 = decode_fixed(m, inps, override, decode_chunks)
            for r, (w, kind) in enumerate(rows):
                chunks.setdefault(w, {})[kind] = (exe[r], cx[r], int(nat1[r]))
            dt = time.time() - td
            for w in dec_ws:
                active[w]["dec_time"] += dt / len(dec_ws)
        for w in gen_ws:
            td = time.time()
            ids, _ = m.generate_reference(active[w]["img"], active[w]["instr"])
            ids = np.asarray(ids); chunks[w] = {"main": (ids, ids, int(ids[0, 0]))}
            active[w]["dec_time"] += time.time() - td
        for w in ws:
            ep = active[w]; t = ep["t"]; rec = {"t": t}; ex = ep["ex"]
            if w in chunks:
                ep["n_gen"] += 1
                e_, c_, g_ = chunks[w]["main"]
                acts = m.tokens_to_actions(e_)
                ex.add(t, acts)
                ex.cf.pop(t, None)
                rec.update({"exec": e_.tolist(), "ctx": c_.tolist() if not np.array_equal(e_, c_) else None, "nat1": g_})
                if "cf" in chunks[w]:
                    ce, cc, _ = chunks[w]["cf"]
                    ex.cf[t] = np.asarray(m.tokens_to_actions(ce), dtype=np.float64)
                    rec["cf_exec"] = ce.tolist(); rec["cf_ctx"] = cc.tolist()
                ex.cf = {g: v for g, v in ex.cf.items() if t - g < HORIZON}
            env_action, preds, ages = ex.step(t)
            env_action = np.asarray(env_action, dtype=np.float64)
            raw, wts, _ = ex.ens_raw(t, False)
            if ex.cf:
                rawcf, _, _ = ex.ens_raw(t, True)
                rec["ctx_dev"] = np.round(raw - rawcf, 6).tolist()        # actual - counterfactual context, pre-sticky
            if w in chunks and ep["mode"] not in ("natural", "gen") and is_inj(t) and w in pinfo:
                g, p, dist = pinfo[w]
                vg, vp = T["norm"][T["id2row"][g]], T["norm"][T["id2row"][p]]
                delta = T["phys"][T["id2row"][p]] - T["phys"][T["id2row"][g]]
                w0 = float(wts[-1]) if ages[-1] == 0 else 0.0
                exec_p = ep["mode"] in ("feedback", "corrected", "feedbackD", "correctedD")
                off = np.zeros(3)
                if ep["mode"] in ("feedbackD", "correctedD"):
                    off = (1.0 - w0) * delta
                    env_action = env_action.copy(); env_action[:3] = env_action[:3] + off
                rec["inj"] = {"g": int(g), "p": int(p), "dnorm": float(np.linalg.norm(vp - vg)), "g_norm": float(np.linalg.norm(vg)),
                              "flip": bool(np.linalg.norm(vg) < D_CLOSED), "delta_phys": delta.tolist(), "w0": w0,
                              "offset": off.tolist(),
                              "direct_exec": ((w0 * delta if exec_p else np.zeros(3)) + off).tolist()}
            rec.update({"action": env_action.tolist(), "preds": np.round(preds, 6).tolist(), "ages": ages})
            ep["rec"].append(rec)
            pipes[w].send(("step", (ep["task"], env_action)))
        for w in ws:
            ep = active[w]
            img, dn, tr, ob = pipes[w].recv()
            ep["img"] = img; ep["rec"][-1].update({k: ob[k] for k in ("pose", "grip", "contacts", "state_hash", "img_hash")})
            ep["rec"][-1]["info"] = {k: v for k, v in ob["info"].items() if isinstance(v, bool)}
            ep["t"] += 1; ep["success"] = dn; ep["final_info"] = ob["info"]
            for k, v in ob["info"].items():
                if isinstance(v, bool):
                    ep["ever"][k] = ep["ever"].get(k, False) or v
            if ep["mode"] == "natural" and ep["t"] == T0:
                key = (ep["task"], ep["seed"], ep["cfg"])
                snaps[key] = {"actions": [r["action"] for r in ep["rec"]], "ex": copy.deepcopy(ep["ex"]),
                              "trace": [ep["init"]] + [{"state_hash": r["state_hash"], "img_hash": r["img_hash"]} for r in ep["rec"]],
                              "rec": copy.deepcopy(ep["rec"]), "ever": dict(ep["ever"])}
                snaps[key]["trace"][0] = {"state_hash": ep["init"]["state_hash"], "img_hash": ep["init"]["img_hash"]}
            if tr or ep["t"] >= a.horizon:
                out.write(json.dumps({"task": ep["task"], "seed": ep["seed"], "cond": ep["cond"], "cfg": ep["cfg"], "mode": ep["mode"],
                                      "instr": ep["instr"], "episode_id": ep["episode_id"], "overlay": ep["overlay"], "success": bool(ep["success"]),
                                      "steps": ep["t"], "init": ep["init"], "restore": ep["restore"], "T0": T0, "inject": a.inject,
                                      "final_info": ep["final_info"], "ever": ep["ever"], "n_gen_calls": ep["n_gen"],
                                      "wall_s": round(time.time() - ep["t_start"], 2), "decode_s": round(ep["dec_time"], 2),
                                      "gpu": gpu, "rec": ep["rec"]}) + "\n"); out.flush()
                n += 1; del active[w]
                if n % 40 == 0:
                    el = time.time() - t0
                    print(f"[prog] {n} episodes {el / 60:.1f} min ({n / el * 60:.2f}/min)", flush=True)
        for w in range(a.workers):
            if w not in active and queue:
                start(w)
        # drop snapshots whose branches are all done
        pend = {(t, s, c) for t, s, c, _ in queue} | {(e["task"], e["seed"], e["cfg"]) for e in active.values()}
        for k in [k for k in snaps if k not in pend]:
            del snaps[k]
    for p_ in pipes:
        p_.send(("close", None)); p_.recv()
    print(f"[done] {n} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
