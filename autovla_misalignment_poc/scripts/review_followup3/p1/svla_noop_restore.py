#!/usr/bin/env python
"""
SpatialVLA P1-B / P1-C audit runner (PROTOCOL_P1.md §8). RTX 5090 only (source /root/VLA/simpler/env.sh).

Built on review_followup/p1d/p1d_runner.py (HEAD 817d166). It is imported, not modified: Exec2, decode_fixed,
perturb_token, parse_cfg. The changes are:
  * extended full-state fingerprint at reset, every replayed step and every executed step, stored per component
    (sim state, robot qpos/qvel/qacc/drive target, every controller's internal targets/integrator/step counters,
    agent.get_state, TCP pose, gripper, elapsed steps (env + TimeLimit), env seeds and both RandomState states,
    scene timestep, camera model matrices, raw and 224-px image) plus the executor/policy state (chunk buffer, cf,
    sticky gripper) after every executed step;
  * arms from reset (root-like) and from T0 (branched), each comparable to the root N step by step:
      natural      U0   root N from reset (snapshot at T0)              [C5 reference]
      natrep       U0'  N again from reset in the same process           [C5a]
      (run 2)      U0'' N from reset in a separate process (--tag)       [C5b]
      hooknone     U1   from reset; at injection steps the override computes p exactly as the experiment does and
                        returns None (hook installed, no edit)           [C1]
      nullfull     U2   from reset; (g, g) at injection steps + the experiment's shadow cf row   [C2]
      branchN      U3   branched at T0: reset(reconfigure) + replay of N's actions + executor deepcopy, no
                        intervention                                    [C3 / P1-C restore]
      feedback / corrected / reverse   branched at T0 as P1-D            [P1-C: four arms from identical state]
  * C4 analogue at T0 of every root: the chunk N decoded in the experiment batch (decode_fixed, 8 rows) vs
    decode_chunks with B = 1 vs HF generate (m.generate_reference) on the identical observation (token arrays).
Records: episodes.jsonl (per step: action float64, fingerprint component hashes, TCP), c4_t0.jsonl.
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

P1D = "/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1d"
sys.path.insert(0, P1D)
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/spatialvla_protection")
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal")
import p1d_runner as R  # noqa: E402
from svla_protection import decode_fixed, parse_cfg  # noqa: E402

H = lambda x: hashlib.sha1(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()).hexdigest()[:16]
HB = lambda b: hashlib.sha1(b).hexdigest()[:16]
ROOTLIKE = ("natural", "natrep", "hooknone", "nullfull")
BRANCH = ("branchN", "feedback", "corrected", "reverse")
CTRL_ATTRS = ("_target_qpos", "_start_qpos", "_last_drive_qpos_targets", "_interpolation_path", "_interpolation_path_vel",
              "_step", "_sim_steps", "_integral")


def ser(v):
    """json-able, exact (float hex) serialisation of nested state (dicts, arrays, sapien Poses)."""
    if isinstance(v, dict):
        return {str(k): ser(x) for k, x in v.items()}
    if hasattr(v, "p") and hasattr(v, "q"):
        return {"p": ser(np.asarray(v.p)), "q": ser(np.asarray(v.q))}
    if isinstance(v, (list, tuple)):
        return [ser(x) for x in v]
    if v is None or isinstance(v, (bool, str)):
        return v
    return np.asarray(v, np.float64).tobytes().hex()


def fingerprint(env, obs_img_raw, img224):
    u = env.unwrapped
    ag = u.agent
    fp = {"sim_state": H(u.get_state()), "agent_state": HB(json.dumps(ser(ag.get_state()), sort_keys=True).encode()),
          "qpos": H(ag.robot.get_qpos()), "qvel": H(ag.robot.get_qvel()), "qacc": H(ag.robot.get_qacc()),
          "drive_target": H(ag.robot.get_drive_target())}
    for n, c in ag.controller.controllers.items():
        parts = []
        for a in CTRL_ATTRS:
            if hasattr(c, a):
                v = getattr(c, a)
                parts.append(f"{a}={np.asarray(v, np.float64).tolist() if v is not None else None}")
        if hasattr(c, "_target_pose") and c._target_pose is not None:
            parts.append(f"_target_pose={np.asarray(c._target_pose.p).tolist()}{np.asarray(c._target_pose.q).tolist()}")
        fp[f"ctrl_{n}"] = HB("|".join(parts).encode())
    p = u.tcp.pose
    fp["tcp"] = [float(x) for x in p.p] + [float(x) for x in p.q]
    fp["grip"] = float(ag.get_gripper_closedness())
    w, tl = env, None
    while hasattr(w, "env"):
        if type(w).__name__ == "TimeLimit":
            tl = getattr(w, "_elapsed_steps", None)
        w = w.env
    fp["steps"] = [int(u._elapsed_steps), tl]
    fp["seeds"] = [int(u._episode_seed), int(u._main_seed)]
    fp["rng"] = HB(np.asarray(u._episode_rng.get_state()[1]).tobytes() + np.asarray(u._main_rng.get_state()[1]).tobytes()
                   + str(u._episode_rng.get_state()[2:]).encode() + str(u._main_rng.get_state()[2:]).encode())
    fp["timestep"] = float(u._scene.get_timestep())
    cams = []
    for name, cam in u._cameras.items():
        try:
            cams.append(np.asarray(cam.camera.get_model_matrix(), np.float64).ravel())
        except Exception:
            cams.append(np.asarray([np.nan]))
    fp["cameras"] = H(np.concatenate(cams))
    fp["img_raw"] = HB(np.ascontiguousarray(obs_img_raw).tobytes())
    fp["img224"] = HB(np.ascontiguousarray(img224).tobytes())
    return fp


def exec_fp(ex):
    """policy / ensemble / sticky state after a step."""
    parts = [f"{g}:{np.asarray(c, np.float64).tobytes().hex()}" for g, c in ex.chunks]
    parts += [f"cf{g}:{np.asarray(c, np.float64).tobytes().hex()}" for g, c in sorted(ex.cf.items())]
    for a in ("sticky_on", "repeat", "sticky_action", "prev_grip", "K", "E"):
        if hasattr(ex, a):
            v = getattr(ex, a)
            parts.append(f"{a}={np.asarray(v, np.float64).tolist() if v is not None else None}")
    return HB("|".join(parts).encode())


def worker(conn):
    for k, v in (("LP_NUM_THREADS", "2"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1")):
        os.environ[k] = v
    import cv2 as cv
    cv.setNumThreads(1)
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    envs = {}

    def observe(env, obs):
        raw = get_image_from_maniskill2_obs_dict(env, obs)
        img = cv.resize(raw, (224, 224), interpolation=cv.INTER_AREA)              # p1d_runner.py:78
        return img, fingerprint(env, raw, img)

    def scal(info):
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in info.items()
                if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating))}
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":                                                            # p1d_runner.py:98-118
            task, seed, replay, eid = arg
            if task not in envs:
                envs[task] = simpler_env.make(task)
            env = envs[task]
            opts = {"reconfigure": True}
            if eid is not None:
                opts["obj_init_options"] = {"episode_id": int(eid)}
            obs, info0 = env.reset(seed=seed, options=opts)
            info0 = dict(info0); info0["overlay"] = getattr(env.unwrapped, "urdf_version", None)
            img, fp = observe(env, obs)
            trace = [fp]
            for a in replay:
                obs, _, done, trunc, info = env.step(np.asarray(a, dtype=np.float64))
                img, fp = observe(env, obs); trace.append(fp)
            i0 = scal(info0); i0["overlay"] = info0["overlay"]
            conn.send((img, env.get_language_instruction(), trace, i0))
        elif cmd == "step":
            task, action = arg
            env = envs[task]
            obs, _, done, trunc, info = env.step(action)
            img, fp = observe(env, obs)
            conn.send((img, bool(done), bool(trunc), fp, {k: v for k, v in scal(info).items() if isinstance(v, bool)}))
        elif cmd == "close":
            conn.send(None); return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True, help="svla_manifest.json (units: task, seed, episode_id, overlay, configs)")
    ap.add_argument("--modes", default="natural,natrep,hooknone,nullfull,branchN,feedback,corrected,reverse")
    ap.add_argument("--units", default="all", help="'pilot' or 'all'")
    ap.add_argument("--configs", default="", help="override the configs of every unit (pilot)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--c4", action="store_true", help="token-level C4 analogue at T0 of every root")
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    inj = [8, 12, 16, 20]; T0 = 8
    is_inj = lambda t: t in inj
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "episodes.jsonl")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    man = json.load(open(a.manifest))
    units = [u for u in man["units"] if a.units == "all" or u["pilot"]]
    modes = a.modes.split(",")
    items = []
    for u in units:
        for c in (a.configs.split(",") if a.configs else u["configs"]):
            for md in ["natural"] + [m for m in modes if m != "natural"]:
                items.append((u["task"], u["seed"], c, md, u.get("episode_id"), u.get("overlay")))
    queue = sorted(items, key=lambda it: (it[1], it[0], it[2], it[3] in BRANCH))
    print(f"[plan] {len(queue)} episodes", flush=True)
    json.dump({"argv": sys.argv, "T0": T0, "inject": inj, "start": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(a.out, f"run_args_{int(time.time())}.json"), "w"))

    from spatialvla_core import SpatialVLA, decode_chunks, translation_table
    from svla_rollouts import perturb_token
    import torch
    m = SpatialVLA(); T = translation_table(m)
    gpu = torch.cuda.get_device_name(0)
    assert "5090" in gpu, gpu
    ctx = mp.get_context("spawn")
    pipes = []
    for _ in range(a.workers):
        p_, c_ = ctx.Pipe(); pr = ctx.Process(target=worker, args=(c_,), daemon=True); pr.start(); pipes.append(p_)
    out = open(dst, "a"); fc4 = open(os.path.join(a.out, "c4_t0.jsonl"), "a")
    t0 = time.time(); n = 0
    snaps, active = {}, {}

    def start(w):
        for qi, (task, seed, cfg, mode, eid, ov) in enumerate(queue):
            if mode in ROOTLIKE or (task, seed, cfg) in snaps:
                break
        else:
            return False
        queue.pop(qi)
        if mode in ROOTLIKE:
            replay, ex, tstart = [], R.Exec2(*parse_cfg(cfg)), 0
        else:
            sn = snaps[(task, seed, cfg)]
            replay, ex, tstart = sn["actions"], copy.deepcopy(sn["ex"]), T0
        ts = time.time()
        pipes[w].send(("reset", (task, seed, replay, eid))); img, instr, trace, info0 = pipes[w].recv()
        if eid is not None:
            assert info0.get("episode_id") == eid and info0.get("overlay") == ov, (task, seed, eid, ov, info0)
        ep = {"task": task, "seed": seed, "cfg": cfg, "mode": mode, "cond": f"{cfg}:{mode}{a.tag}", "img": img, "instr": instr,
              "t": tstart, "ex": ex, "rec": [], "success": False, "ever": {}, "t_start": ts, "init": trace[0],
              "episode_id": info0.get("episode_id"), "overlay": info0.get("overlay"), "restore": None, "n_gen": 0}
        if mode in BRANCH:
            sn = snaps[(task, seed, cfg)]
            diffs = []
            for i, (x, y) in enumerate(zip(trace, sn["trace"])):
                bad = sorted(k for k in x if k in y and x[k] != y[k])
                if bad:
                    diffs.append({"step": i, "components": bad})
            ex_ok = exec_fp(ep["ex"]) == sn["exfp"]
            ep["restore"] = {"ok": bool(not diffs and len(trace) == len(sn["trace"]) and ex_ok), "n_steps": len(replay),
                             "component_diffs": diffs[:5], "executor_fp_equal": ex_ok}
            ep["rec"] = copy.deepcopy(sn["rec"]); ep["ever"] = dict(sn["ever"])
        active[w] = ep
        return True

    for w in range(a.workers):
        if queue:
            start(w)
    while active or queue:
        if not active:
            raise RuntimeError("queue blocked: branch without root")
        ws = sorted(active)
        dec_ws = [w for w in ws if active[w]["ex"].plans_at(active[w]["t"])]
        chunks = {}
        if dec_ws:
            rows = [(w, "main") for w in dec_ws]
            for w in dec_ws:                                     # shadow rows as P1-D (not for hooknone / natural arms)
                ep = active[w]
                if ep["mode"] in ("nullfull", "feedback", "corrected", "reverse") and is_inj(ep["t"]):
                    rows.append((w, "cf"))
            inps_w = {w: m.inputs(active[w]["img"], active[w]["instr"]) for w in dec_ws}
            inps = [inps_w[w] for w, _ in rows]
            pinfo = {}

            def override(r, g):                                  # p1d_runner.py:305-312 + hooknone
                w, kind = rows[r]; ep = active[w]; md = ep["mode"]
                if md in ("natural", "natrep", "branchN") or not is_inj(ep["t"]):
                    return None
                p, dist, _ = perturb_token(T, g, R.D_CLOSED, R.DIRECTION)
                pinfo[w] = (g, p, dist)
                if md == "hooknone":
                    return None
                main_ = {"nullfull": (g, g), "feedback": (p, p), "corrected": (p, g), "reverse": (g, p)}[md]
                if kind == "main":
                    return main_
                return {"nullfull": (g, p), "feedback": (p, g), "corrected": (p, p), "reverse": (g, g)}[md]
            exe, cx, nat1 = decode_fixed(m, inps, override, decode_chunks)
            for r, (w, kind) in enumerate(rows):
                chunks.setdefault(w, {})[kind] = (exe[r], cx[r], int(nat1[r]))
            if a.c4:                                             # C4 analogue at T0 for roots
                for w in dec_ws:
                    ep = active[w]
                    if ep["mode"] == "natural" and ep["t"] == T0:
                        e1, c1, _ = decode_chunks(m, [inps_w[w]], [0], lambda r, g: None)
                        gid, _ = m.generate_reference(ep["img"], ep["instr"])
                        gid = np.asarray(gid).reshape(4, 3)
                        e8 = chunks[w]["main"][0]
                        fc4.write(json.dumps({"task": ep["task"], "seed": ep["seed"], "cfg": ep["cfg"], "t": T0,
                                              "batch8": e8.tolist(), "b1": e1[0].tolist(), "generate": gid.tolist(),
                                              "b1_equal_batch8": bool(np.array_equal(e1[0], e8)),
                                              "generate_equal_batch8": bool(np.array_equal(gid, e8))}) + "\n"); fc4.flush()
        for w in ws:
            ep = active[w]; t = ep["t"]; rec = {"t": t}; ex = ep["ex"]
            if w in chunks:
                ep["n_gen"] += 1
                e_, c_, g_ = chunks[w]["main"]
                ex.add(t, m.tokens_to_actions(e_)); ex.cf.pop(t, None)
                rec.update({"exec": e_.tolist(), "ctx": c_.tolist() if not np.array_equal(e_, c_) else None, "nat1": g_})
                if "cf" in chunks[w]:
                    ce, cc, _ = chunks[w]["cf"]
                    ex.cf[t] = np.asarray(m.tokens_to_actions(ce), dtype=np.float64)
                ex.cf = {g: v for g, v in ex.cf.items() if t - g < R.HORIZON}
                if w in pinfo:
                    rec["inj"] = {"g": int(pinfo[w][0]), "p": int(pinfo[w][1])}
            env_action, preds, ages = ex.step(t)
            env_action = np.asarray(env_action, dtype=np.float64)
            rec.update({"action": env_action.tolist(), "action_hex": env_action.tobytes().hex()})
            ep["rec"].append(rec)
            pipes[w].send(("step", (ep["task"], env_action)))
        for w in ws:
            ep = active[w]
            img, dn, tr, fp, binfo = pipes[w].recv()
            ep["img"] = img; ep["rec"][-1]["fp"] = fp; ep["rec"][-1]["exfp"] = exec_fp(ep["ex"])
            ep["t"] += 1; ep["success"] = dn
            for k, v in binfo.items():
                ep["ever"][k] = ep["ever"].get(k, False) or v
            if ep["mode"] == "natural" and ep["t"] == T0:
                snaps[(ep["task"], ep["seed"], ep["cfg"])] = {
                    "actions": [r["action"] for r in ep["rec"]], "ex": copy.deepcopy(ep["ex"]), "exfp": exec_fp(ep["ex"]),
                    "trace": [ep["init"]] + [r["fp"] for r in ep["rec"]], "rec": copy.deepcopy(ep["rec"]), "ever": dict(ep["ever"])}
            if tr or ep["t"] >= a.horizon:
                out.write(json.dumps({"task": ep["task"], "seed": ep["seed"], "cond": ep["cond"], "cfg": ep["cfg"], "mode": ep["mode"],
                                      "episode_id": ep["episode_id"], "overlay": ep["overlay"], "success": bool(ep["success"]),
                                      "steps": ep["t"], "init": ep["init"], "restore": ep["restore"], "T0": T0,
                                      "ever": ep["ever"], "n_gen_calls": ep["n_gen"], "wall_s": round(time.time() - ep["t_start"], 2),
                                      "gpu": gpu, "rec": ep["rec"]}) + "\n"); out.flush()
                n += 1; del active[w]
                if n % 20 == 0:
                    el = time.time() - t0
                    print(f"[prog] {n} episodes {el / 60:.1f} min ({n / el * 60:.2f}/min)", flush=True)
        for w in range(a.workers):
            if w not in active and queue:
                start(w)
        pend = {(it[0], it[1], it[2]) for it in queue} | {(e["task"], e["seed"], e["cfg"]) for e in active.values()}
        for k in [k for k in snaps if k not in pend]:
            del snaps[k]
    for p_ in pipes:
        p_.send(("close", None)); p_.recv()
    print(f"[done] {n} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
